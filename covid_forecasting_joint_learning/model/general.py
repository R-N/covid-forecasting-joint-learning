import torch
from torch import nn
from .modules.representation import check_conv_kwargs
from .modules.main import SingleModel
from .train import train, test, val, autoclip_gradient
import datetime
from torch.utils.tensorboard import SummaryWriter
from .scheduler import OneCycleLR, LRFinder
from torch.optim import AdamW
from contextlib import suppress
from pathlib import Path
from . import attr as Attribution
from . import util as ModelUtil
from ..data import util as DataUtil
from matplotlib import pyplot as plt
from .early_stopping import EarlyStopping
import gc
from ..pipeline.main import preprocessing_5, preprocessing_6
from copy import deepcopy
from ..data import cols as DataCol
from .loss import MSSELoss, NaNPredException, NaNLossException
import numpy as np
from math import sqrt, log
import pandas as pd

from .util import LINE_PROFILER


class SourcePick:
    ALL = 0
    CLOSEST = 1
    LONGEST = 2
    NONE = 3


class SharedMode:
    NONE = 0
    PRIVATE = 1
    SHARED = 2


def check_key(dict, key):
    return key in dict and dict[key] is not None


class ClusterModel:
    def __init__(
        self,
        cluster,
        sizes,
        model_kwargs={
            "past_model": {
                "representation_model": {
                    "private_representation": {},
                    "pre_shared_representation": {},
                    "shared_representation": {},
                    "combine_representation": {}
                },
                "private_head": {},
                "shared_head": {}
            },
            "representation_future_model": {
                "private_representation": {},
                "pre_shared_representation": {},
                "shared_representation": {},
                "combine_representation": {}
            },
            "private_head_future_cell": {},
            "shared_head_future_cell": {},
            "post_future_model": {},
        },
        source_pick=SourcePick.ALL,
        private_mode=SharedMode.PRIVATE,
        shared_mode=SharedMode.SHARED,
        optimizer_fn=AdamW,
        lr=1e-5,
        div_factor=25,
        grad_clip_percentile=10,
        optimizer_kwargs={
            "weight_decay": 0.00975
        },
        train_kwargs={},
        grad_scaler=None,
        min_epoch=50,
        shared_model=None,
        device="cpu"
    ):
        self.cluster = cluster
        if source_pick == SourcePick.ALL:
            self.sources = cluster.sources
        elif source_pick == SourcePick.CLOSEST:
            self.sources = [cluster.source_closest]
        elif source_pick == SourcePick.LONGEST:
            self.sources = [cluster.source_longest]
        elif source_pick == SourcePick.NONE:
            self.sources = []
        else:
            raise Exception("Invalid source_pick: %s" % (source_pick,))

        self.target = cluster.target
        self.targets = cluster.targets

        self.source_pick = source_pick
        self.private_mode = private_mode
        self.shared_mode = shared_mode

        self.shared_model = shared_model or SingleModel(**sizes, **model_kwargs)

        if self.shared_mode == SharedMode.SHARED:
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["representation_model"]["shared_representation"] =\
                    self.shared_model.past_model.representation_model.shared_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["shared_head"] =\
                    self.shared_model.past_model.shared_head
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["representation_future_model"]["shared_representation"] =\
                    self.shared_model.representation_future_model.shared_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["shared_head_future_cell"] =\
                    self.shared_model.shared_head_future_cell
        if self.private_mode == SharedMode.SHARED:
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["representation_model"]["private_representation"] =\
                    self.shared_model.past_model.representation_model.private_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["representation_model"]["pre_shared_representation"] =\
                    self.shared_model.past_model.representation_model.pre_shared_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["representation_model"]["combine_representation"] =\
                    self.shared_model.past_model.representation_model.combine_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["past_model"]["private_head"] =\
                    self.shared_model.past_model.private_head
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["representation_future_model"]["private_representation"] =\
                    self.shared_model.representation_future_model.private_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["representation_future_model"]["pre_shared_representation"] =\
                    self.shared_model.representation_future_model.pre_shared_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["representation_future_model"]["combine_representation"] =\
                    self.shared_model.representation_future_model.combine_representation
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["private_head_future_cell"] =\
                    self.shared_model.private_head_future_cell
            with suppress(KeyError, TypeError, AttributeError):
                model_kwargs["post_future_model"] =\
                    self.shared_model.post_future_model

        for k in self.members:
            k.model = SingleModel(**sizes, **model_kwargs)

        self.models = nn.ModuleList([k.model for k in self.members])
        self._device = "cpu"
        self.to(device)
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.train_kwargs = train_kwargs

        self.grad_history = []
        self.grad_clip_percentile = grad_clip_percentile
        self.min_epoch = min_epoch
        self.grad_scaler = grad_scaler
        self.div_factor = div_factor
        self.lr = None
        self.optimizer = self.create_optimizer()
        self.scheduler = None
        if lr is None:
            lr_result = self.find_lr(num_iter=int(0.5 * self.min_epoch))
            lr = min(1, lr_result.best_lr)
            div = lr_result.descend_lr
            if div:
                self.div_factor = lr / div
            self.set_onecycle(lr)
        else:
            self.set_lr(lr)

    def create_optimizer(self):
        return self.optimizer_fn(
            self.models.parameters(),
            **self.optimizer_kwargs
        )

    def create_scheduler(self):
        return OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            div_factor=self.div_factor,
            steps_per_epoch=len(self.target.dataloaders[0]),
            epochs=int(0.5 * self.min_epoch)
        )

    def set_lr(self, lr):
        if lr != self.lr:
            self.lr = lr
            if self.optimizer_kwargs:
                self.optimizer_kwargs["lr"] = lr
                self.optimizer = self.create_optimizer()

    def set_onecycle(self, lr):
        if lr != self.lr:
            self.set_lr(lr)
            self.scheduler = self.create_scheduler()

    def clip_grad_norm(self, lr=None):
        """
        lr = lr or self.lr
        autoclip_gradient(
            self.models,
            self.grad_history,
            clip_percentile=self.grad_clip_percentile,
            max_clip=max(1, min(2, 1 + log(lr, 10)))
        )
        """
        torch.nn.utils.clip_grad_norm_(self.models.parameters(), 1)

    @property
    def members(self):
        return self.sources + self.targets


    def freeze_shared(self, freeze=True):
        for k in self.members:
            self.k.model.freeze_shared(freeze)

    def freeze_private(self, freeze=True):
        for k in self.members:
            self.k.model.freeze_private(freeze)

    def find_lr(self, loss_fn=None, **kwargs):
        def objective(scheduler):
            return self.train(
                loss_fn=loss_fn,
                scheduler=scheduler,
                clip_grad_norm=False
            )[0].item()

        lr_finder = LRFinder(objective, self.models, self.optimizer)
        lr_finder.range_test(**kwargs)
        lr_finder.reset_state()
        return lr_finder.result

    def train(self, grad_scaler=None, loss_fn=None, scheduler=None, clip_grad_norm=True):
        grad_scaler = grad_scaler or self.grad_scaler
        scheduler = scheduler or self.scheduler
        # optimizer = self.create_optimizer()
        train_kwargs = self.train_kwargs
        if loss_fn:
            train_kwargs["loss_fn"] = loss_fn
        return train(
            self.sources,
            self.targets,
            optimizer=self.optimizer,
            scheduler=scheduler,
            key=lambda k: k.dataloaders[0],
            clip_grad_norm=self.clip_grad_norm if clip_grad_norm else None,
            grad_scaler=grad_scaler,
            **train_kwargs
        )

    def val(self, loss_fn=None):
        train_kwargs = self.train_kwargs
        if loss_fn:
            train_kwargs["loss_fn"] = loss_fn
        return val(
            self.sources,
            self.targets,
            key=lambda k: k.dataloaders[1],
            **train_kwargs
        )

    def test(self, loss_fn=None):
        test_kwargs = {k: v for k, v in self.train_kwargs.items() if k not in ["source_weight", "loss_fn"]}
        if loss_fn:
            test_kwargs["loss_fn"] = loss_fn
        return {target.name: test(
            target,
            key=lambda k: k.dataloaders[-1],
            # scaler=target.scaler_2,
            **test_kwargs
        ) for target in self.targets}

    def get_target_model_summary(self):
        return self.target.get_model_summary()

    def write_graph(self, path):
        self.target.write_model_graph(path)

    def to(self, device):
        ret = self.models.to(device)
        self._device = device
        return ret

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.to(value)

    def share_memory(self):
        return self.models.share_memory()

    def get_target_input_attr(self, *args, **kwargs):
        return self.target.get_input_attr(*args, **kwargs)

    def get_target_aggregate_layer_attr(self, *args, **kwargs):
        return self.target.get_aggregate_layer_attr(*args, **kwargs)


DEFAULT_ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "SELU": nn.SELU
}

DEFAULT_FUTURE_EXO_COLS = [
    DataCol.future_exo_cols(dates=DataCol.DATES),
    DataCol.future_exo_cols(dates=DataCol.DATES_BETA),
    DataCol.future_exo_cols(dates=DataCol.DATES_DELTA_I),
    DataCol.future_exo_cols(dates=DataCol.DATES_I),
    DataCol.future_exo_cols(dates=DataCol.DATES_CORR),
    DataCol.future_exo_cols(dates=[]),
    []
]

DEFAULT_PAST_COLS = [
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=DataCol.DATES)),
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=DataCol.DATES_BETA)),
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=DataCol.DATES_DELTA_I)),
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=DataCol.DATES_I)),
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=DataCol.DATES_CORR)),
    DataCol.past_cols(future_exo_cols=DataCol.future_exo_cols(dates=[])),
    DataCol.past_cols(future_exo_cols=[]),
    DataCol.past_cols(future_exo_cols=[], cols_non_date=[])
]


class ObjectiveModel:
    def __init__(
        self,
        cluster,
        hidden_size_past=0,
        hidden_size_future=0,
        shared_state_size=0,
        private_state_size=3,
        representation_past_private_depth=0,
        representation_past_private_kernel_size=3,
        representation_past_private_stride=1,
        representation_past_private_dilation=1,
        representation_past_shared_depth=0,
        representation_past_shared_kernel_size=3,
        representation_past_shared_stride=1,
        representation_past_shared_dilation=1,
        representation_past_pre_shared_depth=0,
        combine_representation_past_w0_mean=1.0,
        combine_representation_past_w0_std=0.0,
        representation_future_private_depth=0,
        representation_future_private_kernel_size=3,
        representation_future_private_stride=1,
        representation_future_private_dilation=1,
        representation_future_shared_depth=0,
        representation_future_shared_kernel_size=3,
        representation_future_shared_stride=1,
        representation_future_shared_dilation=1,
        representation_future_pre_shared_depth=0,
        combine_representation_future_w0_mean=1.0,
        combine_representation_future_w0_std=0.0,
        combine_head_w0_mean=1.0,
        combine_head_w0_std=0.0,
        precombine_head_depth=0,
        combine_head_depth=1,
        conv_activation=nn.ReLU,
        fc_activation=nn.ReLU,
        residual_activation=nn.ReLU,
        past_cols=DEFAULT_PAST_COLS[0],
        future_exo_cols=DEFAULT_FUTURE_EXO_COLS[0],
        batch_size=8,
        additional_past_length=0,
        use_last_past=True,
        seed_length=30,
        source_pick=SourcePick.ALL,
        private_mode=SharedMode.PRIVATE,
        shared_mode=SharedMode.SHARED,
        optimizer_fn=AdamW,
        lr=1e-5,
        weight_decay=0.00975,
        loss_fn=MSSELoss(),
        source_weight=1.0,
        teacher_forcing=True,
        grad_scaler=None,
        trial_id=None,
        log_dir="temp/logs/",
        model_dir="temp/model/",
        debug=False,
        min_epoch=50,
        shared_model=None,
        use_shared=True,
        update_hx=True,
        use_exo=True,
        val=None
    ):
        self.cluster = cluster

        seed_length_0 = seed_length
        if representation_future_private_depth <= 0 and representation_future_shared_depth <= 0:
            seed_length = 1

        past_length = 30 + additional_past_length
        future_length = 14

        model_kwargs = {
            "past_model": {
                "representation_model": {
                    "private_representation": {
                        "depth": representation_past_private_depth,
                        "data_length": past_length,
                        "conv_kwargs": {
                            "kernel_size": representation_past_private_kernel_size,
                            "stride": representation_past_private_stride,
                            "dilation": representation_past_private_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs": {
                            "activation": residual_activation
                        }
                    },
                    "pre_shared_representation": {
                        "depth": representation_past_pre_shared_depth,
                        "data_length": past_length,
                        "conv_kwargs": {
                            "kernel_size": representation_past_shared_kernel_size,
                            "stride": representation_past_shared_stride,
                            "dilation": representation_past_shared_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs": {
                            "activation": residual_activation
                        }
                    } if (representation_past_shared_depth and representation_past_pre_shared_depth) else None,
                    "shared_representation": {
                        "depth": representation_past_shared_depth,
                        "data_length": past_length,
                        "conv_kwargs": {
                            "kernel_size": representation_past_shared_kernel_size,
                            "stride": representation_past_shared_stride,
                            "dilation": representation_past_shared_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs": {
                            "activation": residual_activation
                        }
                    } if representation_past_shared_depth else None,
                    "combine_representation": {
                        "w0_mean": combine_representation_past_w0_mean,
                        "w0_std": combine_representation_past_w0_std,
                    } if representation_past_shared_depth else None
                } if representation_past_private_depth else None,
                "private_head": {
                    "use_last_past": use_last_past
                },
                "shared_head": {
                    "use_last_past": use_last_past
                } if representation_past_shared_depth or use_shared else None
            },
            "representation_future_model": {
                "private_representation": {
                    "depth": representation_future_private_depth,
                    "data_length": future_length,
                    "conv_kwargs": {
                        "kernel_size": representation_future_private_kernel_size,
                        "stride": representation_future_private_stride,
                        "dilation": representation_future_private_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs": {
                        "activation": residual_activation
                    }
                } if representation_future_private_depth else None,
                "pre_shared_representation": {
                    "depth": representation_future_pre_shared_depth,
                    "data_length": future_length,
                    "conv_kwargs": {
                        "kernel_size": representation_future_shared_kernel_size,
                        "stride": representation_future_shared_stride,
                        "dilation": representation_future_shared_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs": {
                        "activation": residual_activation
                    }
                } if (representation_future_shared_depth and representation_future_pre_shared_depth) else None,
                "shared_representation": {
                    "depth": representation_future_shared_depth,
                    "data_length": future_length,
                    "conv_kwargs": {
                        "kernel_size": representation_future_shared_kernel_size,
                        "stride": representation_future_shared_stride,
                        "dilation": representation_future_shared_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs": {
                        "activation": residual_activation
                    }
                } if representation_future_shared_depth else None,
                "combine_representation": {
                    "w0_mean": combine_representation_future_w0_mean,
                    "w0_std": combine_representation_future_w0_std,
                } if representation_future_shared_depth else None
            } if representation_future_private_depth else None,
            "private_head_future_cell": {},
            "shared_head_future_cell": {} if representation_future_shared_depth or use_shared else None,
            "post_future_model": {
                "combiner": {
                    "w0_mean": combine_head_w0_mean,
                    "w0_std": combine_head_w0_std
                } if representation_future_shared_depth or use_shared else None,
                "precombine": {
                    "depth": precombine_head_depth,
                    "fc_activation": fc_activation,
                    "residual_activation": residual_activation
                } if representation_future_shared_depth or precombine_head_depth else None,
                "reducer": {
                    "depth": combine_head_depth,
                    "fc_activation": fc_activation,
                    "residual_activation": residual_activation
                }
            },
            "seed_length": seed_length,
            "teacher_forcing": teacher_forcing,
            "update_hx": update_hx
        }

        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["private_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["pre_shared_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["shared_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["private_representation"]["conv_kwargs"], future_length)
        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["pre_shared_representation"]["conv_kwargs"], future_length)
        with suppress(KeyError, TypeError, AttributeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["shared_representation"]["conv_kwargs"], future_length)

        self.preprocessing_5_params = {
            "past_size": past_length,
            "seed_size": seed_length_0,
            "past_cols": past_cols,
            "future_exo_cols": future_exo_cols,
        }
        self.preprocessing_6_params = {
            "batch_size": batch_size
        }
        self.preprocessing(val=val)

        sample = cluster.target.datasets[0][0]
        input_size_past = sample[0].shape[-1]
        input_size_future = sample[3].shape[-1]

        use_exo = use_exo and future_exo_cols is not None and len(future_exo_cols) > 0
        if use_exo:
            # input_size_past += sample[2].shape[-1]
            input_size_future += sample[4].shape[-1]
        model_kwargs["use_exo"] = use_exo

        sizes = {
            "input_size_past": input_size_past,
            "hidden_size_past": hidden_size_past,
            "input_size_future": input_size_future,
            "hidden_size_future": hidden_size_future,
            "private_state_size": private_state_size,
            "shared_state_size": shared_state_size,
            "output_size": 3,
        }

        self.sizes = sizes
        self.model_kwargs = deepcopy(model_kwargs)

        if debug:
            print(ModelUtil.str_dict(sizes))
            print(ModelUtil.str_dict(model_kwargs))

        self.model = ClusterModel(
            cluster,
            sizes=sizes,
            model_kwargs=model_kwargs,
            source_pick=source_pick,
            private_mode=private_mode,
            shared_mode=shared_mode,
            optimizer_fn=optimizer_fn,
            grad_scaler=grad_scaler,
            lr=lr,
            min_epoch=min_epoch,
            optimizer_kwargs={
                "weight_decay": weight_decay
            },
            train_kwargs={
                "loss_fn": loss_fn,
                "source_weight": source_weight
            },
            shared_model=shared_model
        )

        if debug:
            print(self.get_target_model_summary())

        self.trial_id = trial_id if trial_id is not None else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if isinstance(log_dir, str):
            log_dir = ModelUtil.prepare_dir(log_dir)
            # log_dir = f"{log_dir}T{self.trial_id}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        if self.log_dir:
            self.train_summary_writer = SummaryWriter(self.log_dir + '/train')
            self.val_summary_writer = SummaryWriter(self.log_dir + '/val')

        if isinstance(model_dir, str):
            model_dir = ModelUtil.prepare_dir(model_dir)
            # model_dir = f"{model_dir}{self.trial_id}/{self.cluster.group.id}/{self.cluster.id}/"
            model_dir = f"{model_dir}/{self.cluster.group.id}/{self.cluster.id}/"
            Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir

        self.train_epoch = 0
        self.val_epoch = 0

        self.label = f"G{self.cluster.group.id}.C{self.cluster.id}/"


    def preprocessing(
        self,
        val=None
    ):
        preprocessing_5(
            self.cluster.members,
            label_cols=DataCol.SIRD_VARS,
            final_seed_cols=DataCol.SIRD,
            final_cols=DataCol.IRD,
            val=val,
            **self.preprocessing_5_params
        )
        preprocessing_6(
            self.cluster.members,
            **self.preprocessing_6_params
        )


    def _log_scalar(self, writer, loss, epoch):
        writer.add_scalar(f"{self.label}/avg_loss", loss[0].item(), global_step=epoch)
        writer.add_scalar(f"{self.label}/target_loss", loss[1].item(), global_step=epoch)
        losses = loss[2]
        for i in range(len(losses)):
            writer.add_scalar(f"{self.label}/target_loss_{i}_{self.model.targets[i].name}", losses[i], global_step=epoch)
        writer.flush()


    def set_lr(self, lr):
        return self.model.set_lr(lr)

    def find_lr(self):
        return self.model.find_lr()

    def train(self, epoch=None, loss_fn=None):
        loss = self.model.train(loss_fn=loss_fn)
        epoch = epoch if epoch is not None else self.train_epoch
        if self.log_dir:
            self._log_scalar(self.train_summary_writer, loss, epoch)
        self.train_epoch = epoch + 1
        return loss

    def val(self, epoch=None, loss_fn=None):
        loss = self.model.val(loss_fn=loss_fn)
        epoch = epoch if epoch is not None else self.val_epoch
        if self.log_dir:
            self._log_scalar(self.val_summary_writer, loss, epoch)
        self.val_epoch = epoch + 1
        return loss

    def test(self, loss_fn=None):
        return self.model.test(loss_fn=loss_fn)

    def get_target_model_summary(self):
        return self.model.get_target_model_summary()

    def write_graph(self):
        self.model.write_graph(self.log_dir)

    def freeze_shared(self, freeze=True):
        self.model.freeze_shared(freeze)

    def freeze_private(self, freeze=True):
        self.model.freeze_private(freeze)

    def to(self, device):
        return self.model.to(device)

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, value):
        self.to(value)

    def share_memory(self):
        return self.model.share_memory()

    def get_target_input_attr(self, *args, **kwargs):
        return self.model.get_target_input_attr(*args, **kwargs)

    def get_target_aggregate_layer_attr(self, *args, **kwargs):
        return self.target.get_target_aggregate_layer_attr(*args, **kwargs)

    def pretrain_save_model(self, model_dir=None):
        model_dir = model_dir or self.model_dir
        if not model_dir:
            raise ValueError("Please provide or set model_dir")

        DataUtil.write_string(str(self.get_target_model_summary()), model_dir + "target_model_summary.txt")
        DataUtil.write_string(ModelUtil.str_dict(self.sizes), model_dir + "sizes.json")
        DataUtil.write_string(ModelUtil.str_dict(self.model_kwargs), model_dir + "model_kwargs.json")

        device = str(self.device)
        device = device.split(":", 1)[0]
        if device != "cpu":
            DataUtil.write_string("1", model_dir + device)

    def posttrain_save_model(self, model_dir=None, save_state=False):
        model_dir = model_dir or self.model_dir
        if not model_dir:
            raise ValueError("Please provide or set model_dir")

        if save_state:
            torch.save(self.model.models.state_dict(), f"{model_dir}models.pt")
        for i in range(len(self.model.targets)):
            target = self.model.targets[i]
            model_dir_2 = f"{model_dir}{target.name}/"
            Path(model_dir_2).mkdir(parents=True, exist_ok=True)
            if save_state:
                torch.save(target.model.state_dict(), f"{model_dir_2}model.pt")

            excluded_dates = self.cluster.inverse_date_cols

            input_attr = target.get_input_attr()
            labels = target.dataset_labels[:5]
            labeled_input_attr, input_attr_labels = Attribution.label_input_attr(input_attr, labels)
            input_fig = Attribution.plot_attr(
                labeled_input_attr,
                input_attr_labels,
                exclude_inner=excluded_dates
            )
            input_fig.savefig(f"{model_dir_2}input_attr.jpg", bbox_inches="tight")
            plt.close(input_fig)

            input_keys = sorted(labeled_input_attr.keys())
            input_values = [labeled_input_attr[k] for k in input_keys]
            input_df = pd.DataFrame(input_values, columns=input_attr_labels, index=input_keys)
            input_df.T.to_excel(f"{model_dir_2}input_attr.xlsx", sheet_name="input_attr")

            input_fig_exo_only = Attribution.plot_attr(
                labeled_input_attr,
                input_attr_labels,
                exclude_inner=excluded_dates + DataCol.SIRD_VARS
            )
            input_fig_exo_only.savefig(f"{model_dir_2}input_attr_exo_only.jpg", bbox_inches="tight")
            plt.close(input_fig_exo_only)

            input_fig_date_only = Attribution.plot_attr(
                labeled_input_attr,
                input_attr_labels,
                exclude_inner=excluded_dates + DataCol.SIRD_VARS + DataCol.COLS_NON_DATE
            )
            input_fig_date_only.savefig(f"{model_dir_2}input_attr_date_only.jpg", bbox_inches="tight")
            plt.close(input_fig_date_only)

            layer_attrs = target.get_aggregate_layer_attr()
            layer_fig = Attribution.plot_attr(*Attribution.label_layer_attr(layer_attrs), title="Layer importance")
            layer_fig.savefig(f"{model_dir_2}layer_attr.jpg", bbox_inches="tight")
            plt.close(layer_fig)

            DataUtil.write_string(ModelUtil.str_dict(layer_attrs), f"{model_dir_2}layer_attr.json")


def prepare_params(
    params,
    activations=DEFAULT_ACTIVATIONS,
    past_cols=DEFAULT_PAST_COLS,
    future_exo_cols=DEFAULT_FUTURE_EXO_COLS
):
    params["batch_size"] = 16 * (2**params["batch_size"])
    if isinstance(params["past_cols"], int):
        params["past_cols"] = past_cols[params["past_cols"]]
    with suppress(KeyError):
        if isinstance(params["future_exo_cols"], int):
            params["future_exo_cols"] = future_exo_cols[params["future_exo_cols"]]
    with suppress(KeyError):
        params["conv_activation"] = activations[params["conv_activation"]]
    with suppress(KeyError):
        params["fc_activation"] = activations[params["fc_activation"]]
    with suppress(KeyError):
        params["residual_activation"] = activations[params["residual_activation"]]
    return params


def upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id, only_contents=False):
    if log_dir_i and log_dir_id:
        drive.upload_folder(
            log_dir_i,
            parent_id=log_dir_id,
            only_contents=only_contents,
            replace=False
        )
    if model_dir_i and model_dir_id:
        drive.upload_folder(
            model_dir_i,
            parent_id=model_dir_id,
            only_contents=only_contents,
            replace=False
        )


class TrialWrapper:
    def __init__(self, trial):
        self.trial = trial
        self.number = trial.number

    def suggest_int(self, name, param, *args, **kwargs):
        if isinstance(param, tuple) or isinstance(param, list):  # and param[0] != param[1]:
            return self.trial.suggest_int(name, *param, *args, **kwargs)
        return param

    def suggest_float(self, name, param, *args, **kwargs):
        if isinstance(param, tuple) or isinstance(param, list):  # and param[0] != param[1]:
            return self.trial.suggest_float(name, *param, *args, **kwargs)
        return param

    def suggest_categorical(self, name, param, *args, **kwargs):
        if isinstance(param, list) or isinstance(param, tuple):
            return self.trial.suggest_categorical(name, param, *args, **kwargs)
        return param

    def suggest_categorical_list(self, name, param, *args, **kwargs):
        if (isinstance(param, list) or isinstance(param, tuple)) and (isinstance(param[0], list) or isinstance(param[0], tuple)):
            return self.trial.suggest_categorical(name, param, *args, **kwargs)
        return param


def clean_params(
    params,
    source_pick,
    joint_learning,
    use_representation_past,
    use_representation_future,
    use_shared,
    activations,
    past_cols,
    future_exo_cols
):
    if "onecycle" in params:
        onecycle = params["onecycle"]
        if onecycle:
            params["lr"] = None
        params.pop("onecycle", None)

    if not joint_learning:
        params.pop("source_weight", None)
        source_pick = SourcePick.NONE

    if not (use_representation_past or use_representation_future):
        params.pop("conv_activation", None)

    if not use_shared:
        params.pop("shared_state_size", None)
        params.pop("combine_head_w0_mean", None)
        params.pop("combine_head_w0_std", None)
        params.pop("precombine_head_depth", None)

    if not use_representation_past:
        params.pop("hidden_size_past", None)
        params.pop("representation_past_private_depth", None)
        params.pop("representation_past_private_kernel_size", None)
        params.pop("representation_past_private_stride", None)
        params.pop("representation_past_private_dilation", None)

    if use_representation_past and not use_shared:
        params.pop("representation_past_shared_depth", None)
        params.pop("representation_past_shared_kernel_size", None)
        params.pop("representation_past_shared_stride", None)
        params.pop("representation_past_shared_dilation", None)
        params.pop("representation_past_pre_shared_depth", None)
        params.pop("combine_representation_past_w0_mean", None)
        params.pop("combine_representation_past_w0_std", None)

    if not use_representation_future:
        params.pop("hidden_size_future", None)
        params.pop("representation_future_private_depth", None)
        params.pop("representation_future_private_kernel_size", None)
        params.pop("representation_future_private_stride", None)
        params.pop("representation_future_private_dilation", None)

    if use_representation_future and not use_shared:
        params.pop("representation_future_shared_depth", None)
        params.pop("representation_future_shared_kernel_size", None)
        params.pop("representation_future_shared_stride", None)
        params.pop("representation_future_shared_dilation", None)
        params.pop("representation_future_pre_shared_depth", None)
        params.pop("combine_representation_future_w0_mean", None)
        params.pop("combine_representation_future_w0_std", None)

    params = prepare_params(params, activations, past_cols, future_exo_cols)

    if "use_exo" not in params:
        params["use_exo"] = bool(params["future_exo_cols"])

    return params, source_pick


def eval(
    groups,
    params,
    log_dir="temp/logs/",
    model_dir="temp/model/",
    log_dir_copy=None,
    model_dir_copy=None,
    drive=None,
    log_dir_id=None,
    model_dir_id=None,
    device=None,
    write_graph=True,
    early_stopping_interval_mode=1,
    min_epoch=100,
    max_epoch=None,
    teacher_forcing=True,
    activations=DEFAULT_ACTIVATIONS,
    past_cols=DEFAULT_PAST_COLS,
    future_exo_cols=DEFAULT_FUTURE_EXO_COLS,
    source_pick=SourcePick.ALL,
    private_mode=SharedMode.PRIVATE,
    shared_mode=SharedMode.SHARED,
    pretrain_upload=False,
    posttrain_upload=False,
    pretrain_copy=True,
    posttrain_copy=True,
    cleanup=True,
    use_representation_past=True,
    use_representation_future=False,
    use_shared=True,
    update_hx=True,
    joint_learning=True,
    merge_clusters=False,
    debug=False,
    val=0,  # use 2 for pred training
    continue_train=True,
    trial_id=-1,
    copy_group=False,
    early_stopping_kwargs={
        # "rise_patience": 25,
        # "still_patience": 25,
        # "both_patience": 75
    }
):
    if device is None:
        device = ModelUtil.DEVICE

    assert (not log_dir_copy) or log_dir
    assert (not model_dir_copy) or model_dir

    log_dir = ModelUtil.prepare_dir(log_dir)
    model_dir = ModelUtil.prepare_dir(model_dir)
    log_dir_copy = ModelUtil.prepare_dir(log_dir_copy)
    model_dir_copy = ModelUtil.prepare_dir(model_dir_copy)

    log_dir_i, model_dir_i = ModelUtil.prepare_log_model_dir(log_dir, model_dir, trial_id, mkdir=True)
    log_dir_copy_i, model_dir_copy_i = ModelUtil.prepare_log_model_dir(log_dir_copy, model_dir_copy, trial_id, mkdir=False)

    ModelUtil.global_random_seed()

    params, source_pick = clean_params(
        params,
        source_pick,
        joint_learning,
        use_representation_past,
        use_representation_future,
        use_shared,
        activations,
        past_cols,
        future_exo_cols
    )

    target_losses = {}
    epoch_log = {}

    for group_0 in groups:
        group = group_0.copy() if copy_group else group_0
        target_losses[group.id] = {}
        epoch_log[group.id] = {}
        clusters = [group.merge_clusters()] if merge_clusters else group.clusters
        for cluster in clusters:
            if debug and (group.id > 0 or cluster.id > 1):
                continue

            print(f"Model for {trial_id}.{group.id}.{cluster.id}")

            grad_scaler = None  # GradScaler(init_scale=8192)

            model = ObjectiveModel(
                cluster,
                trial_id=trial_id,
                log_dir=log_dir_i,
                model_dir=model_dir_i,
                grad_scaler=grad_scaler,
                # teacher_forcing=True,
                min_epoch=min_epoch,
                use_shared=use_shared,
                source_pick=source_pick,
                private_mode=private_mode,
                shared_mode=shared_mode,
                debug=debug,
                val=val,
                **params
            )
            model.to(device)

            if write_graph and group.id == 0 and cluster.id <= 0:
                model.write_graph()

            if model_dir:
                model.pretrain_save_model()

            if pretrain_copy:
                if model_dir_copy_i:
                    ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
            if drive and pretrain_upload:
                upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)

            early_stopping = EarlyStopping(
                model.model.models,
                debug=1,
                log_dir=model.log_dir,
                label=model.label,
                interval_mode=early_stopping_interval_mode,
                wait=min_epoch,
                max_epoch=max_epoch,
                **early_stopping_kwargs
            )

            while not early_stopping.stopped:
                train_loss_target, val_loss_target = np.nan, np.nan
                try:
                    train_loss, train_loss_target, train_loss_targets = model.train()
                    if torch.isnan(train_loss).any():
                        raise NaNLossException()
                    train_loss, train_loss_target = train_loss.item(), train_loss_target.item()

                    val_loss, val_loss_target, val_loss_targets = model.val()
                    if torch.isnan(val_loss).any():
                        raise NaNLossException()
                    val_loss, val_loss_target = val_loss.item(), val_loss_target.item()

                    early_stopping(train_loss_target, val_loss_target)
                except (NaNPredException, NaNLossException):
                    if not early_stopping.step_nan():
                        raise

            last_epoch, stop_reason, best_epoch = early_stopping.last_epoch, early_stopping.stop_reason, early_stopping.best_epoch
            last_epoch, best_epoch = str(last_epoch), str(best_epoch)

            if continue_train:
                best_epoch = early_stopping.best_epoch
                best_loss = early_stopping.best_val_loss_2
                first_train_count = len(model.model.target.datasets[0])
                model.preprocessing(val=val+1)
                second_train_count = len(model.model.target.datasets[0])

                early_stopping_2 = EarlyStopping(
                    model.model.models,
                    debug=1,
                    log_dir=model.log_dir,
                    label=model.label + "a",
                    interval_mode=early_stopping_interval_mode,
                    wait=0,
                    max_epoch=int(best_epoch * (second_train_count / first_train_count)),
                    update_state_mode=1,
                    **early_stopping_kwargs
                )

                while not early_stopping_2.stopped:
                    train_loss_target, val_loss_target = np.nan, np.nan
                    try:
                        train_loss, train_loss_target, train_loss_targets = model.train()
                        if torch.isnan(train_loss).any():
                            raise NaNLossException()
                        train_loss, train_loss_target = train_loss.item(), train_loss_target.item()

                        val_loss, val_loss_target, val_loss_targets = model.val()
                        if torch.isnan(val_loss).any():
                            raise NaNLossException()
                        val_loss, val_loss_target = val_loss.item(), val_loss_target.item()

                        early_stopping_2(train_loss_target, val_loss_target)
                    except (NaNPredException, NaNLossException):
                        if not early_stopping_2.step_nan():
                            raise

                if best_loss < early_stopping_2.best_val_loss:
                    print("Second training results in higher loss. Loading previous best state")
                    early_stopping.load_best_state()
                else:
                    last_epoch_2, stop_reason_2, best_epoch_2 = early_stopping_2.last_epoch, early_stopping_2.stop_reason, early_stopping_2.best_epoch
                    last_epoch, stop_reason, best_epoch = last_epoch+last_epoch_2, stop_reason+stop_reason_2, best_epoch+best_epoch_2

            test_loss = model.test()
            print("Test loss:", test_loss)
            target_losses[group.id][cluster.id] = test_loss
            epoch_log[group.id][cluster.id] = {
                "last_epoch": last_epoch,
                "stop_reason": stop_reason,
                "best_epoch": best_epoch
            }

            if model_dir:
                model.posttrain_save_model(save_state=True)

            if posttrain_copy:
                if log_dir_copy_i:
                    ModelUtil.copytree(log_dir_i, log_dir_copy_i, dirs_exist_ok=True)
                if model_dir_copy_i:
                    ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
            if drive and posttrain_upload:
                upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)

            del model
            del early_stopping

            torch.cuda.empty_cache()
            gc.collect()

        del group
        del clusters
        gc.collect()

    if not posttrain_copy:
        if log_dir_copy_i:
            ModelUtil.copytree(log_dir_i, log_dir_copy_i, dirs_exist_ok=True)
        if model_dir_copy_i:
            ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
    if drive and not posttrain_upload:
        upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)
    if cleanup:
        if log_dir_i and (log_dir_copy_i or drive):
            ModelUtil.rmtree(log_dir_i)
        if model_dir_i and (model_dir_copy_i or drive):
            ModelUtil.rmtree(model_dir_i)

    return target_losses, epoch_log



def make_objective(
    groups,
    log_dir="temp/logs/",
    model_dir="temp/model/",
    log_dir_copy=None,
    model_dir_copy=None,
    drive=None,
    log_dir_id=None,
    model_dir_id=None,
    device=None,
    write_graph=False,
    early_stopping_interval_mode=1,
    min_epoch=50,
    max_epoch=150,
    teacher_forcing=True,
    activations=DEFAULT_ACTIVATIONS,
    hidden_sizes=(3, 37),
    state_sizes=(3, 56),
    normal_conv_depths=(1, 20),
    pre_conv_depths=(0, 5),
    normal_fc_depths=(1, 20),
    pre_fc_depths=(0, 5),
    past_kernel_sizes=(3, 14),
    past_strides=(1, 1),
    past_dilations=(1, 1),
    future_kernel_sizes=(3, 7),
    future_strides=(1, 1),
    future_dilations=(1, 1),
    w0_means=(0.0, 1.0),
    w0_stds=(0.0, 0.5),
    booleans=(0, 1),
    onecycles=(0, 1),
    lrs=(1e-5, 1e-2),
    weight_decays=(0, 0.3),
    source_weights=(0.5, 1.0),
    batch_sizes=(0, 5),
    additional_past_lengths=(0, 4),
    seed_lengths=30,
    past_cols=DEFAULT_PAST_COLS,
    future_exo_cols=DEFAULT_FUTURE_EXO_COLS,
    source_pick=SourcePick.ALL,
    private_mode=SharedMode.PRIVATE,
    shared_mode=SharedMode.SHARED,
    pretrain_upload=False,
    posttrain_upload=False,
    pretrain_copy=True,
    posttrain_copy=True,
    cleanup=True,
    use_representation_past=True,
    use_representation_future=False,
    use_shared=True,
    update_hx=True,
    joint_learning=True,
    merge_clusters=False,
    debug=False
):
    if device is None:
        device = ModelUtil.DEVICE
    activation_keys = [x for x in activations.keys()]

    assert (not log_dir_copy) or log_dir
    assert (not model_dir_copy) or model_dir

    log_dir = ModelUtil.prepare_dir(log_dir)
    model_dir = ModelUtil.prepare_dir(model_dir)
    log_dir_copy = ModelUtil.prepare_dir(log_dir_copy)
    model_dir_copy = ModelUtil.prepare_dir(model_dir_copy)

    def objective(
        trial
    ):
        trial_id = trial.number

        log_dir_i, model_dir_i = ModelUtil.prepare_log_model_dir(log_dir, model_dir, trial_id, mkdir=True)
        log_dir_copy_i, model_dir_copy_i = ModelUtil.prepare_log_model_dir(log_dir_copy, model_dir_copy, trial_id, mkdir=False)

        trial = TrialWrapper(trial)
        ModelUtil.global_random_seed()

        params = {
            "private_state_size": trial.suggest_int("private_state_size", state_sizes),
            "fc_activation": trial.suggest_categorical("fc_activation", activation_keys),
            "residual_activation": trial.suggest_categorical("residual_activation", activation_keys),
            "combine_head_depth": trial.suggest_int("combine_head_depth", normal_fc_depths),
            "batch_size": trial.suggest_int("batch_size", batch_sizes),
            "additional_past_length": trial.suggest_int("additional_past_length", additional_past_lengths),
            "seed_length": trial.suggest_int("seed_length", seed_lengths),
            "use_last_past": trial.suggest_int("use_last_past", booleans),
            "past_cols": trial.suggest_int("past_cols", (0, len(past_cols) - 1)),
            "future_exo_cols": trial.suggest_int("future_exo_cols", (0, len(future_exo_cols) - 1)),
            "teacher_forcing": trial.suggest_categorical("teacher_forcing", teacher_forcing),
            "update_hx": trial.suggest_categorical("update_hx", update_hx),
            "weight_decay": trial.suggest_float("weight_decay", weight_decays),
        }

        onecycle = trial.suggest_categorical("onecycle", onecycles)
        if onecycle:
            params["lr"] = None
        else:
            params["lr"] = trial.suggest_float("lr", lrs)


        source_pick_1 = source_pick
        if joint_learning:
            params.update({
                "source_weight": trial.suggest_float("source_weight", source_weights)
            })
        else:
            source_pick_1 = SourcePick.NONE

        if use_representation_past or use_representation_future:
            params.update({
                "conv_activation": trial.suggest_categorical("conv_activation", activation_keys)
            })

        if use_shared:
            params.update({
                "shared_state_size": trial.suggest_int("shared_state_size", state_sizes),
                "combine_head_w0_mean": trial.suggest_float("combine_head_w0_mean", w0_means),
                "combine_head_w0_std": trial.suggest_float("combine_head_w0_std", w0_stds),
                "precombine_head_depth": trial.suggest_int("precombine_head_depth", pre_fc_depths)
            })

        if use_representation_past:
            params.update({
                "hidden_size_past": trial.suggest_int("hidden_size_past", hidden_sizes),
                "representation_past_private_depth": trial.suggest_int("representation_past_private_depth", normal_conv_depths),
                "representation_past_private_kernel_size": trial.suggest_int("representation_past_private_kernel_size", past_kernel_sizes),
                "representation_past_private_stride": trial.suggest_int("representation_past_private_stride", past_strides),
                "representation_past_private_dilation": trial.suggest_int("representation_past_private_dilation", past_dilations)
            })
            if use_shared:
                params.update({
                    "representation_past_shared_depth": trial.suggest_int("representation_past_shared_depth", normal_conv_depths),
                    "representation_past_shared_kernel_size": trial.suggest_int("representation_past_shared_kernel_size", past_kernel_sizes),
                    "representation_past_shared_stride": trial.suggest_int("representation_past_shared_stride", past_strides),
                    "representation_past_shared_dilation": trial.suggest_int("representation_past_shared_dilation", past_dilations),
                    "representation_past_pre_shared_depth": trial.suggest_int("representation_past_pre_shared_depth", pre_conv_depths),
                    "combine_representation_past_w0_mean": trial.suggest_float("combine_representation_past_w0_mean", w0_means),
                    "combine_representation_past_w0_std": trial.suggest_float("combine_representation_past_w0_std", w0_stds)
                })

        if use_representation_future:
            params.update({
                "hidden_size_future": trial.suggest_int("hidden_size_future", hidden_sizes),
                "representation_future_private_depth": trial.suggest_int("representation_future_private_depth", normal_conv_depths),
                "representation_future_private_kernel_size": trial.suggest_int("representation_future_private_kernel_size", future_kernel_sizes),
                "representation_future_private_stride": trial.suggest_int("representation_future_private_stride", future_strides),
                "representation_future_private_dilation": trial.suggest_int("representation_future_private_dilation", future_dilations)
            })
            if use_shared:
                params.update({
                    "representation_future_shared_depth": trial.suggest_int("representation_future_shared_depth", normal_conv_depths),
                    "representation_future_shared_kernel_size": trial.suggest_int("representation_future_shared_kernel_size", future_kernel_sizes),
                    "representation_future_shared_stride": trial.suggest_int("representation_future_shared_stride", future_strides),
                    "representation_future_shared_dilation": trial.suggest_int("representation_future_shared_dilation", future_dilations),
                    "representation_future_pre_shared_depth": trial.suggest_int("representation_future_pre_shared_depth", pre_conv_depths),
                    "combine_representation_future_w0_mean": trial.suggest_float("combine_representation_future_w0_mean", w0_means),
                    "combine_representation_future_w0_std": trial.suggest_float("combine_representation_future_w0_std", w0_stds)
                })

        params = prepare_params(params, activations, past_cols, future_exo_cols)
        use_exo = bool(params["future_exo_cols"])
        params["use_exo"] = use_exo

        sum_val_loss_target = 0

        for group_0 in groups:
            group = group_0.copy()
            clusters = [group.merge_clusters()] if merge_clusters else group.clusters
            sum_val_loss_target_group = 0
            for cluster in clusters:

                if debug and (group.id > 0 or cluster.id > 1):
                    continue

                print(f"Model for {trial_id}.{group.id}.{cluster.id}")

                grad_scaler = None  # GradScaler(init_scale=8192)

                model = ObjectiveModel(
                    cluster,
                    trial_id=trial_id,
                    log_dir=log_dir_i,
                    model_dir=model_dir_i,
                    grad_scaler=grad_scaler,
                    # teacher_forcing=True,
                    min_epoch=min_epoch,
                    use_shared=use_shared,
                    source_pick=source_pick_1,
                    private_mode=private_mode,
                    shared_mode=shared_mode,
                    debug=debug,
                    **params
                )
                model.to(device)

                if write_graph and group.id == 0 and cluster.id <= 0:
                    model.write_graph()

                if model_dir:
                    model.pretrain_save_model()

                if pretrain_copy:
                    if model_dir_copy_i:
                        ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
                if drive and pretrain_upload:
                    upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)

                early_stopping = EarlyStopping(
                    model.model.models,
                    debug=1,
                    log_dir=model.log_dir,
                    label=model.label,
                    interval_mode=early_stopping_interval_mode,
                    wait=min_epoch,
                    max_epoch=max_epoch
                )

                best_loss = np.inf
                while not early_stopping.stopped:
                    train_loss_target, val_loss_target = np.nan, np.nan
                    try:
                        train_loss, train_loss_target, train_loss_targets = model.train()
                        if torch.isnan(train_loss).any():
                            raise NaNLossException()
                        train_loss, train_loss_target = train_loss.item(), train_loss_target.item()
                        val_loss, val_loss_target, val_loss_targets = model.val()
                        if torch.isnan(val_loss).any():
                            raise NaNLossException()
                        val_loss, val_loss_target = val_loss.item(), val_loss_target.item()
                        best_loss = min(best_loss, val_loss_target)

                        early_stopping(train_loss_target, val_loss_target)
                    except (NaNPredException, NaNLossException):
                        if not early_stopping.step_nan():
                            raise

                sum_val_loss_target_group += best_loss
                if model_dir:
                    model.posttrain_save_model()

                if posttrain_copy:
                    if log_dir_copy_i:
                        ModelUtil.copytree(log_dir_i, log_dir_copy_i, dirs_exist_ok=True)
                    if model_dir_copy_i:
                        ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
                if drive and posttrain_upload:
                    upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)

                del model
                del early_stopping

                torch.cuda.empty_cache()
                gc.collect()

            sum_val_loss_target_group /= len(clusters)
            sum_val_loss_target += sum_val_loss_target_group
            del group
            del clusters
            gc.collect()

        if not posttrain_copy:
            if log_dir_copy_i:
                ModelUtil.copytree(log_dir_i, log_dir_copy_i, dirs_exist_ok=True)
            if model_dir_copy_i:
                ModelUtil.copytree(model_dir_i, model_dir_copy_i, dirs_exist_ok=True)
        if drive and not posttrain_upload:
            upload_logs(drive, trial_id, log_dir_i, log_dir_id, model_dir_i, model_dir_id)
        if cleanup:
            if log_dir_i and (log_dir_copy_i or drive):
                ModelUtil.rmtree(log_dir_i)
            if model_dir_i and (model_dir_copy_i or drive):
                ModelUtil.rmtree(model_dir_i)

        return sum_val_loss_target / len(groups)

    return objective