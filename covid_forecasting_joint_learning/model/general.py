import torch
from torch import nn
from .modules.representation import check_conv_kwargs
from .modules.main import SingleModel
from .train import train, test
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
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
        max_grad_norm=1.0,
        optimizer_kwargs={},
        train_kwargs={},
        grad_scaler=None,
        min_epochs=50,
        shared_model=None
    ):
        self.cluster = cluster
        if source_pick == SourcePick.ALL:
            self.sources = cluster.sources
        elif source_pick == SourcePick.CLOSEST:
            self.sources = [cluster.source_closest]
        elif source_pick == SourcePick.LONGEST:
            self.sources = [cluster.source_longest]
        elif self.source_pick == SourcePick.NONE:
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
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["representation_model"]["shared_representation"] =\
                    self.shared_model.past_model.representation_model.shared_representation
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["shared_head"] =\
                    self.shared_model.past_model.shared_head
            with suppress(KeyError, TypeError):
                model_kwargs["representation_future_model"]["shared_representation"] =\
                    self.shared_model.representation_future_model.shared_representation
            with suppress(KeyError, TypeError):
                model_kwargs["shared_head_future_cell"] =\
                    self.shared_model.shared_head_future_cell
        if self.private_mode == SharedMode.SHARED:
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["representation_model"]["private_representation"] =\
                    self.shared_model.past_model.representation_model.private_representation
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["representation_model"]["pre_shared_representation"] =\
                    self.shared_model.past_model.representation_model.pre_shared_representation
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["representation_model"]["combine_representation"] =\
                    self.shared_model.past_model.representation_model.combine_representation
            with suppress(KeyError, TypeError):
                model_kwargs["past_model"]["private_head"] =\
                    self.shared_model.past_model.private_head
            with suppress(KeyError, TypeError):
                model_kwargs["representation_future_model"]["private_representation"] =\
                    self.shared_model.representation_future_model.private_representation
            with suppress(KeyError, TypeError):
                model_kwargs["representation_future_model"]["pre_shared_representation"] =\
                    self.shared_model.representation_future_model.pre_shared_representation
            with suppress(KeyError, TypeError):
                model_kwargs["representation_future_model"]["combine_representation"] =\
                    self.shared_model.representation_future_model.combine_representation
            with suppress(KeyError, TypeError):
                model_kwargs["private_head_future_cell"] =\
                    self.shared_model.private_head_future_cell
            with suppress(KeyError, TypeError):
                model_kwargs["post_future_model"] =\
                    self.shared_model.post_future_model

        for k in self.members:
            k.model = SingleModel(**sizes, **model_kwargs)

        self.models = nn.ModuleList([k.model for k in self.members])
        self.optimizer_fn = optimizer_fn
        self.lr = lr
        optimizer_kwargs["lr"] = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.train_kwargs = train_kwargs

        self.max_grad_norm = max_grad_norm
        self.optimizer = self.create_optimizer()
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=len(self.target.datasets[0]) * min_epochs)
        self.min_epochs = min_epochs
        self.grad_scaler = grad_scaler

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.models.parameters(), self.max_grad_norm)

    @property
    def members(self):
        return [*self.sources, self.target]

    def create_optimizer(self):
        return self.optimizer_fn(self.models.parameters(), **self.optimizer_kwargs)

    def freeze_shared(self, freeze=True):
        for k in self.members:
            self.k.model.freeze_shared(freeze)

    def freeze_private(self, freeze=True):
        for k in self.members:
            self.k.model.freeze_private(freeze)

    def train(self, grad_scaler=None):
        grad_scaler = grad_scaler or self.grad_scaler
        # optimizer = self.create_optimizer()
        return train(
            self.sources,
            self.targets,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            key=lambda k: k.dataloaders[0],
            clip_grad_norm=self.clip_grad_norm,
            grad_scaler=grad_scaler,
            **self.train_kwargs
        )

    def val(self):
        return test(
            self.sources,
            self.targets,
            key=lambda k: k.dataloaders[1],
            **self.train_kwargs
        )

    def test(self):
        return test(
            self.sources,
            self.targets,
            key=lambda k: k.dataloaders[2],
            **self.train_kwargs
        )

    def get_target_model_summary(self):
        return self.target.get_model_summary()

    def write_graph(self, path):
        self.target.write_model_graph(path)

    def to(self, device):
        return self.models.to(device)

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
DEFAULT_PAST_COLS = [None]
DEFAULT_FUTURE_EXO_COLS = [["psbb", "ppkm", "ppkm_mikro"]]


class ObjectiveModel:
    def __init__(
        self,
        cluster,
        hidden_size_past=3,
        hidden_size_future=3,
        shared_state_size=3,
        private_state_size=3,
        representation_past_private_depth=1,
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
        combine_head_depth=0,
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
        optimizer_fn=torch.optim.Adam,
        lr=1e-5,
        loss_fn=nn.MSELoss(),
        source_weight=1.0,
        teacher_forcing=True,
        grad_scaler=None,
        trial_id=None,
        log_dir=None,
        model_dir=None,
        debug=False,
        min_epochs=50,
        shared_model=None,
        use_shared=True
    ):
        self.cluster = cluster

        if representation_future_private_depth <= 0 and representation_future_shared_depth <= 0:
            seed_length = 1

        past_length = 30 + additional_past_length
        future_length = 14

        print(precombine_head_depth)

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
                },
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
                },
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
            },
            "private_head_future_cell": {},
            "shared_head_future_cell": {} if representation_future_shared_depth or use_shared else None,
            "post_future_model": {
                "combiner": {
                    "w0_mean": combine_head_w0_mean,
                    "w0_std": combine_head_w0_std
                } if representation_future_shared_depth else None,
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
            "teacher_forcing": teacher_forcing
        }

        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["private_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["pre_shared_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["shared_representation"]["conv_kwargs"], past_length)
        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["private_representation"]["conv_kwargs"], future_length)
        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["pre_shared_representation"]["conv_kwargs"], future_length)
        with suppress(KeyError, TypeError):
            check_conv_kwargs(model_kwargs["representation_future_model"]["shared_representation"]["conv_kwargs"], future_length)

        members = cluster.members
        preprocessing_5(
            members,
            past_size=past_length,
            past_cols=past_cols,
            future_exo_cols=future_exo_cols
        )
        preprocessing_6(
            members,
            batch_size=batch_size
        )

        sample = cluster.target.datasets[0][0]
        input_size_past = sample[0].shape[-1]
        input_size_future = sample[3].shape[-1]

        use_exo_cols = future_exo_cols is not None and len(future_exo_cols) > 0
        if use_exo_cols:
            # input_size_past += sample[2].shape[-1]
            input_size_future += sample[4].shape[-1]
        model_kwargs["use_exo"] = use_exo_cols

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
            min_epochs=min_epochs,
            optimizer_kwargs={
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
            if not log_dir.endswith("/"):
                log_dir = log_dir + "/"
            log_dir = log_dir + str(self.trial_id)
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        if self.log_dir:
            self.train_summary_writer = SummaryWriter(self.log_dir + '/train')
            self.val_summary_writer = SummaryWriter(self.log_dir + '/val')

        if isinstance(model_dir, str):
            if not model_dir.endswith("/"):
                model_dir = model_dir + "/"
            model_dir = model_dir + f"{self.trial_id}/{self.cluster.group.id}/{self.cluster.id}/"
            Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir

        self.train_epoch = 0
        self.val_epoch = 0

        self.label = f"G{self.cluster.group.id}.C{self.cluster.id}/"


    def _log_scalar(self, writer, loss, epoch):
        writer.add_scalar(f"{self.label}/avg_loss", loss[0].item(), global_step=epoch)
        writer.add_scalar(f"{self.label}/target_loss", loss[1].item(), global_step=epoch)
        losses = loss[2]
        for i in range(len(losses)):
            writer.add_scalar(f"{self.label}/target_loss_{i}_{self.model.targets[i].name}", losses[i], global_step=epoch)
        writer.flush()


    def train(self, epoch=None):
        loss = self.model.train()
        epoch = epoch if epoch is not None else self.train_epoch
        if self.log_dir:
            self._log_scalar(self.train_summary_writer, loss, epoch)
        self.train_epoch = epoch + 1
        return loss

    def val(self, epoch=None):
        loss = self.model.val()
        epoch = epoch if epoch is not None else self.val_epoch
        if self.log_dir:
            self._log_scalar(self.val_summary_writer, loss, epoch)
        self.val_epoch = epoch + 1
        return loss

    def test(self):
        return self.model.test()

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

    def posttrain_save_model(self, model_dir=None):
        model_dir = model_dir or self.model_dir
        if not model_dir:
            raise ValueError("Please provide or set model_dir")

        torch.save(self.model.models.state_dict(), f"{model_dir}models.pt")
        for i in range(len(self.model.targets)):
            target = self.model.targets[i]
            suffix = f"_{i}_{target.name}"
            torch.save(target.model.state_dict(), f"{model_dir}target{suffix}.pt")

            input_attr = target.get_input_attr()
            input_fig = Attribution.plot_attr(*Attribution.label_input_attr(input_attr, target.dataset_labels))
            input_fig.savefig(f"{model_dir}input_attr{suffix}.jpg", bbox_inches="tight")
            plt.close(input_fig)

            layer_attrs = target.get_aggregate_layer_attr()
            layer_fig = Attribution.plot_attr(*Attribution.label_layer_attr(layer_attrs), title="Layer importance")
            layer_fig.savefig(f"{model_dir}layer_attr{suffix}.jpg", bbox_inches="tight")
            plt.close(layer_fig)


def prepare_params(
    params,
    activations=DEFAULT_ACTIVATIONS,
    past_cols=DEFAULT_PAST_COLS,
    future_exo_cols=DEFAULT_FUTURE_EXO_COLS
):
    params["conv_activation"] = activations[params["conv_activation"]]
    params["fc_activation"] = activations[params["fc_activation"]]
    params["residual_activation"] = activations[params["residual_activation"]]
    params["batch_size"] = 16 * (2**params["batch_size"])
    params["past_cols"] = past_cols[params["past_cols"]]
    params["future_exo_cols"] = future_exo_cols[params["future_exo_cols"]]
    return params


def upload_logs(drive, trial_id, log_dir, log_dir_id, model_dir, model_dir_id):
    if log_dir and log_dir_id:
        drive.upload_folder(
            log_dir + str(trial_id),
            parent_id=log_dir_id,
            only_contents=False,
            replace=False
        )
    if model_dir and model_dir_id:
        drive.upload_folder(
            model_dir + str(trial_id),
            parent_id=model_dir_id,
            only_contents=False,
            replace=False
        )


class TrialWrapper:
    def __init__(self, trial):
        self.trial = trial
        self.number = trial.number

    def suggest_int(self, name, param, *args, **kwargs):
        if isinstance(param, tuple) or isinstance(param, list) and param[0] != param[1]:
            return self.trial.suggest_int(name, *param, *args, **kwargs)
        return param

    def suggest_float(self, name, param, *args, **kwargs):
        if isinstance(param, tuple) or isinstance(param, list) and param[0] != param[1]:
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

def make_objective(
    groups,
    log_dir=None,
    model_dir=None,
    drive=None,
    log_dir_id=None,
    model_dir_id=None,
    device="cpu",
    write_graph=False,
    early_stopping_interval_mode=2,
    max_epoch=100,
    teacher_forcing=True,
    activations=DEFAULT_ACTIVATIONS,
    hidden_sizes=(3, 50),
    normal_conv_depths=(1, 20),
    pre_conv_depths=(0, 5),
    normal_fc_depths=(1, 20),
    pre_fc_depths=(0, 5),
    past_kernel_sizes=(3, 14),
    past_strides=(1, 7),
    past_dilations=(1, 7),
    future_kernel_sizes=(3, 7),
    future_strides=(1, 7),
    future_dilations=(1, 7),
    w0_means=(0.0, 1.0),
    w0_stds=(0.0, 0.5),
    booleans=(0, 1),
    lrs=(1e-5, 1e-2),
    source_weights=(0.5, 1.0),
    batch_sizes=(0, 5),
    additional_past_lengths=(0, 4),
    seed_lengths=30,
    min_epochs=50,
    past_cols=DEFAULT_PAST_COLS,
    future_exo_cols=DEFAULT_FUTURE_EXO_COLS,
    pretrain_upload=False,
    use_representation_past=True,
    use_representation_future=False,
    use_shared=True,
    joint_learning=True
):
    activation_keys = [x for x in activations.keys()]
    if not use_representation_future:
        seed_lengths = 1

    @LINE_PROFILER
    def objective(
        trial
    ):
        trial = TrialWrapper(trial)
        ModelUtil.global_random_seed()

        params = {
            "private_state_size": trial.suggest_int("private_state_size", hidden_sizes),
            "lr": trial.suggest_float("lr", lrs),
            "batch_size": trial.suggest_int("batch_size", batch_sizes),
            "additional_past_length": trial.suggest_int("additional_past_length", additional_past_lengths),
            "seed_length": trial.suggest_int("seed_length", seed_lengths),
            "use_last_past": trial.suggest_int("use_last_past", booleans),
            "past_cols": trial.suggest_int("past_cols", (0, len(past_cols) - 1)),
            "future_exo_cols": trial.suggest_int("future_exo_cols", (0, len(future_exo_cols) - 1)),
            "teacher_forcing": trial.suggest_categorical("teacher_forcing", teacher_forcing)
        }

        if joint_learning:
            params.update({
                "source_weight": trial.suggest_float("source_weight", source_weights)
            })

        if use_representation_past or use_representation_future or use_shared:
            params.update({
                "residual_activation": trial.suggest_categorical("residual_activation", activation_keys)
            })

        if use_representation_past or use_representation_future:
            params.update({
                "conv_activation": trial.suggest_categorical("conv_activation", activation_keys)
            })

        if use_shared:
            params.update({
                "fc_activation": trial.suggest_categorical("fc_activation", activation_keys),
                "shared_state_size": trial.suggest_int("shared_state_size", hidden_sizes),
                "combine_head_w0_mean": trial.suggest_float("combine_head_w0_mean", w0_means),
                "combine_head_w0_std": trial.suggest_float("combine_head_w0_std", w0_stds),
                "precombine_head_depth": trial.suggest_int("precombine_head_depth", pre_fc_depths),
                "combine_head_depth": trial.suggest_int("combine_head_depth", normal_fc_depths)
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

        sum_val_loss_target = 0

        for group in groups:
            for cluster in group.clusters:

                if group.id > 0 or cluster.id > 1:
                    continue

                print(f"Model for {group.id}.{cluster.id}")

                grad_scaler = None  # GradScaler(init_scale=8192)

                model = ObjectiveModel(
                    cluster,
                    trial_id=trial.number,
                    log_dir=log_dir,  # "%s/%s/%s" % (log_dir, group.id, cluster.id),
                    model_dir=model_dir,
                    grad_scaler=grad_scaler,
                    # teacher_forcing=True,
                    min_epochs=min_epochs,
                    use_shared=use_shared,
                    **params
                )
                model.to(device)

                if write_graph and group.id == 0 and cluster.id <= 0:
                    model.write_graph()

                if model_dir:
                    model.pretrain_save_model()

                if drive and pretrain_upload:
                    upload_logs(drive, model.trial_id, log_dir, log_dir_id, model_dir, model_dir_id)

                early_stopping = EarlyStopping(
                    model.model.models,
                    debug=1,
                    log_dir=model.log_dir,
                    label=model.label,
                    interval_mode=early_stopping_interval_mode,
                    max_epoch=max_epoch
                )

                while not early_stopping.stopped:
                    train_loss, train_loss_target, train_loss_targets = model.train()
                    train_loss, train_loss_target = train_loss.item(), train_loss_target.item()
                    val_loss, val_loss_target, val_loss_targets = model.val()
                    val_loss, val_loss_target = val_loss.item(), val_loss_target.item()
                    early_stopping(train_loss_target, val_loss_target)

                sum_val_loss_target += val_loss_target

                if model_dir:
                    model.posttrain_save_model()

                if drive:
                    upload_logs(drive, model.trial_id, log_dir, log_dir_id, model_dir, model_dir_id)

                torch.cuda.empty_cache()
                gc.collect()

        return sum_val_loss_target

    return objective
