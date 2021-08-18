import torch
from torch import nn
from .modules.representation import check_conv_kwargs
from .modules.main import SingleModel
from .train import train, test
from ..pipeline.main import preprocessing_5, preprocessing_6
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from contextlib import suppress
from pathlib import Path
from . import attr as Attribution
from . import util as ModelUtil
from ..data import util as DataUtil


class SourcePick:
    ALL = 0
    CLOSEST = 1
    LONGEST = 2


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
        min_epochs=50
    ):
        self.cluster = cluster
        if source_pick == SourcePick.ALL:
            self.sources = cluster.sources
        elif source_pick == SourcePick.CLOSEST:
            self.sources = [cluster.source_closest]
        elif source_pick == SourcePick.LONGEST:
            self.sources = [cluster.source_longest]
        else:
            raise Exception("Invalid source_pick: %s" % (source_pick,))

        self.target = cluster.target

        self.source_pick = source_pick
        self.private_mode = private_mode
        self.shared_mode = shared_mode

        self.shared_model = SingleModel(**sizes, **model_kwargs)

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
            self.target,
            self.optimizer,
            self.scheduler,
            key=lambda k: k.dataloaders[0],
            clip_grad_norm=self.clip_grad_norm,
            grad_scaler=grad_scaler,
            **self.train_kwargs
        )

    def val(self):
        return test(
            self.sources,
            self.target,
            key=lambda k: k.dataloaders[1],
            **self.train_kwargs
        )

    def test(self):
        return test(
            self.sources,
            self.target,
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


class ObjectiveModel:
    def __init__(
        self,
        cluster,
        hidden_size_past,
        hidden_size_future,
        shared_state_size,
        private_state_size,
        representation_past_private_depth,
        representation_past_private_kernel_size,
        representation_past_private_stride,
        representation_past_private_dilation,
        representation_past_shared_depth,
        representation_past_shared_kernel_size,
        representation_past_shared_stride,
        representation_past_shared_dilation,
        representation_past_pre_shared_depth,
        combine_representation_past_w0_mean,
        combine_representation_past_w0_std,
        representation_future_private_depth,
        representation_future_private_kernel_size,
        representation_future_private_stride,
        representation_future_private_dilation,
        representation_future_shared_depth,
        representation_future_shared_kernel_size,
        representation_future_shared_stride,
        representation_future_shared_dilation,
        representation_future_pre_shared_depth,
        combine_representation_future_w0_mean,
        combine_representation_future_w0_std,
        combine_head_w0_mean,
        combine_head_w0_std,
        precombine_head_depth,
        combine_head_depth,
        conv_activation,
        fc_activation,
        residual_activation,
        past_cols,
        future_exo_cols,
        batch_size,
        additional_past_length,
        use_last_past,
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
        min_epochs=50
    ):
        self.cluster = cluster

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
                },
                "private_head": {
                    "use_last_past": use_last_past
                },
                "shared_head": {
                    "use_last_past": use_last_past
                } if representation_past_shared_depth else None
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
            "shared_head_future_cell": {} if representation_future_shared_depth else None,
            "post_future_model": {
                "combiner": {
                    "w0_mean": combine_head_w0_mean,
                    "w0_std": combine_head_w0_std
                } if representation_future_shared_depth else None,
                "precombine": {
                    "depth": precombine_head_depth,
                    "fc_activation": fc_activation,
                    "residual_activation": residual_activation
                } if representation_future_shared_depth else None,
                "reducer": {
                    "depth": combine_head_depth,
                    "fc_activation": fc_activation,
                    "residual_activation": residual_activation
                }
            },
            "seed_length": seed_length,
            "teacher_forcing": teacher_forcing
        }

        try:
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["private_representation"]["conv_kwargs"], past_length)
        except (KeyError, TypeError):
            pass
        try:
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["pre_shared_representation"]["conv_kwargs"], past_length)
        except (KeyError, TypeError):
            pass
        try:
            check_conv_kwargs(model_kwargs["past_model"]["representation_model"]["shared_representation"]["conv_kwargs"], past_length)
        except (KeyError, TypeError):
            pass
        try:
            check_conv_kwargs(model_kwargs["representation_future_model"]["private_representation"]["conv_kwargs"], future_length)
        except (KeyError, TypeError):
            pass
        try:
            check_conv_kwargs(model_kwargs["representation_future_model"]["pre_shared_representation"]["conv_kwargs"], future_length)
        except (KeyError, TypeError):
            pass
        try:
            check_conv_kwargs(model_kwargs["representation_future_model"]["shared_representation"]["conv_kwargs"], future_length)
        except (KeyError, TypeError):
            pass

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
        self.model_kwargs = model_kwargs

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
            }
        )

        if debug:
            print(self.get_target_model_summary())

        self.trial_id = trial_id if trial_id is not None else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if isinstance(log_dir, str):
            if not log_dir.endswith("/"):
                log_dir = log_dir + "/"
            log_dir = log_dir + str(self.trial_id)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        if self.log_dir:
            self.train_summary_writer = SummaryWriter(self.log_dir + '/train')
            self.val_summary_writer = SummaryWriter(self.log_dir + '/val')

        if isinstance(model_dir, str):
            if not model_dir.endswith("/"):
                model_dir = model_dir + "/"
            model_dir = model_dir + f"{self.trial_id}/{self.cluster.group.id}/{self.cluster.id}/"
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir

        self.train_epoch = 0
        self.val_epoch = 0

        self.label = f"G{self.cluster.group.id}.C{self.cluster.id}/"

    def train(self, epoch=None):
        loss = self.model.train()
        epoch = epoch if epoch is not None else self.train_epoch
        if self.log_dir:
            self.train_summary_writer.add_scalar(self.label + "avg_loss", loss[0].item(), global_step=epoch)
            self.train_summary_writer.add_scalar(self.label + "target_loss", loss[1].item(), global_step=epoch)
            self.train_summary_writer.flush()
        self.train_epoch = epoch + 1
        return loss

    def val(self, epoch=None):
        loss = self.model.val()
        epoch = epoch if epoch is not None else self.val_epoch
        if self.log_dir:
            self.val_summary_writer.add_scalar(self.label + "avg_loss", loss[0].item(), global_step=epoch)
            self.val_summary_writer.add_scalar(self.label + "target_loss", loss[1].item(), global_step=epoch)
            self.val_summary_writer.flush()
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

    def save_model(self, model_dir=None):
        model_dir = model_dir or self.model_dir
        if not model_dir:
            raise ValueError("Please provide or set model_dir")

        torch.save(self.model.models.state_dict(), model_dir + "models.pt")
        torch.save(self.model.target.model.state_dict(), model_dir + "target.pt")

        input_attr = self.model.target.get_input_attr()
        input_fig = Attribution.plot_attr(*Attribution.label_input_attr(input_attr, self.model.target.dataset_labels))
        input_fig.savefig(model_dir + "input_attr.jpg", bbox_inches = "tight")

        layer_attrs = self.model.target.get_aggregate_layer_attr()
        layer_fig = Attribution.plot_attr(*Attribution.label_layer_attr(layer_attrs))
        layer_fig.savefig(model_dir + "layer_attr.jpg", bbox_inches = "tight")

        DataUtil.write_string(str(self.get_target_model_summary()), model_dir + "target_model_summary.txt")
        DataUtil.write_string(ModelUtil.str_dict(self.sizes), model_dir + "sizes.json")
        DataUtil.write_string(ModelUtil.str_dict(self.model_kwargs), model_dir + "model_kwargs.json")
