import torch
from torch import nn
from .modules.representation import check_conv_kwargs
from .modules.main import SingleModel
from .train import train, test
from ..pipeline.main import preprocessing_5, preprocessing_6
from .util import str_dict
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam


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
        optimizer_fn=Adam,
        lr=1e-5,
        optimizer_kwargs={},
        train_kwargs={}
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
            try:
                model_kwargs["past_model"]["representation_model"]["shared_representation"] =\
                    self.shared_model.past_model.representation_model.shared_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["past_model"]["shared_head"] =\
                    self.shared_model.past_model.shared_head
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["representation_future_model"]["shared_representation"] =\
                    self.shared_model.representation_future_model.shared_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["shared_head_future_cell"] =\
                    self.shared_model.shared_head_future_cell
            except (KeyError, TypeError):
                pass

        if self.private_mode == SharedMode.SHARED:
            try:
                model_kwargs["past_model"]["representation_model"]["private_representation"] =\
                    self.shared_model.past_model.representation_model.private_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["past_model"]["representation_model"]["pre_shared_representation"] =\
                    self.shared_model.past_model.representation_model.pre_shared_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["past_model"]["representation_model"]["combine_representation"] =\
                    self.shared_model.past_model.representation_model.combine_representation
            except (KeyError, TypeError):
                pass

            try:
                model_kwargs["past_model"]["private_head"] =\
                    self.shared_model.past_model.private_head
            except (KeyError, TypeError):
                pass

            try:
                model_kwargs["representation_future_model"]["private_representation"] =\
                    self.shared_model.representation_future_model.private_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["representation_future_model"]["pre_shared_representation"] =\
                    self.shared_model.representation_future_model.pre_shared_representation
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["representation_future_model"]["combine_representation"] =\
                    self.shared_model.representation_future_model.combine_representation
            except (KeyError, TypeError):
                pass

            try:
                model_kwargs["private_head_future_cell"] =\
                    self.shared_model.private_head_future_cell
            except (KeyError, TypeError):
                pass
            try:
                model_kwargs["post_future_model"] =\
                    self.shared_model.post_future_model
            except (KeyError, TypeError):
                pass

        for k in self.members:
            k.model = SingleModel(**sizes, **model_kwargs)

        self.models = nn.ModuleList([k.model for k in self.members])
        self.optimizer_fn = optimizer_fn
        self.lr = lr
        optimizer_kwargs["lr"] = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.train_kwargs = train_kwargs

        self.optimizer = self.create_optimizer()
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr)

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

    def train(self):
        # optimizer = self.create_optimizer()
        return train(
            self.sources,
            self.target,
            self.optimizer,
            self.scheduler,
            key=lambda k: k.dataloaders[0],
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
        optimizer_lr=1e-5,
        loss_fn=nn.MSELoss(),
        source_weight=1.0,
        trial_id=None,
        log_dir=None,
        debug=False
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
                        "conv_kwargs":{
                            "kernel_size": representation_past_private_kernel_size,
                            "stride": representation_past_private_stride,
                            "dilation": representation_past_private_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs":{
                            "activation": residual_activation
                        }
                    },
                    "pre_shared_representation": {
                        "depth": representation_past_pre_shared_depth,
                        "data_length": past_length,
                        "conv_kwargs":{
                            "kernel_size": representation_past_shared_kernel_size,
                            "stride": representation_past_shared_stride,
                            "dilation": representation_past_shared_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs":{
                            "activation": residual_activation
                        }
                    } if (representation_past_shared_depth and representation_past_pre_shared_depth) else None,
                    "shared_representation": {
                        "depth": representation_past_shared_depth,
                        "data_length": past_length,
                        "conv_kwargs":{
                            "kernel_size": representation_past_shared_kernel_size,
                            "stride": representation_past_shared_stride,
                            "dilation": representation_past_shared_dilation,
                            "activation": conv_activation
                        },
                        "residual_kwargs":{
                            "activation": residual_activation
                        }
                    } if representation_past_shared_depth else None,
                    "combine_representation": {
                        "w0_mean": combine_representation_past_w0_mean,
                        "w0_std": combine_representation_past_w0_std,
                    } if representation_past_shared_depth else None
                } ,
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
                    "conv_kwargs":{
                        "kernel_size": representation_future_private_kernel_size,
                        "stride": representation_future_private_stride,
                        "dilation": representation_future_private_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs":{
                        "activation": residual_activation
                    }
                },
                "pre_shared_representation": {
                    "depth": representation_future_pre_shared_depth,
                    "data_length": future_length,
                    "conv_kwargs":{
                        "kernel_size": representation_future_shared_kernel_size,
                        "stride": representation_future_shared_stride,
                        "dilation": representation_future_shared_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs":{
                        "activation": residual_activation
                    }
                } if (representation_future_shared_depth and representation_future_pre_shared_depth) else None,
                "shared_representation": {
                    "depth": representation_future_shared_depth,
                    "data_length": future_length,
                    "conv_kwargs":{
                        "kernel_size": representation_future_shared_kernel_size,
                        "stride": representation_future_shared_stride,
                        "dilation": representation_future_shared_dilation,
                        "activation": conv_activation
                    },
                    "residual_kwargs":{
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
            "seed_length": seed_length
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
        input_size_past = sample["past"].shape[-1]
        input_size_future = sample["future"].shape[-1]

        use_exo_cols = future_exo_cols is not None and len(future_exo_cols) > 0
        if use_exo_cols:
            # input_size_past += sample["past_exo"].shape[-1]
            input_size_future += sample["future_exo"].shape[-1]
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

        if debug:
            print(str_dict(sizes))
            print(str_dict(model_kwargs))

        self.model = ClusterModel(
            cluster,
            sizes=sizes,
            model_kwargs=model_kwargs,
            source_pick=source_pick,
            private_mode=private_mode,
            shared_mode=shared_mode,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs={
                "lr": optimizer_lr
            },
            train_kwargs={
                "loss_fn": loss_fn,
                "source_weight": source_weight
            }
        )

        if debug:
            print(self.get_target_model_summary())

        self.trial_id = trial_id if trial_id is not None else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if isinstance(log_dir, str) and not log_dir.endswith("/"):
            log_dir = log_dir + "/"
        self.log_dir = log_dir

        if self.log_dir:
            trial_log_dir = self.log_dir + str(self.trial_id)
            train_log_dir = trial_log_dir + '/train'
            val_log_dir = trial_log_dir + '/val'

            self.train_summary_writer = SummaryWriter(train_log_dir)
            self.val_summary_writer = SummaryWriter(val_log_dir)


        self.train_epoch = 0
        self.val_epoch = 0

    def train(self, epoch=None):
        loss = self.model.train()
        epoch = epoch if epoch is not None else self.train_epoch
        if self.log_dir:
            self.train_summary_writer.add_scalar(f"{self.cluster.group.id}.{self.cluster.id}/avg_loss", loss[0].item(), global_step=epoch)
            self.train_summary_writer.add_scalar(f"{self.cluster.group.id}.{self.cluster.id}/target_loss", loss[1].item(), global_step=epoch)
            self.train_summary_writer.flush()
        self.train_epoch = epoch + 1
        return loss

    def val(self, epoch=None):
        loss = self.model.val()
        epoch = epoch if epoch is not None else self.val_epoch
        if self.log_dir:
            self.val_summary_writer.add_scalar(f"{self.cluster.group.id}.{self.cluster.id}/avg_loss", loss[0].item(), global_step=epoch)
            self.val_summary_writer.add_scalar(f"{self.cluster.group.id}.{self.cluster.id}/target_loss", loss[1].item(), global_step=epoch)
            self.val_summary_writer.flush()
        self.val_epoch = epoch + 1
        return loss

    def test(self):
        return self.model.test()

    def get_target_model_summary(self):
        return self.model.get_target_model_summary()

    def write_graph(self, path):
        self.model.write_graph(path)

    def freeze_shared(self, freeze=True):
        self.model.freeze_shared(freeze)

    def freeze_private(self, freeze=True):
        self.model.freeze_private(freeze)
