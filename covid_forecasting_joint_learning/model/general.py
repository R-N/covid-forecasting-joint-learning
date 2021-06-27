import torch
from torch import nn
from .modules.main import SingleModel
from .train import train, test
from ..pipeline.main import preprocessing_5, preprocessing_6


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
                "representation_past_model": {
                    "private_representation": {},
                    "pre_shared_representation": {},
                    "shared_representation": {},
                    "combine_representation": {}
                },
                "private_head_past": {},
                "shared_head_past": {}
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
        optimizer_fn=torch.optim.Adam,
        optimizer_kwargs={},
        train_kwargs={}
    ):
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
                model_kwargs["past_model"]["representation_past_model"]["shared_representation"] =\
                    self.shared_model.past_model.representation_past_model.shared_representation
            except KeyError:
                pass
            try:
                model_kwargs["past_model"]["shared_head_past"] =\
                    self.shared_model.past_model.shared_head_past
            except KeyError:
                pass
            try:
                model_kwargs["representation_future_model"]["shared_representation"] =\
                    self.shared_model.representation_future_model.shared_representation
            except KeyError:
                pass
            try:
                model_kwargs["shared_head_future_cell"] =\
                    self.shared_model.shared_head_future_cell
            except KeyError:
                pass

        if self.private_mode == SharedMode.SHARED:
            try:
                model_kwargs["past_model"]["representation_past_model"]["private_representation"] =\
                    self.shared_model.past_model.representation_past_model.private_representation
            except KeyError:
                pass
            try:
                model_kwargs["past_model"]["representation_past_model"]["pre_shared_representation"] =\
                    self.shared_model.past_model.representation_past_model.pre_shared_representation
            except KeyError:
                pass
            try:
                model_kwargs["past_model"]["representation_past_model"]["combine_representation"] =\
                    self.shared_model.past_model.representation_past_model.combine_representation
            except KeyError:
                pass

            try:
                model_kwargs["past_model"]["private_head_past"] =\
                    self.shared_model.past_model.private_head_past
            except KeyError:
                pass

            try:
                model_kwargs["representation_future_model"]["private_representation"] =\
                    self.shared_model.representation_future_model.private_representation
            except KeyError:
                pass
            try:
                model_kwargs["representation_future_model"]["pre_shared_representation"] =\
                    self.shared_model.representation_future_model.pre_shared_representation
            except KeyError:
                pass
            try:
                model_kwargs["representation_future_model"]["combine_representation"] =\
                    self.shared_model.representation_future_model.combine_representation
            except KeyError:
                pass

            try:
                model_kwargs["private_head_future_cell"] =\
                    self.shared_model.private_head_future_cell
            except KeyError:
                pass
            try:
                model_kwargs["post_future_model"] =\
                    self.shared_model.post_future_model
            except KeyError:
                pass

        for k in self.members:
            k.model = SingleModel(**sizes, **model_kwargs)

        self.models = nn.ModuleList([k.model for k in self.members])
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def members(self):
        return [*self.sources, self.target]

    def train(self):
        optimizer = self.optimizer_fn(self.models.parameters(), **self.optimizer_kwargs)
        return train(
            self.sources,
            self.target,
            optimizer,
            key=lambda k: k.dataloaders[0],
            **self.train_kwargs
        )

    def val(self):
        return train(
            self.sources,
            self.target,
            key=lambda k: k.dataloaders[1],
            **self.train_kwargs
        )

    def test(self):
        return train(
            self.sources,
            self.target,
            key=lambda k: k.dataloaders[2],
            **self.train_kwargs
        )

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
        source_pick=SourcePick.ALL,
        private_mode=SharedMode.PRIVATE,
        shared_mode=SharedMode.SHARED,
        optimizer_fn=torch.optim.Adam,
        optimizer_lr=1e-5,
        loss_fn=nn.MSELoss(),
        source_weight=1.0,
        cuda=True
    ):
        self.cluster = cluster

        past_length = 30 + additional_past_length
        future_length = 14

        self.device = "cpu"
        if device == "cuda" and torch.cuda.is_available():
            self.device = device

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

        sample = cluster.target.data.datasets[0][0]
        input_size_past = sample["past"].size(1)
        input_size_future = sample["future"].size(1)
        if use_exo_cols:
            input_size_past += sample["past_exo"].size(1)
            input_size_future += sample["future_exo"].size(1)

        self.model = ClusterModel(
            cluster,
            sizes={
                "input_size_past": input_size_past,
                "hidden_size_past": hidden_size_past,
                "input_size_future": input_size_future,
                "hidden_size_future": hidden_size_future,
                "private_state_size": private_state_size,
                "shared_state_size": shared_state_size,
                "output_size": 3,
            },
            model_kwargs={
                "past_model": {
                    "representation_past_model": {
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
                    "private_head_past": {
                        "use_last_past": use_last_past
                    },
                    "shared_head_past": {
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
                    "combiner_kwargs": {
                        "w0_mean": combine_head_w0_mean,
                        "w0_std": combine_head_w0_std
                    } if representation_future_shared_depth else None,
                    "precombine_kwargs": {
                        "depth": precombine_head_depth,
                        "fc_activation": fc_activation,
                        "residual_activation": residual_activation
                    } if representation_future_shared_depth else None,
                    "reducer_kwargs": {
                        "depth": combine_head_depth,
                        "fc_activation": fc_activation,
                        "residual_activation": residual_activation
                    }
                },
            },
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

    def train(self):
        return self.model.train()

    def val(self):
        return self.model.val()

    def test(self):
        return self.model.test()
