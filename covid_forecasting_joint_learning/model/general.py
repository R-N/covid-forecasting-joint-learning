import torch
from torch import nn
from .modules.main import SingleModel
from .train import train, test


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
        loss_fn=nn.MSELoss(),
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
            loss_fn=self.loss_fn,
            key=lambda k: k.dataloaders[0],
            **self.train_kwargs
        )

    def val(self):
        return train(
            self.sources,
            self.target,
            loss_fn=self.loss_fn,
            key=lambda k: k.dataloaders[1],
            **self.train_kwargs
        )

    def test(self):
        return train(
            self.sources,
            self.target,
            loss_fn=self.loss_fn,
            key=lambda k: k.dataloaders[2],
            **self.train_kwargs
        )
