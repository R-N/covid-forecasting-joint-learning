import torch
from torch import nn
from .. import util as ModelUtil
from .residual import ResidualFC


class CombineRepresentation(nn.Module):
    def __init__(
        self,
        private_size,
        w0_mean=0.1,
        w0_std=0.01
    ):
        super(CombineRepresentation, self).__init__()

        self.w0 = nn.Parameter(ModelUtil.learnable_normal(private_size, w0_mean, w0_std))

    def forward(self, x):
        x_private, x_shared = x
        x_private = self.w0 * x_private
        ret = torch.cat([x_private, x_shared], x_private.dim()-1)
        return ret


class CombineHead(nn.Module):
    def __init__(
        self,
        size,
        w0_mean=0.1,
        w0_std=0.01,
        precombine_kwargs={},
        reducer_kwargs={}
    ):
        super(CombineHead, self).__init__()

        self.precombine = ResidualFC(
            input_size=size,
            output_size=size,
            **precombine_kwargs
        )
        self.combiner = CombineRepresentation(size, w0_mean, w0_std)
        self.reducer = ResidualFC(
            input_size=2*size,
            output_size=size,
            **reducer_kwargs
        )

    def forward(self, x_private, x_shared):
        x_shared = self.precombine(x_shared)
        x = self.combiner((x_private, x_shared))
        x = self.reducer(x)
        return x
