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

        self.w0 = nn.Parameter(ModelUtil.learnable_normal(
            (private_size,),
            mean=w0_mean,
            std=w0_std
        ))
        self.w0.data = self.w0.data.clamp_(0, 1.0)

    def forward(self, x):
        x_private, x_shared = x
        x_private = self.w0 * x_private
        ret = torch.cat([x_private, x_shared], x_private.dim()-1)
        return ret


class CombineHead(nn.Module):
    def __init__(
        self,
        private_size,
        shared_size=0,
        output_size=3,
        combiner_kwargs={},
        precombine_kwargs={},
        reducer_kwargs={}
    ):
        super(CombineHead, self).__init__()

        use_shared_head = False;
        if precombine_kwargs is not None\
            or shared_size:
            assert precombine_kwargs is not None\
                and shared_size
            use_shared_head = True
        self.use_shared_head = use_shared_head

        if self.use_shared_head:
            self.precombine = ResidualFC(
                input_size=shared_size,
                output_size=shared_size,
                **precombine_kwargs
            )
            self.combiner = CombineRepresentation(private_size, **combiner_kwargs)
        else:
            self.precombine = None
            self.combiner = None

        self.reducer = ResidualFC(
            input_size=private_size+shared_size,
            output_size=output_size,
            **reducer_kwargs
        )

    def forward(self, x_private, x_shared=None):
        if self.use_shared_head:
            x_shared = self.precombine(x_shared)
            x = self.combiner((x_private, x_shared))
        else:
            x = x_private
        x = self.reducer(x)
        return x

    def freeze_shared(self, freeze=True):
        pass

    def freeze_private(self, freeze=True):
        self.requires_grad_(not freeze)
