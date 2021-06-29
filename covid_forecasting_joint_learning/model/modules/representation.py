import math
import torch
from torch import nn
from .residual import ResidualStack


class RepresentationSingle(nn.Module):
    DEFAULT_KWARGS = {
        "kernel_size": 7,
        "stride": 0,
        "dilation": 0,
        "padding": 0,
        "activation": None
    }

    def __init__(
            self,
            input_size,
            output_size,
            kernel_size=7,
            stride=0,
            dilation=0,
            padding=0,
            activation=None
    ):
        super(RepresentationSingle, self).__init__()

        activation = activation or nn.Identity

        self.main = nn.Sequential(
            nn.Conv1d(
                input_size,
                output_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(padding, 0)  # For left padding
            ),
            activation()
        )
    
    def forward(self, x):
        return self.main(x)


class RepresentationBlock(nn.Module):
    DEFAULT_KWARGS = {
        **RepresentationSingle.DEFAULT_KWARGS,
    }

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=None,
            data_length=30,
            depth=1,
            conv_kwargs={},
            residual_kwargs={}
    ):
        super(RepresentationBlock, self).__init__()

        conv_kwargs = {**RepresentationSingle.DEFAULT_KWARGS, **conv_kwargs}

        kernel_size, dilation, stride = [conv_kwargs[x] for x in ("kernel_size", "dilation", "stride")]

        dilated_kernel_size = kernel_size + (kernel_size-1) * (dilation - 1)
        output_length = math.ceil((data_length - (dilated_kernel_size - 1)) / stride)
        try:
            assert output_length >= dilated_kernel_size
        except AssertionError:
            raise Exception("output_length can't be smaller than dilated_kernel_size: (%s, %s, %s, %s, %s, %s)" % (kernel_size, dilation, stride, data_length, dilated_kernel_size, output_length))
        padding = data_length - output_length
        conv_kwargs["padding"] = padding

        def block_f(input_size, output_size):
            return RepresentationSingle(
                input_size,
                output_size,
                **conv_kwargs
            )

        self.main = ResidualStack(
            block_f=block_f,
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            depth=depth,
            **residual_kwargs
        )
    
    def forward(self, x):
        return self.main(x)
