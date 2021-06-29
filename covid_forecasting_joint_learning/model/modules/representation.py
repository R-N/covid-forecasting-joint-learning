import math
import torch
from torch import nn
from .residual import ResidualStack
from optuna.structs import TrialPruned


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

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size

        self.main = nn.Sequential(
            nn.Conv1d(
                input_size,
                output_size,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation
                # padding=(padding, 0)  # For left padding
            ),
            nn.ConstantPad1d((padding, 0), 0),
            activation()
        )
    
    def forward(self, x):
        try:
            return self.main(x)
        except RuntimeError as ex:
            print(self.input_size, self.output_size, self.kernel_size)
            raise


def conv_kwargs_default(conv_kwargs):
    conv_kwargs = {**RepresentationSingle.DEFAULT_KWARGS, **conv_kwargs}
    return conv_kwargs


def conv_output_length(kernel_size, dilation, stride, data_length):
    # dilated_kernel_size = kernel_size + (kernel_size-1) * (dilation - 1)
    # output_length = math.ceil((data_length - (dilated_kernel_size - 1)) / stride)
    dilated_kernel_size = dilation * (kernel_size - 1) + 1
    output_length = (data_length - dilated_kernel_size) // stride + 1
    try:
        assert output_length >= dilated_kernel_size
    except AssertionError:
        raise TrialPruned("output_length can't be smaller than dilated_kernel_size: (%s, %s, %s, %s, %s, %s)" % (kernel_size, dilation, stride, data_length, dilated_kernel_size, output_length))
    return output_length


def check_conv_kwargs(conv_kwargs, data_length):
    conv_kwargs = conv_kwargs_default(conv_kwargs)
    kernel_size, dilation, stride = [conv_kwargs[x] for x in ("kernel_size", "dilation", "stride")]
    output_length = conv_output_length(kernel_size, dilation, stride, data_length)
    return output_length


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

        output_length = check_conv_kwargs(conv_kwargs, data_length)
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
