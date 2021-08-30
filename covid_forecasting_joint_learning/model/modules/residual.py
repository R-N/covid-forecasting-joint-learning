import torch
from torch import nn
from ..util import LINE_PROFILER


class ResidualBlock(nn.Module):
    DEFAULT_KWARGS = {
        "highway": False,
        "activation": None
    }
    def __init__(
        self,
        main_block,
        size,
        highway=False,
        activation=None
    ):
        super(ResidualBlock, self).__init__()
        self.w = nn.Parameter(torch.ones(
            1,  # size,
            requires_grad=True
        ))
        self.highway = highway
        if highway:
            self.register_backward_hook(self.constraint_w)
        self.main_block = main_block
        activation = activation or nn.Identity
        self.activation = activation()
        self.residual = nn.Identity()

    @LINE_PROFILER
    def forward(self, x):
        residual = self.residual(x)
        if self.highway:
            self.constraint_w()
            residual = (1-self.w) * residual + self.w * self.main_block(x)
        else:
            residual = residual + self.w * self.main_block(x)
        residual = self.activation(residual)
        return residual

    def constraint_w(self):
        self.w.data = self.w.data.clamp_(0, 1.0)


def try_residual(
    block,
    input_size, output_size,
    highway=False,
    activation=None
):
    if input_size == output_size:
        return ResidualBlock(
            block,
            input_size,
            highway=highway,
            activation=activation
        )
    return block


class ResidualStack(nn.Module):
    DEFAULT_KWARGS = {
        "hidden_size": None,
        "depth": 1,
        "highway": False,
        "activation": None
    }

    def __init__(
        self,
        block_f,
        input_size,
        output_size,
        hidden_size=None,
        depth=1,
        highway=False,
        activation=None
    ):
        super(ResidualStack, self).__init__()
        assert depth >= 0
        hidden_size = hidden_size or max(input_size, output_size)
        if depth == 0:
            self.main = nn.Identity()
        elif depth == 1:
            self.main = try_residual(
                block_f(input_size, output_size),
                input_size, output_size,
                highway=highway,
                activation=activation
            )
        else:
            hidden_blocks = [
                ResidualBlock(
                    block_f(hidden_size, hidden_size),
                    hidden_size,
                    highway=highway,
                    activation=activation
                ) for x in range(depth - 2)
            ]
            block_1 = try_residual(
                block_f(input_size, hidden_size),
                input_size, hidden_size,
                highway=highway,
                activation=activation
            )
            block_d = try_residual(
                block_f(hidden_size, output_size),
                hidden_size, output_size,
                highway=highway,
                activation=activation
            )

            self.main = nn.Sequential(
                block_1,
                *hidden_blocks,
                block_d
            )

    def forward(self, x):
        return self.main(x)

class ResidualFC(nn.Module):
    DEFAULT_KWARGS = {
        "hidden_size": None,
        "depth": 1,
        "highway": False,
        "residual_activation": None
    }

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=None,
        depth=1,
        highway=False,
        residual_activation=None,
        fc_activation=None
    ):
        super(ResidualFC, self).__init__()
        fc_activation = fc_activation or nn.Identity

        def block_f(input_size, output_size):
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                fc_activation()
            )

        self.main = ResidualStack(
            block_f=block_f,
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            depth=depth,
            highway=highway,
            activation=residual_activation
        )

    def forward(self, x):
        return self.main(x)
