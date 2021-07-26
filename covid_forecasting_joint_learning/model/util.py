import torch
import json
from torch import nn


def learnable_normal(size, mean, std):
    t = torch.zeros(*size, dtype=torch.float32, requires_grad=True)
    nn.init.normal_(t, mean, std)
    return t


def learnable_xavier(size):
    t = torch.zeros(1, *size, dtype=torch.float32, requires_grad=True)
    nn.init.xavier_normal_(t)
    return t[0].detach()


def repeat_batch(t, batch_size):
    # t = batch_size * [t]
    # t = torch.stack(t)
    one = torch.ones(batch_size, *t.size())
    t = one * t
    return t


# Linear tensor uses input of shape (Batch, Length, Channel)
# RNN uses input of shape (Length, Batch, Channel)
def linear_to_sequential_tensor(t):
    return t.permute(1, 0, 2)


def sequential_to_linear_tensor(t):
    return t.permute(1, 0, 2)


# Conv1d uses input of shape (Batch, Channel, Length)
def linear_to_conv1d_tensor(t):
    return t.permute(0, 2, 1)


def conv1d_to_linear_tensor(t):
    return t.permute(0, 2, 1)


def str_dict(d):
    return json.dumps(
        d,
        sort_keys=True,
        indent=4,
        default=str
    )
