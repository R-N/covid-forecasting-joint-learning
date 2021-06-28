import torch
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


def to_sequential_tensor(t):
    return t.permute(1, 0, 2)


def to_batch_tensor(t):
    return t.permute(1, 0, 2)