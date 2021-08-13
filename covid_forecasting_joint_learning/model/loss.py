import torch
from torch import nn

def mse(err):
    return torch.mean((err)**2, dim=1)

def naive(past, step=1):
    return past[:, :-step] - past[:, step:]

def msse(past, future, pred):
    return mse(pred - future) / mse(naive(past))

def rmsse(past, future, pred):
    return torch.sqrt(msse(past, future, pred))

def reduce(loss, reduction="mean"):
    while loss.dim > 1:
        loss = torch.sum(loss, dim=-1)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction {reduction}")

class MSSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, past, future, pred):
        return reduce(msse(past, future, pred), reduction=self.reduction)

class RMSSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, past, future, pred):
        return reduce(rmsse(past, future, pred), reduction=self.reduction)
