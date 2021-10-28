import torch
from torch import nn
from optuna.structs import TrialPruned
from . import util as ModelUtil

class NaNPredException(TrialPruned):
    def __init__(self):
        super(NaNPredException, self).__init__("Pred has NaN!")

class NaNLossException(TrialPruned):
    def __init__(self):
        super(NaNLossException, self).__init__("Loss is NaN!")

def mse(err):
    # (batch, horizon, feature)
    return torch.mean(torch.square(err), dim=-2)

def naive(past, step=1, limit=None):
    if past.dim() == 3:
        if limit:
            past = past[:, -limit:]
        return (past[:, :-step] - past[:, step:]).detach()
    elif past.dim() < 3:
        if limit:
            past = past[-limit:]
        return (past[:-step] - past[step:]).detach()
    else:
        raise Exception(f"Invalid input dim {past.dim()}")

def msse(past, future, pred, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    if torch.isnan(pred).any():
        raise NaNPredException()
    return mse(pred - future) / (mse(naive(past, limit=limit_naive)) + eps).detach()

def rmsse(past, future, pred, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    return torch.sqrt(msse(past, future, pred, limit_naive=limit_naive, eps=eps))

def reduce(loss, reduction="sum", reduce_feature=True):
    while reduce_feature and loss.dim() > 1:
        loss = torch.sum(loss, dim=-1)
    if reduction in ("mean", "avg"):
        return torch.mean(loss, dim=0)
    elif reduction == "sum":
        return torch.sum(loss, dim=0)
    else:
        raise ValueError(f"Invalid reduction {reduction}")

class MSSELoss(nn.Module):
    def __init__(self, reduction="sum", limit_naive=30, eps=ModelUtil.NAIVE_EPS):
        super().__init__()
        self.reduction = reduction
        self.limit_naive = limit_naive
        self.eps = eps

    def forward(self, past, future, pred):
        return reduce(msse(past, future, pred, limit_naive=self.limit_naive, eps=self.eps), reduction=self.reduction)

class RMSSELoss(nn.Module):
    def __init__(self, reduction="sum", limit_naive=30, eps=ModelUtil.NAIVE_EPS):
        super().__init__()
        self.reduction = reduction
        self.limit_naive = limit_naive
        self.eps = eps

    def forward(self, past, future, pred):
        return reduce(rmsse(past, future, pred, limit_naive=self.limit_naive, eps=self.eps), reduction=self.reduction)
