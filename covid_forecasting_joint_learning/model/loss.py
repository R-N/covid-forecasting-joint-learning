import torch
from torch import nn
from optuna.structs import TrialPruned

class NaNPredException(TrialPruned):
    def __init__(self):
        super(NaNPredException, self).__init__("Pred has NaN!")

def mse(err):
    return torch.mean(torch.square(err), dim=-2)

def naive(past, step=1, limit=None):
    if past.dim() == 3:
        if limit:
            past = past[:, :limit]
        return (past[:, :-step] - past[:, step:]).detach()
    elif past.dim() < 3:
        if limit:
            past = past[:limit]
        return (past[:-step] - past[step:]).detach()
    else:
        raise Exception(f"Invalid input dim {past.dim()}")

def msse(past, future, pred, limit_naive=30):
    if torch.isnan(pred).any():
        # raise NaNPredException()
        raise Exception("Pred is Nan!")
    div = mse(naive(past, limit=limit_naive)).detach()
    print(div)
    return mse(pred - future) / div

def rmsse(past, future, pred, limit_naive=30):
    return torch.sqrt(msse(past, future, pred, limit_naive=limit_naive))

def reduce(loss, reduction="sum"):
    while loss.dim() > 1:
        loss = torch.sum(loss, dim=-1)
    if reduction in ("mean", "avg"):
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction {reduction}")

class MSSELoss(nn.Module):
    def __init__(self, reduction="sum", limit_naive=30):
        super().__init__()
        self.reduction = reduction
        self.limit_naive = limit_naive

    def forward(self, past, future, pred):
        return reduce(msse(past, future, pred, limit_naive=self.limit_naive), reduction=self.reduction)

class RMSSELoss(nn.Module):
    def __init__(self, reduction="sum", limit_naive=30):
        super().__init__()
        self.reduction = reduction
        self.limit_naive = limit_naive

    def forward(self, past, future, pred):
        return reduce(rmsse(past, future, pred, limit_naive=self.limit_naive), reduction=self.reduction)
