import torch
from torch import nn

def mse(err):
    if torch.isnan(err).any():
        print(err)
        raise("NAN")
    ret = torch.mean((err)**2, dim=-2)
    if torch.isnan(ret).any():
        print(ret)
        raise("NAN")
    return ret

def naive(past, step=1):
    ret = past[:, :-step] - past[:, step:]
    if torch.isnan(ret).any():
        print(ret)
        raise("NAN")
    return ret

def msse(past, future, pred):
    ret = mse(pred - future) / mse(naive(past))
    if torch.isnan(ret).any():
        print(ret)
        raise("NAN")
    return ret

def rmsse(past, future, pred):
    return torch.sqrt(msse(past, future, pred))

def reduce(loss, reduction="sum"):
    while loss.dim() > 1:
        loss = torch.sum(loss, dim=-1)
    if reduction in ("mean", "avg"):
        ret = torch.mean(loss)
        if torch.isnan(ret).any():
            print(ret)
            raise("NAN")
        return ret
    elif reduction == "sum":
        ret = torch.sum(loss)
        if torch.isnan(ret).any():
            print(ret)
            raise("NAN")
        return ret
    else:
        raise ValueError(f"Invalid reduction {reduction}")

class MSSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, past, future, pred):
        ret = reduce(msse(past, future, pred), reduction=self.reduction)
        if torch.isnan(ret).any():
            print(ret)
            raise("NAN")
        return ret

class RMSSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, past, future, pred):
        return reduce(rmsse(past, future, pred), reduction=self.reduction)
