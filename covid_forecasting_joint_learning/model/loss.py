import torch
from torch import nn

def mse(err):
    ret = torch.mean(torch.square(err), dim=-2)
    if torch.isnan(ret).any():
        raise Exception("NAN")
    return ret

def naive(past, step=1):
    ret = past[:, :-step] - past[:, step:]
    if torch.isnan(ret).any():
        print(past)
        print(ret)
        raise Exception("NAN")
    return ret

def msse(past, future, pred):
    ret = mse(pred - future)
    ret = ret / mse(naive(past))
    if torch.isnan(ret).any():
        print(ret)
        raise Exception("NAN")
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
            raise Exception("NAN")
        return ret
    elif reduction == "sum":
        ret = torch.sum(loss)
        if torch.isnan(ret).any():
            print(ret)
            raise Exception("NAN")
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
            raise Exception("NAN")
        return ret

class RMSSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, past, future, pred):
        return reduce(rmsse(past, future, pred), reduction=self.reduction)
