import torch
from torch import nn
from optuna.structs import TrialPruned
from . import util as ModelUtil
from . import torch_sird as TorchSird

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
    if loss.dim() < 2:
        while reduce_feature and loss.dim() > 0:
            loss = torch.sum(loss, axis=-1)
        return loss
    else:
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

class ReconstructedRMSSELoss(nn.Module):
    """RMSSE on reconstructed IRD counts, differentiable through the SIRD
    rebuild recurrence (`model/torch_sird.py`) -- unlike `RMSSELoss`, which
    scores the model's raw scaled-rate output directly. Aligns the training
    objective with what `pipeline/train.py::test` (and hence
    `ClusterModel.metric`, see `model/general.py`) actually reports.

    Needs more context than a plain `(past, future, pred)` loss: population,
    the rate scaler, and the raw final_seed/future_final IRD samples. Opt in
    via the `reconstructed = True` class attribute -- `model/train.py`'s
    per-batch loop branches on that flag to call this with the extended
    signature below instead of the usual one, so `RMSSELoss`/`MSSELoss`
    remain the unchanged default.
    """
    reconstructed = True

    def __init__(self, reduction="sum", limit_naive=30, eps=ModelUtil.NAIVE_EPS):
        super().__init__()
        self.reduction = reduction
        self.limit_naive = limit_naive
        self.eps = eps

    def forward(self, pred_vars, prev, future_final, n, scaler):
        pred_vars = TorchSird.inverse_scale(pred_vars, scaler)
        pred_final = TorchSird.rebuild(pred_vars, prev[:, -1, :], n)
        past_ird = prev[:, :, 1:]  # drop S, matching pipeline.train.test's `prev[i][:, 1:]`
        return reduce(
            rmsse(past_ird, future_final, pred_final, limit_naive=self.limit_naive, eps=self.eps),
            reduction=self.reduction,
        )
