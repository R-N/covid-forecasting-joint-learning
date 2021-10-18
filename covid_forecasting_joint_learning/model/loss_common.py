import numpy as np
from . import util as ModelUtil

def mse(err):
    return np.mean((err)**2, axis=-2)

def naive(past, step=1, limit=None):
    if limit:
        past = past[-limit:]
    return past[:-step] - past[step:]

def msse(future, pred, mse_naive=None, past=None, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    assert mse_naive is not None or past is not None
    if mse_naive is None:
        mse_naive = mse(naive(past, limit=limit_naive))
    return mse(pred - future) / (mse_naive + eps)

def rmsse(future, pred, mse_naive=None, past=None, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    return np.sqrt(msse(
        future,
        pred,
        mse_naive=mse_naive,
        past=past,
        limit_naive=limit_naive,
        eps=eps
    ))

def reduce(loss, reduction="sum"):
    while loss.ndim > 1:
        loss = np.sum(loss, axis=-1)
    if reduction in ("mean", "avg"):
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError(f"Invalid reduction {reduction}")

def wrap_reduce(loss_fn, reduction="sum"):
    def wrapper(*args, **kwargs):
        loss = loss_fn(*args, **kwargs)
        return reduce(loss, reduction=reduction)
    return wrapper