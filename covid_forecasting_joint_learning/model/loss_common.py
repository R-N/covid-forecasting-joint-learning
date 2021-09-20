import numpy as np
from . import util as ModelUtil

def mse(err):
    return np.mean((err)**2, axis=-2)

def naive(past, step=1, limit=None, eps=ModelUtil.NAIVE_EPS):
    if limit:
        past = past[:limit]
    return past[:-step] - past[step:] + eps

def msse(past, future, pred, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    return mse(pred - future) / mse(naive(past, limit=limit_naive, eps=eps))

def rmsse(past, future, pred, limit_naive=30, eps=ModelUtil.NAIVE_EPS):
    return np.sqrt(msse(past, future, pred, limit=limit_naive, eps=eps))

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