import numpy as np

def mse(err):
    return np.mean((err)**2, axis=-2)

def naive(past, step=1):
    return past[:-step] - past[step:]

def msse(past, future, pred):
    return mse(pred - future) / mse(naive(past))

def rmsse(past, future, pred):
    return np.sqrt(msse(past, future, pred))

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