import numpy as np

def mse(err):
    return np.mean((err)**2, dim=1)

def naive(past, step=1):
    return past[:, :-step] - past[:, step:]

def msse(past, future, pred):
    return mse(pred - future) / mse(naive(past))

def rmsse(past, future, pred):
    return np.sqrt(msse(past, future, pred))
