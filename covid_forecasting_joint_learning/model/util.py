import torch
import json
from torch import nn
import numpy as np
import line_profiler
import os
from shutil import copy2, Error, copystat, rmtree
from pathlib import Path
from math import sqrt
import scipy.stats as st
from decimal import Decimal
from math import floor


LINE_PROFILER = line_profiler.LineProfiler()


NAIVE_EPS = 1e-6


CUDA = False
HALF = False
DEFAULT_TENSOR = "torch.FloatTensor"
DEVICE = None


def init(cuda=True, half=False):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    tensor_precision = "HalfTensor" if half else "FloatTensor"
    tensor_device = "torch.cuda" if cuda else "torch"
    default_tensor = f"{tensor_device}.{tensor_precision}"
    torch.set_default_tensor_type(default_tensor)

    global CUDA
    CUDA = cuda
    global HALF
    HALF = half
    global DEFAULT_TENSOR
    DEFAULT_TENSOR = default_tensor
    global DEVICE
    DEVICE = device

    return device


def learnable_normal(size, mean, std):
    t = torch.zeros(*size, requires_grad=True)
    nn.init.normal_(t, mean, std)
    return t


def learnable_xavier(size):
    t = torch.zeros(1, *size, requires_grad=True)
    nn.init.xavier_normal_(t)
    return t[0].detach()


def repeat_batch(t, batch_size):
    # t = batch_size * [t]
    # t = torch.stack(t)
    one = torch.ones(batch_size, *t.size())
    t = one * t
    return t


# Linear tensor uses input of shape (Batch, Length, Channel)
# RNN uses input of shape (Length, Batch, Channel)
def linear_to_sequential_tensor(t):
    if t.dim() == 3:
        return t.permute(1, 0, 2)
    elif t.dim() == 2:
        return t.permute(0, 1)


def sequential_to_linear_tensor(t):
    if t.dim() == 3:
        return t.permute(1, 0, 2)
    elif t.dim() == 2:
        return t.permute(0, 1)


# Conv1d uses input of shape (Batch, Channel, Length)
def linear_to_conv1d_tensor(t):
    if t.dim() == 3:
        return t.permute(0, 2, 1)
    elif t.dim() == 2:
        return t.permute(1, 0)


def conv1d_to_linear_tensor(t):
    if t.dim() == 3:
        return t.permute(0, 2, 1)
    elif t.dim() == 2:
        return t.permute(1, 0)

def __str_dict(x):
    if isinstance(x, torch.nn.Module):
        return type(x).__name__
    elif issubclass(x, torch.nn.Module):
        return x.__name__
    return str(x)


def str_dict(d):
    return json.dumps(
        d,
        sort_keys=True,
        indent=4,
        default=__str_dict
    )


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def progressive_smooth(last, weight, point):
    return last * weight + (1 - weight) * point


def union_lists(lists):
    return sorted(list(set.union(*[set(x) for x in lists])))


def multi_index_dict(d, index):
    return [d[i] for i in index]


def single_batch(t):
    return torch.stack([t[0]])


def global_random_seed(seed=257):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _copytree(entries, src, dst, symlinks, ignore, copy_function,
              ignore_dangling_symlinks, dirs_exist_ok=False):
    if ignore is not None:
        ignored_names = ignore(src, set(os.listdir(src)))
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    use_srcentry = copy_function is copy2 or copy_function is copy

    for srcentry in entries:
        if srcentry.name in ignored_names:
            continue
        srcname = os.path.join(src, srcentry.name)
        dstname = os.path.join(dst, srcentry.name)
        srcobj = srcentry if use_srcentry else srcname
        try:
            if srcentry.is_symlink():
                linkto = os.readlink(srcname)
                if symlinks:
                    os.symlink(linkto, dstname)
                    copystat(srcobj, dstname, follow_symlinks=not symlinks)
                else:
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    if srcentry.is_dir():
                        copytree(srcobj, dstname, symlinks, ignore,
                                 copy_function, dirs_exist_ok=dirs_exist_ok)
                    else:
                        copy_function(srcobj, dstname)
            elif srcentry.is_dir():
                copytree(srcobj, dstname, symlinks, ignore, copy_function,
                         dirs_exist_ok=dirs_exist_ok)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy_function(srcentry, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)
    return dst

def copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2,
             ignore_dangling_symlinks=False, dirs_exist_ok=False):
    with os.scandir(src) as entries:
        return _copytree(entries=entries, src=src, dst=dst, symlinks=symlinks,
                         ignore=ignore, copy_function=copy_function,
                         ignore_dangling_symlinks=ignore_dangling_symlinks,
                         dirs_exist_ok=dirs_exist_ok)

def prepare_dir(path):
    if isinstance(path, str):
        if not path.endswith("/"):
            path = path + "/"
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def delete_dir_contents(target_dir):
    with os.scandir(target_dir) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                rmtree(entry.path)
            else:
                os.remove(entry.path)

def prepare_log_model_dir(log_dir, model_dir, trial_id, mkdir=False):
    if trial_id is None:
        return log_dir, model_dir
    if isinstance(log_dir, str):
        log_dir = f"{log_dir}T{trial_id}"
        if mkdir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(model_dir, str):
        model_dir = f"{model_dir}{trial_id}"
        if mkdir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
    return log_dir, model_dir

def calculate_prediction_interval(series, alpha=0.05, n=None):
    n = (n or len(series))
    mean = sum(series) / max(1, n)
    sum_err = sum([(mean - x)**2 for x in series])
    stdev = sqrt(1 / max(1, n - 2) * sum_err)
    mul = st.norm.ppf(1.0 - alpha) if alpha >= 0 else 2 + alpha
    sigma = mul * stdev
    return mean, sigma

def round_digits(x, n_digits=0):
    if x is None:
        return x
    x = Decimal(x).as_tuple()
    n = min(len(x.digits), n_digits + 2)
    digits = sum([x.digits[i] * 10**(-i) for i in range(n)])
    if n_digits == 0:
        digits = floor(digits)
    else:
        digits = round(digits, n_digits)
    exp = x.exponent + len(x.digits) - 1
    return digits * 10**exp

def torch_max(tensor, dim=0):
    indices = torch.max(torch.abs(tensor), dim=dim, keepdim=True).indices
    result = torch.take_along_dim(tensor, indices, dim=dim)
    result = torch.sum(result, dim=dim)
    return result
