import torch
import json
from torch import nn
import numpy as np
from optuna.trial import TrialState


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
    return t.permute(1, 0, 2)


def sequential_to_linear_tensor(t):
    return t.permute(1, 0, 2)


# Conv1d uses input of shape (Batch, Channel, Length)
def linear_to_conv1d_tensor(t):
    return t.permute(0, 2, 1)


def conv1d_to_linear_tensor(t):
    return t.permute(0, 2, 1)


def str_dict(d):
    return json.dumps(
        d,
        sort_keys=True,
        indent=4,
        default=str
    )


def filter_trials_undone(study):
    return [t.number for t in study.trials if not (t.state == TrialState.COMPLETE or t.state == TrialState.PRUNED)]


def count_trials_done(study):
    return len(study.trials) - len(filter_trials_undone(study))


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
