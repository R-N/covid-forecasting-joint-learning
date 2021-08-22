import torch
import numpy as np
from ..data.util import right_slice, full_combinations


def set_tuple(a):
    if isinstance(a, list):
        return tuple(set([set_tuple(x) for x in a]))
    return a


def set_similarity(a, b):
    return len(a.intersection(b)) / len(a.union(b))


def find_similar_set(a, sets):
    similarities = [(b, set_similarity(a, b)) for b in sets]
    return max(similarities, key=lambda x: x[1])


def global_random_seed(seed=257):
    torch.manual_seed(seed)
    np.random.seed(seed)
