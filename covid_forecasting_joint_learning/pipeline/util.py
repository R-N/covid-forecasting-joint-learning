from itertools import combinations, chain
from ..data.util import right_slice


def set_tuple(a):
    if isinstance(a, list):
        return tuple(set([set_tuple(x) for x in a]))
    return a


def set_similarity(a, b):
    return len(a.intersection(b))/len(a.union(b))


def find_similar_set(a, sets):
    similarities = [(b, set_similarity(a, b)) for b in sets]
    return max(similarities, key=lambda x: x[1])


def full_combinations(src):
    src = list(src)
    n_src = len(src)
    combs = [list(combinations(src, x)) for x in range(0, n_src+1)]
    return list(chain.from_iterable(combs))
