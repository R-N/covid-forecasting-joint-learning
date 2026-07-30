from ..data.util import right_slice, full_combinations
from optuna.trial import TrialState


def set_tuple(a):
    if isinstance(a, list):
        return tuple(set([set_tuple(x) for x in a]))
    return a


def set_similarity(a, b):
    return len(a.intersection(b)) / len(a.union(b))


def find_similar_set(a, sets):
    similarities = [(b, set_similarity(a, b)) for b in sets]
    return max(similarities, key=lambda x: x[1])


def filter_trials_undone(trials, count_pruned=False):
    return [t.number for t in trials if not (t.state == TrialState.COMPLETE or (count_pruned and t.state == TrialState.PRUNED))]


def count_trials_done(trials, count_pruned=True):
    # A pruned trial spent compute and informed the sampler, so it counts against
    # the budget. Counting it as undone makes optimize() loop until n_trials
    # trials survive pruning, which is unbounded once pruning is enabled.
    return len(trials) - len(filter_trials_undone(trials, count_pruned=count_pruned))
