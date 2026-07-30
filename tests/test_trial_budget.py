"""Regression check for the Optuna trial budget once pruning is enabled.

`main.optimize()` loops until `count_trials_done()` reaches `n_trials`. A pruned
trial spent compute, so it has to count; otherwise the loop keeps launching
batches until `n_trials` trials survive pruning, which is unbounded. Run with:

    python tests/test_trial_budget.py

Optuna and the numeric stack are stubbed so this runs in a bare interpreter.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub(name, **attrs):
    module = sys.modules.setdefault(name, types.ModuleType(name))
    for key, value in attrs.items():
        setattr(module, key, value)
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_stub(parent), child, module)
    return module


class TrialState:
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"


def install_stubs():
    if "optuna" not in sys.modules:
        _stub("optuna.trial", TrialState=TrialState)
    if "torch" not in sys.modules:
        _stub("torch", Tensor=object, from_numpy=lambda x: x)
        _stub("torch.nn", Module=object)
    if "numpy" not in sys.modules:
        _stub("numpy", ndarray=object, float32="float32", nan=float("nan"))
    if "pandas" not in sys.modules:
        _stub("pandas", DataFrame=object, Series=object)
    if "sklearn" not in sys.modules:
        _stub("sklearn.preprocessing", MinMaxScaler=object, StandardScaler=object)


def trials(*states):
    return [types.SimpleNamespace(number=i, state=s) for i, s in enumerate(states)]


def test_pruned_trials_count_against_the_budget():
    from covid_forecasting_joint_learning.pipeline.util import count_trials_done, filter_trials_undone

    done = trials(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING)
    assert count_trials_done(done) == 2, "pruned trial not counted, optimize() would not terminate"

    # Failed and still-running trials must stay retryable.
    assert filter_trials_undone(done, count_pruned=True) == [2, 3]

    # The old accounting is still available for callers that want it.
    assert count_trials_done(done, count_pruned=False) == 1


if __name__ == "__main__":
    install_stubs()
    test_pruned_trials_count_against_the_budget()
    print("ok")
