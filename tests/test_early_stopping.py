"""Regression check for EarlyStopping's first call.

`EarlyStopping.__call__()` computes its loss intervals before recording any loss, so
on the first epoch both histories are still empty. Run with:

    python tests/test_early_stopping.py

The research dependencies are stubbed so this check runs in a bare interpreter;
`scipy.stats.norm.ppf` is backed by the stdlib `statistics.NormalDist`.
"""
import statistics
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


def install_stubs():
    if "torch" not in sys.modules:
        _stub("torch", device=lambda *a, **kw: "cpu", Tensor=object)
        _stub("torch.nn", Module=object)
        _stub("torch.utils.tensorboard", SummaryWriter=object)
    if "numpy" not in sys.modules:
        _stub("numpy", ndarray=object)
    if "line_profiler" not in sys.modules:
        _stub("line_profiler", LineProfiler=lambda: (lambda f: f))
    if "scipy" not in sys.modules:
        _stub("scipy.stats", norm=types.SimpleNamespace(ppf=statistics.NormalDist().inv_cdf))


class DummyModel:
    """EarlyStopping only ever snapshots and reloads a state dict."""

    def __init__(self):
        self.state = {"w": 0.0}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def test_first_call_without_history():
    from covid_forecasting_joint_learning.model.early_stopping import EarlyStopping

    # Both interval modes have to survive the empty first epoch: mode 0 takes
    # min()/max() of the history, mode 1 averages it.
    for mode in (0, 1):
        stopper = EarlyStopping(DummyModel(), max_epoch=10, interval_mode=mode)
        assert not stopper.train_loss_history and not stopper.val_loss_history

        assert stopper(1.0, 0.9) is False, f"interval_mode={mode} stopped on the first epoch"
        assert stopper.epoch == 1
        assert stopper.best_val_loss is not None
        assert stopper.train_loss_history == [1.0]
        assert stopper.val_loss_history == [0.9]

        # Keep going so the populated-history path is covered too.
        for i, (train_loss, val_loss) in enumerate([(0.8, 0.7), (0.7, 0.65), (0.6, 0.6)], start=1):
            assert stopper(train_loss, val_loss) is False
            assert stopper.epoch == i + 1


def test_empty_history_interval_falls_back_to_current_loss():
    from covid_forecasting_joint_learning.model.early_stopping import EarlyStopping

    stopper = EarlyStopping(DummyModel(), max_epoch=10)
    for mode in (0, 1):
        stopper.interval_mode = mode
        mid, delta = stopper.calculate_interval(val=True, default=2.5)
        assert mid == 2.5, f"interval_mode={mode} lost the fallback midpoint"
        assert delta == stopper.eps


if __name__ == "__main__":
    install_stubs()
    test_first_call_without_history()
    test_empty_history_interval_falls_back_to_current_loss()
    print("ok")
