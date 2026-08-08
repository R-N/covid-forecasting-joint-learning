"""Regression check for the gradient-boosted-tree lag-feature baseline.

INVESTIGATION.md, Quick wins: "Add a gradient-boosted tree with lag
features -- Three independent literature lines put it ahead of the linear
baseline." This pins `GBTModel`'s window-building (lag features per I/R/D
column, independently), its recursive multi-step forecast (the window
actually rolls forward instead of repeating the same input), non-negative
clipped output of the right shape, `label_dataset_0` sample unpacking, and
`GBTEvalLog`'s per-IRD schema/round-trip -- without exercising sklearn's
real gradient boosting math, which is stubbed out.

sklearn/xlrd/line_profiler are stubbed; this needs no other heavy
dependency (no statsmodels, optuna, lmfit, tslearn). Run with:

    python tests/test_gbt_baseline.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "xlrd" not in sys.modules:
        xlrd = types.ModuleType("xlrd")
        xlrd.XLRDError = type("XLRDError", (Exception,), {})
        sys.modules["xlrd"] = xlrd
    if "line_profiler" not in sys.modules:
        line_profiler = types.ModuleType("line_profiler")
        line_profiler.LineProfiler = type("LineProfiler", (), {})
        sys.modules["line_profiler"] = line_profiler
    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        ensemble = types.ModuleType("sklearn.ensemble")

        class GradientBoostingRegressor:
            """Records fit/predict calls instead of doing any real boosting
            -- only GBTModel's own window-building/recursion logic is under
            test here, not sklearn's math.
            """

            def __init__(self, n_estimators=50, max_depth=3):
                self.n_estimators = n_estimators
                self.max_depth = max_depth
                self.calls = []

            def fit(self, X, y):
                self.X = np.asarray(X)
                self.y = np.asarray(y)
                self.y_mean = float(np.mean(self.y)) if len(self.y) else 0.0
                return self

            def predict(self, X):
                X = np.asarray(X)
                self.calls.append(X.copy())
                # A deterministic function of the window contents (not a
                # constant), so a rolling window produces varying
                # predictions across recursive steps.
                return np.sum(X, axis=1)

        ensemble.GradientBoostingRegressor = GradientBoostingRegressor
        sklearn.ensemble = ensemble
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.ensemble"] = ensemble


def test_fit_pred_final_shape_and_nonnegative():
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    rng = np.random.default_rng(0)
    past_seed = rng.uniform(1.0, 100.0, size=(30, 3))

    model = GBTModel(lag=7)
    model.fit(past_seed)
    pred = model.pred_final(days=14)

    assert pred.shape == (14, 3), pred.shape
    assert np.all(pred >= 0.0), pred


def test_recursive_forecast_rolls_window_forward():
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    # A monotonically increasing series so each recursive step's window
    # differs from the last -- if the loop fed the same window every step,
    # every predict() call for a column would see identical input.
    past_seed = np.tile(np.arange(30, dtype=float).reshape(-1, 1), (1, 3))
    model = GBTModel(lag=7)
    model.fit(past_seed)

    days = 5
    model.pred_final(days=days)

    for col, reg in enumerate(model.regressors):
        assert len(reg.calls) == days, (col, len(reg.calls))
        distinct_windows = {tuple(call.flatten()) for call in reg.calls}
        assert len(distinct_windows) > 1, "recursive window did not advance"


def test_fit_builds_lag_windows_per_column_independently():
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    seed_length = 20
    lag = 4
    past_seed = np.stack([
        np.arange(seed_length, dtype=float),
        np.arange(seed_length, dtype=float) * 2.0,
        np.arange(seed_length, dtype=float) * 3.0,
    ], axis=1)

    model = GBTModel(lag=lag)
    model.fit(past_seed)

    assert len(model.regressors) == 3
    for col, reg in enumerate(model.regressors):
        expected_rows = seed_length - lag
        assert reg.X.shape == (expected_rows, lag), (col, reg.X.shape)
        assert reg.y.shape == (expected_rows,), (col, reg.y.shape)
        # X[t] must be the `lag` values immediately preceding y[t].
        assert np.allclose(reg.X[0], past_seed[:lag, col])
        assert np.isclose(reg.y[0], past_seed[lag, col])


def test_fit_clamps_lag_when_seed_shorter_than_lag():
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    past_seed = np.ones((3, 3))
    model = GBTModel(lag=7)
    model.fit(past_seed)
    assert model.lag == 2  # min(7, seed_length - 1) == min(7, 2)

    pred = model.pred_final(days=2)
    assert pred.shape == (2, 3)


def test_eval_sample_unpacks_standard_dataset():
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    model = GBTModel()
    captured = {}

    def fake_eval(self, past_seed, future_final, loss_fn=None):
        captured.update(past_seed=past_seed, future_final=future_final)
        return np.array([0.1, 0.2, 0.3])

    model.eval = types.MethodType(fake_eval, model)

    # Mirrors label_dataset_0's 8-field tuple: (past, past_seed, past_exo,
    # future, future_exo, final_seed, future_final, index).
    past = np.zeros((30, 12))
    past_seed = np.arange(30, dtype=float).reshape(10, 3)
    past_exo = np.ones((5, 3))
    future = np.ones((14, 3)) * 3.0
    future_exo = np.ones((14, 3)) * 4.0
    final_seed = np.ones((5, 4)) * 5.0
    future_final = np.ones((14, 3)) * 6.0
    index = list(range(14))
    sample = (past, past_seed, past_exo, future, future_exo, final_seed, future_final, index)

    loss = model.eval_sample(sample)
    assert captured["past_seed"] is past_seed
    assert captured["future_final"] is future_final
    assert np.allclose(loss, [0.1, 0.2, 0.3])

    # A 7-field sample (no trailing index) must unpack identically.
    model.eval_sample(sample[:7])
    assert captured["past_seed"] is past_seed
    assert captured["future_final"] is future_final


def test_eval_log_round_trips_per_ird_columns():
    import pandas as pd
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTEvalLog

    # xlrd/openpyxl aren't installed in this bare environment; the schema
    # and lookup logic don't need real file I/O to check, so load_log/
    # save_log are replaced with an in-memory frame.
    log = object.__new__(GBTEvalLog)
    log.log_path = "unused"
    log.log_sheet_name = "Eval"
    log.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "i", "r", "d"])
    log.load_log = lambda *a, **k: log.log_df
    log.save_log = lambda *a, **k: None

    log.log(group=0, cluster=1, kabko="surabaya", loss=[0.1, 0.2, 0.3])
    assert log.is_eval_done(0, 1, "surabaya")
    assert not log.is_eval_done(0, 1, "malang")
    row = log.log_df.iloc[0]
    assert (row["i"], row["r"], row["d"]) == (0.1, 0.2, 0.3)

    try:
        log.log(group=0, cluster=1, kabko="malang", loss=[0.1, 0.2])
        raise AssertionError("expected len(loss) == 3 assertion to fire")
    except AssertionError:
        pass


if __name__ == "__main__":
    install_stubs()
    test_fit_pred_final_shape_and_nonnegative()
    test_recursive_forecast_rolls_window_forward()
    test_fit_builds_lag_windows_per_column_independently()
    test_fit_clamps_lag_when_seed_shorter_than_lag()
    test_eval_sample_unpacks_standard_dataset()
    test_eval_log_round_trips_per_ird_columns()
    print("ok")
