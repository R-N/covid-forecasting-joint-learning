"""Regression check for the Theta and tuned-linear baselines.

INVESTIGATION.md, Quick wins: "Add Theta and a tuned linear baseline --
minutes each, and both are standard". This pins that `ThetaModel` fits one
univariate `statsmodels.tsa.forecasting.theta.ThetaModel` per I/R/D column
and falls back to naive last-value-carried-forward on a per-column
fit-failure (never crashing the whole comparison run on one bad kabko),
that `LinearModel` fits a plain least-squares linear-in-time trend per
column with no new dependency and recovers it exactly on synthetic linear
data, and that both unpack the standard `label_dataset_0` sample and log
into the shared i/r/d schema the same way the naive/SIRD/ARIMA-SIRD
baselines do.

statsmodels (only the `tsa.forecasting.theta` submodule actually imported)
is stubbed with a dummy `ThetaModel`, plus xlrd/line_profiler per the
shared pattern. Run with:

    python tests/test_theta_linear_baselines.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StubThetaResult:
    def __init__(self, series):
        self._last = series[-1]

    def forecast(self, steps):
        return np.full(steps, self._last)


class StubThetaModel:
    """Dummy stand-in for statsmodels.tsa.forecasting.theta.ThetaModel.

    `.fit()` fails on the call numbers listed in `fail_on_calls` (1-indexed,
    reset per test) so tests can force ThetaModel's per-column fit-failure
    fallback without a real statsmodels dependency.
    """
    calls = 0
    fail_on_calls = set()

    def __init__(self, series, period=7):
        self.series = np.asarray(series, dtype=float)
        self.period = period

    def fit(self):
        StubThetaModel.calls += 1
        if StubThetaModel.calls in StubThetaModel.fail_on_calls:
            raise RuntimeError("stub Theta fit failure")
        return StubThetaResult(self.series)


def install_stubs():
    if "xlrd" not in sys.modules:
        xlrd = types.ModuleType("xlrd")
        xlrd.XLRDError = type("XLRDError", (Exception,), {})
        sys.modules["xlrd"] = xlrd
    if "line_profiler" not in sys.modules:
        line_profiler = types.ModuleType("line_profiler")
        line_profiler.LineProfiler = type("LineProfiler", (), {})
        sys.modules["line_profiler"] = line_profiler
    if "statsmodels.tsa.forecasting.theta" not in sys.modules:
        statsmodels = types.ModuleType("statsmodels")
        tsa = types.ModuleType("statsmodels.tsa")
        forecasting = types.ModuleType("statsmodels.tsa.forecasting")
        theta_mod = types.ModuleType("statsmodels.tsa.forecasting.theta")
        theta_mod.ThetaModel = StubThetaModel
        forecasting.theta = theta_mod
        tsa.forecasting = forecasting
        statsmodels.tsa = tsa
        for name, mod in [
            ("statsmodels", statsmodels),
            ("statsmodels.tsa", tsa),
            ("statsmodels.tsa.forecasting", forecasting),
            ("statsmodels.tsa.forecasting.theta", theta_mod),
        ]:
            sys.modules[name] = mod


def test_theta_fit_forecast_round_trip():
    from covid_forecasting_joint_learning.model.comparison.theta import ThetaModel

    StubThetaModel.calls = 0
    StubThetaModel.fail_on_calls = set()

    t = np.arange(21, dtype=float)
    past_seed = np.stack([50.0 + t, 20.0 + 0.5 * t, 5.0 + 0.1 * t], axis=1)

    model = ThetaModel()
    model.fit(past_seed)
    pred = model.pred_final(days=7)
    assert pred.shape == (7, 3), pred.shape
    assert np.all(pred >= 0.0), pred


def test_theta_fit_failure_falls_back_per_column():
    from covid_forecasting_joint_learning.model.comparison.theta import ThetaModel

    StubThetaModel.calls = 0
    # 3 columns fit in order I, R, D -> the 2nd call (R) fails.
    StubThetaModel.fail_on_calls = {2}

    t = np.arange(21, dtype=float)
    past_seed = np.stack([50.0 + t, 20.0 + 0.5 * t, 5.0 + 0.1 * t], axis=1)

    model = ThetaModel()
    model.fit(past_seed)  # must not raise despite the R-column fit failing
    assert model.fits[1] is None
    assert model.fits[0] is not None and model.fits[2] is not None

    pred = model.pred_final(days=5)
    assert pred.shape == (5, 3), pred.shape
    assert np.all(pred >= 0.0), pred
    # fallback column: naive last-value-carried-forward, i.e. flat at the
    # last observed R value.
    assert np.allclose(pred[:, 1], past_seed[-1, 1])


def test_linear_recovers_exact_trend():
    from covid_forecasting_joint_learning.model.comparison.linear import LinearModel

    n = 10
    t = np.arange(n, dtype=float)
    i = 10.0 + 2.0 * t
    r = 5.0 + 1.0 * t
    d = 1.0 + 0.5 * t
    past_seed = np.stack([i, r, d], axis=1)

    model = LinearModel()
    model.fit(past_seed)
    days = 6
    pred = model.pred_final(days)

    future_t = np.arange(n, n + days, dtype=float)
    expected = np.stack([10.0 + 2.0 * future_t, 5.0 + 1.0 * future_t, 1.0 + 0.5 * future_t], axis=1)
    assert pred.shape == (days, 3)
    assert np.allclose(pred, expected, atol=1e-8), (pred, expected)


def test_linear_eval_sample_unpacks_standard_dataset():
    from covid_forecasting_joint_learning.model.comparison.linear import LinearModel

    model = LinearModel()
    # `past` is a wide, differently-shaped decoy: if eval_sample fit on it
    # instead of past_seed the forecast would not exactly match the flat
    # future_final below.
    past = np.zeros((30, 12))
    past_seed = np.array([[85.0, 18.0, 7.0]] * 5)  # flat window -> zero slope
    past_exo = np.ones((5, 3))
    future = np.ones((14, 3)) * 3.0
    future_exo = np.ones((14, 3)) * 4.0
    final_seed = np.array([[900.0, 85.0, 18.0, 7.0]] * 3)
    future_final = np.array([[85.0, 18.0, 7.0]] * 14)
    index = list(range(14))
    sample = (past, past_seed, past_exo, future, future_exo, final_seed, future_final, index)

    loss = model.eval_sample(sample)
    assert np.allclose(loss, 0.0), loss  # flat past_seed -> flat forecast matching flat future_final

    # 7-field sample (no trailing index) unpacks identically.
    loss_7 = model.eval_sample(sample[:7])
    assert np.allclose(loss_7, loss)


def test_theta_eval_log_round_trips_per_ird_columns():
    import pandas as pd
    from covid_forecasting_joint_learning.model.comparison.theta import ThetaEvalLog

    log = object.__new__(ThetaEvalLog)
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


def test_linear_eval_log_round_trips_per_ird_columns():
    import pandas as pd
    from covid_forecasting_joint_learning.model.comparison.linear import LinearEvalLog

    log = object.__new__(LinearEvalLog)
    log.log_path = "unused"
    log.log_sheet_name = "Eval"
    log.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "i", "r", "d"])
    log.load_log = lambda *a, **k: log.log_df
    log.save_log = lambda *a, **k: None

    log.log(group=0, cluster=1, kabko="surabaya", loss=[0.4, 0.5, 0.6])
    assert log.is_eval_done(0, 1, "surabaya")
    assert not log.is_eval_done(0, 1, "malang")
    row = log.log_df.iloc[0]
    assert (row["i"], row["r"], row["d"]) == (0.4, 0.5, 0.6)


if __name__ == "__main__":
    install_stubs()
    test_theta_fit_forecast_round_trip()
    test_theta_fit_failure_falls_back_per_column()
    test_linear_recovers_exact_trend()
    test_linear_eval_sample_unpacks_standard_dataset()
    test_theta_eval_log_round_trips_per_ird_columns()
    test_linear_eval_log_round_trips_per_ird_columns()
    print("ok")
