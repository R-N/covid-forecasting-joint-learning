"""Regression check for the naive (last-value-carried-forward) baseline.

INVESTIGATION.md, Recommendations #1: an explicit naive arm was missing
entirely -- `loss_common.naive()` only supplies the MSSE/RMSSE scaling
denominator, not a standalone forecast to compare against. This pins that
`NaiveModel` predicts a flat line at the last observed I/R/D values, reports
a per-IRD (not scalar) loss so it logs into the same schema as the SIRD and
ARIMA-SIRD baselines, and unpacks the standard `label_dataset_0` sample the
same way `ARIMASIRDModel` does.

xlrd/line_profiler are stubbed; this needs no other heavy dependency (no
lmfit, statsmodels, optuna, sklearn, tslearn). Run with:

    python tests/test_naive_baseline.py
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


def test_pred_final_repeats_the_last_observed_ird():
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveModel

    model = NaiveModel()
    final_seed = np.array([
        [900.0, 80.0, 15.0, 5.0],
        [890.0, 85.0, 18.0, 7.0],  # last row: S=890, I=85, R=18, D=7
    ])
    pred = model.pred_final(days=5, final_seed=final_seed)
    assert pred.shape == (5, 3)
    assert np.allclose(pred, [85.0, 18.0, 7.0]), pred


def test_eval_returns_per_ird_loss_not_scalar():
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveModel

    model = NaiveModel()
    final_seed = np.array([[900.0, 80.0, 15.0, 5.0], [890.0, 85.0, 18.0, 7.0]])
    future_final = np.array([[86.0, 19.0, 8.0], [90.0, 20.0, 9.0]])

    loss = model.eval(final_seed=final_seed, future_final=future_final)
    assert loss.shape == (3,), loss  # one value per I/R/D, not a summed scalar
    assert len(loss) == 3  # NaiveEvalLog.log() asserts exactly this


def test_perfect_flat_series_has_zero_loss():
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveModel

    model = NaiveModel()
    # A series that has already flatlined: the naive forecast is exact.
    final_seed = np.array([[900.0, 85.0, 18.0, 7.0]] * 3)
    future_final = np.array([[85.0, 18.0, 7.0]] * 5)
    loss = model.eval(final_seed=final_seed, future_final=future_final)
    assert np.allclose(loss, 0.0), loss


def test_eval_sample_unpacks_standard_dataset():
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveModel

    model = NaiveModel()
    past = np.zeros((30, 12))
    past_seed = np.ones((5, 3))
    past_exo = np.ones((5, 3))
    future = np.ones((14, 3)) * 3.0
    future_exo = np.ones((14, 3)) * 4.0
    final_seed = np.array([[900.0, 85.0, 18.0, 7.0]] * 3)
    future_final = np.array([[85.0, 18.0, 7.0]] * 14)
    index = list(range(14))
    sample = (past, past_seed, past_exo, future, future_exo, final_seed, future_final, index)

    loss = model.eval_sample(sample)
    assert np.allclose(loss, 0.0), loss  # future_final matches the flat seed exactly

    # 7-field sample (no trailing index) unpacks identically.
    loss_7 = model.eval_sample(sample[:7])
    assert np.allclose(loss_7, loss)


def test_eval_dataset_reduces_over_samples():
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveModel

    model = NaiveModel()
    final_seed = np.array([[900.0, 86.0, 19.0, 8.0], [895.0, 85.0, 18.0, 7.0]])
    future_final_exact = np.array([[85.0, 18.0, 7.0]] * 3)
    future_final_off = np.array([[95.0, 18.0, 7.0]] * 3)  # I is off by 10

    sample_exact = (None, None, None, None, None, final_seed, future_final_exact)
    sample_off = (None, None, None, None, None, final_seed, future_final_off)

    mean_loss = model.eval_dataset([sample_exact, sample_off], reduction="mean")
    assert mean_loss.shape == (3,)
    assert mean_loss[0] > 0  # the I-component loss reflects the miss
    assert np.isclose(mean_loss[1], 0.0) and np.isclose(mean_loss[2], 0.0)


def test_eval_log_round_trips_per_ird_columns():
    import pandas as pd
    from covid_forecasting_joint_learning.model.comparison.naive import NaiveEvalLog

    # xlrd/openpyxl aren't installed in this bare environment; the schema
    # and lookup logic don't need real file I/O to check, so load_log/
    # save_log are replaced with an in-memory frame.
    log = object.__new__(NaiveEvalLog)
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


if __name__ == "__main__":
    install_stubs()
    test_pred_final_repeats_the_last_observed_ird()
    test_eval_returns_per_ird_loss_not_scalar()
    test_perfect_flat_series_has_zero_loss()
    test_eval_sample_unpacks_standard_dataset()
    test_eval_dataset_reduces_over_samples()
    test_eval_log_round_trips_per_ird_columns()
    print("ok")
