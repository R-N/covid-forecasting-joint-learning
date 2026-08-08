"""Regression check for the multi-seed / rolling-forecast-origin support.

INVESTIGATION.md, Required Rerun Design: "Evaluate over multiple seeds and
rolling forecast origins ... every other item is unmeasurable without
this." Two small, previously-missing pieces of infrastructure: staggered
split origins (`calc_rolling_splits`) and a median ensemble across seeds
(`median_ensemble`).

Run with:

    python tests/test_rolling_and_ensemble.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        preprocessing = types.ModuleType("sklearn.preprocessing")
        preprocessing.MinMaxScaler = type("MinMaxScaler", (), {})
        preprocessing.StandardScaler = type("StandardScaler", (), {})
        sklearn.preprocessing = preprocessing
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.preprocessing"] = preprocessing
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")
        utils = types.ModuleType("tslearn.utils")
        utils.to_time_series_dataset = lambda *a, **k: None
        clustering = types.ModuleType("tslearn.clustering")
        clustering.TimeSeriesKMeans = type("TimeSeriesKMeans", (), {})
        clustering.silhouette_score = lambda *a, **k: None
        metrics = types.ModuleType("tslearn.metrics")
        metrics.dtw = lambda *a, **k: None
        tslearn.utils = utils
        tslearn.clustering = clustering
        tslearn.metrics = metrics
        sys.modules["tslearn"] = tslearn
        sys.modules["tslearn.utils"] = utils
        sys.modules["tslearn.clustering"] = clustering
        sys.modules["tslearn.metrics"] = metrics
    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.TrialState = type("TrialState", (), {})
        optuna.trial = trial_mod
        sys.modules["optuna"] = optuna
        sys.modules["optuna.trial"] = trial_mod
    if "line_profiler" not in sys.modules:
        line_profiler = types.ModuleType("line_profiler")
        line_profiler.LineProfiler = type("LineProfiler", (), {})
        sys.modules["line_profiler"] = line_profiler


def test_rolling_splits_origin_zero_matches_calc_split():
    import pandas as pd
    from covid_forecasting_joint_learning.pipeline.preprocessing import calc_split, calc_rolling_splits

    n_rows = 300
    df = pd.DataFrame({"x": range(n_rows)})

    single = calc_split(df, past_size=30, future_size=14)
    rolling = calc_rolling_splits(df, n_origins=5, past_size=30, future_size=14)

    assert len(rolling) >= 1
    assert rolling[0] == single


def test_rolling_splits_are_staggered_and_valid():
    import pandas as pd
    from covid_forecasting_joint_learning.pipeline.preprocessing import calc_split, calc_rolling_splits

    n_rows = 300
    past_size, future_size = 30, 14
    df = pd.DataFrame({"x": range(n_rows)})

    rolling = calc_rolling_splits(df, n_origins=5, past_size=past_size, future_size=future_size)
    assert len(rolling) == 5

    # Each later origin is a strictly earlier, shorter cut: its test_start
    # (an integer label on this RangeIndex) must not exceed the previous
    # origin's, and every boundary respects calc_split's own ordering.
    prev_test_start = None
    for train_end, val_start, val_end, test_start in rolling:
        assert train_end < val_start <= val_end < test_start
        assert test_start + future_size <= n_rows
        if prev_test_start is not None:
            assert test_start <= prev_test_start
        prev_test_start = test_start


def test_rolling_splits_stop_when_too_short_for_another_origin():
    import pandas as pd
    from covid_forecasting_joint_learning.pipeline.preprocessing import calc_rolling_splits

    # A short series can't support many staggered origins; the generator
    # must return fewer than requested rather than error or emit an
    # unusable (too-short) split.
    n_rows = 100
    rolling = calc_rolling_splits(df=pd.DataFrame({"x": range(n_rows)}), n_origins=200, past_size=30, future_size=14)
    assert 0 < len(rolling) < 200


def test_median_ensemble_reduces_seed_noise():
    from covid_forecasting_joint_learning.model.util import median_ensemble

    truth = np.array([10.0, 20.0, 30.0])
    # Three noisy per-seed predictions straddling the truth; an outlier
    # seed (the third) must not drag a median ensemble off target the way
    # it would a mean.
    seed_preds = [
        truth + np.array([1.0, -1.0, 0.5]),
        truth + np.array([-1.0, 1.0, -0.5]),
        truth + np.array([50.0, 50.0, 50.0]),  # one bad seed
    ]
    median = median_ensemble(seed_preds)
    mean = np.mean(np.stack(seed_preds), axis=0)

    median_err = np.abs(median - truth).sum()
    mean_err = np.abs(mean - truth).sum()
    assert median_err < mean_err, (median_err, mean_err)
    assert median.shape == truth.shape


def test_median_ensemble_is_exact_for_identical_runs():
    from covid_forecasting_joint_learning.model.util import median_ensemble

    same = np.array([1.0, 2.0, 3.0])
    result = median_ensemble([same, same, same])
    assert np.allclose(result, same)


if __name__ == "__main__":
    install_stubs()
    test_rolling_splits_origin_zero_matches_calc_split()
    test_rolling_splits_are_staggered_and_valid()
    test_rolling_splits_stop_when_too_short_for_another_origin()
    test_median_ensemble_reduces_seed_noise()
    test_median_ensemble_is_exact_for_identical_runs()
    print("ok")
