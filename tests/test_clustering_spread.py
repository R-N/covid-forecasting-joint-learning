"""Regression check for `clustering_spread` (INVESTIGATION.md ~L430-440):
measure agreement across a n_clusters x seed sweep of TimeSeriesKMeans
partitions instead of committing to a single silhouette-picked clustering.

tslearn and sklearn are stubbed; none of their real clustering/ARI math is
exercised by these checks -- the dummy TimeSeriesKMeans.fit_predict returns
rigged labels per call, and the dummy adjusted_rand_score is a real (if
simplified) exact-match-fraction agreement measure, good enough to make the
mean/min-agreement assertions meaningful. Run with:

    python3 tests/test_clustering_spread.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")

        tslearn_utils = types.ModuleType("tslearn.utils")
        tslearn_utils.to_time_series_dataset = lambda x: x

        tslearn_clustering = types.ModuleType("tslearn.clustering")

        class TimeSeriesKMeans:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_predict(self, dataset):
                raise NotImplementedError("test must monkeypatch fit_predict")

        tslearn_clustering.TimeSeriesKMeans = TimeSeriesKMeans
        tslearn_clustering.silhouette_score = lambda *a, **k: 0.0

        tslearn_metrics = types.ModuleType("tslearn.metrics")
        tslearn_metrics.dtw = lambda *a, **k: 0.0

        for name, mod in (
            ("tslearn", tslearn),
            ("tslearn.utils", tslearn_utils),
            ("tslearn.clustering", tslearn_clustering),
            ("tslearn.metrics", tslearn_metrics),
        ):
            sys.modules[name] = mod

    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        sklearn_metrics = types.ModuleType("sklearn.metrics")

        def adjusted_rand_score(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            return float(np.mean(a == b))

        sklearn_metrics.adjusted_rand_score = adjusted_rand_score

        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = sklearn_metrics

    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        optuna.create_study = lambda *a, **k: None
        samplers = types.ModuleType("optuna.samplers")
        samplers.TPESampler = type("TPESampler", (), {})
        structs = types.ModuleType("optuna.structs")
        structs.TrialPruned = type("TrialPruned", (Exception,), {})
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.TrialState = type("TrialState", (), {})
        optuna.samplers = samplers
        optuna.structs = structs
        optuna.trial = trial_mod
        for name, mod in [
            ("optuna", optuna),
            ("optuna.samplers", samplers),
            ("optuna.structs", structs),
            ("optuna.trial", trial_mod),
        ]:
            sys.modules[name] = mod


def test_identical_clusterings_give_max_agreement():
    from covid_forecasting_joint_learning.pipeline import clustering

    fixed_labels = np.array([0, 0, 1, 1, 2])

    def fit_predict(self, dataset):
        return fixed_labels

    clustering.TimeSeriesKMeans.fit_predict = fit_predict

    result = clustering.clustering_spread(
        dataset=object(),
        n_clusters_range=[2, 3],
        n_seeds=3,
    )

    n = len([2, 3]) * 3
    assert len(result.pairwise_ari) == n * (n - 1) // 2
    assert result.mean_ari == 1.0
    assert result.min_ari == 1.0


def test_disagreeing_run_pulls_min_below_mean():
    from covid_forecasting_joint_learning.pipeline import clustering

    agree_labels = np.array([0, 0, 1, 1, 2])
    disagree_labels = np.array([2, 1, 0, 2, 1])  # completely different assignment

    calls = {"i": 0}

    def fit_predict(self, dataset):
        calls["i"] += 1
        # first call (n_clusters=2, seed=0) disagrees with everything else
        if calls["i"] == 1:
            return disagree_labels
        return agree_labels

    clustering.TimeSeriesKMeans.fit_predict = fit_predict

    result = clustering.clustering_spread(
        dataset=object(),
        n_clusters_range=[2, 3],
        n_seeds=3,
    )

    n = len([2, 3]) * 3
    assert len(result.pairwise_ari) == n * (n - 1) // 2
    assert result.min_ari < result.mean_ari
    assert result.min_ari == 0.0


def test_clustering_spread_does_not_touch_cluster_best():
    from covid_forecasting_joint_learning.pipeline import clustering
    import inspect

    assert "clustering_spread" not in inspect.signature(clustering.cluster_best).parameters
    src = inspect.getsource(clustering.cluster_best)
    assert "clustering_spread" not in src


if __name__ == "__main__":
    install_stubs()
    test_identical_clusterings_give_max_agreement()
    test_disagreeing_run_pulls_min_below_mean()
    test_clustering_spread_does_not_touch_cluster_best()
    print("ok")
