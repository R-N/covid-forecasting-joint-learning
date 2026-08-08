"""Regression check for `Cluster.rotate_targets` (INVESTIGATION.md, Big
wins: "Rotate the cluster target instead of fixing it to the shortest
training series. Multiplies evaluation data at unchanged per-fit cost and
tests transfer in both directions.").

`pipeline/clustering.py` imports tslearn at module level (not installed
dev-side); stubbed below, none of its real DTW/K-Means math is exercised --
`rotate_targets` only touches `Cluster`/`shortest()`, which are pure Python
over plain objects. Uses a minimal fake standing in for `data/kabko.py::
KabkoData` (only the attributes `Cluster`/`shortest()` actually touch:
`.name`, `.data`, `.cluster`, `.copy`), not the real class (which needs a
`DataCenter` and heavier construction than this pure-clustering-logic check
needs). Run with:

    python3 tests/test_cluster_target_rotation.py
"""
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")
        tslearn_utils = types.ModuleType("tslearn.utils")
        tslearn_utils.to_time_series_dataset = lambda x: x
        tslearn_clustering = types.ModuleType("tslearn.clustering")
        tslearn_clustering.TimeSeriesKMeans = type("TimeSeriesKMeans", (), {})
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

    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.TrialState = type("TrialState", (), {})
        optuna.trial = trial_mod
        sys.modules["optuna"] = optuna
        sys.modules["optuna.trial"] = trial_mod


class FakeKabko:
    """Stands in for `data/kabko.py::KabkoData` -- only the surface
    `Cluster`/`shortest()` touch."""

    def __init__(self, name, length, cluster=None):
        self.name = name
        # `shortest()` needs len(), last_valid_index(), first_valid_index().
        self.data = pd.Series(range(length))
        self.cluster = cluster

    def copy(self, cluster=None):
        return FakeKabko(self.name, len(self.data), cluster=cluster or self.cluster)

    def __repr__(self):
        return f"FakeKabko({self.name!r})"


def make_cluster():
    from covid_forecasting_joint_learning.pipeline.clustering import Cluster

    cluster = Cluster(id=0, group=None)
    members = [FakeKabko("a", 30, cluster=cluster), FakeKabko("b", 50, cluster=cluster), FakeKabko("c", 10, cluster=cluster)]
    cluster.members = members
    cluster.targets = []
    cluster.select_target()  # targets=[] falls back to shortest(sources), matching Cluster.copy()'s pattern
    return cluster, members


def test_default_target_is_the_shortest_series():
    cluster, members = make_cluster()
    # shortest() sorts by -len ascending... `select_target` picks max(key=shortest),
    # and shortest()'s first element is -len(x.data), so max picks the
    # *smallest* len -- "c" (10 rows) is the default target.
    assert cluster.target.name == "c"


def test_rotate_targets_covers_every_member_exactly_once():
    cluster, members = make_cluster()
    rotated = list(cluster.rotate_targets())

    assert len(rotated) == len(members)
    target_names = sorted(r.target.name for r in rotated)
    assert target_names == sorted(m.name for m in members)


def test_rotate_targets_keeps_the_full_member_set_and_swaps_sources():
    cluster, members = make_cluster()
    rotated = {r.target.name: r for r in cluster.rotate_targets()}

    for name, r in rotated.items():
        assert sorted(k.name for k in r.members) == sorted(m.name for m in members)
        assert r.target.name == name
        assert sorted(k.name for k in r.sources) == sorted(m.name for m in members if m.name != name)


def test_rotate_targets_does_not_mutate_the_original_or_other_rotations():
    cluster, members = make_cluster()
    original_target_name = cluster.target.name

    rotated = list(cluster.rotate_targets())

    assert cluster.target.name == original_target_name
    for r in rotated:
        assert r is not cluster
        assert r.members is not cluster.members
        for k in r.members:
            assert k is not next(m for m in members if m.name == k.name)

    # Each rotation's members/cluster identity is independent of the others.
    for r1 in rotated:
        for r2 in rotated:
            if r1 is not r2:
                assert set(id(k) for k in r1.members).isdisjoint(id(k) for k in r2.members)


if __name__ == "__main__":
    install_stubs()
    test_default_target_is_the_shortest_series()
    test_rotate_targets_covers_every_member_exactly_once()
    test_rotate_targets_keeps_the_full_member_set_and_swaps_sources()
    test_rotate_targets_does_not_mutate_the_original_or_other_rotations()
    print("ok")
