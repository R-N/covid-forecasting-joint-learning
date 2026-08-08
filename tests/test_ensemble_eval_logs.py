"""Regression check for median-ensembling per-seed EvalLog tables.

INVESTIGATION.md, Quick wins -- accuracy: "Median-ensemble the seeds the
rerun already requires -- the runs are already being paid for; budget
five." `SIRDEvalLog`/`ARIMASIRDEvalLog`/`NaiveEvalLog` (and any neural
per-seed log written the same way) all write one row per
(group, cluster, kabko) with i/r/d loss columns; `ensemble_eval_logs`
combines several such per-seed tables into one via an elementwise median,
so multi-seed reruns feed a single, less noisy loss into the statistical
tests instead of picking one seed arbitrarily.

Orange is stubbed since `pipeline/eval.py` imports it at module level; no
other heavy dependency is needed (pandas/numpy already available). Run
with:

    python tests/test_ensemble_eval_logs.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "Orange" not in sys.modules:
        orange = types.ModuleType("Orange")
        evaluation = types.ModuleType("Orange.evaluation")
        evaluation.compute_CD = lambda *a, **k: None
        evaluation.graph_ranks = lambda *a, **k: None
        orange.evaluation = evaluation
        sys.modules["Orange"] = orange
        sys.modules["Orange.evaluation"] = evaluation


def _log_df(rows):
    return pd.DataFrame(rows, columns=["group", "cluster", "kabko", "i", "r", "d"])


def test_median_across_three_seeds_is_exact():
    from covid_forecasting_joint_learning.pipeline.eval import ensemble_eval_logs

    seed_0 = _log_df([{"group": "g", "cluster": 0, "kabko": "surabaya", "i": 1.0, "r": 4.0, "d": 9.0}])
    seed_1 = _log_df([{"group": "g", "cluster": 0, "kabko": "surabaya", "i": 3.0, "r": 6.0, "d": 3.0}])
    seed_2 = _log_df([{"group": "g", "cluster": 0, "kabko": "surabaya", "i": 2.0, "r": 5.0, "d": 6.0}])

    out = ensemble_eval_logs([seed_0, seed_1, seed_2])

    assert len(out) == 1
    row = out.iloc[0]
    assert (row["i"], row["r"], row["d"]) == (2.0, 5.0, 6.0)


def test_single_log_is_a_no_op():
    from covid_forecasting_joint_learning.pipeline.eval import ensemble_eval_logs

    log = _log_df([
        {"group": "g", "cluster": 0, "kabko": "surabaya", "i": 1.0, "r": 2.0, "d": 3.0},
        {"group": "g", "cluster": 0, "kabko": "malang", "i": 4.0, "r": 5.0, "d": 6.0},
    ])

    out = ensemble_eval_logs([log]).sort_values("kabko").reset_index(drop=True)
    expected = log.sort_values("kabko").reset_index(drop=True)
    assert np.allclose(out[["i", "r", "d"]].to_numpy(), expected[["i", "r", "d"]].to_numpy())


def test_combines_independently_per_group_cluster_kabko():
    from covid_forecasting_joint_learning.pipeline.eval import ensemble_eval_logs

    seed_0 = _log_df([
        {"group": "g", "cluster": 0, "kabko": "surabaya", "i": 1.0, "r": 1.0, "d": 1.0},
        {"group": "g", "cluster": 0, "kabko": "malang", "i": 10.0, "r": 10.0, "d": 10.0},
    ])
    seed_1 = _log_df([
        {"group": "g", "cluster": 0, "kabko": "surabaya", "i": 3.0, "r": 3.0, "d": 3.0},
        {"group": "g", "cluster": 0, "kabko": "malang", "i": 20.0, "r": 20.0, "d": 20.0},
    ])

    out = ensemble_eval_logs([seed_0, seed_1]).set_index("kabko")
    assert np.isclose(out.loc["surabaya", "i"], 2.0)
    assert np.isclose(out.loc["malang", "i"], 15.0)


if __name__ == "__main__":
    install_stubs()
    test_median_across_three_seeds_is_exact()
    test_single_log_is_a_no_op()
    test_combines_independently_per_group_cluster_kabko()
    print("ok")
