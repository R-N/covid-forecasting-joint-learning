"""Regression check for pipeline/eval.py's statistical-testing fixes.

Two defects: `friedman_chi_square` subtracted `k(k+1)^2/4` once per
algorithm instead of once total, and `z_to_p` returned a signed one-sided
p-value, so a control that ranked *worse* than the comparison silently
reported p > 0.5 instead of a small two-sided p-value. Also adds coverage
for the pool-independent sign and Wilcoxon tests recommended alongside MCB.

Orange (needed only for the CD/plot helpers, not the functions checked here)
is stubbed so this runs without that dependency. Run with:

    python tests/test_eval_stats.py
"""
import sys
import types
from pathlib import Path

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


def test_friedman_chi_square_matches_hand_computation():
    from covid_forecasting_joint_learning.pipeline.eval import friedman_chi_square

    # Demsar 2006, eq. 1: X^2_F = 12N/(k(k+1)) * [sum(R_j^2) - k(k+1)^2/4].
    avranks = [1.5, 2.0, 2.5]
    n_datasets = 10
    k = len(avranks)
    k14 = k * (k + 1)**2 / 4
    expected = (12 * n_datasets / (k * (k + 1))) * (sum(r**2 for r in avranks) - k14)
    got = friedman_chi_square(avranks, n_datasets)
    assert abs(got - expected) < 1e-9, (got, expected)

    # The bug subtracted k14 per term (i.e. k*k14 total); pin that the fixed
    # value differs from that buggy computation whenever there's more than
    # one algorithm.
    buggy = (12 * n_datasets / (k * (k + 1))) * sum((r**2 - k14) for r in avranks)
    assert abs(got - buggy) > 1e-6


def test_z_to_p_is_two_sided_and_symmetric():
    from covid_forecasting_joint_learning.pipeline.eval import z_to_p

    # A control that scores worse than the comparison (negative z) must get
    # the same small p-value as the mirror-image positive z, not p > 0.5.
    p_pos = z_to_p(2.5)
    p_neg = z_to_p(-2.5)
    assert abs(p_pos - p_neg) < 1e-12
    assert p_pos < 0.05, p_pos

    # z = 0 (no difference) must be non-significant.
    assert z_to_p(0.0) > 0.9


def test_sign_test_detects_a_consistent_difference():
    from covid_forecasting_joint_learning.pipeline.eval import sign_test

    # a beats b (lower score) in 9 of 10 datasets: should be significant.
    a = [1] * 9 + [5]
    b = [2] * 10
    p = sign_test(a, b)
    assert p < 0.05, p

    # Identical scores: nothing to test, must not claim significance.
    assert sign_test([1, 2, 3], [1, 2, 3]) == 1.0


def test_wilcoxon_test_detects_a_consistent_difference():
    from covid_forecasting_joint_learning.pipeline.eval import wilcoxon_test

    a = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.15, 1.0]
    b = [2.0, 2.2, 1.9, 2.1, 2.05, 1.95, 2.15, 2.0]
    assert wilcoxon_test(a, b) < 0.05
    assert wilcoxon_test(a, a) == 1.0


def test_best_index_picks_lowest_avg_rank():
    from covid_forecasting_joint_learning.pipeline.eval import best_index

    assert best_index([2.5, 1.2, 3.0]) == 1


if __name__ == "__main__":
    install_stubs()
    test_friedman_chi_square_matches_hand_computation()
    test_z_to_p_is_two_sided_and_symmetric()
    test_sign_test_detects_a_consistent_difference()
    test_wilcoxon_test_detects_a_consistent_difference()
    test_best_index_picks_lowest_avg_rank()
    print("ok")
