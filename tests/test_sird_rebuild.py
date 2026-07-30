"""Regression check for SIRD reconstruction bounds.

The network's three rate outputs are unconstrained, so a prediction can be
negative or remove more from I than I holds. `sird.rebuild` clamps for that, and
this pins the compartment invariants it has to keep. Run with:

    python tests/test_sird_rebuild.py

pandas is stubbed so this runs in a bare interpreter; `rebuild` only touches it
for the DataFrame branch, which this check does not use.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = types.ModuleType("pandas")
        # Not `object`: rebuild isinstance-checks against it, and every value is
        # an object.
        sys.modules["pandas"].DataFrame = type("DataFrame", (), {})


N = 1000000.0
PREV = [900000.0, 1000.0, 500.0, 100.0]  # s, i, r, d


def test_valid_rates_are_untouched():
    from covid_forecasting_joint_learning.pipeline.sird import rebuild

    # Hand-computed one step: delta_r = 20, delta_d = 5, delta_i_in = 0.1*1000*0.9
    (s, i, r, d), = rebuild([(0.1, 0.02, 0.005)], list(PREV), N, return_s=True)
    assert abs(s - (900000.0 - 90.0)) < 1e-6, s
    assert abs(i - (1000.0 + 90.0 - 25.0)) < 1e-6, i
    assert abs(r - 520.0) < 1e-6, r
    assert abs(d - 105.0) < 1e-6, d


def test_negative_rates_do_not_shrink_cumulative_counts():
    from covid_forecasting_joint_learning.pipeline.sird import rebuild

    steps = rebuild([(-1.0, -0.5, -0.5)] * 5, list(PREV), N, return_s=True)
    prev = PREV
    for s, i, r, d in steps:
        assert r >= prev[2], "recovered count fell"
        assert d >= prev[3], "dead count fell"
        assert s <= prev[0], "susceptible count rose"
        prev = [s, i, r, d]


def test_compartments_stay_non_negative():
    from covid_forecasting_joint_learning.pipeline.sird import rebuild

    # Removal rates far above 1 would empty I past zero, and a huge beta would
    # take more out of S than S holds.
    steps = rebuild([(1000.0, 40.0, 40.0)] * 10, list(PREV), N, return_s=True)
    for s, i, r, d in steps:
        assert i >= 0, "infected count went negative"
        assert s >= 0, "susceptible count went negative"


def test_nan_still_propagates():
    from covid_forecasting_joint_learning.pipeline.sird import rebuild

    # A NaN prediction has to stay visible rather than being clamped to zero.
    (i, r, d), = rebuild([(float("nan"), 0.02, 0.005)], list(PREV), N)
    assert i != i, "NaN rate was silently clamped"


if __name__ == "__main__":
    install_stubs()
    test_valid_rates_are_untouched()
    test_negative_rates_do_not_shrink_cumulative_counts()
    test_compartments_stay_non_negative()
    test_nan_still_propagates()
    print("ok")
