"""Regression check for pipeline/eval.py's spectral-entropy forecastability
reporting.

INVESTIGATION.md Quick wins: "Report spectral-entropy forecastability per
kabko -- Cheap to compute, and converts an unmeasured confounder into a
covariate" (see lines ~887-919 for the White and Leon, PLOS Comput Biol 2026
citation and rationale). `spectral_entropy` scores a single series in [0, 1],
higher = more forecastable; `forecastability_by_kabko` applies it per kabko
group.

Orange (needed only for the CD/plot helpers elsewhere in eval.py, not the
functions checked here) is stubbed so this runs without that dependency.
Run with:

    python tests/test_forecastability.py
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


def test_sine_wave_is_highly_forecastable():
    from covid_forecasting_joint_learning.pipeline.eval import spectral_entropy

    t = np.arange(200)
    sine = np.sin(2 * np.pi * t / 20.0)
    score = spectral_entropy(sine)
    assert score > 0.6, score


def test_white_noise_is_poorly_forecastable_and_scores_below_sine():
    from covid_forecasting_joint_learning.pipeline.eval import spectral_entropy

    t = np.arange(200)
    sine = np.sin(2 * np.pi * t / 20.0)
    sine_score = spectral_entropy(sine)

    noise = np.random.RandomState(0).randn(200)
    noise_score = spectral_entropy(noise)

    assert noise_score < 0.5, noise_score
    assert noise_score < sine_score, (noise_score, sine_score)


def test_constant_series_does_not_raise():
    from covid_forecasting_joint_learning.pipeline.eval import spectral_entropy

    constant = np.full(50, 7.0)
    score = spectral_entropy(constant)
    # After first-differencing a constant series is all zeros, so the PSD is
    # zero everywhere -- the degenerate-case guard returns 0.0 (minimally
    # forecastable rather than undefined), not NaN.
    assert score == 0.0, score


def test_too_short_series_returns_nan():
    from covid_forecasting_joint_learning.pipeline.eval import spectral_entropy

    assert np.isnan(spectral_entropy([1.0, 2.0]))


def test_forecastability_by_kabko_returns_series_indexed_by_kabko():
    from covid_forecasting_joint_learning.pipeline.eval import forecastability_by_kabko

    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    t = np.arange(200)
    sine = np.sin(2 * np.pi * t / 20.0)
    noise = np.random.RandomState(1).randn(200)

    data = pd.concat([
        pd.DataFrame({"kabko": "sine_kabko", "date": dates, "i": sine}),
        pd.DataFrame({"kabko": "noisy_kabko", "date": dates, "i": noise}),
    ], ignore_index=True)
    # shuffle rows to confirm date-sorting inside the groupby is exercised
    data = data.sample(frac=1.0, random_state=2).reset_index(drop=True)

    result = forecastability_by_kabko(data, value_col="i")
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert set(result.index) == {"sine_kabko", "noisy_kabko"}
    assert result["sine_kabko"] > result["noisy_kabko"], result


if __name__ == "__main__":
    install_stubs()
    test_sine_wave_is_highly_forecastable()
    test_white_noise_is_poorly_forecastable_and_scores_below_sine()
    test_constant_series_does_not_raise()
    test_too_short_series_returns_nan()
    test_forecastability_by_kabko_returns_series_indexed_by_kabko()
    print("ok")
