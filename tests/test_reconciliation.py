"""Tests for pipeline/reconciliation.py -- pure numpy, no heavy deps, so no
install_stubs() is needed (see tests/test_naive_baseline.py for the pattern
this repo uses when stubbing IS required).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_shrink_correlation_intensity_bounds_and_shrinkage_direction():
    from covid_forecasting_joint_learning.pipeline.reconciliation import shrink_correlation

    rs = np.random.RandomState(0)
    # Correlated columns: shared latent factor + independent noise.
    n_obs = 50
    latent = rs.normal(size=n_obs)
    residuals = np.column_stack([
        latent + 0.1 * rs.normal(size=n_obs),
        latent + 0.1 * rs.normal(size=n_obs),
        rs.normal(size=n_obs),
        rs.normal(size=n_obs),
    ])

    std = residuals.std(axis=0, ddof=0)
    z = (residuals - residuals.mean(axis=0)) / std
    r_hat = (z.T @ z) / n_obs

    r_shrink = shrink_correlation(residuals)

    assert r_shrink.shape == (4, 4)
    assert np.allclose(np.diag(r_shrink), 1.0)

    n = r_hat.shape[0]
    iu = np.triu_indices(n, k=1)
    sample_mag = np.abs(r_hat[iu])
    shrunk_mag = np.abs(r_shrink[iu])
    assert np.all(shrunk_mag <= sample_mag + 1e-12)


def test_shrink_correlation_lambda_in_bounds_generic():
    from covid_forecasting_joint_learning.pipeline.reconciliation import shrink_correlation

    rs = np.random.RandomState(1)
    residuals = rs.normal(size=(30, 6))
    r_shrink = shrink_correlation(residuals)
    assert r_shrink.shape == (6, 6)
    assert np.all(np.isfinite(r_shrink))
    # Off-diagonal magnitudes stay within [0, 1] (valid correlation matrix range).
    n = r_shrink.shape[0]
    iu = np.triu_indices(n, k=1)
    assert np.all(np.abs(r_shrink[iu]) <= 1.0 + 1e-9)


def test_shrink_correlation_independent_columns_lambda_close_to_one():
    from covid_forecasting_joint_learning.pipeline.reconciliation import shrink_correlation

    rs = np.random.RandomState(0)
    n_obs = 200
    residuals = rs.normal(size=(n_obs, 5))  # independent columns, no real correlation signal

    std = residuals.std(axis=0, ddof=0)
    z = (residuals - residuals.mean(axis=0)) / std
    r_hat = (z.T @ z) / n_obs

    r_shrink = shrink_correlation(residuals)

    n = r_hat.shape[0]
    iu = np.triu_indices(n, k=1)
    # Recover lambda from the shrinkage relation R_shrink = lam*I + (1-lam)*R_hat
    # on an off-diagonal entry: r_shrink_ij = (1 - lam) * r_hat_ij.
    ratios = r_shrink[iu] / r_hat[iu]
    lam_recovered = 1 - np.mean(ratios)
    assert lam_recovered > 0.7  # little correlation signal -> heavy shrinkage


def test_shrink_correlation_guards_small_n_obs():
    from covid_forecasting_joint_learning.pipeline.reconciliation import shrink_correlation

    residuals = np.array([[1.0, 2.0, 3.0]])  # n_obs = 1
    r_shrink = shrink_correlation(residuals)
    assert np.allclose(r_shrink, np.eye(3))


def test_shrink_covariance_rescales_by_std():
    from covid_forecasting_joint_learning.pipeline.reconciliation import shrink_covariance

    rs = np.random.RandomState(0)
    residuals = rs.normal(scale=[1.0, 2.0, 3.0], size=(40, 3))
    W = shrink_covariance(residuals)
    assert W.shape == (3, 3)
    assert np.all(np.isfinite(W))
    # Diagonal should be close to the per-column sample variance (ddof=1).
    var = residuals.var(axis=0, ddof=1)
    assert np.allclose(np.diag(W), var, rtol=0.05)


def test_mint_reconcile_makes_incoherent_forecasts_coherent():
    from covid_forecasting_joint_learning.pipeline.reconciliation import (
        mint_reconcile,
        two_level_summing_matrix,
    )

    S = two_level_summing_matrix(4)
    assert S.shape == (5, 4)

    bottom = np.array([10.0, 20.0, 5.0, 8.0])
    aggregate = 100.0  # deliberately incoherent: bottom sums to 43, not 100
    base_forecasts = np.concatenate([[aggregate], bottom])
    assert not np.isclose(base_forecasts[0], np.sum(base_forecasts[1:]))

    W = np.eye(5)
    reconciled = mint_reconcile(base_forecasts, S, W)

    assert reconciled.shape == (5,)
    assert np.isclose(reconciled[0], np.sum(reconciled[1:]))


def test_mint_reconcile_noop_on_already_coherent_forecasts():
    from covid_forecasting_joint_learning.pipeline.reconciliation import (
        mint_reconcile,
        two_level_summing_matrix,
    )

    S = two_level_summing_matrix(4)
    bottom = np.array([10.0, 20.0, 5.0, 8.0])
    aggregate = np.sum(bottom)
    base_forecasts = np.concatenate([[aggregate], bottom])

    W = np.eye(5)
    reconciled = mint_reconcile(base_forecasts, S, W)

    assert np.allclose(reconciled, base_forecasts, atol=1e-8)


def test_mint_reconcile_handles_multi_step_horizon():
    from covid_forecasting_joint_learning.pipeline.reconciliation import (
        mint_reconcile,
        two_level_summing_matrix,
    )

    S = two_level_summing_matrix(4)
    horizon = 3
    rs = np.random.RandomState(2)
    bottom = rs.uniform(1, 20, size=(4, horizon))
    aggregate = bottom.sum(axis=0) + rs.uniform(5, 15, size=horizon)  # incoherent each step
    base_forecasts = np.vstack([aggregate[None, :], bottom])
    assert base_forecasts.shape == (5, horizon)

    W = np.eye(5)
    reconciled = mint_reconcile(base_forecasts, S, W)

    assert reconciled.shape == (5, horizon)
    for h in range(horizon):
        assert np.isclose(reconciled[0, h], np.sum(reconciled[1:, h]))


def test_two_level_summing_matrix_structure():
    from covid_forecasting_joint_learning.pipeline.reconciliation import two_level_summing_matrix

    S = two_level_summing_matrix(3)
    assert S.shape == (4, 3)
    assert np.array_equal(S[0], np.ones(3))
    assert np.array_equal(S[1:], np.eye(3))


def main():
    test_shrink_correlation_intensity_bounds_and_shrinkage_direction()
    test_shrink_correlation_lambda_in_bounds_generic()
    test_shrink_correlation_independent_columns_lambda_close_to_one()
    test_shrink_correlation_guards_small_n_obs()
    test_shrink_covariance_rescales_by_std()
    test_mint_reconcile_makes_incoherent_forecasts_coherent()
    test_mint_reconcile_noop_on_already_coherent_forecasts()
    test_mint_reconcile_handles_multi_step_horizon()
    test_two_level_summing_matrix_structure()
    print("ok")


if __name__ == "__main__":
    main()
