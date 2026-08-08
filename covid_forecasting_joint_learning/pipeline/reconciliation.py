"""Post-hoc hierarchical forecast reconciliation (MinT, shrinkage variant).

INVESTIGATION.md, New and actionable: the kabko/province/national hierarchy
is already loaded from `covid_indo`/`covid_jatim` alongside the per-kabko
series, but the aggregation structure is otherwise unused. MinT projects
independently produced base forecasts onto the aggregation constraints and
is provably no worse than the base forecasts in expected squared error under
standard conditions (Wickramasuriya, Athanasopoulos and Hyndman, MinT). It
is pure post-hoc numerics -- no model code, no I/O -- so it composes with
every arm of the comparison equally.

Caveat noted in INVESTIGATION.md: the optimality result assumes a
well-estimated forecast-error covariance, and estimating a full covariance
over ~38 series from a short test period is exactly the regime where the
sample covariance is unstable. This module therefore only exposes the
Schafer-Strimmer shrinkage-to-identity covariance estimator, not full MinT
with a raw sample covariance.
"""
import numpy as np


def shrink_correlation(residuals):
    residuals = np.asarray(residuals, dtype=float)
    n_obs = residuals.shape[0]
    if n_obs < 2:
        n_series = residuals.shape[1]
        return np.eye(n_series)

    # Population-style (ddof=0) standardization is the Schafer-Strimmer
    # convention: with z_i(t) = (x_i(t) - mean_i) / std_i(ddof=0), the
    # correlation is simply r_hat = z.T @ z / n_obs and the diagonal comes
    # out exactly 1, which the pairwise-product variance below relies on.
    std = residuals.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    z = (residuals - residuals.mean(axis=0)) / std_safe

    r_hat = (z.T @ z) / n_obs

    n_series = r_hat.shape[0]
    iu = np.triu_indices(n_series, k=1)
    r_offdiag = r_hat[iu]

    # Var_hat(r_ij): empirical variance, across the n_obs periods, of the
    # per-period products z_i(t) * z_j(t) (r_hat_ij is their mean), scaled
    # by the standard Schafer-Strimmer small-sample correction factor
    # n_obs / (n_obs - 1) ** 2 -- this is the variance-of-the-mean
    # correction (Var(mean of n_obs values) ~ Var(value)/n_obs, adjusted
    # for the n_obs-1 denominator already used by `.var(ddof=1)` below).
    # It does not change the [0, 1] clipping or the shrink-towards-identity
    # direction, which are the properties that matter for downstream use.
    products = z[:, iu[0]] * z[:, iu[1]]  # (n_obs, n_pairs)
    var_r = products.var(axis=0, ddof=1) * n_obs / (n_obs - 1) ** 2

    numerator = var_r.sum()
    denominator = (r_offdiag ** 2).sum()
    if denominator == 0:
        lam = 1.0
    else:
        lam = numerator / denominator
    lam = np.clip(lam, 0.0, 1.0)

    r_shrink = (1 - lam) * r_hat
    np.fill_diagonal(r_shrink, 1.0)
    return r_shrink


def shrink_covariance(residuals):
    residuals = np.asarray(residuals, dtype=float)
    r_shrink = shrink_correlation(residuals)

    eps = 1e-12
    std = residuals.std(axis=0, ddof=1) if residuals.shape[0] >= 2 else np.zeros(residuals.shape[1])
    std_safe = std + eps

    d_sqrt = np.diag(std_safe)
    return d_sqrt @ r_shrink @ d_sqrt


def mint_reconcile(base_forecasts, S, W):
    base_forecasts = np.asarray(base_forecasts, dtype=float)
    S = np.asarray(S, dtype=float)
    W = np.asarray(W, dtype=float)

    was_1d = base_forecasts.ndim == 1
    y_hat = base_forecasts[:, None] if was_1d else base_forecasts

    W_pinv = np.linalg.pinv(W)
    inner = S.T @ W_pinv @ S  # (n_bottom, n_bottom)
    inner_pinv = np.linalg.pinv(inner)

    combination = S @ inner_pinv @ S.T @ W_pinv  # (n_series, n_series)
    y_tilde = combination @ y_hat

    return y_tilde[:, 0] if was_1d else y_tilde


def two_level_summing_matrix(n_bottom):
    top = np.ones((1, n_bottom))
    bottom = np.eye(n_bottom)
    return np.vstack([top, bottom])
