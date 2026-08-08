import Orange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.stats import norm as z, f, binomtest, wilcoxon
from scipy.signal import periodogram

def friedman_chi_square(avranks, n_datasets):
    k_algorithms = len(avranks)
    k14 = k_algorithms * (k_algorithms + 1)**2 / 4
    # Subtract k14 once, not once per algorithm: the term is
    # sum(r_j^2) - k(k+1)^2/4, not sum(r_j^2 - k(k+1)^2/4).
    rk = sum(r**2 for r in avranks) - k14
    x2f = (12 * n_datasets / (k_algorithms * (k_algorithms + 1))) * rk
    return x2f

def _friedman_adj_f(friedman_chi_square, k_algorithms, n_datasets):
    return ((n_datasets - 1) * friedman_chi_square) / (n_datasets * (k_algorithms - 1) - friedman_chi_square)

def friedman_adj_f(avranks, n_datasets):
    k_algorithms = len(avranks)
    x2f = friedman_chi_square(avranks, n_datasets)
    return _friedman_adj_f(x2f, k_algorithms, n_datasets)

def _test_z(rank_i, rank_j, k_algorithms, n_datasets):
    return (rank_i - rank_j) / sqrt((k_algorithms * (k_algorithms + 1)) / (6 * n_datasets))

def test_z(avranks, n_datasets, control_index=0):
    k_algorithms = len(avranks)
    rank_c = avranks[control_index]
    zs = [_test_z(rank_i, rank_c, k_algorithms, n_datasets) for rank_i in avranks]
    return zs

def _test_p(rank_i, rank_j, k_algorithms, n_datasets):
    z = _test_z(rank_i, rank_j, k_algorithms, n_datasets)
    return z_to_p(z)

def test_p(avranks, n_datasets, control_index=0):
    zs = test_z(avranks, n_datasets, control_index=control_index)
    ps = [z_to_p(z) for z in zs]
    return ps

def z_to_p(z_stat):
    # Two-sided: the one-sided 1 - Phi(z) form returned p > 0.5 whenever the
    # control ranked worse than the comparison (z < 0), silently reporting
    # "not significant" for differences that were significant the other way.
    return 2 * (1 - z.cdf(abs(z_stat)))

def dfn(k):
    return k - 1

def dfd(k, n):
    return (k - 1) * (n - 1)

def f_to_p(f_stat, dfn, dfd):
    return 1 - f.cdf(f_stat, dfn, dfd)

def friedman_adj_p(avranks, n_datasets):
    f = friedman_adj_f(avranks, n_datasets)
    k_algorithms = len(avranks)
    p = f_to_p(f, dfn(k_algorithms), dfd(k_algorithms, n_datasets))
    return p

def best_index(avranks):
    """Index of the best-ranked algorithm (lowest average rank)."""
    return min(range(len(avranks)), key=lambda i: avranks[i])

def bonferroni_dunn_cd(avranks, n_datasets, alpha="0.05"):
    return Orange.evaluation.compute_CD(
        avranks,
        n_datasets,
        alpha=str(alpha),
        test="bonferroni-dunn"
    )

def plot_bonferroni_dunn(names, avranks, cd, control_index=0, width=5, textspace=1.5, file_name=None, **kwargs):
    return Orange.evaluation.graph_ranks(
        avranks,
        names,
        cd=cd,
        width=width,
        textspace=textspace,
        cdmethod=control_index,
        filename=file_name,
        **kwargs
    )

def nemenyi_cd(avranks, n_datasets, alpha="0.05"):
    return Orange.evaluation.compute_CD(
        avranks,
        n_datasets,
        alpha=str(alpha)
    )

def plot_nemenyi(names, avranks, cd, width=5, textspace=1.5, file_name=None, **kwargs):
    return Orange.evaluation.graph_ranks(
        avranks,
        names,
        cd=cd,
        width=width,
        textspace=textspace,
        filename=file_name,
        **kwargs
    )

def mcb_cd(avranks, n_datasets, alpha="0.05"):
    """Multiple Comparisons with the Best (Koning et al. 2005): every
    algorithm is compared against the best-ranked one instead of an a priori
    control. Implemented as Bonferroni-Dunn with the control fixed to the
    best average rank -- the standard practical form of MCB when only a
    Nemenyi/Bonferroni-Dunn implementation (Orange) is available.
    """
    return bonferroni_dunn_cd(avranks, n_datasets, alpha=alpha)

def plot_mcb(names, avranks, cd, width=5, textspace=1.5, file_name=None, **kwargs):
    return plot_bonferroni_dunn(
        names,
        avranks,
        cd,
        control_index=best_index(avranks),
        width=width,
        textspace=textspace,
        file_name=file_name,
        **kwargs
    )

def sign_test(scores_a, scores_b):
    """Two-sided sign test on paired per-dataset scores (lower = better).
    Unlike the mean-rank tests above, its verdict does not depend on which
    other methods were in the comparison pool (Benavoli, Corani and Mangili,
    JMLR 2016). Ties are dropped, matching the classic sign test.
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b) if a != b]
    n = len(diffs)
    if n == 0:
        return 1.0
    wins_a = sum(1 for d in diffs if d < 0)
    return binomtest(wins_a, n, 0.5).pvalue

def wilcoxon_test(scores_a, scores_b):
    """Two-sided Wilcoxon signed-rank test on paired per-dataset scores.
    Pool-independent like the sign test, and more powerful when the
    magnitude of each pairwise difference (not just its sign) is meaningful.
    """
    if all(a == b for a, b in zip(scores_a, scores_b)):
        return 1.0
    return wilcoxon(scores_a, scores_b).pvalue

def spectral_entropy(series, eps=1e-12):
    """Normalized Shannon spectral-entropy forecastability score (White and
    Leon, PLOS Comput Biol 2026; INVESTIGATION.md "Forecastability is
    measurable..."). First-differences the series to detrend it (raw case
    counts are non-stationary and forecastability measures assume otherwise),
    takes its periodogram, and treats the normalized power spectral density
    as a probability distribution: power concentrated in one frequency means
    low spectral entropy (predictable), power spread uniformly means high
    spectral entropy (white noise, unpredictable). Returns 1 minus the
    normalized entropy, so higher = more forecastable, in [0, 1].
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 3:
        return np.nan
    diffed = np.diff(series)
    _, psd = periodogram(diffed)
    total_power = psd.sum()
    if total_power <= 0:
        return 0.0
    p = psd / total_power
    entropy = -np.sum(p * np.log(p + eps))
    normalized_entropy = entropy / np.log(len(psd))
    return 1 - normalized_entropy

def forecastability_by_kabko(data, value_col="i"):
    """Per-kabko spectral-entropy forecastability (INVESTIGATION.md Quick
    wins: "Report spectral-entropy forecastability per kabko"). Converts the
    unmeasured population-size/predictability confounder noted in the
    White and Leon 2026 review into a reported per-kabko covariate.
    """
    scores = {}
    for kabko, group in data.groupby("kabko"):
        if "date" in group.columns:
            group = group.sort_values("date")
        scores[kabko] = spectral_entropy(group[value_col].to_numpy())
    return pd.Series(scores)

def ensemble_eval_logs(log_dfs):
    """Median-ensemble multiple `EvalLog.log_df` tables that share the
    `group`/`cluster`/`kabko`/`i`/`r`/`d` schema (`SIRDEvalLog`,
    `ARIMASIRDEvalLog`, `NaiveEvalLog`, and any neural per-seed log written
    into the same schema) into one table: the median i/r/d loss per
    (group, cluster, kabko) across seeds.

    INVESTIGATION.md, Recommendations #5 / Quick wins -- accuracy:
    "Median-ensemble the seeds the rerun already requires -- the runs are
    already being paid for; budget five." The US COVID-19 Forecast Hub, M4
    and M5 retrospectives all find an equally weighted median ensemble at
    least as accurate as any individual member, and every comparison arm
    in this repo already writes its per-(group,cluster,kabko) loss into
    this exact schema, so combining seeds here (rather than re-deriving it
    per arm) is the one place the ensemble has to be built. Feed the
    result's `i`/`r`/`d` columns into `sign_test`/`wilcoxon_test`/`mcb_cd`
    in place of a single seed's losses.
    """
    combined = pd.concat(log_dfs, ignore_index=True)
    return combined.groupby(["group", "cluster", "kabko"], as_index=False)[["i", "r", "d"]].median()
