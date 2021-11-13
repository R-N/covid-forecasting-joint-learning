import Orange
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm as z, f

def friedman_chi_square(avranks, k_algorithms, n_datasets):
    k14 = k_algorithms * (k_algorithms + 1)**2 / 4
    rk = sum([(r**2 - k14) for r in avranks])
    x2f = (12 * n_datasets / (k_algorithms * (k_algorithms + 1))) * rk
    return x2f

def friedman_adj_f(friedman_chi_square, k_algorithms, n_datasets):
    return ((n_datasets - 1) * friedman_chi_square) / (n_datasets * (k_algorithms - 1) - friedman_chi_square)

def bonferroni_dunn_z(rank_i, rank_j, k_algorithms, n_datasets):
    return (rank_i - rank_j) / sqrt((k_algorithms * (k_algorithms + 1)) / (6 * n_datasets))

def bonferroni_dunn_z_multi(avranks, n_datasets, control_index=0):
    k_algorithms = len(avranks)
    rank_c = avranks[control_index]
    zs = [bonferroni_dunn_z(rank_i, rank_c, k_algorithms, n_datasets) for rank_i in avranks]
    return zs

def z_to_p(z_stat):
    return z.cdf(z_stat)

def f_to_p(f_stat):
    return f.cdf(f_stat)

def bonferroni_dunn_cd(avranks, n_datasets, alpha="0.05"):
    return Orange.evaluation.compute_CD(
        avranks,
        n_datasets,
        alpha=str(alpha),
        test="bonferroni-dunn"
    )

def plot_bonferroni_dunn(names, avranks, cd, control_index=0, width=5, textspace=1.5):
    return Orange.evaluation.graph_ranks(
        avranks,
        names,
        cd=cd,
        width=width,
        textspace=textspace,
        cdmethod=control_index
    )

def nemenyi_cd(avranks, n_datasets, alpha="0.05"):
    return Orange.evaluation.compute_CD(
        avranks,
        n_datasets,
        alpha=str(alpha)
    )

def plot_nemenyi(names, avranks, cd, width=5, textspace=1.5):
    return Orange.evaluation.graph_ranks(
        avranks,
        names,
        cd=cd,
        width=width,
        textspace=textspace
    )

