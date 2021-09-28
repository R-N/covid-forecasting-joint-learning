from math import sqrt
from scipy.stats import norm as z, f

def friedman_adj_f(friedman_chi_square, k_algorithms, n_datasets):
    return ((n_datasets - 1) * friedman_chi_square) / (n_datasets * (k_algorithms - 1) - friedman_chi_square)

def bonferroni_dunn_z(rank_i, rank_j, k_algorithms, n_datasets):
    return (rank_i - rank_j) / sqrt((k_algorithms * (k_algorithms + 1)) / (6 * n_datasets))

def z_to_p(z_stat):
    return z.cdf(z_stat)

def f_to_p(f_stat):
    return f.cdf(f_stat)
