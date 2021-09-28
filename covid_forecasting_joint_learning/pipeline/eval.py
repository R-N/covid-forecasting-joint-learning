import Orange
import matplotlib.pyplot as plt

def bonferroni_dunn_cd(avranks, n_datasets, alpha="0.05"):
    return Orange.evaluation.compute_CD(
        avranks,
        n_datasets,
        alpha=alpha,
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
        alpha=alpha
    )

def plot_nemenyi(names, avranks, cd, width=5, textspace=1.5):
    return Orange.evaluation.graph_ranks(
        avranks,
        names,
        cd=cd,
        width=width,
        textspace=textspace
    )

