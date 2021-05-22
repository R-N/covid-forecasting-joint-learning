import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
from statsmodels.stats.diagnostic import kstest_normal as ks_test
# Functionalities only imported shamelessly
from statsmodels.tsa.stattools import adfuller as adf, acf, pacf
from scipy.stats import pearsonr, spearmanr, kendalltau


# Constants
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
FIG_SIZE = (13, 6)
LINE_WIDTH = 2


# IPython
def init_ipython():
    from IPython.display import HTML, display

    def set_css():
        display(HTML('''
        <style>
            pre {
                    white-space: pre-wrap;
            }
        </style>
        '''))

    get_ipython().events.register('pre_run_cell', set_css)
    pd.options.mode.use_inf_as_na = True


# Matplotlib
def init_matplotlib():
    register_matplotlib_converters()
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('figure', figsize=FIG_SIZE)
    plt.rc('lines', linewidth=LINE_WIDTH)


# ADF test
def print_adf(adf, name=""):
    ret = [
        "Augmented Dickey-Fuller Test for %s" % name,
        "ADF Statistic: {:.10f}".format(adf[0]),
        "p-value: {:.10f}".format(adf[1]),
        "Critical Values:",
        *['\t{}: {:.10f}'.format(key, value) for key, value in adf[4].items()]
    ]
    '\n'.join(ret)
    return ret


# Rolling stats
def rolling_stats(series, window, *args, **kwargs):
    rolling = series.rolling(*args, window=window, **kwargs)
    return rolling.mean(), rolling.std()


def plot_rolling_stats(series, mean, std, name="", window="?"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(series, color='blue', label=name)
    ax.plot(mean, color='red', label='Rolling Mean')
    ax.plot(std, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.title("Rolling Mean & Rolling Standard Deviation for %s with window=%s" % (name, window))
    return fig


# ACF plots
def plot_acf(series, *args, name="", **kwargs):
    return tsaplots.plot_acf(series, *args, title="ACF Plot for " + name, **kwargs)


def plot_pacf(series, *args, name="", **kwargs):
    return tsaplots.plot_pacf(series, *args, title="PACF Plot for " + name, **kwargs)


# Classical decomposition
def classical_decompose(series, period, *args, model="additive", **kwargs):
    return seasonal_decompose(series, *args, period=period, model=model, **kwargs)


def plot_classical_decompose(decomposed, name=""):
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
    decomposed.trend.plot(ax=ax1, ylabel="trend")
    decomposed.seasonal.plot(ax=ax2, ylabel="seasonality")
    decomposed.resid.plot(ax=ax3, ylabel="residual")
    ax1.title("Decomposed %s" % name)

    return fig


# Residual stats
def rmse(residual):
    return math.sqrt((residual * residual).sum())


def print_residual_stats(residual, name=""):
    ks = ks_test(residual)
    ret = [
        "Residual stats for %s" % name,
        "Residual RMSE: %s" % rmse(residual),
        "Residual KS Test Stat: %s" % ks[0],
        "Residual KS Test P: %s" % ks[1]
    ]
    ret = '\n'.join(ret)
    return ret


# Differencing
def diff(series, period=1, **kwargs):
    # ret = series - series.shift()
    ret = series.diff(period=period, **kwargs)
    ret.dropna(inplace=True)
    return ret


# Correlation
def corr_pair(x, y, method="pearson", **kwargs):
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return x.corr(y, method=method)
    elif method == "pearson":
        return pearsonr(x, y)
    elif method == "spearman":
        return spearmanr(x, y)
    elif method == "kendall":
        return kendalltau(x, y)
    else:
        raise Exception("Invalid correlation method %s" % method)


def corr_all(df, method="pearson", **kwargs):
    assert isinstance(df, pd.DataFrame)
    return df.corr(method=method)


def corr_matrix(corr):
    fig, ax = plt.subplots(1, 1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, cmap=cmap, fmt='.3f')
    return fig


def scatter_matrix(df, lib="seaborn", **kwargs):
    if lib == "pandas":
        fig, ax = plt.subplots(1, 1)
        plots = df.scatter_matrix(df, ax=ax, **kwargs)
        return fig
    elif lib == "seaborn":
        return sns.pairplot(df, **kwargs)
