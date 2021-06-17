import math
import numpy as np
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
from matplotlib import rcParams
import itertools


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


# Normal plotting
# This doesn't yet support multiple fills
# It just gets mixed
clist = rcParams['axes.prop_cycle']
cgen = itertools.cycle(clist)

def plot_fill(df, lines=[], fills=[], title="", figsize=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df_fills = df[fills].copy()
    non_zero = (df_fills != 0).any(1)
    df_fills.loc[non_zero] = df[lines].max().max()
    zero = (df_fills == 0).any(1)
    min_val = df[lines].min().min()
    df_fills.loc[zero] = min_val
    # x_zero = pd.Series(min_val, index=df.index)
    for fill in fills:
        ax.fill_between(
            df.index,
            min_val,
            df_fills[fill],
            where=df_fills[fill] > min_val,
            label=fill,
            facecolor=next(cgen)['color'],
            alpha=0.15
          )
        # ax.fill(df_fills[fill], label=fill, alpha=0.15)
    for line in lines:
        ax.plot(df[line], label=line)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


# ADF test
def print_adf(adf, name=""):
    ret = [
        "Augmented Dickey-Fuller Test for %s" % name,
        "ADF Statistic: {:.10f}".format(adf[0]),
        "p-value: {:.10f}".format(adf[1]),
        "Critical Values:",
        *["\t{}: {:.10f}".format(key, value) for key, value in adf[4].items()]
    ]
    return '\n'.join(ret)


# Rolling stats
def rolling_stats(series, window, *args, **kwargs):
    rolling = series.rolling(*args, window=window, **kwargs)
    return rolling.mean(), rolling.std()


def plot_rolling_stats(series, mean, std, name="", window="?"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(series, color="blue", label=name)
    ax.plot(mean, color="red", label="Rolling Mean")
    ax.plot(std, color="black", label="Rolling Std")
    ax.legend(loc="best")
    ax.set_title("Rolling Mean & Rolling Standard Deviation for %s with window=%s" % (name, window))
    return fig


# ACF plots
def plot_acf(series, *args, name="", **kwargs):
    return tsaplots.plot_acf(series, *args, title="ACF Plot for %s" % name, **kwargs)


def plot_pacf(series, *args, name="", **kwargs):
    return tsaplots.plot_pacf(series, *args, title="PACF Plot for %s" % name, **kwargs)


# Classical decomposition
def classical_decompose(series, period, *args, model="additive", **kwargs):
    return seasonal_decompose(series, *args, period=period, model=model, **kwargs)


def plot_classical_decompose(decomposed, name=""):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    decomposed.trend.plot(ax=ax1, ylabel="trend")
    decomposed.seasonal.plot(ax=ax2, ylabel="seasonality")
    decomposed.resid.plot(ax=ax3, ylabel="residual")
    ax1.set_title("Decomposed %s" % name)

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
    return '\n'.join(ret)


# Differencing
def diff(series, period=1, **kwargs):
    # ret = series - series.shift()
    ret = series.diff(period=period, **kwargs)
    ret.dropna(inplace=True)
    return ret


# Correlation
def corr_pair(x, y, method="kendall", **kwargs):
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


def corr_multi(df, x_cols, y_cols, method="kendall", y_as_cols=True):
    corr = np.array([[corr_pair(df[x_col], df[y_col], method=method) for y_col in y_cols] for x_col in x_cols])
    corr = pd.DataFrame(corr, columns=y_cols, index=x_cols)
    return corr if y_as_cols else corr.T


def corr_all(df, method="kendall", **kwargs):
    assert isinstance(df, pd.DataFrame)
    return df.corr(method=method)


def corr_matrix(corr, figsize=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, cmap=cmap, fmt='.3f')
    return fig


def scatter_matrix(df, lib="seaborn", name="", **kwargs):
    if lib == "pandas":
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Scatter matrix for %s" % name)
        plots = pd.plotting.scatter_matrix(df, ax=ax, **kwargs)
        return fig
    elif lib == "seaborn":
        return sns.pairplot(df, **kwargs)


def plot_corr_lag_single(corr, ax):
    n_lags = len(corr)-1
    lags = np.linspace(0, n_lags, n_lags+1)
    ax.vlines(lags, [0], corr)
    ax.plot(lags, corr, marker="o", markersize=5, linestyle="None")


# Correlation of a to shifted b
# Note that shift moves the values forward, 
# so the value at an index contains what should've been at previous index
# so shift(1) means lagged 1
# So I'll just name this corr_lag instead of corr_shift
def corr_lag(
    x, y,
    lag_start=0, lag_end=-14,
    method="kendall"
    # pvalue=False
):
    step = -1 if lag_start > lag_end else 1
    lagged = [y.shift(i).dropna() for i in range(lag_start, lag_end + step, step)]
    corr = [corr_pair(
        x[l.index],
        l,
        method=method
    ) for l in lagged]
    # if not pvalue:
    #     corr = [c[0] for c in corr]
    return corr


def corr_lag_multi(df, x_cols, y_cols, lag_start=0, lag_end=-14, method="kendall", lag_as_col=True):
    step = -1 if lag_start > lag_end else 1
    x_lag = np.array(list(range(lag_start, lag_end + step, step)))
    corr = np.array([[corr_lag(
        df[x_col], df[y_col],
        lag_start=lag_start, lag_end=lag_end,
        method=method
        # pvalue=False
    ) for y_col in y_cols] for x_col in x_cols])
    corr = [pd.DataFrame(
        corr[x],
        index=["%s_x_%s" % (x_cols[x], y) for y in y_cols],
        columns=x_lag
    ) for x in range(0, len(x_cols))]
    corr = pd.concat(corr)
    return corr if lag_as_col else corr.T
