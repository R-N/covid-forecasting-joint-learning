import pandas as pd
from matplotlib import pyplot as plt
# from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller as adf, acf, pacf
from statsmodels.graphics import tsaplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
from statsmodels.stats.diagnostic import kstest_normal as ks_test

import math

# Constants
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
FIG_SIZE = (13, 6)
LINE_WIDTH = 2

# Init?
register_matplotlib_converters()


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
    ret = []
    ret.add("Augmented Dickey-Fuller Test for " + name)
    ret.add('ADF Statistic: {:.10f}'.format(adf[0]))
    ret.add('p-value: {:.10f}'.format(adf[1]))
    ret.add('Critical Values:')
    for key, value in adf[4].items():
        ret.add('\t{}: {:.10f}'.format(key, value))
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
    ret = []
    ret.add("Residual stats for " + name)
    ret.add("Residual RMSE: ", rmse(residual))
    ret.add("Residual KS Test Stat: %s" % ks_test(residual)[0])
    ret.add("Residual KS Test P: %s" % ks_test(residual)[1])
    ret = '\n'.join(ret)
    return ret


# Differencing
def diff(series, period=1, **kwargs):
    # ret = series - series.shift()
    ret = series.diff(period=period, **kwargs)
    ret.dropna(inplace=True)
    return ret
