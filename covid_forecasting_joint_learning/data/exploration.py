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
from . import util as DataUtil
from collections import deque
import gc
import mpld3
# from matplotlib import rcParams
# import itertools


# Constants
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
FIG_SIZE = (13, 6)
LINE_WIDTH = 2


def handle_nan(x):
    return x if (x is not None and not np.isnan(x)) else 0

def my_abs(x):
    return abs(handle_nan(x))


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

# clist = rcParams['axes.prop_cycle']
# cgen = itertools.cycle(clist)


def plot_fill(df=None, lines=[], fills=[], title="", figsize=None, bbox=(0, -0.1), legend=True, return_ax=False, alpha=0.2, interactive=True, alpha_unsel=0.35, alpha_over=1.5, line_colors=None, grid=True, ipython=True):
    if lines is not None and len(lines) > 0 and isinstance(lines[0], str):
        lines = [df[line] for line in lines]
    if fills is not None and len(fills) > 0 and isinstance(fills[0], str):
        fills = [df[fill] for fill in fills]

    if interactive:
        legend = False
        alpha = min(1, alpha / alpha_unsel)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    max_val = max([line.max() for line in lines])
    min_val = min([line.min() for line in lines])
    # x_zero = pd.Series(min_val, index=df.index)

    for i in range(len(fills)):
        fill = fills[i]
        df_fill = fill.copy()
        if df_fill.max() <= 0:
            continue
        df_fill.loc[(df_fill != 0)] = max_val
        up = df_fill.loc[(df_fill != 0)]
        # if len(up) == 1:
        plus_1 = up.last_valid_index() + pd.DateOffset(1)
        if plus_1 in df_fill.index:
            df_fill.loc[plus_1] = max_val
        df_fill.loc[(df_fill == 0)] = min_val
        ax.fill_between(
            df.index,
            min_val,
            df_fill,
            where=df_fill > min_val,
            label=fill.name,
            # facecolor=next(cgen)['color'],
            alpha=alpha
        )
    for i in range(len(lines)):
        line = lines[i]
        if line_colors:
            ax.plot(line, label=line.name, color=line_colors[i])
        else:
            ax.plot(line, label=line.name)
    if grid:
        ax.grid(which="both", alpha=0.3)
    ax.set_title(title)
    if legend:
        # ax.legend(bbox_to_anchor=bbox, loc="upper left")
        if fills is not None and len(fills) > 0:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc="best")

    if interactive:
        fig = interactive_legend(fig, ax, alpha_unsel=alpha_unsel, alpha_over=alpha_over, ipython=ipython)
    if return_ax:
        return fig, ax
    return fig


def interactive_legend(fig, ax, alpha_unsel=0.5, alpha_over=1.5, ipython=True):
    handles, labels = ax.get_legend_handles_labels()  # return lines and labels
    interactive_legend = mpld3.plugins.InteractiveLegendPlugin(
        handles,
        labels,
        alpha_unsel=alpha_unsel,
        alpha_over=alpha_unsel,
        start_visible=True
    )
    mpld3.plugins.connect(fig, interactive_legend)
    if ipython:
        return mpld3.display(fig)
    else:
        return mpld3.fig_to_html(fig)

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
    return tsaplots.plot_acf(series, *args, title="ACF Plot for %s" % name, zero=False, **kwargs)


def plot_pacf(series, *args, name="", **kwargs):
    return tsaplots.plot_pacf(series, *args, title="PACF Plot for %s" % name, zero=False, **kwargs)


# Classical decomposition
def classical_decompose(series, period=None, *args, model="additive", **kwargs):
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
    return math.sqrt((residual * residual).sum() / len(residual))


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
def diff(series, order=1, shift=1, **kwargs):
    # ret = series - series.shift()
    for i in range(order):
        series = series.diff(periods=shift, **kwargs)
    series.dropna(inplace=True)
    return series


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
    n_lags = len(corr) - 1
    lags = np.linspace(0, n_lags, n_lags+1)
    ax.vlines(lags, [0], corr)
    ax.plot(lags, corr, marker="o", markersize=5, linestyle="None")


# Correlation of a to shifted b
# Note that shift moves the values forward, 
# so the value at an index contains what should've been at previous index
# so shift(1) means lagged 1
# So I'll just name this corr_lag instead of corr_shift

def lag_range(lag_start, lag_end):
    step = -1 if lag_start > lag_end else 1
    return range(lag_start, lag_end + step, step)

def corr_lag(
    x, y,
    lag_start=0, lag_end=-14,
    method="kendall"
    # pvalue=False
):
    lagged = [y.shift(i).dropna() for i in lag_range(lag_start, lag_end)]
    corr = [corr_pair(
        x[l.index],
        l,
        method=method
    ) for l in lagged]
    # if not pvalue:
    #     corr = [c[0] for c in corr]
    return corr


def corr_lag_multi(df, x_cols, y_cols, lag_start=0, lag_end=-14, method="kendall", lag_as_col=True):
    x_lag = np.array(list(lag_range(lag_start, lag_end)))
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


def corr_lag_best_multi(
    df,
    x_cols, y_cols,
    lag_start=0, lag_end=-14,
    method="kendall",
    reduction=None,
    abs_corr=False,
    as_dict=False,
    y_as_cols=True
):
    corr = np.array([[max(corr_lag(
        df[x_col],
        df[y_col],
        method=method,
        lag_start=lag_start,
        lag_end=lag_end
    ), key=lambda x: my_abs(x)) for y_col in y_cols] for x_col in x_cols])
    if reduction:
        if abs_corr:
            corr = np.abs(corr)
        if reduction == "sum":
            corr = np.sum(corr, axis=1, skipna=True)
        elif reduction == "max":
            corr = np.array([max(row, key=lambda x: my_abs(x)) for row in corr])
        elif reduction in ("avg", "mean"):
            corr = np.mean(corr, axis=1, skipna=True)
        else:
            raise ValueError(f"Invalid reduction '{reduction}'")
        y_cols = ["corr"]
    if as_dict:
        return dict(zip(x_cols, corr))
    df = pd.DataFrame(corr, columns=y_cols, index=x_cols)
    return df if y_as_cols else df.T


def corr_lag_sort_multi(
    df,
    x_cols, y_cols,
    lag_start=0, lag_end=-14,
    method="kendall",
    min_corr_percentile=0.50, max_corr_diff=0.5, min_corr=0.1, mean=True
):
    x_lags = np.array(list(lag_range(lag_start, lag_end)))
    # make x_cols combinations first
    corrs = [(x_col, y_col, corr_lag(
        df[x_col],
        df[y_col],
        method=method,
        lag_start=lag_start,
        lag_end=lag_end
    )) for x_col in x_cols for y_col in y_cols]
    corrs = [(x_col, y_col, zip(x_lags, corr)) for x_col, y_col, corr in corrs]
    corrs = [(x_col, y_col) + max(corr, key=lambda x: x[-1]) for x_col, y_col, corr in corrs]

    corr_values = np.abs(np.array([corr[-1] for corr in corrs]))
    min_corr_percentile = np.percentile(corr_values, min_corr_percentile)
    best_corr = np.max(corr_values)
    mean_corr = np.mean(corr_values) if mean else 0
    corrs = [corr + (my_abs(corr[-1]),) for corr in corrs]
    corrs = [(x_col, y_col, x_lag, corr) for x_col, y_col, x_lag, corr, abs_corr in corrs if abs_corr >= min_corr and abs_corr >= min_corr_percentile and best_corr - abs_corr <= max_corr_diff and abs_corr >= mean_corr]

    corrs = sorted(corrs, key=lambda x: my_abs(x[-1]), reverse=True)
    corrs = [{
        "x_col": x_col,
        "y_col": y_col,
        "x_lag": x_lag,
        "corr": corr
    } for x_col, y_col, x_lag, corr in corrs]
    return corrs

def explore_date_corr(
    kabko,
    single_dates,
    y_cols,
    labeled_dates=None,
    lag_start=0,
    lag_end=-14,
    method="kendall",
    min_corr=0.1,
    min_corr_diff=1e-5,
    date_set=None,
    return_corr=False,
    collect=False
):
    labeled_dates = labeled_dates or {x: (x,) for x in single_dates}
    date_set = date_set or set([tuple(sorted(x)) for x in labeled_dates.values()])

    stack = deque([(labeled_dates, min_corr, None)])
    ret = {}
    while stack:
        labeled_dates, min_corr, obj = stack.pop()

        df = kabko.add_dates(
            kabko.data,
            dates={k: list(v) for k, v in labeled_dates.items()}
        )

        corrs = corr_lag_best_multi(
            df,
            x_cols=list(labeled_dates.keys()),
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=method,
            reduction="max",
            abs_corr=False,
            as_dict=True
        )

        del df

        if corrs:
            for x_col, corr in corrs.items():
                date = labeled_dates[x_col]
                new_dates = [tuple(sorted(date + (x,))) for x in single_dates if x not in date]
                new_dates = [x for x in new_dates if x not in date_set]
                stack.append((
                    DataUtil.label_combinations(new_dates),
                    my_abs(corr),
                    {
                        "x_col": x_col,
                        "corr": corr,
                        "date": date
                    } if return_corr else date
                ))
            del corrs
        else:
            ret[x_col] = obj
        if collect:
            gc.collect()
    return ret


def label_date_grouping(grouping, penalty=False):
    grouping_1 = {i: [] for i in grouping.values()}
    for date, group in grouping.items():
        grouping_1[group].append(date)
    if penalty and 0 in grouping_1:
        del grouping_1[0]
    grouping_2 = [x for x in grouping_1.values() if x]
    labeled_dates = DataUtil.label_combinations(grouping_2)
    return labeled_dates


def corr_lag_best_multi_kabko(
    kabkos,
    labeled_dates,
    y_cols,
    lag_start=0, lag_end=-14,
    method="kendall",
    kabko_reduction="max",
    y_reduction="max"
):
    corrs_0 = None
    for kabko in kabkos:
        df = kabko.add_dates(
            kabko.data,
            dates={k: list(v) for k, v in labeled_dates.items()}
        )

        corrs = corr_lag_best_multi(
            df,
            x_cols=list(labeled_dates.keys()),
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=method,
            reduction=y_reduction,
            abs_corr=True,
            as_dict=True
        )
        del df

        if corrs_0:
            if kabko_reduction in ("avg", "mean", "sum"):
                for x_col, corr in corrs.items():
                    corrs_0[x_col] += corr
            elif kabko_reduction == "max":
                for x_col, corr in corrs.items():
                    corrs_0[x_col] = max((corr, corrs_0[x_col]))
            else:
                raise ValueError(f"Invalid kabko_reduction '{kabko_reduction}'")
        else:
            corrs_0 = corrs
        del corrs

    if kabko_reduction in ("avg", "mean"):
        scale = len(kabkos)
        corrs_0 = {k: v / scale for k, v in corrs_0.items()}

    return corrs_0


def corr_lag_best_multi_dfs(
    dfs,
    x_cols,
    y_cols,
    lag_start=0, lag_end=-14,
    method="kendall",
    kabko_reduction="max",
    as_dict=False
):
    corrs_0 = None
    for df in dfs:

        corrs = corr_lag_best_multi(
            df,
            x_cols=x_cols,
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=method,
            as_dict=True
        )
        del df

        if corrs_0:
            if kabko_reduction in ("avg", "mean", "sum"):
                for x_col, corr in corrs.items():
                    corrs_0[x_col] = [sum(c) for c in zip(corrs_0[x_col], corr)]
            elif kabko_reduction == "max":
                for x_col, corr in corrs.items():
                    corrs_0[x_col] = [max(c, key=lambda x: my_abs(x)) for c in zip(corrs_0[x_col], corr)]
            else:
                raise ValueError(f"Invalid kabko_reduction '{kabko_reduction}'")
        else:
            corrs_0 = corrs
        del corrs

    if kabko_reduction in ("avg", "mean"):
        scale = len(dfs)
        corrs_0 = {k: [c / scale for c in v] for k, v in corrs_0.items()}

    if as_dict:
        return corrs_0
    return pd.DataFrame(corrs_0, index=y_cols).T


def score_labeled_dates(
    corrs,
    min_corr=0.1,
    penalty=False
):
    if penalty:
        return sum([corr - min_corr for corr in corrs.values()])
    else:
        return sum([corr for corr in corrs.values() if corr >= min_corr])


def make_date_corr_objective(
    kabkos,
    single_dates,
    y_cols,
    lag_start=0,
    lag_end=-14,
    method="kendall",
    kabko_reduction="max",
    y_reduction="max",
    min_corr=0.1,
    penalty=False
):
    count = len(single_dates)
    groups = list(range((count + 1) if penalty else count))
    def objective(trial):
        grouping = {single_dates[i]: trial.suggest_categorical(str(single_dates[i]), groups) for i in range(count)}
        labeled_dates = label_date_grouping(grouping, penalty=penalty)

        corrs = corr_lag_best_multi_kabko(
            kabkos,
            labeled_dates,
            y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=method,
            kabko_reduction=kabko_reduction,
            y_reduction=y_reduction
        )

        return score_labeled_dates(
            corrs,
            min_corr=min_corr,
            penalty=penalty
        )

    return objective


def filter_date_corr(
    labeled_dates,
    corrs,
    min_corr=0.1,
):
    return {k: labeled_dates[k] for k, v in corrs.items() if v >= min_corr}
