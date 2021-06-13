import math
import numpy as np
import pandas as pd
from . import sird
# Only imported shamelessly
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..data import cols as DataCol


# Zero handling
def trim_zero(df, labels=DataCol.IRD):
    non_zero = df[labels] if labels is not None else df
    non_zero = (non_zero != 0).any(1)
    non_zero = df[non_zero]
    df = df[non_zero.first_valid_index():non_zero.last_valid_index()].copy()
    return df


def zero_to_nan(series):
    non_zero = series != 0
    series = series.copy()
    series[non_zero.first_valid_index():non_zero.last_valid_index()].loc[series == 0] = np.NaN
    return series


def fill_zero(
    df,
    labels=[
        *DataCol.IRD,
        *DataCol.VAC_ALL,
        DataCol.I_TOT_GLOBAL,
        DataCol.TEST
    ], 
    method="linear"
):
    df = df.copy()
    # TODO: change middle zeroes to NaN
    for l in labels:
        df.loc[:, l] = zero_to_nan(df[l])
    df = df[labels] if labels is not None else df
    df.interpolate(method=method, limit_direction='forward', axis=0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def handle_zero(
    df,
    trim_labels=DataCol.IRD,
    fill_labels=[
        *DataCol.IRD,
        *DataCol.VAC_ALL,
        DataCol.I_TOT_GLOBAL,
        DataCol.TEST
    ],
    interpolation_method="linear"
):
    df = fill_zero(df, labels=fill_labels, method=interpolation_method)
    df = trim_zero(df, labels=trim_labels)
    return df


# Month splitting
class Group:
    def __init__(self, id, members, clusters=None, clustering_info=None):
        self.id = id
        self.members = members
        self.clusters = clusters
        self.clustering_info = clustering_info


# Note that slicing with date index includes the second part as opposed to integer index
def split_groups(
    kabko_dfs,
    limit_length=[60, 180, 366],
    limit_date=["2021-01-21"]
):
    groups = [
        *[[(kabko, df[:l]) for kabko, df in kabko_dfs] for l in limit_length],
        *[[(kabko, df[:l]) for kabko, df in kabko_dfs] for l in limit_date]
    ]
    groups = [Group(i, groups[i]) for i in range(len(groups))]
    return groups


# Train splitting
def calc_split(
    df,
    val_portion=0.25,
    test_portion=0.25,
    past_size=30
):
    n = len(df) - past_size
    val_len, test_len = int(val_portion * n), int(test_portion * n)
    train_len = n - (val_len + test_len)
    val_start, test_start = train_len+past_size, train_len+val_len+past_size
    # Note that slicing with date index includes the second part as opposed to integer index
    train_end, val_end = df.index[val_start-1], df.index[test_start-1]
    val_start, test_start = df.index[val_start], df.index[test_start]
    return train_end, val_start, val_end, test_start


def generate_dataset(
    df,
    future_start=None, future_end=None,
    past_size=30, future_size=14,
    stride=1,
    labels=DataCol.SIRD_VARS
):
    len_df = len(df)
    future_start = max(future_start or past_size, past_size)
    future_end = min(future_end or len_df, len_df)

    future_len = future_end - future_start
    count = (future_len - future_size + stride) // stride
    ids = [i*stride for i in range(count)]

    past_start_1, past_end_1 = future_start - past_size, future_start
    future_start_1, future_end_1 = future_start, future_start + future_size
    past = [df.iloc[past_start_1+i:past_end_1+i].to_numpy() for i in ids]
    future = [df.iloc[future_start_1+i:future_end_1+i][labels].to_numpy() for i in ids]

    return list(zip(past, future))


def split_dataset(
    df,
    splits,
    past_size=30, future_size=14,
    stride=1,
    labels=DataCol.SIRD_VARS
):
    splits = [df.index.get_loc(s) for s in splits]
    train_end, val_start, val_end, test_start = splits
    train_set = generate_dataset(
        df[:val_start],
        future_start=None, future_end=val_start,
        past_size=past_size, future_size=future_size,
        stride=stride,
        labels=labels
    )
    val_set = generate_dataset(
        df[:test_start],
        future_start=val_start, future_end=test_start,
        past_size=past_size, future_size=future_size,
        stride=stride,
        labels=labels
    )
    test_set = generate_dataset(
        df,
        future_start=test_start, future_end=None,
        past_size=past_size, future_size=future_size,
        stride=stride,
        labels=labels
    )
    return train_set, val_set, test_set


# Differencing
def diff(series, period=1, **kwargs):
    # ret = series - series.shift()
    ret = series.diff(period=period, **kwargs)
    ret.dropna(inplace=True)
    return ret
