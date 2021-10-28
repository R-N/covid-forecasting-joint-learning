import math
import numpy as np
import pandas as pd
from . import sird
# Only imported shamelessly
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..data import cols as DataCol
import itertools
from .clustering import shortest, merge_clusters
from ..model.loss_common import mse, naive


# Zero handling
def trim_zero(df, labels=DataCol.IRD):
    non_zero = df[labels] if labels is not None else df
    non_zero = df[(non_zero != 0).any(1)]
    df = df[non_zero.first_valid_index():non_zero.last_valid_index()].copy()
    return df


def trim_zero_crit(df, labels=DataCol.IRD, crit_labels=[DataCol.I]):
    non_crit = [x for x in labels if x not in crit_labels]
    non_crit = df[non_crit]
    non_zero_non_crit = non_crit[(non_crit != 0).any(1)]
    non_zero_non_crit = df[non_zero_non_crit.first_valid_index():non_zero_non_crit.last_valid_index()]

    crit = non_zero_non_crit[crit_labels]
    zero_crit = crit[(crit == 0).any(1)]
    if len(zero_crit > 0):
        df = df[zero_crit.last_valid_index() + np.timedelta64(1, "D"):].copy()
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
    df_full = df
    df = df.loc[:, labels] if labels is not None else df
    df.interpolate(method=method, limit_direction='forward', axis=0, inplace=True)
    df.fillna(0, inplace=True)
    df_full.loc[:, labels] = df
    return df_full


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


# Days
def add_day_of_week(df, use_index=True):
    dates = df.index if use_index else df[DataCol.DATE].dt
    df[DataCol.DAY_DUM] = dates.dayofweek
    dum = pd.get_dummies(df[DataCol.DAY_DUM], prefix=DataCol.DAY)
    df.loc[:, dum.columns] = dum
    del df[DataCol.DAY_DUM]
    return df


def add_day_since(df):
    df[DataCol.DAY] = list(range(len(df.index)))
    return df


# Month splitting
class Group:
    def __init__(self, id, members, clusters=None, clustering_info=None):
        self.id = id
        self.members = members
        self.clusters = clusters or []
        self.clustering_info = clustering_info

    @property
    def sources(self):
        return list(itertools.chain.from_iterable([c.sources for c in self.clusters]))

    @property
    def targets(self):
        return [c.target for c in self.clusters]

    @property
    def target(self):
        return max(self.targets, key=lambda x: shortest(x))

    def merge_clusters(self):
        return merge_clusters(self)

    def copy(self):
        copy_dict = {k: k.copy() for k in self.members}
        group = Group(
            id=self.id,
            members=[copy_dict[k] for k in self.members],
            clustering_info=self.clustering_info
        )
        group.clusters = [c.copy(group=group, copy_dict=copy_dict) for c in self.clusters]
        return group


# Note that slicing with date index includes the second part as opposed to integer index
def split_groups(
    kabko_dfs,
    limit_length=[90, 180, 366],
    limit_date=["2021-01-21"]
):
    groups = [
        *[[(kabko, df[:l].copy()) for kabko, df in kabko_dfs] for l in limit_length],
        *[[(kabko, df[:l].copy()) for kabko, df in kabko_dfs] for l in limit_date]
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
    val_start, test_start = train_len + past_size, train_len + val_len + past_size
    # Note that slicing with date index includes the second part as opposed to integer index
    train_end, val_end = df.index[val_start - 1], df.index[test_start - 1]
    val_start, test_start = df.index[val_start], df.index[test_start]
    return train_end, val_start, val_end, test_start


def check_split_indices(kabko):
    for i in range(len(kabko.split_indices)):
        try:
            assert kabko.split_indices[i] in kabko.data.index
        except Exception:
            raise Exception(
                "Split indices %s=%s doesn't exist for %s.%s.%s" % (
                    i,
                    kabko.split_indices[i],
                    kabko.group.id,
                    kabko.cluster.id,
                    kabko.name
                )
            )


def slice_dataset(
    df,
    future_start=None, future_end=None,
    past_size=30, future_size=14,
    stride=1,
    limit_past=True,
    label_cols=DataCol.SIRD_VARS,
    final_cols=DataCol.IRD
):
    len_df = len(df)
    past_size = min(len_df - future_size, past_size)
    future_start = max(future_start or past_size, past_size)
    future_end = min(future_end or len_df, len_df)

    future_len = future_end - future_start
    count = (future_len - future_size + stride) // stride
    ids = [i * stride for i in range(count)]

    past_start_1, past_end_1 = future_start - past_size, future_start
    future_start_1, future_end_1 = future_start, future_start + future_size
    if limit_past:
        past = [df.iloc[past_start_1 + i:past_end_1 + i] for i in ids]
    else:
        past = [df.iloc[:past_end_1 + i] for i in ids]
    future = [df.iloc[future_start_1 + i:future_end_1 + i] for i in ids]

    mse_naive = [df[label_cols].iloc[:past_end_1 + i] for i in ids]
    mse_naive = [mse(naive(x.to_numpy())) for x in mse_naive]

    mse_naive_final = [df[final_cols].iloc[:past_end_1 + i] for i in ids]
    mse_naive_final = [mse(naive(x.to_numpy())) for x in mse_naive_final]

    return past, future, mse_naive, mse_naive_final



def label_dataset_0(
    past, future,
    mse_naive=None, mse_naive_final=None,
    past_cols=None,
    seed_size=None,
    label_cols=DataCol.SIRD_VARS,
    future_exo_cols=["psbb", "ppkm", "ppkm_mikro"],
    final_seed_cols=DataCol.SIRD,
    final_cols=DataCol.IRD
):
    final_seed = [x.iloc[-seed_size:] for x in past]
    indices = [x.index for x in future]
    past_size = len(past[0])
    seed_size = seed_size or past_size
    assert seed_size <= past_size

    past_seed = [x[label_cols].iloc[-seed_size:].to_numpy() for x in past]
    past_exo = [x[future_exo_cols].iloc[-seed_size:].to_numpy() for x in past]
    future_exo = [x[future_exo_cols].to_numpy() for x in future]
    if past_cols is not None:
        past = [x[past_cols] for x in past]
    else:
        past_cols = past[0].columns
    future_final = [x[final_cols].to_numpy() for x in future]
    final_seed = [x[final_seed_cols].to_numpy() for x in final_seed]
    past = [x.to_numpy() for x in past]
    future = [x[label_cols].to_numpy() for x in future]

    ret = [(
        past[i],
        past_seed[i],
        past_exo[i],
        future[i],
        future_exo[i],
        final_seed[i],
        future_final[i],
        mse_naive[i] if mse_naive else mse_naive,
        mse_naive_final[i] if mse_naive_final else mse_naive_final,
        indices[i]
    ) for i in range(len(past))]

    labels = [
        past_cols,
        label_cols,
        future_exo_cols,
        label_cols,
        future_exo_cols,
        final_seed_cols,
        final_cols,
        label_cols,
        final_cols,
        "index"
    ]

    return ret, labels


def label_dataset_1(
    past, future,
    mse_naive=None, mse_naive_final=None,
    label_cols=DataCol.SIRD,
    **kwargs
):
    indices = [x.index for x in future]

    past = [x[label_cols].to_numpy() for x in past]
    future = [x[label_cols].to_numpy() for x in future]

    ret = [(
        past[i],
        future[i],
        mse_naive[i] if mse_naive else mse_naive,
        indices[i]
    ) for i in range(len(past))]

    labels = [
        label_cols,
        label_cols,
        label_cols,
        "index"
    ]

    return ret, labels

def label_dataset_2(
    past, future,
    mse_naive=None, mse_naive_final=None,
    label_cols=DataCol.SIRD_VARS,
    final_seed_cols=DataCol.SIRD,
    final_cols=DataCol.IRD,
    **kwargs
):
    final_seed = [x.iloc[-1] for x in past]
    indices = [x.index for x in future]

    final_seed = [x[final_seed_cols].to_numpy() for x in final_seed]
    future_final = [x[final_cols].to_numpy() for x in future]
    past = [x[label_cols].to_numpy() for x in past]
    future = [x[label_cols].to_numpy() for x in future]

    ret = [(
        past[i],
        future[i],
        final_seed[i],
        future_final[i],
        mse_naive[i] if mse_naive else mse_naive,
        mse_naive_final[i] if mse_naive_final else mse_naive_final,
        indices[i]
    ) for i in range(len(past))]

    labels = [
        label_cols,
        label_cols,
        final_seed_cols,
        label_cols,
        final_cols,
        "index"
    ]

    return ret, labels


def merge_dataset(datasets):
    return list(itertools.chain.from_iterable(datasets))


def split_dataset(
    df,
    val_start=None, test_start=None,
    past_size=30, future_size=14,
    seed_size=None,
    stride=1,
    past_cols=None,
    label_cols=DataCol.SIRD_VARS,
    future_exo_cols=["psbb", "ppkm", "ppkm_mikro"],
    final_seed_cols=DataCol.SIRD,
    final_cols=DataCol.IRD,
    limit_past=True,
    val=True,
    labeling=label_dataset_0
):
    # if past_cols is not None:
    #     past_cols = list(set(past_cols + future_exo_cols + label_cols))

    train_set, labels = labeling(
        *slice_dataset(
            df[:val_start],
            future_start=None, future_end=val_start,
            past_size=past_size, future_size=future_size,
            stride=stride,
            limit_past=limit_past
        ),
        seed_size=seed_size,
        past_cols=past_cols,
        label_cols=label_cols,
        future_exo_cols=future_exo_cols,
        final_seed_cols=final_seed_cols, final_cols=final_cols
    )
    val_set, labels = labeling(
        *slice_dataset(
            df[:test_start],
            future_start=val_start, future_end=test_start,
            past_size=past_size, future_size=future_size,
            stride=stride,
            limit_past=limit_past
        ),
        seed_size=seed_size,
        past_cols=past_cols,
        label_cols=label_cols,
        future_exo_cols=future_exo_cols,
        final_seed_cols=final_seed_cols, final_cols=final_cols
    )
    test_set, labels = labeling(
        *slice_dataset(
            df,
            future_start=test_start, future_end=None,
            past_size=past_size, future_size=future_size,
            stride=stride,
            limit_past=limit_past
        ),
        seed_size=seed_size,
        past_cols=past_cols,
        label_cols=label_cols,
        future_exo_cols=future_exo_cols,
        final_seed_cols=final_seed_cols, final_cols=final_cols
    )
    if val == 2:
        return (merge_dataset([train_set, val_set]), val_set, test_set), labels
    elif val:
        return (train_set, val_set, test_set), labels
    else:
        return (merge_dataset([train_set, val_set]), test_set), labels


# Differencing
def diff(series, order=1, shift=1, **kwargs):
    # ret = series - series.shift()
    for i in range(order):
        series = series.diff(periods=shift, **kwargs)
    series.dropna(inplace=True)
    return series
