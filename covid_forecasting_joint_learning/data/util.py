import numpy as np
import pandas as pd
from . import cols as DataCol
import torch
from itertools import combinations, chain


DEFAULT_DTYPE = np.float32


def mkparent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def get_sub_folders_files(path):
    childs = list(path.iterdir())
    folders = [p.relative_to(path) for p in childs if p.is_dir()]
    files = [p.relative_to(path) for p in childs if p.is_file()]
    return folders, files


def write_string(s, path):
    text_file = open(path, "w")
    n = text_file.write(s)
    text_file.close()


def add_dates(df, ranges, name):
    ranges = ranges.itertuples(index=False) if isinstance(ranges, pd.DataFrame) else ranges
    if name not in df:
        df.loc[:, name] = pd.Series(0.0, index=df.index, dtype=DEFAULT_DTYPE)
    for start, end, value in ranges:
        # start, end = pd.to_datetime((start, end))
        df.loc[start:end, name] = value
    return df

def prepare_dates(
    df,
    name_col="name",
    start_col="start", end_col="end",
    val_col="value"
):
    df = df.copy()
    df.loc[:, start_col] = pd.to_datetime(df.loc[:, start_col])
    df.loc[:, end_col] = pd.to_datetime(df.loc[:, end_col])
    if val_col not in df.columns:
        df[val_col] = pd.Series(np.array(len(df) * [1.0]), dtype=DEFAULT_DTYPE)
    df.rename(columns={
        name_col: DataCol.NAME,
        start_col: DataCol.START,
        end_col: DataCol.END,
        val_col: DataCol.VAL
    }, inplace=True)
    return df


def right_slice(
    left,
    right
):
    return left[right.first_valid_index():right.last_valid_index()]


def single_batch(t):
    return torch.stack([t[0]])


def full_combinations(src, include_empty=True):
    src = list(src)
    n_src = len(src)
    combs = [list(combinations(src, x)) for x in range(0 if include_empty else 1, n_src + 1)]
    return list(chain.from_iterable(combs))


def label_combinations(combs):
    return {" + ".join(c): c for c in combs}

def combine_batches(batches, tensor_count=7):
    remains = batches[-1][tensor_count:]
    tensors = tuple(torch.cat([batch[i] for batch in batches], dim=0) for i in range(tensor_count))
    return tensors + remains
