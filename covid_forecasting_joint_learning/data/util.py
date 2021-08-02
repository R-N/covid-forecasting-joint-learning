import numpy as np
import pandas as pd
from . import cols as DataCol


def mkparent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def add_dates(df, ranges, name):
    ranges = ranges.itertuples(index=False) if isinstance(ranges, pd.DataFrame) else ranges
    df.loc[:, name] = pd.Series(0.0, index=df.index, dtype=np.float32)
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
        df[val_col] = pd.Series(np.array(len(df) * [1.0]), dtype=np.float32)
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

