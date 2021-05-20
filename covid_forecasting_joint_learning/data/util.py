import pandas as pd


def add_special_dates(df, ranges, name):
    ranges = ranges.itertuples(index=False) if isinstance(ranges, pd.DataFrame) else ranges
    df.loc[:, name] = pd.Series(0, index=df.index)
    for start, end, value in ranges:
        # start, end = pd.to_datetime((start, end))
        df.loc[start:end, name] = value
    return df
