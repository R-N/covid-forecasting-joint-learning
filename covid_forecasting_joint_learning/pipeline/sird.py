import pandas as pd
from ..data import cols as DataCol


def calc_s_global(df, n_global):
    df[DataCol.N_GLOBAL] = pd.Series(n_global, index=df.index)
    df[DataCol.S_GLOBAL_PEOPLE] = df[DataCol.N_GLOBAL] - (df[DataCol.I_TOT_GLOBAL] + df[DataCol.VAC_PEOPLE])
    df[DataCol.S_GLOBAL_FULL] = df[DataCol.N_GLOBAL] - (df[DataCol.I_TOT_GLOBAL] + df[DataCol.VAC_FULL])
    df[DataCol.POS_RATE] = df[DataCol.I_TOT_GLOBAL] / df[DataCol.TEST]
    return df


def calc_delta_global(df, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted

    df[DataCol.DELTA_TEST] = df[DataCol.TEST] - df_shifted[DataCol.TEST]
    df[DataCol.DELTA_VAC_PEOPLE] = df[DataCol.VAC_PEOPLE] - df_shifted[DataCol.VAC_PEOPLE]
    df[DataCol.DELTA_VAC_FULL] = df[DataCol.VAC_FULL] - df_shifted[DataCol.VAC_FULL]
    df[DataCol.DELTA_I_TOT_GLOBAL] = df[DataCol.I_TOT_GLOBAL] - df_shifted[DataCol.I_TOT_GLOBAL]
    return df


def calc_vars_global(df, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted

    df[DataCol.VAC_PEOPLE_S] = df[DataCol.DELTA_VAC_PEOPLE] / df_shifted[DataCol.S_GLOBAL_PEOPLE]
    df[DataCol.VAC_FULL_S] = df[DataCol.DELTA_VAC_FULL] / df_shifted[DataCol.S_GLOBAL_FULL]
    df[DataCol.DAILY_POS_RATE] = df[DataCol.DELTA_I_TOT_GLOBAL] / df[DataCol.DELTA_TEST]

    return df


def calc_s(df, n):
    df[DataCol.N] = pd.Series(n, index=df.index)
    df[DataCol.S] = df[DataCol.N] - (df[DataCol.I] + df[DataCol.R] + df[DataCol.D])
    return df


def calc_delta(df, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted

    df[DataCol.DELTA_R] = df[DataCol.R] - df_shifted[DataCol.R]
    df[DataCol.DELTA_D] = df[DataCol.D] - df_shifted[DataCol.D]
    df[DataCol.DELTA_I] = df[DataCol.I] - df_shifted[DataCol.I]
    return df


def calc_vars(df, n, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted
    df[DataCol.BETA] = (n / df_shifted[DataCol.S]) * (
        df[DataCol.DELTA_I] + df[DataCol.DELTA_R] + df[DataCol.DELTA_D]
    ) / df_shifted[DataCol.I]
    df[DataCol.GAMMA] = df[DataCol.DELTA_R] / df_shifted[DataCol.I]
    df[DataCol.DELTA] = df[DataCol.DELTA_D] / df_shifted[DataCol.I]

    return df


def rebuild(sird_vars, prev, n, index=None, return_s=False):
    is_df = isinstance(sird_vars, pd.DataFrame)
    index = index if index is not None else sird_vars.index if is_df else None
    sird_vars = sird_vars[DataCol.SIRD_VARS].itertuples(index=False) if is_df else sird_vars

    rebuilt = [prev]
    s, i, r, d = prev

    for beta, gamma, delta in sird_vars:

        delta_r = gamma * i
        delta_d = delta * i
        delta_i_in = beta * i * (s / n)
        delta_s = -delta_i_in
        delta_i = delta_i_in - (delta_r + delta_d)

        s += delta_s
        i += delta_i
        r += delta_r
        d += delta_d

        rebuilt.append([s, i, r, d])

    rebuilt.pop(0)
    if not return_s:
        rebuilt = [x[1:] for x in rebuilt]

    rebuilt = rebuilt if index is None else pd.DataFrame(rebuilt, columns=DataCol.SIRD, index=index)

    return rebuilt

