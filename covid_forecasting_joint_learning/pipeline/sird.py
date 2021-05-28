import pandas as pd


def calc_s(df, n):
    df["n"] = pd.Series(n, index=df.index)
    df["s"] = df["n"] - (df["i"] + df["r"] + df["d"])
    return df


def calc_delta(df, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted

    df["delta_r"] = df["r"] - df_shifted["r"]
    df["delta_d"] = df["d"] - df_shifted["d"]
    df["delta_i"] = df["i"] - df_shifted["i"] 
    return df


def calc_vars(df, n, df_shifted=None):
    df_shifted = df.shift() if df_shifted is None else df_shifted
    df["beta"] = (n / df_shifted["s"]) * (df["delta_i"] + df["delta_r"] + df["delta_d"]) / df_shifted["i"]
    df["gamma"] = df["delta_r"] / df_shifted["i"]
    df["delta"] = df["delta_d"] / df_shifted["i"]

    return df


sird_vars_cols = ["beta", "gamma", "delta"]
sird_cols = ["s", "i", "r", "d"]


def rebuild(sird_vars, prev, n, index=None):
    is_df = isinstance(sird_vars, pd.DataFrame)
    index = index if index is not None else sird_vars.index if is_df else None
    sird_vars = sird_vars[sird_vars_cols].itertuples(index=False) if is_df else sird_vars

    rebuilt = [prev]

    for beta, gamma, delta in sird_vars:
        s, i, r, d = rebuilt[-1]

        delta_r = gamma * i
        delta_d = delta * i
        delta_i_in = beta * s * i / n
        delta_s = -delta_i_in
        delta_i = delta_i_in - (delta_r + delta_d)

        s += delta_s
        i += delta_i
        r += delta_r
        d += delta_d

        rebuilt.append([s, i, r, d])

    rebuilt.pop(0)

    rebuilt = rebuilt if index is None else pd.DataFrame(rebuilt, columns=sird_cols, index=index)

    return rebuilt

