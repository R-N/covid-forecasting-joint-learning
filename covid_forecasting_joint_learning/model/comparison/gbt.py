"""Gradient-boosted-tree baseline with lag (window) features.

INVESTIGATION.md, Quick wins / "Three independent lines converge on a
baseline this project lacks": the baseline set was naive, Theta and a
one-layer linear model, while the literature points harder at a
gradient-boosted tree with window features than at any of those -- a study
casting gradient boosting into a window-based regression framework found it
competes with and often outperforms eight deep architectures across nine
datasets (Elsayed et al., 2021, https://arxiv.org/abs/2101.02118),
EpiCastBench found XGBoost/Random Forest competitive with deep models across
40 epidemic datasets, and M5 was dominated by LightGBM-family methods.

Window-regression scheme (Elsayed et al.'s framework): a single continuous
series is turned into a supervised table by sliding a fixed-width window of
`lag` past observations across it, using each window to predict the next
step. With `lag=7` (one weekly cycle -- COVID case/recovery/death counts have
a strong day-of-week reporting artifact) and a column `column` of length
`seed_length`:

    X[t] = column[t - lag : t]   (shape (lag,))
    y[t] = column[t]
    for every valid t in [lag, seed_length)

This is done independently per I/R/D column (three separate regressors,
since the three series have different scales and dynamics -- unlike a
single-target wrapper over a shared feature space, this needs no
horizon-specific model or joint feature matrix). One
`GradientBoostingRegressor(n_estimators=50, max_depth=3)` is fit per column
on that column's `(X, y)` pairs. `seed_length` (30-34 in this project's
seed windows) is comfortably above `lag`, but a defensive clamp
(`lag = min(lag, seed_length - 1)`, minimum 1) guards the degenerate case.

Multi-step forecasting is recursive, not a 14-model direct wrapper: at each
step the last `lag` observed-or-predicted values are fed back in as the next
window, one column at a time, and the freshly predicted value both becomes
part of the output and slides into the window for the following step.
Predictions are clipped at 0 -- case/recovery/death counts cannot be
negative.

Evaluated with the same per-IRD RMSSE the naive/SIRD/ARIMA-SIRD baselines
use (`..loss_common.rmsse`, unreduced on the feature axis), against the
`past_seed`/`future_final` fields of the standard `label_dataset_0` sample,
so it logs into the same i/r/d schema and is directly comparable.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xlrd import XLRDError
from ..loss_common import rmsse as rmsse_per_ird


class GBTModel:
    def __init__(self, lag=7, reduction="mean"):
        self.lag = lag
        self.reduction = reduction
        self.loss = None
        self.regressors = None
        self.windows = None

    def fit(self, past_seed):
        past_seed = np.asarray(past_seed, dtype=float)
        assert past_seed.ndim == 2 and past_seed.shape[1] == 3
        seed_length = past_seed.shape[0]
        lag = max(1, min(self.lag, seed_length - 1))
        self.lag = lag

        regressors = []
        windows = []
        for col in range(3):
            series = past_seed[:, col]
            X = np.array([series[t - lag:t] for t in range(lag, seed_length)])
            y = np.array([series[t] for t in range(lag, seed_length)])
            reg = GradientBoostingRegressor(n_estimators=50, max_depth=3)
            reg.fit(X, y)
            regressors.append(reg)
            windows.append(series[-lag:].copy())
        self.regressors = regressors
        self.windows = windows
        return self

    def pred_final(self, days, past_seed=None):
        if past_seed is not None:
            self.fit(past_seed)
        assert self.regressors is not None, "GBTModel.pred_final requires fit() first or an explicit past_seed"

        preds = np.zeros((days, 3))
        for col in range(3):
            window = self.windows[col].copy()
            reg = self.regressors[col]
            for t in range(days):
                x = window[-self.lag:].reshape(1, -1)
                value = max(float(reg.predict(x)[0]), 0.0)
                preds[t, col] = value
                window = np.append(window, value)
        return preds

    def eval(self, past_seed, future_final, loss_fn=rmsse_per_ird):
        assert future_final.ndim == 2 and future_final.shape[1] == 3
        self.fit(past_seed)
        pred_final = self.pred_final(len(future_final))
        self.loss = loss_fn(past_seed, future_final, pred_final)
        return self.loss

    def eval_sample(self, sample, loss_fn=rmsse_per_ird):
        # Standard label_dataset_0 layout: (past, past_seed, past_exo,
        # future, future_exo, final_seed, future_final, index). past_seed
        # sits at a fixed offset from the front (position 1), so it can be
        # taken directly; future_final is taken from the end (mirroring
        # NaiveModel.eval_sample) so a 7- or 8-field sample (with or
        # without the trailing index) unpacks identically.
        if len(sample) not in (7, 8):
            raise ValueError("GBTModel requires label_dataset_0 samples (7 or 8 fields)")
        past_seed = sample[1]
        *_, future_final, _ = (*sample, None)[:8]
        return self.eval(past_seed=past_seed, future_final=future_final, loss_fn=loss_fn)

    def eval_dataset(self, dataset, loss_fn=rmsse_per_ird, reduction=None):
        reduction = reduction or self.reduction
        losses = [self.eval_sample(sample, loss_fn=loss_fn) for sample in dataset]
        sum_loss = sum(losses)
        count = len(losses)
        if reduction == "sum":
            loss = sum_loss
        elif reduction in ("mean", "avg"):
            loss = sum_loss / count
        else:
            raise Exception(f"Invalid reduction \"{reduction}\"")
        self.loss = loss
        return loss


class GBTEvalLog:
    """Same i/r/d schema as NaiveEvalLog/SIRDEvalLog/ARIMASIRDEvalLog, minus
    the fields (order, limit_fit, ...) a bare window-regression model has
    none of.
    """

    def __init__(self, log_path, log_sheet_name="Eval"):
        self.log_path = log_path
        self.log_sheet_name = log_sheet_name
        self.load_log()

    def load_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        try:
            self.log_df = pd.read_excel(log_path, sheet_name=log_sheet_name)
        except (FileNotFoundError, ValueError, XLRDError):
            self.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "i", "r", "d"])
            self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
        return self.log_df

    def save_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        self.log_df.to_excel(log_path, sheet_name=log_sheet_name, index=False)

    def is_eval_done(self, group, cluster, kabko):
        df = self.log_df
        try:
            return ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko)).any()
        except (ValueError, XLRDError) as ex:
            if "No sheet" in str(ex) or "is not in list" in str(ex):
                return False
            raise

    def log(self, group, cluster, kabko, loss, log_path=None, log_sheet_name=None):
        assert len(loss) == 3
        df = self.load_log()
        df.loc[df.shape[0]] = {
            "group": group,
            "cluster": cluster,
            "kabko": kabko,
            "i": loss[0],
            "r": loss[1],
            "d": loss[2]
        }
        self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
