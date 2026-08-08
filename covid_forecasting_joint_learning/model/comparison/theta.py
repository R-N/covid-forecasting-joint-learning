"""Theta baseline.

INVESTIGATION.md, Quick wins: "Add Theta and a tuned linear baseline --
minutes each, and both are standard". Theta (Assimakopoulos & Nikolopoulos,
2000) decomposes each series into long-run and short-run "theta lines" and
is a perennial top performer in the M3/M4 competitions despite being
univariate and parameter-light -- a natural sanity check between the naive
baseline and the full SIRD/ARIMA-SIRD models.

Fits one independent `statsmodels` `ThetaModel` per I/R/D column (Theta is
inherently univariate) against a weekly (`period=7`) seasonal cycle, since
the raw case data has a weekly reporting cycle (see `data/cols.py`'s
`DAYS`/day-of-week dummy columns). Evaluated with the same per-IRD RMSSE
(`..loss_common.rmsse`, unreduced on the feature axis) against the same
`final_seed`/`future_final` fields of the standard `label_dataset_0` sample
as the naive/SIRD/ARIMA-SIRD baselines, so all log into the same i/r/d
schema and are directly comparable.

A column with too little history or a degenerate (short/constant) series
can make `ThetaModel(...).fit()` raise -- caught per column and replaced
with a naive last-value-carried-forward fallback for that column only, so
one bad kabko/column never crashes the whole comparison run.
"""
import numpy as np
import pandas as pd
from xlrd import XLRDError
from statsmodels.tsa.forecasting.theta import ThetaModel as _ThetaModel
from ..loss_common import rmsse as rmsse_per_ird

THETA_PERIOD = 7  # weekly reporting cycle


class ThetaModel:
    def __init__(self, period=THETA_PERIOD, reduction="mean"):
        self.period = period
        self.reduction = reduction
        self.loss = None
        self.fits = None  # 3 fitted statsmodels results, one per I/R/D column
        self._last = None  # last observed I/R/D, for the per-column fit-failure fallback

    def fit(self, past_seed):
        past_seed = np.asarray(past_seed, dtype=float)
        if past_seed.ndim == 1:
            past_seed = past_seed[None, :]
        ird = past_seed[:, -3:]  # last 3 columns: I/R/D (drops S if S/I/R/D was passed)
        self._last = ird[-1].copy()
        self.fits = [None, None, None]
        for i in range(3):
            try:
                self.fits[i] = _ThetaModel(ird[:, i], period=self.period).fit()
            except Exception:
                self.fits[i] = None
        return self

    def pred_final(self, days, past_seed=None):
        if self.fits is None:
            assert past_seed is not None, "must fit() first or pass past_seed"
            self.fit(past_seed)
        preds = []
        for i in range(3):
            forecast = None
            if self.fits[i] is not None:
                try:
                    forecast = np.asarray(self.fits[i].forecast(days), dtype=float)
                except Exception:
                    forecast = None
            if forecast is None:
                # naive last-value-carried-forward fallback for this column only
                forecast = np.full(days, self._last[i])
            preds.append(forecast)
        pred = np.stack(preds, axis=1)
        return np.maximum(pred, 0)

    def eval(self, past_seed, final_seed, future_final, loss_fn=rmsse_per_ird):
        assert future_final.ndim == 2 and future_final.shape[1] == 3
        past_final = final_seed[:, 1:] if final_seed.ndim == 2 else final_seed[None, 1:]
        self.fit(past_seed)
        pred_final = self.pred_final(len(future_final))
        self.loss = loss_fn(past_final, future_final, pred_final)
        return self.loss

    def eval_sample(self, sample, loss_fn=rmsse_per_ird):
        # Standard label_dataset_0 layout: (past, past_seed, past_exo,
        # future, future_exo, final_seed, future_final, index). Only
        # past_seed (to fit), final_seed and future_final (to eval) are
        # needed; take them from the standard positions so a 7- or
        # 8-field sample (with or without the trailing index) unpacks
        # identically.
        if len(sample) not in (7, 8):
            raise ValueError("ThetaModel requires label_dataset_0 samples (7 or 8 fields)")
        _, past_seed, _, _, _, final_seed, future_final, *_ = (*sample, None)[:8]
        return self.eval(past_seed=past_seed, final_seed=final_seed, future_final=future_final, loss_fn=loss_fn)

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


class ThetaEvalLog:
    """Same i/r/d schema as NaiveEvalLog/SIRDEvalLog/ARIMASIRDEvalLog."""

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
