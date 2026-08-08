"""Naive (last-value-carried-forward) baseline.

INVESTIGATION.md, Recommendations #1: the single result most likely to
decide whether this project has a finding. At kabko/county geographic
scale, most published epidemic forecasting models fail to beat this -- in
the US COVID-19 Forecast Hub retrospective, only 7 of 22 teams beat
`COVIDhub-baseline` (last observed value carried forward) at state level,
and skill was worse at county level.

Forecasts every future step as the last observed I/R/D values -- no
fitting, no parameters. Evaluated with the same per-IRD RMSSE the SIRD and
ARIMA-SIRD baselines use (`..loss_common.rmsse`, unreduced on the feature
axis), against the same `final_seed`/`future_final` fields of the standard
`label_dataset_0` sample, so all three log into the same i/r/d schema and
are directly comparable.
"""
import numpy as np
import pandas as pd
from xlrd import XLRDError
from ..loss_common import rmsse as rmsse_per_ird


class NaiveModel:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.loss = None

    def pred_final(self, days, final_seed):
        last = final_seed[-1] if final_seed.ndim == 2 else final_seed
        assert last.shape[0] == 4, "expected an S/I/R/D seed"
        ird = last[1:]
        return np.tile(ird, (days, 1))

    def eval(self, final_seed, future_final, loss_fn=rmsse_per_ird):
        assert future_final.ndim == 2 and future_final.shape[1] == 3
        pred_final = self.pred_final(len(future_final), final_seed)
        # Matches ARIMASIRDModel.eval: the naive-forecast denominator in
        # loss_fn is built from final_seed's own I/R/D history, not a
        # separate "past" window -- this model takes none.
        past_final = final_seed[:, 1:] if final_seed.ndim == 2 else final_seed[None, 1:]
        self.loss = loss_fn(past_final, future_final, pred_final)
        return self.loss

    def eval_sample(self, sample, loss_fn=rmsse_per_ird):
        # Standard label_dataset_0 layout: (past, past_seed, past_exo,
        # future, future_exo, final_seed, future_final, index). Only the
        # last two fields before the index are needed; take them from the
        # end so a 7- or 8-field sample (with or without the trailing
        # index) unpacks identically.
        if len(sample) not in (7, 8):
            raise ValueError("NaiveModel requires label_dataset_0 samples (7 or 8 fields)")
        *_, final_seed, future_final, _ = (*sample, None)[:8]
        return self.eval(final_seed=final_seed, future_final=future_final, loss_fn=loss_fn)

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


class NaiveEvalLog:
    """Same i/r/d schema as SIRDEvalLog/ARIMASIRDEvalLog, minus the
    fields (order, limit_fit, ...) a parameter-free model has none of.
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
