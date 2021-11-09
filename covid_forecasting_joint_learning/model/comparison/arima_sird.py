import numpy as np
from xlrd import XLRDError
from .arima import ARIMAModel, ARIMASearchLog, ARIMAEvalLog, rmsse, msse
from ...pipeline import sird

class ARIMASIRDModel:
    def __init__(self, models, population, reduction="mean"):
        assert len(models) == 3
        self.models = models
        self.reduction = reduction
        self.population = population

    def fit(self, endo, exo=None):
        assert endo.ndim == 2 and endo.shape[1] == 3
        assert exo is None or (exo.ndim == 2 and exo.shape[1] == 3)
        for i in range(3):
            self.models[i].fit(
                endo[:, i],
                exo=exo[:, i] if (exo is not None and exo.ndim == 2) else exo
            )
        self.first = len(endo)

    def _pred(self, start, end, exo=None):
        if exo is not None:
            assert len(exo) == (end - start + 1) and exo.ndim == 2 and exo.shape[1] == 3
        preds = [self.models[i]._pred(
            start=start,
            end=end,
            exo=exo[:, i] if (exo is not None and exo.ndim == 2) else exo
        ) for i in range(3)]
        return np.stack(preds).T

    def pred_vars(self, days, exo=None):
        return self._pred(start=self.first, end=self.first + days - 1, exo=exo)

    def pred_vars_full(self, days, exo=None):
        return self._pred(start=0, end=days - 1, exo=exo)

    def rebuild_pred(self, pred_vars, final_seed):
        return sird.rebuild(
            pred_vars,
            final_seed,
            self.population
        )

    def pred_final(self, days, final_seed, exo=None):
        return self.rebuild_pred(
            self.pred_vars(days, exo=exo),
            final_seed
        )

    def pred_final_full(self, days, final_seed, exo=None):
        return self.rebuild_pred(
            self.pred_vars_full(days, exo=exo),
            final_seed
        )

    def eval(self, past, final_seed, future_final, exo=None, past_exo=None, loss_fn=rmsse):
        assert past.ndim == 2 and past.shape[1] == 3
        assert future_final.ndim == 2 and future_final.shape[1] == 3
        if final_seed.ndim == 2:
            final_seed = final_seed[-1]
        assert final_seed.ndim == 1 and final_seed.shape[0] == 4
        self.fit(past, exo=past_exo)
        pred_final = self.pred_final(len(future_final), final_seed, exo=exo)
        self.loss = loss_fn(past, future_final, pred_final)
        return self.loss

    def eval_sample(self, sample, loss_fn=rmsse, use_exo=False):
        if use_exo:
            past, past_seed, past_exo, future, future_exo, final_seed, future_final = sample[:7]
        else:
            past, future, final_seed, future_final = sample[:4]
            past_exo, future_exo = None, None
        return self.eval(
            past=past,
            final_seed=final_seed,
            future_final=future_final,
            exo=future_exo,
            past_exo=past_exo,
            loss_fn=loss_fn
        )

    def eval_dataset(self, dataset, loss_fn=rmsse, use_exo=False, reduction=None):
        reduction = reduction or self.reduction
        losses = [self.eval_sample(sample, loss_fn=loss_fn, use_exo=use_exo) for sample in dataset]
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


class ARIMASIRDEvalLog(ARIMAEvalLog):
    def __init__(self, source_path, log_path, source_sheet_name="ARIMA", log_sheet_name="Eval"):
        super().__init__(
            source_path, log_path,
            source_sheet_name=source_sheet_name,
            log_sheet_name=log_sheet_name,
            columns=["group", "cluster", "kabko", "order", "seasonal_order", "limit_fit", "i", "r", "d"]
        )

    def is_eval_done(self, group, cluster, kabko):
        df = self.log_df
        try:
            return ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko)).any()
        except (ValueError, XLRDError) as ex:
            if "No sheet" in str(ex) or "is not in list" in str(ex):
                return False
            raise

    def log(self, group, cluster, kabko, order, seasonal_order, limit_fit, loss, log_path=None, log_sheet_name=None):
        assert len(loss) == 3
        df = self.load_log()
        df.loc[df.shape[0]] = {
            "group": group,
            "cluster": cluster,
            "kabko": kabko,
            "order": order,
            "seasonal_order": seasonal_order,
            "limit_fit": limit_fit,
            "i": loss[0],
            "r": loss[1],
            "d": loss[2]
        }
        self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
