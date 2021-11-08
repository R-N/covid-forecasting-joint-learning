import numpy as np
from .arima import ARIMAModel, ARIMASearchLog, ARIMAEvalLog, rmsse, msse
from ...pipeline import sird

class ARIMASIRDModel:
    def __init__(self, models, population, reduction="mean"):
        assert len(models) == 3
        self.models = models
        self.reduction = reduction

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
        preds = [self.models[i].predict(
            start=start,
            end=end,
            exog=exo[:, i] if (exo is not None and exo.ndim == 2) else exo
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
        assert final_seed.ndim == 1 and final_seed.shape[1] == 4
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
