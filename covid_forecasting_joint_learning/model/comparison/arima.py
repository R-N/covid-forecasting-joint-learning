from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..loss_common import msse, rmsse, wrap_reduce
import numpy as np
from math import sqrt, ceil
import optuna


msse = wrap_reduce(msse)
rmsse = wrap_reduce(rmsse)


class ARIMAModel:
    def __init__(self, order, seasonal_order=None, limit_fit=None, reduction="mean"):
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.fit_result = None
        self.limit_fit = limit_fit
        self.reduction = reduction
        self.loss = None

    def fit(self, endo, exo=None):
        if self.limit_fit:
            endo = endo[:self.limit_fit]
            if exo is not None:
                exo = exo[:self.limit_fit]
        model = SARIMAX(endog=endo, exog=exo, order=self.order, seasonal_order=self.seasonal_order)
        self.fit_result = model.fit()
        self.first = len(endo)

    def _pred(self, start, end, exo=None):
        return self.fit_result.predict(start=start, end=end, exog=exo)

    def pred(self, days, exo=None):
        return self._pred(start=self.first, end=self.first + days - 1, exo=exo)

    def pred_full(self, days, exo=None):
        return self._pred(start=0, end=days - 1, exo=exo)

    def eval(self, past, future, exo=None, past_exo=None, loss_fn=rmsse):
        self.fit(past, exo=past_exo)
        pred = self.pred(len(future), exo=exo)
        self.loss = loss_fn(past, future, pred)
        return self.loss

    def eval_sample(self, sample, loss_fn=rmsse, use_exo=False):
        if use_exo:
            past, past_exo, future, future_exo = sample[:4]
        else:
            past, future = sample[:2]
            past_exo, future_exo = None, None
        return self.eval(
            past=past,
            future=future,
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
            return sum_loss
        elif reduction in ("mean", "avg"):
            return sum_loss / count
        else:
            raise Exception(f"Invalid reduction \"{reduction}\"")


def search_greedy(orders, train_set, loss_fn=msse, use_exo=False, reduction="mean", limit_past_min=7, limit_past_max=366):
    best_loss = np.inf
    best_model = None
    for order_set in orders:
        order = order_set[0]
        seasonal_order = order_set[1] if len(order_set) > 1 else None
        for i in [*range(limit_past_min, limit_past_max), None]:
            model = ARIMAModel(order, seasonal_order, reduction=reduction, limit_fit=i)
            loss = model.eval_dataset(train_set, loss_fn=loss_fn, use_exo=use_exo)
            if loss < best_loss:
                best_loss = loss
                best_model = model
    return best_model


def search_optuna(orders, train_set, loss_fn=msse, use_exo=False, reduction="mean", limit_past_min=7, limit_past_max=366, n_trials=None):
    def objective(trial):
        order_set = trial.suggest_int("order", 0, len(orders) - 1)
        order_set = orders[order_set]
        order = order_set[0]
        seasonal_order = order_set[1] if len(order_set) > 1 else None

        no_limit = trial.suggest_categorical("no_limit", (0, 1))
        if no_limit:
            limit_fit = None
        else:
            limit_fit = trial.suggest_int("limit_fit", limit_past_min, limit_past_max)

        model = ARIMAModel(order, seasonal_order, limit_fit=limit_fit, reduction=reduction)
        loss = model.eval_dataset(train_set, loss_fn=loss_fn, use_exo=use_exo)

        return loss

    if n_trials is None:
        n_trials = (limit_past_max - limit_past_min + 1) * ceil(sqrt(len(orders)))

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study