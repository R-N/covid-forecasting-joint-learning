from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..loss_common import msse, rmsse, wrap_reduce
from numpy import np


msse = wrap_reduce(msse)
rmsse = wrap_reduce(rmsse)


class ARIMAModel:
    def __init__(self, order, seasonal_order=None, limit_fit=None):
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.model = None
        self.limit_fit = limit_fit

    def fit(self, endog, exog=None):
        if self.limit_fit:
            endog = endog[:self.limit_fit]
            if exog is not None:
                exog = exog[:self.limit_fit]
        self.model = SARIMAX(endog=endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
        self.model.fit()
        return self.model

    def predict(self, start, end, exog=None, model=None):
        model = model or self.model
        return self.model.predict(start=start, end=end, exog=exog)

    def eval(self, start, end, past, future, exog=None, past_exog=None, model=None, loss_fn=rmsse):
        model = model or self.model or self.fit(past, exog=past_exog)
        pred = self.predict(start, end, exog=exog, model=model)
        return loss_fn(past, future, pred)

    def eval_sample(self, past, future, past_exo=None, future_exo=None, loss_fn=rmsse):
        model = self.fit(past, exog=past_exo)
        loss = self.eval(
            start=len(past),
            end=len(future),
            future=future,
            exog=future_exo,
            model=model,
            loss_fn=loss_fn
        )
        return loss

    def eval_dataset(self, dataset, loss_fn=rmsse, use_exo=False, reduction="mean"):
        sum_loss = 0
        count = 0
        for samples in dataset:
            if use_exo:
                past, past_exo, future, future_exo = samples[:4]
            else:
                past, future = samples[:2]
                past_exo, future_exo = None, None

            loss = self.eval_sample(past, future, past_exo=past_exo, future_exo=future_exo, loss_fn=loss_fn)

            sum_loss += loss
            count += 1
        if reduction == "sum":
            return sum_loss
        elif reduction in ("mean", "avg"):
            return sum_loss / count
        else:
            raise Exception(f"Invalid reduction \"{reduction}\"")


def search_arima(orders, train_set, loss_fn=msse, use_exo=False):
    best_loss = np.inf
    best_model = None
    for order_set in orders:
        order = order_set[0]
        seasonal_order = order_set[1] if len(order_set) > 1 else None
        for i in [*range(1, 366), None]:
            model = ARIMAModel(order, seasonal_order, limit_fit=i)
            loss = model.eval_dataset(train_set, loss_fn=loss_fn, use_exo=use_exo)
            if loss < best_loss:
                best_loss = loss
                best_model = model
    return best_model
