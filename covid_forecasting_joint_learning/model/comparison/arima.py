from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..loss_common import msse, rmsse, wrap_reduce
from numpy import np


msse = wrap_reduce(msse)
rmsse = wrap_reduce(rmsse)


class ARIMAModel:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, endog, exog=None):
        self.model = SARIMAX(endog=endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
        self.model.fit()
        return self.model

    def predict(self, start, end, exog=None, model=None):
        model = model or self.model
        return self.model.predict(start=start, end=end, exog=exog)

    def eval(self, start, end, future, exog=None, model=None, loss_fn=rmsse):
        pred = self.predict(start, end, exog=exog, model=model)
        return loss_fn(future, pred)

    def eval_dataset(self, dataset, loss_fn=rmsse, use_exo=False, reduction="mean"):
        sum_loss = 0
        count = 0
        for samples in dataset:
            if use_exo:
                past, past_exo, future, future_exo = samples[:4]
            else:
                past, future = samples[:2]
                past_exo, future_exo = None, None

            model = self.fit(past, exog=past_exo)
            loss = self.eval(
                start=len(past),
                end=len(future),
                future=future,
                exog=future_exo,
                model=model,
                loss_fn=loss_fn
            )
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
        model = ARIMAModel(order_set[0], order_set[1])
        loss = model.eval_dataset(train_set, loss_fn=loss_fn, use_exo=use_exo)
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model
