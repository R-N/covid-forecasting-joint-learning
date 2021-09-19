from scipy.integrate import odeint
from lmfit import minimize, Parameters
import numpy as np
from ..loss_common import msse, rmsse, wrap_reduce
from ...data import cols as DataCol
import optuna

msse = wrap_reduce(msse)
rmsse = wrap_reduce(rmsse)

def dpsird(y, t, n, beta, gamma, delta):
    s, i, r, d = y
    i_flow = beta * s * i / n
    r_flow = gamma * i * 1
    d_flow = delta * i * 1
    dSdt = -i_flow
    dIdt = i_flow - r_flow - d_flow
    dRdt = r_flow
    dDdt = d_flow
    dPdt = dSdt + dIdt + dRdt + dDdt
    assert abs(dPdt) <= 1e-6
    return dSdt, dIdt, dRdt, dDdt


def pred(t, y0, n, beta, gamma, delta):
    # Integrate the SIR equations over the time grid, t.
    y0 = np.array([n - y0[0], *y0])
    ret = odeint(dpsird, y0, t, args=(
        n, beta, gamma, delta
    ))
    retT = ret.T
    s, i, r, d = retT

    return s, i, r, d


def pred_full(days, n, beta, gamma, delta, first=1):
    if not (isinstance(first, tuple) or isinstance(first, list) or isinstance(first, np.ndarray)):
        first = (first, 0, 0)

    t = np.linspace(0, days - 1, days)  # days
    y0 = first

    return pred(t, y0, n, beta, gamma, delta)


def make_objective(data, n, x=None):
    def objective(params):
        s, i, r, d = pred_full(len(data), n, first=data[0][0], **params)
        if x is not None:
            i, r, d = i[x], r[x], d[x]
        ret = np.concatenate((i, r, d))
        ret = ret - data.flatten("F")  # Test flatten
        return ret
    return objective


def make_params(params):
    params_1 = Parameters()
    for kwarg, (init, mini, maxi) in params.items():
        params_1.add(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    return params_1


def fit(objective, params, loss_fn=None):
    result = minimize(objective, params, calc_covar=True)
    return result


class SIRDModel:
    def __init__(self, params_hint, n, loss_fn=None):
        self.params_hint = params_hint
        self.n = n
        self.loss_fn = loss_fn
        self.clear()

    @property
    def fit_params(self):
        if not self.fit_result:
            raise Exception("Please fit the model first")
        return {k: self.fit_result.params[k].value for k in self.fit_result.params.keys()}

    def clear(self):
        self.fit_result = None
        self.prev = None
        self.loss = None
        self.pred_start = None
        self.first = 1

    def fit(self, past, loss_fn=msse):
        self.clear()

        def loss_fn(future, pred):
            return loss_fn(past, future, pred)

        objective = make_objective(past, self.n)

        self.fit_result = fit(objective, self.params_hint, loss_fn=loss_fn)

        self.first = past[0]
        self.prev = past[-1]
        self.pred_start = len(past)

        return self.fit_result

    def pred(self, days):
        if not self.fit_result:
            raise Exception("Please fit the model first!")

        full_len = self.pred_start + days
        s, i, r, d = pred(
            np.linspace(self.pred_start, full_len - 1, days),
            self.prev,
            self.n,
            **self.fit_params
        )
        return np.array([i, r, d]).T

    def pred_full(self, days):
        if not self.fit_result:
            raise Exception("Please fit the model first!")

        s, i, r, d = pred_full(
            days,
            self.n,
            first=self.first,
            **self.fit_params
        )
        return np.array([i, r, d]).T

    def test(self, past, future, loss_fn=rmsse):
        loss_fn = self.loss_fn or loss_fn
        pred = self.pred(len(future))
        return loss_fn(past, future, pred)


def eval(past, future, n, params, loss_fn=rmsse, limit_past=None, limit_loss=False):
    model = SIRDModel(params_hint=params, n=n)
    past_1 = past if not limit_past else past[:limit_past]
    model.fit(past_1)
    model.loss = model.test(past_1 if limit_loss else past, future, loss_fn=loss_fn)
    return model


def eval_dataset(dataset, n, params, loss_fn=rmsse, reduction="mean", limit_past=None, limit_loss=False):
    losses = [
        eval(past, future, n, params, loss_fn=loss_fn, limit_past=limit_past, limit_loss=limit_loss).loss
        for past, future, indices in dataset[:2]
    ]
    sum_loss = sum(losses)
    if reduction == "sum":
        return sum_loss
    elif reduction in ("mean", "avg"):
        return sum_loss / len(losses)
    else:
        raise Exception(f"Invalid reduction \"{reduction}\"")


def search(dataset, n, params, loss_fn=msse, reduction="mean", limit_loss=False, limit_past_min=0, limit_past_max=366):
    def objective(trial):
        limit_past = trial.suggest_int("limit_past", limit_past_min, limit_past_max)
        return eval_dataset(dataset, n, params, loss_fn=loss_fn, reduction=reduction, limit_past=limit_past, limit_loss=limit_loss)

    study = optuna.create_study()
    study.optimize(objective, n_trials=(limit_past_max - limit_past_min + 1), n_jobs=1)
    return study


def search_limit_past(dataset, n, params, loss_fn=rmsse, reduction="mean", limit_loss=False, limit_past_min=0, limit_past_max=366):

    results = [(limit_past, eval_dataset(dataset, n, params, loss_fn=loss_fn, reduction=reduction, limit_past=limit_past, limit_loss=limit_loss)) for limit_past in range(limit_past_min, limit_past_max + 1)]

    return min(results, key=lambda x: x[1])
