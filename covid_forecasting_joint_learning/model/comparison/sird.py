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


def fit(objective, params):
    result = minimize(objective, params, calc_covar=True)
    return result


class SIRDModel:
    def __init__(self, params_hint, n, loss_fn=rmsse, limit_past=None, reduction="mean"):
        self.params_hint = params_hint
        self.n = n
        self.loss_fn = loss_fn
        self.loss = None
        self.limit_past = limit_past
        self.reduction = reduction
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

    def fit(self, past, limit_past=None):
        self.clear()

        limit_past = limit_past or self.limit_past
        if limit_past:
            past = past[:limit_past]

        objective = make_objective(past, self.n)

        self.fit_result = fit(objective, self.params_hint)

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

    def test(self, past, future, loss_fn=None):
        loss_fn = loss_fn or self.loss_fn
        pred = self.pred(len(future))
        self.loss = loss_fn(past, future, pred)
        return self.loss


    def eval(self, past, future, loss_fn=rmsse, limit_past=None):
        self.fit(past, limit_past=limit_past)
        return self.test(past, future, loss_fn=loss_fn)


    def eval_dataset(self, dataset, loss_fn=rmsse, reduction=None, limit_past=None):
        reduction = reduction or self.reduction
        losses = [
            self.eval(past, future, loss_fn=loss_fn, limit_past=limit_past)
            for past, future, indices in dataset
        ]
        sum_loss = sum(losses)
        if reduction == "sum":
            return sum_loss
        elif reduction in ("mean", "avg"):
            return sum_loss / len(losses)
        else:
            raise Exception(f"Invalid reduction \"{reduction}\"")


def search_optuna(params_hint, n, dataset, loss_fn=msse, reduction="mean", limit_past_min=7, limit_past_max=366, no_limit=False, n_trials=None):
    def objective(trial):
        no_limit_1 = no_limit
        if no_limit_1 is None:
            trial.suggest_categorical("no_limit", (0, 1))
        if no_limit_1:
            limit_past = None
        else:
            limit_past = trial.suggest_int("limit_past", limit_past_min, limit_past_max)

        model = SIRDModel(params_hint=params_hint, n=n, loss_fn=loss_fn, reduction=reduction, limit_past=limit_past)
        return model.eval_dataset(dataset)

    if n_trials is None:
        n_trials = (limit_past_max - limit_past_min + 1)

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


def search_greedy(params_hint, n, dataset, loss_fn=rmsse, reduction="mean", limit_past_min=7, limit_past_max=366):
    best_model = None
    best_loss = np.inf
    for limit_past in [*range(limit_past_min, limit_past_max + 1), None]:
        model = SIRDModel(params_hint=params_hint, n=n, loss_fn=loss_fn, reduction=reduction, limit_past=limit_past)
        loss = model.eval_dataset(dataset)
        if loss < best_loss:
            best_model = model
            best_loss = loss
    return best_model
