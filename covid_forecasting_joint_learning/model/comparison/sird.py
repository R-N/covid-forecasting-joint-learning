from scipy.integrate import odeint
from lmfit import minimize, Parameters
import numpy as np
from ..loss_common import msse, rmsse
from ...data import cols as DataCol

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
    ret = odeint(dpsird, y0, t, args=(
        n, beta, gamma, delta
    ))
    retT = ret.T
    s, i, r, d = retT

    return s, i, r, d


def pred_full(days, n, beta, gamma, delta, first=1):
    s_0, i_0, r_0, d_0 = n - first, first, 0, 0  # initial conditions: one infectious, rest susceptible

    t = np.linspace(0, days - 1, days)  # days
    y0 = s_0, i_0, r_0, d_0  # Initial conditions tuple

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
    result = minimize(objective, params, reduce_fcn=loss_fn, calc_covar=True)
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

    def fit(self, past, loss_fn=msse):
        self.clear()

        loss_fn = self.loss_fn or loss_fn

        def loss_fn(future, pred):
            return loss_fn(past, future, pred)

        objective = make_objective(past, self.n)

        self.fit_result = fit(objective, self.params_hint, loss_fn=loss_fn)

        first = past[-1][0]
        self.prev = np.array([self.n - first, *past[-1]])
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

    def test(self, past, future, loss_fn=msse):
        loss_fn = self.loss_fn or loss_fn
        pred = self.pred(len(future))
        return loss_fn(past, future, pred)


def eval(past, future, n, params, loss_fn=msse):
    model = SIRDModel(params_hint=params, n=n, loss_fn=loss_fn)
    model.fit(past)
    model.loss = model.test(past, future)
    return model


def eval_dataset(dataset, n, params, loss_fn=msse, reduction="mean"):
    losses = [
        eval(past, future, n, params, loss_fn=loss_fn).loss
        for past, future in dataset[:2]
    ]
    sum_loss = sum(losses)
    if reduction == "sum":
        return sum_loss
    elif reduction in ("mean", "avg"):
        return sum_loss / len(losses)
    else:
        raise Exception(f"Invalid reduction \"{reduction}\"")
