from scipy.integrate import odeint
from lmfit import minimize, Parameters
import numpy as np
from ..loss import msse, rmsse
from ...data import cols as DataCol

def dpsird(y, t, population, beta, gamma, delta):
    population, susceptible, exposed, infectious, recovered, dead = y
    infectious_flow = beta * susceptible * infectious / population
    recovery_flow = gamma * infectious * 1
    death_flow = delta * infectious * 1
    dSdt = -infectious_flow
    dIdt = infectious_flow - recovery_flow - death_flow
    dRdt = recovery_flow
    dDdt = death_flow
    dPdt = dSdt + dIdt + dRdt + dDdt
    return dPdt, dSdt, dIdt, dRdt, dDdt


def pred(t, y0, population, beta, gamma, delta):
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(dpsird, y0, t, args=(
        population, beta, gamma, delta
    ))
    retT = ret.T
    population_2, susceptible, infectious, recovered, dead = retT
    assert np.mean(np.abs(population_2 - population)) <= 1e-6

    return susceptible, infectious, recovered, dead


def pred_full(days, population, beta, gamma, delta, first=1):
    susceptible_init, infectious_init, recovered_init, dead_init = population - first, first, 0, 0  # initial conditions: one infectious, rest susceptible

    t = np.linspace(0, days - 1, days)  # days
    y0 = susceptible_init, infectious_init, recovered_init, dead_init  # Initial conditions tuple

    return pred(t, y0, beta, gamma, delta)


def make_objective(data, population, x=None):
    def objective(beta, gamma, delta):
        s, i, r, d = pred_full(len(data), population, beta, gamma, delta, first=data[0][0])
        if x is not None:
            i, r, d = i[x], r[x], d[x]
        ret = np.concatenate(i, r, d)
        ret = ret - data.flatten("F")  # Test flatten
        return ret
    return objective


def make_params(params):
    params_1 = Parameters()
    for kwarg, (init, mini, maxi) in params.items():
        params_1.add(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    return params_1


def fit(objective, params, loss_fn=msse):
    result = minimize(objective, params, reduce_fcn=loss_fn, calc_covar=True)
    return result


class SIRDModel:
    def __init__(self, params_hint, population, loss_fn=None):
        self.params_hint = params_hint
        self.population = population
        self.loss_fn = loss_fn
        self.clear()

    @property
    def fit_params(self):
        if not self.fit_result:
            raise Exception("Please fit the model first")
        return {k: self.fit_result.params.value for k in self.fit_result.params.keys()}

    def clear(self):
        self.fit_result = None
        self.prev = None
        self.loss = None
        self.pred_start = None

    def fit(self, past, loss_fn=msse):
        self.clear()

        loss_fn = self.loss_fn or loss_fn

        objective = make_objective(past, self.population)

        self.fit_result = fit(objective, self.params_hint, loss_fn=loss_fn)

        first = past[-1][0]
        self.prev = np.array([self.population - first, *past[-1]])
        self.pred_start = len(past)

        return self.fit_result

    def pred(self, days):
        if not self.fit_result:
            raise Exception("Please fit the model first!")

        full_len = self.pred_start + days
        s, i, r, d = pred(
            np.linspace(self.pred_start, full_len - 1, days),
            self.prev,
            **self.fit_params
        )
        return np.array([i, r, d]).T

    def test(self, future, loss_fn=msse):
        loss_fn = self.loss_fn or loss_fn
        pred = self.pred(len(future))
        return loss_fn(future, pred)


def eval(past, future, population, params, loss_fn=msse):
    model = SIRDModel(params_hint=params, population=population, loss_fn=loss_fn)
    model.fit(past)
    model.loss = model.test(future)
    return model


def eval_dataset(dataset, population, params, loss_fn=msse, reduction="mean"):
    losses = [
        eval(past, future, population, params, loss_fn=loss_fn).loss
        for past, future in dataset[:2]
    ]
    sum_loss = sum(losses)
    if reduction == "sum":
        return sum_loss
    elif reduction in ("mean", "avg"):
        return sum_loss / len(losses)
    else:
        raise Exception(f"Invalid reduction \"{reduction}\"")
