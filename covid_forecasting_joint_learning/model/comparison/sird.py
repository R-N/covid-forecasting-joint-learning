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


def eval(past, future, population, params, loss_fn=msse):
    objective = make_objective(past, population)
    result = fit(objective, params, past, loss_fn=loss_fn)
    past_len, future_len = len(past), len(future)
    full_len = past_len + future_len
    t = np.linspace(past_len, full_len - 1, future_len)
    first = past[-1][0]
    y0 = [population - first, *past[-1]]
    pred_1 = pred(
        t,
        y0,
        *[result.params[x].value for x in DataCol.SIRD_VARS]
    )
    result.loss = loss_fn(future, pred_1)
    return result


def eval_dataset(dataset, population, params, loss_fn=msse, reduction="mean"):
    sum_loss = 0
    count = 0
    for past, future in dataset:
        result = eval(past, future, population, params, loss_fn=loss_fn)
        sum_loss += result.loss
    if reduction == "sum":
        return sum_loss
    elif reduction in ("mean", "avg"):
        return sum_loss / count
    else:
        raise Exception(f"Invalid reduction \"{reduction}\"")
