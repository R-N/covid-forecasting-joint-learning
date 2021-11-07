from scipy.integrate import odeint
from lmfit import minimize, Parameters
import numpy as np
from ..loss_common import msse, rmsse, wrap_reduce
from ...data import cols as DataCol
import pandas as pd
import optuna
from xlrd import XLRDError

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
    def __init__(self, params_hint, n, loss_fn=rmsse, limit_fit=None, reduction="mean"):
        self.params_hint = params_hint
        self.n = n
        self.loss_fn = loss_fn
        self.loss = None
        self.limit_fit = limit_fit
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

    def fit(self, past, limit_fit=None):
        self.clear()

        limit_fit = limit_fit or self.limit_fit
        if limit_fit:
            past = past[-limit_fit:]

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


    def eval(self, past, future, loss_fn=rmsse, limit_fit=None):
        self.fit(past, limit_fit=limit_fit)
        return self.test(past, future, loss_fn=loss_fn)


    def eval_dataset(self, dataset, loss_fn=rmsse, reduction=None, limit_fit=None):
        reduction = reduction or self.reduction
        losses = [
            self.eval(past, future, loss_fn=loss_fn, limit_fit=limit_fit)
            for past, future, indices in dataset
        ]
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


def search_optuna(params_hint, n, dataset, loss_fn=msse, reduction="mean", limit_fit_min=7, limit_fit_max=366, no_limit=False, n_trials=None):
    def objective(trial):
        no_limit_1 = no_limit
        if no_limit_1 is None:
            no_limit_1 = trial.suggest_categorical("no_limit", (False, True))

        if no_limit_1:
            limit_fit = None
        else:
            limit_fit = trial.suggest_int("limit_fit", limit_fit_min, limit_fit_max)

        model = SIRDModel(params_hint=params_hint, n=n, loss_fn=loss_fn, reduction=reduction, limit_fit=limit_fit)
        return model.eval_dataset(dataset)

    if n_trials is None:
        n_trials = (limit_fit_max - limit_fit_min + 1)

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


def search_greedy(params_hint, n, dataset, loss_fn=rmsse, reduction="mean", limit_fit_min=7, limit_fit_max=366):
    best_model = None
    best_loss = np.inf
    for limit_fit in [*range(limit_fit_min, limit_fit_max + 1), None]:
        model = SIRDModel(params_hint=params_hint, n=n, loss_fn=loss_fn, reduction=reduction, limit_fit=limit_fit)
        loss = model.eval_dataset(dataset)
        if loss < best_loss:
            best_model = model
            best_loss = loss
    return best_model, best_loss


class SIRDSearchLog:
    def __init__(self, log_path, log_sheet_name="SIRD"):
        self.log_path = log_path
        self.log_sheet_name = log_sheet_name
        self.load_log()

    def load_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        try:
            self.log_df = pd.read_excel(log_path, sheet_name=log_sheet_name)
        except (FileNotFoundError, ValueError, XLRDError):
            self.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "limit_fit", "loss"])
            self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
        return self.log_df

    def save_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        self.log_df.to_excel(log_path, sheet_name=log_sheet_name, index=False)

    def is_search_done(self, group, cluster, kabko):
        df = self.log_df
        try:
            return ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko)).any()
        except (ValueError, XLRDError) as ex:
            if "No sheet" in str(ex) or "is not in list" in str(ex):
                return False
            raise

    def log(self, group, cluster, kabko, limit_fit, loss):
        df = self.load_log()
        df.loc[df.shape[0]] = {
            "group": group,
            "cluster": cluster,
            "kabko": kabko,
            "limit_fit": limit_fit,
            "loss": loss
        }
        self.save_log()

class SIRDEvalLog:
    def __init__(self, source_path, log_path, source_sheet_name="SIRD", log_sheet_name="Eval"):
        self.source_path = source_path
        self.log_path = log_path
        self.source_sheet_name = source_sheet_name
        self.log_sheet_name = log_sheet_name
        self.source_df = pd.read_excel(source_path, sheet_name=source_sheet_name)
        self.load_log()

    def load_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        try:
            self.log_df = pd.read_excel(log_path, sheet_name=log_sheet_name)
        except (FileNotFoundError, ValueError, XLRDError):
            self.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "limit_fit", "i", "r", "d"])
            self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
        return self.log_df

    def save_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        self.log_df.to_excel(log_path, sheet_name=log_sheet_name, index=False)

    def is_search_done(self, group, cluster, kabko, df=None):
        df = self.source_df if df is None else df
        try:
            return ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko)).any()
        except (ValueError, XLRDError) as ex:
            if "No sheet" in str(ex) or "is not in list" in str(ex):
                return False
            raise

    def is_eval_done(self, group, cluster, kabko):
        df = self.log_df
        return self.is_search_done(group, cluster, kabko, df=df)

    def log(self, group, cluster, kabko, limit_fit, loss, log_path=None, log_sheet_name=None):
        df = self.load_log()
        df.loc[df.shape[0]] = {
            "group": group,
            "cluster": cluster,
            "kabko": kabko,
            "limit_fit": limit_fit,
            "i": loss[0],
            "r": loss[1],
            "d": loss[2]
        }
        self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)

    def read_sird(self, group, cluster, kabko):
        df = self.log_df
        cond = ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko))
        return df[cond]
