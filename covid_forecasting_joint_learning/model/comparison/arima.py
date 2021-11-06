from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..loss_common import msse, rmsse, wrap_reduce
import numpy as np
from math import sqrt, ceil
import optuna
import re
from itertools import chain
import pandas as pd
from optuna.structs import TrialPruned
from numpy.linalg import LinAlgError


msse = wrap_reduce(msse)
rmsse = wrap_reduce(rmsse)


class ARIMAModel:
    def __init__(self, order=None, seasonal_order=None, limit_fit=None, reduction="mean"):
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.fit_result = None
        self.limit_fit = limit_fit
        self.reduction = reduction
        self.loss = None

    def fit(self, endo, exo=None):
        if self.limit_fit:
            endo = endo[-self.limit_fit:]
            if exo is not None:
                exo = exo[-self.limit_fit:]
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
            loss = sum_loss
        elif reduction in ("mean", "avg"):
            loss = sum_loss / count
        else:
            raise Exception(f"Invalid reduction \"{reduction}\"")
        self.loss = loss
        return loss


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
    return best_model, best_loss


def search_optuna(orders, train_set, loss_fn=msse, use_exo=False, reduction="mean", limit_past_min=7, limit_past_max=366, no_limit=False, n_trials=None):

    def make_objective(order_set_0, limit_fit_0, no_limit_0):
        def objective(trial):
            order_set = trial.suggest_categorical("order", order_set_0)
            order_set = orders[order_set]
            order = order_set[0]
            seasonal_order = order_set[1] if len(order_set) > 1 else None

            no_limit = no_limit_0
            if no_limit is None:
                no_limit = trial.suggest_categorical("no_limit", (False, True))

            if no_limit:
                limit_fit = None
            else:
                limit_fit = trial.suggest_int("limit_fit", *limit_fit_0)

            try:
                model = ARIMAModel(order, seasonal_order, limit_fit=limit_fit, reduction=reduction)
                loss = model.eval_dataset(train_set, loss_fn=loss_fn, use_exo=use_exo)
                return loss
            except (LinAlgError, IndexError) as ex:
                raise TrialPruned(str(ex))
        return objective

    n_orders = len(orders)

    if n_trials is None:
        n_trials = (limit_past_max - limit_past_min + 1)

    study = optuna.create_study()
    orders_id = list(range(n_orders))
    study.optimize(
        make_objective(
            orders_id,
            (limit_past_max, limit_past_max),
            False
        ),
        n_trials=n_orders * n_orders,
        n_jobs=1
    )

    study.optimize(
        make_objective(
            orders_id,
            (limit_past_min, limit_past_max),
            no_limit
        ),
        n_trials=n_trials,
        n_jobs=1
    )
    return study


ARIMA_REGEX = re.compile(r"(\(([\d]+)\, *(([\d]+)\, *)?([\d]+)\))? *(\(([\d]+)\, *(([\d]+)\, *)?([\d]+)\)([\d]+))?$")

def _parse_arima_regex(m):
    order = (
        int(m.group(2)),
        int(m.group(4)) if m.group(3) else 0,
        int(m.group(5))
    ) if m.group(1) else None
    seasonal_order = (
        int(m.group(7)),
        int(m.group(9)) if m.group(8) else 0,
        int(m.group(10)),
        int(m.group(11))
    ) if m.group(6) else None
    return order, seasonal_order

def combine_arima(a, b):
    if a[0] is None and b[1] is None:
        return (b[0], a[1])
    elif a[1] is None and b[0] is None:
        return (a[0], b[1])
    else:
        raise Exception(f"Invalid ARIMA combination: {a} x {b}")


ARIMA_DEFAULT = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1)
]


def parse_arima_string(s):
    s = [sg.strip() for sg in s.split("|")]
    s = [sg for sg in s if sg]
    if not s:
        return ARIMA_DEFAULT
    if len(s) > 1:
        return list(chain.from_iterable([parse_arima_string(sg) for sg in s]))
    s = [sx.strip() for sg in s for sx in sg.split("x")]
    s = [sx for sx in s if sx]
    if not s:
        return ARIMA_DEFAULT
    if len(s) > 1:
        assert len(s) == 2
        s = [parse_arima_string(sg) for sg in s]
        assert len(s) == 2
        s, sb = s
        s = [[combine_arima(si, sbi) for sbi in sb] for si in s]
        return list(chain.from_iterable([sg for sg in s]))
    s = [si.strip() for sg in s for si in sg.split(";")]
    s = [si for si in s if si]
    if not s:
        return ARIMA_DEFAULT
    m = [ARIMA_REGEX.match(si) for si in s]
    m = [mi for mi in m if mi]
    if not m:
        return ARIMA_DEFAULT
    orders = [_parse_arima_regex(mi) for mi in m]
    return orders

class ARIMASearchLog:
    def __init__(self, source_path, log_path, source_sheet_name="ARIMA", log_sheet_name="ARIMA"):
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
        except FileNotFoundError:
            self.log_df = pd.DataFrame([], columns=["group", "cluster", "kabko", "label", "order", "seasonal_order", "limit_fit", "loss"])
            self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)
        return self.log_df

    def save_log(self, log_path=None, log_sheet_name=None):
        log_path = log_path or self.log_path
        log_sheet_name = log_sheet_name or self.log_sheet_name
        self.log_df.to_excel(log_path, sheet_name=log_sheet_name, index=False)

    def is_search_done(self, group, cluster, kabko, label, df=None):
        df = df or self.log_df
        return ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko) & (df["label"] == label)).any()

    def log(self, group, cluster, kabko, label, order, seasonal_order, limit_fit, loss, log_path=None, log_sheet_name=None):
        df = self.load_log()
        df.loc[df.shape[0]] = {
            "group": group,
            "cluster": cluster,
            "kabko": kabko,
            "label": label,
            "order": order,
            "seasonal_order": seasonal_order,
            "limit_fit": limit_fit,
            "loss": loss
        }
        self.save_log(log_path=log_path, log_sheet_name=log_sheet_name)

    def read_arima(self, kabko, label):
        df = self.source_df
        return df[(df["kabko"] == kabko)][label].item()

class ARIMAEvalLog(ARIMASearchLog):
    def __init__(self, source_path, log_path, source_sheet_name="ARIMA", log_sheet_name="Eval"):
        super().__init__(source_path, log_path, source_sheet_name=source_sheet_name, log_sheet_name=log_sheet_name)

    def is_search_done(self, group, cluster, kabko, label):
        return super().is_search_done(group, cluster, kabko, label, df=self.source_df)

    def is_eval_done(self, group, cluster, kabko, label):
        return super().is_search_done(group, cluster, kabko, label, df=self.log_df)

    def read_arima(self, group, cluster, kabko, label):
        df = self.log_df
        cond = ((df["group"] == group) & (df["cluster"] == cluster) & (df["kabko"] == kabko) & (df["label"] == label))
        return df[cond]
