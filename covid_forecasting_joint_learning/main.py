from .data import cols as DataCol
from .pipeline import main as Pipeline
from .data.center import DataCenter
from .pipeline import util as PipelineUtil
from .model import attr as Attribution
from .model import util as ModelUtil
import gc
import torch
import pandas as pd


def init(ipython=True):
    if ipython:
        Attribution.init_ipython()
    Attribution.init_matplotlib()
    pd.options.mode.chained_assignment = None
    device = ModelUtil.init(cuda=True, half=False)
    return device

def main_0(
    data,
    labeled_dates=DataCol.LABELED_DATES,
    cols=DataCol.COLS,
    limit_length=[90, 180, 366],
    limit_date=["2021-01-21"],
    n_clusters=Pipeline.DEFAULT_GROUPS_N_CLUSTERS,
    limit_data=True,
    kabkos=None,
    exclude=None
):
    if isinstance(data, DataCenter):
        loader = data
    else:
        loader = DataCenter()
        loader.load_excel(data)
        loader = Pipeline.preprocessing_0(loader)

    kabkos = Pipeline.get_kabkos(loader, kabkos=kabkos, exclude=exclude)
    # kabko_indices = {kabkos[i].name: i for i in range(len(kabkos))}
    kabkos = Pipeline.preprocessing_1(kabkos)

    for kabko in kabkos:
        kabko.data = kabko.add_dates(kabko.data, dates=labeled_dates)[cols]

    groups = Pipeline.preprocessing_2(
        kabkos,
        limit_length=limit_length,
        limit_date=limit_date
    )
    # groups = [groups[2]]
    for group in groups:
        Pipeline.preprocessing_3(group.members, limit_split=limit_data)

    if n_clusters:
        clusters = [Pipeline.clustering_1(
            group,
            n_clusters_min=n_cluster,
            n_clusters_max=n_cluster,
            limit_clustering=limit_data
        ) for group, n_cluster in zip(groups, n_clusters)]
    else:
        clusters = [Pipeline.clustering_1(
            group,
            limit_clustering=limit_data
        ) for group in groups]
    return groups

def main_1(
    data,
    labeled_dates=DataCol.LABELED_DATES,
    cols=DataCol.COLS,
    limit_length=[90, 180, 366],
    limit_date=["2021-01-21"],
    n_clusters=Pipeline.DEFAULT_GROUPS_N_CLUSTERS,
    limit_data=True,
    clustering_callback=None,
    kabkos=None,
    exclude=None
):
    groups = main_0(
        data=data,
        labeled_dates=labeled_dates,
        cols=cols,
        limit_length=limit_length,
        limit_date=limit_date,
        n_clusters=n_clusters,
        kabkos=kabkos,
        exclude=exclude
    )
    if clustering_callback:
        print("clustering_callback call")
        clustering_callback(groups)
    else:
        print("clustering_callback is falsy", clustering_callback)
    for group in groups:
        for cluster in group.clusters:
            Pipeline.preprocessing_4(cluster, limit_split=limit_data)

    return groups


def make_objective(
    f, groups,
    drive=None,
    past_cols=[DataCol.PAST_COLS],
    future_exo_cols=[DataCol.FUTURE_EXO_COLS],
    debug=False,
    **kwargs
):
    return f(
        groups,
        drive=drive,
        past_cols=past_cols,
        future_exo_cols=future_exo_cols,
        debug=debug,
        **kwargs
    )

def optimize(study, model_objective, n_jobs=1, batch=None, n_trials=10000):
    # torch.autograd.set_detect_anomaly(True)

    n_trials_remain = n_trials - PipelineUtil.count_trials_done(study.trials)
    batch = batch or n_trials_remain

    cond = True
    while cond:
        torch.cuda.empty_cache()
        gc.collect()
        trials = []

        def callback(study, trial):
            trials.append(trial)

        study.optimize(model_objective, n_trials=min(batch, n_trials_remain), n_jobs=n_jobs, callbacks=[callback])
        # joblib.dump(study, save_file)
        trials_done = PipelineUtil.count_trials_done(study.trials)
        n_trials_remain = n_trials - trials_done
        cond = n_trials_remain > 0 or trials_done == 0
