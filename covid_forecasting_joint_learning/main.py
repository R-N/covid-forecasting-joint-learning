from .data import cols as DataCol
from .pipeline import main as Pipeline
from .data.center import DataCenter
from .pipeline import util as PipelineUtil
from .model import attr as Attribution
from .model import util as ModelUtil
import gc
import torch
import pandas as pd


def init():
    Attribution.init_ipython()
    Attribution.init_matplotlib()
    pd.options.mode.chained_assignment = None
    device = ModelUtil.init(cuda=True, half=False)
    return device


def main_1(data_path, labeled_dates=DataCol.LABELED_DATES, cols=DataCol.COLS):
    loader = DataCenter()
    loader.load_excel(data_path)
    loader = Pipeline.preprocessing_0(loader)

    kabkos = Pipeline.get_kabkos(loader)
    # kabko_indices = {kabkos[i].name: i for i in range(len(kabkos))}
    kabkos = Pipeline.preprocessing_1(kabkos)

    for kabko in kabkos:
        kabko.data = kabko.add_dates(kabko.data, dates=labeled_dates)[cols]

    groups = Pipeline.preprocessing_2(kabkos)
    # groups = [groups[2]]
    for group in groups:
        Pipeline.preprocessing_3(group.members)

    n_clusters = Pipeline.DEFAULT_GROUPS_N_CLUSTERS

    clusters = [Pipeline.clustering_1(
        group,
        n_clusters_min=n_cluster,
        n_clusters_max=n_cluster
    ) for group, n_cluster in zip(groups, n_clusters)]

    for group in groups:
        for cluster in group.clusters:
            Pipeline.preprocessing_4(cluster)

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
        trials_done = ModelUtil.count_trials_done(trials)
        n_trials_remain -= trials_done
        cond = n_trials_remain > 0 or trials_done == 0
