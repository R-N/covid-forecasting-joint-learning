import pandas as pd
from . import preprocessing
from . import sird
from . import clustering
from ..data.kabko import KabkoData
from ..data import cols as DataCol


def get_kabkos(data_center):
    kabkos = data_center.kabko
    kabkos = [KabkoData(k, data_center) for k in kabkos]
    return kabkos


def __preprocessing_1(
    k
):
    k.raw = k.covid.copy()
    k.raw[DataCol.VAC_ALL] = k.vaccine[DataCol.VAC_ALL]
    k.raw[DataCol.TEST] = k.test[DataCol.TEST]
    k.raw[DataCol.I_TOT_GLOBAL] = k.covid_global[DataCol.I_TOT_GLOBAL]
    k.raw = preprocessing.handle_zero(k.raw)
    k.raw = sird.calc_s(k.raw, k.population)
    k.raw = sird.calc_s_global(k.raw, k.population_global)
    k.raw.dropna(inplace=True)
    df_shifted = k.raw.shift()
    delta = k.raw.copy()
    delta = sird.calc_delta(delta, df_shifted)
    delta = sird.calc_delta_global(delta, df_shifted)
    # delta.dropna(inplace=True)
    k.data = delta
    k.data = sird.calc_vars(k.data, k.population, df_shifted)
    k.data = sird.calc_vars_global(k.data, df_shifted)
    k.data.dropna(inplace=True)
    return k


def preprocessing_1(
    kabkos
):
    return [__preprocessing_1(k) for k in kabkos]


def __preprocessing_2(
    kabko,
    df,
    val_portion=0.25,
    test_portion=0.25,
    past_size=30
):
    split_indices = preprocessing.calc_split(
        df,
        val_portion=val_portion,
        test_portion=test_portion,
        past_size=past_size
    )
    kabko = kabko.copy(
        data=df,
        split_indices=split_indices
    )
    return kabko


def preprocessing_2(
    kabkos,
    limit_length=[60, 180, 366],
    limit_date=["2021-01-21"],
    val_portion=0.25,
    test_portion=0.25,
    past_size=30
):
    kabko_dfs = [(k, k.data) for k in kabkos]
    groups = preprocessing.split_groups(
        kabko_dfs,
        limit_length=limit_length,
        limit_date=limit_date
    )
    for g in groups:
        g.members = [__preprocessing_2(
            kabko,
            df,
            val_portion=val_portion,
            test_portion=test_portion,
            past_size=past_size
        ) for kabko, df in g.members]
        for k in g.members:
            k.group = g
    return groups


def __preprocessing_3(
    kabkos,
    cols=["beta", "gamma", "delta"],
    Scaler=preprocessing.MinMaxScaler
):
    data = [kabko.data[:kabko.split_indices[2], cols] for kabko in kabkos]
    full_data = pd.concat(data)
    scaler = Scaler()
    scaler.fit(full_data)
    return scaler


def preprocessing_3(
    kabkos,
    cols=["beta", "gamma", "delta"],
    Scaler=preprocessing.MinMaxScaler
):
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler = scaler
        kabko.datasets.loc[:, cols] = scaler.transform(kabko.data[cols])
    return kabkos


def clustering_1(
    group,
    cols=["beta", "gamma", "delta"],
    n_clusters_min=2,
    n_clusters_max=10,
    n_init=3,
    max_iter=50,
    metric="dtw"
):
    for k in group.members:
        # k.data_clustering = k.scaler.transform(k.data_train_val[cols])
        k.data_clustering = k.datasets[:k.split_indices[2], cols]
    dataset = [k.data_clustering for k in group.members]
    dataset = clustering.to_time_series_dataset(dataset)
    n_cluster, model, labels, silhouette = clustering.cluster_best(
        dataset,
        n_clusters_min=n_clusters_min,
        n_clusters_max=n_clusters_max,
        n_init=n_init,
        max_iter=max_iter,
        metric=metric
    )
    clusters = [clustering.Cluster(i, [], []) for i in range(n_cluster)]
    for k in group.members:
        k.cluster = model.predict(clustering.to_time_series_dataset([k.data_clustering]))[0]
        clusters[k.cluster].sources.append(k)
        k.cluster = clusters[k.cluster]
    for c in clusters:
        target = min(c.sources, key=lambda x: len(x.data))
        c.sources.remove(target)
        c.targets.append(target)
    group.clusters = clusters
    return clusters


def preprocessing_4(
    kabkos,
    cols=["beta", "gamma", "delta"],
    Scaler=preprocessing.MinMaxScaler
):
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler_2 = scaler
        kabko.datasets.loc[:, cols] = scaler.transform(kabko.data[cols])
        kabko.datasets = preprocessing.split_dataset(
            kabko.data,
            kabko.split_indices
        )
    return kabkos

