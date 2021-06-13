import pandas as pd
from . import preprocessing
from . import sird
from . import clustering
from ..data.kabko import KabkoData
from ..data import cols as DataCol


def preprocessing_0(
    data_center
):
    data_center.set_global_ts(
        vaccine=preprocessing.handle_zero(
            data_center.vaccine,
            trim_labels=DataCol.VAC_ALL,
            fill_labels=DataCol.VAC_ALL
        ),
        test=preprocessing.handle_zero(
            data_center.test,
            trim_labels=[DataCol.TEST],
            fill_labels=[DataCol.TEST]
        ),
        covid_global=preprocessing.handle_zero(
            data_center.covid_global,
            trim_labels=[DataCol.I_TOT_GLOBAL],
            fill_labels=[DataCol.I_TOT_GLOBAL]
        )
    )
    data_center.raw_global = data_center.covid_global.copy()
    data_center.raw_global[DataCol.VAC_ALL] = data_center.vaccine[DataCol.VAC_ALL]
    data_center.raw_global[DataCol.TEST] = data_center.test[DataCol.TEST]
    data_center.raw_global = preprocessing.handle_zero(
        data_center.raw_global,
        trim_labels=[DataCol.I_TOT_GLOBAL],
        fill_labels=[
            *DataCol.VAC_ALL,
            DataCol.TEST,
            DataCol.I_TOT_GLOBAL
        ]
    )
    data_center.raw_global = sird.calc_s_global(data_center.raw_global, data_center.population_global)
    data_center.raw_global.dropna(inplace=True)
    df_shifted = data_center.raw_global.shift()
    delta = data_center.raw_global.copy()
    delta = sird.calc_delta_global(delta, df_shifted)
    data_center.data_global = delta
    data_center.data_global = sird.calc_vars_global(data_center.data_global, df_shifted)
    data_center.data_global.dropna(inplace=True)
    return data_center


def get_kabkos(data_center):
    kabkos = data_center.kabko
    kabkos = [KabkoData(k, data_center) for k in kabkos]
    return kabkos


def __preprocessing_1(
    k,
    trim_labels=DataCol.IRD,
    fill_labels=[
        *DataCol.IRD,
        *DataCol.VAC_ALL,
        DataCol.I_TOT_GLOBAL,
        DataCol.TEST
    ],
    interpolation_method="linear"
):
    k.raw = k.covid.copy()
    k.raw[k.data_global.columns] = k.data_global
    k.raw = preprocessing.handle_zero(
        k.raw,
        trim_labels=trim_labels,
        fill_labels=fill_labels,
        interpolation_method=interpolation_method
    )
    k.raw = sird.calc_s(k.raw, k.population)
    k.raw.dropna(inplace=True)
    df_shifted = k.raw.shift()
    delta = k.raw.copy()
    delta = sird.calc_delta(delta, df_shifted)
    # delta.dropna(inplace=True)
    k.data = delta
    k.data = sird.calc_vars(k.data, k.population, df_shifted)
    k.data.dropna(inplace=True)
    return k


def preprocessing_1(
    kabkos
):
    return [__preprocessing_1(k) for k in kabkos]


def filter_cols(
    kabkos,
    cols
):
    for kabko in kabkos:
        kabko.data = kabko.data[cols]
    return kabkos


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
    kabko = kabko.copy()
    kabko.data = df
    kabko.split_indices = split_indices
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
    cols=DataCol.SIRD_VARS,
    Scaler=preprocessing.MinMaxScaler
):
    data = [kabko.data.loc[:kabko.split_indices[2], cols] for kabko in kabkos]
    full_data = pd.concat(data)
    scaler = Scaler()
    scaler.fit(full_data)
    return scaler


def preprocessing_3(
    kabkos,
    cols=DataCol.SIRD_VARS,
    Scaler=preprocessing.MinMaxScaler
):
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler = scaler
        # This produces warnings for whatever reason idk
        # I've used loc everywhere
        kabko.data.loc[:, cols] = scaler.transform(kabko.data.loc[:, cols])
    return kabkos


def clustering_1(
    group,
    cols=DataCol.SIRD_VARS,
    n_clusters_min=2,
    n_clusters_max=10,
    n_init=3,
    max_iter=50,
    metric="dtw",
    good_clustering_non_single=2,
    min_silhouette_percentile=0.75,
    max_silhouette_diff=0.25
):
    for k in group.members:
        # k.data_clustering = k.scaler.transform(k.data_train_val[cols])
        k.data_clustering = k.data.loc[:k.split_indices[2], cols]

    clustering_members = list(group.members)
    outliers = []
    while len(clustering_members) > 0:
        dataset = [k.data_clustering for k in clustering_members]
        dataset = clustering.to_time_series_dataset(dataset)
        best_clustering = clustering.cluster_best(
            dataset,
            n_clusters_min=n_clusters_min,
            n_clusters_max=n_clusters_max,
            n_init=n_init,
            max_iter=max_iter,
            metric=metric,
            good_clustering_non_single=good_clustering_non_single,
            min_silhouette_percentile=min_silhouette_percentile,
            max_silhouette_diff=max_silhouette_diff
        )
        outliers += [k for k in clustering_members if clustering.predict(best_clustering.model, k.data_clustering) in best_clustering.single_clusters]
        if best_clustering.n_clusters_non_single >= good_clustering_non_single:
            break
        print("Removing %s outliers from %s" % (len(outliers), len(clustering_members)))
        clustering_members = [k for k in clustering_members if k not in outliers]

    if len(clustering_members) == 0 or clustering.n_clusters_non_single == 0:
        # I want to see first if this will ever happen
        raise Exception("Group can't be clustered well")
        clusters = [clustering.Cluster(0, [])]
        for k in group.members:
            clusters[0].sources.append(k)
            k.cluster = clusters[0]
    else:
        clusters = [clustering.Cluster(i, []) for i in range(best_clustering.n_clusters)]
        for k in group.members:
            k.cluster = clustering.predict(best_clustering.model, k.data_clustering)
            clusters[k.cluster].sources.append(k)
            k.cluster = clusters[k.cluster]
        # Remove single clusters as outliers
        clusters = [c for c in clusters if len(c.sources) < 2]

    for c in clusters:
        target = min(c.sources, key=lambda x: len(x.data))
        c.sources.remove(target)
        c.target = target
        # Remove outliers if they're not target
        c.sources = [k for k in c.sources if k not in outliers]

    group.clusters = clusters
    return clusters

def preprocessing_4(
    cluster,
    cols=DataCol.SIRD_VARS,
    Scaler=preprocessing.MinMaxScaler
):
    kabkos = [*cluster.sources, cluster.targets]
    for kabko in kabkos:
        kabko.data = kabko.parent.data[:cluster.target.data.last_valid_index()]
        kabko.split_indices = cluster.target.split_indices
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler_2 = scaler
        kabko.data.loc[:, cols] = scaler.transform(kabko.data[cols])
        kabko.data = preprocessing.split_dataset(
            kabko.data,
            kabko.split_indices
        )
    return kabkos

