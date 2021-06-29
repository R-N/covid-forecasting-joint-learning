from itertools import combinations
import pandas as pd
from . import preprocessing
from . import sird
from . import clustering
from ..data.kabko import KabkoData
from ..data import cols as DataCol
import torch
from torch.utils.data import DataLoader, Dataset


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
    prev = len(k.data)
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
    limit_length=[90, 180, 366],
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
    Scaler=preprocessing.MinMaxScaler,
    limit_split=True
):
    if limit_split:
        data = [kabko.data.loc[:kabko.split_indices[2], cols] for kabko in kabkos]
    else:
        data = [kabko.data.loc[:, cols] for kabko in kabkos]
    full_data = pd.concat(data)
    scaler = Scaler()
    scaler.fit(full_data)
    return scaler


def preprocessing_3(
    kabkos,
    cols=DataCol.SIRD_VARS,
    Scaler=preprocessing.MinMaxScaler,
    limit_split=True
):
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler,
        limit_split=limit_split
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
    n_init=1,
    max_iter=50,
    max_iter_barycenter=100,
    metric="dtw",
    random_state=257,
    good_clustering_non_single=2,
    min_silhouette_percentile=0.75,
    max_silhouette_diff=0.25,
    **kwargs
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
            max_iter_barycenter=max_iter_barycenter,
            metric=metric,
            random_state=random_state,
            good_clustering_non_single=good_clustering_non_single,
            min_silhouette_percentile=min_silhouette_percentile,
            max_silhouette_diff=max_silhouette_diff,
            **kwargs
        )
        outliers += [k for k in clustering_members if clustering.predict(best_clustering.model, k.data_clustering) in best_clustering.single_clusters]
        if best_clustering.n_clusters_non_single >= good_clustering_non_single:
            break
        print("Removing %s outliers from %s" % (len(outliers), len(clustering_members)))
        clustering_members = [k for k in clustering_members if k not in outliers]

    if len(clustering_members) == 0 or best_clustering.n_clusters_non_single < good_clustering_non_single:
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
        clusters = [c for c in clusters if len(c.sources) > 1]

    for c in clusters:
        target = max(c.sources, key=lambda x: (-len(x.data), x.data.last_valid_index(), x.data.first_valid_index()))
        c.sources.remove(target)
        c.target = target
        # Remove outliers if they're not target
        c.sources = [k for k in c.sources if k not in outliers]
        for s in c.sources:
          assert c.target.data.last_valid_index() >= s.data.last_valid_index()
    group.clustering_info = best_clustering.get_info()
    group.clusters = clusters
    return clusters


def clustering_consistency(
    groups,
    n_samples=3,
    **kwargs
):
    clustering_results = []
    silhouettes = []
    n_groups = len(groups)
    for i in range(n_samples):
        clusters = [clustering_1(
            group,
            **kwargs
        ) for group in groups]
        silhouettes_i = [group.clustering_info.silhouette for group in groups]
        silhouette_i = sum(silhouettes_i) / n_groups
        silhouettes.append(silhouette_i)
        clusters = [[{k.name for k in cluster.members} for cluster in group.clusters] for group in groups]
        clustering_results.append(clusters)
    comb = list(combinations(list(range(n_samples)), 2))
    total_similarity = 0
    for a, b in comb:
        a, b = clustering_results[a], clustering_results[b]
        similarities = [clustering.pairwise_clustering_similarity(a[i], b[i]) for i in range(len(groups))]
        similarity = sum(similarities) / len(groups)
        total_similarity += similarity
    consistency = total_similarity / len(comb)
    silhouette = sum(silhouettes) / n_samples
    return consistency, silhouette


def preprocessing_4(
    cluster,
    cols=DataCol.SIRD_VARS,
    Scaler=preprocessing.MinMaxScaler
):
    kabkos = cluster.members
    target_split_indices = cluster.target.split_indices
    # target_first_split_index = target_split_indices[0]
    target_last_index = cluster.target.data.last_valid_index()
    for kabko in kabkos:
        assert target_last_index >= kabko.data.last_valid_index()
        kabko.data = kabko.parent.data[:target_last_index]
        kabko.split_indices = target_split_indices
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler_2 = scaler
        kabko.data.loc[:, cols] = scaler.transform(kabko.data[cols])
    return cluster


def preprocessing_5(
    kabkos,
    past_size=30, future_size=14,
    stride=1,
    past_cols=None,
    label_cols=DataCol.SIRD_VARS,
    future_exo_cols=["psbb", "ppkm", "ppkm_mikro"]
):
    for kabko in kabkos:
        split_indices = kabko.split_indices[1], kabko.split_indices[3]
        split_indices = [kabko.data.index.get_loc(s) for s in split_indices]
        val_start, test_start = split_indices

        kabko.datasets = preprocessing.split_dataset(
            kabko.data,
            past_size=past_size, future_size=future_size,
            val_start=val_start, test_start=test_start,
            stride=1,
            past_cols=past_cols, label_cols=label_cols, future_exo_cols=future_exo_cols
        )

def preprocessing_6(
    kabkos,
    batch_size=5
):
    for kabko in kabkos:
        def collate_fn(samples):
            keys = list(samples[0].keys())
            samples_1 = {key: torch.stack(
                [samples[i] for i in range(len(samples))]
            ) for key in keys}
            samples_1["kabko"] = kabko
            return samples_1

        kabko.dataloaders = [DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        ) for dataset in kabko.datasets]
