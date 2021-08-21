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
    data_center.raw_global = preprocessing.trim_zero_crit(
        data_center.raw_global,
        labels=[DataCol.I_TOT_GLOBAL],
        crit_labels=[DataCol.S_GLOBAL_PEOPLE, DataCol.S_GLOBAL_FULL]
    )
    data_center.raw_global.dropna(inplace=True)
    df_shifted = data_center.raw_global.shift()
    delta = data_center.raw_global.copy()
    delta = sird.calc_delta_global(delta, df_shifted)
    delta = preprocessing.trim_zero_crit(
        delta,
        labels=[DataCol.I_TOT_GLOBAL],
        crit_labels=[DataCol.DELTA_TEST]
    )
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
    crit_labels=[DataCol.I],
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
    k.raw = preprocessing.trim_zero_crit(
        k.raw,
        labels=trim_labels,
        crit_labels=crit_labels
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
    limit_split=True,
    scale=False
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
        if scale:
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
    max_silhouette_diff=0.1,
    scale=True,
    **kwargs
):
    for k in group.members:
        k.data_clustering = k.data.loc[:k.split_indices[2], cols].copy()
        if scale:
            k.data_clustering = k.scaler.transform(k.data_clustering)

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
        clusters = [clustering.Cluster(0, group, [])]
        for k in group.members:
            clusters[0].sources.append(k)
            k.cluster = clusters[0]
    else:
        clusters = [clustering.Cluster(i, group, []) for i in range(best_clustering.n_clusters)]
        for k in group.members:
            k.cluster = clustering.predict(best_clustering.model, k.data_clustering)
            clusters[k.cluster].sources.append(k)
            k.cluster = clusters[k.cluster]
        # Remove single clusters as outliers
        clusters = [c for c in clusters if len(c.sources) > 1]

    for c in clusters:
        target = max(c.sources, key=lambda x: clustering.shortest(x))
        c.sources.remove(target)
        c.target = target
        # Remove outliers if they're not target
        c.sources = [k for k in c.sources if k not in outliers]
        for s in c.sources:
            assert c.target.data.last_valid_index() >= s.data.last_valid_index()
    group.clustering_info = best_clustering.get_info()
    group.clusters = clusters
    return clusters


DEFAULT_GROUPS_N_CLUSTERS = (4, 3, 2, 2)


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
    Scaler=preprocessing.MinMaxScaler,
    scale=True
):
    kabkos = cluster.members
    target_split_indices = cluster.target.split_indices
    # target_first_split_index = target_split_indices[0]
    target_last_index = cluster.target.data.last_valid_index()
    for kabko in kabkos:
        assert target_last_index >= kabko.data.last_valid_index()
        kabko.data = kabko.parent.data[:target_last_index].copy()
        kabko.split_indices = target_split_indices
    scaler = __preprocessing_3(
        kabkos,
        cols=cols,
        Scaler=Scaler
    )
    for kabko in kabkos:
        kabko.scaler_2 = scaler
        if scale:
            kabko.data.loc[:, cols] = scaler.transform(kabko.data[cols])
    return cluster


def preprocessing_5(
    kabkos,
    past_size=30, future_size=14,
    stride=1,
    past_cols=None,
    label_cols=DataCol.SIRD_VARS,
    future_exo_cols=["psbb", "ppkm", "ppkm_mikro"],
    final_seed_cols=DataCol.SIRD,
    final_cols=DataCol.IRD,
    # limit_past=True,
    # val=True
):
    for kabko in kabkos:
        split_indices = kabko.split_indices[1], kabko.split_indices[3]
        split_indices = [kabko.data.index.get_loc(s) for s in split_indices]
        val_start, test_start = split_indices

        kabko.datasets, kabko.dataset_labels = preprocessing.split_dataset(
            kabko.data,
            past_size=past_size, future_size=future_size,
            val_start=val_start, test_start=test_start,
            stride=1,
            past_cols=past_cols, label_cols=label_cols, future_exo_cols=future_exo_cols,
            final_seed_cols=final_seed_cols, final_cols=final_cols,
            limit_past=True,
            val=True,
            labeling=preprocessing.label_dataset_0
        )

        kabko.datasets_1, kabko.dataset_labels_1 = preprocessing.split_dataset(
            kabko.data,
            past_size=past_size, future_size=future_size,
            val_start=val_start, test_start=test_start,
            stride=1,
            past_cols=past_cols, label_cols=label_cols, future_exo_cols=future_exo_cols,
            final_seed_cols=final_seed_cols, final_cols=final_cols,
            limit_past=False,
            val=False,
            labeling=preprocessing.label_dataset_1
        )

        kabko.datasets_2, kabko.dataset_labels_2 = preprocessing.split_dataset(
            kabko.data,
            past_size=past_size, future_size=future_size,
            val_start=val_start, test_start=test_start,
            stride=1,
            past_cols=past_cols, label_cols=label_cols, future_exo_cols=future_exo_cols,
            final_seed_cols=final_seed_cols, final_cols=final_cols,
            limit_past=False,
            val=False,
            labeling=preprocessing.label_dataset_2
        )


def preprocessing_6(
    kabkos,
    batch_size=8,
    pin_memory=False,
    shuffle=True,
    tensor_count=7
):
    for kabko in kabkos:
        kabko.datasets_torch = [[
            tuple(
                torch.Tensor(sample[i]) if i < tensor_count else sample[i] for i in range(8)
            ) for sample in dataset
        ] for dataset in kabko.datasets]

        def collate_fn(samples):
            samples_1 = tuple([sample[i] for sample in samples] for i in range(8))
            samples_1 = tuple(torch.stack(samples_1[i]).detach() if i < tensor_count else samples_1[i] for i in range(len(samples_1)))
            samples_1 = samples_1 + (kabko.population, kabko,)
            return samples_1

        dataset_count = len(kabko.datasets_torch)
        last = dataset_count - 1
        test_size = len(kabko.datasets_torch[last])

        kabko.dataloaders = [DataLoader(
            kabko.datasets_torch[i],
            batch_size=batch_size if i != last else test_size,
            shuffle=shuffle if i != last else False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=pin_memory
        ) for i in range(dataset_count)]
