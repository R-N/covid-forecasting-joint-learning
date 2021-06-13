from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
# import itertools
from collections import Counter
import numpy as np


class Cluster:
    def __init__(
        self,
        cluster_id,
        sources=None,
        target=None
    ):
        self.cluster_id = cluster_id
        self.sources = sources
        self.target = target


def cluster(
    dataset,
    n_clusters,
    n_init=3,
    max_iter=50,
    metric="dtw"
):
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        metric=metric
    )
    labels = model.fit_predict(dataset)
    return model, labels


class ClusteringResult:
    def __init__(
        self,
        n_clusters,
        model,
        labels,
        silhouette
    ):
        self.n_clusters = n_clusters
        self.model = model
        self.labels = labels
        self.silhouette = silhouette
        # Count non-single clusters
        counter = list(Counter(labels).items())
        counts = [g[1] for g in counter]
        self.n_clusters_single = len([count for count in counts if count < 2])
        self.n_clusters_non_single = self.n_clusters - self.n_clusters_single
        self.single_clusters = [g[0] for g in counter if g[1] < 2]
        print(self.n_clusters_single, self.single_clusters)


def cluster_best(
    dataset,
    n_clusters_min=2,
    n_clusters_max=10,
    n_init=3,
    max_iter=50,
    metric="dtw",
    good_clustering_non_single=2,
    min_silhouette_percentile=0.75,
    max_silhouette_diff=0.25
):
    trial_labels = [(n, *cluster(
        dataset,
        n,
        n_init=n_init,
        max_iter=max_iter,
        metric=metric
    )) for n in range(n_clusters_min, n_clusters_max+1)]

    trial_results = [ClusteringResult(n, model, labels, silhouette_score(
        dataset,
        labels,
        metric=metric
    )) for n, model, labels in trial_labels]

    # yeah so I forgot that there can be a cluster with single member
    # there can be 2 approach to handle this
    # a. Still pick k with highest silhouette despite having clusters with 
    # single members and just exclude those clusters. The downside is that
    # this can result in one massive cluster and clustering only removes
    # outliers.
    # b. Pick k which will produce least single clusters then select one
    # with highest silhouette. The downside is that it might be a bad k
    # since the silhouette is now a secondary measure. 
    # Or maybe use combination of both, with some formula or rule, idk
    # Maybe slice the upper quartile/median of the silhouette first then pick
    # one with least single cluster count?

    # I decided to first filter the clusters to have at least 2 non-single clusters
    # That way the clustering works
    trial_results_2 = [r for r in trial_results if r.n_clusters_non_single >= good_clustering_non_single]
    # I must first check that it exists before using it
    good_clustering = len(trial_results_2) > 0
    if good_clustering:
        trial_results_3 = trial_results_2
    else:
        # Else I'll just return best one to remove outliers
        trial_results_3 = trial_results

    # Then I will filter it to just the upper quartile of silhouette
    silhouettes = [r.silhouette for r in trial_results_3]
    min_silhouette = np.percentile(silhouettes, min_silhouette_percentile)
    trial_results_3 = [r for r in trial_results_3 if r.silhouette >= min_silhouette]
    # It should also not be that far off the best silhouette
    best_silhouette = max(silhouettes)
    trial_results_3 = [r for r in trial_results_3 if best_silhouette-r.silhouette <= max_silhouette_diff]

    # if good_clustering:
    #     # Then I'll pick the one with most non-single clusters, 
    #     # and the one with best silhouette if there are more than one
    #     best_result = max(trial_results_3, key=lambda r: (r.n_clusters_non_single, r.silhouette))
    # else:
    #     # But if it's useless clustering, then I will pick one with least single clusters
    #     # So that I won't remove too much
    # Nevermind. I shouldn't have maximized cluster number. 
    # There's no good reason for that
    best_result = min(trial_results_3, key=lambda r: (r.n_clusters_single, -r.silhouette))
    return best_result


def predict(model, data):
    return model.predict(to_time_series_dataset([data]))[0]

