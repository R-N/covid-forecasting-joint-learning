from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
# import itertools
from collections import Counter


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


def single_cluster_count(
    labels
):
    counts = [g[1] for g in Counter(labels).items()]
    # counts = [len(list(g[1])) for g in itertools.groupby(labels)]
    return len([count for count in counts if count < 2])


def cluster_best(
    dataset,
    n_clusters_min=2,
    n_clusters_max=10,
    n_init=3,
    max_iter=50,
    metric="dtw"
):
    trial_labels = [(n, *cluster(
        dataset,
        n,
        n_init=n_init,
        max_iter=max_iter,
        metric=metric
    )) for n in range(n_clusters_min, n_clusters_max+1)]
    trial_results = [(n, model, labels, silhouette_score(
        dataset,
        labels,
        metric=metric
    )) for n, model, labels in trial_labels]
    # yeah so I forgot that there can be a cluster with single member
    # there can be 2 approach to handle this
    # a. Still pick k with highest silhouette despite having clusters with 
    # single members and just exclude those clusters. The downside is that
    # this can result in one massive cluster.
    # b. Pick k which will produce least single clusters then select one
    # with highest silhouette. The downside is that it might be a bad k
    # since the silhouette is now a secondary measure. This is done below.
    # Or maybe use combination of both, with some formula or rule, idk
    # Maybe slice the upper quartile/median of the silhouette first then pick
    # one with least single cluster count?
    best_result = min(trial_results, key=lambda x: (single_cluster_count(x[2]), 1-x[3]))
    # n_cluster_best, model_best, labels_best, silhouette_best
    return best_result
