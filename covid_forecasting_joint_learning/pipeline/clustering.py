from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


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
    # n_cluster_best, model_best, labels_best, silhouette_best
    best_result = max(trial_results, key=lambda x: x[3])
    return best_result
