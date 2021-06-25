from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import dtw
# import itertools
from collections import Counter
import numpy as np
from . import util as PipelineUtil


class Cluster:
    def __init__(
        self,
        id,
        sources=None,
        target=None
    ):
        self.id = id
        self.sources = sources
        self.target = target

    @property
    def members(self):
        return [*self.sources, self.target]

    @property
    def source_longest(self):
        return max(self.sources, key=lambda x: len(x))

    @property
    def source_closest(self):
        return min(self.sources, key=lambda x: dtw(self.target, x))


def cluster(
    dataset,
    n_clusters,
    n_init=3,
    max_iter=50,
    max_iter_barycenter=100,
    metric="dtw",
    random_state=None,
    **kwargs
):
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        max_iter_barycenter=max_iter_barycenter,
        metric=metric,
        random_state=random_state,
        **kwargs
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
        self.good_clustering = None
        self.good_clustering_2 = None
        self.best_silhouette = None

    def get_info(self):
        return ClusteringInfo(
            n_clusters=self.n_clusters,
            silhouette=self.silhouette,
            n_clusters_single=self.n_clusters_single,
            n_clusters_non_single=self.n_clusters_non_single,
            single_clusters=self.single_clusters,
            good_clustering=self.good_clustering,
            good_clustering_2=self.good_clustering_2,
            best_silhouette=self.best_silhouette
        )


class ClusteringInfo:
    def __init__(
        self,
        n_clusters,
        silhouette,
        n_clusters_single,
        n_clusters_non_single,
        single_clusters,
        good_clustering,
        good_clustering_2,
        best_silhouette=None
    ):
        self.n_clusters = n_clusters
        self.silhouette = silhouette
        self.n_clusters_single = n_clusters_single
        self.n_clusters_non_single = n_clusters_non_single
        self.single_clusters = single_clusters
        self.good_clustering = good_clustering
        self.good_clustering_2 = good_clustering_2
        self.best_silhouette = best_silhouette


def cluster_best(
    dataset,
    n_clusters_min=2,
    n_clusters_max=10,
    n_init=3,
    max_iter=50,
    max_iter_barycenter=100,
    metric="dtw",
    random_state=None,
    good_clustering_non_single=2,
    min_silhouette_percentile=0.75,
    max_silhouette_diff=0.25,
    **kwargs
):
    trial_labels = [(n, *cluster(
        dataset,
        n,
        n_init=n_init,
        max_iter=max_iter,
        max_iter_barycenter=max_iter_barycenter,
        metric=metric,
        random_state=random_state,
        **kwargs
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
    trial_results_1 = [r for r in trial_results if r.n_clusters_non_single >= good_clustering_non_single]
    # I must first check that it exists before using it
    good_clustering = len(trial_results_1) > 0
    if good_clustering:
        trial_results = trial_results_1
    # Else I'll just return best one to remove outliers

    # Then I will filter it to just the upper quartile of silhouette
    silhouettes = [r.silhouette for r in trial_results]
    min_silhouette = np.percentile(silhouettes, min_silhouette_percentile)
    trial_results = [r for r in trial_results if r.silhouette >= min_silhouette]

    # It should also not be that far off the best silhouette
    best_silhouette = max(silhouettes)
    trial_results = [r for r in trial_results if best_silhouette-r.silhouette <= max_silhouette_diff]

    # Try to not have single clusters more than non-single
    trial_results_2 = [r for r in trial_results if r.n_clusters_non_single >= r.n_clusters_single]
    good_clustering_2 = len(trial_results_2) > 0
    if good_clustering:
        trial_results = trial_results_2

    # if good_clustering:
    #     # Then I'll pick the one with most non-single clusters, 
    #     # and the one with best silhouette if there are more than one
    #     best_result = max(trial_results, key=lambda r: (r.n_clusters_non_single, r.silhouette))
    # else:
    #     # But if it's useless clustering, then I will pick one with least single clusters
    #     # So that I won't remove too much
    # Nevermind. I shouldn't have maximized cluster number. 
    # There's no good reason for that
    best_result = min(trial_results, key=lambda r: (r.n_clusters_single, -r.silhouette))
    best_result.good_clustering = good_clustering
    best_result.good_clustering_2 = good_clustering_2
    best_result.best_silhouette = best_silhouette
    return best_result


def predict(model, data):
    return model.predict(to_time_series_dataset([data]))[0]


def one_sided_clustering_similarity(a, b):
    similars = [PipelineUtil.find_similar_set(ca, b) for ca in a]
    weighted_similarities = [len(s[0])*s[1] for s in similars]
    n = sum([len(ca) for ca in a])
    avg_similarity = sum(weighted_similarities) / n
    return avg_similarity


def pairwise_clustering_similarity(a, b):
    return 0.5 * (one_sided_clustering_similarity(a, b) + one_sided_clustering_similarity(b, a))


def check_cluster_data_indices(
    kabko,
    target_last_index,
    target_first_split_index
):
    try:
        assert kabko.data.last_valid_index() == target_last_index
    except Exception:
        raise Exception("Inequal %s.%s.%s data end %s != %s" % (
            kabko.group.id,
            kabko.cluster.id,
            kabko.name,
            kabko.data.last_valid_index(),
            target_last_index
        ))
    try:
        assert kabko.data.first_valid_index() <= target_first_split_index
    except Exception:
        raise Exception("Late %s.%s.%s data start %s > %s" % (
            kabko.group.id,
            kabko.cluster.id,
            kabko.name,
            kabko.data.first_valid_index(),
            target_first_split_index
        ))
