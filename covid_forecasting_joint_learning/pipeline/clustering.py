from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import dtw
# import itertools
from collections import Counter
import numpy as np
from . import util as PipelineUtil
import itertools
from ..data import cols as DataCol

def shortest(x):
    return (-len(x.data), x.data.last_valid_index(), x.data.first_valid_index())


class Cluster:
    def __init__(
        self,
        id,
        group,
        members=None,
        target=None,
        targets=[]
    ):
        self.id = id
        self.group = group
        self.members = members
        self.__target = None
        self.__targets = []

        if target and targets:
            self.__target = target
            self.__targets = targets
        else:
            self.target = target
            self.targets = targets


    def remove_kabkos(self, kabkos):
        if not kabkos:
            return
        if not isinstance(kabkos[0], str):
            kabkos = [k.name for k in kabkos]
        self.remove_targets(kabkos)
        self.members = [k for k in self.members if k.name not in kabkos]

    def remove_targets(self, targets):
        if not targets:
            return
        if not isinstance(targets[0], str):
            targets = [k.name for k in targets]
        # self.sources += [k for k in self.targets if k.name in targets]
        self.targets = [k for k in self.targets if k.name not in targets]
        self.select_target()

    def set_targets(self, targets):
        if not targets:
            return
        if not isinstance(targets[0], str):
            targets = [k.name for k in targets]
        self.append_targets(targets)
        self.remove_targets([t for t in self.targets if t.name not in targets])

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, value):
        self.__target = value
        if value:
            if (not self.__targets) and value:
                self.__targets = [value]
            elif value not in self.__targets:
                self.__targets.append(value)

    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, value):
        assert value is None or isinstance(value, list)
        self.__targets = value
        if value:
            self.select_target()

    def select_target(self):
        if len(self.__targets) > 0:
            self.__target = max(self.__targets, key=lambda x: shortest(x))
        else:
            self.target = max(self.sources, key=lambda x: shortest(x))
        # self.sources = [k for k in self.sources if k not in self.targets]

    def append_target(self, value):
        if not value:
            return
        if not isinstance(value, str):
            value = value.name
        value = [k for k in self.sources if k.name == value][0]
        self.__targets.append(value)
        self.select_target()

    def append_targets(self, values):
        if not values:
            return
        if not isinstance(values[0], str):
            values = [k.name for k in values]
        values = [k for k in self.sources if k.name in values]
        self.__targets.extend(values)
        self.select_target()

    @property
    def sources(self):
        return [k for k in self.members if k not in self.targets]

    @property
    def source_longest(self):
        return min(self.sources, key=lambda x: shortest(x))

    @property
    def source_closest(self):
        return min(self.sources, key=lambda x: dtw(self.target, x))

    def copy(self, group=None, copy_dict=None):
        cluster = Cluster(
            id=self.id,
            group=group or self.group
        )
        if not copy_dict:
            copy_dict = {k: k.copy(cluster=cluster) for k in self.members}
        cluster.members = [copy_dict[k] for k in self.members]
        cluster.targets = [copy_dict[k] for k in self.targets]
        for k in cluster.members:
            k.cluster = cluster
        cluster.select_target()
        return cluster

    @property
    def date_cols(self):
        date_set = set()
        for k in self.members:
            date_set = date_set.union([d for d in DataCol.DATES if sum(k.data[d]) > 0])
        return [d for d in DataCol.DATES if d in date_set]

    @property
    def inverse_date_cols(self):
        date_set = self.date_cols
        return [d for d in DataCol.DATES if d not in date_set]


def merge_clusters(group):
    return Cluster(
        -1,
        group,
        members=group.members,
        targets=group.targets
    )



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


def log_results(trial_results, title):
    print(title)
    for r in trial_results:
        print(f"n: {r.n_clusters}, silhouette: {r.silhouette}, non_single: {r.n_clusters_non_single}, single: {r.n_clusters_single}")


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
    min_silhouette_percentile=75,
    max_silhouette_diff=0.1,
    verbose=True,
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
    )) for n in range(n_clusters_min, n_clusters_max + 1)]

    trial_results = [ClusteringResult(n, model, labels, silhouette_score(
        dataset,
        labels,
        metric=metric
    )) for n, model, labels in trial_labels]

    trial_results = sorted(
        trial_results,
        key=lambda r: r.silhouette,
        reverse=True
    )

    if verbose:
        log_results(trial_results, "All silhouette")

    if len(trial_results) > 1:

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
        if good_clustering and len(trial_results_1) != len(trial_results):
            trial_results = trial_results_1
            if verbose:
                log_results(trial_results, "At least 2 non-single clusters")
        # Else I'll just return best one to remove outliers

        # Then I will filter it to just the upper quartile of silhouette
        silhouettes = [r.silhouette for r in trial_results]
        min_silhouette = np.percentile(silhouettes, min_silhouette_percentile)
        trial_results = [r for r in trial_results if r.silhouette >= min_silhouette]
        if verbose:
            log_results(trial_results, "Upper quartile")

        # It should also not be that far off the best silhouette
        best_silhouette = max(silhouettes)
        trial_results = [r for r in trial_results if best_silhouette-r.silhouette <= max_silhouette_diff]

        # Try to not have single clusters more than non-single
        trial_results_2 = [r for r in trial_results if r.n_clusters_non_single >= r.n_clusters_single]
        good_clustering_2 = len(trial_results_2) > 0
        if good_clustering and len(trial_results_2) != len(trial_results):
            trial_results = trial_results_2
            if verbose:
                log_results(trial_results, "Single cluster count not higher than non-single")

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
        best_result.best_silhouette = best_silhouette
        best_result.good_clustering = good_clustering
        best_result.good_clustering_2 = good_clustering_2
    else:
        # Provided for single n_clusters
        best_result = trial_results[0]
        best_result.best_silhouette = best_result.silhouette
        best_result.good_clustering = best_result.n_clusters_non_single >= good_clustering_non_single
        best_result.good_clustering_2 = best_result.n_clusters_non_single >= best_result.n_clusters_single

    if verbose:
        log_results([best_result], "Best n cluster")

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
