"""Regression check for the alternating branch-freeze schedule.

INVESTIGATION.md, Big wins: "Use the branch-freezing hooks that already
exist. freeze_shared() and freeze_private() are never called, so source
and target gradients compete every step." `model/util.py::
alternate_branch_freeze` picks which branch to freeze each epoch;
`model/general.py`'s epoch loops (in `eval()` and `make_objective()`) call
`ClusterModel.freeze_shared`/`freeze_private` with it when
`freeze_schedule="alternate"` is passed in (default `None` leaves both
branches always trainable, i.e. unchanged prior behavior -- neither method
was ever called before this).

`ClusterModel.freeze_shared`/`freeze_private` (`model/general.py`, not
touched by this change) just propagate to `member.model.freeze_shared`/
`freeze_private` for every member -- exercised here directly, against real
method code, via `ClusterModel.__new__` + spy member models, instead of
building a full cluster/pipeline (which needs a real dataset). Needs the
same heavy import-chain stubs as tests/test_scheduled_sampling.py, since
importing `model/general.py` pulls in optuna, torchinfo, tensorboard,
captum, seaborn, mpld3 and several statsmodels submodules, none installed
dev-side and none of their real behavior exercised here.

Run with:

    python tests/test_freeze_schedule.py
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    def stub(name, **attrs):
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
        return sys.modules[name]

    stub("line_profiler", LineProfiler=type("LineProfiler", (), {}))
    optuna = stub("optuna")
    optuna.structs = stub("optuna.structs", TrialPruned=type("TrialPruned", (Exception,), {}))
    optuna.exceptions = stub("optuna.exceptions", TrialPruned=optuna.structs.TrialPruned)
    optuna.samplers = stub("optuna.samplers", TPESampler=type("TPESampler", (), {}))
    optuna.trial = stub("optuna.trial", TrialState=type("TrialState", (), {}))
    optuna.create_study = lambda *a, **k: None
    stub("torchinfo", summary=lambda *a, **k: None)
    stub("torch.utils.tensorboard", SummaryWriter=type("SummaryWriter", (), {}))
    captum = stub("captum")
    captum.attr = stub("captum.attr", Saliency=type("Saliency", (), {}), LayerGradCam=type("LayerGradCam", (), {}))
    stub("seaborn")
    stub("mpld3")
    statsmodels = stub("statsmodels")
    statsmodels.graphics = stub("statsmodels.graphics")
    statsmodels.graphics.tsaplots = stub("statsmodels.graphics.tsaplots", plot_acf=lambda *a, **k: None, plot_pacf=lambda *a, **k: None)
    statsmodels.tsa = stub("statsmodels.tsa")
    statsmodels.tsa.seasonal = stub("statsmodels.tsa.seasonal", seasonal_decompose=lambda *a, **k: None)
    statsmodels.tsa.stattools = stub("statsmodels.tsa.stattools", adfuller=lambda *a, **k: None, acf=lambda *a, **k: None, pacf=lambda *a, **k: None)
    statsmodels.stats = stub("statsmodels.stats")
    statsmodels.stats.diagnostic = stub("statsmodels.stats.diagnostic", kstest_normal=lambda *a, **k: None)
    stub("openpyxl")
    stub("xlrd")
    torch_lr_finder = stub("torch_lr_finder")
    torch_lr_finder.lr_finder = stub("torch_lr_finder.lr_finder", ExponentialLR=type("ExponentialLR", (), {}), LinearLR=type("LinearLR", (), {}))
    sklearn = stub("sklearn")
    sklearn.preprocessing = stub("sklearn.preprocessing", MinMaxScaler=type("MinMaxScaler", (), {}), StandardScaler=type("StandardScaler", (), {}))
    tslearn = stub("tslearn")
    tslearn.utils = stub("tslearn.utils", to_time_series_dataset=lambda x: x)
    tslearn.clustering = stub("tslearn.clustering", TimeSeriesKMeans=type("TimeSeriesKMeans", (), {}), silhouette_score=lambda *a, **k: 0.0)
    tslearn.metrics = stub("tslearn.metrics", dtw=lambda *a, **k: 0.0)


def test_alternate_branch_freeze_period_one_flips_every_epoch():
    from covid_forecasting_joint_learning.model.util import alternate_branch_freeze

    assert [alternate_branch_freeze(e, period=1) for e in range(5)] == [
        "shared", "private", "shared", "private", "shared",
    ]


def test_alternate_branch_freeze_period_n_holds_for_n_epochs():
    from covid_forecasting_joint_learning.model.util import alternate_branch_freeze

    # period=3: epochs 0-2 shared, 3-5 private, 6-8 shared, ...
    assert [alternate_branch_freeze(e, period=3) for e in range(9)] == (
        ["shared"] * 3 + ["private"] * 3 + ["shared"] * 3
    )


def test_alternate_branch_freeze_never_freezes_both_or_neither():
    from covid_forecasting_joint_learning.model.util import alternate_branch_freeze

    for period in (1, 2, 5):
        for epoch in range(20):
            assert alternate_branch_freeze(epoch, period) in ("shared", "private")


def test_alternate_branch_freeze_nonpositive_period_treated_as_one():
    from covid_forecasting_joint_learning.model.util import alternate_branch_freeze

    assert alternate_branch_freeze(0, period=0) == alternate_branch_freeze(0, period=1)
    assert alternate_branch_freeze(1, period=-3) == alternate_branch_freeze(1, period=1)


def test_cluster_model_freeze_propagates_to_every_member():
    from covid_forecasting_joint_learning.model.general import ClusterModel

    calls = []

    class SpyModel:
        def freeze_shared(self, freeze):
            calls.append(("shared", freeze))

        def freeze_private(self, freeze):
            calls.append(("private", freeze))

    # Bypass ClusterModel.__init__ (needs a real cluster/dataset/pipeline);
    # freeze_shared/freeze_private only ever touch self.members, so this
    # exercises the real method bodies without the rest of construction.
    cluster_model = ClusterModel.__new__(ClusterModel)
    cluster_model.sources = [SimpleNamespace(model=SpyModel())]
    cluster_model.targets = [SimpleNamespace(model=SpyModel())]

    cluster_model.freeze_shared(True)
    cluster_model.freeze_private(False)

    assert calls == [("shared", True), ("shared", True), ("private", False), ("private", False)]


def test_alternating_schedule_drives_cluster_model_freeze_correctly():
    # End-to-end (minus the real epoch loop): reproduces exactly what
    # model/general.py's `while not early_stopping.stopped:` blocks do each
    # epoch under freeze_schedule="alternate", against the real
    # ClusterModel.freeze_shared/freeze_private and the real schedule fn.
    from covid_forecasting_joint_learning.model.general import ClusterModel
    from covid_forecasting_joint_learning.model.util import alternate_branch_freeze

    calls = []

    class SpyModel:
        def freeze_shared(self, freeze):
            calls.append(("shared", freeze))

        def freeze_private(self, freeze):
            calls.append(("private", freeze))

    cluster_model = ClusterModel.__new__(ClusterModel)
    cluster_model.sources = []
    cluster_model.targets = [SimpleNamespace(model=SpyModel())]

    for epoch in range(4):
        calls.clear()
        branch = alternate_branch_freeze(epoch, period=1)
        cluster_model.freeze_shared(branch == "shared")
        cluster_model.freeze_private(branch == "private")
        if epoch % 2 == 0:
            assert calls == [("shared", True), ("private", False)]
        else:
            assert calls == [("shared", False), ("private", True)]


if __name__ == "__main__":
    install_stubs()
    test_alternate_branch_freeze_period_one_flips_every_epoch()
    test_alternate_branch_freeze_period_n_holds_for_n_epochs()
    test_alternate_branch_freeze_never_freezes_both_or_neither()
    test_alternate_branch_freeze_nonpositive_period_treated_as_one()
    test_cluster_model_freeze_propagates_to_every_member()
    test_alternating_schedule_drives_cluster_model_freeze_correctly()
    print("ok")
