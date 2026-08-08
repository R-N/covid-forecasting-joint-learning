"""Regression check: `ClusterModel`'s shared-branch wiring covers
`DirectFutureHead` too.

Found while wiring up the HFTA fusion end-to-end proof (`model/modules/
fused.py`): `ClusterModel.__init__` (`model/general.py`) shares submodules
across cluster members by assigning the *same object* into `model_kwargs`
before constructing each member's `SingleModel` (`shared_head_future_cell`,
`private_head_future_cell`, `post_future_model`, etc.) -- but this only
covered the recurrent-decoder path. `SingleModel(..., direct_multi_horizon
=True)`'s `direct_shared_head`/`direct_private_head` weren't in that list
at all, so every member built its own *independent* direct-decoder shared
branch instead of actually sharing it -- silently defeating joint
learning for direct-decoder models. Fixed by:

- `SingleModel.__init__` gained `direct_private_head=None`/
  `direct_shared_head=None` params (mirroring `shared_head_future_cell`'s
  pattern: `None` auto-constructs fresh from `direct_future_head`, a
  pre-built module is used as-is).
- `ClusterModel.__init__`'s `shared_mode`/`private_mode` `SHARED` blocks
  now also copy `self.shared_model.direct_shared_head`/
  `direct_private_head` into `model_kwargs` (a no-op, via the existing
  `suppress(...)` guards, for recursive-decoder models where these
  attributes are always `None`).

Needs the same heavy import-chain stubs as tests/test_freeze_schedule.py,
since importing model/general.py pulls in optuna, torchinfo, tensorboard,
captum, seaborn, mpld3, tslearn, sklearn and several statsmodels
submodules, none installed dev-side and none of their real behavior
exercised here.

Run with:

    python tests/test_direct_head_sharing.py
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    import importlib.machinery

    def stub(name, **attrs):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
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


def make_cluster_model(direct_multi_horizon, with_shared, n_members=3):
    from covid_forecasting_joint_learning.model.general import ClusterModel, SharedMode, SourcePick

    output_size, exo_size, future_length = 3, 0, 4
    sizes = {
        "input_size_past": 4,
        "hidden_size_past": 0,
        "input_size_future": output_size,
        "hidden_size_future": 0,
        "private_state_size": 6,
        "shared_state_size": 5 if with_shared else 0,
        "output_size": output_size,
        "seed_length": 5,
        "future_length": future_length,
    }
    model_kwargs = {
        "past_model": {"representation_model": None, "private_head": {}, "shared_head": {} if with_shared else None},
        "representation_future_model": None,
        "private_head_future_cell": {},
        "shared_head_future_cell": {} if with_shared else None,
        "post_future_model": {},
        "teacher_forcing": True,
        "use_exo": False,
        "update_hx": True,
        "direct_multi_horizon": direct_multi_horizon,
    }

    members = [SimpleNamespace(name=f"m{i}", weight=1.0) for i in range(n_members)]
    cluster = SimpleNamespace(
        sources=members,
        target=members[-1],
        targets=[members[-1]],
    )

    torch.manual_seed(0)
    return ClusterModel(
        cluster,
        sizes,
        model_kwargs=model_kwargs,
        source_pick=SourcePick.ALL,
        private_mode=SharedMode.SHARED,
        shared_mode=SharedMode.SHARED,
        lr=1e-4,
    )


def test_direct_shared_head_is_the_same_object_across_members():
    install_stubs()
    cluster_model = make_cluster_model(direct_multi_horizon=True, with_shared=True)

    heads = [m.direct_shared_head for m in cluster_model.models]
    assert all(h is not None for h in heads)
    assert all(h is heads[0] for h in heads[1:]), "direct_shared_head should be one shared object, not per-member copies"


def test_direct_private_head_is_the_same_object_across_members():
    install_stubs()
    cluster_model = make_cluster_model(direct_multi_horizon=True, with_shared=True)

    heads = [m.direct_private_head for m in cluster_model.models]
    assert all(h is not None for h in heads)
    assert all(h is heads[0] for h in heads[1:]), "direct_private_head should be one shared object, not per-member copies"


def test_recursive_decoder_models_unaffected_direct_heads_stay_none():
    install_stubs()
    cluster_model = make_cluster_model(direct_multi_horizon=False, with_shared=True)

    for m in cluster_model.models:
        assert m.direct_private_head is None
        assert m.direct_shared_head is None


def test_shared_gradient_actually_reaches_every_member():
    # Not just "same object" -- prove a gradient computed through one
    # member's forward pass updates the shared head's grad, visible from
    # every other member's reference to it too.
    install_stubs()
    cluster_model = make_cluster_model(direct_multi_horizon=True, with_shared=True)
    for m in cluster_model.models:
        m.train()

    member = cluster_model.models[0]
    batch = 2
    past = torch.randn(batch, 10, 4)
    past_seed = torch.zeros(batch, 5, 3)
    future_exo = None
    out = member(past, past_seed, future=None, future_exo=future_exo)
    out.sum().backward()

    shared_head = cluster_model.models[1].direct_shared_head
    assert shared_head is member.direct_shared_head
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in shared_head.parameters())


if __name__ == "__main__":
    test_direct_shared_head_is_the_same_object_across_members()
    test_direct_private_head_is_the_same_object_across_members()
    test_recursive_decoder_models_unaffected_direct_heads_stay_none()
    test_shared_gradient_actually_reaches_every_member()
    print("ok")
