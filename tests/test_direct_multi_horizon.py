"""Regression check for the direct multi-horizon decoder head.

INVESTIGATION.md, Big wins: "Direct multi-horizon head replacing the
recursive decoder. Now supported by in-domain evidence at 1 to 4 week
horizons, not just by the general strategy literature. Removes exposure
bias and the sequential launch cost together." `model/modules/head.py::
DirectFutureHead` predicts every future step in one batched,
non-autoregressive op from the past encoding + a learned per-horizon
embedding + future_exo, instead of the existing `LILSTMCell2`-looped
recursive decoder (still the default: `SingleModel(..., direct_multi_horizon
=False)`).

Needs the real torch (available in this dev environment); `SingleModel`
also transitively imports optuna, torchinfo, tensorboard, captum, seaborn,
mpld3 and several statsmodels submodules -- stubbed below (same pattern as
tests/test_scheduled_sampling.py), none of their real behavior exercised
by this test. Run with:

    python tests/test_direct_multi_horizon.py
"""
import sys
import types
from pathlib import Path

import torch

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


OUTPUT_SIZE = 3
EXO_SIZE = 2
FUTURE_LENGTH = 4


def make_model(seed=0, direct_multi_horizon=True, with_shared=False):
    from covid_forecasting_joint_learning.model.modules.main import SingleModel
    torch.manual_seed(seed)
    past_model = {"representation_model": None, "private_head": {}, "shared_head": {} if with_shared else None}
    return SingleModel(
        input_size_past=4, hidden_size_past=0,
        input_size_future=OUTPUT_SIZE + EXO_SIZE, hidden_size_future=0,
        private_state_size=6, shared_state_size=5 if with_shared else 0,
        output_size=OUTPUT_SIZE, seed_length=5, future_length=FUTURE_LENGTH,
        past_model=past_model,
        representation_future_model=None,
        private_head_future_cell={}, shared_head_future_cell=({} if with_shared else None),
        post_future_model={}, teacher_forcing=True, use_exo=True, update_hx=True,
        direct_multi_horizon=direct_multi_horizon,
    )


def sample_inputs(batch=2, seed=1):
    torch.manual_seed(seed)
    past = torch.randn(batch, 10, 4)
    past_seed = torch.randn(batch, 5, OUTPUT_SIZE)
    past_exo = torch.randn(batch, 5, EXO_SIZE)
    future = torch.randn(batch, FUTURE_LENGTH, OUTPUT_SIZE)
    future_exo = torch.randn(batch, FUTURE_LENGTH, EXO_SIZE)
    return past, past_seed, past_exo, future, future_exo


def test_direct_head_output_shape_matches_recursive_decoder():
    model = make_model(direct_multi_horizon=True)
    past, past_seed, past_exo, future, future_exo = sample_inputs()
    out = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    assert out.shape == (2, FUTURE_LENGTH, OUTPUT_SIZE)


def test_direct_head_ignores_future_ground_truth_entirely():
    # Unlike the recursive decoder, the direct head never reads `future` --
    # there is no autoregressive feedback loop to teacher-force.
    model = make_model(direct_multi_horizon=True)
    model.train()
    past, past_seed, past_exo, future, future_exo = sample_inputs()

    out_a = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    _, _, _, other_future, _ = sample_inputs(seed=99)
    out_b = model(past, past_seed, past_exo=past_exo, future=other_future, future_exo=future_exo)

    assert torch.equal(out_a, out_b)


def test_direct_head_is_train_eval_invariant():
    # No teacher forcing / self-feedback distinction exists for a
    # non-autoregressive head -- train() vs eval() must not matter.
    model = make_model(direct_multi_horizon=True)
    past, past_seed, past_exo, future, future_exo = sample_inputs()

    model.train()
    out_train = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    model.eval()
    out_eval = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)

    assert torch.equal(out_train, out_eval)


def test_direct_head_responds_to_future_exo():
    model = make_model(direct_multi_horizon=True)
    model.eval()
    past, past_seed, past_exo, future, future_exo = sample_inputs()

    out_a = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    out_b = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo + 5.0)

    assert not torch.equal(out_a, out_b)


def test_direct_head_horizon_steps_are_distinguished():
    # With identical exo across every step, only the learned horizon
    # embedding can tell step 0 apart from step 3 -- if it collapsed, every
    # horizon step would predict the same thing.
    model = make_model(direct_multi_horizon=True)
    model.eval()
    batch = 2
    past = torch.randn(batch, 10, 4)
    past_seed = torch.randn(batch, 5, OUTPUT_SIZE)
    constant_exo = torch.zeros(batch, FUTURE_LENGTH, EXO_SIZE)

    out = model(past, past_seed, future=None, future_exo=constant_exo)
    assert not torch.equal(out[:, 0, :], out[:, 1, :])
    assert not torch.equal(out[:, 0, :], out[:, -1, :])


def test_direct_head_gradient_reaches_past_model():
    model = make_model(direct_multi_horizon=True)
    model.train()
    past, past_seed, past_exo, future, future_exo = sample_inputs()

    out = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    out.sum().backward()

    past_param = next(model.past_model.parameters())
    assert past_param.grad is not None
    assert torch.any(past_param.grad != 0)

    private_head_param = next(model.direct_private_head.parameters())
    assert private_head_param.grad is not None
    assert torch.any(private_head_param.grad != 0)


def test_recursive_decoder_is_unaffected_by_default():
    # direct_multi_horizon=False (the default) must still take the
    # original recursive path -- unchanged output shape, still responds to
    # teacher forcing (unlike the direct head).
    model = make_model(direct_multi_horizon=False)
    model.train()
    past, past_seed, past_exo, future, future_exo = sample_inputs()

    out_a = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    assert out_a.shape == (2, FUTURE_LENGTH, OUTPUT_SIZE)

    _, _, _, other_future, _ = sample_inputs(seed=99)
    out_b = model(past, past_seed, past_exo=past_exo, future=other_future, future_exo=future_exo)
    # Recursive + teacher forcing: a different `future` changes later steps.
    assert not torch.equal(out_a, out_b)


def test_freeze_shared_and_private_toggle_direct_heads_requires_grad():
    model = make_model(direct_multi_horizon=True, with_shared=True)
    assert model.direct_shared_head is not None

    model.freeze_shared(True)
    assert all(not p.requires_grad for p in model.direct_shared_head.parameters())
    assert all(p.requires_grad for p in model.direct_private_head.parameters())

    model.freeze_shared(False)
    model.freeze_private(True)
    assert all(p.requires_grad for p in model.direct_shared_head.parameters())
    assert all(not p.requires_grad for p in model.direct_private_head.parameters())

    model.freeze_private(False)
    assert all(p.requires_grad for p in model.direct_private_head.parameters())


def test_direct_head_with_shared_branch_runs_and_combines():
    model = make_model(direct_multi_horizon=True, with_shared=True)
    model.eval()
    past, past_seed, past_exo, future, future_exo = sample_inputs()
    out = model(past, past_seed, past_exo=past_exo, future=future, future_exo=future_exo)
    assert out.shape == (2, FUTURE_LENGTH, OUTPUT_SIZE)


if __name__ == "__main__":
    install_stubs()
    test_direct_head_output_shape_matches_recursive_decoder()
    test_direct_head_ignores_future_ground_truth_entirely()
    test_direct_head_is_train_eval_invariant()
    test_direct_head_responds_to_future_exo()
    test_direct_head_horizon_steps_are_distinguished()
    test_direct_head_gradient_reaches_past_model()
    test_recursive_decoder_is_unaffected_by_default()
    test_freeze_shared_and_private_toggle_direct_heads_requires_grad()
    test_direct_head_with_shared_branch_runs_and_combines()
    print("ok")
