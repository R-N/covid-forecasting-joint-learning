"""Regression check for scheduled sampling / horizon curriculum.

INVESTIGATION.md, Quick wins: "Teacher forcing is all-or-nothing per trial
... Scheduled sampling, decaying the forcing probability across epochs,
targets the exposure bias expected at a 14-step horizon." `SingleModel`
gained an opt-in `teacher_forcing_ratio` override (`None` preserves the
exact previous all-or-nothing `teacher_forcing` bool behavior) and a
per-step Bernoulli draw decides whether each future step sees the ground
truth or the model's own prediction. `model/util.py::
teacher_forcing_ratio_schedule` linearly decays that ratio across epochs
for callers that opt in.

Needs the real torch (available in this dev environment). `SingleModel`
also transitively imports optuna, torchinfo, tensorboard, captum, seaborn,
mpld3 and several statsmodels submodules -- none installed here and none
of their real behavior is exercised by this test (this repo's model
forward pass has no existing test, so this is the first one to walk that
import chain; every stub below is a bare attribute placeholder). Run with:

    python tests/test_scheduled_sampling.py
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


def make_model(seed=0, teacher_forcing=True):
    from covid_forecasting_joint_learning.model.modules.main import SingleModel
    torch.manual_seed(seed)
    return SingleModel(
        input_size_past=4, hidden_size_past=0,
        input_size_future=3, hidden_size_future=0,
        private_state_size=6, shared_state_size=0,
        output_size=3, seed_length=5, future_length=4,
        past_model={"representation_model": None, "private_head": {}, "shared_head": None},
        representation_future_model=None,
        private_head_future_cell={}, shared_head_future_cell=None,
        post_future_model={}, teacher_forcing=teacher_forcing, use_exo=False, update_hx=True,
    )


def sample_inputs(batch=2, seed=1):
    torch.manual_seed(seed)
    past = torch.randn(batch, 10, 4)
    past_seed = torch.randn(batch, 5, 3)
    future = torch.randn(batch, 4, 3)
    return past, past_seed, future


def test_teacher_forcing_ratio_schedule_linear_decay():
    from covid_forecasting_joint_learning.model.util import teacher_forcing_ratio_schedule

    assert teacher_forcing_ratio_schedule(0, 10) == 1.0
    assert teacher_forcing_ratio_schedule(10, 10) == 0.0
    assert abs(teacher_forcing_ratio_schedule(5, 10) - 0.5) < 1e-9
    # Past the decay window, clamps at `end` rather than overshooting negative.
    assert teacher_forcing_ratio_schedule(100, 10) == 0.0
    # decay_epochs <= 0 -> immediately at `end`.
    assert teacher_forcing_ratio_schedule(0, 0) == 0.0
    # Custom start/end bounds.
    assert abs(teacher_forcing_ratio_schedule(5, 10, start=0.8, end=0.2) - 0.5) < 1e-9


def test_set_teacher_forcing_ratio_mutates_state():
    model = make_model()
    assert model.teacher_forcing_ratio is None
    model.set_teacher_forcing_ratio(0.3)
    assert model.teacher_forcing_ratio == 0.3


def test_ratio_none_with_teacher_forcing_true_matches_ratio_one():
    model = make_model(teacher_forcing=True)
    model.train()
    past, past_seed, future = sample_inputs()

    torch.manual_seed(42)
    out_default = model(past, past_seed, future=future)

    model.set_teacher_forcing_ratio(1.0)
    torch.manual_seed(42)
    out_ratio_one = model(past, past_seed, future=future)

    assert torch.equal(out_default, out_ratio_one)


def test_ratio_zero_matches_teacher_forcing_false_and_ignores_future():
    model_a = make_model(teacher_forcing=True)
    model_b = make_model(teacher_forcing=False)
    model_b.load_state_dict(model_a.state_dict())
    model_a.train()
    model_b.train()

    past, past_seed, future = sample_inputs()
    model_a.set_teacher_forcing_ratio(0.0)

    torch.manual_seed(7)
    out_a = model_a(past, past_seed, future=future)
    torch.manual_seed(7)
    out_b = model_b(past, past_seed, future=None)

    assert torch.equal(out_a, out_b)

    # Ground truth is never read at ratio 0 -- an entirely different
    # `future` tensor must not change the output at all.
    _, _, other_future = sample_inputs(seed=99)
    torch.manual_seed(7)
    out_a_other_future = model_a(past, past_seed, future=other_future)
    assert torch.equal(out_a, out_a_other_future)


def test_intermediate_ratio_is_reproducible_and_between_the_extremes():
    model = make_model(teacher_forcing=True)
    model.train()
    past, past_seed, future = sample_inputs()

    # torch.manual_seed(0) draws [True, False, True, True] < 0.5 across the
    # 4 future steps -- a genuine mix, not all-forced or all-autoregressive.
    model.set_teacher_forcing_ratio(0.5)
    torch.manual_seed(0)
    out_1 = model(past, past_seed, future=future)
    torch.manual_seed(0)
    out_2 = model(past, past_seed, future=future)
    assert torch.equal(out_1, out_2)  # same seed -> same Bernoulli draws -> reproducible

    model.set_teacher_forcing_ratio(1.0)
    torch.manual_seed(0)
    out_full = model(past, past_seed, future=future)
    model.set_teacher_forcing_ratio(0.0)
    torch.manual_seed(0)
    out_none = model(past, past_seed, future=future)

    assert not torch.equal(out_1, out_full)
    assert not torch.equal(out_1, out_none)


def test_eval_mode_ignores_teacher_forcing_ratio():
    model = make_model(teacher_forcing=True)
    model.eval()
    past, past_seed, future = sample_inputs()

    torch.manual_seed(11)
    out_default = model(past, past_seed, future=future)

    for ratio in (0.0, 0.5, 1.0):
        model.set_teacher_forcing_ratio(ratio)
        torch.manual_seed(11)
        out = model(past, past_seed, future=future)
        assert torch.equal(out, out_default), f"eval-mode output changed at ratio={ratio}"


if __name__ == "__main__":
    install_stubs()
    test_teacher_forcing_ratio_schedule_linear_decay()
    test_set_teacher_forcing_ratio_mutates_state()
    test_ratio_none_with_teacher_forcing_true_matches_ratio_one()
    test_ratio_zero_matches_teacher_forcing_false_and_ignores_future()
    test_intermediate_ratio_is_reproducible_and_between_the_extremes()
    test_eval_mode_ignores_teacher_forcing_ratio()
    print("ok")
