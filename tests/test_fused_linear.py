"""Regression check for `model/modules/fused.py` (INVESTIGATION.md, Big
wins #1: "Member batching via HFTA rather than by hand ... Cluster members
within a trial are architecturally identical and fuse cleanly").

`hfta` isn't installed dev-side (not in this dev environment's package
set, same situation as sklearn/tslearn/optuna/statsmodels -- pinned in
requirements-experiment.txt, stubbed here). The stub below is not a dumb
mock: it reproduces `hfta.ops.Linear`'s real `forward`/`snatch_parameters`
verbatim (MIT-licensed, from
https://github.com/UofT-EcoSystem/hfta/blob/12cb0bb5031d36b909605981c1e90e96fdfae57a/hfta/ops/linear.py),
so this test exercises the same batched-matmul math as the genuine
library, not an approximation of it.

That reproduction was cross-checked once against the real, pip-installed
`hfta` package out-of-band (not in this repo's tracked dev environment):

    python3 -m venv /tmp/hfta_venv
    /tmp/hfta_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/hfta_venv/bin/pip install "git+https://github.com/UofT-EcoSystem/hfta@12cb0bb5031d36b909605981c1e90e96fdfae57a"
    /tmp/hfta_venv/bin/python -c "
    import torch, torch.nn as nn
    from hfta.ops import Linear as FusedLinearOp
    torch.manual_seed(0)
    B, BATCH, IN, OUT = 4, 5, 6, 3
    layers = [nn.Linear(IN, OUT) for _ in range(B)]
    fused = FusedLinearOp(IN, OUT, B=B)
    for b, layer in enumerate(layers):
        fused.snatch_parameters(layer, b)
    inputs = [torch.randn(BATCH, IN) for _ in range(B)]
    separate = torch.stack([l(x) for l, x in zip(layers, inputs)])
    fused_out = fused(torch.stack(inputs))
    assert torch.allclose(separate, fused_out, atol=1e-5)
    print('max abs diff:', (separate - fused_out).abs().max().item())
    "
    # -> max abs diff: 0.0

Run this file with:

    python3 tests/test_fused_linear.py
"""
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "hfta" in sys.modules:
        return

    class _StubHFTALinear(nn.Module):
        """Verbatim reproduction of hfta.ops.Linear's forward/
        snatch_parameters (MIT license, see module docstring for source)."""

        def __init__(self, in_features, out_features, bias=True, B=1):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.B = B
            self.weight = nn.Parameter(torch.empty((B, in_features, out_features)))
            if bias:
                self.bias = nn.Parameter(torch.empty((B, 1, out_features)))
            else:
                self.register_parameter("bias", None)
            for b in range(B):
                nn.init.kaiming_uniform_(self.weight[b], a=math.sqrt(5), mode="fan_out")
                if self.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight[b])
                    bound = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
                    nn.init.uniform_(self.bias[b], -bound, bound)

        def forward(self, input):
            old_shape = list(input.size())
            input = input.view(old_shape[0], -1, old_shape[-1])
            if self.bias is None:
                res = torch.bmm(input, self.weight)
            else:
                res = torch.baddbmm(self.bias, input, self.weight)
            old_shape[-1] = self.out_features
            return res.view(old_shape)

        def snatch_parameters(self, other, b):
            assert isinstance(other, nn.Linear)
            assert 0 <= b < self.B
            self.weight.data[b] = other.weight.data.transpose(0, 1)
            if self.bias is not None:
                self.bias.data[b] = other.bias.data.unsqueeze(0)

    hfta = types.ModuleType("hfta")
    hfta_ops = types.ModuleType("hfta.ops")
    hfta_ops.Linear = _StubHFTALinear
    hfta.ops = hfta_ops
    sys.modules["hfta"] = hfta
    sys.modules["hfta.ops"] = hfta_ops


def test_hfta_unavailable_raises_import_error_without_the_stub():
    # Reload fused.py fresh, without the hfta stub installed, to prove the
    # graceful-degradation path (no hfta -> ImportError, not a crash on
    # import) actually works -- this is how every dev environment without
    # hfta installed (including this one) behaves today.
    import importlib
    sys.modules.pop("hfta", None)
    sys.modules.pop("hfta.ops", None)
    sys.modules.pop("covid_forecasting_joint_learning.model.modules.fused", None)
    from covid_forecasting_joint_learning.model.modules import fused
    importlib.reload(fused)

    assert fused.HFTA_AVAILABLE is False
    try:
        fused.fused_linear(3, 4, B=2)
        assert False, "expected ImportError"
    except ImportError as e:
        assert "hfta" in str(e)


def test_fused_linear_matches_separate_layers():
    install_stubs()
    import importlib
    from covid_forecasting_joint_learning.model.modules import fused
    importlib.reload(fused)
    assert fused.HFTA_AVAILABLE is True

    torch.manual_seed(0)
    B, batch, in_features, out_features = 4, 5, 6, 3
    layers = [nn.Linear(in_features, out_features) for _ in range(B)]

    fused_op = fused.fused_linear_from_members(layers)
    inputs = [torch.randn(batch, in_features) for _ in range(B)]

    separate = torch.stack([layer(x) for layer, x in zip(layers, inputs)])
    fused_output = fused_op(torch.stack(inputs))

    assert fused_output.shape == separate.shape
    assert torch.allclose(separate, fused_output, atol=1e-5)


def test_fused_linear_from_members_rejects_mismatched_shapes():
    install_stubs()
    import importlib
    from covid_forecasting_joint_learning.model.modules import fused
    importlib.reload(fused)

    layers = [nn.Linear(4, 3), nn.Linear(5, 3)]  # mismatched in_features
    try:
        fused.fused_linear_from_members(layers)
        assert False, "expected AssertionError on shape mismatch"
    except AssertionError:
        pass


def test_fused_linear_preserves_each_members_own_weights():
    # Not just "the math works" -- prove member b's fused slot actually
    # forecasts with member b's own weights, not some averaged/shared one.
    install_stubs()
    import importlib
    from covid_forecasting_joint_learning.model.modules import fused
    importlib.reload(fused)

    torch.manual_seed(1)
    layer_a = nn.Linear(3, 2)
    layer_b = nn.Linear(3, 2)
    with torch.no_grad():
        layer_b.weight.copy_(layer_a.weight + 10.0)  # deliberately very different

    fused_op = fused.fused_linear_from_members([layer_a, layer_b])
    x = torch.randn(2, 3, 3)  # (B=2, batch=3, in_features=3), same input both slots

    out = fused_op(x)
    assert not torch.allclose(out[0], out[1])
    assert torch.allclose(out[0], layer_a(x[0]), atol=1e-5)
    assert torch.allclose(out[1], layer_b(x[1]), atol=1e-5)


def install_model_stubs():
    """Same stub set as tests/test_direct_multi_horizon.py -- SingleModel's
    import chain pulls in optuna/torchinfo/tensorboard/captum/seaborn/
    mpld3/statsmodels submodules, none installed dev-side and none of
    their real behavior exercised by these fusion tests."""
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


def test_fused_residual_fc_matches_separate_when_sizes_differ():
    # input_size != output_size -> ResidualFC.main is a plain
    # nn.Sequential(Linear, activation), no ResidualBlock wrapping.
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.residual import ResidualFC
    from covid_forecasting_joint_learning.model.modules import fused

    torch.manual_seed(2)
    B, batch, in_size, out_size = 3, 4, 5, 2
    members = [ResidualFC(in_size, out_size) for _ in range(B)]
    fused_module = fused.fused_residual_fc_from_members(members)

    inputs = [torch.randn(batch, in_size) for _ in range(B)]
    separate = torch.stack([m(x) for m, x in zip(members, inputs)])
    fused_output = fused_module(torch.stack(inputs))
    assert torch.allclose(separate, fused_output, atol=1e-5)


def test_fused_residual_fc_matches_separate_with_residual_block():
    # input_size == output_size -> ResidualFC.main wraps a ResidualBlock
    # (the w-weighted residual sum), exercising _fused_residual_block_from_members.
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.residual import ResidualFC
    from covid_forecasting_joint_learning.model.modules import fused

    for highway in (False, True):
        torch.manual_seed(3)
        B, batch, size = 3, 4, 5
        members = [ResidualFC(size, size, highway=highway) for _ in range(B)]
        fused_module = fused.fused_residual_fc_from_members(members)

        inputs = [torch.randn(batch, size) for _ in range(B)]
        separate = torch.stack([m(x) for m, x in zip(members, inputs)])
        fused_output = fused_module(torch.stack(inputs))
        assert torch.allclose(separate, fused_output, atol=1e-5), f"highway={highway}"


def test_fused_residual_fc_rejects_deeper_stacks():
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.residual import ResidualFC
    from covid_forecasting_joint_learning.model.modules import fused

    members = [ResidualFC(4, 4, depth=2) for _ in range(2)]
    try:
        fused.fused_residual_fc_from_members(members)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_fused_combine_head_matches_separate_private_only():
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.combine import CombineHead
    from covid_forecasting_joint_learning.model.modules import fused

    torch.manual_seed(4)
    B, batch, private_size, output_size = 3, 4, 6, 3
    members = [CombineHead(private_size, shared_size=0, output_size=output_size) for _ in range(B)]
    fused_module = fused.fused_combine_head_from_members(members)

    inputs = [torch.randn(batch, private_size) for _ in range(B)]
    separate = torch.stack([m(x) for m, x in zip(members, inputs)])
    fused_output = fused_module(torch.stack(inputs))
    assert torch.allclose(separate, fused_output, atol=1e-5)


def test_fused_combine_head_matches_separate_with_shared():
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.combine import CombineHead
    from covid_forecasting_joint_learning.model.modules import fused

    torch.manual_seed(5)
    B, batch, private_size, shared_size, output_size = 3, 4, 6, 5, 3
    members = [CombineHead(private_size, shared_size=shared_size, output_size=output_size) for _ in range(B)]
    fused_module = fused.fused_combine_head_from_members(members)

    private_inputs = [torch.randn(batch, private_size) for _ in range(B)]
    shared_inputs = [torch.randn(batch, shared_size) for _ in range(B)]
    separate = torch.stack([m(p, s) for m, p, s in zip(members, private_inputs, shared_inputs)])
    fused_output = fused_module(torch.stack(private_inputs), torch.stack(shared_inputs))
    assert torch.allclose(separate, fused_output, atol=1e-5)


def test_fused_direct_future_head_matches_separate():
    install_stubs()
    from covid_forecasting_joint_learning.model.modules.head import DirectFutureHead
    from covid_forecasting_joint_learning.model.modules import fused

    torch.manual_seed(6)
    B, batch, hidden_size, state_size, future_length, exo_size = 3, 4, 6, 6, 5, 2
    members = [DirectFutureHead(hidden_size, state_size, future_length, exo_size=exo_size) for _ in range(B)]
    fused_module = fused.fused_direct_future_head_from_members(members)

    hx_inputs = [torch.randn(batch, hidden_size) for _ in range(B)]
    exo_inputs = [torch.randn(batch, future_length, exo_size) for _ in range(B)]
    separate = torch.stack([m(hx, exo) for m, hx, exo in zip(members, hx_inputs, exo_inputs)])
    fused_output = fused_module(torch.stack(hx_inputs), torch.stack(exo_inputs))
    assert torch.allclose(separate, fused_output, atol=1e-5)


def test_fuse_direct_decoder_matches_unfused_members():
    # Full end-to-end proof: B real SingleModel instances (direct_multi_horizon=True),
    # matching general.py:154's guarantee (identical sizes/model_kwargs,
    # independent weights). This mirrors the out-of-band real-hfta
    # reproduction documented in model/modules/fused.py's docstring,
    # verified there against the genuine pip-installed library:
    #   private-only:      max abs diff 1.19e-07
    #   private+shared:    max abs diff 8.94e-08
    install_stubs()
    install_model_stubs()
    from covid_forecasting_joint_learning.model.modules.main import SingleModel
    from covid_forecasting_joint_learning.model.modules import fused

    output_size, exo_size, future_length = 3, 2, 4

    def make_model(seed, with_shared):
        torch.manual_seed(seed)
        past_model = {"representation_model": None, "private_head": {}, "shared_head": {} if with_shared else None}
        return SingleModel(
            input_size_past=4, hidden_size_past=0,
            input_size_future=output_size + exo_size, hidden_size_future=0,
            private_state_size=6, shared_state_size=5 if with_shared else 0,
            output_size=output_size, seed_length=5, future_length=future_length,
            past_model=past_model,
            representation_future_model=None,
            private_head_future_cell={}, shared_head_future_cell=({} if with_shared else None),
            post_future_model={}, teacher_forcing=True, use_exo=True, update_hx=True,
            direct_multi_horizon=True,
        )

    for with_shared in (False, True):
        B, batch = 3, 5
        models = [make_model(seed=100 + i, with_shared=with_shared) for i in range(B)]
        for m in models:
            m.eval()

        pasts = [torch.randn(batch, 10, 4) for _ in range(B)]
        past_seeds = [torch.randn(batch, 5, output_size) for _ in range(B)]
        past_exos = [torch.randn(batch, 5, exo_size) for _ in range(B)]
        future_exos = [torch.randn(batch, future_length, exo_size) for _ in range(B)]

        unfused_outputs = torch.stack([
            m(pasts[i], past_seeds[i], past_exo=past_exos[i], future=None, future_exo=future_exos[i])
            for i, m in enumerate(models)
        ])

        fused_private_head, fused_shared_head, fused_post_future_model = fused.fuse_direct_decoder_from_members(models)

        hx_privates, hx_shareds = [], []
        for i, m in enumerate(models):
            if m.past_model.use_shared_head:
                hp, hs = m.past_model(pasts[i])
                hx_shareds.append(hs)
            else:
                hp = m.past_model(pasts[i])
            hx_privates.append(hp)

        future_exo_fused = torch.stack(future_exos)
        cx_private = fused_private_head(torch.stack(hx_privates), future_exo_fused)
        cx_shared = fused_shared_head(torch.stack(hx_shareds), future_exo_fused) if with_shared else None
        fused_output = fused_post_future_model(cx_private, cx_shared)

        assert torch.allclose(unfused_outputs, fused_output, atol=1e-4), f"with_shared={with_shared}"


if __name__ == "__main__":
    install_model_stubs()  # needed by residual/combine/head.py's ..util -> line_profiler chain, ahead of every test below
    test_hfta_unavailable_raises_import_error_without_the_stub()
    test_fused_linear_matches_separate_layers()
    test_fused_linear_from_members_rejects_mismatched_shapes()
    test_fused_linear_preserves_each_members_own_weights()
    test_fused_residual_fc_matches_separate_when_sizes_differ()
    test_fused_residual_fc_matches_separate_with_residual_block()
    test_fused_residual_fc_rejects_deeper_stacks()
    test_fused_combine_head_matches_separate_private_only()
    test_fused_combine_head_matches_separate_with_shared()
    test_fused_direct_future_head_matches_separate()
    test_fuse_direct_decoder_matches_unfused_members()
    print("ok")
