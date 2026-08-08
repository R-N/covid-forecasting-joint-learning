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


if __name__ == "__main__":
    test_hfta_unavailable_raises_import_error_without_the_stub()
    test_fused_linear_matches_separate_layers()
    test_fused_linear_from_members_rejects_mismatched_shapes()
    test_fused_linear_preserves_each_members_own_weights()
    print("ok")
