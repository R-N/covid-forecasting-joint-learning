"""Regression check for fusing PastHead's LSTMCell loop into nn.LSTM.

INVESTIGATION.md, Quick wins -- cost: "Fuse the past encoder into nn.LSTM
... An identity, not a research claim". `PastHead.forward` used to unroll
`LILSTMCell` (an `nn.LSTMCell` wrapper) in a Python loop over the whole past
window; since the whole input sequence is known up front (unlike the
decoder, this loop is not autoregressive), a single `nn.LSTM` call over the
sequence computes the identical recurrence. This test proves that identity
numerically: given the same weights, the fused `nn.LSTM` path and a
reference loop built from `nn.LSTMCell` with those same weights copied in
must produce the same final hidden/cell state.

Needs the real torch (available in this dev environment, unlike most other
heavy deps) -- no other stubs beyond `line_profiler`, which `model/util.py`
imports at module load time regardless of whether this test exercises it.

Run with:

    python tests/test_past_head_fuse.py
"""
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "line_profiler" not in sys.modules:
    line_profiler = types.ModuleType("line_profiler")
    line_profiler.LineProfiler = type("LineProfiler", (), {})
    sys.modules["line_profiler"] = line_profiler

from covid_forecasting_joint_learning.model.modules.head import PastHead


def reference_forward(past_head, x):
    """Replays the pre-fuse LSTMCell loop using the exact same weights as
    `past_head.rnn`, to compare against the fused implementation."""
    cell = nn.LSTMCell(past_head.rnn.input_size, past_head.state_size)
    with torch.no_grad():
        cell.weight_ih.copy_(past_head.rnn.weight_ih_l0)
        cell.weight_hh.copy_(past_head.rnn.weight_hh_l0)
        cell.bias_ih.copy_(past_head.rnn.bias_ih_l0)
        cell.bias_hh.copy_(past_head.rnn.bias_hh_l0)

    batch_size = x.size(1)
    hx = past_head.hx_0.unsqueeze(0).expand(batch_size, -1)
    cx = past_head.cx_0.unsqueeze(0).expand(batch_size, -1)

    past_length = x.size(0)
    past_length = past_length if past_head.use_last_past else past_length - 1
    for i in range(past_length):
        hx, cx = cell(x[i], (hx, cx))
    return hx, cx


def test_fused_past_head_matches_unrolled_lstmcell_loop():
    torch.manual_seed(0)
    input_size, state_size, seq_len, batch = 5, 8, 12, 3
    head = PastHead(input_size, state_size, use_last_past=False)
    x = torch.randn(seq_len, batch, input_size)

    hx, cx = head(x)
    ref_hx, ref_cx = reference_forward(head, x)

    assert torch.allclose(hx, ref_hx, atol=1e-5), (hx - ref_hx).abs().max()
    assert torch.allclose(cx, ref_cx, atol=1e-5), (cx - ref_cx).abs().max()


def test_use_last_past_true_matches_unrolled_loop_over_full_window():
    torch.manual_seed(2)
    head = PastHead(4, 6, use_last_past=True)
    x = torch.randn(10, 2, 4)

    hx, cx = head(x)
    ref_hx, ref_cx = reference_forward(head, x)

    assert torch.allclose(hx, ref_hx, atol=1e-5)
    assert torch.allclose(cx, ref_cx, atol=1e-5)


def test_use_last_past_flag_changes_the_consumed_window():
    torch.manual_seed(1)
    head_false = PastHead(4, 6, use_last_past=False)
    head_true = PastHead(4, 6, use_last_past=True)
    x = torch.randn(10, 2, 4)
    hx_false, _ = head_false(x)
    hx_true, _ = head_true(x)
    # Different weights (separate modules) AND a different number of
    # consumed steps -- outputs must differ.
    assert not torch.allclose(hx_false, hx_true)


def test_degenerate_single_step_without_use_last_past_returns_none():
    head = PastHead(3, 4, use_last_past=False)
    x = torch.randn(1, 2, 3)
    hx, cx = head(x)
    assert hx is None and cx is None


if __name__ == "__main__":
    test_fused_past_head_matches_unrolled_lstmcell_loop()
    test_use_last_past_true_matches_unrolled_loop_over_full_window()
    test_use_last_past_flag_changes_the_consumed_window()
    test_degenerate_single_step_without_use_last_past_returns_none()
    print("ok")
