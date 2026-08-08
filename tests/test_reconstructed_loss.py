"""Regression check for the reconstructed-IRD-count training loss.

INVESTIGATION.md, Big wins: "Loss on reconstructed IRD counts rather than
scaled rates. Aligns optimisation with evaluation ... Decide the error
distribution at the same time." `model/torch_sird.py` ports
`pipeline/sird.py::rebuild`'s per-step recurrence to differentiable torch
ops; `model/loss.py::ReconstructedRMSSELoss` uses it to backprop RMSSE
computed on reconstructed IRD counts, instead of RMSSE/MSSE computed on the
model's raw scaled-rate output (`RMSSELoss`/`MSSELoss`); `model/train.py`'s
per-batch loop opts into the extended call signature via a `reconstructed`
flag on the loss object, leaving the default (`RMSSELoss`/`MSSELoss`) path
byte-for-byte unchanged.

Run with:

    python tests/test_reconstructed_loss.py
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covid_forecasting_joint_learning.model import torch_sird
from covid_forecasting_joint_learning.pipeline import sird as np_sird


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


def make_scaler(scale, min_):
    return SimpleNamespace(scale_=np.array(scale, dtype=np.float64), min_=np.array(min_, dtype=np.float64))


def test_torch_rebuild_matches_numpy_reference():
    torch.manual_seed(0)
    np.random.seed(0)
    batch, steps = 3, 5
    n = 10_000.0

    # Deliberately include out-of-range rates so every clamp in
    # pipeline.sird.rebuild actually fires: negative rates, gamma+delta > 1,
    # and an S-inflow that would exceed S itself.
    sird_vars = np.random.uniform(-0.2, 1.5, size=(batch, steps, 3))
    prev = np.stack([
        np.array([n - 50.0, 50.0, 0.0, 0.0]),
        np.array([n - 5.0, 5.0, 3.0, 1.0]),
        np.array([n - 200.0, 200.0, 10.0, 2.0]),
    ])

    expected = np.stack([
        np.array(np_sird.rebuild(sird_vars[b], tuple(prev[b]), n))
        for b in range(batch)
    ])

    actual = torch_sird.rebuild(
        torch.tensor(sird_vars, dtype=torch.float64),
        torch.tensor(prev, dtype=torch.float64),
        n,
    ).numpy()

    assert np.allclose(actual, expected, atol=1e-8), (actual - expected)


def test_inverse_scale_matches_minmax_round_trip():
    scaler = make_scaler(scale=[2.0, 4.0, 0.5], min_=[-1.0, 0.5, 3.0])
    x_physical = torch.tensor([[0.1, 0.2, 100.0], [0.05, 0.4, 50.0]], dtype=torch.float64)

    # sklearn MinMaxScaler.transform is x * scale_ + min_; verify our
    # inverse_scale really undoes that (not just matches an ad hoc formula).
    x_scaled = x_physical * torch.as_tensor(scaler.scale_) + torch.as_tensor(scaler.min_)
    recovered = torch_sird.inverse_scale(x_scaled, scaler)

    assert torch.allclose(recovered, x_physical, atol=1e-10)


def test_reconstructed_loss_matches_independent_numpy_pipeline_and_backprops():
    from covid_forecasting_joint_learning.model.loss import ReconstructedRMSSELoss
    from covid_forecasting_joint_learning.model import loss_common

    torch.manual_seed(1)
    batch, steps, seed_len = 2, 4, 6
    n = 5_000.0
    scaler = make_scaler(scale=[3.0, 2.0, 1.5], min_=[0.0, 0.1, 0.05])

    pred_vars_scaled = torch.rand(batch, steps, 3, dtype=torch.float64, requires_grad=True)
    prev = torch.tensor(np.stack([
        np.stack([np.linspace(n - 100, n - 90, seed_len),
                  np.linspace(100, 90, seed_len),
                  np.linspace(5, 8, seed_len),
                  np.linspace(1, 2, seed_len)], axis=-1)
        for _ in range(batch)
    ]), dtype=torch.float64)
    future_final = torch.rand(batch, steps, 3, dtype=torch.float64) * 50 + 50

    loss_fn = ReconstructedRMSSELoss(reduction="sum")
    assert loss_fn.reconstructed is True
    loss = loss_fn(pred_vars_scaled, prev, future_final, n, scaler)

    loss.backward()
    assert pred_vars_scaled.grad is not None
    assert torch.any(pred_vars_scaled.grad != 0)

    # Independent cross-check: reconstruct via the original numpy pipeline
    # (pipeline.sird.rebuild) and score with the original numpy rmsse
    # (loss_common.rmsse), completely bypassing torch_sird/loss.py.
    pred_vars_np = pred_vars_scaled.detach().numpy()
    pred_vars_physical = (pred_vars_np - scaler.min_) / scaler.scale_
    prev_np = prev.numpy()
    expected_final = np.stack([
        np.array(np_sird.rebuild(pred_vars_physical[b], tuple(prev_np[b][-1]), n))
        for b in range(batch)
    ])
    expected_losses = np.stack([
        loss_common.rmsse(prev_np[b][:, 1:], future_final.numpy()[b], expected_final[b])
        for b in range(batch)
    ])
    expected_loss = np.sum(np.sum(expected_losses, axis=-1), axis=0)

    assert np.isclose(loss.item(), expected_loss, atol=1e-6), (loss.item(), expected_loss)


def test_eval_dispatches_reconstructed_loss_with_extended_signature():
    install_stubs()
    from covid_forecasting_joint_learning.model import train

    pred = torch.randn(2, 4, 3, requires_grad=True)
    past = torch.randn(2, 10, 4)
    past_seed = torch.randn(2, 5, 3)
    future = torch.randn(2, 4, 3)
    final_seed = torch.randn(2, 5, 4)
    future_final = torch.randn(2, 4, 3)
    scaler = make_scaler(scale=[1.0, 1.0, 1.0], min_=[0.0, 0.0, 0.0])

    kabko = SimpleNamespace(
        model=lambda *args: pred,
        weight=1.0,
        is_target=1,
        population=1000.0,
        scaler_2=scaler,
    )
    sample = (past, past_seed, None, future, None, final_seed, future_final, kabko)

    class SpyLoss:
        reconstructed = True

        def __init__(self):
            self.calls = []

        def __call__(self, pred_arg, prev_arg, future_final_arg, n_arg, scaler_arg):
            self.calls.append((pred_arg, prev_arg, future_final_arg, n_arg, scaler_arg))
            return torch.zeros(())

    spy = SpyLoss()
    train.__eval([sample], spy, weights=1.0, target_weights=1.0, train=False)

    assert len(spy.calls) == 1
    called_pred, called_prev, called_future_final, called_n, called_scaler = spy.calls[0]
    assert called_pred is pred
    assert called_prev is final_seed
    assert called_future_final is future_final
    assert called_n == 1000.0
    assert called_scaler is scaler

    class SpyPlainLoss:
        def __init__(self):
            self.calls = []

        def __call__(self, past_arg, future_arg, pred_arg):
            self.calls.append((past_arg, future_arg, pred_arg))
            return torch.zeros(())

    plain_spy = SpyPlainLoss()
    train.__eval([sample], plain_spy, weights=1.0, target_weights=1.0, train=False)

    assert len(plain_spy.calls) == 1
    called_past, called_future, called_pred2 = plain_spy.calls[0]
    assert called_past is past_seed
    assert called_future is future
    assert called_pred2 is pred


if __name__ == "__main__":
    install_stubs()
    test_torch_rebuild_matches_numpy_reference()
    test_inverse_scale_matches_minmax_round_trip()
    test_reconstructed_loss_matches_independent_numpy_pipeline_and_backprops()
    test_eval_dispatches_reconstructed_loss_with_extended_signature()
    print("ok")
