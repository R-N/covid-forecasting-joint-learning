"""Differentiable torch port of `pipeline/sird.py::rebuild`.

`SingleModel.rebuild` (`model/modules/main.py`) detaches predictions to numpy
before reconstructing IRD counts, so the training loss (`MSSELoss`/
`RMSSELoss` in `model/loss.py`) only ever sees the scaled SIRD *rate*
predictions, never the reconstructed counts that `pipeline/train.py::test`
(and hence `ClusterModel.metric`, the value Optuna/EarlyStopping actually
watch) scores against. This module mirrors the same per-step recurrence in
pure torch ops so it can sit inside the autograd graph:
`model/loss.py::ReconstructedRMSSELoss` uses it to backprop through the
reconstruction into the rate prediction, instead of training on rate error
and only *measuring* count error.

The clamps below are the same three guards `pipeline/sird.py::rebuild`
applies, translated from Python `if` branches to elementwise tensor ops
(`torch.clamp`/`torch.minimum`) since batch + time need to run together
here:
- a predicted rate isn't bounded to be non-negative;
- `gamma + delta` (fraction of I leaving each step) isn't bounded to <= 1;
- the inflow from S can't exceed what's actually left in S.
"""
import torch


def inverse_scale(x, scaler):
    """Undo an sklearn `MinMaxScaler`'s `transform` in torch, keeping `x` in
    the autograd graph. `scaler.min_`/`scaler.scale_` are fitted constants
    (not learned parameters), so treating them as plain tensors is exact:
    sklearn's `inverse_transform` is `(x - min_) / scale_`.
    """
    min_ = torch.as_tensor(scaler.min_, dtype=x.dtype, device=x.device)
    scale_ = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
    return (x - min_) / scale_


def rebuild(sird_vars, prev, n):
    """Reconstruct absolute I/R/D counts from predicted beta/gamma/delta
    rates, batched and differentiable.

    Args:
        sird_vars: `(batch, steps, 3)` tensor of (already inverse-scaled,
            physical-units) beta, gamma, delta rates.
        prev: `(batch, 4)` tensor, the last known s, i, r, d state. Ground
            truth, not a model output -- never carries gradient in practice,
            but nothing here forces `.detach()` since that's the caller's
            data, not this function's business.
        n: population -- python scalar, or a `(batch,)`/`(batch, 1)` tensor.

    Returns:
        `(batch, steps, 3)` tensor of rebuilt i, r, d counts, matching
        `pipeline/sird.py::rebuild`'s default `return_s=False` layout.
    """
    beta = torch.clamp(sird_vars[..., 0], min=0.0)
    gamma = torch.clamp(sird_vars[..., 1], min=0.0)
    delta = torch.clamp(sird_vars[..., 2], min=0.0)

    # No more can be removed from I in one step than is in it -- rescale
    # gamma/delta down (keeping their ratio) when they'd remove more than
    # all of it. Elementwise over the whole (batch, steps) grid: this only
    # depends on that step's own gamma/delta, not on accumulated state, so
    # doing it once before the recurrence is exactly the per-step clamp.
    removed_scale = torch.clamp(gamma + delta, min=1.0)
    gamma = gamma / removed_scale
    delta = delta / removed_scale

    s, i, r, d = prev[:, 0], prev[:, 1], prev[:, 2], prev[:, 3]
    outputs = []
    for step in range(sird_vars.shape[1]):
        delta_r = gamma[:, step] * i
        delta_d = delta[:, step] * i
        # No more can leave S in one step than is in it.
        delta_i_in = torch.minimum(beta[:, step] * i * (s / n), s)
        delta_s = -delta_i_in
        delta_i = delta_i_in - (delta_r + delta_d)

        s = s + delta_s
        i = i + delta_i
        r = r + delta_r
        d = d + delta_d

        outputs.append(torch.stack([i, r, d], dim=-1))

    return torch.stack(outputs, dim=1)
