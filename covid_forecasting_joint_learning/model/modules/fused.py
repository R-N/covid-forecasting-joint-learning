"""Optional HFTA-based member fusion for architecturally-identical
`SingleModel` Linear layers within one `ClusterModel` cluster.

INVESTIGATION.md, Big wins #1: "Member batching via HFTA rather than by
hand ... Cluster members within a trial are architecturally identical and
fuse cleanly" -- confirmed by `general.py`: `ClusterModel.__init__` builds
every member's `SingleModel` from the *same* `sizes`/`model_kwargs`
(`k.model = SingleModel(**sizes, **model_kwargs)` for every `k` in
`self.members`), so their Linear layers are shape-identical across
members, just independently weighted.

`hfta`'s fused `Linear` op (`hfta.ops.Linear`) folds a "B" (number of
fused members) dimension into `torch.bmm`/`torch.baddbmm` -- plain,
device-agnostic torch ops, not a CUDA-only kernel -- so correctness is
verifiable, and this module usable, on CPU too. Verified out-of-band
against the real pip-installed library (not this dev environment, which
doesn't have `hfta` installed): 4 architecturally-identical `nn.Linear`
layers fused via `hfta.ops.Linear(B=4)` produced output identical (max abs
diff `0.0`) to running the 4 layers separately and stacking the results --
see `tests/test_fused_linear.py` for the reproduction command and the
snatch/forward math re-verified against a stub reproducing `hfta.ops.
Linear` verbatim (MIT-licensed) for dev-environment testing.

The *training-throughput* benefit HFTA targets is GPU/TPU-specific and
unmeasured here -- this repo's dev machine has no CUDA GPU, so a fused
forward pass on CPU proves correctness, not speed.

Covers `nn.Linear` (`fused_linear_from_members`), `ResidualFC` at
depth<=1 (`fused_residual_fc_from_members` -- this repo's actual configs;
deeper `ResidualStack`s raise `NotImplementedError` rather than silently
mis-fusing), `CombineHead` (`fused_combine_head_from_members`), and
`DirectFutureHead` (`fused_direct_future_head_from_members`), plus
`fuse_direct_decoder_from_members` gluing the last three into the
complete non-autoregressive decoder stage for `SingleModel(...,
direct_multi_horizon=True)` models. Does NOT fuse the recurrent
`LILSTMCell2`/`nn.LSTM` decoder cells (`hfta.ops` has no LSTM op, so
`PastModel`'s `PastHead`, and the `LILSTMCell2`-looped recursive decoder
used when `direct_multi_horizon=False`, both stay per-member) -- only the
direct-decoder path is fully Linear, hence fully fusable.

Not wired into `ClusterModel`'s default training loop (`model/train.py`'s
`__eval` still calls each member separately) -- that requires restructuring
the per-member loop itself, a larger change deferred past this module.
`fuse_direct_decoder_from_members` is real, usable, and numerically
verified against unfused `SingleModel` instances (see
`tests/test_fused_linear.py::test_fuse_direct_decoder_matches_unfused_members`)
for whoever wires it in.
"""
import torch
from torch import nn

try:
    from hfta.ops import Linear as _HFTALinear
    HFTA_AVAILABLE = True
except ImportError:
    _HFTALinear = None
    HFTA_AVAILABLE = False


def _require_hfta():
    if not HFTA_AVAILABLE:
        raise ImportError(
            "model.modules.fused requires the optional `hfta` package: "
            "pip install git+https://github.com/UofT-EcoSystem/hfta"
            "@12cb0bb5031d36b909605981c1e90e96fdfae57a"
            " (see requirements-experiment.txt)"
        )


def fused_linear(in_features, out_features, bias=True, B=1):
    """One `hfta.ops.Linear`, batched over `B` architecturally-identical
    members. Input format: `(B, *, in_features)`."""
    _require_hfta()
    return _HFTALinear(in_features, out_features, bias=bias, B=B)


def fused_linear_from_members(layers):
    """Fuse `B = len(layers)` existing `nn.Linear` layers -- e.g. one
    `ResidualFC`'s inner `nn.Linear` per cluster member -- into a single
    `hfta.ops.Linear`, preserving each member's own weights (via HFTA's
    own `snatch_parameters`). Every layer must share the same
    `(in_features, out_features, bias is not None)`, matching the
    `general.py:154` guarantee that cluster members share `sizes`/
    `model_kwargs`.
    """
    _require_hfta()
    assert layers, "fused_linear_from_members needs at least one layer"
    in_features = layers[0].in_features
    out_features = layers[0].out_features
    has_bias = layers[0].bias is not None
    for layer in layers[1:]:
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert (layer.bias is not None) == has_bias

    fused = _HFTALinear(in_features, out_features, bias=has_bias, B=len(layers))
    for b, layer in enumerate(layers):
        fused.snatch_parameters(layer, b)
    return fused


def fused_residual_fc_from_members(residual_fcs):
    """Fuse `B = len(residual_fcs)` `ResidualFC` instances (depth<=1 --
    this repo's actual usage; every `ResidualFC(...)` call site in
    `model/modules/main.py`/`combine.py`/`head.py` uses the depth=1
    default). Dispatches on `ResidualStack`'s three depth<=1 shapes:
    `nn.Identity` (depth=0), a plain `nn.Sequential(Linear, activation)`
    (depth=1, input_size != output_size), or a `ResidualBlock` wrapping
    that same Sequential (depth=1, input_size == output_size, e.g.
    `CombineHead`'s `precombine`). Deeper `ResidualStack`s (depth>1) raise
    `NotImplementedError` rather than silently mis-fusing.
    """
    _require_hfta()
    assert residual_fcs, "fused_residual_fc_from_members needs at least one ResidualFC"
    mains = [r.main.main for r in residual_fcs]  # ResidualFC.main is a ResidualStack
    first = mains[0]

    if isinstance(first, nn.Identity):
        for m in mains[1:]:
            assert isinstance(m, nn.Identity)
        return nn.Identity()

    if isinstance(first, nn.Sequential) and len(first) == 2 and isinstance(first[0], nn.Linear):
        for m in mains[1:]:
            assert isinstance(m, nn.Sequential) and len(m) == 2 and isinstance(m[0], nn.Linear)
        return _fused_sequential_linear_block(mains)

    from .residual import ResidualBlock
    if isinstance(first, ResidualBlock):
        for m in mains[1:]:
            assert isinstance(m, ResidualBlock)
        return _fused_residual_block_from_members(mains)

    raise NotImplementedError(
        f"fused_residual_fc_from_members only supports depth<=1 ResidualFC "
        f"(nn.Identity / plain Linear+activation / a single ResidualBlock); "
        f"got {type(first).__name__} for a deeper ResidualStack, not implemented."
    )


def _fused_sequential_linear_block(sequentials):
    # Each is nn.Sequential(nn.Linear(...), fc_activation()).
    linears = [seq[0] for seq in sequentials]
    activation_cls = type(sequentials[0][1])
    return nn.Sequential(fused_linear_from_members(linears), activation_cls())


class FusedResidualBlock(nn.Module):
    """Fuses B `ResidualBlock` instances (`model/modules/residual.py`) --
    same highway-residual math, `w` stacked to `(B, 1)` and broadcast over
    every dim after the leading B dim.
    """

    def __init__(self, fused_main_block, w, highway=False, activation_cls=nn.Identity):
        super().__init__()
        self.main_block = fused_main_block
        self.w = nn.Parameter(w)
        self.highway = highway
        self.activation = activation_cls()

    def forward(self, x):
        w = self.w.view(self.w.shape[0], *([1] * (x.dim() - 1)))
        if self.highway:
            w = w.clamp(0, 1.0)
            residual = (1 - w) * x + w * self.main_block(x)
        else:
            residual = x + w * self.main_block(x)
        return self.activation(residual)


def _fused_residual_block_from_members(blocks):
    fused_inner = _fused_sequential_linear_block([b.main_block for b in blocks])
    highway = blocks[0].highway
    for b in blocks[1:]:
        assert b.highway == highway
    activation_cls = type(blocks[0].activation)
    w = torch.stack([b.w.detach().clone() for b in blocks])  # (B, 1)
    return FusedResidualBlock(fused_inner, w, highway=highway, activation_cls=activation_cls)


class FusedCombineHead(nn.Module):
    """Fuses B `CombineHead` instances (`model/modules/combine.py`),
    `post_future_model` in `SingleModel`."""

    def __init__(self, use_shared_head, precombine, w0, reducer):
        super().__init__()
        self.use_shared_head = use_shared_head
        self.precombine = precombine
        self.w0 = None if w0 is None else nn.Parameter(w0)
        self.reducer = reducer

    def forward(self, x_private, x_shared=None):
        if self.use_shared_head:
            x_shared = x_shared if self.precombine is None else self.precombine(x_shared)
            # w0 stored as (B, private_size); reshape to match x_private's
            # rank (x_private is (B, *, private_size) -- (B, batch,
            # private_size) for the recurrent decoder's per-step call, or
            # (B, batch, future_length, private_size) for the direct
            # decoder's one-shot call).
            w0 = self.w0.view(self.w0.shape[0], *([1] * (x_private.dim() - 2)), self.w0.shape[-1])
            x_private = w0 * x_private
            x = torch.cat([x_private, x_shared], dim=-1)
        else:
            x = x_private
        return self.reducer(x)


def fused_combine_head_from_members(combine_heads):
    _require_hfta()
    assert combine_heads, "fused_combine_head_from_members needs at least one CombineHead"
    use_shared_head = combine_heads[0].use_shared_head
    for c in combine_heads[1:]:
        assert c.use_shared_head == use_shared_head

    precombine, w0 = None, None
    if use_shared_head:
        if combine_heads[0].precombine is not None:
            precombine = fused_residual_fc_from_members([c.precombine for c in combine_heads])
        w0s = torch.stack([c.combiner.w0.detach().clone() for c in combine_heads])  # (B, private_size)
        w0 = w0s  # reshaped dynamically in FusedCombineHead.forward to match input rank

    reducer = fused_residual_fc_from_members([c.reducer for c in combine_heads])
    return FusedCombineHead(use_shared_head, precombine, w0, reducer)


class FusedDirectFutureHead(nn.Module):
    """Fuses B `DirectFutureHead` instances (`model/modules/head.py`).
    Input `hx`: `(B, batch, hidden_size)`; output: `(B, batch,
    future_length, state_size)`.
    """

    def __init__(self, horizon_embedding, project):
        super().__init__()
        self.horizon_embedding = nn.Parameter(horizon_embedding)
        self.project = project

    def forward(self, hx, future_exo=None):
        x = hx.unsqueeze(2) + self.horizon_embedding.unsqueeze(1)
        if future_exo is not None:
            x = torch.cat([x, future_exo], dim=-1)
        return self.project(x)


def fused_direct_future_head_from_members(heads):
    _require_hfta()
    assert heads, "fused_direct_future_head_from_members needs at least one DirectFutureHead"
    horizon_embedding = torch.stack([h.horizon_embedding.detach().clone() for h in heads])
    project = fused_residual_fc_from_members([h.project for h in heads])
    return FusedDirectFutureHead(horizon_embedding, project)


def fuse_direct_decoder_from_members(models):
    """Fuse the fully-Linear 'direct decoder' stage (`direct_private_head`,
    `direct_shared_head`, `post_future_model`) of B `SingleModel`
    instances built with identical `sizes`/`model_kwargs` (the
    `general.py:154` guarantee) and `direct_multi_horizon=True`.

    Does NOT touch `past_model` (`PastHead`'s `nn.LSTM` has no fused
    equivalent in `hfta.ops`) -- call each member's own `past_model(past)`
    as before, stack the B members' `hx_private`/`hx_shared` results along
    a new leading dim, and feed them through the returned fused triple to
    get every member's final output in one batched forward call instead
    of B separate `SingleModel.forward()` calls. Not wired into
    `ClusterModel`'s training loop automatically -- see module docstring.

    Returns `(fused_private_head, fused_shared_head_or_None,
    fused_post_future_model)`.
    """
    _require_hfta()
    assert models, "fuse_direct_decoder_from_members needs at least one SingleModel"
    for m in models:
        assert m.direct_multi_horizon, "fuse_direct_decoder_from_members needs direct_multi_horizon=True models"

    fused_private_head = fused_direct_future_head_from_members([m.direct_private_head for m in models])
    fused_shared_head = None
    if models[0].use_shared_head:
        fused_shared_head = fused_direct_future_head_from_members([m.direct_shared_head for m in models])
    fused_post_future_model = fused_combine_head_from_members([m.post_future_model for m in models])
    return fused_private_head, fused_shared_head, fused_post_future_model
