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

Scope: fuses single `nn.Linear` layers only. Does NOT fuse the recurrent
`LILSTMCell2`/`nn.LSTM` decoder cells (`hfta.ops` has no LSTM op) or wire
into `ClusterModel`'s training loop -- replacing the per-member Python
loop in `model/train.py`'s `__eval` with one batched forward call across
all members is a separate, larger architectural change, not attempted
here.
"""
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
