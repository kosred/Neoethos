# GPU-Only Feature Workspace Preflight V3 Design

## Scope

This slice introduces the first production-safe prerequisite for sealing the
complete native Discovery workspace. It does not claim that the workspace can
be sealed today and does not change any App or CLI caller.

## Decision

Data owns a move-only preflight token. Its only constructor consumes an exact
`PinnedCanonicalSeriesV1`, reads row-count metadata from the already-pinned
manifest, validates the frozen row budget, and asks Data's crate-owned producer
factory for the complete ordered capability manifest. It does not decode OHLCV
values or build a host `FeatureFrame`.

The token keeps the immutable generation leases alive. Its fields stay private,
and it has no clone, serialization, raw-byte, hash, ordinal, context, or stream
constructor. The one-shot run-device carrier remains outside the preflight and
is not consumed until every producer and every workspace component receipt can
be sealed.

## Current fail-closed frontier

The current capability census contains resident Classic TA, resident SMC, the
already-real in-tree parallel Merkle SHA-256 kernels, and the f64/u4
feature-major-to-bar-major CUDA layout kernel. Data seals each of the latter two
implementations into a move-only allocation, lifetime, and component receipt
derived only from its private resolved plan. The receipts move through
materialization admission and the seal token. The real owner must match the
layout pack count, f64/u4 extents, zero full feature-major staging, Merkle leaf
and two-level scratch extents, retained 32-byte root, one compact root readback,
and final ready-event/consumer lifetime semantics.

The preflight still fails closed with the exact ordered missing producer list:

1. Quant
2. Session
3. Regime
4. Footprint
5. HigherTimeframeAlignment
6. RobustNormalization

After those producers exist, Data must still seal these ordered component
receipts before it can contribute a resident-feature-store charge to the full
workspace authority:

1. dataset recipe identity
2. feature-plan schema identity
3. route-plan identity
4. ordered resident feature routes
5. exact resident producer-batch ledger
6. normalization scratch extent
7. normalization-fit metadata extent
8. final feature-plan identity
9. normalization-fit identity
10. source-provenance identity

This backlog is descriptive non-authority. No byte count or hash supplied by a
caller may satisfy it.

## Next ownership boundary

The next slice must turn a completed Data preflight into one opaque resident
feature-store workspace component receipt and assemble it with opaque
trim/prefilter, resident-generation, post-GA, and bounded-readback receipts.
That assembly requires a separate decision about the cross-crate owner because
`neoethos-data` depends on `neoethos-gpu-cuda`, while `neoethos-gpu-cuda` cannot
depend back on Data.
