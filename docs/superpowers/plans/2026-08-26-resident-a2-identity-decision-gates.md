# Resident A2 Identity Decision Gates

> **Status:** Reviewed and deliberately blocked before production edits. This is
> not an executable implementation plan until the semantic and ownership
> decisions below are resolved.

**Goal:** Reach an honest, runnable ten-of-ten resident producer manifest and
the first canonical full search without host feature materialization, caller-
minted identities, guessed memory extents, or partial capability claims.

## Frozen current-source facts

- The canonical CPU artifact schema is `SMC -> Classic -> Quant -> Session ->
  Regime -> Footprint`.
- The resident capability manifest order is `Classic -> SMC -> Quant ->
  Session -> Regime -> Footprint -> HTF -> Robust -> SHA -> layout`.
- The current resident runtime executes `Classic -> SMC -> Robust -> seal`.
- Classic route receipts begin at global ordinal zero and bind that ordinal into
  their hashes. They cannot be honestly rebased after receipt creation.
- Exact Classic retained/scratch bytes are known only after VectorTA runtime
  output owners exist. Data has no typed pre-device Classic memory receipt.
- `FeaturePlanV1` identifies a complete canonical plan; it does not provide a
  pre-fit schema identity.
- Quant semantic-v2 is one atomic 63-column capability. Twenty-two columns are
  permanently `MissingInput` with the current inputs, and enabled
  RobustNormalization-v2 rejects any column with no valid training cell.
- Session authority is semantic-v2, not v3. An exact mirror is a serial
  23-column CUDA state machine; a parallel/calendar-corrected implementation is
  a new semantic version and migration.
- HTF requires retained direct-timeframe parents and causal alignment ownership
  that the current single-frame materializer does not possess.

## Required decisions

### 1. Global ordering authority

Choose and version one schema/route order. Keep capability-manifest order and
runtime scheduling as separate contracts. If CPU artifact order remains
authoritative, Classic preflight must accept the final global starting ordinal
or defer route receipt creation until the Data-owned global seal.

### 2. Quant-v2 closure

- **A — typed inputs:** provide exact annualization/session/timeframe inputs so
  all 63 columns can produce meaningful validity.
- **B — versioned normalization:** define a new normalization semantic that
  preserves an all-invalid column without treating it as a fit error.
- **Reject:** advertising only 41 columns under the existing Quant-v2 identity.

### 3. Session closure

- **A — exact semantic-v2:** one deterministic sequential f64 CUDA kernel,
  canonical invalid NaN bits, exact validity codes, mandatory resident volume,
  one ready event, zero feature D2H, and `rows * 23 * 9` retained bytes.
- **B — audited semantic-v3:** first define corrected session/calendar/reset/
  ATR/previous-close/volume semantics, update the CPU authority and manifest,
  and add an explicit migration policy before implementing CUDA.

## Earliest honest implementation sequence after decisions

1. Add a move-only pinned source descriptor carrying exact source bindings,
   leases, normalization split, and canonical frame authority through final
   consumption.
2. Add producer-local route and batch drafts. Owner-supplied preflight receipts
   must carry exact retained/scratch extents; Data may compose but not infer
   them.
3. Add typed Classic pre-device memory ownership and global-ordinal route
   sealing before device admission.
4. Flatten drafts in the chosen schema order and derive complete route receipts,
   a domain-separated pre-fit template identity, and the exact working-set plan.
5. Implement and validate complete Quant, Session, and HTF components. Column
   producers alone participate in route/batch coverage; Robust, SHA, and layout
   remain transforms/infrastructure.
6. Finalize inside `seal_gpu_resident_feature_store_v3()` only after runtime
   normalization evidence exists. Enabled mode adds the fitted normalization
   node; disabled mode validates its canonical disabled evidence and adds no
   fitted node. Retain the exact `FeaturePlanV1` and validated
   `DatasetFeatureArtifactProvenanceV1`, not caller-provided hashes.
7. Promote the manifest only after exact runtime receipt validation for every
   producer, then run warning-denied Cargo/NVCC/device gates and the canonical
   full search on RTX.

## Deliberately forbidden interim states

- No complete identity template while any route, ordinal, source binding,
  producer batch, or normalization mode is unresolved.
- No guessed Classic/Quant/HTF scratch bytes.
- No Session-v3 label over unchanged semantic-v2 behavior.
- No Quant partial-family capability under the atomic semantic-v2 schema.
- No capability-count promotion from isolated tests or unreachable routes.
