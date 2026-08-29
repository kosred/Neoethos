# Resident A2 Schema V4 Design

## Status and approved choice

The architecture owner approved the phased typed-draft design on 2026-08-26.
It preserves the canonical CPU feature order, appends higher-timeframe columns
under an explicit schema migration, mirrors Session semantic-v2 exactly, and
requires typed Quant inputs rather than accepting permanently invalid columns.

Three approaches were considered:

1. Seal a monolithic recipe from the existing Classic-first runtime graph.
   Rejected because it changes schema ordinals and would rebase already-hashed
   Classic routes.
2. Build producer-local move-only drafts, then assign global ordinals exactly
   once in Data after every preceding producer count and memory receipt is
   known. **Approved.**
3. Let the runtime discover routes and memory after device acquisition.
   Rejected because full-workspace admission must know exact bytes before the
   one-shot device carrier is consumed.

## Separate authorities

The implementation names three orders and never treats them as aliases:

- **Schema order:** `SMC -> Classic -> Quant -> Session -> Regime -> Footprint
  -> HigherTimeframeAlignment`.
- **Capability-manifest order:** the existing ten-entry contract order. This is
  capability inventory only.
- **Runtime schedule:** ascending admitted batch-span order, because the current
  assembler accepts only its next destination range. Every pack-ready event is
  retired before the next producer batch is appended.

RobustNormalization, canonical SHA/Merkle, and feature-major-to-bar-major layout
are transforms/infrastructure. They have no schema span and never participate
in producer batch-to-route coverage.

Appending HTF changes the historical six-family CPU schema, so Data defines a
new domain-versioned schema identity and a fail-closed migration policy. No
existing artifact is silently interpreted as the new schema.

## Typed producer drafts

Each column producer creates a move-only `ResidentProducerDraftV4` containing:

- producer identity and exact semantic version;
- ordered route fragments without global ordinal, route id, or route receipt;
- complete route semantics: feature name, indicator/output ids, stage, swept
  period, private typed canonical parameters, and producer-local route domain;
- one or more contiguous producer-local batch fragments of width `1..=64`;
- owner-supplied exact additional-retained and scratch bytes for every batch;
- the implementation capability that will execute the same semantics.

The Data assembler accepts producers only in schema order, rejects duplicate
feature names and gaps, assigns the final global ordinal once, and derives the
route id and receipt from the full typed fragment plus that ordinal. Receipt
creation is not a rebase: no global receipt exists before the seal.

Data derives each canonical parameter-tuple hash during the global seal; no
draft or caller may supply that hash. Capability construction is separate:
Data indexes capabilities by producer and reconstructs the existing ten-entry
manifest order instead of reusing schema-draft insertion order.

Classic retains its local destination indices in the gpu-cuda recipe. Its Data
draft carries no global receipt. A typed pre-device Classic memory receipt must
come from the same runtime-owner logic that defines its output/scratch
allocations; Data composes those bytes but does not duplicate VectorTA formulas.

## Source and identity lifetime

`PinnedCanonicalSeriesV1` gains a crate-private move-only resident descriptor.
The descriptor retains every exact generation manifest and reader lease, and
can derive `SourceArtifactBindingV1` values without decoding OHLCV. The base
generation and selected direct higher timeframes remain pinned through final
store sealing and Search-consumer ownership.

The first resident descriptor admits full-generation segments only, because
manifest metadata proves only full row/timestamp extents without decoding.
Arbitrary windows require a later typed indexed-segment authority.

The recipe seal creates a domain-separated pre-fit template identity for
admission only. It retains typed source bindings and `FeatureNodeV1` construction
inputs; it does not pretend that `FeaturePlanV1` has a schema-only identity.

After all producer runtime receipts and RobustNormalization evidence validate,
`seal_gpu_resident_feature_store_v3()` consumes the template:

- enabled normalization adds the exact fitted normalization node using the
  runtime fit digest;
- disabled normalization validates the canonical disabled receipt and adds no
  fitted node;
- `FeaturePlanV1` and `DatasetFeatureArtifactProvenanceV1` are built and retained;
- only their identities enter the sealed low-level store contract.

The complete move-only resident source descriptor, not only its bindings, moves
through recipe preflight, admission, seal token, sealed resident store, and the
Search consumer. Its generation leases therefore outlive every imported device
view.

No caller can supply recipe, route, plan, provenance, or fit hashes.

## Session semantic-v2

Session remains an atomic 23-column semantic-v2 producer. Resident admission
fails before device acquisition if volume is absent, non-finite, negative, or
has a mismatched extent. One CUDA thread scans
rows in ascending order to preserve the CPU arithmetic and state-update order.
It consumes retained OHLCV, mandatory volume, and canonical millisecond
timestamps on the same primary context and non-default stream.

Exact v2 parity retains its dual-clock quirk: the value lane infers timestamp
units from the first sixteen nonzero timestamps before conversion, while the
validity lane interprets the original canonical millisecond timestamp. The
single kernel therefore owns two independent clock/state lanes. Removing this
quirk is a later semantic-v3 migration, not a resident optimization.

The owner retains exactly `207 * rows` producer bytes (`184N` values and `23N`
logical validity), with zero additional-retained bytes, zero scratch, zero
parent H2D, zero feature/validity D2H, one native launch, one ready event, and
zero host synchronization. Generic pointer tables require 736 bytes; isolated
pointer/schema accounting is 1,377 bytes. Invalid cells receive bits
`0x7ff8000000000000` and, on the present-volume resident domain, only validity
codes Valid `0`, Warmup `1`, or ZeroDenominator `5`. The Data receipt binds
`session_features.rs`, `timestamps.rs`, `features.rs`, allocation, event,
context, stream, route span, and runtime evidence.

Values, validity, context, stream, ordinal, bindings, and the producer-ready
event live through the downstream pack-ready event, followed by stream-ordered
nonblocking release. CPU-only tests retain absent-volume/MissingInput coverage.

Every Session local route uses domain
`neoethos.data.resident-session-route.semantic-v2`, indicator id
`neoethos_session_semantic_v2`, output id equal to the exact feature name,
`ResidentFeatureStageV3::Derived`, no swept period, and typed parameters binding
the dual-clock policy, fixed UTC windows, cumulative ATR policy, and the
output-specific session tag. Unit inference binds all magnitude buckets, the
first-sixteen-nonzero 75-percent vote/fallback rule, and the
`10_000_000_000_000` boundary.

## Quant typed inputs

Quant uses an explicit typed semantic-v3 authority while retaining the exact
ordered 63-column family. No changed input validity may be advertised as v2. A
typed input receipt
must distinguish values derived from the canonical base frame, canonical UTC
timestamps, direct-timeframe leases, and explicit market metadata. Columns
whose required market fact cannot be derived from these authorities are not
silently marked `MissingInput`; they require an explicit Quant-v3 remove/replace
migration before the complete capability can be advertised.

The twenty-two formerly permanent `MissingInput` columns receive one sealed
typed authority: validated uniform `timeframe_millis`; UTC day/session
boundaries identical to Session semantic-v2; versioned
`trading_sessions_per_year = 252`; and checked bars-per-session/day/week.
Eight volatility columns bind the annualization policy, while fourteen temporal
columns bind the exact session/day/week contract. The 63 names remain unchanged
and every v3 formula/validity/warmup rule must be frozen before promotion.

Quant-v3 supports fixed intraday base timeframes only; calendar D1/W1/MN1 base
recipes fail before admission because they have no uniform millisecond duration.
The unchanged `quant_orb_4/8/12` names select the Asian/UTC-day session whose
semantic-v2 open is 00:00 UTC. The base grid must divide that session exactly,
must provide at least twelve bars per eight-hour Asian session (canonical M30 or
finer), and ORB availability resets at the typed UTC-day/session key; no
London/New York choice is inferred from timestamps.

RobustNormalization-v2 remains unchanged, including rejection when a training
column has no valid observation. Quant-v3 must make all 63 columns meaningful or
fail closed; permanent `MissingInput` and normalization relaxation are forbidden.

## Higher-timeframe alignment

HTF owns the selected direct-timeframe resident parents and their generation
leases. It applies existing causal availability rules: a base row may see only
a higher-timeframe bar that is closed and available at that base timestamp.
Aligned columns are appended after Footprint under schema-v4 in exact order:
selected direct-timeframe order, canonical CPU producer order, then each
producer's local output order. Timeframe and causal-availability rule are bound
into every aligned route identity. Parent uploads,
alignment outputs, scratch, events, and zero feature D2H are bound by typed
allocation/lifetime/runtime receipts.

## Failure and testing policy

Every seam fails before device-carrier consumption if a source binding, semantic
version, producer, route, batch, memory extent, or transform receipt is missing.
All move-only authorities forbid `Clone`/`Copy` and public constructors from raw
bytes or hashes.

Local work uses standalone source contracts and direct `rustc --test` fixtures
only. Cargo, NVCC, SASS inspection, and real-device parity remain RTX gates.
Capability promotion occurs only after the complete runtime receipt path exists.
The production schema seal remains unreachable until all seven column-producer
drafts and all three transform/infrastructure receipts exist; synthetic draft
tests do not mint admission or plan authority.
