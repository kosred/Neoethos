# Exact Search Holdout Scope Design

## Problem

`run_discovery_cycle_with_holdout_and_progress` currently slices the first 80% of the values, then calls the value runner with the full canonical receipt. The existing `CanonicalSearchRunInputV1` correctly rejects that combination: its receipt segments must cover exactly the values it borrows. `DiscoveryResult::artifact_scope` later hides the mismatch by reconstructing an entire-receipt scope with any requested role.

## Decision

Keep `CanonicalSearchRunInputV1` as the strict, already-validated full input. Add a private discovery-core typed wrapper that can only be created from that parent input. The wrapper constructs its own value windows and `CanonicalSearchArtifactScopeV1` together, so callers cannot pair arbitrary slices with a receipt or scope.

A no-holdout run has one borrowed `DiscoveryInput` window covering the full parent input. A holdout run owns two windows derived from the same parent:

- selection: `InSample`, prefix `0..is_end`;
- evidence: `Holdout`, contiguous suffix `is_end..row_count`.

For 100 rows and the existing 20% holdout fraction, the exact windows are `0..80` and `80..100`; timestamps are the first/last values of those half-open ranges.

## Invariants

- Both scopes embed the exact parent receipt.
- Selection and holdout roles cannot be swapped.
- Selection begins at the parent scope start.
- Selection end equals holdout start; gaps and overlaps fail.
- Holdout ends at the parent scope end.
- Window value rows and timestamps exactly equal the corresponding parent slices.
- Empty ranges, a missing holdout suffix, and fewer than 64 in-sample rows fail before discovery.
- `DiscoveryResult` stores the exact selection scope and optional holdout scope.
- `selection_scope()` and `holdout_scope()` validate the public stored fields against `search_input_receipt` before returning authority. The fields remain explicit only for existing cross-crate struct-literal compatibility; there is no default or permissive constructor.
- `validate_evaluated_scopes()` accepts only an exact full `DiscoveryInput`, or a contiguous `InSample` prefix plus `Holdout` suffix. No API may relabel or rebuild result authority with `for_entire_receipt`.

## Data flow

1. Validate the full `CanonicalSearchRunInputV1` as today.
2. Construct private typed full or split discovery inputs from it.
3. Pass the typed selection input into the value runner; do not pass raw features, OHLCV, and receipt separately.
4. Use the typed holdout values for forward/prop evaluation.
5. Store both scopes in `DiscoveryResult` at construction.
6. Portfolio and promotion-summary writers clone `selection_scope()`.

`live_portfolio.rs` and app/model consumers are intentionally outside this slice. Their required call-site migrations are reported separately.

## Testing

Tests use the existing 100-row canonical cTrader fixture and exercise real scope/value construction:

- exact 80/20 roles, source rows, and timestamps;
- no-holdout full `DiscoveryInput` scope;
- swapped roles;
- boundary gap and overlap;
- empty range and missing suffix;
- in-sample length below 64.

The focused Rust test must be observed RED before production edits and GREEN after them. Cargo execution requires an explicit parent GO because the shared vector lane is currently gated.
