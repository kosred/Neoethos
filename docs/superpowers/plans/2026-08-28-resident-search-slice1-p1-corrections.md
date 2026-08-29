# Resident Search Slice1 P1 Correction Plan

## Scope and invariants

Correct only the reviewed Slice1 configuration/admission seams. Do not change CPU novelty, add a numerical fallback, expose raw handles, link a CUDA binary, or run a GPU. Production must remain fail closed when the native trim preflight provider is absent.

## TDD sequence

1. Extend `crates/neoethos-search/tests/current_config_resident_slice1_contract.rs` with RED source contracts for:
   - one versioned typed canonical digest that exhaustively destructures all shipped `DiscoveryConfig` fields and nested semantic records;
   - deterministic sorted map encoding, ordered vector encoding, and exact `f64::to_bits` encoding;
   - a distinct trim-preflight receipt/identity, with the old population-byte alias forbidden.
2. Extend `crates/neoethos-search/src/gpu_resident_current_config_plan_v1_tests.rs` with grouped mutation tests covering every top-level and nested Search-semantic field, insertion-order-independent maps, order-sensitive higher timeframes, and EURUSD/GBP `None` versus `Some(last_close)` financial identity.
3. Implement `canonical_discovery_config_digest_v1` as an explicit schema-tagged typed encoder. Exhaustive destructuring is the compile-time field-coverage ratchet; all sizes are checked, floats use raw bits, maps sort keys, and ordered collections preserve order. Bind the digest into `SealedCurrentConfigResidentSearchPlanV1` and its plan identity while retaining existing trim sub-identities.
4. Add an opaque resident-trim native preflight receipt to `full_discovery_workspace_plan_v1`. It binds independently queried peak/retained extents, CUB scratch, population overlap/lifetime, native query/calibration identities, and the full-workspace identity. Remove the population-byte alias. Only a `cfg(test)` fixture may mint synthetic facts; production sealing requires the provider receipt before allocation.
5. Run the direct focused contract after each seam. After the serialized Cargo slot is released, run focused unit tests and the exact no-default/gpu-b-adapter `-Dwarnings --locked --offline -j7` no-link gates. Rustfmt, inspect the complete diff, commit on the isolated branch, produce hashes/log manifest, then clean only this worktree's target.

## Acceptance

- Every shipped Search-semantic configuration mutation changes the canonical digest.
- Equivalent map insertion order does not change it; higher-timeframe order does.
- Current EURUSD/GBP geometry is bit-identical for absent versus supplied last-close.
- Trim reserve is derived only from the distinct opaque native/query/calibration preflight receipt; absence fails before allocation.
- Existing trim/schema/financial/runtime sub-identities remain bound.
