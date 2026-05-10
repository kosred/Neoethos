# Follow-on phases 41-44: runtime safety metadata

This slice continues the contract-only path while Claude Code's Phase 25/26
work remains uncommitted in the search and app worktree. It avoids
`forex-search` and UI wiring, and adds the shared runtime-safety metadata that
downstream integration can consume later.

## Completed in this Codex slice

### Phase 41 — runtime safety issue registry

Added `RuntimeSafetyIssue` in `forex-core::contracts` to classify why an
artifact is not live-safe:

- non-canonical runtime mode,
- degraded or unavailable backend,
- backend/device-assignment mismatch,
- missing degraded-runtime reason,
- canonical artifact carrying degraded metadata.

### Phase 42 — provenance runtime safety report

Added `RuntimeSafetyReport` plus `ArtifactProvenance::runtime_safety_report`.
The report exposes runtime mode, backend kind, assignment backend, live-safety
boolean, degraded reason, and issue list without requiring UI or live code to
parse validation errors.

### Phase 43 — promotion readiness runtime snapshot

Extended `PromotionReadinessReport` with `runtime_safety_passed` and a full
`runtime_safety` snapshot. The promotion report now has a dedicated
`RuntimeSafety` readiness check before the live execution contract check.

### Phase 44 — downstream export surface

Re-exported `RuntimeSafetyIssue` and `RuntimeSafetyReport` from `forex_core`, so
later `forex-search`, `forex-models`, and `forex-app` integration can consume
the same runtime-safety contract.

## Integration guardrail

When Claude's artifact producers land, use this report as the single source for
runtime-safety display and logging. Do not add a second degraded-mode schema in
discovery profiles or UI state.
