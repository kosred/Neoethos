# Follow-on phases 30-40: promotion safety rail

This sidecar sequence is intentionally independent from Claude Code's Phase 25 work.
It does not touch discovery validation artifacts directly. Instead, it defines the
shared promotion contract that later phases can use once live-execution simulation
and prop-firm validation artifacts are produced.

## Completed in this Codex slice

### Phase 30 — validation evidence manifest

Added `ValidationEvidenceManifest` in `forex-core::contracts`.
Live promotion now has one typed place for the required evidence hashes:

- canonical backtest validation,
- walk-forward validation,
- forward-test validation,
- live-execution simulation,
- prop-firm risk validation.

The manifest fails closed when any required evidence hash is missing.

### Phase 31 — live promotion gate

Added `LivePromotionGate`, which combines the existing `LiveExecutionContract`
with the new validation-evidence manifest. Promotion now validates both the
artifact contract and the evidence bundle before accepting a live-ready strategy.

### Phase 32 — deterministic promotion requirement

`LivePromotionGate` defaults to `require_deterministic = true`. Best-effort or
non-deterministic artifacts remain structurally valid, but they cannot pass the
promotion gate unless an operator explicitly disables the deterministic
requirement.

### Phase 33 — degraded runtime rejection at promotion boundary

The promotion gate delegates to `LiveExecutionContract`, so degraded,
fallback, approximate, or diagnostic runtime modes remain rejected even when
all validation evidence hashes are present.

### Phase 34 — operator-facing readiness report

Added `PromotionReadinessReport` via `LivePromotionGate::readiness_report`.
This gives UI/operator layers a non-throwing summary:

- validation evidence completeness,
- live-contract result,
- determinism requirement result,
- final readiness boolean,
- rejection reasons suitable for display/logging.

## Handoff for phases 35-40

These stages should be wired after Claude Code's Phase 25+ artifact work lands:

35. Feed live-execution simulation artifact hashes into `ValidationEvidenceManifest`.
36. Feed prop-firm risk validation artifact hashes into `ValidationEvidenceManifest`.
37. Attach promotion readiness reports to discovery/profile outputs.
38. Surface promotion readiness in the UI hardware/runtime panel.
39. Add operator docs for promotion rejection reasons and remediation.
40. Add end-to-end promotion tests once all artifact producers are merged.
