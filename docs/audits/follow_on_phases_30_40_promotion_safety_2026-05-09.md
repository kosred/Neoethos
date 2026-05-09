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

### Phase 35 — typed validation evidence registry

Added `ValidationEvidenceKind` as the shared registry for the five required
promotion evidence inputs. Future discovery / validation producers can target
the enum instead of stringly-typed field names.

### Phase 36 — evidence completeness telemetry

Added `ValidationEvidenceCheck` plus `ValidationEvidenceManifest::hash_for`,
`missing_kinds`, and `evidence_checks`. Operator and UI layers can now show
which validation artifact is missing without parsing error strings.

### Phase 37 — deterministic promotion clock

Added `LivePromotionGate::validate_at` and `readiness_report_at`, so stale
artifact checks can be tested and reproduced with an injected clock instead of
depending on wall-clock time.

### Phase 38 — structured readiness checks

Extended `PromotionReadinessReport` with structured `PromotionReadinessCheck`
items for validation evidence, live execution contract, and deterministic
execution requirements. This preserves the existing boolean summary while
giving UI / logs stable machine-readable statuses.

### Phase 39 — cross-crate export surface

Re-exported the promotion gate, evidence manifest, evidence checks, and
readiness report types from `forex_core`, so downstream `forex-search` /
`forex-app` code can consume the contract without reaching through internal
modules.

### Phase 40 — integration handoff guardrails

The remaining work is intentionally integration-only and should start after
Claude Code's Phase 25+ artifact producers are committed. The guardrail is:
wire producers into this contract, do not create another live-readiness schema
in discovery or UI code.

## Handoff after Claude Phase 25+ lands

These integration stages should be wired after Claude Code's artifact work
lands:

1. Feed live-execution simulation artifact hashes into `ValidationEvidenceManifest`.
2. Feed prop-firm risk validation artifact hashes into `ValidationEvidenceManifest`.
3. Attach promotion readiness reports to discovery/profile outputs.
4. Surface promotion readiness in the UI hardware/runtime panel.
5. Add operator docs for promotion rejection reasons and remediation.
6. Add end-to-end promotion tests once all artifact producers are merged.
