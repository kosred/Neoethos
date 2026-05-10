# Follow-on phases 28-60 — what landed and what comes next

This doc summarises the 33-phase Claude Code follow-on slice that ran
on top of `master @ 99f634bc` (which already carried Codex's Phases
30-44 promotion contracts). It exists so a future operator or engineer
can answer "what did this batch do, and what is still open" without
walking the audit log entry-by-entry.

## Landed (Phases 28-32 + 45-60)

### Discovery validation chain (28-32, 45-49)

- Phase 28 — `live_validation_evidence_from_discovery` bridge: typed
  `LiveValidationEvidence` from `DiscoveryResult.validation_gates` +
  forward-test artifacts.
- Phase 29 — `compute_discovery_prop_firm_artifacts` +
  `save_prop_firm_validation_artifacts` + bridge populates
  `prop_firm_passed`.
- Phase 45 — forex-app discovery service runs the prop-firm helper
  on the OOS tail and persists alongside forward-test.
- Phase 46 — `DiscoveryRequest::prop_firm_rules` for per-challenge
  customisation.
- Phase 47 — operator artifact safety reference
  (`docs/operator/artifact_safety.md`).
- Phase 48 — `discovery_validation_evidence_manifest` builds Codex's
  typed `ValidationEvidenceManifest` from `DiscoveryResult`.
- Phase 49 — `DiscoveryRunProfile.validation_evidence_*` fields
  surface the per-kind hashes + completeness boolean.

### Promotion verdict + operator surfaces (50-58)

- Phase 50 — operator promotion-readiness runbook
  (`docs/operator/promotion_readiness.md`).
- Phase 51 — `DiscoveryRunProfile.determinism_policy` records the
  typed policy that `LivePromotionGate::PromotionRejectedDeterminism`
  references.
- Phase 52 — `discovery_validation_evidence_manifest_excluding_live_sim`
  + `DiscoveryPerKindEvidenceHashes::all_producer_kinds_present` for
  diagnostic display while the simulator is still deferred.
- Phase 53 — `*_profile.json` field reference
  (`docs/operator/profile_json_reference.md`).
- Phase 54 — `BatchDiscoverySummary.portfolios_with_missing_producer_evidence`
  counter.
- Phase 55 — operator docs index (`docs/operator/README.md`).
- Phase 56 — full validation chain integration test in
  `forex-search::discovery::tests`.
- Phase 57 — `DiscoveryPerKindEvidenceHashes::check_summary` for
  tabular UI / log rendering.
- Phase 58 — `save_promotion_summary_json` + forex-app side-file
  (`*_promotion_summary.json`).

### Determinism rollout (59)

- Phase 59 — `forex-models::genetic::train_with_discovery` logs the
  resolved `DeterminismPolicy` so log lines correlate with the
  persisted profile field.

### Phase 60

This document.

## Test coverage growth

forex-search lib tests grew 86 → 107 over this slice; forex-core
tests stayed at 51 (Codex's promotion contract surface). Every phase
either added tests or modified an existing test in lockstep with the
code change. `cargo check` on forex-cli / forex-app / forex-models
remains clean across the entire slice.

## Still open

- **Live-execution simulator**: `live_execution_simulation_hash` is
  structurally absent until a simulator produces real evidence. Phase
  52's lossy manifest helper exists specifically for this gap and
  should be retired once the simulator lands.
- **P1-3 degraded-mode propagation in `discovery_gpu`**: the typed
  `RuntimeSafetyReport` and `RuntimeSafetyIssue` from Codex Phase 41-44
  are not yet attached to the GPU discovery result's runtime backend
  string. A future phase should replace
  `GpuDiscoveryResult::degraded_reason: Option<String>` and
  `runtime_backend: String` with the typed enums.
- **P0-9 deeper coverage**: forex-models' RL paths, exit-agent, and
  sampling helpers still construct their own RNGs. The pattern from
  Phase 26 / 59 (consume `current_determinism_policy()` and surface
  it in tracing) should be propagated module by module.
- **P2-1 UI scheduler exposure**: hardware/runtime panel that exposes
  detected CPU / GPU / VRAM / precision modes. Outside this slice's
  forex-search/forex-app contract scope.
- **Per-challenge prop-firm UI**: `DiscoveryRequest::prop_firm_rules`
  is plumbed (Phase 46) but the UI form does not yet surface a
  challenge selector. The operator currently has to construct the
  request programmatically to override FTMO defaults.

## Reading order for future engineers

1. The audit's master report
   (`docs/audits/ALL_AUDITS_CONSOLIDATED_2026-05-06.md`) — for the
   engineering working map.
2. This doc — for what the 28-60 slice intentionally did and did not
   touch.
3. `docs/operator/README.md` — for the operator-facing references
   produced by this slice.
4. The Phase 60 commit log — for the chronological list of changes.
