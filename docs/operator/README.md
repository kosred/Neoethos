# Operator documentation index

This directory holds the operator-facing reference for the
search → portfolio → live promotion pipeline. The audit (P2-3 plus
the Codex Phase 30-40 doc handoff list, item 39) calls out that every
operational decision an operator makes must have a runbook
documenting the contract behind it.

The references below are intentionally short and opinionated. They do
not duplicate the consolidated audit ([
`docs/audits/ALL_AUDITS_CONSOLIDATED_2026-05-06.md`](../audits/ALL_AUDITS_CONSOLIDATED_2026-05-06.md))
— that file is the working map for engineering decisions. These docs
exist so an operator can answer "is this portfolio promotable, and if
not, what do I do about it" without reading source code.

## Contents

### [`artifact_safety.md`](artifact_safety.md)

Per-artifact reference. One section for each of the five validation
artifact kinds (`canonical_strategy_backtest`, `walkforward_validation`,
`forward_test_validation`, `live_execution_simulation`,
`prop_firm_risk_validation`) covering what the artifact is, how it is
produced, the required provenance fields, whether it is live-safe,
and how it is invalidated and rebuilt. Closes with a worked example
of `validate_provenance` + `validate_evidence` chained via the
Phase 28 evidence bridge, plus a "trigger → affected artifacts"
rebuild matrix.

### [`promotion_readiness.md`](promotion_readiness.md)

Runbook for the `LivePromotionGate` verdict. Covers the four
orthogonal checks (validation evidence, runtime safety, live
execution contract, determinism requirement), one section per
rejection variant (`MissingValidationEvidence`,
`LiveRejectedRuntimeMode`, `LiveRejectedMismatch`,
`LiveRejectedStaleArtifact`, `LiveRejectedFailedEvidenceGate`,
`LiveRejectedMissingEvidence`, `PromotionRejectedDeterminism`) with
contract source, plain-language meaning, common causes per kind, and
rebuild instructions. Recommends a fix-order walkthrough so operators
do not chase secondary errors.

### [`profile_json_reference.md`](profile_json_reference.md)

Field-by-field reference for the `*_profile.json` written next to
every portfolio export. Grouped into seven sections (run identity,
search filters, resolved runtime overrides, run observations,
validation gates, per-kind artifact counts, promotion-readiness
summary). The forward-compatibility note clarifies that the schema is
additive so consumers should not reject profiles with new keys.

## Reading order

1. Start with [`profile_json_reference.md`](profile_json_reference.md)
   to understand the data on disk.
2. Move to [`artifact_safety.md`](artifact_safety.md) when you need
   to interpret a specific artifact kind or rebuild it after a
   configuration change.
3. Move to [`promotion_readiness.md`](promotion_readiness.md) when a
   live bridge rejects an artifact and you need to walk the
   `rejection_reasons` list back to a fix.

## Conventions

- Every field name in these docs matches the field name in the
  persisted JSON or the typed Rust struct exactly.
- Every contract / helper reference points to its source file in
  `crates/forex-core/src/contracts/` or
  `crates/forex-search/src/`. Consult the source when the doc is
  ambiguous; the doc is wrong.
- "Operator" means the human reading the artifacts, not the
  automated live bridge. Most references describe what the operator
  should *do*; the live bridge's behavior is fixed by the contracts.
