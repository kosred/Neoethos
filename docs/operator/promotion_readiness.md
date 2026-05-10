# Operator-facing live-promotion readiness reference

This is the runbook for reading a portfolio's promotion verdict and
acting on every rejection reason `LivePromotionGate` can produce. It
complements [`artifact_safety.md`](artifact_safety.md), which covers
the per-artifact contracts; this document covers the gate that
consumes them.

The audit (item 39 from the consolidated execution plan) calls out
that operators must be able to answer four questions about any
rejected promotion attempt:

1. Which gate rejected the artifact?
2. What was the rejection reason in plain language?
3. What needs to change in the source data / config to clear it?
4. How do I rebuild the evidence after fixing the cause?

Every gate, error variant, and rejection reason below references the
canonical types in
[`crates/forex-core/src/contracts/promotion.rs`](../../crates/forex-core/src/contracts/promotion.rs)
and [`live.rs`](../../crates/forex-core/src/contracts/live.rs).

---

## The gate at a glance

`LivePromotionGate` chains four orthogonal checks. The order matters
only for the error message you see — every check runs and the report
returns the full set, so one failed gate does not hide the others.

| Order | Check                          | Source field on report        |
|-------|--------------------------------|-------------------------------|
| 1     | Validation evidence            | `validation_evidence_complete`|
| 2     | Runtime safety                 | `runtime_safety_passed`       |
| 3     | Live execution contract        | `live_contract_passed`        |
| 4     | Determinism requirement        | `determinism_requirement_passed` |

The aggregate result is the single boolean `report.ready`. When
`ready == false`, `report.rejection_reasons` lists every plain-language
reason in the order the checks ran, and `report.checks` carries the
machine-readable `PromotionReadinessCheck { kind, status, reason }`
records for the same data.

---

## How to read the persisted profile

Every discovery cycle writes a `*_profile.json` next to the portfolio
export. Phase 49 added three fields that surface the evidence half of
the gate without instantiating it:

```json
{
  "validation_evidence_complete": false,
  "validation_evidence_missing_kinds": ["live_execution_simulation"],
  "validation_evidence_hashes": {
    "canonical_backtest": "fnv64:0123456789abcdef",
    "walkforward":       "fnv64:fedcba9876543210",
    "forward_test":      "fnv64:11223344aabbccdd",
    "prop_firm":         "fnv64:556677889900eeff",
    "live_execution_simulation": null
  }
}
```

The combination above means: the discovery cycle produced four out of
five validation kinds, and the live-execution simulation hash is
missing because the simulator has not run yet. A live bridge would
reject on this profile alone — `LivePromotionGate::validate` would
surface
`MissingValidationEvidence("live_execution_simulation_hash")`.

---

## Rejection reasons and what to do about them

### `MissingValidationEvidence("<field>_hash")`

**Source.** `ValidationEvidenceManifest::validate` on the manifest
the gate received.

**Plain language.** Discovery never produced an artifact of the named
kind, so the typed manifest cannot be validated.

**Common causes per kind.**

- `canonical_backtest_validation_hash` — discovery returned an empty
  portfolio, or `compute_discovery_validation_artifacts` was called
  before the portfolio was selected.
- `walkforward_validation_hash` — same as above, plus the walk-forward
  validator produced zero splits because `walkforward_splits` was set
  to 0 or `embargo_minutes` exceeded the in-sample window.
- `forward_test_validation_hash` — the forex-app discovery service
  did not slice an OOS tail, or the tail had fewer rows than the
  feature columns; check the `wfv_bound` log line.
- `prop_firm_risk_validation_hash` — the same OOS tail produced no
  trades for any portfolio strategy (e.g. all kill-zone-blocked) or
  the rule set rejected every fold.
- `live_execution_simulation_hash` — **always missing today**: the
  live-execution simulator is the last piece of the validation chain
  and has not been wired yet. This rejection is structural, not a bug
  in the discovery cycle.

**How to rebuild.** Re-run discovery on a window large enough to
produce all four producer-side kinds. The fifth (live-sim) requires
the simulator to land in a follow-on phase — until then, operators
must explicitly disable `require_live_execution_simulation` on the
gate they use, or accept that the gate will reject every portfolio.

---

### `LiveRejectedRuntimeMode { mode, backend }`

**Source.** `ArtifactProvenance::runtime_safety_report` (Phase 42),
surfaced through the runtime-safety check on the readiness report.

**Plain language.** The artifact's `runtime_mode` is not `Canonical`,
or its `backend_kind` does not match the assignment recorded in the
provenance. Both indicate the artifact was produced under a degraded
or fallback execution path that the live bridge will not accept.

**Common causes.**

- The discovery cycle ran on CPU because CUDA was requested but
  unavailable (the binary lacks the `gpu` feature, or no GPU device
  matched the configured backend).
- A backend adapter fell back to a surrogate / approximate path and
  the artifact carries `RuntimeMode::Degraded` with a populated
  `runtime_degraded_reason`.

**How to rebuild.** Make the originally-requested execution path
available (rebuild with `--features gpu`, attach the missing GPU,
remove the surrogate fallback) and re-run discovery. Set
`FOREX_BOT_REQUIRE_GPU=1` if you want the binary to fail fast when
CUDA is unavailable, instead of silently falling back to CPU.

---

### `LiveRejectedMismatch { field, actual, expected }`

**Source.** `LiveExecutionContract::validate_provenance`. The check
runs once per hashed contract field (feature schema, timestamp policy,
feature availability policy, symbol universe, runtime config, risk
config, backend kind).

**Plain language.** The artifact was produced under a different
configuration than the live bridge is currently configured to accept.

**Common causes.**

- Feature pipeline changed since the artifact was produced (different
  `feature_schema_hash`).
- Risk model was upgraded (different `risk_config_hash`).
- Live broker configuration changed (different `runtime_config_hash`).
- Backend was switched (different `backend_kind`).

**How to rebuild.** Re-run the discovery cycle under the new
configuration. The artifact_kind and temporal scope checks are part
of the provenance match, so changing the temporal contract requires
rebuilding everything downstream.

---

### `LiveRejectedStaleArtifact { age_seconds, max_age_seconds }`

**Source.** `LiveExecutionContract::validate_provenance_at`. Only
runs when the contract was constructed with
`with_max_artifact_age_seconds`.

**Plain language.** The artifact is older than the maximum age the
live bridge accepts.

**Common causes.**

- The discovery cycle ran more than `max_artifact_age_seconds` ago.
- Wall clock skew between the producer and consumer hosts.

**How to rebuild.** Re-run discovery. If the operator deliberately
wants to relax the staleness check, they can construct the gate
without `with_max_artifact_age_seconds` (omit it entirely; defaults to
"no age limit").

---

### `LiveRejectedFailedEvidenceGate { gate }`

**Source.** `LiveExecutionContract::validate_evidence` (Phase 27).
Only fires when the gate was constructed with `with_required_*_pass()`
and the evidence record reports the gate as failed.

**Plain language.** Validation evidence was produced, but the named
gate (`walkforward`, `cpcv`, `forward_test`, `prop_firm`) reported a
fail outcome.

**Common causes per gate.**

- `walkforward` — at least one fold breached the daily-loss / trade-
  consistency / min-trading-days rule defined by `WalkforwardSummary`.
- `cpcv` — the configured `cpcv_min_phi` was not reached across the
  combinatorial test groups.
- `forward_test` — the held-out tail produced zero trades or negative
  net profit on at least one portfolio strategy.
- `prop_firm` — `compute_prop_firm_risk_summary` reported
  `all_rules_passed = false` for at least one strategy under the
  current `PropFirmRiskRules`.

**How to rebuild.** Address the failure mode that triggered the gate
— relax the rule set if the strategy is acceptable on softer rules,
filter the portfolio differently in discovery, or pick a different
OOS tail. Then re-run discovery.

---

### `LiveRejectedMissingEvidence { gate }`

**Source.** Same as the failed-gate variant. Fires when the gate is
required but the evidence record's `forward_test_passed` /
`prop_firm_passed` is `None`, or the
`required_live_sim_runtime_model_hash` was set but the evidence
carries no `live_sim_runtime_model_hash`.

**Plain language.** The gate is required for promotion, but the
evidence record never carried the data the gate needs to evaluate.

**How to rebuild.** Run the corresponding producer (forward-test on
the OOS tail, prop-firm on the OOS tail, live-execution simulator)
and persist the artifact so the evidence bridge can pick it up next
time.

---

### `PromotionRejectedDeterminism { actual }`

**Source.** `LivePromotionGate::validate_at` when
`require_deterministic = true` (the default) and the artifact's
`determinism_policy` is `BestEffort` or `NonDeterministicAllowed`.

**Plain language.** Promotion requires deterministic execution, but
the artifact was produced under a non-deterministic policy — meaning
re-running discovery on the same dataset is not guaranteed to produce
the same portfolio.

**Common causes.**

- `FOREX_BOT_SEARCH_SEED` was not set, so `build_search_rng` seeded
  from the OS RNG (Phase 26 mapping: `seed: None` →
  `DeterminismPolicy::NonDeterministicAllowed`).
- The artifact was produced before Phase 26 wired the typed policy.

**How to rebuild.** Set `FOREX_BOT_SEARCH_SEED` (or install the typed
override programmatically via
`install_genetic_search_runtime_overrides`), re-run discovery, and
re-validate the artifact. Operators that explicitly want to promote a
non-deterministic artifact can flip the gate's flag with
`require_deterministic(false)` — but the default is on for a reason
and should not be flipped silently.

---

## Walking the chain end-to-end

A typical rejected verdict surfaces multiple reasons at once. The
order in which you fix them matters:

1. **Start with the runtime-safety report.** A degraded artifact will
   never pass any other check; rebuild on the requested backend before
   touching anything else.
2. **Check the validation-evidence section next.** Missing or failed
   gates point to either a discovery configuration that should be
   re-run, or a strategy that should be filtered out of the portfolio.
3. **Then resolve the live-contract mismatches.** A schema / risk /
   runtime mismatch between producer and consumer almost always means
   the live bridge was upgraded to a new contract while older
   artifacts were still on disk; re-run discovery.
4. **Finally, the determinism requirement.** Set the seed and
   re-run. This rejection is the cheapest one to fix and almost
   always correlates with "I forgot to set
   `FOREX_BOT_SEARCH_SEED`".

The single boolean `report.ready` only flips to `true` once every
gate is green. Until then, the gate is doing its job.
