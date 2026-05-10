# Operator-facing artifact safety reference

This document is a short, opinionated reference for every validation
artifact the search → portfolio → live bridge can produce, what each
kind means, when it is created, what guarantees it carries, and how a
live bridge accepts or rejects it.

It exists because the audit (P2-3) calls out that operators must be
able to answer five questions about any persisted artifact without
reading source code:

1. What is it?
2. How is it produced?
3. What hashes / provenance fields does it require?
4. Is it live-safe?
5. How is it invalidated, and how do you rebuild it?

Every artifact kind below is defined in
`crates/forex-search/src/validation.rs` and shares the same atomic IO
helpers (`write_*_atomic` + `read_*` validating loader) plus the same
`TemporalScopeHashes` boundary from `forex-core::contracts`.

---

## canonical_strategy_backtest

**What it is.** The reference outcome of running a single strategy
against a single dataset under the canonical evaluator. One file per
strategy in `<portfolio>_canonical_backtests/`.

**How it is produced.** `forex-search` runs each portfolio gene
through `fast_evaluate_strategy_core` after discovery selects it. The
file is built by `CanonicalBacktestArtifactFile::new(scope, metrics)`
where `metrics` are the canonical `BacktestMetrics` and `scope` binds
dataset / evaluation-config / strategy hashes plus the temporal scope.

**Required provenance fields.**

```text
artifact_kind = "canonical_strategy_backtest_artifact"
artifact_schema_version = 1
scope.dataset_hash
scope.evaluation_config_hash
scope.strategy_hash
scope.temporal_scope.{temporal_contract_hash,
                     timestamp_policy_hash,
                     feature_availability_policy_hash,
                     label_policy_hash}
```

**Live-safe?** Yes for in-sample reference, but **insufficient on its
own** for live execution — a canonical backtest is the calibration
artifact, not the gate. The live bridge accepts it as evidence of
"the strategy was scored", but `LiveExecutionContract::validate_evidence`
will still reject the load unless `walkforward_passed` and `cpcv_passed`
also clear.

**Invalidation.** The artifact is invalidated whenever any of the
hashes in `scope` changes — most commonly when the dataset is
re-fetched (different `dataset_hash`), the evaluation config changes
(different `evaluation_config_hash`), or the temporal contract is
upgraded (different `temporal_scope.temporal_contract_hash`). The
load-time validator rejects automatically; rebuild by re-running
discovery on the same dataset under the new contract.

---

## walkforward_validation

**What it is.** The folded distribution of out-of-sample backtest
metrics produced by the embargoed walk-forward validator. One file per
strategy in `<portfolio>_walkforward_validations/`.

**How it is produced.** During discovery the search engine calls
`embargoed_walkforward_backtest` with the configured `walkforward_splits`
and `embargo_minutes`; the result is folded into a `WalkforwardSummary`
and packaged with the per-strategy scope.

**Required provenance fields.** Same as canonical_strategy_backtest
above (the scope binds dataset / evaluation-config / strategy /
temporal scope), plus the per-split details inside `summary.splits`.

**Live-safe?** Yes when `summary.any_daily_loss_breach` is `false`
**and** `summary.any_consistency_violation` is `false` **and**
`summary.all_min_trading_days_ok` is `true`. The discovery
`walkforward_passed` gate aggregates these into a single boolean that
flows through the typed evidence record.

**Invalidation.** Same hash-driven invalidation as the canonical
backtest. Rebuild by re-running discovery; walk-forward validation
runs automatically as part of every discovery cycle that has at least
one portfolio strategy.

---

## forward_test_validation

**What it is.** A single backtest pass over data that was **withheld
from both training and walk-forward CV** — the unbiased OOS estimate.
One file per strategy in `<portfolio>_forward_tests/`.

**How it is produced.** The forex-app discovery service slices the
final 20% of the dataset (the rows past `wfv_bound`) and feeds it to
`compute_discovery_forward_test_artifacts`. Internally each strategy's
signals are rebuilt with `signals_for_gene_full` against the
column-projected tail and scored through `compute_forward_test_summary`.

**Required provenance fields.** Same shape as the canonical /
walk-forward scope, but `scope.dataset_hash` binds **the tail dataset**
(not the full discovery dataset) so the artifact cannot be confused
with a canonical backtest produced from in-sample data.

**Live-safe?** Yes when `summary.metrics.trade_count > 0` **and**
`summary.metrics.net_profit > 0.0`. The
`live_validation_evidence_from_discovery` helper aggregates these into
the typed `LiveValidationEvidence::forward_test_passed` field;
`LiveExecutionContract::validate_evidence` rejects when the gate is
required but evidence is missing or fails.

**Invalidation.** Same hash-driven invalidation. Note: a forward-test
artifact is **not** equivalent to a canonical backtest artifact — the
live bridge enforces this through `artifact_kind` so a forward-test
file cannot be loaded as a canonical backtest or vice versa.

---

## live_execution_simulation

**What it is.** Canonical metrics under live-realistic execution
assumptions (slippage, latency, partial fills, kill-zone gating), plus
the `LiveExecutionRuntimeModel` that produced them. One file per
strategy in `<portfolio>_live_execution_simulations/` once a simulator
is wired in.

**How it is produced.** The contract surface
(`LiveExecutionSimulationArtifactFile`,
`LiveExecutionRuntimeModel`, atomic IO helpers) is in place. The
production simulator that builds the summary is **deferred** — Phase 25
added the typed contract, but a canonical CTrader-equivalent simulator
that produces the actual `bars_simulated` / `trades_blocked_by_kill_zone`
counters is a separate work item.

**Required provenance fields.**

```text
artifact_kind = "live_execution_simulation_artifact"
artifact_schema_version = 1
scope.runtime_model_hash    (mandatory — must match the live bridge's
                             current runtime configuration)
scope.{dataset_hash, evaluation_config_hash, strategy_hash,
       temporal_scope}
summary.runtime_model       (recorded in full so an operator can
                             diff slippage / latency / spread
                             assumptions without re-running the sim)
```

**Live-safe?** Yes when `LiveExecutionContract` has
`required_live_sim_runtime_model_hash` set and the persisted
`scope.runtime_model_hash` matches. A mismatch surfaces as
`LiveRejectedMismatch { field: "live_sim_runtime_model_hash" }` at
load time — the live bridge will not accept simulation evidence
produced under a different broker / latency profile than the one
configured for live execution.

**Invalidation.** Whenever the live runtime model changes (e.g.
different broker latency, new spread profile, kill-zone re-tuned).
Rebuild by running the simulator with the updated
`LiveExecutionRuntimeModel`.

---

## prop_firm_risk_validation

**What it is.** A pass/fail summary of a strategy's observed trades
against a typed `PropFirmRiskRules` set (max daily loss, overall
drawdown, profit consistency, min trading days, max trades per day,
optional profit target). One file per strategy in
`<portfolio>_prop_firm_validations/`.

**How it is produced.** `compute_discovery_prop_firm_artifacts` runs
`signals_for_gene_full` + `simulate_trades_core` on the same OOS tail
the forward-test uses, then feeds the trades to
`compute_prop_firm_risk_summary`. The forex-app discovery service
calls this automatically and reads the rule set from
`DiscoveryRequest::prop_firm_rules` (defaults to FTMO-style baseline
when the caller does not override).

**Required provenance fields.**

```text
artifact_kind = "prop_firm_risk_validation_artifact"
artifact_schema_version = 1
scope.rules_hash            (binds the artifact to a specific
                             challenge rule set — different challenges
                             produce different files)
scope.{dataset_hash, evaluation_config_hash, strategy_hash,
       temporal_scope}
summary.rules               (recorded in full so an operator can
                             see thresholds without dereferencing
                             the hash)
summary.all_rules_passed    (the single boolean the live bridge
                             aggregates per portfolio)
summary.{daily_loss_breach, overall_drawdown_breach,
         consistency_violation, trade_limit_violation,
         min_trading_days_ok, profit_target_met}
```

**Live-safe?** Yes when `summary.all_rules_passed` is `true` for
every portfolio strategy. The
`live_validation_evidence_from_discovery` helper aggregates this into
`LiveValidationEvidence::prop_firm_passed` — the live bridge can then
require it as a gate via `with_required_prop_firm_pass()`.

**Invalidation.** Whenever the `PropFirmRiskRules` change (different
challenge), the dataset changes, or the temporal contract is upgraded.
Switching from FTMO defaults to a different challenge produces a new
`scope.rules_hash`; both files can coexist on disk.

---

## How the live bridge consumes them

The five artifact kinds above feed `LiveExecutionContract` through one
typed boundary:

```rust
let evidence = live_validation_evidence_from_discovery(&result);
contract.validate_provenance(&envelope.provenance)?;
contract.validate_evidence(&evidence)?;
```

Each `with_required_*_pass()` builder on the contract turns a gate on.
The defaults leave every gate disabled so upgrading a contract
instance never silently tightens the bar — the operator decides which
gates are required for live execution and configures the contract
accordingly.

When a gate is required but evidence is missing, the bridge raises
`LiveRejectedMissingEvidence`; when evidence is present but failed,
it raises `LiveRejectedFailedEvidenceGate`. Both errors carry the
gate name as a `&'static str` so log lines surface the exact rule
that rejected the artifact.

---

## Quick rebuild matrix

| Trigger                                  | Affected artifacts |
|------------------------------------------|--------------------|
| Dataset re-fetched (new bars)            | All five           |
| Evaluation config changed                | All five           |
| Temporal contract upgraded               | All five           |
| Walk-forward splits / embargo changed    | walkforward_validation only |
| OOS tail boundary changed (`wfv_bound`)  | forward_test, prop_firm |
| `PropFirmRiskRules` overridden           | prop_firm_risk_validation only |
| Live `LiveExecutionRuntimeModel` updated | live_execution_simulation only |
| Strategy gene mutated                    | All five (per affected strategy) |

When in doubt: re-run discovery. Every artifact except
`live_execution_simulation` is produced as a side effect of a normal
discovery cycle, so a fresh run rebuilds the full set for the
selected portfolio.
