# Search-to-Live Bridge Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: relationship between discovery output, model genetic artifacts, runtime prediction, trading UI/services, and live execution readiness.

## Summary

Current master shows a strong separation between discovery/search exports and broker/trading execution services.

That separation is safe in the short term because discovered strategies do not appear to be automatically routed into live orders. But it also means the production bridge is not yet a complete, validated pipeline.

A safe production pipeline should look like:

1. discovery/search creates candidate strategies
2. canonical validation accepts/rejects them
3. portfolio artifact is exported with a strict contract
4. runtime loader validates feature schema and artifact version
5. signal engine recreates signals from the exact artifact contract
6. risk/execution gate checks account/session/prop rules
7. broker execution places orders
8. realized live results are written back for forward-test monitoring

The current repo appears to have several pieces of this, but not one complete enforced chain.

## Findings

### 1. Discovery portfolio exports do not appear to be live-loadable artifacts yet

Search did not find a clear `load_portfolio_json` or importer for discovery `GeneExport` artifacts.

The discovery side saves portfolio JSON, profile JSON, quality JSON, and trade logs, but these exports do not appear to be consumed directly by the trading service.

**Risk:** discovery portfolios are useful for inspection but cannot yet be trusted as deployable runtime artifacts.

**Severity:** High.

**Fix direction:** create a strict `PortfolioArtifact` loader with schema validation and round-trip tests.

---

### 2. Trading service appears broker/execution focused, not strategy-signal focused

`crates/forex-app/src/app_services/trading.rs` is primarily concerned with broker auth, cTrader account/runtime, market charting, bootstrap, execution surfaces, and account state.

The visible service layer does not show a direct link from discovery portfolio JSON into live signal generation and order placement.

**Risk:** live execution and discovery results are not yet connected by a verified contract.

**Severity:** Medium-High.

**Fix direction:** introduce a dedicated `strategy_runtime` or `portfolio_runtime` layer between discovery and trading. Trading should not read raw discovery exports directly.

---

### 3. There are at least two portfolio-like artifact concepts

`forex-search` discovery exports portfolio JSONs through `save_portfolio_json`.

`forex-models/src/genetic.rs` uses a model artifact named `genetic_portfolio.json` plus `metadata.json` inside the `GeneticStrategyExpert` runtime model path.

These are not obviously the same contract.

**Risk:** the word “portfolio” can mean different things depending on whether it came from discovery search or model genetic runtime training.

**Severity:** High.

**Fix direction:** rename and separate artifact types:

- `DiscoveryPortfolioArtifact`
- `ModelGeneticArtifact`
- `LiveStrategyRuntimeArtifact`

Then explicitly define which one can be used for live trading.

---

### 4. Model artifact metadata is stricter than discovery portfolio metadata

`forex-models/src/runtime/artifacts.rs` defines `RuntimeArtifactMetadata` with model name, family, capability state, feature columns, label mapping, and train/validation row counts.

Discovery portfolio exports currently do not appear to have an equivalent strict artifact metadata layer.

**Risk:** the search side, which selects the actual trading strategy candidates, has weaker artifact guarantees than the model side.

**Severity:** High.

**Fix direction:** add `DiscoveryPortfolioMetadata` or `PortfolioContract` with stronger fields than the current model metadata, including schema hash, split ranges, validation results, backend, seed, and effective runtime config.

---

### 5. Search-to-live bridge should not bypass risk/execution policy

A discovered strategy is only a signal source. It should never be allowed to place orders directly.

Live execution must pass through account-aware risk checks: position size, max daily loss, total drawdown, max trades/day, session/kill-zone rules, spread/slippage guards, and broker constraints.

**Risk:** if a future bridge loads discovery exports directly into order placement, it could bypass the same rules that search/backtest assumed.

**Severity:** Critical for future live deployment.

**Fix direction:** enforce architecture: `PortfolioArtifact -> SignalEngine -> RiskGate -> ExecutionPlan -> BrokerAdapter`.

---

### 6. Forward-test tracking is not proven as part of the artifact contract

Search did not reveal a mandatory forward-test ledger tied to a specific discovery portfolio artifact.

**Risk:** a portfolio can be discovered/exported but not continuously compared against live/forward behavior under the same contract.

**Severity:** High.

**Fix direction:** each deployed portfolio must have a `forward_run_id`, immutable artifact hash, account/environment tag, paper/live mode, and realized-vs-expected tracking.

---

### 7. Runtime prediction contract does not solve discovery signal contract

`RuntimePrediction` and `RuntimeArtifactMetadata` are useful for supervised model outputs. Discovery genes, however, create rule-based signals using feature indices, thresholds, SMC flags, SL/TP, and execution assumptions.

**Risk:** treating discovery genes as if they were normal model predictions can hide important execution assumptions.

**Severity:** Medium-High.

**Fix direction:** define a separate `StrategySignalContract` for gene-based strategies.

## Recommended implementation order

1. Create `DiscoveryPortfolioArtifact` with strict serde load/save.
2. Add artifact version and schema hash.
3. Include effective feature names and original-to-effective mapping.
4. Include search seed, entrypoint, git commit, dataset id, split ranges, and validation results.
5. Add `StrategySignalEngine` that consumes only validated portfolio artifacts.
6. Add `RiskGate` integration before any broker execution.
7. Add paper-forward ledger keyed by artifact hash.
8. Add round-trip test: discovery -> artifact -> load -> signal -> backtest metrics match.
9. Add live bridge test with stub broker proving risk gate cannot be bypassed.
10. Only then label any discovered portfolio as live-ready.

## Bottom line

The current separation between discovery and live trading is safer than an accidental auto-trader, but it also means the production bridge is incomplete. The next architecture milestone should be a strict `DiscoveryPortfolioArtifact` plus a validated signal-runtime layer before any connection to live execution.
