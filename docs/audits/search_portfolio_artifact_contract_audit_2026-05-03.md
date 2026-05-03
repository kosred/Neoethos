# Search Portfolio Artifact / Export Contract Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: discovery portfolio JSONs, feature-name/index contract, export/load parity, portfolio artifact metadata, CI/test coverage for search artifacts.

## Summary

The current discovery pipeline can export portfolio JSONs, profiles, quality reports, and trade logs, but the exported portfolio does not yet look like a safe deployable contract.

The biggest issue found in this pass is a likely feature-index / feature-name mismatch when feature prefiltering is enabled: `run_discovery_cycle` can internally reduce the feature matrix, but callers save the portfolio with the original `features.names` they had before discovery.

This can make exported indicator names wrong even if the strategy indices were correct during search.

## Findings

### 1. Possible critical mismatch between searched feature indices and exported feature names

`run_discovery_cycle_with_progress` receives a `FeatureFrame`, then may call `prefilter_features`, replacing the feature matrix and feature name list inside the discovery function.

Genes discovered after that point use indices relative to the filtered feature matrix.

However, `save_portfolio_json` is called externally by CLI, UI, and batch orchestration with the caller's original `features.names`, not the post-prefilter feature names used inside discovery.

Observed call pattern:

- CLI `discover`: `save_portfolio_json(&out, &result.portfolio, &features.names)`
- batch orchestrator: `save_portfolio_json(&out_path, &result.portfolio, &features.names)`
- UI discovery service: `save_portfolio_json(&out_path, &result.portfolio, &features.names)`

**Risk:** exported `indicators` can point to the wrong feature names. A gene index that meant column 7 in the filtered matrix may be written as column 7 from the original matrix.

**Severity:** Critical.

**Fix direction:** `DiscoveryResult` must carry `effective_feature_names` and ideally `feature_mapping` from original columns to filtered columns. `save_portfolio_json` should use the effective names from the result, not names from the caller.

---

### 2. Discovery portfolio JSON appears to be export-only, not load-validated

Repo search did not find a clear `load_portfolio_json` or importer for discovery `GeneExport` artifacts.

This suggests the portfolio JSON is currently a report/knowledge artifact more than a validated deployable object.

**Risk:** there is no enforced round-trip test that exported strategies can be reloaded and produce the same signals/trades under the same feature schema.

**Severity:** High.

**Fix direction:** introduce a `PortfolioArtifact` struct with strict serialization/deserialization, schema hash, effective feature names, strategy list, validation results, and artifact version. Add load-time schema checks.

---

### 3. Discovery profile records config/counts but not complete artifact contract

`DiscoveryRunProfile` includes basic config fields and observed counts, but it does not appear to include:

- git commit
- entrypoint
- dataset identity
- feature schema hash
- effective feature names after prefilter
- original-to-effective feature mapping
- search seed
- runtime/env overrides
- evaluator backend per stage
- split ranges
- OOS/WFV/CPCV execution results
- candidate ranking formula
- portfolio selection/rejection reasons

**Risk:** exported portfolios are hard to reproduce and hard to audit.

**Severity:** High.

**Fix direction:** replace or extend `DiscoveryRunProfile` with `SearchRunContract` and `PortfolioContract`.

---

### 4. Model artifact metadata exists, but discovery portfolio metadata is weaker

`forex-models/src/runtime/artifacts.rs` has `RuntimeArtifactMetadata` with model name, family, state, feature columns, label mapping, and train/validation rows.

That is useful for model artifacts, but discovery portfolio artifacts need a stronger contract: feature schema hash, prefilter mapping, timestamp unit, timeframe alignment, execution settings, validation gates, and backend details.

**Risk:** model artifacts and discovery portfolios use different metadata strength. The discovery side is the one directly selecting trade strategies, so it needs at least the same level of rigor.

**Severity:** High.

**Fix direction:** add a discovery-specific artifact contract instead of relying on simple JSON export.

---

### 5. Batch discovery uses empty higher-timeframe list

`DiscoveryOrchestrator::run_batch` calls `prepare_multitimeframe_features(&ds_ready, tf, &[], None)`.

Other entrypoints can pass configured higher timeframes. Therefore batch discovery can search a different feature universe than CLI/UI discovery.

**Risk:** same symbol/timeframe/config can produce different portfolios depending on entrypoint.

**Severity:** High.

**Fix direction:** batch discovery must use the same higher-timeframe policy as CLI/UI/config. The effective higher timeframe list must be exported in the artifact contract.

---

### 6. CI does not currently cover `forex-search` or `forex-models`

Current `.github/workflows/ci.yml` checks/tests:

- `forex-app`
- `forex-core`
- `forex-news`

It does not run `cargo check` / `cargo test` for `forex-search` or `forex-models` in the current master workflow.

**Risk:** search/backtest/model regressions can land without CI catching them.

**Severity:** Critical.

**Fix direction:** CI must include at minimum:

- `cargo check -p forex-search`
- `cargo test -p forex-search`
- `cargo check -p forex-models`
- `cargo test -p forex-models`
- feature matrix for `gpu`, `pure-rust-ml`, tree backends, app defaults, and no-default-features as appropriate.

---

### 7. Missing tests for feature prefilter export parity

Search did not reveal dedicated tests proving that when `prefilter_features` is active, exported indicator names match the exact feature matrix used during search.

**Risk:** the most dangerous artifact bug can persist silently.

**Severity:** Critical.

**Fix direction:** add a test that builds a known feature matrix with unique column names, forces prefiltering, runs discovery, exports portfolio, and verifies every exported indicator name maps to the same effective column used by the gene.

---

### 8. Missing round-trip tests for portfolio artifacts

Search did not reveal tests proving:

- export portfolio
- reload portfolio
- rebuild features with same config
- validate schema hash
- regenerate signals
- reproduce original trade metrics

**Risk:** exported portfolios can drift from the in-memory discovery result.

**Severity:** High.

**Fix direction:** add artifact round-trip tests before treating discovery exports as live-ready.

## Recommended implementation order

1. Add `effective_feature_names` and `feature_mapping` to `DiscoveryResult`.
2. Change `save_portfolio_json` to take a `PortfolioArtifact`, not raw `Vec<Gene> + feature_names`.
3. Add `PortfolioContract` with artifact version, git commit, entrypoint, dataset id, feature schema hash, effective feature names, mapping, seed, backend, split ranges, and validation results.
4. Add load-time schema validation.
5. Fix batch discovery to use the same higher-timeframe policy as the canonical discovery config.
6. Add CI coverage for `forex-search` and `forex-models`.
7. Add prefilter/export parity test.
8. Add export-load-signal-trade round-trip test.
9. Add clean-env reproducibility test.
10. Only after these pass, mark exported discovery portfolios as deployable.

## Bottom line

The discovery portfolio export is currently useful for inspection, but it should not yet be treated as a robust live artifact. The highest-priority fix is the potential mismatch between prefiltered feature indices and externally supplied original feature names during export.
