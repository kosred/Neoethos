# Dead Code / Stale Artifacts Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Summary

The current master branch does not show active Python or active PyO3 bindings, but it does show evidence of stale historical artifacts and possible dead-code debt.

The most important finding is that `cache/audit/2026-03-20-file-manifest.txt` is not a reliable current file list. It references files that are not fetchable from current master, including old Python/PyO3-era binding files and old Python setup scripts.

## Findings

### 1. Confirmed stale Python/PyO3-era references

The old manifest references paths such as:

- `crates/forex-bindings/forex_bindings/__init__.py`
- `crates/forex-bindings/pyproject.toml`
- `scripts/prepare_ubuntu_24_04_py313.sh`

Fetching those paths from current master returned Not Found. Search results for `forex-bindings` and `pyo3` point mostly to documentation, reports, or old audit files.

Conclusion: these are stale references, not active source code in current master.

Action: archive or regenerate old cache audit files, and mark old Python/PyO3 migration docs as historical.

### 2. Root workspace does not include the old binding crate

The current root `Cargo.toml` workspace members are Rust crates only:

- `crates/forex-search`
- `crates/forex-cli`
- `crates/forex-data`
- `crates/forex-models`
- `crates/forex-core`
- `crates/forex-app`
- `crates/forex-news`

There is no active `crates/forex-bindings` member.

Action: keep CI guardrails so no Python/PyO3 binding crate is reintroduced accidentally.

### 3. Some comments still carry Python migration breadcrumbs

`crates/forex-models/src/lib.rs` still contains comments such as:

- `Base classes and utilities (derived from models/base.py)`
- `Hardware detection (derived from models/device.py)`

This is not dead code, but it is stale architecture wording in a Rust-first codebase.

Action: update comments to describe the current Rust architecture, or mark the Python reference as historical migration context.

### 4. Vendor patches are not automatically dead, but they need dependency hygiene

The root `Cargo.toml` patches:

- `lightgbm3-sys`
- `sklears-core`
- `rlkit`

These are connected to optional/default model features in `crates/forex-models/Cargo.toml`. `lightgbm.rs` also has feature-gated native LightGBM code with local fallback behavior.

Conclusion: these vendor directories should not be deleted blindly. They may be active through default or optional model features.

Action: run feature-matrix checks before pruning vendor code. If a backend is not part of the target Rust-first runtime, disable it in default features first, then remove it after CI proves it is unused.

### 5. `allow(dead_code)` appears in active source files

Search found `allow(dead_code)` in several active areas, including cTrader service files, UI theme, session features, and tree model wrappers.

This does not prove the whole file is dead. In many cases it may be suppressing warnings for feature-gated or generated structures. But every suppression should be reviewed.

Action: create a small cleanup issue/list for each `allow(dead_code)` occurrence:

- keep if generated/protocol/feature-gated and documented
- remove if the symbol is genuinely unused
- replace with narrower cfg gating if possible

### 6. Current source has more modules than the old manifest

The old manifest is not only stale because it lists removed files; it also misses current modules such as `forex-data/src/core/all_indicators.rs`, `hpc_ta.rs`, `parquet_migration.rs`, `quant_features.rs`, `regime_detection.rs`, `session_features.rs`, and `vortex_io.rs`.

Action: regenerate a fresh manifest from the current tree before making deletion decisions.

## Recommended cleanup order

1. Regenerate current file manifest from master.
2. Archive or delete stale `cache/audit` snapshots after preserving important history.
3. Update old docs/reports that still imply Python/PyO3 bindings are current.
4. Review every `allow(dead_code)` occurrence.
5. Run feature-matrix `cargo check` for default, no-default-features, tree-models, pure-rust-ml, GPU/search features, and app features.
6. Only then remove unused vendor patches or optional model backends.

## Bottom line

Your instinct is probably right: there is dead-code debt, but it is not mainly active PyO3 code in current master. The confirmed problem is stale historical artifacts plus warning suppressions and optional backend/vendor complexity. Do not delete vendor/model code blindly; first prove which features are part of the target runtime.
