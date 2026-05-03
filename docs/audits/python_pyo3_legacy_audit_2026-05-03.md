# Python / PyO3 Legacy Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Summary

The current master branch appears Rust-first. I did not find active Python source, active Python packaging, or active PyO3 bindings in the current workspace.

Important distinction: old audit artifacts still mention `crates/forex-bindings`, `forex_bindings/__init__.py`, and `pyproject.toml`. Those paths look historical/stale because they are listed in an old manifest but are not fetchable from current master.

## Checked patterns

Searches performed included:

- Python source and packaging: `.py`, `requirements.txt`, `pyproject.toml`, `setup.py`, `Pipfile`, `poetry.lock`
- Python runtime calls: `python`, `python3`, `Command::new`, subprocess-style patterns
- PyO3 patterns: `pyo3`, `PyO3`, `PyResult`, `Python<'py>`, `#[pymodule]`, `#[pyfunction]`, `pymethods`, `pyclass`, `maturin`, `cdylib`, `python-extension`, `extension-module`
- Workspace membership and dependencies in root `Cargo.toml`
- Old audit manifest references under `cache/audit/2026-03-20-file-manifest.txt`

## Findings

### 1. No active Python implementation found

No current active Python source or packaging file was found by repo search. The root workspace `Cargo.toml` lists only Rust workspace members and does not include `crates/forex-bindings`.

### 2. No active PyO3 binding crate found in workspace

The current root `Cargo.toml` workspace members are:

- `crates/forex-search`
- `crates/forex-cli`
- `crates/forex-data`
- `crates/forex-models`
- `crates/forex-core`
- `crates/forex-app`
- `crates/forex-news`

There is no active `crates/forex-bindings` workspace member in the current root workspace file.

### 3. Old manifest shows removed Python/PyO3-era artifacts

The historical manifest `cache/audit/2026-03-20-file-manifest.txt` lists old files including:

- `crates/forex-bindings/forex_bindings/__init__.py`
- `crates/forex-bindings/pyproject.toml`
- several Rust files under `crates/forex-bindings/src/`

But fetching `crates/forex-bindings/pyproject.toml` from current master returned Not Found. Treat the old manifest as stale historical evidence.

### 4. Search hits for PyO3 are mostly docs or stale references

Exact searches for `pyo3`, `pyo3 =`, `PyO3`, `#[pymodule]`, and related patterns did not reveal an active binding crate in current master. Some hits point to old reports/plans or broad search matches, not an active workspace dependency.

### 5. Env vars are Rust runtime/config debt, not necessarily Python legacy

Search for `std::env::var` found active Rust files. This should be audited as Rust configuration hygiene, not automatically classified as Python leftovers.

## Recommended cleanup

1. Regenerate a fresh repo manifest from current master.
2. Archive or clearly label `cache/audit/2026-03-20-file-manifest.txt` as historical.
3. Add CI guardrails to fail if active `.py`, Python packaging, or PyO3 binding files reappear outside an explicit allowlist.
4. Search and update docs that still describe PyO3/Python bindings as current architecture.
5. Run a separate Rust env-var/config audit.

## Bottom line

Your suspicion was reasonable because the old manifest clearly shows a previous Python/PyO3 binding layer. Current master, however, does not appear to contain that layer as active code. The cleanup target is now stale documentation/artifacts plus Rust env/config hygiene, not mass deletion of live Python code.
