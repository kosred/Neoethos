# Follow-on phases 72-75: quick-win audit gaps

### Phase 72 - Python/PyO3 CI guardrail

Added `scripts/check_no_python_legacy.sh` and wired it into the manual CI
workflow. The guard fails if active tracked files reintroduce Python sources,
`pyproject.toml`, PyO3-named paths, Python binding directories, or a `pyo3`
Cargo dependency outside `docs/` and `vendor/`.

The stale `cache/audit/2026-03-20-file-manifest.txt` referenced by older audits
is not tracked in the current tree because `cache/` is ignored. The active
replacement is the CI guard plus the historical audit note.

### Phase 73 - `allow(dead_code)` audit

Created `dead_code_allowlist_2026-05-10.md` and removed the stale
`SessionAccum` suppression in `forex-data`. Remaining suppressions are
documented as generated protocol, broker-integration, feature-gated native
backend, UI contract, or large-module follow-up debt.

### Phase 74 - feature registry metadata

Added `forex-data::core::feature_registry`, a typed metadata layer for generated
feature columns. It covers SMC, session, regime, quantitative, and VectorTA
classic-TA outputs, including parameter metadata for lags, windows, periods,
output lines, sessions, and higher-timeframe prefixes.

### Phase 75 - registry validation surface

`FeatureFrame` now exposes `column_metadata()` and `validate_registry()` so
discovery/training callers can reject feature-pipeline drift without duplicating
feature-name rules. Registry unit tests cover explicit feature groups,
parameterized quant names, VectorTA period/line outputs, MTF prefixes, and
unknown-column rejection.
