# Stabilization Backlog

## Build Blockers

- `cargo test --workspace`
  - fix the GPU distribution contract failure in [`crates/forex-models/src/hardware.rs`](C:/Users/konst/development/forex-ai/crates/forex-models/src/hardware.rs)
  - expected outcome: workspace tests are no longer red on the baseline Windows lane

- `cargo clippy --workspace --all-targets -- -D warnings`
  - clear the current lint baseline starting with runtime-critical crates:
    - [`crates/forex-core`](C:/Users/konst/development/forex-ai/crates/forex-core)
    - [`crates/forex-data`](C:/Users/konst/development/forex-ai/crates/forex-data)
    - [`crates/forex-search`](C:/Users/konst/development/forex-ai/crates/forex-search)
    - [`crates/forex-models`](C:/Users/konst/development/forex-ai/crates/forex-models)
    - [`crates/forex-bindings`](C:/Users/konst/development/forex-ai/crates/forex-bindings)
    - [`crates/forex-app`](C:/Users/konst/development/forex-ai/crates/forex-app)
  - expected outcome: no enforced-clippy warnings on the supported build lane

## Runtime Blockers

- `Tranche 1: Remove false-success training`
  - replace the placeholder training closure in [`crates/forex-models/src/training_orchestrator.rs:48`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L48)
  - drive enabled-model selection from settings instead of the hardcoded list at [`crates/forex-models/src/training_orchestrator.rs:62`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L62)
  - stop [`crates/forex-models/src/parallel_trainer.rs:111`](C:/Users/konst/development/forex-ai/crates/forex-models/src/parallel_trainer.rs#L111) from returning `Ok(...)` on partial or total training failure
  - rationale: this is the highest-risk correctness gap because the current CLI runtime can report successful Rust training without real model work

- `Tranche 2: Make data/discovery failures explicit`
  - stop silent timeframe omission in [`crates/forex-data/src/lib.rs:93`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L93) and [`crates/forex-data/src/lib.rs:103`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L103)
  - remove synthetic zero-timestamp fallback at [`crates/forex-data/src/lib.rs:116`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L116)
  - convert degraded-success discovery in [`crates/forex-search/src/orchestration.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-search/src/orchestration.rs#L22) and [`crates/forex-cli/src/main.rs:282`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs#L282) into explicit result states
  - rationale: the current search/data path can look successful while operating on incomplete inputs or empty outputs

- `Tranche 3: Repair MT5 initialization error surfacing`
  - fix tuple extraction for `last_error()` in [`crates/mt5-bridge/src/lib.rs:29`](C:/Users/konst/development/forex-ai/crates/mt5-bridge/src/lib.rs#L29)
  - expected outcome: broker initialization failures expose the real error code and description instead of a bridge-level type mismatch

## Warning Cleanup

- remove unused imports and dead fields from the baseline warning clusters already recorded in the report
- pay special attention to `runtime-critical` crates first, and `crates/forex-news` second because it is currently warning-heavy despite small size
- expected outcome: warning inventory is empty or explicitly justified on supported lanes

## Contract Repairs

- `Bindings surface cleanup`
  - either implement or remove the public stubs in [`crates/forex-bindings/src/data.rs`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/data.rs)
  - stop exporting placeholder analytics in [`crates/forex-bindings/src/evaluation.rs:532`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/evaluation.rs#L532)
  - stop swallowing gene-serialization failures in [`crates/forex-bindings/src/search.rs:51`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/search.rs#L51)
  - honor Python-supplied validation settings in [`crates/forex-bindings/src/validation.rs:42`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/validation.rs#L42)

- `CLI surface cleanup`
  - pass `--root` through the training command or remove it from [`crates/forex-cli/src/main.rs:147`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs#L147)
  - align printed help with the actual parser at [`crates/forex-cli/src/main.rs:370`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs#L370)

- `Search/runtime contract cleanup`
  - honor configured candidate limits in [`crates/forex-search/src/discovery.rs:80`](C:/Users/konst/development/forex-ai/crates/forex-search/src/discovery.rs#L80)
  - honor the configured walk-forward compliance threshold in [`crates/forex-search/src/validation.rs:111`](C:/Users/konst/development/forex-ai/crates/forex-search/src/validation.rs#L111)
  - either compute or remove placeholder walk-forward summary fields in [`crates/forex-search/src/validation.rs:102`](C:/Users/konst/development/forex-ai/crates/forex-search/src/validation.rs#L102)
  - honor or remove `lookback_days` in [`crates/forex-search/src/portfolio.rs:24`](C:/Users/konst/development/forex-ai/crates/forex-search/src/portfolio.rs#L24)

## Observability And Recovery

- keep the non-blocking log flush guard alive from [`crates/forex-core/src/logging.rs:29`](C:/Users/konst/development/forex-ai/crates/forex-core/src/logging.rs#L29)
- unify app startup logging with the shared contract instead of bypassing it in [`crates/forex-app/src/main.rs:43`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L43)
- make ONNX model loading explicit about empty/missing model directories in [`crates/forex-models/src/lib.rs:89`](C:/Users/konst/development/forex-ai/crates/forex-models/src/lib.rs#L89)
- stop collapsing provider-unavailable states into normal results in:
  - [`crates/forex-news/src/openai.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-news/src/openai.rs#L22)
  - [`crates/forex-news/src/perplexity.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-news/src/perplexity.rs#L22)
- expected outcome: operators can distinguish success, degraded success, and failure without reading source code

## Structural Cleanup Candidates

- remove or hide placeholder public bindings until their implementations exist:
  - [`crates/forex-bindings/src/models.rs`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/models.rs)
  - [`crates/forex-bindings/src/data.rs`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/data.rs)
  - [`crates/forex-bindings/src/evaluation.rs`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/evaluation.rs)
- centralize hardware-thread budgeting so Rayon, ONNX, and any future inference path use one source of truth
- replace stale documentation/comments that describe behavior the code no longer implements
- after correctness fixes, reassess whether `crates/forex-app` should call backend services directly or through a dedicated application-service layer

## UI Integration Entry Criteria

- `forex-app` discovery must call the real backend instead of the mock loop at [`crates/forex-app/src/main.rs:298`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L298)
- `forex-app` training must call the real backend instead of the log-only action at [`crates/forex-app/src/main.rs:327`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L327)
- app startup must use the shared logging contract
- no critical runtime findings may remain open in training, discovery, or MT5 initialization
- the MT5 environment may still be blocked by local authorization, but the bridge itself must surface that state explicitly and correctly
