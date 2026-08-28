# Historical Research CLI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy CLI search path with an exact-receipt, deterministic CpuOnly gross-R candidate scan.

**Architecture:** A focused `neoethos_search::historical_search_cli` adapter owns strict arguments, exact-generation loading, deterministic candidate construction, causal distance generation, and atomic artifact output. A model-free bin target in the same package is the runnable entrypoint; legacy `main.rs` retains only dispatch/help wiring to that same adapter. The existing `neoethos_search::historical_research` module remains the evaluator and identity authority.

**Tech Stack:** Rust 2024 workspace, serde/serde_json, sha2, rand 0.9 `StdRng`, canonical Vortex dataset contracts, `neoethos-search` historical research API.

---

## Chunk 1: RED boundary and domain corrections

### Task 1: Write and observe the RED CLI contract

**Files:**
- Create: `crates/neoethos-search/tests/historical_search_cli_contract.rs`

- [ ] Add source assertions that `search` dispatches to `historical_search::run`, contains no `evolve_search`/`net_profit`, and no longer advertises the retired search flags.
- [ ] Add a source assertion that `search` bypasses every `neoethos_models` and generic search-runtime installer before dispatch.
- [ ] Add behavior assertions for required receipt/seed/non-zero candidate count and an exact canonical fixture producing the requested artifact classifications/identities.
- [ ] Run the focused `neoethos-search` historical CLI contract only after the shared Cargo lane is released; confirm RED because the adapter/bin/behavior does not exist.

### Task 2: Preserve typed warmup/gap validity

**Files:**
- Modify: `crates/neoethos-search/src/historical_research.rs`
- Modify: `crates/neoethos-search/tests/historical_research_contract.rs`

- [ ] Add a failing test whose candidate column has warmup/gap validity but a sufficient valid segment; expect invalid rows to become flat rather than rejecting the whole candidate.
- [ ] Remove the blanket whole-column invalid-cell rejection while retaining bounds checks and rejection of non-finite cells marked Valid.
- [ ] Expose the validated candidate signal identity helper for the CLI's exact-N deduplication.
- [ ] Run the focused historical contract test and confirm GREEN.

## Chunk 2: Strict production caller

### Task 3: Implement strict arguments and exact receipt loading

**Files:**
- Create: `crates/neoethos-search/src/historical_search_cli.rs`
- Create: `crates/neoethos-search/src/bin/neoethos-historical-search.rs`
- Modify: `crates/neoethos-search/src/lib.rs`
- Modify: `crates/neoethos-cli/src/main.rs`

- [ ] Implement a deny-unknown, deny-duplicate parser for `--expected-input-receipt`, `--seed`, `--candidates`, `--max-indicators`, `--stop-multiple`, `--target-multiple`, `--out`, and `--root`.
- [ ] Decode/validate the expected receipt and derive all identities/timeframes from it.
- [ ] Reject mixed symbol/source/account scopes, non-bar-open inputs, duplicate timeframe bindings with different generation/binding, and missing anchor bindings.
- [ ] Open every `SelectedDatasetGenerationV1` exactly before feature computation; rebuild direct-timeframe features; require rebuilt receipt equality; construct `CanonicalSearchRunInputV1`.
- [ ] Replace `cmd_search`, dispatch, and help; remove retired flags and legacy call.
- [ ] Classify `search` before configuration loading; do not require `config.yaml`, and skip every config-dependent hardware/data/model/search installer for this strict lane.
- [ ] Resolve the strict search budget only from automatic detection plus the shared validated parent `--cpu-threads` assignment.

### Task 4: Generate exactly N deterministic candidates

**Files:**
- Modify: `crates/neoethos-search/src/historical_search_cli.rs`

- [ ] Domain-separate the explicit seed with the receipt identity and generator policy.
- [ ] Use `genetic::new_random_gene` with every structural/MTF probability disabled.
- [ ] Admit columns/candidate intersections with the fixed sufficient causal valid-row policy while preserving warmup/gaps.
- [ ] Deduplicate using the historical domain's candidate identity; regenerate deterministic collisions until N or return typed `SearchSpaceExhausted`.
- [ ] Test byte-identical identities across repeated runs and the exhaustion error with a deterministic colliding generator.

### Task 5: Run scan and atomically write the artifact

**Files:**
- Modify: `crates/neoethos-search/src/historical_search_cli.rs`
- Modify: `crates/neoethos-search/tests/historical_search_cli_contract.rs`

- [ ] Build the causal true-range distance series and bind its semantic ID/receipt SHA.
- [ ] Call `scan_historical_candidates_v1` with `CpuOnly` and explicit failure policy.
- [ ] Serialize the versioned generator contract plus complete scan result; create-new write, sync, rename, and refuse overwrite.
- [ ] Print receipt/search/ranking-policy identities plus ResearchOnly/NotPromotionEligible and gross-R summary only.
- [ ] Run the exact CLI and historical tests; inspect INFO, WARNING, and ERROR output.

### Task 5A: Make candidate evaluation lease-bound and parallel

**Files:**
- Modify: `crates/neoethos-search/src/historical_research.rs`
- Modify: `crates/neoethos-search/src/bin/neoethos-historical-search.rs`
- Modify: `crates/neoethos-search/tests/historical_research_contract.rs`
- Modify: `crates/neoethos-search/tests/historical_search_cli_contract.rs`

- [ ] Install `detected_request_with_parent` in the lightweight binary before any pool and require the shared adapter to use the installed broker.
- [ ] Run exact loading and feature construction inside a private `BudgetedCpuExecutor` under a full-width local lease, never global Rayon.
- [ ] Require an executor plus exact-broker lease transfer at `scan_historical_candidates_v1`; reject a foreign-broker transfer with a typed error.
- [ ] Evaluate candidates with indexed `par_iter`, collect `Vec<Result<_>>`, then apply failure policy strictly by input ordinal after the join.
- [ ] Return the scan lease before atomic output I/O while retaining every canonical frame/generation lease until the output commit completes.
- [ ] Prove worker=1 and automatic-width runs emit byte-identical complete JSON artifacts and rankings.

## Chunk 3: Cleanup and verification

### Task 6: Remove superseded active surfaces

**Files:**
- Modify: active CLI tests/scripts/docs found by `rg`

- [ ] Search for active callers/help/scripts using legacy `search --symbol/--base/--higher/--genes/--generations` and update or delete them.
- [ ] Prove there is no compatibility wrapper or legacy `evolve_search` caller in `neoethos-cli`.
- [ ] Run formatting, focused tests, `cargo check -p neoethos-cli --all-targets`, and inspect complete INFO/WARNING/ERROR output.
