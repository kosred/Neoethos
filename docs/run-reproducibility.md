# Run reproducibility: the profile names its own arithmetic

**Slice 5 of the search-correctness campaign (2026-08-08).**

If two identical-config discovery runs differ and nobody can tell why, every
experiment is unfalsifiable. This document records the contract that closes
that hole and how to exercise it.

## What makes a discovery run reproducible

A run is a function of four inputs:

1. **The data** — symbol/timeframe series under the data root.
2. **The config** — `DiscoveryConfig`, resolved from the single `Settings`.
3. **The ambient process state** — every process-wide knob installed at
   startup (RNG seed, GA selection policy, cost/SMC overrides, backtest
   arithmetic, thread counts, adaptive-stop switch, GPU lane + precision +
   budgets) plus two pieces of **cross-run state**: the seen-signature file
   and the discovery ledger, which deliberately make a re-run search
   differently unless cleared or pinned.
4. **Mechanisms coupled to wall clock** — the GA convergence early-stop fires
   only after `convergence_min_elapsed_fraction` of the per-combo time
   budget, so a time-budgeted run can legitimately stop at a different
   generation on a slower machine. Pin `generations` low and leave
   `max_hours` at 0 when you need bit-level reproduction.

## The contract

`DiscoveryRunProfile` (written as `<out>.profile.json` by every discovery
path — CLI, app, orchestrator) must record **every setting that can change
what the search selects**. Enforcement is two-sided:

- **Compiler**: `build_discovery_profile`
  (`crates/neoethos-search/src/discovery.rs`) destructures `DiscoveryConfig`,
  `FilteringConfig`, and `DiscoveryRuntimeOverrides` **without `..`**. Adding
  a config field without deciding where the profile records it is a compile
  error.
- **Test**: `every_env_knob_is_classified_and_recorded_in_the_run_profile`
  (`crates/neoethos-search/src/discovery_tests.rs`) scans the crate sources
  for env-var names (all `NEOETHOS_*` plus `RAYON_NUM_THREADS` /
  `FOREX_TRAIN_PRECISION`) and fails unless each name is either mapped to a
  verified JSON pointer in the serialized profile or explicitly declared
  diagnostic-only with a justification. 102 names are classified as of
  2026-08-08; `NEOETHOS_GPU_TIMING` and `NEOETHOS_BOT_SEARCH_VRAM_LOG` are
  the only diagnostic-only exemptions.

The ambient state itself is captured by
`ExecutionEnvironmentProfile::capture()`
(`crates/neoethos-search/src/execution_profile.rs`) **through the same
accessors the engine reads** — the profile cannot disagree with the engine.
Lazily-memoised GPU decisions (fused eval, memory budgets) are *peeked*, not
forced: `null` in the profile means "never consulted in this process", and
capture never launches GPU work.

## Reading a profile diff

Two runs disagreed? Diff the profiles first:

```sh
diff <(jq -S . a.profile.json) <(jq -S . b.profile.json)
```

- Any difference under `.execution` → the runs did not share an environment;
  the differing line is the cause (seed, lane, precision, threads, budgets,
  cost overrides, seen-memory, …).
- `.execution` identical but results differ → an **unrecorded** source of
  nondeterminism exists. That is a census failure: find the knob, add it to
  `ExecutionEnvironmentProfile`, extend the census table.

## The two-identical-runs proof

`scripts/two_identical_runs_proof.sh` runs the same discovery twice in
isolated scratch working directories and compares every artifact
(jq-normalized sha256). It refuses to run unless the profile proves the setup
is deterministic-by-construction:

- `determinism_policy` must be `{"mode":"deterministic","seed":N}` — set
  `models.search_runtime.seed` in the **canonical** user config (the CLI
  installs runtime overrides once at startup from `Settings::load()`; a
  `--config` flag on the subcommand does not reach those installers).
- No persistent seen-signature file; discovery ledger either disabled or on a
  relative cache dir (per-run scratch cwd isolates it).

Exit 0 = reproducible, 1 = divergent (prints the execution-section diff),
2 = preconditions not met.
