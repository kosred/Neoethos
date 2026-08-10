# Pending edits: files another workflow was writing

Written 2026-08-09 by the indicator-vocabulary restoration. Everything below is
a fix that belongs to that work but lands in a file this workflow was told not
to touch (`crates/neoethos-search/src/discovery.rs`,
`crates/neoethos-search/src/validation.rs`). The supporting code these edits
call is already merged and tested in `crates/neoethos-data`.

Apply in order. Each entry states the exact site, the exact replacement, and
what changes behaviourally — none of these are cosmetic.

## Status, re-verified 2026-08-09 against the working tree

| # | State | Evidence |
|---|-------|----------|
| 1 | **APPLIED** by another workflow — do not re-apply | `discovery.rs:5462` and `:5477` call `neoethos_data::core::stats_f64::pearson_pairwise_f32`, over walk-forward windows, for BOTH the long and short labels, with `is_rankable()` gating and a `had_nonfinite` census flag. The old `fn pearson_correlation` is gone; `discovery.rs:4955` is the comment recording the replacement. |
| 2 | **APPLIED** — `discovery.rs:106` reads `prefilter_top_k: 240`. All four sites (code default, core config default, root `config.yaml:563`, `desktop/src-tauri/resources/config.yaml:141`) now agree. |
| 3 | **STILL PENDING** — no `SplitSkip` or `skipped_splits` anywhere in `validation.rs`. |
| 4 | **STILL PENDING** — the day-index substitution guard is not present. |
| 5 | **STILL PENDING** — `validate_registry()` still has zero callers repo-wide. |

Items 1 and 2 are left in this document as the record of what changed and why,
not as work to do.

---

# Wave 2 — the config consolidation, 2026-08-10

Appended by shard B, merging the handoffs of shards A, C, D and E. Every row is
an edit one of those shards **needed and did not make**, because the file
belongs to the app-side wave or the f64 kernel conversion that were running at
the same time. 💰 = money path.

Forbidden this wave: `crates/neoethos-app/**`, `crates/neoethos-core/src/domain/**`,
`crates/neoethos-cli/**`, `crates/neoethos-models/**`, `crates/neoethos-trader/**`,
`desktop/src/**`, `desktop/src-tauri/src/**`, `vendor/**`.

## Status table

| # | State | File / site | Edit required | Blocks |
|---|-------|-------------|---------------|--------|
| W2-1 💰 | **PENDING — blocks a deletion** | `crates/neoethos-app/src/app_services/live_trading.rs:1226-1272` | Make live trailing read `models.exit_policy.{trailing_enabled, trailing_stop_multiplier, trailing_be_trigger_r, trailing_min_lock_pips}`. Today it trails **unconditionally, with no config recipient of any kind**, under a comment claiming parity with a backtest that now ships trailing OFF. | **Blocks D5** — A deleting `risk.trailing_*` (`config.rs:417-428`). Wire live FIRST. Deleting first converts a visibly-wrong value into an invisible hardcode on the path that spends real money. |
| W2-2 💰 | **PENDING — blocks a deletion** | `crates/neoethos-app/src/server/risk.rs:166`, `:244`; `RiskDto.prop_firm_rules_enabled` (`risk.rs:44`); `desktop/src/…/Risk.tsx:53`, `RiskyMode.tsx:100` | Derive the "currently active rules" display from `system.trading_mode`; drop `risk.prop_firm_rules` from the DTO. One write, one display read, **zero decisions** — every discovery call passes a hardcoded `PropFirmRiskRules::default()`. The card can announce "Prop-firm" while `system.trading_mode` is `risky`. | **Blocks D6** — A deleting `risk.prop_firm_rules`, otherwise the DTO stops compiling. B has already removed the key from the repo `config.yaml`. |
| W2-3 💰 | **PENDING — the writer that regenerates the divergence** | `crates/neoethos-app/src/server/risk.rs` (preset writer) | It persists the prop firm's **RAW** ceiling instead of the **BUFFERED** one. That is the origin of `daily_drawdown_limit: 0.10000000149011612` and `total_drawdown_limit: 0.20000000298023224` — f32 values widened to f64, i.e. machine-written. A UI preset click rewrites them, so B's correction in `config.yaml` **will come back** until this lands. | Nothing, but the fix is one-shot without it. |
| W2-4 | **PENDING** | `crates/neoethos-app/src/server/settings.rs:256-269` | The raw-YAML editor endpoint must reject unknown keys instead of reporting "saved (verbatim)". That editor is the only route to **364 of the 390 knobs**, and `trailing_enabeld:` currently parses, saves and reports success. | Pairs with A's `deny_unknown_fields`; without it the endpoint is a hole straight past the struct-level guard. |
| W2-5 | **PENDING — ORDERING CONSTRAINT** | `%LOCALAPPDATA%\neoethos\config.yaml` | The four tombstones (`export_onnx`, `news_kill_window_min`, `news_lookahead_minutes`, `perplexity_enabled`) become a **hard startup failure** the moment `deny_unknown_fields` lands. | **A's `deny_unknown_fields` must not ship before `scripts/migrate_live_config.ps1` has been run and its DROP section accepted.** |
| W2-6 💰 | **PENDING** | `crates/neoethos-app/src/server/knob_catalog.rs` — `month_capacity` entry (~`:884-894`) | Remove `min: Some(12)` or re-label. It advertises a minimum of 12 while describing the knob as a RAM cap. `month_capacity` sizes `monthly_pnls` → metric slot 7 → `named.rs:161` × **0.45**, the dominant term of the prop-firm objective. Setting the UI-endorsed minimum scores every gene on its **first twelve months** of a ten-year record and returns a plausible number. | — |
| W2-7 | **PENDING** | `crates/neoethos-app/src/server/knob_catalog.rs` — 34 `env_var:` entries | Set to `None` in the same wave as D's deletion of the five `install_*_from_env` wrappers. The catalog is served to the LLM control plane at `mcp/server.rs:716-725` **as authoritative**, so a stale `env_var:` tells the control plane to set a variable nothing reads. | — |
| W2-8 | **PENDING** | `crates/neoethos-cli/src/tui/config_view.rs:50-54`, `:152-162` | The "Discovery mode" list must offer **`strict` and `legacy` only**, and stop rejecting `legacy`. It currently offers `prop_firm`/`strict`/`risky` and rejects `legacy` — two values the engine maps to nothing, and a refusal of one of the two it honours. | Cosmetic once E's fall-through WARN lands; still actively misleading. |
| W2-9 | **PENDING** | `crates/neoethos-cli/src/…/resolved_config.rs:329`, `:332`, `:337-350` | `:329` prints raw `models.discovery_mode` beside a mode resolved from `system.trading_mode`; `:332` help names the wrong field; `:337-350` still reports `normalize_features` and `disable_smc_gate` as `source=env` from variables the engine no longer reads. | This is the ONE diagnostic built to answer "which value won", and it is wrong on three rows. |
| W2-10 💰 | **PENDING — same binary, different behaviour by cwd** | `desktop/src-tauri/src/lib.rs:52`, `prepare_data_root:424-437` | Relative-path install + data-root preparation mean the same binary trades differently depending on the directory it was started from. This is the concrete harm behind removing the bare relative `"config.yaml"` fallback. | Pairs with A's single-resolution-point work. |
| W2-11 💰 | **PENDING — wire or delete, not both** | `crates/neoethos-core/src/domain/risky_mode.rs:149`, `:155`, `:160`; validator `:531-540` | `correlation_cap`, `volatility_sigma_pause`, `require_swarm_confidence_min` are declared, documented as active defences, **validated at startup**, and enforced nowhere. Their sibling on the same struct (`presend_sanity_ceiling_fraction`) **is** enforced, which is what makes the absence conclusive. E's recommendation: **delete the three validations now** (costs nothing, removes the false signal immediately) and mark the fields `⚠ UNWIRED`. Wiring three risky-mode safety gates is a money-path behaviour change that needs its own measurement. | Startup validation of a value nothing uses is the strongest possible false signal: affirmative evidence that the number was checked. |
| W2-12 💰 | **PENDING — blocks four deletions** | `crates/neoethos-models/src/training_orchestrator.rs:878-884`, `:886-892`, `:894-900`, `:2316-2321`, `:2481-2487`, `:902-911` | Six silent arithmetic merges: `transformer_hidden_dim`/`_d_model`, `transformer_heads`/`_n_heads`, `transformer_layers`/`_n_layers` (`.max()`); `label_stop_atr_multiplier`/`risk.atr_stop_multiplier` (`.max()`) 💰; `label_take_profit_rr`/`risk.min_risk_reward` (mode-gated `.max()`) 💰; the three row caps `global_max_rows`/`_per_symbol`/`max_training_rows_per_tf` (`.min()`). A silent `.max()` means **raising EITHER raises the effective value** — an operator who lowered one has not lowered the setting. The transformer trio agree at 256 today *by luck*. | **Blocks A** from deleting any of them. E confirms no caller update this wave, so A MARKS rather than deletes. |
| W2-13 | **PENDING** | `crates/neoethos-models/src/training_orchestrator.rs` — `apply_hardware_plan_params` | Drop the `if canonical_model_name(name) == "genetic"` condition from **both** match arms so every `ModelFamily::Evolutionary` model takes `WorkloadKind::StrategySearch`; delete the three `("device", settings.system.device)` entries in the `neuro_evo`/`neat` param maps; point `preferred_burn_device_policy` at the `DeepTraining` workload device or delete it. **No new `WorkloadKind` variant is needed** — C deliberately did not add one, because a variant no consumer maps to is a second dead surface. ⚠ **Behaviour change to declare:** `neuro_evo` and `neat` currently run on `system.device` (shipped `"cpu"`); afterwards they run on the planner's search device, which on a CUDA box is `"cuda:all"`. Log old and new. | **Blocks A** deleting `system.device`. C reports the `WorkloadKind` precondition **did not land and cannot** from `system.rs` — the mapping site is here. Until then `system.device` and `system.enable_gpu` stay settable, unskipped, undeleted. |
| W2-14 | **PENDING** | `crates/neoethos-models/src/training_orchestrator.rs:1900` | The `rllib_auto` branch — `system.enable_gpu`'s single reader — immediately degrades to native `rlkit` and warns. Branch and field die together. | Pairs with W2-13. |
| W2-15 | **PENDING — same change or neither** | `crates/neoethos-models/src/training_orchestrator.rs:1994-1997` | `models.train_batch_size` has 13 readers, all overwritten by `apply_hardware_plan_params` except the DQN replay-buffer sizing, which falls back to it when `rl_buffer_capacity == 0`. Deleting or `#[serde(skip)]`-ing the field without giving the replay buffer its own hardware-derived default silently shrinks or grows an RL buffer. `neoethos_core::system::training_batch_size(gpu, min_vram_gb)` is now public for exactly this. | — |
| W2-16 | **PENDING** | `crates/neoethos-models/**` | The absolute memory sizers: `swarm_memory_limit_mb` (fixed 256.0 on a 16 GB laptop and a 192 GB box alike), `exit_agent_memory_capacity`, `rl_buffer_capacity`, and the three row budgets collapsed by `.min()`. All are capacity, all should derive from the probe. `WorkloadExecutionPlan::memory_budget_gb` is already computed and already handed to these models under a different key — they ignore it. | Never-OOM invariant: peak memory is a function of available hardware, never of a user parameter. |
| W2-17 | **PENDING — its own commit, with its own measurement** | thread clamp, formerly `system.rs:1343-1350` | `apply_thread_env_defaults` was the **only** code in the workspace that set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`, and its only caller was the dead `AutoTuner::apply`. **So those variables have never been set in any run** — LightGBM's OpenMP wrapper and every XGBoost/BLAS pool have been sizing themselves unclamped, in the subsystem already identified as the thread-oversubscription bottleneck. Real opportunity, **not** a config item. Must land alone with a before/after wall-clock on a real training run, because clamping for the first time changes run times in both directions depending on the model. | Do **not** fold into a config commit claiming to be a no-op. (Note: `set_var` is the process telling a native library what to do — legitimate. The "no env vars" ban is on *reading* configuration out of the environment.) |
| W2-18 | **PENDING** | `crates/neoethos-app/src/app_services/discovery.rs:719-726` | Reads `NEOETHOS_BOT_MIN_HISTORY_YEARS` and passes it into the search as `min_years`. Replace with `settings.models.discovery_runtime.min_history_years`. E deleted the six-name env reader inside `neoethos-search`, so this is the **last surviving env path** for that knob. | Two ways to set one thing is the defect. |
| W2-19 | **PENDING — half-satisfied mandate** | `crates/neoethos-search/src/backend.rs` ↔ `crates/neoethos-core/src/system.rs` | The device decision still has **two independent resolvers**: `EvaluationBackend::from_settings` calls `resolve_with_probe(...)` with its own `HardwareProbe::detect()`, while `HardwareExecutionPlan::from_settings_profile_and_overrides` resolves the same two fields with its own policy and its own probe. Same inputs, two policies. They do not currently disagree in a demonstrable way, so nothing was changed. | "Resolve in ONE place so it cannot be bypassed" is only half true: the plan is authoritative for *training*, and the search path never consults it. |
| W2-20 | **PENDING — doc rows only, no compile break** | `crates/neoethos-core/tests/config_has_recipient.rs` | Four places still cite `AutoTuneHints` as the reason a knob has no recipient — the rows for `ModelsConfig::inference_batch_size` and `SystemConfig::n_jobs`, and two prose justifications quoting `system.rs:1251` / `:1256`. Those lines no longer exist and the type is deleted. The only in-code occurrence is inside a string literal in a parser unit test, which still exercises the parser unchanged. | Update when E rewrites the ledger. |

## Corrections to the decision documents that this wave established

These overturn statements in `docs/knob-second-pass-2026-08-09.md`. Recorded
here so the next reader does not re-derive them.

1. **§4a's "pickling" claim is one field wide, not four.** `available_parallelism()`
   appears **exactly once** in `config.rs`, in the `n_jobs` default.
   `enable_gpu: false`, `num_gpus: 0` and `device: cpu` are **static literals** in
   `SystemConfig::default()` — on the 3090 box they were never probe output. That
   is arguably a worse defect (a config asserting "no GPU" with no detector
   involved), but it means the `#[serde(skip)]` remedy is needed for `n_jobs`
   only, and deleting `n_jobs` discharges it. **Phase C item 11 is therefore not
   blocking anything.**
2. **The `#[serde(skip)]` list is empty, deliberately.** The three fields that
   warranted it get **deleted** instead (strictly stronger: gone from the struct,
   the schema and the UI). Every remaining hardware-shaped field still has a live
   reader, so `#[serde(skip)]` on it would silently substitute the `Default` for
   the operator's configured value on the next load — the exact silent change
   non-negotiable #1 forbids.
3. **§4b is overturned: `apply_thread_env_defaults` was never reachable.** See
   W2-17. The brief's instruction to keep it alive as a precondition was wrong;
   the decision document's own §4b says so.
4. **§4c undercounts `system.device` by one.** There is a fourth reader and it is
   not `neuro_evo`: `training_orchestrator.rs::preferred_burn_device_policy`
   feeds **thirteen** deep/exit model param maps. They are all overwritten by
   `apply_hardware_plan_params` afterwards, so the effect is dead — but the read
   is live and the field cannot be deleted while that function compiles.
5. **The planner's warnings had literally never been printed.** Their only logger
   was inside the dead `AutoTuner::apply`. `HardwareExecutionPlan::announce` now
   logs the probe, every workload assignment, every planner warning and every
   configured value the plan overrides.
6. **`Settings::default()` must stay publicly constructible.** `log_gate_states`
   in `discovery.rs` calls it to print each gate's Rust `Default` beside its
   effective value — the only way a run can say "this gate is off and the Default
   says on" without a fourth hardcoded copy of the defaults. A's
   compiler-enforced single load path must not gate `Default::default()`.
7. **`prop_firm_min_pass_rate` does not collapse this wave.** E does not confirm
   a caller update; the collapse needs a decision about which name survives plus
   a migration of the operator's live value. A marks both. Meanwhile the `.max()`
   at `discovery.rs:7812` logs both values, names the winner, and states that the
   safer (higher) one binds.
8. **The "five money divergences between root and seed" were never really a
   conflict — they were two files answering for two different presets, and
   nothing said so.** The repo root runs `risk.preset: none` (own money) and the
   operator's live store runs `risk.preset: ftmo`. So `0.08 / 0.14` is correct in
   the root (`NONE_OWN_MONEY.daily_dd_stop_trading_pct`, and
   `max_overall_drawdown_pct 0.20 × the 0.7 buffer`) and `0.04 / 0.07` is correct
   in the live store (FTMO's `0.040`, and `0.10 × 0.7`). **The triage note's
   proposed `0.04` for the root was wrong** — tighter, so not dangerous, but it
   would have introduced a third unexplained number and broken agreement with
   `RiskConfig::default()`, which is the exact failure being closed.

## Status of the 2026-08-09 items that landed in shard B's files

| Old item | New state |
|---|---|
| **7** — `config.yaml:158-159` drawdown breakers | **APPLIED 2026-08-10**, at the re-derived `0.08 / 0.14`, not the triage note's `0.04`. See correction 8 above. |
| **8** — desktop seed divergences (#245/#246/#248/#249/#250 + the uncounted sixth) | **DISSOLVED, not applied.** The seed is now GENERATED from `Settings::default()`, so: (a) the "sixth divergence" — `risk.preset` present in the root and absent from the seed, falling between the unique-keys test and the value-disagreement test — is structurally impossible, because a generated file contains every field; (b) `preset`, `daily_drawdown_limit` and `total_drawdown_limit` are guaranteed mutually consistent, because they all come from one `Default`; (c) the item's own warning — *"what must not happen is FTMO's preset with own-money drawdown numbers"* — cannot happen. ⚠ **This declines the loosening #246/#248 proposed.** A fresh install now ships the Rust defaults (FTMO preset, prop-firm rules armed, USD) rather than this machine's `risky` + rules-disarmed + GBP. That is the conservative direction the item itself named as the alternative, and it is recorded here as a deliberate choice, not an oversight. |
| **9** — one home for the ×0.7 drawdown buffer in `config.rs` | Still pending; `config.rs` is shard A's file. Unchanged by this wave. |

---

## 1. `discovery.rs:3835` — replace `pearson_correlation`

### The two defects, both measured

```rust
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    ...
    let num = n_f * sum_xy - sum_x * sum_y;
    let den = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
    if den == 0.0 || !den.is_finite() { 0.0 } else { num / den }
}
```

1. **f32 catastrophic cancellation.** `n*Sx2 - Sx*Sx` subtracts two nearly
   equal large numbers. On a price-scale feature at the real row count this
   underestimates by **0.24x** — enough to flip a rank, which is the only thing
   the value is used for.

2. **One NaN scores an entire column exactly `0.0`.** `sum_x` becomes NaN, so
   `den` becomes NaN, so the `!den.is_finite()` guard fires and the column is
   reported as *uncorrelated* — indistinguishable from a genuinely useless
   feature. `core::features::align_features_by_ns` (features.rs:375) fills every
   aligned higher-timeframe array with `f32::NAN` and only overwrites rows whose
   higher-TF bar has closed, and the prefilter's in-sample slice starts at row 0
   (`discovery.rs:3936-3941`). Normalisation, which would have turned the NaN
   into 0.0, is off by default (`lib.rs:22-29`).

   Therefore **every H1/H4/D1 column scored exactly 0.0**, the stable
   `sort_by` at `discovery.rs:3946` broke the resulting mass tie by original
   column index, and base columns — emitted first in the cube — swept the whole
   top-K. The prefilter was not ranking the higher timeframes badly; it was not
   ranking them at all.

   > ⚠ **The keep-rate figures this paragraph used to quote — `base 217/217,
   > H1 40/217, H4 8/217` — are VOID and have been removed.** They were
   > produced by the very function described above, so they measured column
   > index, not correlation. Re-measured on real EURUSD M5/H1/H4 bars
   > 2026-08-09: the legacy function scored **100% of H1 and 100% of H4
   > columns exactly 0.0** (and 99.3% of base columns against the
   > triple-barrier label), and the median global rank per timeframe landed on
   > the cube's index midpoints — 973 / 2,919 / 4,865. The mechanism above is
   > confirmed; only the numbers are retracted.
   >
   > What replaces them: **H1 carries rankable signal and was being discarded**
   > (0 columns earned on rank before, 79 after; 106 clear a Bonferroni bar on
   > their own effective sample size). **H4 does not clear that bar on a single
   > column**, for a measured arithmetic reason that is not the old one. See
   > [`higher-timeframe-lane-2026-08-09.md`](higher-timeframe-lane-2026-08-09.md),
   > and re-run
   > `crates/neoethos-search/tests/higher_timeframe_lane_measured.rs` before
   > citing any higher-timeframe keep rate.

### The replacement

`neoethos-data` now ships the stable version, with tests including a
reproduction of the legacy behaviour:
`crates/neoethos-data/src/core/stats_f64.rs`.

Delete `fn pearson_correlation` and change the call site to:

```rust
use neoethos_data::core::stats_f64::{PearsonOutcome, pearson_pairwise_f32, MIN_PAIRWISE_SAMPLES};

// … inside prefilter_features, where pearson_correlation(col_train, target_train) was called:
let outcome: PearsonOutcome = pearson_pairwise_f32(col_train, target_train);
if outcome.skipped > 0 {
    tracing::warn!(
        target: "neoethos_search::prefilter",
        column = %name,
        skipped = outcome.skipped,
        used = outcome.used,
        "feature column has non-finite rows; correlation computed pairwise-complete over the \
         finite rows. Before 2026-08-09 a single NaN scored this column exactly 0.0."
    );
    nan_columns += 1;
}
if !outcome.is_rankable() {
    // A correlation from fewer than MIN_PAIRWISE_SAMPLES rows, or from a
    // column with no variance, is not evidence. Do NOT score it 0.0 and let it
    // compete — record it and exclude it explicitly.
    unrankable.push(name.clone());
    continue;
}
let score = outcome.abs() as f32;
```

Add `nan_columns` and `unrankable.len()` to the `FunnelProfile` next to the
existing `features_after_prefilter` stage (`discovery.rs:3577`), and emit one
WARN naming the first few unrankable columns.

### This is a deliberate behaviour change

Feature ranking moves. Higher-timeframe columns stop scoring 0.0 and start
competing on their real correlation, so top-K membership changes — that is the
point. **Any discovery artifact produced before this lands was ranked by the
old function and is not comparable to one produced after.**

---

## 2. `discovery.rs:96` — `DiscoveryRuntimeOverrides::default()`

```rust
prefilter_top_k: 50,
```

becomes

```rust
// 240, matching config.yaml AND
// neoethos_core::config::DiscoveryRuntimeConfig::default(). The three had
// drifted (code 50 / root yaml 240 / desktop yaml 50); the other two are
// fixed and pinned by
// crates/neoethos-core/tests/shipped_config_matches_defaults.rs.
prefilter_top_k: 240,
```

Then update `crates/neoethos-search/src/discovery_tests.rs:578`:

```rust
assert_eq!(defaults.prefilter_top_k, 240);
```

**Already applied elsewhere** (this workflow): `crates/neoethos-core/src/config.rs`
default 50 -> 240, `desktop/src-tauri/resources/config.yaml` 50 -> 240, and the
new drift test.

---

## 3. `validation.rs:1156-1167` — per-split walk-forward drops are invisible

```rust
(0..n_splits).into_par_iter().filter_map(|i| {
    …
    if test_start >= end || (train_end - start) < 40 || (end - test_start) < 40 {
        return None;
    }
    …
})
```

The GLOBAL failure warns at 1145-1152. A PER-SPLIT drop does not: the caller
sees a shorter `split_results` and cannot tell "this split was skipped" from
"this split failed". This is directly implicated in the recorded
`walkforward=false` symptom.

Replace `filter_map` with a `map` returning
`Result<WalkforwardSplitResult, SplitSkip>` where

```rust
struct SplitSkip { i: usize, start: usize, end: usize, train_bars: usize, test_bars: usize, reason: &'static str }
```

partition the results, emit ONE warn with the reason histogram, and carry
`skipped_splits` out in the returned struct so the caller reports
"walkforward inconclusive (N of M splits too short)" instead of
"walkforward failed".

---

## 4. `validation.rs:1175-1179` — day indices silently substituted for timestamps

```rust
let slice_ts = if timestamps.len() == n { &timestamps[test_start..end] } else { slice_days };
```

A length mismatch feeds day INDEX integers into a parameter the evaluator reads
as epoch milliseconds, silently changing every time-derived cost (swap, session
spread) in that split. Silently swapping units into a cost model is not a
recoverable condition:

```rust
anyhow::ensure!(
    timestamps.len() == n,
    "walkforward: timestamps len {} != bar count {n} — refusing to substitute day indices for \
     epoch milliseconds, which would silently change every time-derived broker cost",
    timestamps.len()
);
let slice_ts = &timestamps[test_start..end];
```

---

## 6. `genetic/smc_indicators.rs:353` — substring column matching, now scanning 1,795 names

`find_feature_column` accepts `norm.contains(alias)` and takes the FIRST
matching column in cube order. Two of the eleven SMC aliases have no `smc_`
column to bind to — `choch` (there is no `smc_choch`) and `premium` (the SMC
family ships `smc_pd_array`, not `smc_premium`) — so for those two the search
falls through the whole cube.

That was a 217-name scan. After the vocabulary restoration it is a 1,795-name
scan, so the probability that a restored indicator column silently becomes the
SMC gate's "premium" or "change of character" signal has gone up by an order of
magnitude. **It is latent, not live:** measured, zero of the 342 ids in
`ALL_INDICATORS` contain `premium` or `choch` as a substring.

Two changes, neither large:

1. require an exact `smc_`-prefixed name for every SMC alias, and let the
   absence fall to `derive_smc_arrays` explicitly rather than by accident;
2. log, once, which column index each of the eleven flags bound to — a gate
   that silently binds to the wrong column is indistinguishable from one that
   works.

(Related and already better than reported: `build_smc_arrays`
(`smc_indicators.rs:578`) DOES read the `smc_` columns — it seeds from
`derive_smc_arrays(ohlcv)` and then overrides each flag from its matching
column. The crude 12/20-bar re-derivation is the fallback, not the only path.)

---

## 5. Not an edit — a call that must start happening

`crates/neoethos-data/src/core/feature_registry.rs:241` `validate_feature_names`
and `features.rs:97` `FeatureFrame::validate_registry` both exist and both have
**zero callers repo-wide**. The gate exists, the call never happens.

Call `features.validate_registry()` once immediately after
`prepare_multitimeframe_features_with_options` returns, WARN-and-list by
default (existing drift must not block a run), hard-fail in CI. Then extend the
registry to cover the restored indicator ids — the vocabulary restore widens the
gap by hundreds of names, and the registry is the natural place to record the
before/after column counts.

---

# APPENDED 2026-08-09 — the repair wave, shard 2 (presets / settings / knob catalog)

Everything below belongs to the `docs/audit-status-2026-08-09.md` repair and lands
in a file that shard's territory forbids: the two `config.yaml` files and
`crates/neoethos-core/src/config.rs`. The code half of each item is **already
applied** in `crates/neoethos-app/src/server/` — these are the remaining halves.

Each entry gives file, line, the exact current text, the exact replacement, and
what changes behaviourally. **Every number below was re-derived from
`crates/neoethos-core/src/domain/prop_firm.rs` at the line cited — not copied
from the audit document, which is wrong about one of them.**

| # | File | Item | Blocking? |
|---|------|------|-----------|
| 7 | `config.yaml:158-159` | #213/#214 — the two wrong drawdown breakers | no — the writer is fixed, so any preset click also fixes these |
| 8 | `desktop/src-tauri/resources/config.yaml` | §2.3 #245/#246/#248/#249/#250 + a **sixth, uncounted** divergence | no |
| 9 | `crates/neoethos-core/src/config.rs` | de-duplicate the ×0.7 drawdown buffer | no — hardening |

---

## 7. `config.yaml:158-159` — the two drawdown breakers (#213 / #214 / #269)

This machine runs `risk.preset: none` (`config.yaml:146`) — verified, not assumed.
That is the branch the corrected numbers must come from.

```yaml
# config.yaml:158  CURRENT
  daily_drawdown_limit: 0.10000000149011612
# config.yaml:159  CURRENT
  total_drawdown_limit: 0.20000000298023224
```

```yaml
# config.yaml:158  CORRECTED
  daily_drawdown_limit: 0.08
# config.yaml:159  CORRECTED
  total_drawdown_limit: 0.14
```

### Where each number comes from

| Field | Source | Value for `preset: none` |
|---|---|---|
| `daily_drawdown_limit` | `PropFirmRuntimeDefaults::NONE_OWN_MONEY.daily_dd_stop_trading_pct` — `domain/prop_firm.rs:301`, selected at `:313`. This is the same field `RiskConfig::default()` reads at `config.rs:558`. | **0.08** |
| `total_drawdown_limit` | `PropFirmConstraints::NONE_OWN_MONEY.max_overall_drawdown_pct` (`domain/prop_firm.rs:167`) `= 0.20`, × the 0.7 buffer `config.rs:561` applies. | **0.14** |

### ⚠ Correction to the triage note

The shard triage proposed `daily_drawdown_limit: 0.04`. **0.04 is FTMO's number**
(`PropFirmRuntimeDefaults::FTMO_STANDARD.daily_dd_stop_trading_pct`,
`prop_firm.rs:269`) and this install is not on FTMO. Writing 0.04 here would not
be wrong in the dangerous direction — it is *tighter* than 0.08 — but it would put
a third, unexplained number into the file and break agreement with
`RiskConfig::default()`, which is the exact failure #213 records. `0.20 × 0.7 =
0.14` for the total is confirmed correct.

### What changes

On a £1,000 account: the daily breaker arms at **£80 lost instead of £100**, the
total at **£140 instead of £200**. Both **REFUSE** trading earlier than today.
Nothing is permitted that was not permitted before — this direction is strictly
more conservative. Both values satisfy `validate_safety_bounds`
(`config.rs:2634-2655`) and the new value checks on `POST /settings/raw`.

### This half is no longer urgent, and here is why

`crates/neoethos-app/src/server/risk.rs:171-174` — the writer that produced the
`f32`-widened values now in the file — was fixed in the same wave. It seeds from
the buffered numbers and, additionally, **never widens a breaker that is already
tighter on disk**. So clicking any preset in the UI now writes 0.08 / 0.14 rather
than putting 0.10 / 0.20 back. Applying this edit by hand simply gets there
without a click.

---

## 8. `desktop/src-tauri/resources/config.yaml` — the seed diverges from this machine (§2.3)

Five value disagreements, all money, all confirmed by reading both files today.
Direction: **seed → root** (make a fresh install trade what this machine trades).

| Line | Current (seed) | Replacement | Item |
|---|---|---|---|
| `:9` | `  account_currency: USD` | `  account_currency: GBP` | #245 |
| `:125` | `  discovery_mode: prop_firm  # FTMO-style robust discovery (default)` | `  discovery_mode: risky` | #246 |
| `:36` | `  prop_firm_rules: true` | `  prop_firm_rules: false` | #248 |
| `:31` | `  daily_drawdown_limit: 0.04` | `  daily_drawdown_limit: 0.08` | #249 |
| `:32` | `  total_drawdown_limit: 0.07` | `  total_drawdown_limit: 0.14` | #250 |

**The currency edit is the dangerous one and should go first.** A fresh install
computes every pip value, every risk-per-trade lot size and the prop-firm
daily-loss check in USD against a GBP account. At ~1.27 that mis-sizes every
position by ~27%; against the 30% risky ceiling that is ~38% real risk per trade
the operator never chose.

### ⚠ A SIXTH divergence the audit did not count

The audit reports "0 keys unique to the seed" and five value disagreements. Both
are true, and they miss this: **`risk.preset` exists in the root
(`config.yaml:146`, `none`) and is ABSENT from the seed entirely.** A key present
in one file and missing from the other is neither "unique to the seed" nor a
"value disagreement", so it fell between the two tests. Serde fills the missing
key with `PropFirmPreset::default()` = **Ftmo** (`prop_firm.rs:437` asserts it),
so a fresh install runs the FTMO preset while this machine runs none.

That is also what makes edits `:31`/`:32` above conditional. **Apply them only
together with:**

```yaml
# desktop/src-tauri/resources/config.yaml — ADD inside the `risk:` block
  preset: none
```

Without the added key the seed would carry `preset: <Ftmo>` with `prop_firm_rules:
false` and own-money drawdowns — a combination this machine has never run and no
test covers. **If you would rather not add the key, then leave `:31`/`:32` at
0.04 / 0.07: those are exactly right for FTMO** (`prop_firm.rs:269` → 0.040;
`0.10 × 0.7` → 0.07) and the file stays self-consistent. What must not happen is
FTMO's preset with own-money drawdown numbers.

### Say this out loud before applying #246 and #248

These two make the SHIPPED DEFAULT of a fresh install `risky` mode with the
prop-firm rule set disarmed — i.e. the 30% risky risk ceiling and no challenge
accounting, for anyone who installs the app. That is correct if the installer is
only ever the owner's own machine, and it is the audit's stated intent. It is a
loosening, so it is recorded here explicitly rather than applied quietly. The
alternative direction — move the ROOT to match the SEED — is the conservative
one and closes the same divergence.

---

## 9. `crates/neoethos-core/src/config.rs` — one home for the drawdown buffer

Not a defect; the remaining half of the #213 fix. Two files now compute the same
buffered drawdown seeds from a preset:

- `config.rs:558` `daily_drawdown_limit: runtime.daily_dd_stop_trading_pct`
  and `config.rs:561` `total_drawdown_limit: (constraints.max_overall_drawdown_pct as f64) * 0.7`
- `crates/neoethos-app/src/server/risk.rs` — same two expressions, with a local
  `const TOTAL_DRAWDOWN_BUFFER: f64 = 0.7;` and a comment saying it is a mirror.

A mirror is what drifted last time. Add to `impl RiskConfig` (or beside it):

```rust
/// The internal drawdown breakers for `preset`, buffered SHORT of the firm's
/// published ceilings. Single source for `RiskConfig::default()` and for
/// `POST /risk/preset` (`neoethos-app/src/server/risk.rs`), which used to
/// compute them independently and wrote the RAW ceilings for months (#213).
///
/// Returns `(daily_drawdown_limit, total_drawdown_limit)`.
pub fn preset_drawdown_seeds(preset: PropFirmPreset) -> (f64, f64) {
    let constraints = PropFirmConstraints::for_preset(preset);
    let runtime = PropFirmRuntimeDefaults::for_preset(preset);
    // Daily is NOT `max_daily_loss_pct * 0.7` — the presets publish their own
    // stop-trading threshold (FTMO 0.040 vs a 0.05 ceiling). Total is the
    // published overall cap at 70%.
    (
        runtime.daily_dd_stop_trading_pct,
        (constraints.max_overall_drawdown_pct as f64) * 0.7,
    )
}
```

Then `RiskConfig::default()` uses it at `:558`/`:561`, and `server/risk.rs`
replaces its local constant and both expressions with one call. Behaviour
identical on every preset — this only removes the ability to drift.

---

## W10b — spawn the MCP sidecar by name, and verify what answered

**Filed 2026-08-10 by the main session.** The other half of W10 landed already:
`crates/neoethos-mcp` no longer declares `[[bin]] name = "neoethos-mcp"`; its
binary is `neoethos-codex`. That removes the collision at the source. This
remaining edit lives in `desktop/**`, owned by another workflow at the time of
writing, so it is recorded rather than applied.

**File:** `desktop/src-tauri/src/lib.rs`, around `:231-250`.

**Current shape:** the desktop app spawns the sidecar **by bare filename** from
its own directory:

```rust
"neoethos-mcp.exe"   // windows
"neoethos-mcp"       // unix
```

Nothing checks that the process which started is the outbound client sidecar
rather than something else that happens to carry that name. Before today two
different crates produced that exact filename; the rename means the *repo* can
no longer produce a second one, but an installed directory may still hold a
stale executable from an earlier build.

**The edit:** after spawn, probe the child and refuse to use it if it is not the
sidecar — the sidecar already answers on localhost HTTP, so a single request to
its health/tools route is enough. On mismatch or no answer, log at ERROR naming
the path that was spawned, and treat the sidecar as unavailable rather than
proceeding with a process that will not answer the Supervisor.

**Why it matters:** `crates/neoethos-app/src/app_services/supervisor.rs:593` and
`:625` already surface `"MCP sidecar not reachable — is neoethos-mcp running?"`.
That message is what an operator sees today when the wrong binary is spawned,
and it points at the wrong cause. Update its text in the same change: the
sidecar is `neoethos-mcp` (from the `mcp/` workspace) and the control plane is
`neoethos-codex` (from `crates/neoethos-mcp`) — naming both prevents the next
person conflating them again.

**Also check when that crate is unlocked:** `.github/` and any installer script
that references `neoethos-mcp` by name, to confirm none of them meant the
control-plane binary.

---

## GPU FEATURE SWEEP — 2026-08-10 — three of four findings RETRACTED

**Filed, then largely withdrawn, by the main session.** Recorded in full because
a retraction that is not written down gets re-discovered as a "finding" next
week. I read the whole `[features]` graph, flagged four items, and on checking
each against the current tree, three were wrong.

### RETRACTED — `burn-cuda-backend` is not an oversight

I claimed it was missing from the `gpu-cuda` aggregate by accident. It is
**deliberately excluded and documented in place** (`crates/neoethos-models/Cargo.toml`,
in the `gpu-cuda` block): burn-cuda 0.21 was measured pathological for our very
small neural nets on an A6000 (2026-06-10) — 14 real epochs in 74 minutes for
one combination, plus burn-tensor dtype panics, against ~17 minutes TOTAL on
burn-ndarray CPU. The dependency and the feature are kept precisely so a card
user can opt in explicitly:

```
cargo build --release --features gpu-cuda,burn-cuda-backend
```

**What IS still open is a measurement, not an edit.** That 2026-06-10 figure was
taken at a much smaller scale, and a later observation (2026-08-01) recorded a
training run sitting at GPU 0% / 1 MiB for over an hour at 799,880 rows — which
suggests the cost ratio may have inverted as the work grew. So: on the 4090 run
the same combination twice, with and without `burn-cuda-backend`, and let the
number decide. Do NOT add it to the aggregate on the strength of an argument.

### RETRACTED — `ml-blend` is enabled

I claimed no aggregate pulls it. `crates/neoethos-cli/Cargo.toml:50` reads
`neoethos-trader = { path = "../neoethos-trader", features = ["ml-blend"] }`,
with a comment explaining it drives `trader-replay --blend`. My grep truncated
before reaching it. Nothing to do.

### RETRACTED — LightGBM's device string is already fixed

I claimed `tree_models/lightgbm.rs` still writes `device_type = "gpu"` (the
OpenCL learner our build does not compile). It no longer does.
`effective_device_type()` now returns `"cuda"` or `"cpu"` and never `"gpu"`,
behind four conditions — the operator opting in, the `lightgbm-gpu` feature
being linked, the device preference not being an explicit cpu, and a card
actually being visible — and it WARNS when the knob is set on a binary built
without the learner. Its doc comment states the vocabulary explicitly: "cuda,
never gpu. In LightGBM those name two different tree learners." Landed by the
app-side wave earlier today. Nothing to do.

### STANDS — `nvtx` is enabled by nothing

`crates/neoethos-search/Cargo.toml:4` declares `nvtx = []` and no aggregate, no
CI line and no documented build command turns it on, so the NVTX range macros
never expand and we have no in-kernel timeline on the card. Harmless to
correctness; it costs us the ability to see where device time goes. Suggested:
enable it for the diagnostic build on the 4090 only — `--features gpu-nvidia,nvtx`
— rather than adding it to an aggregate, since it is instrumentation, not
behaviour.

---

## HOST-FALLBACK DEBT — the counter is written, nothing reads it

**Filed 2026-08-10.** Half of this landed; the other half needs a crate that was
being written at the time.

`vector_ta::cuda::host_fallback::record()` had **zero call sites in the entire
crate** while four wrappers computed on the host and returned the result as a
`DeviceArrayF32` — `rvi_batch_dev` (size-guarded at `rows*len <= 2_000_000`, the
common case), and `mass_`, `net_myrsi_` and `vosc_many_series_one_param_time_major_dev`
(the first two unconditional, the third in the `use_ds == false` branch). A caller
holding that pointer cannot tell the device never ran. `total()` therefore
returned zero **by construction rather than by achievement**, which reads as a
clean bill of health — an instrument that structurally cannot report debt is
worse than no instrument.

**Done:** all four now call `record()` with their indicator id, each with the
rule written at the site — card present and a kernel exists, the card runs it;
card present and no kernel, the host may compute it but the call is COUNTED.

**Still needed, in `crates/neoethos-search/src/eval_telemetry.rs` beside the
existing device summary:** read `vector_ta::cuda::host_fallback::total()` and
`per_indicator()` at run end and print them next to the GPU percentage, so a
non-zero debt appears in the same place an operator already looks for "did this
run stay on the card". Gate it on the `cuda` feature — the module is
`#![cfg(feature = "cuda")]`.

**Note for whoever verifies this:** the four edits are inside `cfg(feature =
"cuda")` code. `cargo check -p vector-ta` does not compile them, and the crate
cannot be checked with `--features cuda` from the workspace ("cannot specify
features for packages outside of workspace") nor from inside it ("believes it's
in a workspace when it's not"). They were brace- and paren-balance checked only.
**The card is their first compiler.**
