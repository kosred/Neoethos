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

> **RECONCILED against the working tree 2026-08-10 by the docs owner.**
> W2-1, W2-2, W2-3, W2-4 and W2-10 were carrying **PENDING** while commit
> `64722aa7` ("wip: a full day of remediation across every crate") had already
> applied all five. Two of them — W2-1 and W2-2 — were the stated blockers for
> deletions **D5** and **D6**, so the stale status was actively holding work
> back: D6 is in fact already complete, and D5 is unblocked and waiting only on
> the comment correction now filed as W2-26. Every row below that says DONE
> names the file and line that was read to confirm it, not the commit message.
> Rows W2-21 … W2-26 are new, filed in the same pass.
>
> ⚠ Marking a row DONE **on the strength of a commit message alone is how this
> table went stale in the first place.** `64722aa7` says NOT COMPILE-VERIFIED in
> its own subject. Nothing here was verified by building — this machine has no
> nvcc, no NVIDIA driver and no CUDA toolkit, and the Fix phase runs no cargo.
> What is claimed is what was read.

| # | State | File / site | Edit required | Blocks |
|---|-------|-------------|---------------|--------|
| W2-1 💰 | **DONE — applied by `64722aa7`, re-verified 2026-08-10** | `crates/neoethos-app/src/app_services/live_trading.rs:709-711`, `:713-745`, `:1426-1440` | Live trailing now reads `models.exit_policy`. `:710` binds `sizing.models.exit_policy`; `:1426` `let trailing = exit_policy.filter(\|p\| p.trailing_enabled);` gates the entire amend block; `:1438-1440` take `trailing_be_trigger_r` / `trailing_stop_multiplier` / `trailing_min_lock_pips` from the policy. Three-arm log at `:713-745`: ARMED (warn, all four values), DISABLED (info), UNRESOLVABLE (error + **refuse to trail**, no fallback to the constant). | **D5 IS UNBLOCKED.** ⚠ But see W2-26 — the comment at `config.rs:452-475` still asserts the OLD wiring by file:line, so a reader of `config.rs` is told the opposite of what the tree does. Fix that comment in the same change as the D5 deletion. |
| W2-2 💰 | **DONE — applied by `64722aa7`, re-verified 2026-08-10** | `crates/neoethos-app/src/server/risk.rs:197-224` (write removed), `:418`, `:440`; `crates/neoethos-core/src/config.rs:415-424`, `:3063-3070` | The `settings.risk.prop_firm_rules = preset != None` write is gone from `POST /risk/preset`. `:418` reads `prop_firm_rules_enabled: derive_prop_firm_rules_active(settings)`, defined at `:440`, which derives from `system.trading_mode` — what the engine actually reads. `desktop/src/screens/Risk.tsx:81` and `RiskyMode.tsx:42` carry the matching corrections. | **D6 IS DONE TOO** — the field is deleted from `RiskConfig` (`config.rs:415`, comment in place) and tombstoned in `RETIRED_KEYS` (`config.rs:3063`), so a live store still carrying the key loads with it NAMED at WARN rather than rejected. Nothing left here. |
| W2-3 💰 | **DONE — applied by `64722aa7`, re-verified 2026-08-10** | `crates/neoethos-app/src/server/risk.rs:259-262`, `:271-284`, `:286-302`, `:303+` | The preset writer seeds from the **buffered** numbers (`:261` `runtime.daily_dd_stop_trading_pct`, `:262` `max_overall_drawdown_pct * TOTAL_DRAWDOWN_BUFFER`), never the raw ceiling. It also went further than this row asked: `:271-284` takes `prev.min(seeded)` for both breakers, so **a preset click can tighten a breaker and can never widen one**, and `:303+` WARNs with `preset_would_set` / `kept` when a seed is refused. `:286-302` logs firm-raw, buffered, previous and applied side by side on every click. | Nothing. The `0.10000000149011612` / `0.20000000298023224` values cannot be regenerated. |
| W2-4 | **DONE — applied by `64722aa7`, but NOT where this row said, re-verified 2026-08-10** | `crates/neoethos-app/src/server/settings.rs:280-301` | **No app-side edit was made and none was needed.** `Deserialize for Settings` is now hand-written and *is* the loader (`neoethos-core/src/config.rs`, `mod load_seal`), so the step-2 call `serde_yaml_ng::from_str::<neoethos_core::Settings>(&payload.yaml)` at `:301` runs the retired-key prune and the **unknown-key refusal** on the payload *before* anything is written. `trailing_enabeld:` now returns 400 naming the key. `:280-298` records the reasoning in place. | Nothing. Sealing the deserializer in core closed this as a consequence — which is why sealing beat fixing call sites one at a time. |
| W2-5 | **DISCHARGED — the constraint no longer binds. Re-verified 2026-08-10 (not one of the five this pass was sent to check; found while confirming W2-4).** | `crates/neoethos-core/src/config.rs:3106`, `:3118`, `:3124`, `:3129`; prune at `:3285`, called `:3587`; typed parse `:3600` | All four tombstones are in `RETIRED_KEYS` — `models.export_onnx` (`:3106`), `news.news_kill_window_min` (`:3118`), `news.news_lookahead_minutes` (`:3124`), `news.perplexity_enabled` (`:3129`) — and `from_raw` calls `prune_retired_keys` at `:3587`, **before** `serde_yaml_ng::from_value::<SettingsWire>` at `:3600`. A live store carrying all four therefore loads, with each key NAMED at WARN, and never reaches the unknown-key refusal. | **The ordering constraint is dead: collapsing the live store is NO LONGER a precondition for the sealed loader, which has already shipped.** ⚠ This row said the opposite and would have stopped someone from shipping something already shipped. Verify the operator's actual `%LOCALAPPDATA%\neoethos\config.yaml` against `RETIRED_KEYS` once before the next live start anyway — a FIFTH retired key that is not tombstoned would still be a hard startup failure, and no test covers his file. |
| W2-6 💰 | **PENDING** | `crates/neoethos-app/src/server/knob_catalog.rs` — `month_capacity` entry (~`:884-894`) | Remove `min: Some(12)` or re-label. It advertises a minimum of 12 while describing the knob as a RAM cap. `month_capacity` sizes `monthly_pnls` → metric slot 7 → `named.rs:161` × **0.45**, the dominant term of the prop-firm objective. Setting the UI-endorsed minimum scores every gene on its **first twelve months** of a ten-year record and returns a plausible number. | — |
| W2-7 | **PENDING** | `crates/neoethos-app/src/server/knob_catalog.rs` — 34 `env_var:` entries | Set to `None` in the same wave as D's deletion of the five `install_*_from_env` wrappers. The catalog is served to the LLM control plane at `mcp/server.rs:716-725` **as authoritative**, so a stale `env_var:` tells the control plane to set a variable nothing reads. | — |
| W2-8 | **PENDING** | `crates/neoethos-cli/src/tui/config_view.rs:50-54`, `:152-162` | The "Discovery mode" list must offer **`strict` and `legacy` only**, and stop rejecting `legacy`. It currently offers `prop_firm`/`strict`/`risky` and rejects `legacy` — two values the engine maps to nothing, and a refusal of one of the two it honours. | Cosmetic once E's fall-through WARN lands; still actively misleading. |
| W2-9 | **PENDING** | `crates/neoethos-cli/src/…/resolved_config.rs:329`, `:332`, `:337-350` | `:329` prints raw `models.discovery_mode` beside a mode resolved from `system.trading_mode`; `:332` help names the wrong field; `:337-350` still reports `normalize_features` and `disable_smc_gate` as `source=env` from variables the engine no longer reads. | This is the ONE diagnostic built to answer "which value won", and it is wrong on three rows. |
| W2-10 💰 | **DONE for the config half — applied by `64722aa7`, re-verified 2026-08-10** | `desktop/src-tauri/src/lib.rs:31`, `:35-50`, `:52-72`, `:635-643` | The bare relative `"config.yaml"` is gone from the process's answer to "which config?". `:31` `static RESOLVED_CONFIG_PATH: OnceLock<PathBuf>`; `:35-50` `record_config_path` absolutises against the CWD **at the moment that CWD is still the correct one** (deliberately not `fs::canonicalize` — reasons given in place); `:52-72` `fatal_config_error` replaces the old `unwrap_or_else(\|_\| Settings::default())`, so a load failure stops the app instead of starting it under a different trading policy. `prepare_data_root:559` still *detects* with `Path::new("config.yaml").exists()`, but the hit is immediately absolutised by `record_config_path`, so it is a probe, not a stored path. | ⚠ **Declared, not hidden:** the DATA half is still CWD-dependent. `:635-643` logs the `set_current_dir` failure at the point of failure and names the consequence — "relative data/cache/model paths will resolve against the process CWD instead". That is a loud substitution, not a silent one, which satisfies non-negotiable #5; making `system.data_dir` absolute at the same seam is separate work. |
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
| W2-21 💰 | **PENDING — filed 2026-08-10, half B of the strategy blacklist** | `crates/neoethos-search/src/live_portfolio.rs:153` `save_live_portfolio_json`; callers `crates/neoethos-app/src/app_services/discovery.rs:1160` and `crates/neoethos-cli/src/main.rs:1307`; gate `crates/neoethos-app/src/app_services/strategy_blacklist.rs:298` `is_blacklisted` | **Current value: `grep -rc blacklist crates/neoethos-search/src/` returns ZERO for every file.** Discovery has no idea the blacklist exists. **Target value: at least one consultation before a gene reaches `live_portfolio.json`.** See the detail section below for why the obvious edit does not compile and what does. | See §W2-21 below. This is the item `rediscovery.rs:10-32` names as *"still OPEN"* in its own header. |
| W2-22 💰 | **FILED ELSEWHERE IN THIS DOCUMENT — see [`models.blend_*`](#modelsblend_--the-live-blend-multipliers-need-a-config-recipient) at the end** | `crates/neoethos-core/src/config.rs` (`ModelsConfig`, beside `live_ml_gate` at `:1031` / `:2425`) | Pointer row only, so a reader of this TABLE finds it. The `models.blend.*` recipient was the fourth un-filed item in the 2026-08-10 record; the app/CLI/trader shard filed it in full while this reconciliation was running, and its version is better than mine — it names `blend_gate_floor` / `blend_veto_below` (flat, not nested), the exact `Default` sites, the full six-point drag, and the two waiting readers. **Re-verified against the tree 2026-08-10: `BlendConfig::from_config_values` now has production callers** (`live_trading.rs:668`, `main.rs:676`/`:865`/`:892`) and `operator_blend_gate_floor`/`_veto_below` (`live_trading.rs:520`/`:528`) return `None` pending the fields. Do not file a third copy. | — |
| W2-23 💰 | **PENDING — filed 2026-08-10, two starting balances, neither reads the account** | `crates/neoethos-core/src/config.rs:586` and `:1910` | **Current values: `risk.initial_balance = 10_000.0` (`:586`) and `models.backtest_runtime.initial_equity = 100_000.0` (`:1910`) — a factor of ten apart, in one file, both shipped, neither documented as answering a different question.** Target: see §W2-23 below. Non-negotiable #1 applies — where two numbers conflict the SAFER wins, and the safer starting balance is the SMALLER one, because every percentage-of-equity limit converts to a smaller absolute loss against it. | Nothing compiles against the disagreement, which is why it survived. It reaches the objective through the monthly-PnL buckets. |
| W2-24 | **PENDING — filed 2026-08-10, §6.3 of `docs/audit-status-2026-08-09.md`, one line** | `crates/neoethos-search/src/gpu_native/prototype_a_engine.rs:170` and `:172` | **Current values: `&& scenario.spread_ticks == 0` and `&& scenario.commission_micros == 0`. Target values: `== NO_TICK_OVERRIDE` and `== NO_MICRO_OVERRIDE` (both `-1`).** Confirmed present 2026-08-10 and confirmed still wrong: `scenario.rs:366`/`:368` write `NO_TICK_OVERRIDE`/`NO_MICRO_OVERRIDE` into every base scenario, so `validate_supported_scenarios` REJECTS every descriptor the base path produces — and would WAVE THROUGH a literal-zero, i.e. free-trading, one. | `bench` on Prototype A is broken. No live money (A is bench-only), but the bench cluster is five registered CLI subcommands. These are the LAST two zero-sentinel comparisons in the tree — `prototype_population.rs`, `prototype_b_population.cu:133-134` and `eval.rs:2663` all agree on `-1`. |
| W2-25 | **PENDING — filed 2026-08-10, stale build guidance pointing at a loaded gun** | `crates/neoethos-app/Cargo.toml:33`, `crates/neoethos-data/Cargo.toml:110`, `crates/neoethos-data/src/core/hpc_ta.rs:1539` | Three sites still tell the operator to set the **singular** `CUDA_ARCH`, which `vendor/vector-ta-.../build.rs::target_archs` takes as the single-value branch and turns into a **SINGLE-architecture fatbin that will not load on any other card**. `crates/neoethos-data/build.rs:30-32` names this exact trap in its own header. The build logic was NOT narrowed and is correct — only the guidance is stale. Exact replacements in §W2-25 below. | Zero behaviour change in the tree; it changes what the operator is told to type on the card. |
| W2-26 | **PENDING — a comment that asserts a wiring that is no longer true** | `crates/neoethos-core/src/config.rs:452-475` | The `WARNING UNWIRED / SHADOWED DUPLICATE` block above `risk.trailing_*` still reads *"Live execution trails UNCONDITIONALLY with no config recipient at all (`live_trading.rs:1226-1272`)"* and *"THEY ARE NOT DELETED YET, DELIBERATELY … wire live to `models.exit_policy` FIRST"*. **W2-1 landed; live IS wired** (`live_trading.rs:710`, `:1426-1440`). The comment now describes the opposite of the tree and is the stated precondition for D5, so a reader of `config.rs` concludes D5 is still blocked when it is not. Non-negotiable #4. | Fix the comment in the same change as the D5 deletion of `risk.trailing_enabled` / `_atr_multiplier` / `_be_trigger_r` / `_min_lock_pips` (`config.rs:475-486`, defaults `:640-643`) — do not fix it separately, or the next reader sees a struct whose comment says "deleted" and fields that are still there. |

## The four items the 2026-08-10 record says were never written down

Filed 2026-08-10 by the docs owner, on instruction. All four were REAL and all
four were missing from this document, so anyone reading it concluded they were
closed. Each was re-opened in the tree before being written here — none is
copied from a report.

One of the four, the `models.blend.*` recipient, was filed by the app/CLI/trader
shard **while this reconciliation was running**; it lives at the end of this
document and W2-22 above points at it. The other three are detailed here.

### §W2-21 — the strategy blacklist, half B: discovery never consults it

**Measured, not asserted:** `grep -rc "blacklist" crates/neoethos-search/src/`
returns zero matches in every file. The word does not occur in the search crate.

`crates/neoethos-app/src/app_services/rediscovery.rs:10-32` already documents
this correctly and calls it *"still OPEN"* — that header is the only place it
was recorded, and a module header in a crate is not where the next person looks
for outstanding work. Its own summary:

* **Selection IS guarded.** `server::autonomous` and
  `app_services::federation.rs:198` both call
  `strategy_blacklist::is_blacklisted` before a portfolio can go live, and
  `server::portfolios` hides retired ones from the listing.
* **Identity IS the gene, not the file** (`strategy_blacklist.rs`,
  `GENE_FINGERPRINT_PREFIX` at `:97`, `GENE_MEASUREMENT_FIELDS` deny-list at
  `:115`), so a re-discovered artifact describing the same rule is caught at
  selection even though its bytes differ.
* **Still open:** a rediscovery whose portfolio contains the culled gene
  *alongside different ones* hashes differently and is not blocked, and the
  search burns the whole run re-deriving a rule that was already retired for
  losing real money.

#### Why the obvious edit does not exist yet, and what the real target is

The obvious edit — "call `is_blacklisted` from `neoethos-search`" — **cannot be
written**. `strategy_blacklist` lives in `crates/neoethos-app`, and
`neoethos-search` does not and must not depend on `neoethos-app` (that is a
dependency cycle). Verified from the manifests:

| Crate | Depends on | Can call `is_blacklisted` today? |
|---|---|---|
| `neoethos-app` | `neoethos-search`, `neoethos-core` | yes — it owns it |
| `neoethos-cli` (`Cargo.toml:42-45`) | `neoethos-core`, `neoethos-search`, … — **NOT `neoethos-app`** | **no** |
| `neoethos-search` (`Cargo.toml:43-46`) | `neoethos-data`, `neoethos-core`, gpu crates | **no** |

So the target is two-stage, and the stages must be labelled honestly:

1. **The durable fix — move the store to `neoethos-core`.** `strategy_blacklist`
   depends on nothing app-specific: `neoethos_core::utils::hashing::fnv1a64`
   (`:84`), `system.data_dir` for the path (`:71`), and serde. Moving it to
   `neoethos-core` makes it reachable from `neoethos-search`, `neoethos-cli` and
   `neoethos-app` alike, and only then can discovery drop a blacklisted gene
   before it is ever scored. That is the fix `rediscovery.rs:30-31` asks for.
2. **The cheap interim — filter at the write site.** `save_live_portfolio_json`
   (`crates/neoethos-search/src/live_portfolio.rs:153`) has exactly two callers:
   `crates/neoethos-app/src/app_services/discovery.rs:1160` (CAN call the gate)
   and `crates/neoethos-cli/src/main.rs:1307` (CANNOT). Guarding only the app
   caller closes the GUI/Supervisor path and leaves the CLI path unguarded.

⚠ **If you take the interim, say so in the log line.** A half-covered gate that
prints nothing is worse than no gate, because the next audit will read the app
call site and record the hole as closed — which is precisely how this item came
to be missing from this document. Whatever lands must name which paths are
covered and which are not, at run time, in the run's own output.

**Non-negotiable #2 applies with force here.** `is_blacklisted` is a gate that
was written and that the search never calls. Do not add a second one.

### §W2-23 — two starting balances in one file, ten times apart

| Site | Field | Current value |
|---|---|---|
| `crates/neoethos-core/src/config.rs:586` | `risk.initial_balance` | **`10_000.0`** |
| `crates/neoethos-core/src/config.rs:1910` | `models.backtest_runtime.initial_equity` | **`100_000.0`** |

(The 2026-08-10 record cites `:1890` for the second. The struct field is declared
at `:1899` and the literal is at `:1910`; the defect is there, the line number
had drifted by twenty. Recorded so nobody re-litigates it as "not at the cited
line".)

`:585` says *"Account starting balance is broker-specific. Operators override
this via `config.yaml`'s `risk.initial_balance`."* `:1898` says *"Starting equity
for canonical backtest PnL accounting (> 0)."* Read individually each is
defensible. Read together they mean **the equity a strategy is scored against is
ten times the equity it will be traded on**, and nothing in the tree says so.

Why it matters rather than being cosmetic: `initial_equity` seeds the PnL
accounting whose monthly buckets are sized by `month_capacity` (`:1911`) and
consumed as metric slot 7 — the term W2-6 identifies as carrying **×0.45**, the
dominant weight of the prop-firm objective. A percentage-based risk model is
scale-free in the ideal, but the floors are not: absolute pip floors, the
minimum-lot quantisation and the drawdown breakers all bite differently at £1,000
than at £100,000, and the operator's real account is the £1,000-scale one
(`docs/` goal scenarios: 1,000 → 200,000).

**Target.** One of these two, decided — NOT patched to agree by hand:

* **(a) Preferred — derive the backtest equity from the account.** Make
  `initial_equity` default from `risk.initial_balance` rather than from its own
  literal, so there is one number and the search scores what will be traded.
  This CHANGES SCORING: every gene is re-scored against a 10× smaller account,
  which makes the absolute floors bind harder and will reject strategies that
  pass today. **Any artifact produced before it lands is not comparable to one
  produced after** — say that in the release note, the way §1 of this document
  says it for the correlation change.
* **(b) If they must stay independent**, then each doc comment must state the
  other's existence and why the answer differs, and
  `crates/neoethos-core/tests/config_has_recipient.rs` must carry a row pinning
  both values so the next drift breaks a test instead of a run.

⚠ **Non-negotiable #1 governs the direction.** If anyone reconciles these by
moving one to meet the other, the SAFER number wins, and the safer number is
**10,000**, not 100,000 — a smaller starting equity converts every
percentage-of-equity ceiling into a smaller absolute loss, and it is the one that
matches the account this system actually trades. Raising `initial_balance` to
100,000 to match the backtest would silently multiply every absolute risk figure
the operator reads by ten.

### §W2-25 — three sites still tell the operator to set the singular `CUDA_ARCH`

The build is correct and was not narrowed. `vendor/vector-ta-0.2.9-patched/build.rs::target_archs`
resolves, in order: `CUDA_ARCHS` (list) → `CUDA_ARCH` (**singular, yields
`vec![a]`, i.e. one architecture**) → `DEFAULT_TARGET_ARCHS = [80, 86, 89, 90]`,
intersected with `nvcc --list-gpu-arch`, plus `-gencode arch=compute_90,code=compute_90`
so a newer card JITs instead of failing. So **the correct instruction is to set
NOTHING**, and to narrow with the list form only.

`crates/neoethos-data/build.rs:28-32` already states the trap in its own header:
the old panic's remedy *"told the operator to export `CUDA_ARCH=sm_86` — which in
vector-ta's `target_archs()` takes the single-value branch and builds a
SINGLE-ARCHITECTURE fatbin that will not load on an A100. The error message
recreated the exact trap it existed to prevent."* `docs/vector-ta-cuda-wiring.md:75-85`
says **NEVER set `CUDA_ARCH=`** and is correct as written — it is not one of the
three.

These three are, with exact current text and exact replacement:

**1. `crates/neoethos-app/Cargo.toml:33-34`**

```toml
# CURRENT
    # in the build. Requires nvcc AND `CUDA_ARCH` set to the card's compute
    # capability (see crates/neoethos-data/Cargo.toml for the arch trap).
```
```toml
# REPLACEMENT
    # in the build. Requires nvcc. Do NOT set an arch: vector-ta builds ONE
    # fatbin covering sm_80/86/89/90 plus compute_90 PTX for forward JIT, so
    # the default artifact runs on every card we target. `CUDA_ARCH=` (the
    # SINGULAR form) narrows it to ONE architecture and the binary then fails
    # to load anywhere else — see crates/neoethos-data/build.rs:28-32. To
    # narrow deliberately for a faster iteration loop, use the LIST form,
    # e.g. `CUDA_ARCHS=86`.
```

**2. `crates/neoethos-data/Cargo.toml:107-111`**

```toml
# CURRENT (:110)
#   CUDA_ARCH=sm_86 CUDA_FAST_MATH=0 cargo build -p neoethos-data --features gpu-cuda
```
```toml
# REPLACEMENT
#   CUDA_FAST_MATH=0 cargo build -p neoethos-data --features gpu-cuda
#
# No arch variable. vector-ta compiles the multi-arch fatbin [80, 86, 89, 90]
# + compute_90 PTX by default, which covers the 3090 (sm_86) and every other
# card we target from ONE artifact. To narrow for a faster loop use the LIST
# form — `CUDA_ARCHS=86` — never `CUDA_ARCH=`, which builds a
# single-architecture fatbin that loads on nothing else.
```

The surrounding lines `:107-108` also need correcting: they say the build
*"[resolves] the target arch from the card that is actually present rather than
defaulting to a hardcoded `compute_89`"*. `neoethos-data/build.rs` **no longer
resolves an arch at all** — that probe was deleted precisely because it made the
build require a card on the build host. The sentence describes code that is gone.

**3. `crates/neoethos-data/src/core/hpc_ta.rs:1536-1541`** — an operator-facing
assert message, which is the worst of the three because it is read at the moment
the operator is about to type the command:

```rust
// CURRENT
        "IndicatorComputePolicy::RequireGpu was requested but this binary has no CUDA indicator \
         lane compiled in. Rebuild with `--features gpu-cuda` (and CUDA_ARCH set to the card's \
         compute capability), or use IndicatorComputePolicy::Auto/Cpu."
```
```rust
// REPLACEMENT
        "IndicatorComputePolicy::RequireGpu was requested but this binary has no CUDA indicator \
         lane compiled in. Rebuild with `--features gpu-cuda` — set NO arch variable, the \
         default fatbin covers sm_80/86/89/90 plus compute_90 PTX. (Never `CUDA_ARCH=`: the \
         singular form builds a single-architecture fatbin that will not load on another card. \
         To narrow deliberately, `CUDA_ARCHS=<list>`.) Or use IndicatorComputePolicy::Auto/Cpu."
```

#### Two lookalikes that are NOT defects — do not "fix" them

* `crates/neoethos-data/src/core/indicator_telemetry.rs:124` and `:296` mention
  `CUDA_ARCH`, but both are **historical**: `:124` explains that the constant
  used to be recomputed from it and is now re-exported from
  `vector_ta::cuda::module_loader::COMPILED_PTX_ARCH`, and `:296` is a test
  asserting a card-less build never reports a resolved arch. Neither instructs
  anyone to set anything.
* `crates/neoethos-gpu-cuda/build.rs:19`, `:63-65` uses **`NEOETHOS_CUDA_ARCH`**,
  a different variable in a different crate, defaulting to
  `compute_70,code=compute_70` — virtual PTX that the driver JITs for whatever
  card is present. That is the portable default, not the trap. It is a fourth
  hit on a `CUDA_ARCH` grep and it is fine.

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

**Filed 2026-08-10 by the main session. ⚠ NAME CORRECTED 2026-08-10 — this entry
called the control-plane binary `neoethos-codex` and that is WRONG.** The name
changed after the entry was written, because a crate called `neoethos-codex`
already existed and taking that name for a binary would have swapped one
collision for another. Verified against the manifests today:

| Thing | Crate | `[[bin]] name` | What it is |
|---|---|---|---|
| outbound sidecar | `mcp/` workspace (`mcp/Cargo.toml:10`, `:19`) | **`neoethos-mcp`** | the client the desktop app spawns, bundled beside it |
| inbound control plane | `crates/neoethos-mcp` (`Cargo.toml:34-35`) | **`neoethos-control-plane`** | MCP server over the localhost backend API |
| — | `crates/neoethos-codex` (`Cargo.toml:2`) | *(library)* | a SEPARATE, live crate: PKCE OAuth, token store, callback server. It is why the control-plane binary could not be called `neoethos-codex`. |

Everywhere in the tree already agrees on `neoethos-control-plane`
(`crates/neoethos-mcp/Cargo.toml:22`, `src/main.rs:1`, `:8`;
`neoethos-app/src/app_services/supervisor.rs:467`, `:477`, `:484`;
`server/auth.rs:78`; `server/mcp.rs:218`, `:225`; `docs/codex-control-plane.md:3`,
`:28`, `:29`). **This document was the last place carrying the dead name.**

The other half of W10 landed: `crates/neoethos-mcp` no longer declares
`[[bin]] name = "neoethos-mcp"`, so the repo can no longer produce two
identically-named executables. The remaining edit lives in `desktop/**`, owned by
another workflow, so it is recorded rather than applied.

**Status of this entry's three parts, re-verified 2026-08-10:**

* **Rename — DONE.** `crates/neoethos-mcp/Cargo.toml:35` reads
  `name = "neoethos-control-plane"`.
* **Message text — DONE**, by `64722aa7`. `supervisor.rs:462` records that it
  used to say *"is neoethos-mcp running?"*; `:481` now reads *"MCP sidecar not
  reachable. The process that must be running is …"* and `:477`/`:484` name both
  binaries and which is which. The two line numbers this entry cited (`:593`,
  `:625`) no longer hold that string.
* **Post-spawn identity probe — STILL PENDING.** This is the actual remaining
  work; see below.

**File:** `desktop/src-tauri/src/lib.rs` — `mcp_sidecar::start`, `:243-296`. The
bare-filename spawn is at `:244-248` (`bin_name`) and the launch at `:289`
(`Command::new(&exe)`), with no probe of what answered. Note the neighbouring
improvement that DID land: `NEOETHOS_MCP_PATH` — an env var that redirected which
binary this shell launched — is deleted (`:253-260`), so the search order is now
deterministic: beside the exe, then `resources/`.

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

**Why it still matters after the rename:** the rename stops the *repo* producing
a second `neoethos-mcp.exe`. It does nothing about an *installed* directory that
still holds a stale executable of that name from a build made before today — and
that directory is exactly where `:289` spawns from. The probe is what converts
"the wrong process is running and every MCP tool silently does nothing" into a
named ERROR.

The operator-facing message half is already done (see the status list above), and
it now names both binaries — sidecar `neoethos-mcp` from the `mcp/` workspace,
control plane `neoethos-control-plane` from `crates/neoethos-mcp` — which is what
prevents the next person conflating them.

**Also check when that crate is unlocked:** `.github/` and any installer script
that references `neoethos-mcp` by name, to confirm none of them meant the
control-plane binary. Note that `tauri.windows.conf.json` bundles the `mcp/`
sidecar deliberately — that one IS `neoethos-mcp` and is correct.

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

---

## `models.blend_*` — the live blend multipliers need a config recipient

**Filed 2026-08-10** by the shard that owns `crates/neoethos-app/**`,
`crates/neoethos-cli/src/main.rs` and `crates/neoethos-trader/**`.
Target file: `crates/neoethos-core/src/config.rs` (`ModelsConfig`), plus the
two `config.yaml` copies. Owned by another shard this wave.

### Why

`BlendConfig::from_config_values` (`neoethos-trader/src/blend_signal.rs:119`)
existed with **zero production callers**. Both live construction sites built the
struct by literal instead, so `gate_floor` 0.34 and `veto_below` 0.15 were
hardcoded onto the LIVE sizing path — `live_trading.rs` multiplies the gene's
per-trade risk by the confidence those two numbers produce. That is now fixed on
the code side: every production site goes through the constructor. What is still
missing is the YAML the constructor reads from, so the live path passes `None`
and lands on the shipped defaults. **The numbers do not change; the operator
still cannot see or set them.**

### Exact fields to add

In `ModelsConfig` (`config.rs`, next to `live_ml_gate` at :1031), flat
`blend_`-prefixed to match the surrounding style — NOT a nested `blend:`
sub-struct, which would add a DTO, a UI group and a `knob_catalog` section for
two scalars:

| Field | Type | Default | YAML path |
|---|---|---|---|
| `blend_gate_floor` | `f64` | `0.34` | `models.blend_gate_floor` |
| `blend_veto_below` | `f64` | `0.15` | `models.blend_veto_below` |

Defaults MUST be sourced as the literals `0.34` / `0.15` (core cannot depend on
`neoethos-trader`), and MUST stay equal to
`neoethos_trader::DEFAULT_BLEND_GATE_FLOOR` / `DEFAULT_BLEND_VETO_BELOW`. If you
prefer one source of truth, move the two constants into `neoethos-core` and have
`blend_signal.rs` re-export them — that is a strictly better end state, but it
touches a crate boundary, so it is a decision, not an edit.

Doc lines to paste above the fields:

```rust
    /// Floor on the ML agreement term in the live blend (`models.live_ml_gate`).
    /// A gene bar the ensemble is only lukewarm about still trades at THIS
    /// fraction of its size, so the validated gene edge is never gated to
    /// nothing by a lukewarm model. Range [0,1]; default 0.34. Out of range,
    /// non-finite, or below `blend_veto_below` ⇒ REFUSED back to the default and
    /// logged with both numbers (`BlendConfig::from_config_values`) — this
    /// multiplier scales every entry's risk.
    pub blend_gate_floor: f64,

    /// Effective-multiplier floor below which the live blend SKIPS the bar
    /// entirely (Flat, not confidence 0 — the sizing floor would otherwise open
    /// min volume). In `MlConfirm` it also vetoes when the raw ML `p_side` is
    /// below it. Range [0,1]; default 0.15. Must be <= `blend_gate_floor`, else
    /// every floored bar would be vetoed and the pair is REFUSED back to the
    /// defaults, loudly.
    pub blend_veto_below: f64,
```

### Full drag — all of it, or the knob is a decoration

1. `crates/neoethos-core/src/config.rs` — the two fields + the two `Default`
   entries (beside `live_ml_gate: false` at :2425).
2. `config.yaml` (repo root) — two lines under `models:`, next to
   `live_ml_gate`.
3. `desktop/src-tauri/resources/config.yaml` — the same two lines (that copy has
   `live_ml_gate: false` at :272).
4. `crates/neoethos-app/src/server/settings.rs` — the response struct (beside
   `live_ml_gate` at :112/:115), the patch struct (:153/:154), the apply arm
   (:755) and the read-back (:854). **This file is mine; tell me and I will make
   this edit, or make it yourself once the core fields exist — say which.**
5. `crates/neoethos-app/src/server/knob_catalog.rs` — one row each, so the
   operator can find them. **Mine as well; same offer.**
6. `crates/neoethos-mcp/src/params.rs` + `ops.rs` — the pattern at
   `params.rs:399` / `ops.rs:767` mirrors every settings knob into the control
   plane; these two belong there for parity.

### The reader is already written and waiting

`crates/neoethos-app/src/app_services/live_trading.rs` has two functions whose
entire body is the pending `None`:

```rust
fn operator_blend_gate_floor(settings: Option<&neoethos_core::Settings>) -> Option<f64>
fn operator_blend_veto_below(settings: Option<&neoethos_core::Settings>) -> Option<f64>
```

When the fields land, each becomes a one-liner —
`settings.map(|s| s.models.blend_gate_floor)` and `.blend_veto_below` — and
nothing else in that file moves. They are called from the `live_blend_cfg`
construction beside the `live_ml_gate` read, which is consumed by
`blend_decision` on every entry bar. **Do not add a second reader; use these.**

### What this permits once landed

The operator can raise `blend_gate_floor` (ML gets less authority to shrink a
validated gene entry) or lower it (ML gets more). He can raise
`blend_veto_below` to skip more marginal bars. He CANNOT set an inverted pair, a
value outside `[0,1]`, or a non-finite value — the constructor refuses those back
to the shipped defaults and logs both the configured and the used number. There
is no path by which a bad YAML value silently changes a live position size.

---

# `pattern_recognition` on the f64 device lane — DEFERRED, with the design

Written 2026-08-10 by the `crates/neoethos-data` owner. Every edit below lands
in `vendor/vector-ta-0.2.9-patched/**`, which that workflow does not own.

## The finding, confirmed at the cited code

* `vendor/vector-ta-0.2.9-patched/kernels/cuda/pattern_recognition_kernel.cu`
  holds **71 `*_f64` entry points** — a features kernel, a rolling-stats kernel,
  a doji predicate, five shared row kernels, ~62 per-pattern row kernels, and a
  `pattern_u8_to_f64_kernel` widener at `:7585`. They are written and they are
  compiled.
* `vendor/vector-ta-0.2.9-patched/src/cuda/pattern_recognition_wrapper.rs` —
  5,185 lines — contains **zero occurrences of the substring `_f64`**. Every
  buffer it declares is `DeviceBuffer<f32>` (`DevicePatternFeatures`,
  `DevicePatternRollingStats`) and every launch it issues resolves an f32 entry
  point. So the f64 half is unreachable: no caller can select it.
* `pattern_recognition` has no row in `cuda_f64::F64_KERNELS` (338 rows), and it
  is the one id among the four missing ones that owns real arithmetic.

## Assessment: it CANNOT be expressed through `F64KernelSpec`, and forcing it would break the struct for its other 338 users

`F64KernelSpec { indicator_id, kernel, input, first_valid }` models exactly one
thing: **one entry point, driven by a period list, producing one `double*`
matrix whose rows are periods.** Four separate mismatches, each independently
fatal:

1. **One entry point vs a sequenced plan.** `F64Kernel` is an enum with one
   `entry_point() -> &'static str` per variant. Patterns need a features launch,
   then rolling stats, then ~65 row launches that read those intermediates.
   There is no honest value for `entry_point()` here; any single name is a lie
   about what runs.
2. **Rows are periods, not patterns.** `IndicatorCudaDeviceRequestF64.periods:
   &[i32]` sets `rows`. `pattern_recognition` declares no window parameter at
   all — its rows are the 61 entries of `NATIVE_SUPPORTED_PATTERN_IDS`
   (`pattern_recognition_wrapper.rs:155`). Making `periods` mean "row selector"
   for one caller silently redefines it for all 338.
3. **`u8` out, not `double` out.** `IndicatorCudaSeriesF64` carries `HostF64` /
   `DeviceF64` only. The widener kernel exists, so a `double*` of 0.0/1.0 is
   *technically* producible — but the result would then cost 8 bytes per bar per
   pattern where the CPU path returns `IndicatorSeries::Bool`, and
   `hpc_ta::pattern_matrix_columns` (`hpc_ta.rs:786`) matches on `Bool` and
   returns `None` for anything else, taking the named-hard-error path. So the
   widened form is not even accepted by this crate's consumer.
4. **Intermediate device state.** Features and rolling stats are live buffers
   shared across the row launches. `compute_cuda_device_f64` has no vocabulary
   for a kernel that allocates and hands on intermediates.

Conclusion: **deferred.** `pattern_recognition` stays out of `F64_KERNELS`, and
the number stays **338 of 342** rather than quietly becoming 339.

## The shape the registry would need

A SECOND registry beside `F64_KERNELS`, sharing its upload and first-valid
vocabulary and nothing else:

```rust
/// One indicator whose device form is a sequenced PLAN producing a u8 matrix
/// whose rows are named, not periods.
pub struct F64MatrixKernelSpec {
    pub indicator_id: &'static str,
    /// The launches, in dependency order. Each names its own entry point.
    pub plan: &'static [F64MatrixStage],
    pub input: F64InputKind,          // Ohlc4 for pattern_recognition
    pub first_valid: F64FirstValidRule,
    /// Row r of the output IS this id. Length == rows. No period anywhere.
    pub row_ids: &'static [&'static str],
}

pub enum F64MatrixStage {
    /// Fills the shared f64 feature buffers.
    Features { entry: &'static str },
    /// Fills the shared f64 rolling-stat buffers.
    RollingStats { entry: &'static str },
    /// Writes rows `first..first + count` of the u8 matrix.
    Rows { entry: &'static str, first: usize, count: usize },
}

pub const F64_MATRIX_KERNELS: &[F64MatrixKernelSpec] = &[ /* pattern_recognition */ ];

pub fn compute_cuda_matrix_f64(
    engine: &CudaF64Indicators,
    req: IndicatorCudaMatrixRequestF64,
) -> Result<IndicatorCudaMatrixOutputU8, ...>;
```

with `IndicatorCudaMatrixOutputU8 { series: Host/DeviceU8, rows, cols, row_ids }`
so `hpc_ta::pattern_matrix_columns` keeps its `Bool` contract unchanged. The
existing f32 wrapper is the working template for the plan — it already sequences
exactly these stages; the port is "duplicate the launch sequence against the
`_f64` symbols and the f64 buffers", not new numerics.

## The second, independent reason this is deferred

Even a finished vendor change would NOT put `pattern_recognition` on a card for
this repo. `crates/neoethos-data` has a device lane for **one** stage only: the
multi-period sweep (`hpc_ta::compute_multi_period_columns`, the sole caller of
`GpuIndicatorEngine`). `pattern_recognition` is in the BASE vocabulary
(`all_indicators.rs:213`), and the base vocabulary and the extended sweep have
no device lane at all. Whoever picks this up must land both halves or the work
is still unreachable — which is the exact failure the f64 kernels are already
sitting in.

Also worth knowing before starting: the CPU emits **62** pattern columns
(`PATTERN_RUNNERS.len()`, `pattern_recognition.rs:1146`) and the device native
set is **61** (`NATIVE_SUPPORTED_PATTERN_IDS`). A device path therefore covers
61 of 62 and the 62nd must stay on the CPU, named — not dropped.

---

# `docs/vector-ta-cuda-wiring.md` is stale about the device table — exact edits

Written 2026-08-10 by the `crates/neoethos-data` owner, who does not own `docs/`
beyond this handoff file. The code changed today; these lines still describe the
old behaviour, which makes them claims rather than prose.

## Edit 1 — the heading and the table membership (around line 188)

REPLACE:

```
### Ten indicators on the device, eight on the CPU — and why two were nearly wrong

`hpc_ta::MULTI_PERIOD_IDS` has eighteen entries.
`gpu_indicators::GPU_SWEEP_SPECS` has ten: `sma`, `ema`, `rsi`, `roc`, `mom`,
`atr`, `adx`, `willr`, `cci`, `mfi`. The other eight (`stoch`, `macd`,
`bollinger_bands`, `keltner`, `supertrend`, `tsi`, `obv`, `vwap`) have a
multi-output or non-period device contract and stay on the CPU — enumerated up
front and reported as `CpuIndicatorNotPortable`, never discovered by a failed
launch mid-run.

The ten share one parameter contract.
```

WITH:

```
### Every reachable indicator on the device, the multi-output five on the CPU — and why two were nearly wrong

`hpc_ta::MULTI_PERIOD_IDS` has eighteen entries. `gpu_indicators::GPU_SWEEP_SPECS`
now holds every one of them that is SINGLE-OUTPUT: `sma`, `ema`, `rsi`, `roc`,
`mom`, `atr`, `adx`, `willr`, `cci`, `mfi`, `tsi`, `obv`, `vwap`.

The remaining five — `stoch`, `macd`, `bollinger_bands`, `keltner`,
`supertrend` — stay on the CPU and are reported as `CpuIndicatorNotPortable`,
enumerated up front, never discovered by a failed launch mid-run. Note what that
label does NOT mean: all five HAVE rows in `cuda_f64::F64_KERNELS`. They emit
ZERO columns on EITHER lane because `hpc_ta` calls `compute_cpu` with
`output_id: None`, which returns `Err(InvalidParam)` for a multi-output
indicator (`cpu_batch.rs:2185`) and is swallowed at `hpc_ta.rs:291`. The device
kernel is not the missing piece; the CPU call is.

`vwap` was the last id to move (2026-08-10). It had been withheld because
vector-ta carried a second CPU implementation, `vwap_row_scalar_pv`, reachable
only from the `Kernel::Scalar` arm and accumulating `price * volume` with two
roundings where `vwap_scalar` uses one `mul_add` — so there was no single CPU
answer for a device result to match. vector-ta deleted it; both batch arms now
call `vwap_row_scalar`, and `WITHHELD_PENDING_CPU_SELF_CONSISTENCY` is `&[]`.

Do not restate the size of this table here again. It is asserted by
`gpu_indicators::tests::every_reachable_multi_period_id_with_an_f64_kernel_is_claimed`,
which fails in both directions.

They share one parameter contract.
```

## Edit 2 — the launch count (the "One launch per period" section)

That heading and its body describe a lane that no longer exists. The f64 lane
takes the period LIST directly (`GpuIndicatorEngine::sweep_periods`), so the
five periods cost ONE launch, the output is exactly `rows × n × 8` bytes, and
nothing computed is thrown away. Retitle to "One launch for the whole period
list" and rewrite the body from `gpu_indicators.rs::sweep_periods`, whose doc
comment already states the current design and the measurement that motivated it.

---

# Wave 3 — close the sub-struct bypass: the last unsealed section (2026-08-10)

Appended by the shard that owns `crates/neoethos-core/src/config.rs` and
`crates/neoethos-core/tests/**`. Forbidden to that shard:
`crates/neoethos-core/src/domain/**`.

## What already landed (context, not work)

`SystemConfig`, `RiskConfig`, `NewsConfig` and `AppRuntimeConfig` are sealed
against a second load path: `#[serde(remote = "Self")]` makes the derive emit an
*inherent* `X::deserialize` instead of `impl Deserialize for X`, so
`from_str::<X>` and any `#[derive(Deserialize)]` struct that merely holds one no
longer compile. The single caller is `SettingsWire`'s
`#[serde(deserialize_with = "X::deserialize")]`. See the SUB-STRUCT SEAL block
in `config.rs`, above `pub use load_seal`.

Why it matters, measured: the same bytes `risk: {preset: the5ers}` gave
`daily_drawdown_limit` **0.032 / total 0.042** through `Settings` and **0.040 /
0.070** through the bypass, because the bypass never ran `reconcile_preset`. 💰
The bypass got the LOOSER limit under the correct firm label.

## The one edit left, and why it is here

`ModelsConfig` is **NOT** sealed. Five call sites deserialize it directly, all
inside `#[cfg(test)]` blocks of `crates/neoethos-core/src/domain/**`, which the
sealing shard does not own. Adding `remote = "Self"` without moving them first
is a **build break**, so it was not done.

`ModelsConfig` carries no preset re-derivation, so what its bypass skips is the
top-level retired-key prune and the provenance record — not a money number.
That is why it is a follow-up and not a blocker.

### Edit 1 — `crates/neoethos-core/src/domain/demo_gate.rs`

Two sites, both in the `#[cfg(test)] mod` at the end of the file. Replace the
direct sub-struct parse with a `Settings` parse (which is sealed, in-memory, and
needs no file), then read `.models`. Re-indent the YAML one level under
`models:`.

`demo_gate.rs:280-282` — `a_config_without_the_key_keeps_the_previous_thresholds`

```rust
// before
let models: crate::config::ModelsConfig =
    serde_yaml_ng::from_str("ml_models: [lightgbm]\n").expect("legacy config deserialises");
// after
let settings: crate::config::Settings =
    serde_yaml_ng::from_str("models:\n  ml_models: [lightgbm]\n")
        .expect("legacy config deserialises");
let models = settings.models;
```

`demo_gate.rs:287-294` — `an_operator_set_demo_bar_reaches_the_gate_and_changes_the_verdict`

```rust
// before
let yaml = "\
ml_models: [lightgbm]
demo_forward_gate:
  min_demo_trades: 400
";
let models: crate::config::ModelsConfig =
    serde_yaml_ng::from_str(yaml).expect("operator config deserialises");
// after
let yaml = "\
models:
  ml_models: [lightgbm]
  demo_forward_gate:
    min_demo_trades: 400
";
let settings: crate::config::Settings =
    serde_yaml_ng::from_str(yaml).expect("operator config deserialises");
let models = settings.models;
```

### Edit 2 — `crates/neoethos-core/src/domain/promotion_gate.rs`

Three sites, same shape, same `#[cfg(test)] mod`.

* `:461-463` `a_config_without_the_promotion_gate_key_keeps_the_previous_thresholds`
  — `"ml_models: [lightgbm]\n"` → `"models:\n  ml_models: [lightgbm]\n"`.
* `:468-477` `an_operator_set_threshold_survives_a_yaml_round_trip` — indent the
  `ml_models` / `promotion_gate` block one level under `models:`.
* `:524-526` `a_disabled_gate_set_from_config_actually_disables_the_gate` —
  `"promotion_gate:\n  enabled: false\n"` →
  `"models:\n  promotion_gate:\n    enabled: false\n"`.

In each, bind `let settings: crate::config::Settings = serde_yaml_ng::from_str(..)`
and then `let models = settings.models;`. Nothing else in those tests changes:
they still assert exactly the same fields, and they now assert them through the
path a real run takes, which is strictly stronger.

### Edit 3 — seal it (in `config.rs`, owned by the appending shard)

Once Edits 1 and 2 are in, four mechanical changes finish the job. They are
listed here so whoever applies the above can hand it back:

1. `ModelsConfig`'s attribute becomes
   `#[serde(remote = "Self", default, deny_unknown_fields)]`.
2. Add `impl Serialize for ModelsConfig` delegating to
   `ModelsConfig::serialize(self, serializer)`, beside the other four.
3. `SettingsWire`'s `models` field gains
   `#[serde(deserialize_with = "ModelsConfig::deserialize")]`.
4. Delete the "`ModelsConfig` is NOT sealed" paragraph from the SUB-STRUCT SEAL
   block, and flip its flag to `true` in `SUBSTRUCTS` in
   `crates/neoethos-core/tests/config_single_load_path.rs`.

`a_sealed_substruct_has_no_deserialize_impl_to_route_around` fails until step 4
is done, and `no_second_caller_of_the_inherent_parsers` fails if the attribute
in step 3 is ever deleted. Neither can be satisfied by a comment.
