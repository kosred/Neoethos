# Audit Status — 2026-08-09

**This document changed nothing.** No code was edited, no test was run, no file was
deleted by the workflow that produced it. It is a reconciliation of the 323-item audit
ledger against the working tree as it stood on 2026-08-09.

**Everything marked FIXED below is UNCOMMITTED.** 323 paths are dirty in `git status`
(31 of them untracked). `git log` shows none of today's work. A `git reset --hard`
erases every fix in this report. Commit before doing anything else.

---

## 1. THE SCOREBOARD

Of **323** items:

| Status | Count | Meaning |
|---|---:|---|
| **FIXED** | **81** | verified in the current files, file:line given |
| **PARTIAL** | **21** | one half landed, the other did not |
| **OPEN** | **191** | untouched, defect still visible at the cited line |
| **WAS_WRONG** | **20** | the audit or a refuter established the finding is false |
| **SUPERSEDED** | **9** | overtaken by a later decision; the item no longer means anything |
| **IN_FLIGHT** | **1** | `vendor/vector-ta-0.2.9-patched`, still being written |

**Money path only — the number that decides whether he can trade. Of 84 money-path items:**

| Status | Count |
|---|---:|
| **FIXED** | **23** |
| **PARTIAL** | **9** |
| **OPEN** | **47** |
| **SUPERSEDED** | **3** (operator scope rulings, not defects) |
| **WAS_WRONG** | **2** |

**The plain answer to "can I trade": yes, more safely than yesterday, and with four
specific numbers wrong.** Today closed the three gaps that could have lost the account
outright — the risky-mode kill switch now exists and is called, the demo forward-test
gate no longer scores the wrong account, and `require_stop_loss` is finally enforced on
both order routes. What remains open is not a missing brake; it is a set of brakes set at
the wrong number and one new backtest/live divergence created this morning.

**Three corrections to the shard reports were applied before writing this document,
verified by reading the code:**

1. **#71 (cost band) is PARTIAL, not FIXED.** `discovery.rs:7378-7381` says in its own
   words *"it classifies, it does not reject"*, and `discovery.rs:7718` binds the verdict
   as `_cost_band` and drops it. The comment correcting this sits fifteen lines above the
   evidence the shard cited as proof of gating.
2. **#290 (Default-vs-shipped test) is PARTIAL, not OPEN.** The test exists —
   `crates/neoethos-core/tests/shipped_config_matches_defaults.rs` (untracked, new today)
   — but `MUST_MATCH_DEFAULT` at `:32-36` holds three search-shaping keys and none of the
   five money divergences.
3. **#74 and #208 describe one change and disagreed about it.** #74 is now PARTIAL: the
   trailing default flipped OFF in discovery and did not flip in live.

---

## 2. WHAT HE IS EXPOSED TO TODAY

Every OPEN money-path item, worst first. "Exposed" means: a number that governs real
money is wrong right now, on this machine, with this config.

### 2.1 — Live trails, discovery no longer does. Created TODAY. (#208 / #74)

`crates/neoethos-app/src/app_services/live_trading.rs:1220-1275` — the trailing block runs
**unconditionally** on every open position, reads
`neoethos_core::config::DEFAULT_TRAILING_MIN_LOCK_PIPS` at `:1241`, and hardcodes the +1R
trigger and 1×SL trail distance inline.

`crates/neoethos-core/src/config.rs:1666-1672` — `ExitPolicyConfig::default` now ships
`trailing_enabled: false`, and the search reads it (`strategy_gene.rs:867`).

**The consequence.** Every strategy validated from today onward is scored with the
take-profit reachable, then traded live with the stop pulled to break-even the instant
price touches +1R. That is the exact mechanism `config.rs:1623-1631` measured as capping
realised payoff at **1.08 against a configured floor of 2.0** — the sixteen-month answer.
Live gives back the wins the backtest was scored on.

**And the code now lies about it.** `live_trading.rs:1221-1223` still asserts
*"eval.rs hardcodes break-even + trailing ALWAYS ON"*. That was true at breakfast. It is
false now. Anyone reading that comment reasons wrongly.

**Effort: hours.** Read `models.exit_policy` in the live loop instead of the constant.
This is the single highest-value open item in the report.

### 2.2 — The drawdown breakers are set 6 and 6 points too wide (#214, #213, #269)

`config.yaml:158` `daily_drawdown_limit: 0.10000000149011612`
`config.yaml:159` `total_drawdown_limit: 0.20000000298023224`

Those f32-widened-to-f64 fingerprints are forensic proof: they came through
`crates/neoethos-app/src/server/risk.rs:170-173`, which writes the firm's **raw** ceiling,
not `RiskConfig::default`'s buffered one (`config.rs:561`, `* 0.7`).

**On a £1,000 account:** the total-drawdown breaker at `live_trading.rs:1510-1520` arms at
**£200 lost**, not the £140 the code's own design intends. The daily breaker at `:1532-1546`
arms at **£100**, not £40. The guard fires at the moment the prop firm would already have
failed him, instead of 30% short of it — and one click on any preset in the UI puts it
back every time.

**Effort: trivial** to correct the two values; **hours** to fix the writer so it stops
re-introducing them.

### 2.3 — A fresh install trades a different product than this machine (#245, #246, #248, #249, #250)

Root vs seed, measured today. Root 383 keys, seed 233, **0 keys unique to the seed**,
**exactly 5 value disagreements** — and all five are money.

| Knob | `config.yaml` | `desktop/src-tauri/resources/config.yaml` |
|---|---|---|
| `account_currency` | GBP (`:96`) | **USD** (`:9`) |
| `discovery_mode` | risky (`:535`) | **prop_firm** (`:125`) |
| `prop_firm_rules` | false (`:165`) | **true** (`:36`) |
| `daily_drawdown_limit` | 0.100000001 (`:158`) | 0.04 (`:31`) |
| `total_drawdown_limit` | 0.200000003 (`:159`) | 0.07 (`:32`) |

**The currency one is the dangerous one.** A fresh install computes every pip value, every
risk-per-trade lot size and the prop-firm daily-loss check in **USD against a GBP account**.
At ~1.27 that mis-sizes every position by ~27%. Against the 30% risky ceiling that is a
**38% real risk-per-trade he never chose**.

**Effort: five one-line edits.** Cheapest money fix in the report.

### 2.4 — The UI says Risky and the search runs PropFirm (#267)

`crates/neoethos-app/src/server/settings.rs:397-412` validates `tradingMode`, then writes
`settings.system.trading_mode = mode` — and nothing else. `models.discovery_mode` is never
touched. Discovery reads `models.discovery_mode`.

**Consequence:** he clicks Risky, the banner says Risky, and the next run applies the
PropFirm arm of `apply_mode_overrides` (`discovery.rs:925`) — permissive DD 0.50,
`min_trades_per_day` 0.001. He ranks candidates under rules he believes he switched off.
**Effort: trivial** (one more assignment).

### 2.5 — Backtests are scored at a flat spread the broker does not charge (#69)

The wire is complete: `discovery.rs:1726` builds `SessionSpreadProfile` at the single point
every discovery setting flows through, and `config.rs:636` correctly **refuses** a
partially-configured curve. But all three keys ship `null` — `config.yaml:222-224` and
seed `:69-71`.

**Consequence:** every backtest charges the flat 1.5 pips (`config.yaml:204`) at 03:00 Tokyo
and at the London open alike. Any strategy the search selects whose trades cluster in the
Asian session was priced optimistically and will underperform live by the difference. The
run already tells him the Asian trade count and PnL share it cannot vouch for
(`discovery.rs:7595-7612`) and names the fix: average the hourly means already in
`spread_stats.json` over 22-07 / 07-16 / 16-22 UTC.

**Effort: trivial — three measured numbers, no code.**

### 2.6 — Risky mode can size to 50% while config says 30% (#209, #210)

`risk.risky_max_risk_per_trade: 0.30` (`config.yaml:156`) has exactly one reader and it is
the **search** (`discovery.rs:820`). The live ladder is bounded by
`RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION = 0.50` (`domain/risky_mode.rs:66`), and
`live_trading.rs:1664-1671` takes `base_risk` from the stage ladder — `risk.risk_per_trade`
is only the degenerate-input fallback.

**Consequence:** twenty percentage points of the account per trade, with nothing
reconciling the two numbers. On £1,000 that is £300–£500 at risk per entry. It is now
bounded by the kill switch and the 55% pre-send ceiling landed today, so this is a
*sizing-vs-config* defect rather than an unbounded one. **Effort: hours; decision-required**
on which number wins.

### 2.7 — The kill-zones knob is one-sided (#75 / #217)

`discovery.rs:1738` — `kill_zones_enabled: true,` a literal, in the one settings-template
function every discovery lane flows through. The fields on either side of it read from
`config.`; this one does not. Live reads `risk.kill_zones_enabled` (`live_trading.rs:649`).

**Consequence:** every validated backtest excluded out-of-session hours. Turn the live knob
off and the bot trades hours no backtest ever priced, and holds through weekend gaps.
The knob can only move live **away** from what was validated, never toward it, and its
description does not warn him. Defaults agree today, so exposure is one toggle away.
**Effort: trivial** — read the config field there.

### 2.8 — Nothing validates the trades the card produces (#57)

`crates/neoethos-search/src/gpu_native/trade_invariants.rs:48` `check_trade_invariants` —
written, tested at `:243/:255/:277/:305`, **never called from production**.

**Consequence:** the card produces the trades that produce the metrics that rank the
strategies he deploys. Today nothing checks that a device trade's exit follows its entry,
or that its PnL sign agrees with its direction. A kernel regression does not fail — it
*ranks*. The first signal would be live results diverging from backtest.
**Effort: hours.** The checker is 48 lines from the call site.

### 2.9 — The cost band is measured and then thrown away (#71) — money, PARTIAL

`discovery.rs:7395-7415` computes `CostBandVerdict` at both cost edges and counts it
run-level. `discovery.rs:7718` binds it `_cost_band` and drops it at the export boundary.

**Consequence:** a gene profitable **only at the optimistic 1.6-pip edge** lands in
`live_portfolio.json` indistinguishable from one robust across the whole band. The number
exists; it never reaches him. The code names the follow-up at `discovery.rs:7390`.
**Effort: hours** — carry the verdict into the portfolio struct.

### 2.10 — The demo gate measures the wrong drawdown (#197)

`live_gate.rs:171` takes `query_equity(&data_dir, None, None)` — the **account-wide** curve
— computes `stats.max_drawdown_pct` from it at `:187`, and compares it at `:199` against
**one strategy's** `quality.json`.

**Consequence:** he runs several engines plus manual orders on one cTrader account. The
account curve's drawdown is their union, so a strategy whose own backtest DD was 11.8% is
measured against a curve that easily shows 25% — refused though it qualified. On a quiet
account the reverse: an unqualified strategy is admitted. Neither error is visible; the log
at `:175-185` reports counts, not whose drawdown it measured. **Effort: days.**

### 2.11 — The margin-call feed has no reader (#238)

`ctrader_messages.rs:875` `build_margin_call_list_request` — zero callers outside the file.
cTrader will tell the account it is in margin call and nothing asks. Every breaker keys off
balance and realised P&L (`live_trading.rs:1510`, `:1532`); an **unrealised** margin
emergency reaches him only if he happens to be looking at the broker's own platform.
**Effort: hours.**

### 2.12 — A culled loser comes back after the next discovery run (#218, #219)

`strategy_blacklist.rs:69-71` fingerprints the **file bytes**. Re-running discovery writes
a byte-different artifact for the same gene, so a strategy auto-culled after six
consecutive losses is eligible again with a clean record. And discovery never consults the
blacklist at all — zero hits for `blacklist` in `crates/neoethos-search/src/`. The
rediscovery request fired at `live_trading.rs:1209` can re-derive the shape that just lost
real money. **Effort: hours each.**

### 2.13 — The replay screen is not replaying his strategies (#224–#229)

`POST /autonomous/replay` → `replay_symbol_from_dir`, whose own doc
(`data_replay.rs:50-51`) says *"momentum stub signal + permissive risk gate + mock
execution"*. A $10,000 starting balance (`engine.rs:32`), fills at the mark with zero
spread/commission/slippage (`execution.rs:50-56`), a 0.5%-of-price bracket
(`decision.rs:37`) — on EURUSD that is a 54-pip stop against the GA's 6–20 — and no
trailing anywhere in the crate. **The numbers on his Replay screen are not his numbers.**
**Effort: hours per item, days for the block.**

### 2.14 — The remaining money OPENs, one line each

| # | Defect | file:line | Exposure |
|---|---|---|---|
| #142 | `NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE` documented as required by `risk_gate::prop_firm_pre_trade_check` — **that function does not exist anywhere in the repo** | `env_overrides.rs:56-65` | the exact lever he would pull against the known 192× EURJPY pip-value inflation, wired to nothing |
| #143 | retired `NEOETHOS_PROP_FIRM_PRESET` still reported as an *active override that changes runtime behaviour* at every startup of all three binaries | `env_overrides.rs:199-201` vs `config.rs:516-521` | a stale env var reassures him his tighter preset limits are in force while §2.2's 20% runs |
| #137 | 847-line `RiskManager` with `check_trade_allowed`, prop-firm DD and revenge-trade logic — all three constructors inside its own test module | `domain/risk.rs:332`, `:799/:808/:829` | `risk.daily_dd_warning_pct` and `risk.challenge_mode` reach only this file; setting them does nothing. It is also the decoy that made #204 survive |
| #144 | `symbol_metadata.rs:725` still reads a raw `env::var` string literal, bypassing the registry | `env_overrides.rs:141` | symbol metadata is the denominator of every position size; a registry rename silently stops his override applying |
| #294 | thirteen Settings fields with **no reader at all**, seven of them RiskConfig | `tests/config_has_recipient.rs:126-234` | seven risk knobs he can set from the UI that change nothing |
| #206 | four `risk.trailing_*` fields, zero readers, now formally marked SHADOWED DUPLICATE of `models.exit_policy.*` | `config.rs:417-419`, `:428` | three names for one concept; the one that looks like the risk knob is the dead one |
| #289 | `ModelsConfig::default()` re-arms the export gates the operator deliberately disarmed on 2026-06-06 | `config.rs:2266-2267` vs `config.yaml:536-537` | any `Settings::default()` path — including the desktop fallback at `lib.rs:65` — silently changes which strategies may reach live |
| #125 | desktop installs a bare relative `"config.yaml"` and swallows the load failure with `unwrap_or_else(Settings::default)` | `desktop/src-tauri/src/lib.rs:52`, `:65` | a config-miss becomes a silent policy change (see #289). **And a money gate landed today now reads its setting through this path** (`orders.rs:63-75`) |
| #299 + #310 | three numerically divergent artifacts (`tide` best_loss 1,308,811.5; `tide_nf` 51,699,690; `sac` final_alpha 5.69e9, healthy peers ~0.48) sit in `DEFAULT_BOOTSTRAP_EXPERT_NAMES` | `bootstrap.rs:118/119/140` | harmless **only** because `live_ml_gate` is false and `expert_weights` is permanently empty. There is no sanity check between "artifact on disk" and "artifact votes". **Decide these two together, never separately.** |
| #232 | `BlendConfig` gate_floor 0.34 / veto_below 0.15 are literals in the live sizing path | `blend_signal.rs:79-83` ← `live_trading.rs:1697-1700` | dormant while the ML gate is off; two numbers that scale every entry's risk with no config recipient |
| #191 | `place_pending` still has no bracket rule when `require_stop_loss` is OFF | `orders.rs:195-216` | a resting limit order can fill into a naked position with no override flag ever requested |
| #199 | `amend_position_sltp_blocking` is the one order-path call not bound to the admitted environment | `broker_api.rs:1279-1290` | a Demo→Live flip mid-iteration sends a stop-modification to the other environment; the call is `let _ =` fire-and-forget so a rejection is not even logged |
| #115/#116 | two disjoint preset vocabularies; the safety-posture presets have no apply path | `knob_catalog.rs:1025`, `server/risk.rs:139-141`, `Settings.tsx:38-42` | the catalog advertises Conservative (0.5% risk, require_stop_loss on) and nothing in the product can put him there — which is how §2.2 happened |
| #119 | no credential form exists anywhere in `desktop/src` | `broker_control.rs:51/:82` | nil today (credentials are baked in), but a revoked client_id locks him out of his own broker until someone rebuilds the binary |
| #265 | three starting balances for one concept: 10,000 / 100,000 / 10,000 | `config.rs:528`, `:1767`, `engine.rs:32` | the GPU kernel's compounding denominator is the 100,000 one — every percentage return the search ranks on is computed against a balance 10× his real account |
| #293 | `POST /settings/raw` validates schema, never values — it bypasses `Settings::from_yaml` so `validate_safety_bounds` never runs | `server/settings.rs:255`, `:270` | a hand-edited YAML with an out-of-range risk value is accepted and written |
| #35/#36 | the CubeCL f32 lane is **54% wrong at 200k bars** and says so in its own file; unreachable only while the prototype-B *runtime probe* succeeds | `cubecl_eval.rs:4856-4877` | on any box where the probe fails, strategies are ranked on a net profit less than half the truth, and nothing asserts the intended CPU fallback |

---

## 3. THE PARTIALS — 21 items

The project's signature failure: Phase A lands, Phase B never happens. **Six were created
today.** A partial made this morning is cheap now and expensive in a month.

### 3.1 — Created TODAY

| # | M | In | Missing | To close |
|---|:-:|---|---|---|
| **#71** | ● | cost band measured at both edges, counted run-level (`discovery.rs:7395-7415`) | verdict dropped at the export boundary — `discovery.rs:7718` binds `_cost_band` | hours — carry it into the portfolio struct (`discovery.rs:7390` names it) |
| **#74/#208** | ● | trailing default OFF in discovery, config-driven (`config.rs:1670`, `strategy_gene.rs:867`) | live still trails unconditionally off a constant (`live_trading.rs:1220-1275`) | hours — see §2.1 |
| **#69** | ● | full wire, single point, refuses partial curves (`discovery.rs:1726`, `config.rs:636`) | all three curve keys ship `null` in both YAMLs | trivial — three measured numbers |
| **#68** | | writer, chunking, ledger manifest (`discovery.rs:7191`, `discovery_ledger.rs:98`) | **nothing reads them** — `discovery.rs:2154` says so in its own words | days — DSR/PBO/CSCV, the reason it was persisted |
| **#90** | | `IndicatorComputePolicy` enum + resolver genuinely wired (`hpc_ta.rs:127`, `:79`) | `set_indicator_compute_policy` (`:93`) has **zero callers**; its own doc calls it "the seam the operator's Settings plugs into" | hours — call it where Settings resolves |
| **#166** | | `with_default_config` deleted, tests re-pointed | `voting_expert_count` / `experts_unused_for_voting` still tests-only; the app logs `loaded_count()` (`live_trading.rs:677-681`), not the voting count | trivial — two values into the existing `info!` |
| **#290** | ● | the Default-vs-shipped test **exists** (`tests/shipped_config_matches_defaults.rs`) | `MUST_MATCH_DEFAULT` (`:32-36`) covers three search keys and **none of the five money divergences** | trivial — six lines added to an array |

### 3.2 — Older partials

| # | M | In | Missing | To close |
|---|:-:|---|---|---|
| #20 | ● | kernel writes slot 7 (`prototype_b_population.cu:1380`); `PHASE1_GPU_SIZING_PORTED = true` (`eval.rs:1955`) | the const is still named `RESERVED_INDEX_7`, and `eval.rs:277`/`:1265` still assert the GPU lane is disabled — contradicted by line 1955 of the same file | trivial — rename + two comments |
| #87 | | silent column drop closed by the typed ledger (`hpc_ta.rs:595-660`) | `max_indicator_warmup` still has zero production callers while **three** doc comments claim it gates the sweep | hours — WIRE it (refuter's ruling), do not delete |
| #135 | | core `domain/portfolio.rs` deleted with its comparison table | the survivor `neoethos-search/src/portfolio.rs` is still exported and still dead, now with nothing recording that | trivial |
| #144 | ● | three call sites migrated to the registry | `symbol_metadata.rs:725` raw literal; `broker_config.rs:164` local const; the Phase-A header at `env_overrides.rs:29-37` no longer describes reality | trivial |
| #168 | ● | `abstain_below_confidence` deleted | four knobs still unsettable — `build_ensemble_for_symbol_with_config` has zero non-test callers; `expert_weights` permanently empty ⇒ all 33 experts at weight 1.0 | hours; this is model-verdict Option C |
| #191 | ● | `require_stop_loss` enforced on pending orders (`orders.rs:225-242`) | no bracket rule at all on pendings when the setting is off; `NewPendingOrderBody` has no `risky` field | trivial |
| #199 | ● | `resolve_creds_expecting` closes the check/send window for four order paths | `amend_position_sltp_blocking` (`broker_api.rs:1279-1290`) still calls bare `resolve_creds()` | trivial |
| #228 | ● | same-side signal now holds the position (`decision.rs:105-106`) | still closes on Flat (`:78-83`) and on reversal (`:98-103`); `eval.rs` has neither exit reason | hours |
| #235 | | eight of the eleven builders acquired real callers via D2 | five remain callerless; three of those are the KEEP-and-WIRE items ⇒ deletable set is **two**, not eight | trivial |
| #244 | | seed grew 219→233 today, closing 20 gaps | 150 root keys still absent from the seed, silently taking Rust Default | hours |
| #279 | | `RAYON_NUM_THREADS` confirmed dead in `eval.rs:535` (orphaned `from_env`) | it is **live** at `tree_models/config.rs:119` on every tree train — deleting the var on the eval.rs finding breaks thread control | trivial (a note, not code) |
| #283 | | the scanner no longer false-passes on shadowed names | 146 duplicate field names still exist; the guard just stopped lying about them | days |
| #296 | | `export_onnx` gone entirely with D4 | `regime_router_enabled` (yaml true / default false) and `multi_resolution_enabled` (yaml false / default true) still contradict | trivial |
| #302 | | the flat-accuracy finding stands | it understates: the deep models collapse to exactly **two** values, 0.621756 or 0.378244 — degenerate single-class predictors, not weak learners | decision |

---

## 4. STILL OPEN, BY CRATE

Pick a wave from here. Effort is the shard verifiers' estimate, checked against the code.

### `crates/neoethos-search` — 53 open

- **D1 was not run.** Items 1–24 and 54–64 are byte-for-byte as the audit found them:
  `parity.rs` 324 lines (`:1`), `genetic/regime_labels.rs` 522 (all eight exports
  re-grepped individually, all unreferenced), `gpu_native/ranking.rs` 263 (`:27`),
  `portfolio.rs` 487 (`:23`), `strategy_db.rs` 238 (`:210`) plus a bundled-DuckDB C++
  dependency (`Cargo.toml:47`), `checkpoint.rs` 508 — **zero hits for `checkpoint::`
  anywhere, not even a test**. `scoring/named.rs:274/:308/:345` three dead named scores kept
  alive by a fn-pointer test (`scoring/mod.rs:119-133`). **Effort: trivial each; one wave.**
- **Two HOLDs resolved.** #58 → **HOLD STANDS, delete refused**: `scripts/gpu-bench/preflight.sh:116-118`
  runs `cargo test -p neoethos-search --features gpu-cuda gpu_native::` under
  compute-sanitizer, and `prototype_b_mirror`'s module path is `gpu_native::prototype_b_mirror::*`,
  so the filter **does** select its tests. Deleting it silently shrinks the memcheck probe
  on the rented box. #60 → half-wrong: one declaration *is* exercised
  (`prototype_b_engine.rs:952`).
- **The preflight is decorative.** `capability.rs:211-213` returns `Ok(())` before examining
  a stage, because #35's escalation pins every card-present run to `AllowCpu`
  (`backend.rs:64-68`, `:198`). The manifest it would have read (`capability.rs:88`) is
  staler than "3 of 16": `PopulationEvaluation` still names CubeCL as the strict-GPU engine
  when production routes prototype B; `QualityScreen` is still `CpuOnly` while
  `discovery.rs:7090-7112` runs it as GPU scenarios. **Effort: hours.**
- **`WindowEvaluation::trades` fast path is inert** (`validation.rs:1391` — the only
  production construction hardcodes `trades: None` at `discovery.rs:3022`), and the readback
  half `trades_from_outcomes` has zero non-test callers (`device_trades.rs:29`). The cost
  is measured in-repo: **191,800 calls, 45% of a run** (`validation.rs:1382-1384`).
  **Effort: days — the largest single performance item in the report.**

### `crates/neoethos-data` — 15 open

`cross_pair_features.rs` 552 lines with exactly one external mention (`core/mod.rs:2`);
`loader.rs:62` `resolve_path_to_vortex` with 11 hits all inside its own file **and a header
at `:32-34` falsely asserting it is "actually called"** — the live path is
`data_control.rs:371`; the three `to_vortex.rs:518/551/557` cache helpers orphan with it;
`indicators.rs:1/:29/:71` three functions with one grep hit each;
`normalization.rs:141`/`:42` two dead variants — **keep `norm_fit_rows` +
`normalize_feature_series_in_place`, which are the live normaliser (`lib.rs:1223-1224`)**.
`test_fixtures.rs` ships a 100-bar JSON in every binary (`lib.rs:64`) and its justification
names three callers that do not exist (`:16-21`). **Ordering constraint: #96 before #93** —
a bare `#[cfg(test)]` still breaks three crates and no `test-fixtures` feature exists yet.
**Effort: trivial each, one wave; #93 hours.**

### `crates/neoethos-app` + `desktop/` — 17 open

The knob-catalog surface (#113–#116): the backend emits `kind`/`min`/`max`/`enumChoices`/
`helpLong`/`envVar` (`knob_catalog.rs:90-174`) and 53×3 = 159 preset strings; the only
client renders four fields read-only (`Advanced.tsx:418-428`) while hand-maintaining a
parallel schema (`Advanced.tsx:192`). **The prize:** render editable typed controls from
the catalog and 53 knobs become settable for free. Also: `Dashboard.tsx:70` sends him to a
credentials form that has never existed; `GET /journal/analytics` — the per-trade pips/R
and MFE-MAE view that would have surfaced the payoff-1.08 problem sixteen months earlier —
is reachable only by an LLM tool call (`mcp/ops.rs:864`), never by him.
**Effort: hours each; #114 days.**

### `crates/neoethos-core` — 5 open

#137 (the 847-line dead `RiskManager` D3 left standing), #142/#143/#144 (the env registry
that documents functions which do not exist), #141 (four artifact-contract names — **note:
the stated dependency on `checkpoint.rs` is false**; `checkpoint.rs` defines its own
`*ArtifactFile` types at `:204`/`:264` and identifies kinds with string constants, so
deleting it changes nothing about them — do not plan D1 around that claim), #145 (the
resolved-config table reporting `normalize_features` from a retired env var, on the very
day its default flipped to true — `config.rs:1904` says "this one is not cosmetic").
**Effort: trivial to hours; #137 decision-required.**

### `crates/neoethos-models` — 13 open

Twelve are one pattern: a config knob whose reader is overwritten, absent or purely
descriptive (#177, #178, #180, #182, #183, #184, #186, #188). Plus the two-orphan cluster:
`exit_agent` trained every run and excluded from the load list (`training_orchestrator.rs:547`
vs `bootstrap.rs:93-101`), the `genetic` expert running a second **in-sample** GA
(`genetic.rs:534-551`), and the whitelist that hides both (`ensemble_inference/mod.rs:907`).
#163 is the honest version of the failure mode — `supports_gpu_for_model` says in its own
doc that it is deliberately unwired pending task #35. **Effort: trivial to hours;
#173/#175 decision-required.**

### trading / live — 33 open

Covered in §2. The replay harness (#220–#231) is the largest unowned block.

### params / models verdict — 60 open

Five YAML divergences (§2.3), the env surface (#273–#280: the catalog advertises 47
`NEOETHOS*` names of which **four** are live, and five `install_*_from_env` wrappers have
zero call sites), and the model verdict (#297–#317).

**A caveat that changes how to read the model verdict:** every artifact in the store is
dated 2026-07-01 and byte-identical to what the audit measured (43 instances, 287.46 MiB,
`xgboost_rf` at 113.4 MiB). But **Option D was already true at HEAD** —
`l1_feature_selection_max_features: 256` (`config.rs:2241`, `config.yaml:514`, live readers
at `training_orchestrator.rs:1014/:1117`) — and the artifacts still record `feature_count: 20`.
**Items #297–#306 are true of the artifacts on disk and stale as a verdict on the system.
One retrain re-decides all of them.**

---

## 5. WAS_WRONG — 20 items. Do not re-open.

Re-litigating these is how this project loses weeks. Each was checked again against today's
tree to make sure the edits did not invalidate the refutation.

### The thirteen outright refutations

| # | Killed by | Why |
|---|---|---|
| #41 | R1 §7 | Prototype C + the bench cluster are **reachable from five registered CLI subcommands** (`neoethos-cli/src/main.rs:112-116`). A product decision (#46), not a dead-code fact. Items #30–#34 are marked SUPERSEDED for the same reason. |
| #45 | R1 §4 | `walkforward_risk_diagnostics_from_trades` is the **live** CPU measurement, called unconditionally at `validation.rs:943`. Only the `Some(trades)` arm at `:1605` is inert — that is #42. |
| #89 | R1 §3 | Both halves stale — the function no longer exists, and `neoethos-data/gpu-cuda` is named by `neoethos-cli/Cargo.toml:25` and `neoethos-app/Cargo.toml:35`. |
| #117 | R1 §5 | The three mesh routes **are** called — by the bundled sidecar in the excluded `mesh/` workspace (`mesh/src/main.rs:298/303/316`). A workspace-scoped grep missed it. |
| #134 | R1 §6 | `PortfolioManager` was cited as "the live one" justifying deletion of the search optimizer. It had zero references. **Both** were dead. |
| #170 | R1 §1 | `neuro-evolution-gpu` **is** in the `gpu-cuda` aggregate. |
| #171 | R1 §2 | `reinforcement-learning-cuda` **is** in the aggregate. The open item is a **card run** (#321), not a deletion. |
| #181 | R2 §6 | `models.tree_device_preference` has two live readers (`system.rs:374`, `training_orchestrator.rs:1280`). The audit named the wrong survivor. |
| #185 | R2 §3 | `models.ddp_world_size` **hard-fails** every training-artifact write (`runtime/profile.rs:148-154`), it is not recorded-only. |
| #187 | R2 §1 | The six `prop_search_*` knobs are passed **positionally into a real GA**, not provenance strings. |
| #211 | R2 §5 | `risk.max_portfolio_risk: 0.34` is a **sizing input**, not a max-1-position switch — `live_trading.rs:1748-1770` sizes the first entry down from 0.50 to 0.34. |
| #254/#255 | R2 §4 | `apply_overrides_from_lookup` writes those fields and is production-live; `system.enable_gpu`, `models.train_batch_size` and `models.hpo_trials` all have live readers. |
| #266 | R2 §2 | `backtest_runtime.initial_equity` crosses the FFI with a **frozen ABI offset assertion** (`neoethos-gpu-contracts/src/lib.rs:417`) and is the kernel's denominator. |

**Two more killed during this reconciliation:** #118 (the ledger's own W9, *"nothing POSTs
`/mesh/migration/enable` when the mesh toggle flips"*) — the chain closes:
`Advanced.tsx:22` → `mesh_sidecar::set_enabled` → `start()` → the sidecar POSTs it
idempotently every 20 s at `mesh/src/main.rs:709`. **W9 should be struck.** And #126/#127,
the audit's own NOT-DEAD notes, re-verified (24/24 desktop screens mount, three of them one
level down inside `Configuration.tsx` and `AiDesk.tsx`).

### Corrections — the finding stands, the framing did not

- **#196 — THE LEDGER CORRECTION THAT WOULD HAVE BRICKED THE DEMO GATE.** The ledger's W2
  instruction said to scope the demo forward-test gate on `active_account_id()`. **Applied
  literally, the gate becomes permanently unsatisfiable** — once the operator switches to a
  live account, no demo history can ever match the active account id, so no strategy could
  ever be promoted again. It was replaced with an **environment stamp on the journal row**:
  `live_gate.rs:80-84` retains rows stamped `Demo`, written from `current_environment_label()`
  at `journal_reconcile.rs:118`/`:183`. The correction is recorded in the code at
  `live_gate.rs:44-77` and pinned by a regression test at `:283-297`
  (`a_full_demo_history_is_still_eligible_when_the_live_account_is_active`).
  **Known caveat, stated in-code at `:70-77`:** rows with `environment: None` predate today
  and are excluded, so his demo forward test restarts from zero.
- **#190 — W1 ruled OUT OF SCOPE by the operator.** The manual order path correctly respects
  the operator: no size clamp, no `risk_per_trade` sizing, no daily-slot consumption. That is
  now written into `orders.rs:15-18` rather than being an accident. **Only** the
  `require_stop_loss` inconsistency was in scope, and it was fixed. #192/#193/#195 are marked
  SUPERSEDED on this ruling, not OPEN — their residual exposure is stated so it stays visible.
- **#271 — and this correction has itself gone stale.** The refuter established that
  `orchestration.rs:127` had no `apply_mode_overrides` call. **It has one today**
  (`self.config.clone().apply_mode_overrides()`), making **six** production call sites. The
  non-idempotency in #270 now compounds across one more path than anyone measured. Do not
  re-open the refuted claim; do note the new site.
- #244/#251 (seed counts; `discovery_runtime` **is** in the seed at `:136`), #260 (the
  silent-CPU-run half is closed at `backend.rs:160-198`), #272 (exposure inverted — the
  **fresh install** is exposed, the dev tree is immune), #279 (`RAYON_NUM_THREADS` collateral),
  #87 (WIRE, not delete), #96 (`cfg(any(test, feature="test-fixtures"))`, not bare `cfg(test)`),
  #289 (the export gates were disarmed **deliberately** on 2026-06-06 — the defect is that
  `ModelsConfig::default()` still encodes the pre-mandate posture).

---

## 6. WHAT THE AUDIT NEVER COVERED

The four audit areas were `neoethos-search`, `neoethos-models`, trading, and data-app-ui.
Everything below is outside all four and therefore **invisible in a status report by
construction**. Each was checked against all nine source sections.

### 6.1 — Subsystems with zero findings

| Area | Size | Mentions in the whole audit |
|---|---:|---|
| `crates/neoethos-codex` | 2,236 LOC | **0** |
| `crates/neoethos-mcp` | 3,590 LOC | 1 — the binary *filename* |
| `crates/neoethos-gpu-cuda/native` (CUDA) | 3,282 lines | **0** |
| `crates/neoethos-cli` | 11,058 LOC | incidental; `tui/wizard.rs`: **0** |
| `neoethos-app`: `artifact_io`, `pending_actions`, `broker_persistence` | — | **0 each** |
| `.github/workflows` | 1,240 lines | 6 lines (#37) |
| `scripts/` | 1,885 lines | 2 items |

**The two that matter:**

1. **`neoethos-mcp` is an order-submission path and it was audited for its filename.**
   `ops.rs:301` POSTs `/orders`, `:376` amends, `:417` sets protection, `:452` places
   pendings, `:467` cancels. Its demo guard (`backend.rs:234-260`) is genuinely well built
   — fresh, uncached, fail-closed on unknown — **but it has the same TOCTOU the trading
   shard fixed for the autopilot today.** `ensure_demo()` reads `/broker/status`,
   `trade_post()` then posts, and nothing binds the two to one read. `broker_api.rs:237-260`
   solved exactly this with `resolve_creds_expecting(expected_is_live)`; the MCP path was
   not part of that change because it was not part of the audit. `DemoProof` proves a check
   *happened*, not that the environment still is what it was. **Money path. Not in the 323.**
2. **`neoethos-codex` handles credentials and was never opened.** PKCE OAuth, a token
   exchange, a loopback listener on `127.0.0.1:1455`, read/write of `~/.codex/auth.json` —
   2,236 lines, zero findings.

### 6.2 — Work that landed TODAY carrying zero items

The ledger was flattened before four workflows finished writing. These are real, wired
subsystems the report has no item for:

- **`crates/neoethos-search/src/gpu_native/scenario.rs`, 787 lines, new, fully wired** —
  cost-sensitivity and Monte-Carlo scenario construction, called from `discovery.rs`
  (eight sites) and `eval.rs:2602/:2620`. **Money path, zero items.**
- **`crates/neoethos-data/src/core/feature_budget.rs`, 363 lines, new** — it decides how
  wide the vocabulary is allowed to be (`hpc_ta.rs:274-278`). Shard 2 audited the vocabulary
  **floor** (#99) without noticing the **cap** that binds it.
- **`prototype_b_population.cu`, 1,845 lines changed** — the production kernel. One item
  touches this file.
- **`crates/neoethos-data/tests/zz_throwaway_streaming_measure.rs:4-6`** records an
  **80× feature-build regression measured today**: *"at 6,000 bars with a 4,096-column
  budget the feature pass went from ~9 s to ~12 min."* That is the cost of the vocabulary
  restoration at the hard ceiling, and it is recorded **only** in a test whose own header
  says "delete after the measurement is recorded". **Nothing in the 323 items says discovery
  start-up got two orders of magnitude slower today. Promote this to a first-class item
  before the file is deleted and the number is lost.**
- **192 changed paths under `vendor/`** collapse into one item (#101, IN_FLIGHT). Defensible,
  but "one item" reads as "one thing".

### 6.3 — A new defect found during this reconciliation. One line, fix it today.

The gpu-contracts sentinel changed today: `spread_ticks` / `commission_micros` went from
*"0 = use the settings' costs"* to *"-1 = no override, 0 = charge NOTHING"*, and the derived
`Default` was replaced by hand because *"the derived one was a FREE-TRADING BACKTEST"*
(`neoethos-gpu-contracts/src/lib.rs:146-155`, `:126-128`).

The author found and fixed the consequent guard bug in
`prototype_population.rs:370-385`, with a six-line paragraph explaining it.
**The identical guard one file over was not updated:**

```
crates/neoethos-search/src/gpu_native/prototype_a_engine.rs:170     && scenario.spread_ticks == 0
crates/neoethos-search/src/gpu_native/prototype_a_engine.rs:172     && scenario.commission_micros == 0
```

`scenario::base_scenario` now writes `NO_TICK_OVERRIDE` (`scenario.rs:366`), so
`validate_supported_scenarios` **rejects every descriptor the base path produces** and would
wave through a free-trading one. Prototype A is bench-only, so no live money — but the bench
cluster is five registered CLI subcommands (#41), so `bench` on Prototype A is broken as of
today. These are the only two remaining zero-sentinel comparisons in the tree; the CUDA side
(`prototype_b_population.cu:133-134`) and the CPU oracle (`eval.rs:2663`) both agree on `-1`.

### 6.4 — Four specified-but-unapplied edits, none of them items

`docs/pending-edits-forbidden-territory.md` carries four named defects with file:line that
appear nowhere in the 323:

- **`SplitSkip` / `skipped_splits`** (`:137-166`) — specified, and `grep -rn` across `crates/`
  returns **zero**. The caller still cannot distinguish "this split was too short" from
  "walkforward failed". The doc calls it *"directly implicated in the recorded
  `walkforward=false` symptom"*.
- **`validation.rs:1175-1179`** — day indices silently substituted for timestamps.
- **`FeatureFrame::validate_registry`** (`features.rs:97`, `feature_registry.rs:241`) —
  the gate exists, **zero callers repo-wide**. The one guard that would have caught a
  name-drift in today's 66→674-column expansion has never run.
- **SMC substring matching** (`smc_indicators.rs:358-361`) — `norm.contains(a)` over a
  vocabulary that just grew ~8×. Two of eleven aliases (`choch`, `premium`) bind to nothing
  today, so it is latent — but any future indicator named `*premium*` silently becomes the
  SMC premium gate.

### 6.5 — Three structural gaps

- **Nobody audited the tests.** The project's stated asset is 1,685 tests. No item asks how
  many are `#[ignore]`d, how many assert a round-trip rather than a behaviour (#47 found one
  such by accident), or whether any suite is excluded from CI.
- **Nobody audited data at rest.** The 14,240 corrupt zero-price bars and the "data needs
  re-import" conclusion have no item, no batch and no owner in these 323.
- **Nobody audited the build.** `neoethos-app/build.rs` emits embedded broker credentials
  into the binary; whether they are the right ones or rotatable was never asked — and #119
  establishes there is no in-app way to replace them.

---

## 7. THE RECOMMENDED NEXT WAVE

### Wave 0 — before anything else

**Commit the working tree.** 81 FIXED items and every one of today's money-path repairs are
uncommitted across 323 dirty paths, 31 of them untracked (`run_identity.rs`,
`trial_returns.rs`, `scenario.rs`, `feature_budget.rs`, `indicator_ledger.rs`,
`shipped_config_matches_defaults.rs` among them). A stray `git reset --hard` costs the
entire day.

### Wave 1 — money, cheap, independent. Half a day for all of it.

Each of these is independent of the others. Do them in any order; do them all.

1. **#208 — read `models.exit_policy` in the live trailing block** (`live_trading.rs:1220-1275`)
   and delete the now-false comment at `:1221-1223`. *Hours. Highest value in the report.*
   This is the only item in Wave 1 that is not trivial, and it is the one that matters most.
2. **#245/#246/#248/#249/#250 — the five seed divergences.** Five one-line edits.
   Currency first.
3. **#214 — correct the two drawdown values in `config.yaml:158-159`.** Trivial.
4. **#69 — put the three session-spread numbers in both YAMLs.** The run already tells you
   where to get them (`discovery.rs:7595-7612`). No code.
5. **#267 — write `models.discovery_mode` alongside `system.trading_mode`** in
   `settings.rs:397-412`. One assignment.
6. **#75 — read `kill_zones_enabled` from config at `discovery.rs:1738`** instead of the
   literal. One line.
7. **§6.3 — the two `== 0` sentinel comparisons at `prototype_a_engine.rs:170/:172`.**
   One line, and it un-breaks `bench` on Prototype A.

### Wave 2 — must land together, not separately

Three pairs. Splitting any of them makes the system worse than leaving both halves alone.

| Pair | Why they are coupled |
|---|---|
| **#213 + #214** | Correcting the two YAML values without fixing the preset writer (`server/risk.rs:172-173`) means the next click on any preset puts them back. Fix the writer to apply the same ×0.7 buffer `config.rs:561` uses, or the value edit is cosmetic. |
| **#299 + #310** | Three numerically divergent artifacts are harmless **only** because `live_ml_gate` is off. Flipping the gate without either a numerical-sanity check or non-empty `expert_weights` (#315) puts `tide` (best_loss 1.3 M) into a vote that scales real position size. Decide both, or neither. |
| **#289 + #125** | `ModelsConfig::default()` re-arms the export gates; the desktop swallows a config-load failure into `Settings::default()`. Either alone is survivable; together they mean a CWD accident silently changes which strategies may reach live. And a money gate landed today (`orders.rs:63-75`) now reads through that same path. |

### Wave 3 — one wave each, mechanical

- **D1 (search orphans).** ~3,000 lines across items 1–24 and 54–64. **Two preconditions,
  both verified:** keep `EvalMetrics` (`diversity.rs:3`) and `StrategyQualityAnalyzer`
  (`quality.rs:837`, live at `discovery.rs:1650`); and **do not delete
  `prototype_b_mirror.rs`** — the rented-box preflight selects its tests (#320, resolved).
  Also drop the false `checkpoint.rs` → contracts dependency from the plan (#141).
- **D5 (data orphans).** Items 76–85. **#96 must land before #93** — add a `test-fixtures`
  feature first, or three crates stop compiling.
- **#290 + the ledger.** Add the six money keys to `MUST_MATCH_DEFAULT`
  (`shipped_config_matches_defaults.rs:32-36`). Hours, and it prevents Wave 1 item 2 from
  ever recurring.
- **The env registry (#142/#143/#144/#273–#280).** The catalog advertises 47 names of which
  four are live; two of the dead ones are documented as required by a function that does not
  exist anywhere in the repo. Delete the fiction, wire or delete the getters, migrate
  `symbol_metadata.rs:725`.

### Wave 4 — decisions, not work. He must choose; nobody can choose for him.

- **#46 / D7 / D8** — do Prototype A, Prototype C and the f32 lanes exist at all? The bench
  cluster is five live CLI subcommands, so this is a product call, not a dead-code fact.
- **#137** — wire the 847-line `RiskManager` or delete it. Wiring it changes live sizing.
- **#173 / #175** — ship the exit-side decision loop that consumes `ExitDecision3`, or stop
  training `exit_agent` every run.
- **#209 / #210** — which number bounds risky sizing: the config's 0.30 or the constant's 0.50?
- **#322** — is `require_walkforward_for_export: false` still the right posture? Needs an OOS
  comparison, not a code read.
- **The model verdict (#297–#317)** — and **retrain first**. Option D (`l1_feature_selection_max_features: 256`)
  is already shipped and no model has been trained under it. Every conclusion in #297–#306
  describes a fleet trained under a configuration the repo no longer ships.

### Explicitly independent — no coordination needed

#57 (wire `check_trade_invariants`), #238 (margin-call reader), #124 (the journal-analytics
UI), #218/#219 (blacklist by gene, and let discovery read it), #199 (bind the trailing amend
to the admitted environment), #191 (bracket rule on pendings), and every D1/D5 deletion.
None of these touches the others.

### Not recommended yet

The replay harness (#220–#231, twelve items). It is wrong in twelve independent ways and
owned by no batch. Either scope it as a project — real gene brackets, real costs, trailing,
his real balance — or **remove the Replay button until it is**, because a diagnostic tool
that gives wrong diagnostics is worse than no tool.
