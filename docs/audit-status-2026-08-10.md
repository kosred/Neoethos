# Audit Status — 2026-08-10 (RECOUNT)

**FIXED 115 · PARTIAL 13 · OPEN 163 · WAS_WRONG 21 · SUPERSEDED 10 · IN_FLIGHT 1 — of the same 323.**
**Money path (84 items): FIXED 45 · PARTIAL 4 · OPEN 28 · SUPERSEDED 4 · WAS_WRONG 3.**
**Money OPEN fell 47 → 28.** Twenty commits landed since the 08-09 ledger; the tree is committed.
**The single biggest change: live and discovery now agree about trailing.** That was the 16-month mechanism.
**The single biggest thing this recount found: the file a run actually reads has not been migrated.** §5.

This document changed nothing. No code was edited, no test was run, no cargo command
was issued. It is a re-check of the **same 323 items** against the tree at `afbd912c`.
It is a recount, not an audit: no finding was added, no finding was re-derived.

---

## 1. THE SCOREBOARD, AND HOW IT MOVED

| Status | 08-09 | 08-10 | Δ |
|---|---:|---:|---:|
| FIXED | 81 | **115** | +34 |
| PARTIAL | 21 | **13** | −8 |
| OPEN | 191 | **163** | −28 |
| WAS_WRONG | 20 | **21** | +1 |
| SUPERSEDED | 9 | **10** | +1 |
| IN_FLIGHT | 1 | **1** | 0 |
| | 323 | 323 | |

**Money path — the number that decides whether he can trade. Of 84 money items:**

| Status | 08-09 | 08-10 | Δ |
|---|---:|---:|---:|
| FIXED | 23 | **45** | +22 |
| PARTIAL | 9 | **4** | −5 |
| OPEN | 47 | **28** | −19 |
| SUPERSEDED | 3 | **4** | +1 |
| WAS_WRONG | 2 | **3** | +1 |

**Verification coverage, stated plainly so the number can be trusted for what it is.**
I read the code for **63 items individually** and give file:line for each below. Six env-surface
items (#273–#280) were verified **at the mechanism, not name-by-name** — the retirement
module, the retired `install_*_from_env` wrappers and the two ratchet tests are real, and
they are marked as block-verified where they appear. The large untouched blocks (D1 search
orphans, D5 data orphans, the models-knob cluster) were verified by re-grepping their **cited
representatives**, all of which are still exactly as the audit found them; I did not re-open
all sixty-odd of those files. Anything I could not verify is named in §7.

---

## 2. WHAT CLOSED SINCE YESTERDAY — verified by reading the code, not the reports

### 2.1 The 16-month mechanism: live and discovery now agree (#208, #74 — PARTIAL → FIXED)

`live_trading.rs:762-793` resolves `models.exit_policy` from the loaded `Settings` and logs
which of the three states it got. `:1479` — `let trailing = exit_policy.filter(|p| p.trailing_enabled);`
— gates the whole block, and `:1491-1493` take the geometry (`trailing_be_trigger_r`,
`trailing_stop_multiplier`, `trailing_min_lock_pips`) from the policy instead of the constant.
The false comment yesterday's ledger quoted is corrected in place at `:1451-1462`.
`ExitPolicyConfig::default().trailing_enabled` is `false` (`config.rs:1849`), and the search
reads the same field. **Backtest and live now model the same exit.** This was the highest-value
open item in the report and it is closed.

### 2.2 The drawdown breakers, at the value AND at the writer (#213, #214, #269 — OPEN → FIXED)

`server/risk.rs:46` `const TOTAL_DRAWDOWN_BUFFER: f64 = 0.7`, applied at `:258` —
`(constraints.max_overall_drawdown_pct as f64) * TOTAL_DRAWDOWN_BUFFER` — with the daily
side taking `PropFirmRuntimeDefaults::daily_dd_stop_trading_pct` (`:257`), which is a
separately published number and deliberately not `max_daily_loss * 0.7`. `:263-275` adds a
rule the item did not even ask for: **a preset switch may tighten a breaker, never widen
one**, with the refusal logged at WARN carrying both numbers. The f32 fingerprints are gone
from `config.yaml` (`:191` `0.08`, `:196` `0.14`) and both are registered with their
derivation in `shipped_config_matches_defaults.rs:143-159`.

### 2.3 The five seed divergences — the mechanism, not the values (#245, #246, #248, #249, #250 — OPEN → FIXED)

`desktop/src-tauri/resources/config.yaml` is now **generated** from `Settings::default()`
(header at `:1-12`, generator `crates/neoethos-core/tests/generated_seed_is_current.rs`), and
the test rewrites the file and then fails, naming every key whose shipped value moved. A
hand divergence between root and seed can no longer be typed. `config.yaml` became an
**overrides-only** developer profile (317 lines, header `:1-40`), and 27 money/search keys
are PINNED with their divergence and reason recorded in `ROOT_REGISTERED`
(`shipped_config_matches_defaults.rs:61-175`). Yesterday's "fresh install trades a different
product" cannot be produced by the shipped seed any more.

### 2.4 The rest, one line each

| # | Was | Now | Proof |
|---|---|---|---|
| #290 | PARTIAL | **FIXED** | `shipped_config_matches_defaults.rs` 992 lines; `PINNED` `:61-89` carries all five money divergences plus 22 more; `ROOT_REGISTERED` `:103` carries value + reason; `:674-684` fails if a registry entry names an unpinned path |
| #293 | OPEN | **FIXED** | `server/settings.rs:353-367` — `POST /settings/raw` now range-checks `daily_drawdown_limit` and `total_drawdown_limit` before writing; the stale "shape only" comment is corrected at `:322-325` |
| #125 | OPEN | **FIXED** | `desktop/src-tauri/src/lib.rs:130-156` resolves an ABSOLUTE path once; the `unwrap_or_else(\|_\| Settings::default())` swallow is gone and named at `:155`. Also `config.rs:3677-3689`: the bare relative `"config.yaml"` fallback is **deleted** and a no-config run now says so by name |
| #197 | OPEN | **FIXED** | `live_gate.rs:108-128` — drawdown is reconstructed from the SYMBOL-SCOPED trades and is **fail-closed** (`f64::MAX` when unmeasurable, so it can never pass a cap by accident). Residual, named in-code at `:203-208`: Sharpe is still account-scoped, deliberately |
| #238 | OPEN | **FIXED** | new `app_services/margin_call.rs` (`poll_once` `:328`, `spawn` `:478`); called at `broker_api.rs:926-933` in `prepare_new_order`, the single choke point both order-opening paths pass through. Closes reduce exposure and stay allowed |
| #199 | PARTIAL | **FIXED** | `live_trading.rs:1567` calls `amend_position_sltp_expecting`. `orders.rs:572-580` keeps the unbound variant for the operator's own manual amend, with the reason written at `:573-579` — that is the #190 ruling, not an omission |
| #191 | PARTIAL | **FIXED** | `orders.rs:241` `pub risky: bool` on `NewPendingOrderBody`; `place_pending` `:251-269` enforces `require_stop_loss`, and the bracket rule follows at `:270` |
| #209 / #210 | OPEN | **FIXED** | `live_trading.rs:816` reads `risk.risky_max_risk_per_trade`; `:833-865` clamps the stage ladder to the lower of it and `RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION`, logging which bound won |
| #218 | OPEN | **FIXED** | `strategy_blacklist.rs:232` `gene_fingerprint_bytes` hashes the RULE and drops the measurements; `is_blacklisted` `:298-308` matches on it, with the old file-bytes hash retained so existing entries still match. (#219 is still OPEN — see §3) |
| #143 | OPEN | **FIXED** | `core/env_overrides.rs:80-87` now states the var is dead instead of reporting it as an active override; `app_services/retired_env.rs` (new, 259 lines) prints an ERROR naming variable → replacement, called at `lib.rs:44` |
| #144 | PARTIAL | **FIXED** | `symbol_metadata.rs:726` is now a comment; the raw literal is gone |
| #145 | OPEN | **FIXED** | `resolved_config.rs:109-117` — field renamed and sourced from `models.data_runtime.normalize_features` (`:181`), not a retired env var |
| #119 | OPEN | **FIXED** | `desktop/src/screens/Settings.tsx:165-176` — the cTrader credentials form exists and posts to `/broker/credentials`; `Dashboard.tsx:68-75` no longer points at a form that never existed |
| #124 | OPEN | **FIXED** | `desktop/src/screens/Account.tsx:205` polls `journalAnalytics`, `:249/:253` render the "Analytics (pips · R · MFE)" tab. The view is reachable by him, not only by a tool call |
| #166 | PARTIAL | **FIXED** | `live_trading.rs:897-907` logs `voting_expert_count()` and `experts_unused_for_voting()` |
| #90 | PARTIAL | **FIXED** | `backend.rs:330` calls `set_indicator_compute_policy(policy)`. The seam that had zero callers has one, in production |
| #244 | PARTIAL | **FIXED** | superseded by generation — the seed cannot lack a key the defaults have |
| #267 | OPEN | **WAS_WRONG** | the premise is false. `discovery.rs:6876-6921` `resolve_discovery_mode` decides Risky vs PropFirm from **`system.trading_mode`** — the field the handler writes. `models.discovery_mode` maps only `strict\|legacy`; `risky`/`prop_firm` fall through and now WARN that they decided nothing (`:6900-6910`). `settings.rs:596-604` additionally REFUSES the change when the escape hatch is set. Do not re-open |
| #289 | OPEN | **SUPERSEDED** | the drift mechanism is gone. The Default/shipped disagreement on `require_walkforward_for_export` is now a dated, reasoned entry in `ROOT_REGISTERED` (`:105-112`) that neither side can move silently. The remaining question is #322, which is a posture decision, not a defect |
| #142 | OPEN | **PARTIAL** | the documentation half is fixed — `core/env_overrides.rs:104-118` now says **INERT, setting this changes nothing** and explains that the capability, not the var, is what is missing. The capability is still missing: `prop_firm_quote_to_account_rate` has zero callers |
| #273–#280 | OPEN | **6 FIXED / 2 OPEN** (block-verified) | `app_services/env_overrides.rs:31-43` "Migration COMPLETE — 2026-08-10", behavioural knobs read `AppRuntimeConfig`; every `install_*_from_env` wrapper is retired and warns (`eval.rs:565`, `evolution_math.rs:428`, `runtime_overrides.rs:729`); ratchet tests `neoethos-data/tests/env_surface_is_empty.rs` + its twin in `neoethos-search`. **56 `env::var` reads remain workspace-wide** (core 20, app 12, search 14, models 6, data 2, cli 2) — the core crate is not ratcheted, so 2 items stay OPEN |

---

## 3. EVERY OPEN MONEY ITEM, WITH ITS EXPOSURE — 28

Each was re-read today. "Exposure" means: what is wrong right now, on this code.

| # | Defect, at today's line | Exposure |
|---|---|---|
| **#69** | session-spread curve still unconfigured. The wire is complete and single-point (`discovery.rs:2062-2068`), and `config.rs:725-760` still correctly refuses a partial curve — but all three keys are `null` in the generated seed (`:109-111`) and absent from the root profile | every backtest charges the flat 1.5 pips at 03:00 Tokyo and at the London open alike. Any selected strategy whose trades cluster in the Asian session was priced optimistically. **Still trivial: three measured numbers, no code** |
| **#75 / #217** | `discovery.rs:2074` `kill_zones_enabled: true,` — still a literal, in the one settings-template every discovery lane flows through. The fields on both sides of it read `config.` | the knob can move live **away** from what was validated and never toward it. Defaults agree, so exposure is one toggle away. One line |
| **#57** | `gpu_native/trade_invariants.rs:48` `check_trade_invariants` — re-grepped repo-wide today: **still zero callers outside its own file** | the card produces the trades that produce the metrics that rank what he deploys, and nothing checks that an exit follows its entry or that PnL sign agrees with direction. A kernel regression does not fail — it ranks |
| **#71** (PARTIAL) | the band is measured at both edges and counted run-level (`discovery.rs:8298-8364`, census `:8845-8862`), and `discovery.rs:8926` **still binds `_cost_band`** at the export boundary | a gene profitable only at the optimistic 1.6-pip edge lands in `live_portfolio.json` indistinguishable from one robust across the band |
| **#219** | `grep -rn blacklist crates/neoethos-search/src` → **zero hits**, unchanged | discovery can re-derive the exact shape that was just auto-culled after six consecutive real-money losses. #218 fixed the identity; nothing consults it during search |
| **#220–#231** (12) | see §4 — **this block is worse than yesterday, not better** | the Replay screen's numbers are still not his numbers, and the code now says otherwise |
| **#137** | `domain/risk.rs` — `RiskManager` still has no production constructor; confirmed today from the other side: `config_has_recipient.rs:165-200` records `challenge_mode`, `challenge_phase` and `recovery_mode_enabled` as unread **because every `RiskManager::new` in the workspace is inside a test module** | `risk.daily_dd_warning_pct`, `risk.challenge_mode`, `risk.challenge_phase`, `risk.recovery_mode_enabled` reach nothing. His live store sets `challenge_mode` — see §5 |
| **#294** | still **13** fields with no qualified reader (`config_has_recipient.rs:165-341`), and the count by struct got **worse under a sharper scanner: 9 of the 13 are now `RiskConfig`**, not 7 | nine risk knobs that change nothing. Mitigated, not closed: a new guard at `:1989-2013` fails the build if a UI control is backed by one of them |
| **#206** | the four `risk.trailing_*` remain shadowed duplicates of `models.exit_policy.*`, formally recorded at `config_has_recipient.rs:63` and `shipped_config_matches_defaults.rs:166-175` | three names for one concept, and the one that looks like the risk knob is the dead one. **His live store sets `risk.trailing_enabled: true`** — see §5 |
| **#299 + #310** | `bootstrap.rs:118/119/140` — `tide`, `tide_nf`, `sac` still in `DEFAULT_BOOTSTRAP_EXPERT_NAMES`; no numerical-sanity check exists between "artifact on disk" and "artifact votes" | harmless **only** because `live_ml_gate: false` — verified in his live store (`:284`) and `expert_weights` is empty (`soft_voting.rs:116`). Decide these two together, never separately |
| **#232** | `blend_signal.rs:89/:93` — the literals are now named constants with a validated builder (`:137-165`) and the recipient is **still not there**: `live_trading.rs:529` is literally `let _ = settings; // recipient pending` | two numbers that scale every entry's risk with no config recipient. Dormant while the ML gate is off. This is a written gate whose caller passes nothing — the signature defect, self-declared |
| **#115 / #116** | two disjoint preset vocabularies unchanged — `knob_catalog.rs` advertises Conservative/Balanced/Aggressive per knob (`:259`, `:385`, `:548`…), `Settings.tsx:66/:392` applies only the prop-firm registry presets | the catalog advertises a Conservative posture the product cannot put him in. That is how §2.2 happened in the first place |
| **#265** | three starting balances for one concept, now documented together at `trader/engine.rs:23-36` and still three: `config.rs` `initial_balance` 10 000, the kernel's 100 000, `DEFAULT_REPLAY_STARTING_BALANCE` 10 000 | every percentage return the GPU search ranks on is computed against a balance 10× his real account |
| **#35 / #36** | `cubecl_eval.rs:4827-4841` still states in its own file that this lane is **54 % wrong at 200 000 bars**; the fallback below it is the CPU. Unreachable only while the prototype-B runtime probe succeeds | on any box where the probe fails, strategies are ranked on a net profit less than half the truth |
| §6.3 sentinel | `gpu_native/prototype_a_engine.rs:170` `&& scenario.spread_ticks == 0` and `:172` `&& scenario.commission_micros == 0` — **still the zero sentinel.** `prototype_population.rs:384` was fixed and this one was not; `scenario::base_scenario` writes `NO_TICK_OVERRIDE` | `validate_supported_scenarios` rejects every descriptor the base path produces and would wave through a free-trading one. Prototype A is bench-only, but the bench cluster is five registered CLI subcommands. **This was Wave-1 item 7, "one line", and it did not land** |

The balance of the 28 is the remainder of the replay block (§4) and the two un-ratcheted
core env items from #273–#280.

---

## 4. WHERE IT IS WORSE THAN YESTERDAY

Three places. This is information, not failure.

**4.1 — The replay block regressed for the operator while improving for the CLI (#220–#231).**
`7a4d60b0` landed `ReplayCostModel` (`execution.rs:34-60`), real-gene replay
(`replay_portfolio_from_dir`, `data_replay.rs:186`) and a fidelity-warning disclosure that
enumerates every stub (`data_replay.rs:19-83`). The **CLI** got all of it:
`neoethos-cli/src/main.rs:719-723` passes `replay_engine_config(settings, symbol)`, which
builds the cost model from `risk.backtest_spread_pips` / `slippage_pips` /
`commission_per_lot` (`main.rs:2593-2597`), and `--portfolio` routes to the real-gene path
(`main.rs:658`).

**The app route did not.** `crates/neoethos-app/src/server/autonomous.rs:74-79` still calls
`replay_symbol_from_dir(..., EngineConfig::default())`, and `EngineConfig::default()` sets
`costs: ReplayCostModel::zero()` (`engine.rs:61`). So `POST /autonomous/replay` — the Replay
button — is still the momentum stub, at the mark, zero spread, zero commission, zero
slippage, on a synthetic 10 000 balance, with no portfolio option at all. And the module
header at `data_replay.rs:3-7` now asserts that the two front-ends "produce byte-identical
`EngineStats`". **That claim is false as of today.** The gate was written and one of its two
callers passes nothing — in the same wave that was closing exactly this pattern elsewhere.

**4.2 — #294 got worse by count.** Yesterday: 13 unread fields, 7 of them `RiskConfig`.
Today: still 13, but **9** are `RiskConfig`. The set did not grow; the scanner stopped
false-passing on same-named fields belonging to other structs
(`config_has_recipient.rs:189-200`). The measurement improved and the number it reports is
worse. That is the honest direction.

**4.3 — A comment went stale inside the fix that made it stale.**
`shipped_config_matches_defaults.rs:166-175` justifies keeping `risk.trailing_enabled` by
saying "live execution trails unconditionally with no config recipient at all". That was
true yesterday. `live_trading.rs:1479` made it false in the same tree. One sentence, but it
is the sentence a future reader will reason from.

---

## 5. WHAT THE LEDGER CANNOT SEE — the file a run actually reads

**This is status, not a new item, and it is the thing he should act on first.**

The 08-09 ledger cites `config.yaml:NNN` throughout. As of `b763ca2f` the repo root
`config.yaml` **is read by nothing** unless `$CONFIG_FILE` points at it
(`config.rs:3660-3689`). The only file a run opens is
`%LOCALAPPDATA%\neoethos\config.yaml`. I read it. It is dated **2026-07-31** and
`scripts/migrate_live_config.ps1` has not been run against it.

What that file says, against the code as it stands today:

- **`models.prop_search_min_payoff_ratio: 0.0`** (`:209`). The default is **2.0**
  (`config.rs:2390`) — the payoff floor that was landed as the answer to the sixteen-month
  problem. **On his machine the floor is off.** Every run he has done since it landed
  ranked without it.
- **`risk.max_portfolio_risk: 0.0`** (`:100`). `live_trading.rs:2126` gates the entire
  portfolio-level concurrent-risk budget behind `if portfolio_risk_cap > 0.0`. At 0.0 the
  cap is **silently disabled** — no warning, no error. The `ROOT_REGISTERED` note
  (`shipped_config_matches_defaults.rs:161-165`) says this ambiguity "is being turned into
  a loud startup error naming both readings". **That has not landed.**
- **`models.exit_policy` is absent entirely.** So the default applies —
  `trailing_enabled: false` (`config.rs:1849`) — which means live does not trail and
  discovery does not trail. **They agree.** This is the §2.1 fix behaving correctly on his
  actual file.
- **`risk.trailing_enabled: true`** (`:117`) is set — and is the SHADOWED DUPLICATE with no
  reader (#206). His own config tells him trailing is on while nothing reads it and the
  live loop does not trail. Two knobs, opposite answers, one of them dead.
- **`system.trading_mode: prop_firm` and `models.discovery_mode: prop_firm`** (`:17`,
  `:328`). The repo profile says `risky`. Per §2.4 #267 the first of these is the one that
  decides, so **his runs are prop-firm**, and the second decides nothing and will now WARN
  that it decided nothing.
- `preset: ftmo` with `daily_drawdown_limit: 0.04` / `total_drawdown_limit: 0.07` (`:92`,
  `:101-102`) — FTMO's buffered numbers, i.e. **tighter** than the own-money profile. Safe,
  and not what the repo files describe.
- `live_ml_gate: false` (`:284`) — confirms #299/#310 are dormant.
- `require_walkforward_for_export: false` (`:329`) — #322 stands, unchanged.
- `account_currency: GBP` at `:14` and `account_currency: null` at `:373` — two keys, one
  name, in his live file.

**The recount's plainest sentence: the money-path number improved from 47 to 28 in the
code, and the two settings that matter most on his own disk — the payoff floor and the
portfolio risk cap — are both switched off in the file the code reads.** Running
`scripts/migrate_live_config.ps1` is the cheapest money action available today, and it must
be run before the next card run, not after.

---

## 6. THE REMAINING PARTIALS — 13

| # | In | Missing |
|---|---|---|
| #71 | band measured at both edges + run-level census (`discovery.rs:8298-8362`, `:8845`) | verdict dropped at export — `discovery.rs:8926` |
| #142 | the doc lie is gone (`core/env_overrides.rs:104-118`) | the capability: `prop_firm_quote_to_account_rate` has zero callers; the 192× EURJPY lever is still wired to nothing |
| #68 | **half closed:** the ledger IS read now — `discovery.rs:5647` `load_prior_ledger`, `:5655` `seed_seen_from_ledger`; PBO/CSCV exists and gates (`discovery.rs:589`, `:1518-1523`) | DSR is still absent; the ledger's per-trial returns are read for dedup, not for the deflation it was persisted for |
| #20 | kernel writes slot 7; `PHASE1_GPU_SIZING_PORTED = true` (`eval.rs:2004`) | still named `RESERVED_INDEX_7` (`eval.rs:288`) and `eval.rs:278` still asserts the GPU lane is disabled — contradicted 1 726 lines below it in the same file |
| #87 | typed ledger closed the silent column drop | `max_indicator_warmup` (`hpc_ta.rs:56`) still has zero production callers while `:42` and `:76` claim it gates the sweep |
| #135 | core `domain/portfolio.rs` deleted | `neoethos-search/src/portfolio.rs` still exported (`lib.rs:166`) and still dead |
| #168 | `abstain_below_confidence` deleted | `expert_weights` still permanently empty (`soft_voting.rs:116`) ⇒ all 33 experts at weight 1.0 |
| #228 | same-side signal holds the position | still closes on Flat and on reversal; `eval.rs` has neither exit reason |
| #235 | eight builders acquired callers | five remain callerless; deletable set is two |
| #279 | `RAYON_NUM_THREADS` collateral recorded | still a note, not code |
| #283 | scanner no longer false-passes | 146 duplicate field names still exist |
| #296 | `export_onnx` gone | `regime_router_enabled` and `multi_resolution_enabled` still contradict — now **registered** as decisions (`ROOT_REGISTERED:120-136`), so this is closer to SUPERSEDED than PARTIAL; I left it PARTIAL because the values still disagree |
| #302 | flat-accuracy finding stands | decision, not work |

---

## 7. WHAT I COULD NOT VERIFY, AND WHY

- **The 08-09 ledger does not enumerate its 323 items.** It is a narrative over them. I
  recounted every item it names explicitly (63 of them, individually, with file:line above)
  and the class-level blocks by their cited representatives. The block statuses are
  therefore as reliable as their representatives, and I say which representative I checked.
- **`crates/neoethos-app/**`, `crates/neoethos-core/src/config.rs`, `crates/neoethos-data/**`
  and `vendor/**` were being written by another wave while I read them.** I read only; I
  edited nothing. Eleven of those paths are dirty right now, `live_trading.rs` and
  `config.rs` among them. Any line number I give in those files is from the working tree,
  not from `afbd912c`, and may move.
- **I ran no cargo command**, so "1,710 tests pass" is a claim from the commit message
  `34374c33`, not something I verified. The same applies to anything that depends on
  compilation.
- **Nothing `cfg(feature="cuda")` was checked by compiling** — this machine has no nvcc. The
  CUDA-side claims (`prototype_b_population.cu`) are read-only reads.
- **#101 (`vendor/vector-ta-0.2.9-patched`) stays IN_FLIGHT.** It is committed now, but three
  of its files are dirty in the working tree as I write this.
- **The three uncounted areas from §6.1 of yesterday's report are still uncounted**:
  `neoethos-codex` (2 236 LOC, 0 findings), `neoethos-mcp`, and the CUDA native tree. I
  re-checked the one that matters — the MCP order path still has the TOCTOU the trading
  shard fixed for the autopilot: `ops.rs:285/:308/:406/:434/:466` take a `DemoProof` from
  `ensure_demo()` (`backend.rs:234`), which proves a check *happened*, not that the
  environment still is what it was. `broker_api.rs:257` `resolve_creds_expecting` is the
  pattern that solves it. **Still not in the 323, still money path.**

---

## 8. THE NEXT WAVE, RE-DERIVED FROM THIS RECOUNT

1. **Run `scripts/migrate_live_config.ps1`.** §5. It is the only action that changes what
   the next run actually does. Nothing else on this list competes with it.
2. **`autonomous.rs:74` — pass `replay_engine_config`, or delete the Replay button.** §4.1.
   The CLI already proves the shape.
3. **`prototype_a_engine.rs:170/:172` — the two `== 0` sentinels.** One line. It was on
   yesterday's Wave 1 and did not land.
4. **`discovery.rs:2074` — read `kill_zones_enabled` from config.** One line.
5. **The three session-spread numbers (#69).** No code; `discovery.rs:7595-7612` still tells
   the run where to get them.
6. **`max_portfolio_risk: 0.0` — make it a loud startup error naming both readings**, as
   `ROOT_REGISTERED:161-165` already promises. Never a silent correction.
7. **Wire `check_trade_invariants` (#57)** — still 48 lines from its call site.
8. Then the blocks: D1, D5, the knob-catalog UI, and the model retrain that re-decides
   #297–#317.
