# Knob second pass — what the 390 knobs actually decide

**Date:** 2026-08-09 · **Status:** DECISION DOCUMENT — nothing was changed
**Builds on:** the parameter-chaos audit (54 KB) and its refutation pass (50 KB), both on disk.
**Scope:** read-only. No file was edited, no `cargo` was run, no knob was deleted. Every item below awaits your decision.

---

## The answer, in ten lines

**390 knobs — not 371; the surface grew by 19 while the audit was being written — express roughly 322 independent decisions.**

- **43 decide nothing today.** 13 have no qualified reader at all (`config_has_recipient.rs:126-234`), 4 are unreachable because a discovery override fills their slot first, 4 are shadowed twins of `models.exit_policy`, and the rest are frozen outputs of a detector that no longer runs.
- **~25 are surplus copies** of a decision another knob already makes — merged silently by `.max()`, `.min()`, or a precedence rule that names no winner.
- So **"most of them do the same job" is wrong as arithmetic and exactly right as experience** — because the duplication is concentrated in the knobs you can *see*. Of the 76 knobs with any UI presence (52 catalog rows + 24 controls), **16 display a compile-time constant as their "current" value**, **9 name a dead env var instead of the live config key**, and **7 of the 24 editable controls move something other than what their label says**. The 314 you cannot see are mostly distinct and mostly unreachable except by hand-editing YAML.
- And the sharpest finding of this pass is not about duplication at all: **there is a fourth config file, `%LOCALAPPDATA%\neoethos\config.yaml`, and it is the only one a run reads.** Every prior audit measured the two files in the repo. Neither is opened by `neoethos-cli discover` or the installed app.

---

## What this pass is, and what it is not

**The prior audit established** — and this pass re-verified rather than re-derived — the inventory: the struct count, the YAML key diff, the Default contradictions, the env-var census, `AutoTuner` having zero callers, `knob_catalog` advertising 49 env vars of which ~45 are inert, five orphaned `*_from_env` installers, `resolved_config` misreporting two knobs in both directions, and the `config_has_recipient` matcher being name-blind. **The refutation pass** then killed 6 of its 22 claims outright and corrected 4 — most importantly proving that the six `models.prop_search_*` GA knobs are live (consumed positionally, which is why a name-grep missed them), that `initial_equity` crosses the FFI into the CUDA kernel, that `ddp_world_size` can abort a training run, and that `max_portfolio_risk` binds every trade rather than blocking the second one.

**What has changed since those documents were written** (verified individually at HEAD):

| Change | Status |
|---|---|
| `config_has_recipient` rewritten receiver-aware (1710 lines, resolves receiver types, walks `mesh/` and `mcp/`) | **LANDED** — it was the report's own "highest-leverage single fix" |
| `risk.require_stop_loss` genuinely enforced, fails closed, `risky:true` no longer overrides | **FIXED** (`orders.rs:63-71`, `:116-137`) |
| `prefilter_top_k` reconciled to 240 in all four places, with a pinning test | **FIXED** |
| Trailing geometry promoted out of `strategy_gene.rs` into `models.exit_policy` + `models.gene_stop_bounds` | **LANDED**, correctly seeded in both repo files |
| `news.perplexity_enabled`, `news_lookahead_minutes`, `news_kill_window_min`, `models.export_onnx` deleted | **DONE** — those findings now have no subject |
| `shipped_config_matches_defaults.rs` added | **PARTIAL** — guards 3 of 390 knobs (0.8%); see §7 |

**What this pass adds** — the four questions the prior audit never asked: the **twin table**, the **derive list**, the **promote list** (hardcoded values that should be settable), and the **UI lie list**. Plus one thing nobody looked for: **which file a running process actually reads**.

**Three prior conclusions are now wrong.** Correct them before they propagate:

1. The refuters said `crates/neoethos-search/src/orchestration.rs:127` "does not exist at HEAD". **It exists.** `apply_mode_overrides` has six production call sites, restoring the original list.
2. The refuters' own collateral — "`RAYON_NUM_THREADS` is silently ignored on both binaries" — is **wrong**. `tree_models/config.rs:119` reads it on a live path (`cpu_threads_hint`, called from `parallel_trainer.rs:19`, `catboost.rs:414`/`:702`). It is an asymmetry, not a silence.
3. The refuters' AutoTuner collateral correction — "the `NEOETHOS_BOT_*` env path is live via `Settings::load_with_env()`" — is **wrong at HEAD**. `grep -rn load_with_env crates desktop mesh mcp` returns exactly one line: the definition at `config.rs:2837`. The entire 27-name env-override layer is unreachable. `system.device` now has **no live writer at all**.

---

## 1. THE UI LIE LIST

What you think it does · what it does. Ordered by how much money is behind the lie. 💰 = money path.

| # | Control | You think | It does |
|---|---|---|---|
| 1 💰 | **"RISK / TRADE 3.00%"** — `Risk.tsx:44`, `Settings.tsx:311` | You risk 3% per trade | In the repo posture (`trading_mode: risky`) live sizing ignores `risk.risk_per_trade` entirely — `live_trading.rs:1664-1680` substitutes the hardcoded ladder `risky_mode.rs:130` (0.30) → `:136` (0.50), then `:1749-1770` caps the first entry at `max_portfolio_risk` 0.34. **The card says 3%, the bot risks 34%.** `RiskyMode.tsx:17` renders a *second* card with the same label showing the honest 30–50% band. (On your live store `trading_mode: prop_firm`, so this lie is dormant *for you* — see §7.) |
| 2 💰 | **"Set what you want for this search + live sizing"** — `Discovery.tsx:350` tooltip on `riskPerTrade` | It sets both | Neither. `grep '\.risk_per_trade\b' crates/neoethos-search` → **zero hits**; the search reads the *bands* (`discovery.rs:813-821`). Live discards it in risky mode. |
| 3 💰 | **"Currently active rules: Prop-firm / Risky"** — `RiskyMode.tsx:100`, `Risk.tsx:53` (`risk.prop_firm_rules`) | Which rule set governs the account | Nothing. One write (`server/risk.rs:166`, from the preset dropdown), one read (`:244`, into the display DTO), **zero decisions**. Every discovery call passes the hardcoded `PropFirmRiskRules::default()` (`validation.rs:319`, `main.rs:540`, `engines_control.rs:257`, `cli/main.rs:1117`). It can announce "Prop-firm" while `system.trading_mode` is `risky`. |
| 4 💰 | **`risk.preset: the5ers`** — Risk screen dropdown | Selects that firm's numbers | Selects that firm's **label only**. `#[serde(default)]` on `RiskConfig` (`config.rs:315`) builds `Default` *first*, and `PropFirmPreset` derives `#[default] Ftmo` (`prop_firm.rs:35-38`) — so the YAML `preset:` key lands *after* the six fields it is documented to seed (`config.rs:521-523`), and nothing re-derives (`for_preset` has exactly two callers outside `prop_firm.rs`). You get The5%ers' name with FTMO's drawdown, lot and target numbers. |
| 5 💰 | **"Max portfolio risk (%)" — "entries pause once open positions risk ~5%"** — `Advanced.tsx:198` | It pauses new entries | It **sizes down first**: `live_trading.rs:1749-1770` `effective_risk = base_risk.min(remaining)`, so with zero open positions the first entry is resized to the cap. And `0.0` — the documented "disabled" value and what your live store carries — means **no cap at all**, not "no additional risk". |
| 6 💰 | **`risk.trailing_enabled: true`** in both repo files and your live store | The strategy trails | The search reads `models.exit_policy.trailing_enabled`, which is **`false`** (`config.yaml:387`, seed `:228`), consumed at `strategy_gene.rs:867`. Your live store has your hand-tuned `trailing_atr_multiplier: 0.4` / `trailing_be_trigger_r: 0.1` — deliberate values that move nothing. Meanwhile live execution trails **unconditionally** with no config gate at all (`live_trading.rs:1226-1272`). Three trailing policies, three answers. |
| 7 💰 | **Risk/Settings: "Manual orders are not gated by these"** — `Settings.tsx:317`, `Risk.tsx:36` | Manual orders bypass risk settings | False since 2026-08-09: `orders.rs:116-137` refuses a manual order with 400 when `require_stop_loss` is on. The refusal text points you at *"turn it off in Settings"* — **a control that does not exist anywhere in the app** (no DTO field, no `.tsx` control; `grep requireStopLoss desktop/src` → nothing). |
| 8 | **Compute: auto / cpu / gpu** — `Settings.tsx:287`, `Advanced.tsx:199`, TUI `config_view.rs:40` | Chooses the search device | `backend.rs:126-130`: `models.prop_search_device` **replaces** the global whenever non-empty, and both files set it. Press CPU on a card box and you get a GPU run while `Settings.tsx:295` renders `Active: cpu`. The one value that *would* force CPU — `cpu_forced` — is rejected by `server/settings.rs:428` with a 400 and appears nowhere in `desktop/src`. |
| 9 | **Knob catalog "Current" column** — `Advanced.tsx:420` | What the bot is using right now | **16 of 52 rows are compile-time string literals.** `ctrader.*` (6), `paths.*` (2), `risk.prop_firm_preset`, `risk.pnl_audit_drift_fraction`, `risk.pnl_circuit_breaker_fraction`, `risk.require_stop_loss`, `log.*` (2), `server.bind_addr`. Nine of them have a live installed value one function call away (`env_overrides::ctrader_max_attempts()` etc.). Set `app_runtime.ctrader_max_attempts: 5` and the screen still says 3. |
| 10 | **`envVar` in the catalog** — served to the LLM control plane at `mcp/server.rs:716-725` as *"read this before touching update_settings"* | 49 env vars you can set | ~4 are live. ~35 are read only inside orphaned `from_env()` constructors. **Correcting both prior documents:** the "10 with zero readers anywhere" group is wrong for 9 of the 10 — those knobs are *alive*, migrated onto `settings.app_runtime` (`env_overrides.rs:97-183`, installed at `app/lib.rs:38`, live callers at `server/mod.rs:466`, `ctrader_execution.rs:504/769/779/1078`, `live_trading.rs:991/1011`, `pnl.rs:118/129`). Only `app_runtime.chart_merge_side` is genuinely dead. **The catalog names a dead lever and hides the live one.** |
| 11 | **"Knob catalog (52)" / "~200 long-tail knobs"** — `Advanced.tsx:381`, `:411`, `:415` | The catalog documents every option | 52 of 390, read-only. `knob_catalog.rs:12-22` quotes *your* 2026-05-25 directive ("every possible option plus a help section") as its reason for existing. |
| 12 | **Raw YAML editor "✓ config.yaml saved (verbatim)"** — `Advanced.tsx:310`, validator `settings.rs:256-269` | The typed schema check caught typos | No config struct sets `deny_unknown_fields` (`config.rs:163` says so for `SystemConfig`). `trailing_enabeld:` parses, saves, and reports success. **This is the only route to 364 of the 390 knobs.** |
| 13 | **CLI TUI "Discovery mode: prop_firm \| strict \| risky"** — `config_view.rs:50-54`, `:152-162` | Selects the regime | `discovery.rs:5755-5760` maps only `strict\|legacy`; everything else falls through to `system.trading_mode`, which is **not in the TUI form at all**. So `risky` is a no-op, `prop_firm` is a no-op, and `legacy` — the one value the engine honours — is *rejected* by the validator. |
| 14 | **`neoethos-cli config` resolved table** | Which value won | Three rows wrong. `resolved_config.rs:329` prints raw `models.discovery_mode` beside a mode resolved from `trading_mode`, with help naming the wrong field (`:332`). `:337-350` still emits `normalize_features` and `disable_smc_gate` as `source=env` from env vars the engine no longer reads. **The one diagnostic built to answer "which value won" is wrong on three rows.** |
| 15 | **`risk.account_currency` help: "Operator-supplied via Settings → Broker Setup; never auto-defaulted"** — `knob_catalog.rs:289` | You set it once in the UI | No currency field exists in `Settings.tsx`. It is **auto-synced from the broker** (`bridge.rs:126` `sync_account_currency_to_config`, called `:737`) — the exact opposite. |
| 16 💰 | **`correlation_cap` 0.7 / `volatility_sigma_pause` 3.0 / `require_swarm_confidence_min` 0.65** — `risky_mode.rs:149/155/160`, doc'd as active defences, **validated at startup** `:531-540` | Risky-mode safety gates | Enforced **nowhere**. A grep returns only the declaration, the Default, and the validator. The sibling field on the same struct (`presend_sanity_ceiling_fraction`) *is* enforced (`:739`, `live_trading.rs:1939/1959`), which is what makes the other three's absence conclusive. **Startup validation of a value nothing uses is the strongest possible false signal.** |

**One severity correction in your favour:** `has_edge` (`quality.rs:958`, `edge_score >= 0.70`) gates nothing. Its only two consumers are a `.filter().count()` feeding a display highlight and a format string (`app_services/discovery.rs:465`, `:544`). It reads like a money gate and is a printed number.

---

## 2. THE TWIN TABLE

Winner today, by what mechanism, whether you can tell, which should survive. **✗ = a merge the refuters overturned — do not execute.**

### 2a. Silent arithmetic merges — two names, both read, collapsed by `.max()`/`.min()`, no winner named

| Pair | Wins today | Mechanism | Can you tell? | Survivor |
|---|---|---|---|---|
| `models.transformer_hidden_dim` (`config.rs:923`) vs `transformer_d_model` (`:1116`) | Whichever is larger | `training_orchestrator.rs:878-884` `.max()` | No — both ship 256, agreeing by luck | one field |
| `transformer_heads`/`transformer_n_heads`; `transformer_layers`/`transformer_n_layers` | larger | `:886-892`, `:894-900` | No | one each |
| `models.label_stop_atr_multiplier` (`:1003`) vs `risk.atr_stop_multiplier` (`:415`) 💰 | larger | `:2316-2321` | No | `risk.*`; note a *third* hardcoded 1.5 at `stop_target.rs:226` |
| `models.label_take_profit_rr` (`:1004`) vs `risk.min_risk_reward` (`:359`) 💰 | larger — **and only if `models.label_geometry` selects the Asymmetric arm** (`:2481-2487`); the shipped `symmetric` arm reads neither | mode-gated `.max()` | No. This is why your 2RR floor may never reach the labels | `risk.min_risk_reward` |
| `models.prop_firm_min_pass_rate` (`:1031`) vs `models.discovery_runtime.prop_firm_gate.pass_rate` (`:1420`) 💰 | larger | `discovery.rs:7812` `.max()` | No — and the 2026-06-06 mandate written into both YAMLs names only the *first*, so raising the second silently overrides your disarm | one field |
| `global_max_rows` / `global_max_rows_per_symbol` / `max_training_rows_per_tf` | smallest non-zero | `training_orchestrator.rs:902-911` `.min()` | No — three names imply three scopes; code treats them as three spellings | one, derived (§4) |

### 2b. Precedence chains where the loser is what the UI advertises

| Pair | Wins today | Mechanism | Can you tell? | Survivor |
|---|---|---|---|---|
| `models.eval_runtime.spread_pips` / `.commission_per_trade` (`config.rs:1455-1456`) vs `risk.backtest_spread_pips` + `slippage_pips` + `commission_per_lot` 💰 | `risk.*`, **unconditionally** | `strategy_gene.rs:509-511` is a 4-step chain; step (1) is filled by `discovery.rs:754-756` → `:1090-1091` on *every* discovery run, and `:4086-4104` refuses a non-finite override so the `.filter` can never fall through | **No — the reverse.** These two are exactly what the Settings screen renders as `cost.spread_pips` / `cost.commission_per_trade` with tuning presets (`knob_catalog.rs:655-686`) | `risk.*`; delete the eval_runtime pair or state it is a library fallback |
| `models.tree_device_preference` (`:856`) vs `models.tree_runtime.device` (`:1974`) | ✗ **REFUTER OVERTURN — both live.** `tree_device_preference` has two readers (`system.rs:374`, `training_orchestrator.rs:1280`). But `tree_runtime.device` **is taken for CatBoost on every run**: `CatBoostExpert::new(0)` fixes `device_pref` at construction from it (`catboost.rs:265-266`), the orchestrator then replaces only `config.params` (`:3238-3239`) and never recomputes it, and training reads the captured field (`:388-389`). Three inference loaders do the same (`tree_adapters.rs:161/278/381`) | captured-into-a-second-struct | No, and they ship disagreeing (`gpu` vs `auto`) | **KEEP BOTH** — this is a live capture, not a dead default |
| `system.enable_gpu_preference` vs `models.prop_search_device` | ✗ **REFUTER OVERTURN — do not merge.** `enable_gpu_preference` is the *master gate* for training (`system.rs:310-320`); `prop_search_device` *replaces* the global for the search (`backend.rs:126-130`). `cpu` + `auto` = CPU training with GPU search, which is the A6000 configuration on record as deliberate. One field cannot express it | different axes | No | **KEEP BOTH**, but make one of them accept `cpu_forced` from the UI |
| `system.trading_mode` vs `models.discovery_mode` 💰 | `trading_mode` for the risky/prop_firm axis | `discovery.rs:5772-5783` | No — three surfaces claim otherwise (§1 #13, #14) | ✗ **REFUTER OVERTURN — do not merge.** `discovery_mode` reaches `Strict`, which `trading_mode` structurally cannot. **Restrict it to `strict\|legacy`** and fix the TUI. Also: this **refutes prior finding #23** — one `tradingMode` click flips *both* search and live, so the "validated under one regime, sized under the other" harm does not exist |
| `models.eval_runtime.smc_gate_threshold` vs `search_runtime.smc_gate_{start,end,curve,stagnation_step}` | ✗ **REFUTER OVERTURN.** The static threshold wins on **9** evaluation call sites (`discovery.rs:1689, 1820, 1972, 3142, 3295, 4221, 4284, 4495, 4866`) and loses on 6 (`:2643, 2955, 3235, 3384, 6354, 8165`). Not a losing twin | | No | **KEEP BOTH** |
| `models.statistical_device` (`:1099`) vs `NEOETHOS_BOT_{MODEL}_DEVICE` / `_META_DEVICE` | **ENV wins over config** (`statistical/common.rs:60-68`) — and unlike the `apply_overrides_from_lookup` layer, this path is genuinely live | env-first chain | No — neither env name is in `knob_catalog`, and the per-model key is synthesised so it cannot be enumerated | config; this carries the one precedence direction the project has spent eight months eliminating |

### 2c. Shadows — a knob in a section you read that reaches nothing

| Pair | Wins today | Can you tell? | Survivor |
|---|---|---|---|
| `risk.trailing_enabled` / `_atr_multiplier` / `_be_trigger_r` / `_min_lock_pips` (`config.rs:417-428`) vs `models.exit_policy.*` (`:1650-1663`) 💰 | `models.exit_policy` absolutely; the risk four reach nothing (`config_has_recipient.rs:173-198` ledgers all four as SHADOWED DUPLICATE, and notes `trailing_atr_multiplier` "was never an ATR multiple") | Partly — both repo YAMLs now carry a ⚠ UNWIRED comment. **Your live store does not, and never will** | `models.exit_policy` |
| `risk.prop_firm_rules` (`:385`) vs `risk.preset` — it is literally `preset != None`, written twice (`config.rs:571`, `risk.rs:166`) 💰 | Neither — no engine reads it | No; the Risk screen renders it as a live toggle | delete (derive the display) |
| `system.symbol`/`account_currency` vs `models.eval_runtime.symbol`/`account_currency` 💰 | `system.*` whenever non-empty (`discovery.rs:735`, `:747`); eval_runtime is a last-resort fallback that logs an error and returns a NaN sentinel | No — two `symbol:` keys ~1300 lines apart | `system.*` |
| `system.cache_dir` (`:156`) vs `models.discovery_ledger.cache_dir` (`:1824`) | Both — genuinely different artifacts, different types (`PathBuf` vs relative `String`) | No | **KEEP BOTH**, rename the ledger's to `ledger_dir` |

### 2d. Naming twins — same word, different job. **Keep both, rename.**

| Pair | Why they are not duplicates |
|---|---|
| `models.ml_cpcv_enabled` (`:881`) vs `models.enable_cpcv` (`:1106`) 💰 | Training CPCV (`training_orchestrator.rs:4352`) vs the **search's admission gate** (`discovery.rs:2586-2591`). Disarming the wrong one admits candidates that never passed purged CV. Rename to `training_cpcv_enabled` / `search_cpcv_gate_enabled` |
| `eval_runtime.smc_w_*` (11) vs `smc_search_runtime.p_*` (11) | Scoring weights vs gene-seeding probabilities. Included so nobody deletes half in a duplication sweep |
| `risk.backtest_spread_pips` vs `_asian`/`_overlap`/`_late_ny` 💰 | **The model twin.** `session_spread_pips()` (`config.rs:636`) returns `Ok(Some)`/`Ok(None)`/`Err`, and `discovery.rs:639-686` branches on all three — logs "session spread curve ACTIVE", warns what a flat spread costs, and **refuses** a partial curve rather than repairing it. *This is the shape every other twin here should be refactored into.* |
| `models.gene_stop_bounds` ATR band vs pip band 💰 | Selected by `atr_scaled`, and the fallback is **logged** (`discovery.rs:4247-4258`). Honest. Mark the four pip fields fallback-only in the YAML |

### 2e. The mode family, counted 💰

Seven mode knobs — `trading_mode`, `discovery_mode`, `preset`, `prop_firm_rules`, `challenge_mode`, `challenge_phase`, `recovery_mode_enabled` — produce **6 meaningful states** (3 discovery regimes × 2 live sizing regimes), and **4 of the 7 contribute zero**. Both repo files set `challenge_mode: true` against a Default of `false`: **you have deliberately armed a mode that does not exist.**

### 2f. ✗ Merges the refuters overturned — listed so they are not re-proposed

| Proposed merge | Why it was commuted |
|---|---|
| `triple_barrier_max_bars` / `label_horizon_bars` / `meta_label_max_hold_bars` 💰 | `label_horizon_bars` is the **CV purge width** — the leakage guard (`training_orchestrator.rs:380-382`), validated as a distinct artifact field (`profile.rs:110-112`). And a resolver already exists for two of the three (`:795-799`). Merging silently changes purging |
| The 12 `prop_search_*` / `search_runtime.*` selection knobs | The genetic arm's values come from `config.params`, which **HPO owns and rewrites per trial**, with its own literal defaults (tournament **3** at `:4682-4689` vs `(population/12).max(3)` = 16–341). A parity test cannot be written. A third `immigrant_fraction` exists on NEAT (`neat_impl.rs:72`) and a name-sweep would reach it |
| `initial_balance` / `initial_equity` / `starting_balance` / `risky_start_balance_usd` 💰 | `initial_equity` is not a normaliser — the metrics are **not scale-invariant**. `risk_based_pos_lots` clamps at an absolute `0.0..100.0` lots (`eval.rs:774-777`), so changing the base moves where that ceiling bites. Plus three fitness constants calibrated against 100 000 (`named.rs:162` `net/20_000`, `ingredients.rs:195` `/2_500`, `:209` `/50`). This is a metric re-baseline that invalidates every stored artifact — it must land *with* the constant change and a re-run |
| `cpu_budget` / `rayon_threads` / `n_jobs` / `RAYON_NUM_THREADS` | Under the CLI they are already one knob (`cli/main.rs:29-32`, `:1673` partitions cores across children). And the collapse is not neutral: `rayon_threads: None` means rayon takes **all** cores, while `resolve_cpu_budget` falls back to `total_cores - 1` |
| `multi_resolution_timeframes` vs `higher_timeframes` | Two different filters. The multi-res branch filters only against the base (`config.rs:296-301`); the other additionally filters to `canonical_higher_timeframes` (`:303-308`). Which survives is a product question |
| `gene_stop_bounds.rr_min` as a twin of `risk.min_risk_reward` 💰 | It is the **gene sampling band edge** (`evolution_math.rs:877`, `:716-717`, `:1273-1276`), not a policy floor. Merging changes which genes can exist |

---

## 3. THE GATES THAT SHIP DISABLED 💰

A safety check the code implements and the shipped config switches off. This is the most consequential class in the report.

### 3a. Deliberate — decided, documented, and correct. The defect is only that the Rust `Default` still encodes the *pre*-decision posture

| Gate | Default | root | seed | live store | What it prevented |
|---|---|---|---|---|---|
| `models.require_walkforward_for_export` (`config.rs:2266`) | **true** | false | false | false | Hard OOS export gate. Mandate verbatim at seed `:351`: *"2026-06-06 operator regime-diversity mandate ... Set true to restore"* |
| `models.prop_firm_min_pass_rate` (`:2267`) | **0.40** | 0.0 | 0.0 | 0.0 | All-window consistency floor. Mandate at seed `:357`: *"0.65→0.40→0.0 ... 0.0 = RANKING-ONLY"* |
| `risk.max_trades_per_day_enabled` (`:580`) | false | false | false | absent | The 8-entry daily cap. Off deliberately — `config.rs:577-580` records that a cap of 8 *"would have refused 68.1% of historical entries"* |

**Risk:** any install that loses one of these keys silently re-arms a gate you deliberately disarmed, with no config diff to explain why exports stopped. **Fix:** change the three Defaults to match the decision, then add the gate-parity pin (§9 step 2).

### 3b. Disabled by a hardcoded mode override — no config recipient exists anywhere

`apply_mode_overrides` (`discovery.rs:925-946` PropFirm, `:995-1005` Risky) assigns these from literals. `DiscoveryConfig::from_settings` never sets them (`:553-572` sets seven *other* fields and takes `..Default::default()`), and no `prop_search_max_dd` / `_min_sharpe` / `_min_profit_factor` key exists in `config.rs`:

| Forced value | PropFirm | Risky | `FilteringConfig::default` it replaces (`strategy_gene.rs:103-116`) |
|---|---|---|---|
| `max_dd` | **0.50** | **0.60** | 0.15 |
| `min_sharpe` | −10.0 | −5.0 | 0.3 |
| `min_win_rate` | 0.0 | 0.0 | 0.50 |
| `min_profit_factor` | 0.0 | 0.0 | 1.2 |
| **`anomaly_guard`** | **false** | **false** | true — **the anomaly detector never runs on any real run** |
| `cpcv_min_phi` | 0.0 | 0.0 | — |
| `min_trades_per_day` | 0.001 | — | — |

Whether a candidate that drew down 55% is a survivor or a reject is a Rust literal. The prior record — *1 713 of 2 211 candidates rejected for exceeding a 15% cap the mode had raised* — is this machinery.

### 3c. Disabled by value — a floor of `0.0` that reads like "off" and is opt-in-by-value

`TargetProfile::evaluate` (`discovery.rs:2227-2258`) has five criteria. **Four are guarded by `> 0.0`:**

| Criterion | Default | root | seed | live store |
|---|---|---|---|---|
| `min_expectancy_t_stat` — the noise bound | 0.0 | 0.0 | 0.0 | absent |
| `min_win_rate` | 0.0 | 0.0 | 0.0 | 0.0 |
| `max_in_market` | 0.0 (off) | absent | absent | **0.35** |
| `min_payoff_ratio` — **task #36** | **2.0** | 2.0 | 2.0 | **0.0** |
| `min_net_expectancy_per_trade` | binds **unconditionally** by design (`:2233`) | | | |

The code says it itself at `:2161-2164`: the trade-count floor is carrying the load against a lucky sample, *"It is not a noise bound."* **On the machine that runs your searches, the entire quality screen is `profit_per_trade > 0` plus an in-market cap plus the trade-count floors.** Your 2RR floor is not in force where it matters.

Same family: `models.discovery_runtime.prop_firm_gate` ships with `pass_rate: 0.0` and `n_windows: 0` — but `0` there is a **sentinel for auto-tune** (`discovery.rs:1070-1076`, `:5788-5802`), not a disable. The gate runs, ranks, and rejects nothing.

### 3d. Disabled on your machine only

| Gate | Repo | Your live store |
|---|---|---|
| `risk.max_portfolio_risk` 💰 | 0.34 (binds every trade) | **0.0 — no portfolio-level concurrent-risk cap at all.** Every engine sizes independently with nothing bounding aggregate exposure |
| `models.prop_search_min_payoff_ratio` 💰 | 2.0 | 0.0 |
| `models.cpcv_max_rows` | 200000 | **0 = unbounded**, the pre-fix state the seed comment measures at 1.05M rows |
| `models.prefilter_top_k` | 240 | **50** — the exact defect `shipped_config_matches_defaults.rs` was written to prevent |
| `system.multi_resolution_enabled` | false | **true** — the seed comment calls this "the pre-GA wall" |

### 3e. Implemented and never reached at all

| Subsystem | Evidence |
|---|---|
| **`domain::risk::RiskManager`** 💰 — monthly-target stop, challenge-target stop (`risk.rs:506-509`), revenge detector, night-session block, session windows | `RiskManager::new` (`domain/risk.rs:379`) has **no production constructor**. And `:399-400` fills your monthly and challenge targets from `PropFirmConstraints::FTMO_STANDARD` — so `risk.monthly_profit_target_pct` is not merely unread, it is **actively overridden by a constant** |
| Risky-mode `correlation_cap`, `volatility_sigma_pause`, `require_swarm_confidence_min` 💰 | Declared, documented, **validated at startup** (`risky_mode.rs:525-540`), enforced nowhere |
| The search's daily entry cap | `eval.rs:364` and `discovery.rs:2993` pin `max_trades_per_day: 0`. Arm the live cap and the backtest that selected your strategies took entries live will refuse |
| `kill_zones_enabled` 💰 | Pinned `true` in the search (`discovery.rs:1738`); live reads `risk.kill_zones_enabled` (`live_trading.rs:645-647`, whose own comment at `:626-638` states they must match). They agree only because the Default is also `true` |

---

## 4. DERIVE — what the machine should compute

**The line:** a knob is *capacity* when its value changes only whether the same computation fits — same inputs, same answer, different footprint. Test: *"on a machine with twice the RAM, would you want a different number, and only because of the hardware?"*

**The repo already knows the line and states it.** `discovery.rs:4788-4791`: *"Population is NOT a batching parameter: a bigger one creates different candidates and selects different survivors. So it may only grow when the operator asked (`population_auto`)."* And `cubecl_eval.rs:2255` logs *"memory tracks hardware, not user params"*. Those are the exemplars.

### 4a. The headline is not AutoTuner — it is `Settings::save`

`config.rs:2844-2851` serializes the **entire** 390-field struct on every write, and both `POST /settings` and `POST /risk/preset` load-mutate-save. **So one click on any control pickles every hardware-derived `Default` into `config.yaml` as a literal.** That is the mechanism behind `config.yaml:132 n_jobs: 11` (the fingerprint of `available_parallelism()-1` on a 12-core box, `config.rs:209-211`) and `:134-136 enable_gpu: false / num_gpus: 0 / device: cpu` on a 3090 box. **Wiring AutoTuner without fixing this just re-freezes the same lie on a different machine.** Fix: `#[serde(skip)]` on hardware-derived fields, or move them off `Settings` entirely.

### 4b. The AutoTuner decision: **DELETE**

`grep -rn AutoTuner crates desktop/src-tauri/src mesh mcp` returns **exactly two lines**, both declarations (`system.rs:1218`, `:1240`). No construction, no `.apply()`, not even a test.

Its successor exists **and is called**: `HardwareExecutionPlan` (`system.rs:287-570`), built at `training_orchestrator.rs:684`, computes the identical quantities from the same probe — `training_batch_size(gpu, min_vram_gb)` (`:1451-1463`), `inference_batch_size` (`:1466-1477`), `planned_memory_budget_gb` (`:1421-1449`) — and hands them to the consumer as **params** (`apply_hardware_plan_params`, `:691-762`) rather than writing them back into `Settings`. AutoTuner's `apply()` is the pre-plan writer nobody deleted; its nine assignments (`:1247-1258`) are exactly the poison-the-config pattern the plan was built to replace. Wiring it would additionally overwrite `system.device`, `models.prop_search_device` and `models.tree_device_preference`.

> ✗ **REFUTER OVERTURN on the stated precondition.** The brief said `apply_thread_env_defaults` (`system.rs:1343-1350`) must be re-homed first because it is the only setter of `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`. **It has exactly one caller — `system.rs:1259`, inside `AutoTuner::apply` — so those variables are not being set today.** Acting on the precondition would *newly* clamp OMP/MKL/OpenBLAS for the first time, in the subsystem already identified as the thread-oversubscription bottleneck, inside a commit claiming to be a no-op. **Delete both.** If a thread clamp is wanted, land it separately with its own measurement.

### 4c. Derive from a hardware quantity

| Knob | Declared | Derive from | Note |
|---|---|---|---|
| `system.n_jobs` | `config.rs:157` | `available_parallelism()` — already in the Default at `:209-211` | Zero readers (`config_has_recipient.rs:210-216`). **Delete** |
| `system.num_gpus` | `:166` | `HardwareProfile::num_gpus` — every real consumer already reads the probe (`scheduler.rs:184`, `cli/main.rs:1453`) | Zero readers (`:218-224`). **Delete** |
| `models.inference_batch_size` | `:919` | `WorkloadKind::Inference` plan (`system.rs:538`, `:1466-1477`) | Zero readers (`:226-233`). Ships `32` in both files — a value the plan would never produce. **Delete** |
| `system.enable_gpu` | `:165` | the probe | One reader: `training_orchestrator.rs:1900`, gating `rllib_auto` — which immediately degrades to `rlkit` and warns (`:1902-1903`). Delete with that branch |
| `system.device` | `:167` | `HardwareExecutionPlan` workload device | Survives at **three** sites, all `neuro_evo` (`:2044`, `:2115`, `:2165`), which hits the `_ => None` arm of the workload map (`:705`) so the plan returns early. **Give `neuro_evo` a `WorkloadKind` first**, then it deletes cleanly |
| `models.train_batch_size` | `:918` | `WorkloadKind::DeepTraining` (`system.rs:497`) | 13 readers, **all overwritten** by `apply_hardware_plan_params` (`:733-735`, applied last by design `:665-667`). One survivor: `:1994-1997` uses it for the **DQN replay buffer** when `rl_buffer_capacity: 0`. Give that its own hardware-derived default in the same change |
| `models.seen_signature_runtime.max_entries` | `:1784` | `available_memory_bytes()` | **`0` means UNBOUNDED** (`evolution_math.rs:372-376`) — a `HashSet<u64>` + `VecDeque<u64>` growing for the whole run, and `0` is exactly what someone types for "no limit". Make `0` mean *derive* |
| `models.swarm_memory_limit_mb` | `:933` | `WorkloadExecutionPlan::memory_budget_gb` | Fixed 256.0 on a 16 GB laptop and a 192 GB box alike. The plan's budget is inserted alongside under a different key and ignored |
| `models.exit_agent_memory_capacity` (`:756`), `models.rl_buffer_capacity` | | `WorkloadKind::RlTraining` (`system.rs:510-526`) | Last two absolute-integer memory sizers on the training side |
| `models.search_runtime.archive_cap_override` (`:1181`) | | `available_memory_bytes()` | The derivation already exists (`runtime_overrides.rs:398-401`); the override's only stated purpose is *"Override only if RAM-constrained"* (`knob_catalog.rs:691-706`) — the machine's question, not yours |
| `NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB` (`cubecl_eval.rs:2362`) | | `installed_memory_budgets().host_budget_mb` | Env is tried **first, never clamped**. The comment three lines above records the measured cost: *"a 200-gene M1 pass with no copy margin hit 38 GB RSS and SIGSEGV'd"* |
| `NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB` (`:2332`) | | `installed_memory_budgets().vram_budget_mb` | Same shape on the card. **The fix is already known**: the sibling `gpu_buffer_elem_cap` (`:2286-2288`) *does* `.min()` the env against the probed device cap — *"env is a ceiling, never exceed the device's true limit"* |
| `NEOETHOS_FEATURE_CUBE_MODE=ram` (`neoethos-data/src/lib.rs:1398`) | | `available_memory_bytes()` | The `ram` arm returns **before** the free-RAM check two lines below. ⚠ **IN_FLIGHT** — `crates/neoethos-data` is being modified; land after |
| `models.tree_runtime.gpu_count` (`:1981`) | | ✗ **REFUTER OVERTURN — KEEP.** It sits deliberately *after* `*_VISIBLE_DEVICES` and *before* the `nvidia-smi`/`rocm` subprocess probes (`tree_models/config.rs:369-374`). Those probes return `None` on timeout or a missing binary — at which point every tree model silently trains on CPU. `*_VISIBLE_DEVICES` can *narrow* a detected set; it cannot supply a count when detection fails. This is the only escape on a cuda-devel image with no smi CLI, and it ships `null` | |

### 4d. ⚠ The capacity/preference line was drawn wrong twice — both on the money path

The DERIVE lens closed with *"No money-path knob is a capacity knob... Zero deletions proposed on the money path."* That guarantee was tested by **section**, not by **effect**, and two of its own DERIVE entries decide money:

- **`models.backtest_runtime.month_capacity`** (`config.rs:1758`) 💰 — it sizes `monthly_pnls` and `month_start_equities` (`eval.rs:833-839`), which produce **metric slot 7**, which `named.rs:161` multiplies by **0.45 — the dominant term of the prop-firm objective**. Overflow is **silently dropped**, twice (`eval.rs:879-883`, `:1268`). And `knob_catalog.rs:884-894` advertises `min: Some(12)` while calling it a RAM cap: **setting the UI-endorsed minimum scores every gene on its first twelve months of a ten-year dataset and returns a plausible number.**
- **`models.cpcv_max_rows`** 💰 — `discovery.rs:2597-2601` validates the **tail only** (`offset = n - capped_n`) and returns pass/fail with **no coverage figure anywhere**. 200 000 against 1.05 M bars means the OOS gate that promotes a strategy toward live money saw **19% of history** and reported a clean pass.
- **`seen_signature_runtime.max_entries`** — eviction is FIFO (`evolution_math.rs:483-490`), so a lowered cap silently **re-admits previously-seen genes**, changing what the run explores.

**These three need a validator and a coverage log before anything about their memory behaviour is touched.**

### 4e. The line held — do not derive these

`prop_search_population`, `_generations`, `_max_hours`, `_max_rows`, `hpo_trials`, `cpcv_max_rows`(*value*), `l1_feature_selection_sample_limit`, `global_max_rows`. They change the **answer**. AutoTuner would have written `hpo_trials = if gpu { 50 } else { 20 }` (`system.rs:1325`) — letting the hardware decide how hard to search. That is the mistake this lens must not make in the other direction. `population_auto` (`discovery.rs:4797-4812`) is the correct pattern for any future hardware-aware mode: opt-in, floored at the configured value, and warn-logged as answer-changing.

---

## 5. PROMOTE — hardcoded values that should be settable

Ranked by how much a different value changes your results. **Lead: the four in the same class as the sixteen-month trailing.**

| # | Value | file:line | Range / default | Why it is the trailing's sibling |
|---|---|---|---|---|
| **1** 💰 | `MONTHLY_RETURN_TARGET = 0.04` — **your monthly bar**, in four evaluators plus a doubled fifth copy | `eval.rs:1266`, `cubecl_eval.rs:4252`, `:5173`, `prototype_c_engine/device.rs:855`; `DISCOVERY_MONTHLY_BAR_PER_60D_WINDOW = 0.08` at `discovery.rs:1049` | 0.01–0.15, default 0.04 | It fills slot 7 → **×0.45**, the dominant term of the GA objective. Meanwhile `risk.monthly_profit_target_pct` exists (`config.rs:326`), you can edit it, `POST /risk/preset` writes it (`risk.rs:174`), and **nothing reads it** — while `RiskManager` fills its own copy from `FTMO_STANDARD` (`domain/risk.rs:399`). ⚠ **Unguarded**: `0.0` makes every non-negative month a hit and collapses the objective's main discriminator — the validator must refuse 0 |
| **2** 💰 | The **quality floors** `apply_mode_overrides` assigns: `max_dd` 0.50/0.60, `min_sharpe` −10/−5, `min_win_rate` 0.0, `min_profit_factor` 0.0, `anomaly_guard` false, `min_trades_per_day` 0.001 | `discovery.rs:928-946`, `:998-1005` | `max_dd` 0.10–0.80, default 0.50/0.60 as today | Not overrides of configured values — **there is no config key for any of them**. Whether a 55%-drawdown candidate survives is a Rust literal. ⚠ **Unguarded**: a promoted `max_dd` above `risk.total_drawdown_limit` selects strategies the live breaker halts on — the validator must cross-check the two, and must decide whether `anomaly_guard` may ever be `true` |
| **3** 💰 | `max_regime_loss_pct = 3.0` — the search's daily/regime loss rule, hardcoded in **both** constructors including `from_settings` | `discovery.rs:498` **and `:826`** | fraction, default = `risk.daily_drawdown_limit` | It rejects any gene losing >3% of initial balance in a regime (`:5716`) **and it is the walk-forward `max_daily_loss_pct`** (`:2990`). The live equivalent is a knob shipping at three different values. Strategies are validated against a rule no file states and no operator can change |
| **4** 💰 | `MAX_LEVERAGE = 30.0` | `live_trading.rs:465` | 1–500, default from the broker account | On a £1 000 account at EURUSD ~1.16 the ceiling is ~0.25 lots. A 30% risk instruction silently becomes 8%, **with no log line when the cap binds** — `lots` is simply reduced (`:461-468`). ⚠ **Unguarded**: promoting it turns a knob that only under-sizes into one that can over-size. Needs a log when it fires and a refusal above what the account reports |

**Second tier — changes what is reachable or how results are read**

| Value | file:line | Range / default | Effect |
|---|---|---|---|
| `STATIC_THRESHOLD_LADDER = [0.10,0.20,0.35,0.50,0.70,0.90]` and weight levels `[0.2..1.0]` | `evolution_math.rs:563`, `:555` | `Vec<f32>`, default as today | **The entire alphabet a gene can be written in.** Eleven numbers, forever, unless `adaptive_thresholds` is on — and even then the six percentile points are literals (`:804-811`). Exactly the defect the ATR-scaled stop bounds just fixed, against feature scale instead of timeframe |
| `train_ratio = 0.70` at three production walk-forward sites | `discovery.rs:2427`, `:2981`, `eval.rs:5024` | 0.50–0.90, default 0.70 | The **model** half of the same system uses a configurable `models.global_train_ratio: 0.8` (`config.rs:978`). Same concept, two ratios, one visible. It is the first thing anyone tunes when an OOS result looks fragile |
| **33 of 35 `StopTargetSettings` fields** — `rr_trend` 2.5, `rr_range` 1.5, `stop_k_tail` 1.25, `min_risk_reward` 2.0, `atr_stop_multiplier` 1.5, `atr_period` 14, `vol_horizon_bars` 5, + 26 more | `stop_target.rs:138-240`; every production construction is `::default()` (`search_engine.rs:1603`, `cli/main.rs:2077`, `:1221`) | per-field | 💰 Only `tail_max_bars`/`tail_step` have a config recipient (`:1222-1224`). **Five share a NAME and a value with a `risk.*` knob you can edit** — so raising `risk.min_risk_reward` from 2.0 to 3.0 moves the labels and the live check and leaves the adaptive stop engine at 2.0, silently. The trailing's shape at ten times the surface |
| Half-Kelly cap **0.30** (ranking) vs **0.25** (GA objective) | `discovery.rs:6169` vs `named.rs:245` | `risk.risky_max_risk_per_trade` + new `risk.kelly_fraction` (0.5) | 💰 Both comments claim the two are identical. They are not. The GA evolves under a 0.25 growth model and the ranking orders survivors under 0.30 — **search and selection disagree about which gene compounds fastest**, the exact defect `named.rs` says it prevented |
| PropFirm ranking: drawdown denominator **0.07**, consistency cliff at **0.80** (×2 bonus) | `discovery.rs:6201`, `:6211` | `risk.total_drawdown_limit`; cliff → continuous | 💰 0.07 is exactly the *seed's* `total_drawdown_limit` and exactly not the root's 0.20. And consistency 0.7999 vs 0.8001 differ by a **factor of two** in final rank — a cliff the quality screen's own comment (`quality.rs:895-896`) says the project decided against |
| `min_risk_reward` floored at **1.5**, RR ceilinged at **6.0** | `stop_target.rs:807/917/1107/1124`, `:924` | honour the configured value; add `risk.max_risk_reward` | 💰 A knob that exists, is set, and is overwritten by `.max()` three lines later, with no log |
| Risk-of-ruin = **losing half the account**; reported Kelly = **quarter**-Kelly | `quality.rs:456`, `:859` | `risk.ruin_threshold_fraction` 0.10–0.90; merge the Kelly multiplier | 💰 "Risk of ruin 0.5%" means *probability of losing 50%* — on a prop account that trips at 7%, that is a threshold seven times further away than the one that ends the challenge. And the project now has **three** Kelly multipliers for one concept |
| `risk.max_trades_per_day` / `_enabled`, `risk.commission_per_lot_is_per_side`, `risk.backtest_spread_pips_{asian,overlap,late_ny}`, `models.exit_policy.*`, `models.gene_stop_bounds.*`, `models.prop_search_device` | — | — | 💰 **The reverse list**: all live, all money-deciding, **no control on any screen**. The exit knobs that decide when a winner is cut sit under `models:` while the four dead ones sit under `risk:` where you read |
| GA currency normalisers: `net/20_000`, `/2_500`, `/50`, activity `trades/30` | `named.rs:162`, `ingredients.rs:195`, `:209`, `named.rs:142` | derive from `initial_balance` | The archive net term **saturates at ±3 above 7 500 currency units** — 7.5% on the 100 000 backtest equity, so every strategy above that ranks identically. On the 100→50 000 ladder it is a rounding error |
| PropFirm `min_trades_per_month` TF ladder (8 literal multipliers), `FALLBACK_PORTFOLIO_MAX = 8`, PBO sample 64 / min 8, `stop_vol_mult` init `[0.5,3.0]` vs mutation clamp `[0.3,4.0]`, mid-pair seed probability 0.2, ranking tilt 0.7/0.3 | `discovery.rs:2066-2087`, `:6406`, `:2761-2762`, `evolution_math.rs:886`/`:1145`, `:873`, `discovery.rs:6198` | per-item | Third tier. The `stop_vol_mult` mismatch is notable: mutation can walk a gene into a region the initialiser cannot reach, **eight lines from the comment explaining why that is wrong** |

**Do NOT promote** (checked and cleared): `MAX_TRADES_PER_CANDIDATE = 8192` (`gpu-cuda/population.rs:45`) — an overrun drops **diagnostic** records only; equity, drawdown and trade count are unaffected, fill is measured and logged every launch, and host/kernel are pinned by `trade_slots_match_the_kernel` (`:782-802`). Promoting it would make peak VRAM a function of a user parameter, which the never-OOM invariant forbids. Also `max_pbo: 0.5` — hardcoded on purpose with the reason inline (`discovery.rs:804-807`: *"loosening it should require editing code or raw YAML, not one careless click"*). Also the mutation schedule, the MC iteration count, the four quality labels and the three UTC session boundaries. **371 knobs happened because everything numeric looked promotable.**

---

## 6. DELETE — with the full drag, and the commutations marked

| # | Item | Settings row | YAML (root / seed / live) | DTO / UI | Docs & tests | Verdict |
|---|---|---|---|---|---|---|
| D1 | **`Settings::apply_overrides_from_lookup` + `load_with_env`** — the whole 27-name `NEOETHOS_BOT_*` config-override layer, ~170 lines | `config.rs:2666-2834`, `:2837` | — | — | Only other reference is `config.rs:3078`, a `#[cfg(test)]` block. `load_registry_settings` was deleted in batch D4 (`registry.rs:1-19`) | **DELETE.** Zero behaviour change. ✗ **This overturns the refuters' correction #4**, which the reconciliation repeated after checking only that the assignment line still exists |
| D2 | **AutoTuner + `AutoTuneHints` + `apply_thread_env_defaults`**, ~145 lines | `system.rs:1218-1362` | — | — | — | **DELETE.** ✗ **Precondition overturned** — see §4b; do *not* re-home the OMP vars as a side effect |
| D3 | **5 `install_*_from_env` wrappers + 5 `from_env()` constructors**, ~220 lines carrying **34 env names** | `runtime_overrides.rs:973`, `:874`, `eval.rs:580`, `quality.rs:93`, `evolution_math.rs:406` | — | 34 `knob_catalog` `env_var:` entries go with them | Every hit is a definition, a doc line or a `pub use` | **DELETE.** Largest env-surface reduction available. **Exception:** `SmcSearchConfig::from_env()` is on a production path — **rename to `::current()`**, do not delete |
| D4 | `system.n_jobs`, `system.num_gpus`, `models.inference_batch_size` | `config.rs:157`, `:166`, `:919` | root only (all three absent from seed) / live has `n_jobs: 11`, `num_gpus: 0` | none | `config_has_recipient.rs:210-233` already ledgers all three as `Inert::WrittenNeverRead` | **DELETE** with D2 |
| D5 💰 | `risk.trailing_enabled` / `_atr_multiplier` / `_be_trigger_r` / `_min_lock_pips` | `config.rs:417-428` | `config.yaml:189-191` / seed `:52-54` / live `:117-119` | none | `config_has_recipient.rs:173-198` | **DELETE — but not first.** ⚠ Live execution trails unconditionally with **zero** config recipient (`live_trading.rs:1226-1272`) under a comment (`:1220`) still claiming parity with a backtest that now ships trailing OFF. Deleting these keys converts a visibly-wrong value into an invisible hardcode **on the path that spends real money**. Wire live to `models.exit_policy` in the same commit or before |
| D6 💰 | `risk.prop_firm_rules` | `config.rs:385` | root `:165` false / seed `:36` true | `RiskDto.prop_firm_rules_enabled` (`risk.rs:44`, `:244`); rendered `Risk.tsx:53`, `RiskyMode.tsx:100` | — | **DELETE the field**, derive the display from `system.trading_mode`. One write, one display read, zero decisions |
| D7 | `models.prop_search_async`, `_async_wait` | `config.rs:2147-2148` | `config.yaml:418` / seed `:174` — both shipped **ON** against a Default of false | none | `training_orchestrator.rs:1868`/`:1872` writers, `:2779` self-confessing note | **DELETE.** Both files enable an asynchrony the orchestrator does not have. **KEEP `hpo_backend`** — stronger than "keep": it reaches a validator that **bails on an empty string** (`profile.rs:142-143`) |
| D8 | `models.enable_fsdp`, `models.symbol_hash_buckets` | — | — | — | `training_orchestrator.rs:2771` note | **DELETE.** ✗ **`ddp_world_size` and `enable_ddp` COMMUTED** — `profile.rs:151-153` bails on `ddp_world_size > 1 && !ddp_enabled`, reached from the production artifact-write path (`training_orchestrator.rs:2896`). Set `ddp_world_size: 4` today and every training artifact write fails |
| D9 | `env_overrides.rs`: `ENV_PROP_FIRM_PRESET` + `prop_firm_preset_raw` + the `active_overrides()` arm; `prop_firm_account_currency`, `prop_firm_quote_to_account_rate`, `symbol_metadata_path_override` (0 external callers each) | `env_overrides.rs:114-141`, `:199-200` | — | `knob_catalog.rs:329` | Retired in v0.4.36 (`config.rs:433`) yet still fires the startup warning banner | **DELETE** the preset entry; finish Phase B for the other three |
| D10 | `models.eval_runtime.spread_pips` / `.commission_per_trade` / `.symbol` / `.account_currency` | `config.rs:1450-1456` | ship `null` everywhere | 💰 rendered as `cost.spread_pips` / `cost.commission_per_trade` with presets (`knob_catalog.rs:655-686`) | — | **DELETE or re-label.** Unreachable in every discovery run |
| D11 | `models.discovery_mode` as a three-valued field | `config.rs:1012` | root/seed/live all set values that are **no-ops** | TUI `config_view.rs:50-54` offers `risky` and rejects `legacy` | — | ✗ **COMMUTED — do not delete.** It reaches `Strict`, which `trading_mode` cannot. **Restrict** its accepted values to `strict\|legacy` and fix the TUI |
| D12 💰 | `risk.challenge_mode`, `challenge_phase`, `recovery_mode_enabled`, `monthly_profit_target_pct` | `config.rs:375`, `:384`, `:411`, `:326` | `challenge_mode: true` in both repo files against a Default of false | — | `config_has_recipient.rs:128-148`, `:200-208` | ✗ **COMMUTED — RETAIN AS INTENT.** Their consumer exists and is **hardcoded**: `RiskManager` fills the monthly and challenge targets from `FTMO_STANDARD` (`domain/risk.rs:399-400`), and the whole subsystem (monthly stop, challenge stop, revenge detector, night block) has **no production constructor**. Deleting these deletes your prop-firm *goal*, not garbage. Mark them `⚠ UNWIRED — RiskManager has no production constructor` inline, exactly as the trailing block was marked, and decide the subsystem as one item |
| D13 | `models.tree_runtime.device`, `models.tree_runtime.gpu_count` | — | — | — | — | ✗ **BOTH COMMUTED.** `device` is *captured* by `CatBoostExpert::new` and read at train time (`catboost.rs:265-266`, `:388-389`, orchestrator `:3238-3239`) plus three inference loaders. `gpu_count` is the only escape when the `nvidia-smi`/`rocm` probes fail. See §2b, §4c |
| D14 | The six `models.prop_search_*` selection knobs | — | — | — | — | ✗ **REMAINS REFUTED — do not reopen.** Live via a positional call at `training_orchestrator.rs:4682`. Deleting them reverts the genetic model to hardcoded values with no config diff |

**Also delete, zero risk:** the four tombstoned keys still sitting in your live store — `export_onnx: true` (line 280), `news_kill_window_min` (466), `news_lookahead_minutes` (470), `perplexity_enabled` (487). Serde drops them silently because no struct sets `deny_unknown_fields`.

---

## 7. THE FOUR-WAY DIVERGENCE — and which file a running app gets

### 7a. There are four sources, and the fourth is the only one that decides

`Settings::load()` (`config.rs:2593-2606`) resolves: `$CONFIG_FILE` → **`user_config_path()` if it exists** → literal `"config.yaml"`. On your machine that file exists:

```
C:\Users\konst\AppData\Local\neoethos\config.yaml   402 leaf keys   mtime 2026-07-31 06:54
```

Every `Settings::load()` caller reads it and never the repo: `cli/main.rs:28` and `:2405`, `live_gate.rs:161`, `desktop/src-tauri/src/lib.rs:399`, `tui/config_view.rs:23`.

The CLI already documents this, dated 2026-08-01, at `cli/main.rs:2388-2405`: *"the user config says trading_mode prop_firm, preset ftmo... the repo template says risky, none, and `cpu`... Which one a run got depended on the directory it was started from."* **That was fixed for the CLI. It was not fixed for the desktop shell.**

| | Rust `Default` | root `config.yaml` | desktop seed | **your live store** |
|---|---|---|---|---|
| leaf keys | — | 383 | 233 | **402** |
| missing from seed | — | — | 150 (strict subset; seed−root = **0**) | — |
| keys differing across the three files | — | **5**, all material | | |
| literal-Default contradictions | — | ≥33 | ≥29 | (49 across all three) |

The five differing keys: `discovery_mode` risky/prop_firm · `daily_drawdown_limit` 0.10000000149011612/0.04 · `total_drawdown_limit` 0.20000000298023224/0.07 · `prop_firm_rules` false/true · `account_currency` **GBP/USD**.

### 7b. What your live store actually says — three separately-fixed regressions, all still live

| Key | Default | root | seed | **live** | Consequence |
|---|---|---|---|---|---|
| `prefilter_top_k` | 240 | 240 | 240 | **50** | The exact value `shipped_config_matches_defaults.rs:4-11` was written to prevent: *"the base feature set collapses from 217 columns to roughly 64, and the SMC, session and footprint families die first"* ⚠ **the "217" is stale — the cube is now 1,946 columns per timeframe; see [`higher-timeframe-lane-2026-08-09.md`](higher-timeframe-lane-2026-08-09.md) §7.2** |
| `cpcv_max_rows` | 200000 | 200000 | 200000 | **0** | CPCV unbounded — the 1.05 M-row state the seed comment (`:365`) measured |
| `multi_resolution_enabled` | **true** | false | false | **true** | The seed comment (`:12`) calls this *"the pre-GA wall that stopped combos completing on laptop AND VPS"* — and the **Rust Default is also true**, so a key-less install re-creates it |
| `prop_search_min_payoff_ratio` 💰 | 2.0 | 2.0 | 2.0 | **0.0** | Task #36 is not in force where runs happen |
| `max_portfolio_risk` 💰 | 0.0 | 0.34 | absent | **0.0** | No portfolio-level risk cap at all on your app |
| `trading_mode` 💰 | prop_firm | **risky** | absent | **prop_firm** | ⚠ **Corrects both prior documents on their most-cited money finding:** the 30–50% risky ladder is *not* what sizes your live trades. `risk.risk_per_trade: 0.03` is. It becomes live the moment you flip the UI |
| `daily`/`total_drawdown_limit` 💰 | **0.04 / 0.070000001** | 0.100000001 / 0.200000003 | 0.04 / 0.07 | **0.04 / 0.07** | ⚠ **Second correction:** the audit's *"your live total-drawdown breaker sits at 20%"* is **false for your running app** — it sits at 7%. The raw ceilings are in the *repo* file only. Also: `PropFirmPreset` derives `#[default] Ftmo` (`prop_firm.rs:35-38`), not `None`, so the audit's *"the Default would have been 0.10 and 0.14"* is wrong twice |
| `generations` / `max_hours` / `max_indicators` | 50 / 0.5 / 12 | 20000 / 1.0 / 16 | | **1000 / 24.0 / 0** | `0` indicators means **ALL** (`discovery.rs:768-772`), not none. The run you think you configured is not the run that happens |
| `exit_policy` / `gene_stop_bounds` 💰 | | present | present | **absent** | The one correctly-seeded new block, and your live store will never acquire it |
| tombstones | | | | `export_onnx`, 3 news keys | Serde drops them silently |

### 7c. The desktop shell splits the money path across two files in one process

`desktop/src-tauri/src/lib.rs:52` installs the config path as the literal relative `"config.yaml"`, and `prepare_data_root` (`:424-437`) returns **without chdir** when the CWD already holds one — the dev launch. So `live_trading::run` (`live_trading.rs:566` → `current_config_path()`) reads the **repo** file while `live_gate` — the last gate before real money — reads `Settings::load()` (`live_gate.rs:161`) and gets **`%LOCALAPPDATA%`**.

**In a dev launch: sizing, drawdown breakers, kill zones and the ML gate come from the repo's `risky` / 0.10 / 0.20 / 0.34 posture while the demo-forward gate reads your `prop_firm` / 0.04 / 0.07 / 0.0 posture.** Installed launches are consistent (`lib.rs:445-457` chdirs to the exe dir). **The same binary trades differently depending on the directory it was started from** — the CLI's own comment names this as *"the cause of eight months of discovery never reaching the card."*

### 7d. Does the new test close any of it? **0.8%**

`shipped_config_matches_defaults.rs` guards **exactly three keys** (`:32-36` `prefilter_top_k`, `prefilter_insample_frac`, `prefilter_min_per_timeframe`), only inside `models.discovery_runtime`, only when the key is **present** (`:108-112` skips absent keys with *"Absent is FINE"* — which is precisely the 150-missing-key defect waved through by design), never compares root against seed, and `shipped_configs()` (`:49-58`) cannot reach `%LOCALAPPDATA%`. **Its own motivating example is live in production and the test is green.**

---

## 8. WHAT I COULD NOT DETERMINE

Named, so nobody treats a gap as a finding.

1. **Whether you launch the desktop app installed or from the repo.** The §7c split is proven in code; which posture your last live session ran under, I cannot tell. Nothing logs which file each subsystem opened.
2. **The exact "independent decisions" number.** 322 is an estimate from a stated method (390 − 43 inert − ~25 surplus), not a measurement. The surplus count depends on judgement calls the refuters overturned in six places, and would move again on a third pass.
3. **The prior audit's 73/43 Default-contradiction counts could not be reproduced.** My parser deliberately skips computed, derived and `Vec` defaults (34 of them). I have a **lower bound** of ≥33 root / ≥29 seed literal-comparable, and 49 across all three files. The 73/43 figures are neither confirmed nor refuted.
4. **Whether HPO actually rewrites the genetic model's `config.params` in production.** The refuters' overturn of the `prop_search_*` merge rests on that indirection being live. I confirmed the param map and its literal defaults (`training_orchestrator.rs:4682-4689`); I did **not** trace an HPO trial writing those specific keys.
5. **Whether the `genetic` training model's artifact matters at all.** The refuters proved the six knobs are read; whether anything consumes the resulting model is a separate question the models audit raised and this pass did not settle.
6. **`models.data_runtime.*` and anything downstream of the vocabulary work.** ⚠ **IN_FLIGHT** — `crates/neoethos-data` and `vendor/vector-ta-0.2.9-patched` were being modified as this ran, and the workspace had 303 dirty paths. `NEOETHOS_FEATURE_CUBE_MODE` (§4c) and the whole `data_runtime` block should be judged after those land.
7. **Whether `mesh/` or `mcp/` read any config knob directly.** The rewritten guard walks both, so a true orphan there would fail CI. I did not independently enumerate their reads, and `mcp` is served the knob catalog as authoritative (`mcp/server.rs:716-725`) — what an LLM then does with 45 inert env names I cannot bound.
8. **Which of the 24 `Advanced.tsx` controls you actually use.** The UI lie list is ranked by money at stake, not by your usage. If you never touch the preset dropdown, lie #4 is dormant.
9. **The blast radius of re-baselining `initial_equity`.** I confirmed the three fitness constants calibrated against 100 000 and the absolute 100-lot clamp; I did **not** measure how far a stored artifact's score would move. That measurement must precede the change.
10. **Whether `anomaly_guard: false` in both mode arms was a decision or a drift.** Unlike the two export gates, it carries no dated mandate anywhere in the tree.

---

## 9. THE ORDER TO DO IT IN

Safest and highest-clarity first. Money path last and separately. **Steps 1–5 are independent of each other; steps marked ⟷ must land together.**

### Phase A — pure deletion, zero behaviour change (independent, any order)

1. **D1** delete the `apply_overrides_from_lookup` / `load_with_env` layer (~170 lines, 27 env names). Verified unreachable.
2. **D2 ⟷ D4** delete AutoTuner + `AutoTuneHints` + `apply_thread_env_defaults`, together with `system.n_jobs`, `system.num_gpus`, `models.inference_batch_size`. Do **not** re-home the OMP variables.
3. **D3** delete the five `install_*_from_env` wrappers and their constructors (34 env names) ⟷ set `env_var: None` on the corresponding `knob_catalog` entries in the same commit, and rename `SmcSearchConfig::from_env()` → `::current()`.
4. **D7, D8 (partial), D9** — `prop_search_async`/`_wait`, `enable_fsdp`, `symbol_hash_buckets`, the retired preset env entry. Keep `hpo_backend`, `enable_ddp`, `ddp_world_size`.
5. Delete the four tombstoned keys from the live store.

### Phase B — stop lying, before changing anything (independent)

6. **The gate-parity test.** Pin `require_walkforward_for_export`, `prop_firm_min_pass_rate`, `regime_router_enabled`, `multi_resolution_enabled`, `l1_feature_selection_*`, `challenge_mode`, `max_trades_per_day_enabled` to the same value in `Default`, root, seed **and the live store**. Then change the three deliberate Defaults to match your 2026-06-06 decision. *The cheapest single test in the report; it would have caught 5 of the original 73 contradictions.*
7. **Extend `shipped_config_matches_defaults.rs`**: fail on a key present in one file and absent from another, and read `%LOCALAPPDATA%` too. The seed is a strict subset (seed−root = 0), so **generating it from `Settings::default()` is mechanical with no merge problem**.
8. **`deny_unknown_fields`** (or a diff-against-known-keys check) on the raw-YAML endpoint, so a typo is reported instead of saved.
9. **Fix the three wrong rows in `resolved_config.rs`** (`:329`, `:332`, `:337-350`) and the 16 hardcoded `current` values in `knob_catalog.rs`. Add a `config_key` field beside `env_var`.
10. **Delete the `has_edge` framing** from any plan that treats it as a gate; it is a display count.

### Phase C — derive (independent, but 11 gates the rest)

11. **`#[serde(skip)]` on hardware-derived fields** so `Settings::save` stops pickling detector output as input. *Everything else in this phase is cosmetic until this lands.*
12. Give `neuro_evo` a `WorkloadKind`, then delete `system.device` and `system.enable_gpu`.
13. Clamp `NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB` and `_VRAM_BUDGET_MB` against the probe, copying the `gpu_buffer_elem_cap` pattern already in the same file.
14. Make `seen_signature_runtime.max_entries: 0` mean *derive*, not *unbounded*.
15. Collapse the three row budgets; derive `swarm_memory_limit_mb`, `exit_agent_memory_capacity`, `rl_buffer_capacity` ⟷ delete `train_batch_size`.

### Phase D — validators before promotions (independent)

16. **`risk.preset` must re-derive after deserialisation**, or refuse to load a config whose preset disagrees with its derived fields. *This is the single clearest dangerous knob in the report.*
17. **`month_capacity`**: refuse a value below the months spanned by the loaded frame; log the coverage. 💰
18. **`cpcv_max_rows`**: return the coverage fraction and warn below ~50%. 💰
19. **Ambiguous sentinels**: any knob whose name is a maximum or a minimum must not use `0` to mean "off". Applies to `max_portfolio_risk`, `max_in_market`, `min_payoff_ratio`, `min_expectancy_t_stat`. 💰
20. **`commission_per_lot_is_per_side`** must belong to the *source*, not to `risk` — today it doubles the broker-quoted figure too (`discovery.rs:602-620`), latent only because the cTrader path leaves that field `None`. 💰

### Phase E — money path, separately, one at a time, each with a measurement 💰

21. **⟷ Wire live trailing to `models.exit_policy`, THEN delete the four `risk.trailing_*` shadows (D5).** Never the reverse order.
22. **Promote `MONTHLY_RETURN_TARGET`** to `risk.monthly_profit_target_pct`, with a validator refusing 0, and derive the 60-day window bar as 2× it. This gives your prop-firm goal one owner instead of a constant and an orphan.
23. **Promote the `apply_mode_overrides` quality floors**, with a cross-check against `risk.total_drawdown_limit`, and settle `anomaly_guard`.
24. **Promote `max_regime_loss_pct`** and point it at `risk.daily_drawdown_limit`, so the rule that validated a strategy is the rule the account enforces.
25. **Promote `MAX_LEVERAGE`** — with a log line when the cap binds and a refusal above what the account reports.
26. **Reconcile the half-Kelly caps** (0.30 ranking vs 0.25 objective) and the three Kelly multipliers.
27. **Decide `RiskManager`** as one item: construct it in `live_trading::run`, or remove the subsystem and the four knobs together. Until then keep the knobs and mark them `⚠ UNWIRED`.
28. **Wire or remove** `correlation_cap`, `volatility_sigma_pause`, `require_swarm_confidence_min`. A validated value nothing uses is worse than an absent one.
29. **`initial_equity` re-baseline** — last, alone, with the three fitness constants in the same commit, a note that pre-change scores are not comparable, and a re-run of the baseline. The ABI offset (`gpu-contracts/lib.rs:417`) stays pinned.
30. **The UI surface**: render `Advanced.tsx` from `build_catalog()` and add a per-knob write endpoint, so the visible surface grows with the catalog instead of being retyped. This is your own settings-consolidation directive, quoted in the source that fails to satisfy it (`knob_catalog.rs:12-22`).

---

**Nothing in this document was changed. No file was edited, no `cargo` was run, no knob was deleted, no config was patched. Every item above — including every deletion, every merge and every promotion — awaits your decision.**
