# NeoEthos workspace audit — findings ledger

**Started**: 2026-05-24
**Methodology**: read each file ΟΛΟΚΛΗΡΟ. For every finding, document:
- File + line range
- What's wrong (or smells wrong)
- Why it matters (which downstream lies because of it)
- Proposed fix (NOT applied — collect first, batch later)
- Severity: CRITICAL / HIGH / MEDIUM / LOW

**Build policy**: do NOT compile per-finding. Single workspace build at the END.

## Scope

| Crate | Files | Lines | Audit status |
|------|------:|------:|---|
| neoethos-app    | 84 | 43,435 | not started |
| neoethos-models | 65 | 53,151 | not started (3 dead fns already deleted 2026-05-24) |
| neoethos-search | 31 | 20,810 | **in progress** |
| neoethos-core   | 40 | 14,472 | not started |
| neoethos-data   | 22 |  9,232 | partial (#212 hpc_ta) |
| neoethos-cli    | 20 |  5,038 | not started |
| neoethos-codex  |  7 |  1,406 | clean (new crate, no legacy) |
| **TOTAL**       | **269** | **147,544** | |

## Priority order (engine first — if these lie, everything downstream lies)

1. `crates/neoethos-search/src/eval.rs` — backtest core (1211 lines)
2. `crates/neoethos-search/src/discovery.rs` — orchestrator (2900 lines)
3. `crates/neoethos-search/src/validation.rs` — CPCV / OOS (1855 lines)
4. `crates/neoethos-search/src/genetic/search_engine.rs`
5. `crates/neoethos-search/src/genetic/strategy_gene.rs`
6. `crates/neoethos-search/src/genetic/mod.rs`
7. `crates/neoethos-search/src/genetic/smc_indicators.rs`
8. `crates/neoethos-search/src/quality.rs`
9. `crates/neoethos-search/src/portfolio.rs`
10. `crates/neoethos-data/src/core/hpc_ta.rs` (partial)
11. `crates/neoethos-data/src/core/timestamps.rs`
12. `crates/neoethos-data/src/core/*.rs` (feature pipeline)
13. … engine continues
14. `crates/neoethos-app/src/app_services/*.rs` (75 `#[allow(dead_code)]` here)
15. `crates/neoethos-models/src/**.rs`
16. CLI/TUI

---

# Findings

## eval.rs — `crates/neoethos-search/src/eval.rs` (1211 lines, **COMPLETE**)

### F-001 (HIGH) — `BacktestMetrics` has abandoned slot at array index 7
- **Location**: lines 167-200
- **What**: `from_metric_array` jumps from `metrics[6]` directly to `metrics[8]`. `to_metric_array` writes literal `0.0` at index 7.
- **Why it matters**: 11-element array → 10 fields. Index 7 is a phantom slot. Any caller that builds a `[f64; 11]` manually thinking index 7 is data gets `0.0` silently. Magic-position bug.
- **Fix**: convert to named struct field round-trip instead of positional array. Or shrink to `[f64; 10]`. Decide based on whether any caller relies on the 11-element shape.
- **Severity**: HIGH (data corruption risk if anyone hand-rolls metric arrays)

### F-002 (CRITICAL) — `BacktestSettings::default()` uses synthetic EURUSD cost-profile
- **Location**: lines 215-244
- **What**: `Default::default()` calls `infer_market_cost_profile("", "", None, None, None)` which falls back to EURUSD pip math on USD account.
- **Why it matters**: any caller that constructs `BacktestSettings::default()` then runs an actual backtest is using FAKE spreads/commissions. MaxDD/Sharpe numbers from those paths are unreliable. This is the ROOT of bug #214 — fixing the validation harness was surface-level.
- **Real callers** (verified via grep, all run real evaluation):
  - `crates/neoethos-search/src/discovery.rs:710` (struct-update spread — verify field overrides)
  - `crates/neoethos-search/src/gauntlet.rs:60`
  - `crates/neoethos-search/src/parity.rs:217`
  - `crates/neoethos-search/src/eval.rs:1152, 1181` (tests — likely fine)
- **Fix**: see F-003.
- **Severity**: CRITICAL

### F-003 (CRITICAL) — `BacktestSettings::for_symbol` referenced in doc but DOES NOT EXIST
- **Location**: `eval.rs:218-224` comment says "Every backtest entry point should pass a real symbol via `for_symbol(...)`"
- **What**: `grep -rn "fn for_symbol" crates/` returns ZERO matches in `BacktestSettings`. The method is referenced as the migration target but was never implemented. A sibling struct `EvaluationConfig` at `crates/neoethos-search/src/genetic/strategy_gene.rs:582` DOES have `for_symbol` with the correct pattern — that's the template.
- **Why it matters**: Phantom API. A TODO that points to nothing. The fix path that the original author documented was never taken. This is exactly the user's complaint about "code that exists but is not properly tied together".
- **Fix**:
  1. Add `BacktestSettings::for_symbol(symbol, account_currency, price_hint, spread_override, commission_override) -> Self` mirroring `EvaluationConfig::for_symbol`.
  2. Migrate 3 production callers (`discovery.rs:710`, `gauntlet.rs:60`, `parity.rs:217`).
  3. Update `Default::default()` doc to say "tests-only; production paths must use `for_symbol`".
- **Severity**: CRITICAL — fixes F-002 root cause.

### F-004 (MEDIUM) — Two parallel backtest implementations (drift risk)
- **Location**: `fast_evaluate_strategy_core` lines 332-616 AND `simulate_trades_core`-like (no explicit fn name visible, lines 700-905) inside the same file
- **What**: Two near-identical loops doing the same backtest: same entry causality, same SL/TP geometry, same kill-zone gates. Only difference: the second variant returns per-trade `Trade` records (for journaling). Both reference the "Bug #1 fix: half-spread at entry" pattern, suggesting they were synced once but there is no compile-time guarantee they stay in sync.
- **Why it matters**: When a bug is found in one (e.g. the historical intra-bar lookahead that was fixed at line 595 of `fast_evaluate_strategy_core`), the other path may silently keep the bug. Two separate stores of "what is a trade?" can drift.
- **Fix**: extract the per-bar simulation loop into a single `step` function called by both. Each variant only differs in whether it accumulates a `Trade` record. Estimated refactor ~150 LOC delta, no behavior change.
- **Severity**: MEDIUM (no immediate correctness bug; latent drift risk)

### F-005 (LOW) — `FOREX_BOT_DISABLE_SMC_GATE` env var hidden bypass
- **Location**: lines 947-951 + same pattern in `genetic/search_engine.rs:261`
- **What**: setting this env to "1"/"true"/"TRUE" silently bypasses the SMC indicator gate (sets `active_sum = 0`). Documented in the comment as "Lets operators isolate ... without recompiling".
- **Why it matters**: behavior changes invisibly based on environment. A trader who set this once and forgot will get different results from someone who didn't. No log line announces the bypass.
- **Fix**: at startup, if any `FOREX_BOT_*` env var is set, log a single `tracing::warn!` listing them so the operator can see what's been overridden.
- **Severity**: LOW

### F-006 (NOTE) — `init_rayon` IS wired (false alarm)
- **Location**: defined `eval.rs:40`, called `eval.rs:1016` from `evaluate_population_core`
- **What**: my initial suspicion that `init_rayon` was disconnected was wrong. It IS called from the public eval entrypoint.
- **Severity**: NONE — verification only.

---

# Sessions
- **2026-05-24 session 1**: scaffolded ledger; audited eval.rs COMPLETE (1211/1211 lines); surfaced F-001, F-002, F-003, F-004, F-005, F-006. Critical findings: F-002 + F-003 are the root cause of bug #214 (cost-model with empty symbol). Fixes deferred to "batch implementation" phase per user directive: build only once at end.

## discovery.rs — `crates/neoethos-search/src/discovery.rs` (2900 lines, **COMPLETE**)

### F-007 (HIGH) — `evaluation_account_currency: "USD"` hardcoded in `from_settings`
- **Location**: `discovery.rs:275`
- **What**: `from_settings` builds the `DiscoveryConfig` from `Settings`, but the account currency is a literal `"USD".to_string()` — does NOT read from settings.
- **Why it matters**: cost-model lookup uses this. A user with a GBP demo account (we have one!) gets USD-based pip conversion. Already manifested as bug #181 ("currency shows £ but settings say USD"). The "fix" for #181 was UI-side; the engine still hardcodes USD.
- **Fix**: read `settings.system.account_currency` (need to add to `SystemConfig` first if not present). NO synthetic fallback — bail if missing.
- **Severity**: HIGH

### F-008 (MEDIUM) — `corr_threshold: 0.85` hardcoded in `from_settings`
- **Location**: `discovery.rs:294`
- **What**: hardcoded 0.85 even though `DiscoveryConfig::default()` has the same value at line 215 — `from_settings` should pull from `settings.models.*` like the other fields do, OR document why it deliberately ignores settings.
- **Severity**: MEDIUM (matches default so behavior identical — but the silent ignoring of settings is the smell)

### F-009 (LOW) — `max_regime_loss_pct: 3.0` hardcoded in `from_settings`
- **Location**: `discovery.rs:307`
- **What**: same pattern as F-008. Hardcoded 3.0 instead of reading from settings.
- **Severity**: LOW

### F-010 (MEDIUM) — `portfolio_size: 2000` default seems unrealistic
- **Location**: `discovery.rs:211` (`Default`); from_settings reads from config so this is just the Default-path
- **What**: a portfolio of 2000 discovered strategies cannot be traded — real prop-firm operators run 5-20 simultaneously, broker margins limit far below 2000. Likely set to "use all candidates" sentinel without explicit comment.
- **Fix**: either lower the default to e.g. 20, or document why 2000 is "the candidate pool size, not the deployed-strategy count".
- **Severity**: MEDIUM (UX/semantics confusion, not correctness)

### F-011 (MEDIUM) — `with_env_runtime_overrides` silently switches ~10 fields when mode=PropFirm
- **Location**: `discovery.rs:325-357`
- **What**: when `FOREX_BOT_DISCOVERY_MODE` resolves to `PropFirm`, the method overrides: `filtering.max_dd`, `filtering.min_profit`, `filtering.min_trades`, `filtering.min_sharpe`, `filtering.min_win_rate`, `filtering.min_profit_factor`, `filtering.anomaly_guard`, `cpcv_min_phi`, `min_trades_per_day`, and installs `prop_firm_gate`. Heavy silent behavior change driven by one env var, no startup log line announcing the mode.
- **Why it matters**: identical inputs → wildly different discovery outputs depending on env. Reproducibility hazard. Operator could forget the env was set last week.
- **Fix**: log a single `tracing::info!` at discovery start naming the resolved mode + which fields were overridden.
- **Severity**: MEDIUM

### F-012 (CRITICAL) — `discovery_backtest_settings` inherits EURUSD-synthetic fields via struct-update `..BacktestSettings::default()`
- **Location**: `discovery.rs:684-712` (lines 690 + 710 specifically)
- **What**: this helper builds the per-gene `BacktestSettings` by overriding ~10 named fields (sl_pips, tp_pips, max_hold_bars, trailing*, pip_value, spread_pips, commission_per_trade, pip_value_per_lot, kill_zones_enabled) and ends with `..crate::eval::BacktestSettings::default()`. The default constructor (F-002) builds with synthetic EURUSD cost-profile, so any field NOT in the override list (e.g. `min_hold_bars`, `slippage_pips`, `enable_intrabar`, `commission_min_per_trade`, account-currency-dependent micro-adjustments) leaks the synthetic profile into every backtest the discovery pipeline runs.
- **Why it matters**: this is the production-side EVIDENCE of F-002/F-003. The five public entry points (`build_discovery_validation_artifacts`, `compute_discovery_forward_test_artifacts`, `compute_discovery_prop_firm_artifacts`, `evaluate_cpcv_gate`, the MC perturbation loop) all reach `discovery_backtest_settings`, which all reach `BacktestSettings::default()`. So every persisted artifact (canonical_backtest, walkforward, forward_test, prop_firm) has the same hidden EURUSD bias.
- **Fix**: once F-003 ships `BacktestSettings::for_symbol(...)`, change line 710 from `..BacktestSettings::default()` to `..BacktestSettings::for_symbol(&config.evaluation_symbol, &config.evaluation_account_currency, price_hint, None, None)`. Then the named overrides act as deltas on top of a CORRECT base profile.
- **Severity**: CRITICAL — silently contaminates every discovery artifact.

### F-013 (HIGH) — `validate_regime_robustness` has dead-zones where trades count for nothing
- **Location**: `discovery.rs:1564-1627`, specifically lines 1607-1617
- **What**: the function buckets per-trade PnL by regime:
  - `trend_str > 0.25` → trend bucket
  - `trend_str < 0.15` → range bucket
  - `0.15 ≤ trend_str ≤ 0.25` → NO bucket (silently dropped)
  - `vol_state > 0.5` → high-vol bucket
  - `vol_state < -0.5` → low-vol bucket
  - `-0.5 ≤ vol_state ≤ 0.5` → NO bucket
- **Why it matters**: trades in the dead-zones are not counted toward any regime PnL, so the gate's `if trend_pnl < limit || range_pnl < limit ...` check sees an artificially small set. A strategy that always trades in 0.15-0.25 trend regime would pass the gate trivially with all four buckets at zero. The gate ALSO silently returns `true` (passing) when `regime_trend_strength` or `regime_vol_state` features are absent from `features.names` (line 1576-1578).
- **Fix**:
  1. Decide whether dead-zone trades go into "trend" or "range" (likely range since 0.15-0.25 is weak-trend). Make the boundaries adjacent: e.g. `trend_str >= 0.20 → trend, else range`.
  2. Same for vol: `vol_state >= 0.0 → high, else low` (or document the dead-zone choice explicitly).
  3. When regime features are missing, return `false` (fail the gate) OR log a tracing::warn — silently passing is a "gate that does nothing" failure mode.
- **Severity**: HIGH (gate that silently does nothing)

### F-014 (MEDIUM) — `quality_analyzer_for_config` uses `min_sharpe` as `min_sortino`
- **Location**: `discovery.rs:670-682` (lines 672-673)
- **What**: `min_sortino: config.filtering.min_sharpe.max(0.0)` — both ratios get the same threshold even though Sortino is typically 1.5-2× Sharpe for the same strategy (downside-only denominator). A min_sharpe=1.0 floor effectively becomes min_sortino=1.0, which is much weaker than the natural Sortino bar for a real signal.
- **Why it matters**: gate is weaker than the user thinks. Strategies with poor downside risk-adjusted returns (low Sortino) pass because we're checking against a number meant for Sharpe.
- **Fix**: add `min_sortino: f64` field to `FilteringConfig` with sensible default (e.g. 1.5× the Sharpe floor) and read from settings. Or, document why min_sharpe doubles as min_sortino (which I don't think there's a reason).
- **Severity**: MEDIUM

### F-015 (HIGH) — Magic numbers everywhere in MC + sensitivity + income-score pipeline
- **Locations**: `discovery.rs` lines 1834-1845, 1990, 1996-2005, 2033, 2040-2041
- **What**: the post-search quality screen and ranking use a cluster of hardcoded magic numbers that materially affect which strategies survive:
  - `pf_capped = gene.profit_factor.min(3.0) / 3.0` (line 1834) — PF cap = 3.0
  - `safety = 1.0 - gene.max_drawdown / 0.07` (line 1835) — DD safety floor = 7%
  - `consistency_score > 0.8 → 2x bonus` (line 1843) — bonus threshold + magnitude
  - `mc_runs = 100usize` (line 1990) — Monte Carlo run count
  - `+/- 15%` threshold perturbation (lines 1996-1997)
  - `+/- 20%` weight perturbation (line 1999)
  - `+/- 25%` SL/TP perturbation (lines 2001-2005)
  - `if profitable_runs < 70` (line 2033) — 70% MC pass-rate threshold
  - `spread_pips = 2.0; commission_per_trade = 7.0` (lines 2040-2041) — sensitivity test hardcoded to EURUSD-grade costs
- **Why it matters**:
  1. The 70/100 MC threshold is a HARD gate, not just a ranking. Changing it from 70 to 60 likely changes the surviving portfolio size dramatically — and the operator has no way to do that without recompiling.
  2. The sensitivity test (2pip + $7) is EURUSD-biased. For JPY pairs (where 2pip ≠ same proportion of price as on EURUSD) and exotics (5-15 pip realistic spread) this test is too lenient. Cross-pair fairness is broken.
  3. The income-score magic numbers (PF cap, DD floor, bonus threshold) jointly determine candidate ranking. They should be in the audit log so we can A/B them.
- **Fix**:
  1. Add a `QualityScreenConfig` (or extend `FilteringConfig`) with all magic numbers as named fields.
  2. `compute_sensitivity_settings(symbol, base_settings)` → derive sensitivity spread/comm from the SAME `for_symbol` cost-profile, scaled by some "worst-case multiplier" (e.g. spread × 1.5, comm × 1.5).
  3. Document why 70/100 is the floor (or pick a different floor based on backtested edge significance).
- **Severity**: HIGH (sensitivity test on JPY/exotic is essentially broken — F-016 below specifies the symbol-bias half)

### F-016 (HIGH) — Sensitivity test hardcoded to EURUSD-grade costs
- **Location**: `discovery.rs:2037-2053` (specifically lines 2040-2041)
- **What**: `sensitive_settings.spread_pips = 2.0; sensitive_settings.commission_per_trade = 7.0;` — these are reasonable worst-case for EURUSD, but completely wrong for:
  - EURJPY (1 pip = 0.01, so 2pip = 200 yen × pip_value_per_lot — and the real broker spread is usually 1.0-1.5pip, not 2.0)
  - GBPUSD (real spread 1.5pip typical, 2.0 is reasonable worst-case — OK)
  - XAUUSD (gold, spread varies wildly, 2pip = $0.02 which is far below real)
  - GBPNZD (exotic, 5-10pip real spread → 2pip is OPTIMISTIC, not pessimistic)
- **Why it matters**: a strategy that survives the EURUSD-grade sensitivity test on a GBPNZD signal is not actually robust to GBPNZD costs. F-016 + F-012 together mean the entire cost model in discovery is silently EURUSD-shaped.
- **Fix**: replace literals with `sensitive_settings = BacktestSettings::for_symbol(symbol, ..., spread_override=Some(base_spread * 1.5), commission_override=Some(base_comm * 1.5));` once F-003 lands.
- **Severity**: HIGH (compounds F-012)

### F-017 (MEDIUM) — Train ratio mismatch: eval.rs uses 0.80, discovery.rs uses 0.70
- **Locations**: `eval.rs:953` (`wfv_bound = n_rows * 0.8`) vs `discovery.rs:830, 1077` (both `train_ratio: 0.70`)
- **What**: the GA fitness evaluation (eval.rs `evaluate_population_core`) uses 80/20 train/OOS split. The post-search validation (discovery.rs `build_discovery_validation_artifacts` → `embargoed_walkforward_backtest`) uses 70/30. The discovery temporal-contract hash claims 0.70.
- **Why it matters**: two different walk-forward setups, no explicit doc on why. If both are intended, the choice should be documented. If one is wrong, we have a bug. The hash claims 70/30 but the GA fitness is built on 80/20 — so the persisted contract describes a different split than the GA actually used to select the candidate.
- **Fix**: pick ONE canonical ratio for in-sample fitness vs OOS validation OR document the two-stage design explicitly (e.g. "GA uses 80/20 to maximise the search horizon, post-search validation uses 70/30 to reserve more OOS for the gate"). Promote to a named constant in `discovery::TRAIN_RATIO_VALIDATION` and `eval::TRAIN_RATIO_GA_FITNESS`.
- **Severity**: MEDIUM

### F-018 (MEDIUM) — `evaluate_cpcv_gate` returns "passed" without running CPCV when disabled
- **Location**: `discovery.rs:948-950`
- **What**: `if !config.enable_cpcv { return Ok((true, 0, 1.0)); }` — when CPCV is disabled, the gate returns `(passed=true, fold_count=0, profitable_ratio=1.0)`. Then upstream `validation_gates.cpcv_passed = true` is recorded in the persisted profile and ALSO satisfies `is_portfolio_export_ready` (via `walkforward_passed && cpcv_passed`).
- **Why it matters**: the persisted profile says `cpcv_passed=true` for a run where CPCV NEVER RAN. An operator reading the profile reasonably assumes CPCV validated the portfolio. The `profitable_fold_ratio=1.0` is even more misleading — claims 100% folds profitable when zero folds were run.
- **Fix**: when CPCV is disabled, return `cpcv_passed=true` but also set `cpcv_fold_count=None` (change the type to Option<usize>) OR `cpcv_profitable_fold_ratio=None`. Persist the disabled state distinctly from "ran and passed". The `is_portfolio_export_ready` check then needs to allow `cpcv_disabled || cpcv_passed`.
- **Severity**: MEDIUM (profile lies about what was validated)

### F-019 (LOW) — Hardcoded `min_trading_days: 0, max_trades_per_day: 0` in walkforward call
- **Location**: `discovery.rs:1083-1084`
- **What**: `embargoed_walkforward_backtest` is called with these two limits zeroed (= disabled). But `config.min_trades_per_day` exists at line 220 and is used elsewhere (e.g. `min_trades_required` line 1879). So the gate is disabled in the validation walkforward but enabled in the candidate filter — inconsistent.
- **Fix**: either thread `config.min_trades_per_day` into the walkforward call OR document why we disable it there.
- **Severity**: LOW

---

## validation.rs — `crates/neoethos-search/src/validation.rs` (1855 lines, **COMPLETE**)

Module breakdown: production code lines 1-1277, tests 1280-1855. Most of the production code is artifact-schema/scope/IO boilerplate (Canonical / Walkforward / ForwardTest / LiveExecutionSimulation / PropFirmRisk × { Scope, ArtifactFile, validate, atomic-write, read }). Real logic lives in `compute_prop_firm_risk_summary` (742-836), `walkforward_risk_diagnostics` (884-1008), `embargoed_walkforward_backtest` (1010-1173), and `CombinatorialPurgedCV::split` (1192-1276).

### F-020 (HIGH) — `embargoed_walkforward_backtest` hardcoded min-window thresholds are timeframe-agnostic
- **Location**: `validation.rs:1054, 1061`
- **What**: each candidate split must satisfy:
  - `end - start >= 80` (line 1054, else `break` ends the whole loop)
  - `(train_end - start) >= 40 && (end - test_start) >= 40` (line 1061, else `continue` skips THIS split)
- **Why it matters**: 80 bars is meaningless without a timeframe — on M1 that's 80 minutes (≈1.3h), on D1 it's 80 days (≈4 months). With H4 data and `n_rows = 1500` + `n_splits = 5`, `window = 300`, train = 210, test = 70 — passes. But with the same H4 data and `n_splits = 20`, `window = 75 < 80` → entire loop breaks at first iteration → `split_results` empty → walkforward gate fails for "no splits", not "no edge". The break-not-continue at line 1054 makes ALL subsequent splits dead too. Combined with the **F-017 train_ratio mismatch** this means the validation harness can silently produce zero splits without explanation.
- **Fix**:
  1. Replace `break` with `continue` at line 1054 so a single small window doesn't kill the rest.
  2. Convert the thresholds to be expressed as "min N bars OR equivalent time-span (e.g. >= 2 calendar weeks of data)" — let the caller supply both bounds, default the time-bound from `config.timeframe_label`.
  3. Emit a tracing::warn when a split is skipped so operator sees "skipped split 3/5: train_size 35 < 40 minimum".
- **Severity**: HIGH (silent gate failure mode on small-window timeframes)

### F-021 (MEDIUM) — Walkforward silently degrades timestamps → days when length mismatches
- **Location**: `validation.rs:921-926` (inside `walkforward_risk_diagnostics`) and `1071-1075` (inside `embargoed_walkforward_backtest`)
- **What**: both helpers contain `let ts = if timestamps.len() == close.len() { timestamps } else { days };` — when the timestamps array length doesn't match, simulation falls back to using the `days` array as timestamps. `simulate_trades_core` uses timestamps for: gap detection (lookahead at session boundaries), kill-zone rules, intraday session classification. Day-level "timestamps" make all of these coarse-grained or wrong.
- **Why it matters**: a caller bug (passing wrong-length array) becomes a silent quality degradation instead of a hard failure. The walkforward summary will look valid but the simulation underneath is not.
- **Fix**: change the fallback to a hard `bail!("timestamps length must match close length OR be empty")` — empty is the documented "no timestamps available" path (line 845-846 of `WalkforwardBacktestInput` says "ms or ns" but doesn't mention empty as a valid input — clean this up too).
- **Severity**: MEDIUM (caller-bug-masking)

### F-022 (MEDIUM) — `normalized_pct_threshold` boundary ambiguity at value=1.0
- **Location**: `validation.rs:874-882`
- **What**: the helper accepts pct in two forms: fraction (e.g. `0.05` = 5%) or percentage-points (e.g. `5.0` = 5%). It picks based on `value > 1.0`. The boundary value `1.0` is treated as a fraction (= 100%), NOT as "1 percent". A config writer who sets `max_daily_loss_pct: 1.0` expecting "1% max daily loss" gets `100% max daily loss` (i.e. the gate is disabled).
- **Why it matters**: silent severe misconfiguration. The 1% daily-loss gate is plausible for some prop firms; writing it as `1.0` is the natural form and the auto-detect gets it backwards.
- **Fix**: kill the auto-detect. Pick ONE convention (recommend "always pct-points: 5.0 means 5%") and document it on every caller. Validate that callers use the correct form — or rename the field to `max_daily_loss_fraction` to make the unit explicit.
- **Severity**: MEDIUM (config footgun)

### F-023 (LOW) — `max_profit_consistency_ratio` carries FIXME(hardcoded) marker untouched
- **Location**: `validation.rs:607`
- **What**: `PropFirmRiskRules::default()` sets `max_profit_consistency_ratio: 0.50` with an explicit `// FIXME(hardcoded): config-extract — internal consistency-ratio cap.` comment. Per directive 2026-05-14 the rest of the prop-firm defaults come from `PropFirmConstraints::FTMO_STANDARD`, but this one is still inline.
- **Fix**: add `consistency_ratio_cap` to `PropFirmConstraints` (or `PropFirmChallengeDefaults`) and read it from there.
- **Severity**: LOW

### F-024 (LOW) — Hardcoded `100_000.0` initial-balance fallback in two places
- **Location**: `validation.rs:745-749` (`compute_prop_firm_risk_summary`), `903-907` (`walkforward_risk_diagnostics`)
- **What**: when caller passes a non-finite or non-positive `initial_balance`, both helpers default to `$100,000`. That's the standard FTMO challenge size but the magic number lives in two places and is undocumented.
- **Fix**: either error out instead of defaulting (force callers to be explicit) or promote to a named constant `DEFAULT_PROP_FIRM_BALANCE` exported from `neoethos_core::domain::prop_firm::PropFirmConstants`.
- **Severity**: LOW

---

## gauntlet.rs — `crates/neoethos-search/src/gauntlet.rs` (154 lines, **COMPLETE**)

### F-025 (CRITICAL) — `GauntletConfig::default()` is the second confirmed F-002 caller site
- **Location**: `gauntlet.rs:60`
- **What**: `backtest: BacktestSettings::default()` — same struct-update pattern as F-012 in discovery.rs. `StrategyGauntlet::run()` then only overrides `settings.sl_pips` and `settings.tp_pips` from the gene (lines 91-93). All cost-profile fields (`pip_value`, `spread_pips`, `commission_per_trade`, `pip_value_per_lot`) leak the synthetic EURUSD profile.
- **Why it matters**: every strategy that passes through the gauntlet is evaluated under synthetic EURUSD costs regardless of which symbol it's intended for. Combined with F-012 this means the gauntlet endorses strategies that survive EURUSD economics but might fail on the real symbol.
- **Fix**: same as F-003 — when `BacktestSettings::for_symbol` lands, change line 60 to `backtest: BacktestSettings::for_symbol(...)`. Also: the `StrategyGauntlet::run()` signature takes no symbol; thread the symbol through `Gene` (or pass it as an argument) so the gauntlet can build the correct profile.
- **Severity**: CRITICAL (compounds F-002/F-012)

### F-026 (LOW) — Internal vs prop-firm DD/daily caps documented with `debug_assert` cross-check
- **Location**: `gauntlet.rs:42-53`
- **What**: NOT a bug — this is the GOOD pattern. `DEFAULT_MAX_DRAWDOWN_PCT = 0.07` is intentionally below `FTMO_STANDARD.max_overall_drawdown_pct = 0.10`. The `debug_assert!` catches the inversion at startup.
- **Note**: file this as a reference example for how to handle internal-tunable thresholds across audit-extracted finding fixes.
- **Severity**: NONE — pattern reference.

---

## parity.rs — `crates/neoethos-search/src/parity.rs` (315 lines, **COMPLETE**)

### F-027 (LOW) — Test-only F-002 caller + 11-shape coupling
- **Location**: `parity.rs:217` (test helper) + line 45-46 + 63-64 (signature)
- **What**:
  1. `..BacktestSettings::default()` in test helper `backtest_settings()` — minor; just means parity tests use EURUSD-synthetic context. Not a runtime risk but the tests don't catch F-002 because they all use the same synthetic.
  2. Hardcoded `[f64; 11]` shape in `compare_metric_matrices` signature and tests — directly coupled to F-001. If F-001 shrinks the metric array to `[f64; 10]`, this file needs simultaneous update.
- **Fix**: when F-001 lands, update parity.rs to track the new shape. Also: add at least one parity test that uses `BacktestSettings::for_symbol("EURJPY", "USD", ...)` so JPY-pair parity is covered.
- **Severity**: LOW (test-only)

---

## strategy_gene.rs — `crates/neoethos-search/src/genetic/strategy_gene.rs` (649 lines, **COMPLETE**)

This is the file that holds `EvaluationConfig::for_symbol` (line 582-605) — the **template** F-003 should mirror in `BacktestSettings::for_symbol`. Verified the shape:
```rust
pub fn for_symbol(
    symbol: &str,
    account_currency: &str,
    price_hint: Option<f64>,
    spread_pips_override: Option<f64>,
    commission_override: Option<f64>,
) -> Self
```
Internally calls `infer_market_cost_profile(...)` and overrides the 6 cost-profile fields from the resolved profile. Same signature shape should land in `BacktestSettings::for_symbol`.

### F-028 (HIGH) — `Gene::is_anomalous()` has 4 overlapping anomaly classifications with all magic numbers
- **Location**: `strategy_gene.rs:356-391`
- **What**: four independent thresholds reject strategies as "too good to be true":
  - `suspicious_combo`: trades ≥ 120 AND dd ≤ 0.25% AND win_rate ≥ 92% AND PF ≥ 12 AND profit ≥ $10M
  - `suspicious_ppt`: trades ≥ 40 AND dd ≤ 1% AND profit-per-trade ≥ $100k
  - `suspicious_ultra`: trades ≥ 50 AND dd ≤ 0.1% AND profit ≥ $7.5M AND ppt ≥ $50k
  - `suspicious_low_dd`: trades ≥ 80 AND dd ≤ 0.1% AND profit ≥ $2.5M
- **Why it matters**:
  1. A real prop-firm-grade strategy that hits 4-10%/mo (the documented target per the comment) compounded on a $10k base over 10y gives ~$11M target equity. That's RIGHT NEXT to the `min_profit = 10_000_000` bar. A genuine 4%/mo strategy over 11 years would cross $10M and could trip `suspicious_combo` if its other metrics are also strong. The comment claims "raised 50× so genuine target-hitting strategies are not discarded" but the calibration math is opaque.
  2. The four classifications OVERLAP heavily — a strategy that trips one likely trips two — but the code treats them as OR'd independent gates. The thresholds were tuned in lockstep, not independently.
  3. No way to tune any of these from config — all baked into source. Operators have NO knob to relax this for genuinely-good runs.
- **Fix**:
  1. Promote to `AnomalyGuardConfig` struct on `FilteringConfig` with all 4 classification thresholds.
  2. Default to today's values (preserve behaviour).
  3. Log at `tracing::warn!` when a strategy is anomaly-flagged so operator sees which classification + which threshold tripped.
- **Severity**: HIGH (silent good-strategy rejection risk on the 10y compounded backtest)

### F-029 (MEDIUM) — `infer_market_cost_profile` asset-class default spreads + flat $7 commission
- **Location**: `strategy_gene.rs:331-343`
- **What**: when no spread/commission is provided from runtime overrides OR explicit override, the function falls back to:
  - metal: 2.5 pips
  - crypto: 8.0 pips
  - fx: 1.5 pips
  - other: 1.0 pip
  - commission: $7.00 per trade (universal)
- **Why it matters**: these defaults are EURUSD-grade fx, XAUUSD-grade metal, BTCUSD-grade crypto. They are wrong for:
  - EURGBP (typical 1.5 OK but cross-pair adds non-trivial slippage not modelled)
  - GBPNZD (typical 5-10pip real spread)
  - USDMXN (typical 200+pip "raw" or 2-4pip with normalization)
  - XAGUSD (typical 4-7pip spread, NOT 2.5)
  - ETHUSD (typical 5-15pip on most brokers)
- **The existing TODO(real-data)** at lines 323-330 acknowledges this explicitly — "Once `symbol_metadata::SymbolMetadata` is extended with broker-supplied `typical_spread_pips` and `commission_per_lot` fields (sourced from the cTrader account / commission plan), remove these magic defaults and bail when the metadata is missing."
- **Fix**:
  1. Extend `neoethos_core::symbol_metadata::SymbolMetadata` with `typical_spread_pips: Option<f64>` and `commission_per_lot: Option<f64>`.
  2. Source these from cTrader symbol records (look in `ctrader_data` / `ctrader_messages` for `ProtoOASymbolCategory` parsing) when the user connects.
  3. When metadata is missing, BAIL (fail-loudly) instead of silently using EURUSD-grade defaults. Add a `FOREX_BOT_ALLOW_SYNTHETIC_SPREADS=1` env override for backtests on symbols without real metadata.
- **Severity**: MEDIUM (multi-symbol cost accuracy)

### F-030 (LOW) — `Gene::normalize()` hardcoded fallbacks for invalid genes
- **Location**: `strategy_gene.rs:483, 491-505`
- **What**: when a gene has NaN/invalid fields, normalize() fills in:
  - long_threshold = 0.25, short_threshold = -0.25 (line 491, 494)
  - tp_pips = 40, sl_pips = 20 (lines 502, 505)
  - weights clamped to [-5.0, 5.0] (line 483)
- **Why it matters**: these magic numbers determine the behaviour of a "salvaged" gene. They probably came from "what looks reasonable" but are not derived from any cost-profile or per-symbol consideration. A salvaged gene meant for XAUUSD with `sl_pips = 20` will have a much tighter SL (in price terms) than the operator might expect.
- **Fix**: extract to `GeneNormalizationDefaults` config (or use `FilteringConfig` slots). Or, alternatively: don't salvage genes with NaN fields — discard them entirely so the GA pool stays clean.
- **Severity**: LOW

### F-031 (LOW) — `FilteringConfig::default()` has 8+ undocumented magic numbers
- **Location**: `strategy_gene.rs:76-100`
- **What**: defaults for max_dd, min_profit, min_trades, min_sharpe, min_win_rate, min_profit_factor, trade_log_max are all hardcoded with no derivation. Some are reasonable (min_profit_factor: 1.05 ≈ "make $1.05 for every $1 lost") but others (min_sharpe: 0.3?) are oddly weak for a prop-firm-grade gate.
- **Fix**: add doc-comment per field stating provenance OR sourcing from a published threshold (e.g. "Sharpe ≥ 0.3 per Cliff's quant-edge threshold").
- **Severity**: LOW

---

## genetic/search_engine.rs — `crates/neoethos-search/src/genetic/search_engine.rs` (1060 lines, **COMPLETE**)

### F-032 (CRITICAL) — `signals_for_gene` doc-comment lies about SMC gating
- **Location**: `search_engine.rs:111-203`
- **What**: the function's doc-comment claims "Apply the SMC-flag gate using the same scoring as `synthesize_signals_cpu`" (line 127-131), and item 6 of the search-optimization notes warns the post-search filter used to produce a "signal series that did NOT match what was actually evaluated and archived during search". BUT — the actual implementation at lines 171-202 is identical for both `any_flag=true` and `any_flag=false` cases (both run the bare threshold logic and return). The "Need OHLCV-derived SMC indicator series" comment at line 186 admits this: "Without OHLCV we fall back to the un-gated path so single-arg callers (no Ohlcv handy) keep working; gated callers should use `signals_for_gene_full`."
- **Why it matters**: callers reading the public API doc believe they get gated signals. They get UN-GATED signals. The function's job per its doc and per item 6 is to "Apply the SMC-flag gate" but it does NOT. A caller that built a Gene with `use_ob=true` then called `signals_for_gene(features, &gene)` will get more signals than the evaluator would have generated for the same gene. The post-search min_trades gate then over-counts trades.
- **The only caller in production code is `gauntlet.rs:85`** — so the gauntlet's min_trades / win_rate / pf comparisons are computed against un-gated signals. The gauntlet may pass strategies whose gated trade count is below `min_trades`.
- **Fix**: either
  1. Fix the implementation to actually gate (using `signals_for_gene_full` internally with default SMC arrays built from a `FeatureFrame`-only fallback), OR
  2. Rewrite the doc to plainly say "this is the un-gated path; use `signals_for_gene_full` when you need gating; the gauntlet must migrate to `signals_for_gene_full`".
- **Severity**: CRITICAL (silently wrong gauntlet results)

### F-033 (CRITICAL) — Third + fourth F-002 caller sites in `evaluate_genes` and `evaluate_genes_cached`
- **Location**: `search_engine.rs:355-365` (cached), `450-460` (non-cached)
- **What**: both functions build `BacktestSettings` with `..Default::default()` (the synthetic EURUSD profile from F-002). Identical pattern to F-012 in discovery.rs and F-025 in gauntlet.rs. With these two more sites confirmed, the EURUSD profile leak now hits **4 production sites**:
  1. discovery.rs:710 (validation/forward/prop-firm artifact generation)
  2. gauntlet.rs:60 (gauntlet gate)
  3. search_engine.rs:355 (evaluate_genes_cached — main GA evaluation loop)
  4. search_engine.rs:450 (evaluate_genes — the unbuffered evaluation path)
- **Why it matters**: the GA itself (#3 + #4) selects strategies based on EURUSD-shaped fitness. Strategy selection is built on the wrong cost model. Any "good" candidate the GA finds is good relative to EURUSD economics, not the target symbol.
- **Fix**: when F-003 lands, change all four sites to `..BacktestSettings::for_symbol(...)`. Since the EvaluationConfig already carries the symbol+account_currency (via `config.symbol`, `config.account_currency`), this becomes a 1-line change per site.
- **Severity**: CRITICAL (compounds F-002)

### F-034 (MEDIUM) — `resolve_stop_target_arrays` hardcoded EURUSD-style pip fallback
- **Location**: `search_engine.rs:486-524`
- **What**:
  - Line 491-494: `pip_size = if config.pip_value.is_finite() && > 0 { config.pip_value } else { 0.0001 }` — the fallback `0.0001` is EURUSD-style. For JPY pairs (`0.01`), metals (`0.01`), crypto (`1.0`) this is wrong.
  - Line 505-507: `let (default_sl, default_tp) = default.map(...).unwrap_or((20.0, 40.0))` — same 20/40 magic as F-030 (Gene::normalize), but used here only when `infer_stop_target_pips` returns None.
- **Why it matters**: invalid genes that fall through to fallback get EURUSD-shaped stops. For YEN pairs this means a 20-pip SL is 0.20 yen = ~$0.20 — far too tight.
- **Fix**: read pip_size from `infer_market_cost_profile(config.symbol, ...).pip_value`. Promote 20.0 / 40.0 to named constants `DEFAULT_SL_PIPS_EURUSD` / `DEFAULT_TP_PIPS_EURUSD` and bail when symbol is unknown.
- **Severity**: MEDIUM

### F-035 (MEDIUM) — `best_return_count` formula is opaque magic
- **Location**: `search_engine.rs:868-870`, `903-904`
- **What**: `let best_return_count = population.clamp(2, (population / 2).clamp(100, 500)).min(scored.len());` — what does this even do?
  - population=50 → pop/2=25 → clamp(100,500)=100 → 50.clamp(2,100)=50 (return all 50)
  - population=200 → pop/2=100 → clamp(100,500)=100 → 200.clamp(2,100)=100 (return 100)
  - population=1500 → pop/2=750 → clamp(100,500)=500 → 1500.clamp(2,500)=500 (return 500)
  - population=10000 → pop/2=5000 → clamp(100,500)=500 → 10000.clamp(2,500)=500 (return 500)
- **Why it matters**: behavior changes nonlinearly across population sizes and operators have no obvious knob to control "how many candidates does the GA return for downstream filtering". The hard 500 cap on large populations silently drops the tail of the population.
- **Fix**: replace with a named config field like `return_top_k_fraction` (default 0.5) with explicit min/max bounds and a doc-comment.
- **Severity**: MEDIUM

### F-036 (MEDIUM) — SMC gate stagnation-decrement has no lower bound before clamp
- **Location**: `search_engine.rs:732-734`
- **What**: `gate_now -= gate_stagnation_step * (stagnant_gens as f32);` — multiplied by `stagnant_gens` which can grow unboundedly across generations. The subsequent `.clamp(gate_lo, gate_hi)` saves it, but the intermediate value can be NEGATIVE / very-large-negative. Fine numerically, but the clamp at `gate_lo` is the only thing protecting. If `gate_lo > gate_hi` (because `min().min(max())` etc. was passed garbage), clamp behaves weirdly.
- **Fix**: tighten to `gate_now = (gate_now - gate_stagnation_step * stagnant_gens as f32).max(gate_lo).min(gate_hi);` and add a debug_assert that `gate_lo <= gate_hi`.
- **Severity**: MEDIUM (low blast radius in practice, but the gate's calibration shouldn't depend on a clamp)

### F-037 (LOW) — Magic factors in stagnation-recovery branches
- **Location**: `search_engine.rs:944-948`, `969-973`
- **What**:
  - Line 945: `(search_policy.survivor_fraction * 0.75).clamp(0.0, 0.5)` — magic 0.75 multiplier + magic 0.5 upper bound when stagnant.
  - Line 970: `search_policy.immigrant_fraction.max(0.5)` — magic 0.5 lower bound for immigrants when stagnant.
  - Line 1016: `while b_idx == a_idx && retries < 4` — magic 4 retries to find a distinct second parent.
- **Fix**: extract as named constants in module scope (`STAGNATION_SURVIVOR_MULTIPLIER = 0.75`, etc.) with doc explaining the choice. OR (preferred) add fields to `EvolutionSearchPolicy` so they're tunable per run.
- **Severity**: LOW

---

## genetic/smc_indicators.rs — `crates/neoethos-search/src/genetic/smc_indicators.rs` (659 lines, **COMPLETE**)

This module computes/derives 11 SMC indicator arrays from either OHLCV alone (`derive_smc_arrays`, lines 335-510) or feature-frame columns when present (`build_smc_arrays`, lines 512-659). Both paths produce the same 11-tuple consumed by the evaluator.

### F-038 (HIGH) — SMC derivation lookback windows are timeframe-agnostic
- **Location**: `smc_indicators.rs:365-367` (declaration), used throughout 369-495
- **What**: three hardcoded lookback windows are baked in:
  - `lookback = 12` — used for trend (close[i] vs close[i-12]) and BoS (close vs 12-bar high/low)
  - `eq_lookback = 20` — equal-highs/lows tolerance over 20 bars
  - `displacement_lookback = 20` — body vs average body over 20 bars
- **Why it matters**: on M1 data 12 bars = 12 minutes, on H4 = 48 hours, on D1 = 12 days. The "trend" and "BoS" definitions become wildly different things across timeframes. A "12-bar trend" on M1 is intraday noise; on D1 it's a multi-week move. A strategy whose SMC signals depend on these gives different trade counts on the same symbol at different TFs purely because of the lookback semantics, not because of the strategy logic.
- **Fix**:
  1. Express lookbacks as time-units (minutes/hours) instead of bars, then convert to bar-count at runtime using `timeframe_label` minutes/bar.
  2. Or expose as `SmcDerivationConfig { trend_lookback_bars, eq_lookback_bars, displacement_lookback_bars }` so per-TF tuning is possible.
- **Severity**: HIGH (SMC signal semantics drift across timeframes)

### F-039 (MEDIUM) — `derive_smc_arrays` heuristics diverge from textbook SMC, silent fallback when feature columns absent
- **Location**: `smc_indicators.rs:335-510` — `derive_smc_arrays`
- **What**: when the feature frame doesn't carry pre-computed SMC columns (`smc_ob`, `smc_fvg`, etc.), the engine falls back to these simplified heuristics:
  - "Order Block" (393-406): bull/bear ENGULFING pattern. Textbook OB requires a structure break (BoS) first — this implementation doesn't.
  - "Fair Value Gap" (416-422): bar-pair gap between bar[i-2] and bar[i] — closer to textbook but ignores sweep/context.
  - "Liquidity sweep" (424-436): 3-bar low/high penetrate-and-close-back — textbook requires session/swing context.
  - "Inducement" (408-413): just a high-wick or low-wick > 2× body — textbook needs broader structural context.
- **Why it matters**: a user who doesn't ship SMC columns through the feature pipeline gets "SMC" signals that are simplified caricatures. The system is named "SMC-based" but degrades silently to toy patterns. No warning, no log, no documentation that this fallback is active.
- **Fix**:
  1. At `build_smc_arrays` entry, log `tracing::warn!` listing WHICH SMC columns were missing → derived. Operator can then see "5 of 11 SMC columns derived heuristically".
  2. Add a strict mode env (`FOREX_BOT_REQUIRE_REAL_SMC_COLUMNS=1`) that BAILS when any SMC column is missing — for production users who need real SMC, not toy heuristics.
- **Severity**: MEDIUM (silent fallback to simplified logic)

### F-040 (MEDIUM) — `apply_dir_fill_zeros` pattern conflates separate signals
- **Location**: `smc_indicators.rs:624-629`
- **What**:
  ```
  apply_dir_fill_zeros(&mut ob, cols.bos);     // ob inherits bos's non-zero values
  apply_dir_fill_zeros(&mut ob, cols.choch);   // ob inherits choch's non-zero values too
  apply_dir_fill_zeros(&mut trend, cols.bos);
  apply_dir_fill_zeros(&mut trend, cols.choch);
  apply_dir_fill_zeros(&mut trend, cols.displacement);
  ```
- **Why it matters**: if BoS column has data but OB column doesn't, OB's zeros get filled with BoS's values. The gene's `use_ob` flag then activates against what is really BoS data. The signal carries the wrong semantic label. Likewise trend ends up being a mix of trend/bos/choch/displacement. The gating logic in the evaluator can't distinguish "the user said OB" from "we filled OB with BoS".
- **Fix**: drop the fill-zeros pattern — let signals stay zero when their column is missing. If a gene has `use_ob=true` and no OB data, the gate evaluates against zero (which fails the gate naturally) — better than silently substituting another signal.
- **Severity**: MEDIUM (semantic label drift in gated signals)

### F-041 (LOW) — Magic threshold constants in heuristic SMC derivation
- **Location**: `smc_indicators.rs:411, 463, 485`
- **What**:
  - Line 411: `(upper / body) > 2.0 || (lower / body) > 2.0` — magic 2.0 wick-to-body ratio for inducement
  - Line 463: `tol = (avg_range * 0.1).max(1e-9)` — magic 10% of avg range as equal-highs/lows tolerance
  - Line 485: `body >= (1.8 * avg_body)` — magic 1.8× threshold for displacement
- **Fix**: promote to named constants with doc OR put on the `SmcDerivationConfig` proposed in F-038.
- **Severity**: LOW

---

## quality.rs — `crates/neoethos-search/src/quality.rs` (786 lines, **COMPLETE**)

This module computes per-strategy quality metrics (Sharpe, Sortino, Calmar, etc.) plus a composite `quality_score` and a binary `has_edge` flag used downstream by `StrategyRanker` to pick survivors.

### F-042 (HIGH) — `score_strategy` is a heavily-tuned magic-number scoring function with no config knob
- **Location**: `quality.rs:570-611`
- **What**: the function combines 8 sub-scores into a `quality_score` ∈ [0, 100]. Each sub-score has its own opaque magic constants:
  ```rust
  let sortino_score = 30.0 * (1.0 - (-metrics.sortino_ratio.max(0.0) * 0.6).exp());
  let pf_score      = 20.0 * (1.0 - (-(metrics.profit_factor.max(0.0) - 1.0).max(0.0) * 1.5).exp());
  let wr_score      = 15.0 * ((metrics.win_rate - 0.45) / 0.25).clamp(0.0, 1.0);
  let calmar_score  = 20.0 * (1.0 - (-metrics.calmar_ratio.max(0.0) * 0.8).exp());
  let dd_score      = 15.0 * (1.0 - (metrics.max_drawdown_pct / 0.15).clamp(0.0, 1.0)).max(0.0);
  let pval_score    = 10.0 * (1.0 - pval).powi(3);
  let mwr_score     = 10.0 * metrics.monthly_win_rate.clamp(0.0, 1.0);
  let mr_score      = if avg_monthly >= min_monthly { 10.0 * ratio.min(1.0) } else { 0.0 };
  ```
  Constants: 30, 0.6, 20, 1.5, 15, 0.45, 0.25, 20, 0.8, 15, 0.15, 10 (cubic), 10, 10, min_monthly_return_pct. Pre-saturation weights sum to 30+20+15+20+15+10+10+10 = 130 capped to 100 — so the cap is binding only on great strategies, and "weight" semantics are inconsistent.
- **Why it matters**: this scoring decides which strategies survive the quality screen. A strategy with Sortino=2.0 + PF=1.4 + WR=0.55 + DD=0.10 might score 65 ("ACCEPTABLE"); change Sortino curve constant from 0.6 to 0.4 and the same strategy scores 60 ("POOR"). The operator has zero visibility into these tunables.
- **Fix**:
  1. Move sub-score weights + saturation constants into a new `QualityScoreConfig` struct (or into `QualityRuntimeOverrides`).
  2. Defaults preserve current behaviour.
  3. Document the saturation math (`1 - exp(-k*x)` saturates ~1 at `x = 5/k`, so `k=0.6` means Sortino saturates around 8.3).
- **Severity**: HIGH (silent strategy ranking determined by unaudited constants)

### F-043 (MEDIUM) — MC ruin threshold + iteration count baked into source
- **Location**: `quality.rs:284-287, 336`
- **What**:
  - Line 284: `mc_iterations = 1000` — magic; can't be tuned without recompile
  - Line 287: `ruin_threshold = initial_balance * 0.50` — magic 50% loss-of-capital = "ruin"
  - Line 336: `p95_idx = (mc_iterations as f64 * 0.95)` — magic 95th percentile worst-DD reporting
- **Why it matters**:
  - 50% ruin is harsher than typical (most propfirms call 10% drawdown a fail) but more generous than some risk frameworks (1% ruin probability). The semantics of `mc_risk_of_ruin_pct` therefore depend on this magic threshold.
  - 1000 MC iterations is the speed/accuracy tradeoff. Operators can't ask for 10000 to get tighter confidence intervals.
- **Fix**: extract to `QualityRuntimeOverrides` (mc_iterations, ruin_threshold_pct, p_worst_dd_percentile).
- **Severity**: MEDIUM

### F-044 (LOW) — Recommendation tier thresholds (80/70/60) undocumented
- **Location**: `quality.rs:636-645`
- **What**: `EXCELLENT >= 80`, `GOOD >= 70`, `ACCEPTABLE >= 60`, `POOR < 60`. Magic boundaries.
- **Fix**: name them as constants OR put on `QualityScoreConfig`. Document that 80 = "deploy live", 70 = "trade demo first", 60 = "needs tuning".
- **Severity**: LOW

### F-045 (LOW) — Quarter-Kelly magic multiplier
- **Location**: `quality.rs:546`
- **What**: `kelly = kelly.clamp(0.0, 1.0); kelly * 0.25` — Kelly fraction multiplied by 0.25 (quarter-Kelly). Industry-standard for "conservative Kelly" but no doc comment explaining the choice.
- **Fix**: add doc-comment `// Quarter-Kelly for conservative position sizing (Thorp 1962)` and consider exposing the fraction.
- **Severity**: LOW

### F-046 (LOW) — `profit_factor` hard-capped at 100
- **Location**: `quality.rs:218-221`
- **What**: `if profit_factor > 100.0 { profit_factor = 100.0; }` — caps the ratio gross_profit / gross_loss. Reasonable to cap (prevents one zero-loss winning trade from inflating to infinity) but the value 100 is magic.
- **Fix**: promote to constant `PROFIT_FACTOR_CAP = 100.0` with doc.
- **Severity**: LOW

### F-047 (NOTE) — Sortino threshold correctly differentiated from Sharpe (counter-example to F-014)
- **Location**: `quality.rs:144-145`
- **What**: `min_sharpe: 1.2, min_sortino: 1.2` — wait, these are the SAME default. F-014 noted discovery.rs's `quality_analyzer_for_config` reuses min_sharpe for min_sortino — but the DEFAULT struct here also has them equal. So the F-014 bug ALSO affects the standalone `StrategyQualityAnalyzer::default()` — Sortino floor of 1.2 is too weak relative to Sharpe 1.2 in good strategies.
- **Update F-014**: the bug is wider than discovery.rs; it's also in this default. Same fix applies (raise min_sortino default, OR document why they should be equal).
- **Severity**: REFERENCE — promotes F-014's scope.

---

---

# Sessions (updated)
- **2026-05-24 session 1**: scaffolded ledger; **eval.rs COMPLETE (1211/1211)** F-001..F-006; **discovery.rs COMPLETE (2900/2900)** F-007..F-019; **validation.rs COMPLETE (1855/1855)** F-020..F-024; **gauntlet.rs COMPLETE (154/154)** F-025..F-026; **parity.rs COMPLETE (315/315)** F-027; **strategy_gene.rs COMPLETE (649/649)** F-028..F-031; **search_engine.rs COMPLETE (1060/1060)** F-032..F-037; **smc_indicators.rs COMPLETE (659/659)** F-038..F-041; **quality.rs COMPLETE (786/786)** F-042..F-047. Total findings: 47. **F-002 EURUSD-leak confirmed at 4 production sites**. F-014 scope promoted via F-047 (Sortino floor also weak in `StrategyQualityAnalyzer::default()`).

## Audit progress
| Crate | File | Lines | Status |
|---|---|---|---|
| neoethos-search | eval.rs | 1211 | COMPLETE |
| neoethos-search | discovery.rs | 2900 | COMPLETE |
| neoethos-search | validation.rs | 1855 | COMPLETE |
| neoethos-search | gauntlet.rs | 154 | COMPLETE |
| neoethos-search | parity.rs | 315 | COMPLETE |
| neoethos-search | genetic/strategy_gene.rs | 649 | COMPLETE (F-003 template) |
| neoethos-search | genetic/search_engine.rs | 1060 | COMPLETE |
| neoethos-search | genetic/smc_indicators.rs | 659 | COMPLETE |
| neoethos-search | quality.rs | 786 | COMPLETE |
| neoethos-search | genetic/evolution_math.rs | 946 | next |
| neoethos-search | genetic/runtime_overrides.rs | 795 | pending |
| neoethos-search | genetic/regime_labels.rs | 523 | pending |
| neoethos-search | genetic/diversity.rs | 219 | pending |
| neoethos-search | genetic/mod.rs | 45 | pending |
| neoethos-search | portfolio.rs | 345 | pending |
| neoethos-search | lib.rs | 1017 | pending |
| neoethos-search | stop_target.rs | 958 | pending |
| neoethos-search | cubecl_eval.rs | 1078 | pending |
| neoethos-search | discovery_gpu.rs | 1028 | pending |
| neoethos-search | hpc_gpu_discovery.rs | 894 | pending |
| neoethos-search | hpc.rs | 324 | pending |
| neoethos-search | checkpoint.rs | 494 | pending |
| neoethos-search | cubecl_ga.rs | 324 | pending |
| neoethos-search | challenge.rs | 160 | pending |
| neoethos-search | orchestration.rs | 222 | pending |
| neoethos-search | funnel_profile.rs | 236 | pending |
| neoethos-search | strategy_db.rs | 238 | pending |
| neoethos-search | export_state.rs | 115 | pending |
| neoethos-search | scheduler_assignment.rs | 18 | pending |
| neoethos-search | artifact_io.rs | 4 | pending |
| neoethos-search | discovery_tests.rs | 1238 | pending (test) |
| neoethos-data | core/hpc_ta.rs | ? | partial #212 |
| neoethos-data | core/timestamps.rs | ? | pending |
| neoethos-data | core/*.rs | ? | pending |
| ... | further crates | ... | pending |

**neoethos-search progress: 10 of 31 files COMPLETE (≈ 9610 of 20810 lines = 46%)**
