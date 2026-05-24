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

# Sessions (updated)
- **2026-05-24 session 1**: scaffolded ledger; audited **eval.rs COMPLETE (1211/1211)** — F-001 to F-006. Audited **discovery.rs COMPLETE (2900/2900)** — F-007 to F-019. Total findings so far: 19.

## Next session targets
- `crates/neoethos-search/src/validation.rs` (1855 lines)
- `crates/neoethos-search/src/genetic/search_engine.rs`
- `crates/neoethos-search/src/genetic/strategy_gene.rs` (template for F-003 fix; verify `for_symbol` shape)
- `crates/neoethos-search/src/genetic/mod.rs`, `genetic/smc_indicators.rs`
- `crates/neoethos-search/src/quality.rs`
- `crates/neoethos-search/src/portfolio.rs`
- `crates/neoethos-search/src/gauntlet.rs` (F-002 caller site)
- `crates/neoethos-search/src/parity.rs` (F-002 caller site)
- `crates/neoethos-data/src/core/hpc_ta.rs`, `timestamps.rs` and feature pipeline
- engine continues … then app_services, models, CLI/TUI
