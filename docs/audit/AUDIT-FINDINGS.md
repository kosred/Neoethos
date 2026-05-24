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

## discovery.rs — `crates/neoethos-search/src/discovery.rs` (2900 lines, **PARTIAL** lines 1-500)

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

---

# Sessions (updated)
- **2026-05-24 session 1**: scaffolded ledger; audited **eval.rs COMPLETE (1211/1211)**; surfaced F-001 to F-006. Begun **discovery.rs (500/2900)**; surfaced F-007 to F-011.

## Next session targets
- `crates/neoethos-search/src/discovery.rs` lines 500-2900 (remaining ~2400 lines, probably 1-2 sessions)
- `crates/neoethos-search/src/validation.rs` (1855 lines)
- `crates/neoethos-search/src/genetic/search_engine.rs`
- `crates/neoethos-search/src/genetic/strategy_gene.rs`
- … then continue priority list

