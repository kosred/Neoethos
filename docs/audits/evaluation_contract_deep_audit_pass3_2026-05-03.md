# Evaluation Contract Deep Audit — Pass 3

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: `fast_evaluate_strategy_core`, `simulate_trades_core`, walk-forward validation, metric semantics, evaluator parity, and prop/risk diagnostics.

## Summary

The evaluation layer is the heart of the strategy search system. If search scoring, trade simulation, validation, and quality reports do not share one execution contract, the search can select strategies that do not survive later validation or live/runtime interpretation.

This pass confirms that the project currently has multiple evaluator semantics:

- fast population evaluator
- trade simulator
- walk-forward diagnostics
- GPU/CUDA evaluator path
- label-search scoring path

These must converge into one canonical evaluation contract.

## Findings

### 1. `fast_evaluate_strategy_core` and `simulate_trades_core` are not equivalent

`fast_evaluate_strategy_core` is used for fast scoring and returns metric arrays.

`simulate_trades_core` produces actual trade logs.

They share some behavior, such as causal prior-bar signal entry, SL/TP, max hold, spread, commission, and max trades per day. But they are not identical.

For example, `simulate_trades_core` contains kill-zone/weekend entry blocking and weekend force-exit logic when `kill_zones_enabled` is true. The fast evaluator does not appear to implement the same full session/kill-zone logic.

**Risk:** a gene can rank well during fast evaluation but produce different trades and risk behavior when simulated for logs/validation.

**Severity:** Critical.

**Fix direction:** define one canonical execution state machine and have both metrics and trade logs derive from the same simulation pass.

---

### 2. Walk-forward risk diagnostics pass day keys as timestamps

`walkforward_risk_diagnostics` receives `days: &[i64]` and then calls:

```rust
simulate_trades_core(close, high, low, days, signals, settings)
```

But `simulate_trades_core` expects timestamp values in milliseconds. It computes day keys with `ts / 86_400_000` and duration with `/ 3_600_000.0`.

Passing day IDs instead of millisecond timestamps breaks day/session/duration logic.

**Risk:** walk-forward risk diagnostics can report incorrect daily trade counts, durations, session behavior, daily PnL grouping, and prop compliance.

**Severity:** Critical.

**Fix direction:** `WalkforwardBacktestInput` must include real `timestamps`, not only `months` and `days`. Risk diagnostics must receive timestamps, while daily grouping can separately receive day keys.

---

### 3. Initial equity is still env-driven in `BacktestSettings`

`BacktestSettings::initial_equity()` reads `FOREX_BOT_BACKTEST_INITIAL_EQUITY`.

**Risk:** two identical configs can produce different drawdown, return, and risk metrics depending on environment.

**Severity:** High.

**Fix direction:** make `initial_equity` a field in `BacktestSettings` or a higher-level `ExecutionContract`. Remove env access from evaluation.

---

### 4. Monthly bucket capacity is env-driven

`BacktestSettings::month_capacity()` reads `FOREX_BOT_BACKTEST_MAX_MONTH_BUCKETS`.

**Risk:** consistency/sharpe calculation can silently change with environment.

**Severity:** Medium-High.

**Fix direction:** make month bucket policy explicit and exported in evaluation contract.

---

### 5. Metrics array lacks typed semantics

Evaluation returns `[f64; 11]` with implicit positions:

- 0 net_profit
- 1 sharpe
- 2 peak_equity
- 3 max_drawdown
- 4 win_rate
- 5 profit_factor
- 6 expectancy
- 7 unused
- 8 trade_count
- 9 consistency
- 10 max_daily_dd

This layout is duplicated and manually interpreted by search, validation, and discovery.

**Risk:** index mistakes are easy and dangerous. A slot can change meaning without compiler protection.

**Severity:** High.

**Fix direction:** replace `[f64; 11]` with a typed `BacktestMetrics` struct.

---

### 6. Open-position finalization is not explicit enough in evaluator contract

If a position remains open at the end of the data, the current fast evaluator/trade simulator behavior must be explicitly defined and tested.

**Risk:** different paths can disagree on whether final open positions are closed, ignored, or marked open.

**Severity:** Medium-High.

**Fix direction:** add `final_position_policy`:

- `IgnoreOpen`
- `CloseAtLastClose`
- `MarkToMarketOnly`
- `RejectRunIfOpen`

Export it in the evaluation contract.

---

### 7. Intrabar SL/TP ordering is under-specified

When both SL and TP occur in the same candle, the current logic checks SL before TP. This is conservative for long positions, but the policy is implicit.

**Risk:** metrics depend on implicit intrabar ordering assumptions.

**Severity:** Medium.

**Fix direction:** add `intrabar_fill_policy`:

- `StopFirstConservative`
- `TargetFirstOptimistic`
- `OpenHighLowCloseHeuristic`
- `RejectAmbiguousBars`

---

### 8. Walk-forward still validates fixed signals, not retrain-per-split discovery

The walk-forward path evaluates provided signals over split windows. It does not appear to rerun the full strategy discovery inside each training split and then test the discovered strategy on the out-of-sample split.

**Risk:** this is useful as signal robustness validation, but it is not full walk-forward optimization.

**Severity:** Medium-High.

**Fix direction:** distinguish:

- `SignalWalkForwardValidation`
- `RetrainPerSplitWalkForwardOptimization`

Both are valid, but they answer different questions.

---

### 9. Prop/risk diagnostics use a separate path from search scoring

Risk diagnostics compute max consecutive losses, daily loss breach, profit consistency, trade-limit violation, and min trading days using simulated trades.

Search scoring uses fast metrics. These are not guaranteed to share the same execution details.

**Risk:** the search can optimize for one contract while prop compliance is checked under another.

**Severity:** High.

**Fix direction:** prop/risk gates must be derived from the same canonical trade simulation used for search acceptance.

---

## Recommended refactor

### Step 1: Typed metrics

```rust
pub struct BacktestMetrics {
    pub net_profit: f64,
    pub sharpe: f64,
    pub peak_equity: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trade_count: usize,
    pub consistency: f64,
    pub max_daily_drawdown: f64,
}
```

### Step 2: One execution contract

```rust
pub struct ExecutionContract {
    pub initial_equity: f64,
    pub pip_size: f64,
    pub pip_value_per_lot: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub max_hold_bars: usize,
    pub min_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub gap_threshold_ms: i64,
    pub kill_zones_enabled: bool,
    pub final_position_policy: FinalPositionPolicy,
    pub intrabar_fill_policy: IntrabarFillPolicy,
}
```

### Step 3: One canonical simulator

Create one simulator that returns:

```rust
pub struct SimulationResult {
    pub metrics: BacktestMetrics,
    pub trades: Vec<Trade>,
    pub daily_ledger: Vec<DailyLedgerRow>,
    pub monthly_ledger: Vec<MonthlyLedgerRow>,
    pub diagnostics: RiskDiagnostics,
}
```

Fast population search can use a stripped/optimized version only if parity tests prove equivalence.

### Step 4: Real timestamp contract

Validation and diagnostics must carry both:

- real timestamps in milliseconds or nanoseconds with explicit unit
- precomputed day/month keys

Do not pass day keys where timestamps are expected.

## Required tests

1. `fast_eval_matches_simulated_trade_metrics_basic`
2. `fast_eval_matches_simulator_with_max_trades_per_day`
3. `fast_eval_matches_simulator_with_kill_zones`
4. `walkforward_uses_real_timestamps_for_risk_diagnostics`
5. `initial_equity_is_config_not_env`
6. `metrics_struct_replaces_array_indices`
7. `intrabar_stop_target_policy_is_explicit`
8. `final_open_position_policy_is_explicit`

## Bottom line

The evaluator must become a single canonical execution contract. Right now the search engine, trade simulator, validation diagnostics, and GPU paths can disagree. That is dangerous because the entire evolutionary search learns from whatever the evaluator rewards.
