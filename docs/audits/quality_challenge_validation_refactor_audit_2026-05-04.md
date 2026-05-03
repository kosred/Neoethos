# Quality / Challenge Validation Refactor Audit

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: validation, quality scoring, Monte Carlo checks, and challenge-style risk allocation.

Files inspected:

- `crates/forex-search/src/quality.rs`
- `crates/forex-search/src/challenge.rs`

No production code was changed by this audit.

## Core conclusion

The quality and challenge logic should become part of one unified validation module.

The current code contains useful validation logic, but it should not remain as a separate second layer with its own hidden assumptions, env vars, and random behavior.

Target module:

```rust
validation/
  trade_ledger.rs
  metrics.rs
  quality_policy.rs
  monte_carlo.rs
  challenge_policy.rs
  validation_report.rs
```

or equivalent.

## `quality.rs` findings

`quality.rs` contains valuable logic and should not be deleted.

It defines:

- `Trade`
- `StrategyMetrics`
- `StrategyQualityAnalyzer`
- `StrategyRanker`

It calculates:

- total trades
- win rate
- profit factor
- Sharpe
- Sortino
- Calmar
- total return
- drawdown
- streaks
- expectancy
- Kelly fraction
- statistical significance
- monthly consistency
- trades per month
- Monte Carlo worst drawdown 95%
- risk of ruin
- quality score
- edge flag
- recommendation label

This is useful and should be preserved as validation logic.

## `quality.rs` problems

### 1. Env-driven validation semantics

The visible code reads env vars that affect quality results:

- `FOREX_BOT_PROP_MIN_TRADES_PER_MONTH`
- `FOREX_BOT_TRADING_DAYS_PER_MONTH`

These are not debug flags. They change monthly consistency and trade-frequency calculations, which can change whether a strategy passes quality screening.

They should become typed config:

```rust
pub struct QualityValidationPolicy {
    pub min_trades_per_month: usize,
    pub trading_days_per_month: f64,
    pub min_sharpe: f64,
    pub min_sortino: f64,
    pub min_calmar: f64,
    pub min_profit_factor: f64,
    pub min_win_rate: f64,
    pub min_trades: usize,
    pub max_drawdown: f64,
    pub min_monthly_return_pct: f64,
    pub edge_significance_pvalue: f64,
}
```

### 2. Unseeded Monte Carlo

`StrategyQualityAnalyzer::analyze_strategy` creates a Monte Carlo simulation with:

```rust
rand::rng()
```

This makes validation non-reproducible. The same strategy can produce slightly different Monte Carlo risk values across runs.

Monte Carlo validation should receive a seed from the resolved validation plan.

Suggested policy:

```rust
pub struct MonteCarloValidationPolicy {
    pub enabled: bool,
    pub iterations: usize,
    pub seed: u64,
    pub ruin_threshold_pct: f64,
    pub use_daily_block_bootstrap: bool,
    pub min_days_for_block_bootstrap: usize,
}
```

### 3. Metrics need provenance

A saved quality result should record:

- backtest contract that produced the trades
- quality policy used
- Monte Carlo policy used
- Monte Carlo seed
- initial balance
- trading-days-per-month assumption
- minimum trades per month
- code version
- dataset fingerprint

Without this, quality scores are not fully reproducible.

### 4. Quality score should be a report, not just a field

`StrategyMetrics` currently stores both raw metrics and final scoring fields:

- `quality_score`
- `has_edge`
- `recommendation`

This works, but a more explicit type would be safer:

```rust
pub struct StrategyQualityReport {
    pub metrics: StrategyMetrics,
    pub policy: QualityValidationPolicy,
    pub monte_carlo: Option<MonteCarloReport>,
    pub passed: bool,
    pub reasons: Vec<String>,
}
```

## `challenge.rs` findings

`challenge.rs` defines:

- `ChallengeTarget`
- `ChallengeOptimizer`
- `RiskAllocationInput`

It optimizes risk based on:

- current profit
- days left
- current drawdown
- win rate
- average risk/reward
- daily loss
- realized trades per day

This is useful, but it should live in the same validation/risk policy layer.

## `challenge.rs` positive points

The code is already typed and small.

`ChallengeTarget` is clear:

- total profit target
- daily target
- max daily drawdown
- max total drawdown
- min trading days
- max trading days

`RiskAllocationInput` is also clear and explicit.

This is better than env-driven code.

## `challenge.rs` problems

### 1. Hardcoded default challenge assumptions

Default values are hardcoded:

```text
total profit target = 10%
max daily DD = 4.5%
max total DD = 10%
min trading days = 5
max trading days = 60
```

These should be config/UI settings because prop/challenge rules differ by provider and account type.

Suggested type:

```rust
pub struct ChallengePolicy {
    pub provider: String,
    pub phase: String,
    pub total_profit_target: f64,
    pub max_daily_drawdown: f64,
    pub max_total_drawdown: f64,
    pub min_trading_days: i32,
    pub max_trading_days: i32,
    pub risk_floor: f64,
    pub risk_ceiling: f64,
    pub fail_closed_when_limits_hit: bool,
}
```

### 2. Risk floor can be dangerous when limits are hit

`optimize_risk_allocation` clamps final risk to:

```rust
optimal_risk.clamp(0.001, 0.015)
```

This means it can return at least 0.1% risk even if drawdown room is zero.

For a challenge/prop-style account, if daily or total drawdown room is exhausted, the correct behavior should probably be fail-closed or return zero risk, not force a minimum risk.

The risk floor must be policy-driven:

```rust
if daily_room <= 0.0 || total_room <= 0.0 {
    return 0.0; // or BlockTrading
}
```

### 3. Challenge logic should be separate from validation pass/fail

`ChallengeOptimizer` computes risk allocation. It does not produce a full challenge compliance report.

The validation layer should also be able to say:

- passed daily drawdown rule
- passed total drawdown rule
- passed min trading days rule
- target achieved
- trading should be blocked
- risk should be reduced

Suggested output:

```rust
pub struct ChallengeComplianceReport {
    pub target_progress: f64,
    pub daily_dd_used: f64,
    pub total_dd_used: f64,
    pub days_used: i32,
    pub days_left: i32,
    pub compliant: bool,
    pub block_new_trades: bool,
    pub recommended_risk: f64,
    pub reasons: Vec<String>,
}
```

## Unified validation module direction

The validation layer should consume trades and produce reports.

Suggested flow:

```text
Canonical backtest trades
-> TradeLedger
-> StrategyMetricsCalculator
-> QualityValidationPolicy
-> MonteCarloValidationPolicy
-> ChallengePolicy
-> StrategyValidationReport
```

The final saved candidate should not only contain a fitness number. It should contain a validation report.

## Required integration with earlier refactor findings

This audit connects to previous architecture notes:

- `unified_module_logic_architecture_2026-05-04.md`
- `search_orchestration_refactor_audit_2026-05-03.md`
- `search_gpu_discovery_scheduler_audit_2026-05-03.md`

The quality/challenge layer should be one of the canonical modules.

It should not read production semantics from env vars.

It should not use unseeded random behavior for reproducible validation.

It should record validation provenance in artifacts.

## Required tests

Add tests for:

- deterministic Monte Carlo with fixed seed
- monthly consistency with configurable min trades per month
- trading-days-per-month assumption
- quality score stability
- challenge risk when target already reached
- challenge risk when daily drawdown room is zero
- challenge risk when total drawdown room is zero
- challenge risk under high time pressure
- challenge compliance report pass/fail cases

## Bottom line

`quality.rs` and `challenge.rs` contain useful logic, but they should become a unified validation/risk module with typed policies and reproducible outputs.

The most important fix is to remove hidden env/random behavior from validation and to make challenge risk allocation fail-closed when risk limits are exhausted.
