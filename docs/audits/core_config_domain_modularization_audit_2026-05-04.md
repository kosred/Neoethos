# Core Config / Domain Modularization Audit

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: config/settings, hardware planning, risk domain, portfolio domain, and modularization direction.

Files inspected:

- `crates/forex-core/src/lib.rs`
- `crates/forex-core/src/config.rs`
- `crates/forex-core/src/system.rs`
- `crates/forex-core/src/domain/mod.rs`
- `crates/forex-core/src/domain/risk.rs`
- `crates/forex-core/src/domain/portfolio.rs`

No production code was changed by this audit.

## Core conclusion

The project already has strong foundations for typed configuration, hardware planning, risk domain logic, and portfolio math.

The problem is not that these concepts are missing. The problem is that they are too concentrated in large files and not yet used as the single source of truth across search/model/evaluator/GPU modules.

The direction should be:

```text
large config/system files -> small focused modules
search-specific duplicate logic -> core domain services
scattered env reads -> resolved typed config
hardcoded hardware paths -> generic runtime scheduler
```

## `forex-core/src/lib.rs` findings

`lib.rs` exports:

```rust
pub mod config;
pub mod domain;
pub mod system;
pub use config::Settings;
pub use system::{AcceleratorBackend, AcceleratorDevice, HardwareExecutionPlan, TrainingPrecision, WorkloadExecutionPlan, WorkloadKind};
```

This confirms that `Settings` and hardware planning are already intended as shared core concepts.

The refactor should strengthen this: downstream crates should use core config/runtime/domain types instead of reading env vars or redefining their own partial policies.

## `config.rs` findings

`config.rs` is very large and contains multiple domains in one file.

Visible major sections include:

- `SystemConfig`
- `RiskConfig`
- `ModelsConfig`
- later config sections not fully inspected because the connector truncated the file

Positive finding: many settings that were previously seen as env-driven already exist as typed config fields.

Examples:

- GPU preference
- number of GPUs
- prop search population/generations
- prop search parent/survivor selection
- prop search validation thresholds
- walk-forward and CPCV settings
- risk settings
- challenge settings
- stop target / volatility settings
- spread and commission settings
- model training/inference batch sizes

This means the project does not need to invent config from scratch.

### Main problem

The config exists, but downstream code often bypasses it and reads env vars directly.

Examples found in previous audits:

- search policy env vars in `genetic/search_engine.rs`
- SMC env vars in `smc_indicators.rs`
- quality env vars in `quality.rs`
- backtest env vars in `eval.rs`
- device/precision env vars in GPU files
- runtime precision env vars in `system.rs`

The target should be one resolved config object:

```rust
ResolvedAppConfig
ResolvedRuntimeConfig
ResolvedSearchConfig
ResolvedRiskConfig
ResolvedValidationConfig
```

These should be generated from layered inputs:

```text
built-in defaults
< installer-generated config
< hardware profile / benchmark
< UI settings
< optional CLI override
```

Env vars should not be normal production semantics.

## Proposed `config.rs` split

`config.rs` should be split into smaller files:

```text
config/
  mod.rs
  system.rs
  risk.rs
  models.rs
  search.rs
  validation.rs
  runtime.rs
  features.rs
  artifacts.rs
  news.rs
  resolve.rs
```

Each config file should own one domain.

`resolve.rs` should produce immutable resolved config structs used during a run.

## `system.rs` findings

`system.rs` already contains useful runtime/hardware planning types:

- `HardwareProfile`
- `HardwareProbe`
- `AcceleratorBackend`
- `TrainingPrecision`
- `AcceleratorDevice`
- `WorkloadKind`
- `WorkloadExecutionPlan`
- `HardwareExecutionPlan`
- `AutoTuner`

This is a strong foundation for replacing `hpc.rs` and all scattered GPU env logic.

### Main problem

`system.rs` is too large and owns too many responsibilities:

- hardware profile type
- hardware probing
- CUDA/ROCm/WGPU detection
- backend selection
- precision selection
- workload planning
- autotuning
- env precision override
- thread env defaults

This should be split into small runtime modules.

## Proposed `system.rs` split

```text
runtime/
  mod.rs
  hardware_profile.rs
  hardware_probe.rs
  backend.rs
  precision.rs
  workload_kind.rs
  workload_plan.rs
  execution_plan.rs
  auto_tuner.rs
  device_assignment.rs
  scheduler.rs
  config_resolution.rs
```

The current `system.rs` can become a compatibility re-export module until callers migrate.

## Runtime source-of-truth requirement

Every CUDA/GPU/model/search/evaluator module should receive runtime decisions from the planner.

Target pattern:

```rust
let plan = ResolvedRuntimeConfig::from(settings, hardware_profile, ui_settings);
let assignment = scheduler.assign(work_unit, &plan);
kernel.execute(input, assignment);
```

Avoid this pattern:

```rust
let device = std::env::var("FOREX_BOT_...");
```

inside kernels or search/model modules.

## `domain/mod.rs` findings

`domain/mod.rs` already defines a useful domain structure:

```rust
consistency
errors
events
meta_controller
drift_monitor
news_filter
order_execution
portfolio
risk
```

This is good. The repo already has the start of a domain-driven structure.

The refactor should move duplicated business logic from search/model/runtime into domain modules when appropriate.

## `domain/risk.rs` findings

`domain/risk.rs` is a mature risk domain module.

It defines:

- `ChallengePhase`
- `PropFirmRules`
- `ChallengeRiskPreset`
- `resolve_challenge_risk_preset`
- `TradeRecord`
- `TradeGateInput`
- `PositionSizingInput`
- `RevengeTradeDetector`
- `RiskManager`

It includes logic for:

- prop firm rules
- challenge phase presets
- daily/total drawdown state
- circuit breaker
- recovery mode
- trade session checks
- night session blocking
- news kill window
- revenge-trading detection
- confidence gate
- position sizing
- risk reduction under drawdown
- zero risk at total drawdown limit

This is stronger and more complete than `crates/forex-search/src/challenge.rs`.

### Important duplication

`crates/forex-search/src/challenge.rs` and `forex-core/src/domain/risk.rs` overlap conceptually.

`search/challenge.rs` has a small `ChallengeOptimizer` for risk allocation.

`domain/risk.rs` has a broader, more production-like risk manager and challenge rules.

The search-specific challenge logic should likely be retired or replaced by core-domain risk/validation logic.

Suggested direction:

```text
search/challenge.rs -> remove after migration
core/domain/risk.rs -> owner of challenge/risk rules
validation/challenge_policy.rs -> wrapper/reporting for backtest/validation use
```

### Positive contrast with previous finding

Earlier, `search/challenge.rs` had a forced minimum risk clamp. In `domain/risk.rs`, `calculate_position_size` can return zero when total drawdown limit is hit.

That is closer to the desired fail-closed behavior.

## `domain/portfolio.rs` findings

`domain/portfolio.rs` is a good example of a small, focused module.

It owns:

- correlation matrix calculation
- portfolio weight optimization
- correlation penalty
- exposure budget

It is compact and domain-specific.

This is the file style the project should move toward.

### Improvement needed

`PortfolioManager` should eventually consume `ValidatedStrategyGene` or `PortfolioCandidate` artifacts with provenance, not only raw names/returns.

It should also record portfolio optimization policy and output a `PortfolioConstructionReport`.

Suggested future types:

```rust
PortfolioConstructionPolicy
PortfolioCandidate
PortfolioConstructionReport
PortfolioAllocation
```

## Cross-layer issue: config vs domain vs search duplication

The same business concepts appear in multiple places:

- challenge target rules in `search/challenge.rs`
- challenge/risk rules in `domain/risk.rs`
- risk settings in `config.rs`
- validation/quality rules in `quality.rs`
- backtest risk assumptions in `eval.rs`

These should be unified.

Suggested ownership:

```text
config/risk.rs: serializable settings
core/domain/risk.rs: live/domain risk behavior
validation/challenge_policy.rs: validation report/checks using domain rules
search: calls validation/domain; does not own challenge rules
```

## Recommended migration plan

### Stage 1: Split large files without changing behavior

Split:

- `config.rs`
- `system.rs`

into smaller modules with re-exports.

### Stage 2: Introduce resolved config structs

Add:

```rust
ResolvedAppConfig
ResolvedRuntimeConfig
ResolvedSearchConfig
ResolvedRiskConfig
ResolvedValidationConfig
```

### Stage 3: Replace env reads gradually

Move env-driven search/quality/eval/GPU settings into typed config resolution.

### Stage 4: Retire duplicate challenge logic

Migrate `search/challenge.rs` callers to core domain risk / validation challenge policy.

### Stage 5: Connect portfolio to validated strategy artifacts

Make portfolio construction consume validated candidates with feature/evaluation provenance.

## Bottom line

The core crate already contains many of the right pieces.

The next refactor should not rewrite everything. It should:

- split large files into small focused modules,
- make resolved config the only source of production settings,
- make `system.rs` become a generic runtime scheduler foundation,
- move challenge/risk ownership to core domain risk,
- keep `domain/portfolio.rs` as a model for clean module size,
- remove search-specific duplicate business rules after migration.
