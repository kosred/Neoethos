# Unified Module Logic Architecture Note

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: record the requirement to unify duplicated core logic into reusable modules with clear contracts.

## User idea

The user proposed that core logic should be unified everywhere. Example: feature generation should live in one feature module, and the other bot components should call that module instead of rebuilding features independently.

This idea is correct and should become a core refactor rule.

## Core principle

Move from:

```text
same concept implemented separately in search, models, GPU paths, CPU fallback, validation, and runtime
```

to:

```text
one concept -> one owning module -> many backend implementations -> one artifact/provenance contract
```

Unification does not mean one huge file. It means one owner module per concept.

## Proposed owner modules

### 1. Feature module

Owns:

- OHLCV-to-feature generation
- timestamp normalization
- multi-timeframe alignment
- causal feature availability
- feature prefiltering
- feature scaling / normalization contract
- original-to-effective feature mapping
- feature schema hash

All search, model, validation, runtime, CPU, and GPU paths should use this module.

### 2. SMC module

Owns:

- SMC feature source detection
- derived OHLCV SMC arrays
- SMC arrays from feature columns
- SMC gate policy
- SMC flag randomization policy
- SMC weights
- CPU/GPU SMC parity contract

Artifacts should record whether SMC data came from derived OHLCV, feature columns, or a mixed source.

### 3. Signal module

Owns:

- weighted feature combination
- thresholds
- SMC gating
- long/short/flat signal synthesis
- signal timing policy
- CPU/GPU signal parity tests

The signal timing rule must be explicit, especially prior-bar signal execution.

### 4. Evaluation module

Owns:

- canonical backtest semantics
- spread and commission
- SL/TP rules
- trailing stop rules
- max/min hold rules
- timestamp gap handling
- session/kill-zone rules
- daily/monthly aggregation
- metric schema
- CPU/GPU evaluator parity

Approximate presearch must not be confused with canonical validation.

### 5. Search module

Owns:

- candidate generation
- mutation/crossover
- parent/survivor selection
- archive policy
- seen-candidate memory
- checkpoint/resume for search
- candidate lifecycle stages

The search module should call feature, signal, evaluator, scheduler, and artifact modules instead of owning their logic.

### 6. Model module

Owns:

- model training plans
- model inference plans
- model artifacts
- model backend kernels

Models must consume the same feature schema and preprocessing contract as search and runtime.

### 7. Scheduler/runtime module

Owns:

- hardware profile
- resolved runtime config
- device assignments
- work units
- sharding strategies
- backend kernels
- precision policy
- fallback policy

GPU kernel files should receive a `DeviceAssignment`; they should not choose devices through env vars.

### 8. Artifact/provenance module

Owns:

- dataset fingerprint
- feature schema hash
- runtime plan hash
- config hash
- hardware profile ID
- device assignment record
- canonical vs approximate mode
- validation status

Every saved model, candidate, strategy, portfolio, checkpoint, or evaluation result should record this provenance.

## Candidate lifecycle split

The current `Gene` type should eventually be split into stages:

```rust
CandidateGene
EvaluatedGene
ValidatedStrategyGene
PortfolioStrategy
```

This prevents approximate candidates from being treated the same as canonical validated candidates.

## Refactor rule

Before adding a new helper, check which module owns that concept.

Examples:

- feature alignment belongs to the feature module
- timestamp unit conversion belongs to the timestamp/feature layer
- signal synthesis belongs to the signal module
- backtest metrics belong to the evaluation module
- GPU device selection belongs to the scheduler module
- artifact metadata belongs to the artifact/provenance module

## Migration plan

1. Define shared contracts first.
2. Keep old implementations until parity tests exist.
3. Move one caller at a time to the shared module.
4. Add CPU/GPU parity tests.
5. Add artifact provenance for each unified contract.
6. Remove duplicated helpers only after all callers use the shared module.

## Bottom line

The user's idea is correct.

The project should be refactored around shared canonical modules, especially a single feature pipeline used by all parts of the bot.

This will reduce semantic drift, improve reproducibility, make CPU/GPU/multi-GPU execution safer, and make the application easier to distribute and configure through generated config and UI settings.
