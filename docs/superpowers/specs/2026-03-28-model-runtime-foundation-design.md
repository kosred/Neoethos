# Model Runtime Foundation Design

Date: 2026-03-28

## Goal

Create a truthful, unified runtime foundation for every model family already declared in the repo so the system can move from placeholder inventory to full implementation without more fake support, fake training paths, or inconsistent prediction/artifact behavior.

This subproject does **not** finish the tree, deep, calibration, exit, adaptive, anomaly, or RL families by itself. It defines the runtime shape that all of those families must use.

## Why This Exists

The current codebase already shows the intended architecture:

- tree models for fast tabular scoring
- deep models for sequence and nonlinear alpha
- calibration and conformal gating for trade/no-trade discipline
- meta-blending for combining experts
- exit and adaptive models for lifecycle and drift response
- anomaly and RL families for later specialist behavior

The problem is that these intentions are mixed together with:

- placeholder trainers
- placeholder predictors
- config-only aspirations
- Python bridge remnants
- inconsistent save/load contracts

As a result, the repo currently overstates what is runnable and understates what is actually intended.

## Design Principles

### 1. Capability truthfulness

The runtime must distinguish between:

- `planned`
- `implemented`
- `verified`

No family disappears from the roadmap, but the runtime must stop pretending that a planned family is already production-ready.

### 2. One runtime shape for all families

Every model family must eventually plug into the same core runtime concepts:

- capability metadata
- dispatch metadata
- training result metadata
- artifact metadata
- prediction output contract

### 3. Family-first dispatch

The runtime should dispatch by family first, then by concrete model name.

Families:

- `tree`
- `deep`
- `meta`
- `exit`
- `adaptive`
- `anomaly`
- `rl`

This prevents the orchestrator from degrading into stringly-typed model-specific branching.

### 4. No more generic shell -> bail later

The current orchestrator flow accepts configured models, builds a plan, and then can still fail with generic runtime `bail!` placeholders. The foundation must move those decisions upfront into deterministic capability and dispatch mapping.

## Scope

### In scope

- truthful capability map for all model names already present in config and registry surfaces
- deterministic mapping from model name to family
- shared training dispatch contracts
- shared artifact metadata schema
- shared prediction output schema
- early settings validation for configured model names and families
- removing generic placeholder dispatch behavior from the orchestration layer

### Out of scope

- implementing full tree model training/inference
- implementing full deep model orchestration
- rewriting calibration algorithms
- implementing anomaly, adaptive, exit, or RL families

Those belong to the next tranches after the foundation.

## Current Intent Read From The Repo

### Tree models

Files:

- `crates/forex-models/src/tree_models/xgboost.rs`
- `crates/forex-models/src/tree_models/lightgbm.rs`
- `crates/forex-models/src/tree_models/catboost.rs`

Intended role:

- fast tabular signal scoring
- low-latency baseline alpha
- range/routing-friendly models for engineered features

Current maturity:

- mostly placeholders

### Deep models

File:

- `crates/forex-models/src/burn_models.rs`

Intended role:

- core nonlinear and sequence alpha stack
- MLP, N-BEATS, TiDE, TabNet, KAN, Transformer
- likely trend/regime-sensitive core prediction families

Current maturity:

- real implementation exists in isolation
- not cleanly integrated into the active training runtime

### Calibration, conformal, and meta-blending

Files:

- `crates/forex-models/src/ensemble.rs`
- `crates/forex-bindings/src/calibration.rs`
- `crates/forex-bindings/src/conformal.rs`

Intended role:

- improve probability quality
- abstain when confidence is weak
- combine multiple experts into a tradeability layer

Current maturity:

- partially implemented, but not yet turned into a coherent runtime layer

### Genetic discovery

Files:

- `crates/forex-models/src/genetic.rs`
- `crates/forex-search/src/*`

Intended role:

- discover strategies and signal logic, not just supervised direction predictions

Current maturity:

- `forex-search` is real
- `forex-models/genetic.rs` is still a Python bridge

### Exit and adaptive behavior

Files:

- `crates/forex-models/src/exit_agent.rs`
- `crates/forex-models/src/streaming/adaptive_impl.rs`

Intended role:

- improve exits
- support online adaptation and drift response

Current maturity:

- exit agent is more serious than the generic adaptive path
- adaptive path is still early-stage

### Anomaly and RL families

Files:

- `crates/forex-models/src/anomaly/forest_impl.rs`
- `crates/forex-models/src/rl/dqn_impl.rs`
- `crates/forex-models/src/forecasting/swarm_impl.rs`

Intended role:

- anomaly gating
- policy learning
- experimental forecasting/exploration

Current maturity:

- mostly placeholders or early-stage experiments

## Proposed Runtime Structure

### Runtime module

Create a new runtime namespace:

- `crates/forex-models/src/runtime/mod.rs`
- `crates/forex-models/src/runtime/capabilities.rs`
- `crates/forex-models/src/runtime/artifacts.rs`
- `crates/forex-models/src/runtime/prediction.rs`
- `crates/forex-models/src/runtime/dispatch.rs`

### Capabilities

This layer defines:

- model name
- model family
- current implementation state:
  - `planned`
  - `implemented`
  - `verified`
- artifact type
- trainer type
- inference type

### Artifacts

Shared metadata schema for all families:

- model name
- family
- version
- feature columns
- label mapping
- training dataset summary
- training timestamp
- optional calibration artifact reference
- optional validation summary

### Prediction output

Shared runtime output shape:

- class probabilities
- optional confidence score
- optional abstain recommendation
- model metadata

The runtime should stop allowing ad-hoc family-specific prediction semantics in the active orchestration path.

### Dispatch

Training and inference dispatch should move to a deterministic family-aware plan:

- configured model names are resolved into a dispatch plan
- every planned model is assigned a known family and trainer path
- orchestration code no longer performs a late generic fallback failure

## File Ownership

### Existing files to modify first

- `crates/forex-models/src/registry.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/parallel_trainer.rs`
- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/lib.rs`

### New files

- `crates/forex-models/src/runtime/mod.rs`
- `crates/forex-models/src/runtime/capabilities.rs`
- `crates/forex-models/src/runtime/artifacts.rs`
- `crates/forex-models/src/runtime/prediction.rs`
- `crates/forex-models/src/runtime/dispatch.rs`

## Validation And Testing

### Tests required

- registry-to-family mapping tests
- model-name capability snapshot tests
- dispatch plan tests
- artifact metadata round-trip tests
- prediction contract shape tests
- config validation tests against current configured model names

### Verification required

- `cargo test -p forex-models -- --nocapture`
- `cargo clippy -p forex-models --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`

## Acceptance Criteria

- the runtime knows the family and capability state of every configured model name
- the orchestrator builds a deterministic dispatch plan instead of failing late with generic placeholders
- there is one shared artifact metadata contract
- there is one shared prediction output contract
- the foundation is ready for the next sequence of tranches:
  - tree
  - deep
  - calibration/meta
  - exit/adaptive
  - anomaly/RL

## Planned Sequence After This Foundation

1. Tree Models Productionization
2. Deep Models Productionization
3. Calibration / Meta Layer
4. Exit / Adaptive
5. Anomaly / RL

That order stays fixed unless later evidence from the codebase makes a hard dependency force a change.
