# Truthfulness + Safety Hardening Design

## Goal

Remove misleading operator-facing states from the app, enforce a real backend auto-trade gate, and make every trading/runtime capability surface report honest availability instead of placeholders.

## Problem Statement

The current application contains multiple operator-visible states that look real but are not backed by verified runtime sources:

- synthetic equity data in the dashboard
- hardcoded AI confidence values
- hardcoded compliance/news/latency labels
- misleading top-ribbon metrics
- an `AI Auto-Trade` switch that is not yet a guaranteed backend execution gate

This is more dangerous than an obviously missing feature because it can cause the operator to trust the wrong state during live evaluation or execution.

## Scope

This subproject covers only truthfulness and execution safety for the current app shell.

Included:

- make operator-facing UI panels show only real values or explicit unavailable states
- convert `AI Auto-Trade` into a real backend execution gate for bot/AI actions
- preserve manual operator trading when adapter/risk gates allow it
- add honest capability reporting for partial, stubbed, or unavailable runtime features
- write denied-action journal/log records with exact reasons

Excluded:

- implementing missing ML models
- multi-account copy trading
- new broker adapters
- packaging/distribution work
- news intelligence implementation
- full walk-forward validation redesign

## Product Rules

### 1. Truthful UI Rule

No operator-facing panel may display synthetic or placeholder runtime values as if they are live truth.

Allowed states:

- real measured/runtime-backed value
- `Unavailable`
- `Not connected`
- `Not wired`
- `Stubbed`
- `Unknown`
- `Degraded`

Disallowed states:

- fake probability defaults such as `33/33/34`
- fake compliance labels such as `SAFE`
- fake latency such as `0.00ms`
- fake equity curves
- fake CPU percentage derived from core count

### 2. Execution Gate Rule

`AI Auto-Trade` is a backend permission gate, not a cosmetic checkbox.

When disabled:

- bot/AI initiated execution must be denied
- denial must be explicit and journaled

When enabled:

- bot/AI initiated execution must still pass:
  - adapter connectivity
  - capability availability
  - risk gating
  - symbol/runtime validity

Manual operator trading remains allowed in this tranche as long as the normal runtime/risk gates pass.

### 3. Capability Honesty Rule

Every operator surface that describes trading, intelligence, compliance, or telemetry must report capability truthfully.

Examples:

- if AI probabilities are not backed by a real inference source, show `Model signal unavailable`
- if latency is not measured, show `Latency unavailable`
- if compliance/news gating is not wired, show `Not wired`
- if a model family is known stubbed, show `Stubbed`

## Architectural Changes

### Source-of-Truth Snapshot Layer

The service layer must own operator truth state. UI panels consume typed snapshots and do not invent operational values locally.

The snapshot layer will cover:

- execution gating state
- runtime telemetry availability
- AI/inference availability
- intelligence/compliance availability
- account/equity/feed availability

### Auto-Trade Backend Gate

The trading service must reject bot/AI-initiated execution when:

- auto-trade is disabled
- adapter is disconnected
- runtime capability is unavailable
- risk gate denies the action

Each denial must generate:

- a journal row
- a structured canonical log record

### Honest Capability Registry

The app will treat capabilities as explicit state rather than inferring them from visible UI panels.

Each relevant subsystem should expose a status like:

- available
- unavailable
- stubbed
- degraded
- disconnected

## Target Files

Primary files expected to change:

- `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/dashboard.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/ai_insights.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/system_status.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs`

Secondary files may change if needed for shared rendering or logging:

- `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs`

## Acceptance Criteria

- no synthetic operator metric is presented as live truth
- AI/bot trade paths are denied when auto-trade is off
- manual trades remain available when runtime/risk gates allow them
- every denied AI/bot action produces an explicit journal/log reason
- top-ribbon metrics are either real or explicitly unavailable
- dashboard and AI panels no longer present fake equity/probabilities/compliance state
- no new warning/clippy regressions on verified paths

## Verification

Required:

- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`
- headless smoke run for `forex-app`

Test expectations:

- auto-trade off denies AI execution
- manual execution remains allowed
- unavailable truth states render explicitly
- denied actions are journaled

