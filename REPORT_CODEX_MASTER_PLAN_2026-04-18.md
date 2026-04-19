# Forex-AI — Codex Master Execution Plan

**Repository:** `kosred/forex-ai`  
**Reference date:** 2026-04-18 (Saturday)  
**Purpose:** Full execution report and migration plan for Codex.  
**Primary direction:** Convert the project into a **cTrader-first, Rust-first, hardware-adaptive trading platform** with minimal or zero required Python in the default runtime path.  
**UI direction:** Rebuild the desktop experience so it is **~99% TradingView-like in workflow and ergonomics**, without copying branding or proprietary assets.

---

## 0. How this document must be used

This is **not** a brainstorm note.
This is an **execution document**.

Codex must follow the order in this file.

Do **not** start from cosmetic UI work.
Do **not** start from random model rewrites.
Do **not** start from config cleanup.
Do **not** start from isolated bug fixes unless they block the migration.

The correct order is:

1. Remove runtime ambiguity.
2. Make cTrader the canonical live broker path.
3. Isolate or remove MT5/Python-required runtime pieces.
4. Introduce a universal hardware planner.
5. Unify discovery/training/model execution around that planner.
6. Only then redesign the UI.

If this order is violated, the repo will get noisier and harder to maintain.

---

## 1. Executive summary

The repository already contains serious work.
It is **not** an empty shell.
It has:

- a heavy data and feature engine,
- a serious search/discovery subsystem,
- a broad model runtime layer,
- a strong policy/risk/persistence spine,
- and a much more mature cTrader stack than it may first appear.

The main architectural problem is **not missing functionality**.
The main problem is **execution fragmentation**.

Today the project is split across:

- CPU-first fallbacks,
- GPU-specific paths,
- feature-flag behavior,
- Python-facing surfaces,
- MT5 legacy logic,
- model-family-specific runtime assumptions,
- and UI surfaces that expose only simplified runtime state.

The project is therefore **hardware-aware**, but not yet **hardware-orchestrated**.
It is also **cTrader-capable**, but not yet **cTrader-canonical**.
It is **Rust-heavy**, but not yet **Python-independent by default**.

This plan exists to fix exactly that.

---

## 2. Primary architectural decision

### Canonical target architecture

The project must move toward this canonical structure:

- **Broker runtime:** cTrader only in the default runtime path
- **Default application runtime:** Rust-native app + Rust-native CLI
- **Default model/search runtime:** Rust-native, hardware-adaptive, multi-backend
- **Python:** optional compatibility layer only, never required for the main desktop/runtime flow
- **UI direction:** TradingView-like workflow and layout grammar, without branding or asset copying

### Meaning in practice

This means:

- `mt5-bridge` becomes legacy and then removable
- MT5 startup and runtime awareness disappear from the app default flow
- cTrader auth, streaming, execution, account discovery, bootstrap and market data become the official live path
- hardware selection stops being a boolean `gpu_enabled` concept and becomes a real device/backend planning layer
- discovery, training, inference and evaluation stop making hidden backend decisions independently
- the UI becomes an operator workstation, not a place that hides runtime ambiguity

---

## 3. Hard conclusions from the repository audit

### 3.1 Strong areas to preserve

#### `crates/forex-data`
Strong candidate to remain a core pillar.
Heavy feature/data engine, multi-timeframe preparation, Vortex-native storage and cache patterns.
Do not rewrite unless necessary.

#### `crates/forex-core`
Strong policy/persistence spine.
Risk, meta-controller, drift monitoring, sectioned logs, storage, system behavior.
Do not flatten or dilute this.

#### `crates/forex-models`
Broad runtime/model layer.
A lot of honest runtime metadata, artifact discipline, sidecars, fallback stories, training profiles, and model-family implementations already exist.
The problem is mostly **unification**, not lack of work.

#### `crates/forex-search`
Real search/discovery system exists.
The issue is not “missing search”; the issue is “multiple search execution worlds” (CPU-first baseline vs GPU-special path).

#### cTrader app services in `crates/forex-app/src/app_services`
This is one of the strongest parts of the repo.
The cTrader path already includes:

- auth state,
- browser/loopback login,
- token exchange,
- token refresh,
- secure token persistence,
- account discovery,
- historical data,
- live data,
- execution,
- bootstrap/reconciliation,
- streaming.

This is strong enough to become the canonical live path.

### 3.2 Structural weaknesses or fragmentation

#### MT5 / Python runtime legacy
`crates/mt5-bridge` is a PyO3 bridge to Python `MetaTrader5` and is explicitly read-only.
This is pure architectural debt in the current direction.

#### Hardware/runtime abstraction
The project has device hints, GPU detection, CUDA awareness, some WGPU/Burn structure, and special GPU paths.
But it does **not** yet have a universal planner for:

- CPU-only,
- single GPU,
- multi-GPU,
- CUDA,
- WGPU,
- ROCm,
- ONNX providers,
- backend fallback policy,
- workload-specific backend planning.

#### Search execution split
The repo contains a true GPU path for discovery, but the baseline operational path remains mostly CPU/host-driven.
This creates two worlds instead of one planner-driven execution model.

#### UI hardware surface
The current hardware UI is far too simple for the ambition of the project.
A slider for CPU cores and a CUDA checkbox is not a serious runtime control surface.

#### Python-facing API scope
`crates/forex-bindings` is not small.
It is a large compatibility surface.
Python removal is therefore a staged problem, not a single deletion.

---

## 4. Non-negotiable design principles

### Principle A — One canonical live runtime
There must be exactly one default live broker runtime.
That runtime is cTrader.

### Principle B — Python is optional, never required
Default app runtime, default CLI runtime, default training/discovery runtime must not require Python.

### Principle C — Hardware decisions must be centralized
No model family, search module or UI panel should independently invent execution backend policy.

### Principle D — UI must not hide architecture problems
The UI must present runtime reality clearly.
It is not the place to paper over backend inconsistency.

### Principle E — Artifact/runtime truth must be explicit
If a model used fallback CPU, sidecar surrogate, degraded profile, simple ES backend, or local fallback tree model, it must say so in metadata and runtime state.

### Principle F — Mutation that changes training semantics must invalidate trained state
If a setter or builder changes architecture, feature schema, search semantics, backend semantics, replay memory semantics or calibration semantics, then old trained state must be invalidated.

---

## 5. Repository map

### Workspace members

- `crates/forex-search`
- `crates/forex-cli`
- `crates/forex-data`
- `crates/forex-models`
- `crates/forex-core`
- `crates/forex-bindings`
- `crates/forex-app`
- `crates/mt5-bridge`
- `crates/forex-news`

### Canonical crates going forward

Keep central:

- `forex-data`
- `forex-core`
- `forex-models`
- `forex-search`
- `forex-app`
- `forex-cli`

Optional or legacy direction:

- `forex-bindings` → optional compatibility surface
- `mt5-bridge` → legacy target, remove from default runtime first, then remove entirely
- `forex-news` → keep isolated and optional in business flow

---

## 6. Migration phases — strict order

# Phase 1 — Quarantine MT5 and make cTrader the default live broker path

## Objective
Remove MT5 from the default runtime story without breaking the app.
Do not delete everything immediately. First isolate, then remove.

## Files to inspect and edit first

- `crates/mt5-bridge/Cargo.toml`
- `crates/mt5-bridge/src/lib.rs`
- `crates/forex-app/Cargo.toml`
- `crates/forex-app/src/main.rs`
- `crates/forex-app/src/app_services/trading.rs`
- `crates/forex-app/src/app_services/broker_config.rs`
- `crates/forex-app/src/ui/system/brokers.rs`

## Required search commands

```bash
rg -n "mt5|MetaTrader5|pyo3|Python::with_gil|legacy-mt5" crates/
rg -n "BrokerAdapter|MT5|cTrader|ctrader" crates/forex-app
```

## Required changes

### 1. Make `mt5-bridge` optional
In `crates/forex-app/Cargo.toml`, move MT5 support behind a feature, for example:

```toml
[features]
default = ["ctrader-default"]
ctrader-default = []
legacy-mt5 = ["dep:mt5-bridge"]
```

### 2. Remove MT5 from default startup branch selection
In `crates/forex-app/src/main.rs`, remove any default branch that treats MT5 as a normal startup/runtime peer to cTrader.
The default live path must be cTrader.

### 3. Reduce `trading.rs` adapter ambiguity
In `crates/forex-app/src/app_services/trading.rs`:
- keep cTrader adapter fully live,
- move MT5 adapter creation behind `#[cfg(feature = "legacy-mt5")]`,
- isolate any MT5-specific state from the main runtime path.

### 4. Broker UI must stop presenting MT5 as canonical
In `crates/forex-app/src/ui/system/brokers.rs`:
- cTrader should be first-class and default,
- MT5 should either be hidden when feature-disabled or explicitly marked legacy.

## Phase 1 acceptance criteria

- App builds and runs without Python installed.
- App can complete cTrader auth/account selection flow.
- Default broker path is cTrader.
- MT5 code is feature-gated or clearly isolated.

## Phase 1 definition of done

`cargo build` for default feature set must not require `mt5-bridge` and must not require Python.

---

# Phase 2 — Promote cTrader runtime to canonical broker architecture

## Objective
Turn the already-strong cTrader path into the explicit runtime center.

## Files to inspect

- `crates/forex-app/src/app_services/ctrader_auth.rs`
- `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- `crates/forex-app/src/app_services/ctrader_account.rs`
- `crates/forex-app/src/app_services/ctrader_data.rs`
- `crates/forex-app/src/app_services/ctrader_execution.rs`
- `crates/forex-app/src/app_services/ctrader_streaming.rs`
- `crates/forex-app/src/app_services/ctrader_bootstrap.rs`
- `crates/forex-app/src/app_services/secure_store.rs`
- `crates/forex-app/src/app_services/trading.rs`

## Required change pattern

Create a unified runtime façade.

Recommended new file:

- `crates/forex-app/src/app_services/ctrader_runtime.rs`

Recommended structure:

```rust
pub struct CTraderRuntimeFacade {
    pub auth: CTraderAuthManager,
    pub accounts: CTraderAccountManager,
    pub data: CTraderDataManager,
    pub exec: CTraderExecutionManager,
    pub stream: CTraderStreamingManager,
    pub bootstrap: CTraderBootstrapManager,
}
```

## Required refactor

### 1. Centralize lifecycle
Create a single lifecycle pipeline:

- not configured
- ready
- browser auth
- callback capture
- token exchange
- token restore
- token refresh
- account discovery
- execution target selection
- live data ready
- execution ready

### 2. Centralize token/session ownership
Only one subsystem should own persisted token state.
That should be the secure store + auth manager path.

### 3. Centralize execution requests
Execution should pass through exactly one public service boundary.
No panel or helper should assemble raw execution websocket flow directly.

### 4. Centralize market data subscriptions
Streaming subscriptions should be owned by one service and reflected into runtime snapshots.

## Required grep/search

```bash
rg -n "CTraderAuth|CTraderExecution|CTraderStreaming|CTraderEnvironment|access_token|refresh_token" crates/forex-app/src/app_services
```

## Phase 2 acceptance criteria

- One cTrader runtime façade exists.
- UI panels consume façade-derived state, not scattered cTrader details.
- Execution, streaming, bootstrap and auth are lifecycle-coherent.

---

# Phase 3 — Introduce universal accelerator planner

## Objective
Replace today’s backend guessing with one central execution planner.

## Problem being solved
Today backend logic is fragmented across:

- `crates/forex-core/src/system.rs`
- `crates/forex-models/src/hardware.rs`
- `crates/forex-models/src/tree_models/config.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/parallel_trainer.rs`
- `crates/forex-search/src/lib.rs`
- `crates/forex-app/src/app_state.rs`
- `crates/forex-app/src/ui/hardware.rs`

This must stop.

## New module proposal

Create:

- `crates/forex-core/src/accelerator/mod.rs`
- `crates/forex-core/src/accelerator/probe.rs`
- `crates/forex-core/src/accelerator/plan.rs`
- `crates/forex-core/src/accelerator/policy.rs`

## Required structures

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceleratorBackend {
    Cpu,
    Wgpu,
    Cuda,
    Rocm,
    OnnxCpu,
    OnnxCuda,
    OnnxTensorrt,
}

#[derive(Debug, Clone)]
pub struct AcceleratorDevice {
    pub backend: AcceleratorBackend,
    pub ordinal: usize,
    pub name: String,
    pub total_memory_mb: u64,
    pub usable: bool,
}

#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    pub phase: WorkloadPhase,
    pub memory_intensity: u8,
    pub parallelism_intensity: u8,
    pub prefers_vectorized_execution: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkloadPhase {
    Search,
    Training,
    Inference,
    StreamingInference,
}

#[derive(Debug, Clone)]
pub struct AcceleratorPlan {
    pub search_backend: AcceleratorBackend,
    pub training_backend: AcceleratorBackend,
    pub inference_backend: AcceleratorBackend,
    pub devices: Vec<AcceleratorDevice>,
    pub multi_gpu: bool,
    pub allow_fallback: bool,
}
```

## Mandatory behaviors

### 1. Probe actual runtime availability
Stop depending mostly on `nvidia-smi` and `gpu_enabled: bool`.
Use provider-specific probes where possible.
If only CUDA is available, that must be explicit.
If WGPU is available but CUDA is not, that must be explicit.

### 2. Build a workload-aware plan
Do not just detect devices.
Plan backend per workload phase.
Search may prefer one backend, training another, inference another.

### 3. Expose plan to the whole repo
Make `AcceleratorPlan` the single object passed into:

- training orchestrator
- discovery backend selection
- inference model execution
- UI hardware state

## Required grep/search

```bash
rg -n "gpu_enabled|nvidia-smi|cuda|rocm|wgpu|onnx|ExecutionProvider|device_pref|gpu_only" crates/
```

## Phase 3 acceptance criteria

- A single `AcceleratorPlan` exists.
- Search/training/inference can all read it.
- The UI can render detected devices and selected policy from it.
- No new logic should depend on `gpu_enabled: bool` alone.

---

# Phase 4 — Unify discovery/search backends

## Objective
Remove the split between CPU-baseline discovery and GPU-special discovery.

## Current pain points
Important files:

- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/eval.rs`
- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/hpc_gpu_discovery.rs`
- `crates/forex-search/src/genetic/search_engine.rs`
- `crates/forex-search/src/genetic/evolution_math.rs`
- `crates/forex-search/src/genetic/strategy_gene.rs`

Today the repo contains both:

- CPU/host-driven baseline discovery,
- and a real GPU path,

but they are not under one clean backend contract.

## Required new abstraction

```rust
pub trait DiscoveryBackend {
    fn evaluate_population(
        &self,
        population: &[StrategyGene],
        ctx: &DiscoveryContext,
    ) -> anyhow::Result<Vec<FitnessResult>>;
}

pub struct CpuDiscoveryBackend;
pub struct GpuDiscoveryBackend;
pub struct HybridDiscoveryBackend;
```

## Required behavioral change

`run_discovery_cycle_with_progress(...)` must not choose the execution world indirectly.
It must accept or derive a backend instance from the new `AcceleratorPlan`.

## Required work

### 1. Separate discovery policy from discovery execution
Search policy (filters, thresholds, gauntlet, portfolio selection) should remain stable.
Execution backend should be swappable.

### 2. Keep one canonical result contract
CPU and GPU evaluation must return the same shape of `FitnessResult` and quality metadata.

### 3. Make GPU path no longer look like a side universe
GPU discovery should be one backend implementation, not a separate worldview.

## Required search commands

```bash
rg -n "run_discovery_cycle|used_gpu|discovery_gpu|hpc_gpu_discovery|evaluate_population_core|evolve_search" crates/forex-search
```

## Phase 4 acceptance criteria

- Exactly one discovery orchestration path.
- Multiple interchangeable execution backends.
- CPU and GPU use same high-level pipeline contract.

---

# Phase 5 — Unify model execution targets and invalidation discipline

## Objective
Make all model families obey one runtime execution contract and one invalidation policy.

## Important files

- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/runtime/*`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/hardware.rs`
- `crates/forex-models/src/burn_models.rs`
- `crates/forex-models/src/deep_models.rs`
- `crates/forex-models/src/tree_models/*`
- `crates/forex-models/src/rl/dqn_impl.rs`
- `crates/forex-models/src/exit_agent.rs`
- `crates/forex-models/src/evolution/*`
- `crates/forex-models/src/anomaly/forest_impl.rs`
- `crates/forex-models/src/statistical/*`
- `crates/forex-models/src/streaming/adaptive_impl.rs`
- `crates/forex-models/src/forecasting/swarm_impl.rs`

## Required new abstraction

```rust
#[derive(Debug, Clone)]
pub struct ExecutionTarget {
    pub backend: AcceleratorBackend,
    pub device_ordinal: Option<usize>,
    pub allow_fallback: bool,
}

pub trait HardwareAwareModel {
    fn preferred_target(&self) -> ExecutionTarget;
    fn train_with_target(&mut self, target: &ExecutionTarget) -> anyhow::Result<()>;
    fn predict_with_target(&self, target: &ExecutionTarget, x: &[f32]) -> anyhow::Result<Prediction>;
}
```

## Required refactor

### 1. Standardize requested/effective backend semantics
All model families must use the same language for:

- requested backend
- effective backend
- fallback status
- degraded status
- sidecar/local surrogate usage

### 2. Enforce invalidation discipline
When setters/builders mutate training semantics, invalidate:

- trained artifact
- runtime metadata
- sidecar/fallback artifact if necessary
- training report
- readiness flags

### 3. Standardize model-side execution target selection
No model family should invent its own hidden fallback logic without exposing it in runtime metadata.

## Required search commands

```bash
rg -n "requested_backend|effective_backend|execution_backend|device_pref|gpu_only|fallback|degraded" crates/forex-models
rg -n "with_|set_.*config|pub .*config|pub .*alpha|pub .*learning_rate|pub .*epochs" crates/forex-models/src
```

## Phase 5 acceptance criteria

- One shared execution target vocabulary.
- Honest runtime truth for all model families.
- No silent trained-state desynchronization after semantic mutations.

---

# Phase 6 — Reduce Python dependence strategically

## Objective
Make Python optional instead of foundational.

## Important files

- `crates/forex-bindings/Cargo.toml`
- `crates/forex-bindings/src/lib.rs`
- all files under `crates/forex-bindings/src/`
- `crates/mt5-bridge/*`

## Important reality

Removing Python is not a single delete.
`forex-bindings` is a large Python-facing surface.
Therefore, do this in stages.

## Required strategy

### Stage A — MT5 runtime removal
This is the first and easiest real win.
Make default runtime completely independent of `mt5-bridge`.

### Stage B — Separate Python bindings from default workspace flow
Options:

- make `forex-bindings` optional in default builds,
- or split it into a separate workspace/profile,
- or keep it as a compatibility crate that is not used by the default app/CLI runtime.

### Stage C — Replace placeholder-heavy Python binding surfaces where valuable
Files such as `crates/forex-bindings/src/data.rs` contain placeholder-like behavior in places.
Either complete them or clearly downgrade them to compatibility-only scope.

## Required grep/search

```bash
rg -n "pyo3|#[ ]*pyfunction|#[ ]*pymodule|Python::with_gil|PyResult|PyObject" crates/
rg -n "MetaTrader5|mt5-bridge" crates/
```

## Phase 6 acceptance criteria

- Default desktop runtime requires no Python.
- Default CLI runtime requires no Python.
- `forex-bindings` is optional or clearly separated.
- MT5 dependency no longer influences default architectural decisions.

---

# Phase 7 — UI redesign to TradingView-like workflow

## Objective
Rebuild the UI into a dense operator workstation with TradingView-like ergonomics.

## Important note

This means:

- similar workflow,
- similar layout grammar,
- similar information density,
- similar operator behavior,

but **not** copying branding, visual identity, proprietary assets or trademarked UI details.

## Current files to redesign

- `crates/forex-app/src/ui/hardware.rs`
- `crates/forex-app/src/ui/discovery.rs`
- `crates/forex-app/src/ui/training.rs`
- `crates/forex-app/src/ui/trading.rs`
- `crates/forex-app/src/ui/trading/chart_panel.rs`
- `crates/forex-app/src/ui/trading/execution_panel.rs`
- `crates/forex-app/src/ui/trading/watchlist_panel.rs`
- `crates/forex-app/src/ui/trading/bottom_strip.rs`
- `crates/forex-app/src/workspace/*`
- `crates/forex-app/src/ui/theme.rs`
- `crates/forex-app/src/ui/system/*`

## Target layout

### Top bar
Must include:

- symbol selector
- timeframe selector
- chart type selector
- indicator button
- layout/profile button
- quick search
- quick actions (save layout / reset zoom / crosshair / settings)

### Left vertical rail
Must include:

- cursor
- crosshair
- trend line
- horizontal line
- vertical line
- rectangle / box
- fib tool
- text / note
- ruler / measurement

### Center panel
Large chart area.
This must dominate the screen.

### Right rail
Must include:

- watchlist
- order ticket
- positions/orders summary
- symbol metrics / quick stats

### Bottom dock
Tabbed dock for:

- trades
- journal
- logs
- runtime diagnostics
- model statuses
- training runs
- discovery runs
- strategy explorer

## Required behavior

### 1. Hardware panel redesign
Stop using only:

- CPU core slider
- CUDA checkbox

Replace with:

- detected device list
- backend/provider priority
- training backend policy
- inference backend policy
- search backend policy
- multi-GPU distribution mode
- fallback behavior
- memory safety mode

### 2. Chart panel redesign
`chart_panel.rs` must evolve from a good custom painter into a real workstation chart surface with:

- overlays
- tool state
- crosshair behavior
- visible scale controls
- left/right axes behavior
- bottom time scale behavior
- object selection and deletion

### 3. Discovery/training panes must become dockable operator tabs
They should not feel like separate app screens.
They should feel like workstation panels.

## Suggested egui shell sketch

```rust
egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
    render_symbol_timeframe_bar(ui, state);
});

egui::SidePanel::left("tools")
    .resizable(false)
    .default_width(48.0)
    .show(ctx, |ui| render_drawing_toolbar(ui, state));

egui::SidePanel::right("right_rail")
    .default_width(340.0)
    .show(ctx, |ui| {
        render_watchlist(ui, state);
        ui.separator();
        render_order_ticket(ui, state);
    });

egui::TopBottomPanel::bottom("bottom_dock")
    .default_height(220.0)
    .show(ctx, |ui| render_bottom_tabs(ui, state));

egui::CentralPanel::default().show(ctx, |ui| {
    render_tradingview_like_chart(ui, state);
});
```

## Phase 7 acceptance criteria

- The app visually behaves like a chart workstation.
- Chart is the dominant center.
- Order/watchlist/runtime panels live in clear side/dock zones.
- Hardware/runtime UI exposes real runtime state.

---

## 7. Search commands Codex must run early

```bash
rg -n "mt5|MetaTrader5|pyo3|Python::with_gil" crates/
rg -n "CTrader|ctrader" crates/forex-app crates/forex-core crates/forex-bindings
rg -n "gpu_enabled|nvidia-smi|cuda|rocm|wgpu|tch::|ExecutionProvider|onnx" crates/
rg -n "run_discovery_cycle|evolve_search|used_gpu|discovery_gpu|hpc_gpu_discovery" crates/
rg -n "requested_backend|effective_backend|execution_backend|device_pref|gpu_only" crates/
rg -n "with_|set_.*config|public .*config|pub .*config|pub .*alpha|pub .*epochs" crates/forex-models/src
```

---

## 8. Recommended pull request sequence

### PR1 — `quarantine-mt5-and-make-ctrader-default`
Goal: make cTrader the default live broker path, isolate MT5 behind a feature.

### PR2 — `introduce-universal-accelerator-planner`
Goal: central hardware/device/backend planning layer.

### PR3 — `unify-discovery-backends`
Goal: CPU/GPU discovery under one backend contract.

### PR4 — `unify-model-execution-targets-and-invalidation`
Goal: shared execution target contract and trained-state discipline.

### PR5 — `separate-python-bindings-from-default-runtime`
Goal: make Python optional.

### PR6 — `tradingview-like-workstation-ui-shell`
Goal: chart workstation shell, right rail, bottom dock, hardware panel redesign.

### PR7 — `remove-legacy-mt5-paths`
Goal: full deletion of dead MT5 branches if Phase 1–6 succeeded.

---

## 9. Definition of done

The work is considered complete when all of the following are true:

1. The app runs without Python installed.
2. `mt5-bridge` is not part of the default build.
3. cTrader is the only canonical live broker runtime.
4. The hardware panel shows real devices and backend policy, not a simple checkbox.
5. Search, training and inference all receive a shared `AcceleratorPlan`.
6. The discovery stack uses a unified backend interface.
7. Model families expose honest execution/runtime truth.
8. Python bindings are optional and not default-architectural.
9. The UI feels like a chart workstation and is strongly TradingView-like in flow.
10. Semantic model mutations invalidate stale trained state.

---

## 10. Final instruction to Codex

Do not start from UI cosmetics.
Do not start from config files.
Do not start by “modernizing” everything at once.

Start from:

**runtime simplification → broker canonicalization → hardware planner → discovery/model unification → UI redesign**

If this order is respected, the project will become simpler and stronger.
If not, the project will just accumulate prettier chaos.
