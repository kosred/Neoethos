//! Shared state handed to every axum route via `with_state`.
//!
//! `AppApiState` is intentionally tiny — it holds an `Arc` to whatever
//! TradingSession / cache / settings each route needs to read from. The
//! routes themselves do the work of converting domain objects into wire
//! DTOs. This keeps the server module decoupled from the business code:
//! you can mock `AppApiState` in a test by constructing it with stub data.
//!
//! For the Phase 1 milestone the state holds an `Option<TradingSnapshot>`
//! that the server fills with synthetic-but-realistic numbers when no
//! live broker session is wired in. As soon as we plumb the real
//! `TradingSession` accessors through (next session), we swap the
//! `Option` for an `Arc<TradingSession>` or a dedicated read-snapshot
//! channel.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex, RwLock, broadcast};

use crate::app_services::jobs::{CancellationFlag, JobKind};
use crate::server::codex::CodexFlowState;
use crate::server::engines_control::EngineRunState;

/// Process-wide config-file path, defaulting to `"config.yaml"` when no
/// CLI `--config` override is provided. Set once at startup via
/// [`install_config_path`]; queried via [`current_config_path`].
static CONFIG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// **F-231-related closure (2026-05-25)** — process-wide handle to
/// the bridge's `account_refresh` trigger channel. Set once at
/// startup from `main.rs` after `AppApiState::new()` constructs
/// the channel; readable from anywhere in the crate (notably the
/// cTrader execution-event parser) without threading
/// `AppApiState` through every call site.
///
/// Same pattern as `current_config_path()` — process-global,
/// install-once, accessed via a free function so deep call sites
/// don't depend on the axum router state.
static ACCOUNT_REFRESH_TX: OnceLock<tokio::sync::mpsc::UnboundedSender<()>> =
    OnceLock::new();

/// Install the global account-refresh trigger. Called from `main.rs`
/// exactly once, right after `AppApiState::new()`. Subsequent calls
/// are silent no-ops.
pub fn install_account_refresh_trigger(
    tx: tokio::sync::mpsc::UnboundedSender<()>,
) {
    let _ = ACCOUNT_REFRESH_TX.set(tx);
}

/// Trigger an immediate account refresh from anywhere in the
/// process. Used by the cTrader execution-event parser
/// (`parse_execution_event`) so a fill / close / margin call from
/// our own POST /orders flips the dashboard to the new state
/// without waiting up to 5 s for the bridge's safety poll.
///
/// Silent no-op when the global isn't installed yet (= startup
/// race window, harmless: the bridge's 5 s timer covers it).
pub fn trigger_global_account_refresh() {
    if let Some(tx) = ACCOUNT_REFRESH_TX.get() {
        if tx.send(()).is_err() {
            tracing::warn!(
                target: "neoethos_app::server::state",
                "global account_refresh_tx send failed — bridge receiver dropped?"
            );
        }
    }
}

/// Process-wide install of the resolved config-file path. Called once
/// from `main` after the CLI flag has been parsed; subsequent calls
/// are no-ops (the first install wins).
pub fn install_config_path(path: impl Into<PathBuf>) {
    let _ = CONFIG_PATH.set(path.into());
}

/// F-270 (2026-05-28): process-global record of whether THIS backend
/// was spawned by a Flutter supervisor (via `--launched-by-flutter`).
/// Exposed via `/healthz` so a second Flutter shell starting up while
/// a stale backend (api-test orphan, manually-started server) holds
/// port 7423 can tell "this is a sibling UI's backend, refuse second
/// launch" apart from "this is a stale backend, attach to it instead
/// of exiting the new shell".
static LAUNCHED_BY_FLUTTER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Record at startup whether `--launched-by-flutter` was set. Called
/// once from `main`; subsequent calls are no-ops (first install wins).
pub fn install_launched_by_flutter(flag: bool) {
    let _ = LAUNCHED_BY_FLUTTER.set(flag);
}

/// Read the launched-by-Flutter flag for `/healthz`. Defaults to false
/// when no install has happened (= test/library callers).
pub fn launched_by_flutter() -> bool {
    LAUNCHED_BY_FLUTTER.get().copied().unwrap_or(false)
}

/// Resolved config-file path. Free functions that don't carry
/// `AppApiState` (e.g. `engines_control::resolve_data_root`) consult
/// this to honour the operator's `--config` flag.
pub fn current_config_path() -> PathBuf {
    CONFIG_PATH
        .get()
        .cloned()
        // F-settings-persistence (2026-06-01): the fallback MUST be the same
        // canonical user-data config the engine loads on boot
        // (`%LOCALAPPDATA%\neoethos\config.yaml` via `user_config_path`), NOT a
        // CWD-relative "config.yaml". Otherwise the `/settings` GET/POST
        // handlers read+write a DIFFERENT file than `Settings::load` reads on
        // next launch, so saved settings silently vanish (operator: "settings
        // show defaults / it keeps nothing").
        .unwrap_or_else(neoethos_core::config::user_config_path)
}

/// A minimal, render-ready account snapshot. Same shape as the
/// `AccountSnapshot` Dart class in `backend_client.dart`. Kept here
/// instead of in `account.rs` so other routes can read account state
/// without a circular dep.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSnapshotPayload {
    pub balance: f64,
    pub equity: f64,
    pub free_margin: f64,
    pub used_margin: f64,
    pub currency: String,
    /// Server-side wall-clock when this snapshot was assembled
    /// (Unix milliseconds, UTC). Flutter converts to local time
    /// for the "as of HH:MM:SS" badge on the Dashboard so the
    /// operator always knows whether the displayed numbers are
    /// fresh or stale.
    pub fetched_at_unix_ms: i64,
    pub positions: Vec<PositionPayload>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionPayload {
    pub position_id: i64,
    /// Broker volume in centi-lots (what `POST /positions/close`
    /// wants). 0 if the source feed doesn't expose it — the Flutter
    /// side falls back to a "volume in lots * 100000 * 100" estimate.
    pub volume_units: i64,
    pub symbol: String,
    pub side: String,
    pub volume: f64,
    /// Position open time as Unix milliseconds (UTC). Flutter
    /// converts to local time for the "Open since HH:MM" badge in
    /// the position row. `None` when cTrader didn't include it
    /// (unusual but possible mid-fill).
    pub open_timestamp_ms: Option<i64>,
    pub pnl_pips: f64,
    pub pnl_usd: f64,
}

/// Cheap-to-clone handle to whatever the server needs to read.
///
/// Wrapped in `Arc<RwLock<...>>` so background tasks (e.g. the
/// upcoming spot-stream worker) can write updates without blocking
/// the route layer for reads. `RwLock` over `Mutex` because most
/// requests are reads.
#[derive(Clone)]
pub struct AppApiState {
    inner: Arc<RwLock<AppApiInner>>,
    /// In-flight OAuth state for the Codex (ChatGPT) fallback. Kept
    /// as a separate `Mutex<Option<...>>` rather than inside the main
    /// RwLock because:
    ///   1. It's write-heavy on the slow path (callback completion
    ///      updates it) and the main state is read-heavy — splitting
    ///      avoids reader-starvation pathologies.
    ///   2. Only the `/auth/codex/*` routes touch it, so isolating
    ///      the lock surface keeps cross-handler coupling minimal.
    ///   3. The data is small (`Option<CodexFlowState>` is ≤ 200 B);
    ///      cloning the `Arc<Mutex<...>>` per request is free.
    pub codex: Arc<Mutex<Option<CodexFlowState>>>,
    /// Path to the `config.yaml` (or operator-chosen alternative) that
    /// routes consult via `Settings::from_yaml(state.config_path())`.
    /// `Arc<PathBuf>` so cloning state-per-request is free and the
    /// router stays Send + Sync without an extra lock.
    config_path: Arc<PathBuf>,
    /// **2026-05-25 — operator directive "uniform push everywhere"**:
    /// broadcast channel fired on every `set_account` write. The
    /// `/account/snapshot/stream` SSE endpoint subscribes here and
    /// forwards account updates to Flutter as they arrive — same
    /// pattern as `live_spots::SPOT_BROADCAST` for ticks. Capacity
    /// 64 = generous buffer for slow consumers; the cache always
    /// has the latest value so a dropped broadcast is never a
    /// correctness issue.
    account_broadcast: broadcast::Sender<AccountSnapshotPayload>,
    /// **2026-05-25 — operator directive "uniform push everywhere"**:
    /// account-refresh trigger. The bridge polling loop checks this
    /// channel every tick; when a message arrives, it runs an
    /// immediate `refresh_once` instead of waiting for the 5 s
    /// timer. Senders:
    ///   1. The future cTrader `OAExecutionEvent` handler (fill /
    ///      close / margin-call push from the broker → instant
    ///      account refresh on the bridge).
    ///   2. `POST /account/snapshot/refresh` — operator-triggered
    ///      force-refresh button in the UI.
    ///
    /// Unbounded `mpsc::UnboundedSender` because the events are
    /// rare and we never want to block the broker handler waiting
    /// for the bridge to drain. Each enqueue is a single atomic.
    account_refresh_tx: tokio::sync::mpsc::UnboundedSender<()>,
    /// Receiver paired with `account_refresh_tx`. Wrapped in `Mutex`
    /// so the bridge can take exclusive ownership at startup
    /// without us needing to thread it through the constructor.
    /// One-shot ownership: the bridge takes it once via
    /// `take_account_refresh_rx`; subsequent calls panic in debug
    /// (deliberate — a second taker is a bug).
    account_refresh_rx:
        Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<()>>>>,
    /// Live autonomous trading job handle. `None` when idle.
    pub live_trading: Arc<std::sync::Mutex<Option<crate::app_services::live_trading::Handle>>>,
}

#[derive(Default)]
pub(crate) struct AppApiInner {
    pub account: Option<AccountSnapshotPayload>,
    pub discovery: EngineSlot,
    pub training: EngineSlot,
    /// Cached map from cTrader `symbol_id` (i64) to human-readable
    /// ticker (e.g. `1` → `"EURUSD"`). Populated by:
    ///   1. The `/broker/symbols` route after a successful fetch.
    ///   2. The bridge's first lazy refresh when a position needs a
    ///      name but the cache is empty.
    /// The bridge reads through this map in `position_to_payload` so
    /// the dashboard shows `EURUSD` instead of the previous `sym#1`
    /// placeholder. Empty by default — falls back to `sym#<id>` only
    /// when neither path has populated the cache yet (e.g. broker
    /// not authed at boot).
    pub symbol_catalog: HashMap<i64, String>,
}

/// In-memory tracking of one engine's lifecycle. `state` is what
/// `/engines/status` returns; `cancel` lets `/engines/{kind}/stop`
/// signal the running job; `summary` is the latest one-line status
/// from the job's progress reports.
#[derive(Debug, Clone, Default)]
pub struct EngineSlot {
    pub state: EngineRunState,
    pub cancel: Option<CancellationFlag>,
    pub summary: String,
    /// F-340 (Feature #14): live discovery/training progress mirrored
    /// from the `JobSnapshot` the ServiceEvent drainer processes.
    /// `stage` is the coarse phase label (e.g. `"search_generations"`),
    /// empty when idle. `percent` is 0.0..=1.0, 0.0 when idle.
    /// `counters` is the live `(name, value)` counter list, empty when
    /// idle. Reset to defaults whenever the engine reaches a terminal
    /// (non-Running) state so `/engines/status` reports clean numbers
    /// the instant a run finishes.
    pub stage: String,
    pub percent: f64,
    pub counters: Vec<(String, u64)>,
}

impl Default for EngineRunState {
    fn default() -> Self {
        EngineRunState::Idle
    }
}

impl AppApiState {
    /// Construct with empty state. Routes that hit unfilled fields
    /// return a deterministic placeholder so the Flutter UI never
    /// renders an empty white screen during a fresh boot.
    ///
    /// The `config_path` is sourced from [`current_config_path`] so
    /// state built mid-process inherits the same install that
    /// `main.rs` performed via [`install_config_path`]. Tests that
    /// don't install ahead of time get the default `"config.yaml"`.
    pub fn new() -> Self {
        let (account_broadcast, _) = broadcast::channel(64);
        let (account_refresh_tx, account_refresh_rx) =
            tokio::sync::mpsc::unbounded_channel();
        Self {
            inner: Arc::new(RwLock::new(AppApiInner::default())),
            codex: Arc::new(Mutex::new(None)),
            config_path: Arc::new(current_config_path()),
            account_broadcast,
            account_refresh_tx,
            account_refresh_rx: Arc::new(std::sync::Mutex::new(Some(account_refresh_rx))),
            live_trading: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Clone of the account-refresh sender, suitable for installing
    /// as the process-wide handle via `install_account_refresh_trigger`.
    /// Called by `main.rs` exactly once at startup.
    pub fn account_refresh_tx_clone(
        &self,
    ) -> tokio::sync::mpsc::UnboundedSender<()> {
        self.account_refresh_tx.clone()
    }

    /// Trigger an immediate account refresh. Non-blocking; if the
    /// bridge isn't draining the channel (shouldn't happen — it's
    /// the bridge's job), the send still succeeds and accumulates.
    /// Used by:
    ///   - `POST /account/snapshot/refresh` (operator force-refresh)
    ///   - cTrader `OAExecutionEvent` handler via the global trigger
    ///     (`trigger_global_account_refresh`)
    pub fn trigger_account_refresh(&self) {
        // `send` only fails when the receiver is dropped — that
        // would mean the bridge died, which is a separate problem.
        // Best-effort, log on failure.
        if self.account_refresh_tx.send(()).is_err() {
            tracing::warn!(
                target: "neoethos_app::server::state",
                "account_refresh_tx send failed — bridge receiver dropped? \
                 Account snapshot will still refresh on the 5s timer."
            );
        }
    }

    /// Bridge calls this exactly once at startup to take ownership of
    /// the receiver side. Returns `None` on a second call so a future
    /// regression doesn't silently spawn two consumers.
    pub fn take_account_refresh_rx(
        &self,
    ) -> Option<tokio::sync::mpsc::UnboundedReceiver<()>> {
        self.account_refresh_rx
            .lock()
            .ok()
            .and_then(|mut g| g.take())
    }

    /// Subscribe to account-snapshot broadcasts. Each call returns a
    /// fresh receiver; the SSE endpoint
    /// `/account/snapshot/stream` calls this to wrap the receiver
    /// into a push stream for Flutter.
    pub fn subscribe_account(&self) -> broadcast::Receiver<AccountSnapshotPayload> {
        self.account_broadcast.subscribe()
    }

    /// Resolved config-file path. Routes that previously hardcoded
    /// `Settings::from_yaml("config.yaml")` now read this so the
    /// operator's `--config` flag flows through consistently.
    ///
    /// **F-553/F-576 closure (2026-05-25)**: the `with_config_path`
    /// builder was dropped because `main.rs` now uses
    /// [`install_config_path`] for the process-wide install, and
    /// `AppApiState::new()` already sources its default from
    /// [`current_config_path`]. Keeping an unused builder around
    /// would be the "dead code with attribute" anti-pattern the
    /// operator rejected on 2026-05-25 (see `chart.rs` Broker
    /// variant + `score_from_metrics` shim).
    pub fn config_path(&self) -> &Path {
        self.config_path.as_path()
    }

    /// Bootstrap with a canned snapshot — used by the `#[cfg(test)]`
    /// router fixtures in `account.rs` so axum handler tests can hit
    /// `/account/snapshot` without spinning up a real bridge task.
    /// Production paths leave the inner cache empty and let the bridge
    /// fill it via `set_account` once the broker session is up.
    #[cfg(test)]
    pub fn with_seed_account(mut self, snapshot: AccountSnapshotPayload) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("seeded test state must not be shared yet")
            .get_mut()
            .account = Some(snapshot);
        self
    }

    /// Read the current account snapshot. `None` means the broker
    /// session hasn't produced one yet — the route turns this into a
    /// `503 Service Unavailable` so the Flutter side can render a
    /// "waiting for broker…" placeholder.
    pub async fn account(&self) -> Option<AccountSnapshotPayload> {
        self.inner.read().await.account.clone()
    }

    /// Blocking-thread variant of [`account`] for callers that run
    /// inside `tokio::task::spawn_blocking` and cannot `.await`.
    /// Safe to call only from a thread that does NOT hold a tokio
    /// reactor — calling this on the reactor thread would deadlock
    /// the RwLock.
    #[allow(dead_code)]
    pub fn account_blocking(&self) -> Option<AccountSnapshotPayload> {
        self.inner.blocking_read().account.clone()
    }

    /// Overwrite the cached snapshot AND publish to the broadcast
    /// channel so any subscribed SSE clients receive the new state
    /// immediately. Called from the bridge polling loop on every
    /// successful `refresh_once`.
    ///
    /// **2026-05-25**: the stale `#[allow(dead_code)]` (left from
    /// pre-bridge-wiring days) was removed — the function is now
    /// used both by the bridge AND by the SSE push fanout.
    pub async fn set_account(&self, snapshot: AccountSnapshotPayload) {
        self.inner.write().await.account = Some(snapshot.clone());
        // Best-effort push. `send` returns `Err` when there are no
        // subscribers — fine; the cache write above still serves the
        // polling GET path so nothing is lost.
        let _ = self.account_broadcast.send(snapshot);
    }

    /// Wipe the cached snapshot — used by the bridge when refresh
    /// fails repeatedly so `/account/snapshot` flips back to 503
    /// instead of serving last-known-good numbers from a session
    /// the broker has since invalidated. Without this the dashboard
    /// could lie for hours after `CH_ACCESS_TOKEN_INVALID`.
    pub async fn clear_account(&self) {
        self.inner.write().await.account = None;
    }

    // ─── Symbol catalog accessors ──────────────────────────────────────
    //
    // `/broker/symbols` is the source of truth (the cTrader API call
    // that returns the per-account ticker list). The bridge reads
    // through the cached map to label positions with real names
    // instead of the legacy `sym#<id>` placeholder.

    /// Replace the cached symbol-name lookup table. The
    /// `/broker/symbols` route calls this after every successful
    /// fetch so the freshest names are always available to the
    /// bridge — no staleness window even if the broker re-issues
    /// symbol IDs after a maintenance window.
    pub async fn set_symbol_catalog(&self, catalog: HashMap<i64, String>) {
        self.inner.write().await.symbol_catalog = catalog;
    }

    /// Resolve a `symbol_id` to its ticker name. `None` when the
    /// cache hasn't been populated yet — the caller falls back to a
    /// `sym#<id>` placeholder so the operator still sees *which*
    /// symbol the position is against.
    pub async fn resolve_symbol_name(&self, symbol_id: i64) -> Option<String> {
        self.inner
            .read()
            .await
            .symbol_catalog
            .get(&symbol_id)
            .cloned()
    }

    /// Whether the cache has any entries. The bridge uses this to
    /// decide whether to fire a lazy `/broker/symbols`-equivalent
    /// refresh on the first position it sees.
    pub async fn symbol_catalog_is_empty(&self) -> bool {
        self.inner.read().await.symbol_catalog.is_empty()
    }

    // ─── Engine slot accessors ────────────────────────────────────────

    /// Read the current run state for the given engine.
    pub async fn engine_state(&self, kind: JobKind) -> EngineRunState {
        let inner = self.inner.read().await;
        match kind {
            JobKind::Discovery => inner.discovery.state,
            JobKind::Training => inner.training.state,
            JobKind::Bootstrap => EngineRunState::Idle,
        }
    }

    /// Read the latest one-line summary for the given engine.
    pub async fn engine_summary(&self, kind: JobKind) -> String {
        let inner = self.inner.read().await;
        match kind {
            JobKind::Discovery => inner.discovery.summary.clone(),
            JobKind::Training => inner.training.summary.clone(),
            JobKind::Bootstrap => String::new(),
        }
    }

    /// F-340 (Feature #14): read the live progress triple
    /// `(stage, percent, counters)` for the given engine. Returns
    /// `("", 0.0, [])` for the Bootstrap variant (it has no slot) and
    /// for any engine that is idle / has been reset on terminal.
    pub async fn engine_progress(&self, kind: JobKind) -> (String, f64, Vec<(String, u64)>) {
        let inner = self.inner.read().await;
        let slot = match kind {
            JobKind::Discovery => &inner.discovery,
            JobKind::Training => &inner.training,
            JobKind::Bootstrap => return (String::new(), 0.0, Vec::new()),
        };
        (slot.stage.clone(), slot.percent, slot.counters.clone())
    }

    /// Mark an engine as Running and remember its cancel flag so a
    /// later `/stop` can signal it. Called by the discovery/training
    /// `start` endpoints right after `start_*_job` returns.
    pub async fn install_engine(&self, kind: JobKind, cancel: CancellationFlag) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::Bootstrap => return,
        };
        slot.state = EngineRunState::Running;
        slot.cancel = Some(cancel);
        slot.summary = "starting…".to_string();
    }

    /// Reflect the latest ServiceEvent-derived state into the slot.
    /// Called from the engines_control state-drainer task.
    pub async fn update_engine(&self, kind: JobKind, state: EngineRunState, summary: String) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::Bootstrap => return,
        };
        slot.state = state;
        if !summary.is_empty() {
            slot.summary = summary;
        }
        // Once we hit a terminal state, drop the cancel flag — there's
        // nothing left to cancel.
        if !matches!(state, EngineRunState::Running) {
            slot.cancel = None;
            // F-340 (Feature #14): also wipe the live progress so
            // `/engines/status` reports `("", 0.0, [])` the instant a
            // run finishes — a stale "search_generations / 0.83" line
            // hanging around after a Succeeded run would mislead the UI.
            slot.stage.clear();
            slot.percent = 0.0;
            slot.counters.clear();
        }
    }

    /// F-340 (Feature #14): mirror the live `JobSnapshot` progress
    /// (`stage`, `percent`, `counters`) into the engine slot. Called by
    /// the engines_control ServiceEvent drainer alongside
    /// [`update_engine`] on every non-terminal update so
    /// `/engines/status` exposes the rich counters the discovery job
    /// accumulates. `percent` is clamped to 0.0..=1.0. Terminal cleanup
    /// is handled by [`update_engine`], so this setter only ever writes
    /// "live" values.
    pub async fn set_engine_progress(
        &self,
        kind: JobKind,
        stage: String,
        percent: f64,
        counters: Vec<(String, u64)>,
    ) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::Bootstrap => return,
        };
        slot.stage = stage;
        slot.percent = percent.clamp(0.0, 1.0);
        slot.counters = counters;
    }

    /// Defensive guard: if the ServiceEvent channel closes without a
    /// terminal event (shouldn't happen in practice but worth
    /// covering), flip Running → Idle so the UI doesn't get stuck.
    pub async fn finalize_engine_if_running(&self, kind: JobKind) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::Bootstrap => return,
        };
        if matches!(slot.state, EngineRunState::Running) {
            slot.state = EngineRunState::Idle;
            slot.cancel = None;
        }
    }

    /// Fire the cancel flag on the named engine if one is registered.
    /// Returns `true` if a job was actually running; `false` is the
    /// idempotent no-op case.
    pub async fn cancel_engine(&self, kind: JobKind) -> bool {
        let inner = self.inner.read().await;
        let slot = match kind {
            JobKind::Discovery => &inner.discovery,
            JobKind::Training => &inner.training,
            JobKind::Bootstrap => return false,
        };
        if let Some(cancel) = &slot.cancel {
            cancel.request();
            true
        } else {
            false
        }
    }
}

impl Default for AppApiState {
    fn default() -> Self {
        Self::new()
    }
}
