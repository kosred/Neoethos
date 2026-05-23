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
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::app_services::jobs::{CancellationFlag, JobKind};
use crate::server::engines_control::EngineRunState;

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
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(AppApiInner::default())),
        }
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
    /// inside `tokio::task::spawn_blocking` and cannot `.await`
    /// (e.g. the Gemma tool-loop dispatcher, which executes inside
    /// the LLM inference blocking thread). Safe to call only from a
    /// thread that does NOT hold a tokio reactor — calling this on
    /// the reactor thread would deadlock the RwLock.
    pub fn account_blocking(&self) -> Option<AccountSnapshotPayload> {
        self.inner.blocking_read().account.clone()
    }

    /// Overwrite the cached snapshot. Called from whatever background
    /// task pulls live data off the cTrader stream.
    #[allow(dead_code)] // wired up next session when the streaming worker lands
    pub async fn set_account(&self, snapshot: AccountSnapshotPayload) {
        self.inner.write().await.account = Some(snapshot);
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
        }
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
