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

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::app_services::jobs::{CancellationFlag, JobKind};
use crate::server::engines_control::EngineRunState;

/// A minimal, render-ready account snapshot. Same shape as the
/// `AccountSnapshot` Dart class in `backend_client.dart`. Kept here
/// instead of in `account.rs` so other routes can read account state
/// without a circular dep.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AccountSnapshotPayload {
    pub balance: f64,
    pub equity: f64,
    pub free_margin: f64,
    pub used_margin: f64,
    pub currency: String,
    pub positions: Vec<PositionPayload>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PositionPayload {
    pub symbol: String,
    pub side: String,
    pub volume: f64,
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

    /// Bootstrap with a canned snapshot so the Flutter dashboard has
    /// something to render before live data arrives. Production paths
    /// overwrite this via `set_account` once the broker session is up.
    pub fn with_seed_account(self, snapshot: AccountSnapshotPayload) -> Self {
        {
            let inner = self.inner.clone();
            // We're sync here (constructor path), so block_on the RwLock
            // write — there's no contention yet because nothing else
            // holds the Arc.
            let mut guard = inner.blocking_write();
            guard.account = Some(snapshot);
        }
        self
    }

    /// Read the current account snapshot. `None` means the broker
    /// session hasn't produced one yet — the route turns this into a
    /// `503 Service Unavailable` so the Flutter side can render a
    /// "waiting for broker…" placeholder.
    pub async fn account(&self) -> Option<AccountSnapshotPayload> {
        self.inner.read().await.account.clone()
    }

    /// Overwrite the cached snapshot. Called from whatever background
    /// task pulls live data off the cTrader stream.
    #[allow(dead_code)] // wired up next session when the streaming worker lands
    pub async fn set_account(&self, snapshot: AccountSnapshotPayload) {
        self.inner.write().await.account = Some(snapshot);
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
    pub async fn update_engine(
        &self,
        kind: JobKind,
        state: EngineRunState,
        summary: String,
    ) {
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
