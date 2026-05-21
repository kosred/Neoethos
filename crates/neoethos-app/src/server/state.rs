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
}

impl Default for AppApiState {
    fn default() -> Self {
        Self::new()
    }
}
