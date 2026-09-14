//! Shared state handed to every axum route via `with_state`.
//!
//! `AppApiState` is intentionally tiny — it holds an `Arc` to whatever
//! TradingSession / cache / settings each route needs to read from. The
//! routes themselves do the work of converting domain objects into wire
//! DTOs. This keeps the server module decoupled from the business code:
//! you can mock `AppApiState` in a test by constructing it with stub data.
//!
//! Account values start unknown. Only a verified broker response whose scope
//! still matches the persisted execution selection may populate the cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex, RwLock, broadcast};

use crate::app_services::canonical_native_discovery::{
    CanonicalNativeResearchSnapshotV1, CanonicalNativeResearchStateV1,
    CanonicalNativeResearchTerminalSnapshotV1,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::jobs::{CancellationFlag, JobKind, JobSnapshot};
use crate::server::codex::CodexFlowState;
use crate::server::engines_control::EngineRunState;
use neoethos_core::Settings;
use neoethos_search::{
    CanonicalNativeCancellationTokenV1, CanonicalNativeDiscoveryRequestErrorV1,
    CanonicalNativeRuntimeInstallReceiptV1, install_and_seal_canonical_native_runtime_authority_v1,
};

/// Process-wide config-file path, defaulting to `"config.yaml"` when no
/// CLI `--config` override is provided. Set once at startup via
/// [`install_config_path`]; queried via [`current_config_path`].
static CONFIG_PATH: OnceLock<PathBuf> = OnceLock::new();

#[derive(Clone)]
pub(crate) struct CanonicalNativeStartupAuthorityV1 {
    settings: Arc<Settings>,
    runtime_install_receipt: Arc<CanonicalNativeRuntimeInstallReceiptV1>,
}

impl CanonicalNativeStartupAuthorityV1 {
    pub(crate) fn settings(&self) -> Arc<Settings> {
        self.settings.clone()
    }

    pub(crate) fn runtime_install_receipt(&self) -> Arc<CanonicalNativeRuntimeInstallReceiptV1> {
        self.runtime_install_receipt.clone()
    }
}

static CANONICAL_NATIVE_STARTUP_AUTHORITY_V1: OnceLock<CanonicalNativeStartupAuthorityV1> =
    OnceLock::new();

/// Install and seal the process-wide canonical-native authority from the exact
/// startup Settings. This must run once after the ordinary runtime overrides
/// are installed and before any [`AppApiState`] is constructed.
pub fn install_canonical_native_startup_authority_v1(
    settings: &Settings,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    let receipt = install_and_seal_canonical_native_runtime_authority_v1(settings)?;
    if let Some(installed) = CANONICAL_NATIVE_STARTUP_AUTHORITY_V1.get() {
        if installed.runtime_install_receipt.identity_sha256() == receipt.identity_sha256()
            && installed.runtime_install_receipt.startup_settings_sha256()
                == receipt.startup_settings_sha256()
        {
            return Ok(());
        }
        return Err(CanonicalNativeDiscoveryRequestErrorV1::RuntimeAuthority(
            "conflicting app startup authority was already installed".to_owned(),
        ));
    }
    CANONICAL_NATIVE_STARTUP_AUTHORITY_V1
        .set(CanonicalNativeStartupAuthorityV1 {
            settings: Arc::new(settings.clone()),
            runtime_install_receipt: Arc::new(receipt),
        })
        .map_err(|_| {
            CanonicalNativeDiscoveryRequestErrorV1::RuntimeAuthority(
                "app startup authority raced with another installer".to_owned(),
            )
        })
}

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
static ACCOUNT_REFRESH_TX: OnceLock<tokio::sync::mpsc::UnboundedSender<()>> = OnceLock::new();

/// Install the global account-refresh trigger. Called from `main.rs`
/// exactly once, right after `AppApiState::new()`. Subsequent calls
/// are silent no-ops.
pub fn install_account_refresh_trigger(tx: tokio::sync::mpsc::UnboundedSender<()>) {
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

// REMOVED 2026-08-09 (dead-code purge, batch D2): the `LAUNCHED_BY_FLUTTER`
// OnceLock and its `install_launched_by_flutter` / `launched_by_flutter`
// accessors. Flutter's `BackendSupervisor` — the only reader of the flag it
// fed into `/healthz` — died in the 2026-06-22 Tauri migration.

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
    /// Identity retained from the verified broker response/request, never current settings.
    /// Internal status provenance; the existing account wire DTO remains unchanged.
    #[serde(skip)]
    pub(crate) source_account_id: i64,
    #[serde(skip)]
    pub(crate) source_environment: crate::app_services::ctrader_live_auth::CTraderEnvironment,
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
pub struct AccountRefreshFailure {
    pub code: String,
    pub detail: String,
    pub observed_at_unix_ms: i64,
}

impl AccountRefreshFailure {
    fn from_error(error: &anyhow::Error) -> Self {
        let code = if error
            .downcast_ref::<neoethos_core::BrokerFinancialTruthErrorV1>()
            .is_some()
        {
            neoethos_core::BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1
        } else {
            "account_refresh_failed"
        };
        Self {
            code: code.to_owned(),
            detail: format!("{error:#}"),
            observed_at_unix_ms: chrono::Utc::now().timestamp_millis(),
        }
    }
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
    /// `None` until exact broker symbol/conversion provenance supports a pips
    /// derivation. Never serialize unavailable pips as a financial zero.
    pub pnl_pips: Option<f64>,
    pub pnl_usd: f64,
    /// Entry (open) price, stop-loss and take-profit as the broker reports
    /// them. Server-provided so the client renders the full row with ZERO
    /// conversion or cross-source merging.
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    /// Volume in LOTS (= base units / contract_size), so the UI shows what
    /// cTrader shows (1.17) instead of raw units (117000). `None` when the
    /// symbol isn't in the metadata table. Parity-correct, server-computed.
    pub volume_lots: Option<f64>,
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
    canonical_native_startup_authority: Option<CanonicalNativeStartupAuthorityV1>,
    /// The only async admission coordinator for CPU-heavy app work. Production
    /// startup installs the process budget before constructing this state; a
    /// state built without that preflight (small router tests) keeps this
    /// `None`, and heavy routes fail closed instead of inventing a broker.
    execution: Option<Arc<crate::app_state::AppExecutionState>>,
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
    /// broadcast channel fired on every accepted account refresh. The
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
    account_refresh_rx: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<()>>>>,
    /// Live autonomous trading engines — one per running portfolio so several
    /// discovered strategies can trade concurrently (each is internally
    /// multi-timeframe). Empty when idle.
    pub live_trading: Arc<std::sync::Mutex<Vec<crate::app_services::live_trading::Handle>>>,
    /// Process-lifetime, account-keyed risk ownership shared by every live
    /// portfolio engine. Strategy workers never own independent account limits.
    pub account_risk: Arc<crate::app_services::account_risk::AccountRiskRegistry>,
}

#[derive(Default)]
pub(crate) struct AppApiInner {
    pub account: Option<AccountSnapshotPayload>,
    pub account_failure: Option<AccountRefreshFailure>,
    account_scope: Option<(i64, CTraderEnvironment)>,
    account_failures: usize,
    pub discovery: EngineSlot,
    pub training: EngineSlot,
    pub canonical_native_research: CanonicalNativeResearchSlotV1,
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

impl AppApiInner {
    fn retain_account_scope(
        &mut self,
        resolved: anyhow::Result<(i64, CTraderEnvironment)>,
    ) -> anyhow::Result<(i64, CTraderEnvironment)> {
        let current = resolved.as_ref().ok().copied();
        if current.is_none() || self.account_scope != current {
            self.account = None;
            self.account_failure = None;
            self.account_failures = 0;
            self.account_scope = current;
        }
        resolved
    }
}

/// In-memory tracking of one engine's lifecycle. `state` is what
/// `/engines/status` returns; `cancel` lets `/engines/{kind}/stop`
/// signal the running job; `summary` preserves the reported status and,
/// for a degraded outcome, its warning/error details.
#[derive(Debug, Clone, Default)]
pub struct EngineSlot {
    pub state: EngineRunState,
    /// Existing process-lease identity; late observations cannot replace a new run.
    lease_token: Option<u64>,
    pub cancel: Option<CancellationFlag>,
    pub summary: String,
    /// F-340 (Feature #14): live discovery/training progress mirrored
    /// from the `JobSnapshot` the ServiceEvent drainer processes.
    /// `stage` is the coarse phase label (e.g. `"search_generations"`),
    /// empty when idle. `percent` is 0.0..=1.0 when measured and `None`
    /// when the engine cannot report a fraction (including input preparation).
    /// `counters` retains the latest observed run census after termination.
    /// Only stage/percent are live-only; installing the next run clears its
    /// predecessor's counters so the UI cannot mix separate searches.
    pub stage: String,
    pub percent: Option<f64>,
    pub counters: Vec<(String, u64)>,
}

/// Native research owns an exact Search cancellation token and its own typed
/// terminal evidence. It must never be represented by the legacy Discovery
/// slot because that would permit the Discovery-to-Training auto-chain.
#[derive(Debug, Clone, Default)]
pub struct CanonicalNativeResearchSlotV1 {
    snapshot: Option<CanonicalNativeResearchSnapshotV1>,
    cancellation: Option<CanonicalNativeCancellationTokenV1>,
    cancellation_requested: bool,
    terminal: Option<CanonicalNativeResearchTerminalSnapshotV1>,
}

impl CanonicalNativeResearchSlotV1 {
    pub fn snapshot(&self) -> Option<&CanonicalNativeResearchSnapshotV1> {
        self.snapshot.as_ref()
    }

    pub const fn cancellation_requested(&self) -> bool {
        self.cancellation_requested
    }

    pub fn terminal(&self) -> Option<&CanonicalNativeResearchTerminalSnapshotV1> {
        self.terminal.as_ref()
    }
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
        let (account_refresh_tx, account_refresh_rx) = tokio::sync::mpsc::unbounded_channel();
        let execution =
            neoethos_core::execution_budget::installed_process_budget().map(|installed| {
                Arc::new(
                    crate::app_state::AppExecutionState::new(
                        installed.broker().clone(),
                        installed.resolved().effective_worker_limit,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "failed to start the process execution admission coordinator: {error}"
                        )
                    }),
                )
            });
        Self {
            inner: Arc::new(RwLock::new(AppApiInner::default())),
            canonical_native_startup_authority: CANONICAL_NATIVE_STARTUP_AUTHORITY_V1
                .get()
                .cloned(),
            execution,
            codex: Arc::new(Mutex::new(None)),
            config_path: Arc::new(current_config_path()),
            account_broadcast,
            account_refresh_tx,
            account_refresh_rx: Arc::new(std::sync::Mutex::new(Some(account_refresh_rx))),
            live_trading: Arc::new(std::sync::Mutex::new(Vec::new())),
            account_risk: Arc::new(crate::app_services::account_risk::AccountRiskRegistry::new()),
        }
    }

    /// Clone the process-lifetime admission owner. `None` means construction
    /// happened before the immutable process budget was installed; callers
    /// must return a startup/admission error instead of creating capacity.
    pub fn execution_state(&self) -> Option<Arc<crate::app_state::AppExecutionState>> {
        self.execution.clone()
    }

    pub(crate) fn canonical_native_startup_authority_v1(
        &self,
    ) -> Option<CanonicalNativeStartupAuthorityV1> {
        self.canonical_native_startup_authority.clone()
    }

    /// Clone of the account-refresh sender, suitable for installing
    /// as the process-wide handle via `install_account_refresh_trigger`.
    /// Called by `main.rs` exactly once at startup.
    pub fn account_refresh_tx_clone(&self) -> tokio::sync::mpsc::UnboundedSender<()> {
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
    pub fn take_account_refresh_rx(&self) -> Option<tokio::sync::mpsc::UnboundedReceiver<()>> {
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
    /// fill it via the scope-checked refresh completion boundary.
    #[cfg(test)]
    pub fn with_seed_account(mut self, snapshot: AccountSnapshotPayload) -> Self {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("seeded test state must not be shared yet")
            .get_mut();
        inner.account_scope = Some((snapshot.source_account_id, snapshot.source_environment));
        inner.account = Some(snapshot);
        self
    }

    /// Read the current account snapshot. `None` means the broker
    /// session hasn't produced one yet — the route turns this into a
    /// `503 Service Unavailable` so the Flutter side can render a
    /// "waiting for broker…" placeholder.
    #[cfg(test)]
    pub async fn account(&self) -> Option<AccountSnapshotPayload> {
        self.inner.read().await.account.clone()
    }

    /// Read the cached snapshot and its latest refresh failure atomically.
    pub async fn account_observation(
        &self,
    ) -> (
        Option<AccountSnapshotPayload>,
        Option<AccountRefreshFailure>,
    ) {
        let inner = self.inner.read().await;
        (inner.account.clone(), inner.account_failure.clone())
    }

    /// Observe only the currently selected account, including its failure state.
    /// Call from the blocking pool: the resolver may read persisted selection.
    /// Resolving under the cache lock orders readers and publishers, but does not
    /// make external settings-file writers part of an atomic account switch.
    pub(crate) fn current_account_observation(
        &self,
        resolve_scope: impl FnOnce() -> anyhow::Result<(i64, CTraderEnvironment)>,
    ) -> anyhow::Result<(
        Option<AccountSnapshotPayload>,
        Option<AccountRefreshFailure>,
    )> {
        let mut inner = self.inner.blocking_write();
        inner.retain_account_scope(resolve_scope())?;
        Ok((inner.account.clone(), inner.account_failure.clone()))
    }

    /// Accept a completed refresh only for its still-current request scope.
    /// Response identity is checked independently before publishing. A late
    /// foreign failure cannot overwrite current failure evidence or clear a
    /// current snapshot. Three consecutive accepted failures retain the existing
    /// stale-cache policy; counters reset when selection changes or refresh works.
    /// Call on the blocking pool, with no await between resolution and mutation.
    pub(crate) fn complete_account_refresh(
        &self,
        request_scope: Option<(i64, CTraderEnvironment)>,
        result: anyhow::Result<AccountSnapshotPayload>,
        resolve_scope: impl FnOnce() -> anyhow::Result<(i64, CTraderEnvironment)>,
    ) -> anyhow::Result<bool> {
        const STALE_THRESHOLD: usize = 3;
        let mut inner = self.inner.blocking_write();
        let current = inner.retain_account_scope(resolve_scope())?;
        if request_scope != Some(current) {
            return Ok(false);
        }
        match result {
            Ok(snapshot) => {
                if (snapshot.source_account_id, snapshot.source_environment) != current {
                    return Ok(false);
                }
                inner.account = Some(snapshot.clone());
                inner.account_failure = None;
                inner.account_failures = 0;
                // Keep the response timestamp, and order push publication with
                // the cache write. No subscribers is an ordinary outcome.
                let _ = self.account_broadcast.send(snapshot);
            }
            Err(error) => {
                inner.account_failure = Some(AccountRefreshFailure::from_error(&error));
                inner.account_failures = inner.account_failures.saturating_add(1);
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %error,
                    consecutive_failures = inner.account_failures,
                    "Current account refresh failed; retained values are not confirmed current."
                );
                if inner.account_failures >= STALE_THRESHOLD {
                    inner.account = None;
                }
            }
        }
        Ok(true)
    }

    #[cfg(test)]
    pub async fn set_account_failure(&self, error: &anyhow::Error) {
        self.inner.write().await.account_failure = Some(AccountRefreshFailure::from_error(error));
    }

    /// Test seeding only; production publishes through `complete_account_refresh`.
    #[cfg(test)]
    pub async fn set_account(&self, snapshot: AccountSnapshotPayload) {
        let scope = (snapshot.source_account_id, snapshot.source_environment);
        let state = self.clone();
        assert!(
            tokio::task::spawn_blocking(move || {
                state.complete_account_refresh(Some(scope), Ok(snapshot), || Ok(scope))
            })
            .await
            .expect("test account publisher")
            .expect("test account scope")
        );
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

    /// Clone of the full symbol_id → name map, for callers that need it off the
    /// async runtime (e.g. the journal reconcile on the blocking pool, so closed
    /// trades store the real pair name instead of `#<id>`).
    pub async fn symbol_catalog_snapshot(&self) -> HashMap<i64, String> {
        self.inner.read().await.symbol_catalog.clone()
    }

    // ─── Engine slot accessors ────────────────────────────────────────

    /// One internally consistent observation for the Discovery status response.
    pub(crate) async fn discovery_observation(&self) -> EngineSlot {
        self.inner.read().await.discovery.clone()
    }

    /// Read the current run state for the given engine.
    pub async fn engine_state(&self, kind: JobKind) -> EngineRunState {
        let inner = self.inner.read().await;
        match kind {
            JobKind::Discovery => inner.discovery.state,
            JobKind::Training => inner.training.state,
            JobKind::CanonicalNativeResearch => inner
                .canonical_native_research
                .snapshot
                .as_ref()
                .map(|snapshot| native_engine_state_v1(snapshot.state()))
                .unwrap_or(EngineRunState::Idle),
        }
    }

    /// Read the latest summary and any degraded-outcome details for the engine.
    pub async fn engine_summary(&self, kind: JobKind) -> String {
        let inner = self.inner.read().await;
        match kind {
            JobKind::Discovery => inner.discovery.summary.clone(),
            JobKind::Training => inner.training.summary.clone(),
            JobKind::CanonicalNativeResearch => inner
                .canonical_native_research
                .snapshot
                .as_ref()
                .map(|snapshot| snapshot.stage().to_owned())
                .unwrap_or_default(),
        }
    }

    /// F-340 (Feature #14): read the live progress triple
    /// `(stage, percent, counters)` for the given engine. Returns
    /// `("", None, [])` before the first run. A terminal engine retains its
    /// observed counters, but has no live stage or percent.
    pub async fn engine_progress(
        &self,
        kind: JobKind,
    ) -> (String, Option<f64>, Vec<(String, u64)>) {
        let inner = self.inner.read().await;
        let slot = match kind {
            JobKind::Discovery => &inner.discovery,
            JobKind::Training => &inner.training,
            JobKind::CanonicalNativeResearch => {
                let Some(snapshot) = inner.canonical_native_research.snapshot.as_ref() else {
                    return (String::new(), None, Vec::new());
                };
                return (
                    snapshot.stage().to_owned(),
                    Some(f64::from(snapshot.percent_basis_points()) / 10_000.0),
                    Vec::new(),
                );
            }
        };
        (slot.stage.clone(), slot.percent, slot.counters.clone())
    }

    /// Mark an engine as Running and remember its cancel flag so a
    /// later `/stop` can signal it. Called by the discovery/training
    /// `start` endpoints right after `start_*_job` returns.
    pub async fn install_engine(&self, kind: JobKind, cancel: CancellationFlag, lease_token: u64) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::CanonicalNativeResearch => return,
        };
        slot.state = EngineRunState::Running;
        slot.lease_token = Some(lease_token);
        slot.cancel = Some(cancel);
        slot.summary = "starting…".to_string();
        slot.stage.clear();
        slot.percent = None;
        slot.counters.clear();
    }

    /// Publish one worker observation atomically, including terminal counters.
    /// A failure synthesized without a report preserves the last actual census;
    /// `install_engine` is the boundary that clears previous-run evidence.
    pub(crate) async fn update_engine_snapshot(
        &self,
        kind: JobKind,
        snapshot: &JobSnapshot,
        lease_token: u64,
    ) {
        let mut inner = self.inner.write().await;
        let slot = match kind {
            JobKind::Discovery => &mut inner.discovery,
            JobKind::Training => &mut inner.training,
            JobKind::CanonicalNativeResearch => return,
        };
        if slot.lease_token != Some(lease_token) {
            return;
        }
        slot.state = EngineRunState::from(snapshot.state);
        let running = slot.state == EngineRunState::Running;
        let summary = if running && !snapshot.progress.message.is_empty() {
            &snapshot.progress.message
        } else {
            &snapshot.report.summary
        };
        if !summary.is_empty() {
            slot.summary.clone_from(summary);
        }
        if slot.state == EngineRunState::Degraded {
            // The wire has one summary field. Keep the producer's refusal
            // reason as well as its completed-research census; never infer a
            // calibration verdict from counts or recast it as a worker crash.
            slot.summary = std::iter::once(summary)
                .chain(&snapshot.report.warnings)
                .chain(&snapshot.report.errors)
                .filter(|detail| !detail.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
        }
        if running || !snapshot.report.counters.is_empty() {
            slot.counters.clone_from(&snapshot.report.counters);
        }
        if running {
            slot.stage.clone_from(&snapshot.progress.stage);
            slot.percent = snapshot
                .progress
                .percent
                .map(f64::from)
                .filter(|value| value.is_finite())
                .map(|value| value.clamp(0.0, 1.0));
        } else {
            slot.cancel = None;
            slot.stage.clear();
            slot.percent = None;
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
            JobKind::CanonicalNativeResearch => {
                let Some(cancel) = inner.canonical_native_research.cancellation.clone() else {
                    return false;
                };
                drop(inner);
                cancel.cancel();
                let mut inner = self.inner.write().await;
                inner.canonical_native_research.cancellation_requested = true;
                return true;
            }
        };
        if let Some(cancel) = &slot.cancel {
            cancel.request();
            true
        } else {
            false
        }
    }

    pub(crate) async fn cancel_canonical_native_research_exact_v1(
        &self,
        expected_lease_token: u64,
    ) -> CanonicalNativeResearchCancellationOutcomeV1 {
        let mut inner = self.inner.write().await;
        let slot = &mut inner.canonical_native_research;
        let Some(snapshot) = slot.snapshot.as_ref() else {
            return CanonicalNativeResearchCancellationOutcomeV1::NotRunning;
        };
        if snapshot.lease_token() != expected_lease_token {
            return CanonicalNativeResearchCancellationOutcomeV1::TokenMismatch;
        }
        let Some(cancellation) = slot.cancellation.as_ref() else {
            return CanonicalNativeResearchCancellationOutcomeV1::NotRunning;
        };
        cancellation.cancel();
        let already_requested = slot.cancellation_requested;
        slot.cancellation_requested = true;
        if already_requested {
            CanonicalNativeResearchCancellationOutcomeV1::AlreadyRequested
        } else {
            CanonicalNativeResearchCancellationOutcomeV1::Requested
        }
    }

    pub async fn install_canonical_native_research_v1(
        &self,
        cancellation: CanonicalNativeCancellationTokenV1,
        snapshot: CanonicalNativeResearchSnapshotV1,
    ) {
        let mut inner = self.inner.write().await;
        inner.canonical_native_research = CanonicalNativeResearchSlotV1 {
            snapshot: Some(snapshot),
            cancellation: Some(cancellation),
            cancellation_requested: false,
            terminal: None,
        };
    }

    pub async fn update_canonical_native_research_v1(
        &self,
        snapshot: CanonicalNativeResearchSnapshotV1,
    ) {
        let mut inner = self.inner.write().await;
        let slot = &mut inner.canonical_native_research;
        if let Some(terminal) = snapshot.terminal().cloned() {
            slot.terminal = Some(terminal);
            slot.cancellation = None;
        }
        slot.snapshot = Some(snapshot);
    }

    pub async fn canonical_native_research_slot_v1(&self) -> CanonicalNativeResearchSlotV1 {
        self.inner.read().await.canonical_native_research.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeResearchCancellationOutcomeV1 {
    Requested,
    AlreadyRequested,
    NotRunning,
    TokenMismatch,
}

fn native_engine_state_v1(state: CanonicalNativeResearchStateV1) -> EngineRunState {
    match state {
        CanonicalNativeResearchStateV1::Queued | CanonicalNativeResearchStateV1::Running => {
            EngineRunState::Running
        }
        CanonicalNativeResearchStateV1::Published => EngineRunState::Succeeded,
        CanonicalNativeResearchStateV1::Failed | CanonicalNativeResearchStateV1::WorkerPanicked => {
            EngineRunState::Failed
        }
        CanonicalNativeResearchStateV1::Cancelled => EngineRunState::Cancelled,
    }
}

impl Default for AppApiState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod account_scope_tests {
    use super::*;

    fn observed(scope: (i64, CTraderEnvironment), balance: f64) -> AccountSnapshotPayload {
        AccountSnapshotPayload {
            source_account_id: scope.0,
            source_environment: scope.1,
            balance,
            equity: balance,
            free_margin: balance,
            used_margin: 0.0,
            currency: "USD".into(),
            fetched_at_unix_ms: 123_456,
            positions: Vec::new(),
        }
    }

    #[tokio::test]
    async fn delayed_foreign_success_and_errors_preserve_current_state_and_failure_count() {
        let old_scope = (11, CTraderEnvironment::Demo);
        for new_scope in [
            (22, CTraderEnvironment::Demo),
            (11, CTraderEnvironment::Live),
        ] {
            for failed in [false, true] {
                let state = AppApiState::new();
                let selection = Arc::new(std::sync::Mutex::new(old_scope));
                let mut events = state.subscribe_account();
                let (release, delayed) = tokio::sync::oneshot::channel::<()>();
                let old_state = state.clone();
                let old_selection = selection.clone();
                let old = tokio::spawn(async move {
                    delayed.await.unwrap();
                    tokio::task::spawn_blocking(move || {
                        for _ in 0..3 {
                            let result = if failed {
                                Err(anyhow::anyhow!("old account failure"))
                            } else {
                                Ok(observed(old_scope, 111.0))
                            };
                            assert!(
                                !old_state
                                    .complete_account_refresh(Some(old_scope), result, || {
                                        Ok(*old_selection.lock().unwrap())
                                    })
                                    .unwrap()
                            );
                        }
                    })
                    .await
                    .unwrap();
                });
                *selection.lock().unwrap() = new_scope;
                let current = state.clone();
                tokio::task::spawn_blocking(move || {
                    assert!(
                        current
                            .complete_account_refresh(
                                Some(new_scope),
                                Ok(observed(new_scope, 222.0)),
                                || Ok(new_scope)
                            )
                            .unwrap()
                    );
                    assert!(
                        current
                            .complete_account_refresh(
                                Some(new_scope),
                                Err(anyhow::anyhow!("current failure")),
                                || Ok(new_scope)
                            )
                            .unwrap()
                    );
                })
                .await
                .unwrap();
                assert_eq!(events.try_recv().unwrap().balance, 222.0);
                release.send(()).unwrap();
                old.await.unwrap();
                let current = state.clone();
                tokio::task::spawn_blocking(move || {
                    let (saved, failure) = current
                        .current_account_observation(|| Ok(new_scope))
                        .unwrap();
                    let saved = saved.unwrap();
                    assert_eq!(
                        (saved.source_account_id, saved.source_environment),
                        new_scope
                    );
                    assert_eq!(saved.balance, 222.0);
                    assert_eq!(saved.fetched_at_unix_ms, 123_456);
                    assert_eq!(failure.unwrap().detail, "current failure");
                    // Old failures must not spend the new account's three-failure budget.
                    for attempt in 2..=3 {
                        assert!(
                            current
                                .complete_account_refresh(
                                    Some(new_scope),
                                    Err(anyhow::anyhow!("current failure")),
                                    || Ok(new_scope)
                                )
                                .unwrap()
                        );
                        let (saved, failure) = current
                            .current_account_observation(|| Ok(new_scope))
                            .unwrap();
                        assert_eq!(saved.is_some(), attempt < 3);
                        assert!(failure.is_some());
                    }
                })
                .await
                .unwrap();
                assert!(matches!(
                    events.try_recv(),
                    Err(broadcast::error::TryRecvError::Empty)
                ));
            }
        }
    }

    #[test]
    fn both_request_and_response_identity_are_required_before_publication() {
        let scope = (22, CTraderEnvironment::Demo);
        let other = (11, CTraderEnvironment::Demo);
        let state = AppApiState::new();
        let mut events = state.subscribe_account();
        state
            .complete_account_refresh(Some(scope), Ok(observed(scope, 222.0)), || Ok(scope))
            .unwrap();
        events.try_recv().unwrap();
        for (requested, response) in [
            (Some(other), scope),
            (None, scope),
            (Some(scope), other),
            (Some(scope), (22, CTraderEnvironment::Live)),
        ] {
            assert!(
                !state
                    .complete_account_refresh(requested, Ok(observed(response, 999.0)), || Ok(
                        scope
                    ))
                    .unwrap()
            );
        }
        assert!(
            !state
                .complete_account_refresh(None, Err(anyhow::anyhow!("unqualified failure")), || Ok(
                    scope
                ))
                .unwrap()
        );
        let (saved, failure) = state.current_account_observation(|| Ok(scope)).unwrap();
        assert_eq!(saved.unwrap().balance, 222.0);
        assert!(failure.is_none());
        assert!(matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn read_rechecks_selection_even_without_a_new_broker_completion() {
        let old = (11, CTraderEnvironment::Demo);
        for current in [
            (22, CTraderEnvironment::Demo),
            (11, CTraderEnvironment::Live),
        ] {
            let state = AppApiState::new();
            state
                .complete_account_refresh(Some(old), Ok(observed(old, 111.0)), || Ok(old))
                .unwrap();
            state
                .complete_account_refresh(Some(old), Err(anyhow::anyhow!("old failure")), || {
                    Ok(old)
                })
                .unwrap();
            let (saved, failure) = state.current_account_observation(|| Ok(current)).unwrap();
            assert!(saved.is_none());
            assert!(failure.is_none());
            assert!(
                !state
                    .complete_account_refresh(Some(old), Ok(observed(old, 999.0)), || Ok(current))
                    .unwrap()
            );
            assert!(
                state
                    .current_account_observation(|| Ok(current))
                    .unwrap()
                    .0
                    .is_none()
            );
        }
    }

    #[test]
    fn unavailable_selection_clears_unqualified_values_for_read_and_completion() {
        let scope = (11, CTraderEnvironment::Demo);
        for on_read in [true, false] {
            let state = AppApiState::new();
            state
                .complete_account_refresh(Some(scope), Ok(observed(scope, 111.0)), || Ok(scope))
                .unwrap();
            let resolve = || Err(anyhow::anyhow!("no unique persisted execution account"));
            let error = if on_read {
                state.current_account_observation(resolve).unwrap_err()
            } else {
                state
                    .complete_account_refresh(Some(scope), Ok(observed(scope, 999.0)), resolve)
                    .unwrap_err()
            };
            assert_eq!(error.to_string(), "no unique persisted execution account");
            let (saved, failure) = state.current_account_observation(|| Ok(scope)).unwrap();
            assert!(saved.is_none(), "unknown is not a zero or an old balance");
            assert!(failure.is_none());
        }
    }

    #[test]
    fn accepted_success_preserves_timestamp_and_resets_current_failure_budget() {
        let scope = (11, CTraderEnvironment::Demo);
        let state = AppApiState::new();
        let mut events = state.subscribe_account();
        for _ in 0..2 {
            state
                .complete_account_refresh(
                    Some(scope),
                    Err(anyhow::anyhow!("auth unavailable")),
                    || Ok(scope),
                )
                .unwrap();
        }
        assert!(
            state
                .complete_account_refresh(Some(scope), Ok(observed(scope, 444.0)), || Ok(scope))
                .unwrap()
        );
        let (saved, failure) = state.current_account_observation(|| Ok(scope)).unwrap();
        assert_eq!(saved.unwrap().fetched_at_unix_ms, 123_456);
        assert!(failure.is_none());
        assert_eq!(events.try_recv().unwrap().fetched_at_unix_ms, 123_456);
        state
            .complete_account_refresh(
                Some(scope),
                Err(anyhow::anyhow!("first failure after success")),
                || Ok(scope),
            )
            .unwrap();
        assert!(
            state
                .current_account_observation(|| Ok(scope))
                .unwrap()
                .0
                .is_some()
        );
    }
}

#[cfg(test)]
mod terminal_census_tests {
    use super::*;
    use crate::app_services::jobs::JobState;

    #[tokio::test]
    async fn degraded_research_retains_reason_and_census_without_success_or_active_stop() {
        let state = AppApiState::new();
        for kind in [JobKind::Discovery, JobKind::Training] {
            state.install_engine(kind, CancellationFlag::new(), 1).await;
            let mut snapshot = JobSnapshot::new(kind);
            snapshot.state = JobState::Degraded;
            snapshot.report.summary = "research saved: 1 report, 0 handoffs".into();
            snapshot.report.warnings = vec![
                "portfolio/handoff refused: no calibration-surviving strategies".into(),
            ];
            snapshot.report.errors = vec!["requested output unavailable".into()];
            snapshot.report.counters = vec![
                ("working_set_saved_results".into(), 1),
                ("working_set_training_handoffs".into(), 0),
                ("working_set_publication_failures".into(), 1),
            ];
            // A repeated terminal observation must not append diagnostics twice.
            for _ in 0..2 {
                state.update_engine_snapshot(kind, &snapshot, 1).await;
                assert_eq!(state.engine_state(kind).await, EngineRunState::Degraded);
                assert_eq!(state.engine_state(kind).await.as_str(), "Degraded");
                assert_eq!(
                    state.engine_summary(kind).await,
                    "research saved: 1 report, 0 handoffs\nportfolio/handoff refused: no calibration-surviving strategies\nrequested output unavailable"
                );
                assert_eq!(
                    state.engine_progress(kind).await,
                    (String::new(), None, snapshot.report.counters.clone())
                );
                assert!(!state.cancel_engine(kind).await);
            }
            state.install_engine(kind, CancellationFlag::new(), 2).await;
            state.update_engine_snapshot(kind, &snapshot, 1).await;
            assert_eq!(state.engine_state(kind).await, EngineRunState::Running);
            assert_eq!(state.engine_summary(kind).await, "starting…");
            assert!(state.engine_progress(kind).await.2.is_empty());
        }
    }

    #[tokio::test]
    async fn delayed_old_run_updates_cannot_replace_new_discovery_or_mix_its_observation() {
        let state = AppApiState::new();
        state
            .install_engine(JobKind::Discovery, CancellationFlag::new(), 1)
            .await;
        let mut old = JobSnapshot::new(JobKind::Discovery);
        old.state = JobState::Succeeded;
        old.report.summary = "old result".into();
        old.report.counters = vec![("walkforward_tested".into(), 200)];
        state
            .update_engine_snapshot(JobKind::Discovery, &old, 1)
            .await;
        let old_observation = state.discovery_observation().await;

        state
            .install_engine(JobKind::Discovery, CancellationFlag::new(), 2)
            .await;
        let mut new = JobSnapshot::new(JobKind::Discovery);
        new.state = JobState::Running;
        new.progress.stage = "search_generations".into();
        new.progress.message = "new run".into();
        new.report.counters = vec![("candidates_evaluated".into(), 12)];
        state
            .update_engine_snapshot(JobKind::Discovery, &new, 2)
            .await;
        for terminal in [JobState::Succeeded, JobState::Failed, JobState::Cancelled] {
            old.state = terminal;
            state
                .update_engine_snapshot(JobKind::Discovery, &old, 1)
                .await;
            let current = state.discovery_observation().await;
            assert_eq!(current.state, EngineRunState::Running);
            assert_eq!(current.summary, "new run");
            assert_eq!(current.stage, "search_generations");
            assert_eq!(current.counters, new.report.counters);
        }
        assert_eq!(old_observation.state, EngineRunState::Succeeded);
        assert_eq!(old_observation.summary, "old result");
        assert_eq!(
            old_observation.counters,
            vec![("walkforward_tested".into(), 200)]
        );
    }

    #[tokio::test]
    async fn terminal_census_survives_every_outcome_until_the_next_same_engine_run() {
        let state = AppApiState::new();
        for outcome in [
            JobState::Succeeded,
            JobState::Failed,
            JobState::Cancelled,
            JobState::Degraded,
        ] {
            state
                .install_engine(JobKind::Discovery, CancellationFlag::new(), 1)
                .await;
            assert!(state.engine_progress(JobKind::Discovery).await.2.is_empty());
            let mut snapshot = JobSnapshot::new(JobKind::Discovery);
            snapshot.state = outcome;
            snapshot.progress.stage = "obsolete_live_stage".into();
            snapshot.progress.percent = Some(0.75);
            snapshot.report.counters = vec![("walkforward_tested".into(), 37)];
            state
                .update_engine_snapshot(JobKind::Discovery, &snapshot, 1)
                .await;
            assert_eq!(
                state.engine_state(JobKind::Discovery).await,
                EngineRunState::from(outcome)
            );
            assert_eq!(
                state.engine_progress(JobKind::Discovery).await,
                (String::new(), None, snapshot.report.counters.clone())
            );
            assert!(!state.cancel_engine(JobKind::Discovery).await);
            state
                .install_engine(JobKind::Training, CancellationFlag::new(), 2)
                .await;
            assert_eq!(
                state.engine_progress(JobKind::Discovery).await.2,
                snapshot.report.counters
            );
        }
    }

    #[tokio::test]
    async fn missing_terminal_report_preserves_observed_counts_without_inventing_completion() {
        let state = AppApiState::new();
        state
            .install_engine(JobKind::Discovery, CancellationFlag::new(), 1)
            .await;
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        snapshot.report.counters = vec![("walkforward_tested".into(), 12)];
        state
            .update_engine_snapshot(JobKind::Discovery, &snapshot, 1)
            .await;
        let mut failed = JobSnapshot::new(JobKind::Discovery);
        failed.state = JobState::Failed;
        failed.report.summary = "channel closed before terminal evidence".into();
        state
            .update_engine_snapshot(JobKind::Discovery, &failed, 1)
            .await;
        assert_eq!(
            state.engine_state(JobKind::Discovery).await,
            EngineRunState::Failed
        );
        assert_eq!(
            state.engine_progress(JobKind::Discovery).await,
            (String::new(), None, snapshot.report.counters)
        );
    }
}

#[cfg(test)]
mod canonical_native_startup_authority_tests {
    use super::*;

    #[test]
    fn app_state_captures_only_an_already_installed_native_startup_authority() {
        let before_install = AppApiState::new();
        assert!(
            before_install
                .canonical_native_startup_authority_v1()
                .is_none()
        );

        let mut settings = Settings::default();
        settings.models.seen_signature_runtime.max_entries = 3_000_000;
        install_canonical_native_startup_authority_v1(&settings).unwrap();

        let after_install = AppApiState::new();
        let installed = after_install
            .canonical_native_startup_authority_v1()
            .expect("startup authority must be captured by newly constructed state");
        let receipt = installed.runtime_install_receipt();
        assert_eq!(
            receipt.startup_settings_sha256(),
            install_and_seal_canonical_native_runtime_authority_v1(&settings)
                .unwrap()
                .startup_settings_sha256()
        );
        assert_eq!(
            receipt.identity_sha256(),
            install_and_seal_canonical_native_runtime_authority_v1(&settings)
                .unwrap()
                .identity_sha256()
        );
        assert_eq!(
            installed
                .settings()
                .models
                .seen_signature_runtime
                .max_entries,
            settings.models.seen_signature_runtime.max_entries
        );
        assert!(
            before_install
                .canonical_native_startup_authority_v1()
                .is_none(),
            "state construction must capture authority rather than ambiently reread it"
        );
    }
}
