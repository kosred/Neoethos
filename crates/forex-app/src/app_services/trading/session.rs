//! cTrader session lifecycle, OAuth refresh-retry orchestration,
//! account / application auth, and reconcile-request build.
//!
//! Carved out of `trading/mod.rs` (Batch 5 follow-up). This module owns:
//! - Authorization flow start (`start_ctrader_auth`,
//!   `receive_ctrader_authorization_code`, `build_ctrader_token_exchange_request`).
//! - Loopback-callback live-auth driver (`start_ctrader_live_auth`,
//!   `poll_ctrader_live_auth`).
//! - Persisted-session restore / clear (`restore_ctrader_session`,
//!   `clear_ctrader_saved_session`).
//! - Account discovery against `id.ctrader.com` (`discover_ctrader_accounts`).
//! - Adapter connect / disconnect (`start_connect`,
//!   `handle_ctrader_connect_result`, `connect`, `disconnect`).
//! - Reconcile-request build / runtime fetch
//!   (`build_ctrader_account_runtime_request`, `load_ctrader_account_runtime`,
//!   `resolve_ctrader_bootstrap_context`).
//! - Proactive + force token refresh (`ensure_fresh_ctrader_token_bundle`,
//!   `force_refresh_ctrader_token_bundle`, `refresh_ctrader_token_bundle`).
//! - Historical-data bootstrap kickoff (`start_ctrader_bootstrap_batch`).
//! - Adapter selection (`select_adapter`), connection-state reset and the
//!   per-session background-task handles.
//!
//! PRESERVED FIXES (do not change without auditor sign-off):
//! - cTrader OAuth flow uses the documented `GET ?client_secret=...` form per
//!   <https://help.ctrader.com/open-api/account-authentication/>. The
//!   refresh path (`refresh_ctrader_token_bundle`) MUST go through the
//!   `CTraderLiveAuthBackend::refresh_token_bundle` abstraction — we do not
//!   re-implement the URL-query token call here, so log-redaction stays
//!   centralised in the backend.
//! - D11 / Batch 1 audit-fix F3 idempotent retry: when the broker rejects
//!   the access token (`CTRADER_TOKEN_EXPIRED_SENTINEL`),
//!   `force_refresh_ctrader_token_bundle` bypasses the local expiry-window
//!   check and refreshes immediately. The actual reconcile-before-retry
//!   guard lives next to the execute path in `orders.rs`.
//! - Batch 1 OAuth state CSRF: the `state` parameter is generated and
//!   verified by `CTraderAuthSession` (not duplicated here) so this module
//!   merely orchestrates the calls to that abstraction.

use super::{
    AdapterReadinessSnapshot, AppState, BrokerSessionState, BrokerSettingsState,
    CTRADER_DEFAULT_SCOPE, CTRADER_TOKEN_REFRESH_WINDOW_SECS, CTraderAccountDiscoveryRequest,
    CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot, CTraderAccountSummary,
    CTraderAuthSession, CTraderAuthSnapshot, CTraderBootstrapContext,
    CTraderBrokerEnvironment, CTraderEnvironment, CTraderLiveAuthRequest,
    CTraderTokenBundle, CTraderTokenExchangeRequest,
    CTraderTokenRefreshRequest, DataSource, JobKind, JobSnapshot, JobState, ServiceEvent,
    TaskKind, TradingAdapter, TradingAdapterKind, TradingSession, build_default_loopback_config,
    current_unix_seconds, format_ctrader_connect_error, format_ctrader_terminal_info,
    record_app_event, run_ctrader_bootstrap_batch_with_context,
    sync_ctrader_discovered_accounts_into_targets, sync_discovered_accounts_with_targets,
};
use std::sync::Arc;
use std::sync::mpsc::{self, TryRecvError};
use std::time::Instant;

impl TradingSession {
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn configured_adapter(&self) -> TradingAdapterKind {
        self.configured_adapter
    }

    pub fn broker_settings_mut(&mut self) -> &mut BrokerSettingsState {
        &mut self.broker_settings
    }

    pub fn adapter_readiness(&self) -> AdapterReadinessSnapshot {
        let mut readiness = self.broker_settings.readiness(self.configured_adapter);
        if self.connected {
            readiness.session_state = BrokerSessionState::Authenticated;
            readiness.status_line = format!("{} connected.", self.active_adapter_kind().as_str());
            readiness.can_attempt_connect = false;
        }
        readiness
    }

    pub fn can_attempt_connect(&self) -> bool {
        self.adapter_readiness().can_attempt_connect
    }

    pub fn ctrader_auth_snapshot(&self) -> Option<CTraderAuthSnapshot> {
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                if let Some(session) = &self.ctrader_auth {
                    Some(session.snapshot())
                } else {
                    let client_id = self.broker_settings.ctrader.client_id.trim();
                    let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim();
                    if client_id.is_empty() || redirect_uri.is_empty() {
                        None
                    } else {
                        Some(CTraderAuthSession::new(client_id, redirect_uri).snapshot())
                    }
                }
            }
            _ => None,
        }
    }

    pub fn start_ctrader_bootstrap_batch(
        &mut self,
        data_root: std::path::PathBuf,
        symbols: Vec<String>,
        timeframes: Vec<String>,
        years: u32,
        tx: tokio::sync::mpsc::Sender<ServiceEvent>,
    ) -> anyhow::Result<()> {
        self.reap_finished_background_tasks();
        if symbols.is_empty() || timeframes.is_empty() {
            return Err(anyhow::anyhow!(
                "bootstrap requires at least one symbol and one timeframe"
            ));
        }
        if self.background_task_running(TaskKind::Bootstrap) {
            return Err(anyhow::anyhow!("cTrader data bootstrap is already running"));
        }
        let context = self.resolve_ctrader_bootstrap_context()?;
        let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
        snapshot.state = JobState::Running;
        snapshot.progress.stage = "bootstrap_queued".to_string();
        snapshot.progress.message = format!(
            "Queued {} symbols across {} timeframes",
            symbols.len(),
            timeframes.len()
        );

        let tx_clone = tx.clone();
        let _ = tx.blocking_send(ServiceEvent::BootstrapUpdated(snapshot.clone()));

        let handle = std::thread::spawn(move || {
            let mut running_snapshot = snapshot;
            running_snapshot.state = JobState::Running;
            running_snapshot.progress.stage = "bootstrap_running".to_string();
            running_snapshot.progress.message =
                "Fetching and normalizing historical data".to_string();
            let _ =
                tx_clone.blocking_send(ServiceEvent::BootstrapUpdated(running_snapshot.clone()));

            let final_snapshot = match run_ctrader_bootstrap_batch_with_context(
                &context,
                &data_root,
                &symbols,
                &timeframes,
                years,
            ) {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    let mut failed = running_snapshot;
                    failed.state = JobState::Failed;
                    failed.progress.stage = "bootstrap_failed".to_string();
                    failed.progress.message = err.to_string();
                    failed.report.summary = format!("Bootstrap failed: {err}");
                    failed.report.errors.push(err.to_string());
                    failed
                }
            };
            let _ = tx_clone.blocking_send(ServiceEvent::BootstrapUpdated(final_snapshot));
        });
        self.bootstrap_handle = Some(handle);

        Ok(())
    }

    pub fn start_ctrader_auth(&mut self) -> anyhow::Result<CTraderAuthSnapshot> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || redirect_uri.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader auth requires client_id and redirect_uri"
            ));
        }
        let mut session = CTraderAuthSession::new(client_id, redirect_uri);
        session.start_authorization("trading");
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);
        Ok(snapshot)
    }

    pub fn receive_ctrader_authorization_code(
        &mut self,
        code: impl Into<String>,
    ) -> CTraderAuthSnapshot {
        let session = self.ctrader_auth.get_or_insert_with(|| {
            CTraderAuthSession::new(
                self.broker_settings.ctrader.client_id.clone(),
                self.broker_settings.ctrader.redirect_uri.clone(),
            )
        });
        session.receive_authorization_code(code);
        session.snapshot()
    }

    pub fn build_ctrader_token_exchange_request(
        &mut self,
    ) -> anyhow::Result<CTraderTokenExchangeRequest> {
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_secret.is_empty() {
            if let Some(session) = self.ctrader_auth.as_mut() {
                session.mark_failed("cTrader token exchange requires client_secret");
            }
            return Err(anyhow::anyhow!(
                "cTrader token exchange requires client_secret"
            ));
        }
        let session = self
            .ctrader_auth
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("cTrader auth has not started"))?;
        let request = session.build_token_exchange_request(client_secret);
        if !self.broker_settings.ctrader.accounts.is_empty() {
            session.set_accounts(
                self.broker_settings
                    .ctrader
                    .accounts
                    .iter()
                    .map(|account| CTraderAccountSummary {
                        account_id: account.account_id.clone(),
                        broker_title: account.label.clone(),
                        enabled_for_execution: account.enabled_for_execution,
                    })
                    .collect(),
            );
        }
        Ok(request)
    }

    pub fn start_ctrader_live_auth(
        &mut self,
        _tx: tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
    ) -> anyhow::Result<CTraderAuthSnapshot> {
        self.reap_finished_background_tasks();
        if self.ctrader_live_auth_rx.is_some() {
            return Err(anyhow::anyhow!("cTrader live auth is already in progress"));
        }
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || client_secret.is_empty() || redirect_uri.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader live auth requires client_id, client_secret, and redirect_uri"
            ));
        }
        let loopback = build_default_loopback_config(&redirect_uri)?;
        let callback_port = *loopback
            .allowed_ports()
            .first()
            .ok_or_else(|| anyhow::anyhow!("cTrader live auth has no callback ports configured"))?;
        let mut session = CTraderAuthSession::new(client_id.clone(), redirect_uri.clone());
        session.start_authorization(CTRADER_DEFAULT_SCOPE);
        session.mark_listening_for_callback(callback_port);
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);

        let request = CTraderLiveAuthRequest {
            client_id,
            client_secret,
            redirect_uri,
            scope: CTRADER_DEFAULT_SCOPE.to_string(),
            loopback,
        };
        let backend = Arc::clone(&self.ctrader_live_auth_backend);
        let (local_tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = backend.run(request).map_err(|err| err.to_string());
            let _ = local_tx.send(result.clone());
            if let Ok(_auth_result) = result {
                // Background update via ServiceEvent if needed
                // For now just prevent the unused variable warning
            }
        });
        self.ctrader_live_auth_rx = Some(rx);
        Ok(snapshot)
    }

    pub fn poll_ctrader_live_auth(&mut self) -> Option<CTraderAuthSnapshot> {
        let receiver = self.ctrader_live_auth_rx.as_ref()?;
        let outcome = match receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => {
                if let Some(session) = self.ctrader_auth.as_mut() {
                    session.mark_failed(
                        "cTrader live auth worker disconnected before returning a result",
                    );
                    let snapshot = session.snapshot();
                    self.ctrader_live_auth_rx = None;
                    return Some(snapshot);
                }
                self.ctrader_live_auth_rx = None;
                return None;
            }
        };
        self.ctrader_live_auth_rx = None;

        match outcome {
            Ok(result) => {
                {
                    let session = self.ctrader_auth.get_or_insert_with(|| {
                        CTraderAuthSession::new(
                            self.broker_settings.ctrader.client_id.clone(),
                            self.broker_settings.ctrader.redirect_uri.clone(),
                        )
                    });
                    session.mark_listening_for_callback(result.callback_port);
                    session.receive_authorization_code(result.authorization_code);
                    if let Err(err) = self
                        .ctrader_token_store
                        .save_token_bundle(&result.token_bundle)
                    {
                        session.mark_failed(format!("failed to save cTrader token bundle: {err}"));
                        return Some(session.snapshot());
                    }
                    session.restore_from_storage(result.token_bundle);
                }
                match self.discover_ctrader_accounts() {
                    Ok(Some(snapshot)) => Some(snapshot),
                    Ok(None) => self.ctrader_auth.as_ref().map(|session| session.snapshot()),
                    Err(err) => {
                        let session = self.ctrader_auth.as_mut()?;
                        session.mark_failed(format!("cTrader account discovery failed: {err}"));
                        Some(session.snapshot())
                    }
                }
            }
            Err(message) => {
                let session = self.ctrader_auth.get_or_insert_with(|| {
                    CTraderAuthSession::new(
                        self.broker_settings.ctrader.client_id.clone(),
                        self.broker_settings.ctrader.redirect_uri.clone(),
                    )
                });
                session.mark_failed(message);
                Some(session.snapshot())
            }
        }
    }

    pub fn restore_ctrader_session(&mut self) -> anyhow::Result<Option<CTraderAuthSnapshot>> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || redirect_uri.is_empty() {
            return Ok(None);
        }

        let Some(bundle) = self.ctrader_token_store.load_token_bundle()? else {
            self.ctrader_auth = None;
            return Ok(None);
        };

        let mut session = CTraderAuthSession::new(client_id, redirect_uri);
        session.restore_from_storage(bundle);
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);
        Ok(Some(snapshot))
    }

    pub fn clear_ctrader_saved_session(&mut self) -> anyhow::Result<()> {
        self.ctrader_token_store.clear_token_bundle()?;
        self.reset_connection_state();
        self.ctrader_auth = None;
        self.ctrader_live_auth_rx = None;
        Ok(())
    }

    pub fn discover_ctrader_accounts(&mut self) -> anyhow::Result<Option<CTraderAuthSnapshot>> {
        let auth_state = self.ctrader_auth.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cTrader account discovery requires a restored token session")
        })?;
        if !matches!(
            auth_state.snapshot().state,
            crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
                | crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
        ) {
            return Err(anyhow::anyhow!(
                "cTrader account discovery requires a restored token session"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader account discovery requires a stored token bundle",
            )?
            .access_token;
        let request = CTraderAccountDiscoveryRequest {
            client_id: self.broker_settings.ctrader.client_id.clone(),
            client_secret: self.broker_settings.ctrader.client_secret.clone(),
            access_token,
            environment: match self.broker_settings.ctrader.environment {
                CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
                CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
            },
        };
        let result = self
            .ctrader_account_discovery_backend
            .discover_accounts(&request)?;

        let synced_accounts = sync_ctrader_discovered_accounts_into_targets(
            &self.broker_settings.ctrader.accounts,
            &result.accounts,
        );
        self.broker_settings.ctrader.accounts = synced_accounts.clone();
        let session = self.ctrader_auth.as_mut().ok_or_else(|| {
            anyhow::anyhow!("cTrader account discovery requires a restored token session")
        })?;
        session.set_discovered_accounts(sync_discovered_accounts_with_targets(
            &result.accounts,
            &self.broker_settings.ctrader.accounts,
        ));

        Ok(Some(session.snapshot()))
    }

    pub fn start_connect(
        &mut self,
        tx: tokio::sync::mpsc::Sender<ServiceEvent>,
    ) -> anyhow::Result<()> {
        self.reap_finished_background_tasks();
        if self.background_task_running(TaskKind::Connect) {
            return Err(anyhow::anyhow!("connection attempt already in progress"));
        }
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                let request = self.build_ctrader_account_runtime_request()?;
                let backend = Arc::clone(&self.ctrader_account_runtime_backend);
                let handle =
                    std::thread::spawn(move || match backend.load_account_runtime(&request) {
                        Ok(runtime) => {
                            let _ = tx.blocking_send(ServiceEvent::CTraderConnectUpdated(runtime));
                        }
                        Err(err) => {
                            let _ = tx
                                .blocking_send(ServiceEvent::ConnectOutcome(Err(err.to_string())));
                        }
                    });
                self.connect_handle = Some(handle);
            }
            _ => {
                let _ = tx.blocking_send(ServiceEvent::ConnectOutcome(Err(format!(
                    "Adapter {:?} not implemented",
                    self.configured_adapter
                ))));
            }
        }
        Ok(())
    }

    pub fn handle_ctrader_connect_result(
        &mut self,
        state: &mut AppState,
        runtime: CTraderAccountRuntimeSnapshot,
    ) {
        self.connected = true;
        self.ctrader_runtime_refreshed_at = Some(Instant::now());
        self.terminal_info =
            format_ctrader_terminal_info(&runtime.trader, self.selected_ctrader_environment());
        if self.initial_equity.is_none() {
            self.initial_equity = Some(runtime.trader.balance);
        }
        self.day_start_equity = Some(self.day_start_equity.unwrap_or(runtime.trader.balance));
        self.adapter = Some(TradingAdapter::CTrader(runtime));
        self.market_chart_cache = None;
        self.execution_surface_cache = None;
        state.status_msg = "cTrader connected".to_string();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn connect(&mut self, state: &mut AppState) {
        let readiness = self.adapter_readiness();
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                self.reset_connection_state();
                if !readiness.can_attempt_connect {
                    let mut failed_readiness = readiness;
                    failed_readiness.session_state = BrokerSessionState::Failed;
                    state.status_msg = failed_readiness.status_line.clone();
                    record_app_event(
                        "ui_adapter_connect",
                        "FAILED",
                        format!(
                            "{} connect blocked: {}",
                            self.configured_adapter.as_str(),
                            failed_readiness.status_line
                        ),
                    );
                    return;
                }

                match self.load_ctrader_account_runtime() {
                    Ok(runtime) => {
                        self.connected = true;
                        self.terminal_info = format_ctrader_terminal_info(
                            &runtime.trader,
                            self.selected_ctrader_environment(),
                        );
                        if self.initial_equity.is_none() {
                            self.initial_equity = Some(runtime.trader.balance); // Fixed: Initial eq is balance + 0
                        }
                        self.day_start_equity =
                            Some(self.day_start_equity.unwrap_or(runtime.trader.balance));
                        self.adapter = Some(TradingAdapter::CTrader(runtime));
                        self.market_chart_cache = None;
                        self.execution_surface_cache = None;
                        state.status_msg = "cTrader connected".to_string();
                        record_app_event(
                            "ui_adapter_connect",
                            "SUCCESS",
                            "cTrader account runtime connected",
                        );
                    }
                    Err(err) => {
                        self.reset_connection_state();
                        state.status_msg = format_ctrader_connect_error(&err);
                        record_app_event(
                            "ui_adapter_connect",
                            "FAILED",
                            format!("cTrader connect failed: {err}"),
                        );
                    }
                }
            }
            TradingAdapterKind::DxTrade => {
                self.reset_connection_state();
                if !readiness.can_attempt_connect {
                    let mut failed_readiness = readiness;
                    failed_readiness.session_state = BrokerSessionState::Failed;
                    state.status_msg = failed_readiness.status_line.clone();
                    record_app_event(
                        "ui_adapter_connect",
                        "FAILED",
                        format!(
                            "{} connect blocked: {}",
                            self.configured_adapter.as_str(),
                            failed_readiness.status_line
                        ),
                    );
                } else {
                    state.status_msg = format!(
                        "{} credentials ready · live auth is not wired yet",
                        self.configured_adapter.as_str()
                    );
                    record_app_event(
                        "ui_adapter_connect",
                        "DEGRADED",
                        format!(
                            "{} credentials ready but live auth is not wired yet",
                            self.configured_adapter.as_str()
                        ),
                    );
                }
            }
        }
    }

    pub fn disconnect(&mut self, state: &mut AppState) {
        self.reset_connection_state();
        state.status_msg = "Offline".to_string();
        record_app_event("ui_disconnect", "SUCCESS", "UI broker connection closed");
    }

    pub fn select_adapter(&mut self, state: &mut AppState, kind: TradingAdapterKind) {
        let previous = self.active_adapter_kind();
        self.reset_connection_state();
        self.configured_adapter = kind;
        state.status_msg = match state.data_source {
            DataSource::Local => "Local Mode".to_string(),
            DataSource::CTrader => format!("{} selected · disconnected", kind.as_str()),
        };
        if matches!(kind, TradingAdapterKind::CTrader)
            && let Ok(Some(_)) = self.restore_ctrader_session()
        {
            state.status_msg = "cTrader selected · session restored".to_string();
        }
        record_app_event(
            "ui_adapter_select",
            "SUCCESS",
            format!(
                "selected trading adapter {} (previous {})",
                kind.as_str(),
                previous.as_str()
            ),
        );
    }

    pub(super) fn reset_connection_state(&mut self) {
        self.adapter = None;
        self.connected = false;
        self.terminal_info.clear();
        self.market_chart_cache = None;
        self.execution_surface_cache = None;
        self.ctrader_live_spot_cache = None;
        self.ctrader_runtime_refreshed_at = None;
    }

    pub(super) fn selected_ctrader_environment(&self) -> CTraderEnvironment {
        match self.broker_settings.ctrader.environment {
            CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
            CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
        }
    }

    pub(super) fn load_ctrader_account_runtime(
        &mut self,
    ) -> anyhow::Result<CTraderAccountRuntimeSnapshot> {
        let request = self.build_ctrader_account_runtime_request()?;
        self.ctrader_account_runtime_backend
            .load_account_runtime(&request)
    }

    pub(super) fn build_ctrader_account_runtime_request(
        &mut self,
    ) -> anyhow::Result<CTraderAccountRuntimeRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader account runtime requires configured client_id and client_secret"
            ));
        }

        if self.ctrader_auth.is_none() {
            self.restore_ctrader_session()?;
        }

        let auth_state = self
            .ctrader_auth
            .as_ref()
            .map(|session| session.snapshot().state);
        if !matches!(
            auth_state,
            Some(crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage)
                | Some(crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable)
        ) {
            return Err(anyhow::anyhow!(
                "cTrader connect requires a restored token session"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle("cTrader connect requires a stored token bundle")?
            .access_token;

        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader account runtime requires at least one discovered account")
            })?;

        Ok(CTraderAccountRuntimeRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            return_protection_orders: true,
        })
    }

    pub(super) fn resolve_ctrader_bootstrap_context(
        &mut self,
    ) -> anyhow::Result<CTraderBootstrapContext> {
        let request = self.build_ctrader_account_runtime_request()?;
        Ok(CTraderBootstrapContext {
            client_id: request.client_id,
            client_secret: request.client_secret,
            access_token: request.access_token,
            environment: request.environment,
            account_id: request.account_id,
        })
    }

    /// Server-side single-call drill-down: fetch every order tied to a
    /// `positionId` via `ProtoOAOrderListByPositionIdReq` (payload 2183).
    /// Replaces what would otherwise be an N-call client-side scan of
    /// `ProtoOAOrderListReq` results filtered by `position_id`. See
    /// `docs/audits/research/ctrader_api_full_reference.md` Appendix C
    /// item #5 for the rationale (was: 1 reconcile + iterate N pending
    /// orders / 1 ProtoOAOrderListReq + filter the order history
    /// client-side; now: 1 ProtoOAOrderListByPositionIdReq).
    ///
    /// `from_timestamp_ms` / `to_timestamp_ms` are optional per the
    /// proto; when both are `None` the broker returns every order ever
    /// linked to the position. The helper clamps locally to the window
    /// before returning per `fetch_orders_by_position_id_with_transport`.
    pub(super) fn fetch_ctrader_orders_for_position(
        &mut self,
        position_id: i64,
        from_timestamp_ms: Option<i64>,
        to_timestamp_ms: Option<i64>,
    ) -> anyhow::Result<Vec<crate::app_services::ctrader_account::CTraderPendingOrderSnapshot>>
    {
        let runtime_request = self.build_ctrader_account_runtime_request()?;
        let request = crate::app_services::ctrader_history::CTraderPositionLookupRequest {
            client_id: runtime_request.client_id,
            client_secret: runtime_request.client_secret,
            access_token: runtime_request.access_token,
            environment: runtime_request.environment,
            account_id: runtime_request.account_id,
            position_id,
            from_timestamp_ms,
            to_timestamp_ms,
        };
        tracing::debug!(
            target: "forex_app::ctrader_history",
            position_id,
            from_ms = from_timestamp_ms,
            to_ms = to_timestamp_ms,
            "trading::session fetch_ctrader_orders_for_position dispatching ProtoOAOrderListByPositionIdReq"
        );
        let orders = self
            .ctrader_position_order_history_backend
            .fetch_orders_by_position_id(&request)?;
        tracing::debug!(
            target: "forex_app::ctrader_history",
            position_id,
            orders = orders.len(),
            "trading::session fetch_ctrader_orders_for_position returned"
        );
        Ok(orders)
    }

    /// Server-side single-call drill-down: fetch a single order plus
    /// every child deal via `ProtoOAOrderDetailsReq` (payload 2181).
    /// Replaces what would otherwise be a re-scan of the reconcile
    /// snapshot for the matching `order_id` plus a separate deal lookup.
    /// See `docs/audits/research/ctrader_api_full_reference.md` §10
    /// item #5 (per-order detail is more granular than reconcile).
    pub(super) fn fetch_ctrader_order_details(
        &mut self,
        order_id: i64,
    ) -> anyhow::Result<crate::app_services::ctrader_account::CTraderOrderDetailsSnapshot> {
        let runtime_request = self.build_ctrader_account_runtime_request()?;
        let request = crate::app_services::ctrader_history::CTraderOrderDetailsRequest {
            client_id: runtime_request.client_id,
            client_secret: runtime_request.client_secret,
            access_token: runtime_request.access_token,
            environment: runtime_request.environment,
            account_id: runtime_request.account_id,
            order_id,
        };
        tracing::debug!(
            target: "forex_app::ctrader_history",
            order_id,
            "trading::session fetch_ctrader_order_details dispatching ProtoOAOrderDetailsReq"
        );
        let snapshot =
            crate::app_services::ctrader_history::fetch_order_details(&request)?;
        tracing::debug!(
            target: "forex_app::ctrader_history",
            order_id,
            deal_count = snapshot.deals.len(),
            "trading::session fetch_ctrader_order_details returned"
        );
        Ok(snapshot)
    }

    pub(super) fn background_task_running(&self, kind: TaskKind) -> bool {
        match kind {
            TaskKind::Connect => self
                .connect_handle
                .as_ref()
                .is_some_and(|handle| !handle.is_finished()),
            TaskKind::Bootstrap => self
                .bootstrap_handle
                .as_ref()
                .is_some_and(|handle| !handle.is_finished()),
        }
    }

    pub(super) fn reap_finished_background_tasks(&mut self) {
        if self
            .connect_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = self.connect_handle.take()
        {
            let _ = handle.join();
        }
        if self
            .bootstrap_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = self.bootstrap_handle.take()
        {
            let _ = handle.join();
        }
    }

    pub(super) fn selected_ctrader_execution_account_id(&self) -> Option<String> {
        self.broker_settings
            .ctrader
            .accounts
            .iter()
            .find(|account| account.enabled_for_execution)
            .or_else(|| self.broker_settings.ctrader.accounts.first())
            .map(|account| account.account_id.clone())
    }

    pub(super) fn ensure_fresh_ctrader_token_bundle(
        &mut self,
        missing_bundle_message: &str,
    ) -> anyhow::Result<CTraderTokenBundle> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader token refresh requires configured client_id and client_secret"
            ));
        }

        let bundle = self
            .ctrader_token_store
            .load_token_bundle()?
            .ok_or_else(|| anyhow::anyhow!(missing_bundle_message.to_string()))?;
        let now_unix = current_unix_seconds()?;
        if !bundle.needs_refresh_at(now_unix, CTRADER_TOKEN_REFRESH_WINDOW_SECS) {
            return Ok(bundle);
        }

        self.refresh_ctrader_token_bundle(&client_id, &client_secret, &bundle)
    }

    /// D11: force a token refresh regardless of the local expiry timer.
    /// Used when the broker rejects the current `access_token` mid-session
    /// (e.g. server-side revocation, clock skew). Returns the new bundle so
    /// the next execution request rebuilds with a valid token.
    pub(super) fn force_refresh_ctrader_token_bundle(
        &mut self,
    ) -> anyhow::Result<CTraderTokenBundle> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader token refresh requires configured client_id and client_secret"
            ));
        }
        let bundle = self
            .ctrader_token_store
            .load_token_bundle()?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cTrader force-refresh requires a stored token bundle with a refresh token"
                )
            })?;
        self.refresh_ctrader_token_bundle(&client_id, &client_secret, &bundle)
    }

    pub(super) fn refresh_ctrader_token_bundle(
        &mut self,
        client_id: &str,
        client_secret: &str,
        bundle: &CTraderTokenBundle,
    ) -> anyhow::Result<CTraderTokenBundle> {
        let refreshed_bundle =
            self.ctrader_live_auth_backend
                .refresh_token_bundle(&CTraderTokenRefreshRequest {
                    client_id: client_id.to_string(),
                    client_secret: client_secret.to_string(),
                    refresh_token: bundle.refresh_token.clone(),
                    scope: bundle.scope.clone(),
                })?;
        self.ctrader_token_store
            .save_token_bundle(&refreshed_bundle)
            .map_err(|err| {
                anyhow::anyhow!("failed to persist refreshed cTrader token bundle: {err}")
            })?;
        if let Some(session) = self.ctrader_auth.as_mut() {
            session.replace_persisted_token_bundle(refreshed_bundle.clone());
        }
        Ok(refreshed_bundle)
    }
}
