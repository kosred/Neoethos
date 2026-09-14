//! Polls verified cTrader account-runtime responses every five seconds and on
//! explicit refresh triggers. Cache publication and reads recheck the selected
//! account/environment; unavailable or failed observations remain explicit.
//! Snapshot identities come from the verified response and its request endpoint.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use crate::app_services::broker_api::{
    broker_credentials_configured, fetch_broker_symbols_blocking,
};
use crate::app_services::broker_config::BrokerSettingsState;
use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeRequest, CTraderPositionSnapshot, load_account_runtime,
};
use crate::app_services::ctrader_auth::CTraderTokenBundle;
use crate::app_services::ctrader_live_auth::{
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderTokenRefreshRequest,
    ProductionCTraderLiveAuthBackend,
};
use crate::app_services::ctrader_messages::CTraderPositionUnrealizedPnL;
use crate::app_services::secure_store::production_ctrader_token_store;

use super::state::{AccountSnapshotPayload, AppApiState, PositionPayload};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

/// Auto-sync `system.account_currency` in config.yaml to the broker's real
/// deposit currency (known 3-letter codes only — never the UNKNOWN sentinel).
///
/// Only accepted snapshots initiate this best-effort write. Recheck selection
/// inside the blocking task. This is not transactional with other config or
/// selection writers. Read config each time so failed saves and later operator
/// edits are reconsidered; there is no process-lifetime currency memo.
fn sync_account_currency_to_config(broker_ccy: &str, scope: (i64, CTraderEnvironment)) {
    let ccy = broker_ccy.trim().to_ascii_uppercase();
    if ccy.len() != 3 || ccy == "UNK" {
        return; // UNKNOWN sentinel or malformed — never write a guess to config
    }
    tokio::task::spawn_blocking(move || {
        if current_execution_account_scope().ok() != Some(scope) {
            return;
        }
        let path = crate::server::state::current_config_path();
        let mut settings = match neoethos_core::Settings::from_yaml(&path) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "neoethos_app::bridge",
                    error = %e,
                    "account-currency sync: config.yaml not loadable — skipping"
                );
                return;
            }
        };
        let current = settings.system.account_currency.trim().to_ascii_uppercase();
        // Loading config may have blocked while a different account was selected.
        if current_execution_account_scope().ok() != Some(scope) {
            return;
        }
        if current == ccy {
            return; // config already correct
        }
        settings.system.account_currency = ccy.clone();
        match settings.save(&path) {
            Ok(()) => {
                tracing::info!(
                    target: "neoethos_app::bridge",
                    from = %current, to = %ccy,
                    "account-currency synced from broker → config.yaml"
                );
            }
            Err(e) => tracing::warn!(
                target: "neoethos_app::bridge",
                error = %e,
                "account-currency sync: failed to save config.yaml"
            ),
        }
    });
}

/// Spawn the long-running refresh task. Returns immediately; the
/// task lives for the lifetime of the tokio runtime (and therefore
/// the server process).
pub fn spawn(state: AppApiState) {
    tokio::spawn(async move {
        run(state).await;
    });
}

async fn run(state: AppApiState) {
    let mut ticker = tokio::time::interval(REFRESH_INTERVAL);
    // **2026-05-25 — uniform-push doctrine**: alongside the 5 s safety
    // timer, listen on the account-refresh trigger channel. Senders
    // (force-refresh endpoint and broker-operation refresh triggers)
    // ping the channel to demand an immediate refresh — no waiting
    // for the next 5 s tick.
    // Graceful degradation: if a future regression spawns a second
    // bridge, the second receive-take returns `None`. Log and run the
    // bridge in poll-only mode (the 5 s safety timer still works) so
    // the dashboard keeps updating even though the push-trigger path
    // is degraded. This is per the doctrine "log loud, never panic".
    let refresh_rx_opt = state.take_account_refresh_rx();
    if refresh_rx_opt.is_none() {
        tracing::error!(
            target: "neoethos_app::bridge",
            "account_refresh_rx already taken — running bridge in poll-only mode \
             (push refresh trigger disabled). This indicates a duplicate `bridge::spawn` call."
        );
    }
    let mut refresh_rx = refresh_rx_opt;
    // Run an immediate first refresh so the dashboard isn't blank for
    // the first 5 seconds after server start.
    ticker.tick().await;
    // **F-201/F-202 closure (2026-05-25 — operator directive
    // "periodic refresh 24h")**: the symbol-catalog cache used to be
    // lazy-loaded only on first position with `sym#<id>` and then
    // pinned for the lifetime of the process. A broker maintenance
    // window that re-issues symbol IDs (rare but real) would silently
    // mislabel positions until the operator restarted. Now the
    // bridge proactively refreshes the catalog every 24 hours so
    // symbol-ID drift is caught within a day automatically.
    const SYMBOL_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(86_400);
    let mut last_symbol_refresh: Option<std::time::Instant> = None;

    loop {
        // **F-231/F-501/F-630 closure (2026-05-25)**: Risky Mode
        // kill-switch expiry check. Each tick of the polling loop (every 5s)
        // we ask the persistence layer "has the 24h cooldown elapsed
        // since the last kill-switch trip?" — when yes, it clears the persisted
        // kill timestamp. Cheap
        // (single file read; only writes on the rare day-cadence
        // expiry event), and the 5s granularity is more than sufficient for a
        // 24-hour safety window.
        match tokio::task::spawn_blocking(
            crate::app_services::risky_mode_persistence::clear_expired_kill_switch,
        )
        .await
        {
            Ok(Ok(true)) => {
                tracing::info!(
                    target: "neoethos_app::server::bridge",
                    "Risky Mode kill-switch cooldown expired and was cleared"
                );
            }
            Ok(Ok(false)) => {
                // No state file, or cooldown still in progress, or
                // already cleared — all benign. No log.
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    "Risky Mode kill-switch expiry check failed; will retry next cycle"
                );
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %join_err,
                    "Risky Mode kill-switch expiry task panicked"
                );
            }
        }

        // Coalesce refresh triggers even when no broker is configured. The
        // normal wait below still runs, and configuration is checked again on
        // the next tick/push; setup never requires restarting the bridge.
        if let Some(rx) = refresh_rx.as_mut() {
            while let Ok(()) = rx.try_recv() {}
        }

        if broker_refresh_configured(&state, broker_credentials_configured).await {
            // **F-201/F-202**: 24h periodic symbol-catalog refresh.
            // Independent of the account-snapshot refresh because broker
            // catalogs change on a different timescale (rarely vs.
            // every 5s).
            let needs_symbol_refresh = match last_symbol_refresh {
                None => true,
                Some(t) => t.elapsed() >= SYMBOL_REFRESH_INTERVAL,
            };
            if needs_symbol_refresh {
                match tokio::task::spawn_blocking(fetch_broker_symbols_blocking).await {
                    Ok(Ok(bundle)) => {
                        let catalog: HashMap<i64, String> = bundle
                            .symbols
                            .into_iter()
                            .map(|s| (s.symbol_id, s.symbol_name))
                            .collect();
                        let count = catalog.len();
                        state.set_symbol_catalog(catalog).await;
                        last_symbol_refresh = Some(std::time::Instant::now());
                        tracing::info!(
                            target: "neoethos_app::server::bridge",
                            symbol_count = count,
                            "periodic symbol-catalog refresh complete (24h cadence)"
                        );
                    }
                    Ok(Err(err)) => {
                        tracing::warn!(
                            target: "neoethos_app::server::bridge",
                            error = %err,
                            "periodic symbol-catalog refresh failed; will retry next cycle"
                        );
                    }
                    Err(join_err) => {
                        tracing::warn!(
                            target: "neoethos_app::server::bridge",
                            error = %join_err,
                            "periodic symbol-catalog blocking task panicked; will retry"
                        );
                    }
                }
            }

            let mut request_scope = None;
            let result = refresh_once(&state, &mut request_scope).await;
            let currency = result.as_ref().ok().map(|payload| payload.currency.clone());
            let publisher = state.clone();
            let completion = tokio::task::spawn_blocking(move || {
                publisher.complete_account_refresh(
                    request_scope,
                    result,
                    current_execution_account_scope,
                )
            })
            .await
            .map_err(|error| anyhow::anyhow!("account completion task panicked: {error}"))
            .and_then(|result| result);
            match completion {
                Ok(true) => {
                    if let (Some(currency), Some(scope)) = (currency, request_scope) {
                        sync_account_currency_to_config(&currency, scope);
                        tracing::debug!(target: "neoethos_app::server::bridge", "/account/snapshot refreshed from cTrader");
                    }
                }
                Ok(false) => tracing::debug!(
                    target: "neoethos_app::server::bridge",
                    "discarded account refresh from a superseded or unresolved execution scope"
                ),
                Err(error) => tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %error,
                    "Account refresh completion could not be confirmed."
                ),
            }
        } else {
            // A later configured profile must refresh its catalog immediately,
            // not inherit the previous profile's 24-hour timestamp.
            last_symbol_refresh = None;
        }
        // **2026-05-25 — push-trigger or timer, whichever fires first**.
        // The 5 s ticker is the safety floor; `refresh_rx.recv()` lets
        // a force-refresh button or a broker-operation refresh trigger
        // skip the wait. `tokio::select!` ensures both wakeups are
        // honoured without spinning. The drain-loop at the top of the
        // outer loop body collapses any burst of triggers into a
        // single refresh per iteration.
        //
        // If `refresh_rx` is `None` (degraded mode — see the
        // graceful-degradation note at the top of `run`), we fall
        // back to ticker-only — the operator still gets a refresh
        // every 5 s, just without the push acceleration.
        match refresh_rx.as_mut() {
            Some(rx) => {
                tokio::select! {
                    _ = ticker.tick() => {},
                    _ = rx.recv() => {},
                }
            }
            None => {
                ticker.tick().await;
            }
        }
    }
}

/// A local setup check only, not account, token or trading authorization. Keep
/// all configured-account validation inside the existing refresh path.
async fn broker_refresh_configured(
    state: &AppApiState,
    configured: impl FnOnce() -> bool + Send + 'static,
) -> bool {
    let observer = state.clone();
    match tokio::task::spawn_blocking(move || {
        let configured = configured();
        if !configured {
            // A removed/unconfigured broker cannot keep displaying a previously
            // confirmed account. Reuse the same scope-invalidation boundary as
            // account reads; no broker/keyring/transport operation is needed.
            let _ = observer.current_account_observation(|| {
                Err(anyhow::anyhow!("broker credentials are not configured"))
            });
        }
        configured
    })
    .await
    {
        Ok(configured) => configured,
        Err(error) => {
            tracing::warn!(
                target: "neoethos_app::server::bridge",
                error = %error,
                "broker configuration check failed; skipping refresh this cycle"
            );
            false
        }
    }
}

/// Best-effort cTrader OAuth token refresh. If the saved bundle is within
/// the refresh-ahead window (or already expired) and has a `refresh_token`,
/// exchange it for a fresh access token and persist the new bundle to the
/// keyring. On ANY failure the original bundle is returned unchanged, so the
/// caller proceeds exactly as before — a stale token simply fails the next
/// account call as it would have anyway (no regression).
///
/// This closes the production token-expiry gap: before v0.4.36 the legacy
/// `TradingSession` heartbeat refreshed tokens, but it never ran in
/// production. Without this, a long-running server's OAuth token silently
/// expired at the first TTL boundary and every account fetch broke until a
/// manual interactive browser re-auth. Runs blocking (HTTP + keyring I/O) —
/// call only from inside a `spawn_blocking` task.
fn refresh_ctrader_token_if_needed(
    settings: &BrokerSettingsState,
    bundle: CTraderTokenBundle,
) -> CTraderTokenBundle {
    // 30-minute refresh-ahead window: refresh once the token is within half
    // an hour of expiry (or already expired) so an active session never
    // races the boundary mid-request.
    const REFRESH_WINDOW_SECS: i64 = 1800;
    let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => return bundle, // clock before epoch — skip the refresh
    };
    if !bundle.needs_refresh_at(now, REFRESH_WINDOW_SECS) || bundle.refresh_token.is_empty() {
        return bundle;
    }
    let ctrader = &settings.ctrader;
    if ctrader.client_id.is_empty() || ctrader.client_secret.is_empty() {
        return bundle;
    }
    let request = CTraderTokenRefreshRequest {
        client_id: ctrader.client_id.clone(),
        client_secret: ctrader.client_secret.clone(),
        refresh_token: bundle.refresh_token.clone(),
        scope: bundle.scope.clone(),
    };
    let backend = ProductionCTraderLiveAuthBackend;
    match backend.refresh_token_bundle(&request) {
        Ok(fresh) => {
            if let Err(e) = production_ctrader_token_store().save_token_bundle(&fresh) {
                tracing::warn!(
                    target: "neoethos_app::ctrader_auth",
                    error = %e,
                    "refreshed cTrader OAuth token but could not persist it to the keyring; \
                     using the fresh token for this session only"
                );
            } else {
                tracing::info!(
                    target: "neoethos_app::ctrader_auth",
                    "refreshed cTrader OAuth token ahead of expiry and persisted the new bundle"
                );
            }
            fresh
        }
        Err(e) => {
            // 2026-06-10: distinguish "refresh failed but the current token is
            // still valid for a while" (benign — we'll retry next cycle) from
            // "refresh failed AND the token is already expired" (the next broker
            // call WILL 401/403 and the operator must re-auth NOW). The latter
            // is an operational emergency, not a warning.
            if bundle.is_expired_at(now) {
                tracing::error!(
                    target: "neoethos_app::ctrader_auth",
                    error = %e,
                    "cTrader OAuth token is EXPIRED and the refresh failed — account/trading \
                     calls will fail until you re-authenticate. Manual re-auth required immediately."
                );
            } else {
                tracing::warn!(
                    target: "neoethos_app::ctrader_auth",
                    error = %e,
                    "cTrader OAuth token refresh failed; the current token is still valid, \
                     will retry on the next refresh cycle"
                );
            }
            bundle
        }
    }
}

/// Pull saved creds + access token, hit cTrader, return a render-ready
/// snapshot. Reads through `state.symbol_catalog` so positions are
/// labelled with real tickers (`EURUSD`) instead of the legacy
/// `sym#<id>` placeholder. If the catalog is empty (Markets tab never
/// opened), this triggers a one-time lazy fetch so the dashboard
/// shows correct names from the very first refresh.
async fn refresh_once(
    state: &AppApiState,
    request_scope: &mut Option<(i64, CTraderEnvironment)>,
) -> anyhow::Result<AccountSnapshotPayload> {
    // Current account figures come from the authenticated trader/reconcile/PnL
    // responses below. Historical quote-replay certification is not a
    // prerequisite for reading the broker's current balance and position PnL.
    // Step 1: resolve credentials. `load_broker_settings` and the
    // secure store are both sync filesystem / keyring ops; we run
    // them on a blocking task so the tokio reactor stays free.
    let settings = tokio::task::spawn_blocking(load_broker_settings)
        .await
        .map_err(|error| anyhow::anyhow!("blocking settings task panicked: {error}"))?;
    let account_id = execution_account_id(&settings)?;
    let environment = match settings.ctrader.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            CTraderEnvironment::Demo
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            CTraderEnvironment::Live
        }
    };
    // Qualify failures too, before keyring/token refresh or broker work starts.
    *request_scope = Some((account_id.parse::<i64>()?, environment));
    let (settings, token_bundle) = tokio::task::spawn_blocking(move || {
        let t = production_ctrader_token_store()
            .load_token_bundle_with_legacy_fallback()
            .map_err(|e| anyhow::anyhow!("load_token_bundle failed: {e}"))?;
        // Best-effort OAuth token refresh ahead of expiry (see the fn's
        // doc-comment). Closes the production token-expiry gap left when
        // the legacy TradingSession heartbeat — which used to drive token
        // refresh — was removed in v0.4.36. Non-fatal: on any failure the
        // existing token is kept, so this never regresses the refresh path.
        let t = t.map(|bundle| refresh_ctrader_token_if_needed(&settings, bundle));
        Ok::<_, anyhow::Error>((settings, t))
    })
    .await
    .map_err(|e| anyhow::anyhow!("blocking creds task panicked: {e}"))??;

    let access_token = token_bundle
        .ok_or_else(|| {
            anyhow::anyhow!("no saved cTrader OAuth token bundle — operator must sign in")
        })?
        .access_token;

    let ctrader = &settings.ctrader;
    if ctrader.client_id.is_empty() || ctrader.client_secret.is_empty() {
        anyhow::bail!("broker_credentials.toml has no cTrader client_id / client_secret");
    }
    let request = CTraderAccountRuntimeRequest {
        client_id: ctrader.client_id.clone(),
        client_secret: ctrader.client_secret.clone(),
        access_token,
        environment,
        account_id,
        // Pending protection orders not needed for the dashboard's
        // balance/equity summary — saves an extra round-trip.
        return_protection_orders: false,
    };

    // Step 2: the actual cTrader API call. `load_account_runtime`
    // is blocking (synchronous reqwest under the hood), so wrap it.
    let snapshot = tokio::task::spawn_blocking(move || load_account_runtime(&request))
        .await
        .map_err(|e| anyhow::anyhow!("blocking account-runtime task panicked: {e}"))??;

    // Reconcile the trade journal from this fresh snapshot's realized deals.
    // This is the production replacement for the retired legacy TradingSession
    // heartbeat that used to drive journal reconcile (removed with the egui
    // surface in v0.4.36). This account/dashboard endpoint is the live
    // cTrader-account fetch the Flutter UI polls, so reconciling here captures
    // every closing deal on the next refresh — idempotent on `position_id`.
    // Fire-and-forget on the blocking pool so journal disk I/O never delays
    // this response (the journal contract: never blocks the refresh).
    let snapshot_for_journal = snapshot.clone();
    // Thread the symbol_id→name catalog in so closed trades store the real pair
    // name (EURUSD) instead of `#<id>`. Populated from prior cycles once any
    // position/symbol has been seen; empty map falls back to `#<id>`.
    let journal_names = state.symbol_catalog_snapshot().await;
    tokio::task::spawn_blocking(move || {
        crate::app_services::journal_reconcile::reconcile_best_effort(
            &snapshot_for_journal,
            &journal_names,
        );
    });

    // Step 3: convert the broker account snapshot to the wire payload.
    // Equity is calculated only after the exact account-scoped
    // ProtoOAGetPositionUnrealizedPnL response is validated below.
    let trader = &snapshot.trader;
    let balance = trader.balance;
    let used_margin = snapshot.reconcile.positions.iter().try_fold(
        0.0_f64,
        |running_total, position| -> anyhow::Result<f64> {
            let margin = position.used_margin.ok_or_else(|| {
                anyhow::Error::new(neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                    neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
                ))
            })?;
            let total = running_total + margin;
            if !total.is_finite() {
                return Err(anyhow::Error::new(
                    neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                        neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
                    ),
                ));
            }
            Ok(total)
        },
    )?;
    // `equity` and `free_margin` are computed only from that validated
    // position set. A missing response, row, or conversion never becomes zero.

    // Resolve symbol_id → ticker name from the cached catalog. If the
    // catalog is empty *and* we actually have positions to label, do a
    // one-time blocking fetch so the dashboard doesn't show `sym#1`
    // until the operator visits the Markets tab. Empty positions →
    // skip the fetch (no point paying for the catalog if we don't
    // need names).
    let has_positions = !snapshot.reconcile.positions.is_empty();
    if has_positions && state.symbol_catalog_is_empty().await {
        match tokio::task::spawn_blocking(fetch_broker_symbols_blocking).await {
            Ok(Ok(bundle)) => {
                let catalog: HashMap<i64, String> = bundle
                    .symbols
                    .into_iter()
                    .map(|s| (s.symbol_id, s.symbol_name))
                    .collect();
                state.set_symbol_catalog(catalog).await;
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    "lazy symbol-catalog fetch failed — positions will \
                     fall back to `sym#<id>` placeholders this cycle"
                );
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %join_err,
                    "symbol-catalog blocking task panicked"
                );
            }
        }
    }

    // The account-runtime request already fetched and reconciled one exact
    // ProtoOAGetPositionUnrealizedPnLRes against this same open-position set.
    // Reuse those rows so the bridge cannot mix two different broker instants.
    let pnl_by_position = &snapshot.unrealized_pnl_by_position;
    let account_unrealized = snapshot.unrealized_pnl;
    let equity = balance + account_unrealized;
    let free_margin = equity - used_margin;
    if !equity.is_finite() || !free_margin.is_finite() {
        return Err(anyhow::Error::new(
            neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
            ),
        ));
    }

    // Compute the deposit currency once so the snapshot labels the broker's
    // authoritative monetary PnL in the correct account currency.
    let account_currency = snapshot.deposit_asset_name.clone();

    let mut positions = Vec::with_capacity(snapshot.reconcile.positions.len());
    for p in &snapshot.reconcile.positions {
        let resolved_name = state.resolve_symbol_name(p.symbol_id).await;
        positions.push(position_to_payload(p, resolved_name, pnl_by_position)?);
    }

    Ok(AccountSnapshotPayload {
        source_account_id: snapshot.trader.account_id,
        source_environment: snapshot.environment,
        balance,
        equity,
        free_margin,
        used_margin,
        currency: account_currency,
        // Wall-clock at the moment we finished assembling this
        // snapshot. The Flutter Dashboard converts to local time
        // for the "as of HH:MM:SS" freshness badge so the
        // operator can tell at a glance whether the numbers are
        // live or carried over from a stale cycle.
        fetched_at_unix_ms: chrono::Utc::now().timestamp_millis(),
        positions,
    })
}

/// Read only the actual persisted execution selection. No credentials healing,
/// embedded fallback, keyring access or authentication occurs at this boundary.
/// Call on the blocking pool; file writers are not serialized with cache locks.
pub(crate) fn current_execution_account_scope() -> anyhow::Result<(i64, CTraderEnvironment)> {
    let path = neoethos_core::broker_config::credentials_file_path()?;
    let settings = neoethos_core::broker_config::load_from_disk(&path)
        // A TOML error chain may echo credential source lines. Keep only its
        // safe outer file/category context when returning an error to the UI.
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .ok_or_else(|| anyhow::anyhow!("no persisted cTrader account selection is available"))?;
    let account_id = execution_account_id(&settings)?.parse::<i64>()?;
    let environment = match settings.ctrader.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            CTraderEnvironment::Demo
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            CTraderEnvironment::Live
        }
    };
    Ok((account_id, environment))
}

/// The dashboard must describe the account that execution uses, not whichever
/// account happened to be first in the saved OAuth list.
pub(super) fn execution_account_id(settings: &BrokerSettingsState) -> anyhow::Result<String> {
    let mut enabled = settings
        .ctrader
        .accounts
        .iter()
        .filter(|a| a.enabled_for_execution);
    let selected = enabled
        .next()
        .ok_or_else(|| anyhow::anyhow!("no cTrader account enabled for execution"))?;
    anyhow::ensure!(
        enabled.next().is_none(),
        "more than one cTrader account enabled for execution"
    );
    anyhow::ensure!(
        selected.account_id.parse::<i64>().is_ok_and(|id| id > 0),
        "selected cTrader account id is not a positive integer"
    );
    Ok(selected.account_id.clone())
}

fn position_to_payload(
    p: &CTraderPositionSnapshot,
    resolved_name: Option<String>,
    pnl_by_position: &BTreeMap<i64, CTraderPositionUnrealizedPnL>,
) -> anyhow::Result<PositionPayload> {
    // The close endpoint takes the original broker centi-units. The separate
    // display value is base units and can lose integer precision through f64.
    let volume_units = p.volume_raw_centi_units;

    // Broker-authoritative net unrealized PnL in the deposit currency. Missing
    // rows are an integrity failure, not zero profit.
    let pnl_usd = pnl_by_position
        .get(&p.position_id)
        .map(|b| b.net_unrealized_pnl)
        .ok_or_else(|| {
            anyhow::Error::new(neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
            ))
        })?;

    Ok(PositionPayload {
        position_id: p.position_id,
        volume_units,
        // Resolved from the cached cTrader symbol catalog. Falls back
        // to the legacy `sym#<id>` placeholder only when neither
        // `/broker/symbols` nor the bridge's lazy refresh has populated
        // the cache — e.g. when the broker is briefly unreachable for
        // the catalog call but the account-runtime call succeeded.
        symbol: resolved_name.unwrap_or_else(|| format!("sym#{}", p.symbol_id)),
        side: p.trade_side.clone(),
        volume: p.volume,
        // Server-side timestamp from the cTrader fill event. Flutter
        // converts to local time for the "Open since HH:MM" badge in
        // the position row. None on the rare cTrader payload where
        // the fill happened literally microseconds before we polled
        // and the broker hadn't stamped it yet — UI shows "—" in
        // that case rather than guessing.
        open_timestamp_ms: p.open_timestamp_ms,
        // cTrader returns authoritative PnL in deposit currency, not pips.
        // Keep this explicitly unavailable until exact ProtoOASymbol
        // pipPosition plus conversion-leg provenance is connected.
        pnl_pips: None,
        pnl_usd,
        entry_price: p.price,
        stop_loss: p.stop_loss,
        take_profit: p.take_profit,
        // Exact lots require this position's broker `ProtoOASymbol.lotSize`.
        // The account snapshot does not carry that joined row yet, so the UI
        // receives an explicit absence instead of local contract-size math.
        volume_lots: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unconfigured_cycles_skip_refresh_clear_stale_account_and_resume_after_setup() {
        let scope = (22, CTraderEnvironment::Demo);
        let state = AppApiState::new().with_seed_account(AccountSnapshotPayload {
            source_account_id: scope.0,
            source_environment: scope.1,
            balance: 100.0,
            equity: 100.0,
            free_margin: 100.0,
            used_margin: 0.0,
            currency: "USD".into(),
            fetched_at_unix_ms: 123_456,
            positions: Vec::new(),
        });
        state
            .set_account_failure(&anyhow::anyhow!("previous configured refresh failed"))
            .await;
        let mut refresh_calls = 0;
        for configured in [false, false, false, true] {
            if broker_refresh_configured(&state, move || configured).await {
                refresh_calls += 1;
            }
            if !configured {
                let (account, failure) = state.account_observation().await;
                assert!(account.is_none());
                assert!(failure.is_none());
                assert_eq!(refresh_calls, 0);
            }
        }
        assert_eq!(
            refresh_calls, 1,
            "setup re-enables the existing refresh path"
        );
    }

    #[tokio::test]
    async fn configured_check_preserves_the_current_account_observation() {
        let state = AppApiState::new();
        state
            .set_account_failure(&anyhow::anyhow!("configured account is unreachable"))
            .await;
        assert!(broker_refresh_configured(&state, || true).await);
        assert!(state.account_observation().await.1.is_some());
    }

    #[test]
    fn account_snapshot_follows_the_execution_selection_not_list_order() {
        let mut settings = BrokerSettingsState::default();
        let disabled = neoethos_core::broker_config::BrokerAccountTarget {
            account_id: "11".into(),
            label: "unselected".into(),
            enabled_for_execution: false,
        };
        let mut selected = disabled.clone();
        selected.account_id = "22".into();
        selected.enabled_for_execution = true;
        settings.ctrader.accounts = vec![disabled, selected];
        assert_eq!(execution_account_id(&settings).unwrap(), "22");
        settings.ctrader.accounts[0].enabled_for_execution = true;
        assert!(execution_account_id(&settings).is_err());
        settings.ctrader.accounts.clear();
        assert!(execution_account_id(&settings).is_err());
    }

    fn sample_position() -> CTraderPositionSnapshot {
        CTraderPositionSnapshot {
            position_id: 42,
            symbol_id: 1,
            trade_side: "BUY".to_string(),
            // **E.1 fix (2026-05-27)**: `CTraderPositionSnapshot.volume`
            // is in **base-currency UNITS** — not lots. For 0.1 lot
            // EURUSD that's 10,000 EUR (= 0.1 × contract_size 100,000).
            // Previously this fixture stored `0.1` which was lot-shaped
            // and masked the A.3 bug because the broken legacy formula
            // `pnl / (pip_value_quote × volume)` happened to produce the
            // right number when `volume` was passed as lots. Now the
            // fixture is wire-shape-accurate.
            volume_raw_centi_units: 1_000_000,
            volume: 10_000.0,
            price: Some(1.0840),
            stop_loss: None,
            take_profit: None,
            open_timestamp_ms: Some(1_716_422_400_000),
            swap: None,
            commission: None,
            mirroring_commission: None,
            used_margin: None,
            label: None,
            comment: None,
            client_order_id: None,
        }
    }

    #[test]
    fn position_to_payload_uses_broker_pnl_when_present() {
        let p = sample_position();
        let mut map = BTreeMap::new();
        map.insert(
            42,
            CTraderPositionUnrealizedPnL {
                position_id: 42,
                gross_unrealized_pnl: 12.5,
                net_unrealized_pnl: 11.3,
            },
        );
        let payload = position_to_payload(&p, Some("EURUSD".to_string()), &map)
            .expect("a complete broker PnL row is renderable");
        assert!((payload.pnl_usd - 11.3).abs() < 1e-9);
        assert_eq!(
            payload.pnl_pips, None,
            "pips stay unavailable until exact ProtoOASymbol/conversion provenance is wired"
        );
    }

    #[test]
    fn position_to_payload_preserves_exact_raw_close_volume() {
        let mut position = sample_position();
        position.volume_raw_centi_units = 9_007_199_254_740_993;
        position.volume = position.volume_raw_centi_units as f64 / 100.0;
        let pnl = BTreeMap::from([(
            position.position_id,
            CTraderPositionUnrealizedPnL {
                position_id: position.position_id,
                gross_unrealized_pnl: 0.0,
                net_unrealized_pnl: 0.0,
            },
        )]);
        let payload = position_to_payload(&position, Some("EURUSD".to_string()), &pnl)
            .expect("broker position payload");
        assert_eq!(payload.volume_units, 9_007_199_254_740_993);
        assert_eq!(payload.volume, position.volume);
    }

    #[test]
    fn position_to_payload_rejects_missing_broker_pnl_instead_of_zero_filling() {
        let p = sample_position();
        let error = position_to_payload(&p, Some("EURUSD".to_string()), &BTreeMap::new());
        let error = error.expect_err("missing broker PnL must disable the snapshot");
        assert!(
            error
                .to_string()
                .contains(neoethos_core::BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1)
        );
    }
}
