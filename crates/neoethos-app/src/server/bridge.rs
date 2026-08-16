//! Live data bridge between the broker integration and the HTTP server.
//!
//! Phase-1 implementation of task #87: a tokio task that polls the
//! cTrader account-runtime endpoint every `REFRESH_INTERVAL` seconds
//! and writes the latest snapshot into [`AppApiState`]. The axum
//! route layer reads from the same `AppApiState`, so the HTTP surface
//! always serves the **most-recent broker-fed numbers** without
//! holding any locks across an outgoing HTTP request.
//!
//! ## Why polling and not push
//!
//! cTrader's Open API supports a streaming `ProtoOAGetAccountInfoRes`
//! event, but wiring that into our existing
//! `ProductionCTraderOpenApiTransport` is a separate piece of work
//! (it shares the same websocket as quote streaming, which lands in
//! Session 2). A 5-second poll is acceptable for the dashboard's
//! balance/equity numbers — those fields move on every trade close,
//! not every tick.
//!
//! ## Credential resolution
//!
//! 1. `broker_persistence::load_broker_settings()` — TOML + embedded
//!    fallback. Source of `client_id`, `client_secret`, account-id,
//!    and `CTraderEnvironment` (demo vs. live).
//! 2. `secure_store::production_ctrader_token_store().load_token_bundle()`
//!    — keyring-stored `access_token`. Empty / missing means the
//!    operator hasn't OAuthed yet; the bridge logs a warning and
//!    keeps retrying (the operator might OAuth at any moment).
//!
//! If either lookup fails the bridge waits one full interval and
//! tries again — no point spamming the cTrader API with calls that
//! will all 401.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use crate::app_services::broker_api::fetch_broker_symbols_blocking;
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

/// Number of consecutive refresh failures before the cached account
/// snapshot is wiped (= `STALE_THRESHOLD * REFRESH_INTERVAL` of
/// continuous broker silence — 15s with the current 3 × 5s tuning).
/// Lower → faster "broker not ready" surface but more flapping on a
/// flaky network; higher → dashboard lies for longer when the token
/// has actually expired. The v0.4.20 symptom that motivated the cache
/// invalidation is documented in `run()` below.
#[allow(dead_code)] // referenced inside the cTrader-gated run() loop.
const STALE_THRESHOLD: usize = 3;

/// Auto-sync `system.account_currency` in config.yaml to the broker's real
/// deposit currency (known 3-letter codes only — never the UNKNOWN sentinel).
///
/// Cheap on the hot path: a process-level memo of the last currency we synced
/// means config.yaml is only READ/WRITTEN when the broker currency actually
/// changes (first snapshot of the process, or an account switch) — not on
/// every 5s refresh. Best-effort: failures log and never affect the snapshot.
fn sync_account_currency_to_config(broker_ccy: &str) {
    static LAST_SYNCED: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

    let ccy = broker_ccy.trim().to_ascii_uppercase();
    if ccy.len() != 3 || ccy == "UNK" {
        return; // UNKNOWN sentinel or malformed — never write a guess to config
    }
    {
        let Ok(mut last) = LAST_SYNCED.lock() else {
            return;
        };
        if last.as_deref() == Some(ccy.as_str()) {
            return; // already synced this currency in this process
        }
        *last = Some(ccy.clone());
    }

    tokio::task::spawn_blocking(move || {
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
        if current == ccy {
            return; // config already correct
        }
        settings.system.account_currency = ccy.clone();
        match settings.save(&path) {
            Ok(()) => tracing::info!(
                target: "neoethos_app::bridge",
                from = %current, to = %ccy,
                "account-currency synced from broker → config.yaml (discovery \
                 cost model + money views now use the real deposit currency)"
            ),
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
    // (force-refresh endpoint + future `OAExecutionEvent` handler)
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
    // Consecutive-failure counter. After 3 failed refreshes (= 15s of
    // continuous error), wipe the cached snapshot so /account/snapshot
    // returns 503 instead of last-known-good numbers. Without this the
    // dashboard would silently lie for hours — the v0.4.20 user-visible
    // symptom was "balance shows €1000 forever even though token is
    // CH_ACCESS_TOKEN_INVALID since 30 minutes ago". One transient blip
    // (1-2 missed ticks) does NOT clear the cache; only sustained failure.
    // The threshold itself lives at module scope (#148) as STALE_THRESHOLD.
    let mut failures: usize = 0;

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
        // auto re-arm check. Each tick of the polling loop (every 5s)
        // we ask the persistence layer "has the 24h cooldown elapsed
        // since the last kill-switch trip?" — when yes, it flips
        // `armed = true` on disk and clears the kill timestamp. Cheap
        // (single file read; only writes on the rare day-cadence
        // re-arm event), and the 5s granularity is way faster than the
        // human-visible "operator notices kill switch came back".
        match tokio::task::spawn_blocking(
            crate::app_services::risky_mode_persistence::auto_re_arm_if_ready,
        )
        .await
        {
            Ok(Ok(true)) => {
                tracing::info!(
                    target: "neoethos_app::server::bridge",
                    "Risky Mode auto re-armed (24h cooldown elapsed)"
                );
            }
            Ok(Ok(false)) => {
                // No state file, or cooldown still in progress, or
                // already armed — all benign. No log.
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    "Risky Mode auto re-arm check failed; will retry next cycle"
                );
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %join_err,
                    "Risky Mode auto re-arm blocking task panicked"
                );
            }
        }

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

        // **2026-05-25 — drain any pending push-triggers** before the
        // refresh so a burst of `OAExecutionEvent`s collapses into a
        // single refresh per polling iteration (idempotent — the
        // refresh reads broker-of-record state, not deltas).
        if let Some(rx) = refresh_rx.as_mut() {
            while let Ok(()) = rx.try_recv() {
                // Drain only; the refresh below covers them all.
            }
        }

        match refresh_once(&state).await {
            Ok(payload) => {
                state.set_account(payload).await;
                failures = 0;
                tracing::debug!(
                    target: "neoethos_app::server::bridge",
                    "/account/snapshot refreshed from cTrader"
                );
            }
            Err(err) => {
                failures = failures.saturating_add(1);
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    consecutive_failures = failures,
                    "cTrader account refresh failed — Flutter dashboard \
                     will keep showing the previous snapshot until the \
                     next interval. Common causes: OAuth token expired, \
                     broker session not yet established, or no network."
                );
                if failures >= STALE_THRESHOLD && state.account().await.is_some() {
                    tracing::warn!(
                        target: "neoethos_app::server::bridge",
                        consecutive_failures = failures,
                        "clearing cached account snapshot — dashboard \
                         will now show 'broker not ready' instead of \
                         stale balance/equity numbers. Re-authenticate \
                         (Broker Setup → Re-authenticate) or correct \
                         the account_id (Settings) to restore the feed."
                    );
                    state.clear_account().await;
                }
            }
        }
        // **2026-05-25 — push-trigger or timer, whichever fires first**.
        // The 5 s ticker is the safety floor; `refresh_rx.recv()` lets
        // a force-refresh button or a future `OAExecutionEvent` push
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
async fn refresh_once(state: &AppApiState) -> anyhow::Result<AccountSnapshotPayload> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl)
        .map_err(anyhow::Error::new)?;

    // Step 1: resolve credentials. `load_broker_settings` and the
    // secure store are both sync filesystem / keyring ops; we run
    // them on a blocking task so the tokio reactor stays free.
    let (settings, token_bundle) = tokio::task::spawn_blocking(|| {
        let s = load_broker_settings();
        let t = production_ctrader_token_store()
            .load_token_bundle_with_legacy_fallback()
            .map_err(|e| anyhow::anyhow!("load_token_bundle failed: {e}"))?;
        // Best-effort OAuth token refresh ahead of expiry (see the fn's
        // doc-comment). Closes the production token-expiry gap left when
        // the legacy TradingSession heartbeat — which used to drive token
        // refresh — was removed in v0.4.36. Non-fatal: on any failure the
        // existing token is kept, so this never regresses the refresh path.
        let t = t.map(|bundle| refresh_ctrader_token_if_needed(&s, bundle));
        Ok::<_, anyhow::Error>((s, t))
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
    let account_target = ctrader
        .accounts
        .first()
        .ok_or_else(|| anyhow::anyhow!("broker_credentials.toml has no cTrader account picked"))?
        .clone();

    let environment = match ctrader.environment {
        // The on-disk enum mirrors the live-auth one but they're
        // independent types so we can't blanket-cast. Explicit
        // match keeps a compile error if either gains a variant.
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            CTraderEnvironment::Demo
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            CTraderEnvironment::Live
        }
    };

    let request = CTraderAccountRuntimeRequest {
        client_id: ctrader.client_id.clone(),
        client_secret: ctrader.client_secret.clone(),
        access_token,
        environment,
        account_id: account_target.account_id,
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
                anyhow::Error::new(
                    neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                        neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
                    ),
                )
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

    // Auto-sync `system.account_currency` in config.yaml to the broker's REAL
    // deposit currency. Live sizing already reads the broker value, but the
    // DISCOVERY cost model + €/£ views read config — a stale value (USD while
    // the account is GBP) makes discovery optimize with wrong costs. Fire-and-
    // forget on the blocking pool; never delays this snapshot.
    sync_account_currency_to_config(&account_currency);

    let mut positions = Vec::with_capacity(snapshot.reconcile.positions.len());
    for p in &snapshot.reconcile.positions {
        let resolved_name = state.resolve_symbol_name(p.symbol_id).await;
        positions.push(position_to_payload(p, resolved_name, pnl_by_position)?);
    }

    Ok(AccountSnapshotPayload {
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

fn position_to_payload(
    p: &CTraderPositionSnapshot,
    resolved_name: Option<String>,
    pnl_by_position: &BTreeMap<i64, CTraderPositionUnrealizedPnL>,
) -> anyhow::Result<PositionPayload> {
    // **2026-05-26 fix v2 (Κωνσταντίνος)**: corrected unit conversion
    // for the Close-Position endpoint. Empirical chain from live trace
    // against cTrader Demo account 47367144, position 262647379:
    //
    //   * cTrader proto wire field `tradeData.volume` is in CENTS of
    //     base currency (1 lot EURUSD = 100,000 EUR × 100 = 10,000,000
    //     wire units).
    //   * `volume_to_units(wire) = wire / 100.0` in
    //     `ctrader_account.rs:885`, so `p.volume` stored in the
    //     snapshot is base-currency UNITS — not cents and not lots.
    //     For a 1.0 standard lot EURUSD: p.volume = 100,000.
    //   * The Close-Position endpoint (`ProtoOAClosePositionReq.volume`)
    //     wants the same unit as `tradeData.volume`, i.e. CENTS.
    //   * Therefore: `volume_units = p.volume * 100`.
    //
    // History:
    //   v1 (this session, earlier): assumed `p.volume` was already in
    //   cents — passed through → still 100× too small.
    //   pre-v1 (the dev's original): assumed `p.volume` was in lots —
    //   computed `lots * 100_000 * 100 = 10^7` → 10^7× too large.
    //   v2 (here): `p.volume * 100` produces the correct wire volume.
    //
    // Verified against the broker's TRADING_BAD_VOLUME error trace:
    //   "Order closeVolume 10000000000 is bigger than position
    //    volume 100000" — broker displays in `wire / 100` units, so a
    //   1.0-lot position shows 100,000 there too. To close it, the
    //   close request must send wire volume = 10,000,000, which is
    //   `snapshot.volume (100_000) * 100`.
    let volume_units = (p.volume * 100.0).round() as i64;

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
