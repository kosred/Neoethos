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

use std::collections::HashMap;
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
use crate::app_services::ctrader_messages::ProductionCTraderOpenApiTransport;
use crate::app_services::live_spots::get_tick;
use crate::app_services::pnl::{BrokerPositionPnL, fetch_unrealized_pnl_for_all_positions};
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

/// Map cTrader's numeric `depositAssetId` to a 3-letter ISO code
/// for the dashboard currency badge. The full source of truth is
/// `ProtoOAAssetListReq`, but pulling that registry on every refresh
/// is wasteful when 95% of operators use one of the 8 majors below.
///
/// Returns `"EUR"` as the conservative fallback for unknown ids —
/// most demo / FTMO accounts ARE EUR, and rendering an unfamiliar
/// numeric id in the UI is strictly worse than rendering a slightly-
/// wrong-but-readable label. When this returns the fallback we log
/// the unknown id so #144's follow-up can grow the table.
pub(crate) fn asset_id_to_currency(asset_id: Option<i64>) -> &'static str {
    match asset_id {
        // Sourced from public Spotware OpenAPI samples + the cTrader
        // sandbox catalog. Conservative subset — additions here are
        // safe (purely widens the supported set).
        Some(4) => "GBP",
        Some(5) => "CHF",
        Some(6) => "EUR",
        Some(8) => "USD",
        Some(14) => "JPY",
        Some(23) => "AUD",
        Some(25) => "NZD",
        Some(27) => "CAD",
        Some(36) => "PLN",
        Some(id) => {
            // F-285 fix (2026-05-25): the previous fallback returned
            // "EUR" silently, mislabelling USD/CHF/GBP accounts at the
            // UI. We now return an explicit "UNKNOWN" sentinel and
            // emit a structured warn naming the unknown asset_id.
            // The UI renders "UNKNOWN" as a banner-tagged warning
            // (instead of showing wrong currency) and the operator
            // sees the structured log line with the asset_id to
            // add to this lookup.
            tracing::warn!(
                target: "neoethos_app::bridge",
                asset_id = id,
                "unknown cTrader depositAssetId; emitting UNKNOWN sentinel \
                 (was: silently EUR). Add to asset_id_to_currency() to fix."
            );
            "UNKNOWN"
        }
        None => {
            // F-285: same fix for the missing-asset_id path.
            tracing::warn!(
                target: "neoethos_app::bridge",
                "cTrader account has no depositAssetId; emitting UNKNOWN sentinel"
            );
            "UNKNOWN"
        }
    }
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
    const SYMBOL_REFRESH_INTERVAL: std::time::Duration =
        std::time::Duration::from_secs(86_400);
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
    tokio::task::spawn_blocking(move || {
        crate::app_services::journal_reconcile::reconcile_best_effort(&snapshot_for_journal);
    });

    // Step 3: convert to wire payload. Equity is balance + unrealized
    // PnL — the prop-firm-correct number per the comment in
    // `CTraderTraderSnapshot::unrealized_pnl`.
    let trader = &snapshot.trader;
    let balance = trader.balance;
    let equity = trader.balance + trader.unrealized_pnl;
    let used_margin: f64 = snapshot
        .reconcile
        .positions
        .iter()
        .filter_map(|p| p.used_margin)
        .sum();
    let free_margin = (equity - used_margin).max(0.0);

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

    // #134 — pull broker-authoritative unrealized PnL per position
    // and pass into `position_to_payload`. Broker does the FX
    // conversion server-side so we don't have to chase per-symbol
    // pip-value-in-account-currency conversion. Failure is
    // non-fatal: positions still flow through with pnl_usd = 0.0
    // (the pre-#134 behaviour) and a `warn!` documents the gap so
    // the operator sees in the log that the dashboard's PnL
    // column will be quiet until the next refresh.
    let pnl_by_position: HashMap<i64, BrokerPositionPnL> = if has_positions {
        let open_ids: Vec<i64> = snapshot
            .reconcile
            .positions
            .iter()
            .map(|p| p.position_id)
            .collect();
        let client_id_clone = ctrader.client_id.clone();
        let client_secret_clone = ctrader.client_secret.clone();
        let access_token_for_pnl = match production_ctrader_token_store()
            .load_token_bundle_with_legacy_fallback()
            .ok()
            .flatten()
        {
            Some(tb) => tb.access_token,
            None => {
                // #149 follow-up: this branch early-returns with pnl_usd=0.0
                // for every open position. That used to be a `debug!` which
                // meant the user saw quiet zeroes in the dashboard with no
                // signal anywhere unless RUST_LOG=debug. Promoted to `warn!`
                // and called out the user-visible effect so an operator can
                // tell from the log whether the column is actually $0.00 or
                // just the keyring lookup blanked mid-refresh.
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    "skipped authoritative PnL fetch — token bundle vanished \
                     mid-refresh; dashboard will show pnl_usd=0.0 for all \
                     positions until the next bridge tick recovers it"
                );
                let currency =
                    asset_id_to_currency(snapshot.trader.deposit_asset_id).to_string();
                let positions = snapshot
                    .reconcile
                    .positions
                    .iter()
                    .map(|p| position_to_payload(p, None, &HashMap::new(), &currency))
                    .collect();
                return Ok(AccountSnapshotPayload {
                    balance,
                    equity,
                    free_margin,
                    used_margin: if used_margin.is_sign_negative() && used_margin == 0.0 {
                        0.0
                    } else {
                        used_margin
                    },
                    currency,
                    fetched_at_unix_ms: chrono::Utc::now().timestamp_millis(),
                    positions,
                });
            }
        };
        let endpoint_host = environment.endpoint_host();
        // Both `snapshot.trader.account_id` and `snapshot.reconcile.account_id`
        // exist; they're the same value the broker echoes back on
        // each call. Prefer the trader-side because the
        // ProtoOATraderRes carries it as the canonical i64
        // discriminator; reconcile sometimes elides it on empty
        // result sets.
        let account_id_i64 = snapshot.trader.account_id;
        let pnl_result = tokio::task::spawn_blocking(move || {
            let transport = ProductionCTraderOpenApiTransport::new(endpoint_host);
            fetch_unrealized_pnl_for_all_positions(
                &transport,
                &client_id_clone,
                &client_secret_clone,
                &access_token_for_pnl,
                account_id_i64,
                &open_ids,
            )
        })
        .await;
        match pnl_result {
            Ok(Ok(auth)) => auth.by_position,
            Ok(Err(err)) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    "authoritative PnL fetch failed — positions will report \
                     pnl_usd=0.0 this refresh cycle"
                );
                HashMap::new()
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %join_err,
                    "authoritative PnL blocking task panicked"
                );
                HashMap::new()
            }
        }
    } else {
        HashMap::new()
    };

    // Compute the deposit currency once so every position payload
    // gets the same account_currency string (needed by
    // `compute_pnl_pips` for the A.3 fix) and the snapshot's
    // top-level `currency` field stays in sync.
    let account_currency = asset_id_to_currency(snapshot.trader.deposit_asset_id).to_string();

    let mut positions = Vec::with_capacity(snapshot.reconcile.positions.len());
    for p in &snapshot.reconcile.positions {
        let resolved_name = state.resolve_symbol_name(p.symbol_id).await;
        positions.push(position_to_payload(
            p,
            resolved_name,
            &pnl_by_position,
            &account_currency,
        ));
    }

    Ok(AccountSnapshotPayload {
        balance,
        equity,
        free_margin,
        used_margin: if used_margin.is_sign_negative() && used_margin == 0.0 {
            // serde renders f64 `-0.0` literally as `-0.0`; the Flutter
            // dashboard surfaced that as a janky "-$0.00 used margin"
            // pill. Normalize the sign for any sum-derived zero so the
            // wire shape stays clean.
            0.0
        } else {
            used_margin
        },
        // #144: 8-currency lookup table from the trader payload's
        // depositAssetId. EUR fallback for unknown ids, logged so
        // we can grow the table. Full ProtoOAAssetListReq still
        // a follow-up for the very-long-tail currencies.
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
    pnl_by_position: &HashMap<i64, BrokerPositionPnL>,
    account_currency: &str,
) -> PositionPayload {
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

    // #134 — broker-authoritative net unrealized PnL in the account
    // currency. The pnl module already handles money-digit scaling +
    // FX conversion to the deposit currency, so we just plug it
    // through. Missing rows (broker omitted the position, or the
    // fetch failed earlier) fall back to 0.0 — pre-#134 behaviour.
    let pnl_usd = pnl_by_position
        .get(&p.position_id)
        .map(|b| b.net_unrealized_pnl)
        .unwrap_or(0.0);

    // Derive pnl_pips from the account-currency PnL via the symbol
    // metadata table. **A.3 fix**: route through `SymbolMetadata::
    // account_pnl_to_pips` which folds in the quote→account FX
    // step explicitly. For quote == account_currency this is
    // exact. For base == account we feed the live mid (below)
    // so the helper can do the per-tick FX. For cross pairs
    // (account is neither base nor quote) we don't have an FX
    // rate yet — the helper returns NaN → 0.0 pips here, but the
    // #142 live-tick override below recomputes pips directly from
    // price_diff / pip_size, which is currency-free and works in
    // every account configuration.
    //
    // We pull the live tick **once** here so the helper and the
    // #142 override share the same observation; `get_tick` is a
    // cheap mutex read but doing it twice would be a needless
    // race window if a fresh tick arrived between the calls.
    let tick = get_tick(p.symbol_id);
    let live_mid = tick.as_ref().and_then(|t| t.mid_price());

    let mut pnl_pips = compute_pnl_pips(
        resolved_name.as_deref(),
        pnl_usd,
        p.volume,
        account_currency,
        // No FX registry wired yet — Phase B follow-up will
        // populate this from a `BridgeBrokerFxCache` that
        // subscribes to the symbols needed to bridge quote↔
        // deposit currency. Until then, cross-currency accounts
        // get 0.0 from this code path and rely on the
        // currency-free live-tick override (#142, below).
        None,
        live_mid,
    );

    // #142: if the live_spots streamer has a fresh tick for this
    // position's symbol AND we have an entry price, recompute
    // pnl_pips directly from the live mid-price. The broker-derived
    // pnl_pips above is up to 5 s stale (one bridge refresh
    // interval); the live tick is < 2 s old. We OVERRIDE pnl_pips
    // only — pnl_usd is left as the broker-authoritative number
    // since recomputing it in account currency requires per-symbol
    // FX conversion data we don't track yet. So the UI sees pips
    // update at sub-2 s cadence while the dollar number refreshes
    // on the 5 s bridge tick. Good enough for the live-overlay
    // case; full live USD PnL is a follow-up.
    if let Some(entry_price) = p.price {
        if let Some(tick) = tick.as_ref() {
            // Use the tick's freshness as the gate — > 5 s old and
            // we'd just be replacing one stale number with another.
            let now_ms = chrono::Utc::now().timestamp_millis();
            let freshness_ms = now_ms - tick.received_at_unix_ms;
            if freshness_ms <= 5_000 {
                if let Some(mid) = live_mid {
                    let price_diff = if p.trade_side.eq_ignore_ascii_case("Buy") {
                        mid - entry_price
                    } else {
                        entry_price - mid
                    };
                    // GROUP D remediation (operator directive 2026-05-25):
                    // pip_size via canonical symbol_metadata, falling back
                    // to the legacy JPY heuristic only when metadata is
                    // genuinely absent (preserves backwards-compat for
                    // exotics not yet in the registry).
                    let pip_size = resolved_name
                        .as_deref()
                        .and_then(neoethos_core::symbol_metadata::resolve)
                        .map(|meta| meta.pip_size)
                        .unwrap_or_else(|| {
                            if resolved_name
                                .as_deref()
                                .map(|n| n.to_ascii_uppercase().ends_with("JPY"))
                                .unwrap_or(false)
                            {
                                0.01
                            } else {
                                0.0001
                            }
                        });
                    pnl_pips = price_diff / pip_size;
                }
            }
        }
    }

    // Lots = base units / contract_size — SAME metadata the pips calc uses, so
    // the UI shows cTrader's lots (1.17) not raw units (117000). Computed before
    // the literal because `symbol:` moves `resolved_name`.
    let volume_lots = resolved_name
        .as_deref()
        .and_then(neoethos_core::symbol_metadata::resolve)
        .filter(|m| m.contract_size.is_finite() && m.contract_size > 0.0)
        .map(|m| p.volume / m.contract_size);

    PositionPayload {
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
        pnl_pips,
        pnl_usd,
        entry_price: p.price,
        stop_loss: p.stop_loss,
        take_profit: p.take_profit,
        volume_lots,
    }
}

/// Derive PnL in pips from broker-side net unrealized PnL (already
/// in account currency, broker did the FX conversion) and the
/// position's base-currency volume. Returns 0.0 when:
///   - `pnl_account_ccy` is 0.0 (nothing to convert),
///   - `volume_base_units` is 0.0 (defensive — shouldn't happen but
///     a div-by-zero is unhelpful),
///   - the symbol isn't in the metadata table (use 0.0 as a
///     visible "unknown" rather than NaN which breaks JSON),
///   - the symbol's `contract_size` is missing (defensive),
///   - the account currency is **cross** to both base and quote AND
///     no `quote_to_account_rate` is supplied — we fail loud (0.0)
///     rather than guess; the operator sees the broken column and
///     the live-tick override (#142) still produces correct pips
///     from price_diff/pip_size without any FX dependency.
///
/// **A.3 fix (2026-05-27)**: prior implementation divided by
/// `pip_value_quote * volume_lots`, treating pip-value-in-quote as
/// if it were pip-value-in-account. For a GBP account trading
/// EURUSD that under-reported pips by the USD/GBP factor (~25 %).
/// We now route through `SymbolMetadata::account_pnl_to_pips`,
/// which folds in the FX step explicitly.
fn compute_pnl_pips(
    resolved_name: Option<&str>,
    pnl_account_ccy: f64,
    volume_base_units: f64,
    account_currency: &str,
    quote_to_account_rate: Option<f64>,
    live_price: Option<f64>,
) -> f64 {
    if !pnl_account_ccy.is_finite() || pnl_account_ccy == 0.0 {
        return 0.0;
    }
    if !volume_base_units.is_finite() || volume_base_units <= 0.0 {
        return 0.0;
    }
    let Some(name) = resolved_name else {
        return 0.0;
    };
    let Some(meta) = neoethos_core::symbol_metadata::resolve(name) else {
        return 0.0;
    };
    if !meta.contract_size.is_finite() || meta.contract_size <= 0.0 {
        return 0.0;
    }
    // CTraderPositionSnapshot.volume is in base-currency UNITS
    // (see the `volume_units` comment block above for the wire
    // derivation). Convert to lots before handing off to the
    // unit-conversion helper.
    let lots = volume_base_units / meta.contract_size;
    meta.account_pnl_to_pips(
        pnl_account_ccy,
        lots,
        account_currency,
        quote_to_account_rate,
        live_price,
    )
    .unwrap_or(0.0)
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
        let mut map = HashMap::new();
        map.insert(
            42,
            BrokerPositionPnL {
                position_id: 42,
                gross_unrealized_pnl: 12.5,
                net_unrealized_pnl: 11.3,
                money_digits: 2,
            },
        );
        // EURUSD on a USD account: account == quote, so
        // pip_value_in_account = pip_value_quote = $10/lot. Position
        // base-units = 10,000 → lots = 0.1 → $1/pip.
        // PnL 11.3 USD → 11.3 pips.
        let payload = position_to_payload(&p, Some("EURUSD".to_string()), &map, "USD");
        assert!((payload.pnl_usd - 11.3).abs() < 1e-9);
        assert!((payload.pnl_pips - 11.3).abs() < 0.01);
    }

    #[test]
    fn position_to_payload_zero_when_no_pnl_entry() {
        let p = sample_position();
        let payload =
            position_to_payload(&p, Some("EURUSD".to_string()), &HashMap::new(), "USD");
        assert_eq!(payload.pnl_usd, 0.0);
        assert_eq!(payload.pnl_pips, 0.0);
    }

    #[test]
    fn position_to_payload_zero_pips_when_symbol_not_in_metadata() {
        let p = sample_position();
        let mut map = HashMap::new();
        map.insert(
            42,
            BrokerPositionPnL {
                position_id: 42,
                gross_unrealized_pnl: 5.0,
                net_unrealized_pnl: 5.0,
                money_digits: 2,
            },
        );
        // Resolved name is the placeholder — symbol_metadata::resolve
        // returns None → pnl_pips falls back to 0.0 (visible
        // "unknown", not NaN).
        let payload = position_to_payload(&p, Some("sym#999".to_string()), &map, "USD");
        assert_eq!(payload.pnl_usd, 5.0);
        assert_eq!(payload.pnl_pips, 0.0);
    }

    /// **A.3 regression guard**: GBP account holding 0.1 lot EURUSD.
    /// The broker tells us pnl_usd = £8 (it already did USD→GBP
    /// server-side). Without an FX registry to convert pip values
    /// from USD→GBP, `compute_pnl_pips` must return 0.0 (fail loud)
    /// rather than the legacy ~25%-off approximation. The live-tick
    /// override is the real source of pips for cross-currency
    /// accounts until the FX cache lands in Phase B.
    #[test]
    fn compute_pnl_pips_cross_currency_returns_zero_without_fx_rate() {
        let pips = compute_pnl_pips(
            Some("EURUSD"),
            8.0,         // £8 — broker-converted
            10_000.0,    // base units = 0.1 lot
            "GBP",       // account ccy is neither base (EUR) nor quote (USD)
            None,        // no FX rate → helper returns None → 0.0 here
            Some(1.0840),
        );
        assert_eq!(pips, 0.0, "must fail loud (0.0) when FX rate is unknown");
    }

    /// **A.3 happy path**: account == quote (USD account, EURUSD).
    /// 0.1 lot, 20 USD PnL → exactly 20 pips.
    #[test]
    fn compute_pnl_pips_account_equals_quote_is_exact() {
        let pips = compute_pnl_pips(
            Some("EURUSD"),
            20.0,
            10_000.0,
            "USD",
            None,
            None,
        );
        assert!(
            (pips - 20.0).abs() < 1e-9,
            "expected 20.0 pips, got {pips}"
        );
    }

    /// **A.3 happy path**: account == base (EUR account, EURUSD).
    /// 0.1 lot, broker tells us €9.23 PnL at a live mid of 1.0840:
    /// pip_value_in_account = $10/1.0840 ≈ €9.225 per lot, so per
    /// 0.1 lot that's ≈ €0.9225/pip → 9.23 / 0.9225 ≈ 10 pips.
    #[test]
    fn compute_pnl_pips_account_equals_base_uses_live_price() {
        let pips = compute_pnl_pips(
            Some("EURUSD"),
            9.225,
            10_000.0,
            "EUR",
            None,
            Some(1.0840),
        );
        assert!(
            (pips - 10.0).abs() < 0.05,
            "expected ~10.0 pips, got {pips}"
        );
    }

    #[test]
    fn compute_pnl_pips_handles_zero_volume() {
        assert_eq!(
            compute_pnl_pips(Some("EURUSD"), 10.0, 0.0, "USD", None, None),
            0.0
        );
        assert_eq!(
            compute_pnl_pips(Some("EURUSD"), 10.0, f64::NAN, "USD", None, None),
            0.0
        );
    }

    #[test]
    fn compute_pnl_pips_handles_nonfinite_pnl() {
        assert_eq!(
            compute_pnl_pips(Some("EURUSD"), f64::NAN, 10_000.0, "USD", None, None),
            0.0
        );
        assert_eq!(
            compute_pnl_pips(Some("EURUSD"), f64::INFINITY, 10_000.0, "USD", None, None),
            0.0
        );
    }
}
