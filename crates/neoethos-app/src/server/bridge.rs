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
use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeRequest, CTraderPositionSnapshot, load_account_runtime,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::secure_store::production_ctrader_token_store;

use super::state::{AccountSnapshotPayload, AppApiState, PositionPayload};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

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
    const STALE_THRESHOLD: usize = 3;
    let mut failures: usize = 0;
    loop {
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
        ticker.tick().await;
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

    let mut positions = Vec::with_capacity(snapshot.reconcile.positions.len());
    for p in &snapshot.reconcile.positions {
        let resolved_name = state.resolve_symbol_name(p.symbol_id).await;
        positions.push(position_to_payload(p, resolved_name));
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
        // The ProtoOATraderRes payload only exposes the *integer*
        // `depositAssetId` (e.g. 6 for EUR, 8 for USD) — resolving
        // that to a 3-letter ISO code needs a follow-up
        // ProtoOAAssetListReq call against the broker's asset
        // registry. Until that endpoint ships, default to "EUR"
        // (the Spotware sandbox + most FTMO challenges fall in that
        // bucket). The previous code was passing the *account_type*
        // enum label ("HEDGED", "NETTED") into the currency field,
        // which surfaced as a wrong currency badge on the Flutter
        // dashboard — fixed here.
        currency: "EUR".to_string(),
        positions,
    })
}

fn position_to_payload(
    p: &CTraderPositionSnapshot,
    resolved_name: Option<String>,
) -> PositionPayload {
    // cTrader feeds `volume` as already-converted lots (f64). The
    // close-position endpoint wants broker volume units (centi-lots).
    // Convert via the standard FX lot_size: 1 lot = 100,000 units;
    // 1 lot in centi-units = 100,000 * 100 = 10_000_000. Non-FX
    // instruments may have other lot_sizes — once we plumb the
    // symbol catalog through here we'll look up the real lot_size
    // per symbol. For the MVP, EURUSD-shaped FX is the common case.
    let volume_units = (p.volume * 100_000.0 * 100.0).round() as i64;
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
        // cTrader's mark-to-market pip and USD PnL are derived
        // server-side from the position's open price + the latest
        // spot — we don't have the spot stream here. Report 0 for
        // now; the spot stream worker (Session 2) will overwrite
        // these with live values. swap + commission ARE available
        // (cTrader feeds them on the reconcile response) but they
        // are cost-of-carry, not PnL — surfacing them in the PnL
        // column would mislead the operator, so they stay 0 here.
        pnl_pips: 0.0,
        pnl_usd: 0.0,
    }
}
