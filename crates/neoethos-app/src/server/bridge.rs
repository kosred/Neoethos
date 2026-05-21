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

use std::time::Duration;

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
    loop {
        match refresh_once().await {
            Ok(payload) => {
                state.set_account(payload).await;
                tracing::debug!(
                    target: "neoethos_app::server::bridge",
                    "/account/snapshot refreshed from cTrader"
                );
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::server::bridge",
                    error = %err,
                    "cTrader account refresh failed — Flutter dashboard \
                     will keep showing the previous snapshot until the \
                     next interval. Common causes: OAuth token expired, \
                     broker session not yet established, or no network."
                );
            }
        }
        ticker.tick().await;
    }
}

/// Pull saved creds + access token, hit cTrader, return a render-ready
/// snapshot. Pure-async, no shared state — the caller is responsible
/// for writing the result into [`AppApiState`].
async fn refresh_once() -> anyhow::Result<AccountSnapshotPayload> {
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
        .ok_or_else(|| anyhow::anyhow!("no saved cTrader OAuth token bundle — operator must sign in"))?
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
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
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

    let positions = snapshot
        .reconcile
        .positions
        .iter()
        .map(position_to_payload)
        .collect();

    Ok(AccountSnapshotPayload {
        balance,
        equity,
        free_margin,
        used_margin,
        // cTrader doesn't report the account currency in the trader
        // snapshot — it's per-account static config we'd need a
        // separate `/symbols-for-account` call to discover. For now,
        // hard-code EUR (most FTMO challenges) and revisit once the
        // symbol-metadata fetch lands.
        currency: trader.account_type.clone().unwrap_or_else(|| "EUR".to_string()),
        positions,
    })
}

fn position_to_payload(p: &CTraderPositionSnapshot) -> PositionPayload {
    PositionPayload {
        // The cTrader feed gives us numeric symbol IDs, not names —
        // resolving id→ticker needs the symbol-list endpoint which
        // is its own session of work. For the dashboard MVP we stamp
        // the id with a `sym#` prefix so the operator at least sees
        // *which* symbol; a follow-up will pipe the resolved ticker
        // (we already cache the map in app_services::ctrader_symbols
        // for the egui side — that wiring just needs to be exposed).
        symbol: format!("sym#{}", p.symbol_id),
        side: p.trade_side.clone(),
        volume: p.volume,
        // cTrader's mark-to-market pip and USD PnL are derived
        // server-side from the position's open price + the latest
        // spot — we don't have the spot stream here. Report 0 for
        // now; the spot stream worker (Session 2) will overwrite
        // these with live values.
        pnl_pips: 0.0,
        pnl_usd: 0.0,
    }
}
