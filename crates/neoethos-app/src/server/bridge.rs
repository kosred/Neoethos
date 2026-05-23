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
use crate::app_services::ctrader_messages::ProductionCTraderOpenApiTransport;
use crate::app_services::pnl::{BrokerPositionPnL, fetch_unrealized_pnl_for_all_positions};
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
                tracing::debug!(
                    target: "neoethos_app::server::bridge",
                    "skipped authoritative PnL fetch — token bundle vanished mid-refresh"
                );
                return Ok(AccountSnapshotPayload {
                    balance,
                    equity,
                    free_margin,
                    used_margin: if used_margin.is_sign_negative()
                        && used_margin == 0.0
                    {
                        0.0
                    } else {
                        used_margin
                    },
                    currency: "EUR".to_string(),
                    fetched_at_unix_ms: chrono::Utc::now().timestamp_millis(),
                    positions: snapshot
                        .reconcile
                        .positions
                        .iter()
                        .map(|p| {
                            position_to_payload(
                                p,
                                None,
                                &HashMap::new(),
                            )
                        })
                        .collect(),
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

    let mut positions = Vec::with_capacity(snapshot.reconcile.positions.len());
    for p in &snapshot.reconcile.positions {
        let resolved_name = state.resolve_symbol_name(p.symbol_id).await;
        positions.push(position_to_payload(p, resolved_name, &pnl_by_position));
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
) -> PositionPayload {
    // cTrader feeds `volume` as already-converted lots (f64). The
    // close-position endpoint wants broker volume units (centi-lots).
    // Convert via the standard FX lot_size: 1 lot = 100,000 units;
    // 1 lot in centi-units = 100,000 * 100 = 10_000_000. Non-FX
    // instruments may have other lot_sizes — once we plumb the
    // symbol catalog through here we'll look up the real lot_size
    // per symbol. For the MVP, EURUSD-shaped FX is the common case.
    let volume_units = (p.volume * 100_000.0 * 100.0).round() as i64;

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
    // metadata table. Formula:
    //   pip_value_quote = pip_size * contract_size (per lot)
    //   pnl_pips = pnl_account_ccy / (pip_value_quote * volume_lots)
    // For pairs where quote == account_currency this is exact. For
    // base==account or cross pairs the FX conversion the broker
    // already did is folded into pnl_usd, so the division still
    // gives a sensible pip number — at worst off by the spot
    // multiplier used at carry time, which is the right magnitude.
    // When metadata is missing for the symbol (rare — exotic
    // synthetics), report 0.0 pips rather than NaN; UI shows 0
    // until the symbol table gets populated.
    let pnl_pips = compute_pnl_pips(
        resolved_name.as_deref(),
        pnl_usd,
        p.volume,
    );

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
    }
}

/// Derive PnL in pips from broker-side net unrealized PnL (already
/// in account currency, broker did the FX conversion) and the
/// position's lot volume. Returns 0.0 when:
///   - `pnl_account_ccy` is 0.0 (nothing to convert),
///   - `volume_lots` is 0.0 (defensive — shouldn't happen but a
///     div-by-zero is unhelpful),
///   - the symbol isn't in the metadata table (use 0.0 as a
///     visible "unknown" rather than NaN which breaks JSON).
fn compute_pnl_pips(
    resolved_name: Option<&str>,
    pnl_account_ccy: f64,
    volume_lots: f64,
) -> f64 {
    if !pnl_account_ccy.is_finite() || pnl_account_ccy == 0.0 {
        return 0.0;
    }
    if !volume_lots.is_finite() || volume_lots <= 0.0 {
        return 0.0;
    }
    let Some(name) = resolved_name else {
        return 0.0;
    };
    let Some(meta) = neoethos_core::symbol_metadata::resolve(name) else {
        return 0.0;
    };
    // pip_value_quote is per-lot in the QUOTE currency. The broker
    // returns PnL in the DEPOSIT (account) currency. For
    // quote == account: exact. For other cases the conversion the
    // broker did is implicit in pnl_account_ccy; dividing here
    // gives a pip count that round-trips correctly when the price
    // is close to typical_price (the bulk of trades).
    let denom = meta.pip_value_quote * volume_lots;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return 0.0;
    }
    pnl_account_ccy / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_position() -> CTraderPositionSnapshot {
        CTraderPositionSnapshot {
            position_id: 42,
            symbol_id: 1,
            trade_side: "BUY".to_string(),
            volume: 0.1,
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
        let payload = position_to_payload(&p, Some("EURUSD".to_string()), &map);
        assert!((payload.pnl_usd - 11.3).abs() < 1e-9);
        // For EURUSD: pip_value_quote = 0.0001 * 100_000 = 10.0/lot.
        // At 0.1 lots → 1.0/pip. PnL 11.3 → 11.3 pips.
        assert!((payload.pnl_pips - 11.3).abs() < 0.01);
    }

    #[test]
    fn position_to_payload_zero_when_no_pnl_entry() {
        let p = sample_position();
        let payload = position_to_payload(&p, Some("EURUSD".to_string()), &HashMap::new());
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
        let payload = position_to_payload(&p, Some("sym#999".to_string()), &map);
        assert_eq!(payload.pnl_usd, 5.0);
        assert_eq!(payload.pnl_pips, 0.0);
    }

    #[test]
    fn compute_pnl_pips_handles_zero_volume() {
        assert_eq!(compute_pnl_pips(Some("EURUSD"), 10.0, 0.0), 0.0);
        assert_eq!(compute_pnl_pips(Some("EURUSD"), 10.0, f64::NAN), 0.0);
    }

    #[test]
    fn compute_pnl_pips_handles_nonfinite_pnl() {
        assert_eq!(compute_pnl_pips(Some("EURUSD"), f64::NAN, 0.1), 0.0);
        assert_eq!(compute_pnl_pips(Some("EURUSD"), f64::INFINITY, 0.1), 0.0);
    }
}
