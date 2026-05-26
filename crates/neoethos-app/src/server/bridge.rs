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
fn asset_id_to_currency(asset_id: Option<i64>) -> &'static str {
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
                return Ok(AccountSnapshotPayload {
                    balance,
                    equity,
                    free_margin,
                    used_margin: if used_margin.is_sign_negative() && used_margin == 0.0 {
                        0.0
                    } else {
                        used_margin
                    },
                    currency: asset_id_to_currency(snapshot.trader.deposit_asset_id).to_string(),
                    fetched_at_unix_ms: chrono::Utc::now().timestamp_millis(),
                    positions: snapshot
                        .reconcile
                        .positions
                        .iter()
                        .map(|p| position_to_payload(p, None, &HashMap::new()))
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
        // #144: 8-currency lookup table from the trader payload's
        // depositAssetId. EUR fallback for unknown ids, logged so
        // we can grow the table. Full ProtoOAAssetListReq still
        // a follow-up for the very-long-tail currencies.
        currency: asset_id_to_currency(snapshot.trader.deposit_asset_id).to_string(),
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
    let mut pnl_pips = compute_pnl_pips(resolved_name.as_deref(), pnl_usd, p.volume);

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
        if let Some(tick) = get_tick(p.symbol_id) {
            // Use the tick's freshness as the gate — > 5 s old and
            // we'd just be replacing one stale number with another.
            let now_ms = chrono::Utc::now().timestamp_millis();
            let freshness_ms = now_ms - tick.received_at_unix_ms;
            if freshness_ms <= 5_000 {
                if let Some(mid) = tick.mid_price() {
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
fn compute_pnl_pips(resolved_name: Option<&str>, pnl_account_ccy: f64, volume_lots: f64) -> f64 {
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
