//! Long-running cTrader spot-stream task (#137).
//!
//! Owns a single WebSocket connection to the cTrader streaming
//! endpoint, authenticates, subscribes to N symbols at once, and
//! reads incoming `ProtoOASpotEvent` payloads in a hot loop —
//! routing each one to the shared `live_spots` cache.
//!
//! Re-uses the existing `parse_spot_event_loose` parser
//! (added here because the in-tree one is strict about
//! `expected_symbol_id`, which can't be predicted for a
//! multi-symbol subscription) and the existing connect/auth
//! message builders.
//!
//! ## Design
//!
//! - **One blocking thread** holds the tungstenite socket.
//!   `tokio::task::spawn_blocking` so the read loop doesn't
//!   starve other tokio tasks.
//! - **Outer reconnect loop** in the async parent waits 5 s on
//!   error before re-entering the blocking section. cTrader
//!   does drop streaming sessions periodically (token expiry,
//!   network blips, planned maintenance), so this is expected
//!   behaviour, not an exception.
//! - **Symbol list discovered once** at startup from a hardcoded
//!   forex-majors whitelist + lookup against `/broker/symbols`'s
//!   underlying loader to translate names → numeric IDs. We
//!   subscribe to all of them at once via a single
//!   `ProtoOASubscribeSpotsReq`.
//!
//! ## Limitations (deferred to phase 2)
//!
//! - Symbol list is static at startup. When the user opens a
//!   chart for a symbol we didn't pre-subscribe to, that chart's
//!   "live" price won't update via this stream until a restart.
//!   The chart's on-demand `/chart` endpoint (broker-API pull via
//!   `broker_api::fetch_recent_chart_bars_blocking`) still serves
//!   fresh bars in the meantime.
//! - No heartbeat send. cTrader's docs say streaming clients
//!   should heartbeat every ~30 s; today's flow just reads
//!   incoming PING and replies PONG, which keeps the connection
//!   alive in practice but isn't perfectly spec-conformant.

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};

/// F-338 (Feature #12): monotonically-increasing "which streamer
/// generation is current" counter. Every spawned streamer captures the
/// value at spawn time (`my_gen`); when [`restart_streamer`] bumps it,
/// the in-flight read loop notices `STREAM_GENERATION != my_gen` on its
/// next ~5 s read-timeout tick, closes its socket, and self-terminates
/// — while a freshly-spawned streamer (carrying the new generation)
/// takes over with the updated watchlist. This lets a Market Watch edit
/// re-subscribe the live stream within ~5 s with no app restart.
static STREAM_GENERATION: AtomicU64 = AtomicU64::new(0);

use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_HEARTBEAT_PAYLOAD_TYPE, CTRADER_OA_MARGIN_CALL_TRIGGER_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_MARGIN_CALL_UPDATE_EVENT_PAYLOAD_TYPE, CTRADER_OA_MARGIN_CHANGED_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE, CTRADER_OA_TRADER_UPDATE_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_TRAILING_SL_CHANGED_EVENT_PAYLOAD_TYPE, build_account_auth_request,
    build_application_auth_request, build_subscribe_spots_request, parse_ctrader_error_payload,
    parse_open_api_envelope,
};
use crate::app_services::live_spots;

/// Forex majors we subscribe to by default. Names are matched
/// case-insensitively against the broker's symbol list at
/// startup to recover their numeric IDs. Picked to cover the
/// 80% case for retail forex trading; bigger lists can grow
/// here without touching the streamer logic.
pub const DEFAULT_STREAMED_SYMBOLS: &[&str] = &[
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP",
];

type CTraderSocket = WebSocket<MaybeTlsStream<TcpStream>>;

/// Inputs the streamer needs at connection time. Parameterised
/// here (rather than reading from env at startup) so tests can
/// inject a stub and the spawn site can resolve creds + symbol
/// IDs once and pass them down.
#[derive(Debug, Clone)]
pub struct LiveSpotsStreamerConfig {
    pub endpoint_host: String,
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub account_id: i64,
    /// Pre-resolved `(symbol_id, symbol_name, digits)` rows.
    /// `digits` is the broker's price-scaling factor — we need
    /// it to convert the raw integer bid/ask back to a floating
    /// price before caching.
    pub symbols: Vec<StreamedSymbol>,
}

#[derive(Debug, Clone)]
pub struct StreamedSymbol {
    pub symbol_id: i64,
    pub symbol_name: String,
    pub digits: i32,
}

/// Best-effort wiring helper. Calls the existing broker symbol
/// list endpoint to translate the `DEFAULT_STREAMED_SYMBOLS`
/// names to numeric IDs, then spawns the streamer. Returns
/// `true` when the streamer was spawned, `false` when something
/// failed (creds missing, token expired, etc.) — caller logs and
/// moves on. The HTTP server still comes up either way; the
/// `/live/spots` endpoint just returns an empty list until the
/// streamer eventually connects.
///
/// `digits` is hardcoded by symbol-name suffix because the
/// existing `CTraderLightSymbolInfo` doesn't carry it (a proper
/// ProtoOASymbolByIdReq would add a round-trip we don't need
/// for forex majors). JPY pairs → 3 digits; everything else → 5.
pub fn try_spawn_with_defaults_blocking() -> bool {
    use crate::app_services::broker_api::fetch_broker_symbols_blocking;
    use crate::app_services::broker_persistence::load_broker_settings;
    use crate::app_services::secure_store::production_ctrader_token_store;

    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    if ct.client_id.is_empty() || ct.client_secret.is_empty() {
        tracing::warn!(
            target: "neoethos_app::live_spots_streamer",
            "skipping spawn — broker credentials are empty"
        );
        return false;
    }
    let token_store = production_ctrader_token_store();
    let token_bundle = match token_store.load_token_bundle_with_legacy_fallback() {
        Ok(Some(b)) => b,
        Ok(None) => {
            tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                "skipping spawn — no token bundle in keyring"
            );
            return false;
        }
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                error = %err,
                "skipping spawn — failed to load token bundle"
            );
            return false;
        }
    };
    let primary_account = ct
        .accounts
        .iter()
        .find(|a| a.enabled_for_execution)
        .or_else(|| ct.accounts.first());
    let Some(account_row) = primary_account else {
        tracing::warn!(
            target: "neoethos_app::live_spots_streamer",
            "skipping spawn — no cTrader account configured"
        );
        return false;
    };
    let account_id: i64 = match account_row.account_id.parse() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                account_id = %account_row.account_id,
                "skipping spawn — account_id not numeric"
            );
            return false;
        }
    };

    // Resolve symbol names → ids by hitting the broker once. The
    // call also confirms creds + token still work; if it fails,
    // we bail cleanly rather than spawn a streamer that will just
    // loop on auth errors.
    let bundle = match fetch_broker_symbols_blocking() {
        Ok(b) => b,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                error = %err,
                "skipping spawn — could not list broker symbols"
            );
            return false;
        }
    };

    // F-338: subscribe to the operator's Market Watch set (config
    // `system.watchlist`); fall back to the 8 majors when it's unset.
    let watchlist: Vec<String> =
        neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
            .map(|s| s.system.watchlist)
            .unwrap_or_default();
    let want_symbols: Vec<String> = if watchlist.is_empty() {
        DEFAULT_STREAMED_SYMBOLS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        watchlist
    };

    let mut resolved: Vec<StreamedSymbol> = Vec::new();
    for want in &want_symbols {
        if let Some(s) = bundle
            .symbols
            .iter()
            .find(|s| s.symbol_name.eq_ignore_ascii_case(want.as_str()))
        {
            // GROUP D remediation (operator directive 2026-05-25):
            // route pip-digits through the canonical
            // `neoethos_core::symbol_metadata` registry instead of
            // hand-rolling the JPY heuristic. Defends against
            // silent-wrong-digits for symbols outside the simple
            // "ends with JPY" rule (e.g. XAUUSD = 2 digits, BTCUSD = 1).
            let digits = neoethos_core::symbol_metadata::resolve(&s.symbol_name)
                .map(|meta| meta.digits as i32)
                .unwrap_or_else(|| {
                    if s.symbol_name.to_ascii_uppercase().ends_with("JPY") {
                        3
                    } else {
                        5
                    }
                });
            resolved.push(StreamedSymbol {
                symbol_id: s.symbol_id,
                symbol_name: s.symbol_name.clone(),
                digits,
            });
        }
    }

    if resolved.is_empty() {
        tracing::warn!(
            target: "neoethos_app::live_spots_streamer",
            "skipping spawn — none of the DEFAULT_STREAMED_SYMBOLS found in broker catalog"
        );
        return false;
    }

    // Endpoint host inferred from environment label — matches the
    // pattern used in fetch_broker_symbols_blocking via
    // CTraderEnvironment::endpoint_host().
    let endpoint_host = match ct.environment.as_str() {
        "Live" => "live.ctraderapi.com",
        _ => "demo.ctraderapi.com",
    }
    .to_string();

    let config = LiveSpotsStreamerConfig {
        endpoint_host,
        client_id: ct.client_id.clone(),
        client_secret: ct.client_secret.clone(),
        access_token: token_bundle.access_token,
        account_id,
        symbols: resolved,
    };

    spawn(config);
    true
}

/// F-338 (Feature #12): re-subscribe the live spot stream to the
/// current `system.watchlist` without an app restart.
///
/// Bumps [`STREAM_GENERATION`] so any in-flight streamer self-terminates
/// on its next ~5 s read tick (see the generation check at the top of
/// the `run_blocking` read loop), then re-runs the canonical spawn
/// entrypoint [`try_spawn_with_defaults_blocking`] — which RE-READS
/// `system.watchlist` from `config.yaml`, re-resolves symbol ids against
/// the broker, and spawns a fresh streamer carrying the bumped
/// generation. The new streamer therefore subscribes to the edited
/// symbol set while the old one cleanly exits.
///
/// Returns whatever the entrypoint returns: `false` when the new
/// streamer could not be spawned (missing creds/token, broker
/// unreachable, none of the watchlist symbols resolvable, …). Note the
/// generation is bumped UNCONDITIONALLY — even on a `false` return the
/// old streamer still stops, which is the correct behaviour: a
/// watchlist edit should never leave a stream subscribed to the stale
/// symbol set.
///
/// Runs the same blocking work as [`try_spawn_with_defaults_blocking`]
/// (broker round-trip to list symbols), so callers on the async runtime
/// (e.g. the `POST /watchlist` handler) must invoke it via
/// `tokio::task::spawn_blocking`.
pub fn restart_streamer() -> bool {
    STREAM_GENERATION.fetch_add(1, Relaxed);
    try_spawn_with_defaults_blocking()
}

/// Spawn the streamer as a background async task. Returns
/// immediately; the task owns its own retry loop and won't be
/// observable to the caller.
///
/// The task is fire-and-forget by design — it has no parent
/// future to bubble errors to, and the cache stays in whatever
/// state it was in when the connection died. The next successful
/// reconnect refreshes it. Operators who want to know the
/// connection state should read the `live_spots::snapshot_all()`
/// freshness timestamps.
pub fn spawn(config: LiveSpotsStreamerConfig) {
    // F-338 (Feature #12): snapshot the current generation. The read
    // loop carries this `my_gen` and self-terminates the moment
    // `restart_streamer` bumps the global past it.
    let my_gen = STREAM_GENERATION.load(Relaxed);
    tokio::spawn(async move {
        loop {
            tracing::info!(
                target: "neoethos_app::live_spots_streamer",
                symbols = config.symbols.len(),
                generation = my_gen,
                "connecting to cTrader spot stream"
            );
            let cfg = config.clone();
            let outcome = tokio::task::spawn_blocking(move || run_blocking(cfg, my_gen)).await;
            match outcome {
                Ok(Ok(())) => {
                    tracing::warn!(
                        target: "neoethos_app::live_spots_streamer",
                        "spot stream ended cleanly (read loop returned Ok); will reconnect"
                    );
                }
                Ok(Err(err)) => {
                    tracing::warn!(
                        target: "neoethos_app::live_spots_streamer",
                        error = %err,
                        "spot stream errored; will reconnect after backoff"
                    );
                }
                Err(join_err) => {
                    tracing::error!(
                        target: "neoethos_app::live_spots_streamer",
                        error = %join_err,
                        "spot stream blocking task panicked; will reconnect"
                    );
                }
            }
            // F-338 (Feature #12): if a newer streamer generation has
            // been installed (watchlist edit → `restart_streamer`), this
            // streamer has been superseded — STOP rather than reconnect.
            // The freshly-spawned streamer owns the live stream now.
            if STREAM_GENERATION.load(Relaxed) != my_gen {
                tracing::info!(
                    target: "neoethos_app::live_spots_streamer",
                    generation = my_gen,
                    current = STREAM_GENERATION.load(Relaxed),
                    "spot streamer superseded by a newer generation — exiting reconnect loop"
                );
                break;
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
}

fn run_blocking(config: LiveSpotsStreamerConfig, my_gen: u64) -> Result<()> {
    let url = format!("wss://{}:5036", config.endpoint_host);
    crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
    let (mut socket, _) = connect(url.as_str())
        .with_context(|| format!("failed to connect to cTrader spot stream {url}"))?;

    // 1. App auth
    send_and_await(
        &mut socket,
        &serde_json::to_string(&build_application_auth_request(
            &config.client_id,
            &config.client_secret,
            "spot-app-auth",
        ))?,
    )?;

    // 2. Account auth
    send_and_await(
        &mut socket,
        &serde_json::to_string(&build_account_auth_request(
            config.account_id,
            &config.access_token,
            "spot-account-auth",
        ))?,
    )?;

    // 3. Subscribe to all symbols in one request — cTrader's
    //    subscribe-spots payload takes a list, so we don't need
    //    one round-trip per symbol.
    //
    //    No unsubscribe-before-subscribe is needed on reconnect: every
    //    reconnect runs a FRESH `run_blocking` that opens a brand-new
    //    `connect()` socket (above) → a new cTrader session. Spot
    //    subscriptions are per-session, so the dropped connection's
    //    subscriptions die with it; there is nothing to carry over and
    //    therefore no duplicate-subscription to guard against. Even if a
    //    stray duplicate spot event did arrive, `live_spots::update_tick`
    //    overwrites by `symbol_id`, so the cache stays correct.
    let symbol_ids: Vec<i64> = config.symbols.iter().map(|s| s.symbol_id).collect();
    send_and_await(
        &mut socket,
        &serde_json::to_string(&build_subscribe_spots_request(
            config.account_id,
            &symbol_ids,
            true,
            "spot-subscribe",
        ))?,
    )?;

    tracing::info!(
        target: "neoethos_app::live_spots_streamer",
        symbols = symbol_ids.len(),
        "spot stream subscribed; entering read loop"
    );

    // **2026-05-31 fix — outgoing app heartbeat.** cTrader's Open API
    // closes a streaming connection (CloseFrame "Bye") after ~60 s when
    // the client doesn't send a ProtoHeartbeatEvent, EVEN if the
    // transport-level WebSocket ping/pong is healthy. The old loop only
    // replied to incoming pings and never sent its own heartbeat, so the
    // spot stream died every ~65 s and Market Watch showed "no live
    // spots" (and position pnl_pips fell back to 0 with no live price).
    // We now (a) set a short read timeout so the blocking `read()`
    // returns periodically, and (b) send a JSON heartbeat (payloadType
    // 51) every ~10 s — the same cadence the account session uses.
    set_spot_read_timeout(&mut socket, Duration::from_secs(5));
    let heartbeat_every = Duration::from_secs(10);
    let mut last_heartbeat = Instant::now();

    // 4. Read loop. Spot events flow in forever; everything else
    //    (ping/pong, account disconnect, errors) is handled inline.
    loop {
        // F-338 (Feature #12): bail out the moment the operator edits
        // the watchlist (which bumps STREAM_GENERATION via
        // `restart_streamer`). Returning `Ok(())` lets the outer reconnect
        // loop observe the generation mismatch and exit cleanly instead
        // of reconnecting. The 5 s read-timeout cadence set above
        // guarantees this check runs within ~5 s of the bump.
        if STREAM_GENERATION.load(Relaxed) != my_gen {
            tracing::info!(
                target: "neoethos_app::live_spots_streamer",
                "watchlist changed — closing spot stream to re-subscribe"
            );
            return Ok(());
        }
        // Send an app-level heartbeat on schedule so cTrader keeps the
        // stream open. A half-duplex send between reads is safe on a
        // sync tungstenite socket.
        if last_heartbeat.elapsed() >= heartbeat_every {
            let hb = format!(
                r#"{{"clientMsgId":"spot-hb","payloadType":{CTRADER_OA_HEARTBEAT_PAYLOAD_TYPE},"payload":{{}}}}"#
            );
            socket
                .send(Message::Text(hb.into()))
                .context("failed to send spot-stream heartbeat")?;
            last_heartbeat = Instant::now();
        }

        let frame = match socket.read() {
            Ok(f) => f,
            // Read timeout (set above) — no frame this interval. Loop
            // back so the heartbeat scheduler runs. tungstenite surfaces
            // the socket timeout as WouldBlock (*nix) or TimedOut
            // (Windows); both just mean "nothing to read right now".
            Err(tungstenite::Error::Io(e))
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(e) => {
                return Err(anyhow!(
                    "failed to read frame from cTrader spot stream: {e}"
                ));
            }
        };
        let payload_text = match frame {
            Message::Text(t) => t.to_string(),
            Message::Binary(b) => {
                String::from_utf8(b.to_vec()).context("non-utf8 binary frame on spot stream")?
            }
            Message::Ping(p) => {
                socket
                    .send(Message::Pong(p))
                    .context("failed to reply pong on spot stream")?;
                continue;
            }
            Message::Pong(_) => continue,
            Message::Close(reason) => {
                return Err(anyhow!(
                    "cTrader spot stream closed by server: {:?}",
                    reason
                ));
            }
            Message::Frame(_) => continue,
        };

        // **2026-05-25 — real-data fixture capture** (operator
        // directive). No-op when `NEOETHOS_CAPTURE_FIXTURES_DIR` is
        // unset (production default). When set, writes every parsed
        // payload to disk so the `TODO(real-data)` tests can be
        // backed by captured fixtures from a live cTrader session.
        // Best-effort; never blocks the stream.
        crate::app_services::env_overrides::capture_fixture(
            "OpenApiSpotFrame",
            payload_text.as_bytes(),
        );

        // 2026-06-10 defensive parse: a single malformed frame must NOT tear
        // down the whole stream (a reconnect costs a full re-auth + re-subscribe
        // round-trip and a Market-Watch price gap). Skip it and keep reading —
        // a genuinely dead socket still surfaces via the read error / Close arm.
        let envelope = match parse_open_api_envelope(&payload_text) {
            Ok(env) => env,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::live_spots_streamer",
                    error = %err,
                    "skipping unparseable cTrader spot-stream frame"
                );
                continue;
            }
        };
        match envelope.payload_type {
            CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE => {
                if let Some((symbol_id, bid, ask, ts)) =
                    parse_spot_event_loose(&payload_text, &config.symbols)
                {
                    let symbol_name = config
                        .symbols
                        .iter()
                        .find(|s| s.symbol_id == symbol_id)
                        .map(|s| s.symbol_name.clone())
                        .unwrap_or_default();
                    live_spots::update_tick(symbol_id, symbol_name, bid, ask, ts);
                }
            }
            CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE => {
                return Err(anyhow!(
                    "cTrader account disconnect event received on spot stream"
                ));
            }
            CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE => {
                let detail = parse_ctrader_error_payload(&envelope.payload)
                    .unwrap_or_else(|_| "unparseable error payload".to_string());
                return Err(anyhow!("cTrader error on spot stream: {detail}"));
            }
            // **2026-06-10 API-completeness pass.** The Open API multiplexes
            // ALL account push events onto this one authed socket, not just
            // spots. Surface the high-value ones instead of dropping them in
            // the `_` arm — silence here is why an invalidated token or a
            // margin call used to go unnoticed until the next failed request.
            CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE => {
                // Token revoked broker-side: the whole session is now invalid.
                // Tear down so the reconnect re-auths; if the token is truly
                // dead the re-auth surfaces it loudly. This is an auth
                // emergency, not a routine reconnect.
                tracing::error!(
                    target: "neoethos_app::live_spots_streamer",
                    "cTrader ACCOUNTS_TOKEN_INVALIDATED event — the access token was revoked; \
                     a manual re-authentication is required"
                );
                return Err(anyhow!(
                    "cTrader access token invalidated (token-invalidated event on spot stream)"
                ));
            }
            CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE => {
                tracing::warn!(
                    target: "neoethos_app::live_spots_streamer",
                    payload = %payload_text,
                    "cTrader CLIENT_DISCONNECT event — broker dropped the application session"
                );
                return Err(anyhow!("cTrader client disconnect event on spot stream"));
            }
            CTRADER_OA_MARGIN_CALL_TRIGGER_EVENT_PAYLOAD_TYPE => {
                // A live-money risk event — never bury this.
                tracing::warn!(
                    target: "neoethos_app::live_spots_streamer",
                    payload = %payload_text,
                    "cTrader MARGIN_CALL_TRIGGER event — a margin-call threshold was breached"
                );
                continue;
            }
            CTRADER_OA_MARGIN_CALL_UPDATE_EVENT_PAYLOAD_TYPE => {
                tracing::info!(
                    target: "neoethos_app::live_spots_streamer",
                    "cTrader MARGIN_CALL_UPDATE event — a margin-call threshold changed"
                );
                continue;
            }
            CTRADER_OA_MARGIN_CHANGED_EVENT_PAYLOAD_TYPE
            | CTRADER_OA_TRADER_UPDATE_EVENT_PAYLOAD_TYPE
            | CTRADER_OA_TRAILING_SL_CHANGED_EVENT_PAYLOAD_TYPE => {
                // Informational account-state pushes the bridge's periodic
                // snapshot will also pick up. Trace at debug so they are
                // observable without spamming the default log level.
                tracing::debug!(
                    target: "neoethos_app::live_spots_streamer",
                    payload_type = envelope.payload_type,
                    "cTrader account push event (margin/trader/trailing-SL changed)"
                );
                continue;
            }
            _ => {
                // Heartbeat, symbol-changed, execution events, etc. — the
                // regular bridge owns those. Just keep reading.
                continue;
            }
        }
    }
}

/// Set a read timeout on the underlying TCP socket so the blocking
/// [`WebSocket::read`] returns periodically (instead of blocking until
/// the next frame arrives), letting the heartbeat scheduler in the read
/// loop run. Best-effort: if setting the timeout fails we just fall back
/// to the old blocking behaviour — no worse than before the fix.
fn set_spot_read_timeout(socket: &mut CTraderSocket, dur: Duration) {
    match socket.get_mut() {
        MaybeTlsStream::Plain(tcp) => {
            let _ = tcp.set_read_timeout(Some(dur));
        }
        MaybeTlsStream::Rustls(tls) => {
            // rustls 0.23 exposes the wrapped TcpStream as the public
            // `sock` field on StreamOwned.
            let _ = tls.sock.set_read_timeout(Some(dur));
        }
        // native-tls isn't compiled in for this target; nothing to do.
        _ => {}
    }
}

/// Send a single message and read replies until we see the
/// matching response (matched by clientMsgId). Errors / closes
/// are propagated up so the outer reconnect loop kicks in.
fn send_and_await(socket: &mut CTraderSocket, message_json: &str) -> Result<()> {
    let envelope = parse_open_api_envelope(message_json)?;
    let expected_msg_id = envelope.client_msg_id.clone();

    socket
        .send(Message::Text(message_json.to_string().into()))
        .context("failed to send cTrader spot-stream message")?;

    loop {
        let frame = socket
            .read()
            .context("failed to read cTrader spot-stream response")?;
        let text = match frame {
            Message::Text(t) => t.to_string(),
            Message::Binary(b) => String::from_utf8(b.to_vec())
                .context("non-utf8 binary frame during spot-stream handshake")?,
            Message::Ping(p) => {
                socket
                    .send(Message::Pong(p))
                    .context("failed to reply pong during handshake")?;
                continue;
            }
            Message::Pong(_) => continue,
            Message::Close(reason) => {
                return Err(anyhow!(
                    "cTrader closed spot stream during handshake: {:?}",
                    reason
                ));
            }
            Message::Frame(_) => continue,
        };
        // 2026-06-10 defensive parse: skip an unparseable frame during the
        // handshake exactly like an unrelated one (below) — keep awaiting the
        // matching clientMsgId rather than aborting the connection on one bad
        // frame.
        let env = match parse_open_api_envelope(&text) {
            Ok(env) => env,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::live_spots_streamer",
                    error = %err,
                    "skipping unparseable cTrader frame during spot-stream handshake"
                );
                continue;
            }
        };
        if env.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
            let detail = parse_ctrader_error_payload(&env.payload)
                .unwrap_or_else(|_| "unparseable error payload".to_string());
            return Err(anyhow!("cTrader handshake error: {detail}"));
        }
        if env.client_msg_id == expected_msg_id {
            return Ok(());
        }
        // Otherwise: drop unrelated frame (e.g. an early spot
        // event arriving before the subscribe response). The
        // post-handshake loop will catch it on the next pass.
    }
}

#[derive(Debug, Deserialize)]
struct LooseSpotEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: LooseSpotPayload,
}

#[derive(Debug, Deserialize)]
struct LooseSpotPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    bid: Option<u64>,
    ask: Option<u64>,
    timestamp: Option<i64>,
}

/// Lenient cTrader spot-event parser.
/// Returns `(symbol_id, bid, ask, broker_timestamp_ms)` for any
/// spot event whose symbol is in our subscription list. Returns
/// `None` for events with an unknown symbol (e.g. a leftover
/// subscription we forgot to unsub from) so we silently drop
/// those rather than crashing the read loop.
fn parse_spot_event_loose(
    response_json: &str,
    known_symbols: &[StreamedSymbol],
) -> Option<(i64, Option<f64>, Option<f64>, Option<i64>)> {
    let env: LooseSpotEnvelope = serde_json::from_str(response_json).ok()?;
    if env.payload_type != CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
        return None;
    }
    let symbol_meta = known_symbols
        .iter()
        .find(|s| s.symbol_id == env.payload.symbol_id)?;
    let bid = env
        .payload
        .bid
        .map(|v| scale_price(v as i64, symbol_meta.digits));
    let ask = env
        .payload
        .ask
        .map(|v| scale_price(v as i64, symbol_meta.digits));
    Some((symbol_meta.symbol_id, bid, ask, env.payload.timestamp))
}

/// Scales a cTrader integer price to a float. The math is:
/// `raw_int / 100_000 * 10^digits`, rounded to `digits` decimals.
fn scale_price(value: i64, digits: i32) -> f64 {
    let raw = value as f64 / 100_000.0;
    let factor = 10_f64.powi(digits.max(0));
    (raw * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spot_event_loose_picks_up_known_symbol() {
        let symbols = vec![StreamedSymbol {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            digits: 5,
        }];
        // cTrader sends bid/ask scaled by 10^5: bid=108500 → 1.08500.
        // The `scale_price` formula is `value / 100000 * 10^digits`,
        // which at digits=5 reduces to `value / 100000`.
        let payload = r#"{
            "payloadType": 2131,
            "payload": {
                "ctidTraderAccountId": 42,
                "symbolId": 1,
                "bid": 108500,
                "ask": 108520,
                "timestamp": 1700000000
            }
        }"#;
        let parsed = parse_spot_event_loose(payload, &symbols).expect("parsed");
        assert_eq!(parsed.0, 1);
        assert_eq!(parsed.1, Some(1.085));
        assert_eq!(parsed.2, Some(1.0852));
        assert_eq!(parsed.3, Some(1_700_000_000));
    }

    #[test]
    fn parse_spot_event_loose_drops_unknown_symbol() {
        let symbols = vec![StreamedSymbol {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            digits: 5,
        }];
        let payload = r#"{
            "payloadType": 2131,
            "payload": {
                "ctidTraderAccountId": 42,
                "symbolId": 999,
                "bid": 108500,
                "ask": 108520
            }
        }"#;
        assert!(parse_spot_event_loose(payload, &symbols).is_none());
    }

    #[test]
    fn parse_spot_event_loose_drops_non_spot_payload_types() {
        let symbols = vec![StreamedSymbol {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            digits: 5,
        }];
        // payloadType 2104 = account auth res, not a spot
        let payload = r#"{
            "payloadType": 2104,
            "payload": {
                "ctidTraderAccountId": 42,
                "symbolId": 1
            }
        }"#;
        assert!(parse_spot_event_loose(payload, &symbols).is_none());
    }

    #[test]
    fn scale_price_handles_5_digit_forex_pair() {
        // EURUSD: raw 108500 in 5-digit form represents 1.08500
        // The cTrader scaling is integer / 10^digits with the
        // 100_000 normalisation. value 108500 here ≠ 1.085 with
        // digits=5; the production stream sends value 108_500
        // (no scaling) and 100_000 cancels out at digits=5:
        //   108_500 / 100_000 * 10^5 = 1.085 * 100_000 = 108_500
        // rounded /factor → 108_500 / 100_000 = 1.085
        let p = scale_price(108_500, 5);
        assert!((p - 1.085).abs() < 1e-6, "got {p}");
    }
}
