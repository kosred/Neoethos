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
//! - **Outer reconnect loop** uses capped exponential backoff with
//!   jitter. Repeated authentication/routing failures cannot create
//!   an indefinite one-second retry loop. Only a subscribed session
//!   lasting at least a minute resets the retry history.
//! - **Symbol list** comes from the saved watchlist (forex majors are
//!   the empty-watchlist fallback), resolved against broker symbol IDs.
//!   A watchlist edit supersedes the old stream and its pending retries.
//! - **Application heartbeat** is sent every ten seconds, independently
//!   of WebSocket ping/pong. Cached quotes retain their actual freshness
//!   timestamps; a reconnect attempt does not make stale quotes current.

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};

/// F-338 (Feature #12): monotonically-increasing "which streamer
/// generation is current" counter. Every logical start reserves a new
/// value before blocking preparation (`my_gen`); [`restart_streamer`] does this once.
/// The in-flight read loop notices `STREAM_GENERATION != my_gen` on its
/// next ~5 s read-timeout tick, closes its socket, and self-terminates
/// — while a freshly-spawned streamer (carrying the new generation)
/// takes over with the updated watchlist. This lets a Market Watch edit
/// re-subscribe the live stream within ~5 s with no app restart.
static STREAM_GENERATION: AtomicU64 = AtomicU64::new(0);

const MAX_RECONNECT_DELAY_MS: u64 = 60_000;
const STABLE_SUBSCRIPTION_DURATION: Duration = Duration::from_secs(60);

#[derive(Default)]
struct SpotReconnectBackoff {
    consecutive_failures: u32,
}

impl SpotReconnectBackoff {
    fn next_delay(
        &mut self,
        configured_base_ms: u64,
        subscribed_for: Option<Duration>,
        jitter: u64,
    ) -> Duration {
        if subscribed_for.is_some_and(|elapsed| elapsed >= STABLE_SUBSCRIPTION_DURATION) {
            self.consecutive_failures = 0;
        }
        let base_ms = configured_base_ms.clamp(1_000, MAX_RECONNECT_DELAY_MS);
        let ceiling_ms = base_ms
            .saturating_mul(1_u64 << self.consecutive_failures.min(6))
            .min(MAX_RECONNECT_DELAY_MS);
        // Equal jitter retains a meaningful minimum pause at the cap. A
        // healthy-but-idle weekend session counts only after subscription,
        // never from time spent waiting for connect/authentication to fail.
        let floor_ms = (ceiling_ms / 2).max(base_ms);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        Duration::from_millis(floor_ms + jitter % (ceiling_ms - floor_ms + 1))
    }
}

async fn wait_for_stream_retry(generation: &AtomicU64, my_gen: u64, delay: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + delay;
    loop {
        if generation.load(Relaxed) != my_gen {
            return false;
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return true;
        }
        tokio::time::sleep(remaining.min(Duration::from_millis(250))).await;
    }
}

use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_HEARTBEAT_PAYLOAD_TYPE, CTRADER_OA_MARGIN_CALL_TRIGGER_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_MARGIN_CALL_UPDATE_EVENT_PAYLOAD_TYPE, CTRADER_OA_MARGIN_CHANGED_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE, CTRADER_OA_TRADER_UPDATE_EVENT_PAYLOAD_TYPE,
    CTRADER_OA_TRAILING_SL_CHANGED_EVENT_PAYLOAD_TYPE, CTraderOpenApiJsonMessage,
    build_account_auth_request, build_application_auth_request, build_subscribe_spots_request,
    ctrader_json_wss_url, expected_response_payload_type, parse_ctrader_error_payload,
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
    /// `digits` controls display rounding only. Checked financial reads retain
    /// the original protocol sides, whose scale is always 1/100000.
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
    let my_gen =
        reserve_stream_generation(&STREAM_GENERATION, live_spots::invalidate_stream_generation);
    try_spawn_with_defaults_for_generation(my_gen)
}

fn try_spawn_with_defaults_for_generation(my_gen: u64) -> bool {
    if STREAM_GENERATION.load(Relaxed) != my_gen {
        return false;
    }
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
        Ok(v) if v > 0 => v,
        _ => {
            tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                account_id = %account_row.account_id,
                "skipping spawn — account_id is not a positive integer"
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

    // The catalog resolver captures its own credentials. A settings switch
    // between the two reads must not relabel another account/environment's IDs.
    if bundle.account_id != account_id || bundle.environment != ct.environment.as_str() {
        tracing::warn!(
            target: "neoethos_app::live_spots_streamer",
            "skipping spawn — broker symbol catalog differs from the captured account/environment"
        );
        return false;
    }

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

    finish_preparation_for_generation(&STREAM_GENERATION, my_gen, config, spawn_for_generation)
}

// All public logical starts supersede old loops, even if preparation then fails.
// The injected invalidator keeps the same reservation path testable without a
// process-global cache or a broker connection.
fn reserve_stream_generation(generation: &AtomicU64, invalidate: impl FnOnce(u64)) -> u64 {
    let reserved = generation.fetch_add(1, Relaxed).wrapping_add(1);
    invalidate(reserved);
    reserved
}

// Carry the generation reserved before blocking preparation all the way to launch.
// A slower old catalog/config read must never adopt a newer restart's generation.
fn finish_preparation_for_generation(
    generation: &AtomicU64,
    my_gen: u64,
    config: LiveSpotsStreamerConfig,
    launch: impl FnOnce(LiveSpotsStreamerConfig, u64) -> bool,
) -> bool {
    if generation.load(Relaxed) != my_gen {
        return false;
    }
    launch(config, my_gen)
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
    // The public startup helper reserves exactly once, including failed starts.
    try_spawn_with_defaults_blocking()
}

/// Spawn the streamer as a background async task. Returns
/// immediately; the task owns its own retry loop and won't be
/// observable to the caller.
///
/// Display rows may remain after disconnect, but their financial provenance
/// is invalidated by the connection guard. Only fresh sides from the current
/// authenticated connection can satisfy the checked cache read.
pub fn spawn(config: LiveSpotsStreamerConfig) {
    let my_gen =
        reserve_stream_generation(&STREAM_GENERATION, live_spots::invalidate_stream_generation);
    let _ = spawn_for_generation(config, my_gen);
}

fn spawn_for_generation(config: LiveSpotsStreamerConfig, my_gen: u64) -> bool {
    if STREAM_GENERATION.load(Relaxed) != my_gen {
        return false;
    }
    tokio::spawn(async move {
        let mut retry = SpotReconnectBackoff::default();
        loop {
            if STREAM_GENERATION.load(Relaxed) != my_gen {
                break;
            }
            tracing::info!(
                target: "neoethos_app::live_spots_streamer",
                symbols = config.symbols.len(),
                generation = my_gen,
                "connecting to cTrader spot stream"
            );
            let mut cfg = config.clone();
            // 2026-07-18 deep-audit fix: refresh the access token before EVERY
            // (re)connect. cTrader access tokens live ~30 minutes; the old loop
            // re-authenticated each reconnect with the SPAWN-TIME token, so
            // after the first expiry every reconnect failed auth with the same
            // stale token forever — Market Watch went permanently dark until an
            // app restart. Fail-soft: on refresh failure keep the previous
            // token (the reconnect may still succeed if it hasn't expired).
            {
                let client_id = cfg.client_id.clone();
                let client_secret = cfg.client_secret.clone();
                match tokio::task::spawn_blocking(move || {
                    crate::app_services::broker_api::ensure_fresh_token_bundle(
                        &client_id,
                        &client_secret,
                    )
                })
                .await
                {
                    Ok(Ok(bundle)) => cfg.access_token = bundle.access_token,
                    Ok(Err(err)) => tracing::warn!(
                        target: "neoethos_app::live_spots_streamer",
                        error = %err,
                        "token refresh before spot-stream connect failed — using previous token"
                    ),
                    Err(join_err) => tracing::warn!(
                        target: "neoethos_app::live_spots_streamer",
                        error = %join_err,
                        "token refresh task failed — using previous token"
                    ),
                }
            }
            if STREAM_GENERATION.load(Relaxed) != my_gen {
                break;
            }
            let outcome = tokio::task::spawn_blocking(move || {
                let mut subscribed_at = None;
                let result = run_blocking(cfg, my_gen, &mut subscribed_at);
                (
                    result,
                    subscribed_at.map(|started: Instant| started.elapsed()),
                )
            })
            .await;
            let subscribed_for = outcome.as_ref().ok().and_then(|(_, elapsed)| *elapsed);
            match outcome {
                Ok((Ok(()), _)) => {
                    tracing::warn!(
                        target: "neoethos_app::live_spots_streamer",
                        "spot stream ended cleanly (read loop returned Ok); will reconnect"
                    );
                }
                Ok((Err(err), _)) => {
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
            let backoff = retry.next_delay(
                crate::app_services::env_overrides::ctrader_stream_backoff_base_ms(),
                subscribed_for,
                rand::random::<u64>(),
            );
            tracing::info!(
                target: "neoethos_app::live_spots_streamer",
                retry_delay_ms = backoff.as_millis() as u64,
                consecutive_failures = retry.consecutive_failures,
                "spot stream reconnect scheduled"
            );
            if !wait_for_stream_retry(&STREAM_GENERATION, my_gen, backoff).await {
                break;
            }
        }
    });
    true
}

fn run_blocking(
    config: LiveSpotsStreamerConfig,
    my_gen: u64,
    subscribed_at: &mut Option<Instant>,
) -> Result<()> {
    if STREAM_GENERATION.load(Relaxed) != my_gen {
        return Ok(());
    }
    let environment = stream_environment(&config.endpoint_host)?;
    let session =
        live_spots::begin_session(config.account_id, environment, my_gen).ok_or_else(|| {
            anyhow!("spot session is invalid, superseded, or the cache is unavailable")
        })?;
    // The guard starts before connect/auth and invalidates on every return or unwind.
    let url = ctrader_json_wss_url(&config.endpoint_host);
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

    // Resolve conversion dependencies on this same authenticated connection,
    // before any spot subscriptions. Recomputed after every reconnect; never
    // mutate the operator's watchlist or borrow another connection's quotes.
    let (symbols, subscription) =
        prepare_spot_subscription(config.account_id, &config.symbols, |request| {
            send_and_await(&mut socket, &serde_json::to_string(request)?)
        })?;
    send_and_await(&mut socket, &serde_json::to_string(&subscription)?)?;

    *subscribed_at = Some(Instant::now());
    tracing::info!(
        target: "neoethos_app::live_spots_streamer",
        symbols = symbols.len(),
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
                    parse_spot_event_loose(&payload_text, config.account_id, &symbols)
                {
                    if let Some(symbol) = symbols.iter().find(|s| s.symbol_id == symbol_id) {
                        live_spots::update_session_tick(
                            &session,
                            symbol_id,
                            &symbol.symbol_name,
                            symbol.digits,
                            bid,
                            ask,
                            ts,
                        );
                    }
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

fn prepare_spot_subscription(
    account_id: i64,
    configured: &[StreamedSymbol],
    mut exchange: impl FnMut(&CTraderOpenApiJsonMessage) -> Result<CTraderOpenApiJsonMessage>,
) -> Result<(Vec<StreamedSymbol>, CTraderOpenApiJsonMessage)> {
    use crate::app_services::ctrader_account::parse_trader_response;
    use crate::app_services::ctrader_data::parse_symbols_list_response;
    use crate::app_services::ctrader_messages::{build_symbols_list_request, build_trader_request};

    anyhow::ensure!(account_id > 0, "spot subscription account must be positive");
    let mut checked = |request: CTraderOpenApiJsonMessage| -> Result<CTraderOpenApiJsonMessage> {
        let response = exchange(&request)?;
        anyhow::ensure!(
            handshake_response_matches(
                &response,
                &request.client_msg_id,
                expected_response_payload_type(request.payload_type)?,
                Some(account_id),
            )?,
            "spot metadata response does not match its request"
        );
        Ok(response)
    };
    let trader = parse_trader_response(&serde_json::to_string(&checked(build_trader_request(
        account_id,
        "spot-trader",
    ))?)?)?;
    let catalog = parse_symbols_list_response(&serde_json::to_string(&checked(
        build_symbols_list_request(account_id, false, "spot-symbols"),
    )?)?)?;
    anyhow::ensure!(
        trader.account_id == account_id && catalog.account_id == account_id,
        "spot subscription metadata belongs to another account"
    );
    let deposit = trader
        .deposit_asset_id
        .filter(|id| *id > 0)
        .context("spot subscription trader omitted a valid depositAssetId")?;
    let mut symbols = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut primaries = Vec::new();
    for configured in configured {
        anyhow::ensure!(
            configured.symbol_id > 0 && !configured.symbol_name.trim().is_empty(),
            "configured spot symbol identity is invalid"
        );
        let mut matches = catalog
            .symbols
            .iter()
            .filter(|symbol| symbol.symbol_id == configured.symbol_id);
        let primary = matches
            .next()
            .context("configured spot symbol is absent from current account catalog")?;
        anyhow::ensure!(
            matches.next().is_none() && primary.symbol_name == configured.symbol_name,
            "configured spot symbol identity differs from current account catalog"
        );
        if seen.insert(configured.symbol_id) {
            symbols.push(configured.clone());
            primaries.push(primary);
        }
    }
    for primary in primaries {
        match crate::app_services::broker_api::required_entry_conversion_symbol(
            primary,
            deposit,
            &catalog.symbols,
        ) {
            Ok(Some(leg)) if seen.insert(leg.symbol_id) => symbols.push(StreamedSymbol {
                symbol_id: leg.symbol_id,
                symbol_name: leg.symbol_name.clone(),
                // Light symbols have no broker digits. Keep full wire precision
                // for these display rows, not a guessed broker pip/digits contract.
                // Checked financial reads use the original 1/100000 sides anyway.
                digits: 5,
            }),
            Ok(_) => {}
            Err(error) => tracing::warn!(
                target: "neoethos_app::live_spots_streamer",
                symbol = %primary.symbol_name, %error,
                "conversion subscription unavailable for this primary; retaining display quotes, entry conversion remains refused"
            ),
        }
    }
    anyhow::ensure!(!symbols.is_empty(), "spot subscription set is empty");
    let ids = symbols
        .iter()
        .map(|symbol| symbol.symbol_id)
        .collect::<Vec<_>>();
    let subscription = build_subscribe_spots_request(account_id, &ids, true, "spot-subscribe");
    Ok((symbols, subscription))
}

fn stream_environment(endpoint_host: &str) -> Result<CTraderEnvironment> {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        if endpoint_host == environment.endpoint_host() {
            return Ok(environment);
        }
    }
    Err(anyhow!(
        "spot endpoint does not identify a supported cTrader environment"
    ))
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
/// matching response (message ID, expected type, and account). Errors / closes
/// are propagated up so the outer reconnect loop kicks in.
fn send_and_await(
    socket: &mut CTraderSocket,
    message_json: &str,
) -> Result<CTraderOpenApiJsonMessage> {
    let envelope = parse_open_api_envelope(message_json)?;
    anyhow::ensure!(
        !envelope.client_msg_id.is_empty(),
        "spot handshake request has no message ID"
    );
    let expected_type = expected_response_payload_type(envelope.payload_type)?;
    let expected_account = envelope
        .payload
        .get("ctidTraderAccountId")
        .map(|value| {
            value
                .as_i64()
                .filter(|account| *account > 0)
                .ok_or_else(|| anyhow!("spot handshake request has an invalid account"))
        })
        .transpose()?;

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
        if handshake_response_matches(
            &env,
            &envelope.client_msg_id,
            expected_type,
            expected_account,
        )? {
            return Ok(env);
        }
        // Unrelated frames cannot complete authentication. An early spot event
        // discarded here is not replayed or given freshness in the cache.
    }
}

fn handshake_response_matches(
    response: &CTraderOpenApiJsonMessage,
    expected_msg_id: &str,
    expected_type: u32,
    expected_account: Option<i64>,
) -> Result<bool> {
    if response.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        let detail = parse_ctrader_error_payload(&response.payload)
            .unwrap_or_else(|_| "unparseable error payload".to_owned());
        return Err(anyhow!("cTrader handshake error: {detail}"));
    }
    if matches!(
        response.payload_type,
        CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE
            | CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE
            | CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE
    ) {
        return Err(anyhow!("cTrader session invalidated during spot handshake"));
    }
    if response.client_msg_id != expected_msg_id {
        return Ok(false);
    }
    anyhow::ensure!(
        response.payload_type == expected_type,
        "spot handshake response type differs from the requested operation"
    );
    anyhow::ensure!(
        response.payload.is_object(),
        "spot handshake response payload is not an object"
    );
    if let Some(account) = expected_account {
        anyhow::ensure!(
            account > 0
                && response
                    .payload
                    .get("ctidTraderAccountId")
                    .and_then(serde_json::Value::as_i64)
                    == Some(account),
            "spot handshake response belongs to a missing, invalid, or different account"
        );
    }
    Ok(true)
}

#[derive(Debug, Deserialize)]
struct LooseSpotEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: LooseSpotPayload,
}

#[derive(Debug, Deserialize)]
struct LooseSpotPayload {
    #[serde(rename = "ctidTraderAccountId")]
    account_id: i64,
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    bid: Option<u64>,
    ask: Option<u64>,
    timestamp: Option<i64>,
}

/// Lenient cTrader spot-event parser.
/// Returns `(symbol_id, raw_bid, raw_ask, broker_timestamp_ms)` only for the
/// expected account and a symbol in our subscription list. Returns
/// `None` for events with an unknown symbol (e.g. a leftover
/// subscription we forgot to unsub from) so we silently drop
/// those rather than crashing the read loop.
fn parse_spot_event_loose(
    response_json: &str,
    expected_account_id: i64,
    known_symbols: &[StreamedSymbol],
) -> Option<(i64, Option<u64>, Option<u64>, Option<i64>)> {
    let env: LooseSpotEnvelope = serde_json::from_str(response_json).ok()?;
    if env.payload_type != CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE
        || expected_account_id <= 0
        || env.payload.account_id != expected_account_id
        || env.payload.symbol_id <= 0
        || (env.payload.bid.is_none() && env.payload.ask.is_none())
    {
        return None;
    }
    let symbol_meta = known_symbols
        .iter()
        .find(|s| s.symbol_id == env.payload.symbol_id)?;
    Some((
        symbol_meta.symbol_id,
        env.payload.bid,
        env.payload.ask,
        env.payload.timestamp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconnect_backoff_escalates_and_caps_with_bounded_jitter() {
        let mut retry = SpotReconnectBackoff::default();
        for ceiling in [1_000_u64, 2_000, 4_000, 8_000, 16_000, 32_000, 60_000] {
            let floor = (ceiling / 2).max(1_000);
            assert_eq!(retry.next_delay(1_000, None, 0).as_millis(), floor as u128);
        }
        for jitter in [0, 1, 30_000, u64::MAX] {
            let delay = retry.next_delay(1_000, None, jitter);
            assert!((Duration::from_secs(30)..=Duration::from_secs(60)).contains(&delay));
        }
        retry.consecutive_failures = u32::MAX;
        assert_eq!(
            retry.next_delay(1_000, None, 30_000),
            Duration::from_secs(60)
        );
        assert_eq!(retry.consecutive_failures, u32::MAX);
    }

    #[test]
    fn reconnect_backoff_only_resets_after_a_stable_subscription() {
        let mut retry = SpotReconnectBackoff {
            consecutive_failures: 20,
        };
        assert_eq!(retry.next_delay(1_000, None, 0), Duration::from_secs(30));
        assert_eq!(
            retry.next_delay(1_000, Some(Duration::from_secs(59)), 0),
            Duration::from_secs(30)
        );
        assert_eq!(
            retry.next_delay(1_000, Some(STABLE_SUBSCRIPTION_DURATION), 0),
            Duration::from_secs(1)
        );
        assert_eq!(retry.consecutive_failures, 1);
    }

    #[test]
    fn reconnect_backoff_bounds_invalid_operator_delay_values() {
        let mut retry = SpotReconnectBackoff::default();
        assert_eq!(retry.next_delay(0, None, 0), Duration::from_secs(1));
        assert_eq!(retry.next_delay(u64::MAX, None, 0), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn superseded_stream_does_not_wait_or_retry() {
        let generation = AtomicU64::new(2);
        assert!(!wait_for_stream_retry(&generation, 1, Duration::from_secs(60)).await);
        assert!(wait_for_stream_retry(&generation, 2, Duration::ZERO).await);
    }

    #[tokio::test]
    async fn watchlist_change_interrupts_a_pending_backoff() {
        let generation = AtomicU64::new(1);
        let waiting = wait_for_stream_retry(&generation, 1, Duration::from_secs(60));
        let change = async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            generation.store(2, Relaxed);
        };
        let (should_retry, ()) = tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(waiting, change)
        })
        .await
        .expect("watchlist change should not wait for the full retry delay");
        assert!(!should_retry);
    }

    #[test]
    fn parse_spot_event_loose_picks_up_known_symbol() {
        let symbols = vec![StreamedSymbol {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            digits: 5,
        }];
        // Preserve original unsigned 1/100000 protocol units for the checked cache.
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
        let parsed = parse_spot_event_loose(payload, 42, &symbols).expect("parsed");
        assert_eq!(parsed.0, 1);
        assert_eq!(parsed.1, Some(108_500));
        assert_eq!(parsed.2, Some(108_520));
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
        assert!(parse_spot_event_loose(payload, 42, &symbols).is_none());
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
        assert!(parse_spot_event_loose(payload, 42, &symbols).is_none());
    }

    #[test]
    fn handshake_requires_matching_type_and_exact_account_before_success() {
        let request = build_account_auth_request(42, "synthetic-test-token", "expected");
        let expected = expected_response_payload_type(request.payload_type).unwrap();
        let response = |payload_type, message_id: &str, account: serde_json::Value| {
            parse_open_api_envelope(
                &serde_json::json!({
                    "clientMsgId": message_id,
                    "payloadType": payload_type,
                    "payload": {"ctidTraderAccountId": account}
                })
                .to_string(),
            )
            .unwrap()
        };
        assert!(
            handshake_response_matches(
                &response(expected, "expected", 42.into()),
                "expected",
                expected,
                Some(42)
            )
            .unwrap()
        );
        assert!(
            !handshake_response_matches(
                &response(expected, "other", 42.into()),
                "expected",
                expected,
                Some(42)
            )
            .unwrap()
        );
        assert!(
            handshake_response_matches(
                &response(CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE, "expected", 42.into()),
                "expected",
                expected,
                Some(42)
            )
            .is_err()
        );
        for account in [
            serde_json::Value::Null,
            99.into(),
            0.into(),
            (-1).into(),
            "42".into(),
            serde_json::json!(42.5),
            serde_json::json!(u64::MAX),
        ] {
            assert!(
                handshake_response_matches(
                    &response(expected, "expected", account),
                    "expected",
                    expected,
                    Some(42)
                )
                .is_err()
            );
        }
        let missing = parse_open_api_envelope(
            &serde_json::json!({
                "clientMsgId":"expected", "payloadType":expected, "payload":{}
            })
            .to_string(),
        )
        .unwrap();
        assert!(handshake_response_matches(&missing, "expected", expected, Some(42)).is_err());
        for message in [
            build_application_auth_request("synthetic-client", "synthetic-secret", "application"),
            build_subscribe_spots_request(42, &[1], true, "subscribe"),
        ] {
            let expected_type = expected_response_payload_type(message.payload_type).unwrap();
            let expected_account = message
                .payload
                .get("ctidTraderAccountId")
                .and_then(serde_json::Value::as_i64);
            let good = parse_open_api_envelope(&serde_json::json!({
                "clientMsgId":message.client_msg_id, "payloadType":expected_type,
                "payload":expected_account.map_or_else(|| serde_json::json!({}), |account| serde_json::json!({"ctidTraderAccountId":account}))
            }).to_string()).unwrap();
            assert!(
                handshake_response_matches(
                    &good,
                    &message.client_msg_id,
                    expected_type,
                    expected_account
                )
                .unwrap()
            );
            if expected_account.is_some() {
                assert!(
                    handshake_response_matches(
                        &response(expected_type, &message.client_msg_id, 99.into()),
                        &message.client_msg_id,
                        expected_type,
                        expected_account
                    )
                    .is_err()
                );
                assert_eq!(message.payload["subscribeToSpotTimestamp"], true);
            }
        }
    }

    #[test]
    fn handshake_propagates_broker_error_and_disconnect_without_authenticating() {
        for payload_type in [
            CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE,
            CTRADER_OA_ACCOUNTS_TOKEN_INVALIDATED_EVENT_PAYLOAD_TYPE,
            CTRADER_OA_CLIENT_DISCONNECT_EVENT_PAYLOAD_TYPE,
            CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
        ] {
            let response = parse_open_api_envelope(
                &serde_json::json!({
                    "clientMsgId":"unrelated",
                    "payloadType":payload_type,
                    "payload":{"errorCode":"SYNTHETIC_REFUSAL","description":"test"}
                })
                .to_string(),
            )
            .unwrap();
            let error =
                handshake_response_matches(&response, "expected", 2103, Some(42)).unwrap_err();
            if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                assert!(error.to_string().contains("SYNTHETIC_REFUSAL"));
            }
        }
    }

    #[test]
    fn spot_parser_requires_wire_account_and_preserves_unsigned_protocol_sides() {
        let symbols = [StreamedSymbol {
            symbol_id: 1,
            symbol_name: "EURUSD".to_owned(),
            digits: 2,
        }];
        let mut message = serde_json::json!({
            "payloadType":CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE,
            "payload":{"ctidTraderAccountId":42,"symbolId":1,"bid":108521,"ask":108531,"timestamp":1000}
        });
        let parsed = parse_spot_event_loose(&message.to_string(), 42, &symbols).unwrap();
        assert_eq!((parsed.1, parsed.2), (Some(108521), Some(108531)));
        for account in [
            serde_json::Value::Null,
            99.into(),
            0.into(),
            (-1).into(),
            "42".into(),
            serde_json::json!(42.5),
            serde_json::json!(u64::MAX),
        ] {
            message["payload"]["ctidTraderAccountId"] = account;
            assert!(parse_spot_event_loose(&message.to_string(), 42, &symbols).is_none());
        }
        message["payload"]
            .as_object_mut()
            .unwrap()
            .remove("ctidTraderAccountId");
        assert!(parse_spot_event_loose(&message.to_string(), 42, &symbols).is_none());
        message["payload"]["ctidTraderAccountId"] = 42.into();
        message["payload"]["bid"] = serde_json::json!(u64::MAX);
        message["payload"].as_object_mut().unwrap().remove("ask");
        let unsigned = parse_spot_event_loose(&message.to_string(), 42, &symbols).unwrap();
        assert_eq!(
            unsigned.1,
            Some(u64::MAX),
            "checked cache refuses oversized raw prices; no signed wrap"
        );
        assert_eq!(unsigned.2, None);
        message["payload"].as_object_mut().unwrap().remove("bid");
        assert!(parse_spot_event_loose(&message.to_string(), 42, &symbols).is_none());
    }

    #[test]
    fn older_preparation_finishing_last_cannot_adopt_the_newer_generation() {
        let generation = AtomicU64::new(0);
        let invalidations = std::cell::RefCell::new(Vec::new());
        let launches = std::cell::RefCell::new(Vec::new());
        let config = |name: &str| LiveSpotsStreamerConfig {
            endpoint_host: "demo.ctraderapi.com".to_owned(),
            client_id: String::new(),
            client_secret: String::new(),
            access_token: String::new(),
            account_id: 42,
            symbols: vec![StreamedSymbol {
                symbol_id: 1,
                symbol_name: name.to_owned(),
                digits: 5,
            }],
        };

        // A starts and reserves generation 1 before blocking preparation.
        let older_generation = reserve_stream_generation(&generation, |reserved| {
            invalidations.borrow_mut().push(reserved);
        });
        let older_config = config("OLDER");
        // A second logical start must reserve 2, not reuse 1; it finishes first.
        let newer_generation = reserve_stream_generation(&generation, |reserved| {
            invalidations.borrow_mut().push(reserved);
        });
        assert_eq!((older_generation, newer_generation), (1, 2));
        assert_eq!(*invalidations.borrow(), vec![1, 2]);
        assert!(finish_preparation_for_generation(
            &generation,
            newer_generation,
            config("NEWER"),
            |ready, captured| {
                launches
                    .borrow_mut()
                    .push((ready.symbols[0].symbol_name.clone(), captured));
                true
            },
        ));
        // The actual preparation completion seam must refuse A before launch,
        // rather than relabelling its stale config as generation 2.
        assert!(!finish_preparation_for_generation(
            &generation,
            older_generation,
            older_config,
            |_, _| panic!("superseded preparation must never launch"),
        ));
        assert_eq!(*launches.borrow(), vec![("NEWER".to_owned(), 2)]);
        assert_eq!(generation.load(Relaxed), 2);
        // A later start also invalidates the prior generation when its launcher
        // refuses; it cannot leave the old loop authoritative.
        let failed_generation = reserve_stream_generation(&generation, |reserved| {
            invalidations.borrow_mut().push(reserved);
        });
        assert!(!finish_preparation_for_generation(
            &generation,
            failed_generation,
            config("FAILED"),
            |_, _| false,
        ));
        assert_eq!(failed_generation, 3);
        assert_eq!(*invalidations.borrow(), vec![1, 2, 3]);
    }

    #[test]
    fn stream_environment_comes_only_from_the_actual_supported_endpoint() {
        assert_eq!(
            stream_environment("demo.ctraderapi.com").unwrap(),
            CTraderEnvironment::Demo
        );
        assert_eq!(
            stream_environment("live.ctraderapi.com").unwrap(),
            CTraderEnvironment::Live
        );
        assert!(stream_environment("demo.ctraderapi.com.attacker.invalid").is_err());
        assert!(stream_environment("").is_err());
    }
}

#[cfg(test)]
#[path = "live_spots_subscription_tests.rs"]
mod subscription_tests;
