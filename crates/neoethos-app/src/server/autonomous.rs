//! `POST /autonomous/replay` — offline dry-run of the autonomous-trader engine
//! over on-disk history.
//!
//! Returns the SAME `EngineStats` the CLI `trader-replay` command prints, from
//! the SAME `neoethos_trader::replay_symbol_from_dir` helper — so the two
//! front-ends are byte-identical (the UI↔CLI parity mandate, applied to the
//! trader from day one). ZERO broker calls: the engine runs the mock execution
//! adapter over replayed bars.
//!
//! Symbol/base resolve through the shared `SystemConfig` resolvers — exactly as
//! `/engines/discovery/start` does — so an omitted field defaults from
//! `config.yaml` identically to the CLI.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use neoethos_core::Settings;

use crate::app_services::live_trading::{LiveTradingStatus, StartRequest};
use super::errors::actionable_error;
use super::state::AppApiState;

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct ReplayBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

pub async fn replay(State(state): State<AppApiState>, body: Option<Json<ReplayBody>>) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();

    let settings = Settings::from_yaml(state.config_path()).ok();
    let symbol = body
        .symbol
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|s| s.system.resolve_symbol()))
        .unwrap_or_default();
    let base_tf = body
        .base_tf
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|s| s.system.resolve_base_timeframe()))
        .unwrap_or_default();

    if symbol.is_empty() || base_tf.is_empty() {
        return actionable_error(
            StatusCode::BAD_REQUEST,
            "Replay can't run — no symbol / base timeframe was supplied and config.yaml \
             couldn't provide a default. Set them in Settings or include them in the request.",
            &anyhow::anyhow!("symbol='{symbol}' base_tf='{base_tf}'"),
        );
    }

    let data_dir = match settings.as_ref() {
        Some(s) => s.system.data_dir.clone(),
        None => {
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Replay can't run — config.yaml couldn't be read to locate the data folder.",
                &anyhow::anyhow!("settings unavailable"),
            );
        }
    };

    // The replay reads + crunches a whole history synchronously — keep it off the
    // async runtime's worker threads.
    let sym = symbol.clone();
    let base = base_tf.clone();
    let result = tokio::task::spawn_blocking(move || {
        neoethos_trader::replay_symbol_from_dir(
            &data_dir,
            &sym,
            &base,
            neoethos_trader::EngineConfig::default(),
        )
    })
    .await;

    match result {
        Ok(Ok(stats)) => Json(stats).into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Replay failed — make sure the data folder has this symbol + base timeframe \
             (run Data Bootstrap or import a file first).",
            &err,
        ),
        Err(join_err) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The replay task panicked.",
            &anyhow::anyhow!("{join_err}"),
        ),
    }
}

// ── Live autonomous trading ───────────────────────────────────────────────────

/// `POST /autonomous/start` — begin live trading from a discovered portfolio.
///
/// Body: `StartRequest` JSON (portfolio_path required; lot_size, sl/tp optional).
/// Returns 409 if already running, 200 with the initial status on success.
/// HTTP body for `/autonomous/start`. Accepts EITHER a single `portfolio_path`
/// or a list `portfolio_paths` — run several discovered strategies at once, each
/// in its own concurrent (internally multi-timeframe) engine. Sizing / SL / TP /
/// warmup apply to every started engine.
#[derive(Debug, serde::Deserialize)]
pub struct StartLiveBody {
    pub portfolio_path: Option<String>,
    #[serde(default)]
    pub portfolio_paths: Vec<String>,
    #[serde(default = "crate::app_services::live_trading::default_lot_size")]
    pub lot_size: f64,
    pub stop_loss_pips: Option<f64>,
    pub take_profit_pips: Option<f64>,
    #[serde(default = "crate::app_services::live_trading::default_warmup_bars")]
    pub warmup_bars: usize,
    /// Auto-cull: retire a strategy after this many consecutive losing trades
    /// (0 disables). Applies to every engine started in this request.
    #[serde(default = "crate::app_services::live_trading::default_cull_losses")]
    pub cull_after_consecutive_losses: u32,
}

/// Aggregate status across every running engine.
fn live_overview(handles: &[crate::app_services::live_trading::Handle]) -> serde_json::Value {
    let engines: Vec<LiveTradingStatus> = handles.iter().map(|h| h.snapshot()).collect();
    serde_json::json!({
        "running": engines.iter().any(|e| e.running),
        "engineCount": engines.len(),
        "engines": engines,
    })
}

pub async fn start_live(
    State(state): State<AppApiState>,
    Json(body): Json<StartLiveBody>,
) -> Response {
    // Resolve the set of portfolios to run (list + optional single, de-duped).
    let mut paths: Vec<String> = body.portfolio_paths.clone();
    if let Some(p) = body.portfolio_path.clone() {
        paths.push(p);
    }
    paths.retain(|p| !p.trim().is_empty());
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "no portfolio_path / portfolio_paths provided"
            })),
        )
            .into_response();
    }

    let mut slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };
    // Keep only still-running engines so the registry reflects reality.
    slot.retain(|h| h.is_running());
    let already: std::collections::HashSet<String> = slot
        .iter()
        .filter_map(|h| h.snapshot().portfolio_path)
        .collect();

    let mut started: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut blacklisted: Vec<String> = Vec::new();
    let mut failed: Vec<serde_json::Value> = Vec::new();
    for path in paths {
        if already.contains(&path) {
            skipped.push(path);
            continue;
        }
        // Auto-cull enforcement: a retired strategy can NEVER be traded again.
        if crate::app_services::strategy_blacklist::is_blacklisted(&path) {
            blacklisted.push(path);
            continue;
        }
        let req = StartRequest {
            portfolio_path: path.clone(),
            lot_size: body.lot_size,
            stop_loss_pips: body.stop_loss_pips,
            take_profit_pips: body.take_profit_pips,
            warmup_bars: body.warmup_bars,
            cull_after_consecutive_losses: body.cull_after_consecutive_losses,
        };
        match crate::app_services::live_trading::start(req) {
            Ok(handle) => {
                started.push(path);
                slot.push(handle);
            }
            Err(e) => {
                failed.push(serde_json::json!({"portfolio": path, "error": e.to_string()}))
            }
        }
    }

    let code = if started.is_empty() && !failed.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    (
        code,
        Json(serde_json::json!({
            "started": started,
            "skipped": skipped,
            "blacklisted": blacklisted,
            "failed": failed,
            "overview": live_overview(&slot),
        })),
    )
        .into_response()
}

/// `GET /strategy/blacklist` — the permanent list of auto-retired strategies.
/// These can never be selected for live trading or re-surfaced by discovery.
pub async fn blacklist() -> Response {
    Json(crate::app_services::strategy_blacklist::load()).into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct ParityQuery {
    pub portfolio: String,
    /// Live-style window size (defaults to the live engine's 1000).
    pub window: Option<usize>,
    /// Long-history reference size (default 3000).
    pub reference: Option<usize>,
}

/// `GET /autonomous/parity?portfolio=..&window=1000&reference=3000` — the
/// live↔backtest parity harness: does the live 1000-bar window produce the
/// SAME signals as a long-history computation for this portfolio? FAIL means
/// live trading cannot match the validated backtest (warmup-sensitive
/// features) — fix BEFORE trusting live results.
pub async fn parity(Query(q): Query<ParityQuery>) -> Response {
    let portfolio = q.portfolio;
    let window = q.window.unwrap_or(crate::app_services::live_trading::default_warmup_bars());
    let reference = q.reference.unwrap_or(3000);
    let result = tokio::task::spawn_blocking(move || {
        crate::app_services::live_parity::run_live_parity_check(&portfolio, window, reference)
    })
    .await;
    match result {
        Ok(Ok(report)) => Json(report).into_response(),
        Ok(Err(e)) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Parity check failed — make sure the portfolio path exists and the broker \
             connection is alive (it fetches recent bars).",
            &e,
        ),
        Err(join_err) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The parity task panicked.",
            &anyhow::anyhow!("{join_err}"),
        ),
    }
}

/// `POST /autonomous/stop` — gracefully stop ALL running live engines.
pub async fn stop_live(State(state): State<AppApiState>) -> Response {
    let mut slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };
    let count = slot.len();
    for handle in slot.iter() {
        handle.stop();
    }
    slot.clear();
    (
        StatusCode::OK,
        Json(serde_json::json!({"stopped": count > 0, "enginesStopped": count})),
    )
        .into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct GateQuery {
    pub portfolio: String,
}

/// `GET /autonomous/gate?portfolio=...` — demo forward-test eligibility for a
/// portfolio, so the UI can show WHY live is (not) yet allowed BEFORE the
/// operator clicks Start. `enforced` is true only on a Live (real-money) env;
/// on Demo the gate is informational (eligibility still tracked, never blocks).
pub async fn gate(Query(q): Query<GateQuery>) -> Response {
    let portfolio = q.portfolio;
    let env_is_live = crate::app_services::live_gate::active_env_is_live();
    let result = tokio::task::spawn_blocking(move || {
        crate::app_services::live_gate::evaluate_for_portfolio(&portfolio)
    })
    .await;
    match result {
        Ok(Ok(decision)) => Json(serde_json::json!({
            "envIsLive": env_is_live,
            "enforced": env_is_live,
            "eligible": decision.eligible,
            "summary": decision.summary,
            "criteria": decision.criteria,
        }))
        .into_response(),
        Ok(Err(e)) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Couldn't evaluate the demo forward-test gate — make sure the portfolio path \
             and its sibling *.quality.json exist.",
            &e,
        ),
        Err(join_err) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The gate evaluation task panicked.",
            &anyhow::anyhow!("{join_err}"),
        ),
    }
}

/// `GET /autonomous/status` — poll all live engines. Returns
/// `{ running, engineCount, engines: [LiveTradingStatus...] }`.
pub async fn live_status(State(state): State<AppApiState>) -> Response {
    let mut slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };
    slot.retain(|h| h.is_running());
    Json(live_overview(&slot)).into_response()
}
