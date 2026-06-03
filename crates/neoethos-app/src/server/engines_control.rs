//! Control endpoints for the Discovery and Training engines.
//!
//! POST /engines/discovery/start  — kick off a discovery job
//! POST /engines/discovery/stop   — request cancellation
//! POST /engines/training/start   — kick off a training job
//! POST /engines/training/stop    — request cancellation
//!
//! Each engine has at most one in-flight job at a time. Starting while
//! one is already running returns 409 Conflict. Stopping when nothing is
//! running returns 200 with `{"running": false}` — idempotent.
//!
//! Engine state ("Idle" / "Running" / "Failed: …" / "Succeeded") is
//! tracked through a `EngineSlot` held inside `AppApiState`. The
//! background task that drives each job drains the `ServiceEvent`
//! channel and writes the latest `JobState` back into the slot, which
//! `/engines/status` then reads.

use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::app_services::ServiceEvent;
use crate::app_services::discovery::{DiscoveryRequest, start_discovery_job};
use crate::app_services::jobs::{JobKind, JobState};
use crate::app_services::training::{TrainingRequest, start_training_job};

use super::errors::actionable_error;
use super::state::AppApiState;

/// Shared request body for `start` endpoints — picks the symbol +
/// timeframe to operate on. Empty fields fall back to "EURUSD" / "M1"
/// so the dashboard "Start" button can fire without any params.
///
/// `higher_tfs` is the MTF context discovery considers alongside
/// `base_tf`. When omitted, falls back to [`DEFAULT_HIGHER_TFS`]. The
/// dashboard exposes this as a comma-separated text field; the
/// canonical wire form is a JSON array of canonical timeframe labels
/// (`["M5", "M15", "H1"]`).
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct StartJobBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
    pub higher_tfs: Option<Vec<String>>,
    /// #194: optional GA hyperparameter overrides. When `None` the
    /// engine uses the defaults baked into
    /// `neoethos_search::DiscoveryConfig::default()`; sending any field
    /// here replaces only that knob. The UI's "Advanced" expander
    /// builds this struct from the operator's sliders.
    pub population: Option<usize>,
    pub generations: Option<usize>,
    pub max_indicators: Option<usize>,
    pub target_candidates: Option<usize>,
    pub portfolio_size: Option<usize>,
}

/// Default MTF context for discovery when the caller does not supply
/// one. Mirrors the dashboard preset that ships in `DiscoveryFormState`
/// in `app_state.rs` so the server-driven start and the UI-driven
/// start operate on the same context unless the caller explicitly
/// overrides.
pub const DEFAULT_HIGHER_TFS: &[&str] = &["M5", "M15", "H1"];

#[derive(Debug, serde::Serialize)]
pub struct StartResponse {
    pub started: bool,
    pub kind: &'static str,
    pub symbol: String,
    pub base_tf: String,
}

#[derive(Debug, serde::Serialize)]
pub struct StopResponse {
    pub running: bool,
    pub kind: &'static str,
}

// ─── Discovery ────────────────────────────────────────────────────────────

pub async fn discovery_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let symbol = body
        .symbol
        .unwrap_or_else(|| "EURUSD".to_string())
        .trim()
        .to_uppercase();
    let base_tf = body
        .base_tf
        .unwrap_or_else(|| "M1".to_string())
        .trim()
        .to_uppercase();
    let higher_tfs: Vec<String> = body
        .higher_tfs
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            DEFAULT_HIGHER_TFS
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        })
        .into_iter()
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();

    if state.engine_state(JobKind::Discovery).await == EngineRunState::Running {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "discovery already running — stop the current job first",
            })),
        )
            .into_response();
    }

    let data_root = match resolve_data_root().await {
        Ok(p) => p,
        Err(err) => {
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Discovery can't start — config.yaml couldn't be loaded. \
                 Check the data directory path in Settings, then try again.",
                &err,
            );
        }
    };

    // #153: pre-flight gate. A bare-install user clicks "Start Discovery"
    // before running Data Bootstrap; discovery then runs for ~2 seconds
    // before bailing with a deep "no matching files" panic that the
    // dashboard surfaces as a useless "crash". Catch the empty / missing
    // data root here and explain what to do instead.
    if let Err(err) = preflight_discovery_data_root(&data_root, &symbol, &base_tf).await {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": err.to_string(),
                "code": "discovery_no_data",
                "data_root": data_root.display().to_string(),
                "hint": "Run Data Bootstrap or import a CSV/Parquet \
                         file for this symbol + base timeframe first.",
            })),
        )
            .into_response();
    }

    // #194: stitch operator overrides into the default DiscoveryConfig.
    // Anything the body omits stays at the engine default (which itself
    // pulls from config.yaml).
    //
    // F-304 fix (2026-05-28): seed the config via `from_settings` rather
    // than `default()` so `evaluation_symbol` + `evaluation_account_currency`
    // arrive populated from `Settings.system.*`, then explicitly override
    // the symbol with the request-body value. Without this seed, every
    // /engines/discovery/start request ran with empty symbol + empty
    // account_currency + NaN spread/commission, tripping the cost-model
    // NaN guard and producing zero-trade GA candidates that the
    // sanitizer scrubbed to 0.0 — invisible failure mode.
    let mut config = match Settings::from_yaml(state.config_path()) {
        Ok(settings) => neoethos_search::DiscoveryConfig::from_settings(&settings),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::engines_control",
                error = %err,
                config_path = %state.config_path().display(),
                "failed to load Settings; falling back to DiscoveryConfig::default() \
                 (evaluation_symbol/account_currency will be empty — discovery will \
                 fail at the cost-model NaN guard until config.yaml is fixed)"
            );
            neoethos_search::DiscoveryConfig::default()
        }
    };
    // Body-supplied symbol always wins (operator picked it on the UI).
    config.evaluation_symbol = symbol.clone();
    // Account currency comes from Settings (loaded into config by
    // from_settings). An empty value propagates to the guard, which bails with
    // an actionable error. Config is the single source — the legacy
    // `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env fallback was removed in v0.4.36.
    if let Some(p) = body.population.filter(|&p| p > 0) {
        config.population = p;
    }
    if let Some(g) = body.generations.filter(|&g| g > 0) {
        config.generations = g;
    }
    if let Some(m) = body.max_indicators.filter(|&m| m > 0) {
        config.max_indicators = m;
    }
    if let Some(t) = body.target_candidates.filter(|&t| t > 0) {
        config.candidate_count = t;
    }
    if let Some(s) = body.portfolio_size.filter(|&s| s > 0) {
        config.portfolio_size = s;
    }

    // F-314 fix (2026-05-29): wire env runtime overrides into the
    // UI-driven Discovery path. Until today, only `neoethos-cli`
    // (`main.rs:487`) called this helper; the UI route silently ran
    // in accidental "Strict" mode with `prop_firm_gate=None`, no
    // F-305 timeframe-aware `min_trades_per_month` scaling, and no
    // F-277 adaptive threshold ladder — hitting the legacy walk-
    // forward consistency gate that rejects almost every candidate.
    // The visible symptom: every UI Discovery run produced an empty
    // portfolio while CLI runs found strategies for the developer.
    //
    // Apply this AFTER the body-supplied GA knob overrides above so
    // the operator's UI-tuned population/generations/etc. survive,
    // and AFTER the `evaluation_symbol`/`evaluation_account_currency`
    // mode-dependent overrides (config-driven mode from
    // models.discovery_mode + the prop-firm window-pass gate) so they see
    // the final config — not the from-yaml-defaults version.
    config = config.apply_mode_overrides();

    let request = DiscoveryRequest {
        data_root,
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
        higher_tfs: higher_tfs.clone(),
        config,
        prop_firm_rules: neoethos_search::PropFirmRiskRules::default(),
    };

    let (tx, rx) = mpsc::channel::<ServiceEvent>(1000);
    let handle = match start_discovery_job(request, tx) {
        Ok(h) => h,
        Err(err) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Discovery failed to start. Make sure a symbol and timeframe are selected, \
                 then try again.",
                &err,
            );
        }
    };

    state
        .install_engine(JobKind::Discovery, handle.cancel.clone())
        .await;
    // The fourth arg arms the auto-chain: when this discovery run
    // hits a terminal "Succeeded" state, the drainer fires
    // `start_training_job` with the same (symbol, base_tf) — that's
    // the "natural sequence" the user explicitly asked for
    // (discovery → training → trading). Skipped if the user is
    // already running training manually when discovery finishes.
    spawn_state_drainer(
        state.clone(),
        JobKind::Discovery,
        rx,
        Some((symbol.clone(), base_tf.clone())),
    );

    Json(StartResponse {
        started: true,
        kind: "discovery",
        symbol,
        base_tf,
    })
    .into_response()
}

pub async fn discovery_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Discovery).await;
    Json(StopResponse {
        running,
        kind: "discovery",
    })
}

// ─── Training ─────────────────────────────────────────────────────────────

pub async fn training_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let symbol = body
        .symbol
        .unwrap_or_else(|| "EURUSD".to_string())
        .trim()
        .to_uppercase();
    let base_tf = body
        .base_tf
        .unwrap_or_else(|| "M1".to_string())
        .trim()
        .to_uppercase();

    if state.engine_state(JobKind::Training).await == EngineRunState::Running {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "training already running — stop the current job first",
            })),
        )
            .into_response();
    }

    let request = TrainingRequest {
        config_path: "config.yaml".to_string(),
        models_dir: PathBuf::from("models"),
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
    };

    let (tx, rx) = mpsc::channel::<ServiceEvent>(1000);
    let handle = match start_training_job(request, tx) {
        Ok(h) => h,
        Err(err) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Training failed to start. Make sure Discovery finished for this \
                 symbol/timeframe and model_targets.json exists in the models folder.",
                &err,
            );
        }
    };

    state
        .install_engine(JobKind::Training, handle.cancel.clone())
        .await;
    // Training has no further auto-chain step yet (the auto-trader
    // wiring lands in a follow-up), so the drainer gets None for the
    // chain arg — a Succeeded training run leaves the operator on
    // the dashboard, not in autonomous mode.
    spawn_state_drainer(state.clone(), JobKind::Training, rx, None);

    Json(StartResponse {
        started: true,
        kind: "training",
        symbol,
        base_tf,
    })
    .into_response()
}

pub async fn training_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Training).await;
    Json(StopResponse {
        running,
        kind: "training",
    })
}

// ─── shared helpers ───────────────────────────────────────────────────────

/// Where the engines pull their input data from. Mirrors backend startup
/// wiring: load the CLI-configured `config.yaml` (see
/// `server::state::install_config_path`), then take `system.data_dir`
/// from it.
async fn resolve_data_root() -> Result<PathBuf> {
    // F-553/F-576 closure (2026-05-25): resolved via the process-wide
    // install so a non-default `--config` flag still works.
    let config_path = super::state::current_config_path();
    tokio::task::spawn_blocking(move || {
        let settings = Settings::from_yaml(&config_path)
            .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
        Ok(settings.system.data_dir)
    })
    .await
    .map_err(|e| anyhow::anyhow!("blocking task panicked: {e}"))?
}

/// #153 pre-flight: refuse to start discovery if the data root is
/// missing, unreadable, or contains zero files for the requested
/// `symbol`+`base_tf`. The deep-stack failure that motivated this
/// gate was a `panic!("no matching files")` two layers below
/// `start_discovery_job` after ~2s — fast enough to look like a
/// crash, slow enough that the user thought the engine "broke".
///
/// #203 refactor: the original v1 only scanned the TOP LEVEL of the
/// data directory for filenames containing both the symbol and the
/// timeframe strings. That gave a false negative for the actual
/// hive-style layout the rest of the codebase uses
/// (`data/symbol=EURUSD/timeframe=H1/data.vortex`) — the only
/// top-level entry is `symbol=EURUSD` which doesn't contain "H1".
/// Now we delegate to the data layer's `discover_timeframes` which
/// already understands the hive layout and is cached per #79.
async fn preflight_discovery_data_root(
    data_root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> Result<()> {
    if !data_root.exists() {
        anyhow::bail!(
            "data directory does not exist: {} (configured via \
             config.yaml `system.data_dir`)",
            data_root.display()
        );
    }
    if !data_root.is_dir() {
        anyhow::bail!("data directory is not a directory: {}", data_root.display());
    }

    // Empty-directory branch: distinguish "user hasn't run Data
    // Bootstrap at all" from "user ran it for a different symbol".
    let mut entries = tokio::fs::read_dir(data_root)
        .await
        .map_err(|e| anyhow::anyhow!("cannot read data directory {}: {e}", data_root.display()))?;
    let mut total = 0usize;
    while let Some(_entry) = entries.next_entry().await? {
        total += 1;
    }
    if total == 0 {
        anyhow::bail!(
            "data directory is empty: {} — run Data Bootstrap or import \
             a CSV/Parquet file first",
            data_root.display()
        );
    }

    let symbol_up = symbol.to_uppercase();
    let base_tf_up = base_tf.to_uppercase();
    let data_root_owned = data_root.to_path_buf();
    let symbol_for_blocking = symbol_up.clone();
    // `discover_timeframes` walks `symbol=*/timeframe=*/` and does
    // some std::fs work; run it on the blocking pool so the async
    // runtime stays responsive even when the data dir is on a slow
    // network drive.
    let discovered: Vec<String> = tokio::task::spawn_blocking(move || {
        neoethos_data::discover_timeframes(&data_root_owned, &symbol_for_blocking)
            .unwrap_or_default()
    })
    .await
    .map_err(|e| anyhow::anyhow!("timeframe-discovery task panicked: {e}"))?;

    if discovered.is_empty() {
        anyhow::bail!(
            "no data on disk for {} in {} — import OHLCV for this \
             symbol first (data dir has {} top-level entries but \
             none under symbol={})",
            symbol_up,
            data_root.display(),
            total,
            symbol_up,
        );
    }
    if !discovered.iter().any(|tf| tf.eq_ignore_ascii_case(&base_tf_up)) {
        anyhow::bail!(
            "{} is on disk but timeframe {} is missing — available: {} ({})",
            symbol_up,
            base_tf_up,
            discovered.join(", "),
            data_root.display(),
        );
    }
    Ok(())
}

/// Spawn a background task that drains the ServiceEvent rx channel
/// emitted by the job and reflects the latest `JobState` into the
/// `AppApiState` engine slot. The task exits when the channel closes
/// (job's send end dropped after terminal event).
///
/// `auto_chain_args` is `Some((symbol, base_tf))` for Discovery only —
/// when discovery terminates with `Succeeded`, the drainer fires
/// `start_training_job` against the same pair. That's the
/// "natural sequence" the operator expects:
///
///     Discovery (GA-evolves a portfolio)
///        ↓ writes model_targets.json
///     Training (34-model ensemble fits per model_targets.json)
///        ↓ writes models/*.{pkl,joblib,pt}
///     (Auto-Trader — lands in a follow-up)
///
/// Auto-chain is suppressed if the user already started Training
/// manually before Discovery finishes (Training is single-job:
/// `state.engine_state(Training)` would be Running). Failed,
/// Cancelled, or Degraded discoveries also skip the chain — we only
/// promote a clean Success.
fn spawn_state_drainer(
    state: AppApiState,
    kind: JobKind,
    mut rx: mpsc::Receiver<ServiceEvent>,
    auto_chain_args: Option<(String, String)>,
) {
    tokio::spawn(async move {
        let mut terminal_state: Option<JobState> = None;
        while let Some(event) = rx.recv().await {
            let snapshot = match (&event, kind) {
                (ServiceEvent::DiscoveryUpdated(s), JobKind::Discovery) => Some(s),
                (ServiceEvent::TrainingUpdated(s), JobKind::Training) => Some(s),
                _ => None,
            };
            let Some(snap) = snapshot else { continue };
            terminal_state = Some(snap.state);
            let run_state = EngineRunState::from(snap.state);
            state
                .update_engine(kind, run_state, snap.report.summary.clone())
                .await;
            // F-340 (Feature #14): mirror the live discovery/training
            // progress (stage + percent + counters) from the same
            // JobSnapshot into the slot so `/engines/status` can expose
            // the rich counters the GA loop accumulates. Only do this
            // while the engine is still Running — `update_engine` wipes
            // the progress on any terminal state, and re-populating it
            // here would leave a stale "search_generations / 0.83"
            // line hanging around after the run finished.
            if matches!(run_state, EngineRunState::Running) {
                // `JobProgress::percent` is `Option<f32>` in 0.0..=1.0;
                // default to 0.0 when the job hasn't reported a fraction
                // yet. `JobReport::counters` is already a `(name, u64)`
                // list — forward it verbatim.
                let percent = snap.progress.percent.unwrap_or(0.0) as f64;
                state
                    .set_engine_progress(
                        kind,
                        snap.progress.stage.clone(),
                        percent,
                        snap.report.counters.clone(),
                    )
                    .await;
            }
        }
        // Channel closed — make sure we don't leave a dangling
        // "Running" state if the producer side dropped without a
        // terminal event (shouldn't happen, defensive guard).
        state.finalize_engine_if_running(kind).await;

        // Auto-chain Discovery → Training when:
        //   1. We're the discovery drainer (Some auto_chain_args).
        //   2. Discovery succeeded (Degraded counts as success in
        //      EngineRunState but Training needs the strictly-clean
        //      `model_targets.json` from a Succeeded run).
        //   3. Training isn't already running (idempotency — the user
        //      might have hit Train manually while Discovery was
        //      still grinding).
        if let Some((symbol, base_tf)) = auto_chain_args {
            if matches!(terminal_state, Some(JobState::Succeeded)) {
                let already_training = matches!(
                    state.engine_state(JobKind::Training).await,
                    EngineRunState::Running
                );
                if already_training {
                    tracing::info!(
                        target: "neoethos_app::server::engines_control",
                        "Discovery succeeded but Training is already \
                         running — skipping auto-chain to avoid 409"
                    );
                } else {
                    tracing::info!(
                        target: "neoethos_app::server::engines_control",
                        symbol = %symbol,
                        base_tf = %base_tf,
                        "Discovery succeeded — auto-chaining Training \
                         on the same (symbol, base_tf) per natural \
                         pipeline sequence"
                    );
                    spawn_auto_chained_training(state, symbol, base_tf);
                }
            } else {
                tracing::info!(
                    target: "neoethos_app::server::engines_control",
                    ?terminal_state,
                    "Discovery did NOT succeed cleanly — skipping \
                     auto-chain. Operator can re-trigger Discovery \
                     or start Training manually."
                );
            }
        }
    });
}

/// Helper: kick off a Training job from inside the Discovery drainer
/// (not an HTTP path), wiring up its own drainer with no further
/// auto-chain. Pulled out so the recursive shape stays readable.
fn spawn_auto_chained_training(state: AppApiState, symbol: String, base_tf: String) {
    let request = TrainingRequest {
        config_path: "config.yaml".to_string(),
        models_dir: PathBuf::from("models"),
        symbol,
        base_tf,
    };
    let (tx, rx) = mpsc::channel::<ServiceEvent>(1000);
    match start_training_job(request, tx) {
        Ok(handle) => {
            let state_for_install = state.clone();
            tokio::spawn(async move {
                state_for_install
                    .install_engine(JobKind::Training, handle.cancel.clone())
                    .await;
            });
            spawn_state_drainer(state, JobKind::Training, rx, None);
        }
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::engines_control",
                error = %err,
                "auto-chained Training failed to start — operator \
                 must launch it manually from the Training screen"
            );
        }
    }
}

// ─── EngineRunState (wire-friendly subset of JobState) ────────────────────

/// Compact engine state for `/engines/status`. We collapse Queued and
/// Running into the same "Running" label (the UI only cares whether
/// it should show a green dot + a "Stop" button), and Degraded into
/// Succeeded (still a terminal-OK outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineRunState {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl EngineRunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineRunState::Idle => "Idle",
            EngineRunState::Running => "Running",
            EngineRunState::Succeeded => "Succeeded",
            EngineRunState::Failed => "Failed",
            EngineRunState::Cancelled => "Cancelled",
        }
    }
}

impl From<JobState> for EngineRunState {
    fn from(value: JobState) -> Self {
        match value {
            JobState::Queued | JobState::Running => EngineRunState::Running,
            JobState::Succeeded | JobState::Degraded => EngineRunState::Succeeded,
            JobState::Failed => EngineRunState::Failed,
            JobState::Cancelled => EngineRunState::Cancelled,
        }
    }
}
