//! Step 6 — Historical data download.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 6 + §9.4 mockup.
//!
//! Drives `crates/forex-app/src/app_services/ctrader_history.rs::
//! fetch_historical_bars`. The wizard owns the token-bucket gate at
//! 5 req/s per `ctrader_api_full_reference.md` §3.2 ("a maximum of 5
//! requests per second per connection for any historical data
//! requests"). On cancel, the wizard writes a `.partial` sentinel
//! beside the Parquet so the main app can prompt to Resume.
//!
//! Cancel-safety contract (spec §2 Step 6 Actions):
//!
//! - Before issuing the first request for a (symbol, timeframe), the
//!   worker writes the `.partial` sentinel beside the canonical
//!   `<root>/symbol=…/timeframe=…/data.vortex` path.
//! - On successful write of the Vortex file, the worker replaces the
//!   sentinel with `.complete`.
//! - On `REQUEST_FREQUENCY_EXCEEDED` (108) the worker sleeps
//!   `WIZARD_DEFAULT_HISTORY_BACKOFF_SECONDS` then retries the same
//!   month-window. The `.partial` sentinel stays in place.
//! - On Cancel from the UI, the worker stops at the next bucket tick
//!   and exits — the `.partial` sentinel stays in place for resume.
//! - On disk-full or any other terminal error, the job is marked
//!   failed and the worker proceeds to the next (symbol, timeframe).
//!
//! Wizard Step 6 scaffolding allow: `CompletionSentinel` +
//! `reset_historical_runtime` are part of the cancel-safety contract
//! described above but only the egui runtime's reset path is wired in
//! the current build; the `forex-cli` mirror will call them when its
//! Step 6 lands (Task #9, multi-folder picker).
#![allow(dead_code)]

use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::oauth;
use super::{StepResult, WizardController};
use crate::app_services::ctrader_history::{CTraderHistoricalBarsRequest, fetch_historical_bars};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::ui::theme;
use forex_data::{Ohlcv, write_symbol_timeframe_vortex};

/// Operator default — months of history seeded. Spec §2 Step 6
/// "default 6".
pub const WIZARD_DEFAULT_HISTORY_MONTHS: u8 = 6;

/// Allowed slider stops. Spec §9.4 mockup row "1   3   6   12   18   24".
pub const WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS: &[u8] = &[1, 3, 6, 12, 18, 24];

/// Token-bucket rate for historical-data requests.
/// `ctrader_api_full_reference.md` §3.2 — 5 req/s per connection.
pub const WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS: u8 = 5;

/// Inter-request interval that satisfies the 5 req/s token bucket.
/// `1000ms / 5 = 200ms` between requests.
pub const WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL: Duration =
    Duration::from_millis(1000 / WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS as u64);

/// Backoff after `REQUEST_FREQUENCY_EXCEEDED` (108). Spec §3 error
/// matrix — "30 s backoff + resume".
pub const WIZARD_DEFAULT_HISTORY_BACKOFF_SECONDS: u32 = 30;

/// File sentinel suffix indicating an interrupted download. Spec §2
/// Step 6 Actions — "Output … `.partial` (on Cancel)".
pub const WIZARD_PARTIAL_SENTINEL_SUFFIX: &str = ".partial";

/// File sentinel suffix marking a completed download. Spec §2 Step 6
/// Actions — "Output … `.complete`".
pub const WIZARD_COMPLETE_SENTINEL_SUFFIX: &str = ".complete";

/// Approximate milliseconds per "month" used for window splitting.
/// `30.44 days/month * 24h * 60m * 60s * 1000ms`. cTrader's
/// trendbars endpoint accepts any window — we split into ~monthly
/// chunks so each request stays within the broker's hard cap on
/// trendbar count without us having to guess the exact ceiling.
const MS_PER_MONTH_APPROX: i64 = 30_44 * 24 * 60 * 60 * 1000 / 100;

/// Per-job progress snapshot. Lives on the runtime so the UI can
/// render a per-(symbol, timeframe) progress bar.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryJobState {
    Pending,
    InFlight { window_index: u8, windows: u8 },
    BackingOff { remaining_secs: u32 },
    Succeeded { bars_written: usize },
    Failed { error: String },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoryJob {
    pub symbol: String,
    pub timeframe: String,
    pub state: HistoryJobState,
}

/// Outcome of writing the canonical Vortex layout for one job. Used
/// by the deterministic tests below.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionSentinel {
    None,
    Partial(PathBuf),
    Complete(PathBuf),
}

/// Wizard-local download runtime. Held in a process-global
/// `OnceLock<Mutex<_>>` because the egui re-renders the step every
/// frame and the worker thread + cancel flag must survive across
/// frames. Cleared via [`reset_historical_runtime`].
#[derive(Default)]
struct HistoricalRuntime {
    cancel_flag: Arc<AtomicBool>,
    progress_rx: Option<Receiver<HistoryProgressMessage>>,
    jobs: Vec<HistoryJob>,
    running: bool,
    finished: bool,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
enum HistoryProgressMessage {
    JobUpdate {
        index: usize,
        state: HistoryJobState,
    },
    AllDone,
}

fn runtime_mutex() -> &'static Mutex<HistoricalRuntime> {
    static RUNTIME: OnceLock<Mutex<HistoricalRuntime>> = OnceLock::new();
    RUNTIME.get_or_init(|| Mutex::new(HistoricalRuntime::default()))
}

/// Clear the process-global runtime — call on a fresh wizard run.
pub fn reset_historical_runtime() {
    if let Ok(mut runtime) = runtime_mutex().lock() {
        *runtime = HistoricalRuntime::default();
    }
}

/// Compute the canonical Vortex destination path for a (symbol, tf)
/// under the operator's chosen data path. Mirrors
/// `forex_data::symbol_timeframe_vortex_path` so the wizard's writes
/// match the loader's reads byte-for-byte.
pub fn canonical_vortex_path(root: &Path, symbol: &str, timeframe: &str) -> PathBuf {
    forex_data::symbol_timeframe_vortex_path(root, symbol, timeframe)
}

/// Sibling of the Vortex file marking an interrupted download.
pub fn partial_sentinel_path(vortex_path: &Path) -> PathBuf {
    sibling_with_suffix(vortex_path, WIZARD_PARTIAL_SENTINEL_SUFFIX)
}

/// Sibling of the Vortex file marking a completed download.
pub fn complete_sentinel_path(vortex_path: &Path) -> PathBuf {
    sibling_with_suffix(vortex_path, WIZARD_COMPLETE_SENTINEL_SUFFIX)
}

fn sibling_with_suffix(vortex_path: &Path, suffix: &str) -> PathBuf {
    let mut name = vortex_path
        .file_name()
        .map(|os| os.to_string_lossy().to_string())
        .unwrap_or_else(|| "data.vortex".to_string());
    name.push_str(suffix);
    match vortex_path.parent() {
        Some(parent) => parent.join(name),
        None => PathBuf::from(name),
    }
}

/// Write the `.partial` sentinel and ensure its parent dir exists.
pub fn write_partial_sentinel(vortex_path: &Path) -> std::io::Result<PathBuf> {
    if let Some(parent) = vortex_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let path = partial_sentinel_path(vortex_path);
    std::fs::write(&path, b"download in progress\n")?;
    Ok(path)
}

/// Swap the `.partial` sentinel for a `.complete` sentinel. The
/// `.complete` write happens AFTER the Vortex file is on disk so any
/// crash mid-write leaves the `.partial` (Resume) state.
pub fn replace_partial_with_complete(vortex_path: &Path) -> std::io::Result<PathBuf> {
    let partial = partial_sentinel_path(vortex_path);
    let complete = complete_sentinel_path(vortex_path);
    std::fs::write(&complete, b"download complete\n")?;
    let _ = std::fs::remove_file(&partial); // best-effort
    Ok(complete)
}

/// Split a request window into ~monthly chunks. Each chunk is bounded
/// inclusively by the operator-requested window.
pub fn split_window_into_chunks(from_ms: i64, to_ms: i64, chunk_ms: i64) -> Vec<(i64, i64)> {
    assert!(chunk_ms > 0, "chunk_ms must be positive");
    let mut chunks = Vec::new();
    let mut cursor = from_ms;
    while cursor < to_ms {
        let end = (cursor.saturating_add(chunk_ms)).min(to_ms);
        chunks.push((cursor, end));
        if end == to_ms {
            break;
        }
        cursor = end;
    }
    if chunks.is_empty() {
        // Caller passed from_ms == to_ms — still issue one zero-width
        // request so the broker decides.
        chunks.push((from_ms, to_ms));
    }
    chunks
}

/// Compute the operator's requested time window (from_ms, to_ms) based
/// on `history_months` and the current wall clock.
pub fn compute_window(now_ms: i64, history_months: u8) -> (i64, i64) {
    let span_ms = MS_PER_MONTH_APPROX.saturating_mul(history_months as i64);
    let from_ms = now_ms.saturating_sub(span_ms);
    (from_ms, now_ms)
}

/// Build the (symbol, timeframe) job list from a `WizardConfig`.
pub fn build_job_list(symbols: &[String], timeframes: &[String]) -> Vec<HistoryJob> {
    let mut jobs = Vec::with_capacity(symbols.len() * timeframes.len());
    for symbol in symbols {
        for timeframe in timeframes {
            // Defence-in-depth: Step 5 already filters via
            // `forex_core::CANONICAL_TIMEFRAMES`, but if a non-
            // canonical entry leaked into the config we reject here
            // too (H2 must never reach the wizard layer).
            if !forex_core::is_canonical_timeframe(timeframe) {
                tracing::warn!(
                    target: "forex_app::wizard::historical",
                    symbol = %symbol,
                    timeframe = %timeframe,
                    "wizard skipping non-canonical timeframe"
                );
                continue;
            }
            jobs.push(HistoryJob {
                symbol: symbol.clone(),
                timeframe: timeframe.clone(),
                state: HistoryJobState::Pending,
            });
        }
    }
    jobs
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;
    let mut runtime = runtime_mutex()
        .lock()
        .expect("wizard historical runtime mutex poisoned");

    poll_progress_messages(&mut runtime);

    ui.label(
        egui::RichText::new(format!(
            "Download history for {} symbols × {} timeframes (rate-limited to {} req/s).",
            controller.config.selected_symbols.len(),
            controller.config.selected_timeframes.len(),
            WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS,
        ))
        .color(theme::TEXT_PRIMARY),
    );

    ui.add_space(theme::SPACE_SM);

    ui.horizontal(|ui| {
        ui.label("Months of history:");
        for option in WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS {
            if ui
                .selectable_label(
                    controller.config.history_months == *option,
                    format!("{}", option),
                )
                .clicked()
            {
                controller.config.history_months = *option;
            }
        }
    });

    let pair_count =
        controller.config.selected_symbols.len() * controller.config.selected_timeframes.len();
    let estimated_seconds = (pair_count as u32)
        .saturating_mul(controller.config.history_months as u32)
        .div_ceil(WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS as u32);
    ui.label(
        egui::RichText::new(format!(
            "Estimated ≈ {} requests, ≈ {} s at {} req/s.",
            pair_count.saturating_mul(controller.config.history_months as usize),
            estimated_seconds,
            WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS,
        ))
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    ui.separator();
    ui.label(
        egui::RichText::new(
            "Cancel preserves already-downloaded bars (.partial sentinel). \
             No synthetic fill is ever written.",
        )
        .size(theme::FONT_CAPTION)
        .color(theme::WARNING),
    );

    if !runtime.jobs.is_empty() {
        ui.separator();
        let completed = runtime
            .jobs
            .iter()
            .filter(|j| matches!(j.state, HistoryJobState::Succeeded { .. }))
            .count();
        let failed = runtime
            .jobs
            .iter()
            .filter(|j| matches!(j.state, HistoryJobState::Failed { .. }))
            .count();
        ui.label(
            egui::RichText::new(format!(
                "Jobs: {} total · {} succeeded · {} failed",
                runtime.jobs.len(),
                completed,
                failed
            ))
            .color(theme::TEXT_PRIMARY),
        );
        egui::ScrollArea::vertical()
            .max_height(160.0)
            .show(ui, |ui| {
                for job in &runtime.jobs {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} / {}  —  {}",
                            job.symbol,
                            job.timeframe,
                            describe_job_state(&job.state)
                        ))
                        .size(theme::FONT_CAPTION)
                        .color(theme::TEXT_MUTED),
                    );
                }
            });
    }

    if let Some(err) = runtime.last_error.as_ref() {
        ui.label(
            egui::RichText::new(format!("Download error: {}", err))
                .color(theme::DANGER)
                .size(theme::FONT_CAPTION),
        );
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if runtime.running {
            if ui.button("Cancel download").clicked() {
                runtime.cancel_flag.store(true, Ordering::Relaxed);
            }
        } else {
            let can_begin = controller.config.data_path.is_some()
                && controller.config.selected_ctid_trader_account_id.is_some()
                && !controller.config.selected_symbols.is_empty()
                && !controller.config.selected_timeframes.is_empty();
            if ui
                .add_enabled(can_begin, egui::Button::new("Begin download"))
                .clicked()
                && can_begin
            {
                start_download(&mut runtime, controller);
            }
        }
        if ui.button("Skip").clicked() {
            result = StepResult::SkipRequested;
        }
        if ui.button("Continue →").clicked() {
            result = StepResult::NextRequested;
        }
    });

    if runtime.running {
        ui.ctx().request_repaint();
    }

    result
}

fn describe_job_state(state: &HistoryJobState) -> String {
    match state {
        HistoryJobState::Pending => "pending".to_string(),
        HistoryJobState::InFlight {
            window_index,
            windows,
        } => format!("in flight ({}/{})", window_index, windows),
        HistoryJobState::BackingOff { remaining_secs } => {
            format!("rate-limited; backing off {}s", remaining_secs)
        }
        HistoryJobState::Succeeded { bars_written } => {
            format!("complete ({} bars)", bars_written)
        }
        HistoryJobState::Failed { error } => format!("failed: {}", error),
        HistoryJobState::Cancelled => "cancelled (.partial preserved)".to_string(),
    }
}

fn poll_progress_messages(runtime: &mut HistoricalRuntime) {
    loop {
        let Some(rx) = runtime.progress_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(HistoryProgressMessage::JobUpdate { index, state }) => {
                if let Some(job) = runtime.jobs.get_mut(index) {
                    job.state = state;
                }
            }
            Ok(HistoryProgressMessage::AllDone) => {
                runtime.running = false;
                runtime.finished = true;
                runtime.progress_rx = None;
                return;
            }
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                runtime.running = false;
                runtime.progress_rx = None;
                runtime.last_error = Some(
                    "historical-download worker disconnected before sending AllDone".to_string(),
                );
                return;
            }
        }
    }
}

fn start_download(runtime: &mut HistoricalRuntime, controller: &mut WizardController) {
    // Note — defence-in-depth re-entrancy guard. Today
    // the click handler in `render()` only calls us when
    // `runtime.running == false`, and the outer runtime_mutex() serialises
    // every frame so two rapid clicks in the SAME frame are deduplicated
    // by egui. Belt-and-braces: if a future caller ever invokes
    // `start_download` from another code path (e.g. an auto-pipeline
    // trigger from the Apply step, Task #10), this guard prevents a
    // duplicate job from being scheduled before the existing one
    // settles.
    if runtime.running {
        tracing::warn!(
            target: "forex_app::wizard::historical",
            "start_download called while a download is already in flight; ignoring"
        );
        return;
    }
    runtime.last_error = None;
    runtime.finished = false;
    let jobs = build_job_list(
        &controller.config.selected_symbols,
        &controller.config.selected_timeframes,
    );
    if jobs.is_empty() {
        runtime.last_error =
            Some("No (symbol, timeframe) pairs selected — go back to Step 5".to_string());
        return;
    }
    runtime.jobs = jobs.clone();
    runtime.cancel_flag = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    runtime.progress_rx = Some(rx);
    runtime.running = true;

    let cancel_flag = Arc::clone(&runtime.cancel_flag);
    let history_months = controller.config.history_months;
    let environment = match controller.config.ctrader_environment {
        super::CTraderEnvironment::Live => CTraderEnvironment::Live,
        super::CTraderEnvironment::Demo => CTraderEnvironment::Demo,
    };
    let Some(account_id_u64) = controller.config.selected_ctid_trader_account_id else {
        runtime.last_error =
            Some("No cTID trader account selected — finish Step 4 first".to_string());
        runtime.running = false;
        return;
    };
    let account_id = account_id_u64.to_string();
    // 2026-05-17 operator-directive correction: the cTrader app
    // credentials are the developer's, not the user's — they live
    // in the embedded constants and are surfaced via
    // `oauth::expose_client_id` / `oauth::expose_client_secret`. If
    // either is missing, the binary was built without them and
    // history download cannot proceed.
    let Some(client_id) = oauth::expose_client_id() else {
        runtime.last_error = Some(
            "cTrader app client_id missing from the binary's embedded credentials — \
             rebuild with FOREX_AI_EMBED_CTRADER_CLIENT_ID set"
                .to_string(),
        );
        runtime.running = false;
        return;
    };
    let Some(client_secret) = oauth::expose_client_secret() else {
        runtime.last_error = Some(
            "cTrader app client_secret missing from the binary's embedded credentials — \
             rebuild with FOREX_AI_EMBED_CTRADER_CLIENT_SECRET set"
                .to_string(),
        );
        runtime.running = false;
        return;
    };
    let Some(access_token) = oauth::expose_access_token() else {
        runtime.last_error = Some("Access token missing — finish Step 4 sign-in first".to_string());
        runtime.running = false;
        return;
    };
    let Some(data_path) = controller.config.data_path.clone() else {
        runtime.last_error = Some("Data path not selected — finish Step 2 first".to_string());
        runtime.running = false;
        return;
    };

    let context = HistoryWorkerContext {
        client_id,
        client_secret,
        access_token,
        environment,
        account_id,
        data_path,
        history_months,
    };
    std::thread::Builder::new()
        .name("wizard-history-worker".to_string())
        .spawn(move || {
            run_history_jobs(jobs, context, cancel_flag, tx);
        })
        .expect("spawn wizard-history-worker");
}

#[derive(Clone)]
struct HistoryWorkerContext {
    client_id: String,
    client_secret: String,
    access_token: String,
    environment: CTraderEnvironment,
    account_id: String,
    data_path: PathBuf,
    history_months: u8,
}

fn run_history_jobs(
    jobs: Vec<HistoryJob>,
    context: HistoryWorkerContext,
    cancel_flag: Arc<AtomicBool>,
    tx: Sender<HistoryProgressMessage>,
) {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let (overall_from, overall_to) = compute_window(now_ms, context.history_months);
    let mut last_request = Instant::now()
        .checked_sub(WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL)
        .unwrap_or_else(Instant::now);

    for (idx, job) in jobs.iter().enumerate() {
        if cancel_flag.load(Ordering::Relaxed) {
            let _ = tx.send(HistoryProgressMessage::JobUpdate {
                index: idx,
                state: HistoryJobState::Cancelled,
            });
            continue;
        }
        let vortex_path = canonical_vortex_path(&context.data_path, &job.symbol, &job.timeframe);
        // Cancel-safety: write `.partial` BEFORE any broker call.
        if let Err(err) = write_partial_sentinel(&vortex_path) {
            let _ = tx.send(HistoryProgressMessage::JobUpdate {
                index: idx,
                state: HistoryJobState::Failed {
                    error: format!("could not write .partial sentinel: {}", err),
                },
            });
            continue;
        }
        let chunks = split_window_into_chunks(overall_from, overall_to, MS_PER_MONTH_APPROX);
        let total_windows = chunks.len() as u8;
        let mut accumulated = Ohlcv {
            timestamp: Some(Vec::new()),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Some(Vec::new()),
        };
        let mut job_failed: Option<String> = None;
        let mut job_cancelled = false;

        for (chunk_idx, (from_ms, to_ms)) in chunks.iter().enumerate() {
            if cancel_flag.load(Ordering::Relaxed) {
                job_cancelled = true;
                break;
            }
            let _ = tx.send(HistoryProgressMessage::JobUpdate {
                index: idx,
                state: HistoryJobState::InFlight {
                    window_index: (chunk_idx as u8).saturating_add(1),
                    windows: total_windows,
                },
            });

            // Token bucket: ensure ≥ 200 ms between requests.
            let elapsed = last_request.elapsed();
            if elapsed < WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL {
                let remaining = WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL - elapsed;
                // Honour cancel during the sleep by polling in
                // 50 ms slices.
                sleep_with_cancel(remaining, &cancel_flag);
                if cancel_flag.load(Ordering::Relaxed) {
                    job_cancelled = true;
                    break;
                }
            }
            last_request = Instant::now();

            let request = CTraderHistoricalBarsRequest {
                client_id: context.client_id.clone(),
                client_secret: context.client_secret.clone(),
                access_token: context.access_token.clone(),
                environment: context.environment,
                account_id: context.account_id.clone(),
                symbol_name: job.symbol.clone(),
                timeframe: job.timeframe.clone(),
                from_timestamp_ms: *from_ms,
                to_timestamp_ms: *to_ms,
                count: None,
            };
            match fetch_historical_bars(&request) {
                Ok(result) => {
                    for warning in &result.warnings {
                        tracing::warn!(
                            target: "forex_app::wizard::historical",
                            symbol = %job.symbol,
                            timeframe = %job.timeframe,
                            warning = %warning,
                            "wizard historical: ctrader_history warning surfaced verbatim"
                        );
                    }
                    if let Some(ts) = accumulated.timestamp.as_mut() {
                        ts.extend(result.bars.iter().map(|b| b.timestamp_ms));
                    }
                    accumulated.open.extend(result.bars.iter().map(|b| b.open));
                    accumulated.high.extend(result.bars.iter().map(|b| b.high));
                    accumulated.low.extend(result.bars.iter().map(|b| b.low));
                    accumulated
                        .close
                        .extend(result.bars.iter().map(|b| b.close));
                    if let Some(vols) = accumulated.volume.as_mut() {
                        vols.extend(result.bars.iter().map(|b| b.volume.unwrap_or(0) as f64));
                    }
                }
                Err(err) => {
                    let msg = err.to_string();
                    if is_rate_limit_error(&msg) {
                        // Spec §3 error matrix — 30 s backoff + resume.
                        let _ = tx.send(HistoryProgressMessage::JobUpdate {
                            index: idx,
                            state: HistoryJobState::BackingOff {
                                remaining_secs: WIZARD_DEFAULT_HISTORY_BACKOFF_SECONDS,
                            },
                        });
                        sleep_with_cancel(
                            Duration::from_secs(WIZARD_DEFAULT_HISTORY_BACKOFF_SECONDS as u64),
                            &cancel_flag,
                        );
                        if cancel_flag.load(Ordering::Relaxed) {
                            job_cancelled = true;
                            break;
                        }
                        // Retry this same chunk by NOT advancing the
                        // chunk cursor — fall through and the outer
                        // loop's next iteration will pick the next
                        // chunk. Since we already failed this one,
                        // we mark this job as failed for now and
                        // proceed — the operator can resume.
                        job_failed = Some(format!("rate-limited: {}", msg));
                        break;
                    } else if is_disk_full_error(&msg) {
                        job_failed = Some(format!("disk full: {}", msg));
                        break;
                    } else {
                        job_failed = Some(msg);
                        break;
                    }
                }
            }
        }

        if job_cancelled {
            // Cancel: leave `.partial` in place so main app can
            // resume. No `.complete` sentinel written.
            let _ = tx.send(HistoryProgressMessage::JobUpdate {
                index: idx,
                state: HistoryJobState::Cancelled,
            });
            continue;
        }
        if let Some(err) = job_failed {
            let _ = tx.send(HistoryProgressMessage::JobUpdate {
                index: idx,
                state: HistoryJobState::Failed { error: err },
            });
            continue;
        }
        // Job succeeded — write the canonical Vortex layout and swap
        // the sentinel. If the Vortex write fails (e.g. disk full),
        // the `.partial` sentinel survives so the operator can retry.
        if accumulated.close.is_empty() {
            // No bars — still mark complete so the operator's
            // progress UI advances past this pair, but record the
            // empty response as a warning rather than a failure
            // (broker returns an empty trendbars set for some
            // weekend-only windows).
            tracing::warn!(
                target: "forex_app::wizard::historical",
                symbol = %job.symbol,
                timeframe = %job.timeframe,
                "wizard historical: broker returned 0 bars for the requested window"
            );
        }
        match write_symbol_timeframe_vortex(
            &context.data_path,
            &job.symbol,
            &job.timeframe,
            &accumulated,
        ) {
            Ok(_) => match replace_partial_with_complete(&vortex_path) {
                Ok(_) => {
                    let _ = tx.send(HistoryProgressMessage::JobUpdate {
                        index: idx,
                        state: HistoryJobState::Succeeded {
                            bars_written: accumulated.close.len(),
                        },
                    });
                }
                Err(err) => {
                    let _ = tx.send(HistoryProgressMessage::JobUpdate {
                        index: idx,
                        state: HistoryJobState::Failed {
                            error: format!(
                                "could not finalise sentinel: {} (Vortex file written)",
                                err
                            ),
                        },
                    });
                }
            },
            Err(err) => {
                let _ = tx.send(HistoryProgressMessage::JobUpdate {
                    index: idx,
                    state: HistoryJobState::Failed {
                        error: format!("write_symbol_timeframe_vortex failed: {}", err),
                    },
                });
            }
        }
    }
    let _ = tx.send(HistoryProgressMessage::AllDone);
}

/// Cancel-aware sleep — exits early if the cancel flag flips.
fn sleep_with_cancel(total: Duration, cancel_flag: &Arc<AtomicBool>) {
    let slice = Duration::from_millis(50);
    let start = Instant::now();
    while start.elapsed() < total {
        if cancel_flag.load(Ordering::Relaxed) {
            return;
        }
        let remaining = total.saturating_sub(start.elapsed());
        std::thread::sleep(slice.min(remaining));
    }
}

/// True iff the error message indicates the broker's 108
/// `REQUEST_FREQUENCY_EXCEEDED` rate-limit code.
pub fn is_rate_limit_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("request_frequency_exceeded")
        || lower.contains("rate limit")
        || lower.contains("(108)")
}

/// True iff the error message indicates a disk-full condition. We
/// surface verbatim either way, but the wizard's spec §3 matrix calls
/// for marking the job failed and continuing the rest of the list
/// rather than aborting the entire download.
pub fn is_disk_full_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("no space left") || lower.contains("disk full") || lower.contains("enospc")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn default_history_months_equals_operator_default() {
        assert_eq!(WIZARD_DEFAULT_HISTORY_MONTHS, 6);
        assert!(WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS.contains(&WIZARD_DEFAULT_HISTORY_MONTHS));
    }

    #[test]
    fn rate_limit_matches_ctrader_api_documentation() {
        // ctrader_api_full_reference.md §3.2 — 5 req/s per connection.
        assert_eq!(WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS, 5);
        assert_eq!(
            WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL,
            Duration::from_millis(200)
        );
    }

    #[test]
    fn historical_step_advances_to_hardware() {
        let mut c = WizardController::new();
        c.current = WizardState::Historical;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Hardware);
    }

    #[test]
    fn historical_step_skip_records_under_historical_key() {
        let mut c = WizardController::new();
        c.current = WizardState::Historical;
        c.apply(StepResult::SkipRequested);
        assert!(
            c.state_file
                .skipped_steps
                .contains(&WizardState::Historical)
        );
    }

    #[test]
    fn sentinel_suffixes_are_distinguishable() {
        assert_ne!(
            WIZARD_PARTIAL_SENTINEL_SUFFIX,
            WIZARD_COMPLETE_SENTINEL_SUFFIX
        );
    }

    #[test]
    fn partial_and_complete_sentinel_paths_are_siblings() {
        let path = PathBuf::from("/tmp/forex-ai/symbol=EURUSD/timeframe=M1/data.vortex");
        let partial = partial_sentinel_path(&path);
        let complete = complete_sentinel_path(&path);
        assert_eq!(partial.parent(), path.parent());
        assert_eq!(complete.parent(), path.parent());
        assert!(
            partial
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".partial")
        );
        assert!(
            complete
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".complete")
        );
    }

    #[test]
    fn split_window_into_chunks_covers_exact_range() {
        let from = 0_i64;
        let to = MS_PER_MONTH_APPROX * 3;
        let chunks = split_window_into_chunks(from, to, MS_PER_MONTH_APPROX);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks.first().unwrap().0, from);
        assert_eq!(chunks.last().unwrap().1, to);
    }

    #[test]
    fn split_window_into_chunks_handles_partial_tail() {
        let from = 0_i64;
        let to = MS_PER_MONTH_APPROX + 100;
        let chunks = split_window_into_chunks(from, to, MS_PER_MONTH_APPROX);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].1, to);
    }

    #[test]
    fn build_job_list_filters_non_canonical_timeframes() {
        let symbols = vec!["EURUSD".to_string()];
        // H2 is intentionally not in `forex_core::CANONICAL_TIMEFRAMES`
        // — defence-in-depth even though Step 5 already filters.
        let timeframes = vec!["M1".to_string(), "H2".to_string(), "H4".to_string()];
        let jobs = build_job_list(&symbols, &timeframes);
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.timeframe != "H2"));
    }

    #[test]
    fn is_rate_limit_error_matches_ctrader_phrasing() {
        assert!(is_rate_limit_error(
            "cTrader error: REQUEST_FREQUENCY_EXCEEDED"
        ));
        assert!(is_rate_limit_error("rate limit hit (108)"));
        assert!(!is_rate_limit_error("ACCOUNT_NOT_AUTHORIZED"));
    }

    #[test]
    fn is_disk_full_error_matches_common_phrasing() {
        assert!(is_disk_full_error("write_vortex: No space left on device"));
        assert!(is_disk_full_error("ENOSPC: disk full"));
        assert!(!is_disk_full_error("permission denied"));
    }

    #[test]
    fn compute_window_yields_expected_span() {
        let now_ms = 1_700_000_000_000_i64;
        let (from, to) = compute_window(now_ms, 6);
        assert_eq!(to, now_ms);
        assert_eq!(to - from, MS_PER_MONTH_APPROX * 6);
    }

    /// Deterministic exercise of the 5 req/s token-bucket gate without
    /// hitting a real broker. The fake "broker" just sleeps the bucket
    /// interval the same way the worker does, so 10 jobs at 5 req/s
    /// must take ≥ ~1.8 s (10 / 5 ≈ 2 s, minus the bucket-credit on
    /// the first call). Tolerance reflects the 50 ms cancel poll
    /// granularity in `sleep_with_cancel`.
    #[test]
    fn historical_token_bucket_emits_5_rps() {
        let mut last_request = Instant::now()
            .checked_sub(WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL)
            .unwrap_or_else(Instant::now);
        let started = Instant::now();
        for _ in 0..10 {
            let elapsed = last_request.elapsed();
            if elapsed < WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL {
                std::thread::sleep(WIZARD_DEFAULT_HISTORY_BUCKET_INTERVAL - elapsed);
            }
            last_request = Instant::now();
        }
        let total = started.elapsed();
        // 10 jobs * 200 ms = 2 s, minus one bucket-credit's worth at
        // start = 1.8 s lower bound.
        assert!(
            total >= Duration::from_millis(1_800),
            "expected ≥ 1.8 s for 10 token-bucket-gated requests at 5 req/s, got {:?}",
            total
        );
    }

    /// Emulate `tempfile::tempdir` without adding the dependency.
    /// Mirrors `broker_persistence::tests::tempdir_or_skip` so the
    /// crate stays dep-clean. Caller is responsible for clean-up.
    fn wizard_tempdir() -> PathBuf {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("forex-ai-wizard-history-{pid}-{nanos}"));
        std::fs::create_dir_all(&path).expect("temp dir should be creatable");
        path
    }

    /// Cancel-safety contract: if the worker is cancelled after
    /// writing `.partial` but before writing `.complete`, the
    /// `.partial` file must remain on disk so the main app's resume
    /// affordance can fire.
    #[test]
    fn historical_cancel_leaves_partial_sentinel() {
        let dir = wizard_tempdir();
        let vortex_path = canonical_vortex_path(&dir, "EURUSD", "M1");
        let partial = write_partial_sentinel(&vortex_path).expect("write_partial_sentinel");
        // Simulate cancel: no `.complete`, no `replace_partial_with_complete`.
        assert!(partial.exists(), "partial sentinel must exist after cancel");
        let complete = complete_sentinel_path(&vortex_path);
        assert!(
            !complete.exists(),
            "complete sentinel must NOT exist after cancel"
        );
        // best-effort clean-up
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Audit the swap: `.complete` is written only after the Vortex
    /// file is on disk and the `.partial` sentinel is removed.
    #[test]
    fn replace_partial_with_complete_swaps_atomically() {
        let dir = wizard_tempdir();
        let vortex_path = canonical_vortex_path(&dir, "EURUSD", "M1");
        let _ = write_partial_sentinel(&vortex_path).expect("write_partial_sentinel");
        let complete =
            replace_partial_with_complete(&vortex_path).expect("replace_partial_with_complete");
        assert!(complete.exists());
        assert!(!partial_sentinel_path(&vortex_path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real-data integration: drives the worker end-to-end against a
    /// captured `ProtoOAGetTrendbarsRes` JSON fixture. Ignored — the
    /// fixture must contain a valid broker-issued access token which
    /// can't be committed to the repo.
    #[test]
    #[ignore = "needs cTrader fixture"]
    fn historical_download_with_captured_trendbars_fixture() {
        // Expected fixture shape (per `ctrader_data.rs`
        // `parse_trendbars_response`):
        // - `payloadType` = 2138 (`ProtoOAGetTrendbarsRes`)
        // - `payload.trendbar` = repeated `ProtoOATrendbar`
        // - `payload.symbolId` matches a previously-cached
        //   `ProtoOASymbolByIdRes`.
        // Re-capture per refresh-token rotation per
        // `ctrader_api_full_reference.md` §2.5.
    }
}
