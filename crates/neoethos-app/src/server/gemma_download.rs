//! Server-side downloader for the bundled Gemma GGUF.
//!
//! The Lite installer ships without the 5 GiB GGUF and fetches it
//! during NSIS install. The Flutter AI Helper screen ALSO needs a
//! first-launch fallback for when (a) the install-time download was
//! skipped, (b) the download was interrupted, or (c) the user just
//! copied the bundle dir manually without running the installer at
//! all. This module exposes three endpoints:
//!
//!   POST /gemma/download         — kick off a streaming HF download
//!                                  to the canonical runtime slot.
//!   GET  /gemma/download/status  — polled by Flutter to drive the
//!                                  progress bar (bytes done / total,
//!                                  elapsed time, state machine).
//!   POST /gemma/download/cancel  — aborts an in-flight download.
//!
//! The state machine has four terminal-or-active variants:
//!   Idle / Downloading / Completed / Failed
//! The single-flight invariant means at most one download is in
//! progress per server lifetime — a second POST /gemma/download
//! while one is running returns 409 with the current progress so the
//! UI can re-attach to the existing transfer instead of starting a
//! second one.
//!
//! Why `OnceLock<Arc<Mutex<...>>>` instead of going through
//! `AppApiState`: keeping the download surface self-contained means
//! state.rs stays small and tests for the rest of the server don't
//! need to mock a download manager. The route handlers in the parent
//! `gemma` module call into this module via the public functions.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::state::AppApiState;

/// One of four states the download manager can be in. Serialised to
/// JSON via the camelCase DTO at the bottom of this file.
#[derive(Debug, Clone)]
enum Phase {
    Idle,
    Downloading {
        bytes_done: u64,
        bytes_total: u64,
        started_at: Instant,
    },
    Completed {
        path: PathBuf,
        bytes_total: u64,
        completed_at: Instant,
    },
    Failed {
        error: String,
        failed_at: Instant,
    },
    Cancelled {
        cancelled_at: Instant,
    },
}

struct DownloadManager {
    phase: Phase,
    /// Set when a download is in-flight. The background task polls
    /// this between chunk writes and aborts the transfer cleanly if
    /// it flips to true. Always reset to false when a new download
    /// starts so a stale cancel doesn't poison the next attempt.
    cancel: Arc<AtomicBool>,
}

impl DownloadManager {
    fn new() -> Self {
        Self {
            phase: Phase::Idle,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Global download manager handle. `OnceLock` because we lazily
/// construct on first request; `Mutex` because all reads and writes
/// must be serialised so the state machine stays consistent.
fn manager() -> &'static Arc<Mutex<DownloadManager>> {
    static MANAGER: OnceLock<Arc<Mutex<DownloadManager>>> = OnceLock::new();
    MANAGER.get_or_init(|| Arc::new(Mutex::new(DownloadManager::new())))
}

/// Where the streamed bytes land. Same path the runtime resolver
/// looks at as the "user_data_dir" fallback (see runtime.rs's lookup
/// chain), so a successful download is picked up by the very next
/// `/gemma/status` call without any further intervention.
fn target_path() -> PathBuf {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("neoethos")
        .join("models")
        .join(neoethos_gemma::runtime::BUNDLED_MODEL_FILENAME)
}

// ─── POST /gemma/download ─────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartDownloadDto {
    pub started: bool,
    pub already_running: bool,
    pub target_path: String,
    pub source_url: String,
    pub expected_bytes: u64,
}

pub async fn start(State(_state): State<AppApiState>) -> Response {
    let manager = manager().clone();
    let mut guard = manager.lock().await;

    if matches!(guard.phase, Phase::Downloading { .. }) {
        let body = StartDownloadDto {
            started: false,
            already_running: true,
            target_path: target_path().display().to_string(),
            source_url: neoethos_gemma::runtime::BUNDLED_MODEL_DOWNLOAD_URL.to_string(),
            expected_bytes: neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES,
        };
        return (StatusCode::CONFLICT, Json(body)).into_response();
    }

    // Reset state for a fresh attempt — Completed/Failed/Cancelled
    // shouldn't block a re-try, and we want a clean cancel flag.
    guard.phase = Phase::Downloading {
        bytes_done: 0,
        bytes_total: neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES,
        started_at: Instant::now(),
    };
    guard.cancel = Arc::new(AtomicBool::new(false));
    let cancel = guard.cancel.clone();
    drop(guard);

    // Fire the actual download in the background. We don't hold the
    // mutex across the network call — every progress update relocks
    // the mutex for a few microseconds.
    let target = target_path();
    let manager_for_task = manager.clone();
    tokio::spawn(async move {
        let result = run_download(manager_for_task.clone(), target.clone(), cancel.clone()).await;
        let mut guard = manager_for_task.lock().await;
        match result {
            Ok(total) => {
                guard.phase = Phase::Completed {
                    path: target,
                    bytes_total: total,
                    completed_at: Instant::now(),
                };
            }
            Err(err) if cancel.load(Ordering::SeqCst) => {
                guard.phase = Phase::Cancelled {
                    cancelled_at: Instant::now(),
                };
                tracing::info!(
                    target: "neoethos_app::server::gemma_download",
                    error = %err,
                    "Gemma download cancelled by user"
                );
            }
            Err(err) => {
                guard.phase = Phase::Failed {
                    error: err.to_string(),
                    failed_at: Instant::now(),
                };
                tracing::warn!(
                    target: "neoethos_app::server::gemma_download",
                    error = %err,
                    "Gemma download failed"
                );
            }
        }
    });

    Json(StartDownloadDto {
        started: true,
        already_running: false,
        target_path: target_path().display().to_string(),
        source_url: neoethos_gemma::runtime::BUNDLED_MODEL_DOWNLOAD_URL.to_string(),
        expected_bytes: neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES,
    })
    .into_response()
}

// ─── GET /gemma/download/status ────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadStatusDto {
    /// One of: idle / downloading / completed / failed / cancelled.
    pub state: String,
    /// Bytes already on disk for the in-flight (or last-completed) run.
    pub bytes_done: u64,
    /// Total expected bytes — server-reported when known, otherwise
    /// the approx constant from `runtime::BUNDLED_MODEL_APPROX_BYTES`.
    pub bytes_total: u64,
    /// Seconds since the current state was entered.
    pub elapsed_seconds: u64,
    /// Where the file ended up (only set on `completed`).
    pub written_path: Option<String>,
    /// Error message when state == "failed".
    pub error: Option<String>,
}

pub async fn status(State(_state): State<AppApiState>) -> Json<DownloadStatusDto> {
    let manager = manager().clone();
    let guard = manager.lock().await;
    let now = Instant::now();
    Json(match &guard.phase {
        Phase::Idle => DownloadStatusDto {
            state: "idle".to_string(),
            bytes_done: 0,
            bytes_total: 0,
            elapsed_seconds: 0,
            written_path: None,
            error: None,
        },
        Phase::Downloading {
            bytes_done,
            bytes_total,
            started_at,
        } => DownloadStatusDto {
            state: "downloading".to_string(),
            bytes_done: *bytes_done,
            bytes_total: *bytes_total,
            elapsed_seconds: now.saturating_duration_since(*started_at).as_secs(),
            written_path: None,
            error: None,
        },
        Phase::Completed {
            path,
            bytes_total,
            completed_at,
        } => DownloadStatusDto {
            state: "completed".to_string(),
            bytes_done: *bytes_total,
            bytes_total: *bytes_total,
            elapsed_seconds: now.saturating_duration_since(*completed_at).as_secs(),
            written_path: Some(path.display().to_string()),
            error: None,
        },
        Phase::Failed { error, failed_at } => DownloadStatusDto {
            state: "failed".to_string(),
            bytes_done: 0,
            bytes_total: 0,
            elapsed_seconds: now.saturating_duration_since(*failed_at).as_secs(),
            written_path: None,
            error: Some(error.clone()),
        },
        Phase::Cancelled { cancelled_at } => DownloadStatusDto {
            state: "cancelled".to_string(),
            bytes_done: 0,
            bytes_total: 0,
            elapsed_seconds: now.saturating_duration_since(*cancelled_at).as_secs(),
            written_path: None,
            error: None,
        },
    })
}

// ─── POST /gemma/download/cancel ───────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelDownloadDto {
    pub cancelled: bool,
    /// True only when there was a download running to cancel.
    pub was_running: bool,
}

pub async fn cancel(State(_state): State<AppApiState>) -> Json<CancelDownloadDto> {
    let manager = manager().clone();
    let guard = manager.lock().await;
    let was_running = matches!(guard.phase, Phase::Downloading { .. });
    guard.cancel.store(true, Ordering::SeqCst);
    Json(CancelDownloadDto {
        cancelled: true,
        was_running,
    })
}

// ─── streaming download impl ──────────────────────────────────────────────

async fn run_download(
    manager: Arc<Mutex<DownloadManager>>,
    target: PathBuf,
    cancel: Arc<AtomicBool>,
) -> anyhow::Result<u64> {
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| anyhow::anyhow!("create dir {}: {e}", parent.display()))?;
    }
    let tmp_path = target.with_extension("gguf.partial");

    let client = reqwest::Client::builder()
        .user_agent(concat!("neoethos/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| anyhow::anyhow!("reqwest client: {e}"))?;

    let mut response = client
        .get(neoethos_gemma::runtime::BUNDLED_MODEL_DOWNLOAD_URL)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET HF: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("HTTP error from HF: {e}"))?;

    let bytes_total = response.content_length().unwrap_or(0);
    // Update the manager with the server-reported total so the
    // progress bar normalises against the real number, not the
    // approximate one from the constant.
    if bytes_total > 0 {
        let mut guard = manager.lock().await;
        if let Phase::Downloading {
            bytes_total: ref mut t,
            ..
        } = guard.phase
        {
            *t = bytes_total;
        }
    }

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| anyhow::anyhow!("create {}: {e}", tmp_path.display()))?;
    let mut bytes_done: u64 = 0;
    // Update phase at most ~5 Hz to avoid lock contention on every
    // single chunk write. `response.chunk()` returns 16–64 KiB blocks
    // typically; updating every chunk would lock the mutex thousands
    // of times per second for no UX benefit.
    let mut last_update = Instant::now();

    // `chunk()` is provided by reqwest's default async client without
    // requiring the optional `stream` feature, so the dep tree stays
    // small.
    loop {
        if cancel.load(Ordering::SeqCst) {
            // Best-effort cleanup. If delete fails the next attempt
            // will overwrite anyway.
            let _ = tokio::fs::remove_file(&tmp_path).await;
            anyhow::bail!("cancelled");
        }
        let chunk = match response
            .chunk()
            .await
            .map_err(|e| anyhow::anyhow!("read chunk: {e}"))?
        {
            Some(c) => c,
            None => break,
        };
        file.write_all(&chunk)
            .await
            .map_err(|e| anyhow::anyhow!("write {}: {e}", tmp_path.display()))?;
        bytes_done += chunk.len() as u64;
        if last_update.elapsed().as_millis() >= 200 {
            let mut guard = manager.lock().await;
            if let Phase::Downloading {
                bytes_done: ref mut bd,
                ..
            } = guard.phase
            {
                *bd = bytes_done;
            }
            last_update = Instant::now();
        }
    }
    file.flush()
        .await
        .map_err(|e| anyhow::anyhow!("flush {}: {e}", tmp_path.display()))?;
    drop(file);

    // Atomic-rename pattern: only after the entire transfer + flush
    // succeeds do we move `.gguf.partial` into the canonical name.
    // Crash mid-download leaves a `.partial` that the next run can
    // either resume (future enhancement) or just delete + redo.
    tokio::fs::rename(&tmp_path, &target)
        .await
        .map_err(|e| anyhow::anyhow!("rename to {}: {e}", target.display()))?;

    Ok(bytes_done)
}
