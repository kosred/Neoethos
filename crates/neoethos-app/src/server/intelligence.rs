//! `/intelligence` — what the model swarm currently knows.
//!
//! Surfaces the contents of the `models/` directory plus the
//! `model_targets.json` written by the last completed discovery run,
//! in a shape the Flutter Intelligence screen can render directly.
//! Read-only — the actual training happens via `/engines/training/*`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use super::state::AppApiState;

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntelligenceDto {
    /// Path of the `models/` directory (informational).
    pub models_dir: String,
    /// Whether the directory exists on disk.
    pub models_dir_exists: bool,
    /// Number of artifact files (joblib / pt / cbm / onnx / json) found.
    pub artifact_count: usize,
    /// Names of the discovered artifact files (sorted; no path).
    pub artifacts: Vec<String>,
    /// mtime of the most-recently-touched artifact, Unix-millis.
    /// `None` when the directory is empty.
    pub last_touched_unix_ms: Option<u64>,
    /// Targets list from the latest discovery (`model_targets.json`).
    /// Empty when discovery hasn't run yet, or when the file failed
    /// to parse.
    pub discovery_targets: Vec<DiscoveryTargetDto>,
    /// Top-level metrics from `walkforward_metrics.json` if present.
    pub walkforward_splits: Option<u32>,
    pub walkforward_avg_accuracy: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryTargetDto {
    pub symbol: String,
    pub base_tf: String,
    pub strategy_id: String,
    pub sharpe: Option<f64>,
    pub win_rate: Option<f64>,
}

pub async fn intelligence(State(_state): State<AppApiState>) -> Response {
    let result = tokio::task::spawn_blocking(scan_intelligence).await;
    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("blocking task panicked: {join_err}")
            })),
        )
            .into_response(),
    }
}

fn scan_intelligence() -> anyhow::Result<IntelligenceDto> {
    // The backend currently scans the hardcoded "models" path so the
    // Flutter screen surfaces the same artifacts every run.
    // If Settings ever grows a `models_dir` we'll switch over here.
    let _settings = Settings::from_yaml("config.yaml").ok();
    let models_dir = std::path::PathBuf::from("models");
    let models_dir_str = models_dir.display().to_string();
    if !models_dir.exists() {
        return Ok(IntelligenceDto {
            models_dir: models_dir_str,
            models_dir_exists: false,
            artifact_count: 0,
            artifacts: Vec::new(),
            last_touched_unix_ms: None,
            discovery_targets: Vec::new(),
            walkforward_splits: None,
            walkforward_avg_accuracy: None,
        });
    }

    let mut artifacts: Vec<String> = Vec::new();
    let mut latest_mtime: Option<SystemTime> = None;

    if let Ok(read_dir) = std::fs::read_dir(&models_dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if is_artifact(name) {
                        artifacts.push(name.to_string());
                    }
                }
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        latest_mtime = Some(match latest_mtime {
                            Some(prev) if prev > mtime => prev,
                            _ => mtime,
                        });
                    }
                }
            }
        }
    }
    artifacts.sort();

    let last_touched_unix_ms = latest_mtime
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);

    let discovery_targets = parse_model_targets(&models_dir);
    let (wf_splits, wf_avg) = parse_walkforward(&models_dir);

    Ok(IntelligenceDto {
        models_dir: models_dir_str,
        models_dir_exists: true,
        artifact_count: artifacts.len(),
        artifacts,
        last_touched_unix_ms,
        discovery_targets,
        walkforward_splits: wf_splits,
        walkforward_avg_accuracy: wf_avg,
    })
}

fn is_artifact(name: &str) -> bool {
    // Whitelist of extensions written by the training pipeline. We
    // exclude `.txt` log files and the leading `_healthcheck` /
    // `_workers` dot-prefixed sentinels so the UI shows only models.
    if name.starts_with('_') {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    [".joblib", ".pkl", ".pt", ".cbm", ".onnx", ".json"]
        .iter()
        .any(|ext| lower.ends_with(ext))
}

fn parse_model_targets(models_dir: &Path) -> Vec<DiscoveryTargetDto> {
    let path = models_dir.join("model_targets.json");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) else {
        return Vec::new();
    };

    // The shape is `{ "EURUSD/M1": [{strategy_id, sharpe, win_rate, ...}, ...] }`.
    // We flatten that to a Vec for the wire.
    let mut out = Vec::new();
    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            let (symbol, base_tf) = match key.split_once('/') {
                Some((s, t)) => (s.to_string(), t.to_string()),
                None => (key.clone(), String::new()),
            };
            if let Some(list) = val.as_array() {
                for entry in list {
                    let strategy_id = entry
                        .get("strategy_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if strategy_id.is_empty() {
                        continue;
                    }
                    out.push(DiscoveryTargetDto {
                        symbol: symbol.clone(),
                        base_tf: base_tf.clone(),
                        strategy_id,
                        sharpe: entry.get("sharpe").and_then(|v| v.as_f64()),
                        win_rate: entry.get("win_rate").and_then(|v| v.as_f64()),
                    });
                }
            }
        }
    }
    out
}

fn parse_walkforward(models_dir: &Path) -> (Option<u32>, Option<f64>) {
    let path = models_dir.join("walkforward_metrics.json");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return (None, None);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) else {
        return (None, None);
    };
    let splits = value
        .get("walkforward_splits")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let avg = value.get("avg_accuracy").and_then(|v| v.as_f64());
    (splits, avg)
}
