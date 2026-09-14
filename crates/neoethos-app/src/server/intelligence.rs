//! `/intelligence` — what the model swarm currently knows.
//!
//! Surfaces installed models and exact Discovery training selections.
//! Read-only — the actual training happens via `/engines/training/*`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app_services::training::handoff::{
    self, TrainingHandoffSummary, TrainingHandoffUnavailable,
};
use crate::app_services::training::{
    CombinedResearchReportsDto, load_saved_final_research_context,
    read_saved_final_research_reports_with_context,
};
use axum::Json;
use axum::extract::{Query, State, rejection::QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use super::errors::{actionable_error, internal_panic};
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
    /// Strategies from validated, immutable Discovery training handoffs.
    /// Empty before publication; unavailable handoffs are reported separately.
    pub discovery_targets: Vec<DiscoveryTargetDto>,
    pub training_handoffs: Vec<TrainingHandoffSummary>,
    pub training_handoff_unavailable: Vec<TrainingHandoffUnavailable>,
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
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read the models directory. Run Discovery and Training first, \
             or check the app data folder is accessible.",
            &err,
        ),
        Err(join_err) => internal_panic("Loading intelligence", join_err),
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchQuery {
    training_handoff: String,
}

/// Explicit, selected-handoff read. This is deliberately not part of the
/// periodically polled intelligence inventory. Strategy reports need no models;
/// combined reports verify their installed tree without loading models or prices.
pub async fn research(
    State(state): State<AppApiState>,
    query: Result<Query<ResearchQuery>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => {
            return research_error_response(ResearchReadError {
                status: StatusCode::BAD_REQUEST,
                message: "Select one training handoff using the training_handoff query field.",
                source: anyhow::anyhow!(error.body_text()),
            });
        }
    };
    // Validate before config or filesystem access. Reuse the same canonical
    // selector policy as Discovery -> Training, rather than accepting a path.
    if let Err(source) = handoff::handoff_path(Path::new(""), &query.training_handoff) {
        return research_error_response(ResearchReadError {
            status: StatusCode::BAD_REQUEST,
            message: "Select a canonical lowercase training handoff identity, not a path.",
            source,
        });
    }
    let config_path = state.config_path().to_owned();
    let result = tokio::task::spawn_blocking(move || {
        let settings = Settings::from_yaml(config_path).map_err(|source| ResearchReadError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Could not read the app configuration for the selected research report.",
            source: source.into(),
        })?;
        // This is the same root used by the typed TrainingRequest producer.
        // Settings currently has no alternative models-directory authority.
        read_selected_research(
            &settings.system.data_dir,
            &Path::new("models").join("candidates"),
            &query.training_handoff,
        )
    })
    .await;
    let mut response = match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(error)) => research_error_response(error),
        Err(error) => internal_panic("Loading selected saved research", error),
    };
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

#[derive(Debug)]
struct ResearchReadError {
    status: StatusCode,
    message: &'static str,
    source: anyhow::Error,
}

fn research_error_response(error: ResearchReadError) -> Response {
    // Keep the complete validation chain; no broker operation occurs here and
    // these failures must not be translated into a trading-permission message.
    (
        error.status,
        Json(serde_json::json!({
            "error": error.message,
            "detail": format!("{:#}", error.source),
        })),
    )
        .into_response()
}

fn io_error_kind(error: &anyhow::Error) -> Option<std::io::ErrorKind> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind)
    })
}

fn read_selected_research(
    data_root: &Path,
    candidate_root: &Path,
    identity: &str,
) -> Result<CombinedResearchReportsDto, ResearchReadError> {
    handoff::handoff_path(data_root, identity).map_err(|source| ResearchReadError {
        status: StatusCode::BAD_REQUEST,
        message: "Select a canonical lowercase training handoff identity, not a path.",
        source,
    })?;
    // Qualify missing/invalid selection separately from missing report evidence.
    // Carry this exact checked projection into the report phase; do not discard
    // it and decode/validate the same large handoff again in this request.
    let context = load_saved_final_research_context(data_root, identity).map_err(|source| {
        ResearchReadError {
            status: match io_error_kind(&source) {
                Some(std::io::ErrorKind::NotFound) => StatusCode::NOT_FOUND,
                Some(_) => StatusCode::INTERNAL_SERVER_ERROR,
                None => StatusCode::BAD_REQUEST,
            },
            message: "The selected Discovery training handoff is missing or invalid.",
            source,
        }
    })?;
    // The reader owns any required model verification once for all attempts.
    read_saved_final_research_reports_with_context(data_root, candidate_root, context)
        .map_err(candidate_read_error)
}

fn candidate_read_error(source: anyhow::Error) -> ResearchReadError {
    ResearchReadError {
        status: if io_error_kind(&source).is_some() {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::BAD_REQUEST
        },
        message: "The selected candidate research could not be verified. Its saved evidence may be incomplete or invalid.",
        source,
    }
}

fn scan_intelligence() -> anyhow::Result<IntelligenceDto> {
    // The backend currently scans the hardcoded "models" path so the
    // Flutter screen surfaces the same artifacts every run.
    // If Settings ever grows a `models_dir` we'll switch over here.
    // F-553/F-576 closure (2026-05-25): resolved via the process-wide
    // install so a non-default `--config` flag still works.
    let settings = Settings::from_yaml(super::state::current_config_path())?;
    scan_intelligence_at(&settings.system.data_dir, Path::new("models"))
}

fn scan_intelligence_at(data_root: &Path, models_dir: &Path) -> anyhow::Result<IntelligenceDto> {
    let inventory = handoff::list(data_root)?;
    let training_handoffs = inventory.available;
    let training_handoff_unavailable = inventory.unavailable;
    let mut discovery_targets = Vec::new();
    for summary in &training_handoffs {
        for strategy in &summary.strategies {
            discovery_targets.push(DiscoveryTargetDto {
                symbol: summary.symbol.clone(),
                base_tf: summary.base_tf.clone(),
                strategy_id: strategy.strategy_id.clone(),
                sharpe: Some(strategy.sharpe),
                win_rate: Some(strategy.win_rate),
            });
        }
    }
    // Absolute path so out-of-process helpers (the P2P mesh sidecar) can locate
    // the model store to transfer trained models. Falls back to the relative
    // form if the CWD is somehow unreadable.
    let models_dir_str = std::fs::canonicalize(&models_dir)
        .ok()
        .map(|p| {
            p.display()
                .to_string()
                .trim_start_matches(r"\\?\")
                .to_string()
        })
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|c| c.join(models_dir).display().to_string())
        })
        .unwrap_or_else(|| models_dir.display().to_string());
    if !models_dir.exists() {
        return Ok(IntelligenceDto {
            models_dir: models_dir_str,
            models_dir_exists: false,
            artifact_count: 0,
            artifacts: Vec::new(),
            last_touched_unix_ms: None,
            discovery_targets,
            training_handoffs,
            training_handoff_unavailable,
        });
    }

    let (artifacts, latest_mtime) = scan_models_dir(&models_dir);

    let last_touched_unix_ms = latest_mtime
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);

    Ok(IntelligenceDto {
        models_dir: models_dir_str,
        models_dir_exists: true,
        artifact_count: artifacts.len(),
        artifacts,
        last_touched_unix_ms,
        discovery_targets,
        training_handoffs,
        training_handoff_unavailable,
    })
}

#[cfg(test)]
#[path = "intelligence_research_tests.rs"]
mod research_tests;

/// Scan the models directory for TRAINED MODELS (2026-07-17 fix — operator:
/// "τα βλέπει σαν αρχείο αλλά δεν τα αξιοποιεί").
///
/// Trained models live in NESTED directories — `models/<SYMBOL>/<TF>/<name>/`
/// — exactly the layout the ensemble loader (`load_experts_for_symbol`)
/// consumes. The old scan looked only at TOP-LEVEL FILES, so every trained
/// expert was invisible to the Intelligence screen: the operator saw a full
/// models directory on disk but "0 models" in the app. Each model dir is now
/// reported as `SYMBOL/TF/model_name`, alongside any top-level loose artifact
/// files (model_targets.json, walkforward.json, legacy single-file models).
fn scan_models_dir(models_dir: &Path) -> (Vec<String>, Option<SystemTime>) {
    let mut artifacts: Vec<String> = Vec::new();
    let mut latest_mtime: Option<SystemTime> = None;
    let mut touch = |meta: Option<std::fs::Metadata>| {
        if let Some(mtime) = meta.and_then(|m| m.modified().ok()) {
            latest_mtime = Some(match latest_mtime {
                Some(prev) if prev > mtime => prev,
                _ => mtime,
            });
        }
    };

    if let Ok(read_dir) = std::fs::read_dir(models_dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if is_artifact(name) {
                        artifacts.push(name.to_string());
                    }
                }
                touch(entry.metadata().ok());
            } else if path.is_dir() {
                let symbol = entry.file_name().to_string_lossy().to_string();
                if symbol == "candidates" {
                    continue;
                } // candidate trees are not deployed experts
                let Ok(tf_dirs) = std::fs::read_dir(&path) else {
                    continue;
                };
                for tf_entry in tf_dirs.flatten() {
                    if !tf_entry.path().is_dir() {
                        continue;
                    }
                    let tf = tf_entry.file_name().to_string_lossy().to_string();
                    let Ok(model_dirs) = std::fs::read_dir(tf_entry.path()) else {
                        continue;
                    };
                    for model_entry in model_dirs.flatten() {
                        let model_path = model_entry.path();
                        let name = model_entry.file_name().to_string_lossy().to_string();
                        if name.starts_with('_') {
                            continue; // sentinels, not models
                        }
                        if model_path.is_dir() {
                            artifacts.push(format!("{symbol}/{tf}/{name}"));
                            touch(model_entry.metadata().ok());
                        } else if model_path.is_file() && is_artifact(&name) {
                            // Flat per-TF artifact files (older layouts).
                            artifacts.push(format!("{symbol}/{tf}/{name}"));
                            touch(model_entry.metadata().ok());
                        }
                    }
                }
            }
        }
    }
    artifacts.sort();
    (artifacts, latest_mtime)
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

#[cfg(test)]
mod scan_tests {
    use super::*;

    #[test]
    fn nested_trained_model_dirs_are_visible() {
        // 2026-07-17: the Intelligence scan must surface trained models in
        // the nested models/<SYMBOL>/<TF>/<name>/ layout the ensemble loader
        // consumes — the old top-level-files-only scan showed 0 models.
        let root = std::env::temp_dir().join(format!(
            "neoethos_intel_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for model in ["dqn", "hmm_regime", "swarm_forecaster"] {
            std::fs::create_dir_all(root.join("EURUSD").join("M15").join(model)).unwrap();
        }
        std::fs::create_dir_all(root.join("EURUSD").join("M15").join("_healthcheck")).unwrap();
        std::fs::write(root.join("model_targets.json"), "{}").unwrap();
        std::fs::write(root.join("training.txt"), "log").unwrap(); // not an artifact

        let (artifacts, mtime) = scan_models_dir(&root);
        assert!(
            artifacts.contains(&"EURUSD/M15/dqn".to_string()),
            "nested model dirs must be listed: {artifacts:?}"
        );
        assert!(artifacts.contains(&"EURUSD/M15/hmm_regime".to_string()));
        assert!(artifacts.contains(&"EURUSD/M15/swarm_forecaster".to_string()));
        assert!(artifacts.contains(&"model_targets.json".to_string()));
        assert!(
            !artifacts.iter().any(|a| a.contains("_healthcheck")),
            "sentinels must be excluded"
        );
        assert!(
            !artifacts.iter().any(|a| a.contains("training.txt")),
            "non-artifact files excluded"
        );
        assert_eq!(artifacts.len(), 4, "3 models + 1 json: {artifacts:?}");
        assert!(mtime.is_some());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
