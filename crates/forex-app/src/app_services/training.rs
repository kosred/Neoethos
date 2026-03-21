use crate::app_services::{
    jobs::{
        push_recent_event, CancellationFlag, JobEventLevel, JobKind, JobProgress, JobReport,
        JobSnapshot, JobState,
    },
    ServiceEvent,
};
use anyhow::Result;
use forex_core::{
    logging::{canonical_log_path, write_subsystem_record},
    sectioned_log::{SectionedRunRecord, SubsystemSection},
    Settings,
};
use forex_models::TrainingOrchestrator;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub config_path: String,
    pub models_dir: PathBuf,
    pub symbol: String,
    pub base_tf: String,
}

impl TrainingRequest {
    pub fn validate(&self) -> Result<()> {
        if self.symbol.trim().is_empty() {
            anyhow::bail!("training request symbol must not be empty");
        }
        if self.base_tf.trim().is_empty() {
            anyhow::bail!("training request base timeframe must not be empty");
        }
        if self.config_path.trim().is_empty() {
            anyhow::bail!("training request config path must not be empty");
        }
        if self.models_dir.as_os_str().is_empty() {
            anyhow::bail!("training request models directory must not be empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrainingJobHandle {
    pub snapshot: JobSnapshot,
    pub cancel: CancellationFlag,
}

impl TrainingJobHandle {
    pub fn new() -> Self {
        Self {
            snapshot: JobSnapshot::new(JobKind::Training),
            cancel: CancellationFlag::new(),
        }
    }
}

fn requested_training_highlights(request: &TrainingRequest) -> Vec<(String, String)> {
    vec![
        ("symbol".to_string(), request.symbol.clone()),
        ("base_tf".to_string(), request.base_tf.clone()),
        ("config_path".to_string(), request.config_path.clone()),
        (
            "models_dir".to_string(),
            request.models_dir.display().to_string(),
        ),
    ]
}

#[cfg(test)]
pub fn completed_snapshot(
    completed_models: Vec<String>,
    failed_models: Vec<String>,
) -> JobSnapshot {
    completed_snapshot_from(
        JobSnapshot::new(JobKind::Training),
        completed_models,
        failed_models,
    )
}

fn completed_snapshot_from(
    mut snapshot: JobSnapshot,
    completed_models: Vec<String>,
    failed_models: Vec<String>,
) -> JobSnapshot {
    let summary = format!(
        "completed models: [{}]; failed models: [{}]",
        completed_models.join(", "),
        failed_models.join(", ")
    );

    snapshot.state = if failed_models.is_empty() {
        JobState::Succeeded
    } else {
        JobState::Degraded
    };
    let mut warnings = Vec::new();
    if !failed_models.is_empty() {
        warnings.push(format!(
            "{} model(s) failed during training",
            failed_models.len()
        ));
    }
    snapshot.report = JobReport {
        counters: vec![
            (
                "requested_models".to_string(),
                (completed_models.len() + failed_models.len()) as u64,
            ),
            (
                "completed_models".to_string(),
                completed_models.len() as u64,
            ),
            ("failed_models".to_string(), failed_models.len() as u64),
        ],
        highlights: vec![
            (
                "completed_models".to_string(),
                completed_models.len().to_string(),
            ),
            ("failed_models".to_string(), failed_models.len().to_string()),
            (
                "requested_models".to_string(),
                (completed_models.len() + failed_models.len()).to_string(),
            ),
        ],
        entries: completed_models
            .iter()
            .map(|model| format!("completed | {model}"))
            .chain(
                failed_models
                    .iter()
                    .map(|model| format!("failed | {model}")),
            )
            .collect(),
        events: push_recent_event(
            &snapshot.report.events,
            if failed_models.is_empty() {
                JobEventLevel::Info
            } else {
                JobEventLevel::Warning
            },
            format!(
                "training finished with {} completed and {} failed model(s)",
                completed_models.len(),
                failed_models.len()
            ),
        ),
        warnings,
        summary,
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

pub fn failed_snapshot(err: anyhow::Error) -> JobSnapshot {
    failed_snapshot_from(JobSnapshot::new(JobKind::Training), err)
}

fn failed_snapshot_from(mut snapshot: JobSnapshot, err: anyhow::Error) -> JobSnapshot {
    let message = err.to_string();
    snapshot.state = JobState::Failed;
    snapshot.report = JobReport {
        errors: vec![message.clone()],
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Error,
            format!("training failed: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

#[cfg(test)]
pub fn cancelled_snapshot(message: impl Into<String>) -> JobSnapshot {
    cancelled_snapshot_from(JobSnapshot::new(JobKind::Training), message)
}

fn cancelled_snapshot_from(mut snapshot: JobSnapshot, message: impl Into<String>) -> JobSnapshot {
    let message = message.into();
    snapshot.state = JobState::Cancelled;
    snapshot.report = JobReport {
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Warning,
            format!("training cancelled: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

pub fn start_training_job(
    request: TrainingRequest,
    tx: mpsc::UnboundedSender<ServiceEvent>,
) -> Result<TrainingJobHandle> {
    request.validate()?;

    let handle = TrainingJobHandle::new();
    let cancel = handle.cancel.clone();
    let mut snapshot = handle.snapshot.clone();
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.1),
        stage: "loading_settings".to_string(),
        message: format!("loading training settings from {}", request.config_path),
    };
    snapshot.report = JobReport {
        highlights: requested_training_highlights(&request),
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "loading training settings for {} {} from {}",
                request.symbol, request.base_tf, request.config_path
            ),
        ),
        summary: format!(
            "loading training settings for {} on {}",
            request.symbol, request.base_tf
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    send_event(&tx, ServiceEvent::TrainingUpdated(snapshot.clone()));
    log_training_event(
        "ui_training_job",
        "STARTED",
        format!("starting training for {}", request.symbol),
    );

    tokio::spawn(async move {
        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled training before settings load",
            );
            send_event(&tx, ServiceEvent::TrainingUpdated(cancelled.clone()));
            log_training_event(
                "ui_training_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let settings_request = request.clone();
        let settings_and_models = match tokio::task::spawn_blocking(move || {
            let settings = Settings::from_yaml(&settings_request.config_path)?;
            let planned_models = settings.models.ml_models.clone();
            Ok::<_, anyhow::Error>((settings, planned_models))
        })
        .await
        {
            Ok(Ok(parts)) => parts,
            Ok(Err(err)) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::TrainingUpdated(failed.clone()));
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
                return;
            }
            Err(err) => {
                let failed = failed_snapshot_from(
                    snapshot,
                    anyhow::anyhow!("training settings join error: {err}"),
                );
                send_event(&tx, ServiceEvent::TrainingUpdated(failed.clone()));
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        let (settings, planned_models) = settings_and_models;

        snapshot.progress = JobProgress {
            percent: Some(0.35),
            stage: "planning_models".to_string(),
            message: format!(
                "loaded model plan for {} on {}",
                request.symbol, request.base_tf
            ),
        };
        snapshot.report = JobReport {
            counters: vec![("planned_models".to_string(), planned_models.len() as u64)],
            highlights: requested_training_highlights(&request),
            entries: planned_models
                .iter()
                .take(8)
                .map(|model| format!("planned | {model}"))
                .collect(),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "loaded {} planned model(s) for {}",
                    planned_models.len(),
                    request.symbol
                ),
            ),
            summary: format!(
                "loaded {} planned model(s) for training",
                planned_models.len()
            ),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };
        send_event(&tx, ServiceEvent::TrainingUpdated(snapshot.clone()));

        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled training before backend execution",
            );
            send_event(&tx, ServiceEvent::TrainingUpdated(cancelled.clone()));
            log_training_event(
                "ui_training_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        snapshot.progress = JobProgress {
            percent: Some(0.7),
            stage: "dispatching_backend".to_string(),
            message: format!(
                "dispatching backend training for {} planned model(s)",
                planned_models.len()
            ),
        };
        snapshot.report = JobReport {
            counters: vec![("planned_models".to_string(), planned_models.len() as u64)],
            highlights: requested_training_highlights(&request),
            entries: planned_models
                .iter()
                .take(8)
                .map(|model| format!("planned | {model}"))
                .collect(),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "dispatching backend training for {} planned model(s)",
                    planned_models.len()
                ),
            ),
            summary: format!(
                "dispatching backend training for {} planned model(s)",
                planned_models.len()
            ),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };
        send_event(&tx, ServiceEvent::TrainingUpdated(snapshot.clone()));

        let train_request = request.clone();
        let train_result = tokio::task::spawn_blocking(move || {
            let orchestrator =
                TrainingOrchestrator::new(settings, train_request.models_dir.clone());
            orchestrator.train_symbol(&train_request.symbol, &train_request.base_tf)
        })
        .await;

        match train_result {
            Ok(Ok(())) => {
                let completed = completed_snapshot_from(snapshot, planned_models, Vec::new());
                send_event(&tx, ServiceEvent::TrainingUpdated(completed.clone()));
                log_training_event(
                    "ui_training_job",
                    "SUCCESS",
                    completed.report.summary.clone(),
                );
            }
            Ok(Err(err)) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::TrainingUpdated(failed.clone()));
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
            }
            Err(err) => {
                let failed =
                    failed_snapshot_from(snapshot, anyhow::anyhow!("training join error: {err}"));
                send_event(&tx, ServiceEvent::TrainingUpdated(failed.clone()));
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
            }
        }
    });

    Ok(handle)
}

fn send_event(tx: &mpsc::UnboundedSender<ServiceEvent>, event: ServiceEvent) {
    if let Err(err) = tx.send(event) {
        tracing::error!("Failed to send training service event: {}", err);
    }
}

fn log_training_event(operation: &str, status: &str, message: String) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::Training,
        training_record(operation, status, message),
    ) {
        tracing::error!("Failed to write TRAINING section log: {}", err);
    }
}

fn training_record(operation: &str, status: &str, message: String) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!("training-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::Training,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message,
        body: String::new(),
    }
}

fn system_time_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch");
    format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::{
        jobs::{JobEventLevel, JobState},
        ServiceEvent,
    };
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    fn sample_request() -> TrainingRequest {
        TrainingRequest {
            config_path: "config.yaml".to_string(),
            models_dir: PathBuf::from("models"),
            symbol: "EURUSD".to_string(),
            base_tf: "M1".to_string(),
        }
    }

    #[test]
    fn invalid_request_fails_before_launch() {
        let mut request = sample_request();
        request.symbol.clear();

        let err = request
            .validate()
            .expect_err("expected invalid request to fail");
        assert!(err.to_string().contains("symbol"));
    }

    #[test]
    fn cancellation_request_maps_to_cancelled_snapshot() {
        let snapshot = cancelled_snapshot("operator cancelled training");

        assert_eq!(snapshot.state, JobState::Cancelled);
        assert_eq!(snapshot.report.summary, "operator cancelled training");
    }

    #[test]
    fn failed_training_maps_to_failed_snapshot() {
        let snapshot = failed_snapshot(anyhow::anyhow!("training backend unavailable"));

        assert_eq!(snapshot.state, JobState::Failed);
        assert_eq!(
            snapshot.report.errors,
            vec!["training backend unavailable".to_string()]
        );
    }

    #[test]
    fn completed_training_snapshot_keeps_completed_and_failed_models() {
        let snapshot = completed_snapshot(
            vec!["xgboost".to_string(), "lightgbm".to_string()],
            vec!["mlp".to_string()],
        );

        assert_eq!(snapshot.state, JobState::Degraded);
        assert_eq!(
            snapshot.report.counters,
            vec![
                ("requested_models".to_string(), 3),
                ("completed_models".to_string(), 2),
                ("failed_models".to_string(), 1),
            ]
        );
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "completed_models" && value == "2" }));
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "failed_models" && value == "1" }));
        assert!(snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry == "completed | xgboost"));
        assert!(snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry == "failed | mlp"));
        assert!(snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("training finished")));
        assert_eq!(
            snapshot.report.events.last().map(|event| event.level),
            Some(JobEventLevel::Warning)
        );
        assert!(snapshot.report.summary.contains("xgboost"));
        assert!(snapshot.report.summary.contains("mlp"));
    }

    #[tokio::test]
    async fn start_training_job_emits_initial_snapshot_with_runtime_context() {
        let request = sample_request();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let _handle = start_training_job(request.clone(), tx).expect("job should start");
        let event = rx.recv().await.expect("expected initial training event");
        let ServiceEvent::TrainingUpdated(snapshot) = event else {
            panic!("expected training update event");
        };

        assert_eq!(snapshot.state, JobState::Running);
        assert_eq!(snapshot.progress.stage, "loading_settings");
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "symbol" && value == "EURUSD"));
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "config_path" && value == "config.yaml"));
        assert!(snapshot.report.events.iter().any(|event| {
            event.message.contains("loading training settings") && event.message.contains("EURUSD")
        }));
        assert_eq!(
            snapshot.report.log_path,
            Some(canonical_log_path().display().to_string())
        );
    }
}
