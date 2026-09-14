mod combined;
mod final_holdout;
pub mod handoff;

pub(crate) use final_holdout::{
    CombinedResearchReportsDto, load_saved_final_research_context,
    read_saved_combined_research_reports, read_saved_final_research_reports_with_context,
};
#[cfg(test)]
pub(crate) use final_holdout::{
    install_saved_research_test_fixture, install_saved_strategy_research_test_fixture,
    read_saved_final_research_reports,
};

use crate::app_services::{
    ServiceEvent,
    jobs::{
        CancellationFlag, JobEventLevel, JobKind, JobProgress, JobReport, JobSnapshot, JobState,
        push_recent_event,
    },
};
use anyhow::{Context, Result};
use neoethos_core::{
    Settings,
    logging::{canonical_log_path, write_subsystem_record},
    sectioned_log::{SectionedRunRecord, SubsystemSection},
};
use neoethos_models::{
    ModelTrainingProgress, PromotionCandidateTrainingTerminalV1, TrainingOrchestrator,
    TrainingRunSummary, train_and_deploy_promotion_candidate_v1,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub config_path: String,
    pub models_dir: PathBuf,
    pub symbol: String,
    pub base_tf: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    #[default]
    TrainModels,
    StrategyOnly,
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

fn upsert_counter(counters: &mut Vec<(String, u64)>, name: &str, value: u64) {
    if let Some((_, existing)) = counters.iter_mut().find(|(key, _)| key == name) {
        *existing = value;
    } else {
        counters.push((name.to_string(), value));
    }
}

fn push_recent_entry(entries: &[String], entry: impl Into<String>) -> Vec<String> {
    let mut next = entries.to_vec();
    next.push(entry.into());
    if next.len() > 12 {
        next.drain(0..(next.len() - 12));
    }
    next
}

fn push_unique_text(items: &[String], value: impl Into<String>) -> Vec<String> {
    let value = value.into();
    let mut next = items.to_vec();
    if !next.iter().any(|item| item == &value) {
        next.push(value);
    }
    next
}

fn backend_progress_percent(
    completed_models: usize,
    failed_models: usize,
    total_models: usize,
) -> Option<f32> {
    if total_models == 0 {
        return None;
    }

    let done = (completed_models + failed_models) as f32;
    let total = total_models as f32;
    Some((0.7 + 0.15 * (done / total)).clamp(0.7, 0.85))
}

fn combined_progress_percent(stage: &str, completed: usize, total: usize) -> Option<f32> {
    let (start, end) = match stage {
        "search_features" => (0.85, 0.88),
        "model_features" => (0.88, 0.91),
        "causal_model_inference" => (0.91, 0.97),
        "account_replay" => (0.97, 0.99),
        _ => return None,
    };
    let fraction = if total == 0 {
        0.0
    } else {
        completed.min(total) as f32 / total as f32
    };
    Some(start + (end - start) * fraction)
}

fn apply_backend_progress_event(snapshot: &mut JobSnapshot, event: &ModelTrainingProgress) {
    let (
        model,
        completed_models,
        failed_models,
        total_models,
        event_level,
        event_message,
        entry,
        error_text,
    ) = match event {
        ModelTrainingProgress::Started {
            model,
            total_models,
        } => (
            model.as_str(),
            0usize,
            0usize,
            *total_models,
            JobEventLevel::Info,
            format!("backend started model {model}"),
            format!("started | {model}"),
            None,
        ),
        ModelTrainingProgress::Succeeded {
            model,
            completed_models,
            failed_models,
            total_models,
        } => (
            model.as_str(),
            *completed_models,
            *failed_models,
            *total_models,
            JobEventLevel::Info,
            format!(
                "model {model} completed ({}/{total_models} finished, {} failed)",
                completed_models + failed_models,
                failed_models
            ),
            format!("completed | {model}"),
            None,
        ),
        ModelTrainingProgress::Failed {
            model,
            error,
            completed_models,
            failed_models,
            total_models,
        } => (
            model.as_str(),
            *completed_models,
            *failed_models,
            *total_models,
            JobEventLevel::Warning,
            format!(
                "model {model} failed ({}/{total_models} finished): {error}",
                completed_models + failed_models
            ),
            format!("failed | {model} | {error}"),
            Some(format!("model {model}: {error}")),
        ),
    };

    snapshot.progress = JobProgress {
        percent: backend_progress_percent(completed_models, failed_models, total_models),
        stage: "backend_running".to_string(),
        // Name the CURRENT model, not just a count. On `Started` this is the
        // model now training; during a long/stuck model the message stays on it
        // so the operator can see WHICH model is running (a 20h hang used to
        // show only an anonymous "N of M finished"). Failures land in
        // report.warnings/events with the reason, so which+why are both visible.
        message: format!(
            "training model `{model}` — {} of {} finished ({} failed)",
            completed_models + failed_models,
            total_models,
            failed_models
        ),
    };
    upsert_counter(
        &mut snapshot.report.counters,
        "requested_models",
        total_models as u64,
    );
    upsert_counter(
        &mut snapshot.report.counters,
        "planned_models",
        total_models as u64,
    );
    upsert_counter(
        &mut snapshot.report.counters,
        "completed_models",
        completed_models as u64,
    );
    upsert_counter(
        &mut snapshot.report.counters,
        "failed_models",
        failed_models as u64,
    );
    snapshot.report.entries = push_recent_entry(&snapshot.report.entries, entry);
    snapshot.report.events = push_recent_event(&snapshot.report.events, event_level, event_message);
    snapshot.report.summary = format!(
        "backend running with {} completed and {} failed model(s) out of {}",
        completed_models, failed_models, total_models
    );
    snapshot.report.log_path = Some(canonical_log_path().display().to_string());

    if let Some(error_text) = error_text {
        snapshot.report.warnings = push_unique_text(
            &snapshot.report.warnings,
            format!("model `{model}` failed during backend training"),
        );
        snapshot.report.errors = push_unique_text(&snapshot.report.errors, error_text);
    }
}

fn completed_snapshot_from_run_summary(
    snapshot: JobSnapshot,
    summary: TrainingRunSummary,
) -> JobSnapshot {
    let failed_names: Vec<String> = summary
        .failed_models
        .iter()
        .map(|failure| failure.name.clone())
        .collect();
    let mut completed = completed_snapshot_from(snapshot, summary.completed_models, failed_names);

    if !summary.failed_models.is_empty() {
        completed.report.errors =
            summary
                .failed_models
                .iter()
                .fold(completed.report.errors, |errors, failure| {
                    push_unique_text(
                        &errors,
                        format!("model {}: {}", failure.name, failure.error),
                    )
                });
        completed.report.warnings =
            summary
                .failed_models
                .iter()
                .fold(completed.report.warnings, |warnings, failure| {
                    push_unique_text(
                        &warnings,
                        format!("model `{}` failed during backend training", failure.name),
                    )
                });
        completed.report.entries =
            summary
                .failed_models
                .iter()
                .fold(completed.report.entries, |entries, failure| {
                    push_recent_entry(
                        &entries,
                        format!("failed | {} | {}", failure.name, failure.error),
                    )
                });
    }

    completed
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
    snapshot.progress.percent = Some(1.0);
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

#[cfg(test)]
pub fn failed_snapshot(err: anyhow::Error) -> JobSnapshot {
    failed_snapshot_from(JobSnapshot::new(JobKind::Training), err)
}

fn failed_snapshot_from(mut snapshot: JobSnapshot, err: anyhow::Error) -> JobSnapshot {
    let message = format!("{err:#}");
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
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<TrainingJobHandle> {
    start_training_job_impl(request, None, TrainingMode::TrainModels, tx)
}

pub fn start_discovery_training_job(
    request: TrainingRequest,
    identity: String,
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<TrainingJobHandle> {
    handoff::handoff_path(Path::new("data"), &identity)?;
    start_training_job_impl(request, Some(identity), TrainingMode::TrainModels, tx)
}

pub(crate) fn start_strategy_research_job(
    request: TrainingRequest,
    identity: String,
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<TrainingJobHandle> {
    handoff::handoff_path(Path::new("data"), &identity)?;
    start_training_job_impl(request, Some(identity), TrainingMode::StrategyOnly, tx)
}

fn start_training_job_impl(
    request: TrainingRequest,
    handoff_identity: Option<String>,
    mode: TrainingMode,
    tx: mpsc::Sender<ServiceEvent>,
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
            send_terminal_snapshot(&tx, &cancelled).await;
            log_training_event(
                "ui_training_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let settings_request = request.clone();
        let settings_and_models = match tokio::task::spawn_blocking(move || {
            let mut settings = Settings::from_yaml(&settings_request.config_path)?;
            let selected = handoff_identity
                .as_deref()
                .map(|identity| handoff::load(&settings.system.data_dir, identity))
                .transpose()?;
            let planned_models = if let Some(selected) = &selected {
                settings = handoff::settings_for_series(&settings, selected.canonical_series());
                anyhow::ensure!(
                    settings_request.symbol == settings.system.symbol
                        && settings_request.base_tf == selected.base_timeframe().as_str(),
                    "Training request labels differ from the selected handoff"
                );
                if mode == TrainingMode::TrainModels {
                    selected.validate_against_settings_v1(&settings)?;
                    selected.training_config().planned_models().to_vec()
                } else {
                    // load() already validated the complete sealed handoff.
                    // Strategy-only research uses its archived policy/recipe,
                    // not ambient model settings or a new model hardware plan.
                    Vec::new()
                }
            } else {
                settings.models.ml_models.clone()
            };
            Ok::<_, anyhow::Error>((settings, planned_models, selected))
        })
        .await
        {
            Ok(Ok(parts)) => parts,
            Ok(Err(err)) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_terminal_snapshot(&tx, &failed).await;
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
                return;
            }
            Err(err) => {
                let failed = failed_snapshot_from(
                    snapshot,
                    anyhow::anyhow!("training settings join error: {err}"),
                );
                send_terminal_snapshot(&tx, &failed).await;
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        let (settings, planned_models, selected_handoff) = settings_and_models;

        if mode == TrainingMode::StrategyOnly {
            let Some(selected) = selected_handoff else {
                let failed = failed_snapshot_from(
                    snapshot,
                    anyhow::anyhow!(
                        "strategy-only final research requires an exact Discovery handoff"
                    ),
                );
                send_terminal_snapshot(&tx, &failed).await;
                return;
            };
            run_strategy_research_job(
                settings,
                request.models_dir.join("candidates"),
                selected,
                cancel,
                snapshot,
                tx,
            )
            .await;
            return;
        }

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
            send_terminal_snapshot(&tx, &cancelled).await;
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

        let live_snapshot = Arc::new(Mutex::new(snapshot.clone()));
        let train_request = request.clone();
        let tx_progress = tx.clone();
        let live_snapshot_for_progress = Arc::clone(&live_snapshot);
        // Install the cancel flag the training loop polls between models so Stop
        // halts training mid-run (single-instance → a process-global is safe).
        neoethos_models::set_training_cancel(Some(cancel.cancel_arc()));
        let combined_cancel = cancel.clone();
        let combined_snapshot = Arc::clone(&live_snapshot);
        let combined_tx = tx.clone();
        let train_result = tokio::task::spawn_blocking(move || {
            let installed = neoethos_core::execution_budget::installed_process_budget()
                .context("training requires the immutable process CPU budget")?;
            let width = installed.resolved().effective_worker_limit;
            let lease = installed
                .broker()
                .acquire_cancellable(
                    neoethos_core::execution_budget::CpuPermitRequest::local(width),
                    combined_cancel.cpu_budget_token(),
                )
                .context("acquire process CPU budget for model training")?;
            let progress = move |event| {
                if let Ok(mut snapshot) = live_snapshot_for_progress.lock() {
                    apply_backend_progress_event(&mut snapshot, &event);
                    send_event(
                        &tx_progress,
                        ServiceEvent::TrainingUpdated(snapshot.clone()),
                    );
                }
            };
            if let Some(handoff) = selected_handoff {
                let identity = handoff.identity_sha256()?;
                let candidate_root = train_request.models_dir.join("candidates");
                let terminal = train_and_deploy_promotion_candidate_v1(
                    &settings,
                    handoff,
                    &candidate_root,
                    &settings.system.data_dir,
                    &lease,
                    progress,
                );
                let manifest = match terminal {
                    PromotionCandidateTrainingTerminalV1::Installed(manifest)
                    | PromotionCandidateTrainingTerminalV1::ExistingIdentical(manifest) => manifest,
                    PromotionCandidateTrainingTerminalV1::Refused(error) => {
                        return Err(error.into());
                    }
                };
                let model_names = manifest
                    .model_artifacts()
                    .iter()
                    .map(|model| model.model_name().to_owned())
                    .collect::<Vec<_>>();
                let installed_path = candidate_root.join(manifest.candidate_relative_dir());
                neoethos_core::storage::json::write_json_atomic(
                    candidate_root.join(format!("{identity}.manifest.json")),
                    &manifest,
                )?;
                // Training output and the inference consumer must meet here:
                // never call a saved candidate usable while the loader still
                // reads an unrelated legacy models/<symbol>/<tf> directory.
                // Keep the manifest if loading fails so the actual trained
                // output remains identifiable and the failure is diagnosable.
                let ensemble = neoethos_models::ensemble_inference::bootstrap::build_ensemble_for_candidate(
                    &candidate_root,
                    &manifest,
                    &settings,
                )
                .with_context(|| format!(
                    "candidate models were saved at {}, but their ensemble could not load; combined validation and deployment remain unavailable",
                    installed_path.display()
                ))?;
                tracing::info!(
                    candidate = %identity,
                    models = model_names.len(),
                    voters = ensemble.voting_expert_count(),
                    "exact candidate models reloaded through inference adapters; combined OOS validation is still required"
                );
                // The training lease cannot remain reserved while the next
                // phase acquires/splits the same process CPU capacity.
                drop(lease);
                let report_path = combined::evaluate_candidate(
                    &settings,
                    &candidate_root,
                    &manifest,
                    &ensemble,
                    &combined_cancel,
                    &|stage, completed, total| {
                        if let Ok(mut snapshot) = combined_snapshot.lock() {
                            let Some(percent) = combined_progress_percent(stage, completed, total) else {
                                return;
                            };
                            let stage = format!("combined_{stage}");
                            if snapshot.progress.percent.is_some_and(|old| old > percent)
                            {
                                return;
                            }
                            snapshot.progress = JobProgress {
                                stage,
                                percent: Some(percent),
                                message: format!("combined strategy/model bar research: {completed}/{total}"),
                            };
                            send_event(&combined_tx, ServiceEvent::TrainingUpdated(snapshot.clone()));
                        }
                    },
                ).with_context(|| format!(
                    "candidate models remain saved at {}, but combined bar research failed; no combined result or promotion is claimed",
                    installed_path.display()
                ))?;
                Ok((
                    TrainingRunSummary {
                        planned_models: model_names.clone(),
                        completed_models: model_names,
                        failed_models: Vec::new(),
                    },
                    Some((installed_path, report_path)),
                ))
            } else {
                TrainingOrchestrator::new(settings, train_request.models_dir.clone())
                    .train_symbol_with_progress(
                        &train_request.symbol,
                        &train_request.base_tf,
                        &lease,
                        progress,
                    )
                    .map(|summary| (summary, None))
            }
        })
        .await;
        // Clear the training cancel flag now the blocking run has returned.
        neoethos_models::set_training_cancel(None);

        // Operator Stop during training: report a clean CANCELLED (the loop broke
        // after the current model) rather than a partial SUCCESS.
        if cancel.is_requested() {
            let base_snapshot = live_snapshot
                .lock()
                .map(|snapshot| snapshot.clone())
                .unwrap_or(snapshot);
            let cancelled = cancelled_snapshot_from(
                base_snapshot,
                "operator cancelled training during model training",
            );
            send_terminal_snapshot(&tx, &cancelled).await;
            log_training_event(
                "ui_training_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        match train_result {
            Ok(Ok((summary, installed_path))) => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let mut completed = completed_snapshot_from_run_summary(base_snapshot, summary);
                if let Some((path, report_path)) = installed_path {
                    completed
                        .report
                        .highlights
                        .push(("candidate_models".to_owned(), path.display().to_string()));
                    completed.report.highlights.push((
                        "combined_bar_research".to_owned(),
                        report_path.display().to_string(),
                    ));
                    completed.report.summary.push_str(&format!("; candidate models installed at {}; gene-only and combined model bar research saved at {} (reserved final window, not broker/live admission)", path.display(), report_path.display()));
                }
                send_terminal_snapshot(&tx, &completed).await;
                log_training_event(
                    "ui_training_job",
                    "SUCCESS",
                    completed.report.summary.clone(),
                );
            }
            Ok(Err(err)) => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let failed = failed_snapshot_from(base_snapshot, err);
                send_terminal_snapshot(&tx, &failed).await;
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
            }
            Err(err) => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let failed = failed_snapshot_from(
                    base_snapshot,
                    anyhow::anyhow!("training join error: {err}"),
                );
                send_terminal_snapshot(&tx, &failed).await;
                log_training_event("ui_training_job", "FAILED", failed.report.summary.clone());
            }
        }
    });

    Ok(handle)
}

async fn run_strategy_research_job(
    settings: Settings,
    candidate_root: PathBuf,
    handoff: neoethos_models::PromotionCandidateTrainingHandoffV1,
    cancel: CancellationFlag,
    mut snapshot: JobSnapshot,
    tx: mpsc::Sender<ServiceEvent>,
) {
    snapshot.progress = JobProgress {
        percent: Some(0.35),
        stage: "strategy_search_features".to_owned(),
        message: "reconstructing saved strategy features for the reserved final window".to_owned(),
    };
    snapshot
        .report
        .highlights
        .push(("evaluation_mode".to_owned(), "strategy_only".to_owned()));
    snapshot.report.summary =
        "evaluating locked strategies without model training or inference".to_owned();
    send_event(&tx, ServiceEvent::TrainingUpdated(snapshot.clone()));
    let observed = Arc::new(Mutex::new(snapshot.clone()));
    let progress_snapshot = Arc::clone(&observed);
    let progress_tx = tx.clone();
    let worker_cancel = cancel.clone();
    let result = tokio::task::spawn_blocking(move || {
        combined::evaluate_strategies(
            &settings,
            &candidate_root,
            &handoff,
            &worker_cancel,
            &|stage, completed, total| {
                let Some(percent) = strategy_progress_percent(stage, completed, total) else {
                    return;
                };
                if let Ok(mut current) = progress_snapshot.lock() {
                    if current.progress.percent.is_some_and(|old| old > percent) {
                        return;
                    }
                    current.progress = JobProgress {
                        percent: Some(percent),
                        stage: format!("strategy_{stage}"),
                        message: format!(
                            "strategy-only final research: {stage} {completed}/{total}"
                        ),
                    };
                    send_event(&progress_tx, ServiceEvent::TrainingUpdated(current.clone()));
                }
            },
        )
    })
    .await;
    let snapshot = observed
        .lock()
        .map(|current| current.clone())
        .unwrap_or(snapshot);
    let terminal = if cancel.is_requested() {
        cancelled_snapshot_from(snapshot, "operator cancelled strategy-only final research")
    } else {
        match result {
            Ok(Ok(path)) => {
                let mut completed = snapshot;
                completed.state = JobState::Succeeded;
                completed.progress = JobProgress {
                    percent: Some(1.0),
                    stage: "strategy_research_completed".to_owned(),
                    message: "strategy-only final research saved; no promotion or live admission"
                        .to_owned(),
                };
                completed.report.highlights.push((
                    "strategy_bar_research".to_owned(),
                    path.display().to_string(),
                ));
                completed.report.summary = format!(
                    "locked strategies evaluated on the reserved final window; report saved at {} (research only; no models trained or trading permission)",
                    path.display()
                );
                completed
            }
            Ok(Err(error)) => failed_snapshot_from(
                snapshot,
                error.context("strategy-only final research failed"),
            ),
            Err(error) => failed_snapshot_from(
                snapshot,
                anyhow::anyhow!("strategy-only final research worker failed: {error}"),
            ),
        }
    };
    send_terminal_snapshot(&tx, &terminal).await;
    log_training_event(
        "ui_strategy_research",
        &format!("{:?}", terminal.state),
        terminal.report.summary.clone(),
    );
}

fn strategy_progress_percent(stage: &str, completed: usize, total: usize) -> Option<f32> {
    let (start, end) = match stage {
        "search_features" => (0.35, 0.9),
        "account_replay" => (0.9, 0.99),
        _ => return None,
    };
    let fraction = if total == 0 {
        0.0
    } else {
        completed.min(total) as f32 / total as f32
    };
    Some(start + (end - start) * fraction)
}

async fn send_terminal_snapshot(tx: &mpsc::Sender<ServiceEvent>, snapshot: &JobSnapshot) {
    if let Err(error) = tx
        .send(ServiceEvent::TrainingUpdated(snapshot.clone()))
        .await
    {
        tracing::error!(%error, "Training terminal receiver is closed");
    }
}

fn send_event(tx: &mpsc::Sender<ServiceEvent>, event: ServiceEvent) {
    if let Err(err) = tx.try_send(event) {
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
    // F-282 fix (2026-05-25): never panic on pre-1970 clock skew.
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(now) => format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::training",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "pre-1970.000000000Z".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::{
        ServiceEvent,
        jobs::{JobEventLevel, JobState},
    };
    use neoethos_models::ModelTrainingProgress;
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

    #[tokio::test]
    async fn training_terminal_waits_for_capacity_and_survives_progress_backpressure() {
        for terminal in [
            cancelled_snapshot("operator stop"),
            failed_snapshot(anyhow::anyhow!("model failure")),
            completed_snapshot(vec!["xgboost".to_owned()], Vec::new()),
        ] {
            let (tx, mut rx) = mpsc::channel(1);
            tx.send(ServiceEvent::TrainingUpdated(JobSnapshot::new(
                JobKind::Training,
            )))
            .await
            .unwrap();
            let expected = terminal.clone();
            let delivery = tokio::spawn(async move {
                send_terminal_snapshot(&tx, &terminal).await;
            });
            tokio::task::yield_now().await;
            assert!(!delivery.is_finished());
            rx.recv().await.unwrap();
            let delivered = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap();
            let ServiceEvent::TrainingUpdated(delivered) = delivered else {
                panic!("wrong event")
            };
            assert_eq!(delivered, expected);
            delivery.await.unwrap();
        }
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
        assert!(
            snapshot
                .report
                .highlights
                .iter()
                .any(|(name, value)| { name == "completed_models" && value == "2" })
        );
        assert!(
            snapshot
                .report
                .highlights
                .iter()
                .any(|(name, value)| { name == "failed_models" && value == "1" })
        );
        assert!(
            snapshot
                .report
                .entries
                .iter()
                .any(|entry| entry == "completed | xgboost")
        );
        assert!(
            snapshot
                .report
                .entries
                .iter()
                .any(|entry| entry == "failed | mlp")
        );
        assert!(
            snapshot
                .report
                .events
                .iter()
                .any(|event| event.message.contains("training finished"))
        );
        assert_eq!(
            snapshot.report.events.last().map(|event| event.level),
            Some(JobEventLevel::Warning)
        );
        assert!(snapshot.report.summary.contains("xgboost"));
        assert!(snapshot.report.summary.contains("mlp"));
    }

    #[test]
    fn combined_progress_occupies_ordered_final_phases_without_restarting_or_finishing_early() {
        let mut previous = backend_progress_percent(3, 0, 3).unwrap();
        for stage in [
            "search_features",
            "model_features",
            "causal_model_inference",
            "account_replay",
        ] {
            for done in [0, 1, 2] {
                let current = combined_progress_percent(stage, done, 2).unwrap();
                assert!(
                    current >= previous && current < 1.0,
                    "{stage}: {previous} -> {current}"
                );
                previous = current;
            }
        }
        assert_eq!(previous, 0.99);
        assert_eq!(
            combined_progress_percent("search_features", 99, 2),
            Some(0.88)
        );
        assert_eq!(
            combined_progress_percent("model_features", 0, 0),
            Some(0.88)
        );
        assert_eq!(combined_progress_percent("unknown", 1, 1), None);
        assert_eq!(
            completed_snapshot(vec!["xgboost".into()], vec![])
                .progress
                .percent,
            Some(1.0)
        );
    }

    #[test]
    fn strategy_progress_tracks_features_and_replay_without_a_model_phase() {
        let mut previous = 0.35;
        for stage in ["search_features", "account_replay"] {
            for done in [0, 1, 2] {
                let current = strategy_progress_percent(stage, done, 2).unwrap();
                assert!(current >= previous && current < 1.0, "{stage}: {current}");
                previous = current;
            }
        }
        assert_eq!(previous, 0.99);
        assert_eq!(
            strategy_progress_percent("search_features", 99, 2),
            Some(0.9)
        );
        assert_eq!(strategy_progress_percent("account_replay", 0, 0), Some(0.9));
        for unentered in ["model_features", "causal_model_inference", "unknown"] {
            assert_eq!(strategy_progress_percent(unentered, 1, 1), None);
        }
    }

    #[test]
    fn training_failure_keeps_the_causal_error_chain_in_the_ui_report() {
        let error = anyhow::anyhow!("fitted column bits differ")
            .context("validate expert input")
            .context("candidate saved but combined research failed");
        let snapshot = failed_snapshot(error);
        for detail in [
            "candidate saved",
            "validate expert input",
            "fitted column bits differ",
        ] {
            assert!(snapshot.report.summary.contains(detail));
            assert!(snapshot.report.errors[0].contains(detail));
            assert!(
                snapshot
                    .report
                    .events
                    .last()
                    .unwrap()
                    .message
                    .contains(detail)
            );
        }
    }

    #[tokio::test]
    async fn start_training_job_emits_initial_snapshot_with_runtime_context() {
        let request = sample_request();
        let (tx, mut rx) = mpsc::channel(10000);

        let _handle = start_training_job(request.clone(), tx).expect("job should start");
        let event = rx.recv().await.expect("expected initial training event");
        let ServiceEvent::TrainingUpdated(snapshot) = event else {
            panic!("expected training update event");
        };

        assert_eq!(snapshot.state, JobState::Running);
        assert_eq!(snapshot.progress.stage, "loading_settings");
        assert!(
            snapshot
                .report
                .highlights
                .iter()
                .any(|(name, value)| name == "symbol" && value == "EURUSD")
        );
        assert!(
            snapshot
                .report
                .highlights
                .iter()
                .any(|(name, value)| name == "config_path" && value == "config.yaml")
        );
        assert!(snapshot.report.events.iter().any(|event| {
            event.message.contains("loading training settings") && event.message.contains("EURUSD")
        }));
        assert_eq!(
            snapshot.report.log_path,
            Some(canonical_log_path().display().to_string())
        );
    }

    #[test]
    fn backend_failure_event_updates_training_snapshot_with_live_model_status() {
        let request = sample_request();
        let mut snapshot = JobSnapshot::new(JobKind::Training);
        snapshot.state = JobState::Running;
        snapshot.progress = JobProgress {
            percent: Some(0.7),
            stage: "dispatching_backend".to_string(),
            message: "dispatching backend training for 3 planned model(s)".to_string(),
        };
        snapshot.report = JobReport {
            counters: vec![("planned_models".to_string(), 3)],
            highlights: requested_training_highlights(&request),
            entries: vec!["planned | xgboost".to_string()],
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };

        apply_backend_progress_event(
            &mut snapshot,
            &ModelTrainingProgress::Failed {
                model: "mlp".to_string(),
                error: "synthetic backend failure".to_string(),
                completed_models: 1,
                failed_models: 1,
                total_models: 3,
            },
        );

        assert_eq!(snapshot.state, JobState::Running);
        assert_eq!(snapshot.progress.stage, "backend_running");
        assert_eq!(snapshot.progress.percent, Some(0.8));
        assert!(
            snapshot
                .report
                .counters
                .iter()
                .any(|(name, value)| name == "completed_models" && *value == 1)
        );
        assert!(
            snapshot
                .report
                .counters
                .iter()
                .any(|(name, value)| name == "failed_models" && *value == 1)
        );
        assert!(
            snapshot
                .report
                .warnings
                .iter()
                .any(|warning| warning.contains("mlp"))
        );
        assert!(
            snapshot
                .report
                .errors
                .iter()
                .any(|error| error.contains("synthetic backend failure"))
        );
        assert!(snapshot
            .report
            .events
            .iter()
            .any(|event| event.level == JobEventLevel::Warning && event.message.contains("mlp")));
        assert!(
            snapshot
                .report
                .entries
                .iter()
                .any(|entry| entry.contains("failed | mlp"))
        );
    }
}
