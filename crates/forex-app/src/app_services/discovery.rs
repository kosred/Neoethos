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
};
use forex_data::{
    ensure_timeframes_with_resample, load_symbol_dataset, prepare_multitimeframe_features,
    FeatureCache, MANDATORY_TFS,
};
use forex_search::{
    ensure_non_empty_portfolio, run_discovery_cycle_with_progress, save_portfolio_json,
    DiscoveryConfig, DiscoveryProgress, DiscoveryResult,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct DiscoveryRequest {
    pub data_root: PathBuf,
    pub symbol: String,
    pub base_tf: String,
    pub higher_tfs: Vec<String>,
    pub config: DiscoveryConfig,
}

impl DiscoveryRequest {
    pub fn validate(&self) -> Result<()> {
        if self.symbol.trim().is_empty() {
            anyhow::bail!("discovery request symbol must not be empty");
        }
        if self.base_tf.trim().is_empty() {
            anyhow::bail!("discovery request base timeframe must not be empty");
        }
        if self.data_root.as_os_str().is_empty() {
            anyhow::bail!("discovery request data root must not be empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryJobHandle {
    pub snapshot: JobSnapshot,
    pub cancel: CancellationFlag,
}

impl DiscoveryJobHandle {
    pub fn new() -> Self {
        Self {
            snapshot: JobSnapshot::new(JobKind::Discovery),
            cancel: CancellationFlag::new(),
        }
    }
}

fn requested_discovery_counters(request: &DiscoveryRequest) -> Vec<(String, u64)> {
    vec![
        (
            "target_candidates".to_string(),
            request.config.candidate_count as u64,
        ),
        (
            "target_portfolio".to_string(),
            request.config.portfolio_size as u64,
        ),
        ("generations".to_string(), request.config.generations as u64),
        ("population".to_string(), request.config.population as u64),
    ]
}

fn requested_discovery_highlights(request: &DiscoveryRequest) -> Vec<(String, String)> {
    vec![
        ("symbol".to_string(), request.symbol.clone()),
        ("base_tf".to_string(), request.base_tf.clone()),
        (
            "higher_tfs".to_string(),
            if request.higher_tfs.is_empty() {
                "-".to_string()
            } else {
                request.higher_tfs.join(", ")
            },
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

fn apply_backend_discovery_event(snapshot: &mut JobSnapshot, event: &DiscoveryProgress) {
    match event {
        DiscoveryProgress::SearchStarted {
            population,
            generations,
            max_indicators,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.78),
                stage: "search_started".to_string(),
                message: format!(
                    "genetic search started with population={} and generations={}",
                    population, generations
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "population",
                *population as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "generations",
                *generations as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "max_indicators",
                *max_indicators as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "search started with population={} generations={} max_indicators={}",
                    population, generations, max_indicators
                ),
            );
        }
        DiscoveryProgress::GenerationCompleted {
            generation,
            total_generations,
            best_fitness,
            stagnant_generations,
            archived_profitable,
        } => {
            let ratio = if *total_generations == 0 {
                0.0
            } else {
                *generation as f32 / *total_generations as f32
            };
            snapshot.progress = JobProgress {
                percent: Some((0.8 + 0.1 * ratio).clamp(0.8, 0.9)),
                stage: "search_generations".to_string(),
                message: format!(
                    "generation {}/{} complete (best fitness {:.2})",
                    generation, total_generations, best_fitness
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "generation",
                *generation as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "archived_profitable",
                *archived_profitable as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "stagnant_generations",
                *stagnant_generations as u64,
            );
            snapshot.report.entries = push_recent_entry(
                &snapshot.report.entries,
                format!(
                    "generation | {}/{} | best_fitness={:.2} | archived={}",
                    generation, total_generations, best_fitness, archived_profitable
                ),
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "generation {}/{} completed with best fitness {:.2}",
                    generation, total_generations, best_fitness
                ),
            );
        }
        DiscoveryProgress::CandidatesRanked {
            candidate_count,
            truncated_to,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.91),
                stage: "ranking_candidates".to_string(),
                message: format!(
                    "ranked {} candidates and kept top {}",
                    candidate_count, truncated_to
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "candidates",
                *candidate_count as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "truncated_candidates",
                *truncated_to as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "ranked {} candidates and truncated to {}",
                    candidate_count, truncated_to
                ),
            );
        }
        DiscoveryProgress::CandidatesFiltered {
            passed_filters,
            evaluated_candidates,
            min_trades_required,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.94),
                stage: "filtering_candidates".to_string(),
                message: format!(
                    "{} of {} candidates passed filters",
                    passed_filters, evaluated_candidates
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "filtered_candidates",
                *passed_filters as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "min_trades_required",
                *min_trades_required as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "{} of {} candidates passed filters (min trades {})",
                    passed_filters, evaluated_candidates, min_trades_required
                ),
            );
        }
        DiscoveryProgress::PortfolioSelected {
            portfolio_size,
            rejected_by_correlation,
            target_portfolio,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.97),
                stage: "portfolio_construction".to_string(),
                message: format!(
                    "portfolio selection accepted {} of target {}",
                    portfolio_size, target_portfolio
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "portfolio",
                *portfolio_size as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "rejected_by_correlation",
                *rejected_by_correlation as u64,
            );
            snapshot.report.entries = push_recent_entry(
                &snapshot.report.entries,
                format!(
                    "portfolio | accepted={} | rejected_by_correlation={} | target={}",
                    portfolio_size, rejected_by_correlation, target_portfolio
                ),
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "portfolio selection accepted {} and rejected {} by correlation",
                    portfolio_size, rejected_by_correlation
                ),
            );
        }
        DiscoveryProgress::Completed {
            candidate_count,
            filtered_count,
            portfolio_size,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.99),
                stage: "finalizing_discovery".to_string(),
                message: format!(
                    "discovery finalized with {} portfolio strategies",
                    portfolio_size
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "candidates",
                *candidate_count as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "filtered_candidates",
                *filtered_count as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "portfolio",
                *portfolio_size as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "discovery finalized with {} candidates, {} filtered, {} portfolio",
                    candidate_count, filtered_count, portfolio_size
                ),
            );
        }
    }

    snapshot.report.log_path = Some(canonical_log_path().display().to_string());
}

pub fn completed_snapshot(mut snapshot: JobSnapshot, result: &DiscoveryResult) -> JobSnapshot {
    let candidates = result.candidates.len() as u64;
    let portfolio = result.portfolio.len() as u64;
    let rejected = candidates.saturating_sub(portfolio);
    let best_gene = result.portfolio.iter().max_by(|left, right| {
        left.fitness
            .partial_cmp(&right.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut highlights = vec![
        ("accepted".to_string(), portfolio.to_string()),
        ("rejected".to_string(), rejected.to_string()),
    ];
    if let Some(best) = best_gene {
        highlights.push(("best_strategy".to_string(), best.strategy_id.clone()));
        highlights.push((
            "best_sharpe".to_string(),
            format!("{:.2}", best.sharpe_ratio),
        ));
        highlights.push(("best_win_rate".to_string(), format!("{:.2}", best.win_rate)));
    }
    let entries = result
        .portfolio
        .iter()
        .take(3)
        .map(|gene| {
            format!(
                "{} | fitness={:.2} | sharpe={:.2} | win_rate={:.2} | trades={}",
                gene.strategy_id, gene.fitness, gene.sharpe_ratio, gene.win_rate, gene.trades_count
            )
        })
        .collect();

    snapshot.state = JobState::Succeeded;
    snapshot.report = JobReport {
        counters: vec![
            ("candidates".to_string(), candidates),
            ("portfolio".to_string(), portfolio),
            ("rejected".to_string(), rejected),
        ],
        highlights,
        entries,
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "completed discovery with {portfolio} portfolio strategies out of {candidates} candidates"
            ),
        ),
        summary: format!(
            "discovery completed with {} portfolio strategies out of {} candidates",
            portfolio, candidates
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

pub fn failed_snapshot(kind: JobKind, err: anyhow::Error) -> JobSnapshot {
    failed_snapshot_from(JobSnapshot::new(kind), err)
}

fn failed_snapshot_from(mut snapshot: JobSnapshot, err: anyhow::Error) -> JobSnapshot {
    let message = err.to_string();
    snapshot.state = JobState::Failed;
    snapshot.report = JobReport {
        errors: vec![message.clone()],
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Error,
            format!("discovery failed: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

#[cfg(test)]
pub fn cancelled_snapshot(kind: JobKind, message: impl Into<String>) -> JobSnapshot {
    cancelled_snapshot_from(JobSnapshot::new(kind), message)
}

fn cancelled_snapshot_from(mut snapshot: JobSnapshot, message: impl Into<String>) -> JobSnapshot {
    let message = message.into();
    snapshot.state = JobState::Cancelled;
    snapshot.report = JobReport {
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Warning,
            format!("discovery cancelled: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

pub fn start_discovery_job(
    request: DiscoveryRequest,
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<DiscoveryJobHandle> {
    request.validate()?;

    let handle = DiscoveryJobHandle::new();
    let cancel = handle.cancel.clone();
    let mut snapshot = handle.snapshot.clone();
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.05),
        stage: "loading_data".to_string(),
        message: format!("loading symbol dataset for {}", request.symbol),
    };
    snapshot.report = JobReport {
        counters: requested_discovery_counters(&request),
        highlights: requested_discovery_highlights(&request),
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "planned discovery for {} {} with population={}, generations={}, candidate_count={}, portfolio_size={}",
                request.symbol,
                request.base_tf,
                request.config.population,
                request.config.generations,
                request.config.candidate_count,
                request.config.portfolio_size
            ),
        ),
        summary: format!(
            "loading discovery dataset for {} on {}",
            request.symbol, request.base_tf
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));
    log_discovery_event(
        "ui_discovery_job",
        "STARTED",
        format!("starting discovery for {}", request.symbol),
    );

    tokio::spawn(async move {
        if cancel.is_requested() {
            let cancelled =
                cancelled_snapshot_from(snapshot, "operator cancelled discovery before data load");
            send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let load_request = request.clone();
        let dataset = match tokio::task::spawn_blocking(move || {
            load_symbol_dataset(&load_request.data_root, &load_request.symbol)
        })
        .await
        {
            Ok(Ok(dataset)) => dataset,
            Ok(Err(err)) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
            Err(err) => {
                let failed = failed_snapshot_from(
                    snapshot,
                    anyhow::anyhow!("discovery data load join error: {err}"),
                );
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        snapshot.progress = JobProgress {
            percent: Some(0.35),
            stage: "preparing_features".to_string(),
            message: format!("preparing multi-timeframe features for {}", request.symbol),
        };
        snapshot.report = JobReport {
            counters: requested_discovery_counters(&request)
                .into_iter()
                .chain(std::iter::once((
                    "loaded_timeframes".to_string(),
                    dataset.frames.len() as u64,
                )))
                .collect(),
            highlights: requested_discovery_highlights(&request),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "loaded {} timeframe frame(s) for {}",
                    dataset.frames.len(),
                    request.symbol
                ),
            ),
            summary: format!(
                "loaded {} timeframe frames for {}",
                dataset.frames.len(),
                request.symbol
            ),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };
        send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

        if cancel.is_requested() {
            let cancelled =
                cancelled_snapshot_from(snapshot, "operator cancelled discovery after data load");
            send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let feature_request = request.clone();
        let feature_build = tokio::task::spawn_blocking(move || {
            let dataset =
                ensure_timeframes_with_resample(&dataset, &feature_request.base_tf, MANDATORY_TFS)?;
            let higher_refs: Vec<&str> = feature_request
                .higher_tfs
                .iter()
                .map(|tf| tf.as_str())
                .collect();
            let features = prepare_multitimeframe_features(
                &dataset,
                &feature_request.base_tf,
                &higher_refs,
                Some(&FeatureCache::new("cache/features", 60, true)),
            )?;
            let base_ohlcv = dataset
                .frames
                .get(&feature_request.base_tf)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("base timeframe missing: {}", feature_request.base_tf)
                })?;
            Ok::<_, anyhow::Error>((features, base_ohlcv))
        })
        .await;

        let (features, base_ohlcv) = match feature_build {
            Ok(Ok(parts)) => parts,
            Ok(Err(err)) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
            Err(err) => {
                let failed = failed_snapshot_from(
                    snapshot,
                    anyhow::anyhow!("feature preparation join error: {err}"),
                );
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        snapshot.progress = JobProgress {
            percent: Some(0.75),
            stage: "running_discovery".to_string(),
            message: format!("evaluating strategy candidates for {}", request.symbol),
        };
        snapshot.report = JobReport {
            counters: requested_discovery_counters(&request)
                .into_iter()
                .chain([
                    ("feature_rows".to_string(), features.data.nrows() as u64),
                    ("feature_columns".to_string(), features.names.len() as u64),
                ])
                .collect(),
            highlights: requested_discovery_highlights(&request),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "prepared feature frame {}x{} for {}",
                    features.data.nrows(),
                    features.names.len(),
                    request.symbol
                ),
            ),
            summary: format!(
                "prepared {} rows x {} columns for discovery",
                features.data.nrows(),
                features.names.len()
            ),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };
        send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled discovery before portfolio construction",
            );
            send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let live_snapshot = Arc::new(Mutex::new(snapshot.clone()));
        let search_request = request.clone();
        let tx_progress = tx.clone();
        let live_snapshot_for_progress = Arc::clone(&live_snapshot);
        let search_result = tokio::task::spawn_blocking(move || {
            // Apply Walk-Forward Validation (WFV) strict bounds
            // Using 80% In-Sample for training, 20% Out-Of-Sample strictly withheld to eliminate look-ahead bias
            let n_rows = base_ohlcv.close.len();
            let wfv_bound = (n_rows as f64 * 0.8).floor() as usize;

            let mut is_ohlcv = base_ohlcv.clone();
            if let Some(ref mut ts) = is_ohlcv.timestamp {
                ts.truncate(wfv_bound);
            }
            is_ohlcv.open.truncate(wfv_bound);
            is_ohlcv.high.truncate(wfv_bound);
            is_ohlcv.low.truncate(wfv_bound);
            is_ohlcv.close.truncate(wfv_bound);
            if let Some(ref mut vol) = is_ohlcv.volume {
                vol.truncate(wfv_bound);
            }

            let wfv_feat_bound = wfv_bound.min(features.data.nrows());
            let mut is_features = features.clone();
            is_features.timestamps.truncate(wfv_feat_bound);
            let rows = features.data.nrows().min(wfv_feat_bound);
            is_features.data = features.data.slice(ndarray::s![..rows, ..]).to_owned();

            let result = run_discovery_cycle_with_progress(
                &is_features,
                &is_ohlcv,
                &search_request.config,
                move |event| {
                    if let Ok(mut snapshot) = live_snapshot_for_progress.lock() {
                        apply_backend_discovery_event(&mut snapshot, &event);
                        send_event(
                            &tx_progress,
                            ServiceEvent::DiscoveryUpdated(snapshot.clone()),
                        );
                    }
                },
            )?;
            ensure_non_empty_portfolio(
                &result,
                &format!("{} {}", search_request.symbol, search_request.base_tf),
            )?;

            let out_path = PathBuf::from("cache").join("discovery").join(format!(
                "{}_{}.json",
                search_request.symbol, search_request.base_tf
            ));
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            save_portfolio_json(&out_path, &result.portfolio, &features.names)?;
            Ok::<_, anyhow::Error>(result)
        })
        .await;

        let result = match search_result {
            Ok(Ok(result)) => result,
            Ok(Err(err)) => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let failed = failed_snapshot_from(base_snapshot, err);
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
            Err(err) => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let failed = failed_snapshot_from(
                    base_snapshot,
                    anyhow::anyhow!("discovery join error: {err}"),
                );
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        let base_snapshot = live_snapshot
            .lock()
            .map(|snapshot| snapshot.clone())
            .unwrap_or(snapshot);
        let completed = completed_snapshot(base_snapshot, &result);
        send_event(&tx, ServiceEvent::DiscoveryUpdated(completed.clone()));
        log_discovery_event(
            "ui_discovery_job",
            "SUCCESS",
            completed.report.summary.clone(),
        );
    });

    Ok(handle)
}

fn send_event(tx: &mpsc::Sender<ServiceEvent>, event: ServiceEvent) {
    if let Err(err) = tx.try_send(event) {
        tracing::error!("Failed to send discovery service event: {}", err);
    }
}

fn log_discovery_event(operation: &str, status: &str, message: String) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::Discovery,
        discovery_record(operation, status, message),
    ) {
        tracing::error!("Failed to write DISCOVERY section log: {}", err);
    }
}

fn discovery_record(operation: &str, status: &str, message: String) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!("discovery-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::Discovery,
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
        jobs::{JobKind, JobSnapshot, JobState},
        ServiceEvent,
    };
    use forex_search::Gene;
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    fn sample_request() -> DiscoveryRequest {
        DiscoveryRequest {
            data_root: PathBuf::from("data"),
            symbol: "EURUSD".to_string(),
            base_tf: "M1".to_string(),
            higher_tfs: vec!["M5".to_string(), "M15".to_string()],
            config: forex_search::DiscoveryConfig::default(),
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
        let snapshot = cancelled_snapshot(JobKind::Discovery, "operator cancelled discovery");

        assert_eq!(snapshot.state, JobState::Cancelled);
        assert_eq!(snapshot.report.summary, "operator cancelled discovery");
    }

    #[test]
    fn empty_portfolio_failure_maps_to_failed_snapshot() {
        let snapshot = failed_snapshot(
            JobKind::Discovery,
            anyhow::anyhow!("Discovery produced an empty portfolio for EURUSD M1 (candidates=4)"),
        );

        assert_eq!(snapshot.state, JobState::Failed);
        assert_eq!(snapshot.report.errors.len(), 1);
        assert!(snapshot.report.errors[0].contains("empty portfolio"));
    }

    #[test]
    fn success_snapshot_carries_candidate_and_portfolio_counters() {
        let best = Gene {
            strategy_id: "alpha-1".to_string(),
            fitness: 1450.0,
            sharpe_ratio: 1.82,
            win_rate: 0.64,
            ..Gene::default()
        };

        let second = Gene {
            strategy_id: "alpha-2".to_string(),
            fitness: 1200.0,
            sharpe_ratio: 1.55,
            win_rate: 0.59,
            ..Gene::default()
        };

        let result = DiscoveryResult {
            portfolio: vec![best.clone(), second],
            candidates: vec![best, Gene::default(), Gene::default()],
        };

        let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);

        assert_eq!(snapshot.state, JobState::Succeeded);
        assert_eq!(
            snapshot.report.counters,
            vec![
                ("candidates".to_string(), 3),
                ("portfolio".to_string(), 2),
                ("rejected".to_string(), 1),
            ]
        );
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_strategy" && value == "alpha-1" }));
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_sharpe" && value == "1.82" }));
        assert!(snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("alpha-1") && entry.contains("win_rate=0.64")));
        assert!(snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("completed discovery")));
    }

    #[tokio::test]
    async fn start_discovery_job_emits_initial_snapshot_with_requested_targets() {
        let mut request = sample_request();
        request.higher_tfs = vec!["M5".to_string(), "M15".to_string(), "H1".to_string()];
        request.config.population = 96;
        request.config.generations = 7;
        request.config.candidate_count = 144;
        request.config.portfolio_size = 24;
        let (tx, mut rx) = mpsc::channel(10000);

        let _handle = start_discovery_job(request.clone(), tx).expect("job should start");
        let event = rx.recv().await.expect("expected initial discovery event");
        let ServiceEvent::DiscoveryUpdated(snapshot) = event else {
            panic!("expected discovery update event");
        };

        assert_eq!(snapshot.state, JobState::Running);
        assert_eq!(snapshot.progress.stage, "loading_data");
        assert_eq!(
            snapshot.report.counters,
            vec![
                ("target_candidates".to_string(), 144),
                ("target_portfolio".to_string(), 24),
                ("generations".to_string(), 7),
                ("population".to_string(), 96),
            ]
        );
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "symbol" && value == "EURUSD"));
        assert!(snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "higher_tfs" && value == "M5, M15, H1"));
        assert!(snapshot.report.events.iter().any(|event| {
            event.message.contains("planned discovery")
                && event.message.contains("candidate_count=144")
                && event.message.contains("portfolio_size=24")
        }));
        assert_eq!(
            snapshot.report.log_path,
            Some(canonical_log_path().display().to_string())
        );
    }

    #[test]
    fn backend_portfolio_milestone_updates_discovery_snapshot_with_live_counts() {
        let request = sample_request();
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        snapshot.progress = JobProgress {
            percent: Some(0.75),
            stage: "running_discovery".to_string(),
            message: "evaluating strategy candidates for EURUSD".to_string(),
        };
        snapshot.report = JobReport {
            counters: requested_discovery_counters(&request),
            highlights: requested_discovery_highlights(&request),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };

        apply_backend_discovery_event(
            &mut snapshot,
            &forex_search::DiscoveryProgress::PortfolioSelected {
                portfolio_size: 12,
                rejected_by_correlation: 5,
                target_portfolio: 24,
            },
        );

        assert_eq!(snapshot.state, JobState::Running);
        assert_eq!(snapshot.progress.stage, "portfolio_construction");
        assert!(snapshot.progress.percent.expect("percent should exist") >= 0.9);
        assert!(snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "portfolio" && *value == 12));
        assert!(snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "rejected_by_correlation" && *value == 5));
        assert!(snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("portfolio selection")));
        assert!(snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("portfolio | accepted=12")));
    }
}
