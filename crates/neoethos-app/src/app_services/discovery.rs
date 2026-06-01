use crate::app_services::{
    ServiceEvent,
    jobs::{
        CancellationFlag, JobEventLevel, JobKind, JobProgress, JobReport, JobSnapshot, JobState,
        push_recent_event,
    },
};
use anyhow::{Context, Result};
use neoethos_core::{
    logging::{canonical_log_path, write_subsystem_record},
    sectioned_log::{SectionedRunRecord, SubsystemSection},
};
use neoethos_data::{
    FeatureCache, MANDATORY_TFS, ensure_timeframes_with_resample, load_symbol_dataset,
    prepare_multitimeframe_features,
};
// `DiscoveryValidationGates` is used by the sibling tests file
// (`discovery_tests.rs::success_snapshot_carries_candidate_and_portfolio_counters`),
// not by anything in this module. Importing it gated on `#[cfg(test)]`
// keeps the release build clean while staying visible to tests via
// `use super::*;`.
#[cfg(test)]
use neoethos_search::DiscoveryValidationGates;
use neoethos_search::{
    DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PropFirmRiskRules,
    compute_discovery_forward_test_artifacts, compute_discovery_prop_firm_artifacts,
    ensure_non_empty_portfolio, run_discovery_cycle_with_progress,
    save_canonical_backtest_artifacts, save_discovery_profile_json,
    save_forward_test_validation_artifacts, save_funnel_json, save_portfolio_json,
    save_promotion_summary_json, save_prop_firm_validation_artifacts, save_quality_report_json,
    save_trade_log_json, save_walkforward_validation_artifacts,
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
    /// Prop-firm rule set applied to the OOS prop-firm validation pass.
    /// Defaults to `PropFirmRiskRules::default()` (FTMO-style) when the
    /// caller does not need to override per-challenge thresholds.
    pub prop_firm_rules: PropFirmRiskRules,
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
    let mut counters = vec![
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
    ];
    if request.config.max_rows > 0 {
        counters.push(("max_rows".to_string(), request.config.max_rows as u64));
    }
    counters
}

fn requested_discovery_highlights(request: &DiscoveryRequest) -> Vec<(String, String)> {
    let mut highlights = vec![
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
    ];
    if request.config.max_hours > 0.0 {
        highlights.push((
            "time_budget".to_string(),
            format!("{:.2}h", request.config.max_hours),
        ));
    }
    if request.config.filtering.use_opportunistic_candidates {
        highlights.push((
            "quality_lane".to_string(),
            "strict+opportunistic".to_string(),
        ));
    }
    highlights
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
        DiscoveryProgress::QualityScreened {
            strict_passed,
            opportunistic_passed,
            evaluated_candidates,
            logged_trade_sets,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.955),
                stage: "quality_screen".to_string(),
                message: format!(
                    "quality screen kept {} strict and {} opportunistic candidates",
                    strict_passed, opportunistic_passed
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "quality_screened",
                (*strict_passed + *opportunistic_passed) as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "opportunistic_candidates",
                *opportunistic_passed as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "trade_logs",
                *logged_trade_sets as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "quality screen kept {} strict + {} opportunistic out of {} candidates",
                    strict_passed, opportunistic_passed, evaluated_candidates
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
    let quality_by_strategy = result
        .quality_metrics
        .iter()
        .map(|metrics| (metrics.strategy_id.as_str(), metrics))
        .collect::<std::collections::HashMap<_, _>>();
    let best_gene = result.portfolio.iter().max_by(|left, right| {
        left.fitness
            .partial_cmp(&right.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut highlights = vec![
        ("accepted".to_string(), portfolio.to_string()),
        ("rejected".to_string(), rejected.to_string()),
    ];
    if !result.quality_metrics.is_empty() {
        let strict_count = result
            .quality_metrics
            .iter()
            .filter(|metrics| metrics.has_edge)
            .count();
        highlights.push((
            "quality_scored".to_string(),
            result.quality_metrics.len().to_string(),
        ));
        highlights.push(("quality_edge".to_string(), strict_count.to_string()));
    }
    if !result.logged_trades.is_empty() {
        highlights.push((
            "trade_logs".to_string(),
            result.logged_trades.len().to_string(),
        ));
    }
    if let Some(best_quality) = result.quality_metrics.iter().max_by(|left, right| {
        left.quality_score
            .partial_cmp(&right.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push((
            "best_quality".to_string(),
            format!("{:.1}", best_quality.quality_score),
        ));
        highlights.push((
            "best_quality_strategy".to_string(),
            best_quality.strategy_id.clone(),
        ));
    }
    if let Some(best) = best_gene {
        highlights.push(("best_strategy".to_string(), best.strategy_id.clone()));
        highlights.push((
            "best_sharpe".to_string(),
            format!("{:.2}", best.sharpe_ratio),
        ));
        highlights.push(("best_win_rate".to_string(), format!("{:.2}", best.win_rate)));
        // Surface the best gene's max-drawdown (as percent of equity)
        // so `--validation-mode` can record an OOS risk metric per TF
        // without re-reading the on-disk portfolio JSON. Additive
        // highlight — no existing reader keys off the highlights list
        // length, and the UI ignores unknown keys.
        highlights.push((
            "best_max_dd".to_string(),
            format!("{:.4}", best.max_drawdown),
        ));
    }
    // #211: surface the BEST Sharpe across the forward-test (OOS) tail
    // artifacts so `--validation-mode` can record both in-sample and
    // out-of-sample top-Sharpe per TF. `best_sharpe` above is in-sample
    // (stage-1) and is by construction what the GA optimized against —
    // it always looks inflated. The forward-test artifact is the
    // strictly-held-out 20% tail that the discovery cycle never trained
    // on, so its Sharpe is an unbiased OOS estimate.
    //
    // Empty `forward_test_validation_artifacts` (e.g. when the tail
    // window was too short or `compute_discovery_forward_test_artifacts`
    // failed) → no highlight emitted. The validation reader treats the
    // absence as `None` and falls back to in-sample reporting.
    if let Some(best_oos) = result
        .forward_test_validation_artifacts
        .iter()
        .map(|artifact| artifact.summary.metrics.sharpe)
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        highlights.push(("best_oos_sharpe".to_string(), format!("{:.4}", best_oos)));
    }
    let entries = result
        .portfolio
        .iter()
        .take(3)
        .map(|gene| {
            if let Some(metrics) = quality_by_strategy.get(gene.strategy_id.as_str()) {
                format!(
                    "{} | fitness={:.2} | quality={:.1} | monthly_win={:.2} | trades/mo={:.1} | edge={}",
                    gene.strategy_id,
                    gene.fitness,
                    metrics.quality_score,
                    metrics.monthly_win_rate,
                    metrics.trades_per_month,
                    metrics.has_edge
                )
            } else {
                format!(
                    "{} | fitness={:.2} | sharpe={:.2} | win_rate={:.2} | trades={}",
                    gene.strategy_id,
                    gene.fitness,
                    gene.sharpe_ratio,
                    gene.win_rate,
                    gene.trades_count
                )
            }
        })
        .collect();

    snapshot.state = JobState::Succeeded;
    snapshot.report = JobReport {
        counters: vec![
            ("candidates".to_string(), candidates),
            ("portfolio".to_string(), portfolio),
            ("rejected".to_string(), rejected),
            (
                "quality_scored".to_string(),
                result.quality_metrics.len() as u64,
            ),
            ("trade_logs".to_string(), result.logged_trades.len() as u64),
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

#[cfg(test)]
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
    mut request: DiscoveryRequest,
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<DiscoveryJobHandle> {
    request.config.timeframe_label = request.base_tf.clone();
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

        // ── Auto-fetch missing history (operator-chosen over a hard
        // fail) ───────────────────────────────────────────────────────
        // Discovery's `ensure_sufficient_history` floor (set via
        // FOREX_BOT_MIN_HISTORY_YEARS) aborts when the local cache is too
        // short. If the base timeframe is below the floor, pull the
        // required window straight from cTrader and reload before
        // building features — so Discovery runs on real broker history
        // instead of failing.
        let mut dataset = dataset;
        let min_years = std::env::var("FOREX_BOT_MIN_HISTORY_YEARS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        if min_years > 0 {
            let have = dataset
                .frames
                .get(&request.base_tf)
                .map(|f| f.close.len())
                .unwrap_or(0);
            let required = (min_years as usize).saturating_mul(
                neoethos_search::discovery::approx_bars_per_year(&request.base_tf),
            );
            if required > 0 && have < required {
                snapshot.progress = JobProgress {
                    percent: Some(0.15),
                    stage: "fetching_history".to_string(),
                    message: format!(
                        "history short for {} {} ({}/{} bars) — auto-fetching ~{}y from cTrader…",
                        request.symbol, request.base_tf, have, required, min_years
                    ),
                };
                send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));
                log_discovery_event(
                    "ui_discovery_job",
                    "FETCHING",
                    format!(
                        "auto-fetching ~{}y of {} {} from cTrader (have {} / need {} bars)",
                        min_years, request.symbol, request.base_tf, have, required
                    ),
                );
                let fetch_symbol = request.symbol.clone();
                let fetch_tf = request.base_tf.clone();
                let fetch_root = request.data_root.clone();
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);
                // Calendar window + 1-month cushion so weekend/holiday
                // gaps don't starve the trading-bar count.
                let day_ms: i64 = 24 * 60 * 60 * 1000;
                let from_ms = now_ms - ((min_years as i64) * 365 + 30) * day_ms;
                let fetched = tokio::task::spawn_blocking(move || {
                    crate::app_services::broker_api::download_history_blocking(
                        &fetch_symbol,
                        &fetch_tf,
                        from_ms,
                        now_ms,
                        &fetch_root,
                    )
                })
                .await;
                match fetched {
                    Ok(Ok(outcome)) => {
                        log_discovery_event(
                            "ui_discovery_job",
                            "FETCHED",
                            format!(
                                "auto-fetched {} bars of {} {} from cTrader",
                                outcome.bar_count, request.symbol, request.base_tf
                            ),
                        );
                        let reload = request.clone();
                        match tokio::task::spawn_blocking(move || {
                            load_symbol_dataset(&reload.data_root, &reload.symbol)
                        })
                        .await
                        {
                            Ok(Ok(reloaded)) => dataset = reloaded,
                            Ok(Err(err)) => {
                                let failed = failed_snapshot_from(snapshot, err);
                                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                                log_discovery_event(
                                    "ui_discovery_job",
                                    "FAILED",
                                    failed.report.summary.clone(),
                                );
                                return;
                            }
                            Err(err) => {
                                let failed = failed_snapshot_from(
                                    snapshot,
                                    anyhow::anyhow!("history reload join error: {err}"),
                                );
                                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                                log_discovery_event(
                                    "ui_discovery_job",
                                    "FAILED",
                                    failed.report.summary.clone(),
                                );
                                return;
                            }
                        }
                    }
                    Ok(Err(err)) => {
                        // Broker fetch failed — proceed and let the
                        // downstream `ensure_sufficient_history` surface
                        // the precise error rather than masking it here.
                        log_discovery_event(
                            "ui_discovery_job",
                            "FETCH_FAILED",
                            format!(
                                "auto-fetch of {} {} failed: {err}",
                                request.symbol, request.base_tf
                            ),
                        );
                    }
                    Err(err) => {
                        log_discovery_event(
                            "ui_discovery_job",
                            "FETCH_FAILED",
                            format!(
                                "auto-fetch join error for {} {}: {err}",
                                request.symbol, request.base_tf
                            ),
                        );
                    }
                }
            }
        }

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

            let resolved_config = search_request.config.clone().with_env_runtime_overrides();
            let mut result = run_discovery_cycle_with_progress(
                &is_features,
                &is_ohlcv,
                &resolved_config,
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
            // 2026-05-26 operator directive (dual-mode product): save the funnel
            // BEFORE the empty-portfolio check returns an error. The funnel is
            // the operator's main debugging artifact when the GA returns
            // nothing — bailing out with `ensure_non_empty_portfolio` here
            // without persisting the funnel would mean every empty run leaves
            // no trace of WHICH stage rejected everything.
            let funnel_out_path = PathBuf::from("cache").join("discovery").join(format!(
                "{}_{}.json",
                search_request.symbol, search_request.base_tf
            ));
            if let Some(parent) = funnel_out_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(err) = save_funnel_json(&funnel_out_path, &result) {
                tracing::warn!(
                    target: "neoethos_app::discovery",
                    error = %err,
                    "failed to save funnel JSON (non-fatal — discovery continues)"
                );
            }

            ensure_non_empty_portfolio(
                &result,
                &format!("{} {}", search_request.symbol, search_request.base_tf),
            )?;

            // Forward-test the portfolio on the strictly held-out 20% tail
            // (`wfv_bound..`). This is the OOS slice the discovery cycle
            // never saw, so the resulting forward-test summary is an
            // unbiased estimate of out-of-sample behavior.
            if !result.portfolio.is_empty() && wfv_bound < base_ohlcv.close.len() {
                let tail_ohlcv = neoethos_data::Ohlcv {
                    timestamp: base_ohlcv
                        .timestamp
                        .as_ref()
                        .map(|ts| ts[wfv_bound..].to_vec()),
                    open: base_ohlcv.open[wfv_bound..].to_vec(),
                    high: base_ohlcv.high[wfv_bound..].to_vec(),
                    low: base_ohlcv.low[wfv_bound..].to_vec(),
                    close: base_ohlcv.close[wfv_bound..].to_vec(),
                    volume: base_ohlcv.volume.as_ref().map(|v| v[wfv_bound..].to_vec()),
                };
                let tail_feat_start = wfv_bound.min(features.data.nrows());
                let tail_feat_rows = features.data.nrows().saturating_sub(tail_feat_start);
                if tail_feat_rows > 0 && !tail_ohlcv.close.is_empty() {
                    let tail_features = neoethos_data::FeatureFrame {
                        timestamps: features.timestamps[tail_feat_start..].to_vec(),
                        names: features.names.clone(),
                        data: features
                            .data
                            .slice(ndarray::s![tail_feat_start.., ..])
                            .to_owned(),
                    };
                    match compute_discovery_forward_test_artifacts(
                        &result.portfolio,
                        &result.effective_feature_names,
                        &tail_features,
                        &tail_ohlcv,
                        &resolved_config,
                    ) {
                        Ok(artifacts) => {
                            result.forward_test_validation_artifacts = artifacts;
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "neoethos_app::discovery",
                                error = %err,
                                "forward-test artifact computation failed; portfolio export \
                                 will proceed without forward-test evidence"
                            );
                        }
                    }

                    // Reuse the same OOS tail to compute prop-firm
                    // validation evidence. The rule set is sourced from
                    // the typed `DiscoveryRequest::prop_firm_rules`
                    // field so non-FTMO challenges drive the gate
                    // without code changes.
                    match compute_discovery_prop_firm_artifacts(
                        &result.portfolio,
                        &result.effective_feature_names,
                        &tail_features,
                        &tail_ohlcv,
                        &resolved_config,
                        search_request.prop_firm_rules,
                    ) {
                        Ok(artifacts) => {
                            result.prop_firm_validation_artifacts = artifacts;
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "neoethos_app::discovery",
                                error = %err,
                                "prop-firm artifact computation failed; portfolio export \
                                 will proceed without prop-firm evidence"
                            );
                        }
                    }
                }
            }

            let out_path = PathBuf::from("cache").join("discovery").join(format!(
                "{}_{}.json",
                search_request.symbol, search_request.base_tf
            ));
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            save_portfolio_json(&out_path, &result)?;
            save_discovery_profile_json(
                out_path.with_extension("profile.json"),
                &resolved_config,
                &result,
            )?;
            if !result.quality_metrics.is_empty() {
                save_quality_report_json(out_path.with_extension("quality.json"), &result)?;
            }
            if !result.logged_trades.is_empty() {
                save_trade_log_json(out_path.with_extension("trades.json"), &result)?;
            }
            if !result.canonical_backtest_artifacts.is_empty() {
                save_canonical_backtest_artifacts(
                    out_path.with_extension("canonical_backtests"),
                    &result,
                )?;
            }
            if !result.walkforward_validation_artifacts.is_empty() {
                save_walkforward_validation_artifacts(
                    out_path.with_extension("walkforward_validations"),
                    &result,
                )?;
            }
            if !result.forward_test_validation_artifacts.is_empty() {
                save_forward_test_validation_artifacts(
                    out_path.with_extension("forward_tests"),
                    &result,
                )?;
            }
            if !result.prop_firm_validation_artifacts.is_empty() {
                save_prop_firm_validation_artifacts(
                    out_path.with_extension("prop_firm_validations"),
                    &result,
                )?;
            }
            // Always emit a focused promotion summary so a UI / scraper
            // can poll one small file instead of parsing the full
            // profile JSON. Failures here are diagnostic, not blocking.
            if let Err(err) = save_promotion_summary_json(
                out_path.with_extension("promotion_summary.json"),
                &result,
            ) {
                tracing::warn!(
                    target: "neoethos_app::discovery",
                    error = %err,
                    "promotion summary export failed; profile JSON still carries the same data"
                );
            }
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
        // Task #6 — write `model_targets.json` so the training step
        // (Task #10's auto-trigger, or any operator-driven "Load
        // discovered targets" button in the Training panel) has a
        // stable on-disk hand-off from the discovery output. The
        // write is best-effort: a write failure logs a warning but
        // does NOT fail the discovery job, because the in-memory
        // snapshot we just emitted is the authoritative result.
        if let Err(err) = write_model_targets_for_discovery(&request, &result) {
            tracing::warn!(
                target: "neoethos_app::discovery::targets",
                error = %err,
                symbol = %request.symbol,
                "failed to write model_targets.json — operator can still inspect the discovery snapshot in-memory"
            );
        }
        send_event(&tx, ServiceEvent::DiscoveryUpdated(completed.clone()));
        log_discovery_event(
            "ui_discovery_job",
            "SUCCESS",
            completed.report.summary.clone(),
        );
    });

    Ok(handle)
}

/// On-disk contract between Discovery output and Training input.
/// Written by `start_discovery_job` after each successful job.
/// Filename: `<data_root>/discovery_targets/<symbol>_<base_tf>_model_targets.json`.
///
/// The schema is intentionally small — Training only needs to know
/// (a) which symbol/timeframe the portfolio targets and (b) which
/// strategies were accepted (so the operator can pick an ensemble
/// configuration around them). Quality metrics, candidates list,
/// and trade logs stay in the in-memory `DiscoveryResult` /
/// JobSnapshot path; this file is the minimal hand-off only.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelTargetsFile {
    /// Bump this whenever the schema changes incompatibly. Readers
    /// that see a version they don't recognise should refuse the
    /// file (NOT silently fall back).
    pub schema_version: u32,
    pub symbol: String,
    pub base_tf: String,
    pub higher_tfs: Vec<String>,
    /// ISO-8601 UTC at the moment the file was written.
    pub discovered_at_utc: String,
    pub portfolio: Vec<ModelTargetEntry>,
}

/// One accepted strategy from the portfolio.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelTargetEntry {
    pub strategy_id: String,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub trades_count: u64,
    /// **F-330**: peak-to-trough drawdown as a PERCENTAGE. The GA's
    /// `Gene::max_drawdown` is a fraction (0.25 = 25%); we ×100 at
    /// write time so the promotion gate + UI speak percentages
    /// consistently. `#[serde(default)]` keeps pre-F-330
    /// model_targets.json files readable (they get 0.0, which the gate
    /// treats as "no drawdown recorded" — a permissive default for
    /// legacy files rather than a spurious rejection).
    #[serde(default)]
    pub max_drawdown_pct: f64,
    /// **F-330**: gross profit / gross loss. Defaulted for backward
    /// compat with pre-F-330 files.
    #[serde(default)]
    pub profit_factor: f64,
}

/// Current `ModelTargetsFile::schema_version`. Bump when the schema
/// changes; the reader on the Training side asserts on this.
pub const MODEL_TARGETS_SCHEMA_VERSION: u32 = 1;

/// `discovery_targets/<symbol>_<base_tf>_model_targets.json` path
/// resolver. Public so Training can read the same path Discovery
/// writes (Task #10's job).
pub fn model_targets_path_for(
    data_root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> std::path::PathBuf {
    data_root
        .join("discovery_targets")
        .join(format!("{symbol}_{base_tf}_model_targets.json"))
}

/// Write the model-targets file using neoethos_core's atomic-rename
/// helper (no partial files, no half-fsync risks).
fn write_model_targets_for_discovery(
    request: &DiscoveryRequest,
    result: &neoethos_search::DiscoveryResult,
) -> Result<()> {
    use neoethos_core::storage::json::write_json_atomic;
    let path = model_targets_path_for(&request.data_root, &request.symbol, &request.base_tf);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create discovery_targets dir at {}", parent.display()))?;
    }
    let now = chrono::Utc::now().to_rfc3339();
    let portfolio: Vec<ModelTargetEntry> = result
        .portfolio
        .iter()
        .map(|gene| ModelTargetEntry {
            strategy_id: gene.strategy_id.clone(),
            fitness: gene.fitness,
            sharpe_ratio: gene.sharpe_ratio,
            win_rate: gene.win_rate,
            trades_count: gene.trades_count as u64,
            // F-330: Gene stores drawdown as a fraction; gate + UI use %.
            max_drawdown_pct: gene.max_drawdown * 100.0,
            profit_factor: gene.profit_factor,
        })
        .collect();
    let file = ModelTargetsFile {
        schema_version: MODEL_TARGETS_SCHEMA_VERSION,
        symbol: request.symbol.clone(),
        base_tf: request.base_tf.clone(),
        higher_tfs: request.higher_tfs.clone(),
        discovered_at_utc: now,
        portfolio,
    };
    write_json_atomic(&path, &file)
        .with_context(|| format!("write model_targets.json at {}", path.display()))?;
    tracing::info!(
        target: "neoethos_app::discovery::targets",
        path = %path.display(),
        portfolio_size = file.portfolio.len(),
        symbol = %file.symbol,
        base_tf = %file.base_tf,
        "wrote model_targets.json"
    );
    Ok(())
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
    // F-282 fix (2026-05-25): never panic on pre-1970 clock skew.
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(now) => format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::discovery",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "pre-1970.000000000Z".to_string()
        }
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
