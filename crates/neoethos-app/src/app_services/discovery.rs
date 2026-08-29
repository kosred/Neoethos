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
    CanonicalDatasetIdentity, CanonicalDatasetScope, CanonicalDatasetSeriesReceiptV1,
    CanonicalTimeframe, DatasetDiscovery, PinnedCanonicalSeriesV1, SelectedDatasetGenerationV1,
    SymbolDataset, discover_canonical_dataset_identities, pin_exact_canonical_series_v1,
    prepare_multitimeframe_features, require_direct_timeframes,
};
// `DiscoveryValidationGates` is used by the sibling tests file
// (`discovery_tests.rs::success_snapshot_carries_candidate_and_portfolio_counters`),
// not by anything in this module. Importing it gated on `#[cfg(test)]`
// keeps the release build clean while staying visible to tests via
// `use super::*;`.
#[cfg(test)]
use neoethos_search::DiscoveryValidationGates;
use neoethos_search::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchInputReceiptV2,
};
use neoethos_search::{
    DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
    PromotionSummaryAuthorityPayloadV3, PropFirmRiskRules, ensure_non_empty_portfolio,
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
    /// Exact manifests plus reader leases, with no decoded OHLCV values. The
    /// selected prepared CPU/native factory consumes this pin after device
    /// admission, so the worker cannot follow a newer `current` pointer.
    pub pinned_input: Arc<PinnedDiscoveryInput>,
    pub higher_tfs: Vec<String>,
    pub config: DiscoveryConfig,
    /// Prop-firm rule set applied to the OOS prop-firm validation pass.
    /// Defaults to `PropFirmRiskRules::default()` (FTMO-style) when the
    /// caller does not need to override per-challenge thresholds.
    pub prop_firm_rules: PropFirmRiskRules,
}

impl DiscoveryRequest {
    pub fn symbol(&self) -> &str {
        self.dataset_identity().symbol_name()
    }

    pub fn base_tf(&self) -> &'static str {
        self.dataset_identity().timeframe().as_str()
    }

    pub fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        self.pinned_input.receipt().anchor().identity()
    }

    pub fn validate(&self) -> Result<()> {
        if self.data_root.as_os_str().is_empty() {
            anyhow::bail!("discovery request data root must not be empty");
        }
        let higher = self.canonical_higher_timeframes()?;
        self.pinned_input.validate(&higher)?;
        Ok(())
    }

    fn canonical_higher_timeframes(&self) -> Result<Vec<String>> {
        canonical_higher_timeframes(self.dataset_identity().timeframe(), &self.higher_tfs)
    }
}

/// One immutable, directly downloaded/imported timeframe set held for the
/// complete discovery lifetime. Construction is private to the exact pinning
/// functions below so callers cannot pair a receipt with unrelated values.
#[derive(Debug)]
pub struct PinnedDiscoveryInput {
    receipt: CanonicalDatasetSeriesReceiptV1,
    pinned_series: Mutex<Option<PinnedCanonicalSeriesV1>>,
}

impl PinnedDiscoveryInput {
    pub const fn receipt(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.receipt
    }

    fn validate(&self, higher_tfs: &[String]) -> Result<()> {
        self.receipt.validate()?;
        let anchor = self.receipt.anchor().identity();
        let mut required = vec![anchor.timeframe()];
        for label in higher_tfs {
            let timeframe = label
                .parse::<CanonicalTimeframe>()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            if !required.contains(&timeframe) {
                required.push(timeframe);
            }
        }
        let received = self
            .receipt
            .direct_timeframes()
            .iter()
            .map(|selected| selected.identity().timeframe())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            required.len() == received.len()
                && required
                    .iter()
                    .all(|timeframe| received.contains(timeframe)),
            "pinned discovery receipt does not exactly match the requested direct timeframe set"
        );
        Ok(())
    }

    fn take_pinned_series_v1(&self) -> Result<PinnedCanonicalSeriesV1> {
        self.pinned_series
            .lock()
            .map_err(|_| anyhow::anyhow!("pinned canonical series lock is poisoned"))?
            .take()
            .context("pinned canonical series was already consumed by another factory")
    }
}

#[derive(Debug)]
pub struct DirectTimeframeAcquisitionRequired {
    missing: Vec<CanonicalDatasetIdentity>,
}

impl std::fmt::Display for DirectTimeframeAcquisitionRequired {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let missing = self
            .missing
            .iter()
            .map(CanonicalDatasetIdentity::to_path_component)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            formatter,
            "direct timeframe acquisition required before discovery: [{missing}]"
        )
    }
}

impl std::error::Error for DirectTimeframeAcquisitionRequired {}

/// Resolve metadata for the operator-selected anchor and every explicitly
/// requested higher timeframe, then pin exact manifests and reader leases
/// without decoding values. Missing data is an acquisition request, never an
/// instruction for the discovery worker to mutate its own inputs.
pub fn pin_discovery_input(
    root: &std::path::Path,
    anchor: SelectedDatasetGenerationV1,
    higher_tfs: &[String],
) -> Result<PinnedDiscoveryInput> {
    anchor.validate()?;
    anyhow::ensure!(!root.as_os_str().is_empty(), "discovery data root is empty");
    anyhow::ensure!(
        root.is_dir(),
        "discovery data root is not a directory: {}",
        root.display()
    );

    let base = anchor.identity().timeframe();
    let higher = canonical_higher_timeframes(base, higher_tfs)?;
    let mut required = vec![base];
    required.extend(higher.iter().map(|label| {
        label
            .parse::<CanonicalTimeframe>()
            .expect("canonical higher timeframe was already parsed")
    }));

    let inventory = DatasetDiscovery::scan_metadata(root)?;
    let mut selected = Vec::with_capacity(required.len());
    let mut missing = Vec::new();
    for timeframe in required {
        if timeframe == base {
            selected.push(anchor.clone());
            continue;
        }
        let identity = identity_for_timeframe(anchor.identity(), timeframe)?;
        let identity_path = identity.to_path_component();
        let matches = inventory
            .entries
            .iter()
            .filter(|entry| entry.dataset_identity == identity_path)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => {
                let expected_root = root.join(&identity_path);
                if expected_root.exists() {
                    let diagnostics = inventory
                        .skipped
                        .iter()
                        .filter(|skipped| skipped.path.starts_with(&expected_root))
                        .map(|skipped| {
                            format!("{}: {}", skipped.reason.category(), skipped.reason.detail())
                        })
                        .collect::<Vec<_>>();
                    anyhow::bail!(
                        "direct timeframe dataset {identity_path} exists but is not a verified canonical generation: [{}]",
                        diagnostics.join("; ")
                    );
                }
                missing.push(identity);
            }
            [entry] => selected.push(SelectedDatasetGenerationV1::new(
                identity,
                entry.generation.clone(),
                entry.manifest_binding_sha256.clone(),
            )?),
            _ => anyhow::bail!(
                "canonical inventory contains duplicate entries for exact identity {identity_path}"
            ),
        }
    }
    if !missing.is_empty() {
        return Err(DirectTimeframeAcquisitionRequired { missing }.into());
    }

    let receipt = CanonicalDatasetSeriesReceiptV1::new(anchor, selected.clone())?;
    let pinned_series = pin_exact_canonical_series_v1(root, receipt.clone())?;
    let pinned = PinnedDiscoveryInput {
        receipt,
        pinned_series: Mutex::new(Some(pinned_series)),
    };
    pinned.validate(&higher)?;
    Ok(pinned)
}

/// Non-interactive callers first resolve exactly one canonical identity, then
/// snapshot its current metadata into a typed receipt and enter the same exact
/// generation pinning path as the HTTP API.
pub fn pin_current_discovery_input(
    root: &std::path::Path,
    identity: &CanonicalDatasetIdentity,
    higher_tfs: &[String],
) -> Result<PinnedDiscoveryInput> {
    let manifest =
        neoethos_data::core::dataset_manifest::read_current_manifest_metadata(root, identity)?;
    pin_discovery_input(
        root,
        SelectedDatasetGenerationV1::from_manifest(&manifest)?,
        higher_tfs,
    )
}

fn canonical_higher_timeframes(
    base: CanonicalTimeframe,
    higher_tfs: &[String],
) -> Result<Vec<String>> {
    let mut parsed = Vec::with_capacity(higher_tfs.len());
    for raw in higher_tfs {
        let label = raw.trim().to_uppercase();
        let timeframe = label
            .parse::<CanonicalTimeframe>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            timeframe > base,
            "higher timeframe {timeframe} must be strictly above base {base}"
        );
        anyhow::ensure!(
            !parsed.contains(&timeframe),
            "duplicate higher timeframe {timeframe}"
        );
        parsed.push(timeframe);
    }
    Ok(parsed
        .into_iter()
        .map(|timeframe| timeframe.as_str().to_owned())
        .collect())
}

/// Background jobs do not have an interactive dataset picker. They may reuse
/// a symbol/timeframe hint only when it resolves to exactly one canonical
/// source/account series. Zero or multiple matches fail closed and print every
/// candidate identity so the operator can make the selection explicit.
pub fn resolve_unique_background_dataset_identity(
    root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> Result<CanonicalDatasetIdentity> {
    let identities = discover_canonical_dataset_identities(root, symbol)?;
    select_unique_background_identity(identities, symbol, base_tf)
}

fn select_unique_background_identity(
    identities: Vec<CanonicalDatasetIdentity>,
    symbol: &str,
    base_tf: &str,
) -> Result<CanonicalDatasetIdentity> {
    let timeframe = base_tf
        .trim()
        .to_uppercase()
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut all_candidates = identities
        .iter()
        .map(CanonicalDatasetIdentity::to_path_component)
        .collect::<Vec<_>>();
    all_candidates.sort();
    let mut matches = identities
        .into_iter()
        .filter(|identity| {
            identity.symbol_name().eq_ignore_ascii_case(symbol) && identity.timeframe() == timeframe
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(CanonicalDatasetIdentity::to_path_component);
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => anyhow::bail!(
            "background discovery found no exact canonical identity for {} {}; candidates=[{}]",
            symbol.trim().to_uppercase(),
            timeframe,
            all_candidates.join(", ")
        ),
        count => anyhow::bail!(
            "background discovery found {count} canonical identities for {} {}; explicit dataset identity required; candidates=[{}]",
            symbol.trim().to_uppercase(),
            timeframe,
            matches
                .iter()
                .map(CanonicalDatasetIdentity::to_path_component)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn identity_for_timeframe(
    selected: &CanonicalDatasetIdentity,
    timeframe: CanonicalTimeframe,
) -> Result<CanonicalDatasetIdentity> {
    let convention = selected.bar_timestamp_convention();
    match selected.scope() {
        CanonicalDatasetScope::External { source_namespace } => CanonicalDatasetIdentity::external(
            source_namespace.clone(),
            selected.symbol_name(),
            timeframe,
            convention,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string())),
        CanonicalDatasetScope::CTrader {
            environment,
            server,
            account_id,
            symbol_id,
        } => CanonicalDatasetIdentity::ctrader(
            *environment,
            server.clone(),
            *account_id,
            *symbol_id,
            selected.symbol_name(),
            timeframe,
            convention,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string())),
    }
}

fn required_direct_timeframes(request: &DiscoveryRequest) -> Result<Vec<CanonicalTimeframe>> {
    let mut required = Vec::new();
    let mut push = |timeframe: CanonicalTimeframe| {
        if !required.contains(&timeframe) {
            required.push(timeframe);
        }
    };
    push(request.dataset_identity().timeframe());
    for label in &request.higher_tfs {
        push(
            label
                .trim()
                .to_uppercase()
                .parse::<CanonicalTimeframe>()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        );
    }
    Ok(required)
}

fn validate_direct_timeframe_artifacts(
    dataset: &SymbolDataset,
    selected: &CanonicalDatasetIdentity,
    required: &[CanonicalTimeframe],
) -> Result<()> {
    require_direct_timeframes(dataset, selected, required)
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
        ("symbol".to_string(), request.symbol().to_owned()),
        ("base_tf".to_string(), request.base_tf().to_owned()),
        (
            "dataset_identity".to_string(),
            request.dataset_identity().to_path_component(),
        ),
        (
            "dataset_generation".to_string(),
            request
                .pinned_input
                .receipt()
                .anchor()
                .generation_id()
                .to_owned(),
        ),
        (
            "dataset_manifest_binding_sha256".to_string(),
            request
                .pinned_input
                .receipt()
                .anchor()
                .manifest_binding_sha256()
                .to_owned(),
        ),
        (
            "direct_dataset_generations".to_string(),
            request
                .pinned_input
                .receipt()
                .direct_timeframes()
                .iter()
                .map(|selected| {
                    format!(
                        "{}:{}:{}",
                        selected.identity().timeframe(),
                        selected.generation_id(),
                        selected.manifest_binding_sha256()
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
        ),
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
        DiscoveryProgress::StageAdvanced { stage, detail } => {
            // Boundary markers for the long, otherwise-silent post-GA blocks
            // (2026-07-20: a healthy run frozen at "quality_screen 95.5%" for
            // hours was mistaken for a hang and killed). Percent is a fixed,
            // monotonic per-stage map inside the 0.945–0.99 tail window.
            let percent = match *stage {
                "quality_screen" => 0.945,
                "selecting_portfolio" => 0.96,
                "validation_gates" => 0.975,
                "robustness_filters" => 0.985,
                // The holdout replay runs in the WRAPPER, after the inner
                // cycle's Completed (0.99) — same value avoids a visible
                // percent regression while the message switches to the tail.
                "holdout_forward_test" => 0.99,
                _ => 0.95,
            };
            snapshot.progress = JobProgress {
                percent: Some(percent),
                stage: (*stage).to_string(),
                message: detail.clone(),
            };
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!("stage advanced: {stage} — {detail}"),
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
        .map(|artifact| artifact.summary().metrics.sharpe)
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
    request.validate()?;
    request.higher_tfs = request.canonical_higher_timeframes()?;
    request.config.timeframe_label = request.base_tf().to_owned();
    request.config.evaluation_symbol = request.symbol().to_owned();
    request.config.higher_timeframes = request.higher_tfs.clone();

    let handle = DiscoveryJobHandle::new();
    let cancel = handle.cancel.clone();
    let mut snapshot = handle.snapshot.clone();
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.05),
        stage: "using_pinned_data".to_string(),
        message: format!(
            "using pinned exact dataset generation {} @ {}",
            request.pinned_input.receipt().anchor().generation_id(),
            request.dataset_identity().to_path_component()
        ),
    };
    snapshot.report = JobReport {
        counters: requested_discovery_counters(&request),
        highlights: requested_discovery_highlights(&request),
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "planned discovery for {} {} with population={}, generations={}, candidate_count={}, portfolio_size={}",
                request.symbol(),
                request.base_tf(),
                request.config.population,
                request.config.generations,
                request.config.candidate_count,
                request.config.portfolio_size
            ),
        ),
        summary: format!(
            "using pinned discovery dataset for {} on {}",
            request.symbol(),
            request.base_tf()
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));
    log_discovery_event(
        "ui_discovery_job",
        "STARTED",
        format!(
            "starting discovery for {} ({})",
            request.symbol(),
            request.dataset_identity().to_path_component()
        ),
    );

    tokio::spawn(async move {
        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled discovery before feature preparation",
            );
            send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let required_direct = match required_direct_timeframes(&request) {
            Ok(required) => required,
            Err(err) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };
        let selected_timeframe_count = request.pinned_input.receipt().direct_timeframes().len();

        snapshot.progress = JobProgress {
            percent: Some(0.35),
            stage: "preparing_features".to_string(),
            message: format!(
                "preparing multi-timeframe features for {}",
                request.symbol()
            ),
        };
        snapshot.report = JobReport {
            counters: requested_discovery_counters(&request)
                .into_iter()
                .chain(std::iter::once((
                    "selected_timeframes".to_string(),
                    selected_timeframe_count as u64,
                )))
                .collect(),
            highlights: requested_discovery_highlights(&request),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "selected {} exact timeframe generation(s) for {}",
                    selected_timeframe_count,
                    request.symbol()
                ),
            ),
            summary: format!(
                "selected {} exact timeframe generations for {}",
                selected_timeframe_count,
                request.symbol()
            ),
            log_path: Some(canonical_log_path().display().to_string()),
            ..JobReport::default()
        };
        send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled discovery after pinned-data validation",
            );
            send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        // Label the feature-cube build clearly so it never looks frozen. On
        // dense timeframes (M1–M5) this step can take hours; the operator used
        // to see a static bar with no idea whether it was working or stuck.
        snapshot.progress = JobProgress {
            percent: Some(0.4),
            stage: "building_features".to_string(),
            message: format!(
                "building the feature cube for {} ({}) — dense timeframes (M1–M5) \
                 can take a long time; the app is working, not frozen",
                request.symbol(),
                request.base_tf()
            ),
        };
        send_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

        let feature_request = request.clone();
        let feature_input = Arc::clone(&request.pinned_input);
        let feature_handle = tokio::task::spawn_blocking(move || {
            #[cfg(feature = "gpu-nvidia")]
            {
                neoethos_search::prepare_canonical_discovery_run_input_v3(
                    |no_physical_gpu_admission| {
                        let pinned_series = feature_input.take_pinned_series_v1()?;
                        let dataset = pinned_series.into_cpu_dataset_after_no_physical_gpu_v1(
                            &no_physical_gpu_admission,
                        )?;
                        validate_direct_timeframe_artifacts(
                            &dataset,
                            feature_request.dataset_identity(),
                            &required_direct,
                        )?;
                        let higher_refs: Vec<&str> = feature_request
                            .higher_tfs
                            .iter()
                            .map(|tf| tf.as_str())
                            .collect();
                        let features = prepare_multitimeframe_features(
                            &dataset,
                            feature_request.base_tf(),
                            &higher_refs,
                        )?;
                        let base_frame = dataset.canonical_frame(feature_request.base_tf())?;
                        let input = neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame(
                            feature_request.dataset_identity().clone(),
                            base_frame,
                            features,
                        )?;
                        Ok((input, no_physical_gpu_admission))
                    },
                    || {
                        anyhow::bail!(
                            "full native Discovery workspace-plan sealing is not integrated; refusing host feature materialization on a physical GPU"
                        )
                    },
                    |_admitted_native_run| {
                        anyhow::bail!(
                            "resident native Data materialization is unreachable until the complete workspace plan is sealed"
                        )
                    },
                )
            }
            #[cfg(not(feature = "gpu-nvidia"))]
            {
                let pinned_series = feature_input.take_pinned_series_v1()?;
                let dataset = pinned_series.into_cpu_dataset_without_native_adapter_v1()?;
                validate_direct_timeframe_artifacts(
                    &dataset,
                    feature_request.dataset_identity(),
                    &required_direct,
                )?;
                let higher_refs: Vec<&str> = feature_request
                    .higher_tfs
                    .iter()
                    .map(|tf| tf.as_str())
                    .collect();
                let features = prepare_multitimeframe_features(
                    &dataset,
                    feature_request.base_tf(),
                    &higher_refs,
                )?;
                let base_frame = dataset.canonical_frame(feature_request.base_tf())?;
                neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame(
                    feature_request.dataset_identity().clone(),
                    base_frame,
                    features,
                )
                .map_err(anyhow::Error::new)
            }
        });

        // Heartbeat: the build above is a single opaque call that can run for
        // hours, so tick an elapsed-time message every 15s. This proves the app
        // is alive (fixing the "frozen at a low %" confusion) without touching
        // the feature engine. It does NOT interrupt the build — closing the app
        // hard-stops everything; the pre-build cancel check already prevents a
        // fresh build from starting after Stop.
        let hb_tx = tx.clone();
        let mut hb_snapshot = snapshot.clone();
        let hb_symbol = request.symbol().to_owned();
        let hb_started = std::time::Instant::now();
        let heartbeat = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(15));
            ticker.tick().await; // consume the immediate first tick
            loop {
                ticker.tick().await;
                let secs = hb_started.elapsed().as_secs();
                hb_snapshot.progress.message = format!(
                    "building feature cube for {hb_symbol}… {}m {:02}s elapsed \
                     (dense timeframes can take hours; closing the app stops it)",
                    secs / 60,
                    secs % 60
                );
                send_event(&hb_tx, ServiceEvent::DiscoveryUpdated(hb_snapshot.clone()));
            }
        });

        let feature_build = feature_handle.await;
        heartbeat.abort();

        let prepared_input = match feature_build {
            Ok(Ok(prepared_input)) => prepared_input,
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
        #[cfg(feature = "gpu-nvidia")]
        let prepared_shape = prepared_input.shape();
        #[cfg(not(feature = "gpu-nvidia"))]
        let prepared_shape = Ok((
            prepared_input.features().n_samples(),
            prepared_input.features().n_features(),
        ));
        let (feature_rows, feature_columns) = match prepared_shape {
            Ok(shape) => shape,
            Err(err) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_event(&tx, ServiceEvent::DiscoveryUpdated(failed.clone()));
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };

        snapshot.progress = JobProgress {
            percent: Some(0.75),
            stage: "running_discovery".to_string(),
            message: format!("evaluating strategy candidates for {}", request.symbol()),
        };
        snapshot.report = JobReport {
            counters: requested_discovery_counters(&request)
                .into_iter()
                .chain([
                    ("feature_rows".to_string(), feature_rows as u64),
                    ("feature_columns".to_string(), feature_columns as u64),
                ])
                .collect(),
            highlights: requested_discovery_highlights(&request),
            events: push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "prepared feature frame {}x{} for {}",
                    feature_rows,
                    feature_columns,
                    request.symbol()
                ),
            ),
            summary: format!(
                "prepared {} rows x {} columns for discovery",
                feature_rows, feature_columns
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
        // Staleness heartbeat (2026-07-20): the post-GA stages can run for
        // HOURS between progress events on dense timeframes, and a frozen
        // percent reads as a hang (a healthy EURCAD M3 run was killed at
        // 95.5% for exactly this reason). Track when the last REAL event
        // landed; every 20s of silence past 45s, re-emit the current
        // snapshot with an appended "still working — Xm in this stage"
        // liveness note. The note is transient (never stored in the
        // snapshot), so real events always show their own clean message.
        let last_event_at = Arc::new(Mutex::new(std::time::Instant::now()));
        let last_event_for_progress = Arc::clone(&last_event_at);
        let hb_live_snapshot = Arc::clone(&live_snapshot);
        let hb_tx = tx.clone();
        let stale_heartbeat = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(20));
            ticker.tick().await; // consume the immediate first tick
            loop {
                ticker.tick().await;
                let silent_secs = last_event_at
                    .lock()
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                if silent_secs < 45 {
                    continue;
                }
                let Ok(current) = hb_live_snapshot.lock().map(|s| s.clone()) else {
                    continue;
                };
                let mut beat = current;
                beat.progress.message = format!(
                    "{} · still working — {}m {:02}s since the last update (long silent \
                     validation stages are NORMAL on dense timeframes; closing the app \
                     kills the run)",
                    beat.progress.message,
                    silent_secs / 60,
                    silent_secs % 60
                );
                send_event(&hb_tx, ServiceEvent::DiscoveryUpdated(beat));
            }
        });
        // Install the cancel flag the GA polls EACH GENERATION (discovery is
        // single-instance, so a process-global flag is safe). This makes Stop
        // interrupt the search mid-run instead of only at coarse phase boundaries.
        // Cleared right after the blocking search returns.
        neoethos_search::set_search_cancel(Some(cancel.cancel_arc()));
        let cancel_arc_for_closure = cancel.cancel_arc();
        let search_result = tokio::task::spawn_blocking(move || {
            // Outer OOS holdout (audit B02/B03, 2026-07-13): the 80/20
            // split + held-out-tail forward-test/prop-firm artifacts moved
            // INTO neoethos-search (`run_discovery_cycle_with_holdout_*`),
            // the single source of truth shared with the CLI and the batch
            // orchestrator — those two used to run discovery on the FULL
            // series with no holdout at all.
            let resolved_config = search_request.config.clone().apply_mode_overrides();
            let progress = move |event| {
                if let Ok(mut last) = last_event_for_progress.lock() {
                    *last = std::time::Instant::now();
                }
                if let Ok(mut snapshot) = live_snapshot_for_progress.lock() {
                    apply_backend_discovery_event(&mut snapshot, &event);
                    send_event(
                        &tx_progress,
                        ServiceEvent::DiscoveryUpdated(snapshot.clone()),
                    );
                }
            };
            #[cfg(feature = "gpu-nvidia")]
            let result =
                neoethos_search::run_prepared_canonical_discovery_with_holdout_and_progress_v3(
                    prepared_input,
                    &resolved_config,
                    search_request.prop_firm_rules,
                    progress,
                )?;
            #[cfg(not(feature = "gpu-nvidia"))]
            let result = {
                let run_input = prepared_input.as_run_input().map_err(anyhow::Error::new)?;
                neoethos_search::run_discovery_cycle_with_holdout_and_progress(
                    &run_input,
                    &resolved_config,
                    search_request.prop_firm_rules,
                    progress,
                )?
            };

            // Operator Stop during the GA: the search returned early with
            // whatever it had. Do NOT export a cancelled run — bail with a
            // sentinel the async side maps to a clean CANCELLED result.
            if cancel_arc_for_closure.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("__DISCOVERY_CANCELLED__");
            }

            // 2026-05-26 operator directive (dual-mode product): save the funnel
            // BEFORE the empty-portfolio check returns an error. The funnel is
            // the operator's main debugging artifact when the GA returns
            // nothing — bailing out with `ensure_non_empty_portfolio` here
            // without persisting the funnel would mean every empty run leaves
            // no trace of WHICH stage rejected everything.
            let funnel_out_path = PathBuf::from("cache").join("discovery").join(format!(
                "{}_{}.json",
                search_request.symbol(),
                search_request.base_tf()
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
                &format!("{} {}", search_request.symbol(), search_request.base_tf()),
            )?;

            // Forward-test + prop-firm artifacts on the held-out 20% tail are
            // already attached: `run_discovery_cycle_with_holdout_and_progress`
            // computes them inside neoethos-search (audit B02/B03).

            let out_path = PathBuf::from("cache").join("discovery").join(format!(
                "{}_{}.json",
                search_request.symbol(),
                search_request.base_tf()
            ));
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            save_portfolio_json(&out_path, &result)?;
            // Phase 4 (2026-06-04): self-describing live portfolio artifact for
            // the autonomous trader (parity with the CLI discover emit).
            // Additive + non-fatal.
            {
                let live_path = out_path.with_extension("live_portfolio.json");
                if let Err(err) = neoethos_search::save_live_portfolio_json(&live_path, &result) {
                    tracing::warn!(
                        target: "neoethos_app::discovery",
                        error = %err,
                        path = %live_path.display(),
                        "save_live_portfolio_json failed (non-fatal)"
                    );
                }
            }
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
        stale_heartbeat.abort();
        // Clear the GA cancel flag now the blocking search has returned.
        neoethos_search::set_search_cancel(None);

        let result = match search_result {
            Ok(Ok(result)) => result,
            // Operator Stop mid-search: a clean CANCELLED, not a failure.
            Ok(Err(err)) if err.to_string().contains("__DISCOVERY_CANCELLED__") => {
                let base_snapshot = live_snapshot
                    .lock()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or(snapshot);
                let cancelled = cancelled_snapshot_from(
                    base_snapshot,
                    "operator cancelled discovery during the search",
                );
                send_event(&tx, ServiceEvent::DiscoveryUpdated(cancelled.clone()));
                log_discovery_event(
                    "ui_discovery_job",
                    "CANCELLED",
                    cancelled.report.summary.clone(),
                );
                return;
            }
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
                symbol = %request.symbol(),
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
/// Version 3 is the fail-closed promotion evidence hand-off. It carries the
/// exact search receipt/config identity plus a typed copy of the canonical
/// promotion-summary envelope written from the same [`DiscoveryResult`]. The
/// currently embedded summary is diagnostic v3 evidence and cannot mint a
/// live-copy permit; search-core must first produce exact composite v3 scope.
/// Symbol/timeframe labels alone never authorize a copy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    pub search_input_receipt: CanonicalSearchInputReceiptV2,
    pub search_input_receipt_sha256: String,
    pub search_config_hash: String,
    pub promotion_summary_authority: StoredPromotionSummaryAuthorityV3,
    pub portfolio: Vec<ModelTargetEntry>,
}

/// Exact typed authority copied from the canonical promotion-summary sidecar.
/// The loader reloads the canonical file and requires whole-envelope equality;
/// this embedded value is not a reconstruction from request labels.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredPromotionSummaryAuthorityV3 {
    pub canonical_file_name: String,
    pub envelope: CanonicalSearchArtifactEnvelopeV2<PromotionSummaryAuthorityPayloadV3>,
}

/// One accepted strategy from the portfolio.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTargetEntry {
    pub strategy_id: String,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub trades_count: u64,
    /// **F-330**: peak-to-trough drawdown as a PERCENTAGE. The GA's
    /// `Gene::max_drawdown` is a fraction (0.25 = 25%); we ×100 at
    /// write time so the promotion gate + UI speak percentages
    /// consistently. Version 3 rejects a missing value; legacy targets are
    /// intentionally not promotion authorities.
    pub max_drawdown_pct: f64,
    /// **F-330**: gross profit / gross loss. Required by strict v3.
    pub profit_factor: f64,
}

/// Current `ModelTargetsFile::schema_version`. Bump when the schema
/// changes; the reader on the Training side asserts on this.
pub const MODEL_TARGETS_SCHEMA_VERSION: u32 = 3;

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

/// Canonical promotion authority stored beside the v3 target hand-off. Keeping
/// both under `data_root` avoids ambient-CWD lookup and makes the bound file
/// name deterministic for the strict reader.
pub fn promotion_summary_path_for(
    data_root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> std::path::PathBuf {
    data_root
        .join("discovery_targets")
        .join(format!("{symbol}_{base_tf}_promotion_summary.json"))
}

/// Write the model-targets file using neoethos_core's atomic-rename
/// helper (no partial files, no half-fsync risks).
fn write_model_targets_for_discovery(
    request: &DiscoveryRequest,
    result: &neoethos_search::DiscoveryResult,
) -> Result<()> {
    use neoethos_core::storage::json::write_json_atomic;
    let path = model_targets_path_for(&request.data_root, request.symbol(), request.base_tf());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create discovery_targets dir at {}", parent.display()))?;
    }
    let summary_path =
        promotion_summary_path_for(&request.data_root, request.symbol(), request.base_tf());
    save_promotion_summary_json(&summary_path, result).with_context(|| {
        format!(
            "write canonical promotion summary authority at {}",
            summary_path.display()
        )
    })?;
    let summary_bytes = std::fs::read(&summary_path).with_context(|| {
        format!(
            "reload canonical promotion summary authority at {}",
            summary_path.display()
        )
    })?;
    let summary_authority =
        CanonicalSearchArtifactEnvelopeV2::<PromotionSummaryAuthorityPayloadV3>::from_json_bytes(
            &summary_bytes,
        )
        .map_err(anyhow::Error::new)?;
    result.validate_evaluated_scopes()?;
    let expected_scope = result.selection_scope()?.clone();
    summary_authority
        .validate_against(
            PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
            &result.search_config_hash,
            &result.search_input_receipt,
            expected_scope.evaluated_window(),
        )
        .map_err(anyhow::Error::new)?;
    let search_input_receipt_sha256 = result.search_input_receipt_sha256()?;
    anyhow::ensure!(
        summary_authority.scope().receipt_sha256() == search_input_receipt_sha256,
        "canonical promotion summary receipt digest disagrees with DiscoveryResult"
    );
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
        symbol: request.symbol().to_owned(),
        base_tf: request.base_tf().to_owned(),
        higher_tfs: request.higher_tfs.clone(),
        discovered_at_utc: now,
        search_input_receipt: result.search_input_receipt.clone(),
        search_input_receipt_sha256: result.search_input_receipt_sha256()?,
        search_config_hash: result.search_config_hash.clone(),
        promotion_summary_authority: StoredPromotionSummaryAuthorityV3 {
            canonical_file_name: summary_path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .context("canonical promotion summary filename is not UTF-8")?
                .to_owned(),
            envelope: summary_authority,
        },
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
