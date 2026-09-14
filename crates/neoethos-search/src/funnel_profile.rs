//! Per-work-unit funnel profile — the JSON dump P4 calls for.
//!
//! Whenever a discovery work-unit (one symbol × one timeframe) finishes
//! we save a `<symbol>_<tf>_funnel.json` next to the portfolio output.
//! It captures the full rejection funnel so "no strategies" is debuggable
//! without re-running.
//!
//! Stages follow evaluation order. Portfolio capacity is downstream of WF;
//! planned evaluations, capacity exclusions and real failures are not synonyms.
//! Stages:
//!   1.  data_loaded
//!   2.  rows_after_trimming
//!   3.  features_built
//!   4.  features_after_prefilter
//!   5.  stage1_candidates_generated
//!   6.  profitable_archive_size
//!   7.  validation_candidates_admitted
//!   8.  passed_base_filter
//!   9.  nonzero_signals
//!   10. passed_min_trades
//!   11. full_is_evaluated
//!   12. passed_quality
//!   13. passed_prop_firm_window
//!   14. passed_walkforward
//!   15. passed_selection_calibration
//!   16. passed_correlation
//!   17. portfolio_selected
//!   18. portfolio_after_robustness
//!   19. passed_cpcv
//!   20. export_ready

use serde::{Deserialize, Serialize};

/// Compact research diagnostics for a window explicitly USED FOR SELECTION.
/// This is not an untouched final test or a portfolio/deployment authority.
/// The existing result carries the receipt once; each trial links to the full
/// genome in `DiscoveryResult::candidates` by both index and exact identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCalibrationCohort {
    pub scope: crate::data_selection::CanonicalSearchArtifactScopeRefV1,
    pub search_config_hash: String,
    pub trials: Vec<SelectionCalibrationTrial>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCalibrationTrial {
    pub candidate_archive_index: usize,
    pub strategy_identity: crate::validation::ValidationStrategyIdentityV2,
    /// Null means invalid/non-finite metrics, with the reason retained below.
    /// Ordinary losing but finite replays keep their complete measured summary.
    pub summary: Option<crate::validation::ForwardTestSummary>,
    /// Passed the existing numerical sizing gate, not final validation.
    pub profitable_for_selection: bool,
    pub rejection_reason: Option<String>,
}

/// Actual candidate coverage, separate from planned GA evaluation slots and
/// from portfolio capacity. Absent on legacy profiles rather than invented.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryCandidateCensus {
    pub ga_returned_candidates: usize,
    pub validation_candidate_limit: usize,
    pub validation_candidates_admitted: usize,
    pub validation_candidates_capped: usize,
    pub quality_evaluated: usize,
    pub walkforward_tested: usize,
    pub walkforward_passed: usize,
    pub walkforward_failed: usize,
    /// Admitted candidates without a completed non-empty WF result. Includes
    /// earlier quality/window exclusions, but NOT the separately counted cap.
    pub walkforward_not_tested: usize,
    pub correlation_tested: usize,
    pub rejected_by_correlation: usize,
    pub portfolio_capacity_not_selected: usize,
    /// Actual removals after portfolio selection, not WF/correlation failures.
    /// Absent before that transition and in older profiles means unobserved.
    /// A skipped screen or the retained-all policy removes nothing; this is not
    /// a claim that the retained genes passed the robustness tests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robustness_removed: Option<usize>,
    pub portfolio_selected: usize,
}

impl DiscoveryCandidateCensus {
    pub fn counters(&self) -> Vec<(&'static str, usize)> {
        let mut counters = vec![
            ("ga_returned_candidates", self.ga_returned_candidates),
            (
                "validation_candidate_limit",
                self.validation_candidate_limit,
            ),
            (
                "validation_candidates_admitted",
                self.validation_candidates_admitted,
            ),
            (
                "validation_candidates_capped",
                self.validation_candidates_capped,
            ),
            ("quality_evaluated", self.quality_evaluated),
            ("walkforward_tested", self.walkforward_tested),
            ("walkforward_passed", self.walkforward_passed),
            ("walkforward_failed", self.walkforward_failed),
            ("walkforward_not_tested", self.walkforward_not_tested),
            ("correlation_tested", self.correlation_tested),
            ("rejected_by_correlation", self.rejected_by_correlation),
            (
                "portfolio_capacity_not_selected",
                self.portfolio_capacity_not_selected,
            ),
            ("portfolio_selected", self.portfolio_selected),
        ];
        if let Some(removed) = self.robustness_removed {
            counters.push(("robustness_removed", removed));
        }
        counters
    }
}

/// How many named reject reasons one stage may keep.
///
/// The widest stage today is `passed_quality` at fourteen (six gate-level
/// reasons plus the eight base-quality criteria). 32 leaves room for the next
/// split without a silent drop, and a stage that ever reaches it is a stage
/// whose reasons need grouping, not truncating.
pub const MAX_REJECT_REASONS_PER_STAGE: usize = 32;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunnelStage {
    pub name: String,
    pub count_in: usize,
    pub count_out: usize,
    pub rejected: usize,
    /// Top reject reasons with their counts. Empty when the stage
    /// either lets everything through or doesn't track reasons.
    pub top_reasons: Vec<(String, usize)>,
    /// Wall time from the previous stage to this one.
    ///
    /// The funnel already named every stage; it never said how long any took,
    /// so finding where a run spends its time meant reading log timestamps by
    /// hand and guessing at the silent stretches. One such stretch was 25.9 s
    /// of a 136.5 s run with nothing logged inside it.
    ///
    /// `default` so profiles written before this existed still deserialize.
    #[serde(default)]
    pub elapsed_ms: u64,
    /// Reject reasons DISCARDED by the `MAX_REJECT_REASONS_PER_STAGE` cap.
    ///
    /// Raising the cap from 10 to 32 fixed the incidence, not the class:
    /// truncation still drops the SMALLEST counts, which are exactly the rare
    /// causes a funnel exists to surface. A drop on a decision path that nobody
    /// counts is a silent drop, so it is counted here.
    #[serde(default)]
    pub reasons_truncated: usize,
}

impl FunnelStage {
    pub fn new(name: &'static str) -> Self {
        Self {
            name: name.to_string(),
            count_in: 0,
            count_out: 0,
            rejected: 0,
            top_reasons: Vec::new(),
            elapsed_ms: 0,
            reasons_truncated: 0,
        }
    }

    pub fn passthrough(name: &'static str, count: usize) -> Self {
        Self {
            name: name.to_string(),
            count_in: count,
            count_out: count,
            rejected: 0,
            top_reasons: Vec::new(),
            elapsed_ms: 0,
            reasons_truncated: 0,
        }
    }

    pub fn record(&mut self, count_in: usize, count_out: usize) {
        self.count_in = count_in;
        self.count_out = count_out;
        self.rejected = count_in.saturating_sub(count_out);
    }
}

/// Persistent funnel profile written next to the portfolio JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunnelProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_census: Option<DiscoveryCandidateCensus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_calibration_cohort: Option<SelectionCalibrationCohort>,
    pub symbol: String,
    pub timeframe: String,
    pub started_at: String,
    pub finished_at: String,
    /// 2026-05-26 operator directive (dual-mode product): which mode this run
    /// used. Either "PropFirm" (steady ~4%/month, respects FTMO DD caps) or
    /// "Risky" (aggressive €100→€100k, Kelly-aligned ~30% per trade). Empty
    /// when constructed via `FunnelProfile::new` (caller must set this).
    #[serde(default)]
    pub mode: String,
    /// Each canonical pipeline stage. Stages
    /// the run didn't reach get count_in/out=0.
    pub stages: Vec<FunnelStage>,
    /// Bottleneck = the stage with the highest `rejected` count.
    pub bottleneck_stage: String,
    pub bottleneck_rejected: usize,
    /// Final outcome state per P10.
    pub outcome: String,
    /// In-memory authority for the arithmetic engines that completed this exact
    /// discovery run. It is attached once from the closed canonical-scope-bound
    /// receipt and is never reconstructed from funnel JSON or process state.
    #[serde(skip)]
    population_execution_run_receipt_v2:
        Option<crate::population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2>,
    /// Exact live execution policy resolved from the immutable search
    /// authority. This is an in-memory carrier only: the dedicated
    /// `live_portfolio` artifact persists and validates it explicitly.
    #[serde(skip)]
    live_trading_policy_v1: Option<crate::live_portfolio::LiveTradingPolicyV1>,
    /// When the previous stage finished, so each stage can report its own cost.
    /// Not persisted — it only has meaning during the run that set it.
    #[serde(skip)]
    last_mark: Option<std::time::Instant>,
}

impl FunnelProfile {
    pub fn new(symbol: impl Into<String>, timeframe: impl Into<String>) -> Self {
        Self {
            candidate_census: None,
            selection_calibration_cohort: None,
            symbol: symbol.into(),
            timeframe: timeframe.into(),
            started_at: now_iso8601(),
            finished_at: String::new(),
            mode: String::new(),
            stages: canonical_empty_stages(),
            bottleneck_stage: String::new(),
            bottleneck_rejected: 0,
            outcome: "pending".to_string(),
            population_execution_run_receipt_v2: None,
            live_trading_policy_v1: None,
            last_mark: Some(std::time::Instant::now()),
        }
    }

    pub(crate) fn attach_live_trading_policy_v1(
        &mut self,
        policy: crate::live_portfolio::LiveTradingPolicyV1,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.live_trading_policy_v1.is_none(),
            "live trading policy v1 is already attached"
        );
        policy.validate()?;
        self.live_trading_policy_v1 = Some(policy);
        Ok(())
    }

    pub(crate) const fn live_trading_policy_v1(
        &self,
    ) -> Option<&crate::live_portfolio::LiveTradingPolicyV1> {
        self.live_trading_policy_v1.as_ref()
    }

    pub(crate) fn attach_population_execution_run_receipt_v2(
        &mut self,
        receipt: crate::population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2,
    ) -> Result<(), &'static str> {
        if self.population_execution_run_receipt_v2.is_some() {
            return Err("population execution run receipt v2 is already attached");
        }
        self.population_execution_run_receipt_v2 = Some(receipt);
        Ok(())
    }

    pub(crate) const fn population_execution_run_receipt_v2(
        &self,
    ) -> Option<&crate::population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2>
    {
        self.population_execution_run_receipt_v2.as_ref()
    }

    /// 2026-05-26 operator directive (dual-mode product): record which trading
    /// mode produced this funnel. The save path is the same JSON file, but
    /// downstream diagnostics (UI panel, post-mortem scripts) need to know
    /// whether the rejection pattern was generated under PropFirm or Risky
    /// thresholds before drawing conclusions.
    pub fn set_mode(&mut self, mode: impl Into<String>) {
        self.mode = mode.into();
    }

    pub fn record_stage(&mut self, name: &str, count_in: usize, count_out: usize) {
        let elapsed = self
            .last_mark
            .replace(std::time::Instant::now())
            .map(|mark| mark.elapsed().as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or_default();
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == name) {
            s.record(count_in, count_out);
            s.elapsed_ms = elapsed;
        }
    }

    /// Add a reject-reason bucket to a stage's top-reasons list.
    ///
    /// The cap was 10 until 2026-08-09. The quality screen now records six
    /// gate-level reasons PLUS the eight named base-quality criteria — fourteen
    /// buckets — so a 10-entry cap would have SILENTLY DISCARDED four of them,
    /// and the ones discarded would be the smallest counts, i.e. exactly the
    /// rare causes a funnel exists to surface. A truncation that drops evidence
    /// without saying so is the same defect the per-criterion split was landed
    /// to fix, so the cap is raised to a number no stage approaches rather than
    /// left to bite the next stage that grows.
    pub fn add_reject_reason(&mut self, stage_name: &str, reason: impl Into<String>, count: usize) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == stage_name) {
            s.top_reasons.push((reason.into(), count));
            // Descending by count, ties broken by name so the list is stable.
            s.top_reasons
                .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let before = s.top_reasons.len();
            s.top_reasons.truncate(MAX_REJECT_REASONS_PER_STAGE);
            s.reasons_truncated += before - s.top_reasons.len();
        }
    }

    pub fn finalize(&mut self, outcome: &str) {
        self.finished_at = now_iso8601();
        self.outcome = outcome.to_string();
        // Where the run actually went. Counts alone say which stage rejects the
        // most, never which one costs the most — and those are rarely the same
        // stage.
        let mut by_cost: Vec<&FunnelStage> =
            self.stages.iter().filter(|s| s.elapsed_ms > 0).collect();
        by_cost.sort_by(|a, b| b.elapsed_ms.cmp(&a.elapsed_ms));
        let total_ms: u64 = by_cost.iter().map(|s| s.elapsed_ms).sum();
        for stage in by_cost.iter().take(6) {
            tracing::info!(
                target: "neoethos_search::funnel",
                stage = %stage.name,
                elapsed_ms = stage.elapsed_ms,
                share_pct = format!(
                    "{:.1}",
                    stage.elapsed_ms as f64 * 100.0 / total_ms.max(1) as f64
                ),
                count_in = stage.count_in,
                count_out = stage.count_out,
                "stage cost — the six most expensive, wall time since the previous stage"
            );
        }
        // Recompute bottleneck.
        if let Some(b) = self.stages.iter().max_by_key(|s| s.rejected) {
            self.bottleneck_stage = b.name.clone();
            self.bottleneck_rejected = b.rejected;
        }
    }

    pub fn save_next_to(&self, portfolio_json_path: &std::path::Path) -> std::io::Result<()> {
        let funnel_path = funnel_path_for(portfolio_json_path);
        if let Some(dir) = funnel_path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&funnel_path, text)
    }
}

/// `EURUSD_M30.json` → `EURUSD_M30_funnel.json` next to it.
pub fn funnel_path_for(portfolio_json_path: &std::path::Path) -> std::path::PathBuf {
    let stem = portfolio_json_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let parent = portfolio_json_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    parent.join(format!("{}_funnel.json", stem))
}

fn canonical_empty_stages() -> Vec<FunnelStage> {
    vec![
        FunnelStage::new("data_loaded"),
        FunnelStage::new("rows_after_trimming"),
        FunnelStage::new("features_built"),
        FunnelStage::new("features_after_prefilter"),
        FunnelStage::new("stage1_candidates_generated"),
        FunnelStage::new("profitable_archive_size"),
        FunnelStage::new("validation_candidates_admitted"),
        FunnelStage::new("passed_base_filter"),
        FunnelStage::new("nonzero_signals"),
        FunnelStage::new("passed_min_trades"),
        FunnelStage::new("full_is_evaluated"),
        FunnelStage::new("passed_quality"),
        FunnelStage::new("passed_prop_firm_window"),
        FunnelStage::new("passed_walkforward"),
        FunnelStage::new("passed_selection_calibration"),
        FunnelStage::new("passed_correlation"),
        FunnelStage::new("portfolio_selected"),
        FunnelStage::new("portfolio_after_robustness"),
        FunnelStage::new("passed_cpcv"),
        FunnelStage::new("export_ready"),
    ]
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn legacy_profile_has_unknown_candidate_coverage_and_new_census_roundtrips() {
        let mut profile = FunnelProfile::new("EURUSD", "M5");
        let legacy = serde_json::to_string(&profile).unwrap();
        assert!(!legacy.contains("candidate_census"));
        assert!(
            serde_json::from_str::<FunnelProfile>(&legacy)
                .unwrap()
                .candidate_census
                .is_none()
        );
        let census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 200,
            walkforward_tested: 20,
            walkforward_passed: 5,
            walkforward_failed: 15,
            walkforward_not_tested: 180,
            robustness_removed: Some(1),
            ..Default::default()
        };
        profile.candidate_census = Some(census.clone());
        let restored: FunnelProfile =
            serde_json::from_str(&serde_json::to_string(&profile).unwrap()).unwrap();
        assert_eq!(restored.candidate_census, Some(census));
        let names = profile
            .stages
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>();
        assert!(
            names.iter().position(|n| *n == "passed_walkforward")
                < names.iter().position(|n| *n == "passed_correlation")
        );
        assert!(
            names.iter().position(|n| *n == "portfolio_selected")
                < names
                    .iter()
                    .position(|n| *n == "portfolio_after_robustness")
        );
        assert!(
            names
                .iter()
                .position(|n| *n == "portfolio_after_robustness")
                < names.iter().position(|n| *n == "passed_cpcv")
        );
    }

    #[test]
    fn original_thirteen_counter_census_still_deserializes() {
        let original = r#"{
            "ga_returned_candidates":10,
            "validation_candidate_limit":8,
            "validation_candidates_admitted":8,
            "validation_candidates_capped":2,
            "quality_evaluated":6,
            "walkforward_tested":5,
            "walkforward_passed":4,
            "walkforward_failed":1,
            "walkforward_not_tested":3,
            "correlation_tested":4,
            "rejected_by_correlation":1,
            "portfolio_capacity_not_selected":0,
            "portfolio_selected":3
        }"#;
        let census: DiscoveryCandidateCensus = serde_json::from_str(original).unwrap();
        assert_eq!(census.robustness_removed, None);
        assert_eq!(census.quality_evaluated, 6);
        assert_eq!(census.portfolio_selected, 3);
        assert_eq!(census.counters().len(), 13);
        assert!(
            !serde_json::to_string(&census)
                .unwrap()
                .contains("robustness_removed")
        );
        let observed = DiscoveryCandidateCensus {
            robustness_removed: Some(0),
            ..census
        };
        assert!(observed.counters().contains(&("robustness_removed", 0)));
        let restored: DiscoveryCandidateCensus =
            serde_json::from_str(&serde_json::to_string(&observed).unwrap()).unwrap();
        assert_eq!(restored.robustness_removed, Some(0));
    }

    #[test]
    fn funnel_path_appends_suffix() {
        let p = PathBuf::from("/tmp/EURUSD_M30.json");
        assert_eq!(
            funnel_path_for(&p),
            PathBuf::from("/tmp/EURUSD_M30_funnel.json")
        );
    }

    #[test]
    fn a_stage_reports_what_it_cost_not_only_what_it_rejected() {
        let mut profile = FunnelProfile::new("EURUSD", "H1");
        let first = profile
            .stages
            .first()
            .map(|s| s.name.clone())
            .expect("canonical stages exist");
        std::thread::sleep(std::time::Duration::from_millis(12));
        profile.record_stage(&first, 100, 40);
        let stage = profile
            .stages
            .iter()
            .find(|s| s.name == first)
            .expect("the stage was recorded");
        assert!(
            stage.elapsed_ms >= 10,
            "a stage that took 12 ms reported {} ms",
            stage.elapsed_ms
        );
        assert_eq!(stage.rejected, 60, "counts still work");
    }

    #[test]
    fn a_profile_written_before_stages_were_timed_still_loads() {
        // The field is additive; an older artifact must not fail to parse.
        let older = r#"{"name":"stage1_candidates_generated","count_in":9,
            "count_out":4,"rejected":5,"top_reasons":[]}"#;
        let stage: FunnelStage = serde_json::from_str(older).expect("older profiles still load");
        assert_eq!(stage.elapsed_ms, 0);
        assert_eq!(stage.count_out, 4);
    }

    #[test]
    fn record_stage_updates_counts_and_rejected() {
        let mut f = FunnelProfile::new("EURJPY", "D1");
        f.record_stage("passed_base_filter", 100, 25);
        let s = f
            .stages
            .iter()
            .find(|s| s.name == "passed_base_filter")
            .unwrap();
        assert_eq!(s.count_in, 100);
        assert_eq!(s.count_out, 25);
        assert_eq!(s.rejected, 75);
    }

    #[test]
    fn finalize_picks_bottleneck() {
        let mut f = FunnelProfile::new("EURJPY", "D1");
        f.record_stage("passed_base_filter", 100, 25); // 75 rejected
        f.record_stage("nonzero_signals", 25, 10); // 15 rejected
        f.record_stage("passed_quality", 10, 10);
        f.finalize("no_candidates");
        assert_eq!(f.bottleneck_stage, "passed_base_filter");
        assert_eq!(f.bottleneck_rejected, 75);
        assert_eq!(f.outcome, "no_candidates");
    }

    #[test]
    fn add_reject_reason_keeps_the_cap_descending() {
        // The cap was 10 when this test was written. It is now
        // MAX_REJECT_REASONS_PER_STAGE = 32, because the quality screen alone
        // records fourteen reasons and truncation drops the SMALLEST counts —
        // i.e. exactly the rare causes a funnel exists to surface. The test now
        // asserts against the constant rather than a literal, so raising the cap
        // again cannot leave a stale number behind.
        let mut f = FunnelProfile::new("EURJPY", "D1");
        let offered = MAX_REJECT_REASONS_PER_STAGE + 10;
        for i in 0..offered {
            f.add_reject_reason("passed_base_filter", format!("reason_{}", i), i);
        }
        let s = f
            .stages
            .iter()
            .find(|s| s.name == "passed_base_filter")
            .unwrap();
        assert_eq!(s.top_reasons.len(), MAX_REJECT_REASONS_PER_STAGE);
        // First entry = highest count, last = lowest kept.
        assert!(s.top_reasons[0].1 >= s.top_reasons[MAX_REJECT_REASONS_PER_STAGE - 1].1);
    }
}
