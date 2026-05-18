//! Per-work-unit funnel profile — the JSON dump P4 calls for.
//!
//! Whenever a discovery work-unit (one symbol × one timeframe) finishes
//! we save a `<symbol>_<tf>_funnel.json` next to the portfolio output.
//! It captures the full rejection funnel so "no strategies" is debuggable
//! without re-running.
//!
//! Stages (in spec order):
//!   1.  data_loaded
//!   2.  rows_after_trimming
//!   3.  features_built
//!   4.  features_after_prefilter
//!   5.  stage1_candidates_generated
//!   6.  profitable_archive_size
//!   7.  full_is_evaluated
//!   8.  passed_base_filter
//!   9.  nonzero_signals
//!   10. passed_min_trades
//!   11. passed_quality
//!   12. passed_prop_firm_window
//!   13. passed_correlation
//!   14. passed_walkforward
//!   15. passed_cpcv
//!   16. export_ready

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunnelStage {
    pub name: String,
    pub count_in: usize,
    pub count_out: usize,
    pub rejected: usize,
    /// Top reject reasons with their counts. Empty when the stage
    /// either lets everything through or doesn't track reasons.
    pub top_reasons: Vec<(String, usize)>,
}

impl FunnelStage {
    pub fn new(name: &'static str) -> Self {
        Self {
            name: name.to_string(),
            count_in: 0,
            count_out: 0,
            rejected: 0,
            top_reasons: Vec::new(),
        }
    }

    pub fn passthrough(name: &'static str, count: usize) -> Self {
        Self {
            name: name.to_string(),
            count_in: count,
            count_out: count,
            rejected: 0,
            top_reasons: Vec::new(),
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
    pub symbol: String,
    pub timeframe: String,
    pub started_at: String,
    pub finished_at: String,
    /// Each canonical pipeline stage (16 total per spec). Stages
    /// the run didn't reach get count_in/out=0.
    pub stages: Vec<FunnelStage>,
    /// Bottleneck = the stage with the highest `rejected` count.
    pub bottleneck_stage: String,
    pub bottleneck_rejected: usize,
    /// Final outcome state per P10.
    pub outcome: String,
}

impl FunnelProfile {
    pub fn new(symbol: impl Into<String>, timeframe: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            timeframe: timeframe.into(),
            started_at: now_iso8601(),
            finished_at: String::new(),
            stages: canonical_empty_stages(),
            bottleneck_stage: String::new(),
            bottleneck_rejected: 0,
            outcome: "pending".to_string(),
        }
    }

    pub fn record_stage(&mut self, name: &str, count_in: usize, count_out: usize) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == name) {
            s.record(count_in, count_out);
        }
    }

    /// Add a reject-reason bucket to a stage's top-reasons list. The
    /// caller is responsible for keeping the list bounded.
    pub fn add_reject_reason(&mut self, stage_name: &str, reason: impl Into<String>, count: usize) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == stage_name) {
            s.top_reasons.push((reason.into(), count));
            // Keep only the top 10, descending by count.
            s.top_reasons
                .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            s.top_reasons.truncate(10);
        }
    }

    pub fn finalize(&mut self, outcome: &str) {
        self.finished_at = now_iso8601();
        self.outcome = outcome.to_string();
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
        FunnelStage::new("full_is_evaluated"),
        FunnelStage::new("passed_base_filter"),
        FunnelStage::new("nonzero_signals"),
        FunnelStage::new("passed_min_trades"),
        FunnelStage::new("passed_quality"),
        FunnelStage::new("passed_prop_firm_window"),
        FunnelStage::new("passed_correlation"),
        FunnelStage::new("passed_walkforward"),
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
    fn funnel_path_appends_suffix() {
        let p = PathBuf::from("/tmp/EURUSD_M30.json");
        assert_eq!(
            funnel_path_for(&p),
            PathBuf::from("/tmp/EURUSD_M30_funnel.json")
        );
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
    fn add_reject_reason_keeps_top_10_descending() {
        let mut f = FunnelProfile::new("EURJPY", "D1");
        for i in 0..20 {
            f.add_reject_reason("passed_base_filter", format!("reason_{}", i), i);
        }
        let s = f
            .stages
            .iter()
            .find(|s| s.name == "passed_base_filter")
            .unwrap();
        assert_eq!(s.top_reasons.len(), 10);
        // First entry = highest count.
        assert!(s.top_reasons[0].1 >= s.top_reasons[9].1);
    }
}
