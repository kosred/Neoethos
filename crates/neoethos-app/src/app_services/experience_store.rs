//! Live trade EXPERIENCE store — the raw material of live learning.
//!
//! Operator finding (2026-07-02): the online-learning experts
//! (OnlinePassiveAggressive / OnlineHoeffding / AdaptiveGradientBooster, the
//! exit agent, SAC) exist in neoethos-models but NOTHING in the live path fed
//! them — every live trade's experience evaporated.
//!
//! This store fixes the FOUNDATION first, with zero parity risk: the live
//! engine records, for every position it opens, the exact FEATURE ROW it acted
//! on plus the trade's context, and pairs it with the realized outcome when
//! the position closes. Pure data collection — live behavior is unchanged.
//!
//! Consumers (follow-ups, explicitly validated before they may influence
//! live): offline fine-tuning of the exit agent / meta-label models on REAL
//! live outcomes, drift monitoring (live feature distributions vs discovery),
//! and an online-expert layer once it can pass the same OOS discipline as
//! everything else.
//!
//! Format: append-only JSONL at `<data_dir>/experience/live_experience.jsonl`.
//! Best-effort by contract — a store hiccup never touches trading.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveExperience {
    pub schema_version: u32,
    /// Broker position id — join key to the journal.
    pub position_id: i64,
    pub symbol: String,
    pub base_tf: String,
    /// Which portfolio produced the signal (its artifact carries the feature
    /// names that give `features` meaning).
    pub portfolio_path: String,
    /// +1 long / -1 short.
    pub direction: i8,
    /// The gene's brackets at entry (pips; 0 = none).
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub lots: f64,
    pub entry_ts_ms: i64,
    pub entry_price: Option<f64>,
    /// The EXACT projected feature row the signal was computed from.
    pub features: Vec<f32>,
    // ── Filled at close ────────────────────────────────────────────────────
    pub close_ts_ms: Option<i64>,
    pub net_profit: Option<f64>,
}

fn store_path() -> Option<PathBuf> {
    neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
        .ok()
        .map(|s| s.system.data_dir.join("experience").join("live_experience.jsonl"))
}

/// Append one COMPLETED experience (entry snapshot + realized outcome).
/// Best-effort: logs and swallows failures.
pub fn record(exp: &LiveExperience) {
    let Some(path) = store_path() else { return };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    match serde_json::to_string(exp) {
        Ok(line) => {
            use std::io::Write;
            match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                Ok(mut f) => {
                    if let Err(e) = writeln!(f, "{line}") {
                        tracing::warn!(
                            target: "neoethos_app::experience",
                            error = %e, "failed to append live experience"
                        );
                    }
                }
                Err(e) => tracing::warn!(
                    target: "neoethos_app::experience",
                    error = %e, "failed to open live experience store"
                ),
            }
        }
        Err(e) => tracing::warn!(
            target: "neoethos_app::experience",
            error = %e, "failed to serialize live experience"
        ),
    }
}

/// Count stored experiences (for the UI/status surfaces).
pub fn count() -> usize {
    store_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|raw| raw.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}
