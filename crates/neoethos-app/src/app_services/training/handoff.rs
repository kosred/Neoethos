//! Durable Discovery → Training selection. No ambient-symbol dataset lookup.

use anyhow::{Context, Result};
use neoethos_core::Settings;
use neoethos_data::CanonicalDatasetSeriesReceiptV1;
use neoethos_models::{
    MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1, PromotionCandidateTrainingHandoffV1,
};
use std::path::{Path, PathBuf};

pub fn handoff_path(data_root: &Path, identity: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        identity.len() == 64
            && identity
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "training_handoff must be a canonical lowercase SHA-256 identity, not a path"
    );
    Ok(data_root
        .join("discovery_targets")
        .join(format!("{identity}.training-handoff.json")))
}

/// Copy only the exact selected input ladder into the training plan. Unselected
/// lower/higher timeframes in ambient Settings cannot enter this training run.
pub fn settings_for_series(
    settings: &Settings,
    series: &CanonicalDatasetSeriesReceiptV1,
) -> Settings {
    let mut selected = settings.clone();
    let anchor = series.anchor().identity();
    selected.system.symbol = anchor.symbol_name().to_owned();
    selected.system.base_timeframe = anchor.timeframe().as_str().to_owned();
    selected.system.higher_timeframes = series
        .direct_timeframes()
        .iter()
        .filter(|source| source.identity().timeframe() != anchor.timeframe())
        .map(|source| source.identity().timeframe().as_str().to_owned())
        .collect();
    selected.system.multi_resolution_timeframes = selected.system.higher_timeframes.clone();
    selected
}

pub fn publish(data_root: &Path, handoff: &PromotionCandidateTrainingHandoffV1) -> Result<String> {
    let (bytes, identity) = handoff.canonical_bytes_and_identity_sha256()?;
    let path = handoff_path(data_root, &identity)?;
    neoethos_core::storage::json::write_bytes_atomic(&path, &bytes)?;
    // Reopen the bounded regular file and compare to the canonical bytes we
    // just validated. Exact equality proves the disk write without decoding
    // and validating the same immutable content again. Later loads revalidate.
    anyhow::ensure!(
        read_selected_bytes(data_root, &identity)? == bytes,
        "published training handoff bytes changed during atomic publication"
    );
    Ok(identity)
}

pub fn load(data_root: &Path, identity: &str) -> Result<PromotionCandidateTrainingHandoffV1> {
    let handoff = read_selected_unvalidated(data_root, identity)?;
    anyhow::ensure!(
        handoff.identity_sha256()? == identity,
        "selected training handoff identity changed"
    );
    Ok(handoff)
}

/// One strict selected read for callers that also need the live portfolio.
/// Return the value already decoded by validation; never cache it across reads.
pub(crate) fn load_with_live_portfolio(
    data_root: &Path,
    identity: &str,
) -> Result<(
    PromotionCandidateTrainingHandoffV1,
    neoethos_search::live_portfolio::LivePortfolioArtifact,
)> {
    let handoff = read_selected_unvalidated(data_root, identity)?;
    let (portfolio, actual_identity) = handoff.validated_live_portfolio_and_identity_sha256()?;
    anyhow::ensure!(
        actual_identity == identity,
        "selected training handoff identity changed"
    );
    Ok((handoff, portfolio))
}

/// Decode only. Each caller must validate semantics and match the selected
/// identity before exposing anything from this untrusted persisted value.
fn read_selected_unvalidated(
    data_root: &Path,
    identity: &str,
) -> Result<PromotionCandidateTrainingHandoffV1> {
    Ok(serde_json::from_slice(&read_selected_bytes(
        data_root, identity,
    )?)?)
}

fn read_selected_bytes(data_root: &Path, identity: &str) -> Result<Vec<u8>> {
    let path = handoff_path(data_root, identity)?;
    neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
        &path,
        MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64,
    )
    .with_context(|| {
        format!(
            "read selected Discovery training handoff {}",
            path.display()
        )
    })
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingHandoffSummary {
    pub identity: String,
    pub symbol: String,
    pub base_tf: String,
    pub dataset_identity: String,
    pub generation: String,
    pub strategy_count: usize,
    pub planned_models: Vec<String>,
    pub oos_cutoff_ms: i64,
    pub purge_bars: usize,
    // Inventory-only calibration diagnostics from this same validated read.
    // Not serialized as additional handoff fields or independent final evidence.
    #[serde(skip)]
    pub(crate) strategies: Vec<TrainingHandoffStrategySummary>,
}

#[derive(Debug)]
pub(crate) struct TrainingHandoffStrategySummary {
    pub strategy_id: String,
    pub sharpe: f64,
    pub win_rate: f64,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingHandoffUnavailable {
    /// Filename selector for diagnostics; it may itself be malformed.
    pub identity: String,
    pub reason: String,
}

#[derive(Debug, Default)]
pub struct TrainingHandoffInventory {
    pub available: Vec<TrainingHandoffSummary>,
    pub unavailable: Vec<TrainingHandoffUnavailable>,
}

fn summarize(data_root: &Path, identity: &str) -> Result<TrainingHandoffSummary> {
    let (handoff, portfolio) = load_with_live_portfolio(data_root, identity)?;
    let strategies = portfolio
        .genes
        .iter()
        .zip(&portfolio.sizing_evidence)
        .map(|(gene, evidence)| {
            let metrics = evidence.oos_metrics();
            TrainingHandoffStrategySummary {
                strategy_id: gene.strategy_id.clone(),
                sharpe: metrics.sharpe,
                win_rate: metrics.win_rate,
            }
        })
        .collect();
    let anchor = handoff.canonical_series().anchor();
    Ok(TrainingHandoffSummary {
        identity: identity.to_owned(),
        symbol: portfolio.symbol,
        base_tf: portfolio.base_tf,
        dataset_identity: anchor.identity().to_path_component(),
        generation: anchor.generation_id().to_owned(),
        strategy_count: portfolio.genes.len(),
        planned_models: handoff.training_config().planned_models().to_vec(),
        oos_cutoff_ms: handoff.oos_cutoff_ms(),
        purge_bars: handoff.purge_bars(),
        strategies,
    })
}

/// Per-file corruption remains visible without hiding other valid selections.
/// Directory/enumeration errors still fail the inventory; selected `load` stays strict.
pub fn list(data_root: &Path) -> Result<TrainingHandoffInventory> {
    let root = data_root.join("discovery_targets");
    if !root.try_exists()? {
        return Ok(TrainingHandoffInventory::default());
    }
    let mut inventory = TrainingHandoffInventory::default();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(identity) = name.strip_suffix(".training-handoff.json") else {
            continue;
        };
        match summarize(data_root, identity) {
            Ok(summary) => inventory.available.push(summary),
            Err(error) => inventory.unavailable.push(TrainingHandoffUnavailable {
                identity: identity.to_owned(),
                reason: format!("{error:#}"),
            }),
        }
    }
    inventory.available.sort_by(|left, right| {
        right
            .oos_cutoff_ms
            .cmp(&left.oos_cutoff_ms)
            .then(left.identity.cmp(&right.identity))
    });
    inventory
        .unavailable
        .sort_by(|left, right| left.identity.cmp(&right.identity));
    Ok(inventory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_cannot_escape_the_owned_handoff_directory() {
        let root = Path::new("data");
        for invalid in [
            "",
            "../config.yaml",
            "C:\\secrets",
            &"A".repeat(64),
            &"a".repeat(63),
        ] {
            assert!(handoff_path(root, invalid).is_err());
        }
        assert_eq!(
            handoff_path(root, &"a".repeat(64))
                .unwrap()
                .parent()
                .unwrap(),
            root.join("discovery_targets")
        );
    }
}
