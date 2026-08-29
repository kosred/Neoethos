//! Direct-timeframe truth boundary.
//!
//! Every requested timeframe must be backed by its own immutable canonical
//! generation. NeoEthos never manufactures a higher timeframe from M1 (or
//! from any other timeframe): missing broker data is a download/import error,
//! not a resampling opportunity.

use anyhow::{Context, Result, bail, ensure};
use neoethos_dataset_contracts::{CanonicalDatasetIdentity, CanonicalTimeframe};

use crate::SymbolDataset;

pub use neoethos_dataset_contracts::CANONICAL_TIMEFRAMES;

/// Prove that every requested timeframe is a direct generation in the exact
/// selected source/account series.
pub fn require_direct_timeframes(
    dataset: &SymbolDataset,
    selected: &CanonicalDatasetIdentity,
    required: &[CanonicalTimeframe],
) -> Result<()> {
    ensure!(
        dataset.symbol == selected.symbol_name(),
        "dataset symbol {} disagrees with selected identity symbol {}",
        dataset.symbol,
        selected.symbol_name()
    );

    let selected_artifact = dataset
        .source_artifacts
        .get(selected.timeframe().as_str())
        .context("selected base timeframe has no direct canonical artifact")?;
    ensure!(
        selected_artifact.identity() == selected,
        "selected base timeframe artifact does not match the exact requested identity"
    );

    for timeframe in required {
        let key = timeframe.as_str();
        let frame = dataset
            .frames
            .get(key)
            .with_context(|| format!("missing direct canonical timeframe {timeframe}"))?;
        if frame.is_empty() {
            bail!("direct canonical timeframe {timeframe} is empty");
        }
        let artifact = dataset.source_artifacts.get(key).with_context(|| {
            format!("timeframe {timeframe} has no direct canonical source artifact")
        })?;
        let identity = artifact.identity();
        ensure!(
            identity.timeframe() == *timeframe,
            "timeframe {timeframe} is backed by a {} artifact instead of a direct generation",
            identity.timeframe()
        );
        ensure!(
            identity.scope() == selected.scope()
                && identity.symbol_name() == selected.symbol_name()
                && identity.bar_timestamp_convention() == selected.bar_timestamp_convention(),
            "timeframe {timeframe} belongs to a different source/account series"
        );
    }
    Ok(())
}
