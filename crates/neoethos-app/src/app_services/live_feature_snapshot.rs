//! Canonical bridge from bounded live cTrader responses to the existing
//! multi-timeframe feature pipeline.
//!
//! Live broker rows are authoritative current data, but a bare in-memory
//! `SymbolDataset` cannot prove which environment/account/symbol/timeframe
//! produced them. This module publishes each direct broker response into one
//! isolated, short-lived canonical Vortex root, reopens the exact artifacts,
//! and keeps their leases alive for feature computation. The root is deleted
//! only after every loaded artifact has been dropped.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalOhlcvChunk, CanonicalTimeframe, CanonicalVolumeChunk,
    SymbolDataset,
};

use crate::app_services::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use crate::app_services::broker_api::RecentBrokerTrendbarSnapshot;

static LIVE_SNAPSHOT_NONCE: AtomicU64 = AtomicU64::new(1);
const LIVE_SNAPSHOT_PREFIX: &str = "neoethos-live-canonical-v1-";

/// Owns the reopened canonical dataset and its isolated temporary root. The
/// dataset is an `Option` solely so `Drop` can release all generation leases
/// before recursively removing that exact, newly-created root on Windows.
pub(crate) struct LiveCanonicalFeatureSnapshot {
    dataset: Option<SymbolDataset>,
    root: PathBuf,
}

impl LiveCanonicalFeatureSnapshot {
    pub(crate) fn publish_for_artifact(
        artifact: &neoethos_search::LivePortfolioArtifact,
        snapshots: Vec<RecentBrokerTrendbarSnapshot>,
    ) -> Result<Self> {
        artifact.validate()?;
        let anchor = artifact
            .search_scope
            .receipt()
            .validate()
            .map_err(anyhow::Error::new)?;
        let required = std::iter::once(artifact.base_tf.as_str())
            .chain(artifact.higher_tfs.iter().map(String::as_str))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        Self::publish_for_series(&anchor, &required, snapshots)
    }

    fn publish_for_series(
        anchor: &CanonicalDatasetIdentity,
        required_timeframes: &[String],
        snapshots: Vec<RecentBrokerTrendbarSnapshot>,
    ) -> Result<Self> {
        anyhow::ensure!(
            anchor.is_broker_real(),
            "live feature snapshot requires a cTrader receipt anchor"
        );
        let required = required_timeframes
            .iter()
            .map(|timeframe| {
                timeframe
                    .parse::<CanonicalTimeframe>()
                    .with_context(|| format!("unsupported live direct timeframe {timeframe}"))
            })
            .collect::<Result<BTreeSet<_>>>()?;
        anyhow::ensure!(
            required.len() == required_timeframes.len(),
            "live feature snapshot contains duplicate required timeframes"
        );
        anyhow::ensure!(
            required.contains(&anchor.timeframe()),
            "live feature snapshot requirements omit receipt anchor timeframe {}",
            anchor.timeframe()
        );

        let mut by_timeframe = BTreeMap::new();
        for snapshot in snapshots {
            snapshot.validate()?;
            let timeframe = snapshot.identity().timeframe();
            {
                let identity = snapshot.identity();
                anyhow::ensure!(
                    identity.scope() == anchor.scope()
                        && identity.symbol_name() == anchor.symbol_name()
                        && identity.bar_timestamp_convention() == anchor.bar_timestamp_convention(),
                    "live broker snapshot {} is outside receipt series {}",
                    identity.to_path_component(),
                    anchor.to_path_component()
                );
                anyhow::ensure!(
                    required.contains(&timeframe),
                    "live broker snapshot supplied unexpected direct timeframe {timeframe}"
                );
            }
            anyhow::ensure!(
                by_timeframe.insert(timeframe, snapshot).is_none(),
                "live broker snapshot repeats direct timeframe {timeframe}"
            );
        }
        anyhow::ensure!(
            by_timeframe.len() == required.len(),
            "live broker snapshot provides {} of {} required direct timeframes",
            by_timeframe.len(),
            required.len()
        );

        let root = create_isolated_snapshot_root()?;
        let mut output = Self {
            dataset: None,
            root,
        };
        let mut expected_identities = BTreeMap::new();
        for timeframe in &required {
            let snapshot = by_timeframe
                .remove(timeframe)
                .with_context(|| format!("missing validated live timeframe {timeframe}"))?;
            let chunk = canonical_chunk(&snapshot)?;
            let first = snapshot
                .bars
                .first()
                .context("validated live snapshot lost its first row")?
                .timestamp_ms;
            let last = snapshot
                .bars
                .last()
                .context("validated live snapshot lost its last row")?
                .timestamp_ms;
            let row_count = u64::try_from(snapshot.bars.len())
                .context("live trendbar row count exceeds u64")?;
            let published = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
                configured_root: &output.root,
                identity: &snapshot.identity,
                expected_generation: None,
                requested_from_ms: snapshot.requested_from_ms,
                requested_to_ms: snapshot.requested_to_ms,
                retrieved_unix_ms: snapshot.retrieved_unix_ms,
                returned_from_ms: first,
                returned_to_ms: last,
                row_count,
                chunks: std::iter::once(Ok(chunk)),
            })
            .with_context(|| format!("publish live canonical {timeframe} snapshot"))?;
            anyhow::ensure!(
                published.manifest().identity() == &snapshot.identity
                    && published.row_count() == row_count,
                "reopened live publication disagrees with broker snapshot {timeframe}"
            );
            expected_identities.insert(*timeframe, snapshot.identity);
        }

        let requested_names = required
            .iter()
            .map(|timeframe| timeframe.as_str())
            .collect::<Vec<_>>();
        let dataset = neoethos_data::load_dataset_for_identity_with_timeframes(
            &output.root,
            anchor,
            &requested_names,
        )
        .context("reopen live canonical direct-timeframe series")?;
        for (timeframe, expected_identity) in expected_identities {
            let loaded = dataset
                .source_artifacts
                .get(timeframe.as_str())
                .with_context(|| format!("reopened live series lost {timeframe} artifact"))?;
            anyhow::ensure!(
                loaded.identity() == &expected_identity,
                "reopened live {timeframe} artifact changed broker identity"
            );
        }
        output.dataset = Some(dataset);
        Ok(output)
    }

    pub(crate) fn dataset(&self) -> &SymbolDataset {
        self.dataset
            .as_ref()
            .expect("live canonical dataset exists until snapshot drop")
    }

    #[cfg(test)]
    fn root(&self) -> &std::path::Path {
        &self.root
    }
}

impl Drop for LiveCanonicalFeatureSnapshot {
    fn drop(&mut self) {
        drop(self.dataset.take());
        match fs::remove_dir_all(&self.root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                target: "neoethos_app::live_feature_snapshot",
                root = %self.root.display(),
                error = %error,
                "failed to remove isolated live canonical snapshot root"
            ),
        }
    }
}

fn canonical_chunk(snapshot: &RecentBrokerTrendbarSnapshot) -> Result<CanonicalOhlcvChunk> {
    let carries_volume = snapshot
        .bars
        .first()
        .context("cannot encode an empty live trendbar snapshot")?
        .volume
        .is_some();
    if snapshot
        .bars
        .iter()
        .any(|bar| bar.volume.is_some() != carries_volume)
    {
        bail!("cTrader tick-volume presence changes inside one live response");
    }
    let volume = if carries_volume {
        CanonicalVolumeChunk::Int64(
            snapshot
                .bars
                .iter()
                .map(|bar| bar.volume.expect("volume presence validated"))
                .collect(),
        )
    } else {
        CanonicalVolumeChunk::Absent
    };
    Ok(CanonicalOhlcvChunk {
        timestamp_ms: snapshot.bars.iter().map(|bar| bar.timestamp_ms).collect(),
        open: snapshot.bars.iter().map(|bar| bar.open).collect(),
        high: snapshot.bars.iter().map(|bar| bar.high).collect(),
        low: snapshot.bars.iter().map(|bar| bar.low).collect(),
        close: snapshot.bars.iter().map(|bar| bar.close).collect(),
        volume,
    })
}

fn create_isolated_snapshot_root() -> Result<PathBuf> {
    let parent = std::env::temp_dir();
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    for _ in 0..128 {
        let nonce = LIVE_SNAPSHOT_NONCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            "{LIVE_SNAPSHOT_PREFIX}{}-{unix_ms}-{nonce}",
            std::process::id()
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => {
                return fs::canonicalize(&candidate).with_context(|| {
                    format!(
                        "resolve newly-created live snapshot root {}",
                        candidate.display()
                    )
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create isolated live snapshot root {}", candidate.display())
                });
            }
        }
    }
    Err(anyhow!(
        "could not allocate a unique isolated live canonical snapshot root"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_data::HistoricalBar;
    use neoethos_data::{BarTimestampConvention, CTraderEnvironment};

    fn snapshot(timeframe: CanonicalTimeframe, start_ms: i64) -> RecentBrokerTrendbarSnapshot {
        let step = timeframe.fixed_duration_ms().expect("fixed test timeframe");
        let identity = CanonicalDatasetIdentity::ctrader(
            CTraderEnvironment::Demo,
            "demo.ctraderapi.com",
            42,
            1,
            "EURUSD",
            timeframe,
            BarTimestampConvention::BarOpen,
        )
        .expect("test cTrader identity");
        let bar = |timestamp_ms, price| HistoricalBar {
            timestamp_ms,
            open: price,
            high: price + 0.0002,
            low: price - 0.0002,
            close: price + 0.0001,
            volume: Some(10),
        };
        RecentBrokerTrendbarSnapshot {
            identity,
            requested_from_ms: start_ms - step,
            requested_to_ms: start_ms + step * 2 + 1,
            retrieved_unix_ms: 1_700_000_000_000,
            bars: vec![bar(start_ms, 1.1), bar(start_ms + step, 1.2)],
        }
    }

    #[test]
    fn canonical_live_series_has_exact_artifacts_and_deletes_its_isolated_root() {
        let base = snapshot(CanonicalTimeframe::M1, 1_700_000_040_000);
        let anchor = base.identity.clone();
        let higher = snapshot(CanonicalTimeframe::M5, 1_700_000_000_000);
        let root;
        {
            let live = LiveCanonicalFeatureSnapshot::publish_for_series(
                &anchor,
                &["M1".to_owned(), "M5".to_owned()],
                vec![base, higher],
            )
            .expect("publish isolated live canonical series");
            root = live.root().to_path_buf();
            assert!(root.is_dir());
            assert_eq!(live.dataset().source_artifacts.len(), 2);
            assert!(live.dataset().source_artifacts.contains_key("M1"));
            assert!(live.dataset().source_artifacts.contains_key("M5"));
        }
        assert!(
            !root.exists(),
            "isolated live canonical root must be deleted after leases drop"
        );
    }
}
