//! Machine-readable GPU discovery benchmark contracts.

use crate::gpu_native::engine::{EngineStatus, TransferSnapshot};
use crate::gpu_native::semantics::{
    DISCOVERY_SEMANTICS_VERSION, TRADING_SEMANTICS_VERSION, discovery_semantics_hash,
    semantics_hash_hex, trading_semantics_hash,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const BENCHMARK_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPass {
    CleanTiming,
    Diagnostics,
    NsightSystems,
    NsightCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMode {
    Tiny,
    Snapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrototypeId {
    Legacy,
    A,
    B,
    C,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkIdentity {
    pub git_sha: String,
    pub baseline_sha: String,
    pub dataset_hash: String,
    pub config_hash: String,
    pub seed: u64,
    pub timeframe: String,
    pub backend: String,
    pub prototype: PrototypeId,
    pub fixture: FixtureMode,
    pub pass: BenchmarkPass,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareMetadata {
    pub cpu: String,
    pub ram_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub device_class: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_runtime_version: Option<String>,
    pub cuda_toolkit_version: Option<String>,
    pub gpu_clock_mhz: Option<u64>,
    pub power_limit_watts: Option<f64>,
    pub temperature_celsius: Option<f64>,
    pub dedicated_vram_bytes: Option<u64>,
    pub cuda_occupancy: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepPoint {
    pub population: usize,
    pub batch_size: usize,
    pub bar_count: usize,
    pub feature_count: usize,
    pub scenario_count: usize,
    pub calendar_days: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub count: usize,
    pub median: f64,
    pub p95: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl DistributionSummary {
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        let median = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let mean = sorted.iter().sum::<f64>() / count as f64;
        let variance = sorted
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count as f64;
        Some(Self {
            count,
            median,
            p95,
            variance,
            min: sorted[0],
            max: sorted[count - 1],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub candidates_per_second: Option<f64>,
    pub candidate_bars_per_second: Option<f64>,
    pub trades_per_second: Option<f64>,
    pub peak_vram_bytes: Option<u64>,
    pub event_density: Option<f64>,
    pub hold_bars: Option<DistributionSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCoverage {
    pub total_candidates: usize,
    pub supported_candidates: usize,
    pub unsupported_candidates: usize,
    pub unsupported_reasons: BTreeMap<String, usize>,
}

impl CapabilityCoverage {
    pub fn supported_fraction(&self) -> f64 {
        if self.total_candidates == 0 {
            0.0
        } else {
            self.supported_candidates as f64 / self.total_candidates as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityStatus {
    pub matched: bool,
    pub first_divergent_level: Option<u8>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub identity: BenchmarkIdentity,
    pub hardware: HardwareMetadata,
    pub trading_semantics_version: u32,
    pub trading_semantics_hash: String,
    pub discovery_semantics_version: u32,
    pub discovery_semantics_hash: String,
    pub engine_status: String,
    pub warmup_count: usize,
    pub measured_repetitions: usize,
    pub sweep: SweepPoint,
    pub total_wall_seconds: DistributionSummary,
    pub stage_wall_seconds: BTreeMap<String, DistributionSummary>,
    pub throughput: ThroughputMetrics,
    pub transfers: TransferSnapshotReport,
    pub coverage: CapabilityCoverage,
    pub parity: ParityStatus,
    pub notes: Vec<String>,
}

impl BenchmarkReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: BenchmarkIdentity,
        hardware: HardwareMetadata,
        engine_status: EngineStatus,
        warmup_count: usize,
        measured_repetitions: usize,
        sweep: SweepPoint,
        total_wall_seconds: DistributionSummary,
        stage_wall_seconds: BTreeMap<String, DistributionSummary>,
        throughput: ThroughputMetrics,
        transfers: TransferSnapshot,
        coverage: CapabilityCoverage,
        parity: ParityStatus,
        notes: Vec<String>,
    ) -> Self {
        Self {
            schema_version: BENCHMARK_REPORT_SCHEMA_VERSION,
            identity,
            hardware,
            trading_semantics_version: TRADING_SEMANTICS_VERSION,
            trading_semantics_hash: semantics_hash_hex(trading_semantics_hash()),
            discovery_semantics_version: DISCOVERY_SEMANTICS_VERSION,
            discovery_semantics_hash: semantics_hash_hex(discovery_semantics_hash()),
            engine_status: format!("{engine_status:?}"),
            warmup_count,
            measured_repetitions,
            sweep,
            total_wall_seconds,
            stage_wall_seconds,
            throughput,
            transfers: transfers.into(),
            coverage,
            parity,
            notes,
        }
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, bytes)?;
        fs::rename(temporary, path)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferSnapshotReport {
    pub dataset_uploads: u64,
    pub gene_uploads: u64,
    pub scenario_uploads: u64,
    pub full_d2h_readbacks: u64,
    pub compact_d2h_readbacks: u64,
    pub chained_reuploads: u64,
    pub synchronization_events: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

impl From<TransferSnapshot> for TransferSnapshotReport {
    fn from(value: TransferSnapshot) -> Self {
        Self {
            dataset_uploads: value.dataset_uploads,
            gene_uploads: value.gene_uploads,
            scenario_uploads: value.scenario_uploads,
            full_d2h_readbacks: value.full_d2h_readbacks,
            compact_d2h_readbacks: value.compact_d2h_readbacks,
            chained_reuploads: value.chained_reuploads,
            synchronization_events: value.synchronization_events,
            h2d_bytes: value.h2d_bytes,
            d2h_bytes: value.d2h_bytes,
        }
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_reports_median_p95_and_variance() {
        let summary = DistributionSummary::from_samples(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(summary.count, 4);
        assert_eq!(summary.median, 2.5);
        assert!(summary.p95 > 3.0);
        assert!(summary.variance > 0.0);
    }

    #[test]
    fn empty_and_non_finite_samples_are_rejected() {
        assert!(DistributionSummary::from_samples(&[]).is_none());
        assert!(DistributionSummary::from_samples(&[f64::NAN]).is_none());
    }

    #[test]
    fn coverage_fraction_is_explicit() {
        let coverage = CapabilityCoverage {
            total_candidates: 10,
            supported_candidates: 7,
            unsupported_candidates: 3,
            unsupported_reasons: BTreeMap::new(),
        };
        assert_eq!(coverage.supported_fraction(), 0.7);
    }
}
