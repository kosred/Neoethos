//! Persistent CubeCL `BacktestEngine` adapter for Prototype C.
//!
//! The engine owns one long-lived CubeCL client and one device-resident
//! workspace. `evaluate` submits the complete sparse chain — signal synthesis,
//! causal entry emission, compact first-hit search, deterministic
//! non-overlapping trade stitching and the exact cost/sizing/metric reduction —
//! without reading anything back. `readback_compact` is the only metrics D2H
//! boundary.
//!
//! Device storage is shader-portable SoA `i32`/`f32`; the C-ABI structs used by
//! the native-CUDA Prototype B never enter this runtime, and no native pointer
//! ever crosses into CubeCL. Canonical `u64` candidate/scenario identity is held
//! in resident host tables and re-attached at the single compact readback, after
//! the device-emitted table indices have been range-validated.
//!
//! Prices are carried in pip units. That is the same conditioning the fused
//! Prototype A path uses and it keeps an `f32` device path meaningful for FX
//! prices; all comparisons and P&L terms are algebraically identical to the
//! canonical price-space form.

use crate::gpu_native::engine::{EngineError, HostSurvivorSummary};
use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
};
use crate::gpu_native::prototype_b_engine::compact_rank_fields;

pub const C_METRIC_WIDTH: usize = 11;
pub(crate) const SMC_WIDTH: usize = 11;

// ---------------------------------------------------------------------------
// Host-side device state
// ---------------------------------------------------------------------------

/// Backend-independent host projection of one workload into shader-portable
/// buffers. Building it never touches a device, so it is unit-testable in a
/// CPU-only build.
#[derive(Debug, Clone, PartialEq)]
pub struct CPopulationHostBuffers {
    pub close_pips: Vec<f32>,
    pub high_pips: Vec<f32>,
    pub low_pips: Vec<f32>,
    pub gap_flags: Vec<i32>,
    pub timestamp_pair: Vec<i32>,
    pub months: Vec<i32>,
    pub days: Vec<i32>,
    pub smc_rows: Vec<i32>,
    pub adaptive_base_pips: Vec<f32>,
    pub has_adaptive_base: bool,
    pub bars: usize,
    pub feature_count: usize,
}

impl CPopulationHostBuffers {
    pub fn from_dataset(dataset: &PrototypeADatasetUpload) -> Result<Self, EngineError> {
        let bars = dataset.bars();
        if bars == 0 {
            return Err(EngineError::Backend(
                "Prototype C dataset upload has no bars".into(),
            ));
        }
        let settings = dataset.settings.to_settings();
        let pip = if settings.pip_value.abs() < 1.0e-12 {
            1.0e-12
        } else {
            settings.pip_value
        };
        let to_pips = |values: &[f64]| {
            values
                .iter()
                .map(|value| (value / pip) as f32)
                .collect::<Vec<f32>>()
        };

        // Gap detection is dataset-only preprocessing in exact f64: it depends on
        // no candidate and is computed once per logical upload.
        let gap_flags = (0..bars)
            .map(|bar| {
                if bar == 0 || settings.gap_threshold_ms <= 0 {
                    return 0;
                }
                let previous = dataset.timestamps[bar - 1];
                let current = dataset.timestamps[bar];
                i32::from(current > previous && current - previous >= settings.gap_threshold_ms)
            })
            .collect::<Vec<i32>>();

        let mut timestamp_pair = Vec::with_capacity(bars * 2);
        for timestamp in &dataset.timestamps {
            let days = timestamp.div_euclid(86_400_000);
            let ms = timestamp.rem_euclid(86_400_000);
            timestamp_pair.push(saturating_i32(days));
            timestamp_pair.push(saturating_i32(ms));
        }

        let adaptive_base_pips = settings
            .adaptive_base_pips
            .as_ref()
            .filter(|values| values.len() == bars)
            .map(|values| values.iter().map(|value| *value as f32).collect::<Vec<_>>());
        let has_adaptive_base = adaptive_base_pips.is_some();

        Ok(Self {
            close_pips: to_pips(&dataset.close),
            high_pips: to_pips(&dataset.high),
            low_pips: to_pips(&dataset.low),
            gap_flags,
            timestamp_pair,
            months: dataset.months.iter().map(|v| saturating_i32(*v)).collect(),
            days: dataset.days.iter().map(|v| saturating_i32(*v)).collect(),
            smc_rows: {
                let rows = dataset
                    .smc_data
                    .iter()
                    .flatten()
                    .map(|value| i32::from(*value))
                    .collect::<Vec<i32>>();
                debug_assert_eq!(rows.len(), bars * SMC_WIDTH);
                rows
            },
            adaptive_base_pips: adaptive_base_pips.unwrap_or_else(|| vec![0.0]),
            has_adaptive_base,
            bars,
            feature_count: dataset.feature_count,
        })
    }
}

fn saturating_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Widen validated device metrics into the canonical compact summary.
///
/// Table indices are range-checked and non-finite values are rejected: a device
/// result never becomes a survivor row on trust alone.
pub fn survivor_summary_from_device_metrics(
    metrics: &[f32],
    genes: &PrototypeAGeneUpload,
    scenarios: &PrototypeAScenarioUpload,
) -> Result<HostSurvivorSummary, EngineError> {
    let population = genes.population();
    if metrics.len() != population * C_METRIC_WIDTH {
        return Err(EngineError::Backend(format!(
            "Prototype C returned {} metric values for a population of {population}",
            metrics.len()
        )));
    }
    let mut candidate_ids = Vec::with_capacity(population);
    let mut scenario_ids = Vec::with_capacity(population);
    let mut rows = Vec::with_capacity(population);
    let mut rank_keys = Vec::with_capacity(population);
    for candidate in 0..population {
        let mut row = [0.0_f64; C_METRIC_WIDTH];
        for slot in 0..C_METRIC_WIDTH {
            let value = metrics[candidate * C_METRIC_WIDTH + slot];
            if !value.is_finite() {
                return Err(EngineError::Backend(format!(
                    "Prototype C metric [{candidate}][{slot}] is non-finite"
                )));
            }
            row[slot] = f64::from(value);
        }
        candidate_ids.push(genes.candidate_ids[candidate]);
        scenario_ids.push(scenarios.scenarios[candidate].scenario_id);
        rank_keys.push(compact_rank_fields(&row)?);
        rows.push(row);
    }
    Ok(HostSurvivorSummary {
        candidate_ids,
        scenario_ids,
        metrics: rows,
        rank_keys,
    })
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
pub use device::{PrototypeCBacktestEngine, PrototypeCResources, create_prototype_c_engine};

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
mod device;

#[cfg(all(test, any(feature = "gpu-cuda", feature = "gpu-vulkan")))]
mod device_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_native::population_fixture::TinyPopulationFixture;

    #[test]
    fn host_buffers_preserve_shape_and_precompute_gaps_exactly() {
        let (mut dataset, _, _) = TinyPopulationFixture::new(2, 128, 4).prototype_a_uploads();
        dataset.settings.gap_threshold_ms = 120_000;
        // One deliberate 10-minute hole in an otherwise 1-minute series.
        for timestamp in dataset.timestamps.iter_mut().skip(64) {
            *timestamp += 600_000;
        }

        let buffers = CPopulationHostBuffers::from_dataset(&dataset).unwrap();

        assert_eq!(buffers.bars, dataset.bars());
        assert_eq!(buffers.close_pips.len(), dataset.bars());
        assert_eq!(buffers.timestamp_pair.len(), dataset.bars() * 2);
        assert_eq!(buffers.smc_rows.len(), dataset.bars() * SMC_WIDTH);
        assert_eq!(buffers.gap_flags[0], 0);
        assert_eq!(buffers.gap_flags[64], 1, "the injected hole must be a gap");
        assert_eq!(buffers.gap_flags[65], 0);
        assert!(!buffers.has_adaptive_base);
    }

    #[test]
    fn pip_space_conversion_is_reversible_within_f32_resolution() {
        let (dataset, _, _) = TinyPopulationFixture::new(1, 96, 2).prototype_a_uploads();
        let pip = dataset.settings.pip_value;
        let buffers = CPopulationHostBuffers::from_dataset(&dataset).unwrap();
        for (bar, close) in dataset.close.iter().enumerate() {
            let restored = f64::from(buffers.close_pips[bar]) * pip;
            assert!(
                (restored - close).abs() < pip * 1.0e-2,
                "bar {bar}: {restored} vs {close}"
            );
        }
    }

    #[test]
    fn device_metric_widening_rejects_wrong_shape_and_non_finite_values() {
        let (_, genes, scenarios) = TinyPopulationFixture::new(3, 96, 4).prototype_a_uploads();
        let mut metrics = vec![0.0_f32; genes.population() * C_METRIC_WIDTH];
        let summary = survivor_summary_from_device_metrics(&metrics, &genes, &scenarios).unwrap();
        assert_eq!(summary.candidate_ids, genes.candidate_ids);
        assert_eq!(summary.metrics.len(), genes.population());

        metrics[5] = f32::NAN;
        assert!(survivor_summary_from_device_metrics(&metrics, &genes, &scenarios).is_err());

        assert!(survivor_summary_from_device_metrics(&metrics[..3], &genes, &scenarios).is_err());
    }
}
