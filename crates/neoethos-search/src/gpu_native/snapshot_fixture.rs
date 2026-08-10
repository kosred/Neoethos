//! Versioned, validated real-data population snapshot fixture.

use crate::backend::{EvaluationBackend, evaluate_population_core_with_backend_and_audit};
use crate::eval::{BacktestSettings, PopulationEvalInputs, SmcRow};
use crate::gpu_native::cpu_strategy::CpuStrategyAuditContext;
use crate::gpu_native::parity_hierarchy::TraceComparisonReport;
use crate::gpu_native::population_fixture::TinyPopulationFixture;
use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
};
use crate::gpu_native::prototype_population::{
    PrototypeBcRequirements, PrototypePopulationError, PrototypePopulationWorkload,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub const SNAPSHOT_FIXTURE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotSettingsDto {
    pub max_hold_bars: usize,
    pub min_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub gap_threshold_ms: i64,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    /// The three session buckets, or `None` for a flat spread.
    ///
    /// This DTO is the only thing the device settings are built from, and it
    /// used to overwrite the profile with `None` on the way through — so a run
    /// configured with per-session spread reached the card as a flat charge,
    /// whatever the ABI could express. The CPU meanwhile resolved per bar. The
    /// two lanes priced the same trade differently and no fixture noticed,
    /// because every fixture leaves the profile unset.
    pub session_spread_profile: Option<crate::eval::SessionSpreadProfile>,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    pub pnl_conversion_fee_rate: f64,
    pub risk_based_sizing: bool,
    pub risk_per_trade_min: f64,
    pub risk_per_trade_max: f64,
    pub high_quality_confidence: f64,
    pub adaptive_base_pips: Option<Vec<f64>>,
    pub adaptive_rr: f64,
}

impl SnapshotSettingsDto {
    pub(crate) fn from_settings(settings: &BacktestSettings) -> Self {
        Self {
            max_hold_bars: settings.max_hold_bars,
            min_hold_bars: settings.min_hold_bars,
            max_trades_per_day: settings.max_trades_per_day,
            gap_threshold_ms: settings.gap_threshold_ms,
            trailing_enabled: settings.trailing_enabled,
            trailing_atr_multiplier: settings.trailing_atr_multiplier,
            trailing_be_trigger_r: settings.trailing_be_trigger_r,
            pip_value: settings.pip_value,
            spread_pips: settings.spread_pips,
            session_spread_profile: settings.session_spread_profile,
            commission_per_trade: settings.commission_per_trade,
            pip_value_per_lot: settings.pip_value_per_lot,
            swap_long_pips_per_day: settings.swap_long_pips_per_day,
            swap_short_pips_per_day: settings.swap_short_pips_per_day,
            pnl_conversion_fee_rate: settings.pnl_conversion_fee_rate,
            risk_based_sizing: settings.risk_based_sizing,
            risk_per_trade_min: settings.risk_per_trade_min,
            risk_per_trade_max: settings.risk_per_trade_max,
            high_quality_confidence: settings.high_quality_confidence,
            adaptive_base_pips: settings
                .adaptive_base_pips
                .as_ref()
                .map(|values| values.to_vec()),
            adaptive_rr: settings.adaptive_rr,
        }
    }

    pub(crate) fn to_settings(&self) -> BacktestSettings {
        let mut settings = BacktestSettings::default();
        settings.max_hold_bars = self.max_hold_bars;
        settings.min_hold_bars = self.min_hold_bars;
        settings.max_trades_per_day = self.max_trades_per_day;
        settings.gap_threshold_ms = self.gap_threshold_ms;
        settings.trailing_enabled = self.trailing_enabled;
        settings.trailing_atr_multiplier = self.trailing_atr_multiplier;
        settings.trailing_be_trigger_r = self.trailing_be_trigger_r;
        settings.pip_value = self.pip_value;
        settings.spread_pips = self.spread_pips;
        settings.commission_per_trade = self.commission_per_trade;
        settings.pip_value_per_lot = self.pip_value_per_lot;
        settings.kill_zones_enabled = false;
        settings.session_spread_profile = self.session_spread_profile;
        settings.swap_long_pips_per_day = self.swap_long_pips_per_day;
        settings.swap_short_pips_per_day = self.swap_short_pips_per_day;
        settings.pnl_conversion_fee_rate = self.pnl_conversion_fee_rate;
        settings.risk_based_sizing = self.risk_based_sizing;
        settings.risk_per_trade_min = self.risk_per_trade_min;
        settings.risk_per_trade_max = self.risk_per_trade_max;
        settings.high_quality_confidence = self.high_quality_confidence;
        settings.adaptive_base_pips = self
            .adaptive_base_pips
            .clone()
            .map(|values| Arc::<[f64]>::from(values));
        settings.adaptive_vol_mult = 0.0;
        settings.adaptive_rr = self.adaptive_rr;
        settings
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotFixtureDto {
    pub schema_version: u32,
    pub timeframe: String,
    pub source_description: String,
    pub close: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    /// Feature-major contiguous `[feature][bar]` values.
    pub indicators: Vec<f32>,
    pub feature_count: usize,
    pub gene_offsets: Vec<i32>,
    pub gene_indices: Vec<i32>,
    pub gene_weights: Vec<f32>,
    pub long_thresholds: Vec<f32>,
    pub short_thresholds: Vec<f32>,
    pub months: Vec<i64>,
    pub days: Vec<i64>,
    pub timestamps: Vec<i64>,
    pub stop_pips: Vec<f64>,
    pub target_pips: Vec<f64>,
    pub stop_vol_multipliers: Vec<f64>,
    pub smc_data: Vec<SmcRow>,
    pub gene_smc_flags: Vec<SmcRow>,
    pub smc_weights: [f32; 11],
    pub settings: SnapshotSettingsDto,
}

#[derive(Debug, Clone)]
pub struct SnapshotPopulationFixture {
    timeframe: String,
    source_description: String,
    close: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    indicators: Array2<f32>,
    gene_offsets: Vec<i32>,
    gene_indices: Vec<i32>,
    gene_weights: Vec<f32>,
    long_thresholds: Vec<f32>,
    short_thresholds: Vec<f32>,
    months: Vec<i64>,
    days: Vec<i64>,
    timestamps: Vec<i64>,
    stop_pips: Vec<f64>,
    target_pips: Vec<f64>,
    stop_vol_multipliers: Vec<f64>,
    smc_data: Vec<SmcRow>,
    gene_smc_flags: Vec<SmcRow>,
    smc_weights: [f32; 11],
    settings: BacktestSettings,
}

impl SnapshotPopulationFixture {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).map_err(|error| format!("read snapshot {}: {error}", path.display()))?;
        let dto: SnapshotFixtureDto = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse snapshot {}: {error}", path.display()))?;
        Self::from_dto(dto)
    }

    pub fn from_dto(dto: SnapshotFixtureDto) -> Result<Self, String> {
        if dto.schema_version != SNAPSHOT_FIXTURE_SCHEMA_VERSION {
            return Err(format!(
                "snapshot schema {} is unsupported; expected {}",
                dto.schema_version, SNAPSHOT_FIXTURE_SCHEMA_VERSION
            ));
        }
        let bars = dto.close.len();
        let population = dto.long_thresholds.len();
        if bars < 2 || dto.feature_count == 0 || population == 0 {
            return Err("snapshot requires at least 2 bars, 1 feature and 1 candidate".into());
        }
        for (name, len) in [
            ("high", dto.high.len()),
            ("low", dto.low.len()),
            ("months", dto.months.len()),
            ("days", dto.days.len()),
            ("timestamps", dto.timestamps.len()),
            ("smc_data", dto.smc_data.len()),
        ] {
            if len != bars {
                return Err(format!("snapshot {name} length {len} != bars {bars}"));
            }
        }
        if dto.indicators.len() != dto.feature_count.saturating_mul(bars) {
            return Err("snapshot indicator shape is inconsistent".into());
        }
        for (name, len) in [
            ("short_thresholds", dto.short_thresholds.len()),
            ("stop_pips", dto.stop_pips.len()),
            ("target_pips", dto.target_pips.len()),
            ("stop_vol_multipliers", dto.stop_vol_multipliers.len()),
            ("gene_smc_flags", dto.gene_smc_flags.len()),
        ] {
            if len != population {
                return Err(format!(
                    "snapshot {name} length {len} != population {population}"
                ));
            }
        }
        if dto.gene_offsets.len() != population + 1
            || dto.gene_offsets.first().copied() != Some(0)
            || dto.gene_offsets.last().copied() != Some(dto.gene_indices.len() as i32)
            || dto.gene_indices.len() != dto.gene_weights.len()
        {
            return Err("snapshot gene CSR arrays are inconsistent".into());
        }
        if dto
            .gene_indices
            .iter()
            .any(|index| *index < 0 || (*index as usize) >= dto.feature_count)
        {
            return Err("snapshot gene index is outside the feature range".into());
        }
        if dto
            .close
            .iter()
            .chain(&dto.high)
            .chain(&dto.low)
            .any(|value| !value.is_finite())
            || dto.indicators.iter().any(|value| !value.is_finite())
            || dto.gene_weights.iter().any(|value| !value.is_finite())
            || dto.long_thresholds.iter().any(|value| !value.is_finite())
            || dto.short_thresholds.iter().any(|value| !value.is_finite())
        {
            return Err("snapshot contains non-finite numerical values".into());
        }
        if let Some(base) = dto.settings.adaptive_base_pips.as_ref() {
            if base.len() != bars || base.iter().any(|value| !value.is_finite()) {
                return Err("snapshot adaptive base must be finite and aligned to bars".into());
            }
        }

        let indicators = Array2::from_shape_vec((dto.feature_count, bars), dto.indicators)
            .map_err(|error| format!("construct snapshot indicator matrix: {error}"))?;
        let settings = dto.settings.to_settings();
        Ok(Self {
            timeframe: dto.timeframe,
            source_description: dto.source_description,
            close: dto.close,
            high: dto.high,
            low: dto.low,
            indicators,
            gene_offsets: dto.gene_offsets,
            gene_indices: dto.gene_indices,
            gene_weights: dto.gene_weights,
            long_thresholds: dto.long_thresholds,
            short_thresholds: dto.short_thresholds,
            months: dto.months,
            days: dto.days,
            timestamps: dto.timestamps,
            stop_pips: dto.stop_pips,
            target_pips: dto.target_pips,
            stop_vol_multipliers: dto.stop_vol_multipliers,
            smc_data: dto.smc_data,
            gene_smc_flags: dto.gene_smc_flags,
            smc_weights: dto.smc_weights,
            settings,
        })
    }

    pub fn timeframe(&self) -> &str {
        &self.timeframe
    }

    pub fn source_description(&self) -> &str {
        &self.source_description
    }

    pub fn population(&self) -> usize {
        self.long_thresholds.len()
    }

    pub fn bars(&self) -> usize {
        self.close.len()
    }

    pub fn features(&self) -> usize {
        self.indicators.nrows()
    }

    pub fn candidate_bars(&self) -> u64 {
        (self.population() as u64).saturating_mul(self.bars() as u64)
    }

    pub fn prototype_a_uploads(
        &self,
    ) -> (
        PrototypeADatasetUpload,
        PrototypeAGeneUpload,
        PrototypeAScenarioUpload,
    ) {
        let candidate_ids = (0..self.population())
            .map(|index| index as u64)
            .collect::<Vec<_>>();
        let scenarios = candidate_ids
            .iter()
            .copied()
            // Through the ONE builder. A hand-written descriptor with
            // `..default()` wrote `spread_ticks: 0, commission_micros: 0`, which
            // stopped meaning "use the settings' costs" and started meaning
            // "charge nothing" when the sentinel became -1.
            .map(|candidate_id| {
                crate::gpu_native::scenario::base_scenario(candidate_id, candidate_id, self.bars())
            })
            .collect();
        (
            PrototypeADatasetUpload {
                close: self.close.clone(),
                high: self.high.clone(),
                low: self.low.clone(),
                indicators: self.indicators.iter().copied().collect(),
                feature_count: self.features(),
                months: self.months.clone(),
                days: self.days.clone(),
                timestamps: self.timestamps.clone(),
                smc_data: self.smc_data.clone(),
                settings: SnapshotSettingsDto::from_settings(&self.settings),
            },
            PrototypeAGeneUpload {
                candidate_ids,
                offsets: self.gene_offsets.clone(),
                indices: self.gene_indices.clone(),
                weights: self.gene_weights.clone(),
                long_thresholds: self.long_thresholds.clone(),
                short_thresholds: self.short_thresholds.clone(),
                stop_pips: self.stop_pips.clone(),
                target_pips: self.target_pips.clone(),
                stop_vol_multipliers: self.stop_vol_multipliers.clone(),
                smc_flags: self.gene_smc_flags.clone(),
                smc_weights: self.smc_weights,
                gate_threshold: 0.0,
            },
            PrototypeAScenarioUpload { scenarios },
        )
    }

    pub fn population_workload(
        &self,
        requirements: PrototypeBcRequirements,
    ) -> Result<PrototypePopulationWorkload, PrototypePopulationError> {
        let (dataset, genes, scenarios) = self.prototype_a_uploads();
        PrototypePopulationWorkload::from_uploads(dataset, genes, scenarios, requirements)
    }

    pub fn evaluate(
        &self,
        backend: EvaluationBackend,
        audit: &CpuStrategyAuditContext,
    ) -> Result<Vec<[f64; 11]>, String> {
        evaluate_population_core_with_backend_and_audit(
            PopulationEvalInputs {
                close: &self.close,
                high: &self.high,
                low: &self.low,
                indicators: self.indicators.view(),
                gene_offsets: &self.gene_offsets,
                gene_indices: &self.gene_indices,
                gene_weights: &self.gene_weights,
                long_thr: &self.long_thresholds,
                short_thr: &self.short_thresholds,
                month_idx: &self.months,
                day_idx: &self.days,
                timestamps: &self.timestamps,
                sl_pips: &self.stop_pips,
                tp_pips: &self.target_pips,
                stop_vol_mult: &self.stop_vol_multipliers,
                smc_data: &self.smc_data,
                gene_smc_flags: &self.gene_smc_flags,
                gate_threshold: 0.0,
                weights: &self.smc_weights,
                settings: &self.settings,
            },
            backend,
            audit,
        )
    }

    pub fn compare_final_metrics(
        reference: &[[f64; 11]],
        candidate: &[[f64; 11]],
    ) -> TraceComparisonReport {
        TinyPopulationFixture::compare_final_metrics(reference, candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dto() -> SnapshotFixtureDto {
        SnapshotFixtureDto {
            schema_version: SNAPSHOT_FIXTURE_SCHEMA_VERSION,
            timeframe: "M1".into(),
            source_description: "unit fixture".into(),
            close: vec![1.0, 1.1, 1.2],
            high: vec![1.1, 1.2, 1.3],
            low: vec![0.9, 1.0, 1.1],
            indicators: vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
            feature_count: 2,
            gene_offsets: vec![0, 2],
            gene_indices: vec![0, 1],
            gene_weights: vec![0.5, -0.5],
            long_thresholds: vec![0.2],
            short_thresholds: vec![-0.2],
            months: vec![0, 0, 0],
            days: vec![0, 0, 0],
            timestamps: vec![1, 2, 3],
            stop_pips: vec![10.0],
            target_pips: vec![20.0],
            stop_vol_multipliers: vec![0.0],
            smc_data: vec![[0; 11]; 3],
            gene_smc_flags: vec![[0; 11]],
            smc_weights: [0.0; 11],
            settings: SnapshotSettingsDto {
                max_hold_bars: 2,
                min_hold_bars: 0,
                max_trades_per_day: 5,
                gap_threshold_ms: 0,
                trailing_enabled: false,
                trailing_atr_multiplier: 0.0,
                trailing_be_trigger_r: 0.0,
                pip_value: 0.0001,
                spread_pips: 0.0,
                session_spread_profile: None,
                commission_per_trade: 0.0,
                pip_value_per_lot: 10.0,
                swap_long_pips_per_day: 0.0,
                swap_short_pips_per_day: 0.0,
                pnl_conversion_fee_rate: 0.0,
                risk_based_sizing: false,
                risk_per_trade_min: 0.005,
                risk_per_trade_max: 0.01,
                high_quality_confidence: 0.65,
                adaptive_base_pips: None,
                adaptive_rr: 2.0,
            },
        }
    }

    #[test]
    fn valid_snapshot_builds_a_shape_stable_fixture() {
        let fixture = SnapshotPopulationFixture::from_dto(dto()).unwrap();
        assert_eq!(fixture.population(), 1);
        assert_eq!(fixture.bars(), 3);
        assert_eq!(fixture.features(), 2);
        assert_eq!(fixture.timeframe(), "M1");
    }

    #[test]
    fn malformed_snapshot_is_rejected_before_execution() {
        let mut malformed = dto();
        malformed.high.pop();
        assert!(
            SnapshotPopulationFixture::from_dto(malformed)
                .unwrap_err()
                .contains("high length")
        );
    }

    #[test]
    fn snapshot_exports_an_explicit_bc_population_workload() {
        let fixture = SnapshotPopulationFixture::from_dto(dto()).unwrap();
        let (dataset, genes, scenarios) = fixture.prototype_a_uploads();
        let workload = fixture
            .population_workload(
                crate::gpu_native::prototype_population::PrototypeBcRequirements {
                    prop_firm_state:
                        crate::gpu_native::prototype_population::PropFirmRequirement::NotRequested,
                },
            )
            .unwrap();

        assert_eq!(workload.dataset, dataset);
        assert_eq!(workload.genes, genes);
        assert_eq!(workload.scenarios, scenarios);
        assert_eq!(
            workload.dataset.encode().unwrap(),
            dataset.encode().unwrap()
        );
        assert_eq!(workload.genes.encode().unwrap(), genes.encode().unwrap());
        assert_eq!(
            workload.scenarios.encode().unwrap(),
            scenarios.encode().unwrap()
        );
    }
}
