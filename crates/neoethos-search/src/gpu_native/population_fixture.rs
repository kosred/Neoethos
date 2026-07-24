//! Deterministic engine-only population fixture for executable GPU benchmarks.

use crate::backend::{EvaluationBackend, evaluate_population_core_with_backend_and_audit};
use crate::eval::{BacktestSettings, PopulationEvalInputs, SmcRow};
use crate::gpu_native::cpu_strategy::CpuStrategyAuditContext;
use crate::gpu_native::parity_hierarchy::{
    FloatTolerance, ParityPolicy, ParityTrace, TraceComparisonReport, compare_traces,
};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct TinyPopulationFixture {
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

impl TinyPopulationFixture {
    pub fn new(population: usize, bars: usize, features: usize) -> Self {
        let population = population.max(1);
        let bars = bars.max(64);
        let features = features.max(2);
        let close: Vec<f64> = (0..bars)
            .map(|bar| {
                let x = bar as f64;
                1.10 + (x * 0.031).sin() * 0.003 + (x * 0.007).cos() * 0.001
            })
            .collect();
        let high = close.iter().map(|price| price + 0.0007).collect();
        let low = close.iter().map(|price| price - 0.0007).collect();
        let indicators = Array2::from_shape_fn((features, bars), |(feature, bar)| {
            let phase = feature as f32 * 0.37;
            let x = bar as f32 * (0.017 + feature as f32 * 0.0003);
            (x + phase).sin() * 0.75 + (x * 0.31 - phase).cos() * 0.20
        });

        let terms_per_gene = features.min(4).max(2);
        let mut gene_offsets = Vec::with_capacity(population + 1);
        let mut gene_indices = Vec::with_capacity(population * terms_per_gene);
        let mut gene_weights = Vec::with_capacity(population * terms_per_gene);
        gene_offsets.push(0);
        for candidate in 0..population {
            for term in 0..terms_per_gene {
                gene_indices.push(((candidate + term * 3) % features) as i32);
                let magnitude = 0.35 + ((candidate + term) % 5) as f32 * 0.11;
                gene_weights.push(if (candidate + term) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                });
            }
            gene_offsets.push(gene_indices.len() as i32);
        }

        let timestamps: Vec<i64> = (0..bars)
            .map(|bar| 1_700_000_000_000_i64 + bar as i64 * 60_000)
            .collect();
        let months = (0..bars).map(|bar| (bar / 43_200) as i64).collect();
        let days = (0..bars).map(|bar| (bar / 1_440) as i64).collect();
        let long_thresholds = (0..population)
            .map(|candidate| 0.20 + (candidate % 3) as f32 * 0.03)
            .collect();
        let short_thresholds = (0..population)
            .map(|candidate| -0.20 - (candidate % 3) as f32 * 0.03)
            .collect();
        let stop_pips = vec![18.0; population];
        let target_pips = vec![36.0; population];
        let stop_vol_multipliers = vec![0.0; population];
        let smc_data = vec![[0_i8; 11]; bars];
        let gene_smc_flags = vec![[0_i8; 11]; population];

        let mut settings = BacktestSettings::default();
        settings.max_hold_bars = 12;
        settings.trailing_enabled = false;
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.01;
        settings.high_quality_confidence = 0.65;

        Self {
            close,
            high,
            low,
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thresholds,
            short_thresholds,
            months,
            days,
            timestamps,
            stop_pips,
            target_pips,
            stop_vol_multipliers,
            smc_data,
            gene_smc_flags,
            smc_weights: [0.0; 11],
            settings,
        }
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
        let reference = ParityTrace {
            final_metrics: reference.to_vec(),
            ..ParityTrace::default()
        };
        let candidate = ParityTrace {
            final_metrics: candidate.to_vec(),
            ..ParityTrace::default()
        };
        let policy = ParityPolicy {
            metrics: FloatTolerance {
                absolute: 1.0e-3,
                relative: 1.0e-3,
                max_ulps: 64,
            },
            ..ParityPolicy::default()
        };
        compare_traces("cpu_reference", "gpu_candidate", &reference, &candidate, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_reference_fixture_is_deterministic_and_shape_stable() {
        let fixture = TinyPopulationFixture::new(8, 256, 6);
        let first_audit = CpuStrategyAuditContext::validation_reference(1);
        let second_audit = CpuStrategyAuditContext::validation_reference(2);
        let first = fixture
            .evaluate(EvaluationBackend::CPU_CANONICAL, &first_audit)
            .unwrap();
        let second = fixture
            .evaluate(EvaluationBackend::CPU_CANONICAL, &second_audit)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 8);
        assert_eq!(fixture.candidate_bars(), 8 * 256);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn strict_fixture_fails_without_executing_cpu_when_gpu_is_not_compiled() {
        let fixture = TinyPopulationFixture::new(4, 128, 4);
        let audit = CpuStrategyAuditContext::production(3);
        let error = fixture
            .evaluate(EvaluationBackend::GPU_REQUIRED, &audit)
            .unwrap_err();
        assert!(error.contains("compiled without a GPU backend"));
        audit.snapshot().assert_zero_executed().unwrap();
    }
}
