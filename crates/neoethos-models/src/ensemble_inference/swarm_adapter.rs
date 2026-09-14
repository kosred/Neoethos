//! [`super::ExpertModel`] adapter for the **swarm_forecaster** — the last
//! "trained but never voting" model (D1.2.8, operator directive 2026-07-11:
//! every trained model votes unless its job is search).
//!
//! ## Why this adapter is shaped differently
//!
//! [`SwarmForecaster`] is a stateful univariate PRICE forecaster
//! (`fit_series` on a close series, then `forecast(&mut self, horizon)`),
//! not a per-row classifier. Two honest constraints follow:
//!
//! 1. **It votes only on the LAST row.** A per-row historical vote would
//!    require an O(n) walk-forward refit per row (unusable) or forecasting
//!    from the full series for early rows (LOOKAHEAD). The live ML gate
//!    reads exactly one row — the latest bar — so live it votes every bar;
//!    on historical/batch frames every row before the last is explicitly
//!    invalid with `Warmup`. No fake probability, no lookahead.
//! 2. **It is stateless per `predict` call.** `forecast` needs `&mut self`;
//!    instead of interior mutability, each call constructs a fresh
//!    forecaster, restores the trained artifact (configuration: horizon,
//!    ensemble strategy, agent selection), refits on the CURRENT price
//!    series from the incoming frame, and forecasts. A univariate
//!    fit-then-forecast per closed bar costs well under a second.
//!
//! ## Forecast → Classification3 mapping
//!
//! `lean = clamp(relative_return / scale, -1, 1)` where `relative_return`
//! is the mean point-forecast vs the last price and `scale` is the 80 %
//! band half-width (forecast uncertainty). Probabilities for an UP lean of
//! strength `s = |lean|`: `[1/3 - s/6, 1/3 + s/3, 1/3 - s/6]` (sums to 1;
//! caps at 2/3 — a deliberately modest voter), mirrored for DOWN.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use neoethos_data::{FeatureCellValidity, FeatureFrame};
use neoethos_execution_budget::CpuLease;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction, project_expert_frame};
use crate::forecasting::swarm_impl::SwarmForecaster;
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::feature_input::{MODEL_FEATURE_INPUT_FILE_V1, ModelFeatureInputV1};

const SWARM_PRICE_COLUMN: &str = "quant_close";

/// [`ExpertModel`] adapter for [`SwarmForecaster`]. See the module doc for
/// the last-row-only voting contract.
pub struct SwarmForecasterAdapter {
    artifact_dir: PathBuf,
    feature_columns: Vec<String>,
}

impl SwarmForecasterAdapter {
    pub fn new(artifact_dir: PathBuf) -> Self {
        Self {
            artifact_dir,
            feature_columns: vec![SWARM_PRICE_COLUMN.to_string()],
        }
    }

    fn from_artifact_dir(artifact_dir: &Path) -> Result<Self> {
        let mut adapter = Self::new(artifact_dir.to_path_buf());
        if let Some(parent) = artifact_dir.parent() {
            let input_path = parent.join(MODEL_FEATURE_INPUT_FILE_V1);
            match std::fs::symlink_metadata(&input_path) {
                Ok(_) => {
                    let input = ModelFeatureInputV1::read_from_path(&input_path)?;
                    adapter.feature_columns = vec![input.base_feature_name(SWARM_PRICE_COLUMN)?];
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error).context("inspect swarm model input contract"),
            }
        }
        Ok(adapter)
    }

    /// Map a forecast vs the last price into a modest 3-class lean.
    fn lean_probs(
        last_price: f64,
        result: &crate::forecasting::swarm_impl::SwarmForecastResult,
    ) -> Result<[f64; 3]> {
        if result.point_forecast.is_empty()
            || result.level_80_upper.len() != result.point_forecast.len()
            || result.level_80_lower.len() != result.point_forecast.len()
        {
            bail!("swarm forecast returned inconsistent or empty interval arrays");
        }
        let n = result.point_forecast.len() as f64;
        let mean_forecast = result
            .point_forecast
            .iter()
            .copied()
            .map(f64::from)
            .sum::<f64>()
            / n;
        if !mean_forecast.is_finite() || last_price <= 0.0 {
            bail!("swarm forecast or last price is invalid");
        }
        let rel = (mean_forecast - last_price) / last_price;
        // Uncertainty scale: mean 80% band half-width, relative to price.
        // Wider bands ⇒ larger scale ⇒ smaller lean for the same move.
        let half_widths = result
            .level_80_upper
            .iter()
            .zip(result.level_80_lower.iter())
            .map(|(u, l)| f64::from((u - l).abs()) * 0.5)
            .sum::<f64>()
            / n;
        let scale = (half_widths / last_price).max(1e-6);
        let lean = (rel / scale).clamp(-1.0, 1.0);
        let s = lean.abs();
        if lean >= 0.0 {
            Ok([
                1.0 / 3.0 - s / 6.0,
                1.0 / 3.0 + s / 3.0,
                1.0 / 3.0 - s / 6.0,
            ])
        } else {
            Ok([
                1.0 / 3.0 - s / 6.0,
                1.0 / 3.0 - s / 6.0,
                1.0 / 3.0 + s / 3.0,
            ])
        }
    }
}

impl ExpertModel for SwarmForecasterAdapter {
    fn name(&self) -> &str {
        "swarm_forecaster"
    }
    fn family(&self) -> ModelFamily {
        ModelFamily::Forecasting
    }
    fn output_kind(&self) -> ExpertOutputKind {
        ExpertOutputKind::Classification3
    }
    fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }
    fn predict(&self, frame: &FeatureFrame, lease: &CpuLease) -> Result<Vec<ExpertPrediction>> {
        let projected = project_expert_frame(frame, self.feature_columns(), self.name())?;
        let n_rows = projected.n_samples();
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        let mut out = (0..n_rows)
            .map(|_| {
                ExpertPrediction::invalid(
                    ExpertOutputKind::Classification3,
                    FeatureCellValidity::Warmup,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if n_rows < 32 {
            return Ok(out);
        }
        // A univariate price forecaster consumes the observed price, not its
        // robust z-score. The model view retains exact raw lineage through
        // row/column projections; never invert a clipped normalization fit.
        let column = projected.raw_model_column(&self.feature_columns[0])?;
        if let Some(reason) = column
            .validity
            .iter()
            .copied()
            .find(|reason| !reason.is_valid())
        {
            out[n_rows - 1] = ExpertPrediction::invalid(ExpertOutputKind::Classification3, reason)?;
            return Ok(out);
        }
        let series = column
            .values
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| {
                if !value.is_finite() || value <= 0.0 || value > f32::MAX as f64 {
                    bail!("swarm f64-to-f32 adapter rejected price row {row}: {value}");
                }
                let narrowed = value as f32;
                if !narrowed.is_finite() || narrowed <= 0.0 {
                    bail!("swarm f64-to-f32 adapter produced invalid price row {row}");
                }
                Ok(narrowed)
            })
            .collect::<Result<Vec<_>>>()?;
        let last_price = *column.values.last().expect("non-empty frame checked above");

        // Fresh forecaster per call (stateless): restore the trained
        // configuration, refit on the CURRENT series, forecast.
        let result = lease.scope(|| {
            let mut model = SwarmForecaster::new(256.0);
            model.load(&self.artifact_dir).with_context(|| {
                format!("SwarmForecaster::load({})", self.artifact_dir.display())
            })?;
            let horizon = model.config.horizon.max(1);
            let timestamps = projected
                .timestamps
                .iter()
                .copied()
                .map(|timestamp| timestamp as f64)
                .collect::<Vec<_>>();
            model
                .fit_series(&series, &timestamps, "live")
                .context("swarm refit on the live price series")?;
            model.forecast(horizon).context("swarm forecast")
        })?;

        let probs = Self::lean_probs(last_price, &result)?;
        out[n_rows - 1] =
            ExpertPrediction::valid(ExpertOutputKind::Classification3, probs.to_vec())?;
        Ok(out)
    }
}

/// Loader for [`SwarmForecasterAdapter`]. Validates the artifact exists and
/// is loadable ONCE at ensemble build (fail loud into `degraded`), then the
/// adapter reloads it per prediction (cheap JSON read).
pub struct SwarmForecasterAdapterLoader;

impl ExpertLoader for SwarmForecasterAdapterLoader {
    fn name(&self) -> &str {
        "swarm_forecaster"
    }
    fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
        let mut probe = SwarmForecaster::new(256.0);
        probe
            .load(artifact_dir)
            .with_context(|| format!("SwarmForecaster::load({}) failed", artifact_dir.display()))?;
        Ok(Box::new(SwarmForecasterAdapter::from_artifact_dir(
            artifact_dir,
        )?))
    }
}

/// Register the swarm voter. Called by
/// [`super::bootstrap::build_default_registry`].
pub fn register_swarm_loader(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(SwarmForecasterAdapterLoader))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SwarmTestDirectory(PathBuf);

    impl SwarmTestDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neoethos-swarm-input-{}-{sequence}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for SwarmTestDirectory {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0) {
                eprintln!(
                    "ERROR cleaning owned Swarm fixture {}: {error}",
                    self.0.display()
                );
            }
        }
    }

    #[test]
    fn adapter_identity() {
        let a = SwarmForecasterAdapter::new(PathBuf::from("x"));
        assert_eq!(a.name(), "swarm_forecaster");
        assert_eq!(a.family(), ModelFamily::Forecasting);
        assert_eq!(a.output_kind(), ExpertOutputKind::Classification3);
        assert_eq!(a.feature_columns(), &[SWARM_PRICE_COLUMN.to_string()]);
    }

    #[test]
    fn loader_fails_loud_on_missing_artifact() {
        let dir = std::env::temp_dir().join("neoethos_swarm_adapter_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(SwarmForecasterAdapterLoader.load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lean_probs_sum_to_one_and_stay_bounded() {
        let res = crate::forecasting::swarm_impl::SwarmForecastResult {
            point_forecast: vec![101.0, 102.0],
            level_80_lower: vec![99.0, 99.5],
            level_80_upper: vec![103.0, 104.0],
            diversity_score: 0.5,
            effective_models: 3.0,
            prediction_variance: 0.1,
            models_used: 3,
            runtime_backend_kind: None,
            runtime_mode: None,
            runtime_degraded_reason: None,
        };
        let p = SwarmForecasterAdapter::lean_probs(100.0, &res).expect("valid lean");
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1, got {sum}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(p[1] > p[2], "upward forecast must lean buy");
    }

    #[test]
    fn trained_swarm_reloads_exact_base_alias_and_predicts_raw_prices_from_frozen_views() {
        use neoethos_data::{FeatureBuildControl, FeatureBuildOptions, FeatureColumnF64};
        use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
        use std::sync::Arc;

        let width = WorkerLimit::new(1).unwrap();
        let lease = CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .unwrap();
        for prefixed in [false, true] {
            let name = if prefixed {
                "M1_quant_close"
            } else {
                SWARM_PRICE_COLUMN
            };
            let prices: Vec<f64> = (0..64).map(|row| 1.1 + row as f64 * 0.0001).collect();
            let raw = Arc::new(
                neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns_with_options(
                    neoethos_data::test_fixtures::canonical_test_timestamps(64),
                    vec![
                        FeatureColumnF64::new(
                            name,
                            prices.clone(),
                            vec![FeatureCellValidity::Valid; 64],
                        )
                        .unwrap(),
                    ],
                    FeatureBuildOptions {
                        prefix_base_features: prefixed,
                        normalization_training_rows: Some(0..40),
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let fit = raw
                .fit_normalization(0..40, true, &FeatureBuildControl::default())
                .unwrap();
            let normalized = raw.with_fitted_normalization(&fit).unwrap();
            assert!(
                normalized
                    .feature_column(0)
                    .unwrap()
                    .values
                    .iter()
                    .any(|value| *value < 0.0)
            );
            let input = ModelFeatureInputV1::from_training_frame(
                raw.provenance().bindings()[0].dataset_identity(),
                &normalized,
                Some(raw.timestamps[44]),
                4,
            )
            .unwrap();
            let root = SwarmTestDirectory::new();
            let artifact_dir = root.path().join("swarm_forecaster");
            neoethos_core::storage::json::write_bytes_atomic(
                &root.path().join(MODEL_FEATURE_INPUT_FILE_V1),
                &input.to_json_bytes().unwrap(),
            )
            .unwrap();
            let mut model = SwarmForecaster::new(64.0);
            model.config.horizon = 2;
            model
                .fit_from_frame(&normalized.row_window(0, 40).unwrap(), "EURUSD", &lease)
                .unwrap();
            assert_eq!(
                model.values,
                prices[..40]
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>()
            );
            model.save(&artifact_dir).unwrap();
            let adapter = SwarmForecasterAdapterLoader.load(&artifact_dir).unwrap();
            assert_eq!(adapter.feature_columns(), &[name.to_owned()]);
            let short = adapter
                .predict(&normalized.row_window(26, 48).unwrap(), &lease)
                .unwrap();
            assert_eq!(short.len(), 22);
            assert!(
                short
                    .iter()
                    .all(|prediction| prediction.validity == FeatureCellValidity::Warmup)
            );
            // Identical observed window, different numerical wrapper: neither
            // normalized values nor future rows may become the price series.
            let actual = adapter
                .predict(&normalized.row_window(16, 48).unwrap(), &lease)
                .unwrap();
            let expected = adapter
                .predict(&raw.row_window(16, 48).unwrap(), &lease)
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.kind, expected.kind);
                assert_eq!(actual.validity, expected.validity);
                assert_eq!(
                    actual
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(actual.len(), 32);
            assert!(
                actual[..31]
                    .iter()
                    .all(|prediction| !prediction.validity.is_valid())
            );
            assert!(actual[31].validity.is_valid());
        }
    }

    #[test]
    fn malformed_swarm_input_contract_cannot_fall_back_to_legacy_column() {
        let root = SwarmTestDirectory::new();
        let input_path = root.path().join(MODEL_FEATURE_INPUT_FILE_V1);
        std::fs::write(&input_path, b"{}").unwrap();
        assert!(
            SwarmForecasterAdapter::from_artifact_dir(&root.path().join("swarm_forecaster"))
                .is_err()
        );
    }
}
