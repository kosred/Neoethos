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
        if n_rows < 16 {
            return Ok(out);
        }
        let column = projected.feature_column(0)?;
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
        Ok(Box::new(SwarmForecasterAdapter::new(
            artifact_dir.to_path_buf(),
        )))
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
}
