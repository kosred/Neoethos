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
//!    on historical/batch frames every row before the last gets the
//!    neutral abstain `[1/3, 1/3, 1/3]`. No fake history, no lookahead.
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

use anyhow::{Context, Result, anyhow, bail};
use polars::prelude::{DataFrame, DataType};

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::forecasting::swarm_impl::SwarmForecaster;
use crate::runtime::capabilities::ModelFamily;

/// Price columns the forecaster can drive from, in preference order.
/// Mirrors the (private) list `SwarmForecaster::fit_from_frame` trains
/// with, so live prediction reads the SAME series family as training.
const PRICE_COLUMNS: &[&str] = &[
    "close",
    "base_close",
    "mid",
    "price",
    "bid",
    "ask",
    "last",
    "target_price",
    "future_close",
    "next_close",
    "close_M1",
    "close_m1",
];

/// [`ExpertModel`] adapter for [`SwarmForecaster`]. See the module doc for
/// the last-row-only voting contract.
pub struct SwarmForecasterAdapter {
    artifact_dir: PathBuf,
    /// Always empty: the ensemble's shared column contract skips experts
    /// with no declared columns; this adapter self-selects its price
    /// column from whatever frame the contract produced (which contains
    /// the close series — training consumed the same frame family).
    feature_columns: Vec<String>,
}

impl SwarmForecasterAdapter {
    pub fn new(artifact_dir: PathBuf) -> Self {
        Self {
            artifact_dir,
            feature_columns: Vec::new(),
        }
    }

    /// Extract the price series (f32) from the frame, preferring the same
    /// columns training used.
    fn price_series(df: &DataFrame) -> Result<Vec<f32>> {
        for name in PRICE_COLUMNS {
            if let Ok(col) = df.column(name) {
                let casted = col
                    .cast(&DataType::Float64)
                    .with_context(|| format!("cast price column {name} to f64"))?;
                let vals = casted
                    .f64()
                    .with_context(|| format!("read price column {name} as f64"))?;
                let mut out = Vec::with_capacity(vals.len());
                for (idx, v) in vals.into_iter().enumerate() {
                    let v = v.ok_or_else(|| {
                        anyhow!("price column {name} has a null at row {idx}")
                    })?;
                    if !v.is_finite() {
                        bail!("price column {name} has non-finite value at row {idx}");
                    }
                    out.push(v as f32);
                }
                return Ok(out);
            }
        }
        bail!(
            "no price column found in the ensemble frame (looked for {:?}) — \
             swarm_forecaster abstains",
            PRICE_COLUMNS
        )
    }

    /// Map a forecast vs the last price into a modest 3-class lean.
    fn lean_probs(last_price: f32, result: &crate::forecasting::swarm_impl::SwarmForecastResult) -> [f32; 3] {
        let n = result.point_forecast.len().max(1) as f32;
        let mean_forecast: f32 = result.point_forecast.iter().sum::<f32>() / n;
        if !mean_forecast.is_finite() || last_price <= 0.0 {
            return [1.0 / 3.0; 3];
        }
        let rel = (mean_forecast - last_price) / last_price;
        // Uncertainty scale: mean 80% band half-width, relative to price.
        // Wider bands ⇒ larger scale ⇒ smaller lean for the same move.
        let half_widths: f32 = result
            .level_80_upper
            .iter()
            .zip(result.level_80_lower.iter())
            .map(|(u, l)| (u - l).abs() * 0.5)
            .sum::<f32>()
            / n;
        let scale = (half_widths / last_price).max(1e-6);
        let lean = (rel / scale).clamp(-1.0, 1.0);
        let s = lean.abs();
        if lean >= 0.0 {
            [1.0 / 3.0 - s / 6.0, 1.0 / 3.0 + s / 3.0, 1.0 / 3.0 - s / 6.0]
        } else {
            [1.0 / 3.0 - s / 6.0, 1.0 / 3.0 - s / 6.0, 1.0 / 3.0 + s / 3.0]
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
    fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
        let n_rows = df.height();
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        let neutral = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        };
        // Rows before the last: honest abstain (see module doc).
        let mut out = vec![neutral.clone(); n_rows];

        let series = Self::price_series(df)?;
        // A forecast needs history to condition on; a 1-row live frame can't
        // feed the snapshot builder (min 8 points) — abstain gracefully.
        if series.len() < 16 {
            return Ok(out);
        }
        let last_price = *series.last().expect("non-empty checked above");

        // Fresh forecaster per call (stateless): restore the trained
        // configuration, refit on the CURRENT series, forecast.
        let mut model = SwarmForecaster::new(256.0);
        model
            .load(&self.artifact_dir)
            .with_context(|| format!("SwarmForecaster::load({})", self.artifact_dir.display()))?;
        let horizon = model.config.horizon.max(1);
        let timestamps: Vec<f64> = (0..series.len()).map(|i| i as f64).collect();
        model
            .fit_series(&series, &timestamps, "live")
            .context("swarm refit on the live price series")?;
        let result = model.forecast(horizon).context("swarm forecast")?;

        let probs = Self::lean_probs(last_price, &result);
        out[n_rows - 1] = ExpertPrediction {
            kind: ExpertOutputKind::Classification3,
            values: probs.to_vec(),
        };
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
        probe.load(artifact_dir).with_context(|| {
            format!("SwarmForecaster::load({}) failed", artifact_dir.display())
        })?;
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
        assert!(a.feature_columns().is_empty());
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
        let p = SwarmForecasterAdapter::lean_probs(100.0, &res);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1, got {sum}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(p[1] > p[2], "upward forecast must lean buy");
    }
}
