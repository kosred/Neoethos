//! Bridge between the trained-ensemble inference layer
//! ([`forex_models::EnsemblePredictor`]) and the live-bar
//! producer's [`super::auto_trade_producer::ModelPredictor`].
//!
//! Phase D1.3.1. This is the missing-piece glue that ties the
//! whole inference pipeline together:
//!
//! ```text
//!   producer thread
//!     ├─ poll cTrader for newest closed bar
//!     ├─ append to rolling window
//!     ▼
//!   EnsembleModelPredictor::predict(&[bars])    ← THIS module
//!     ├─ convert &[HistoricalBar] → forex_data::Ohlcv
//!     ├─ forex_data::compute_hpc_feature_frame  → FeatureFrame
//!     ├─ FeatureFrame → polars::DataFrame
//!     ├─ ensemble.predict(&df) → Array2<f32> (n_rows × 3)
//!     ├─ take the LAST row (latest-bar prediction)
//!     ├─ argmax → AutoTradeSide (Flat/Buy/Sell)
//!     ├─ confidence = max probability
//!     ▼
//!   PredictionOutput { side, confidence }
//!     ▼
//!   AutoTradeSignal → dispatch_auto_trade_signal (§7.1 gate chain)
//!     ▼
//!   execute_ctrader_order → broker
//! ```
//!
//! ## Column ordering convention
//!
//! `forex_models::base::ExpertModel::predict_proba` documents
//! `Array2<f32>` shape `(N, 3)` with **`[neutral, buy, sell]`**
//! column order. The aggregator [`forex_models::SoftVotingEnsemble`]
//! preserves that ordering through its weighted average. This
//! module maps argmax → [`super::auto_trade::AutoTradeSide`]
//! accordingly:
//!
//! - argmax col 0 (neutral) → `AutoTradeSide::Flat`
//! - argmax col 1 (buy)     → `AutoTradeSide::Buy`
//! - argmax col 2 (sell)    → `AutoTradeSide::Sell`
//!
//! ## Warmup
//!
//! The feature builder needs at least ~100 bars before its
//! longer-window indicators (Hurst, regime, ATR-21) stop emitting
//! NaN tails. We use **200 bars** as the producer's
//! `warmup_bars()` floor so the latest-row prediction is reliably
//! non-degenerate. Below the warmup the producer's run loop
//! short-circuits to Flat — see
//! [`super::auto_trade_producer::LiveInferenceProducer::run_predict`].

use std::sync::Arc;

use anyhow::{Context, Result};
use forex_data::{FeatureFrame, FeatureProfile, Ohlcv, compute_hpc_feature_frame};
use forex_models::EnsemblePredictor;
use polars::prelude::{Column, DataFrame, NamedFrom, Series};

use super::HistoricalBar;
use super::auto_trade::AutoTradeSide;
use super::auto_trade_producer::{ModelPredictor, PredictionOutput};

/// Minimum rolling-window length the ensemble adapter requires
/// before it will emit a non-Flat prediction. 200 bars covers the
/// longest feature lookback (Hurst exponent at 100, plus a safety
/// margin for NaN-tail spillover) while still being a comfortable
/// fit in the producer's default 512-bar rolling window.
pub const ENSEMBLE_PREDICTOR_WARMUP_BARS: usize = 200;

/// Bridge predictor that runs a trained [`EnsemblePredictor`]
/// (SoftVotingEnsemble, MoeEnsemble, …) against the live rolling
/// window of bars and returns one [`PredictionOutput`] for the
/// LATEST bar — the only row the producer cares about per tick.
pub struct EnsembleModelPredictor {
    ensemble: Arc<dyn EnsemblePredictor>,
    feature_profile: FeatureProfile,
    warmup_bars: usize,
}

impl EnsembleModelPredictor {
    /// Wrap an [`EnsemblePredictor`] with the default feature
    /// profile (`Standard`) and the documented 200-bar warmup.
    pub fn new(ensemble: Arc<dyn EnsemblePredictor>) -> Self {
        Self {
            ensemble,
            feature_profile: FeatureProfile::Standard,
            warmup_bars: ENSEMBLE_PREDICTOR_WARMUP_BARS,
        }
    }

    /// Override the feature profile (e.g. `Full` for HPC training
    /// runs that include extra microstructure columns). Returns
    /// `self` for builder-style chaining.
    ///
    /// `#[allow(dead_code)]` 2026-05-21: AUTO ON (Task #7) calls
    /// `new()` with the default profile today; the builder chain
    /// stays public for the operator-config UI that's queued behind
    /// Task #30 (AppState split). Drop the allow once the chain is
    /// wired from the Settings tab.
    #[allow(dead_code)]
    pub fn with_feature_profile(mut self, profile: FeatureProfile) -> Self {
        self.feature_profile = profile;
        self
    }

    /// Override the warmup bars floor. Operators with extreme
    /// configurations (e.g. shorter-lookback feature subsets) may
    /// lower this. Below 50 the bar-feature builder cannot produce
    /// stable indicator values; the validator floors at 50.
    #[allow(dead_code)] // see `with_feature_profile` above
    pub fn with_warmup_bars(mut self, bars: usize) -> Self {
        self.warmup_bars = bars.max(50);
        self
    }

    /// Read-only handle to the inner ensemble — useful for
    /// rendering the operator-facing "X/Y experts active" banner
    /// without re-querying the registry.
    #[allow(dead_code)] // chrome banner widget not yet wired —
                        // status_msg string carries the count today
    pub fn ensemble(&self) -> &dyn EnsemblePredictor {
        self.ensemble.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a chronologically-ordered slice of [`HistoricalBar`]
/// into a [`forex_data::Ohlcv`] suitable for the feature builder.
fn bars_to_ohlcv(bars: &[HistoricalBar]) -> Ohlcv {
    let mut timestamps = Vec::with_capacity(bars.len());
    let mut open = Vec::with_capacity(bars.len());
    let mut high = Vec::with_capacity(bars.len());
    let mut low = Vec::with_capacity(bars.len());
    let mut close = Vec::with_capacity(bars.len());
    let mut volume = Vec::with_capacity(bars.len());
    for bar in bars {
        timestamps.push(bar.timestamp_ms);
        open.push(bar.open);
        high.push(bar.high);
        low.push(bar.low);
        close.push(bar.close);
        // HistoricalBar.volume is Option<i64>; the feature builder
        // wants Option<Vec<f64>>. We collect every bar's volume as
        // an f64 and emit `Some(vec)` iff at least one bar carried
        // a real volume reading.
        volume.push(bar.volume.unwrap_or(0) as f64);
    }
    Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

/// Convert a [`FeatureFrame`] into a polars [`DataFrame`]. Each
/// named column carries the corresponding column of
/// `frame.data` as an `f32` Series. The conversion mirrors
/// `forex_models::parallel_trainer::TrainingPayload::from_named_dense`
/// so the runtime DataFrame layout matches what the experts saw at
/// fit time (column names + ordering).
fn feature_frame_to_dataframe(frame: &FeatureFrame) -> Result<DataFrame> {
    if frame.names.len() != frame.data.ncols() {
        anyhow::bail!(
            "FeatureFrame name/column mismatch: {} names vs {} columns",
            frame.names.len(),
            frame.data.ncols()
        );
    }
    let columns: Vec<Column> = frame
        .names
        .iter()
        .enumerate()
        .map(|(col_idx, name)| {
            let values: Vec<f32> = frame.data.column(col_idx).iter().copied().collect();
            Column::from(Series::new(name.as_str().into(), values))
        })
        .collect();
    DataFrame::new(columns).context("build DataFrame from FeatureFrame")
}

/// Map the trained-ensemble's column ordering `[neutral, buy, sell]`
/// into a [`PredictionOutput`] by argmax.
fn row_to_prediction(row: [f32; 3]) -> PredictionOutput {
    let mut max_idx = 0usize;
    let mut max_val = row[0];
    for (i, v) in row.iter().enumerate().skip(1) {
        if *v > max_val {
            max_val = *v;
            max_idx = i;
        }
    }
    let side = match max_idx {
        0 => AutoTradeSide::Flat,
        1 => AutoTradeSide::Buy,
        2 => AutoTradeSide::Sell,
        _ => unreachable!("3-class argmax index out of range"),
    };
    PredictionOutput {
        side,
        confidence: max_val.clamp(0.0, 1.0) as f64,
    }
}

// ---------------------------------------------------------------------------
// ModelPredictor impl
// ---------------------------------------------------------------------------

impl ModelPredictor for EnsembleModelPredictor {
    fn warmup_bars(&self) -> usize {
        self.warmup_bars
    }

    fn predict(&self, bars: &[HistoricalBar]) -> Result<PredictionOutput> {
        if bars.len() < self.warmup_bars {
            // The producer's run_predict already short-circuits to
            // Flat when bars.len() < warmup_bars, but a redundant
            // guard here costs nothing and protects callers that
            // don't honour the warmup contract.
            return Ok(PredictionOutput {
                side: AutoTradeSide::Flat,
                confidence: 0.0,
            });
        }
        let ohlcv = bars_to_ohlcv(bars);
        let feature_frame = compute_hpc_feature_frame(&ohlcv, self.feature_profile)
            .with_context(|| "compute_hpc_feature_frame failed on live rolling window")?;
        let df = feature_frame_to_dataframe(&feature_frame)?;
        let probs = self
            .ensemble
            .predict(&df)
            .with_context(|| "EnsemblePredictor::predict failed")?;
        if probs.nrows() == 0 || probs.ncols() != 3 {
            anyhow::bail!(
                "EnsemblePredictor returned unexpected shape {:?}",
                probs.shape()
            );
        }
        // The LATEST bar's prediction is on the LAST row of the
        // result matrix.
        let last = probs.nrows() - 1;
        let row = [probs[(last, 0)], probs[(last, 1)], probs[(last, 2)]];
        Ok(row_to_prediction(row))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use forex_models::SoftVotingEnsemble;
    use forex_models::ensemble_inference::{
        ExpertLoadOutcome, ExpertModel, ExpertOutputKind, ExpertPrediction,
    };
    use forex_models::runtime::capabilities::ModelFamily;

    fn bar(timestamp_ms: i64, close: f64) -> HistoricalBar {
        HistoricalBar {
            timestamp_ms,
            open: close,
            high: close + 0.01,
            low: close - 0.01,
            close,
            volume: Some(1),
        }
    }

    /// Constant-output expert used to drive the bridge predict()
    /// path with a known prediction.
    struct ConstantExpert {
        probs: [f32; 3],
    }

    impl ExpertModel for ConstantExpert {
        fn name(&self) -> &str {
            "test_constant"
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Tree
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(&self, df: &polars::prelude::DataFrame) -> Result<Vec<ExpertPrediction>> {
            Ok((0..df.height())
                .map(|_| ExpertPrediction {
                    kind: ExpertOutputKind::Classification3,
                    values: self.probs.to_vec(),
                })
                .collect())
        }
    }

    fn ensemble_with_constant(probs: [f32; 3]) -> Arc<dyn EnsemblePredictor> {
        let outcome = ExpertLoadOutcome {
            loaded: vec![Box::new(ConstantExpert { probs })],
            missing: vec![],
            degraded: vec![],
        };
        Arc::new(SoftVotingEnsemble::with_default_config(outcome).expect("soft voting"))
    }

    fn synthetic_bars(n: usize) -> Vec<HistoricalBar> {
        // Slight upward drift so the feature builder produces
        // non-degenerate indicator values.
        (0..n)
            .map(|i| bar(i as i64 * 60_000, 1.0 + (i as f64) * 0.0001))
            .collect()
    }

    #[test]
    fn warmup_bars_default_matches_constant() {
        let ens = ensemble_with_constant([0.5, 0.3, 0.2]);
        let predictor = EnsembleModelPredictor::new(ens);
        assert_eq!(predictor.warmup_bars(), ENSEMBLE_PREDICTOR_WARMUP_BARS);
    }

    #[test]
    fn with_warmup_bars_clamps_to_minimum_50() {
        let ens = ensemble_with_constant([0.5, 0.3, 0.2]);
        let predictor = EnsembleModelPredictor::new(ens).with_warmup_bars(10);
        assert_eq!(predictor.warmup_bars(), 50);
    }

    #[test]
    fn predict_returns_flat_during_warmup() {
        let ens = ensemble_with_constant([0.1, 0.8, 0.1]); // Buy if computed
        let predictor = EnsembleModelPredictor::new(ens);
        // Below warmup → Flat regardless of expert output.
        let bars = synthetic_bars(50);
        let out = predictor.predict(&bars).expect("predict");
        assert_eq!(out.side, AutoTradeSide::Flat);
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn predict_returns_buy_when_buy_probability_dominant() {
        let ens = ensemble_with_constant([0.05, 0.85, 0.10]);
        let predictor = EnsembleModelPredictor::new(ens);
        let bars = synthetic_bars(250);
        let out = predictor.predict(&bars).expect("predict");
        assert_eq!(out.side, AutoTradeSide::Buy);
        assert!((out.confidence - 0.85).abs() < 1e-5);
    }

    #[test]
    fn predict_returns_sell_when_sell_probability_dominant() {
        let ens = ensemble_with_constant([0.05, 0.10, 0.85]);
        let predictor = EnsembleModelPredictor::new(ens);
        let bars = synthetic_bars(250);
        let out = predictor.predict(&bars).expect("predict");
        assert_eq!(out.side, AutoTradeSide::Sell);
        assert!((out.confidence - 0.85).abs() < 1e-5);
    }

    #[test]
    fn predict_returns_flat_when_neutral_probability_dominant() {
        let ens = ensemble_with_constant([0.7, 0.15, 0.15]);
        let predictor = EnsembleModelPredictor::new(ens);
        let bars = synthetic_bars(250);
        let out = predictor.predict(&bars).expect("predict");
        assert_eq!(out.side, AutoTradeSide::Flat);
        assert!((out.confidence - 0.7).abs() < 1e-5);
    }

    #[test]
    fn row_to_prediction_argmax_invariants() {
        // Pin the column-order convention [neutral, buy, sell].
        assert_eq!(row_to_prediction([0.6, 0.2, 0.2]).side, AutoTradeSide::Flat);
        assert_eq!(row_to_prediction([0.2, 0.6, 0.2]).side, AutoTradeSide::Buy);
        assert_eq!(row_to_prediction([0.2, 0.2, 0.6]).side, AutoTradeSide::Sell);
    }

    #[test]
    fn bars_to_ohlcv_preserves_chronological_order() {
        let bars = vec![bar(1_000, 1.0), bar(2_000, 1.1), bar(3_000, 1.2)];
        let ohlcv = bars_to_ohlcv(&bars);
        assert_eq!(
            ohlcv.timestamp.as_ref().unwrap(),
            &vec![1_000, 2_000, 3_000]
        );
        assert_eq!(ohlcv.close, vec![1.0, 1.1, 1.2]);
        assert_eq!(ohlcv.volume.as_ref().unwrap(), &vec![1.0, 1.0, 1.0]);
    }
}
