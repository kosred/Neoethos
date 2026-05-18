//! Gemma-as-expert integration point for the `SoftVotingEnsemble`.
//!
//! Phase G0 — stub + design intent. Real wiring in G6.
//!
//! ## Operator directive
//!
//! > Gemma is just another expert in the ensemble — equal vote,
//! > no veto, no meta-decider role.
//!
//! ## Contract — what `GemmaExpert` will implement in G6
//!
//! `forex_models::ensemble_inference::ExpertModel`:
//!   - name() -> &str    ("gemma_e4b")
//!   - family() -> ModelFamily   (new Llm variant)
//!   - output_kind() -> ExpertOutputKind   (Classification3)
//!   - feature_columns() -> &[String]
//!   - predict(&DataFrame) -> Vec<ExpertPrediction>
//!
//! ## Bar-timeout safeguard
//!
//! If Gemma can't return within `max_inference_latency_ms`, the
//! ensemble proceeds WITHOUT its vote. Never blocks the trading
//! loop.
//!
//! ## Calibration
//!
//! Initial weight = 0.0 ("voice without a vote"). Operator bumps
//! to ~0.10 in the wizard once backtests justify.

use crate::error::GemmaError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GemmaExpertConfig {
    #[serde(default = "default_max_inference_latency_ms")]
    pub max_inference_latency_ms: u64,
    #[serde(default)]
    pub initial_ensemble_weight: f32,
    #[serde(default = "default_expert_name")]
    pub name: String,
}

fn default_max_inference_latency_ms() -> u64 {
    2_000
}
fn default_expert_name() -> String {
    "gemma_e4b".to_string()
}

impl Default for GemmaExpertConfig {
    fn default() -> Self {
        Self {
            max_inference_latency_ms: default_max_inference_latency_ms(),
            initial_ensemble_weight: 0.0,
            name: default_expert_name(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GemmaVoteDirection {
    Long,
    Short,
    Flat,
}

impl GemmaVoteDirection {
    /// Project a (direction, confidence) pair into a 3-class
    /// probability vector `[p_sell, p_neutral, p_buy]` summing
    /// to 1.0. confidence is clamped to [0, 1].
    pub fn into_classification3(self, confidence: f32) -> [f32; 3] {
        let c = confidence.clamp(0.0, 1.0);
        let rest = (1.0 - c) / 2.0;
        match self {
            Self::Short => [c, rest, rest],
            Self::Flat => [rest, c, rest],
            Self::Long => [rest, rest, c],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GemmaRawVote {
    pub direction: GemmaVoteDirection,
    pub confidence: f32,
}

pub trait GemmaExpertInferenceAdapter: Send + Sync {
    fn infer(&self, prompt_payload: &str) -> Result<Option<GemmaRawVote>, GemmaError>;
}

pub struct StubGemmaExpertInferenceAdapter;

impl GemmaExpertInferenceAdapter for StubGemmaExpertInferenceAdapter {
    fn infer(&self, _prompt_payload: &str) -> Result<Option<GemmaRawVote>, GemmaError> {
        Err(GemmaError::pending(
            "G6 GemmaExpert ensemble adapter (forex-models integration)",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaPromptTemplate {
    #[default]
    BarWindowJsonClassification,
    MultiBarRegimeAware,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_operator_directive() {
        let c = GemmaExpertConfig::default();
        assert_eq!(c.max_inference_latency_ms, 2_000);
        assert_eq!(c.initial_ensemble_weight, 0.0);
        assert_eq!(c.name, "gemma_e4b");
    }

    #[test]
    fn config_round_trips_through_json() {
        let c = GemmaExpertConfig::default();
        let s = serde_json::to_string(&c).unwrap();
        let back: GemmaExpertConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn classification3_sums_to_one_for_long() {
        let p = GemmaVoteDirection::Long.into_classification3(0.7);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[2] > p[0] && p[2] > p[1]);
    }

    #[test]
    fn classification3_sums_to_one_for_short() {
        let p = GemmaVoteDirection::Short.into_classification3(0.6);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[0] > p[1] && p[0] > p[2]);
    }

    #[test]
    fn classification3_sums_to_one_for_flat() {
        let p = GemmaVoteDirection::Flat.into_classification3(0.5);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[1] > p[0] && p[1] > p[2]);
    }

    #[test]
    fn classification3_clamps_confidence_high_and_low() {
        // c > 1 clamps to 1 → directed bucket gets all mass.
        let h = GemmaVoteDirection::Long.into_classification3(1.5);
        assert!((h[2] - 1.0).abs() < 1e-5);
        // c < 0 clamps to 0 → directed bucket gets 0, the other
        // two split the remaining 1.0 evenly (NOT uniform).
        let l = GemmaVoteDirection::Long.into_classification3(-0.2);
        assert!((l[0] - 0.5).abs() < 1e-5);
        assert!((l[1] - 0.5).abs() < 1e-5);
        assert!((l[2] - 0.0).abs() < 1e-5);
        let sum: f32 = l.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn raw_vote_round_trips_through_json() {
        let v = GemmaRawVote {
            direction: GemmaVoteDirection::Long,
            confidence: 0.78,
        };
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("\"direction\":\"long\""));
        let back: GemmaRawVote = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn stub_adapter_fails_loud_with_g6_phase_tag() {
        let adapter = StubGemmaExpertInferenceAdapter;
        let err = adapter.infer("anything").expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("G6"));
        assert!(msg.contains("ensemble"));
    }

    #[test]
    fn prompt_template_default_is_bar_window_json() {
        assert_eq!(
            GemmaPromptTemplate::default(),
            GemmaPromptTemplate::BarWindowJsonClassification
        );
    }
}

// ---------------------------------------------------------------------------
// G6a — `GemmaExpert` scaffolding (forex-models dep deferred behind feature)
// ---------------------------------------------------------------------------

/// G6a feature gate. Enabling pulls in `forex-models` and turns
/// [`GemmaExpert`] into a real `ExpertModel`. While the feature
/// is off (default), `GemmaExpert` exposes the same constructor
/// surface but its `predict_classification3` returns a uniform
/// `[1/3, 1/3, 1/3]` vector — voice without a vote.
///
/// Why not just import `forex-models` unconditionally? Two
/// reasons:
/// - `forex-models` pulls candle + Burn + linfa + a hefty
///   compile-time payload. The bare `forex-gemma` crate stays
///   ~10s to compile in CI; with `forex-models` it climbs to
///   minutes.
/// - The two-way dep `forex-gemma → forex-models` is fine but
///   we want the reverse direction (a model adapter pulling
///   into `forex-models` for the registry) to land in a single
///   focused commit, not as a side effect of every G* change.

/// Marker — the cargo feature `ensemble-integration` is OFF by
/// default. Toggling this constant has no effect at runtime;
/// it's a compile-time `cfg` that downstream code can read.
pub const ENSEMBLE_INTEGRATION_FEATURE_NAME: &str = "ensemble-integration";

/// Lightweight Gemma-expert handle that downstream code can hold
/// without pulling in `forex-models`. The G6b commit will land:
///
/// ```ignore
/// #[cfg(feature = "ensemble-integration")]
/// impl forex_models::ensemble_inference::ExpertModel for GemmaExpert { ... }
/// ```
///
/// For now, the struct stores the config + adapter trait object
/// and a frozen list of feature column names so the registry-
/// loader integration path stays testable without the heavy
/// dep tree.
pub struct GemmaExpert {
    config: GemmaExpertConfig,
    adapter: std::sync::Arc<dyn GemmaExpertInferenceAdapter>,
    feature_columns: Vec<String>,
}

impl GemmaExpert {
    /// Build with the operator-approved defaults — name
    /// `gemma_e4b`, initial weight 0.0, 2-sec latency budget.
    pub fn new(adapter: std::sync::Arc<dyn GemmaExpertInferenceAdapter>) -> Self {
        Self {
            config: GemmaExpertConfig::default(),
            adapter,
            feature_columns: Self::default_feature_columns(),
        }
    }

    pub fn with_config(
        config: GemmaExpertConfig,
        adapter: std::sync::Arc<dyn GemmaExpertInferenceAdapter>,
    ) -> Self {
        Self {
            config,
            adapter,
            feature_columns: Self::default_feature_columns(),
        }
    }

    /// Default feature-column list the G6 adapter expects: same
    /// OHLCV-window + indicators the rest of the ensemble's
    /// classification experts consume. The list is the source
    /// of truth for the registry's column-drift detector.
    pub fn default_feature_columns() -> Vec<String> {
        vec![
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "volume".to_string(),
            "rsi_14".to_string(),
            "atr_14".to_string(),
            "ema_20".to_string(),
            "ema_50".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
            "regime_label".to_string(),
        ]
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }
    pub fn config(&self) -> &GemmaExpertConfig {
        &self.config
    }
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }
    pub fn initial_weight(&self) -> f32 {
        self.config.initial_ensemble_weight
    }

    /// Run a one-shot inference. Returns the 3-class
    /// probability vector `[p_sell, p_neutral, p_buy]` ready for
    /// the soft-voter to consume.
    ///
    /// Returns the uniform `[1/3, 1/3, 1/3]` vector when:
    /// - The adapter errors (bar-timeout safeguard).
    /// - The adapter returns `Ok(None)` (LLM output unparseable).
    ///
    /// "Uniform = no vote" means the soft-voter weighting still
    /// works without special-casing missing experts.
    pub fn predict_classification3(&self, prompt_payload: &str) -> [f32; 3] {
        match self.adapter.infer(prompt_payload) {
            Ok(Some(vote)) => vote.direction.into_classification3(vote.confidence),
            Ok(None) | Err(_) => {
                let third = 1.0 / 3.0;
                [third, third, third]
            }
        }
    }
}

#[cfg(test)]
mod expert_wiring_tests {
    use super::*;
    use std::sync::Arc;

    /// Test adapter that returns a canned vote — useful to pin
    /// the projection at the boundary.
    struct FixedVoteAdapter(GemmaVoteDirection, f32);
    impl GemmaExpertInferenceAdapter for FixedVoteAdapter {
        fn infer(&self, _prompt_payload: &str) -> Result<Option<GemmaRawVote>, GemmaError> {
            Ok(Some(GemmaRawVote {
                direction: self.0,
                confidence: self.1,
            }))
        }
    }

    /// Test adapter that returns Ok(None) — simulates the
    /// LLM-output-unparseable path.
    struct NoneAdapter;
    impl GemmaExpertInferenceAdapter for NoneAdapter {
        fn infer(&self, _prompt_payload: &str) -> Result<Option<GemmaRawVote>, GemmaError> {
            Ok(None)
        }
    }

    /// Test adapter that errors — simulates bar-timeout.
    struct ErroringAdapter;
    impl GemmaExpertInferenceAdapter for ErroringAdapter {
        fn infer(&self, _prompt_payload: &str) -> Result<Option<GemmaRawVote>, GemmaError> {
            Err(GemmaError::pending("simulated timeout"))
        }
    }

    fn third_third_third() -> [f32; 3] {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }

    fn close_enough(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn gemma_expert_constructor_carries_defaults() {
        let e = GemmaExpert::new(Arc::new(StubGemmaExpertInferenceAdapter));
        assert_eq!(e.name(), "gemma_e4b");
        assert_eq!(e.initial_weight(), 0.0);
        assert!(e.feature_columns().contains(&"close".to_string()));
        assert!(e.feature_columns().contains(&"regime_label".to_string()));
    }

    #[test]
    fn predict_emits_directed_probs_when_adapter_returns_vote() {
        let e = GemmaExpert::new(Arc::new(FixedVoteAdapter(GemmaVoteDirection::Long, 0.8)));
        let p = e.predict_classification3("any-prompt");
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[2] > p[0] && p[2] > p[1]);
    }

    #[test]
    fn predict_emits_uniform_when_adapter_returns_none() {
        let e = GemmaExpert::new(Arc::new(NoneAdapter));
        assert!(close_enough(
            e.predict_classification3("anything"),
            third_third_third()
        ));
    }

    #[test]
    fn predict_emits_uniform_when_adapter_errors_bar_timeout() {
        // Bar-timeout safeguard: never block, never NaN — just
        // emit a uniform vote that the soft-voter can ingest.
        let e = GemmaExpert::new(Arc::new(ErroringAdapter));
        assert!(close_enough(
            e.predict_classification3("anything"),
            third_third_third()
        ));
    }

    #[test]
    fn feature_columns_match_ensemble_classifier_convention() {
        // OHLCV + indicators — the same shape the LightGBM /
        // XGBoost / TabNet experts consume. The registry's
        // column-drift detector pins these.
        let cols = GemmaExpert::default_feature_columns();
        for required in &["open", "high", "low", "close", "volume"] {
            assert!(
                cols.contains(&(*required).to_string()),
                "missing OHLCV column {required}"
            );
        }
    }

    #[test]
    fn with_config_overrides_default_weight() {
        let mut cfg = GemmaExpertConfig::default();
        cfg.initial_ensemble_weight = 0.10;
        let e = GemmaExpert::with_config(cfg, Arc::new(StubGemmaExpertInferenceAdapter));
        assert!((e.initial_weight() - 0.10).abs() < 1e-5);
    }
}
