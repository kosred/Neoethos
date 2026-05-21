//! Embedding-based topic gate (Layer 2.2) and the
//! `SessionWatchdog` (Layer 2.5).
//!
//! Phase G2 lands the production candle-backed
//! `MultilingualE5Provider` behind the `gate-embedding` cargo
//! feature; this module ships the trait surface + a
//! deterministic `FakeEmbeddingProvider` (for tests) + the full
//! `EmbeddingGate` scoring + post-filter + session-watchdog
//! pipeline that the runtime stitches into `TopicGateStack`.
//!
//! ## Architecture
//!
//! ```text
//!   user text ─► EmbeddingProvider (candle / fake) ─► Vec<f32>
//!                       │
//!                       ▼
//!   AnchorCorpus ─► EmbeddingGate::score
//!                       │
//!                       ▼
//!   margin = max(cos(text, in_scope_i))
//!          − max(cos(text, out_of_scope_j))
//!                       │
//!                       ▼
//!     thresholds  ►  Allow / SoftWarning / Refuse
//!                       │
//!                       ▼
//!   SessionWatchdog tightens thresholds when soft refusals
//!   accumulate inside the last K turns
//! ```
//!
//! ## Why dependency injection
//!
//! The real candle path needs ~470 MB of model weights and a
//! significant compile-time investment. By isolating the
//! provider behind a trait we keep:
//!
//! - Unit tests fast (FakeEmbeddingProvider returns 8-dim
//!   deterministic vectors based on character hashes).
//! - The gate logic + the watchdog testable in isolation.
//! - The real candle wiring (G2.1, behind `gate-embedding`)
//!   shippable as a small follow-up.

use crate::anchors::{AnchorCorpus, AnchorSentence};
use crate::config::TopicGateConfig;
use crate::error::GemmaError;
use crate::gate::{LanguageHint, TopicCheck, TopicGate, refusal_text};
use std::sync::Mutex;

/// Dimensions a `FakeEmbeddingProvider` emits. Small enough to
/// keep tests trivially fast; the real `MultilingualE5Provider`
/// emits 384-dim vectors.
pub const FAKE_EMBEDDING_DIM: usize = 8;

/// Provider trait — converts text into a fixed-size embedding
/// vector. The real production impl is candle-backed
/// (`gate-embedding` feature, lands in G2.1); the fake is
/// deterministic and used in tests + when the embedding feature
/// is off.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single string. Returns a vector with `dim()`
    /// elements. Implementations SHOULD normalise to unit length
    /// so cosine similarity reduces to a dot product.
    fn embed(&self, text: &str) -> Result<Vec<f32>, GemmaError>;
    /// Dimensionality of returned vectors. Stable across calls.
    fn dim(&self) -> usize;
}

/// Deterministic fake provider. Produces unit vectors from a
/// stable per-character mixing routine. Anchor sentences and
/// queries that share substrings get high cosine similarity;
/// completely different text gets low similarity. Good enough
/// to exercise the scoring + thresholding logic without
/// loading a real model.
pub struct FakeEmbeddingProvider;

impl FakeEmbeddingProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FakeEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for FakeEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, GemmaError> {
        Ok(fake_embed(text, FAKE_EMBEDDING_DIM))
    }
    fn dim(&self) -> usize {
        FAKE_EMBEDDING_DIM
    }
}

/// Pure helper — turn a string into a deterministic unit vector
/// of the given dimension. Each character contributes to one
/// component via a simple character-class hash; the result is
/// L2-normalised. Exposed `pub` so tests can construct expected
/// embeddings directly.
pub fn fake_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    if dim == 0 {
        return v;
    }
    // Token-aware bucketing: lowercase ASCII letters and Greek
    // letters get reproducible buckets so words like
    // "position" / "θέση" stay similar across calls.
    for ch in text.chars() {
        let bucket = match ch {
            'a'..='z' => (ch as u32 - 'a' as u32) as usize % dim,
            'A'..='Z' => (ch as u32 - 'A' as u32) as usize % dim,
            'α'..='ω' => ((ch as u32 - 'α' as u32) as usize + 3) % dim,
            'Α'..='Ω' => ((ch as u32 - 'Α' as u32) as usize + 3) % dim,
            '0'..='9' => ((ch as u32 - '0' as u32) as usize + 5) % dim,
            _ => (ch as u32 as usize) % dim,
        };
        v[bucket] += 1.0;
    }
    // L2 normalise so cosine similarity == dot product.
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Cosine similarity between two equal-length unit-vector
/// embeddings. With L2-normalised vectors this reduces to a
/// dot product, but we compute defensively in case a caller
/// passes a non-normalised vector.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Precomputed anchor embeddings — held inside `EmbeddingGate`
/// after the first `score` call (lazily) or inline at
/// construction time via `EmbeddingGate::with_corpus`.
struct EmbeddedAnchors {
    in_scope: Vec<(AnchorSentence, Vec<f32>)>,
    out_of_scope: Vec<(AnchorSentence, Vec<f32>)>,
}

impl EmbeddedAnchors {
    fn from_corpus(
        corpus: &AnchorCorpus,
        provider: &dyn EmbeddingProvider,
    ) -> Result<Self, GemmaError> {
        let mut in_scope = Vec::with_capacity(corpus.in_scope.len());
        for s in &corpus.in_scope {
            let v = provider.embed(&s.text)?;
            in_scope.push((s.clone(), v));
        }
        let mut out_of_scope = Vec::with_capacity(corpus.out_of_scope.len());
        for s in &corpus.out_of_scope {
            let v = provider.embed(&s.text)?;
            out_of_scope.push((s.clone(), v));
        }
        Ok(Self {
            in_scope,
            out_of_scope,
        })
    }

    fn max_similarity(&self, query: &[f32], in_scope: bool) -> f32 {
        let pool = if in_scope {
            &self.in_scope
        } else {
            &self.out_of_scope
        };
        pool.iter()
            .map(|(_, v)| cosine_similarity(query, v))
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Layer 2.2 — embedding-similarity topic gate.
///
/// Holds a provider + a corpus + thresholds, and produces a
/// `TopicCheck` for each input/output. Cloneable handles wrap
/// the inner state in `Arc` indirectly through ownership at
/// the runtime layer; this struct itself is single-instance.
pub struct EmbeddingGate {
    provider: Box<dyn EmbeddingProvider>,
    anchors: EmbeddedAnchors,
    config: TopicGateConfig,
}

impl EmbeddingGate {
    /// Build from a provider + corpus + config. Pre-computes
    /// anchor embeddings up front so per-turn work is just one
    /// embedding call + 2N cosine similarities.
    pub fn new(
        provider: Box<dyn EmbeddingProvider>,
        corpus: &AnchorCorpus,
        config: TopicGateConfig,
    ) -> Result<Self, GemmaError> {
        let anchors = EmbeddedAnchors::from_corpus(corpus, &*provider)?;
        Ok(Self {
            provider,
            anchors,
            config,
        })
    }

    /// G0 / G2 default — fake provider + placeholder corpus +
    /// default thresholds. Useful for tests and for the
    /// `gate-embedding` feature OFF build path (where the real
    /// candle provider isn't available).
    pub fn with_g2_defaults() -> Result<Self, GemmaError> {
        Self::new(
            Box::new(FakeEmbeddingProvider::new()),
            &AnchorCorpus::g0_placeholder(),
            TopicGateConfig::default(),
        )
    }

    /// Compute the in-scope / out-of-scope similarity margin
    /// for a single text. Exposed `pub` so calibration tools
    /// can drive it directly.
    ///
    /// `margin = max(cos(text, in_scope_i))
    ///         - max(cos(text, out_of_scope_j))`.
    pub fn similarity_margin(&self, text: &str) -> Result<f64, GemmaError> {
        let q = self.provider.embed(text)?;
        let in_s = self.anchors.max_similarity(&q, true);
        let out_s = self.anchors.max_similarity(&q, false);
        Ok((in_s - out_s) as f64)
    }

    /// Apply the configured thresholds to a margin value.
    /// Pure function — testable in isolation.
    pub fn classify_margin(&self, margin: f64) -> TopicCheck {
        if margin < self.config.in_scope_threshold {
            TopicCheck::Refuse {
                reason: format!(
                    "embedding gate: margin {margin:.3} < {:.3} (out-of-scope)",
                    self.config.in_scope_threshold
                ),
                canned_response: refusal_text(LanguageHint::Unknown),
            }
        } else if margin < self.config.soft_warning_threshold {
            TopicCheck::SoftWarning {
                reason: format!(
                    "embedding gate: margin {margin:.3} below soft threshold {:.3}",
                    self.config.soft_warning_threshold
                ),
            }
        } else {
            TopicCheck::Allow
        }
    }
}

impl TopicGate for EmbeddingGate {
    fn check_input(&self, text: &str, language: LanguageHint) -> TopicCheck {
        if !self.config.embedding_gate_enabled {
            return TopicCheck::Allow;
        }
        let margin = match self.similarity_margin(text) {
            Ok(m) => m,
            Err(_) => {
                // If the embedder errored we don't pretend it
                // passed — fail closed (refuse) and pin the
                // language so the canned response matches the
                // user's locale.
                return TopicCheck::Refuse {
                    reason: "embedding gate: provider error".to_string(),
                    canned_response: refusal_text(language),
                };
            }
        };
        let mut verdict = self.classify_margin(margin);
        // Re-route refusal canned text to the right language.
        if let TopicCheck::Refuse {
            canned_response, ..
        } = &mut verdict
        {
            *canned_response = refusal_text(language);
        }
        verdict
    }

    fn check_output(&self, text: &str) -> TopicCheck {
        if !self.config.post_filter_enabled || !self.config.embedding_gate_enabled {
            return TopicCheck::Allow;
        }
        let margin = match self.similarity_margin(text) {
            Ok(m) => m,
            Err(_) => return TopicCheck::Allow, // benign-fail on output side
        };
        self.classify_margin(margin)
    }
}

// ---------------------------------------------------------------------------
// Layer 2.5 — Session watchdog
// ---------------------------------------------------------------------------

/// Tracks the last N gate verdicts for a chat session and
/// reports when the conversation is drifting (soft refusals
/// piling up). The runtime queries `should_tighten` before each
/// turn and lowers the gate thresholds when the watchdog says
/// yes.
///
/// Thread-safe — multiple turns from the same session can
/// arrive on different threads via the SSE bridge.
pub struct SessionWatchdog {
    inner: Mutex<WatchdogState>,
    window_size: usize,
    soft_refusal_trigger: usize,
}

struct WatchdogState {
    last_verdicts: Vec<WatchdogVerdict>,
}

/// Compact form of `TopicCheck` we actually need to track —
/// stores only the discriminant, not the strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogVerdict {
    Allow,
    SoftWarning,
    Refuse,
}

impl From<&TopicCheck> for WatchdogVerdict {
    fn from(c: &TopicCheck) -> Self {
        match c {
            TopicCheck::Allow => Self::Allow,
            TopicCheck::SoftWarning { .. } => Self::SoftWarning,
            TopicCheck::Refuse { .. } => Self::Refuse,
        }
    }
}

impl SessionWatchdog {
    /// Default window: last 10 turns; tighten when 3+ of them
    /// were soft refusals OR hard refusals.
    pub const DEFAULT_WINDOW: usize = 10;
    pub const DEFAULT_SOFT_REFUSAL_TRIGGER: usize = 3;

    pub fn new() -> Self {
        Self::with_window(Self::DEFAULT_WINDOW, Self::DEFAULT_SOFT_REFUSAL_TRIGGER)
    }

    pub fn with_window(window_size: usize, soft_refusal_trigger: usize) -> Self {
        Self {
            inner: Mutex::new(WatchdogState {
                last_verdicts: Vec::with_capacity(window_size.max(1)),
            }),
            window_size: window_size.max(1),
            soft_refusal_trigger: soft_refusal_trigger.max(1),
        }
    }

    /// Record a verdict from the most recent turn. The window
    /// is FIFO; the oldest verdict drops off when full.
    pub fn record(&self, verdict: WatchdogVerdict) {
        let mut g = self.inner.lock().expect("watchdog mutex poisoned");
        if g.last_verdicts.len() == self.window_size {
            g.last_verdicts.remove(0);
        }
        g.last_verdicts.push(verdict);
    }

    /// Number of soft-warning OR refusal verdicts in the window.
    pub fn flagged_count(&self) -> usize {
        self.inner
            .lock()
            .expect("watchdog mutex poisoned")
            .last_verdicts
            .iter()
            .filter(|v| !matches!(v, WatchdogVerdict::Allow))
            .count()
    }

    /// `true` when the runtime should tighten the embedding-gate
    /// thresholds for the rest of the session.
    pub fn should_tighten(&self) -> bool {
        self.flagged_count() >= self.soft_refusal_trigger
    }

    /// Clear the rolling window — useful when the operator
    /// recycles the chat session or after a manual reset.
    pub fn reset(&self) {
        self.inner
            .lock()
            .expect("watchdog mutex poisoned")
            .last_verdicts
            .clear();
    }
}

impl Default for SessionWatchdog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchors::AnchorCorpus;

    #[test]
    fn fake_embed_is_unit_norm() {
        let v = fake_embed("Show my positions", FAKE_EMBEDDING_DIM);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fake_embed_is_deterministic() {
        let v1 = fake_embed("EUR/USD", 8);
        let v2 = fake_embed("EUR/USD", 8);
        assert_eq!(v1, v2);
    }

    #[test]
    fn fake_embed_different_inputs_produce_different_vectors() {
        let v1 = fake_embed("Show my positions", 8);
        let v2 = fake_embed("Tell me a joke", 8);
        assert_ne!(v1, v2);
    }

    #[test]
    fn cosine_similarity_of_identical_unit_vectors_is_one() {
        let v = fake_embed("hello", 8);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_handles_empty_inputs() {
        assert_eq!(cosine_similarity(&[], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0); // mismatched dim
    }

    #[test]
    fn embedded_anchors_construction_from_placeholder_corpus() {
        let provider = FakeEmbeddingProvider::new();
        let corpus = AnchorCorpus::g0_placeholder();
        let anchors = EmbeddedAnchors::from_corpus(&corpus, &provider).unwrap();
        assert_eq!(anchors.in_scope.len(), corpus.in_scope.len());
        assert_eq!(anchors.out_of_scope.len(), corpus.out_of_scope.len());
    }

    #[test]
    fn embedding_gate_with_g2_defaults_constructs_cleanly() {
        let _gate = EmbeddingGate::with_g2_defaults().expect("ok");
    }

    #[test]
    fn embedding_gate_classify_margin_thresholds() {
        let gate = EmbeddingGate::with_g2_defaults().unwrap();
        // Default thresholds: in_scope = -0.05, soft_warning = 0.15
        assert!(matches!(
            gate.classify_margin(-0.5),
            TopicCheck::Refuse { .. }
        ));
        assert!(matches!(
            gate.classify_margin(0.05),
            TopicCheck::SoftWarning { .. }
        ));
        assert!(matches!(gate.classify_margin(0.25), TopicCheck::Allow));
    }

    #[test]
    fn embedding_gate_disabled_via_config_always_allows() {
        let mut cfg = TopicGateConfig::default();
        cfg.embedding_gate_enabled = false;
        let gate = EmbeddingGate::new(
            Box::new(FakeEmbeddingProvider::new()),
            &AnchorCorpus::g0_placeholder(),
            cfg,
        )
        .unwrap();
        // Even a trash query passes when the gate is off.
        assert_eq!(
            gate.check_input("Tell me a joke", LanguageHint::English),
            TopicCheck::Allow
        );
    }

    #[test]
    fn embedding_gate_check_input_uses_language_for_refusal_text() {
        // Force a refusal via threshold manipulation.
        let mut cfg = TopicGateConfig::default();
        cfg.in_scope_threshold = 1.0; // impossibly strict
        cfg.soft_warning_threshold = 1.0;
        let gate = EmbeddingGate::new(
            Box::new(FakeEmbeddingProvider::new()),
            &AnchorCorpus::g0_placeholder(),
            cfg,
        )
        .unwrap();
        let v = gate.check_input("Random text here", LanguageHint::Greek);
        if let TopicCheck::Refuse {
            canned_response, ..
        } = v
        {
            assert!(canned_response.contains("Μπορώ"));
        } else {
            panic!("expected refuse");
        }
    }

    #[test]
    fn embedding_gate_post_filter_can_be_disabled() {
        let mut cfg = TopicGateConfig::default();
        cfg.in_scope_threshold = 1.0;
        cfg.soft_warning_threshold = 1.0;
        cfg.post_filter_enabled = false;
        let gate = EmbeddingGate::new(
            Box::new(FakeEmbeddingProvider::new()),
            &AnchorCorpus::g0_placeholder(),
            cfg,
        )
        .unwrap();
        // Output check returns Allow when post_filter is off,
        // even with impossibly strict thresholds.
        assert_eq!(gate.check_output("anything"), TopicCheck::Allow);
    }

    #[test]
    fn watchdog_starts_quiet_and_records_verdicts() {
        let w = SessionWatchdog::with_window(5, 2);
        assert_eq!(w.flagged_count(), 0);
        assert!(!w.should_tighten());
        w.record(WatchdogVerdict::Allow);
        w.record(WatchdogVerdict::SoftWarning);
        assert_eq!(w.flagged_count(), 1);
        assert!(!w.should_tighten());
    }

    #[test]
    fn watchdog_tightens_when_soft_refusals_pile_up() {
        let w = SessionWatchdog::with_window(10, 3);
        w.record(WatchdogVerdict::SoftWarning);
        w.record(WatchdogVerdict::Allow);
        w.record(WatchdogVerdict::SoftWarning);
        assert!(!w.should_tighten());
        w.record(WatchdogVerdict::Refuse);
        assert!(w.should_tighten());
    }

    #[test]
    fn watchdog_window_drops_oldest_when_full() {
        let w = SessionWatchdog::with_window(3, 2);
        w.record(WatchdogVerdict::SoftWarning);
        w.record(WatchdogVerdict::SoftWarning);
        w.record(WatchdogVerdict::SoftWarning);
        assert_eq!(w.flagged_count(), 3);
        assert!(w.should_tighten());
        // Adding an Allow pushes the oldest SoftWarning out.
        w.record(WatchdogVerdict::Allow);
        assert_eq!(w.flagged_count(), 2);
    }

    #[test]
    fn watchdog_reset_clears_window() {
        let w = SessionWatchdog::new();
        w.record(WatchdogVerdict::Refuse);
        assert_eq!(w.flagged_count(), 1);
        w.reset();
        assert_eq!(w.flagged_count(), 0);
    }

    #[test]
    fn watchdog_verdict_from_topic_check_maps_discriminant() {
        let allow: WatchdogVerdict = (&TopicCheck::Allow).into();
        assert_eq!(allow, WatchdogVerdict::Allow);
        let soft: WatchdogVerdict = (&TopicCheck::SoftWarning {
            reason: "x".to_string(),
        })
            .into();
        assert_eq!(soft, WatchdogVerdict::SoftWarning);
        let refuse: WatchdogVerdict = (&TopicCheck::Refuse {
            reason: "x".to_string(),
            canned_response: "y".to_string(),
        })
            .into();
        assert_eq!(refuse, WatchdogVerdict::Refuse);
    }

    #[test]
    fn watchdog_window_size_clamped_to_minimum_one() {
        // A zero window would deadlock the ring buffer logic;
        // clamp to 1 defensively.
        let w = SessionWatchdog::with_window(0, 0);
        w.record(WatchdogVerdict::Refuse);
        assert_eq!(w.flagged_count(), 1);
    }
}
