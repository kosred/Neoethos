//! Canonical regime-classifier module — operator-approved
//! consolidation per the audit's F-013 / F-048 / F-064 cluster.
//!
//! ## Why this module exists
//!
//! Pre-2026-05-25 the workspace had THREE independent regime
//! classification systems, each with its own thresholds + semantics:
//!
//! | System | Where it lived | What it classified |
//! |--------|----------------|---------------------|
//! | **Feature-bucket** (F-013) | `discovery::validate_regime_robustness` | Rolling-window feature buckets with explicit dead-zones |
//! | **Time-window** (F-048) | `genetic::regime_labels` | Calendar-time chunks (e.g. 30-day windows) labelled by realised return shape |
//! | **ADX/Hurst/EMA cascade** (F-064) | `stop_target::infer_regime` | Live bar-by-bar hard cascade (most rigorous) |
//!
//! Each used different cutoffs (ADX 22 vs 25 vs 28; Hurst 0.55 vs
//! 0.60 vs threshold-pair-by-symbol). Strategies promoted by one
//! system could be rejected by another with no shared vocabulary.
//!
//! ## The new layout (operator-approved doctrine §3 §4)
//!
//! - **`regime/classifier.rs`** — Phase A: re-exports `stop_target::infer_regime`
//!   (F-064 promoted to canonical) under a stable name + adds a
//!   typed `Regime` enum that the other two systems will migrate to.
//! - **`regime/feature_view.rs`** — Phase B: refactor of F-013
//!   feature-bucket logic to consume the canonical classifier and
//!   drop the dead-zone heuristic.
//! - **`regime/time_window.rs`** — Phase B: refactor of F-048
//!   time-window labelling to ALSO call the canonical classifier
//!   per-bar then aggregate. This eliminates the bucket-disagreement
//!   that the audit flagged.
//!
//! Phase A (this commit) is **non-behavioural**: re-exports +
//! typed enum + migration doc. Phase B (follow-up) migrates the
//! two divergent callers to consume the canonical classifier.
//!
//! ## Schema versioning
//!
//! Like `scoring::ScoringVersion`, a `RegimeClassifierVersion`
//! constant pins the current behaviour. When Phase B unifies the
//! callers + bumps `RegimeClassifierVersion` to 2, persisted
//! `DiscoveryRunProfile` artifacts tag which classifier produced
//! their regime labels so old runs are still interpretable.

pub mod classifier;

pub use classifier::{infer_regime_canonical, Regime, RegimeClassifierVersion,
    REGIME_CLASSIFIER_VERSION_CURRENT};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_module_publishes_canonical_classifier_and_version() {
        // Compile-time API pin — if any of these vanishes, this test
        // stops compiling and the audit's F-013/F-048/F-064 closure
        // contract is broken.
        let _v: RegimeClassifierVersion = REGIME_CLASSIFIER_VERSION_CURRENT;
        let _r: fn(f64, f64, f64, f64) -> Regime = infer_regime_canonical;
    }

    #[test]
    fn version_starts_at_one() {
        assert_eq!(REGIME_CLASSIFIER_VERSION_CURRENT.0, 1);
    }
}
