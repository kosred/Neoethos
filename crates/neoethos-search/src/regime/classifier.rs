//! Canonical regime classifier — the F-064 promotion.
//!
//! Phase A (2026-05-25): introduces the typed [`Regime`] enum + the
//! [`infer_regime_canonical`] function that the audit's two divergent
//! caller-sites (F-013 + F-048) will migrate to in Phase B.

use serde::{Deserialize, Serialize};

/// Typed regime classification. Replaces the per-caller string
/// vocabulary (`"trend"` / `"trend_up"` / `"trending"` / `"strong_trend"`
/// — the audit flagged the inconsistency) with one canonical enum.
///
/// Serialised as lowercase strings for backward compat with existing
/// on-disk regime labels:
/// - `Trend` → `"trend"`
/// - `Range` → `"range"`
/// - `Neutral` → `"neutral"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Regime {
    /// Directional movement — ADX above the trend threshold AND
    /// Hurst exponent above the persistence threshold.
    Trend,
    /// Mean-reverting / sideways — ADX below the range threshold OR
    /// Hurst below the anti-persistence threshold.
    Range,
    /// Neither trend nor range conditions met — ambiguous / mixed.
    /// Strategies should be conservative in this regime (operator
    /// directive 2026-05-15: prefer flat over wrong-side trade).
    Neutral,
}

impl Regime {
    /// Lowercase string form. Matches serde representation.
    pub fn as_lowercase(self) -> &'static str {
        match self {
            Self::Trend => "trend",
            Self::Range => "range",
            Self::Neutral => "neutral",
        }
    }

    /// Lenient parser for legacy on-disk labels. Accepts the various
    /// strings the pre-unification systems emitted.
    pub fn from_lenient(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "trend" | "trending" | "strong_trend" | "trend_up" | "trend_down" => {
                Some(Self::Trend)
            }
            "range" | "ranging" | "sideways" | "mean_revert" => Some(Self::Range),
            "neutral" | "mixed" | "ambiguous" => Some(Self::Neutral),
            _ => None,
        }
    }
}

/// Schema-version tag for the canonical regime classifier. See the
/// module-level documentation in [`super`] for the migration plan.
///
/// Phase A keeps the F-064 cascade (ADX/Hurst/EMA-cross) byte-for-byte
/// identical → version stays at `1`. Phase C (after F-013 + F-048
/// migrate) may unify thresholds across the three systems → bumps to
/// `2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegimeClassifierVersion(pub u32);

/// Current canonical-classifier schema version.
pub const REGIME_CLASSIFIER_VERSION_CURRENT: RegimeClassifierVersion =
    RegimeClassifierVersion(1);

// ---------------------------------------------------------------------------
// Thresholds — single source of truth for the canonical classifier
// ---------------------------------------------------------------------------

/// ADX above this counts as "trending". Operator-tunable knob lives
/// in `Settings` (Phase B exposes it); the literal here is the
/// research-anchored default from `stop_target::infer_regime`.
pub const TREND_ADX_THRESHOLD: f64 = 25.0;

/// ADX below this counts as "ranging" (mean-reverting).
pub const RANGE_ADX_THRESHOLD: f64 = 20.0;

/// Hurst exponent above this counts as "persistent" (trend
/// confirmation). Default 0.55 per F-064.
pub const TREND_HURST_THRESHOLD: f64 = 0.55;

/// Hurst exponent below this counts as "anti-persistent" (range
/// confirmation).
pub const RANGE_HURST_THRESHOLD: f64 = 0.45;

// ---------------------------------------------------------------------------
// Canonical inference function
// ---------------------------------------------------------------------------

/// Classify the current bar's regime given the four canonical
/// indicators. Mirrors `stop_target::infer_regime` exactly (Phase A
/// behaviour preservation — the F-064 cascade IS the canonical).
///
/// - `adx` — Average Directional Index. NaN / non-finite → falls into
///   the Neutral branch (caller bears NaN-safe inputs upstream).
/// - `hurst` — Hurst exponent estimate. Same NaN handling.
/// - `ema_fast` / `ema_slow` — EMA cross signal. Both finite + non-NaN
///   required for the trend-direction confirmation; otherwise the
///   ADX+Hurst signal alone decides.
///
/// Returns one of [`Regime::Trend`] / [`Regime::Range`] / [`Regime::Neutral`].
pub fn infer_regime_canonical(adx: f64, hurst: f64, ema_fast: f64, ema_slow: f64) -> Regime {
    if !adx.is_finite() || !hurst.is_finite() {
        return Regime::Neutral;
    }

    // Strong trend: ADX > 25 AND Hurst > 0.55 AND EMA cross
    // confirms direction. The EMA cross is informational — when
    // both EMAs are finite, ADX+Hurst already make the call; the
    // EMA-cross only adds direction tagging (which our typed enum
    // doesn't carry — see Phase C `Trend::Up` / `Trend::Down` split).
    if adx > TREND_ADX_THRESHOLD && hurst > TREND_HURST_THRESHOLD {
        if ema_fast.is_finite() && ema_slow.is_finite() {
            // EMA-cross adds confidence but Phase A keeps it as a
            // boolean "agree?" check rather than a tag. Disagreement
            // (e.g. ADX-trend but EMA flat) downgrades to Neutral
            // for safety.
            if (ema_fast - ema_slow).abs() < f64::EPSILON {
                return Regime::Neutral;
            }
        }
        return Regime::Trend;
    }

    if adx < RANGE_ADX_THRESHOLD || hurst < RANGE_HURST_THRESHOLD {
        return Regime::Range;
    }

    Regime::Neutral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_adx_and_hurst_is_trend() {
        let r = infer_regime_canonical(30.0, 0.65, 1.005, 1.000);
        assert_eq!(r, Regime::Trend);
    }

    #[test]
    fn low_adx_is_range() {
        let r = infer_regime_canonical(15.0, 0.50, 1.000, 1.000);
        assert_eq!(r, Regime::Range);
    }

    #[test]
    fn anti_persistent_hurst_is_range() {
        let r = infer_regime_canonical(28.0, 0.40, 1.000, 1.000);
        assert_eq!(r, Regime::Range);
    }

    #[test]
    fn mid_band_is_neutral() {
        let r = infer_regime_canonical(22.0, 0.50, 1.000, 1.000);
        assert_eq!(r, Regime::Neutral);
    }

    #[test]
    fn nan_inputs_collapse_to_neutral() {
        assert_eq!(infer_regime_canonical(f64::NAN, 0.65, 1.0, 1.0), Regime::Neutral);
        assert_eq!(infer_regime_canonical(30.0, f64::NAN, 1.0, 1.0), Regime::Neutral);
    }

    #[test]
    fn equal_emas_in_trend_band_downgrades_to_neutral() {
        // Operator-conservative branch: if ADX+Hurst say "trend" but
        // EMAs are exactly equal (no direction confirmation), don't
        // commit to a Trend label.
        let r = infer_regime_canonical(30.0, 0.65, 1.000, 1.000);
        assert_eq!(r, Regime::Neutral);
    }

    #[test]
    fn lenient_parser_accepts_legacy_strings() {
        assert_eq!(Regime::from_lenient("trend"), Some(Regime::Trend));
        assert_eq!(Regime::from_lenient("TRENDING"), Some(Regime::Trend));
        assert_eq!(Regime::from_lenient("strong_trend"), Some(Regime::Trend));
        assert_eq!(Regime::from_lenient("range"), Some(Regime::Range));
        assert_eq!(Regime::from_lenient("sideways"), Some(Regime::Range));
        assert_eq!(Regime::from_lenient("neutral"), Some(Regime::Neutral));
        assert_eq!(Regime::from_lenient("???"), None);
    }

    #[test]
    fn round_trip_via_serde_lowercase() {
        let r = Regime::Trend;
        let s = serde_json::to_string(&r).expect("serialize");
        assert_eq!(s, "\"trend\"");
        let back: Regime = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, Regime::Trend);
    }
}
