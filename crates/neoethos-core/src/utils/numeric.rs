//! Shared numeric guards and activation helpers.
//!
//! Phase 65 extraction: previously `finite_or` lived in
//! `neoethos-search::genetic::diversity` and
//! `neoethos-search::genetic::regime_labels`; `clamp_probability` /
//! `clamp_unit` lived in three neoethos-models files; and
//! `sigmoid` had a naive copy in `ensemble.rs` and a
//! numerically-stable copy in `bayesian_impl.rs`. They now live here.

/// Returns `value` when finite, otherwise `fallback`. Use this at the
/// boundary between caller-supplied (potentially NaN / ±∞) data and
/// downstream code that expects a usable number.
pub fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

/// `f32` variant of [`finite_or`] for ML/feature paths.
pub fn finite_or_f32(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Clamp `value` into `[0.0, 1.0]`. Use this whenever a probability
/// crosses a noisy boundary (post-sigmoid, post-softmax, etc.).
pub fn clamp_unit_f32(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

/// `f64` variant of [`clamp_unit_f32`].
pub fn clamp_unit_f64(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

/// Numerically-stable logistic sigmoid for `f32`. The split branch
/// avoids overflow when `value` is large in magnitude (e.g. logits at
/// the tail of a model's output distribution).
pub fn stable_sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_or_passes_through_finite() {
        assert_eq!(finite_or(1.5, 0.0), 1.5);
        assert_eq!(finite_or(f64::NAN, 9.0), 9.0);
        assert_eq!(finite_or(f64::INFINITY, -1.0), -1.0);
    }

    #[test]
    fn clamp_unit_f32_handles_nan_and_bounds() {
        assert_eq!(clamp_unit_f32(0.5), 0.5);
        assert_eq!(clamp_unit_f32(1.5), 1.0);
        assert_eq!(clamp_unit_f32(-0.5), 0.0);
        assert_eq!(clamp_unit_f32(f32::NAN), 0.0);
    }

    #[test]
    fn stable_sigmoid_avoids_overflow_at_extremes() {
        // Naive `1.0 / (1.0 + (-x).exp())` overflows for very negative x.
        assert!((stable_sigmoid_f32(0.0) - 0.5).abs() < 1e-6);
        assert!(stable_sigmoid_f32(50.0) > 0.999);
        assert!(stable_sigmoid_f32(-50.0) < 0.001);
        assert!(stable_sigmoid_f32(f32::INFINITY).is_finite());
        assert!(stable_sigmoid_f32(f32::NEG_INFINITY).is_finite());
    }
}
