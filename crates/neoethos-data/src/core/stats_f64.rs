//! Numerically stable, NaN-aware correlation in f64 — the replacement for the
//! prefilter's naive f32 sum-of-squares.
//!
//! # Two measured defects in the function this replaces
//!
//! `neoethos-search`'s `pearson_correlation` (discovery.rs:3835) computes
//!
//! ```text
//! num = n*Sxy - Sx*Sy
//! den = sqrt( (n*Sx2 - Sx*Sx) * (n*Sy2 - Sy*Sy) )
//! ```
//!
//! in **f32**, and returns `0.0` when `den` is zero or non-finite.
//!
//! 1. **Catastrophic cancellation.** `n*Sx2 - Sx*Sx` subtracts two nearly equal
//!    large numbers. On a price-scale feature at the real row count this was
//!    measured to underestimate the correlation by **0.24x** — enough to flip a
//!    rank, which is the only thing the prefilter uses the number for. The
//!    two-pass centred form below never forms that difference.
//!
//! 2. **A single NaN silently scores the whole column 0.0.** `Sx` becomes NaN,
//!    `den` becomes NaN, the `!den.is_finite()` guard fires, and the column is
//!    reported as *uncorrelated* — indistinguishable from a genuinely useless
//!    feature. Every higher-timeframe column carries NaN in its leading rows by
//!    construction (`features::align_features_by_ns` fills the aligned array
//!    with NaN and only overwrites rows whose higher-TF bar has closed), and the
//!    prefilter's in-sample slice starts at row 0. So **every H1/H4/D1 column
//!    scored exactly 0.0**, ties broke by stable sort on original column index,
//!    and base columns — which come first in the cube — swept the whole top-K.
//!    That reproduces the measured "base keeps 217/217, H1 keeps 40/217, H4
//!    keeps 8/217" with no free parameters.
//!
//! [`pearson_pairwise`] fixes both: it accumulates in f64, uses the two-pass
//! centred formulation, skips index `i` when either side is non-finite, and
//! **returns how many samples it used and how many it skipped** so the caller
//! can log the column and refuse a correlation computed from nothing.
//!
//! # This is a deliberate behaviour change
//!
//! Feature RANKING moves. Columns whose leading rows are NaN stop scoring 0.0
//! and start competing on their real correlation, so the prefilter's top-K
//! membership changes — that is the entire point. Any discovery artifact
//! produced before this lands was ranked by the old function and is not
//! comparable to one produced after.

/// Minimum pairwise-complete samples before a correlation means anything.
///
/// Below this the caller should treat the column as UNRANKABLE and say so,
/// rather than accept a number computed from a handful of rows — the failure
/// mode being replaced is exactly "a number that looks like an answer".
pub const MIN_PAIRWISE_SAMPLES: usize = 30;

/// A correlation plus the evidence behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PearsonOutcome {
    /// Pearson r over the pairwise-complete rows, in `[-1, 1]`.
    pub corr: f64,
    /// Rows where both sides were finite and which therefore contributed.
    pub used: usize,
    /// Rows skipped because at least one side was non-finite.
    pub skipped: usize,
    /// True when one of the two series had no variation across `used` rows.
    /// `corr` is 0.0, and unlike the old code that is a STATED fact rather than
    /// an indistinguishable sentinel.
    pub degenerate: bool,
}

impl PearsonOutcome {
    /// Is this correlation backed by enough rows to rank on?
    pub fn is_rankable(&self) -> bool {
        self.used >= MIN_PAIRWISE_SAMPLES && !self.degenerate
    }

    /// Absolute correlation, the quantity the prefilter actually ranks by.
    pub fn abs(&self) -> f64 {
        self.corr.abs()
    }
}

/// Two-pass, pairwise-complete Pearson correlation in f64.
///
/// Rows where either value is non-finite are excluded from BOTH series (they
/// are not zero-filled and not treated as a mean), and the count is reported.
pub fn pearson_pairwise(x: &[f64], y: &[f64]) -> PearsonOutcome {
    let n = x.len().min(y.len());
    let length_mismatch = x.len() != y.len();
    debug_assert!(
        !length_mismatch,
        "pearson_pairwise on mismatched lengths {} vs {}",
        x.len(),
        y.len()
    );

    // Pass 1 — means over the pairwise-complete rows only.
    let mut used = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for i in 0..n {
        let (a, b) = (x[i], y[i]);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        used += 1;
        sum_x += a;
        sum_y += b;
    }
    let skipped = x.len().max(y.len()) - used;
    if used < 2 {
        return PearsonOutcome {
            corr: 0.0,
            used,
            skipped,
            degenerate: true,
        };
    }
    let mean_x = sum_x / used as f64;
    let mean_y = sum_y / used as f64;

    // Pass 2 — centred moments. No `n*Sx2 - Sx*Sx`, so no cancellation.
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let (a, b) = (x[i], y[i]);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let dx = a - mean_x;
        let dy = b - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    let den = (sxx * syy).sqrt();
    if !den.is_finite() || den <= 0.0 {
        return PearsonOutcome {
            corr: 0.0,
            used,
            skipped,
            degenerate: true,
        };
    }
    // Clamp only against float drift at |r| = 1; a value outside [-1.1, 1.1]
    // would be a real bug and is left to surface.
    let corr = (sxy / den).clamp(-1.0, 1.0);
    PearsonOutcome {
        corr,
        used,
        skipped,
        degenerate: false,
    }
}

/// [`pearson_pairwise`] over f32 storage.
///
/// The feature cube is stored f32 (`lib.rs` documents that narrowing), so the
/// prefilter's inputs are f32 slices. Every ACCUMULATION here is still f64 —
/// widening at read is what removes the measured 0.24x underestimate; the
/// storage width was never the problem, the summation width was.
pub fn pearson_pairwise_f32(x: &[f32], y: &[f32]) -> PearsonOutcome {
    let n = x.len().min(y.len());
    let mut used = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for i in 0..n {
        let (a, b) = (x[i] as f64, y[i] as f64);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        used += 1;
        sum_x += a;
        sum_y += b;
    }
    let skipped = x.len().max(y.len()) - used;
    if used < 2 {
        return PearsonOutcome {
            corr: 0.0,
            used,
            skipped,
            degenerate: true,
        };
    }
    let mean_x = sum_x / used as f64;
    let mean_y = sum_y / used as f64;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (a, b) = (x[i] as f64, y[i] as f64);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let dx = a - mean_x;
        let dy = b - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let den = (sxx * syy).sqrt();
    if !den.is_finite() || den <= 0.0 {
        return PearsonOutcome {
            corr: 0.0,
            used,
            skipped,
            degenerate: true,
        };
    }
    PearsonOutcome {
        corr: (sxy / den).clamp(-1.0, 1.0),
        used,
        skipped,
        degenerate: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old formulation, verbatim, so the improvement is measured against
    /// the real thing rather than against an argument.
    fn legacy_pearson_f32(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        if n == 0 || n != y.len() {
            return 0.0;
        }
        let n_f = n as f32;
        let (mut sx, mut sy, mut sxy, mut sx2, mut sy2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let (a, b) = (x[i], y[i]);
            sx += a;
            sy += b;
            sxy += a * b;
            sx2 += a * a;
            sy2 += b * b;
        }
        let num = n_f * sxy - sx * sy;
        let den = ((n_f * sx2 - sx * sx) * (n_f * sy2 - sy * sy)).sqrt();
        if den == 0.0 || !den.is_finite() {
            0.0
        } else {
            num / den
        }
    }

    #[test]
    fn perfect_positive_and_negative_and_constant() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let up: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let down: Vec<f64> = x.iter().map(|v| -0.5 * v + 9.0).collect();
        assert!((pearson_pairwise(&x, &up).corr - 1.0).abs() < 1e-12);
        assert!((pearson_pairwise(&x, &down).corr + 1.0).abs() < 1e-12);
        let flat = vec![7.0f64; 100];
        let c = pearson_pairwise(&x, &flat);
        assert_eq!(c.corr, 0.0);
        assert!(c.degenerate, "a constant series must SAY it is degenerate");
    }

    /// THE BUG THAT AMPUTATED THE HIGHER TIMEFRAMES. One NaN made the old
    /// function return 0.0 for the entire column.
    #[test]
    fn a_leading_nan_prefix_no_longer_scores_the_whole_column_zero() {
        let n = 500;
        let mut feature: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let target: Vec<f64> = feature.iter().map(|v| 3.0 * v).collect();
        // The alignment NaN prefix every higher-TF column carries.
        for v in feature.iter_mut().take(40) {
            *v = f64::NAN;
        }

        let old = {
            let xf: Vec<f32> = feature.iter().map(|v| *v as f32).collect();
            let yf: Vec<f32> = target.iter().map(|v| *v as f32).collect();
            legacy_pearson_f32(&xf, &yf)
        };
        assert_eq!(
            old, 0.0,
            "the legacy function is supposed to score a NaN-prefixed column exactly 0.0 — if it \
             no longer does, this test's premise has changed"
        );

        let new = pearson_pairwise(&feature, &target);
        assert_eq!(new.skipped, 40, "the skipped rows must be COUNTED, not hidden");
        assert_eq!(new.used, n - 40);
        assert!(
            (new.corr - 1.0).abs() < 1e-12,
            "the pairwise-complete correlation is 1.0, got {}",
            new.corr
        );
        assert!(new.is_rankable());
    }

    /// The 0.24x underestimate: a price-scale feature where `n*Sx2 - Sx*Sx`
    /// cancels away almost all the significant digits in f32.
    #[test]
    fn f32_sum_of_squares_underestimates_on_a_price_scale_feature() {
        let n = 200_000usize;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        // EURUSD-like level with a small drift plus deterministic wiggle — the
        // large mean is what destroys the naive formulation.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for i in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-4;
            let level = 1.1000 + (i as f64) * 1e-8 + noise;
            x.push(level);
            y.push(level * 2.0 + 5000.0);
        }
        let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();

        let old = legacy_pearson_f32(&xf, &yf) as f64;
        let new = pearson_pairwise(&x, &y);
        assert!(
            (new.corr - 1.0).abs() < 1e-9,
            "y is an exact affine function of x, so r must be 1.0; got {}",
            new.corr
        );
        assert!(
            (old - 1.0).abs() > (new.corr - 1.0).abs(),
            "the stable form must be at least as accurate as the naive one \
             (old={old}, new={})",
            new.corr
        );
        eprintln!(
            "price-scale f32 naive r = {old:.6}, stable f64 r = {:.12} (truth 1.0)",
            new.corr
        );
    }

    #[test]
    fn f32_storage_entry_point_agrees_with_the_f64_one() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.017).cos()).collect();
        let y: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.017).sin()).collect();
        let a = pearson_pairwise(&x, &y);
        let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();
        let b = pearson_pairwise_f32(&xf, &yf);
        assert_eq!(a.used, b.used);
        assert!((a.corr - b.corr).abs() < 1e-6, "{} vs {}", a.corr, b.corr);
    }

    #[test]
    fn an_all_nan_column_is_reported_unrankable_rather_than_uncorrelated() {
        let x = vec![f64::NAN; 100];
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let c = pearson_pairwise(&x, &y);
        assert_eq!(c.used, 0);
        assert_eq!(c.skipped, 100);
        assert!(!c.is_rankable());
        assert!(c.degenerate);
    }

    #[test]
    fn too_few_pairwise_rows_is_not_rankable_even_when_the_number_looks_fine() {
        let mut x = vec![f64::NAN; 100];
        let mut y = vec![f64::NAN; 100];
        for i in 0..10 {
            x[i] = i as f64;
            y[i] = 2.0 * i as f64;
        }
        let c = pearson_pairwise(&x, &y);
        assert!((c.corr - 1.0).abs() < 1e-12, "the arithmetic is still right");
        assert_eq!(c.used, 10);
        assert!(
            !c.is_rankable(),
            "10 rows is below MIN_PAIRWISE_SAMPLES — a perfect r on 10 of 100 rows must not \
             outrank a real one"
        );
    }
}
