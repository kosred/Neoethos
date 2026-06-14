use crate::core::feature_registry::{
    FeatureColumnMetadata, feature_metadata_for_names, validate_feature_names,
};
use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FeatureProfile {
    #[default]
    Standard,
    Full,
    HPC,
    Adaptive,
}

impl FromStr for FeatureProfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "full" => Ok(Self::Full),
            "hpc" => Ok(Self::HPC),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(format!("unknown feature profile: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBuildOptions {
    pub profile: FeatureProfile,
    pub include_smc: bool,
    pub include_hpc_ta: bool,
    pub include_regime: bool,
    pub include_quant: bool,
    pub prefix_base_features: bool,
    pub higher_tfs: Vec<String>,
}

impl Default for FeatureBuildOptions {
    fn default() -> Self {
        Self {
            profile: FeatureProfile::Standard,
            include_smc: true,
            include_hpc_ta: true,
            include_regime: true,
            include_quant: true,
            prefix_base_features: false,
            higher_tfs: Vec::new(),
        }
    }
}

/// Backing storage for a [`FeatureFrame`]'s feature matrix.
///
/// Small frames (validation-fold windows, prefiltered sub-frames, per-TF
/// blocks, tests) stay in RAM as a `[samples × features]` `Array2`. The big
/// multi-resolution discovery frame is held out-of-core in a feature-major
/// mmap ([`crate::core::feature_store::FeatureStore`]) so the GA reads only
/// the feature rows it references instead of materialising the full (~100 GB
/// for M1) `[samples × features]` matrix AND its `[features × samples]`
/// transpose in RAM.
#[derive(Debug, Clone)]
pub enum FeatureData {
    InMemory(Array2<f32>),
    Mmap(std::sync::Arc<crate::core::feature_store::FeatureStore>),
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub timestamps: Vec<i64>,
    pub names: Vec<String>,
    pub data: FeatureData,
}

impl FeatureFrame {
    pub fn column_metadata(&self) -> Result<Vec<FeatureColumnMetadata>> {
        feature_metadata_for_names(&self.names)
    }

    pub fn validate_registry(&self) -> Result<()> {
        validate_feature_names(&self.names)
    }

    /// Build an in-RAM frame from a `[samples × features]` matrix.
    pub fn from_array(timestamps: Vec<i64>, names: Vec<String>, data: Array2<f32>) -> Self {
        Self {
            timestamps,
            names,
            data: FeatureData::InMemory(data),
        }
    }

    // ── Out-of-core access layer ──────────────────────────────────────────
    //
    // Accessors abstract over the physical backing so call sites never touch a
    // dense matrix directly. `InMemory` serves small frames from a
    // `[samples × features]` `Array2`; `Mmap` serves the big discovery frame
    // from a feature-major mmap without ever materialising the full matrix.

    /// Number of time samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        match &self.data {
            FeatureData::InMemory(a) => a.nrows(),
            FeatureData::Mmap(s) => s.n_samples(),
        }
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_features(&self) -> usize {
        match &self.data {
            FeatureData::InMemory(a) => a.ncols(),
            FeatureData::Mmap(s) => s.n_features(),
        }
    }

    /// Feature `idx`'s full time series (`[samples]`).
    #[inline]
    pub fn feature_column(&self, idx: usize) -> ndarray::ArrayView1<'_, f32> {
        match &self.data {
            FeatureData::InMemory(a) => a.column(idx),
            FeatureData::Mmap(s) => ndarray::ArrayView1::from(s.feature_row(idx)),
        }
    }

    /// Owned `[(end-start) × n_features]` sample-window sub-matrix (all
    /// features over a contiguous time slice) — small, used by folds/regime.
    pub fn sample_window(&self, start: usize, end: usize) -> Array2<f32> {
        match &self.data {
            FeatureData::InMemory(a) => a.slice(ndarray::s![start..end, ..]).to_owned(),
            FeatureData::Mmap(s) => {
                let rows = end - start;
                let ncols = s.n_features();
                let mut out = Array2::zeros((rows, ncols));
                for f in 0..ncols {
                    out.column_mut(f)
                        .assign(&ndarray::ArrayView1::from(&s.feature_row(f)[start..end]));
                }
                out
            }
        }
    }

    /// Single feature value at logical `(sample, feature)`.
    #[inline]
    pub fn feature_at(&self, sample: usize, feature: usize) -> f32 {
        match &self.data {
            FeatureData::InMemory(a) => a[(sample, feature)],
            FeatureData::Mmap(s) => s.feature_row(feature)[sample],
        }
    }

    /// Total number of feature values (`n_samples * n_features`).
    #[inline]
    pub fn n_values(&self) -> usize {
        self.n_samples() * self.n_features()
    }

    /// `[features × samples]` view — the GA eval's `indicators` layout. The
    /// mmap backing yields this natively (contiguous, zero-copy); the in-RAM
    /// backing yields a (small, strided) transposed view.
    pub fn as_indicators_view(&self) -> ndarray::ArrayView2<'_, f32> {
        match &self.data {
            FeatureData::InMemory(a) => a.t(),
            FeatureData::Mmap(s) => s.as_view(),
        }
    }

    /// Iterate every feature value (order-independent; used by NaN/zero/min/max
    /// diagnostics that only need aggregate stats).
    pub fn iter_values(&self) -> Box<dyn Iterator<Item = f32> + '_> {
        match &self.data {
            FeatureData::InMemory(a) => Box::new(a.iter().copied()),
            FeatureData::Mmap(s) => {
                Box::new((0..s.n_features()).flat_map(move |f| s.feature_row(f).iter().copied()))
            }
        }
    }

    /// Materialise the full `[samples × features]` matrix in RAM. In-memory
    /// frames clone; mmap frames reconstruct from the feature rows.
    ///
    /// WARNING: allocates `n_samples * n_features * 4` bytes — for the full M1
    /// discovery cube that is ~13-32 GB. Only call where the dense matrix is
    /// genuinely required (e.g. ML training on a bounded dataset), NEVER on the
    /// big discovery frame in the GA path (which reads feature rows on demand).
    pub fn to_dense_samples_major(&self) -> Array2<f32> {
        match &self.data {
            FeatureData::InMemory(a) => a.clone(),
            FeatureData::Mmap(s) => {
                let (nf, ns) = (s.n_features(), s.n_samples());
                let mut out = Array2::zeros((ns, nf));
                for f in 0..nf {
                    out.column_mut(f)
                        .assign(&ndarray::ArrayView1::from(s.feature_row(f)));
                }
                out
            }
        }
    }
}

/// Align a higher-timeframe feature matrix onto the base-timeframe
/// timestamp grid via as-of join (binary-search style forward scan).
///
/// `ffill` controls behaviour when a base timestamp falls strictly
/// between two higher-TF bars: with `ffill = true` the most-recent
/// prior higher-TF row is forwarded; with `ffill = false` only exact
/// timestamp matches survive.
///
/// **F-308 (2026-05-28) — max_age_ns parameter**: when `ffill = true`,
/// previous behaviour silently propagated the LAST higher-TF row to
/// every subsequent base row forever, even when the higher-TF source
/// had ended weeks or months before the base. A stale D1 (last bar 2
/// months ago) on a fresh M1 grid → every M1 bar from those 2 months
/// got the SAME stale close/high/low/RSI/etc., feeding a frozen-
/// constant column into the indicator stack. GA candidates would then
/// see indicator outputs that don't change over the held-out window,
/// produce zero or look-alike signals, and the discovery funnel would
/// report `ranked=N, post_passes_filter=0` with no diagnostic.
///
/// `max_age_ns` (Some) caps the forward-fill: if the chosen previous
/// higher-TF timestamp is more than `max_age_ns` older than the
/// current base timestamp, the row is left NaN. NaN propagates
/// through the downstream NaN counter (`discovery.rs::feature_cube_summary`)
/// so the operator sees explicit "stale higher-TF" warnings instead
/// of silent zero-trade GA output.
///
/// `max_age_ns = None` preserves the legacy unbounded behaviour for
/// callers that explicitly want it (e.g. UI chart preview where
/// indicators on yesterday's last-known close are fine).
pub fn align_features_by_ns(
    base_ns: &[i64],
    feature_ns: &[i64],
    feature_data: &Array2<f32>,
    ffill: bool,
    max_age_ns: Option<i64>,
) -> Array2<f32> {
    let n_base = base_ns.len();
    let n_feat = feature_ns.len();
    let n_cols = feature_data.ncols();
    let mut out = Array2::from_elem((n_base, n_cols), f32::NAN);

    if n_feat == 0 {
        return out;
    }

    let mut feat_idx = 0usize;
    for i in 0..n_base {
        let ts = base_ns[i];
        while feat_idx < n_feat && feature_ns[feat_idx] <= ts {
            feat_idx += 1;
        }

        let best_idx = if feat_idx > 0 {
            let prev = feat_idx - 1;
            if feature_ns[prev] == ts {
                Some(prev)
            } else if ffill {
                // F-308 max-age guard: drop the forward-fill when the
                // most-recent higher-TF bar is older than the cap.
                match max_age_ns {
                    Some(max_age) if ts - feature_ns[prev] > max_age => None,
                    _ => Some(prev),
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(idx) = best_idx {
            for j in 0..n_cols {
                out[(i, j)] = feature_data[(idx, j)];
            }
        }
    }
    out
}

#[cfg(test)]
mod align_tests {
    use super::*;
    use ndarray::array;

    fn ns_grid(start_min: i64, step_min: i64, n: usize) -> Vec<i64> {
        (0..n as i64)
            .map(|i| (start_min + i * step_min) * 60 * 1_000_000_000)
            .collect()
    }

    #[test]
    fn align_unbounded_forward_fills_to_end() {
        // Legacy behaviour preserved when max_age = None.
        let base_ns = ns_grid(0, 1, 10);   // M1 × 10 bars
        let feat_ns = ns_grid(0, 5, 2);    // M5 × 2 bars: t=0, t=5
        let feat_data = array![[1.0_f32], [2.0_f32]];
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, None);
        // Without max_age, every base bar past t=5 keeps value 2.0.
        assert_eq!(aligned[(0, 0)], 1.0); // t=0
        assert_eq!(aligned[(4, 0)], 1.0); // t=4 (before first M5 close at 5)
        assert_eq!(aligned[(5, 0)], 2.0); // t=5
        assert_eq!(aligned[(9, 0)], 2.0); // t=9 — frozen, what F-308 calls the bug
    }

    #[test]
    fn align_max_age_caps_stale_forward_fill() {
        // F-308 fix: max_age = 3 minutes (in ns) drops values past 3 min lag.
        let base_ns = ns_grid(0, 1, 10);
        let feat_ns = ns_grid(0, 5, 2);
        let feat_data = array![[1.0_f32], [2.0_f32]];
        let max_age_ns = Some(3_i64 * 60 * 1_000_000_000);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns);
        // t=0 → exact, 1.0
        assert_eq!(aligned[(0, 0)], 1.0);
        // t=1,2,3 → within 3min of t=0, still ffill to 1.0
        assert_eq!(aligned[(3, 0)], 1.0);
        // t=4 → 4 min after t=0, EXCEEDS max_age → NaN
        assert!(aligned[(4, 0)].is_nan(), "expected NaN at t=4, got {}", aligned[(4, 0)]);
        // t=5 → exact match on second feat row, value 2.0
        assert_eq!(aligned[(5, 0)], 2.0);
        // t=6,7,8 → within 3min of t=5, ffill 2.0
        assert_eq!(aligned[(8, 0)], 2.0);
        // t=9 → 4 min after t=5, exceeds → NaN. This is what kills the
        // frozen-constant downstream propagation in the F-308 scenario.
        assert!(aligned[(9, 0)].is_nan(), "expected NaN at t=9, got {}", aligned[(9, 0)]);
    }

    #[test]
    fn align_max_age_zero_preserves_exact_matches() {
        // Edge case: max_age = 0 forbids any forward-fill, only exact ts hits.
        let base_ns = ns_grid(0, 1, 5);
        let feat_ns = ns_grid(0, 5, 1); // single feat row at t=0
        let feat_data = array![[42.0_f32]];
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, Some(0));
        assert_eq!(aligned[(0, 0)], 42.0); // exact match
        for i in 1..5 {
            assert!(aligned[(i, 0)].is_nan(), "expected NaN at i={i}");
        }
    }

    #[test]
    fn align_max_age_with_ffill_false_is_consistent() {
        // When ffill is false, max_age has no effect — only exact matches.
        let base_ns = ns_grid(0, 1, 5);
        let feat_ns = ns_grid(0, 5, 1);
        let feat_data = array![[7.0_f32]];
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, false, Some(i64::MAX));
        assert_eq!(aligned[(0, 0)], 7.0);
        for i in 1..5 {
            assert!(aligned[(i, 0)].is_nan());
        }
    }

    #[test]
    fn align_empty_feature_ns_returns_all_nan() {
        let base_ns = ns_grid(0, 1, 5);
        let feat_ns: Vec<i64> = Vec::new();
        let feat_data: Array2<f32> = Array2::zeros((0, 2));
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, Some(60_000_000_000));
        assert_eq!(aligned.shape(), &[5, 2]);
        for i in 0..5 {
            for j in 0..2 {
                assert!(aligned[(i, j)].is_nan());
            }
        }
    }

    #[test]
    fn align_higher_tf_ends_before_base_last_creates_nan_tail() {
        // The F-308 production scenario: base = M1 × 100 fresh bars,
        // higher TF = D1 with only 1 bar at t=0. Without max_age the
        // entire 100-bar base would have constant D1 values. With
        // max_age = 2 × D1_period = 2 days, all but the first ~2*1440 min
        // of base bars become NaN.
        let base_ns = ns_grid(0, 1, 100); // M1 × 100 = 100 min span
        let feat_ns = ns_grid(0, 1440, 1); // single D1 bar at t=0
        let feat_data = array![[99.0_f32]];
        // max_age = 2 × D1_period = 2 × 1440 × 60 × 1e9 ns
        let max_age_ns = Some(2_i64 * 1440 * 60 * 1_000_000_000);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns);
        // All 100 base bars are within 2 days of t=0, so ALL get 99.0.
        for i in 0..100 {
            assert_eq!(aligned[(i, 0)], 99.0);
        }
        // Now tighten max_age to 50 minutes — only first 51 base bars
        // (t=0..50) survive; rest become NaN.
        let max_age_ns = Some(50_i64 * 60 * 1_000_000_000);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns);
        for i in 0..=50 {
            assert_eq!(aligned[(i, 0)], 99.0, "i={i}");
        }
        for i in 51..100 {
            assert!(aligned[(i, 0)].is_nan(), "expected NaN at i={i}, got {}", aligned[(i, 0)]);
        }
    }
}
