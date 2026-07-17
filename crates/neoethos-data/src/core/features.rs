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
    /// A contiguous ROW WINDOW `[start, end)` over a mmap store — a VIEW, no
    /// materialization (2026-07-18 never-OOM fix). The discovery holdout
    /// split used to copy 80% of a multi-GB disk cube into an in-RAM
    /// `Array2` before the GA even started — peak RAM became a function of
    /// the DATASET size instead of the hardware, freezing small boxes on
    /// dense timeframes (EURUSD M5 = 7.3 GB cube → ~5.8 GB surprise
    /// allocation). This variant serves the same accessors zero-copy off
    /// the OS page cache; only the small slices callers explicitly request
    /// (`sample_window`) are ever materialized.
    MmapWindow {
        store: std::sync::Arc<crate::core::feature_store::FeatureStore>,
        start: usize,
        end: usize,
    },
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
            FeatureData::MmapWindow { start, end, .. } => end - start,
        }
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_features(&self) -> usize {
        match &self.data {
            FeatureData::InMemory(a) => a.ncols(),
            FeatureData::Mmap(s) => s.n_features(),
            FeatureData::MmapWindow { store, .. } => store.n_features(),
        }
    }

    /// Feature `idx`'s full time series (`[samples]`).
    #[inline]
    pub fn feature_column(&self, idx: usize) -> ndarray::ArrayView1<'_, f32> {
        match &self.data {
            FeatureData::InMemory(a) => a.column(idx),
            FeatureData::Mmap(s) => ndarray::ArrayView1::from(s.feature_row(idx)),
            FeatureData::MmapWindow { store, start, end } => {
                ndarray::ArrayView1::from(&store.feature_row(idx)[*start..*end])
            }
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
            FeatureData::MmapWindow {
                store,
                start: w_start,
                ..
            } => {
                // Offsets are window-relative; map onto the underlying store.
                let (abs_start, abs_end) = (w_start + start, w_start + end);
                let rows = end - start;
                let ncols = store.n_features();
                let mut out = Array2::zeros((rows, ncols));
                for f in 0..ncols {
                    out.column_mut(f).assign(&ndarray::ArrayView1::from(
                        &store.feature_row(f)[abs_start..abs_end],
                    ));
                }
                out
            }
        }
    }

    /// A new owned `FeatureFrame` over the contiguous row range `[start, end)`
    /// (a temporal slice). Because the multi-timeframe features are already
    /// flattened — each row is one base bar carrying all its higher-TF context —
    /// a plain row split is a clean temporal cut with NO re-alignment needed.
    /// Used by the sealed-lockbox split (train on the early rows, judge on the
    /// untouched recent rows). Always materialises in-memory (slices are small).
    pub fn row_slice(&self, start: usize, end: usize) -> FeatureFrame {
        let start = start.min(self.n_samples());
        let end = end.min(self.n_samples()).max(start);
        FeatureFrame {
            timestamps: self.timestamps[start..end].to_vec(),
            names: self.names.clone(),
            data: FeatureData::InMemory(self.sample_window(start, end)),
        }
    }

    /// A `FeatureFrame` over the contiguous row range `[start, end)` that is a
    /// zero-copy VIEW when the backing is a mmap store (never-OOM fix
    /// 2026-07-18) and a materialized copy only for the (already-in-RAM)
    /// in-memory backing. This is what the discovery holdout split uses: the
    /// old path materialized 80% of a multi-GB disk cube into RAM before the
    /// GA even started, freezing small machines on dense timeframes.
    pub fn row_window(&self, start: usize, end: usize) -> FeatureFrame {
        let start = start.min(self.n_samples());
        let end = end.min(self.n_samples()).max(start);
        let data = match &self.data {
            FeatureData::InMemory(_) => FeatureData::InMemory(self.sample_window(start, end)),
            FeatureData::Mmap(store) => FeatureData::MmapWindow {
                store: store.clone(),
                start,
                end,
            },
            FeatureData::MmapWindow {
                store,
                start: w_start,
                ..
            } => FeatureData::MmapWindow {
                store: store.clone(),
                start: w_start + start,
                end: w_start + end,
            },
        };
        FeatureFrame {
            timestamps: self.timestamps[start..end].to_vec(),
            names: self.names.clone(),
            data,
        }
    }

    /// Single feature value at logical `(sample, feature)`.
    #[inline]
    pub fn feature_at(&self, sample: usize, feature: usize) -> f32 {
        match &self.data {
            FeatureData::InMemory(a) => a[(sample, feature)],
            FeatureData::Mmap(s) => s.feature_row(feature)[sample],
            FeatureData::MmapWindow { store, start, .. } => {
                store.feature_row(feature)[start + sample]
            }
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
            // Column-window of the [features x samples] store view: a valid
            // STRIDED ArrayView2 (row stride = full n_samples) - zero-copy.
            FeatureData::MmapWindow { store, start, end } => {
                store.as_view().slice_move(ndarray::s![.., *start..*end])
            }
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
            FeatureData::MmapWindow { store, start, end } => {
                let (s0, s1) = (*start, *end);
                Box::new(
                    (0..store.n_features())
                        .flat_map(move |f| store.feature_row(f)[s0..s1].iter().copied()),
                )
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
            FeatureData::MmapWindow { store, start, end } => {
                let (nf, rows) = (store.n_features(), end - start);
                let mut out = Array2::zeros((rows, nf));
                for f in 0..nf {
                    out.column_mut(f).assign(&ndarray::ArrayView1::from(
                        &store.feature_row(f)[*start..*end],
                    ));
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
/// `availability_lag_ns` (audit D02, 2026-07-13) is the delay between a
/// feature row's TIMESTAMP and the moment its values become knowable.
/// Bars are OPEN-stamped everywhere in this codebase (the resampler
/// buckets by `div_euclid(period)`; cTrader trendbars are open-stamped),
/// so a higher-TF bar's final OHLC-derived features only exist at
/// `stamp + period` — its close. The old alignment (`stamp <= ts`, i.e.
/// lag 0) handed every base bar the CONTAINING higher-TF bucket, whose
/// final values lie up to one full period in the future: 5 minutes of
/// lookahead per M5 feature, 4 HOURS per H4, a DAY per D1 — inflating
/// every offline evaluation while live saw different (partial-bar)
/// values, silently breaking backtest↔live parity. It also contradicted
/// the declared `MultiTimeframeAvailabilityPolicy::ClosedHigherTimeframeOnly`.
///
/// Pass the higher TF's period as the lag for closed-bar-only alignment;
/// pass `0` for the legacy same-stamp semantics (exact-match / non-HTF
/// callers — byte-identical to the old behavior). Staleness
/// (`max_age_ns`) is measured from the row's AVAILABILITY time
/// (`stamp + lag`), not its stamp.
pub fn align_features_by_ns(
    base_ns: &[i64],
    feature_ns: &[i64],
    feature_data: &Array2<f32>,
    ffill: bool,
    max_age_ns: Option<i64>,
    availability_lag_ns: i64,
) -> Array2<f32> {
    let n_base = base_ns.len();
    let n_feat = feature_ns.len();
    let n_cols = feature_data.ncols();
    let mut out = Array2::from_elem((n_base, n_cols), f32::NAN);

    if n_feat == 0 {
        return out;
    }

    let lag = availability_lag_ns.max(0);
    let mut feat_idx = 0usize;
    for i in 0..n_base {
        let ts = base_ns[i];
        // Advance past every row whose values are AVAILABLE at `ts`
        // (stamp + lag <= ts). With lag 0 this is the legacy `stamp <= ts`.
        while feat_idx < n_feat && feature_ns[feat_idx].saturating_add(lag) <= ts {
            feat_idx += 1;
        }

        let best_idx = if feat_idx > 0 {
            let prev = feat_idx - 1;
            let available_at = feature_ns[prev].saturating_add(lag);
            if available_at == ts {
                Some(prev)
            } else if ffill {
                // F-308 max-age guard: drop the forward-fill when the
                // most-recent AVAILABLE row is older than the cap.
                match max_age_ns {
                    Some(max_age) if ts - available_at > max_age => None,
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
        // Legacy behaviour preserved when max_age = None (lag 0 — note this
        // legacy mode hands t=0..4 the CONTAINING M5 bucket, i.e. lookahead;
        // production HTF alignment passes the period as lag since D02).
        let base_ns = ns_grid(0, 1, 10);   // M1 × 10 bars
        let feat_ns = ns_grid(0, 5, 2);    // M5 × 2 bars: t=0, t=5
        let feat_data = array![[1.0_f32], [2.0_f32]];
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, None, 0);
        // Without max_age, every base bar past t=5 keeps value 2.0.
        assert_eq!(aligned[(0, 0)], 1.0); // t=0
        assert_eq!(aligned[(4, 0)], 1.0); // t=4 (before first M5 close at 5)
        assert_eq!(aligned[(5, 0)], 2.0); // t=5
        assert_eq!(aligned[(9, 0)], 2.0); // t=9 — frozen, what F-308 calls the bug
    }

    #[test]
    fn align_close_availability_never_reads_the_forming_bar() {
        // Audit D02: with lag = the higher-TF period, a base bar may only
        // read higher-TF bars that have CLOSED at or before its stamp.
        let base_ns = ns_grid(0, 1, 12); // M1 × 12: t=0..11
        let feat_ns = ns_grid(0, 5, 2); //  M5 × 2: opens t=0 (closes 5), t=5 (closes 10)
        let feat_data = array![[1.0_f32], [2.0_f32]];
        let lag = 5 * 60 * 1_000_000_000_i64; // one M5 period
        let max_age = Some(10 * 60 * 1_000_000_000_i64); // 2× period, from close
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age, lag);
        // t=0..4: bar[0] is still FORMING (closes at t=5) — its final values
        // must be invisible. The old alignment leaked 1.0 here.
        for i in 0..5 {
            assert!(
                aligned[(i, 0)].is_nan(),
                "t={i}: forming-bar leak — got {}",
                aligned[(i, 0)]
            );
        }
        // t=5..9: bar[0] closed at t=5 → its values become available; bar[1]
        // is forming (closes t=10) and must stay invisible.
        for i in 5..10 {
            assert_eq!(aligned[(i, 0)], 1.0, "t={i}");
        }
        // t=10,11: bar[1] closed at t=10.
        assert_eq!(aligned[(10, 0)], 2.0);
        assert_eq!(aligned[(11, 0)], 2.0);
    }

    #[test]
    fn align_close_availability_staleness_measured_from_close() {
        // One M5 bar opening t=0 (closes t=5), max_age = 3 min FROM CLOSE:
        // available t=5..8, stale (NaN) from t=9.
        let base_ns = ns_grid(0, 1, 12);
        let feat_ns = ns_grid(0, 5, 1);
        let feat_data = array![[7.0_f32]];
        let lag = 5 * 60 * 1_000_000_000_i64;
        let max_age = Some(3 * 60 * 1_000_000_000_i64);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age, lag);
        for i in 0..5 {
            assert!(aligned[(i, 0)].is_nan(), "t={i}: not yet closed");
        }
        for i in 5..9 {
            assert_eq!(aligned[(i, 0)], 7.0, "t={i}: fresh after close");
        }
        for i in 9..12 {
            assert!(aligned[(i, 0)].is_nan(), "t={i}: stale past max_age from close");
        }
    }

    #[test]
    fn align_max_age_caps_stale_forward_fill() {
        // F-308 fix: max_age = 3 minutes (in ns) drops values past 3 min lag.
        let base_ns = ns_grid(0, 1, 10);
        let feat_ns = ns_grid(0, 5, 2);
        let feat_data = array![[1.0_f32], [2.0_f32]];
        let max_age_ns = Some(3_i64 * 60 * 1_000_000_000);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0);
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
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, Some(0), 0);
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
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, false, Some(i64::MAX), 0);
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
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, Some(60_000_000_000), 0);
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
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0);
        // All 100 base bars are within 2 days of t=0, so ALL get 99.0.
        for i in 0..100 {
            assert_eq!(aligned[(i, 0)], 99.0);
        }
        // Now tighten max_age to 50 minutes — only first 51 base bars
        // (t=0..50) survive; rest become NaN.
        let max_age_ns = Some(50_i64 * 60 * 1_000_000_000);
        let aligned = align_features_by_ns(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0);
        for i in 0..=50 {
            assert_eq!(aligned[(i, 0)], 99.0, "i={i}");
        }
        for i in 51..100 {
            assert!(aligned[(i, 0)].is_nan(), "expected NaN at i={i}, got {}", aligned[(i, 0)]);
        }
    }
}

#[cfg(test)]
mod mmap_window_tests {
    use super::*;
    use crate::core::feature_store::{FeatureStore, FeatureStoreWriter};

    /// Build a tiny on-disk store (3 features × 10 samples), open it, and
    /// prove every accessor of a `row_window` VIEW matches the materialized
    /// equivalent — the never-OOM fix must be a pure representation change.
    #[test]
    fn mmap_window_view_matches_materialized_slice() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_mmap_window_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.fstore");
        let mut w = FeatureStoreWriter::create(&path, 10).unwrap();
        for f in 0..3u32 {
            let series: Vec<f32> = (0..10).map(|i| (f * 100 + i) as f32).collect();
            w.append_feature(&series).unwrap();
        }
        w.finish().unwrap();
        let store = FeatureStore::open(&path, 3, 10, false).unwrap();

        let frame = FeatureFrame {
            timestamps: (0..10).collect(),
            names: vec!["a".into(), "b".into(), "c".into()],
            data: FeatureData::Mmap(std::sync::Arc::new(store)),
        };

        let win = frame.row_window(2, 7); // rows 2..7
        assert!(matches!(win.data, FeatureData::MmapWindow { .. }), "mmap → view, no copy");
        assert_eq!(win.n_samples(), 5);
        assert_eq!(win.n_features(), 3);
        assert_eq!(win.timestamps, vec![2, 3, 4, 5, 6]);

        // feature_column / feature_at
        assert_eq!(win.feature_column(1).to_vec(), vec![102.0, 103.0, 104.0, 105.0, 106.0]);
        assert_eq!(win.feature_at(0, 2), 202.0);
        assert_eq!(win.feature_at(4, 0), 6.0);

        // sample_window with window-relative offsets
        let sw = win.sample_window(1, 3); // absolute rows 3..5
        assert_eq!(sw[(0, 0)], 3.0);
        assert_eq!(sw[(1, 2)], 204.0);

        // as_indicators_view: [features × window] strided view
        let iv = win.as_indicators_view();
        assert_eq!(iv.shape(), &[3, 5]);
        assert_eq!(iv[(2, 0)], 202.0);
        assert_eq!(iv[(0, 4)], 6.0);

        // iter_values covers exactly the window
        assert_eq!(win.iter_values().count(), 15);
        assert!(win.iter_values().all(|v| v.rem_euclid(100.0) >= 2.0 && v.rem_euclid(100.0) <= 6.0));

        // Nested window narrows onto the same store
        let inner = win.row_window(1, 4); // absolute 3..6
        assert_eq!(inner.feature_column(0).to_vec(), vec![3.0, 4.0, 5.0]);

        // to_dense of the window equals the direct materialized slice
        assert_eq!(win.to_dense_samples_major(), frame.sample_window(2, 7));

        drop(win);
        drop(inner);
        drop(frame);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
