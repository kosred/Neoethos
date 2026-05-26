pub mod clock;
pub mod hashing;
pub mod numeric;
pub mod series;
pub mod stats;
// F-101 fix (2026-05-25): `pub mod window_control;` removed.
// The 140-LOC Win32 GUI-automation module had ZERO external callers
// across the workspace (verified via grep). It was a relic from the
// pre-Flutter wizard automation that never made it into the new
// architecture. Deleted per operator directive 2026-05-24 to keep
// the foundation crate focused.

pub use clock::now_unix_ms;

pub use hashing::{fnv1a64, fnv1a64_update};
pub use numeric::{clamp_unit_f32, clamp_unit_f64, finite_or, finite_or_f32, stable_sigmoid_f32};
pub use series::{
    ewma_f32, median_ignore_nan, median_sorted_f32, moving_average_f32, percentile_sorted_f32,
    rolling_mean_f64,
};
pub use stats::{mean, mean_std, mean_vector_f32, pearson_correlation_f32, stddev, stddev_sample};
