pub mod hashing;
pub mod numeric;
pub mod series;
pub mod stats;
pub mod window_control;

pub use hashing::{fnv1a64, fnv1a64_update};
pub use numeric::{clamp_unit_f32, clamp_unit_f64, finite_or, finite_or_f32, stable_sigmoid_f32};
pub use series::{
    ewma_f32, median_ignore_nan, median_sorted_f32, moving_average_f32, percentile_sorted_f32,
    rolling_mean_f64,
};
pub use stats::{mean, mean_std, mean_vector_f32, pearson_correlation_f32, stddev, stddev_sample};
