pub mod hashing;
pub mod numeric;
pub mod stats;
pub mod window_control;

pub use hashing::{fnv1a64, fnv1a64_update};
pub use numeric::{clamp_unit_f32, clamp_unit_f64, finite_or, finite_or_f32, stable_sigmoid_f32};
pub use stats::{mean, mean_std, mean_vector_f32, pearson_correlation_f32, stddev, stddev_sample};
