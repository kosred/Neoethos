#[cfg(feature = "statistical-gpu")]
mod adaptive_gpu;
pub mod adaptive_impl;
#[cfg(all(test, feature = "statistical-gpu"))]
#[path = "adaptive_pa_full_gpu_tests.rs"]
mod adaptive_pa_full_gpu_tests;
#[cfg(all(test, feature = "statistical-gpu"))]
#[path = "adaptive_pa_gpu_tests.rs"]
mod adaptive_pa_gpu_tests;

pub use adaptive_impl::{
    AdaptiveGradientBooster, OnlineHoeffdingExpert, OnlinePassiveAggressiveExpert,
};
