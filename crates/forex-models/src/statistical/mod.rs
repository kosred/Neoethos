pub mod bayesian_impl;
pub mod common;
#[cfg(feature = "statistical-gpu")]
mod linear_gpu;
pub mod linear_impl;

pub use bayesian_impl::BayesianLogitExpert;
pub use linear_impl::{ElasticNetExpert, LogisticExpert};
