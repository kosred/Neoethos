pub mod linear_impl;
pub mod bayesian_impl;

pub use linear_impl::{ElasticNetExpert, LogisticExpert};
pub use bayesian_impl::BayesianLogitExpert;
