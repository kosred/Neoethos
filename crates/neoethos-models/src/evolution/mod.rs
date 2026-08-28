#[cfg(feature = "neuro-evolution-gpu")]
mod crfmnes_gpu;
pub mod crfmnes_impl;
#[cfg(feature = "neuro-evolution-gpu")]
mod neat_gpu;
pub mod neat_impl;

pub use crfmnes_impl::{NeuroEvoExpert, NeuroEvoOptimizer};
pub use neat_impl::NeatExpert;
