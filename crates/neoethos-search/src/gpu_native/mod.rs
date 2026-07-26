//! GPU-native discovery foundation modules.

pub mod benchmark;
pub mod capability;
pub mod cpu_strategy;
pub mod engine;
pub mod instrumentation;
pub mod parity_hierarchy;
pub mod population_fixture;
pub mod prototype_a;
#[cfg(feature = "gpu")]
mod prototype_a_engine;
pub mod prototype_b_engine;
pub mod prototype_bc;
pub mod prototype_c_engine;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
pub mod prototype_c_gpu;
pub mod prototype_population;
pub mod prototype_population_oracle;
pub mod ranking;
pub mod semantics;
#[cfg(all(test, any(feature = "gpu-cuda", feature = "gpu-vulkan")))]
pub mod signal_trace_gpu;
pub mod snapshot_fixture;
#[cfg(test)]
mod trade_trace_gpu;
