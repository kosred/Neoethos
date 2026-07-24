//! GPU-native discovery foundation modules.

pub mod benchmark;
pub mod capability;
pub mod cpu_strategy;
pub mod engine;
pub mod instrumentation;
pub mod parity_hierarchy;
pub mod population_fixture;
pub mod prototype_a;
pub mod prototype_bc;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
pub mod prototype_c_gpu;
pub mod ranking;
pub mod semantics;
