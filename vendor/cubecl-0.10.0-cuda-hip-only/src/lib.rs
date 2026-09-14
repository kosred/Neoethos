//! NeoEthos' narrow CubeCL facade.
//!
//! CUDA is the only active accelerator runtime. HIP/ROCm is retained as an
//! explicit future feature so it can be implemented and validated without
//! being aliased to CUDA or a host fallback. The retired WGPU/Vulkan family is
//! intentionally absent from both this API and its dependency graph.

pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "hip")]
pub use cubecl_hip as hip;

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;
