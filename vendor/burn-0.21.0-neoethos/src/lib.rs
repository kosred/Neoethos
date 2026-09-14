#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! Narrow Burn facade for NeoEthos.
//!
//! The application consumes Burn's shared tensor, module, neural-network,
//! optimizer, and Autodiff APIs. Concrete execution backends are dependencies
//! selected by NeoEthos itself: `burn-ndarray` for the CPU path and
//! `burn-cuda` for explicit CUDA builds. Keeping backend selection outside this
//! facade prevents the retired WGPU/Vulkan family from entering the lockfile.

pub use burn_core::*;

/// Backend decorators used by NeoEthos.
pub mod backend;

/// Neural-network building blocks.
pub mod nn {
    pub use burn_nn::*;
}

pub use burn_std::config::config as runtime_config;

/// Optimizers.
pub mod optim {
    pub use burn_optim::*;
}

/// Learning-rate schedulers.
#[cfg(feature = "std")]
pub mod lr_scheduler {
    pub use burn_optim::lr_scheduler::*;
}

/// Gradient clipping.
pub mod grad_clipping {
    pub use burn_optim::grad_clipping::*;
}

/// Common Burn imports.
pub mod prelude {
    pub use burn_core::prelude::*;

    pub use crate::nn;
}
