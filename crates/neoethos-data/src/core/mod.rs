pub mod all_indicators;
pub mod canonical_ohlcv;
pub mod canonical_ohlcv_stream;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod classic_cuda_plan;
pub mod cross_pair_features;
pub mod dataset_candidate_lease;
pub mod dataset_generation_lease;
pub mod dataset_manifest;
pub mod direct_timeframes;
pub mod discover;
/// Hardware-derived ceiling on the indicator vocabulary's width, with the
/// memory arithmetic at the real bar counts. Restoring hundreds of columns
/// multiplies the widest object the system builds; this is what keeps peak
/// memory a function of the machine and never of a user parameter.
pub mod feature_budget;
pub mod feature_registry;
pub mod feature_run_lease;
pub mod features;
pub mod footprint_features;
/// vector-ta CUDA indicator lane, f64 end to end. Compiled ONLY under
/// `gpu-cuda`, so a card-less build never sees it — the module's own docs
/// explain the exact-architecture native cubin registry, why the lane no longer
/// narrows to f32 anywhere, and why the device table is short.
#[cfg(feature = "gpu-cuda")]
pub mod gpu_indicators;
#[cfg(feature = "gpu-cuda")]
pub mod gpu_only_feature_workspace_preflight_v3;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_classic_ta_v3;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_feature_recipe_v4;
#[cfg(feature = "gpu-cuda")]
pub mod gpu_resident_feature_store_v3;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_higher_timeframe_alignment_v3;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_quant_v3;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_regime_v3;
#[cfg(feature = "gpu-cuda")]
pub mod gpu_resident_robust_normalization_v2;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_session_v2;
#[cfg(feature = "gpu-cuda")]
pub(crate) mod gpu_resident_temporal_grid_v1;
pub mod hpc_ta;
pub mod import_discover;
pub mod import_limits;
pub mod import_provenance;
pub mod import_service;
/// Counted, reasoned outcomes for every indicator column the feature build
/// attempts. The mechanism that makes the 341-silent-drop defect impossible to
/// repeat: no discard on this path may be un-named or un-counted.
pub mod indicator_ledger;
pub mod indicator_telemetry;
pub mod indicators;
pub mod normalization;
pub mod pinned_canonical_series_v1;
pub mod quant_features;
pub mod regime_detection;
mod regime_exact_math_v1;
pub mod session_features;
pub mod slicing;
pub mod smc;
mod smc_log1p_exact_v1;
mod source_seal;
pub use source_seal::{initialize_source_seal_before_runtime, source_seal_slot_limit};
pub mod source_snapshot;
/// Numerically stable, NaN-aware f64 correlation — the replacement for the
/// prefilter's naive f32 sum-of-squares (measured 0.24x underestimate, and a
/// single NaN scored an entire column 0.0).
pub mod stats_f64;
pub mod timestamps;
pub mod vortex_feature_store;
pub mod vortex_io;
