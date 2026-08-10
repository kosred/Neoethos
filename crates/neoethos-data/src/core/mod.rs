pub mod all_indicators;
pub mod cross_pair_features;
pub mod discover;
/// Hardware-derived ceiling on the indicator vocabulary's width, with the
/// memory arithmetic at the real bar counts. Restoring hundreds of columns
/// multiplies the widest object the system builds; this is what keeps peak
/// memory a function of the machine and never of a user parameter.
pub mod feature_budget;
pub mod feature_registry;
pub mod feature_store;
pub mod features;
pub mod footprint_features;
/// vector-ta CUDA indicator lane, f64 end to end. Compiled ONLY under
/// `gpu-cuda`, so a card-less build never sees it — the module's own docs
/// explain how the arch trap was closed (one multi-arch fatbin plus forward
/// PTX), why the lane no longer narrows to f32 anywhere, and why the device
/// table is short.
#[cfg(feature = "gpu-cuda")]
pub mod gpu_indicators;
pub mod hpc_ta;
/// Counted, reasoned outcomes for every indicator column the feature build
/// attempts. The mechanism that makes the 341-silent-drop defect impossible to
/// repeat: no discard on this path may be un-named or un-counted.
pub mod indicator_ledger;
pub mod indicator_telemetry;
pub mod indicators;
pub mod loader;
pub mod normalization;
pub mod parquet_migration;
pub mod quant_features;
pub mod regime_detection;
pub mod resample;
pub mod session_features;
pub mod slicing;
pub mod smc;
/// Numerically stable, NaN-aware f64 correlation — the replacement for the
/// prefilter's naive f32 sum-of-squares (measured 0.24x underestimate, and a
/// single NaN scored an entire column 0.0).
pub mod stats_f64;
pub mod timestamps;
pub mod to_vortex;
pub mod universal_importer;
pub mod vortex_io;
