//! Shared helpers for forex-models internals.
//!
//! Phase 67 extraction: `flatten_features` was duplicated across
//! `evolution::crfmnes_gpu`, `evolution::neat_gpu`, and
//! `statistical::linear_gpu`. Each emitted a different error label
//! ("neuro-evo cuda…", "NEAT cuda…", "statistical cuda…") but the
//! validation + flattening math was identical.
//!
//! Phase 78 extension: the per-kernel `*_cuda_kernel_enabled`,
//! `cuda_device_id`, and `kernel_units` helpers (also duplicated 3x
//! across the same three GPU files) are now collapsed here, plus the
//! shared core of the four `normalize_*_device_policy` functions
//! (statistical/runtime/rl/burn). See
//! `docs/audits/research/gpu_consolidation_audit.md` for the matrix.

use anyhow::{Result, bail};
use ndarray::Array2;

/// Validate that `features` has exactly `input_dim` columns and return
/// the flattened row-major buffer ready for upload to a CUDA kernel.
///
/// `caller_label` is folded into the error message so the operator
/// knows which subsystem produced the mismatch (e.g. `"NEAT"` for the
/// neuro-evolution path, `"statistical"` for linear softmax).
pub fn cuda_flatten_features(
    features: &Array2<f32>,
    input_dim: usize,
    caller_label: &str,
) -> Result<Vec<f32>> {
    if features.ncols() != input_dim {
        bail!(
            "{caller_label} cuda feature dimension mismatch: expected {}, received {}",
            input_dim,
            features.ncols()
        );
    }
    Ok(features.iter().copied().collect())
}

/// Returns `true` if the given env var holds one of the standard
/// "disabled" tokens. Used to gate per-model CUDA kernels via
/// `FOREX_BOT_<NAME>_CUDA_KERNEL=0`. Centralized so we do not have
/// three exact-copy `matches!` blocks across the GPU files.
pub fn is_kernel_disabled_env(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

/// Returns `true` if a device policy requests GPU AND the kernel is
/// not disabled by env var. Both inputs are normalized (trimmed,
/// lowercased) before the prefix/equality test.
pub fn cuda_kernel_enabled(policy: &str, kernel_env_name: &str) -> bool {
    let normalized = policy.trim().to_ascii_lowercase();
    let requested_gpu = normalized == "gpu" || normalized.starts_with("gpu:");
    requested_gpu && !is_kernel_disabled_env(kernel_env_name)
}

/// Resolve which CUDA ordinal to bind to:
///   1. honour the explicit `<DEVICE_ENV>` env var if set + parseable;
///   2. honour the fallback env var (subsystem-wide) if set + parseable;
///   3. parse `gpu:N` out of the requested policy;
///   4. default to ordinal 0.
pub fn cuda_device_id_from_policy(
    policy: &str,
    device_env_name: &str,
    fallback_env_name: Option<&str>,
) -> usize {
    let read = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<usize>().ok());
    if let Some(id) = read(device_env_name) {
        return id;
    }
    if let Some(fallback) = fallback_env_name
        && let Some(id) = read(fallback)
    {
        return id;
    }
    policy
        .trim()
        .to_ascii_lowercase()
        .strip_prefix("gpu:")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Resolve the kernel's units-per-cube count, clamped to
/// `[1, max_units]`. `max_units` is supplied by the caller (typically
/// from `client.properties().hardware.max_units_per_cube`) so this
/// helper stays free of any specific compute-runtime types.
pub fn cuda_kernel_units(max_units: u32, units_env_name: &str) -> u32 {
    let max_units = max_units.max(1);
    std::env::var(units_env_name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

/// Collapse vendor-specific device labels into the canonical
/// `auto|cpu|gpu|gpu:N` set used by the runtime capability layer.
///
/// `extra_prefixes` lets callers extend the recognised vendor set
/// (e.g. burn passes `["wgpu"]` because the burn backend accepts the
/// `wgpu:N` form that statistical / runtime callers do not).
///
/// Unknown tokens are returned unchanged (lowercased) so callers can
/// layer their own validation on top.
pub fn normalize_vendor_device_policy(policy: &str, extra_prefixes: &[&str]) -> String {
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return "auto".to_string();
    }
    if matches!(
        normalized.as_str(),
        "cuda" | "rocm" | "metal" | "vulkan" | "nvidia"
    ) || extra_prefixes.contains(&normalized.as_str())
    {
        return "gpu".to_string();
    }

    let mut suffix = normalized
        .strip_prefix("cuda:")
        .or_else(|| normalized.strip_prefix("rocm:"))
        .or_else(|| normalized.strip_prefix("metal:"))
        .or_else(|| normalized.strip_prefix("vulkan:"))
        .or_else(|| normalized.strip_prefix("gpu:"));
    if suffix.is_none() {
        for prefix in extra_prefixes {
            let with_colon = format!("{prefix}:");
            if let Some(rest) = normalized.strip_prefix(&with_colon) {
                suffix = Some(rest);
                break;
            }
        }
    }
    if let Some(index) = suffix {
        return format!("gpu:{index}");
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn flatten_accepts_matching_dimension() {
        let features = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let flat = cuda_flatten_features(&features, 3, "test").expect("matching dim");
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flatten_rejects_mismatched_dimension() {
        let features = array![[1.0_f32, 2.0, 3.0]];
        let err = cuda_flatten_features(&features, 5, "neuro-evo")
            .expect_err("mismatched dim must reject");
        let msg = err.to_string();
        assert!(msg.contains("neuro-evo"));
        assert!(msg.contains("expected 5"));
        assert!(msg.contains("received 3"));
    }

    #[test]
    fn kernel_enabled_requires_gpu_policy() {
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_CUDA_KERNEL");
        }
        assert!(cuda_kernel_enabled("gpu", "FOREX_BOT_TEST_CUDA_KERNEL"));
        assert!(cuda_kernel_enabled("gpu:1", "FOREX_BOT_TEST_CUDA_KERNEL"));
        assert!(!cuda_kernel_enabled("cpu", "FOREX_BOT_TEST_CUDA_KERNEL"));
        assert!(!cuda_kernel_enabled("auto", "FOREX_BOT_TEST_CUDA_KERNEL"));
    }

    #[test]
    fn kernel_enabled_respects_disable_env() {
        unsafe {
            std::env::set_var("FOREX_BOT_TEST_DISABLE_KERNEL", "false");
        }
        assert!(!cuda_kernel_enabled(
            "gpu",
            "FOREX_BOT_TEST_DISABLE_KERNEL"
        ));
        unsafe {
            std::env::set_var("FOREX_BOT_TEST_DISABLE_KERNEL", "1");
        }
        assert!(cuda_kernel_enabled(
            "gpu",
            "FOREX_BOT_TEST_DISABLE_KERNEL"
        ));
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_DISABLE_KERNEL");
        }
    }

    #[test]
    fn cuda_device_id_parses_policy_suffix() {
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_DEVICE");
        }
        assert_eq!(
            cuda_device_id_from_policy("gpu:3", "FOREX_BOT_TEST_DEVICE", None),
            3
        );
        assert_eq!(
            cuda_device_id_from_policy("gpu", "FOREX_BOT_TEST_DEVICE", None),
            0
        );
    }

    #[test]
    fn cuda_device_id_prefers_env_var() {
        unsafe {
            std::env::set_var("FOREX_BOT_TEST_DEVICE_EXPLICIT", "5");
        }
        assert_eq!(
            cuda_device_id_from_policy("gpu:1", "FOREX_BOT_TEST_DEVICE_EXPLICIT", None),
            5
        );
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_DEVICE_EXPLICIT");
        }
    }

    #[test]
    fn cuda_kernel_units_clamps_to_max() {
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_UNITS");
        }
        assert_eq!(cuda_kernel_units(64, "FOREX_BOT_TEST_UNITS"), 64);
        unsafe {
            std::env::set_var("FOREX_BOT_TEST_UNITS", "32");
        }
        assert_eq!(cuda_kernel_units(64, "FOREX_BOT_TEST_UNITS"), 32);
        unsafe {
            std::env::set_var("FOREX_BOT_TEST_UNITS", "9999");
        }
        assert_eq!(cuda_kernel_units(64, "FOREX_BOT_TEST_UNITS"), 64);
        unsafe {
            std::env::remove_var("FOREX_BOT_TEST_UNITS");
        }
    }

    #[test]
    fn normalize_vendor_device_policy_collapses_aliases() {
        assert_eq!(normalize_vendor_device_policy("cuda:1", &[]), "gpu:1");
        assert_eq!(normalize_vendor_device_policy("rocm:2", &[]), "gpu:2");
        assert_eq!(normalize_vendor_device_policy("metal", &[]), "gpu");
        assert_eq!(normalize_vendor_device_policy("vulkan:0", &[]), "gpu:0");
        assert_eq!(normalize_vendor_device_policy("", &[]), "auto");
    }

    #[test]
    fn normalize_vendor_device_policy_respects_extras() {
        assert_eq!(
            normalize_vendor_device_policy("wgpu:2", &["wgpu"]),
            "gpu:2"
        );
        assert_eq!(normalize_vendor_device_policy("wgpu", &["wgpu"]), "gpu");
    }
}
