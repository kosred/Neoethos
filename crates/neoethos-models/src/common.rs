//! Shared helpers for neoethos-models internals.
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

use anyhow::{Context, Result, bail};
use ndarray::Array2;

/// ROCm aliases are intentionally separate from CUDA normalization: explicit
/// NVIDIA requests must remain visible and fail in a ROCm-only neural build.
pub fn normalize_rocm_device_policy(policy: &str) -> String {
    let value = policy.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "default" => "auto".into(),
        "hip" | "gpu" => "rocm".into(),
        _ => value
            .strip_prefix("hip:")
            .or_else(|| value.strip_prefix("gpu:"))
            .map(|ordinal| format!("rocm:{ordinal}"))
            .unwrap_or(value),
    }
}

/// Checked against CubeCL 0.10 AmdDevice's u16 DeviceId representation. This
/// only parses intent; it does not assert a device exists or permit fallback.
pub fn parse_rocm_device_ordinal(policy: &str) -> Result<usize> {
    let normalized = normalize_rocm_device_policy(policy);
    let ordinal = match normalized.as_str() {
        "auto" | "rocm" => 0,
        value if value.starts_with("rocm:") => {
            let digits = &value[5..];
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                bail!("invalid explicit ROCm device policy `{policy}`");
            }
            digits
                .parse::<usize>()
                .context("ROCm device ordinal overflow")?
        }
        _ => bail!(
            "native ROCm cannot honor device policy `{policy}`; select the matching CPU or CUDA build explicitly"
        ),
    };
    if ordinal > u16::MAX as usize {
        bail!("ROCm device ordinal exceeds CubeCL's exact u16 device identity");
    }
    Ok(ordinal)
}

/// Operator intent for an NVIDIA CUDA execution path.
///
/// This type is deliberately CUDA-specific. Vendor-neutral labels such as
/// ROCm, Vulkan, Metal, and WGPU must not be collapsed into this enum: doing
/// so makes an AMD/Vulkan host look like a usable CUDA device and can silently
/// bind malformed requests to ordinal zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDevicePolicy {
    Auto,
    Cpu,
    Gpu { ordinal: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedCudaDevicePolicy {
    Cpu,
    Cuda { ordinal: usize },
}

/// Parse the exact CUDA policy vocabulary without repairing invalid input.
pub fn parse_cuda_device_policy(policy: &str) -> Result<CudaDevicePolicy> {
    let normalized = policy.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "auto" => Ok(CudaDevicePolicy::Auto),
        "cpu" => Ok(CudaDevicePolicy::Cpu),
        "gpu" | "cuda" | "nvidia" => Ok(CudaDevicePolicy::Gpu { ordinal: 0 }),
        "rocm" | "hip" | "metal" | "vulkan" | "wgpu" | "dx12" => {
            bail!("non-CUDA device policy cannot select a CUDA backend: `{policy}`")
        }
        _ => {
            for prefix in ["gpu:", "cuda:", "nvidia:"] {
                if let Some(raw_ordinal) = normalized.strip_prefix(prefix) {
                    if raw_ordinal.is_empty() {
                        bail!("CUDA device policy `{policy}` is missing an ordinal");
                    }
                    let ordinal = raw_ordinal.parse::<usize>().with_context(|| {
                        format!("invalid CUDA device ordinal in policy `{policy}`")
                    })?;
                    return Ok(CudaDevicePolicy::Gpu { ordinal });
                }
            }
            if ["rocm:", "hip:", "metal:", "vulkan:", "wgpu:", "dx12:"]
                .iter()
                .any(|prefix| normalized.starts_with(prefix))
            {
                bail!("non-CUDA device policy cannot select a CUDA backend: `{policy}`");
            }
            bail!(
                "unsupported CUDA device policy `{policy}`; expected auto, cpu, gpu[:N], cuda[:N], or nvidia[:N]"
            )
        }
    }
}

/// Resolve Auto against NVIDIA visibility without probing CUDA itself.
///
/// A visible NVIDIA card makes CUDA mandatory for Auto. The subsequent
/// runtime/device constructor therefore gets to report a broken driver or
/// toolkit instead of this resolver disguising that failure as a CPU host.
pub fn resolve_cuda_device_policy(
    policy: &str,
    visible_nvidia_devices: usize,
) -> Result<ResolvedCudaDevicePolicy> {
    match parse_cuda_device_policy(policy)? {
        CudaDevicePolicy::Auto if visible_nvidia_devices == 0 => Ok(ResolvedCudaDevicePolicy::Cpu),
        CudaDevicePolicy::Auto => Ok(ResolvedCudaDevicePolicy::Cuda { ordinal: 0 }),
        CudaDevicePolicy::Cpu => Ok(ResolvedCudaDevicePolicy::Cpu),
        CudaDevicePolicy::Gpu { ordinal } => {
            if visible_nvidia_devices == 0 {
                bail!(
                    "CUDA device policy `{policy}` requested ordinal {ordinal}, but no NVIDIA device is visible"
                );
            }
            if ordinal >= visible_nvidia_devices {
                bail!(
                    "CUDA device policy `{policy}` requested ordinal {ordinal}, but only {visible_nvidia_devices} NVIDIA device(s) are visible"
                );
            }
            Ok(ResolvedCudaDevicePolicy::Cuda { ordinal })
        }
    }
}

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

/// Returns `true` if a device policy requests GPU. The input is
/// normalized (trimmed, lowercased) before the prefix/equality test.
///
/// ## 2026-08-10 — the env kill-switch is gone
///
/// This used to take a `kernel_env_name` and AND the answer with
/// `NEOETHOS_BOT_<MODEL>_CUDA_KERNEL` not being set to a "disabled"
/// token. Five such names existed (`_NEAT_`, `_NEURO_EVO_`,
/// `_STATISTICAL_`, plus a per-model spelling of each) and none had a
/// config field, a knob-catalog row or a line in `config.yaml`. That
/// made them exactly the `NEOETHOS_GPU_F64` failure mode: a variable
/// that silently moves the run onto a different execution path and
/// leaves no trace in the artifact.
///
/// The device decision is now made once, by the configured device
/// policy (`models.statistical_device` for the statistical models, the
/// caller-supplied policy string for the evolutionary ones), and that
/// string is what the artifact records. Anyone who wants the CPU path
/// sets the policy to `cpu`.
pub fn cuda_kernel_enabled(policy: &str) -> Result<bool> {
    Ok(matches!(
        parse_cuda_device_policy(policy)?,
        CudaDevicePolicy::Gpu { .. }
    ))
}

/// Resolve which CUDA ordinal to bind to: parse `gpu:N` out of the
/// requested policy, else ordinal 0.
///
/// ## 2026-08-10 — the two env overrides are gone
///
/// This used to consult `NEOETHOS_BOT_<MODEL>_CUDA_DEVICE` and a
/// subsystem-wide fallback name BEFORE the policy string. That is two
/// more ways to say the one thing the policy already says — and the
/// two env names outranked the configured value, so a stale export
/// pinned every kernel to a card the config never named. The policy
/// carries the ordinal (`gpu:1`); there is no second channel.
pub fn cuda_device_id_from_policy(policy: &str) -> Result<usize> {
    match parse_cuda_device_policy(policy)? {
        CudaDevicePolicy::Gpu { ordinal } => Ok(ordinal),
        CudaDevicePolicy::Auto => {
            bail!("CUDA device policy `auto` must be resolved before selecting an ordinal")
        }
        CudaDevicePolicy::Cpu => {
            bail!("CPU device policy cannot select a CUDA ordinal")
        }
    }
}

/// The kernel's units-per-cube count: the hardware maximum, floored at 1.
/// `max_units` is supplied by the caller (typically from
/// `client.properties().hardware.max_units_per_cube`) so this helper
/// stays free of any specific compute-runtime types.
///
/// ## 2026-08-10 — `NEOETHOS_BOT_*_KERNEL_UNITS` deleted
///
/// Three env names could shrink the launch geometry below what the card
/// reports. Occupancy is a property of the hardware, not an operator
/// preference (never-OOM invariant: peak memory is a function of the
/// available hardware, never of a user parameter), so the value is now
/// read from the device and nowhere else.
pub fn cuda_kernel_units(max_units: u32) -> u32 {
    max_units.max(1)
}

/// Normalize CUDA labels into the canonical `auto|cpu|gpu|gpu:N` set used by
/// the runtime capability layer.
///
/// Retired or future vendor tokens are returned unchanged so the CUDA parser
/// or family-specific caller can reject them; they are never aliased to CUDA.
pub fn normalize_vendor_device_policy(policy: &str, _extra_prefixes: &[&str]) -> String {
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return "auto".to_string();
    }
    if matches!(normalized.as_str(), "cuda" | "nvidia") {
        return "gpu".to_string();
    }

    let suffix = normalized
        .strip_prefix("cuda:")
        .or_else(|| normalized.strip_prefix("nvidia:"))
        .or_else(|| normalized.strip_prefix("gpu:"));
    if let Some(index) = suffix {
        return format!("gpu:{index}");
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_policy_preserves_vendor_and_exact_representable_ordinal() {
        for (input, ordinal) in [("auto", 0), ("HIP", 0), ("gpu:3", 3), ("rocm:65535", 65535)] {
            assert_eq!(parse_rocm_device_ordinal(input).unwrap(), ordinal);
        }
        assert_eq!(normalize_rocm_device_policy(" CUDA:2 "), "cuda:2");
        for invalid in [
            "cpu",
            "cuda",
            "cuda:0",
            "nvidia",
            "vulkan:0",
            "rocm:",
            "rocm:-1",
            "rocm:+1",
            "rocm:1.0",
            "rocm:65536",
            "rocm:18446744073709551616",
        ] {
            assert!(
                parse_rocm_device_ordinal(invalid).is_err(),
                "accepted {invalid}"
            );
        }
        // Existing NVIDIA consumers must continue rejecting the AMD vocabulary.
        assert!(parse_cuda_device_policy("rocm:0").is_err());
    }
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
        assert!(cuda_kernel_enabled("gpu").expect("valid gpu policy"));
        assert!(cuda_kernel_enabled("gpu:1").expect("valid gpu ordinal"));
        assert!(!cuda_kernel_enabled("cpu").expect("valid cpu policy"));
        assert!(!cuda_kernel_enabled("auto").expect("valid auto policy"));
    }

    /// The kernel decision must depend on NOTHING but the policy string.
    /// Setting the three retired kill-switches must not move it — that is
    /// the whole point of deleting them, and a test that only checked the
    /// happy path would not have noticed a leftover reader.
    #[test]
    fn kernel_enabled_ignores_the_retired_env_kill_switches() {
        unsafe {
            std::env::set_var("NEOETHOS_BOT_STATISTICAL_CUDA_KERNEL", "0");
            std::env::set_var("NEOETHOS_BOT_NEAT_CUDA_KERNEL", "off");
            std::env::set_var("NEOETHOS_BOT_NEURO_EVO_CUDA_KERNEL", "disabled");
        }
        assert!(cuda_kernel_enabled("gpu").expect("valid gpu policy"));
        assert!(cuda_kernel_enabled("gpu:2").expect("valid gpu ordinal"));
        unsafe {
            std::env::remove_var("NEOETHOS_BOT_STATISTICAL_CUDA_KERNEL");
            std::env::remove_var("NEOETHOS_BOT_NEAT_CUDA_KERNEL");
            std::env::remove_var("NEOETHOS_BOT_NEURO_EVO_CUDA_KERNEL");
        }
    }

    #[test]
    fn cuda_device_id_parses_policy_suffix() {
        assert_eq!(cuda_device_id_from_policy("gpu:3").expect("ordinal"), 3);
        assert_eq!(
            cuda_device_id_from_policy("gpu").expect("default ordinal"),
            0
        );
        assert!(cuda_device_id_from_policy("cpu").is_err());
        assert!(cuda_device_id_from_policy("gpu:bad").is_err());
    }

    /// The retired `NEOETHOS_BOT_*_CUDA_DEVICE` names used to OUTRANK the
    /// policy. A stale export must no longer be able to move the kernel to
    /// a card the configured policy did not name.
    #[test]
    fn cuda_device_id_ignores_the_retired_device_env() {
        unsafe {
            std::env::set_var("NEOETHOS_BOT_STATISTICAL_CUDA_DEVICE", "5");
            std::env::set_var("NEOETHOS_BOT_NEAT_CUDA_DEVICE", "7");
        }
        assert_eq!(cuda_device_id_from_policy("gpu:1").expect("ordinal"), 1);
        unsafe {
            std::env::remove_var("NEOETHOS_BOT_STATISTICAL_CUDA_DEVICE");
            std::env::remove_var("NEOETHOS_BOT_NEAT_CUDA_DEVICE");
        }
    }

    #[test]
    fn cuda_kernel_units_is_the_hardware_maximum() {
        assert_eq!(cuda_kernel_units(64), 64);
        assert_eq!(cuda_kernel_units(0), 1);
        unsafe {
            std::env::set_var("NEOETHOS_BOT_STATISTICAL_KERNEL_UNITS", "32");
        }
        assert_eq!(cuda_kernel_units(64), 64);
        unsafe {
            std::env::remove_var("NEOETHOS_BOT_STATISTICAL_KERNEL_UNITS");
        }
    }

    #[test]
    fn normalize_vendor_device_policy_collapses_only_cuda_aliases() {
        assert_eq!(normalize_vendor_device_policy("cuda:1", &[]), "gpu:1");
        assert_eq!(normalize_vendor_device_policy("nvidia", &[]), "gpu");
        assert_eq!(normalize_vendor_device_policy("", &[]), "auto");
    }

    #[test]
    fn normalize_vendor_device_policy_does_not_alias_retired_backends() {
        for retired in ["rocm:2", "hip", "metal", "vulkan:0", "wgpu", "dx12"] {
            assert_eq!(normalize_vendor_device_policy(retired, &["wgpu"]), retired);
        }
    }

    #[test]
    fn strict_cuda_policy_rejects_non_cuda_vendors_and_malformed_ordinals() {
        for invalid in [
            "rocm", "rocm:1", "hip:0", "vulkan:0", "wgpu:0", "gpu:", "gpu:-1", "gpu:nope",
        ] {
            assert!(
                parse_cuda_device_policy(invalid).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn auto_cuda_resolution_uses_cpu_only_without_nvidia() {
        assert_eq!(
            resolve_cuda_device_policy("auto", 0).expect("no-card auto"),
            ResolvedCudaDevicePolicy::Cpu
        );
        assert_eq!(
            resolve_cuda_device_policy("auto", 1).expect("one-card auto"),
            ResolvedCudaDevicePolicy::Cuda { ordinal: 0 }
        );
        assert_eq!(
            resolve_cuda_device_policy("cuda:1", 2).expect("explicit ordinal"),
            ResolvedCudaDevicePolicy::Cuda { ordinal: 1 }
        );
        assert!(resolve_cuda_device_policy("cuda:2", 2).is_err());
    }
}
