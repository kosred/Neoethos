use serde::{Deserialize, Serialize};

use crate::contracts::BackendKind;

use super::HardwareProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorBackend {
    Cpu,
    Cuda,
    Rocm,
    Wgpu,
    Vulkan,
    Metal,
    Dx12,
}

impl AcceleratorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Wgpu => "wgpu",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Dx12 => "dx12",
        }
    }

    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub fn is_cuda_native(self) -> bool {
        matches!(self, Self::Cuda)
    }

    pub fn is_wgpu_family(self) -> bool {
        matches!(self, Self::Wgpu | Self::Vulkan | Self::Metal | Self::Dx12)
    }

    pub fn backend_kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::NativeCpu,
            Self::Cuda => BackendKind::NativeCuda,
            Self::Rocm | Self::Wgpu | Self::Vulkan | Self::Metal | Self::Dx12 => {
                BackendKind::BurnWgpu
            }
        }
    }
}

pub(super) fn normalize_accelerator_preference(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => "auto".to_string(),
        "cpu" | "false" | "0" | "no" | "off" => "cpu".to_string(),
        "gpu" | "true" | "1" | "yes" | "on" | "nvidia" | "amd" => "gpu".to_string(),
        "cuda" | "cuda:0" => "cuda".to_string(),
        "rocm" | "hip" | "rocm:0" | "hip:0" => "rocm".to_string(),
        "wgpu" | "wgpu:0" | "wgpu_vulkan" => "wgpu".to_string(),
        "vulkan" | "vulkan:0" => "vulkan".to_string(),
        "metal" | "metal:0" => "metal".to_string(),
        "dx12" | "directx12" | "d3d12" | "dx12:0" => "dx12".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn choose_primary_backend(
    preference: &str,
    profile: &HardwareProfile,
) -> AcceleratorBackend {
    if profile.accelerator_devices.is_empty() || preference == "cpu" || preference == "off" {
        return AcceleratorBackend::Cpu;
    }

    let has_cuda = !profile
        .devices_for_backend(AcceleratorBackend::Cuda)
        .is_empty();
    let has_rocm = !profile
        .devices_for_backend(AcceleratorBackend::Rocm)
        .is_empty();
    let has_wgpu = !profile.wgpu_native_devices().is_empty();

    // F-CORE2-014: previously these branches silently downgraded to CPU when
    // the user explicitly asked for a GPU backend that wasn't probed. That
    // makes hour-long discovery runs land on CPU without any signal. Emit a
    // structured warn at the decision site so the downgrade is visible.
    fn warn_downgrade(requested: &str, reason: &str) {
        tracing::warn!(
            target: "forex_core::backends",
            requested = requested,
            reason = reason,
            "GPU backend requested but unavailable; downgrading to CPU (F-CORE2-014)"
        );
    }

    match preference {
        "cuda" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else {
                warn_downgrade("cuda", "no CUDA device detected by hardware probe");
                AcceleratorBackend::Cpu
            }
        }
        "rocm" => {
            if has_rocm {
                AcceleratorBackend::Rocm
            } else {
                warn_downgrade("rocm", "no ROCm device detected by hardware probe");
                AcceleratorBackend::Cpu
            }
        }
        "wgpu" => {
            if has_wgpu {
                AcceleratorBackend::Wgpu
            } else {
                warn_downgrade("wgpu", "no wgpu-compatible device detected");
                AcceleratorBackend::Cpu
            }
        }
        "vulkan" => {
            if has_wgpu {
                AcceleratorBackend::Vulkan
            } else {
                warn_downgrade("vulkan", "no wgpu/Vulkan-compatible device detected");
                AcceleratorBackend::Cpu
            }
        }
        "metal" => {
            if has_wgpu {
                AcceleratorBackend::Metal
            } else {
                warn_downgrade("metal", "no wgpu/Metal-compatible device detected");
                AcceleratorBackend::Cpu
            }
        }
        "dx12" => {
            if has_wgpu {
                AcceleratorBackend::Dx12
            } else {
                warn_downgrade("dx12", "no wgpu/DX12-compatible device detected");
                AcceleratorBackend::Cpu
            }
        }
        "gpu" | "auto" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else if has_rocm {
                AcceleratorBackend::Rocm
            } else if has_wgpu {
                AcceleratorBackend::Wgpu
            } else {
                // "auto" / "gpu" with no GPU is the documented contract for
                // CPU-only hosts; log at info so it's still observable.
                tracing::info!(
                    target: "forex_core::backends",
                    requested = preference,
                    "no GPU device available; using CPU backend"
                );
                AcceleratorBackend::Cpu
            }
        }
        other if has_cuda => {
            warn_downgrade(
                other,
                "unknown preference; CUDA available, falling back to it",
            );
            AcceleratorBackend::Cuda
        }
        other if has_wgpu => {
            warn_downgrade(
                other,
                "unknown preference; wgpu available, falling back to it",
            );
            AcceleratorBackend::Wgpu
        }
        other if has_rocm => {
            warn_downgrade(
                other,
                "unknown preference; ROCm available, falling back to it",
            );
            AcceleratorBackend::Rocm
        }
        other => {
            warn_downgrade(other, "unknown preference and no GPU available");
            AcceleratorBackend::Cpu
        }
    }
}
