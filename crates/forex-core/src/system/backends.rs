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

    match preference {
        "cuda" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "rocm" => {
            if has_rocm {
                AcceleratorBackend::Rocm
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "wgpu" => {
            if has_wgpu {
                AcceleratorBackend::Wgpu
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "vulkan" => {
            if has_wgpu {
                AcceleratorBackend::Vulkan
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "metal" => {
            if has_wgpu {
                AcceleratorBackend::Metal
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "dx12" => {
            if has_wgpu {
                AcceleratorBackend::Dx12
            } else {
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
                AcceleratorBackend::Cpu
            }
        }
        _ if has_cuda => AcceleratorBackend::Cuda,
        _ if has_wgpu => AcceleratorBackend::Wgpu,
        _ if has_rocm => AcceleratorBackend::Rocm,
        _ => AcceleratorBackend::Cpu,
    }
}
