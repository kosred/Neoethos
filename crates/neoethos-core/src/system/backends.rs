use serde::{Deserialize, Serialize};

use crate::contracts::BackendKind;

use super::HardwareProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorBackend {
    Cpu,
    Cuda,
    /// Explicit model-only assignment; not a global Data/Search backend switch.
    Rocm,
}

impl AcceleratorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }

    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub fn backend_kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::NativeCpu,
            Self::Cuda => BackendKind::NativeCuda,
            Self::Rocm => BackendKind::NativeRocm,
        }
    }
}

pub(super) fn normalize_accelerator_preference(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => "auto".to_string(),
        "cpu" | "false" | "0" | "no" | "off" => "cpu".to_string(),
        "gpu" | "true" | "1" | "yes" | "on" => "gpu".to_string(),
        "cuda" | "cuda:0" | "nvidia" => "cuda".to_string(),
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
    // F-CORE2-014: previously these branches silently downgraded to CPU when
    // the user explicitly asked for a GPU backend that wasn't probed. That
    // makes hour-long discovery runs land on CPU without any signal. Emit a
    // structured warn at the decision site so the downgrade is visible.
    fn warn_downgrade(requested: &str, reason: &str) {
        tracing::warn!(
            target: "neoethos_core::backends",
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
        "gpu" | "auto" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else {
                // "auto" / "gpu" with no GPU is the documented contract for
                // CPU-only hosts; log at info so it's still observable.
                tracing::info!(
                    target: "neoethos_core::backends",
                    requested = preference,
                    "no GPU device available; using CPU backend"
                );
                AcceleratorBackend::Cpu
            }
        }
        other => {
            warn_downgrade(
                other,
                "unsupported or retired accelerator preference; CUDA is the only active GPU backend",
            );
            AcceleratorBackend::Cpu
        }
    }
}
