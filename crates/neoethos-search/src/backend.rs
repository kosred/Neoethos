//! Typed discovery-evaluation backend policy.
//!
//! Stage 1 / Commit 0.1 of the GPU-native discovery redesign.  The historical
//! configuration exposed overlapping string knobs (`system.enable_gpu_preference`
//! and `models.prop_search_device`) plus `NEOETHOS_REQUIRE_GPU`.  This module is
//! the single typed boundary that resolves those inputs without changing the
//! legacy meanings of `cpu`, `auto`, or `gpu`.

use neoethos_core::Settings;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePreference {
    Cpu,
    Auto,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackPolicy {
    AllowCpu,
    ForbidCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorHint {
    Any,
    Cuda,
    Wgpu,
    Vulkan,
    Rocm,
    Metal,
    Dx12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvaluationBackend {
    pub device: DevicePreference,
    pub fallback: FallbackPolicy,
    pub accelerator_hint: AcceleratorHint,
}

impl Default for EvaluationBackend {
    fn default() -> Self {
        Self::AUTO
    }
}

impl EvaluationBackend {
    pub const CPU_CANONICAL: Self = Self {
        device: DevicePreference::Cpu,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const AUTO: Self = Self {
        device: DevicePreference::Auto,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const GPU_PREFERRED: Self = Self {
        device: DevicePreference::Gpu,
        fallback: FallbackPolicy::AllowCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub const GPU_REQUIRED: Self = Self {
        device: DevicePreference::Gpu,
        fallback: FallbackPolicy::ForbidCpu,
        accelerator_hint: AcceleratorHint::Any,
    };

    pub fn parse(raw: &str) -> Result<Self, BackendConfigError> {
        let normalized = normalize(raw);
        let parsed = match normalized.as_str() {
            "" | "auto" => Self::AUTO,
            "cpu" | "off" | "false" => Self::CPU_CANONICAL,
            "gpu" | "on" | "true" => Self::GPU_PREFERRED,
            "gpu_required" | "gpu-required" => Self::GPU_REQUIRED,
            "cuda" => Self::gpu_with(AcceleratorHint::Cuda, FallbackPolicy::AllowCpu),
            "cuda_required" | "cuda-required" => {
                Self::gpu_with(AcceleratorHint::Cuda, FallbackPolicy::ForbidCpu)
            }
            "wgpu" => Self::gpu_with(AcceleratorHint::Wgpu, FallbackPolicy::AllowCpu),
            "wgpu_required" | "wgpu-required" => {
                Self::gpu_with(AcceleratorHint::Wgpu, FallbackPolicy::ForbidCpu)
            }
            "vulkan" => Self::gpu_with(AcceleratorHint::Vulkan, FallbackPolicy::AllowCpu),
            "vulkan_required" | "vulkan-required" => {
                Self::gpu_with(AcceleratorHint::Vulkan, FallbackPolicy::ForbidCpu)
            }
            "rocm" => Self::gpu_with(AcceleratorHint::Rocm, FallbackPolicy::AllowCpu),
            "rocm_required" | "rocm-required" => {
                Self::gpu_with(AcceleratorHint::Rocm, FallbackPolicy::ForbidCpu)
            }
            "metal" => Self::gpu_with(AcceleratorHint::Metal, FallbackPolicy::AllowCpu),
            "metal_required" | "metal-required" => {
                Self::gpu_with(AcceleratorHint::Metal, FallbackPolicy::ForbidCpu)
            }
            "dx12" => Self::gpu_with(AcceleratorHint::Dx12, FallbackPolicy::AllowCpu),
            "dx12_required" | "dx12-required" => {
                Self::gpu_with(AcceleratorHint::Dx12, FallbackPolicy::ForbidCpu)
            }
            _ => return Err(BackendConfigError::UnknownPreference(raw.trim().to_owned())),
        };
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn resolve_for_discovery(
        global_preference: &str,
        discovery_preference: &str,
        require_gpu_env: Option<&str>,
    ) -> Result<Self, BackendConfigError> {
        // The discovery-specific value wins whenever it is present.  This keeps
        // the existing `models.prop_search_device` behaviour intact; an empty
        // value deliberately inherits the global system preference.
        let selected = if discovery_preference.trim().is_empty() {
            global_preference
        } else {
            discovery_preference
        };
        let mut backend = Self::parse(selected)?;

        if parse_optional_bool("NEOETHOS_REQUIRE_GPU", require_gpu_env)?.unwrap_or(false) {
            // The environment override can only escalate.  It forces a GPU as
            // well as forbidding CPU fallback, avoiding the invalid
            // `Cpu + ForbidCpu` state.
            backend.device = DevicePreference::Gpu;
            backend.fallback = FallbackPolicy::ForbidCpu;
        }
        backend.validate()?;
        Ok(backend)
    }

    pub fn from_settings(
        settings: &Settings,
        require_gpu_env: Option<&str>,
    ) -> Result<Self, BackendConfigError> {
        Self::resolve_for_discovery(
            &settings.system.enable_gpu_preference,
            &settings.models.prop_search_device,
            require_gpu_env,
        )
    }

    pub fn from_settings_and_process_env(settings: &Settings) -> Result<Self, BackendConfigError> {
        let env_value = std::env::var("NEOETHOS_REQUIRE_GPU").ok();
        Self::from_settings(settings, env_value.as_deref())
    }

    pub fn cpu_fallback_allowed(self) -> bool {
        self.fallback == FallbackPolicy::AllowCpu
    }

    pub fn gpu_required(self) -> bool {
        self.device == DevicePreference::Gpu && self.fallback == FallbackPolicy::ForbidCpu
    }

    pub fn validate(self) -> Result<(), BackendConfigError> {
        if self.device == DevicePreference::Cpu && self.fallback == FallbackPolicy::ForbidCpu {
            return Err(BackendConfigError::ConflictingPolicy {
                device: self.device,
                fallback: self.fallback,
            });
        }
        Ok(())
    }

    const fn gpu_with(hint: AcceleratorHint, fallback: FallbackPolicy) -> Self {
        Self {
            device: DevicePreference::Gpu,
            fallback,
            accelerator_hint: hint,
        }
    }
}

/// Transitional typed entry point. Commit 0.1 deliberately preserves the
/// evaluator's runtime behaviour; Commit 0.2 consumes the policy in the GPU
/// failure/fallback decision and later commits thread it through every stage.
pub fn evaluate_population_core_with_backend(
    inputs: crate::eval::PopulationEvalInputs<'_>,
    backend: EvaluationBackend,
) -> Result<Vec<[f64; 11]>, String> {
    backend.validate().map_err(|error| error.to_string())?;
    crate::eval::evaluate_population_core(inputs)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendConfigError {
    UnknownPreference(String),
    InvalidBoolean { key: &'static str, value: String },
    ConflictingPolicy {
        device: DevicePreference,
        fallback: FallbackPolicy,
    },
}

impl fmt::Display for BackendConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPreference(value) => write!(
                f,
                "unknown discovery compute preference `{value}`; expected cpu, auto, gpu, gpu_required, or a supported accelerator hint"
            ),
            Self::InvalidBoolean { key, value } => write!(
                f,
                "invalid boolean value `{value}` for {key}; expected 1/0, true/false, yes/no, or on/off"
            ),
            Self::ConflictingPolicy { device, fallback } => write!(
                f,
                "invalid evaluation backend policy: device={device:?}, fallback={fallback:?}"
            ),
        }
    }
}

impl Error for BackendConfigError {}

fn normalize(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

fn parse_optional_bool(
    key: &'static str,
    raw: Option<&str>,
) -> Result<Option<bool>, BackendConfigError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let normalized = normalize(raw);
    match normalized.as_str() {
        "" => Ok(None),
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(BackendConfigError::InvalidBoolean {
            key,
            value: raw.trim().to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_values_keep_their_meaning() {
        assert_eq!(EvaluationBackend::parse("cpu").unwrap(), EvaluationBackend::CPU_CANONICAL);
        assert_eq!(EvaluationBackend::parse("auto").unwrap(), EvaluationBackend::AUTO);
        assert_eq!(EvaluationBackend::parse("gpu").unwrap(), EvaluationBackend::GPU_PREFERRED);
    }

    #[test]
    fn gpu_required_is_a_new_strict_value() {
        assert_eq!(EvaluationBackend::parse("gpu_required").unwrap(), EvaluationBackend::GPU_REQUIRED);
        assert!(EvaluationBackend::parse("gpu_required").unwrap().gpu_required());
    }

    #[test]
    fn accelerator_hint_is_preserved() {
        let backend = EvaluationBackend::parse("cuda_required").unwrap();
        assert_eq!(backend.device, DevicePreference::Gpu);
        assert_eq!(backend.fallback, FallbackPolicy::ForbidCpu);
        assert_eq!(backend.accelerator_hint, AcceleratorHint::Cuda);
    }

    #[test]
    fn discovery_specific_value_overrides_global() {
        let backend = EvaluationBackend::resolve_for_discovery("gpu", "cpu", None).unwrap();
        assert_eq!(backend, EvaluationBackend::CPU_CANONICAL);

        let inherited = EvaluationBackend::resolve_for_discovery("gpu", "", None).unwrap();
        assert_eq!(inherited, EvaluationBackend::GPU_PREFERRED);
    }

    #[test]
    fn require_gpu_env_only_escalates() {
        for value in ["1", "true", "YES", "on"] {
            let backend =
                EvaluationBackend::resolve_for_discovery("auto", "cpu", Some(value)).unwrap();
            assert_eq!(backend.device, DevicePreference::Gpu);
            assert_eq!(backend.fallback, FallbackPolicy::ForbidCpu);
        }
    }

    #[test]
    fn false_and_empty_env_values_do_not_escalate() {
        for value in ["0", "false", "NO", "off", ""] {
            let backend =
                EvaluationBackend::resolve_for_discovery("auto", "gpu", Some(value)).unwrap();
            assert_eq!(backend, EvaluationBackend::GPU_PREFERRED);
        }
    }

    #[test]
    fn invalid_boolean_fails_loud() {
        let error = EvaluationBackend::resolve_for_discovery("auto", "gpu", Some("maybe"))
            .unwrap_err();
        assert!(matches!(error, BackendConfigError::InvalidBoolean { .. }));
    }

    #[test]
    fn cpu_forbid_cpu_is_rejected() {
        let invalid = EvaluationBackend {
            device: DevicePreference::Cpu,
            fallback: FallbackPolicy::ForbidCpu,
            accelerator_hint: AcceleratorHint::Any,
        };
        assert!(matches!(
            invalid.validate(),
            Err(BackendConfigError::ConflictingPolicy { .. })
        ));
    }
}
