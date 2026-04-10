// Model Registry
//
// The registry now resolves model names to capability records first and only
// derives compatibility metadata from those records.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tracing::warn;

use crate::runtime::capabilities::{
    model_capability, CapabilityState, ModelCapability, ModelFamily, KNOWN_MODEL_NAMES,
};

fn dynamic_registry() -> &'static Mutex<HashMap<String, ModelCapability>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, ModelCapability>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_registry_settings() -> forex_core::Settings {
    match forex_core::Settings::load_with_env() {
        Ok(settings) => settings,
        Err(err) => {
            warn!(
                "failed to load settings for model registry/runtime device selection: {err}; falling back to defaults"
            );
            forex_core::Settings::default()
        }
    }
}

fn infer_dynamic_family(name: &str, module_path: &str, class_name: &str) -> Option<ModelFamily> {
    let haystack = format!(
        "{} {} {}",
        name.to_ascii_lowercase(),
        module_path.to_ascii_lowercase(),
        class_name.to_ascii_lowercase()
    );
    if haystack.contains("lightgbm")
        || haystack.contains("xgboost")
        || haystack.contains("catboost")
        || haystack.contains("tree")
    {
        Some(ModelFamily::Tree)
    } else if haystack.contains("swarm") || haystack.contains("forecast") {
        Some(ModelFamily::Forecasting)
    } else if haystack.contains("mlp")
        || haystack.contains("nbeats")
        || haystack.contains("tide")
        || haystack.contains("tabnet")
        || haystack.contains("transformer")
        || haystack.contains("patch")
        || haystack.contains("timesnet")
        || haystack.contains("kan")
    {
        Some(ModelFamily::Deep)
    } else if haystack.contains("meta")
        || haystack.contains("calibr")
        || haystack.contains("conformal")
        || haystack.contains("bayes")
        || haystack.contains("logit")
        || haystack.contains("elastic")
    {
        Some(ModelFamily::Meta)
    } else if haystack.contains("genetic")
        || haystack.contains("evo")
        || haystack.contains("crfmnes")
        || haystack.contains("neat")
    {
        Some(ModelFamily::Evolutionary)
    } else if haystack.contains("exit") {
        Some(ModelFamily::Exit)
    } else if haystack.contains("adaptive")
        || haystack.contains("online")
        || haystack.contains("hoeffding")
        || haystack.contains("passive")
    {
        Some(ModelFamily::Adaptive)
    } else if haystack.contains("anomaly")
        || haystack.contains("forest")
        || haystack.contains("isolation")
    {
        Some(ModelFamily::Anomaly)
    } else {
        None
    }
}

/// Capability-aware categories used by legacy helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    Tree,
    Deep,
    Forecasting,
    Meta,
    Evolutionary,
    Exit,
    Adaptive,
    Anomaly,
    Rl,
}

impl From<ModelFamily> for ModelCategory {
    fn from(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Tree => Self::Tree,
            ModelFamily::Deep => Self::Deep,
            ModelFamily::Forecasting => Self::Forecasting,
            ModelFamily::Meta => Self::Meta,
            ModelFamily::Evolutionary => Self::Evolutionary,
            ModelFamily::Exit => Self::Exit,
            ModelFamily::Adaptive => Self::Adaptive,
            ModelFamily::Anomaly => Self::Anomaly,
            ModelFamily::Rl => Self::Rl,
        }
    }
}

/// Model metadata resolved from the capability layer.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub capability: ModelCapability,
    pub category: ModelCategory,
    pub supports_gpu: bool,
    pub prefers_gpu: bool,
    pub description: String,
}

fn description_for_capability(capability: &ModelCapability) -> String {
    let kind = match capability.family {
        ModelFamily::Tree => "tree ensemble",
        ModelFamily::Deep => "deep sequence model",
        ModelFamily::Forecasting => "forecasting ensemble",
        ModelFamily::Meta => "meta/statistical model",
        ModelFamily::Evolutionary => "evolutionary search model",
        ModelFamily::Exit => "exit policy model",
        ModelFamily::Adaptive => "adaptive online model",
        ModelFamily::Anomaly => "anomaly detector",
        ModelFamily::Rl => "reinforcement learning policy",
    };

    format!(
        "{} capability for {} ({})",
        capability.name, kind, capability.state
    )
}

fn supports_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => cfg!(feature = "lightgbm-gpu"),
        "xgboost" | "xgboost_rf" | "xgboost_dart" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(feature = "burn-wgpu-backend"),
            _ => false,
        },
    }
}

fn prefers_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => cfg!(feature = "lightgbm-gpu"),
        "xgboost" | "xgboost_rf" | "xgboost_dart" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(feature = "burn-wgpu-backend"),
            _ => false,
        },
    }
}

fn default_gpu_device_for_capability(capability: &ModelCapability) -> &'static str {
    match capability.family {
        ModelFamily::Tree | ModelFamily::Rl => "cuda:0",
        ModelFamily::Deep | ModelFamily::Exit => {
            if cfg!(feature = "burn-wgpu-backend") {
                "wgpu"
            } else {
                "cpu"
            }
        }
        _ => "cpu",
    }
}

fn gpu_runtime_available_for_capability(capability: &ModelCapability) -> bool {
    match capability.family {
        ModelFamily::Deep | ModelFamily::Exit => cfg!(feature = "burn-wgpu-backend"),
        _ => crate::tree_models::config::gpu_count() > 0,
    }
}

fn normalize_recommended_gpu_device(
    configured_device: &str,
    capability: &ModelCapability,
) -> Option<String> {
    let normalized = configured_device.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "auto" {
        return None;
    }

    match capability.family {
        ModelFamily::Deep | ModelFamily::Exit => {
            if normalized == "gpu"
                || normalized == "wgpu"
                || normalized == "wgpu_vulkan"
                || normalized == "wgpu_dx12"
                || normalized == "wgpu_metal"
            {
                Some("wgpu".to_string())
            } else {
                None
            }
        }
        ModelFamily::Tree | ModelFamily::Rl => {
            if normalized == "gpu" || normalized == "cuda" {
                Some("cuda:0".to_string())
            } else if normalized.starts_with("cuda:") {
                Some(normalized)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve capability-backed model information.
pub fn get_model_info(name: &str) -> Option<ModelInfo> {
    let capability = get_model_capability(name)?;
    let category = capability.family.into();
    let supports_gpu = supports_gpu_for_model(&capability.name, capability.family);
    let prefers_gpu = prefers_gpu_for_model(&capability.name, capability.family);
    let description = description_for_capability(&capability);

    Some(ModelInfo {
        name: capability.name.clone(),
        capability,
        category,
        supports_gpu,
        prefers_gpu,
        description,
    })
}

/// Resolve the capability record for a known model name.
pub fn get_model_capability(name: &str) -> Option<ModelCapability> {
    model_capability(name).or_else(|| {
        dynamic_registry()
            .lock()
            .ok()
            .and_then(|registry| registry.get(name).cloned())
    })
}

fn default_inventory_names(settings: &forex_core::Settings) -> Vec<String> {
    let mut names = KNOWN_MODEL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();

    let num_transformers = settings.models.num_transformers.max(1);
    if num_transformers > 1 {
        for replica_idx in 1..=num_transformers {
            names.push(format!("transformer_{replica_idx:02}"));
        }
    }

    names
}

/// List all available models by capability family.
pub fn list_models_by_category() -> HashMap<ModelCategory, Vec<String>> {
    let mut result = HashMap::new();
    let settings = load_registry_settings();

    for model_name in default_inventory_names(&settings) {
        if let Some(info) = get_model_info(&model_name) {
            result
                .entry(info.category)
                .or_insert_with(Vec::new)
                .push(info.name);
        }
    }

    if let Ok(registry) = dynamic_registry().lock() {
        for capability in registry.values() {
            result
                .entry(capability.family.into())
                .or_insert_with(Vec::new)
                .push(capability.name.clone());
        }
    }

    result
}

/// Check if a model name is valid.
pub fn is_valid_model(name: &str) -> bool {
    get_model_capability(name).is_some()
}

/// Get recommended device for a model.
pub fn get_recommended_device(model_name: &str) -> Result<String> {
    let capability = get_model_capability(model_name)
        .context(format!("Model '{}' not found in registry", model_name))?;
    let settings = load_registry_settings();

    if !settings.system.enable_gpu {
        return Ok("cpu".to_string());
    }

    let configured_device = settings.system.device.trim();
    if configured_device.eq_ignore_ascii_case("cpu") {
        return Ok("cpu".to_string());
    }

    if !supports_gpu_for_model(&capability.name, capability.family) {
        return Ok("cpu".to_string());
    }

    if prefers_gpu_for_model(&capability.name, capability.family)
        && gpu_runtime_available_for_capability(&capability)
    {
        if let Some(device) = normalize_recommended_gpu_device(configured_device, &capability) {
            return Ok(device);
        }
        if !configured_device.is_empty() && !configured_device.eq_ignore_ascii_case("auto") {
            return Ok(default_gpu_device_for_capability(&capability).to_string());
        }
        return Ok(default_gpu_device_for_capability(&capability).to_string());
    }

    Ok("cpu".to_string())
}

// ============================================================================
// PYTHON COMPATIBILITY
// ============================================================================

/// Compatibility hook for dynamic model registration.
/// Built-in models stay statically registered; custom models are appended here.
pub fn register_model(name: &str, module_path: &str, class_name: &str) -> Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("model registration requires a non-empty name");
    }

    if get_model_capability(name).is_some() {
        return Ok(());
    }

    let family = infer_dynamic_family(name, module_path, class_name).with_context(|| {
        format!(
            "dynamic model registration requires an inferable family for {name} ({module_path}::{class_name})"
        )
    })?;
    let capability = ModelCapability::new(name.trim(), family, CapabilityState::Planned);
    let mut registry = dynamic_registry()
        .lock()
        .map_err(|_| anyhow::anyhow!("dynamic model registry mutex poisoned"))?;
    registry.insert(name.trim().to_string(), capability);
    Ok(())
}

// ============================================================================
// SUMMARY
// ============================================================================
//
// The registry now resolves every known configured model name to a
// capability-backed record. Legacy helpers remain available, but they are
// derived from the runtime capability layer instead of being a loose string
// table.
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::capabilities::{CapabilityState, ModelFamily};
    use forex_core::Settings;

    #[test]
    fn known_configured_models_resolve_to_capabilities() {
        let expectations = [
            ("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
            ("xgboost", ModelFamily::Tree, CapabilityState::Implemented),
            (
                "xgboost_rf",
                ModelFamily::Tree,
                CapabilityState::Implemented,
            ),
            (
                "xgboost_dart",
                ModelFamily::Tree,
                CapabilityState::Implemented,
            ),
            ("catboost", ModelFamily::Tree, CapabilityState::Implemented),
            (
                "catboost_alt",
                ModelFamily::Tree,
                CapabilityState::Implemented,
            ),
            (
                "sklears_tree",
                ModelFamily::Tree,
                CapabilityState::Implemented,
            ),
            ("mlp", ModelFamily::Deep, CapabilityState::Implemented),
            (
                "elasticnet",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "bayes_logit",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "meta_blender",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "probability_calibrator",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "conformal_gate",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "meta_stack",
                ModelFamily::Meta,
                CapabilityState::Implemented,
            ),
            (
                "genetic",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
            ),
            (
                "exit_agent",
                ModelFamily::Exit,
                CapabilityState::Implemented,
            ),
            (
                "online_pa",
                ModelFamily::Adaptive,
                CapabilityState::Implemented,
            ),
            (
                "online_hoeffding",
                ModelFamily::Adaptive,
                CapabilityState::Implemented,
            ),
            (
                "isolation_forest",
                ModelFamily::Anomaly,
                CapabilityState::Implemented,
            ),
            ("dqn", ModelFamily::Rl, CapabilityState::Implemented),
            (
                "transformer",
                ModelFamily::Deep,
                CapabilityState::Implemented,
            ),
            ("nbeats", ModelFamily::Deep, CapabilityState::Implemented),
            ("tide", ModelFamily::Deep, CapabilityState::Implemented),
            ("tabnet", ModelFamily::Deep, CapabilityState::Implemented),
            ("kan", ModelFamily::Deep, CapabilityState::Implemented),
            ("patchtst", ModelFamily::Deep, CapabilityState::Implemented),
            ("timesnet", ModelFamily::Deep, CapabilityState::Implemented),
            (
                "nbeatsx_nf",
                ModelFamily::Deep,
                CapabilityState::Implemented,
            ),
            ("tide_nf", ModelFamily::Deep, CapabilityState::Implemented),
            (
                "swarm_forecaster",
                ModelFamily::Forecasting,
                CapabilityState::Implemented,
            ),
            (
                "neuro_evo",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
            ),
            (
                "neat",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
            ),
        ];

        for (name, family, state) in expectations {
            let capability = get_model_capability(name)
                .unwrap_or_else(|| panic!("missing capability for {name}"));

            assert_eq!(capability.name, name);
            assert_eq!(capability.family, family);
            assert_eq!(capability.state, state);
        }
    }

    #[test]
    fn all_default_configured_model_names_have_capabilities() {
        let settings = Settings::default();

        for name in &settings.models.ml_models {
            assert!(
                get_model_capability(name).is_some(),
                "configured model {name} should resolve to a capability"
            );
        }
    }

    #[test]
    fn dynamic_registration_adds_capability_entry() {
        register_model("custom_patch_router", "custom.models.patch", "PatchRouter")
            .expect("dynamic model registration should succeed");

        let capability = get_model_capability("custom_patch_router")
            .expect("dynamic capability should be discoverable");
        assert_eq!(capability.family, ModelFamily::Deep);
        assert_eq!(capability.state, CapabilityState::Planned);
    }

    #[test]
    fn dynamic_registration_rejects_unknown_family_inference() {
        let err = register_model("custom_unknown", "custom.models.misc", "Router")
            .expect_err("unknown dynamic model family must fail");
        assert!(err.to_string().contains("inferable family"));
    }

    #[test]
    fn default_inventory_names_follow_runtime_transformer_replica_settings() {
        let mut settings = Settings::default();
        settings.models.num_transformers = 3;

        let names = default_inventory_names(&settings);
        assert!(names.contains(&"transformer_01".to_string()));
        assert!(names.contains(&"transformer_02".to_string()));
        assert!(names.contains(&"transformer_03".to_string()));
    }

    #[test]
    fn default_inventory_lists_transformer_replicas() {
        let listed = list_models_by_category();
        let deep = listed
            .get(&ModelCategory::Deep)
            .expect("deep category should exist");
        assert!(
            deep.iter().any(|name| name == "transformer_01"),
            "default inventory should expose transformer_01"
        );
    }

    #[test]
    fn known_model_names_are_unique_and_resolve() {
        let mut seen = std::collections::HashSet::new();
        for name in KNOWN_MODEL_NAMES {
            assert!(seen.insert(*name), "duplicate known model name {name}");
            assert!(
                get_model_capability(name).is_some(),
                "known model name {name} should resolve"
            );
        }
    }

    #[test]
    fn normalize_recommended_gpu_device_maps_generic_tokens_per_family() {
        let deep = ModelCapability::new("mlp", ModelFamily::Deep, CapabilityState::Implemented);
        let tree =
            ModelCapability::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented);
        let rl = ModelCapability::new("dqn", ModelFamily::Rl, CapabilityState::Implemented);

        assert_eq!(
            normalize_recommended_gpu_device("gpu", &deep).as_deref(),
            Some("wgpu")
        );
        assert_eq!(
            normalize_recommended_gpu_device("wgpu_vulkan", &deep).as_deref(),
            Some("wgpu")
        );
        assert_eq!(
            normalize_recommended_gpu_device("gpu", &tree).as_deref(),
            Some("cuda:0")
        );
        assert_eq!(
            normalize_recommended_gpu_device("cuda:2", &rl).as_deref(),
            Some("cuda:2")
        );
        assert!(
            normalize_recommended_gpu_device("wgpu", &tree).is_none(),
            "tree models should not claim wgpu devices"
        );
    }
}
