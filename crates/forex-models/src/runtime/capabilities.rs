use serde::{Deserialize, Serialize};
use std::fmt;

use forex_core::{BackendKind, RuntimeDegradedReason, RuntimeMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
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

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Tree => "tree",
            Self::Deep => "deep",
            Self::Forecasting => "forecasting",
            Self::Meta => "meta",
            Self::Evolutionary => "evolutionary",
            Self::Exit => "exit",
            Self::Adaptive => "adaptive",
            Self::Anomaly => "anomaly",
            Self::Rl => "rl",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityState {
    Planned,
    Implemented,
    Verified,
}

impl fmt::Display for CapabilityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Planned => "planned",
            Self::Implemented => "implemented",
            Self::Verified => "verified",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapability {
    pub name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
}

impl ModelCapability {
    pub fn new(name: impl Into<String>, family: ModelFamily, state: CapabilityState) -> Self {
        let name = name.into();
        assert!(
            !name.trim().is_empty(),
            "model capability requires a non-empty name"
        );
        Self {
            name,
            family,
            state,
        }
    }
}

pub const KNOWN_MODEL_NAMES: &[&str] = &[
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
    "sklears_tree",
    "mlp",
    "nbeats",
    "tide",
    "tabnet",
    "kan",
    "transformer",
    "patchtst",
    "timesnet",
    "nbeatsx_nf",
    "tide_nf",
    "swarm_forecaster",
    "elasticnet",
    "logistic",
    "bayes_logit",
    "meta_blender",
    "probability_calibrator",
    "conformal_gate",
    "meta_stack",
    "genetic",
    "neuro_evo",
    "neat",
    "exit_agent",
    "online_pa",
    "online_hoeffding",
    "isolation_forest",
    "dqn",
];

pub fn normalize_runtime_device_policy(policy: &str) -> String {
    crate::common::normalize_vendor_device_policy(policy, &[])
}

pub fn requested_runtime_device_policy(model_name: &str) -> String {
    let model_key = format!(
        "FOREX_BOT_{}_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    let requested = std::env::var(&model_key)
        .or_else(|_| std::env::var("FOREX_BOT_META_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string());
    normalize_runtime_device_policy(&requested)
}

pub fn append_runtime_degraded_reason(
    degraded_reason: Option<String>,
    appended_reason: Option<String>,
) -> Option<String> {
    match (degraded_reason, appended_reason) {
        (Some(primary), Some(secondary)) => Some(format!("{primary}; {secondary}")),
        (Some(primary), None) => Some(primary),
        (None, Some(secondary)) => Some(secondary),
        (None, None) => None,
    }
}

pub fn runtime_backend_kind_from_label(label: Option<&str>) -> Option<BackendKind> {
    let label = label?.trim();
    if label.is_empty() {
        return None;
    }

    let normalized = label.to_ascii_lowercase();
    if normalized.contains("unknown") || normalized.contains("unavailable") {
        return Some(BackendKind::Unavailable);
    }
    if normalized.contains("fallback")
        || normalized.contains("simple_es_restart")
        || normalized.contains("diagonal_profile")
    {
        return Some(BackendKind::LocalSurrogateFallback);
    }
    if normalized.contains("cuda_kernel") || normalized.contains("cuda_fitness") {
        return Some(BackendKind::CudaKernel);
    }
    if normalized.contains("tree") && (normalized.contains("gpu") || normalized.contains("cuda")) {
        return Some(BackendKind::NativeTreeGpu);
    }
    if normalized.contains("tree") && normalized.contains("cpu") {
        return Some(BackendKind::NativeTreeCpu);
    }
    if normalized.contains("wgpu") {
        return Some(BackendKind::BurnWgpu);
    }
    if normalized.contains("burn") && normalized.contains("cpu") {
        return Some(BackendKind::BurnCpu);
    }
    if normalized.contains("cuda") || normalized.contains("gpu") {
        return Some(BackendKind::NativeCuda);
    }
    if normalized.contains("cpu") {
        return Some(BackendKind::NativeCpu);
    }
    if normalized.contains("external") || normalized.contains("swarm") {
        return Some(BackendKind::ExternalRuntime);
    }

    Some(BackendKind::ExternalRuntime)
}

pub fn runtime_mode_from_details(
    backend_kind: Option<BackendKind>,
    degraded_reason: Option<&str>,
) -> Option<RuntimeMode> {
    let has_runtime_details = backend_kind.is_some() || degraded_reason.is_some();
    if !has_runtime_details {
        return None;
    }

    if backend_kind.is_some_and(BackendKind::is_degraded) || degraded_reason.is_some() {
        Some(RuntimeMode::Degraded)
    } else {
        Some(RuntimeMode::Canonical)
    }
}

pub fn typed_runtime_degraded_reason(reason: Option<&str>) -> Option<RuntimeDegradedReason> {
    let reason = reason?.trim();
    if reason.is_empty() {
        return None;
    }

    let first_segment = reason
        .split(';')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("runtime_degraded");
    let mut code = first_segment
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    while code.contains("__") {
        code = code.replace("__", "_");
    }
    let code = code.trim_matches('_');
    let code = if code.is_empty() {
        "runtime_degraded"
    } else {
        code
    };

    Some(RuntimeDegradedReason::new(code, reason))
}

pub fn gpu_policy_cpu_fallback_reason(model_name: &str) -> Option<String> {
    let normalized = requested_runtime_device_policy(model_name);
    if normalized == "gpu" || normalized.starts_with("gpu:") {
        Some(format!(
            "requested device policy `{normalized}`; runtime currently executes on CPU"
        ))
    } else {
        None
    }
}

pub fn normalize_training_precision_policy(policy: &str) -> String {
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return "auto".to_string();
    }
    match normalized.as_str() {
        "auto" | "fp32" | "bf16" | "fp8" | "bf4" => normalized,
        "float32" | "f32" => "fp32".to_string(),
        "bfloat16" => "bf16".to_string(),
        "float8" => "fp8".to_string(),
        _ => "auto".to_string(),
    }
}

pub fn requested_training_precision_policy(model_name: &str) -> String {
    let model_key = format!(
        "FOREX_BOT_{}_TRAIN_PRECISION",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    let requested = std::env::var(&model_key)
        .or_else(|_| std::env::var("FOREX_BOT_TRAIN_PRECISION"))
        .or_else(|_| std::env::var("FOREX_TRAIN_PRECISION"))
        .unwrap_or_else(|_| "auto".to_string());
    normalize_training_precision_policy(&requested)
}

pub fn model_capability(name: &str) -> Option<ModelCapability> {
    let is_transformer_replica = name
        .strip_prefix("transformer_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()));

    let capability = match name {
        // Tree models
        "lightgbm" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "xgboost" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "xgboost_rf" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "xgboost_dart" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "catboost" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "catboost_alt" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),
        "sklears_tree" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Verified),

        // Deep models
        "mlp" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "nbeats" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "tide" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "tabnet" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "kan" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "transformer" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        _ if is_transformer_replica => {
            ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified)
        }
        "patchtst" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "timesnet" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "nbeatsx_nf" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),
        "tide_nf" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Verified),

        // Forecasting models
        "swarm_forecaster" => {
            ModelCapability::new(name, ModelFamily::Forecasting, CapabilityState::Verified)
        }

        // Meta models
        "elasticnet" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified),
        "logistic" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified),
        "bayes_logit" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified),
        "meta_blender" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified),
        "probability_calibrator" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified)
        }
        "conformal_gate" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified)
        }
        "meta_stack" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Verified),

        // Evolutionary models
        "genetic" => ModelCapability::new(
            name,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        ),
        "neuro_evo" => ModelCapability::new(
            name,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        ),
        "neat" => ModelCapability::new(name, ModelFamily::Evolutionary, CapabilityState::Verified),

        // Exit models
        "exit_agent" => ModelCapability::new(name, ModelFamily::Exit, CapabilityState::Verified),

        // Adaptive models
        "online_pa" => ModelCapability::new(name, ModelFamily::Adaptive, CapabilityState::Verified),
        "online_hoeffding" => {
            ModelCapability::new(name, ModelFamily::Adaptive, CapabilityState::Verified)
        }

        // Anomaly models
        "isolation_forest" => {
            ModelCapability::new(name, ModelFamily::Anomaly, CapabilityState::Verified)
        }

        // Reinforcement-learning models
        "dqn" => ModelCapability::new(name, ModelFamily::Rl, CapabilityState::Verified),

        _ => return None,
    };

    Some(capability)
}

#[cfg(test)]
mod tests {
    use super::{
        CapabilityState, KNOWN_MODEL_NAMES, ModelCapability, ModelFamily,
        append_runtime_degraded_reason, gpu_policy_cpu_fallback_reason, model_capability,
        normalize_runtime_device_policy, normalize_training_precision_policy,
        requested_training_precision_policy, runtime_backend_kind_from_label,
        runtime_mode_from_details, typed_runtime_degraded_reason,
    };
    use forex_core::{BackendKind, RuntimeMode};
    use std::collections::HashSet;

    #[test]
    fn model_family_has_expected_variants() {
        assert_eq!(ModelFamily::Tree.to_string(), "tree");
        assert_eq!(ModelFamily::Deep.to_string(), "deep");
        assert_eq!(ModelFamily::Forecasting.to_string(), "forecasting");
        assert_eq!(ModelFamily::Meta.to_string(), "meta");
        assert_eq!(ModelFamily::Evolutionary.to_string(), "evolutionary");
        assert_eq!(ModelFamily::Exit.to_string(), "exit");
        assert_eq!(ModelFamily::Adaptive.to_string(), "adaptive");
        assert_eq!(ModelFamily::Anomaly.to_string(), "anomaly");
        assert_eq!(ModelFamily::Rl.to_string(), "rl");
    }

    #[test]
    fn capability_state_has_expected_variants() {
        assert_eq!(CapabilityState::Planned.to_string(), "planned");
        assert_eq!(CapabilityState::Implemented.to_string(), "implemented");
        assert_eq!(CapabilityState::Verified.to_string(), "verified");
    }

    #[test]
    fn model_capability_can_be_constructed() {
        let capability =
            ModelCapability::new("lightgbm", ModelFamily::Tree, CapabilityState::Planned);

        assert_eq!(capability.name, "lightgbm");
        assert_eq!(capability.family, ModelFamily::Tree);
        assert_eq!(capability.state, CapabilityState::Planned);
    }

    #[test]
    fn known_configured_model_names_resolve_to_capabilities() {
        let capability = model_capability("transformer").expect("transformer should resolve");
        assert_eq!(capability.family, ModelFamily::Deep);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("sklears_tree").expect("sklears_tree should resolve");
        assert_eq!(capability.family, ModelFamily::Tree);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("patchtst").expect("patchtst should resolve");
        assert_eq!(capability.family, ModelFamily::Deep);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability =
            model_capability("swarm_forecaster").expect("swarm_forecaster should resolve");
        assert_eq!(capability.family, ModelFamily::Forecasting);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("online_pa").expect("online_pa should resolve");
        assert_eq!(capability.family, ModelFamily::Adaptive);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("elasticnet").expect("elasticnet should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("logistic").expect("logistic should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("bayes_logit").expect("bayes_logit should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("meta_blender").expect("meta_blender should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("probability_calibrator")
            .expect("probability_calibrator should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("conformal_gate").expect("conformal_gate should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("meta_stack").expect("meta_stack should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("genetic").expect("genetic should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("neuro_evo").expect("neuro_evo should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("neat").expect("neat should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("exit_agent").expect("exit_agent should resolve");
        assert_eq!(capability.family, ModelFamily::Exit);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability =
            model_capability("isolation_forest").expect("isolation_forest should resolve");
        assert_eq!(capability.family, ModelFamily::Anomaly);
        assert_eq!(capability.state, CapabilityState::Verified);

        let capability = model_capability("dqn").expect("dqn should resolve");
        assert_eq!(capability.family, ModelFamily::Rl);
        assert_eq!(capability.state, CapabilityState::Verified);
    }

    #[test]
    fn known_model_names_are_unique_and_resolve() {
        let mut seen = HashSet::new();
        for name in KNOWN_MODEL_NAMES {
            assert!(seen.insert(*name), "duplicate known model name {name}");
            assert!(
                model_capability(name).is_some(),
                "known model name {name} should resolve"
            );
        }
    }

    #[test]
    fn tree_model_names_resolve_to_verified_capabilities() {
        let tree_models = [
            "lightgbm",
            "xgboost",
            "xgboost_rf",
            "xgboost_dart",
            "catboost",
            "catboost_alt",
            "sklears_tree",
        ];

        for model in tree_models {
            let capability = model_capability(model).expect("tree model should resolve");
            assert_eq!(capability.family, ModelFamily::Tree);
            assert_eq!(capability.state, CapabilityState::Verified);
        }
    }

    #[test]
    fn statistical_meta_model_names_resolve_to_verified_capabilities() {
        let verified_models = [
            "elasticnet",
            "logistic",
            "bayes_logit",
            "meta_blender",
            "probability_calibrator",
            "conformal_gate",
            "meta_stack",
        ];
        for model in verified_models {
            let capability = model_capability(model).expect("meta model should resolve");
            assert_eq!(capability.family, ModelFamily::Meta);
            assert_eq!(capability.state, CapabilityState::Verified);
        }
    }

    #[test]
    fn deferred_evolutionary_search_models_remain_implemented() {
        for model in ["genetic", "neuro_evo"] {
            let capability = model_capability(model).expect("evolutionary model should resolve");
            assert_eq!(capability.family, ModelFamily::Evolutionary);
            assert_eq!(capability.state, CapabilityState::Implemented);
        }
    }

    #[test]
    fn non_search_models_promoted_to_verified() {
        let verified_models = [
            "mlp",
            "nbeats",
            "tide",
            "tabnet",
            "kan",
            "transformer",
            "patchtst",
            "timesnet",
            "nbeatsx_nf",
            "tide_nf",
            "swarm_forecaster",
            "neat",
            "exit_agent",
            "online_pa",
            "online_hoeffding",
            "isolation_forest",
            "dqn",
        ];

        for model in verified_models {
            let capability = model_capability(model).expect("model should resolve");
            assert_eq!(capability.state, CapabilityState::Verified);
        }
    }

    #[test]
    #[should_panic(expected = "non-empty name")]
    fn model_capability_new_rejects_blank_name() {
        let _ = ModelCapability::new("   ", ModelFamily::Tree, CapabilityState::Implemented);
    }

    #[test]
    fn normalize_runtime_device_policy_accepts_vendor_aliases() {
        assert_eq!(normalize_runtime_device_policy(" CUDA:1 "), "gpu:1");
        assert_eq!(normalize_runtime_device_policy("rocm:2"), "gpu:2");
        assert_eq!(normalize_runtime_device_policy("metal"), "gpu");
        assert_eq!(normalize_runtime_device_policy("vulkan:0"), "gpu:0");
    }

    #[test]
    fn append_runtime_degraded_reason_preserves_primary_and_secondary() {
        assert_eq!(
            append_runtime_degraded_reason(
                Some("primary".to_string()),
                Some("secondary".to_string())
            ),
            Some("primary; secondary".to_string())
        );
        assert_eq!(
            append_runtime_degraded_reason(None, Some("secondary".to_string())),
            Some("secondary".to_string())
        );
    }

    #[test]
    fn gpu_policy_cpu_fallback_reason_detects_model_override() {
        unsafe {
            std::env::set_var("FOREX_BOT_NEAT_DEVICE", "cuda:3");
        }
        let reason = gpu_policy_cpu_fallback_reason("neat");
        unsafe {
            std::env::remove_var("FOREX_BOT_NEAT_DEVICE");
        }
        assert_eq!(
            reason.as_deref(),
            Some("requested device policy `gpu:3`; runtime currently executes on CPU")
        );
    }

    #[test]
    fn runtime_backend_kind_from_label_maps_known_backend_families() {
        assert_eq!(
            runtime_backend_kind_from_label(Some("symbios_neat_cpu")),
            Some(BackendKind::NativeCpu)
        );
        assert_eq!(
            runtime_backend_kind_from_label(Some("simple_es_restart_cpu")),
            Some(BackendKind::LocalSurrogateFallback)
        );
        assert_eq!(
            runtime_backend_kind_from_label(Some("symbios_neat_cuda_fitness")),
            Some(BackendKind::CudaKernel)
        );
        assert_eq!(
            runtime_backend_kind_from_label(Some("neat_unknown")),
            Some(BackendKind::Unavailable)
        );
    }

    #[test]
    fn runtime_mode_and_degraded_reason_are_typed_from_legacy_details() {
        assert_eq!(
            runtime_mode_from_details(Some(BackendKind::NativeCpu), None),
            Some(RuntimeMode::Canonical)
        );
        assert_eq!(
            runtime_mode_from_details(
                Some(BackendKind::LocalSurrogateFallback),
                Some("fallback_active")
            ),
            Some(RuntimeMode::Degraded)
        );

        let reason = typed_runtime_degraded_reason(Some(
            "requested device policy `gpu:0`; runtime currently executes on CPU",
        ))
        .expect("typed degraded reason");
        assert_eq!(reason.code, "requested_device_policy_gpu_0");
        assert!(reason.message.contains("runtime currently executes on CPU"));
    }

    #[test]
    fn normalize_training_precision_policy_accepts_aliases() {
        assert_eq!(normalize_training_precision_policy(" bfloat16 "), "bf16");
        assert_eq!(normalize_training_precision_policy("f32"), "fp32");
        assert_eq!(normalize_training_precision_policy("float8"), "fp8");
        assert_eq!(normalize_training_precision_policy("unknown"), "auto");
    }

    #[test]
    fn requested_training_precision_policy_prefers_model_scoped_env() {
        unsafe {
            std::env::set_var("FOREX_BOT_DQN_TRAIN_PRECISION", "bf16");
            std::env::set_var("FOREX_BOT_TRAIN_PRECISION", "fp32");
        }
        let requested = requested_training_precision_policy("dqn");
        unsafe {
            std::env::remove_var("FOREX_BOT_DQN_TRAIN_PRECISION");
            std::env::remove_var("FOREX_BOT_TRAIN_PRECISION");
        }
        assert_eq!(requested, "bf16");
    }
}
