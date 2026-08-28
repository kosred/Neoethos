use std::collections::BTreeSet;

const COMMON: &str = include_str!("../src/common.rs");
const TREE_CONFIG: &str = include_str!("../src/tree_models/config.rs");
const XGBOOST: &str = include_str!("../src/tree_models/xgboost.rs");
const LIGHTGBM: &str = include_str!("../src/tree_models/lightgbm.rs");
const CATBOOST: &str = include_str!("../src/tree_models/catboost.rs");
const ORCHESTRATOR: &str = include_str!("../src/training_orchestrator.rs");
const DQN: &str = include_str!("../src/rl/dqn_impl.rs");
const NEAT: &str = include_str!("../src/evolution/neat_impl.rs");
const CRFMNES: &str = include_str!("../src/evolution/crfmnes_impl.rs");
const STATISTICAL_IMPL: &str = include_str!("../src/statistical/linear_impl.rs");
const REGISTRY: &str = include_str!("../src/registry.rs");
const TREE_CUDA_LIFECYCLE: &str = include_str!("tree_cuda_device.rs");
const DQN_CUDA_LIFECYCLE: &str = include_str!("rl_cuda_contract.rs");
const EVOLUTION_CUDA_LIFECYCLE: &str = include_str!("neuro_evolution_cuda_contract.rs");
const STATISTICAL_CUDA_LIFECYCLE: &str = include_str!("statistical_cuda_contract.rs");

#[test]
fn cuda_policy_is_typed_fallible_and_nvidia_only() {
    for required in [
        "pub enum CudaDevicePolicy",
        "pub fn parse_cuda_device_policy",
        "pub fn resolve_cuda_device_policy",
        "pub fn nvidia_gpu_count",
    ] {
        assert!(
            COMMON.contains(required) || TREE_CONFIG.contains(required),
            "missing strict CUDA policy authority `{required}`"
        );
    }

    assert!(
        !COMMON.contains(".and_then(|value| value.parse::<usize>().ok())\n        .unwrap_or(0)"),
        "malformed gpu:<ordinal> still silently aliases CUDA ordinal zero"
    );
    assert!(
        COMMON.contains("ROCm device policies cannot select a CUDA backend"),
        "ROCm/vendor-non-CUDA policy must fail instead of selecting CUDA"
    );
}

#[test]
fn auto_cuda_routing_never_masks_a_present_nvidia_failure() {
    assert!(
        !DQN.contains(
            "Err(_) => return Ok((Device::Cpu, \"cpu\".to_string(), \"rlkit_cpu\".to_string()))"
        ),
        "DQN training Auto still converts every CUDA initialization error to CPU"
    );
    assert!(
        !DQN.contains("Err(_) => (Device::Cpu, \"cpu\".to_string(), \"rlkit_cpu\".to_string())"),
        "DQN inference Auto still converts every CUDA initialization error to CPU"
    );
    for (surface, source) in [
        ("DQN", DQN),
        ("NEAT", NEAT),
        ("CR-FM-NES", CRFMNES),
        ("statistical", STATISTICAL_IMPL),
    ] {
        assert!(
            source.contains("resolve_cuda_device_policy"),
            "{surface} does not route Auto through the shared strict CUDA resolver"
        );
    }
}

#[test]
fn evolution_artifacts_persist_and_validate_exact_effective_cuda_ordinals() {
    for (surface, source) in [("NEAT", NEAT), ("CR-FM-NES", CRFMNES)] {
        for required in [
            "effective_device_policy: String",
            "effective_device_policy: self.effective_device_policy.clone()",
            "artifact.effective_device_policy",
            "format!(\"gpu:{ordinal}\")",
        ] {
            assert!(
                source.contains(required),
                "{surface} artifact lifecycle is missing `{required}`"
            );
        }
        assert!(
            source.contains("let current_device = crate::common::resolve_cuda_device_policy(")
                && source.contains("&artifact.requested_device_policy"),
            "{surface} artifact load does not re-resolve the requested policy against current NVIDIA hardware"
        );
        assert!(
            source.contains("effective_label == format!(\"gpu:{ordinal}\")"),
            "{surface} artifact accepts a non-canonical effective CUDA device label"
        );
        assert!(
            source.contains("recorded CPU fitness, but NVIDIA is now visible"),
            "{surface} Auto artifact can remain CPU-backed after reload on an NVIDIA host"
        );
    }
}

#[test]
fn tree_cuda_policy_preserves_and_routes_exact_ordinals() {
    for required in [
        "parse_tree_cuda_device_policy",
        "cuda_ordinal",
        "nvidia_gpu_count",
    ] {
        assert!(
            TREE_CONFIG.contains(required),
            "tree CUDA config is missing `{required}`"
        );
    }
    assert!(
        XGBOOST.contains("format!(\"cuda:{cuda_ordinal}\")"),
        "XGBoost does not route an exact CUDA ordinal"
    );
    assert!(
        include_str!("../src/tree_models/lightgbm.rs").contains("gpu_device_id"),
        "LightGBM does not receive the selected CUDA ordinal"
    );
    assert!(
        CATBOOST.contains(".arg(\"--devices\")"),
        "CatBoost does not receive the selected CUDA ordinal"
    );
}

#[test]
fn xgboost_reload_reapplies_the_resolved_cuda_device() {
    assert!(
        XGBOOST.contains("apply_runtime_device(&mut model)"),
        "XGBoost load does not reapply the resolved device to the restored booster"
    );
    assert!(
        !XGBOOST.contains("XGBoost runtime sidecar did not fully match the restored config"),
        "XGBoost device drift is still warning-only"
    );
}

#[test]
fn every_persisted_cuda_model_revalidates_device_identity_on_load() {
    for (surface, source, required) in [
        (
            "LightGBM",
            LIGHTGBM,
            "Self::validate_runtime_device_for_load(&runtime_profile)",
        ),
        (
            "CatBoost",
            CATBOOST,
            "self.validate_runtime_device_for_load(runtime_artifact)",
        ),
        (
            "DQN",
            DQN,
            "validate_rl_artifact_device_for_load(&artifact)",
        ),
        (
            "statistical",
            STATISTICAL_IMPL,
            "validate_linear_artifact_device_for_load(&model)",
        ),
    ] {
        assert!(
            source.contains(required),
            "{surface} reload is missing live requested/effective CUDA identity validation `{required}`"
        );
    }
}

#[test]
fn catboost_receives_final_params_at_construction() {
    assert!(
        CATBOOST.contains("pub fn new_with_params("),
        "CatBoost lacks a params-aware constructor"
    );
    assert!(
        ORCHESTRATOR.contains("CatBoostExpert::new_with_params("),
        "orchestrator still constructs CatBoost before installing its final params"
    );
    assert!(
        !ORCHESTRATOR.contains("model.config.params = parse_tree_params(&seeded);"),
        "orchestrator still overwrites CatBoost params after derived device fields were computed"
    );
}

#[test]
fn registry_exposes_exact_cuda_surface_census() {
    let expected = BTreeSet::from([
        "catboost",
        "catboost_alt",
        "conformal_gate",
        "dqn",
        "elasticnet",
        "lightgbm",
        "logistic",
        "meta_blender",
        "meta_stack",
        "neat",
        "neuro_evo",
        "probability_calibrator",
        "xgboost",
        "xgboost_dart",
        "xgboost_rf",
    ]);

    let marker = "pub const CUDA_CAPABLE_MODEL_NAMES: &[&str] = &[";
    let start = REGISTRY
        .find(marker)
        .expect("registry must declare CUDA_CAPABLE_MODEL_NAMES");
    let remainder = &REGISTRY[start + marker.len()..];
    let end = remainder
        .find("];")
        .expect("CUDA_CAPABLE_MODEL_NAMES must be a closed array");
    let actual = remainder[..end]
        .split(',')
        .map(str::trim)
        .filter_map(|value| value.strip_prefix('"').and_then(|v| v.strip_suffix('"')))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "CUDA model capability census drifted");
}

#[test]
fn all_fifteen_cuda_surfaces_have_mandatory_lifecycle_gates() {
    let gates = [
        (
            "xgboost_cuda_named_surfaces_train_infer_save_load",
            TREE_CUDA_LIFECYCLE,
        ),
        (
            "xgboost_cuda_meta_surfaces_train_infer_save_load",
            TREE_CUDA_LIFECYCLE,
        ),
        (
            "lightgbm_cuda_surface_train_infer_save_load",
            TREE_CUDA_LIFECYCLE,
        ),
        (
            "catboost_cuda_named_surfaces_train_infer_save_load",
            TREE_CUDA_LIFECYCLE,
        ),
        ("dqn_cuda_surface_train_infer_save_load", DQN_CUDA_LIFECYCLE),
        (
            "neat_and_neuro_evo_cuda_surfaces_train_infer_save_load",
            EVOLUTION_CUDA_LIFECYCLE,
        ),
        (
            "logistic_and_elasticnet_cuda_surfaces_train_infer_save_load",
            STATISTICAL_CUDA_LIFECYCLE,
        ),
    ];
    for (gate, source) in gates {
        assert!(
            source.contains(gate),
            "missing mandatory RTX lifecycle gate `{gate}`"
        );
    }
    for (gate, source) in gates {
        let start = source
            .find(gate)
            .unwrap_or_else(|| panic!("missing mandatory RTX lifecycle gate `{gate}`"));
        let remainder = &source[start..];
        let end = remainder.find("\n#[test]").unwrap_or(remainder.len());
        let body = &remainder[..end];
        assert!(
            !body.contains("#[ignore]")
                && !body.contains("return; // skip")
                && !body.contains("cuda_if_available"),
            "mandatory RTX gate `{gate}` contains a skip/fallback marker"
        );
    }
}
