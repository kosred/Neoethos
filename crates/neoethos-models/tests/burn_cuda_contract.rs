const BURN_MODELS: &str = include_str!("../src/burn_models.rs");
const MODELS_MANIFEST: &str = include_str!("../Cargo.toml");
const DEEP_MODELS: &str = include_str!("../src/deep_models.rs");
const EXIT_AGENT: &str = include_str!("../src/exit_agent.rs");
const REGISTRY: &str = include_str!("../src/registry.rs");
const SOFT_ACTOR_CRITIC: &str = include_str!("../src/soft_actor_critic.rs");
const BURN_CUDA_LIFECYCLE: &str = include_str!("burn_cuda_lifecycle.rs");

#[test]
fn burn_cuda_resolution_is_exact_fail_loud_and_device_tested() {
    assert!(
        BURN_MODELS
            .contains("pub fn resolve_train_device(\n    policy: &str,\n) -> anyhow::Result<("),
        "Burn training device resolution must be fallible"
    );
    assert!(
        BURN_MODELS
            .contains("pub fn resolve_infer_device(\n    policy: &str,\n) -> anyhow::Result<("),
        "Burn inference device resolution must be fallible"
    );
    assert!(
        !BURN_MODELS.contains("fn resolve_cuda_device_policy(_normalized"),
        "Burn CUDA must not ignore the selected ordinal"
    );
    assert!(
        BURN_MODELS.contains("CudaDevice::new(device_index)"),
        "Burn CUDA must construct the exact selected ordinal"
    );
    assert!(
        BURN_MODELS.contains("burn_cuda_auto_precision_runs_three_epoch_real_kernels_in_fp32"),
        "Burn CUDA needs a mandatory real-device forward and training test"
    );
    assert!(
        BURN_MODELS.contains("burn_cuda_rejects_cpu_and_malformed_device_policies"),
        "Burn CUDA needs strict policy refusal coverage"
    );
    assert!(
        !BURN_MODELS.contains("cuda_if_available") && !BURN_MODELS.contains("return; // skip"),
        "Burn CUDA real-device test must not skip"
    );
}

#[test]
fn burn_cuda_training_precision_is_fail_closed_to_fp32_for_the_complete_lifecycle() {
    for required in [
        "let native_cuda_training = matches!(",
        "let bf16_supported = supports_bf16 && !native_cuda_training;",
        "native Burn CUDA 0.21 BF16 optimizer/fusion graph is not production-safe across the complete model lifecycle",
        "assert_eq!(report.training_precision, \"fp32\");",
        "assert_eq!(report.epochs_ran, 3);",
    ] {
        assert!(
            BURN_MODELS.contains(required),
            "Burn CUDA FP32 lifecycle contract is missing `{required}`"
        );
    }
    assert!(
        !BURN_MODELS.contains("backend `training_dtype` resolves to bf16"),
        "Burn CUDA retains the stale claim that native training resolves to BF16"
    );
}

#[test]
fn burn_cuda_reachable_surface_census_is_explicit_before_the_separate_rtx_matrix() {
    for (variant, model_name) in [
        ("Mlp", "mlp"),
        ("NBeats", "nbeats"),
        ("NBeatsxNf", "nbeatsx_nf"),
        ("TiDE", "tide"),
        ("TiDENf", "tide_nf"),
        ("TabNet", "tabnet"),
        ("Kan", "kan"),
        ("Transformer", "transformer"),
        ("PatchTst", "patchtst"),
        ("TimesNet", "timesnet"),
    ] {
        assert!(
            DEEP_MODELS.contains(&format!("Self::{variant} => \"{model_name}\"")),
            "Burn CUDA census lost deep surface {model_name}"
        );
    }
    for (surface, source) in [
        ("exit_agent", EXIT_AGENT),
        ("soft_actor_critic", SOFT_ACTOR_CRITIC),
    ] {
        assert!(
            source.contains("use crate::burn_models")
                && source.contains("TrainBackend")
                && source.contains("resolve_train_device"),
            "Burn CUDA census lost {surface}'s reachable TrainBackend surface"
        );
    }
}

#[test]
fn burn_cuda_capability_registry_reports_the_compiled_backend() {
    let supports = REGISTRY
        .split("pub fn supports_gpu_for_model")
        .nth(1)
        .expect("the per-model GPU support registry is missing")
        .split("/// Whether the GPU path is the one this model SHOULD take")
        .next()
        .expect("GPU preference documentation must follow GPU support");
    let prefers = REGISTRY
        .split("pub fn prefers_gpu_for_model")
        .nth(1)
        .expect("the per-model GPU preference registry is missing")
        .split("#[cfg(test)]")
        .next()
        .expect("registry tests must follow GPU preference");

    for (name, body) in [("supports", supports), ("prefers", prefers)] {
        assert!(
            body.contains("feature = \"burn-wgpu-backend\"")
                && body.contains("feature = \"burn-cuda-backend\""),
            "Burn {name} capability must include both compiled GPU backends"
        );
        assert!(
            body.contains("\"sac\" =>") && body.contains("ModelFamily::Deep | ModelFamily::Exit"),
            "Burn {name} capability must cover SAC, Deep, and Exit surfaces"
        );
    }
}

#[test]
fn burn_cuda_full_lifecycle_matrix_is_mandatory_and_non_skipping() {
    for gate in [
        "burn_cuda_deep_mlp_lifecycle_gpu0",
        "burn_cuda_deep_nbeats_lifecycle_gpu0",
        "burn_cuda_deep_nbeatsx_nf_lifecycle_gpu0",
        "burn_cuda_deep_tide_lifecycle_gpu0",
        "burn_cuda_deep_tide_nf_lifecycle_gpu0",
        "burn_cuda_deep_tabnet_lifecycle_gpu0",
        "burn_cuda_deep_kan_lifecycle_gpu0",
        "burn_cuda_deep_transformer_lifecycle_gpu0",
        "burn_cuda_deep_patchtst_lifecycle_gpu0",
        "burn_cuda_deep_timesnet_lifecycle_gpu0",
        "burn_cuda_exit_agent_lifecycle_gpu0",
        "burn_cuda_sac_lifecycle_gpu0",
    ] {
        assert!(
            BURN_CUDA_LIFECYCLE.contains(gate),
            "Burn CUDA lifecycle matrix is missing mandatory gate {gate}"
        );
    }

    for required in [
        "#![cfg(feature = \"burn-cuda-backend\")]",
        "resolve_train_device(EXPLICIT_CUDA_POLICY)",
        "assert_eq!(device.index, CUDA_ORDINAL)",
        "fit_with_validation",
        "predict_runtime",
        ".save(",
        "::load(",
        "assert_cuda_artifact_identity",
        "assert_prediction_parity",
    ] {
        assert!(
            BURN_CUDA_LIFECYCLE.contains(required),
            "Burn CUDA lifecycle matrix is missing `{required}`"
        );
    }

    for forbidden in [
        "#[ignore]",
        "cuda_if_available",
        "is_cuda_available",
        "CUDA_VISIBLE_DEVICES",
        "skip_cuda",
    ] {
        assert!(
            !BURN_CUDA_LIFECYCLE.contains(forbidden),
            "Burn CUDA lifecycle matrix may not contain skip path `{forbidden}`"
        );
    }
}

#[test]
fn burn_cuda_feature_is_fail_closed_against_cpu_backends_and_artifact_drift() {
    for required in [
        "ensure_burn_cuda_backend_type::<B>()?;",
        "validate_loaded_burn_device_identity(",
        "Burn CUDA backend requires native CUDA execution",
        "Burn CUDA artifact runtime identity",
    ] {
        assert!(
            BURN_MODELS.contains(required),
            "Burn CUDA strict boundary is missing `{required}`"
        );
    }
    assert!(
        EXIT_AGENT.contains("validate_loaded_burn_device_identity("),
        "ExitAgent load must fail closed on Burn CUDA artifact drift"
    );
    assert!(
        SOFT_ACTOR_CRITIC.contains("validate_loaded_burn_device_identity("),
        "SAC load must fail closed on Burn CUDA artifact drift"
    );
    assert!(
        SOFT_ACTOR_CRITIC.contains("pub fn with_device_policy("),
        "SAC needs an explicit exact-ordinal device constructor"
    );
}

#[test]
fn burn_cuda_comments_name_the_actual_feature_and_burn_version() {
    assert!(BURN_MODELS.contains("Pure-Rust deep learning models using Burn 0.21."));
    assert!(!BURN_MODELS.contains("Burn 0.20"));
    for source in [BURN_MODELS, DEEP_MODELS] {
        assert!(
            !source.contains("Native Burn CUDA backend (gpu-cuda build)"),
            "Burn CUDA comments must name burn-cuda-backend, not gpu-cuda"
        );
    }
    assert!(!BURN_MODELS.contains("the A6000 is exposed"));
    assert!(
        MODELS_MANIFEST
            .contains("CPU operation remains available only when this feature is absent"),
        "the manifest must state the strict Burn CUDA/CPU feature boundary"
    );
    for stale_claim in [
        "These models train in ~17 min TOTAL on burn-ndarray (CPU)",
        "belong on the CPU",
    ] {
        assert!(
            !MODELS_MANIFEST.contains(stale_claim),
            "the manifest retains stale Burn CUDA policy `{stale_claim}`"
        );
    }
}
