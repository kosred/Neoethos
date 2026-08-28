#[test]
fn f64_model_configuration_uses_f64_parsing() {
    let source = include_str!("../src/training_orchestrator.rs");
    for required in [
        "model.learning_rate = parse_f64_param(params, \"lr\", model.learning_rate);",
        "model.alpha = parse_f64_param(params, \"alpha\", model.alpha);",
        "parse_f64_param(params, \"alpha\", 0.10),",
    ] {
        assert!(
            source.contains(required),
            "training orchestrator is missing f64 configuration boundary {required}"
        );
    }
    for retired in [
        "model.learning_rate = parse_f32_param(params, \"lr\", model.learning_rate);",
        "model.alpha = parse_f32_param(params, \"alpha\", model.alpha);",
        "parse_f32_param(params, \"alpha\", 0.10),",
    ] {
        assert!(
            !source.contains(retired),
            "training orchestrator restored lossy f32 configuration boundary {retired}"
        );
    }
}

#[test]
fn f32_backends_promote_probabilities_at_the_shared_runtime_boundary() {
    let exit = include_str!("../src/exit_agent.rs");
    let sac = include_str!("../src/soft_actor_critic.rs");
    let dqn = include_str!("../src/rl/dqn_impl.rs");

    assert!(exit.contains("runtime_probabilities.map(f64::from)"));
    assert!(sac.contains("probabilities.map(f64::from)"));
    assert!(dqn.contains("probabilities.map(f64::from)"));
    assert!(sac.contains("*value <= f64::EPSILON"));
    assert!(dqn.contains("*value <= f64::EPSILON"));
    assert!(dqn.contains("let (_effective_precision, precision_degraded_reason)"));
}

#[test]
fn native_tree_only_import_is_gated_with_its_native_feature() {
    let source = include_str!("../src/tree_models/catboost.rs").replace("\r\n", "\n");
    assert!(
        source.contains(
            "#[cfg(feature = \"catboost\")]\nuse crate::base::feature_columns_from_frame;"
        )
    );
    assert!(source.contains("use crate::common::CudaDevicePolicy;"));
    assert!(source.contains(
        "#[cfg(feature = \"catboost\")]\nuse crate::common::{ResolvedCudaDevicePolicy, resolve_cuda_device_policy};"
    ));
    assert!(!source.contains(
        "use crate::common::{CudaDevicePolicy, ResolvedCudaDevicePolicy, resolve_cuda_device_policy};"
    ));
    assert!(!source.contains("use crate::base::{ExpertModel, feature_columns_from_frame};"));
}

#[test]
fn native_tree_sources_do_not_hide_dead_code_or_unused_imports() {
    for (name, source) in [
        ("xgboost", include_str!("../src/tree_models/xgboost.rs")),
        ("lightgbm", include_str!("../src/tree_models/lightgbm.rs")),
        ("catboost", include_str!("../src/tree_models/catboost.rs")),
    ] {
        for forbidden in [
            "#![allow(dead_code",
            "#![allow(unused_imports",
            "#![allow(dead_code, unused_imports)]",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} hides compiler evidence with `{forbidden}`"
            );
        }
    }
}
