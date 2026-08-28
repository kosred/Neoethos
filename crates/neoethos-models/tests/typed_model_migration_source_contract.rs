fn assert_forbidden_absent(source_name: &str, source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "{source_name} still contains retired typed-model compatibility token `{token}`"
        );
    }
}

fn assert_required_present(source_name: &str, source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "{source_name} is missing canonical typed-model contract token `{token}`"
        );
    }
}

#[test]
fn burn_deep_models_use_typed_frames_caller_lease_and_f64_trait_output() {
    let source = include_str!("../src/deep_models.rs");
    assert_forbidden_absent(
        "deep_models.rs",
        source,
        &[
            "polars::",
            "DataFrame",
            "Series::new(",
            "&Series",
            "dataframe_to_float32_array",
            "feature_columns_from_dataframe",
        ],
    );
    assert_required_present(
        "deep_models.rs",
        source,
        &[
            "FeatureFrame",
            "CpuLease",
            "deep_backend_f32_matrix",
            "feature_columns_from_frame",
            "Result<Array2<f64>>",
            "lease.scope",
        ],
    );
}

#[test]
fn training_orchestrator_preserves_typed_frame_row_identity_and_forwards_lease() {
    let source = include_str!("../src/training_orchestrator.rs");
    assert_forbidden_absent(
        "training_orchestrator.rs",
        source,
        &[
            "polars::",
            "DataFrame",
            "Series::new(",
            "&Series",
            "TrainingPayload::from_named_dense",
            "TrainingPayload::from_dense",
            "labels_to_series",
            ".height()",
            ".width()",
        ],
    );
    assert_required_present(
        "training_orchestrator.rs",
        source,
        &[
            "FeatureFrame",
            "CpuLease",
            "TrainingPayload::from_frame_with_source_rows",
            ".n_samples()",
            ".n_features()",
            ".select_rows(",
        ],
    );
}

#[test]
fn remaining_model_implementations_do_not_import_retired_dataframe_helpers() {
    let sources = [
        (
            "evolution/crfmnes_impl.rs",
            include_str!("../src/evolution/crfmnes_impl.rs"),
        ),
        (
            "evolution/neat_impl.rs",
            include_str!("../src/evolution/neat_impl.rs"),
        ),
        (
            "streaming/adaptive_impl.rs",
            include_str!("../src/streaming/adaptive_impl.rs"),
        ),
        ("rl/dqn_impl.rs", include_str!("../src/rl/dqn_impl.rs")),
        ("genetic.rs", include_str!("../src/genetic.rs")),
        (
            "soft_actor_critic.rs",
            include_str!("../src/soft_actor_critic.rs"),
        ),
        ("exit_agent.rs", include_str!("../src/exit_agent.rs")),
    ];
    let forbidden = [
        "polars::prelude",
        "DataFrame",
        "Series::new(",
        "&Series",
        "feature_matrix_from_dataframe",
        "dataframe_to_float32_array",
        "feature_columns_from_dataframe",
    ];

    for (source_name, source) in sources {
        assert_forbidden_absent(source_name, source, &forbidden);
    }
}
