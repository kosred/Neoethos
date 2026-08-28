fn assert_typed_model_contract(source_name: &str, source: &str, expert_probability: bool) {
    for forbidden in [
        "polars::",
        "DataFrame",
        "Series",
        "feature_matrix_from_dataframe",
        "dataframe_to_float32_array",
        "feature_columns_from_dataframe",
    ] {
        assert!(
            !source.contains(forbidden),
            "{source_name} still contains retired compatibility token `{forbidden}`"
        );
    }
    for required in ["FeatureFrame", "CpuLease", "lease.scope"] {
        assert!(
            source.contains(required),
            "{source_name} is missing canonical typed-model token `{required}`"
        );
    }
    if expert_probability {
        assert!(
            source.contains("Result<Array2<f64>>"),
            "{source_name} must expose f64 probabilities at its ExpertModel boundary"
        );
    }
}

#[test]
fn dqn_uses_typed_frames_and_caller_owned_cpu_lease() {
    assert_typed_model_contract(
        "rl/dqn_impl.rs",
        include_str!("../src/rl/dqn_impl.rs"),
        false,
    );
}

#[test]
fn genetic_strategy_uses_typed_frames_and_caller_owned_cpu_lease() {
    assert_typed_model_contract("genetic.rs", include_str!("../src/genetic.rs"), true);
}

#[test]
fn soft_actor_critic_uses_typed_frames_and_caller_owned_cpu_lease() {
    assert_typed_model_contract(
        "soft_actor_critic.rs",
        include_str!("../src/soft_actor_critic.rs"),
        false,
    );
}

#[test]
fn exit_agent_uses_typed_frames_and_caller_owned_cpu_lease() {
    assert_typed_model_contract("exit_agent.rs", include_str!("../src/exit_agent.rs"), false);
}
