fn assert_absent(surface: &str, source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            !source.contains(token),
            "{surface} still contains retired token `{token}`"
        );
    }
}

fn assert_present(surface: &str, source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "{surface} is missing canonical token `{token}`"
        );
    }
}

#[test]
fn hmm_training_consumes_exact_versioned_feature_columns() {
    let model = include_str!("../src/forecasting/hmm_regime.rs");
    let orchestrator = include_str!("../src/training_orchestrator.rs");

    assert_absent(
        "training_orchestrator.rs",
        orchestrator,
        &["ohlcv_to_features", "derived from raw OHLCV"],
    );
    assert_present(
        "hmm_regime.rs",
        model,
        &[
            "training_observations_from_feature_frame",
            "quant_log_return",
            "quant_log_volatility",
            "FeatureCellValidity",
        ],
    );
    assert_present(
        "training_orchestrator.rs",
        orchestrator,
        &[
            "training_observations_from_feature_frame",
            "&budgeted_frame",
        ],
    );
}

#[test]
fn swarm_training_consumes_only_exact_quant_close_with_caller_lease() {
    let source = include_str!("../src/forecasting/swarm_impl.rs");

    assert_absent(
        "swarm_impl.rs",
        source,
        &[
            "polars::",
            "DataFrame",
            "&Series",
            "Series::",
            "extract_continuous_label_series",
            "extract_series_from_frame",
            "preferred_columns",
            "fallback swarm source column",
        ],
    );
    assert_present(
        "swarm_impl.rs",
        source,
        &[
            "FeatureFrame",
            "CpuLease",
            "quant_close",
            "feature_column",
            "frame.timestamps",
            "lease.scope",
        ],
    );
}

#[test]
fn genetic_model_does_not_run_receiptless_strategy_discovery() {
    let source = include_str!("../src/genetic.rs");

    assert_absent(
        "genetic.rs",
        source,
        &[
            "run_discovery_cycle",
            "train_with_discovery",
            "DiscoveryBacked",
            "Option<&Ohlcv>",
            "genetic_expert_holdout",
            "GENETIC_EXPERT_HOLDOUT",
        ],
    );
    assert_present(
        "genetic.rs",
        source,
        &["train_with_labels", "GeneticBackendMode::LabelSearch"],
    );
}
