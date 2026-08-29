const NATIVE_TREE_SOURCES: [(&str, &str); 3] = [
    ("xgboost", include_str!("../src/tree_models/xgboost.rs")),
    ("lightgbm", include_str!("../src/tree_models/lightgbm.rs")),
    ("catboost", include_str!("../src/tree_models/catboost.rs")),
];

fn compact_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn native_tree_experts_use_typed_frames_and_never_restore_retired_surrogates() {
    for (model, source) in NATIVE_TREE_SOURCES {
        for forbidden in [
            "TreeLocalFallbackArtifact",
            "build_tree_local_fallback_artifact",
            "predict_tree_local_fallback",
            "validate_tree_local_fallback_artifact",
            "dataframe_to_row_major_vec",
            "feature_columns_from_dataframe",
            "polars::",
            "DataFrame",
            "Series",
            "local_fallback",
            "LOCAL_FALLBACK",
        ] {
            assert!(
                !source.contains(forbidden),
                "{model} still contains retired compatibility/fallback token `{forbidden}`"
            );
        }

        let compact = compact_whitespace(source);
        for required in [
            "use neoethos_data::FeatureFrame;",
            "use neoethos_execution_budget::CpuLease;",
            "feature_frame_to_tree_f32_row_major",
            "fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()>",
            "fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>>",
            "lease.scope(||",
        ] {
            assert!(
                compact.contains(required),
                "{model} is missing canonical native-tree contract fragment `{required}`"
            );
        }
    }
}
