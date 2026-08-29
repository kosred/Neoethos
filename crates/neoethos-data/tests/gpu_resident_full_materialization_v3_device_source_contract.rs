use std::{fs, path::PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("neoethos-data lives below the workspace root")
        .to_path_buf()
}

fn read(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"))
}

#[test]
fn fixture_only_admitted_run_and_bar_major_readback_do_not_expand_production_d2h() {
    let plan = read("crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3_device_fixture.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");

    assert!(plan.contains("pub fn seal_test_full_discovery_run_v1("));
    assert!(library.contains("pub mod resident_feature_store_v3_device_fixture;"));
    assert!(fixture.contains("pub struct ResidentFeatureStoreDeviceReadbackV3"));
    assert!(fixture.contains("pub(crate) fn copy_bar_major_for_device_fixture_v3("));
    assert!(runtime.contains("self.ready_event.synchronize()?"));
    assert!(fixture.contains("values.copy_to(&mut host_values)?"));
    assert!(fixture.contains("validity_u4.copy_to(&mut host_validity_u4)?"));
    assert!(runtime.contains("pub fn copy_bar_major_for_device_fixture_v3("));
    assert!(runtime.contains(
        "resident_feature_store_v3_device_fixture::copy_bar_major_for_device_fixture_v3("
    ));
    assert!(data.contains("pub fn copy_bar_major_for_device_fixture_v3("));

    assert_eq!(
        runtime.matches(".copy_to(").count(),
        4,
        "fixture D2H must stay out of the production runtime source file"
    );
}

#[test]
fn integrated_fixture_uses_the_real_ten_producer_materializer() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let hpc = read("crates/neoethos-data/src/core/hpc_ta.rs");
    let classic_plan = read("crates/neoethos-data/src/core/classic_cuda_plan.rs");
    let classic_recipe = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");
    let recipe_v4 = read("crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs");
    let store = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for token in [
        "integrated_resident_feature_store_v3_matches_all_route_value_bits_and_validity_codes",
        "preflight_gpu_only_feature_workspace_v3(",
        "materialize_gpu_only_feature_store_v3(",
        "seal_test_full_discovery_run_v1(",
        "copy_bar_major_for_device_fixture_v3(",
        "ResidentFeatureProducerV3::ALL.len()",
        "integrated mismatch census",
        "integrated_resident_feature_store_v3_parity=GREEN",
    ] {
        assert!(
            oracle.contains(token),
            "missing integrated fixture token `{token}`"
        );
    }
    assert!(oracle.contains("FeatureProfile::Standard"));
    assert!(oracle.contains("const M15_MILLIS: i64"));
    assert!(oracle.contains("publish(CanonicalTimeframe::M15, &base)"));
    assert!(oracle.contains("publish(CanonicalTimeframe::M30, &parent)"));
    assert!(
        oracle
            .contains("compute_classic_ta_gpu_exact_parity_feature_columns_for_device_fixture_v3(")
    );
    assert!(hpc.contains(
        "pub fn compute_classic_ta_gpu_exact_parity_feature_columns_for_device_fixture_v3("
    ));
    assert!(hpc.contains("run_plan.policy = IndicatorComputePolicy::CpuOnly;"));
    assert!(hpc.contains("run_plan.resident_cuda_launches = None;"));
    assert!(hpc.contains("prepare_classic_ta_gpu_exact_parity_run_plan_v3"));
    assert!(hpc.contains("gpu_only_exact_parity_subset_v3"));
    assert!(hpc.contains("GPU_ONLY_PARITY_DEFERRED_INDICATORS_V3"));
    for deferred in [
        "geometric_bias_oscillator",
        "gopalakrishnan_range_index",
        "historical_volatility",
        "ift_rsi",
        "l1_ehlers_phasor",
        "maaq",
        "natr",
        "nma",
        "pfe",
        "premier_rsi_oscillator",
        "pwma",
        "sgf",
        "sinwma",
        "squeeze_index",
        "sqwma",
        "supersmoother_3_pole",
        "trendflex",
        "ttm_trend",
        "ultosc",
        "uma",
        "vidya",
        "volatility_adjusted_ma",
        "vpwma",
        "wave_smoother",
        "wma",
    ] {
        assert!(hpc.contains(&format!("\"{deferred}\"")));
    }
    assert!(
        hpc.contains("deferred.extend(GPU_ONLY_PARITY_DEFERRED_INDICATORS_V3.iter().copied())")
    );
    assert!(hpc.contains("capability_deferred_indicator_ids"));
    assert!(hpc.contains("capability_deferred_output_count"));
    assert!(hpc.contains("allocation_deferred"));
    assert!(hpc.contains("ResolvedClassicCudaLaunch::Primary(_)"));
    assert!(hpc.contains("admission.capability_deferred_output_count = full_plan"));
    assert!(classic_plan.contains("pub(crate) fn output_count(&self) -> usize"));
    assert!(recipe_v4.contains("pub(crate) fn canonical_parameter_tuple_sha256_v4("));
    assert!(classic_recipe.contains("local_route.canonical_parameter_tuple_sha256_v4()?"));
    assert!(
        !classic_recipe
            .contains("fn parameter_tuple_sha256(parameters: &[ResidentClassicTaParameterV3])")
    );
    assert!(store.contains("FeatureProfile::Full => {"));
    assert!(
        store
            .contains("prepare_classic_ta_run_plan(budget_rows, IndicatorComputePolicy::GpuOnly)?")
    );
    assert!(store.contains(
        "FeatureProfile::Standard | FeatureProfile::HPC | FeatureProfile::Adaptive => {"
    ));
    assert!(store.contains("prepare_classic_ta_gpu_exact_parity_run_plan_v3("));
    assert!(!store.contains("integrated-resident-stage="));
    assert!(!store.contains("integrated-resident-seal="));
    assert!(!store.contains("integrated-resident-preflight="));
}
