use std::fs;
use std::path::PathBuf;

fn workspace_read(relative: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root");
    let path = workspace.join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing {signature}"));
    let tail = &source[start..];
    let end = tail
        .find("\n}\n")
        .map_or(tail.len(), |offset| offset + "\n}".len());
    &tail[..end]
}

#[test]
fn contract_builder_uses_the_resident_v3_classic_subset_as_a_cpu_reference() {
    let hpc = workspace_read("crates/neoethos-data/src/core/hpc_ta.rs");
    let data = workspace_read("crates/neoethos-data/src/lib.rs");
    let search = workspace_read("crates/neoethos-search/src/data_selection.rs");
    let cli = workspace_read("crates/neoethos-cli/src/canonical_full_run.rs");

    let cpu_plan = function_body(
        &hpc,
        "fn prepare_classic_ta_gpu_exact_parity_cpu_reference_run_plan_v3(",
    );
    for required in [
        "resolved_indicator_compute_policy() != IndicatorComputePolicy::GpuOnly",
        "prepare_classic_ta_gpu_exact_parity_run_plan_v3(budget_rows)?",
        "run_plan.policy = IndicatorComputePolicy::CpuOnly",
        "run_plan.resident_cuda_launches = None",
    ] {
        assert!(
            cpu_plan.contains(required),
            "exact-subset CPU plan omits `{required}`"
        );
    }

    let data_entry = function_body(
        &data,
        "pub fn prepare_multitimeframe_features_gpu_exact_parity_cpu_reference_v3(",
    );
    assert!(
        data_entry.contains("FeatureProfile::Standard")
            && data_entry.contains("FeatureProfile::HPC")
            && data_entry.contains("FeatureProfile::Adaptive")
            && data_entry.contains("Full must retain the complete fail-closed Classic graph"),
        "Data production entry does not reject the Full profile before selecting the subset"
    );
    let shared_data_builder = function_body(
        &data,
        "fn prepare_multitimeframe_features_with_classic_plan_authority_v3(",
    );
    assert!(
        shared_data_builder
            .contains("prepare_classic_ta_gpu_exact_parity_cpu_reference_run_plan_v3"),
        "Data shared builder does not select the resident V3 Classic subset CPU reference"
    );

    let search_entry = function_body(
        &search,
        "pub fn from_exact_series_receipt_gpu_exact_parity_cpu_reference_v3(",
    );
    assert!(
        search_entry.contains("prepare_multitimeframe_features_gpu_exact_parity_cpu_reference_v3"),
        "Search production entry does not call the named Data production entry"
    );
    assert!(
        search_entry.contains("from_exact_series_receipt_with_builder_v3("),
        "Search exact-subset entry does not reuse the canonical validation/provenance builder"
    );
    let shared_search_builder =
        function_body(&search, "fn from_exact_series_receipt_with_builder_v3(");
    for required in [
        "resolved_canonical_feature_execution_authority_v1()",
        "verify_search_input_provenance(",
    ] {
        assert!(
            shared_search_builder.contains(required),
            "shared exact-series builder omits `{required}`"
        );
    }

    let contract_builder = function_body(&cli, "pub fn build_contract(");
    assert!(
        contract_builder.contains(
            "CanonicalSearchInput::from_exact_series_receipt_gpu_exact_parity_cpu_reference_v3("
        ),
        "canonical-contract-build does not call the named exact-subset CPU production entry"
    );
    assert!(
        !contract_builder.contains("CanonicalSearchInput::from_exact_series_receipt("),
        "canonical-contract-build still calls the ordinary full Classic builder"
    );
    assert!(
        contract_builder.contains("#[cfg(not(feature = \"gpu-nvidia\"))]"),
        "non-GPU canonical-contract-build does not fail closed explicitly"
    );
    for preserved in [
        "feature_options.normalization_training_rows = Some(normalization_training_rows)",
        "canonical_discovery_normalization_training_rows(base_row_count)?",
        "oos_from_ms",
    ] {
        assert!(
            contract_builder.contains(preserved),
            "canonical-contract-build lost `{preserved}`"
        );
    }
}
