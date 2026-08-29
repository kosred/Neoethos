use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read_required(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_local_cfg_before(source: &str, needle: &str, cfg: &str) {
    let position = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing required item {needle:?}"));
    let prefix = &source[..position];
    let local_start = prefix.len().saturating_sub(384);
    let local = &prefix[local_start..];
    assert!(
        local.contains(cfg),
        "{needle:?} is not guarded locally by {cfg:?}; nearby prefix was {local:?}"
    );
}

#[test]
fn search_only_population_bridge_is_cuda_gated_and_has_one_owned_authority() {
    let population = read_required("src/population.rs");
    for (ffi, cfg) in [
        (
            "fn neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(",
            "#[cfg(feature = \"cuda\")]",
        ),
        (
            "fn neoethos_gpu_cuda_population_export_resident_scoring_source_v2(",
            "#[cfg(feature = \"cuda\")]",
        ),
        (
            "fn neoethos_gpu_cuda_population_finish_resident_scoring_source_v2(",
            "#[cfg(feature = \"cuda\")]",
        ),
    ] {
        assert_eq!(
            population.matches(ffi).count(),
            1,
            "one FFI authority required for {ffi}"
        );
        require_local_cfg_before(&population, ffi, cfg);
    }

    assert!(
        !population.contains("pub(crate) fn enqueue_resident_gene_metrics_v2("),
        "the superseded borrowed Search bridge must be deleted, not lint-suppressed"
    );
    let search = read_required("src/resident_search_v2.rs");
    assert!(
        !search.contains("pub(crate) fn enqueue_resident_gene_metrics_v2("),
        "the superseded borrowed Search wrapper must be deleted with its bridge"
    );

    let owned = "pub(crate) fn enqueue_resident_gene_metrics_owned_v2(";
    assert_eq!(
        population.matches(owned).count(),
        1,
        "one move-only Search population bridge required"
    );
    require_local_cfg_before(&population, owned, "#[cfg(feature = \"cuda\")]");

    for owner in [
        "pub(crate) struct ResidentSearchPopulationCompletionLeaseV2 {",
        "impl ResidentSearchPopulationCompletionLeaseV2 {",
        "impl Drop for ResidentSearchPopulationCompletionLeaseV2 {",
    ] {
        require_local_cfg_before(&population, owner, "#[cfg(feature = \"cuda\")]");
    }
}

#[test]
fn terminal_host_metrics_oracle_is_explicitly_fixture_only() {
    let population = read_required("src/population.rs");
    let search = read_required("src/resident_search_v2.rs");
    let store = read_required("src/resident_feature_store_v3.rs");
    let device_test = read_required("src/resident_population_session_v3_device_tests.rs");

    let fixture_method = "pub(crate) fn enqueue_resident_gene_metrics_fixture_v2(";
    for (name, source) in [
        ("population.rs", population.as_str()),
        ("resident_search_v2.rs", search.as_str()),
        ("resident_feature_store_v3.rs", store.as_str()),
    ] {
        assert_eq!(
            source.matches(fixture_method).count(),
            1,
            "{name} must expose exactly one explicitly named fixture bridge"
        );
        require_local_cfg_before(
            source,
            fixture_method,
            "#[cfg(feature = \"cuda-device-fixtures\")]",
        );
        assert!(
            !source.contains("pub(crate) fn enqueue_resident_gene_metrics_v2("),
            "{name} reintroduced the superseded production borrowed bridge"
        );
    }
    require_local_cfg_before(
        &search,
        "use crate::population::ResidentPopulationMetricsV1;",
        "#[cfg(feature = \"cuda-device-fixtures\")]",
    );
    assert!(
        device_test.contains(".enqueue_resident_gene_metrics_fixture_v2(&settings)?"),
        "the real-card fixture must call the explicitly fixture-only bridge"
    );
}

#[test]
fn fixture_only_receipt_facts_are_absent_from_the_production_feature_shape() {
    let search = read_required("src/resident_search_v2.rs");
    let run = section(&search, "pub struct ResidentSearchRunV2 {", "\n}");
    require_local_cfg_before(
        run,
        "expected_survivor_count: u64,",
        "#[cfg(feature = \"cuda-device-fixtures\")]",
    );

    let population = read_required("src/population.rs");
    let lease = section(
        &population,
        "pub(crate) struct ResidentSearchPopulationCompletionLeaseV2 {",
        "\n}",
    );
    require_local_cfg_before(
        lease,
        "counters: NeoPopulationCounters,",
        "#[cfg(feature = \"cuda-device-fixtures\")]",
    );
}

#[test]
fn feature_matrix_fix_does_not_hide_dead_code_broadly() {
    let population = read_required("src/population.rs");
    let search = read_required("src/resident_search_v2.rs");
    for (name, source) in [
        ("population.rs", population),
        ("resident_search_v2.rs", search),
    ] {
        for forbidden in [
            "#![allow(dead_code)]",
            "#![allow(unused_imports)]",
            "#![allow(unused_variables)]",
            "#[allow(clashing_extern_declarations)]",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} hides the feature-matrix defect through {forbidden:?}"
            );
        }
    }
}
