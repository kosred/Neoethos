#![cfg(feature = "cuda")]

use std::fs;
use std::path::PathBuf;

use neoethos_gpu_cuda::resident_search_v2::resident_search_v2_production_readiness;

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

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "production V2 seam is missing {token:?}"
        );
    }
}

fn assert_cpp_symbol_not_fixture_gated(source: &str, symbol: &str) {
    let mut fixture_depth = 0_usize;
    let mut conditional_stack = Vec::<bool>::new();
    let mut found = false;

    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#if") {
            let fixture = trimmed.contains("NEOETHOS_CUDA_DEVICE_FIXTURES_V2");
            conditional_stack.push(fixture);
            if fixture {
                fixture_depth += 1;
            }
        }
        if line.contains(symbol) {
            found = true;
            assert_eq!(
                fixture_depth, 0,
                "production symbol {symbol:?} remains inside NEOETHOS_CUDA_DEVICE_FIXTURES_V2"
            );
        }
        if trimmed.starts_with("#endif") {
            if conditional_stack.pop().unwrap_or(false) {
                fixture_depth -= 1;
            }
        }
    }

    assert!(found, "production symbol {symbol:?} is absent");
}

fn assert_rust_ffi_not_fixture_gated(source: &str, symbol: &str) {
    let needle = format!("fn {symbol}(");
    let offset = source
        .find(&needle)
        .unwrap_or_else(|| panic!("missing Rust FFI declaration {symbol:?}"));
    let prefix_start = source[..offset]
        .rfind("\n    fn ")
        .or_else(|| source[..offset].rfind("unsafe extern \"C\""))
        .unwrap_or(0);
    let attached = &source[prefix_start..offset];
    assert!(
        !attached.contains("cuda-device-fixtures"),
        "Rust FFI declaration {symbol:?} remains fixture-gated"
    );
}

#[test]
fn v2_create_export_and_gene_evaluator_are_in_the_normal_cuda_archive() {
    let rust = read_required("src/resident_search_v2.rs");
    let population_rust = read_required("src/population.rs");
    let public_header = read_required("native/neoethos_gpu_cuda.h");
    let v2_header = read_required("native/resident_generation_v2_abi.cuh");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let population_cuda = read_required("native/prototype_b_population.cu");
    let build = read_required("build.rs");

    let create = "neoethos_gpu_cuda_population_create_resident_generation_run_v2";
    assert_cpp_symbol_not_fixture_gated(&public_header, create);
    assert_cpp_symbol_not_fixture_gated(&population_cuda, create);
    assert_rust_ffi_not_fixture_gated(&rust, create);

    let enqueue = "neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2";
    assert_cpp_symbol_not_fixture_gated(&public_header, enqueue);
    assert_cpp_symbol_not_fixture_gated(&population_cuda, enqueue);
    assert_rust_ffi_not_fixture_gated(&population_rust, enqueue);
    require_all(
        &population_rust,
        &["pub(crate) fn enqueue_resident_gene_metrics_v2("],
    );

    for symbol in [
        "configure_resident_generation_evaluator_v2",
        "export_current_resident_gene_view_v2",
    ] {
        assert_cpp_symbol_not_fixture_gated(&v2_header, symbol);
        assert_cpp_symbol_not_fixture_gated(&generation_cuda, symbol);
        assert_rust_ffi_not_fixture_gated(&rust, symbol);
    }
    assert_cpp_symbol_not_fixture_gated(&v2_header, "validate_resident_gene_view_owner_v2");
    assert_cpp_symbol_not_fixture_gated(&generation_cuda, "validate_resident_gene_view_owner_v2");

    require_all(
        &build,
        &[
            "native/resident_generation_v1.cu",
            "native/prototype_b_population.cu",
            "native/resident_generation_v2_abi.cuh",
        ],
    );
}

#[test]
fn generation_run_owns_the_device_control_and_gene_view_carries_it_privately() {
    let rust = read_required("src/resident_search_v2.rs");
    let header = read_required("native/resident_generation_v2_abi.cuh");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let population_header = read_required("native/neoethos_gpu_cuda.h");
    let population_cuda = read_required("native/prototype_b_population.cu");

    let native_run = section(
        &generation_cuda,
        "struct NeoResidentGenerationRunV1 {",
        "\n};",
    );
    require_all(
        native_run,
        &["NeoResidentSearchDeviceControlV2* resident_control_device_v2;"],
    );
    require_all(
        &header,
        &[
            "struct NeoResidentSearchDeviceControlV2",
            "const NeoResidentSearchDeviceControlV2* control_device;",
        ],
    );
    require_all(
        &generation_cuda,
        &[
            "sizeof(NeoResidentSearchDeviceControlV2)",
            "run->resident_control_device_v2",
            "view->control_device = run->resident_control_device_v2",
        ],
    );

    let enqueue_declaration = section(
        &population_header,
        "neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(",
        ");",
    );
    assert!(
        !enqueue_declaration.contains("NeoResidentSearchDeviceControlV2"),
        "the production evaluator still accepts a caller-supplied control pointer"
    );
    let enqueue_definition = section(
        &population_cuda,
        "neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(",
        "\n}",
    );
    require_all(
        enqueue_definition,
        &[
            "genes->control_device",
            "PopulationEvaluationModeV1::StrictMetricsOnly",
        ],
    );
    assert!(
        !enqueue_definition.contains("control_device,"),
        "the production evaluator forwards a caller-supplied control pointer"
    );

    let raw_view = section(&rust, "struct RawResidentGenerationGeneViewV2 {", "\n}");
    require_all(raw_view, &["control_device: *const c_void"]);
    assert!(
        !rust.contains("pub fn control_device") && !rust.contains("pub fn raw_gene_view"),
        "a public Rust API exposes the private device control or gene view"
    );
}

#[test]
fn resident_feature_store_session_is_consumed_into_one_move_only_v2_owner() {
    let feature_store = read_required("src/resident_feature_store_v3.rs");
    let search = read_required("src/resident_search_v2.rs");

    require_all(
        &feature_store,
        &[
            "pub struct ResidentFeatureStoreSearchRunV2",
            "search_run: Option<ResidentSearchRunV2>",
            "resident_import: Option<ResidentFeatureStoreImportV3>",
            "pub(crate) fn consume_into_resident_search_run_v2(",
            "self,",
            "plan: SealedResidentGenerationPlanV1",
            "pub fn record_consumer_completion(",
            "attach_population_session_v3",
        ],
    );
    let owner = section(
        &feature_store,
        "pub struct ResidentFeatureStoreSearchRunV2 {",
        "\n}",
    );
    assert!(
        !owner.contains("pub "),
        "V2/V3 owner fields must stay private"
    );
    for forbidden in [
        "impl Clone for ResidentFeatureStoreSearchRunV2",
        "impl Copy for ResidentFeatureStoreSearchRunV2",
        "impl Default for ResidentFeatureStoreSearchRunV2",
        "Deserialize for ResidentFeatureStoreSearchRunV2",
        "pub fn into_population_session",
    ] {
        assert!(
            !feature_store.contains(forbidden),
            "move-only V2/V3 authority escapes through {forbidden:?}"
        );
    }

    assert!(
        search.contains("pub(crate) fn close_v2(") && !search.contains("pub fn close_v2("),
        "raw PopulationSession return must be crate-private and used only by the V3 completion owner"
    );
}

#[test]
fn production_owner_has_no_intermediate_host_boundary_or_public_raw_handle() {
    let search = read_required("src/resident_search_v2.rs");
    let feature_store = read_required("src/resident_feature_store_v3.rs");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let population_cuda = read_required("native/prototype_b_population.cu");

    for (name, source) in [
        ("resident_search_v2.rs", search.as_str()),
        ("resident_feature_store_v3.rs", feature_store.as_str()),
    ] {
        for line in source.lines().filter(|line| line.contains("pub fn ")) {
            for forbidden in ["*mut ", "*const ", "c_void", "raw_", "native_handle"] {
                assert!(
                    !line.contains(forbidden),
                    "{name} public API exposes {forbidden:?}: {line}"
                );
            }
        }
    }

    let production_begin = section(
        &search,
        "pub fn begin_resident_search_v2(",
        "#[cfg(feature = \"cuda-device-fixtures\")]",
    );
    let v3_consume = section(
        &feature_store,
        "pub(crate) fn consume_into_resident_search_run_v2(",
        "\n}",
    );
    let generation_v2 = section(
        &generation_cuda,
        "namespace neoethos::resident_generation_v2 {",
        "}  // namespace neoethos::resident_generation_v2",
    );
    let gene_evaluator = section(
        &population_cuda,
        "neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(",
        "\n}",
    );

    for (name, source) in [
        (
            "PopulationSession::begin_resident_search_v2",
            production_begin,
        ),
        (
            "ResidentPopulationSessionV3::consume_into_resident_search_run_v2",
            v3_consume,
        ),
        ("resident_generation_v2 native namespace", generation_v2),
        ("resident gene evaluator enqueue", gene_evaluator),
    ] {
        for forbidden in [
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyDeviceToHost",
            "consume_host_metrics_v1",
            "read_metrics(",
            "read_diagnostics(",
        ] {
            assert!(
                !source.contains(forbidden),
                "production V2 owner crosses the host boundary in {name} via {forbidden:?}"
            );
        }
    }
}

#[test]
fn first_production_seam_turns_green_only_the_owned_bridge_facts() {
    let readiness = resident_search_v2_production_readiness();
    assert!(!readiness.exact_generation_semantics());
    assert!(!readiness.device_resident_generation_advance());
    assert!(readiness.device_owned_search_control());
    assert!(!readiness.immutable_scenario_admission());
    assert!(!readiness.whole_workspace_preallocated());
    assert!(!readiness.unified_device_fault_authority());
    assert!(readiness.native_bridge_production_sealed());
    assert!(readiness.terminal_cleanup_lease());
    assert!(!readiness.production_ready());
}

#[test]
fn rust_abi_and_diagnostics_have_one_production_authority() {
    let generation = read_required("src/resident_generation_v1.rs");
    let search = read_required("src/resident_search_v2.rs");
    let population = read_required("src/population.rs");

    assert_eq!(
        generation
            .matches("struct RawAllocationReceiptV1 {")
            .count(),
        1,
        "the V1 generation module must own the sole allocation-receipt type"
    );
    assert_eq!(
        search.matches("struct RawAllocationReceiptV1 {").count(),
        0,
        "Search V2 must reuse, not redeclare, the allocation-receipt ABI"
    );
    require_all(&generation, &["pub(crate) struct RawAllocationReceiptV1 {"]);
    require_all(
        &search,
        &[
            "RawAllocationReceiptV1,",
            "InvalidAdmission(#[source] CudaPopulationError)",
            "InvalidPlan(&'static str)",
            ".map_err(ResidentSearchV2Error::InvalidAdmission)?",
            "exact GA/RNG semantics, device-resident generation advance, immutable scenario admission, whole-workspace admission and unified device fault authority remain fail-closed",
        ],
    );
    assert!(
        !search.contains("InvalidFixture"),
        "production owner errors must not diagnose sealed-plan failures as fixture failures"
    );
    require_all(
        &population,
        &["self.require_strict_idle_v1(\"begin_resident_search_v2\")?;"],
    );
    assert!(
        !population.contains("begin_resident_search_fixture_v2"),
        "production admission must not use a fixture operation label"
    );
}

#[test]
fn dead_code_exceptions_are_narrow_and_counted() {
    let sources = [
        ("src/lib.rs", 1_usize),
        ("src/resident_search_v2.rs", 9),
        ("src/resident_feature_store_v3.rs", 3),
        ("src/population.rs", 5),
        ("src/resident_generation_v1.rs", 0),
    ];

    for (relative, expected_count) in sources {
        let source = read_required(relative);
        assert!(
            !source.contains("#![allow(dead_code)]"),
            "{relative} must not suppress dead-code warnings module-wide"
        );
        assert_eq!(
            source.matches("#[allow(dead_code)]").count(),
            expected_count,
            "{relative} changed the bounded dead-code allowance budget"
        );
    }

    let lib = read_required("src/lib.rs");
    require_all(
        &lib,
        &[
            "The pre-existing V1 generation owner was source-contract-only.",
            "#[allow(dead_code)]\nmod resident_generation_v1;",
        ],
    );
}

#[test]
fn real_card_v3_to_search_owner_test_is_part_of_the_normal_cuda_test_module() {
    let device = read_required("src/resident_population_session_v3_device_tests.rs");
    require_all(
        &device,
        &[
            "fn resident_store_v3_moves_into_search_v2_and_enqueues_on_real_cuda()",
            "NEOETHOS_REQUIRE_GPU",
            "discovery_generation_semantics_sha256_v1()",
            "feature_count: RESIDENT_SMC_COLUMN_NAMES_V3.len()",
            ".consume_into_resident_search_run_v2(",
            ".upload_resident_scenarios_v2(",
            ".enqueue_resident_gene_metrics_v2(&settings)?",
            ".consume_host_metrics_v1()?",
            "assert_eq!(counters.gene_upload_bytes, 0)",
            "let lease = search.record_consumer_completion()?;",
        ],
    );
    require_all(
        &device,
        &[
            "#[cfg(feature = \"cuda-device-fixtures\")]\nuse sha2::{Digest, Sha256};",
            "#[cfg(feature = \"cuda-device-fixtures\")]\n#[test]\nfn resident_store_v3_terminal_metrics_only_path_is_one_session_and_leak_free()",
        ],
    );
}
