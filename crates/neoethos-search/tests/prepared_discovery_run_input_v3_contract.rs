use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).unwrap_or_default()
}

fn read_sibling_or_empty(crate_name: &str, relative: &str) -> String {
    fs::read_to_string(manifest_dir().join("..").join(crate_name).join(relative))
        .unwrap_or_default()
}

fn normalized(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
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
            "missing prepared-input token {token:?}"
        );
    }
}

fn require_none(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "prepared-input authority contains forbidden token {token:?}"
        );
    }
}

#[test]
fn prepared_input_is_an_exclusive_move_only_cpu_or_native_typestate() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    require_all(
        &source,
        &[
            "pub enum PreparedCanonicalDiscoveryRunInputV3",
            "Cpu(PreparedCpuCanonicalDiscoveryRunInputV3)",
            "NativeCuda(PreparedNativeCudaCanonicalDiscoveryRunInputV3)",
            "CanonicalSearchInput",
            "SealedCpuNoPhysicalGpuRunDeviceAdmissionV1",
            "CanonicalSearchInputReceiptV2",
            "SealedGpuResidentFeatureStoreV3",
        ],
    );
    require_none(
        &source,
        &[
            "impl Clone for PreparedCanonicalDiscoveryRunInputV3",
            "impl Default for PreparedCanonicalDiscoveryRunInputV3",
            "Deserialize for PreparedCanonicalDiscoveryRunInputV3",
            "Option<SealedGpuResidentFeatureStoreV3>",
            "host_and_resident",
        ],
    );
}

#[test]
fn owned_cpu_input_constructor_revalidates_receipt_frame_and_runtime_math_authority() {
    let data_selection = read_or_empty("src/data_selection.rs");
    let constructor = section(
        &data_selection,
        "pub fn from_prepared_canonical_frame(",
        "\n    }",
    );
    require_all(
        constructor,
        &[
            "CanonicalFeatureExecutionReceiptV1::from_runtime_authority",
            "CanonicalSearchInputReceiptV2::from_feature_frame_with_execution",
            "CanonicalSearchRunInputV2::new",
            "base_frame.artifact().identity()",
        ],
    );
    require_none(
        constructor,
        &["unsafe", "unwrap", "from_env", "caller_feature_execution"],
    );
}

#[test]
fn dispatcher_acquires_once_and_defers_the_gpu_workspace_plan_to_the_native_arm() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let dispatcher = section(
        &source,
        "pub fn dispatch_canonical_discovery_data_preparation_v3",
        "\n}\n",
    );
    let compact = normalized(dispatcher);
    require_all(
        dispatcher,
        &[
            "FnOnce",
            "native_workspace_plan_factory",
            "cpu_factory",
            "native_factory",
            "acquire_discovery_run_device_admission_v1",
            "SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu",
            "SealedDiscoveryRunDeviceAdmissionV1::NativeCuda",
            "bind_full_discovery_workspace_plan_v1",
            "AdmittedFullDiscoveryGpuRunV1::NativeCuda",
        ],
    );
    assert_eq!(
        dispatcher
            .matches("acquire_discovery_run_device_admission_v1(")
            .count(),
        1,
        "one prepared run must perform exactly one physical/CUDA admission"
    );
    let native_arm = compact
        .find("SealedDiscoveryRunDeviceAdmissionV1::NativeCuda")
        .expect("native arm must consume the one-shot admission");
    let plan = compact
        .rfind("native_workspace_plan_factory")
        .expect("native-only workspace plan factory is missing");
    let bind = compact
        .find("bind_full_discovery_workspace_plan_v1")
        .expect("full workspace must bind the selected run");
    let materialize = compact
        .rfind("native_factory")
        .expect("native factory must receive the admitted full run");
    assert!(
        native_arm < plan && plan < bind && bind < materialize,
        "native plan/bind/materialization order is not one-shot and fail-closed"
    );
    require_none(
        dispatcher,
        &[
            "acquire_strict_discovery_device_admission_v1",
            "probe_cuda_device_count_v1",
            "runtime_available",
            "device_count",
            "selected_ordinal",
            "device_override",
            "cpu_forced",
            "allow_cpu",
        ],
    );
}

#[test]
fn cpu_factory_receives_and_returns_the_same_opaque_zero_physical_gpu_authority() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let prepare = section(
        &source,
        "pub fn prepare_canonical_discovery_run_input_v3",
        "\n}\n",
    );
    require_all(
        &normalized(prepare),
        &[
            "CpuFactory:FnOnce(",
            "SealedCpuNoPhysicalGpuRunDeviceAdmissionV1",
            "CanonicalSearchInput",
            "PreparedCpuCanonicalDiscoveryRunInputV3",
            "dispatch_canonical_discovery_data_preparation_v3",
        ],
    );
    let dispatcher = section(
        &source,
        "pub fn dispatch_canonical_discovery_data_preparation_v3",
        "\n}\n",
    );
    let cpu_arm = section(
        dispatcher,
        "SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu",
        "SealedDiscoveryRunDeviceAdmissionV1::NativeCuda",
    );
    require_none(
        cpu_arm,
        &[
            "SealedGpuResidentFeatureStoreV3",
            "materialize_gpu_only_feature_store_v3",
            "bind_full_discovery_workspace_plan_v1",
            "native_workspace_plan_factory",
        ],
    );
}

#[test]
fn prepared_cpu_run_consumes_the_physical_absence_authority_without_a_second_probe() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let route = read_or_empty("src/strict_discovery_device_route_v1.rs");
    let cpu_runner = section(
        &source,
        "fn run_cpu_prepared_discovery_v3_with_input",
        "\n}\n",
    );
    require_all(
        cpu_runner,
        &[
            "SealedStrictDiscoveryDeviceAdmissionV1::from_no_physical_gpu_admission_v1",
            "run_discovery_cycle_with_prepared_cpu_admission_v3",
        ],
    );
    require_none(
        cpu_runner,
        &[
            "acquire_discovery_run_device_admission_v1",
            "acquire_strict_discovery_device_admission_v1",
            "probe_",
            "device_count",
            "runtime_available",
        ],
    );
    require_all(
        &route,
        &[
            "SealedCpuDiscoveryRouteReceiptV2",
            "PhysicalGpuAbsence",
            "from_no_physical_gpu_admission_v1",
        ],
    );
    require_all(
        &source,
        &[
            "physical_inventory_probe_count == 1",
            "cuda_enumeration_count == 1",
            "primary_context_acquisition_count == 0",
            "run_stream_creation_count == 0",
        ],
    );
}

#[test]
fn native_factory_receives_only_the_moved_admitted_full_workspace_run() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let prepare = section(
        &source,
        "pub fn prepare_canonical_discovery_run_input_v3",
        "\n}\n",
    );
    require_all(
        &normalized(prepare),
        &[
            "NativeFactory:FnOnce(",
            "AdmittedNativeCudaFullDiscoveryRunV1",
            "CanonicalSearchInputReceiptV2",
            "SealedGpuResidentFeatureStoreV3",
            "PreparedNativeCudaCanonicalDiscoveryRunInputV3",
        ],
    );
    let dispatcher = section(
        &source,
        "pub fn dispatch_canonical_discovery_data_preparation_v3",
        "\n}\n",
    );
    let native_arm = section(
        dispatcher,
        "SealedDiscoveryRunDeviceAdmissionV1::NativeCuda",
        "\n        }",
    );
    require_none(
        native_arm,
        &[
            "CanonicalSearchInput::from_exact_series_receipt",
            "CanonicalSearchRunInputV2::new",
            "prepare_multitimeframe_features",
            "FeatureFrame",
            "Ohlcv",
            "begin_exact_population_execution_run_v1",
            "unwrap_or",
            "or_else",
        ],
    );
}

#[test]
fn prepared_runner_keeps_cpu_and_native_execution_bodies_disjoint() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let runner = section(
        &source,
        "pub fn run_prepared_canonical_discovery_with_holdout_and_progress_v3",
        "\n}\n",
    );
    require_all(
        runner,
        &[
            "PreparedCanonicalDiscoveryRunInputV3::Cpu",
            "PreparedCanonicalDiscoveryRunInputV3::NativeCuda",
            "run_cpu_prepared_discovery_v3",
            "run_native_cuda_prepared_discovery_v3",
        ],
    );
    let native_arm = section(
        runner,
        "PreparedCanonicalDiscoveryRunInputV3::NativeCuda",
        "\n        }",
    );
    require_all(
        native_arm,
        &[
            "bind_strict_resident_feature_store_v3_run_input",
            "seal_gpu_native_trim_prefilter_view_identity_v3",
            "record_resident_feature_store_consumer_completion_v3",
        ],
    );
    let compact_native_arm = normalized(native_arm);
    require_all(
        &compact_native_arm,
        &[
            "letexpected_completion_shape=(run.row_count(),run.column_count());",
            "letconsumer_completion_lease=record_resident_feature_store_consumer_completion_v3(run).context(",
            "consumer_completion_lease.rows()==expected_completion_shape.0",
            "consumer_completion_lease.columns()==expected_completion_shape.1",
        ],
    );
    let completion = compact_native_arm
        .find("letconsumer_completion_lease=record_resident_feature_store_consumer_completion_v3")
        .expect("native run must retain its move-only completion lease");
    let return_outcome = compact_native_arm
        .rfind("outcome")
        .expect("native run must return its recorded execution outcome");
    assert!(
        completion < return_outcome,
        "the resident consumer lease must remain in scope through the native outcome return"
    );
    require_none(
        native_arm,
        &[
            "CanonicalSearchRunInputV2",
            "CanonicalSearchInput::",
            ".features()",
            ".ohlcv()",
            "FeatureFrame",
            "Ohlcv",
            "Cow<",
            "begin_exact_population_execution_run_v1",
            "upload_dataset",
            "upload_parent_dataset_v1",
            "acquire_",
            "cpu",
            "fallback",
            "let _ =",
            "let _completion",
            "#[allow",
            "#[expect",
        ],
    );
}

#[test]
fn strict_v3_binder_owns_a_native_run_instead_of_attaching_to_the_host_v1_run() {
    let source = read_or_empty("src/strict_resident_feature_store_v3.rs");
    require_all(
        &source,
        &[
            "pub(crate) struct StrictResidentPopulationExecutionRunV3",
            "pub(crate) fn bind_strict_resident_feature_store_v3_run_input",
            "Result<StrictResidentPopulationExecutionRunV3",
            "pub(crate) fn record_resident_feature_store_consumer_completion_v3",
            "run: StrictResidentPopulationExecutionRunV3",
        ],
    );
    let bind = section(
        &source,
        "pub(crate) fn bind_strict_resident_feature_store_v3_run_input",
        "\n}\n",
    );
    require_none(
        bind,
        &[
            "ExactPopulationExecutionRunV1",
            "FeatureFrame",
            "Ohlcv",
            "&mut",
            "install_resident_feature_store_session_v3",
        ],
    );
}

#[test]
fn data_materialization_remains_fail_before_carrier_consumption_when_producers_are_missing() {
    let data = read_sibling_or_empty("neoethos-data", "src/core/gpu_resident_feature_store_v3.rs");
    let materialize = section(
        &data,
        "pub fn materialize_gpu_only_feature_store_v3",
        "\n}\n",
    );
    let compact = normalized(materialize);
    let resolve = compact
        .find("CrateOwnedResidentProducerFactoryV3::resolve")
        .expect("Data must resolve the complete producer census");
    let preflight = compact
        .find("preflight_gpu_only_feature_recipe_v3")
        .expect("Data must preflight before consuming the run");
    let consume = compact
        .find("into_gpu_only_run_device_admission_v3")
        .expect("Data must consume exactly one admitted full run");
    assert!(
        resolve < preflight && preflight < consume,
        "missing producers must fail before the one-shot device carrier is consumed"
    );
}

#[test]
fn app_cli_and_autoresearch_switch_the_real_entrypoints_to_the_prepared_typestate() {
    let callers = [
        (
            read_sibling_or_empty("neoethos-app", "src/app_services/discovery.rs"),
            "prepare_canonical_discovery_run_input_v3",
            "run_prepared_canonical_discovery_with_holdout_and_progress_v3",
        ),
        (
            read_sibling_or_empty("neoethos-cli", "src/canonical_full_run.rs"),
            "prepare_canonical_discovery_run_input_v3",
            "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
        ),
        (
            read_sibling_or_empty("neoethos-cli", "src/main.rs"),
            "prepare_canonical_discovery_run_input_v3",
            "run_prepared_canonical_discovery_with_holdout_and_progress_v3",
        ),
        (
            read_sibling_or_empty("neoethos-autoresearch", "src/runner/streaming.rs"),
            "run_prepared_streaming_working_set_v3",
            "run_prepared_canonical_discovery_with_holdout_and_progress_v3",
        ),
    ];
    for (index, (caller, prepare, run)) in callers.iter().enumerate() {
        require_all(caller, &[*prepare, *run]);
        assert!(
            caller.find(*prepare) < caller.find(*run),
            "caller {index} must prepare before it runs"
        );
    }
}

#[test]
fn real_prepared_callers_do_not_build_host_features_before_the_one_shot_dispatch() {
    let app = read_sibling_or_empty("neoethos-app", "src/app_services/discovery.rs");
    let cli = read_sibling_or_empty("neoethos-cli", "src/canonical_full_run.rs");
    let cli_main = read_sibling_or_empty("neoethos-cli", "src/main.rs");
    let autoresearch = read_sibling_or_empty("neoethos-autoresearch", "src/runner/streaming.rs");
    let compact_cli_main = normalized(&cli_main);
    let discover_command = section(
        &cli_main,
        "fn cmd_discover(args: &[String])",
        "\nfn cmd_batch_discover",
    );
    let compact_discover_command = normalized(discover_command);
    require_all(
        &compact_discover_command,
        &["#[cfg(not(feature=\"gpu-nvidia\"))]lethigher_refs:Vec<&str>="],
    );
    require_none(
        discover_command,
        &["let _higher_refs", "#[allow(unused_variables)]", "#[expect"],
    );
    let typed_cli_repin = "take_or_repin(std::path::Path::new(root.as_str()))";
    assert_eq!(
        compact_cli_main.matches(typed_cli_repin).count(),
        2,
        "both CLI prepared paths must pass the canonical data root through the exact Path boundary"
    );
    assert!(
        compact_cli_main
            .find(typed_cli_repin)
            .expect("CLI immutable series pin")
            < compact_cli_main
                .find("prepare_canonical_discovery_run_input_v3")
                .expect("CLI prepared admission"),
        "the CLI must pin its immutable series before acquiring the prepared run admission"
    );
    for (label, caller, dispatcher) in [
        ("app", app, "prepare_canonical_discovery_run_input_v3"),
        ("cli", cli, "prepare_canonical_discovery_run_input_v3"),
        (
            "autoresearch",
            autoresearch,
            "run_prepared_streaming_working_set_v3",
        ),
    ] {
        let prepare = caller
            .find(dispatcher)
            .unwrap_or_else(|| panic!("{label} is not migrated to prepared V3"));
        let prefix = &caller[..prepare];
        require_none(
            prefix,
            &[
                "CanonicalSearchRunInputV2::new",
                "CanonicalSearchInput::from_exact_series_receipt",
                "prepare_multitimeframe_features(",
                "run_discovery_cycle_with_holdout(",
            ],
        );
    }
}

#[test]
fn combined_canonical_run_carries_the_cpu_training_input_and_refuses_a_native_host_rebuild() {
    let prepared = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let cli = read_sibling_or_empty("neoethos-cli", "src/canonical_full_run.rs");
    require_all(
        &prepared,
        &[
            "pub struct PreparedCpuCanonicalTrendbarResearchRunV3",
            "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
            "CanonicalSearchInput",
            "into_parts",
            "GPU-resident training handoff",
        ],
    );
    require_all(
        &cli,
        &[
            "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
            "let (research, training_input) = prepared_research.into_parts();",
            "train_canonical_series_with_progress",
            "training_input",
        ],
    );
    require_none(
        &cli,
        &[
            "drop(run_input)",
            "search_input,\n        &contract",
            "CanonicalSearchInput::from_exact_series_receipt(\n        &data_root",
        ],
    );
}
