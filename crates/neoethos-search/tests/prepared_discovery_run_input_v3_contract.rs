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
        "pub fn from_prepared_canonical_frame_with_control(",
        "\n    }",
    );
    require_all(
        constructor,
        &[
            "CanonicalFeatureExecutionReceiptV1::from_runtime_authority",
            "canonical_feature_execution_authority_for_policy_v1",
            "control.resolved_indicator_compute_policy()",
            "CanonicalSearchRunInputV2::from_fresh_feature_frame_with_execution",
            "base_frame.artifact().identity()",
        ],
    );
    require_none(constructor, &["resolved_canonical_feature_execution_authority_v1()"]);
    let fresh_bind = section(
        &data_selection,
        "fn from_fresh_feature_frame_with_execution(",
        "\n    }",
    );
    require_all(
        fresh_bind,
        &[
            "CanonicalSearchInputReceiptV2::from_feature_frame_with_execution",
            "Self::validate_values",
            "Self::bind_base_frame",
        ],
    );
    require_none(
        &format!("{constructor}\n{fresh_bind}"),
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
fn prepared_cpu_run_revalidates_its_carried_receipt_before_search() {
    let source = read_or_empty("src/prepared_discovery_run_input_v3.rs");
    let cpu_runner = section(
        &source,
        "fn run_cpu_prepared_discovery_v3_with_input",
        "\n}\n",
    );
    let compact = normalized(cpu_runner);
    require_all(
        &compact,
        &[
            "CanonicalSearchRunInputV2::new(prepared.receipt,prepared.input.features(),prepared.input.base_frame(),)",
            "run_discovery_cycle_with_prepared_cpu_admission_v3(&input,",
        ],
    );
    require_none(cpu_runner, &[".as_run_input()", "from_fresh_feature_frame"]);
    assert!(
        compact.find("CanonicalSearchRunInputV2::new(")
            < compact.find("run_discovery_cycle_with_prepared_cpu_admission_v3("),
        "prepared values must match the carried receipt before Search can consume them"
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
            "CanonicalGpuResidentSearchInputReceiptV3",
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
            "consume_strict_resident_population_execution_run_v3",
            "seal_gpu_native_trim_prefilter_view_identity_v3",
            "retain_resident_completion_until_ready_v1",
            "drop(consumer_completion_lease)",
        ],
    );
    let compact_native_arm = normalized(native_arm);
    let completion = compact_native_arm
        .find("retain_resident_completion_until_ready_v1(consumer_completion_lease)")
        .expect("native run must await its move-only completion lease");
    let release = compact_native_arm
        .find("drop(consumer_completion_lease)")
        .expect("native run must release only its completed consumer lease");
    let return_outcome = compact_native_arm
        .rfind("outcome")
        .expect("native run must return its recorded execution outcome");
    assert!(
        completion < release && release < return_outcome,
        "the resident consumer lease must complete before release and native outcome return"
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
            "pub struct StrictResidentPopulationExecutionRunV3",
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
    let prepare = section(
        &data,
        "pub fn prepare_gpu_only_feature_materialization_v3",
        "\n}\n",
    );
    let compact_prepare = normalized(prepare);
    let resolve = compact_prepare
        .find("CrateOwnedResidentProducerFactoryV3::resolve")
        .expect("Data must resolve the complete producer census");
    let preflight = compact_prepare
        .find("preflight_gpu_only_feature_recipe_v3")
        .expect("Data must preflight the complete recipe");
    assert!(
        resolve < preflight,
        "missing producers must fail while preparing the recipe"
    );
    let compatibility = section(&data, "pub fn materialize_gpu_only_feature_store_v3", "\n}");
    let compact_compatibility = normalized(compatibility);
    let prepare_at = compact_compatibility
        .find("prepare_gpu_only_feature_materialization_v3")
        .expect("compatibility entrypoint must prepare before binding");
    let materialize_at = compact_compatibility
        .find("materialize_prepared_gpu_only_feature_store_v3")
        .expect("compatibility entrypoint must consume only a prepared recipe");
    assert!(prepare_at < materialize_at);
    let materialize = section(
        &data,
        "pub fn materialize_prepared_gpu_only_feature_store_v3",
        "\n}",
    );
    require_all(materialize, &["into_gpu_only_run_device_admission_v3"]);
}

#[test]
fn app_cli_and_autoresearch_switch_the_real_entrypoints_to_the_prepared_typestate() {
    let callers = [
        (
            read_sibling_or_empty("neoethos-app", "src/app_services/discovery.rs"),
            "prepare_canonical_discovery_run_input_v3",
            // The UI uses the explicit trendbar-research boundary, not the
            // quote-validated financial execution entrypoint used by CLI.
            "run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3",
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
        (
            "app",
            app.as_str(),
            "prepare_canonical_discovery_run_input_v3",
        ),
        (
            "cli",
            discover_command,
            "prepare_canonical_discovery_run_input_v3",
        ),
        (
            "autoresearch",
            autoresearch.as_str(),
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
fn receipt_bound_training_reopens_the_exact_series_without_discovery_output_reconstruction() {
    let cli = read_sibling_or_empty("neoethos-cli", "src/canonical_full_run.rs");
    let training = section(
        &cli,
        "pub fn train_receipt_bound(args: &[String], settings: &neoethos_core::Settings)",
        "#[cfg(not(feature = \"gpu-nvidia-full\"))]",
    );
    require_all(
        training,
        &[
            "CanonicalSearchInputReceiptV2::from_json_bytes",
            "validate_input_receipt_against_series",
            "CanonicalTrendbarResearchExecutionContractV3::new",
            "train_canonical_series_receipt_with_progress",
            "&input_receipt",
            "&contract",
        ],
    );
    require_none(
        training,
        &[
            "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
            "CanonicalSearchInput::from_exact_series_receipt",
            "prepare_multitimeframe_features(",
        ],
    );
}
