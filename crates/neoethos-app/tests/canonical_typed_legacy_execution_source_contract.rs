#[test]
fn typed_legacy_boundary_acquires_before_settings_and_keeps_same_lease_for_training() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/server/engines_control/typed_execution_v1.rs"
    ))
    .expect("typed legacy execution source must exist");

    for required in [
        "start_typed_discovery_execution_v1",
        "start_typed_training_execution_v1",
        "try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery)",
        "try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)",
        "transition_discovery_to_training_v1()",
        "TypedHigherTimeframePolicyV1::Configured",
        "RequireAutoRediscoveryEnabled",
        "await_terminal",
    ] {
        assert!(
            source.contains(required),
            "missing typed execution seam: {required}"
        );
    }

    let discovery_start = source
        .find("fn start_typed_discovery_execution_v1")
        .expect("typed Discovery start");
    let discovery_tail = &source[discovery_start..];
    let acquire = discovery_tail
        .find("try_acquire_process_execution_lease_v1")
        .expect("lease acquisition");
    let settings = discovery_tail
        .find("Settings::from_yaml")
        .expect("leased Settings load");
    let dataset = discovery_tail
        .find("resolve_unique_background_dataset_identity")
        .expect("leased dataset resolution");
    assert!(acquire < settings && settings < dataset);

    let intent_start = source
        .find("struct TypedDiscoveryExecutionIntentV1")
        .expect("intent declaration");
    let intent_end = source[intent_start..]
        .find("struct TypedTrainingExecutionIntentV1")
        .map(|offset| intent_start + offset)
        .expect("next declaration");
    let intent = &source[intent_start..intent_end];
    assert!(intent.contains("TypedDiscoveryDatasetPolicyV1"));
    assert!(source.contains("Exact(SelectedDatasetGenerationV1)"));
}
