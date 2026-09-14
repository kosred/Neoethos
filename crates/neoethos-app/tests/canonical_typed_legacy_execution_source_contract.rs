#[test]
fn typed_boundary_acquires_before_settings_and_retains_the_discovery_owner() {
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
        "let _lease = lease;",
        "if intent.training_after_success {",
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
        .find("DiscoverySettingsSource::load(&config_path)")
        .expect("leased Settings load");
    let dataset = discovery_tail
        .find("resolve_unique_background_dataset_identity")
        .expect("leased dataset resolution");
    assert!(acquire < settings && settings < dataset);
    let continuation = source
        .split_once("async fn continue_discovery_training_v1<")
        .unwrap()
        .1
        .split_once("async fn prepare_discovery_request_v1(")
        .unwrap()
        .0;
    let succeeded = continuation
        .find("discovery_snapshot.state != JobState::Succeeded")
        .unwrap();
    let identity = continuation.find("handoff::handoff_path(").unwrap();
    let transition = continuation
        .find("transition_discovery_to_training_v1()")
        .unwrap();
    let training = continuation.find("run_training(intent).await").unwrap();
    assert!(succeeded < identity && identity < transition && transition < training);
    assert!(continuation.contains("TypedTrainingSelectionPolicyV1::DiscoveryHandoff"));
    assert!(continuation.contains("handoffs.next().is_some()"));
    assert!(continuation.contains("cancel.is_requested()"));
    assert!(!continuation.contains("TypedTrainingSelectionPolicyV1::Exact"));
    assert!(!continuation.contains("try_acquire_process_execution_lease_v1"));
    let binding = source
        .split_once("let selected = handoff::load(")
        .unwrap()
        .1
        .split_once("let settings = handoff::settings_for_series(")
        .unwrap()
        .0;
    for required in [
        "validate_follow_on_series_v1(",
        "expected_series.as_ref()",
        "selected.canonical_series()",
    ] {
        assert!(
            binding.contains(required),
            "missing exact follow-on binding: {required}"
        );
    }

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
