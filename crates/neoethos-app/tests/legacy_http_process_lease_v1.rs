use std::path::PathBuf;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use neoethos_app::server::router;
use neoethos_app::server::state::{AppApiState, install_config_path};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
    SelectedDatasetGenerationV1,
};
use neoethos_search::{ProcessExecutionKindV1, try_acquire_process_execution_lease_v1};
use serde_json::json;
use tower::ServiceExt;

fn exact_selection() -> SelectedDatasetGenerationV1 {
    let identity = CanonicalDatasetIdentity::external(
        "operator-upload",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid exact dataset identity");
    SelectedDatasetGenerationV1::new(
        identity,
        format!("g1-{}.vortex", "1".repeat(64)),
        "2".repeat(64),
    )
    .expect("valid exact selected generation")
}

#[tokio::test(flavor = "current_thread")]
async fn busy_native_lease_rejects_both_legacy_http_starts_before_settings_or_data_work() {
    install_config_path(PathBuf::from(
        "/definitely/missing/chunk4e-busy-loser/config.yaml",
    ));
    let held = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::NativeResearch)
        .expect("test owns the process execution lease");

    let discovery = router(AppApiState::new())
        .oneshot(
            Request::post("/engines/discovery/start")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "dataset_selection": exact_selection(),
                        "symbol": "EURUSD",
                        "base_tf": "M5",
                    }))
                    .expect("discovery request JSON"),
                ))
                .expect("discovery request"),
        )
        .await
        .expect("discovery response");
    assert_eq!(
        discovery.status(),
        StatusCode::CONFLICT,
        "Busy must win before the deliberately missing Settings path or nonexistent generation is touched",
    );

    let training = router(AppApiState::new())
        .oneshot(
            Request::post("/engines/training/start")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "training_handoff": "a".repeat(64),
                    }))
                    .unwrap(),
                ))
                .expect("training request"),
        )
        .await
        .expect("training response");
    assert_eq!(
        training.status(),
        StatusCode::CONFLICT,
        "Busy must win before the deliberately missing Settings path is touched",
    );

    drop(held);
    let released = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration)
        .expect("Busy HTTP losers must not retain or replace the held authority");
    drop(released);
}

#[tokio::test(flavor = "current_thread")]
async fn training_without_a_discovery_handoff_is_rejected_before_work() {
    let response = router(AppApiState::new())
        .oneshot(
            Request::post("/engines/training/start")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

fn function_body<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start}"));
    let tail = &source[start..];
    let end = tail.find(end).unwrap_or_else(|| panic!("missing {end}"));
    &tail[..end]
}

#[test]
fn legacy_http_routes_delegate_to_the_typed_lease_lane_and_keep_exact_selection() {
    let adapter = include_str!("../src/server/engines_control.rs");
    let typed = include_str!("../src/server/engines_control/typed_execution_v1.rs");

    let discovery = function_body(
        adapter,
        "pub async fn discovery_start(",
        "pub async fn discovery_stop(",
    );
    for required in [
        "TypedDiscoveryDatasetPolicyV1::Exact",
        "start_typed_discovery_execution_v1",
        "await_admission_v1().await",
        "training_after_success: false",
        "detach_typed_legacy_execution_observer_v1",
    ] {
        assert!(
            discovery.contains(required),
            "missing Discovery adapter seam: {required}"
        );
    }
    for forbidden in [
        "Settings::from_yaml",
        "resolve_data_root",
        "pin_discovery_input",
        "start_discovery_job",
        "spawn_state_drainer",
    ] {
        assert!(
            !discovery.contains(forbidden),
            "legacy Discovery HTTP handler bypasses leased worker through {forbidden}",
        );
    }

    let training = function_body(
        adapter,
        "pub async fn training_start(",
        "pub async fn training_stop(",
    );
    assert!(training.contains("start_typed_training_execution_v1"));
    assert!(training.contains("await_admission_v1().await"));
    assert!(training.contains("detach_typed_legacy_execution_observer_v1"));
    assert!(training.contains("TypedTrainingSelectionPolicyV1::DiscoveryHandoff"));
    assert!(training.contains("identity_sha256: body.training_handoff"));
    assert!(!training.contains("TypedTrainingSelectionPolicyV1::Configured"));
    assert!(!training.contains("TypedTrainingSelectionPolicyV1::Exact"));
    for forbidden in [
        "Settings::from_yaml",
        "start_training_job",
        "spawn_state_drainer",
    ] {
        assert!(
            !training.contains(forbidden),
            "legacy Training HTTP handler bypasses leased worker through {forbidden}",
        );
    }
    assert!(
        !adapter.contains("spawn_auto_chained_training"),
        "the old drop/restart auto-chain must not remain callable",
    );

    for required in [
        "enum TypedDiscoveryDatasetPolicyV1",
        "Exact(SelectedDatasetGenerationV1)",
        "enum TypedTrainingSelectionPolicyV1",
        "enum TypedLegacyExecutionAdmissionV1",
        "let _lease = lease;",
    ] {
        assert!(
            typed.contains(required),
            "missing typed exact/owned-lease seam: {required}"
        );
    }
    let start = typed
        .find("fn start_typed_discovery_execution_v1")
        .expect("typed Discovery start");
    let tail = &typed[start..];
    let acquire = tail
        .find("try_acquire_process_execution_lease_v1")
        .expect("lease acquisition");
    let worker = tail
        .find("spawn_discovery_worker_v1")
        .expect("leased worker spawn");
    let settings = tail
        .find("DiscoverySettingsSource::load(&config_path)")
        .expect("leased Settings load");
    assert!(acquire < worker && worker < settings);

    let worker = function_body(
        typed,
        "fn spawn_discovery_worker_v1(",
        "fn spawn_training_worker_v1(",
    );
    assert!(worker.contains("let mut lease = lease;"));
    assert!(worker.contains("let expected_series = request.pinned_input.receipt().clone();"));
    assert!(worker.contains("if intent.training_after_success {"));
    assert!(worker.contains("continue_discovery_training_v1("));
    assert!(worker.contains("run_training_intent_v1("));
    assert!(worker.contains("Some(expected_series)"));
    assert!(!worker.contains("start_typed_training_execution_v1("));
    assert!(!worker.contains("start_training_job("));
    let continuation = function_body(
        typed,
        "async fn continue_discovery_training_v1<",
        "async fn prepare_discovery_request_v1(",
    );
    assert!(continuation.contains("TypedTrainingSelectionPolicyV1::DiscoveryHandoff"));
    assert!(!continuation.contains("TypedTrainingSelectionPolicyV1::Exact"));
    assert!(!continuation.contains("try_acquire_process_execution_lease_v1"));
    let transition = continuation
        .find("transition_discovery_to_training_v1()")
        .unwrap();
    assert!(
        continuation
            .find("discovery_snapshot.state != JobState::Succeeded")
            .unwrap()
            < transition
    );
    assert!(continuation.find("handoff::handoff_path(").unwrap() < transition);
    assert!(transition < continuation.find("run_training(intent).await").unwrap());
}
