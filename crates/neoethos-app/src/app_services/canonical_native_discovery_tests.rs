use super::*;

#[test]
fn native_lane_source_cannot_enter_legacy_discovery_or_training() {
    let source = include_str!("canonical_native_discovery.rs");
    for forbidden in [
        "DiscoveryResult",
        "DiscoveryUpdated",
        "model_targets",
        "start_training_job",
        "TrainingRequest",
    ] {
        assert!(
            !source.contains(forbidden),
            "native lane must not contain legacy seam {forbidden}"
        );
    }
    assert!(source.contains("ProcessExecutionKindV1::NativeResearch"));
    assert!(source.contains("run_canonical_native_discovery_generation_zero_from_ref_v1"));
}

#[tokio::test]
async fn awaiting_terminal_waits_until_the_worker_owned_lease_is_released() {
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::NativeResearch)
        .expect("native lease");
    let lease_token = lease.token();
    let cancellation = CanonicalNativeCancellationTokenV1::new();
    let (_snapshot_tx, snapshots) =
        watch::channel(CanonicalNativeResearchSnapshotV1::queued(lease_token));
    let (terminal_tx, terminal) = oneshot::channel();
    let worker = tokio::spawn(async move {
        let _lease = lease;
        let failure = CanonicalNativeResearchFailureV1::worker_panicked("synthetic terminal");
        let _ = terminal_tx.send(CanonicalNativeResearchTerminalSnapshotV1::WorkerPanicked(
            failure,
        ));
    });
    let handle = CanonicalNativeResearchJobHandleV1 {
        cancellation,
        snapshots,
        terminal,
        worker,
    };

    assert!(matches!(
        handle.await_terminal().await,
        CanonicalNativeResearchTerminalSnapshotV1::WorkerPanicked(_)
    ));
    let replacement = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery)
        .expect("await_terminal must join through lease release");
    assert_ne!(replacement.token(), lease_token);
}

#[test]
fn failure_stage_and_code_names_are_stable_not_debug_strings() {
    assert_eq!(
        stage_name_v1(CanonicalNativeDiscoveryExecutionStageV1::ExactSourcePin),
        "exact_source_pin"
    );
    assert_eq!(
        code_name_v1(CanonicalNativeDiscoveryExecutionErrorCodeV1::ExactGenerationConflict),
        "exact_generation_conflict"
    );
    let panic = CanonicalNativeResearchFailureV1::worker_panicked("panic");
    assert_eq!(panic.stable_stage(), "native_worker");
    assert_eq!(panic.stable_code(), "worker_panicked");
}

#[tokio::test]
async fn native_status_is_nested_and_cancellation_stays_active_until_terminal() {
    use crate::server::state::{AppApiState, CanonicalNativeResearchCancellationOutcomeV1};
    use crate::server::system_status;
    use axum::{Json, extract::State};

    let state = AppApiState::new();
    let token = CanonicalNativeCancellationTokenV1::new();
    let observed_token = token.clone();
    state
        .install_canonical_native_research_v1(
            token,
            CanonicalNativeResearchSnapshotV1::running(71, "native_preflight", 2_500),
        )
        .await;
    assert_eq!(
        state.cancel_canonical_native_research_exact_v1(70).await,
        CanonicalNativeResearchCancellationOutcomeV1::TokenMismatch
    );
    assert!(!observed_token.is_cancelled());
    assert_eq!(
        state.cancel_canonical_native_research_exact_v1(71).await,
        CanonicalNativeResearchCancellationOutcomeV1::Requested
    );
    assert!(observed_token.is_cancelled());

    let Json(dto) = system_status::engines(State(state.clone())).await;
    let value = serde_json::to_value(dto).expect("serialize status DTO");
    assert_eq!(
        value["canonicalNativeResearch"]["state"],
        serde_json::json!("Running")
    );
    assert_eq!(
        value["canonicalNativeResearch"]["cancellationRequested"],
        serde_json::json!(true)
    );
    assert_eq!(
        value["canonicalNativeResearch"]["leaseToken"],
        serde_json::json!("71")
    );
    assert_eq!(value["discovery"], serde_json::json!("Idle"));
    assert_eq!(value["training"], serde_json::json!("Idle"));

    let terminal = CanonicalNativeResearchTerminalSnapshotV1::WorkerPanicked(
        CanonicalNativeResearchFailureV1::worker_panicked("synthetic panic"),
    );
    state
        .update_canonical_native_research_v1(CanonicalNativeResearchSnapshotV1::from_terminal(
            71, terminal,
        ))
        .await;
    let Json(dto) = system_status::engines(State(state)).await;
    let value = serde_json::to_value(dto).expect("serialize terminal status DTO");
    assert_eq!(
        value["canonicalNativeResearch"]["state"],
        serde_json::json!("WorkerPanicked")
    );
    assert_eq!(
        value["canonicalNativeResearch"]["failureCode"],
        serde_json::json!("worker_panicked")
    );
    assert_eq!(
        value["canonicalNativeResearch"]["leaseToken"],
        serde_json::Value::Null
    );
}
