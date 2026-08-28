use super::*;

#[test]
fn generation_policy_distinguishes_exact_api_from_validation_floor() {
    let mut exact = DiscoveryConfig::default();
    exact.generations = 20;
    TypedDiscoveryOverridesV1::exact_generations(7)
        .expect("exact override")
        .apply(&mut exact);
    assert_eq!(exact.generations, 7);

    let mut floor = DiscoveryConfig::default();
    floor.generations = 20;
    TypedDiscoveryOverridesV1::minimum_generations(30)
        .expect("floor override")
        .apply(&mut floor);
    assert_eq!(floor.generations, 30);
}

#[tokio::test]
async fn panicking_training_worker_preserves_token_and_updates_training_slot() {
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)
        .expect("training lease");
    let lease_token = lease.token();
    let cancel = CancellationFlag::new();
    let initial = TypedLegacyExecutionSnapshotV1::new(
        lease_token,
        ProcessExecutionKindV1::Training,
        queued_snapshot_v1(JobKind::Training),
    );
    let (snapshot_tx, snapshots) = watch::channel(initial);
    let (admission_tx, admission) = oneshot::channel();
    let (terminal_tx, terminal) = oneshot::channel();
    let worker = tokio::spawn(async move {
        let _lease = lease;
        let _snapshot_tx = snapshot_tx;
        let _admission_tx = admission_tx;
        let _terminal_tx = terminal_tx;
        panic!("synthetic training worker panic");
    });
    let handle = TypedLegacyExecutionJobHandleV1 {
        lease_token,
        initial_kind: JobKind::Training,
        cancel: cancel.clone(),
        snapshots,
        admission,
        terminal,
        worker,
    };
    let state = AppApiState::new();
    state.install_engine(JobKind::Training, cancel).await;
    detach_typed_legacy_execution_observer_v1(state.clone(), handle);

    for _ in 0..100 {
        if state.engine_state(JobKind::Training).await == EngineRunState::Failed {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        state.engine_state(JobKind::Training).await,
        EngineRunState::Failed
    );
    let replacement = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration)
        .expect("observer must await worker lease release");
    assert_ne!(replacement.token(), lease_token);
}

#[test]
fn cancelled_preparation_is_terminal_cancelled_not_failed() {
    let cancel = CancellationFlag::new();
    cancel.request();
    let snapshot =
        preparation_error_snapshot_v1(JobKind::Discovery, &cancel, "cancelled before Settings");
    assert_eq!(snapshot.state, JobState::Cancelled);
}
