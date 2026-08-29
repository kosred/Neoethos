use neoethos_app::app_services::execution_admission::{
    AdmissionError, ExecutionAdmissionCoordinator,
};
use neoethos_app::app_state::AppExecutionState;
use neoethos_core::execution_budget::{
    AcquireError, AuxiliarySlotLease, AuxiliarySlotLimit, CpuLease, CpuPermitBroker,
    CpuPermitRequest, WorkerLimit,
};
use std::time::Duration;

fn workers(value: usize) -> WorkerLimit {
    WorkerLimit::new(value).expect("test worker count is positive")
}

fn auxiliary_slots(value: usize) -> AuxiliarySlotLimit {
    AuxiliarySlotLimit::new(value).expect("test auxiliary slot count is positive")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn auxiliary_saturation_reserves_no_cpu_and_cancelled_waiter_leaks_nothing() {
    let broker = CpuPermitBroker::new(workers(2));
    let coordinator = ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
        broker.clone(),
        auxiliary_slots(1),
    )
    .expect("coordinator starts");
    let client = coordinator.client();
    let held = client
        .admit_import(CpuPermitRequest::local(workers(1)))
        .await
        .expect("first import receives CPU and the only auxiliary slot");
    let cancelled = client
        .submit_import(CpuPermitRequest::local(workers(1)))
        .expect("second import queues atomically");

    tokio::time::timeout(Duration::from_millis(250), async {
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    })
    .await
    .expect("auxiliary-slot waiting never blocks the sole Tokio worker");

    let snapshot = coordinator.admission_snapshot();
    assert_eq!(snapshot.cpu.available_permits, 1);
    assert_eq!(snapshot.cpu.live_reserved_sum, 1);
    assert_eq!(snapshot.available_auxiliary_slots, 0);
    assert_eq!(snapshot.live_auxiliary_slots, 1);

    drop(cancelled);
    drop(held);

    let full_width = tokio::time::timeout(
        Duration::from_secs(2),
        client.admit_import(CpuPermitRequest::local(workers(2))),
    )
    .await
    .expect("lease return wakes the coordinator")
    .expect("cancelled waiter retained neither resource");
    drop(full_width);

    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cpu_saturation_reserves_no_auxiliary_slot() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator = ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
        broker.clone(),
        auxiliary_slots(1),
    )
    .expect("coordinator starts");
    let client = coordinator.client();
    let held_cpu = client
        .admit(CpuPermitRequest::local(workers(1)))
        .await
        .expect("ordinary CPU request is admitted without an auxiliary slot");
    let pending_import = client
        .submit_import(CpuPermitRequest::local(workers(1)))
        .expect("import queues while CPU is exhausted");

    tokio::time::sleep(Duration::from_millis(10)).await;
    let snapshot = coordinator.admission_snapshot();
    assert_eq!(snapshot.cpu.available_permits, 0);
    assert_eq!(snapshot.available_auxiliary_slots, 1);
    assert_eq!(snapshot.live_auxiliary_slots, 0);

    drop(held_cpu);
    let admitted_import = tokio::time::timeout(Duration::from_secs(2), pending_import.wait())
        .await
        .expect("CPU return wakes the coordinator")
        .expect("the complete import grant is admitted");
    drop(admitted_import);

    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn direct_shared_broker_return_cannot_leave_the_coordinator_asleep() {
    let broker = CpuPermitBroker::new(workers(1));
    let raw_lease = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .expect("raw request is valid")
        .expect("CPU capacity is initially free");
    let coordinator = ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
        broker.clone(),
        auxiliary_slots(1),
    )
    .expect("coordinator starts");
    let client = coordinator.client();
    let initial_grant_cycles = coordinator.admission_snapshot().grant_cycles_completed;
    let pending = client
        .submit_import(CpuPermitRequest::local(workers(1)))
        .expect("import queues behind the shared raw lease");

    tokio::time::timeout(Duration::from_secs(2), async {
        while coordinator.admission_snapshot().grant_cycles_completed <= initial_grant_cycles {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("coordinator completed a failed grant pass while CPU was held");
    assert_eq!(
        coordinator.admission_snapshot().available_auxiliary_slots,
        1
    );
    drop(raw_lease);

    let admitted = tokio::time::timeout(Duration::from_millis(500), pending.wait())
        .await
        .expect("raw broker return is eventually observed without another command")
        .expect("the complete import grant is admitted");
    drop(admitted);
    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn mixed_requests_preserve_child_priority_and_fifo() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator = ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
        broker.clone(),
        auxiliary_slots(1),
    )
    .expect("coordinator starts");
    let client = coordinator.client();
    let held = client
        .admit(CpuPermitRequest::local(workers(1)))
        .await
        .expect("initial request occupies CPU capacity");

    let local_import = client
        .submit_import(CpuPermitRequest::local(workers(1)))
        .expect("local import queues first");
    let first_child = client
        .submit(CpuPermitRequest::child(workers(1)))
        .expect("ordinary child queues second");
    let second_child = client
        .submit_import(CpuPermitRequest::child(workers(1)))
        .expect("import child queues third");
    let local_task = tokio::spawn(async move { local_import.wait().await });
    let second_child_task = tokio::spawn(async move { second_child.wait().await });

    drop(held);
    let first_child_lease = tokio::time::timeout(Duration::from_secs(2), first_child.wait())
        .await
        .expect("first child wakes")
        .expect("first child is admitted before the later child");
    assert!(!second_child_task.is_finished(), "child FIFO was bypassed");
    assert!(
        !local_task.is_finished(),
        "local work bypassed child priority"
    );

    drop(first_child_lease);
    let second_child_lease = tokio::time::timeout(Duration::from_secs(2), second_child_task)
        .await
        .expect("second child wakes")
        .expect("second child task joins")
        .expect("second child is admitted before local work");
    assert!(
        !local_task.is_finished(),
        "local work bypassed child priority"
    );

    drop(second_child_lease);
    let local_lease = tokio::time::timeout(Duration::from_secs(2), local_task)
        .await
        .expect("local import wakes after both children")
        .expect("local task joins")
        .expect("local import is eventually admitted");
    drop(local_lease);

    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn import_wrapper_transfers_raw_leases_to_blocking_work_and_wakes_waiters() {
    let broker = CpuPermitBroker::new(workers(1));
    let app_state =
        AppExecutionState::new_with_auxiliary_slots(broker.clone(), workers(1), auxiliary_slots(1))
            .expect("app execution state starts");
    let client = app_state.admission_client();
    let admitted = client
        .admit_import(CpuPermitRequest::local(workers(1)))
        .await
        .expect("import receives a complete grant");
    let next = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("ordinary work queues behind the import");
    let next_task = tokio::spawn(async move { next.wait().await });
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let observed_broker = broker.clone();

    let blocking_task = tokio::task::spawn_blocking(move || {
        admitted.execute(
            move |cpu_lease: &mut CpuLease, auxiliary_slot: &AuxiliarySlotLease| {
                assert_eq!(cpu_lease.width(), workers(1));
                assert_eq!(auxiliary_slot.index(), 0);
                assert_eq!(observed_broker.snapshot().live_reserved_sum, 1);
                assert!(matches!(
                    observed_broker.try_acquire(CpuPermitRequest::local(workers(1))),
                    Err(AcquireError::NestedAcquisition)
                ));
                entered_tx.send(()).expect("async test is waiting");
                release_rx.recv().expect("test releases blocking import");
                42
            },
        )
    });

    entered_rx.await.expect("blocking import started");
    assert_eq!(app_state.admission_snapshot().live_auxiliary_slots, 1);
    assert!(
        !next_task.is_finished(),
        "CPU capacity returned before import exit"
    );
    tokio::time::timeout(Duration::from_millis(250), async {
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    })
    .await
    .expect("blocking import does not occupy the sole Tokio worker");

    release_tx.send(()).expect("blocking import is still live");
    assert_eq!(blocking_task.await.expect("blocking task joins"), 42);
    let next_lease = tokio::time::timeout(Duration::from_secs(2), next_task)
        .await
        .expect("resource-return wake reaches the coordinator")
        .expect("ordinary task joins")
        .expect("ordinary work is admitted after both resources return");
    drop(next_lease);

    let snapshot = app_state.admission_snapshot();
    assert_eq!(snapshot.cpu.available_permits, 1);
    assert_eq!(snapshot.available_auxiliary_slots, 1);
    app_state.shutdown().expect("app state shuts down");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn invalid_import_head_is_rejected_without_consuming_the_auxiliary_slot() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator = ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
        broker.clone(),
        auxiliary_slots(1),
    )
    .expect("coordinator starts");
    let client = coordinator.client();
    let invalid = client
        .submit_import(CpuPermitRequest::child(workers(2)))
        .expect("invalid import is reported asynchronously");
    let valid = client
        .submit_import(CpuPermitRequest::local(workers(1)))
        .expect("valid import queues behind the invalid head");

    assert!(matches!(
        invalid.wait().await,
        Err(AdmissionError::Acquire(
            AcquireError::ExceedsInstalledLimit {
                requested: 2,
                installed: 1
            }
        ))
    ));
    let valid_lease = tokio::time::timeout(Duration::from_secs(2), valid.wait())
        .await
        .expect("invalid head does not stall the queue")
        .expect("valid import is admitted");
    drop(valid_lease);

    let snapshot = coordinator.admission_snapshot();
    assert_eq!(snapshot.available_auxiliary_slots, 1);
    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[test]
fn app_default_uses_the_authoritative_source_seal_slot_limit() {
    assert_eq!(
        neoethos_app::app_state::platform_import_auxiliary_slot_limit().get(),
        neoethos_data::source_seal_slot_limit()
    );
}
