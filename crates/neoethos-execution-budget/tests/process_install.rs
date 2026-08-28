use std::process::Command;
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use neoethos_execution_budget::{
    AcquireError, BudgetCap, BudgetCapProvenance, CancellationToken, CapacityDetection,
    CapacityDetectionSource, CoordinationScope, CpuPermitBroker, CpuPermitRequest, CpuPriority,
    DetectionDiagnosticCode, ExecutionBudgetRequest, InstallError, LogicalThreadCount, WorkerLimit,
    install_process_budget, resolve_execution_budget,
};

fn logical(value: usize) -> LogicalThreadCount {
    LogicalThreadCount::new(value).expect("test logical-thread count must be positive")
}

fn workers(value: usize) -> WorkerLimit {
    WorkerLimit::new(value).expect("test worker limit must be positive")
}

fn supplied_request(effective_logical_threads: usize) -> ExecutionBudgetRequest {
    ExecutionBudgetRequest {
        host_logical_threads: None,
        detection: CapacityDetection::supplied(logical(effective_logical_threads)),
        persistent_limit: None,
        legacy_persistent_limit: None,
        parent_limit: None,
        coordination_scope: CoordinationScope::ProcessLocal,
    }
}

#[test]
fn resolution_automatic_limit_reserves_exactly_two_when_possible() {
    let cases = [
        (1, 1, 0),
        (2, 1, 1),
        (3, 1, 2),
        (4, 2, 2),
        (12, 10, 2),
        (23, 21, 2),
        (96, 94, 2),
    ];

    for (effective, expected_automatic, expected_reserved) in cases {
        let resolved = resolve_execution_budget(supplied_request(effective)).unwrap();
        assert_eq!(resolved.effective_logical_threads.get(), effective);
        assert_eq!(resolved.reserved_logical_threads, expected_reserved);
        assert_eq!(resolved.automatic_worker_limit.get(), expected_automatic);
        assert_eq!(resolved.effective_worker_limit.get(), expected_automatic);
        assert_eq!(
            resolved.capacity_source,
            CapacityDetectionSource::SuppliedForResolution
        );
    }
}

#[test]
fn resolution_caps_compose_by_minimum_and_preserve_provenance() {
    let request = ExecutionBudgetRequest {
        host_logical_threads: Some(logical(96)),
        detection: CapacityDetection::supplied(logical(23)),
        persistent_limit: Some(BudgetCap::persistent(workers(18))),
        legacy_persistent_limit: Some(BudgetCap::legacy(workers(16))),
        parent_limit: Some(BudgetCap::parent(workers(7))),
        coordination_scope: CoordinationScope::ManagedProcessTree,
    };

    let resolved = resolve_execution_budget(request).unwrap();
    assert_eq!(resolved.host_logical_threads.unwrap().get(), 96);
    assert_eq!(resolved.effective_logical_threads.get(), 23);
    assert_eq!(resolved.automatic_worker_limit.get(), 21);
    assert_eq!(resolved.effective_worker_limit.get(), 7);
    assert_eq!(
        resolved.persistent_limit.unwrap().provenance,
        BudgetCapProvenance::PersistentSetting
    );
    assert_eq!(
        resolved.legacy_persistent_limit.unwrap().provenance,
        BudgetCapProvenance::LegacyPersistentSetting
    );
    assert_eq!(
        resolved.parent_limit.unwrap().provenance,
        BudgetCapProvenance::ParentAssignment
    );
    assert_eq!(
        resolved.coordination_scope,
        CoordinationScope::ManagedProcessTree
    );
}

#[test]
fn resolution_oversized_request_never_enlarges_the_automatic_limit() {
    let mut request = supplied_request(12);
    request.persistent_limit = Some(BudgetCap::persistent(workers(9_999)));
    let resolved = resolve_execution_budget(request).unwrap();
    assert_eq!(resolved.automatic_worker_limit.get(), 10);
    assert_eq!(resolved.effective_worker_limit.get(), 10);
}

#[test]
fn resolution_detects_current_process_capacity_and_subtracts_the_reserve_once() {
    let detection = CapacityDetection::detect();
    let effective = detection.effective_logical_threads.get();
    let expected_reserved = effective.saturating_sub(1).min(2);
    let expected_automatic = effective - expected_reserved;
    let resolved = resolve_execution_budget(ExecutionBudgetRequest {
        host_logical_threads: None,
        detection,
        persistent_limit: None,
        legacy_persistent_limit: None,
        parent_limit: None,
        coordination_scope: CoordinationScope::ProcessLocal,
    })
    .unwrap();

    assert_eq!(resolved.reserved_logical_threads, expected_reserved);
    assert_eq!(resolved.automatic_worker_limit.get(), expected_automatic);
    assert_eq!(resolved.effective_worker_limit.get(), expected_automatic);
    println!(
        "detected_effective_logical_threads={effective} reserved_logical_threads={expected_reserved} automatic_worker_limit={expected_automatic} source={:?}",
        resolved.capacity_source
    );
}

#[test]
fn resolution_rejects_zero_and_mismatched_cap_sources() {
    assert!(LogicalThreadCount::new(0).is_err());
    assert!(WorkerLimit::new(0).is_err());

    let mut request = supplied_request(12);
    request.persistent_limit = Some(BudgetCap::parent(workers(4)));
    let error = resolve_execution_budget(request).unwrap_err();
    assert!(error.to_string().contains("persistent_limit"));
    assert!(error.to_string().contains("ParentAssignment"));
}

#[test]
fn process_install_subprocess_cases() {
    const CASE_ENV: &str = "NEOETHOS_EXECUTION_BUDGET_TEST_CASE";
    if let Ok(case) = std::env::var(CASE_ENV) {
        run_install_case(&case);
        return;
    }

    for case in [
        "equal_is_idempotent",
        "conflict_fails",
        "detection_failure_falls_back",
        "broker_uses_only_final_limit",
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "process_install_subprocess_cases", "--nocapture"])
            .env(CASE_ENV, case)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "subprocess case {case} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn run_install_case(case: &str) {
    match case {
        "equal_is_idempotent" => {
            let mut request = supplied_request(12);
            request.persistent_limit = Some(BudgetCap::persistent(workers(4)));
            let first = install_process_budget(request.clone()).unwrap();
            let second = install_process_budget(request).unwrap();
            assert!(std::ptr::eq(first, second));
        }
        "conflict_fails" => {
            let first = supplied_request(12);
            install_process_budget(first).unwrap();
            let conflicting = supplied_request(8);
            let error = install_process_budget(conflicting).unwrap_err();
            assert!(matches!(
                error,
                InstallError::ConflictingInstallation { .. }
            ));
        }
        "detection_failure_falls_back" => {
            let request = ExecutionBudgetRequest {
                host_logical_threads: Some(logical(96)),
                detection: CapacityDetection::failed("simulated available_parallelism EPERM"),
                persistent_limit: None,
                legacy_persistent_limit: None,
                parent_limit: None,
                coordination_scope: CoordinationScope::ProcessLocal,
            };
            let installed = install_process_budget(request).unwrap();
            assert_eq!(installed.resolved().effective_logical_threads.get(), 1);
            assert_eq!(installed.resolved().effective_worker_limit.get(), 1);
            assert_eq!(
                installed.resolved().capacity_source,
                CapacityDetectionSource::FallbackOneAfterDetectionFailure
            );
            let diagnostic = installed
                .resolved()
                .capacity_diagnostic
                .as_ref()
                .expect("fallback must retain a structured diagnostic");
            assert_eq!(
                diagnostic.code,
                DetectionDiagnosticCode::AvailableParallelismFailed
            );
            assert!(diagnostic.detail.contains("EPERM"));
        }
        "broker_uses_only_final_limit" => {
            let request = ExecutionBudgetRequest {
                host_logical_threads: Some(logical(96)),
                detection: CapacityDetection::supplied(logical(23)),
                persistent_limit: Some(BudgetCap::persistent(workers(18))),
                legacy_persistent_limit: Some(BudgetCap::legacy(workers(16))),
                parent_limit: Some(BudgetCap::parent(workers(4))),
                coordination_scope: CoordinationScope::ManagedProcessTree,
            };
            let installed = install_process_budget(request).unwrap();
            assert_eq!(installed.resolved().automatic_worker_limit.get(), 21);
            assert_eq!(installed.resolved().effective_worker_limit.get(), 4);
            assert_eq!(installed.broker().snapshot().installed_limit.get(), 4);
            let full = installed
                .broker()
                .try_acquire(CpuPermitRequest::local(workers(4)))
                .unwrap()
                .expect("the final four permits must be available");
            assert_eq!(full.width().get(), 4);
            assert!(
                installed
                    .broker()
                    .try_acquire(CpuPermitRequest::local(workers(1)))
                    .unwrap()
                    .is_none()
            );
        }
        other => panic!("unknown subprocess case {other}"),
    }
}

#[test]
fn permit_broker_immediate_split_transfer_and_drop_return() {
    let broker = CpuPermitBroker::new(workers(4));
    let mut parent = broker
        .try_acquire(CpuPermitRequest::local(workers(4)))
        .unwrap()
        .expect("all permits should be immediately available");
    assert_eq!(broker.snapshot().live_reserved_sum, 4);

    let child = parent.split(workers(2)).unwrap();
    assert_eq!(parent.width().get(), 2);
    assert_eq!(child.width().get(), 2);
    assert_eq!(broker.snapshot().live_reserved_sum, 4);

    let transferred = child.into_transfer().accept();
    assert_eq!(transferred.width().get(), 2);
    drop(transferred);
    assert_eq!(broker.snapshot().live_reserved_sum, 2);
    drop(parent);
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
    assert_eq!(broker.snapshot().available_permits, 4);
}

#[test]
fn permit_broker_blocks_then_wakes_when_a_lease_drops() {
    let broker = CpuPermitBroker::new(workers(1));
    let held = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .unwrap()
        .unwrap();
    let waiter_broker = broker.clone();
    let (tx, rx) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let lease = waiter_broker
            .acquire(CpuPermitRequest::local(workers(1)))
            .unwrap();
        tx.send(lease.width().get()).unwrap();
    });

    wait_for_queue(&broker, 1);
    assert!(rx.try_recv().is_err());
    drop(held);
    assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), 1);
    waiter.join().unwrap();
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
}

#[test]
fn permit_broker_is_fifo_within_priority_and_children_overtake_local_work() {
    let broker = CpuPermitBroker::new(workers(1));
    let held = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .unwrap()
        .unwrap();
    let (order_tx, order_rx) = mpsc::channel();

    let (release_local_one_tx, release_local_one_rx) = mpsc::channel();
    let local_one = spawn_ordered_waiter(
        broker.clone(),
        CpuPriority::Local,
        "local-one",
        order_tx.clone(),
        release_local_one_rx,
    );
    wait_for_queue(&broker, 1);

    let (release_local_two_tx, release_local_two_rx) = mpsc::channel();
    let local_two = spawn_ordered_waiter(
        broker.clone(),
        CpuPriority::Local,
        "local-two",
        order_tx.clone(),
        release_local_two_rx,
    );
    wait_for_queue(&broker, 2);

    let (release_child_tx, release_child_rx) = mpsc::channel();
    let child = spawn_ordered_waiter(
        broker.clone(),
        CpuPriority::Child,
        "child",
        order_tx,
        release_child_rx,
    );
    wait_for_queue(&broker, 3);

    drop(held);
    assert_eq!(
        order_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        "child"
    );
    release_child_tx.send(()).unwrap();
    assert_eq!(
        order_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        "local-one"
    );
    release_local_one_tx.send(()).unwrap();
    assert_eq!(
        order_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        "local-two"
    );
    release_local_two_tx.send(()).unwrap();

    child.join().unwrap();
    local_one.join().unwrap();
    local_two.join().unwrap();
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
}

#[test]
fn permit_broker_cancellation_removes_waiter_without_leaking_permits() {
    let broker = CpuPermitBroker::new(workers(1));
    let held = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .unwrap()
        .unwrap();
    let cancellation = CancellationToken::new();
    let waiter_token = cancellation.clone();
    let waiter_broker = broker.clone();
    let waiter = thread::spawn(move || {
        waiter_broker.acquire_cancellable(CpuPermitRequest::local(workers(1)), &waiter_token)
    });
    wait_for_queue(&broker, 1);
    cancellation.cancel();
    assert!(matches!(
        waiter.join().unwrap(),
        Err(AcquireError::Cancelled)
    ));
    assert_eq!(broker.snapshot().queued_total, 0);
    assert_eq!(broker.snapshot().live_reserved_sum, 1);
    drop(held);
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
}

#[test]
fn permit_broker_rejects_nested_acquisition_inside_a_lease_scope() {
    let broker = CpuPermitBroker::new(workers(2));
    let lease = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .unwrap()
        .unwrap();
    let result = lease.scope(|| broker.try_acquire(CpuPermitRequest::local(workers(1))));
    assert!(matches!(result, Err(AcquireError::NestedAcquisition)));
    assert_eq!(broker.snapshot().live_reserved_sum, 1);
}

#[test]
fn permit_broker_panic_unwind_returns_the_lease() {
    let broker = CpuPermitBroker::new(workers(1));
    let panic_broker = broker.clone();
    let caught = std::panic::catch_unwind(move || {
        let lease = panic_broker
            .try_acquire(CpuPermitRequest::local(workers(1)))
            .unwrap()
            .unwrap();
        lease.scope(|| panic!("intentional lease-scope panic"));
    });
    assert!(caught.is_err());
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[test]
fn permit_broker_rejects_requests_larger_than_the_installed_limit() {
    let broker = CpuPermitBroker::new(workers(2));
    let error = broker
        .try_acquire(CpuPermitRequest::local(workers(3)))
        .unwrap_err();
    assert!(matches!(error, AcquireError::ExceedsInstalledLimit { .. }));
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
}

#[test]
fn permit_broker_live_reserved_sum_never_exceeds_the_installed_limit() {
    const LIMIT: usize = 4;
    const THREADS: usize = 8;
    const ITERATIONS: usize = 64;

    let broker = CpuPermitBroker::new(workers(LIMIT));
    let start = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::with_capacity(THREADS);
    for worker_index in 0..THREADS {
        let broker = broker.clone();
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
            start.wait();
            for iteration in 0..ITERATIONS {
                let width = if (worker_index + iteration) % 3 == 0 {
                    2
                } else {
                    1
                };
                let mut lease = broker
                    .acquire(CpuPermitRequest::local(workers(width)))
                    .unwrap();
                assert_broker_accounting(&broker, LIMIT);
                if width == 2 {
                    let child = lease.split(workers(1)).unwrap();
                    assert_broker_accounting(&broker, LIMIT);
                    thread::yield_now();
                    drop(child);
                    assert_broker_accounting(&broker, LIMIT);
                }
                thread::yield_now();
                drop(lease);
                assert_broker_accounting(&broker, LIMIT);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let final_snapshot = broker.snapshot();
    assert_eq!(final_snapshot.live_reserved_sum, 0);
    assert_eq!(final_snapshot.available_permits, LIMIT);
    assert_eq!(final_snapshot.queued_total, 0);
}

fn spawn_ordered_waiter(
    broker: CpuPermitBroker,
    priority: CpuPriority,
    label: &'static str,
    order: mpsc::Sender<&'static str>,
    release: mpsc::Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let lease = broker
            .acquire(CpuPermitRequest::new(workers(1), priority))
            .unwrap();
        order.send(label).unwrap();
        release.recv().unwrap();
        drop(lease);
    })
}

fn wait_for_queue(broker: &CpuPermitBroker, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while broker.snapshot().queued_total != expected {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} queued requests; snapshot={:?}",
            broker.snapshot()
        );
        thread::yield_now();
    }
}

fn assert_broker_accounting(broker: &CpuPermitBroker, limit: usize) {
    let snapshot = broker.snapshot();
    assert!(snapshot.live_reserved_sum <= limit, "{snapshot:?}");
    assert_eq!(
        snapshot.live_reserved_sum + snapshot.available_permits,
        limit,
        "{snapshot:?}"
    );
}
