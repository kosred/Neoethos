use neoethos_app::app_services::execution_admission::{
    AdmissionError, ExecutionAdmissionCoordinator,
};
use neoethos_app::app_state::AppExecutionState;
use neoethos_core::execution::{BudgetedCpuExecutor, BudgetedCpuExecutorError};
use neoethos_core::execution_budget::{
    AcquireError, CpuPermitBroker, CpuPermitRequest, WorkerLimit,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

fn workers(value: usize) -> WorkerLimit {
    WorkerLimit::new(value).expect("test worker count is positive")
}

#[test]
fn executor_rejects_a_lease_from_a_different_capacity_authority() {
    let authority = CpuPermitBroker::new(workers(1));
    let unrelated = CpuPermitBroker::new(workers(1));
    let executor = BudgetedCpuExecutor::new_for_broker(authority, workers(1));
    let unrelated_lease = unrelated
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .expect("valid request")
        .expect("unrelated capacity is free");
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = Arc::clone(&ran);

    let result = executor.execute(unrelated_lease.into_transfer(), move || {
        observed.store(true, Ordering::SeqCst);
    });

    assert!(matches!(
        result,
        Err(BudgetedCpuExecutorError::MismatchedLeaseAuthority)
    ));
    assert!(!ran.load(Ordering::SeqCst));
    assert_eq!(unrelated.snapshot().available_permits, 1);
}

#[test]
fn budgeted_executor_matches_the_lease_and_idle_cache_is_not_admitted_work() {
    let broker = CpuPermitBroker::new(workers(2));
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), workers(2));
    let lease = broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .expect("valid request")
        .expect("capacity is initially free");
    let observed_broker = broker.clone();

    let pool_width = executor
        .execute(lease.into_transfer(), move || {
            let snapshot = observed_broker.snapshot();
            assert_eq!(snapshot.live_reserved_sum, 2);
            assert_eq!(snapshot.available_permits, 0);
            BudgetedCpuExecutor::current_pool_width()
        })
        .expect("matching pool builds");

    assert_eq!(pool_width, 2);
    assert_eq!(executor.cached_idle_worker_threads(), 2);
    let snapshot = broker.snapshot();
    assert_eq!(snapshot.live_reserved_sum, 0);
    assert_eq!(snapshot.available_permits, 2);
}

#[test]
fn nested_work_uses_a_split_lease_and_fresh_acquisition_is_rejected() {
    let broker = CpuPermitBroker::new(workers(2));
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), workers(2));
    let mut parent = broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .expect("valid request")
        .expect("capacity is initially free");
    let child = parent
        .split(workers(1))
        .expect("one permit remains with the parent");
    let rendezvous = Arc::new(Barrier::new(2));

    let child_executor = executor.clone();
    let child_rendezvous = Arc::clone(&rendezvous);
    let child_thread = std::thread::spawn(move || {
        child_executor
            .execute(child.into_transfer(), move || {
                child_rendezvous.wait();
                assert_eq!(BudgetedCpuExecutor::current_pool_width(), 1);
            })
            .expect("child pool builds")
    });

    let nested_broker = broker.clone();
    executor
        .execute(parent.into_transfer(), move || {
            rendezvous.wait();
            assert_eq!(BudgetedCpuExecutor::current_pool_width(), 1);
            assert!(matches!(
                nested_broker.try_acquire(CpuPermitRequest::local(workers(1))),
                Err(AcquireError::NestedAcquisition)
            ));
        })
        .expect("parent pool builds");
    child_thread.join().expect("child executor does not panic");

    assert_eq!(broker.snapshot().available_permits, 2);
}

#[test]
fn stolen_scoped_work_cannot_reacquire_from_a_second_pool_worker() {
    let broker = CpuPermitBroker::new(workers(2));
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), workers(2));
    let lease = broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .expect("valid request")
        .expect("capacity is initially free");
    let rendezvous = Arc::new(Barrier::new(2));
    let child_rejected = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let child_broker = broker.clone();
    let child_rendezvous = Arc::clone(&rendezvous);
    let observed_child = Arc::clone(&child_rejected);

    executor
        .execute_scoped(lease.into_transfer(), move |scope| {
            scope.spawn(move |_| {
                child_rendezvous.wait();
                observed_child.store(
                    matches!(
                        child_broker.try_acquire(CpuPermitRequest::local(workers(1))),
                        Err(AcquireError::NestedAcquisition)
                    ),
                    Ordering::SeqCst,
                );
            });
            rendezvous.wait();
        })
        .expect("matching pool builds");

    assert!(
        child_rejected.load(Ordering::SeqCst),
        "a stolen child ran on an unmarked worker and escaped nested-acquisition rejection"
    );
    assert_eq!(broker.snapshot().available_permits, 2);
}

#[test]
fn panic_returns_permits_and_concurrent_active_width_never_exceeds_budget() {
    let panic_broker = CpuPermitBroker::new(workers(2));
    let panic_executor = BudgetedCpuExecutor::new_for_broker(panic_broker.clone(), workers(2));
    let lease = panic_broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .expect("valid request")
        .expect("capacity is initially free");

    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let _: Result<(), _> = panic_executor.execute(lease.into_transfer(), || {
            panic!("injected executor panic");
        });
    }));
    assert!(panic_result.is_err());
    assert_eq!(panic_broker.snapshot().available_permits, 2);

    let broker = CpuPermitBroker::new(workers(3));
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), workers(3));
    let wide = broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .expect("valid wide request")
        .expect("wide capacity is free");
    let narrow = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .expect("valid narrow request")
        .expect("remaining capacity is free");
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let rendezvous = Arc::new(Barrier::new(2));

    let wide_executor = executor.clone();
    let wide_active = Arc::clone(&active);
    let wide_peak = Arc::clone(&peak);
    let wide_rendezvous = Arc::clone(&rendezvous);
    let wide_thread = std::thread::spawn(move || {
        wide_executor
            .execute(wide.into_transfer(), move || {
                let now = wide_active.fetch_add(2, Ordering::SeqCst) + 2;
                wide_peak.fetch_max(now, Ordering::SeqCst);
                wide_rendezvous.wait();
                wide_active.fetch_sub(2, Ordering::SeqCst);
            })
            .expect("wide pool builds")
    });

    let narrow_active = Arc::clone(&active);
    let narrow_peak = Arc::clone(&peak);
    executor
        .execute(narrow.into_transfer(), move || {
            let now = narrow_active.fetch_add(1, Ordering::SeqCst) + 1;
            narrow_peak.fetch_max(now, Ordering::SeqCst);
            rendezvous.wait();
            narrow_active.fetch_sub(1, Ordering::SeqCst);
        })
        .expect("narrow pool builds");
    wide_thread.join().expect("wide executor does not panic");

    assert_eq!(peak.load(Ordering::SeqCst), 3);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert_eq!(broker.snapshot().available_permits, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn later_child_overtakes_local_waiter_without_blocking_tokio() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator =
        ExecutionAdmissionCoordinator::start(broker.clone()).expect("coordinator starts");
    let client = coordinator.client();
    let held = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("request is queued")
        .wait()
        .await
        .expect("initial request is admitted");

    let local = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("local request is queued");
    let child = client
        .submit(CpuPermitRequest::child(workers(1)))
        .expect("child request is queued");
    let local_task = tokio::spawn(async move { local.wait().await });

    tokio::time::timeout(Duration::from_millis(250), async {
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    })
    .await
    .expect("the sole Tokio worker stays responsive while admission waits");

    drop(held);
    let child_lease = tokio::time::timeout(Duration::from_secs(2), child.wait())
        .await
        .expect("child admission wakes")
        .expect("child is admitted first");
    assert!(
        !local_task.is_finished(),
        "local work bypassed child priority"
    );

    drop(child_lease);
    let local_lease = tokio::time::timeout(Duration::from_secs(2), local_task)
        .await
        .expect("local admission wakes after child exit")
        .expect("local task joins")
        .expect("local request is admitted");
    drop(local_lease);

    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn fifo_cancellation_and_shutdown_are_leak_free() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator =
        ExecutionAdmissionCoordinator::start(broker.clone()).expect("coordinator starts");
    let client = coordinator.client();
    let held = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("request is queued")
        .wait()
        .await
        .expect("initial request is admitted");

    let cancelled_head = client
        .submit(CpuPermitRequest::child(workers(1)))
        .expect("cancelled head is queued");
    let first = client
        .submit(CpuPermitRequest::child(workers(1)))
        .expect("first live child is queued");
    let second = client
        .submit(CpuPermitRequest::child(workers(1)))
        .expect("second live child is queued");
    drop(cancelled_head);
    let second_task = tokio::spawn(async move { second.wait().await });

    drop(held);
    let first_lease = tokio::time::timeout(Duration::from_secs(2), first.wait())
        .await
        .expect("first live child wakes")
        .expect("first live child is admitted");
    assert!(
        !second_task.is_finished(),
        "FIFO within child priority was bypassed"
    );
    drop(first_lease);
    let second_lease = tokio::time::timeout(Duration::from_secs(2), second_task)
        .await
        .expect("second child wakes")
        .expect("second child task joins")
        .expect("second child is admitted");
    drop(second_lease);

    let local_blocker = client
        .admit(CpuPermitRequest::local(workers(1)))
        .await
        .expect("local blocker is admitted");
    let first_local = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("first local request is queued");
    let second_local = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("second local request is queued");
    let second_local_task = tokio::spawn(async move { second_local.wait().await });
    drop(local_blocker);
    let first_local_lease = tokio::time::timeout(Duration::from_secs(2), first_local.wait())
        .await
        .expect("first local request wakes")
        .expect("first local request is admitted");
    assert!(
        !second_local_task.is_finished(),
        "FIFO within local priority was bypassed"
    );
    drop(first_local_lease);
    let second_local_lease = tokio::time::timeout(Duration::from_secs(2), second_local_task)
        .await
        .expect("second local request wakes")
        .expect("second local task joins")
        .expect("second local request is admitted");
    drop(second_local_lease);

    let held_again = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("request is queued")
        .wait()
        .await
        .expect("capacity returns after FIFO jobs");
    let pending_at_shutdown = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("pending shutdown request is queued");
    coordinator.shutdown().expect("coordinator joins cleanly");
    assert!(matches!(
        pending_at_shutdown.wait().await,
        Err(AdmissionError::CoordinatorStopped)
    ));
    drop(held_again);
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cancellation_after_admission_and_app_state_drop_return_every_permit() {
    let broker = CpuPermitBroker::new(workers(1));
    let app_state =
        AppExecutionState::new(broker.clone(), workers(1)).expect("app execution state starts");
    let client = app_state.admission_client();
    let (admitted_tx, admitted_rx) = tokio::sync::oneshot::channel();

    let task = tokio::spawn(async move {
        let lease = client
            .submit(CpuPermitRequest::local(workers(1)))
            .expect("request is queued")
            .wait()
            .await
            .expect("request is admitted");
        admitted_tx.send(()).expect("test receiver is live");
        let _lease = lease;
        std::future::pending::<()>().await;
    });
    admitted_rx.await.expect("task owns an admitted lease");
    assert_eq!(broker.snapshot().live_reserved_sum, 1);

    task.abort();
    assert!(task.await.expect_err("task is cancelled").is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), async {
        while broker.snapshot().available_permits != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("aborted task returns its lease");

    let stopped_client = app_state.admission_client();
    app_state
        .shutdown()
        .expect("app state joins its coordinator");
    assert!(matches!(
        stopped_client.submit(CpuPermitRequest::local(workers(1))),
        Err(AdmissionError::CoordinatorStopped)
    ));
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn admitted_execution_holds_capacity_through_work_and_wakes_the_next_request() {
    let broker = CpuPermitBroker::new(workers(1));
    let app_state =
        AppExecutionState::new(broker.clone(), workers(1)).expect("app execution state starts");
    let client = app_state.admission_client();
    let admitted = client
        .admit(CpuPermitRequest::local(workers(1)))
        .await
        .expect("first request is admitted");
    let next = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("next request is queued");
    let executor = app_state.executor().clone();
    let observed_broker = broker.clone();

    let pool_width = tokio::task::spawn_blocking(move || {
        admitted.execute(&executor, move || {
            assert_eq!(observed_broker.snapshot().live_reserved_sum, 1);
            BudgetedCpuExecutor::current_pool_width()
        })
    })
    .await
    .expect("blocking task joins")
    .expect("budgeted pool builds");
    assert_eq!(pool_width, 1);

    let next_lease = tokio::time::timeout(Duration::from_secs(2), next.wait())
        .await
        .expect("lease-return wake reaches the coordinator")
        .expect("next request is admitted");
    drop(next_lease);
    app_state.shutdown().expect("app state shuts down");
    assert_eq!(broker.snapshot().available_permits, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn invalid_head_is_rejected_without_stalling_a_valid_request() {
    let broker = CpuPermitBroker::new(workers(1));
    let coordinator =
        ExecutionAdmissionCoordinator::start(broker.clone()).expect("coordinator starts");
    let client = coordinator.client();
    let invalid = client
        .submit(CpuPermitRequest::child(workers(2)))
        .expect("invalid width is reported asynchronously");
    let valid = client
        .submit(CpuPermitRequest::local(workers(1)))
        .expect("valid request is queued");

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
        .expect("valid request is not stalled")
        .expect("valid request is admitted");
    drop(valid_lease);
    coordinator.shutdown().expect("coordinator joins cleanly");
    assert_eq!(broker.snapshot().available_permits, 1);
}
