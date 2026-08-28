use std::sync::{Arc, Barrier, Condvar, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::Duration;

use neoethos_search::{
    ProcessExecutionKindV1, ProcessExecutionLeaseTransitionErrorV1, ProcessExecutionLeaseV1,
    try_acquire_process_execution_lease_v1,
};

static TEST_SERIAL_V1: Mutex<()> = Mutex::new(());
const TEST_TIMEOUT_V1: Duration = Duration::from_secs(10);

fn serial_test_v1() -> MutexGuard<'static, ()> {
    TEST_SERIAL_V1
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn acquire_v1(kind: ProcessExecutionKindV1) -> ProcessExecutionLeaseV1 {
    try_acquire_process_execution_lease_v1(kind).expect("execution coordinator is idle")
}

trait AmbiguousIfCloneV1<Marker> {
    fn marker() {}
}

impl<T: ?Sized> AmbiguousIfCloneV1<()> for T {}
impl<T: ?Sized + Clone> AmbiguousIfCloneV1<u8> for T {}

#[test]
fn exactly_one_simultaneous_acquisition_wins_and_busy_names_both_kinds() {
    let _serial = serial_test_v1();
    let start = Arc::new(Barrier::new(3));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let (outcome_tx, outcome_rx) = mpsc::channel();
    let mut workers = Vec::new();

    for kind in [
        ProcessExecutionKindV1::Discovery,
        ProcessExecutionKindV1::Migration,
    ] {
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);
        let outcome_tx = outcome_tx.clone();
        workers.push(thread::spawn(move || {
            start.wait();
            match try_acquire_process_execution_lease_v1(kind) {
                Ok(lease) => {
                    outcome_tx
                        .send(Ok((kind, lease.token())))
                        .expect("report acquired lease");
                    let (lock, wake) = &*release;
                    let mut released = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    drop(lease);
                }
                Err(busy) => outcome_tx.send(Err(busy)).expect("report busy outcome"),
            }
        }));
    }
    drop(outcome_tx);
    start.wait();

    let first = outcome_rx
        .recv_timeout(TEST_TIMEOUT_V1)
        .expect("first acquisition outcome");
    let second = outcome_rx
        .recv_timeout(TEST_TIMEOUT_V1)
        .expect("second acquisition outcome");
    let outcomes = [first, second];
    let acquired: Vec<_> = outcomes
        .iter()
        .filter_map(|value| value.as_ref().ok())
        .collect();
    let busy: Vec<_> = outcomes
        .iter()
        .filter_map(|value| value.as_ref().err())
        .collect();

    assert_eq!(acquired.len(), 1, "exactly one contender acquires");
    assert_eq!(busy.len(), 1, "the other contender receives typed Busy");
    assert_ne!(acquired[0].1, 0, "tokens are never the zero sentinel");
    assert_eq!(busy[0].active(), acquired[0].0);
    assert_ne!(busy[0].requested(), busy[0].active());

    *release
        .0
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
    release.1.notify_all();
    for worker in workers {
        worker.join().expect("acquisition contender exits");
    }
    drop(acquire_v1(ProcessExecutionKindV1::NativeResearch));
}

#[test]
fn dropping_the_join_handle_and_cancelling_do_not_release_the_worker_owned_lease() {
    let _serial = serial_test_v1();
    let lease = acquire_v1(ProcessExecutionKindV1::Discovery);
    let token = lease.token();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let (cancel_observed_tx, cancel_observed_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel();
    let (terminal_tx, terminal_rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        cancel_rx.recv().expect("worker observes cancellation");
        cancel_observed_tx
            .send(lease.token())
            .expect("worker reports retained authority");
        finish_rx.recv().expect("worker reaches actual terminal");
        drop(lease);
        terminal_tx
            .send(())
            .expect("worker reports terminal release");
    });
    drop(handle);
    cancel_tx.send(()).expect("request cancellation");
    assert_eq!(
        cancel_observed_rx
            .recv_timeout(TEST_TIMEOUT_V1)
            .expect("cancellation observation"),
        token
    );

    let busy = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)
        .expect_err("detached/cancelled worker still owns the lease");
    assert_eq!(busy.requested(), ProcessExecutionKindV1::Training);
    assert_eq!(busy.active(), ProcessExecutionKindV1::Discovery);

    finish_tx.send(()).expect("allow actual terminal");
    terminal_rx
        .recv_timeout(TEST_TIMEOUT_V1)
        .expect("worker releases only at terminal");
    drop(acquire_v1(ProcessExecutionKindV1::Training));
}

#[test]
fn discovery_to_training_transition_is_gap_free_same_token_and_one_shot() {
    let _serial = serial_test_v1();
    let mut lease = acquire_v1(ProcessExecutionKindV1::Discovery);
    let token = lease.token();
    let race = Arc::new(Barrier::new(2));
    let contender_race = Arc::clone(&race);
    let contender = thread::spawn(move || {
        contender_race.wait();
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration)
    });

    race.wait();
    lease
        .transition_discovery_to_training_v1()
        .expect("authorized one-shot handoff");
    assert_eq!(lease.kind(), ProcessExecutionKindV1::Training);
    assert_eq!(lease.token(), token, "handoff retains the exact token");

    let raced = contender.join().expect("relabel contender exits");
    let busy = raced.expect_err("there is no idle acquisition gap during relabel");
    assert_eq!(busy.requested(), ProcessExecutionKindV1::Migration);
    assert!(matches!(
        busy.active(),
        ProcessExecutionKindV1::Discovery | ProcessExecutionKindV1::Training
    ));

    assert_eq!(
        lease.transition_discovery_to_training_v1(),
        Err(ProcessExecutionLeaseTransitionErrorV1::AlreadyTransitioned)
    );
    let busy = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::NativeResearch)
        .expect_err("Training keeps the same process-wide authority");
    assert_eq!(busy.active(), ProcessExecutionKindV1::Training);
    drop(lease);
    drop(acquire_v1(ProcessExecutionKindV1::Migration));
}

#[test]
fn every_kind_including_migration_uses_the_same_process_wide_mutex() {
    let _serial = serial_test_v1();
    let kinds = [
        ProcessExecutionKindV1::Discovery,
        ProcessExecutionKindV1::Training,
        ProcessExecutionKindV1::NativeResearch,
        ProcessExecutionKindV1::Migration,
    ];

    for active in kinds {
        let lease = acquire_v1(active);
        assert_eq!(lease.kind(), active);
        for requested in kinds {
            let busy = try_acquire_process_execution_lease_v1(requested)
                .expect_err("all execution kinds share one authority");
            assert_eq!(busy.requested(), requested);
            assert_eq!(busy.active(), active);
        }
        drop(lease);
    }

    let _ = <ProcessExecutionLeaseV1 as AmbiguousIfCloneV1<_>>::marker;
}
