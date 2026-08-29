use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use neoethos_execution_budget::{
    AcquireError, AuxiliarySlotLimit, AuxiliarySlotRequest, CancellationToken,
    CompositeAdmissionAuthority, CompositeAdmissionRequest, CpuPermitBroker, CpuPermitRequest,
    CpuPriority, WorkerLimit,
};

fn workers(value: usize) -> WorkerLimit {
    WorkerLimit::new(value).expect("test worker limit must be positive")
}

fn slots(value: usize) -> AuxiliarySlotLimit {
    AuxiliarySlotLimit::new(value).expect("test auxiliary-slot limit must be positive")
}

fn with_slot(width: usize, priority: CpuPriority) -> CompositeAdmissionRequest {
    CompositeAdmissionRequest::new(
        CpuPermitRequest::new(workers(width), priority),
        AuxiliarySlotRequest::One,
    )
}

fn cpu_only(width: usize, priority: CpuPriority) -> CompositeAdmissionRequest {
    CompositeAdmissionRequest::new(
        CpuPermitRequest::new(workers(width), priority),
        AuxiliarySlotRequest::None,
    )
}

#[test]
fn immediate_grant_is_atomic_optional_and_drop_returns_both_resources() {
    let broker = CpuPermitBroker::new(workers(2));
    let authority = CompositeAdmissionAuthority::new(broker.clone(), slots(1));

    let grant = authority
        .try_acquire(with_slot(1, CpuPriority::Local))
        .unwrap()
        .expect("CPU and the auxiliary slot are initially available");
    assert_eq!(grant.cpu_lease().width().get(), 1);
    assert_eq!(grant.auxiliary_slot().unwrap().index(), 0);

    let held = authority.snapshot();
    assert_eq!(held.cpu.available_permits, 1);
    assert_eq!(held.cpu.live_reserved_sum, 1);
    assert_eq!(held.available_auxiliary_slots, 0);
    assert_eq!(held.live_auxiliary_slots, 1);

    assert!(
        authority
            .try_acquire(with_slot(1, CpuPriority::Local))
            .unwrap()
            .is_none()
    );
    let still_atomic = authority.snapshot();
    assert_eq!(still_atomic.cpu.available_permits, 1);
    assert_eq!(still_atomic.cpu.live_reserved_sum, 1);

    drop(grant);
    let returned = authority.snapshot();
    assert_eq!(returned.cpu.available_permits, 2);
    assert_eq!(returned.cpu.live_reserved_sum, 0);
    assert_eq!(returned.available_auxiliary_slots, 1);
    assert_eq!(returned.live_auxiliary_slots, 0);

    let no_slot = authority
        .try_acquire(cpu_only(2, CpuPriority::Local))
        .unwrap()
        .expect("an optional-slot request must not consume an auxiliary slot");
    assert!(no_slot.auxiliary_slot().is_none());
    assert_eq!(authority.snapshot().available_auxiliary_slots, 1);
    drop(no_slot);
}

#[test]
fn oversized_cpu_width_is_rejected_before_either_resource_changes() {
    let broker = CpuPermitBroker::new(workers(2));
    let authority = CompositeAdmissionAuthority::new(broker, slots(1));

    let error = authority
        .try_acquire(with_slot(3, CpuPriority::Local))
        .unwrap_err();
    assert!(matches!(
        error,
        AcquireError::ExceedsInstalledLimit {
            requested: 3,
            installed: 2
        }
    ));
    let snapshot = authority.snapshot();
    assert_eq!(snapshot.cpu.available_permits, 2);
    assert_eq!(snapshot.available_auxiliary_slots, 1);
    assert_eq!(snapshot.cpu.queued_total, 0);
    assert!(AuxiliarySlotLimit::new(0).is_err());
}

#[test]
fn opposite_order_saturation_never_partially_reserves_a_resource() {
    // CPU exhausted first, auxiliary slot still free.
    let broker = CpuPermitBroker::new(workers(2));
    let authority = CompositeAdmissionAuthority::new(broker.clone(), slots(1));
    let held_cpu = broker
        .try_acquire(CpuPermitRequest::local(workers(2)))
        .unwrap()
        .unwrap();
    let (granted_tx, granted_rx) = mpsc::channel();
    let cpu_first_authority = authority.clone();
    let cpu_first = thread::spawn(move || {
        let grant = cpu_first_authority
            .acquire(with_slot(1, CpuPriority::Local))
            .unwrap();
        granted_tx.send(grant).unwrap();
    });
    wait_for_queue(&authority, 1);
    let cpu_first_waiting = authority.snapshot();
    assert_eq!(cpu_first_waiting.cpu.available_permits, 0);
    assert_eq!(cpu_first_waiting.available_auxiliary_slots, 1);
    assert!(granted_rx.try_recv().is_err());
    drop(held_cpu);
    let granted = granted_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    drop(granted);
    cpu_first.join().unwrap();

    // Auxiliary slot exhausted first, all CPU permits free.
    let seed = authority
        .try_acquire(with_slot(1, CpuPriority::Local))
        .unwrap()
        .unwrap();
    let (seed_cpu, held_slot) = seed.into_parts();
    drop(seed_cpu);
    let held_slot = held_slot.expect("seed requested one auxiliary slot");
    assert_eq!(authority.snapshot().cpu.available_permits, 2);
    assert_eq!(authority.snapshot().available_auxiliary_slots, 0);

    let (slot_granted_tx, slot_granted_rx) = mpsc::channel();
    let slot_first_authority = authority.clone();
    let slot_first = thread::spawn(move || {
        let grant = slot_first_authority
            .acquire(with_slot(2, CpuPriority::Local))
            .unwrap();
        slot_granted_tx.send(grant).unwrap();
    });
    wait_for_queue(&authority, 1);
    let slot_first_waiting = authority.snapshot();
    assert_eq!(slot_first_waiting.cpu.available_permits, 2);
    assert_eq!(slot_first_waiting.cpu.live_reserved_sum, 0);
    assert_eq!(slot_first_waiting.available_auxiliary_slots, 0);
    assert!(slot_granted_rx.try_recv().is_err());

    drop(held_slot);
    let granted = slot_granted_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert_eq!(granted.cpu_lease().width().get(), 2);
    drop(granted);
    slot_first.join().unwrap();
    assert_fully_returned(&authority, 2, 1);
}

#[test]
fn cancellation_while_auxiliary_is_exhausted_holds_no_cpu_and_leaks_nothing() {
    let broker = CpuPermitBroker::new(workers(2));
    let authority = CompositeAdmissionAuthority::new(broker, slots(1));
    let seed = authority
        .try_acquire(with_slot(1, CpuPriority::Local))
        .unwrap()
        .unwrap();
    let (seed_cpu, held_slot) = seed.into_parts();
    drop(seed_cpu);
    let held_slot = held_slot.unwrap();

    let cancellation = CancellationToken::new();
    let waiter_token = cancellation.clone();
    let waiter_authority = authority.clone();
    let waiter = thread::spawn(move || {
        waiter_authority.acquire_cancellable(with_slot(2, CpuPriority::Local), &waiter_token)
    });
    wait_for_queue(&authority, 1);
    let waiting = authority.snapshot();
    assert_eq!(waiting.cpu.available_permits, 2);
    assert_eq!(waiting.cpu.live_reserved_sum, 0);
    assert_eq!(waiting.available_auxiliary_slots, 0);

    cancellation.cancel();
    assert!(matches!(
        waiter.join().unwrap(),
        Err(AcquireError::Cancelled)
    ));
    let cancelled = authority.snapshot();
    assert_eq!(cancelled.cpu.queued_total, 0);
    assert_eq!(cancelled.cpu.available_permits, 2);
    assert_eq!(cancelled.available_auxiliary_slots, 0);

    drop(held_slot);
    assert_fully_returned(&authority, 2, 1);
}

#[test]
fn children_overtake_local_requests_and_fifo_is_preserved_within_priority() {
    let broker = CpuPermitBroker::new(workers(1));
    let authority = CompositeAdmissionAuthority::new(broker.clone(), slots(1));
    let held = broker
        .try_acquire(CpuPermitRequest::local(workers(1)))
        .unwrap()
        .unwrap();
    let (order_tx, order_rx) = mpsc::channel();

    let (release_local_one_tx, release_local_one_rx) = mpsc::channel();
    let local_one = spawn_ordered_waiter(
        authority.clone(),
        CpuPriority::Local,
        "local-one",
        order_tx.clone(),
        release_local_one_rx,
    );
    wait_for_queue(&authority, 1);

    let (release_local_two_tx, release_local_two_rx) = mpsc::channel();
    let local_two = spawn_ordered_waiter(
        authority.clone(),
        CpuPriority::Local,
        "local-two",
        order_tx.clone(),
        release_local_two_rx,
    );
    wait_for_queue(&authority, 2);

    let (release_child_tx, release_child_rx) = mpsc::channel();
    let child = spawn_ordered_waiter(
        authority.clone(),
        CpuPriority::Child,
        "child",
        order_tx,
        release_child_rx,
    );
    wait_for_queue(&authority, 3);
    let queued = authority.snapshot();
    assert_eq!(queued.cpu.queued_children, 1);
    assert_eq!(queued.cpu.queued_local, 2);
    assert_eq!(queued.queued_requiring_auxiliary, 3);

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
    assert_fully_returned(&authority, 1, 1);
}

#[test]
fn cpu_only_request_can_run_while_the_auxiliary_pool_is_exhausted() {
    let broker = CpuPermitBroker::new(workers(2));
    let authority = CompositeAdmissionAuthority::new(broker, slots(1));
    let seed = authority
        .try_acquire(with_slot(1, CpuPriority::Local))
        .unwrap()
        .unwrap();
    let (seed_cpu, held_slot) = seed.into_parts();
    drop(seed_cpu);
    let held_slot = held_slot.unwrap();

    let cpu_only_grant = authority
        .try_acquire(cpu_only(2, CpuPriority::Local))
        .unwrap()
        .expect("a request that needs no auxiliary slot may use the free CPU permits");
    assert!(cpu_only_grant.auxiliary_slot().is_none());
    assert_eq!(authority.snapshot().available_auxiliary_slots, 0);
    drop(cpu_only_grant);
    drop(held_slot);
    assert_fully_returned(&authority, 2, 1);
}

fn spawn_ordered_waiter(
    authority: CompositeAdmissionAuthority,
    priority: CpuPriority,
    label: &'static str,
    order_tx: mpsc::Sender<&'static str>,
    release_rx: mpsc::Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let grant = authority.acquire(with_slot(1, priority)).unwrap();
        order_tx.send(label).unwrap();
        release_rx.recv().unwrap();
        drop(grant);
    })
}

fn wait_for_queue(authority: &CompositeAdmissionAuthority, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while authority.snapshot().cpu.queued_total != expected {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} queued requests; snapshot={:?}",
            authority.snapshot()
        );
        thread::yield_now();
    }
}

fn assert_fully_returned(
    authority: &CompositeAdmissionAuthority,
    cpu_limit: usize,
    auxiliary_limit: usize,
) {
    let snapshot = authority.snapshot();
    assert_eq!(snapshot.cpu.available_permits, cpu_limit);
    assert_eq!(snapshot.cpu.live_reserved_sum, 0);
    assert_eq!(snapshot.cpu.queued_total, 0);
    assert_eq!(snapshot.available_auxiliary_slots, auxiliary_limit);
    assert_eq!(snapshot.live_auxiliary_slots, 0);
}
