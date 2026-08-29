//! Async-safe admission for lease-bound CPU and composite import work.
//!
//! Tokio tasks enqueue requests and await a oneshot. One dedicated OS thread
//! owns both priority/FIFO queues and only calls the permit broker's
//! non-blocking `try_acquire`; it never waits on a request-specific condition
//! variable. Lease return sends an explicit wake command so the queue can be
//! reconsidered without consuming an async runtime worker. While a queue is
//! active, a bounded coordinator-thread recheck also observes raw leases that
//! share the broker but cannot emit an app command when they drop.

use neoethos_core::execution::{BudgetedCpuExecutor, BudgetedCpuExecutorError};
use neoethos_core::execution_budget::{
    AcquireError, AuxiliarySlotLease, AuxiliarySlotLimit, AuxiliarySlotRequest, BrokerSnapshot,
    CompositeAdmissionAuthority, CompositeAdmissionRequest, CompositeAdmissionSnapshot, CpuLease,
    CpuPermitBroker, CpuPermitRequest, CpuPriority, LeaseError, WorkerLimit,
    enter_lease_bound_worker_scope,
};
use neoethos_data::source_seal_slot_limit;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::oneshot;

const ACTIVE_QUEUE_RECHECK_INTERVAL: Duration = Duration::from_millis(25);

pub fn platform_import_auxiliary_slot_limit() -> AuxiliarySlotLimit {
    AuxiliarySlotLimit::new(source_seal_slot_limit())
        .expect("the authoritative source-seal slot count is positive")
}

/// A consistent resource snapshot. Queue counts are intentionally absent:
/// pending app requests live in the coordinator's priority queues rather than
/// the leaf broker queue, so exposing the leaf queue counters here would
/// truthfully describe neither the app queue nor total admission pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionAdmissionSnapshot {
    pub cpu: ExecutionCpuCapacitySnapshot,
    pub auxiliary_limit: AuxiliarySlotLimit,
    pub available_auxiliary_slots: usize,
    pub live_auxiliary_slots: usize,
    /// Monotonic diagnostic epoch incremented after each complete coordinator
    /// grant pass. It lets probes distinguish an unprocessed command from a
    /// request that was examined but could not yet receive all resources.
    pub grant_cycles_completed: u64,
}

/// Capacity-only view of the shared CPU broker. Leaf-broker waiter counters
/// are deliberately omitted because they exclude requests waiting in the app
/// coordinator and would therefore be misleading as aggregate queue metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionCpuCapacitySnapshot {
    pub installed_limit: WorkerLimit,
    pub available_permits: usize,
    pub live_reserved_sum: usize,
}

impl From<BrokerSnapshot> for ExecutionCpuCapacitySnapshot {
    fn from(snapshot: BrokerSnapshot) -> Self {
        Self {
            installed_limit: snapshot.installed_limit,
            available_permits: snapshot.available_permits,
            live_reserved_sum: snapshot.live_reserved_sum,
        }
    }
}

impl ExecutionAdmissionSnapshot {
    fn from_resources(snapshot: CompositeAdmissionSnapshot, grant_cycles_completed: u64) -> Self {
        Self {
            cpu: snapshot.cpu.into(),
            auxiliary_limit: snapshot.auxiliary_limit,
            available_auxiliary_slots: snapshot.available_auxiliary_slots,
            live_auxiliary_slots: snapshot.live_auxiliary_slots,
            grant_cycles_completed,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    Acquire(AcquireError),
    Cancelled,
    CoordinatorStopped,
    RequestSequenceExhausted,
    CoordinatorPanicked,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acquire(error) => write!(formatter, "execution admission failed: {error}"),
            Self::Cancelled => write!(formatter, "execution admission request was cancelled"),
            Self::CoordinatorStopped => {
                write!(formatter, "execution admission coordinator stopped")
            }
            Self::RequestSequenceExhausted => {
                write!(formatter, "execution admission request sequence exhausted")
            }
            Self::CoordinatorPanicked => {
                write!(formatter, "execution admission coordinator panicked")
            }
        }
    }
}

impl Error for AdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Acquire(error) => Some(error),
            Self::Cancelled
            | Self::CoordinatorStopped
            | Self::RequestSequenceExhausted
            | Self::CoordinatorPanicked => None,
        }
    }
}

impl From<AcquireError> for AdmissionError {
    fn from(value: AcquireError) -> Self {
        Self::Acquire(value)
    }
}

#[derive(Clone)]
pub struct ExecutionAdmissionClient {
    command_tx: mpsc::Sender<CoordinatorCommand>,
    next_request_id: Arc<AtomicU64>,
}

impl ExecutionAdmissionClient {
    /// Enqueue without blocking the caller. Dropping the returned pending
    /// handle cancels the queued request, including while its future is being
    /// aborted by Tokio.
    pub fn submit(&self, request: CpuPermitRequest) -> Result<PendingAdmission, AdmissionError> {
        let id = self.allocate_request_id()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(CoordinatorCommand::Enqueue {
                id,
                request: CompositeAdmissionRequest::new(request, AuxiliarySlotRequest::None),
                reply: AdmissionReply::Cpu(reply_tx),
            })
            .map_err(|_| AdmissionError::CoordinatorStopped)?;
        Ok(PendingAdmission {
            id,
            command_tx: self.command_tx.clone(),
            reply: Some(reply_rx),
            settled: false,
        })
    }

    pub async fn admit(
        &self,
        request: CpuPermitRequest,
    ) -> Result<AdmittedCpuLease, AdmissionError> {
        self.submit(request)?.wait().await
    }

    /// Atomically queue one CPU reservation and one import auxiliary slot.
    /// The returned future never waits on a blocking primitive on its Tokio
    /// worker, and cancellation while queued retains neither resource.
    pub fn submit_import(
        &self,
        request: CpuPermitRequest,
    ) -> Result<PendingImportAdmission, AdmissionError> {
        let id = self.allocate_request_id()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(CoordinatorCommand::Enqueue {
                id,
                request: CompositeAdmissionRequest::new(request, AuxiliarySlotRequest::One),
                reply: AdmissionReply::Import(reply_tx),
            })
            .map_err(|_| AdmissionError::CoordinatorStopped)?;
        Ok(PendingImportAdmission {
            id,
            command_tx: self.command_tx.clone(),
            reply: Some(reply_rx),
            settled: false,
        })
    }

    pub async fn admit_import(
        &self,
        request: CpuPermitRequest,
    ) -> Result<AdmittedImportLease, AdmissionError> {
        self.submit_import(request)?.wait().await
    }

    fn allocate_request_id(&self) -> Result<u64, AdmissionError> {
        self.next_request_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| AdmissionError::RequestSequenceExhausted)
    }
}

impl fmt::Debug for ExecutionAdmissionClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionAdmissionClient")
            .finish_non_exhaustive()
    }
}

pub struct PendingAdmission {
    id: u64,
    command_tx: mpsc::Sender<CoordinatorCommand>,
    reply: Option<oneshot::Receiver<Result<AdmittedCpuLease, AdmissionError>>>,
    settled: bool,
}

impl PendingAdmission {
    pub async fn wait(mut self) -> Result<AdmittedCpuLease, AdmissionError> {
        let reply = self
            .reply
            .take()
            .expect("a pending admission reply is awaited only once");
        let result = reply
            .await
            .unwrap_or(Err(AdmissionError::CoordinatorStopped));
        self.settled = true;
        result
    }
}

impl fmt::Debug for PendingAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingAdmission")
            .field("id", &self.id)
            .field("settled", &self.settled)
            .finish_non_exhaustive()
    }
}

impl Drop for PendingAdmission {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self
                .command_tx
                .send(CoordinatorCommand::Cancel { id: self.id });
        }
    }
}

/// A cancellable queued request for one indivisible import grant.
pub struct PendingImportAdmission {
    id: u64,
    command_tx: mpsc::Sender<CoordinatorCommand>,
    reply: Option<oneshot::Receiver<Result<AdmittedImportLease, AdmissionError>>>,
    settled: bool,
}

impl PendingImportAdmission {
    pub async fn wait(mut self) -> Result<AdmittedImportLease, AdmissionError> {
        let reply = self
            .reply
            .take()
            .expect("a pending import admission reply is awaited only once");
        let result = reply
            .await
            .unwrap_or(Err(AdmissionError::CoordinatorStopped));
        self.settled = true;
        result
    }
}

impl fmt::Debug for PendingImportAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingImportAdmission")
            .field("id", &self.id)
            .field("settled", &self.settled)
            .finish_non_exhaustive()
    }
}

impl Drop for PendingImportAdmission {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self
                .command_tx
                .send(CoordinatorCommand::Cancel { id: self.id });
        }
    }
}

/// A coordinator-granted lease. Its destructor returns capacity first and
/// then explicitly wakes the admission thread. This wrapper is intentionally
/// non-cloneable.
pub struct AdmittedCpuLease {
    lease: Option<CpuLease>,
    command_tx: mpsc::Sender<CoordinatorCommand>,
}

impl AdmittedCpuLease {
    pub fn width(&self) -> WorkerLimit {
        self.lease
            .as_ref()
            .expect("an admitted lease owns capacity until drop")
            .width()
    }

    pub fn split(&mut self, width: WorkerLimit) -> Result<Self, LeaseError> {
        let child = self
            .lease
            .as_mut()
            .expect("an admitted lease owns capacity until drop")
            .split(width)?;
        Ok(Self {
            lease: Some(child),
            command_tx: self.command_tx.clone(),
        })
    }

    /// Move the reservation into blocking/scoped CPU execution. On normal
    /// return or panic unwinding, the executor first drops the raw lease and
    /// this wrapper then wakes the coordinator.
    pub fn execute<R, Work>(
        mut self,
        executor: &BudgetedCpuExecutor,
        work: Work,
    ) -> Result<R, BudgetedCpuExecutorError>
    where
        R: Send,
        Work: FnOnce() -> R + Send,
    {
        let transfer = self
            .lease
            .take()
            .expect("an admitted lease can be executed only once")
            .into_transfer();
        executor.execute(transfer, work)
    }
}

impl fmt::Debug for AdmittedCpuLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedCpuLease")
            .field("width", &self.lease.as_ref().map(CpuLease::width))
            .finish_non_exhaustive()
    }
}

impl Drop for AdmittedCpuLease {
    fn drop(&mut self) {
        drop(self.lease.take());
        let _ = self.command_tx.send(CoordinatorCommand::LeaseReturned);
    }
}

/// One coordinator-granted import reservation. It owns both resources until
/// they are moved together into synchronous top-level or `spawn_blocking`
/// work. It is deliberately non-cloneable.
#[must_use = "dropping an admitted import lease immediately returns both resources"]
pub struct AdmittedImportLease {
    cpu_lease: Option<CpuLease>,
    auxiliary_slot: Option<AuxiliarySlotLease>,
    command_tx: Option<mpsc::Sender<CoordinatorCommand>>,
}

impl AdmittedImportLease {
    pub fn width(&self) -> WorkerLimit {
        self.cpu_lease
            .as_ref()
            .expect("an admitted import owns CPU capacity until transfer")
            .width()
    }

    pub fn auxiliary_slot_index(&self) -> usize {
        self.auxiliary_slot
            .as_ref()
            .expect("an admitted import owns an auxiliary slot until transfer")
            .index()
    }

    /// Transfer access to both raw leases into work that the caller has already
    /// moved onto a blocking thread. The borrowed arguments cannot escape the
    /// closure. A lease-bound worker scope rejects fresh nested acquisition;
    /// nested stages must split the supplied CPU lease instead.
    ///
    /// On normal return or panic unwinding, both leases drop before the wake
    /// guard asks the coordinator to reconsider its queue.
    pub fn execute<R, Work>(mut self, work: Work) -> R
    where
        R: Send,
        Work: FnOnce(&mut CpuLease, &AuxiliarySlotLease) -> R + Send,
    {
        let wake = CoordinatorWake {
            command_tx: self.command_tx.take(),
        };
        let mut cpu_lease = self
            .cpu_lease
            .take()
            .expect("an admitted import CPU lease can be transferred only once");
        let auxiliary_slot = self
            .auxiliary_slot
            .take()
            .expect("an admitted import auxiliary slot can be transferred only once");
        let worker_scope = enter_lease_bound_worker_scope();
        let result = work(&mut cpu_lease, &auxiliary_slot);
        drop(worker_scope);
        drop(auxiliary_slot);
        drop(cpu_lease);
        drop(wake);
        result
    }
}

impl fmt::Debug for AdmittedImportLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedImportLease")
            .field("cpu_width", &self.cpu_lease.as_ref().map(CpuLease::width))
            .field(
                "auxiliary_slot",
                &self.auxiliary_slot.as_ref().map(AuxiliarySlotLease::index),
            )
            .finish_non_exhaustive()
    }
}

impl Drop for AdmittedImportLease {
    fn drop(&mut self) {
        drop(self.auxiliary_slot.take());
        drop(self.cpu_lease.take());
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(CoordinatorCommand::LeaseReturned);
        }
    }
}

struct CoordinatorWake {
    command_tx: Option<mpsc::Sender<CoordinatorCommand>>,
}

impl Drop for CoordinatorWake {
    fn drop(&mut self) {
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(CoordinatorCommand::LeaseReturned);
        }
    }
}

pub struct ExecutionAdmissionCoordinator {
    client: ExecutionAdmissionClient,
    authority: CompositeAdmissionAuthority,
    grant_cycles_completed: Arc<AtomicU64>,
    join: Option<JoinHandle<()>>,
}

impl ExecutionAdmissionCoordinator {
    pub fn start(broker: CpuPermitBroker) -> io::Result<Self> {
        Self::start_with_auxiliary_slots(broker, platform_import_auxiliary_slot_limit())
    }

    pub fn start_with_auxiliary_slots(
        broker: CpuPermitBroker,
        auxiliary_slots: AuxiliarySlotLimit,
    ) -> io::Result<Self> {
        let authority = CompositeAdmissionAuthority::new(broker, auxiliary_slots);
        let (command_tx, command_rx) = mpsc::channel();
        let client = ExecutionAdmissionClient {
            command_tx: command_tx.clone(),
            next_request_id: Arc::new(AtomicU64::new(0)),
        };
        let coordinator_tx = command_tx;
        let coordinator_authority = authority.clone();
        let grant_cycles_completed = Arc::new(AtomicU64::new(0));
        let coordinator_grant_cycles = Arc::clone(&grant_cycles_completed);
        let join = std::thread::Builder::new()
            .name("neoethos-cpu-admission".to_owned())
            .spawn(move || {
                run_coordinator(
                    coordinator_authority,
                    command_rx,
                    coordinator_tx,
                    coordinator_grant_cycles,
                );
            })?;
        Ok(Self {
            client,
            authority,
            grant_cycles_completed,
            join: Some(join),
        })
    }

    pub fn client(&self) -> ExecutionAdmissionClient {
        self.client.clone()
    }

    pub fn admission_snapshot(&self) -> ExecutionAdmissionSnapshot {
        ExecutionAdmissionSnapshot::from_resources(
            self.authority.snapshot(),
            self.grant_cycles_completed.load(Ordering::Acquire),
        )
    }

    pub fn shutdown(mut self) -> Result<(), AdmissionError> {
        self.stop_and_join()
    }

    fn stop_and_join(&mut self) -> Result<(), AdmissionError> {
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        let (ack_tx, ack_rx) = mpsc::sync_channel(0);
        if self
            .client
            .command_tx
            .send(CoordinatorCommand::Shutdown { ack: ack_tx })
            .is_ok()
        {
            let _ = ack_rx.recv();
        }
        join.join().map_err(|_| AdmissionError::CoordinatorPanicked)
    }
}

impl fmt::Debug for ExecutionAdmissionCoordinator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionAdmissionCoordinator")
            .field("running", &self.join.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for ExecutionAdmissionCoordinator {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

enum CoordinatorCommand {
    Enqueue {
        id: u64,
        request: CompositeAdmissionRequest,
        reply: AdmissionReply,
    },
    Cancel {
        id: u64,
    },
    LeaseReturned,
    Shutdown {
        ack: mpsc::SyncSender<()>,
    },
}

enum AdmissionReply {
    Cpu(oneshot::Sender<Result<AdmittedCpuLease, AdmissionError>>),
    Import(oneshot::Sender<Result<AdmittedImportLease, AdmissionError>>),
}

impl AdmissionReply {
    fn is_closed(&self) -> bool {
        match self {
            Self::Cpu(reply) => reply.is_closed(),
            Self::Import(reply) => reply.is_closed(),
        }
    }

    fn send_error(self, error: AdmissionError) {
        match self {
            Self::Cpu(reply) => {
                let _ = reply.send(Err(error));
            }
            Self::Import(reply) => {
                let _ = reply.send(Err(error));
            }
        }
    }
}

struct QueuedAdmission {
    id: u64,
    request: CompositeAdmissionRequest,
    reply: AdmissionReply,
}

#[derive(Default)]
struct AdmissionQueues {
    children: VecDeque<QueuedAdmission>,
    local: VecDeque<QueuedAdmission>,
}

impl AdmissionQueues {
    fn is_empty(&self) -> bool {
        self.children.is_empty() && self.local.is_empty()
    }

    fn enqueue(&mut self, queued: QueuedAdmission) {
        match queued.request.cpu.priority {
            CpuPriority::Child => self.children.push_back(queued),
            CpuPriority::Local => self.local.push_back(queued),
        }
    }

    fn cancel(&mut self, id: u64) {
        for queue in [&mut self.children, &mut self.local] {
            if let Some(position) = queue.iter().position(|queued| queued.id == id) {
                if let Some(queued) = queue.remove(position) {
                    queued.reply.send_error(AdmissionError::Cancelled);
                }
                return;
            }
        }
    }

    fn discard_closed(&mut self) {
        self.children.retain(|queued| !queued.reply.is_closed());
        self.local.retain(|queued| !queued.reply.is_closed());
    }

    fn next_request(&self) -> Option<CompositeAdmissionRequest> {
        self.children
            .front()
            .or_else(|| self.local.front())
            .map(|queued| queued.request)
    }

    fn pop_next(&mut self) -> Option<QueuedAdmission> {
        self.children.pop_front().or_else(|| self.local.pop_front())
    }

    fn stop_all(&mut self) {
        for queued in self.children.drain(..).chain(self.local.drain(..)) {
            queued.reply.send_error(AdmissionError::CoordinatorStopped);
        }
    }
}

fn run_coordinator(
    authority: CompositeAdmissionAuthority,
    command_rx: mpsc::Receiver<CoordinatorCommand>,
    command_tx: mpsc::Sender<CoordinatorCommand>,
    grant_cycles_completed: Arc<AtomicU64>,
) {
    let mut queues = AdmissionQueues::default();
    loop {
        let command = if queues.is_empty() {
            match command_rx.recv() {
                Ok(command) => Some(command),
                Err(_) => {
                    queues.stop_all();
                    return;
                }
            }
        } else {
            match command_rx.recv_timeout(ACTIVE_QUEUE_RECHECK_INTERVAL) {
                Ok(command) => Some(command),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => {
                    queues.stop_all();
                    return;
                }
            }
        };
        if let Some(command) = command {
            if handle_command(command, &mut queues) {
                return;
            }

            // Make every already-enqueued request/cancellation visible before
            // a grant decision. This is what lets a later child overtake a
            // local request that was waiting for capacity.
            while let Ok(command) = command_rx.try_recv() {
                if handle_command(command, &mut queues) {
                    return;
                }
            }
        }

        grant_ready_requests(&authority, &command_tx, &mut queues);
        let _ = grant_cycles_completed.fetch_add(1, Ordering::Release);
    }
}

/// Returns true when the coordinator must stop.
fn handle_command(command: CoordinatorCommand, queues: &mut AdmissionQueues) -> bool {
    match command {
        CoordinatorCommand::Enqueue { id, request, reply } => {
            queues.enqueue(QueuedAdmission { id, request, reply })
        }
        CoordinatorCommand::Cancel { id } => queues.cancel(id),
        CoordinatorCommand::LeaseReturned => {}
        CoordinatorCommand::Shutdown { ack } => {
            queues.stop_all();
            let _ = ack.send(());
            return true;
        }
    }
    false
}

fn grant_ready_requests(
    authority: &CompositeAdmissionAuthority,
    command_tx: &mpsc::Sender<CoordinatorCommand>,
    queues: &mut AdmissionQueues,
) {
    loop {
        queues.discard_closed();
        let Some(request) = queues.next_request() else {
            return;
        };
        match authority.try_acquire(request) {
            Ok(Some(grant)) => {
                let queued = queues
                    .pop_next()
                    .expect("a request exists because it was just inspected");
                let (cpu_lease, auxiliary_slot) = grant.into_parts();
                match queued.reply {
                    AdmissionReply::Cpu(reply) => {
                        assert!(
                            auxiliary_slot.is_none(),
                            "an ordinary CPU request cannot receive an auxiliary slot"
                        );
                        let admitted = AdmittedCpuLease {
                            lease: Some(cpu_lease),
                            command_tx: command_tx.clone(),
                        };
                        if let Err(returned) = reply.send(Ok(admitted)) {
                            drop(returned);
                        }
                    }
                    AdmissionReply::Import(reply) => {
                        let admitted = AdmittedImportLease {
                            cpu_lease: Some(cpu_lease),
                            auxiliary_slot: Some(
                                auxiliary_slot
                                    .expect("an import request receives one auxiliary slot"),
                            ),
                            command_tx: Some(command_tx.clone()),
                        };
                        if let Err(returned) = reply.send(Ok(admitted)) {
                            drop(returned);
                        }
                    }
                }
            }
            Ok(None) => return,
            Err(error) => {
                let queued = queues
                    .pop_next()
                    .expect("a request exists because it was just inspected");
                queued.reply.send_error(error.into());
            }
        }
    }
}
