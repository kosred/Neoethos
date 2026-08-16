//! Async-safe admission for lease-bound CPU work.
//!
//! Tokio tasks enqueue requests and await a oneshot. One dedicated OS thread
//! owns both priority/FIFO queues and only calls the permit broker's
//! non-blocking `try_acquire`; it never waits on a request-specific condition
//! variable. Lease return sends an explicit wake command so the queue can be
//! reconsidered without consuming an async runtime worker.

use neoethos_core::execution::{BudgetedCpuExecutor, BudgetedCpuExecutorError};
use neoethos_core::execution_budget::{
    AcquireError, CpuLease, CpuPermitBroker, CpuPermitRequest, CpuPriority, LeaseError, WorkerLimit,
};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use tokio::sync::oneshot;

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
            Self::Acquire(error) => write!(formatter, "CPU admission failed: {error}"),
            Self::Cancelled => write!(formatter, "CPU admission request was cancelled"),
            Self::CoordinatorStopped => write!(formatter, "CPU admission coordinator stopped"),
            Self::RequestSequenceExhausted => {
                write!(formatter, "CPU admission request sequence exhausted")
            }
            Self::CoordinatorPanicked => write!(formatter, "CPU admission coordinator panicked"),
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
        let id = self
            .next_request_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| AdmissionError::RequestSequenceExhausted)?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(CoordinatorCommand::Enqueue {
                id,
                request,
                reply: reply_tx,
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

pub struct ExecutionAdmissionCoordinator {
    client: ExecutionAdmissionClient,
    join: Option<JoinHandle<()>>,
}

impl ExecutionAdmissionCoordinator {
    pub fn start(broker: CpuPermitBroker) -> io::Result<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let client = ExecutionAdmissionClient {
            command_tx: command_tx.clone(),
            next_request_id: Arc::new(AtomicU64::new(0)),
        };
        let coordinator_tx = command_tx;
        let join = std::thread::Builder::new()
            .name("neoethos-cpu-admission".to_owned())
            .spawn(move || run_coordinator(broker, command_rx, coordinator_tx))?;
        Ok(Self {
            client,
            join: Some(join),
        })
    }

    pub fn client(&self) -> ExecutionAdmissionClient {
        self.client.clone()
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
        request: CpuPermitRequest,
        reply: oneshot::Sender<Result<AdmittedCpuLease, AdmissionError>>,
    },
    Cancel {
        id: u64,
    },
    LeaseReturned,
    Shutdown {
        ack: mpsc::SyncSender<()>,
    },
}

struct QueuedAdmission {
    id: u64,
    request: CpuPermitRequest,
    reply: oneshot::Sender<Result<AdmittedCpuLease, AdmissionError>>,
}

#[derive(Default)]
struct AdmissionQueues {
    children: VecDeque<QueuedAdmission>,
    local: VecDeque<QueuedAdmission>,
}

impl AdmissionQueues {
    fn enqueue(&mut self, queued: QueuedAdmission) {
        match queued.request.priority {
            CpuPriority::Child => self.children.push_back(queued),
            CpuPriority::Local => self.local.push_back(queued),
        }
    }

    fn cancel(&mut self, id: u64) {
        for queue in [&mut self.children, &mut self.local] {
            if let Some(position) = queue.iter().position(|queued| queued.id == id) {
                if let Some(queued) = queue.remove(position) {
                    let _ = queued.reply.send(Err(AdmissionError::Cancelled));
                }
                return;
            }
        }
    }

    fn discard_closed(&mut self) {
        self.children.retain(|queued| !queued.reply.is_closed());
        self.local.retain(|queued| !queued.reply.is_closed());
    }

    fn next_request(&self) -> Option<CpuPermitRequest> {
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
            let _ = queued.reply.send(Err(AdmissionError::CoordinatorStopped));
        }
    }
}

fn run_coordinator(
    broker: CpuPermitBroker,
    command_rx: mpsc::Receiver<CoordinatorCommand>,
    command_tx: mpsc::Sender<CoordinatorCommand>,
) {
    let mut queues = AdmissionQueues::default();
    loop {
        let Ok(command) = command_rx.recv() else {
            queues.stop_all();
            return;
        };
        if handle_command(command, &mut queues) {
            return;
        }

        // Make every already-enqueued request/cancellation visible before a
        // grant decision. This is what lets a later child overtake a local
        // request that was waiting for capacity.
        while let Ok(command) = command_rx.try_recv() {
            if handle_command(command, &mut queues) {
                return;
            }
        }

        grant_ready_requests(&broker, &command_tx, &mut queues);
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
    broker: &CpuPermitBroker,
    command_tx: &mpsc::Sender<CoordinatorCommand>,
    queues: &mut AdmissionQueues,
) {
    loop {
        queues.discard_closed();
        let Some(request) = queues.next_request() else {
            return;
        };
        match broker.try_acquire(request) {
            Ok(Some(lease)) => {
                let queued = queues
                    .pop_next()
                    .expect("a request exists because it was just inspected");
                let admitted = AdmittedCpuLease {
                    lease: Some(lease),
                    command_tx: command_tx.clone(),
                };
                if let Err(returned) = queued.reply.send(Ok(admitted)) {
                    drop(returned);
                }
            }
            Ok(None) => return,
            Err(error) => {
                let queued = queues
                    .pop_next()
                    .expect("a request exists because it was just inspected");
                let _ = queued.reply.send(Err(error.into()));
            }
        }
    }
}
