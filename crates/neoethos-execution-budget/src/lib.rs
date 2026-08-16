//! One immutable CPU-capacity authority and a synchronous RAII permit broker.
//!
//! This crate deliberately depends only on the Rust standard library. It can
//! therefore be installed before Tokio, Rayon, tracing, GPU runtimes, model
//! libraries, or any other dependency-owned global worker pool is created.

#![forbid(unsafe_code)]

use std::cell::Cell;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};

/// A positive count of logical threads visible to an inventory or process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalThreadCount(NonZeroUsize);

impl LogicalThreadCount {
    pub fn new(value: usize) -> Result<Self, InvalidPositiveCount> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InvalidPositiveCount {
                kind: "logical threads",
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// A positive upper bound on CPU workers or reserved permits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerLimit(NonZeroUsize);

impl WorkerLimit {
    pub fn new(value: usize) -> Result<Self, InvalidPositiveCount> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InvalidPositiveCount { kind: "workers" })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }

    fn from_known_positive(value: usize) -> Self {
        Self(NonZeroUsize::new(value).expect("internal worker count must remain positive"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidPositiveCount {
    kind: &'static str,
}

impl fmt::Display for InvalidPositiveCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} must be greater than zero", self.kind)
    }
}

impl Error for InvalidPositiveCount {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityDetectionSource {
    AvailableParallelism,
    SuppliedForResolution,
    FallbackOneAfterDetectionFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionDiagnosticCode {
    AvailableParallelismFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityDiagnostic {
    pub code: DetectionDiagnosticCode,
    pub detail: String,
}

/// The result of effective process-capacity detection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityDetection {
    pub effective_logical_threads: LogicalThreadCount,
    pub source: CapacityDetectionSource,
    pub diagnostic: Option<CapacityDiagnostic>,
}

impl CapacityDetection {
    /// Read the OS/cgroup/affinity-aware process capacity once.
    pub fn detect() -> Self {
        match std::thread::available_parallelism() {
            Ok(value) => Self {
                effective_logical_threads: LogicalThreadCount(value),
                source: CapacityDetectionSource::AvailableParallelism,
                diagnostic: None,
            },
            Err(error) => Self::failed(error.to_string()),
        }
    }

    /// Explicit deterministic input for pure resolution and startup tests.
    pub fn supplied(effective_logical_threads: LogicalThreadCount) -> Self {
        Self {
            effective_logical_threads,
            source: CapacityDetectionSource::SuppliedForResolution,
            diagnostic: None,
        }
    }

    /// Structured fail-safe result when `available_parallelism` is unavailable.
    pub fn failed(detail: impl Into<String>) -> Self {
        Self {
            effective_logical_threads: LogicalThreadCount(
                NonZeroUsize::new(1).expect("one is non-zero"),
            ),
            source: CapacityDetectionSource::FallbackOneAfterDetectionFailure,
            diagnostic: Some(CapacityDiagnostic {
                code: DetectionDiagnosticCode::AvailableParallelismFailed,
                detail: detail.into(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetCapProvenance {
    PersistentSetting,
    LegacyPersistentSetting,
    ParentAssignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetCap {
    pub limit: WorkerLimit,
    pub provenance: BudgetCapProvenance,
}

impl BudgetCap {
    pub const fn new(limit: WorkerLimit, provenance: BudgetCapProvenance) -> Self {
        Self { limit, provenance }
    }

    pub const fn persistent(limit: WorkerLimit) -> Self {
        Self::new(limit, BudgetCapProvenance::PersistentSetting)
    }

    pub const fn legacy(limit: WorkerLimit) -> Self {
        Self::new(limit, BudgetCapProvenance::LegacyPersistentSetting)
    }

    pub const fn parent(limit: WorkerLimit) -> Self {
        Self::new(limit, BudgetCapProvenance::ParentAssignment)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinationScope {
    ProcessLocal,
    ManagedProcessTree,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionBudgetRequest {
    pub host_logical_threads: Option<LogicalThreadCount>,
    pub detection: CapacityDetection,
    pub persistent_limit: Option<BudgetCap>,
    pub legacy_persistent_limit: Option<BudgetCap>,
    pub parent_limit: Option<BudgetCap>,
    pub coordination_scope: CoordinationScope,
}

impl ExecutionBudgetRequest {
    /// Construct a production request from the process's current effective
    /// `available_parallelism()` result and no optional caps.
    pub fn detect(coordination_scope: CoordinationScope) -> Self {
        Self {
            host_logical_threads: None,
            detection: CapacityDetection::detect(),
            persistent_limit: None,
            legacy_persistent_limit: None,
            parent_limit: None,
            coordination_scope,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedExecutionBudget {
    pub host_logical_threads: Option<LogicalThreadCount>,
    pub effective_logical_threads: LogicalThreadCount,
    pub reserved_logical_threads: usize,
    pub automatic_worker_limit: WorkerLimit,
    pub persistent_limit: Option<BudgetCap>,
    pub legacy_persistent_limit: Option<BudgetCap>,
    pub parent_limit: Option<BudgetCap>,
    pub effective_worker_limit: WorkerLimit,
    pub capacity_source: CapacityDetectionSource,
    pub capacity_diagnostic: Option<CapacityDiagnostic>,
    pub coordination_scope: CoordinationScope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError {
    MismatchedCapProvenance {
        field: &'static str,
        expected: BudgetCapProvenance,
        actual: BudgetCapProvenance,
    },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedCapProvenance {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} has provenance {actual:?}, expected {expected:?}"
            ),
        }
    }
}

impl Error for ResolutionError {}

/// Resolve the process ceiling without reading global state or initializing a
/// runtime. Optional limits only narrow the automatic `effective - reserve`
/// result; an oversized limit never consumes the two-thread stability reserve.
pub fn resolve_execution_budget(
    request: ExecutionBudgetRequest,
) -> Result<ResolvedExecutionBudget, ResolutionError> {
    validate_cap(
        "persistent_limit",
        request.persistent_limit,
        BudgetCapProvenance::PersistentSetting,
    )?;
    validate_cap(
        "legacy_persistent_limit",
        request.legacy_persistent_limit,
        BudgetCapProvenance::LegacyPersistentSetting,
    )?;
    validate_cap(
        "parent_limit",
        request.parent_limit,
        BudgetCapProvenance::ParentAssignment,
    )?;

    let effective = request.detection.effective_logical_threads.get();
    let reserved = effective.saturating_sub(1).min(2);
    let automatic = WorkerLimit::from_known_positive(effective - reserved);
    let mut final_limit = automatic.get();
    for cap in [
        request.persistent_limit,
        request.legacy_persistent_limit,
        request.parent_limit,
    ]
    .into_iter()
    .flatten()
    {
        final_limit = final_limit.min(cap.limit.get());
    }

    Ok(ResolvedExecutionBudget {
        host_logical_threads: request.host_logical_threads,
        effective_logical_threads: request.detection.effective_logical_threads,
        reserved_logical_threads: reserved,
        automatic_worker_limit: automatic,
        persistent_limit: request.persistent_limit,
        legacy_persistent_limit: request.legacy_persistent_limit,
        parent_limit: request.parent_limit,
        effective_worker_limit: WorkerLimit::from_known_positive(final_limit),
        capacity_source: request.detection.source,
        capacity_diagnostic: request.detection.diagnostic,
        coordination_scope: request.coordination_scope,
    })
}

fn validate_cap(
    field: &'static str,
    cap: Option<BudgetCap>,
    expected: BudgetCapProvenance,
) -> Result<(), ResolutionError> {
    if let Some(cap) = cap
        && cap.provenance != expected
    {
        return Err(ResolutionError::MismatchedCapProvenance {
            field,
            expected,
            actual: cap.provenance,
        });
    }
    Ok(())
}

/// The immutable budget and its one process-global permit broker.
pub struct InstalledExecutionBudget {
    resolved: ResolvedExecutionBudget,
    broker: CpuPermitBroker,
}

impl InstalledExecutionBudget {
    pub fn resolved(&self) -> &ResolvedExecutionBudget {
        &self.resolved
    }

    pub fn broker(&self) -> &CpuPermitBroker {
        &self.broker
    }
}

impl fmt::Debug for InstalledExecutionBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstalledExecutionBudget")
            .field("resolved", &self.resolved)
            .field("broker", &self.broker.snapshot())
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    Resolution(ResolutionError),
    ConflictingInstallation {
        installed: Box<ResolvedExecutionBudget>,
        requested: Box<ResolvedExecutionBudget>,
    },
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution(error) => write!(f, "cannot resolve process CPU budget: {error}"),
            Self::ConflictingInstallation {
                installed,
                requested,
            } => write!(
                f,
                "process CPU budget is already installed as {:?}, conflicting request is {:?}",
                installed.effective_worker_limit, requested.effective_worker_limit
            ),
        }
    }
}

impl Error for InstallError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Resolution(error) => Some(error),
            Self::ConflictingInstallation { .. } => None,
        }
    }
}

impl From<ResolutionError> for InstallError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

static INSTALLED_PROCESS_BUDGET: OnceLock<InstalledExecutionBudget> = OnceLock::new();

/// Install the immutable process budget. Equal reinstallation is idempotent;
/// any conflicting resolved record fails instead of silently accepting the
/// first caller's value.
pub fn install_process_budget(
    request: ExecutionBudgetRequest,
) -> Result<&'static InstalledExecutionBudget, InstallError> {
    let resolved = resolve_execution_budget(request)?;
    if let Some(installed) = INSTALLED_PROCESS_BUDGET.get() {
        return compare_installation(installed, resolved);
    }

    let candidate = InstalledExecutionBudget {
        broker: CpuPermitBroker::new(resolved.effective_worker_limit),
        resolved: resolved.clone(),
    };
    match INSTALLED_PROCESS_BUDGET.set(candidate) {
        Ok(()) => Ok(INSTALLED_PROCESS_BUDGET
            .get()
            .expect("OnceLock contains the value just installed")),
        Err(_candidate) => compare_installation(
            INSTALLED_PROCESS_BUDGET
                .get()
                .expect("a failed OnceLock set has an installed winner"),
            resolved,
        ),
    }
}

pub fn installed_process_budget() -> Option<&'static InstalledExecutionBudget> {
    INSTALLED_PROCESS_BUDGET.get()
}

fn compare_installation(
    installed: &'static InstalledExecutionBudget,
    requested: ResolvedExecutionBudget,
) -> Result<&'static InstalledExecutionBudget, InstallError> {
    if installed.resolved == requested {
        Ok(installed)
    } else {
        Err(InstallError::ConflictingInstallation {
            installed: Box::new(installed.resolved.clone()),
            requested: Box::new(requested),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuPriority {
    Child,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuPermitRequest {
    pub width: WorkerLimit,
    pub priority: CpuPriority,
}

impl CpuPermitRequest {
    pub const fn new(width: WorkerLimit, priority: CpuPriority) -> Self {
        Self { width, priority }
    }

    pub const fn child(width: WorkerLimit) -> Self {
        Self::new(width, CpuPriority::Child)
    }

    pub const fn local(width: WorkerLimit) -> Self {
        Self::new(width, CpuPriority::Local)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcquireError {
    ExceedsInstalledLimit { requested: usize, installed: usize },
    Cancelled,
    NestedAcquisition,
    WaiterSequenceExhausted,
}

impl fmt::Display for AcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceedsInstalledLimit {
                requested,
                installed,
            } => write!(
                f,
                "requested {requested} CPU permits, but the installed limit is {installed}"
            ),
            Self::Cancelled => write!(f, "CPU permit request was cancelled"),
            Self::NestedAcquisition => write!(
                f,
                "fresh CPU permit acquisition is forbidden inside an active lease scope; split the existing lease"
            ),
            Self::WaiterSequenceExhausted => write!(f, "CPU permit waiter sequence exhausted"),
        }
    }
}

impl Error for AcquireError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseError {
    SplitWouldEmptyParent { parent: usize, requested: usize },
    SplitExceedsParent { parent: usize, requested: usize },
}

impl fmt::Display for LeaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplitWouldEmptyParent { parent, requested } => write!(
                f,
                "cannot split all {requested} permits from a {parent}-permit parent; transfer the lease instead"
            ),
            Self::SplitExceedsParent { parent, requested } => write!(
                f,
                "cannot split {requested} permits from a {parent}-permit parent lease"
            ),
        }
    }
}

impl Error for LeaseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrokerSnapshot {
    pub installed_limit: WorkerLimit,
    pub available_permits: usize,
    pub live_reserved_sum: usize,
    pub queued_total: usize,
    pub queued_children: usize,
    pub queued_local: usize,
}

#[derive(Clone)]
pub struct CpuPermitBroker {
    inner: Arc<BrokerInner>,
}

struct BrokerInner {
    installed_limit: WorkerLimit,
    state: Mutex<BrokerState>,
    changed: Condvar,
}

struct BrokerState {
    available: usize,
    next_waiter_id: u64,
    waiters: VecDeque<Waiter>,
}

#[derive(Clone, Copy)]
struct Waiter {
    id: u64,
    request: CpuPermitRequest,
}

impl CpuPermitBroker {
    pub fn new(installed_limit: WorkerLimit) -> Self {
        Self {
            inner: Arc::new(BrokerInner {
                installed_limit,
                state: Mutex::new(BrokerState {
                    available: installed_limit.get(),
                    next_waiter_id: 0,
                    waiters: VecDeque::new(),
                }),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn snapshot(&self) -> BrokerSnapshot {
        let state = lock_unpoisoned(&self.inner.state);
        let queued_children = state
            .waiters
            .iter()
            .filter(|waiter| waiter.request.priority == CpuPriority::Child)
            .count();
        let queued_local = state.waiters.len() - queued_children;
        BrokerSnapshot {
            installed_limit: self.inner.installed_limit,
            available_permits: state.available,
            live_reserved_sum: self.inner.installed_limit.get() - state.available,
            queued_total: state.waiters.len(),
            queued_children,
            queued_local,
        }
    }

    /// Acquire immediately without bypassing an already queued request.
    pub fn try_acquire(&self, request: CpuPermitRequest) -> Result<Option<CpuLease>, AcquireError> {
        reject_nested_acquisition()?;
        self.validate_width(request.width)?;
        let mut state = lock_unpoisoned(&self.inner.state);
        if !state.waiters.is_empty() || state.available < request.width.get() {
            return Ok(None);
        }
        state.available -= request.width.get();
        Ok(Some(CpuLease::new(
            Arc::clone(&self.inner),
            request.width.get(),
        )))
    }

    /// Blocking acquisition for a dedicated coordinator or synchronous
    /// top-level thread. Async runtime workers must not call this method.
    pub fn acquire(&self, request: CpuPermitRequest) -> Result<CpuLease, AcquireError> {
        self.acquire_inner(request, None)
    }

    pub fn acquire_cancellable(
        &self,
        request: CpuPermitRequest,
        cancellation: &CancellationToken,
    ) -> Result<CpuLease, AcquireError> {
        self.acquire_inner(request, Some(cancellation))
    }

    fn acquire_inner(
        &self,
        request: CpuPermitRequest,
        cancellation: Option<&CancellationToken>,
    ) -> Result<CpuLease, AcquireError> {
        reject_nested_acquisition()?;
        self.validate_width(request.width)?;
        if let Some(cancellation) = cancellation {
            cancellation.register(&self.inner);
            if cancellation.is_cancelled() {
                return Err(AcquireError::Cancelled);
            }
        }

        let mut state = lock_unpoisoned(&self.inner.state);
        let id = state.next_waiter_id;
        state.next_waiter_id = state
            .next_waiter_id
            .checked_add(1)
            .ok_or(AcquireError::WaiterSequenceExhausted)?;
        state.waiters.push_back(Waiter { id, request });

        loop {
            if cancellation.is_some_and(CancellationToken::is_cancelled) {
                remove_waiter(&mut state, id);
                self.inner.changed.notify_all();
                return Err(AcquireError::Cancelled);
            }

            let eligible = next_eligible_waiter(&state).is_some_and(|waiter| waiter.id == id);
            if eligible && state.available >= request.width.get() {
                remove_waiter(&mut state, id);
                state.available -= request.width.get();
                self.inner.changed.notify_all();
                return Ok(CpuLease::new(Arc::clone(&self.inner), request.width.get()));
            }

            state = wait_unpoisoned(&self.inner.changed, state);
        }
    }

    fn validate_width(&self, requested: WorkerLimit) -> Result<(), AcquireError> {
        if requested > self.inner.installed_limit {
            Err(AcquireError::ExceedsInstalledLimit {
                requested: requested.get(),
                installed: self.inner.installed_limit.get(),
            })
        } else {
            Ok(())
        }
    }
}

impl fmt::Debug for CpuPermitBroker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.snapshot().fmt(f)
    }
}

fn next_eligible_waiter(state: &BrokerState) -> Option<&Waiter> {
    state
        .waiters
        .iter()
        .find(|waiter| waiter.request.priority == CpuPriority::Child)
        .or_else(|| state.waiters.front())
}

fn remove_waiter(state: &mut BrokerState, id: u64) {
    if let Some(position) = state.waiters.iter().position(|waiter| waiter.id == id) {
        state.waiters.remove(position);
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wait_unpoisoned<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar
        .wait(guard)
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Cooperative cancellation that wakes every broker on which this token has
/// a registered waiter. Notification takes the broker mutex before signalling,
/// preventing a cancellation between the waiter's final check and `wait()`
/// from becoming a lost wakeup.
#[derive(Clone, Default)]
pub struct CancellationToken {
    inner: Arc<CancellationInner>,
}

#[derive(Default)]
struct CancellationInner {
    cancelled: AtomicBool,
    brokers: Mutex<Vec<Weak<BrokerInner>>>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub fn cancel(&self) {
        let brokers = {
            let mut registered = lock_unpoisoned(&self.inner.brokers);
            let brokers = registered
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>();
            registered.retain(|broker| broker.strong_count() > 0);
            brokers
        };

        // Serialize the cancellation linearization point with each registered
        // broker's grant/removal decision. Acquisition never owns more than
        // one broker lock, so taking these in stable address order cannot form
        // a lock cycle. If a grant already owns its lock, it linearizes first;
        // otherwise the waiter observes cancellation before it can grant.
        let mut brokers = brokers;
        brokers.sort_unstable_by_key(Arc::as_ptr);
        let _states = brokers
            .iter()
            .map(|broker| lock_unpoisoned(&broker.state))
            .collect::<Vec<_>>();
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }
        for broker in &brokers {
            broker.changed.notify_all();
        }
    }

    fn register(&self, broker: &Arc<BrokerInner>) {
        let mut registered = lock_unpoisoned(&self.inner.brokers);
        let weak = Arc::downgrade(broker);
        if !registered.iter().any(|existing| existing.ptr_eq(&weak)) {
            registered.push(weak);
        }
    }
}

thread_local! {
    static ACTIVE_LEASE_SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn reject_nested_acquisition() -> Result<(), AcquireError> {
    let active = ACTIVE_LEASE_SCOPE_DEPTH.with(|depth| depth.get() > 0);
    if active {
        Err(AcquireError::NestedAcquisition)
    } else {
        Ok(())
    }
}

struct ActiveLeaseScope;

impl ActiveLeaseScope {
    fn enter() -> Self {
        ACTIVE_LEASE_SCOPE_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_add(1)
                    .expect("active CPU lease scope depth overflow"),
            );
        });
        Self
    }
}

impl Drop for ActiveLeaseScope {
    fn drop(&mut self) {
        ACTIVE_LEASE_SCOPE_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "active CPU lease scope underflow");
            depth.set(current.saturating_sub(1));
        });
    }
}

/// A non-cloneable reservation. Dropping it returns exactly its current width.
pub struct CpuLease {
    broker: Arc<BrokerInner>,
    width: usize,
}

impl CpuLease {
    fn new(broker: Arc<BrokerInner>, width: usize) -> Self {
        debug_assert!(width > 0);
        Self { broker, width }
    }

    pub fn width(&self) -> WorkerLimit {
        WorkerLimit::from_known_positive(self.width)
    }

    /// Split an existing reservation without consulting or waiting on the
    /// broker. Splitting the full width is rejected; transfer it instead.
    pub fn split(&mut self, width: WorkerLimit) -> Result<CpuLease, LeaseError> {
        if width.get() > self.width {
            return Err(LeaseError::SplitExceedsParent {
                parent: self.width,
                requested: width.get(),
            });
        }
        if width.get() == self.width {
            return Err(LeaseError::SplitWouldEmptyParent {
                parent: self.width,
                requested: width.get(),
            });
        }
        self.width -= width.get();
        Ok(CpuLease::new(Arc::clone(&self.broker), width.get()))
    }

    /// Mark the synchronous dynamic extent in which this lease authorizes CPU
    /// work. Fresh broker acquisition in the same extent is rejected; nested
    /// work must receive a split lease.
    pub fn scope<R>(&self, work: impl FnOnce() -> R) -> R {
        let _scope = ActiveLeaseScope::enter();
        work()
    }

    pub fn into_transfer(self) -> CpuLeaseTransfer {
        CpuLeaseTransfer { lease: Some(self) }
    }
}

impl fmt::Debug for CpuLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuLease")
            .field("width", &self.width)
            .finish_non_exhaustive()
    }
}

impl Drop for CpuLease {
    fn drop(&mut self) {
        let mut state = lock_unpoisoned(&self.broker.state);
        state.available = state
            .available
            .checked_add(self.width)
            .expect("CPU permit count overflow while returning lease");
        assert!(
            state.available <= self.broker.installed_limit.get(),
            "returned CPU permits exceed installed limit"
        );
        self.broker.changed.notify_all();
    }
}

/// Explicit ownership handoff used by coordinator-to-worker boundaries.
/// Dropping an unaccepted transfer still returns its permits through RAII.
pub struct CpuLeaseTransfer {
    lease: Option<CpuLease>,
}

impl CpuLeaseTransfer {
    pub fn accept(mut self) -> CpuLease {
        self.lease
            .take()
            .expect("a CPU lease transfer can be accepted only once")
    }
}

impl fmt::Debug for CpuLeaseTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuLeaseTransfer")
            .field("width", &self.lease.as_ref().map(CpuLease::width))
            .finish()
    }
}
