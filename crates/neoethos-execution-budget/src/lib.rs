//! One immutable CPU-capacity authority and a synchronous RAII permit broker.
//!
//! This crate deliberately depends only on the Rust standard library. It can
//! therefore be installed before Tokio, Rayon, tracing, GPU runtimes, model
//! libraries, or any other dependency-owned global worker pool is created.

#![forbid(unsafe_code)]

mod x86_64_v3;

pub use x86_64_v3::{
    DetectedCpuArchitectureV1, X86_64_V3_REQUIREMENTS_V1, X8664V3FeatureSetV1,
    X8664V3PreflightErrorCodeV1, X8664V3PreflightErrorV1, X8664V3RequirementV1, X8664V3SnapshotV1,
    detect_current_x86_64_v3_snapshot_v1, evaluate_x86_64_v3_snapshot_v1,
    require_current_x86_64_v3_v1,
};

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

impl CoordinationScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProcessLocal => "process_local",
            Self::ManagedProcessTree => "managed_process_tree",
        }
    }
}

pub const CPU_THREADS_FLAG: &str = "--cpu-threads";
pub const STARTUP_DIAGNOSTICS_FLAG: &str = "--startup-diagnostics";

/// Parse the ephemeral parent-to-child CPU assignment before any async
/// runtime exists. The parser is intentionally shared by every executable so
/// zero, malformed, inline, missing, and duplicate values cannot acquire
/// subtly different meanings at different process boundaries.
pub fn parse_parent_cpu_assignment(
    args: &[String],
) -> Result<Option<WorkerLimit>, ParentCpuAssignmentError> {
    let mut parsed = None;
    let mut index = 1;
    while index < args.len() {
        let arg = &args[index];
        let raw = if arg == CPU_THREADS_FLAG {
            index += 1;
            args.get(index)
                .ok_or(ParentCpuAssignmentError::MissingValue)?
                .as_str()
        } else if let Some(raw) = arg.strip_prefix("--cpu-threads=") {
            raw
        } else {
            index += 1;
            continue;
        };

        if parsed.is_some() {
            return Err(ParentCpuAssignmentError::Duplicate);
        }
        let value = raw
            .parse::<usize>()
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| ParentCpuAssignmentError::InvalidValue(raw.to_string()))?;
        parsed = Some(WorkerLimit(value));
        index += 1;
    }
    Ok(parsed)
}

pub fn startup_diagnostics_requested(args: &[String]) -> bool {
    args.iter()
        .skip(1)
        .any(|arg| arg == STARTUP_DIAGNOSTICS_FLAG)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParentCpuAssignmentError {
    MissingValue,
    InvalidValue(String),
    Duplicate,
}

impl fmt::Display for ParentCpuAssignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue => {
                write!(f, "{CPU_THREADS_FLAG} requires one positive integer value")
            }
            Self::InvalidValue(value) => write!(
                f,
                "{CPU_THREADS_FLAG} expects a positive integer, got `{value}`"
            ),
            Self::Duplicate => write!(f, "{CPU_THREADS_FLAG} may be supplied only once"),
        }
    }
}

impl Error for ParentCpuAssignmentError {}

/// Construct the zero-config request used by isolated sidecars. A parent cap
/// always changes the coordination scope and is re-clamped against the
/// child's own effective process capacity by the normal resolver.
pub fn detected_request_with_parent(parent: Option<WorkerLimit>) -> ExecutionBudgetRequest {
    let coordination_scope = if parent.is_some() {
        CoordinationScope::ManagedProcessTree
    } else {
        CoordinationScope::ProcessLocal
    };
    let mut request = ExecutionBudgetRequest::detect(coordination_scope);
    request.parent_limit = parent.map(BudgetCap::parent);
    request
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StartupEvent {
    ImportSignalPreflightCompleted,
    ConfigurationSeededOrLocated,
    ConfigurationLoaded,
    ParentCpuCapParsed,
    CpuBudgetResolved,
    CpuBudgetInstalled,
    RuntimeSettingsInstalled,
    TokioRuntimeBuilt,
    TauriAsyncRuntimeInstalled,
    ApplicationBuilderStarted,
}

impl StartupEvent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ImportSignalPreflightCompleted => "import_signal_preflight_completed",
            Self::ConfigurationSeededOrLocated => "configuration_seeded_or_located",
            Self::ConfigurationLoaded => "configuration_loaded",
            Self::ParentCpuCapParsed => "parent_cpu_cap_parsed",
            Self::CpuBudgetResolved => "cpu_budget_resolved",
            Self::CpuBudgetInstalled => "cpu_budget_installed",
            Self::RuntimeSettingsInstalled => "runtime_settings_installed",
            Self::TokioRuntimeBuilt => "tokio_runtime_built",
            Self::TauriAsyncRuntimeInstalled => "tauri_async_runtime_installed",
            Self::ApplicationBuilderStarted => "application_builder_started",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StartupTrace {
    events: Vec<StartupEvent>,
}

impl StartupTrace {
    pub fn record(&mut self, event: StartupEvent) -> Result<(), StartupOrderError> {
        if let Some(previous) = self.events.last().copied()
            && event <= previous
        {
            return Err(StartupOrderError {
                previous,
                next: event,
            });
        }
        self.events.push(event);
        Ok(())
    }

    pub fn events(&self) -> &[StartupEvent] {
        &self.events
    }

    fn csv(&self) -> String {
        self.events
            .iter()
            .map(|event| event.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartupOrderError {
    pub previous: StartupEvent,
    pub next: StartupEvent,
}

impl fmt::Display for StartupOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "startup event `{}` cannot follow `{}`",
            self.next.as_str(),
            self.previous.as_str()
        )
    }
}

impl Error for StartupOrderError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupRuntimeKind {
    Synchronous,
    Tokio,
    Tauri,
}

impl StartupRuntimeKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Synchronous => "synchronous",
            Self::Tokio => "tokio",
            Self::Tauri => "tauri",
        }
    }
}

/// Stable, one-line startup evidence suitable for stderr (including stdio MCP
/// processes whose stdout must remain JSON-RPC clean).
pub fn format_startup_diagnostics(
    executable: &str,
    installed: &InstalledExecutionBudget,
    runtime_kind: StartupRuntimeKind,
    runtime_worker_threads: Option<usize>,
    trace: &StartupTrace,
) -> String {
    let resolved = installed.resolved();
    let capacity_source = match resolved.capacity_source {
        CapacityDetectionSource::AvailableParallelism => "available_parallelism",
        CapacityDetectionSource::SuppliedForResolution => "supplied_for_resolution",
        CapacityDetectionSource::FallbackOneAfterDetectionFailure => {
            "fallback_one_after_detection_failure"
        }
    };
    let runtime_workers = runtime_worker_threads
        .map(|workers| workers.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!(
        "NEOETHOS_STARTUP_V1 executable={executable} effective_logical_threads={} \
         reserved_logical_threads={} automatic_worker_limit={} effective_worker_limit={} \
         capacity_source={capacity_source} coordination_scope={} runtime_kind={} \
         runtime_worker_threads={runtime_workers} events={}",
        resolved.effective_logical_threads.get(),
        resolved.reserved_logical_threads,
        resolved.automatic_worker_limit.get(),
        resolved.effective_worker_limit.get(),
        resolved.coordination_scope.as_str(),
        runtime_kind.as_str(),
        trace.csv(),
    )
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

/// A positive bound on an auxiliary resource pool coordinated with CPU
/// permits. The resource remains deliberately unnamed so callers can use the
/// same authority for signal slots, codec slots, device-init slots, or another
/// bounded one-at-a-time prerequisite without creating a second CPU budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuxiliarySlotLimit(NonZeroUsize);

impl AuxiliarySlotLimit {
    pub fn new(value: usize) -> Result<Self, InvalidPositiveCount> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InvalidPositiveCount {
                kind: "auxiliary slots",
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuxiliarySlotRequest {
    None,
    One,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompositeAdmissionRequest {
    pub cpu: CpuPermitRequest,
    pub auxiliary: AuxiliarySlotRequest,
}

impl CompositeAdmissionRequest {
    pub const fn new(cpu: CpuPermitRequest, auxiliary: AuxiliarySlotRequest) -> Self {
        Self { cpu, auxiliary }
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

#[derive(Clone)]
struct Waiter {
    id: u64,
    request: CpuPermitRequest,
    auxiliary_pool: Option<Arc<AuxiliaryPoolInner>>,
}

struct AuxiliaryPoolInner {
    limit: AuxiliarySlotLimit,
    state: Mutex<AuxiliaryPoolState>,
}

struct AuxiliaryPoolState {
    free_slots: Vec<usize>,
}

struct AcquiredResources {
    cpu_lease: CpuLease,
    auxiliary_slot: Option<AuxiliarySlotLease>,
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
        broker_snapshot_from_state(&self.inner, &state)
    }

    /// True only when the transfer was issued by this exact broker instance.
    /// Width equality is insufficient: a lease from another broker would
    /// create a second capacity authority and allow aggregate oversubscription.
    pub fn owns_transfer(&self, transfer: &CpuLeaseTransfer) -> bool {
        transfer
            .lease
            .as_ref()
            .is_some_and(|lease| Arc::ptr_eq(&self.inner, &lease.broker))
    }

    /// Acquire immediately without bypassing an already queued request.
    pub fn try_acquire(&self, request: CpuPermitRequest) -> Result<Option<CpuLease>, AcquireError> {
        self.try_acquire_resources(request, None)
            .map(|resources| resources.map(|resources| resources.cpu_lease))
    }

    fn try_acquire_resources(
        &self,
        request: CpuPermitRequest,
        auxiliary_pool: Option<Arc<AuxiliaryPoolInner>>,
    ) -> Result<Option<AcquiredResources>, AcquireError> {
        reject_nested_acquisition()?;
        self.validate_width(request.width)?;
        let mut state = lock_unpoisoned(&self.inner.state);
        if !state.waiters.is_empty()
            || state.available < request.width.get()
            || !auxiliary_slot_available(auxiliary_pool.as_ref())
        {
            return Ok(None);
        }
        Ok(Some(reserve_resources(
            &self.inner,
            &mut state,
            request,
            auxiliary_pool,
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
        self.acquire_resources_inner(request, None, cancellation)
            .map(|resources| resources.cpu_lease)
    }

    fn acquire_resources_inner(
        &self,
        request: CpuPermitRequest,
        auxiliary_pool: Option<Arc<AuxiliaryPoolInner>>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<AcquiredResources, AcquireError> {
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
        state.waiters.push_back(Waiter {
            id,
            request,
            auxiliary_pool: auxiliary_pool.clone(),
        });

        loop {
            if cancellation.is_some_and(CancellationToken::is_cancelled) {
                remove_waiter(&mut state, id);
                self.inner.changed.notify_all();
                return Err(AcquireError::Cancelled);
            }

            let eligible = next_eligible_waiter(&state).is_some_and(|waiter| waiter.id == id);
            if eligible
                && state.available >= request.width.get()
                && auxiliary_slot_available(auxiliary_pool.as_ref())
            {
                remove_waiter(&mut state, id);
                let resources = reserve_resources(&self.inner, &mut state, request, auxiliary_pool);
                self.inner.changed.notify_all();
                return Ok(resources);
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

fn broker_snapshot_from_state(inner: &BrokerInner, state: &BrokerState) -> BrokerSnapshot {
    let queued_children = state
        .waiters
        .iter()
        .filter(|waiter| waiter.request.priority == CpuPriority::Child)
        .count();
    let queued_local = state.waiters.len() - queued_children;
    BrokerSnapshot {
        installed_limit: inner.installed_limit,
        available_permits: state.available,
        live_reserved_sum: inner.installed_limit.get() - state.available,
        queued_total: state.waiters.len(),
        queued_children,
        queued_local,
    }
}

fn auxiliary_slot_available(pool: Option<&Arc<AuxiliaryPoolInner>>) -> bool {
    pool.is_none_or(|pool| !lock_unpoisoned(&pool.state).free_slots.is_empty())
}

fn reserve_resources(
    broker: &Arc<BrokerInner>,
    state: &mut BrokerState,
    request: CpuPermitRequest,
    auxiliary_pool: Option<Arc<AuxiliaryPoolInner>>,
) -> AcquiredResources {
    let auxiliary_slot = auxiliary_pool.map(|pool| {
        let index = {
            let mut auxiliary = lock_unpoisoned(&pool.state);
            auxiliary
                .free_slots
                .pop()
                .expect("an auxiliary slot was checked before reservation")
        };
        AuxiliarySlotLease {
            broker: Arc::clone(broker),
            pool,
            index,
        }
    });
    state.available -= request.width.get();
    AcquiredResources {
        cpu_lease: CpuLease::new(Arc::clone(broker), request.width.get()),
        auxiliary_slot,
    }
}

/// One CPU broker plus one independently bounded auxiliary pool. Composite
/// requests join the CPU broker's existing priority/FIFO queue, so they are
/// coordinated with ordinary [`CpuPermitBroker`] callers. Neither resource is
/// removed until the selected waiter can receive both in one critical section.
#[derive(Clone)]
pub struct CompositeAdmissionAuthority {
    broker: CpuPermitBroker,
    auxiliary_pool: Arc<AuxiliaryPoolInner>,
}

impl CompositeAdmissionAuthority {
    pub fn new(broker: CpuPermitBroker, auxiliary_limit: AuxiliarySlotLimit) -> Self {
        let free_slots = (0..auxiliary_limit.get()).rev().collect();
        Self {
            broker,
            auxiliary_pool: Arc::new(AuxiliaryPoolInner {
                limit: auxiliary_limit,
                state: Mutex::new(AuxiliaryPoolState { free_slots }),
            }),
        }
    }

    pub fn snapshot(&self) -> CompositeAdmissionSnapshot {
        let broker_state = lock_unpoisoned(&self.broker.inner.state);
        let auxiliary_state = lock_unpoisoned(&self.auxiliary_pool.state);
        let queued_requiring_auxiliary = broker_state
            .waiters
            .iter()
            .filter(|waiter| {
                waiter
                    .auxiliary_pool
                    .as_ref()
                    .is_some_and(|pool| Arc::ptr_eq(pool, &self.auxiliary_pool))
            })
            .count();
        let available_auxiliary_slots = auxiliary_state.free_slots.len();
        CompositeAdmissionSnapshot {
            cpu: broker_snapshot_from_state(&self.broker.inner, &broker_state),
            auxiliary_limit: self.auxiliary_pool.limit,
            available_auxiliary_slots,
            live_auxiliary_slots: self.auxiliary_pool.limit.get() - available_auxiliary_slots,
            queued_requiring_auxiliary,
        }
    }

    pub fn try_acquire(
        &self,
        request: CompositeAdmissionRequest,
    ) -> Result<Option<CompositeAdmissionGrant>, AcquireError> {
        self.broker
            .try_acquire_resources(request.cpu, self.requested_pool(request.auxiliary))
            .map(|resources| resources.map(CompositeAdmissionGrant::from_resources))
    }

    /// Blocking acquisition for a coordinator or synchronous top-level
    /// thread. Async runtime workers must queue through such a coordinator.
    pub fn acquire(
        &self,
        request: CompositeAdmissionRequest,
    ) -> Result<CompositeAdmissionGrant, AcquireError> {
        self.broker
            .acquire_resources_inner(request.cpu, self.requested_pool(request.auxiliary), None)
            .map(CompositeAdmissionGrant::from_resources)
    }

    pub fn acquire_cancellable(
        &self,
        request: CompositeAdmissionRequest,
        cancellation: &CancellationToken,
    ) -> Result<CompositeAdmissionGrant, AcquireError> {
        self.broker
            .acquire_resources_inner(
                request.cpu,
                self.requested_pool(request.auxiliary),
                Some(cancellation),
            )
            .map(CompositeAdmissionGrant::from_resources)
    }

    fn requested_pool(&self, request: AuxiliarySlotRequest) -> Option<Arc<AuxiliaryPoolInner>> {
        match request {
            AuxiliarySlotRequest::None => None,
            AuxiliarySlotRequest::One => Some(Arc::clone(&self.auxiliary_pool)),
        }
    }
}

impl fmt::Debug for CompositeAdmissionAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.snapshot().fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompositeAdmissionSnapshot {
    pub cpu: BrokerSnapshot,
    pub auxiliary_limit: AuxiliarySlotLimit,
    pub available_auxiliary_slots: usize,
    pub live_auxiliary_slots: usize,
    pub queued_requiring_auxiliary: usize,
}

/// A grant created only after CPU permits and the optional auxiliary slot were
/// simultaneously available. The two RAII parts may be retained together or
/// separated when a caller must release the auxiliary prerequisite earlier
/// than its CPU reservation.
#[must_use = "dropping the grant returns its CPU permits and auxiliary slot"]
pub struct CompositeAdmissionGrant {
    cpu_lease: CpuLease,
    auxiliary_slot: Option<AuxiliarySlotLease>,
}

impl CompositeAdmissionGrant {
    fn from_resources(resources: AcquiredResources) -> Self {
        Self {
            cpu_lease: resources.cpu_lease,
            auxiliary_slot: resources.auxiliary_slot,
        }
    }

    pub fn cpu_lease(&self) -> &CpuLease {
        &self.cpu_lease
    }

    pub fn auxiliary_slot(&self) -> Option<&AuxiliarySlotLease> {
        self.auxiliary_slot.as_ref()
    }

    pub fn into_parts(self) -> (CpuLease, Option<AuxiliarySlotLease>) {
        (self.cpu_lease, self.auxiliary_slot)
    }
}

impl fmt::Debug for CompositeAdmissionGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositeAdmissionGrant")
            .field("cpu_width", &self.cpu_lease.width())
            .field(
                "auxiliary_slot",
                &self.auxiliary_slot.as_ref().map(AuxiliarySlotLease::index),
            )
            .finish()
    }
}

/// One non-cloneable slot from the authority's bounded auxiliary pool.
#[must_use = "dropping the lease returns its auxiliary slot"]
pub struct AuxiliarySlotLease {
    broker: Arc<BrokerInner>,
    pool: Arc<AuxiliaryPoolInner>,
    index: usize,
}

impl AuxiliarySlotLease {
    pub const fn index(&self) -> usize {
        self.index
    }
}

impl fmt::Debug for AuxiliarySlotLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuxiliarySlotLease")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl Drop for AuxiliarySlotLease {
    fn drop(&mut self) {
        let _broker_state = lock_unpoisoned(&self.broker.state);
        let mut auxiliary = lock_unpoisoned(&self.pool.state);
        assert!(
            !auxiliary.free_slots.contains(&self.index),
            "auxiliary slot {} was returned more than once",
            self.index
        );
        auxiliary.free_slots.push(self.index);
        assert!(
            auxiliary.free_slots.len() <= self.pool.limit.get(),
            "returned auxiliary slots exceed the configured limit"
        );
        self.broker.changed.notify_all();
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

/// Opaque worker-lifetime proof used by a private lease-bound executor pool.
///
/// A cached pool retains idle OS threads between calls, so wrapping only the
/// parent `install` closure is insufficient: Rayon may steal a nested job onto
/// another worker whose thread-local scope was never marked. The budgeted
/// executor enters this scope once in each private worker's spawn closure and
/// retains the guard until that worker exits. The private pool itself remains
/// unreachable unless its owner accepts a matching transferred lease.
#[must_use = "dropping the worker scope re-enables fresh permit acquisition on this thread"]
pub struct LeaseBoundWorkerScope {
    _active: ActiveLeaseScope,
}

/// Enter the active-lease scope for the lifetime of one private executor
/// worker. General workload code must use [`CpuLease::scope`] instead; this
/// function exists for the worker-spawn boundary in `BudgetedCpuExecutor`.
pub fn enter_lease_bound_worker_scope() -> LeaseBoundWorkerScope {
    LeaseBoundWorkerScope {
        _active: ActiveLeaseScope::enter(),
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
