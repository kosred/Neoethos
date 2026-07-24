//! Device-resident backtest-engine contract and handle validation.

use neoethos_gpu_contracts::device::HandleToken;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BufferKind {
    Dataset = 1,
    Genes = 2,
    Scenarios = 3,
    Metrics = 4,
    Selection = 5,
    Event = 6,
}

impl BufferKind {
    const fn id(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    Ready,
    NotImplemented,
    NotBenchmarked,
    UnsupportedCapability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineCapabilities {
    pub fixed_stops: bool,
    pub adaptive_stops: bool,
    pub break_even: bool,
    pub trailing: bool,
    pub prop_firm_state: bool,
    pub device_filtering: bool,
    pub compact_readback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynchronizationMode {
    Blocking,
    ExplicitEvents,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineIdentity {
    pub session_id: u64,
    pub backend_id: u32,
    pub device_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetHandle(HandleToken);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneBufferHandle(HandleToken);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioBufferHandle(HandleToken);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMetricsHandle(HandleToken);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSelectionHandle(HandleToken);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceEventHandle(HandleToken);

macro_rules! impl_handle {
    ($name:ident, $kind:expr) => {
        impl $name {
            pub fn new(identity: EngineIdentity, generation: u64) -> Self {
                Self(HandleToken {
                    session_id: identity.session_id,
                    backend_id: identity.backend_id,
                    device_id: identity.device_id,
                    generation,
                    buffer_kind: $kind.id(),
                    reserved: 0,
                })
            }

            pub fn token(self) -> HandleToken {
                self.0
            }
        }
    };
}

impl_handle!(DatasetHandle, BufferKind::Dataset);
impl_handle!(GeneBufferHandle, BufferKind::Genes);
impl_handle!(ScenarioBufferHandle, BufferKind::Scenarios);
impl_handle!(DeviceMetricsHandle, BufferKind::Metrics);
impl_handle!(DeviceSelectionHandle, BufferKind::Selection);
impl_handle!(DeviceEventHandle, BufferKind::Event);

pub trait TypedDeviceHandle: Copy {
    const KIND: BufferKind;
    fn token(self) -> HandleToken;
}

macro_rules! impl_typed_handle {
    ($name:ident, $kind:expr) => {
        impl TypedDeviceHandle for $name {
            const KIND: BufferKind = $kind;
            fn token(self) -> HandleToken {
                self.0
            }
        }
    };
}

impl_typed_handle!(DatasetHandle, BufferKind::Dataset);
impl_typed_handle!(GeneBufferHandle, BufferKind::Genes);
impl_typed_handle!(ScenarioBufferHandle, BufferKind::Scenarios);
impl_typed_handle!(DeviceMetricsHandle, BufferKind::Metrics);
impl_typed_handle!(DeviceSelectionHandle, BufferKind::Selection);
impl_typed_handle!(DeviceEventHandle, BufferKind::Event);

#[derive(Debug, Clone)]
pub struct GpuDiscoverySession {
    identity: EngineIdentity,
    generation: Arc<AtomicU64>,
    transfers: Arc<TransferCounters>,
    synchronization: SynchronizationMode,
}

impl GpuDiscoverySession {
    pub fn new(identity: EngineIdentity, synchronization: SynchronizationMode) -> Self {
        Self {
            identity,
            generation: Arc::new(AtomicU64::new(1)),
            transfers: Arc::new(TransferCounters::default()),
            synchronization,
        }
    }

    pub fn identity(&self) -> EngineIdentity {
        self.identity
    }

    pub fn synchronization_mode(&self) -> SynchronizationMode {
        self.synchronization
    }

    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn reset_workspace(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn validate<H: TypedDeviceHandle>(&self, handle: H) -> Result<(), HandleValidationError> {
        let token = handle.token();
        if token.session_id != self.identity.session_id {
            return Err(HandleValidationError::WrongSession {
                expected: self.identity.session_id,
                actual: token.session_id,
            });
        }
        if token.backend_id != self.identity.backend_id {
            return Err(HandleValidationError::WrongBackend {
                expected: self.identity.backend_id,
                actual: token.backend_id,
            });
        }
        if token.device_id != self.identity.device_id {
            return Err(HandleValidationError::WrongDevice {
                expected: self.identity.device_id,
                actual: token.device_id,
            });
        }
        if token.generation != self.current_generation() {
            return Err(HandleValidationError::StaleGeneration {
                expected: self.current_generation(),
                actual: token.generation,
            });
        }
        if token.buffer_kind != H::KIND.id() {
            return Err(HandleValidationError::WrongBufferKind {
                expected: H::KIND,
                actual: token.buffer_kind,
            });
        }
        Ok(())
    }

    pub fn transfers(&self) -> &TransferCounters {
        &self.transfers
    }

    pub fn transfer_snapshot(&self) -> TransferSnapshot {
        self.transfers.snapshot()
    }
}

#[derive(Debug, Default)]
pub struct TransferCounters {
    dataset_uploads: AtomicU64,
    gene_uploads: AtomicU64,
    scenario_uploads: AtomicU64,
    full_d2h_readbacks: AtomicU64,
    compact_d2h_readbacks: AtomicU64,
    chained_reuploads: AtomicU64,
    synchronization_events: AtomicU64,
    h2d_bytes: AtomicU64,
    d2h_bytes: AtomicU64,
}

impl TransferCounters {
    pub fn record_dataset_upload(&self, bytes: u64) {
        self.dataset_uploads.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_gene_upload(&self, bytes: u64) {
        self.gene_uploads.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_scenario_upload(&self, bytes: u64) {
        self.scenario_uploads.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_full_readback(&self, bytes: u64) {
        self.full_d2h_readbacks.fetch_add(1, Ordering::Relaxed);
        self.d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_compact_readback(&self, bytes: u64) {
        self.compact_d2h_readbacks.fetch_add(1, Ordering::Relaxed);
        self.d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_chained_reupload(&self, bytes: u64) {
        self.chained_reuploads.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_synchronization_event(&self) {
        self.synchronization_events.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> TransferSnapshot {
        TransferSnapshot {
            dataset_uploads: self.dataset_uploads.load(Ordering::Relaxed),
            gene_uploads: self.gene_uploads.load(Ordering::Relaxed),
            scenario_uploads: self.scenario_uploads.load(Ordering::Relaxed),
            full_d2h_readbacks: self.full_d2h_readbacks.load(Ordering::Relaxed),
            compact_d2h_readbacks: self.compact_d2h_readbacks.load(Ordering::Relaxed),
            chained_reuploads: self.chained_reuploads.load(Ordering::Relaxed),
            synchronization_events: self.synchronization_events.load(Ordering::Relaxed),
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            d2h_bytes: self.d2h_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferSnapshot {
    pub dataset_uploads: u64,
    pub gene_uploads: u64,
    pub scenario_uploads: u64,
    pub full_d2h_readbacks: u64,
    pub compact_d2h_readbacks: u64,
    pub chained_reuploads: u64,
    pub synchronization_events: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

impl TransferSnapshot {
    pub fn assert_device_resident_chain(self) -> Result<(), TransferInvariantError> {
        if self.dataset_uploads != 1 {
            return Err(TransferInvariantError::DatasetUploadCount(
                self.dataset_uploads,
            ));
        }
        if self.full_d2h_readbacks != 0 {
            return Err(TransferInvariantError::FullReadbacks(
                self.full_d2h_readbacks,
            ));
        }
        if self.chained_reuploads != 0 {
            return Err(TransferInvariantError::ChainedReuploads(
                self.chained_reuploads,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSurvivorSummary {
    pub candidate_ids: Vec<u64>,
    pub rank_keys: Vec<Vec<i64>>,
}

pub trait BacktestEngine {
    fn status(&self) -> EngineStatus;
    fn capabilities(&self) -> EngineCapabilities;
    fn session(&self) -> &GpuDiscoverySession;

    fn upload_dataset(&mut self, bytes: &[u8]) -> Result<DatasetHandle, EngineError>;
    fn upload_genes(&mut self, bytes: &[u8]) -> Result<GeneBufferHandle, EngineError>;
    fn upload_scenarios(&mut self, bytes: &[u8]) -> Result<ScenarioBufferHandle, EngineError>;

    fn evaluate(
        &mut self,
        dataset: DatasetHandle,
        genes: GeneBufferHandle,
        scenarios: ScenarioBufferHandle,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<(DeviceMetricsHandle, Option<DeviceEventHandle>), EngineError>;

    fn filter(
        &mut self,
        metrics: DeviceMetricsHandle,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<(DeviceSelectionHandle, Option<DeviceEventHandle>), EngineError>;

    fn readback_compact(
        &mut self,
        selection: DeviceSelectionHandle,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<HostSurvivorSummary, EngineError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleValidationError {
    WrongSession { expected: u64, actual: u64 },
    WrongBackend { expected: u32, actual: u32 },
    WrongDevice { expected: u32, actual: u32 },
    StaleGeneration { expected: u64, actual: u64 },
    WrongBufferKind { expected: BufferKind, actual: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferInvariantError {
    DatasetUploadCount(u64),
    FullReadbacks(u64),
    ChainedReuploads(u64),
}

#[derive(Debug)]
pub enum EngineError {
    Handle(HandleValidationError),
    TransferInvariant(TransferInvariantError),
    Status(EngineStatus),
    Backend(String),
}

impl From<HandleValidationError> for EngineError {
    fn from(value: HandleValidationError) -> Self {
        Self::Handle(value)
    }
}

impl From<TransferInvariantError> for EngineError {
    fn from(value: TransferInvariantError) -> Self {
        Self::TransferInvariant(value)
    }
}

impl fmt::Display for HandleValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid device handle: {self:?}")
    }
}

impl fmt::Display for TransferInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "device-residency transfer invariant failed: {self:?}")
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handle(error) => error.fmt(f),
            Self::TransferInvariant(error) => error.fmt(f),
            Self::Status(status) => write!(f, "engine is not ready: {status:?}"),
            Self::Backend(message) => write!(f, "engine backend error: {message}"),
        }
    }
}

impl Error for HandleValidationError {}
impl Error for TransferInvariantError {}
impl Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(session_id: u64) -> EngineIdentity {
        EngineIdentity {
            session_id,
            backend_id: 7,
            device_id: 1,
        }
    }

    #[test]
    fn rejects_cross_session_and_stale_handles() {
        let first = GpuDiscoverySession::new(identity(1), SynchronizationMode::ExplicitEvents);
        let second = GpuDiscoverySession::new(identity(2), SynchronizationMode::ExplicitEvents);
        let handle = DatasetHandle::new(first.identity(), first.current_generation());
        assert!(matches!(
            second.validate(handle),
            Err(HandleValidationError::WrongSession { .. })
        ));
        first.reset_workspace();
        assert!(matches!(
            first.validate(handle),
            Err(HandleValidationError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn transfer_invariant_accepts_one_upload_and_compact_readback() {
        let session = GpuDiscoverySession::new(identity(3), SynchronizationMode::Blocking);
        session.transfers().record_dataset_upload(100);
        session.transfers().record_gene_upload(20);
        session.transfers().record_compact_readback(8);
        session
            .transfer_snapshot()
            .assert_device_resident_chain()
            .unwrap();
    }

    #[test]
    fn transfer_invariant_rejects_full_readback_and_reupload() {
        let session = GpuDiscoverySession::new(identity(4), SynchronizationMode::Blocking);
        session.transfers().record_dataset_upload(100);
        session.transfers().record_full_readback(80);
        assert!(matches!(
            session.transfer_snapshot().assert_device_resident_chain(),
            Err(TransferInvariantError::FullReadbacks(1))
        ));
    }
}
