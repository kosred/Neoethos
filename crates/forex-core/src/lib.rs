pub mod config;
pub mod contracts;
pub mod domain;
pub mod logging;
pub mod resolved_config;
pub mod sectioned_log;
pub mod storage;
pub mod symbol_metadata;
pub mod system;
pub mod utils;

pub use config::Settings;
pub use contracts::{
    ARTIFACT_SCHEMA_VERSION, ArtifactContractError, ArtifactContractKind, ArtifactEnvelope,
    ArtifactKind, ArtifactProvenance, BackendKind, CANONICAL_TIMEFRAMES, CandleTimestampPolicy,
    DeterminismPolicy, DeviceAssignment, FeatureAvailabilityPolicy, LiveExecutionContract,
    LivePromotionGate, LiveReadyStrategyArtifact, LiveReadyStrategyArtifactContract,
    LiveValidationEvidence, ModelRuntimeArtifact, ModelRuntimeArtifactContract,
    MultiTimeframeAvailabilityPolicy, PortfolioSelectionArtifact,
    PortfolioSelectionArtifactContract, PromotionReadinessCheck, PromotionReadinessCheckKind,
    PromotionReadinessReport, PromotionReadinessStatus, RuntimeDegradedReason, RuntimeMode,
    RuntimeSafetyIssue, RuntimeSafetyReport, SearchCheckpointArtifact,
    SearchCheckpointArtifactContract, TimestampPolicy, TimestampUnit, TrainingModelArtifact,
    TrainingModelArtifactContract, TypedArtifactEnvelope, ValidationEvidenceCheck,
    ValidationEvidenceKind, ValidationEvidenceManifest, is_canonical_timeframe,
};
pub use domain::PropFirmConstraints;
pub use system::{
    AcceleratorBackend, AcceleratorDevice, CpuBudget, GpuBudget, HardwareExecutionPlan,
    HardwareRuntimeOverrides, PrecisionPolicy, ResolvedWorkloadAssignment, TrainingPrecision,
    WorkloadExecutionPlan, WorkloadKind,
};
