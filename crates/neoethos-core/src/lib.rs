pub mod broker_config;
pub mod config;
pub mod contracts;
pub mod domain;
// F-150 fix (2026-05-25 — F-CORE3 consolidation): canonical env-var
// registry for neoethos-core. Phase A introduces typed getters; Phase
// B migrates the 6 existing call sites (config / symbol_metadata /
// system / logging / broker_config / resolved_config) to use them.
pub mod env_overrides;
pub mod logging;
pub mod resolved_config;
pub mod schema_version;
pub mod sectioned_log;
pub mod storage;
pub mod symbol_metadata;
pub mod system;
pub mod utils;

pub use broker_config::{
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, BrokerSettingsState,
    CTRADER_CREATE_DEMO_ACCOUNT_URL, CTRADER_CREATE_LIVE_ACCOUNT_URL, CTraderBrokerEnvironment,
    CTraderBrokerSettings, DxTradeBrokerSettings, credentials_file_path,
    load_from_disk as load_broker_credentials_from_disk,
    save_to_disk as save_broker_credentials_to_disk,
};
pub use config::{NewsTradingMode, Settings};
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
pub use domain::{
    DEFAULT_RISKY_TRADES_PER_DAY, KillSwitchTier, MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY,
    RiskyModeConfig, RiskyModeManager, RiskyStage, build_logarithmic_stages,
};
pub use schema_version::{
    HasSchemaVersion, SchemaVersion, SchemaVersionError, check_schema_version_readable, default_v1,
    ensure_schema_version_readable,
};
pub use system::{
    AcceleratorBackend, AcceleratorDevice, CpuBudget, GpuBudget, HardwareExecutionPlan,
    HardwareRuntimeOverrides, PrecisionPolicy, ResolvedWorkloadAssignment, TrainingPrecision,
    WorkloadExecutionPlan, WorkloadKind,
};
