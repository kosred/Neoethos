pub mod broker_config;
pub mod broker_truth;
pub mod config;
pub mod contracts;
pub mod domain;
// F-150 fix (2026-05-25 — F-CORE3 consolidation): canonical env-var
// registry for neoethos-core. Phase A introduces typed getters; Phase
// B migrates the 6 existing call sites (config / symbol_metadata /
// system / logging / broker_config / resolved_config) to use them.
pub mod env_overrides;
pub mod execution;
pub mod logging;
pub mod resolved_config;
pub mod scheduler;
pub mod schema_version;
pub mod sectioned_log;
pub mod storage;
// The ONE definition of "this is the same trading rule" (#219, 2026-08-10).
// It lived in `neoethos-app`, above `neoethos-search` in the dependency graph,
// so the search could not consult the blacklist the live side writes.
pub mod strategy_identity;
pub mod symbol_metadata;
pub mod system;
pub mod utils;

/// Immutable broker-financial evidence contracts and the fail-closed release
/// gate. This is a dependency-leaf re-export; core does not own or install a
/// mutable process-global authority.
pub use neoethos_broker_truth as broker_financial_truth;
/// The single process-capacity and CPU-permit authority. Re-exporting the
/// zero-dependency leaf keeps root-engine consumers on the exact same types as
/// the isolated MCP and mesh workspaces without moving runtime policy into
/// this larger foundation crate.
pub use neoethos_execution_budget as execution_budget;

pub use broker_config::{
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, BrokerSettingsState,
    CTRADER_CREATE_DEMO_ACCOUNT_URL, CTRADER_CREATE_LIVE_ACCOUNT_URL, CTraderBrokerEnvironment,
    CTraderBrokerSettings, credentials_file_path,
    load_from_disk as load_broker_credentials_from_disk,
    save_to_disk as save_broker_credentials_to_disk,
};
pub use broker_truth::{
    BROKER_FINANCIAL_TRUTH_SCHEMA_VERSION_V1, BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1,
    BrokerFinancialOperationV1, BrokerFinancialTruthCapabilityV1, BrokerFinancialTruthErrorV1,
    BrokerFinancialTruthPermitV1, MissingBrokerFinancialEvidenceV1,
    current_broker_financial_truth_capability_v1,
};
pub use config::{NewsTradingMode, Settings, default_news_rss_feeds};
pub use contracts::{
    ARTIFACT_SCHEMA_VERSION, ArtifactContractError, ArtifactContractKind, ArtifactEnvelope,
    ArtifactKind, ArtifactProvenance, BackendKind, BarTimestampConvention, CANONICAL_TIMEFRAMES,
    CandleTimestampPolicy, CanonicalTimeframe, DeterminismPolicy, DeviceAssignment,
    FeatureAvailabilityPolicy, LiveExecutionContract, LivePromotionGate, LiveReadyStrategyArtifact,
    LiveReadyStrategyArtifactContract, LiveValidationEvidence, ModelRuntimeArtifact,
    ModelRuntimeArtifactContract, MultiTimeframeAvailabilityPolicy, PortfolioSelectionArtifact,
    PortfolioSelectionArtifactContract, PromotionReadinessCheck, PromotionReadinessCheckKind,
    PromotionReadinessReport, PromotionReadinessStatus, RuntimeDegradedReason, RuntimeMode,
    RuntimeSafetyIssue, RuntimeSafetyReport, SearchCheckpointArtifact,
    SearchCheckpointArtifactContract, TimestampPolicy, TimestampUnit, TrainingModelArtifact,
    TrainingModelArtifactContract, TypedArtifactEnvelope, ValidationEvidenceCheck,
    ValidationEvidenceKind, ValidationEvidenceManifest, canonical_higher_timeframes,
    is_canonical_timeframe,
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
    AcceleratorBackend, AcceleratorDevice, CpuBudget, CpuCapacityDiagnostics,
    ExecutionBudgetInputError, ExecutionBudgetInputs, GpuBudget, HardwareExecutionPlan,
    HardwareRuntimeOverrides, PrecisionPolicy, ResolvedWorkloadAssignment, TrainingPrecision,
    WorkloadDemand, WorkloadExecutionPlan, WorkloadKind, available_memory_bytes,
    total_memory_bytes,
};
