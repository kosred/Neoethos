use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{ARTIFACT_SCHEMA_VERSION, ArtifactContractError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    TrainingModel,
    SearchCheckpoint,
    PortfolioSelection,
    ModelRuntime,
    LiveReadyStrategy,
    RuntimeDiagnostic,
}

impl ArtifactKind {
    pub fn is_live_eligible(self) -> bool {
        matches!(self, Self::LiveReadyStrategy | Self::PortfolioSelection)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandleTimestampPolicy {
    OpenTime,
    CloseTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiTimeframeAvailabilityPolicy {
    ClosedHigherTimeframeOnly,
    CurrentPartialAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimestampPolicy {
    pub unit: TimestampUnit,
    pub candle_timestamp: CandleTimestampPolicy,
    pub timezone: String,
}

impl TimestampPolicy {
    pub fn new(
        unit: TimestampUnit,
        candle_timestamp: CandleTimestampPolicy,
        timezone: impl Into<String>,
    ) -> Self {
        let timezone = timezone.into();
        assert!(
            !timezone.trim().is_empty(),
            "timestamp policy requires a non-empty timezone"
        );
        Self {
            unit,
            candle_timestamp,
            timezone,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureAvailabilityPolicy {
    pub multi_timeframe: MultiTimeframeAvailabilityPolicy,
    pub embargo_bars: u32,
    pub allow_lookahead: bool,
    pub alignment_policy_hash: String,
}

impl FeatureAvailabilityPolicy {
    pub fn strict_closed_mtf(alignment_policy_hash: impl Into<String>) -> Self {
        let alignment_policy_hash = alignment_policy_hash.into();
        assert!(
            !alignment_policy_hash.trim().is_empty(),
            "feature availability policy requires an alignment policy hash"
        );
        Self {
            multi_timeframe: MultiTimeframeAvailabilityPolicy::ClosedHigherTimeframeOnly,
            embargo_bars: 0,
            allow_lookahead: false,
            alignment_policy_hash,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum DeterminismPolicy {
    Deterministic { seed: u64 },
    BestEffort,
    NonDeterministicAllowed,
}

impl DeterminismPolicy {
    pub fn seed(&self) -> Option<u64> {
        match self {
            Self::Deterministic { seed } => Some(*seed),
            Self::BestEffort | Self::NonDeterministicAllowed => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    NativeCpu,
    NativeCuda,
    CudaKernel,
    BurnWgpu,
    BurnCpu,
    NativeTreeGpu,
    NativeTreeCpu,
    CpuReference,
    LocalSurrogateFallback,
    ExternalRuntime,
    Unavailable,
}

impl BackendKind {
    pub fn is_degraded(self) -> bool {
        matches!(
            self,
            Self::CpuReference | Self::LocalSurrogateFallback | Self::Unavailable
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Canonical,
    Approximate,
    Fallback,
    Degraded,
    DiagnosticOnly,
}

impl RuntimeMode {
    pub fn is_live_safe(self) -> bool {
        matches!(self, Self::Canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDegradedReason {
    pub code: String,
    pub message: String,
}

impl RuntimeDegradedReason {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        let code = code.into();
        let message = message.into();
        assert!(
            !code.trim().is_empty(),
            "runtime degraded reason requires a non-empty code"
        );
        assert!(
            !message.trim().is_empty(),
            "runtime degraded reason requires a non-empty message"
        );
        Self { code, message }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSafetyIssue {
    NonCanonicalRuntimeMode,
    DegradedBackend,
    BackendAssignmentMismatch,
    MissingDegradedReason,
    CanonicalArtifactHasDegradedReason,
}

impl RuntimeSafetyIssue {
    pub fn label(self) -> &'static str {
        match self {
            Self::NonCanonicalRuntimeMode => "runtime mode is not canonical",
            Self::DegradedBackend => "backend is degraded or unavailable",
            Self::BackendAssignmentMismatch => "backend does not match device assignment",
            Self::MissingDegradedReason => "degraded runtime reason is missing",
            Self::CanonicalArtifactHasDegradedReason => {
                "canonical runtime carries a degraded reason"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSafetyReport {
    pub runtime_mode: RuntimeMode,
    pub backend_kind: BackendKind,
    pub assignment_backend: BackendKind,
    pub live_safe: bool,
    pub backend_assignment_matches: bool,
    pub degraded_reason: Option<RuntimeDegradedReason>,
    pub issues: Vec<RuntimeSafetyIssue>,
}

impl RuntimeSafetyReport {
    pub fn has_issue(&self, issue: RuntimeSafetyIssue) -> bool {
        self.issues.contains(&issue)
    }

    pub fn issue_labels(&self) -> Vec<&'static str> {
        self.issues.iter().map(|issue| issue.label()).collect()
    }

    pub fn rejection_reason(&self) -> Option<String> {
        (!self.live_safe).then(|| self.issue_labels().join("; "))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceAssignment {
    pub backend: BackendKind,
    pub device: String,
    pub device_ids: Vec<usize>,
}

impl DeviceAssignment {
    pub fn cpu() -> Self {
        Self {
            backend: BackendKind::NativeCpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactProvenance {
    pub artifact_kind: ArtifactKind,
    pub artifact_schema_version: u16,
    pub feature_schema_hash: String,
    pub dataset_fingerprint: String,
    pub symbol_universe_hash: String,
    pub timeframe_set_hash: String,
    pub timestamp_policy_hash: String,
    pub feature_availability_policy_hash: String,
    pub label_policy_hash: String,
    pub training_config_hash: String,
    pub search_config_hash: String,
    pub runtime_config_hash: String,
    pub risk_config_hash: String,
    pub determinism_policy: DeterminismPolicy,
    pub hardware_profile_id: String,
    pub device_assignment: DeviceAssignment,
    pub backend_kind: BackendKind,
    pub runtime_mode: RuntimeMode,
    pub runtime_degraded_reason: Option<RuntimeDegradedReason>,
    pub created_at: DateTime<Utc>,
    pub source_commit: String,
}

impl ArtifactProvenance {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        artifact_kind: ArtifactKind,
        feature_schema_hash: impl Into<String>,
        dataset_fingerprint: impl Into<String>,
        symbol_universe_hash: impl Into<String>,
        timeframe_set_hash: impl Into<String>,
        timestamp_policy_hash: impl Into<String>,
        feature_availability_policy_hash: impl Into<String>,
        label_policy_hash: impl Into<String>,
        training_config_hash: impl Into<String>,
        search_config_hash: impl Into<String>,
        runtime_config_hash: impl Into<String>,
        risk_config_hash: impl Into<String>,
        determinism_policy: DeterminismPolicy,
        hardware_profile_id: impl Into<String>,
        device_assignment: DeviceAssignment,
        backend_kind: BackendKind,
        runtime_mode: RuntimeMode,
        runtime_degraded_reason: Option<RuntimeDegradedReason>,
        source_commit: impl Into<String>,
    ) -> Result<Self, ArtifactContractError> {
        let provenance = Self {
            artifact_kind,
            artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash: feature_schema_hash.into(),
            dataset_fingerprint: dataset_fingerprint.into(),
            symbol_universe_hash: symbol_universe_hash.into(),
            timeframe_set_hash: timeframe_set_hash.into(),
            timestamp_policy_hash: timestamp_policy_hash.into(),
            feature_availability_policy_hash: feature_availability_policy_hash.into(),
            label_policy_hash: label_policy_hash.into(),
            training_config_hash: training_config_hash.into(),
            search_config_hash: search_config_hash.into(),
            runtime_config_hash: runtime_config_hash.into(),
            risk_config_hash: risk_config_hash.into(),
            determinism_policy,
            hardware_profile_id: hardware_profile_id.into(),
            device_assignment,
            backend_kind,
            runtime_mode,
            runtime_degraded_reason,
            created_at: Utc::now(),
            source_commit: source_commit.into(),
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub fn runtime_safety_report(&self) -> RuntimeSafetyReport {
        let backend_assignment_matches = self.backend_kind == self.device_assignment.backend;
        let mut issues = Vec::new();
        if !self.runtime_mode.is_live_safe() {
            issues.push(RuntimeSafetyIssue::NonCanonicalRuntimeMode);
        }
        if self.backend_kind.is_degraded() {
            issues.push(RuntimeSafetyIssue::DegradedBackend);
        }
        if !backend_assignment_matches {
            issues.push(RuntimeSafetyIssue::BackendAssignmentMismatch);
        }
        if self.runtime_mode == RuntimeMode::Canonical && self.runtime_degraded_reason.is_some() {
            issues.push(RuntimeSafetyIssue::CanonicalArtifactHasDegradedReason);
        }
        if (!self.runtime_mode.is_live_safe() || self.backend_kind.is_degraded())
            && self.runtime_degraded_reason.is_none()
        {
            issues.push(RuntimeSafetyIssue::MissingDegradedReason);
        }

        RuntimeSafetyReport {
            runtime_mode: self.runtime_mode,
            backend_kind: self.backend_kind,
            assignment_backend: self.device_assignment.backend,
            live_safe: issues.is_empty(),
            backend_assignment_matches,
            degraded_reason: self.runtime_degraded_reason.clone(),
            issues,
        }
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        if self.artifact_schema_version != ARTIFACT_SCHEMA_VERSION {
            return Err(ArtifactContractError::UnsupportedSchemaVersion {
                actual: self.artifact_schema_version,
                expected: ARTIFACT_SCHEMA_VERSION,
            });
        }

        let required_fields = [
            ("feature_schema_hash", &self.feature_schema_hash),
            ("dataset_fingerprint", &self.dataset_fingerprint),
            ("symbol_universe_hash", &self.symbol_universe_hash),
            ("timeframe_set_hash", &self.timeframe_set_hash),
            ("timestamp_policy_hash", &self.timestamp_policy_hash),
            (
                "feature_availability_policy_hash",
                &self.feature_availability_policy_hash,
            ),
            ("label_policy_hash", &self.label_policy_hash),
            ("training_config_hash", &self.training_config_hash),
            ("search_config_hash", &self.search_config_hash),
            ("runtime_config_hash", &self.runtime_config_hash),
            ("risk_config_hash", &self.risk_config_hash),
            ("hardware_profile_id", &self.hardware_profile_id),
            ("device_assignment.device", &self.device_assignment.device),
            ("source_commit", &self.source_commit),
        ];
        for (field, value) in required_fields {
            if value.trim().is_empty() {
                return Err(ArtifactContractError::MissingProvenanceField(field));
            }
        }

        if self.backend_kind != self.device_assignment.backend {
            return Err(ArtifactContractError::BackendAssignmentMismatch {
                backend_kind: self.backend_kind,
                assignment_backend: self.device_assignment.backend,
            });
        }

        if self.runtime_mode == RuntimeMode::Canonical && self.runtime_degraded_reason.is_some() {
            return Err(ArtifactContractError::CanonicalArtifactHasDegradedReason);
        }
        if !self.runtime_mode.is_live_safe() && self.runtime_degraded_reason.is_none() {
            return Err(ArtifactContractError::MissingDegradedReason);
        }
        Ok(())
    }
}
