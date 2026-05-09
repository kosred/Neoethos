use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ArtifactContractError, ArtifactEnvelope, DeterminismPolicy, LiveExecutionContract,
    RuntimeSafetyReport,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationEvidenceKind {
    CanonicalBacktest,
    WalkForward,
    ForwardTest,
    LiveExecutionSimulation,
    PropFirmRisk,
}

impl ValidationEvidenceKind {
    pub const ALL: [Self; 5] = [
        Self::CanonicalBacktest,
        Self::WalkForward,
        Self::ForwardTest,
        Self::LiveExecutionSimulation,
        Self::PropFirmRisk,
    ];

    pub fn field_name(self) -> &'static str {
        match self {
            Self::CanonicalBacktest => "canonical_backtest_validation_hash",
            Self::WalkForward => "walkforward_validation_hash",
            Self::ForwardTest => "forward_test_validation_hash",
            Self::LiveExecutionSimulation => "live_execution_simulation_hash",
            Self::PropFirmRisk => "prop_firm_risk_validation_hash",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::CanonicalBacktest => "canonical backtest validation",
            Self::WalkForward => "walk-forward validation",
            Self::ForwardTest => "forward-test validation",
            Self::LiveExecutionSimulation => "live-execution simulation",
            Self::PropFirmRisk => "prop-firm risk validation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationEvidenceCheck {
    pub kind: ValidationEvidenceKind,
    pub hash: Option<String>,
    pub present: bool,
}

impl ValidationEvidenceCheck {
    fn from_manifest(manifest: &ValidationEvidenceManifest, kind: ValidationEvidenceKind) -> Self {
        let hash = manifest
            .hash_for(kind)
            .filter(|hash| !hash.trim().is_empty())
            .map(str::to_string);
        Self {
            kind,
            present: hash.is_some(),
            hash,
        }
    }

    fn missing(kind: ValidationEvidenceKind) -> Self {
        Self {
            kind,
            hash: None,
            present: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationEvidenceManifest {
    pub canonical_backtest_validation_hash: String,
    pub walkforward_validation_hash: String,
    pub forward_test_validation_hash: String,
    pub live_execution_simulation_hash: String,
    pub prop_firm_risk_validation_hash: String,
}

impl ValidationEvidenceManifest {
    pub fn new(
        canonical_backtest_validation_hash: impl Into<String>,
        walkforward_validation_hash: impl Into<String>,
        forward_test_validation_hash: impl Into<String>,
        live_execution_simulation_hash: impl Into<String>,
        prop_firm_risk_validation_hash: impl Into<String>,
    ) -> Result<Self, ArtifactContractError> {
        let manifest = Self {
            canonical_backtest_validation_hash: canonical_backtest_validation_hash.into(),
            walkforward_validation_hash: walkforward_validation_hash.into(),
            forward_test_validation_hash: forward_test_validation_hash.into(),
            live_execution_simulation_hash: live_execution_simulation_hash.into(),
            prop_firm_risk_validation_hash: prop_firm_risk_validation_hash.into(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        for kind in ValidationEvidenceKind::ALL {
            if self
                .hash_for(kind)
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(ArtifactContractError::MissingValidationEvidence(
                    kind.field_name(),
                ));
            }
        }
        Ok(())
    }

    pub fn hash_for(&self, kind: ValidationEvidenceKind) -> Option<&str> {
        let value = match kind {
            ValidationEvidenceKind::CanonicalBacktest => &self.canonical_backtest_validation_hash,
            ValidationEvidenceKind::WalkForward => &self.walkforward_validation_hash,
            ValidationEvidenceKind::ForwardTest => &self.forward_test_validation_hash,
            ValidationEvidenceKind::LiveExecutionSimulation => &self.live_execution_simulation_hash,
            ValidationEvidenceKind::PropFirmRisk => &self.prop_firm_risk_validation_hash,
        };
        Some(value.as_str()).filter(|value| !value.trim().is_empty())
    }

    pub fn missing_kinds(&self) -> Vec<ValidationEvidenceKind> {
        ValidationEvidenceKind::ALL
            .iter()
            .copied()
            .filter(|kind| self.hash_for(*kind).is_none())
            .collect()
    }

    pub fn evidence_checks(&self) -> Vec<ValidationEvidenceCheck> {
        ValidationEvidenceKind::ALL
            .iter()
            .copied()
            .map(|kind| ValidationEvidenceCheck::from_manifest(self, kind))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionReadinessCheckKind {
    ValidationEvidence,
    RuntimeSafety,
    LiveExecutionContract,
    DeterminismRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionReadinessStatus {
    Passed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionReadinessCheck {
    pub kind: PromotionReadinessCheckKind,
    pub status: PromotionReadinessStatus,
    pub reason: Option<String>,
}

impl PromotionReadinessCheck {
    fn passed(kind: PromotionReadinessCheckKind) -> Self {
        Self {
            kind,
            status: PromotionReadinessStatus::Passed,
            reason: None,
        }
    }

    fn failed(kind: PromotionReadinessCheckKind, reason: String) -> Self {
        Self {
            kind,
            status: PromotionReadinessStatus::Failed,
            reason: Some(reason),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LivePromotionGate {
    pub live_execution_contract: LiveExecutionContract,
    pub require_deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionReadinessReport {
    pub validation_evidence_complete: bool,
    pub runtime_safety_passed: bool,
    pub live_contract_passed: bool,
    pub determinism_requirement_passed: bool,
    pub ready: bool,
    pub rejection_reasons: Vec<String>,
    pub runtime_safety: RuntimeSafetyReport,
    pub evidence_checks: Vec<ValidationEvidenceCheck>,
    pub checks: Vec<PromotionReadinessCheck>,
}

impl LivePromotionGate {
    pub fn new(live_execution_contract: LiveExecutionContract) -> Self {
        Self {
            live_execution_contract,
            require_deterministic: true,
        }
    }

    pub fn require_deterministic(mut self, require_deterministic: bool) -> Self {
        self.require_deterministic = require_deterministic;
        self
    }

    pub fn validate<T>(
        &self,
        artifact: &ArtifactEnvelope<T>,
        evidence: &ValidationEvidenceManifest,
    ) -> Result<(), ArtifactContractError> {
        self.validate_at(artifact, evidence, Utc::now())
    }

    pub fn validate_at<T>(
        &self,
        artifact: &ArtifactEnvelope<T>,
        evidence: &ValidationEvidenceManifest,
        now: DateTime<Utc>,
    ) -> Result<(), ArtifactContractError> {
        evidence.validate()?;
        self.live_execution_contract
            .validate_provenance_at(&artifact.provenance, now)?;
        if self.require_deterministic
            && !matches!(
                artifact.provenance.determinism_policy,
                DeterminismPolicy::Deterministic { .. }
            )
        {
            return Err(ArtifactContractError::PromotionRejectedDeterminism {
                actual: artifact.provenance.determinism_policy.clone(),
            });
        }
        Ok(())
    }

    pub fn readiness_report<T>(
        &self,
        artifact: &ArtifactEnvelope<T>,
        evidence: Option<&ValidationEvidenceManifest>,
    ) -> PromotionReadinessReport {
        self.readiness_report_at(artifact, evidence, Utc::now())
    }

    pub fn readiness_report_at<T>(
        &self,
        artifact: &ArtifactEnvelope<T>,
        evidence: Option<&ValidationEvidenceManifest>,
        now: DateTime<Utc>,
    ) -> PromotionReadinessReport {
        let mut rejection_reasons = Vec::new();
        let mut checks = Vec::new();
        let runtime_safety = artifact.provenance.runtime_safety_report();
        let runtime_safety_passed = runtime_safety.live_safe;

        let validation_evidence_complete = match evidence {
            Some(evidence) => match evidence.validate() {
                Ok(()) => {
                    checks.push(PromotionReadinessCheck::passed(
                        PromotionReadinessCheckKind::ValidationEvidence,
                    ));
                    true
                }
                Err(err) => {
                    let reason = err.to_string();
                    rejection_reasons.push(reason.clone());
                    checks.push(PromotionReadinessCheck::failed(
                        PromotionReadinessCheckKind::ValidationEvidence,
                        reason,
                    ));
                    false
                }
            },
            None => {
                let reason = "validation evidence manifest is missing".to_string();
                rejection_reasons.push(reason.clone());
                checks.push(PromotionReadinessCheck::failed(
                    PromotionReadinessCheckKind::ValidationEvidence,
                    reason,
                ));
                false
            }
        };
        let evidence_checks = evidence.map_or_else(
            || {
                ValidationEvidenceKind::ALL
                    .iter()
                    .copied()
                    .map(ValidationEvidenceCheck::missing)
                    .collect()
            },
            ValidationEvidenceManifest::evidence_checks,
        );

        if runtime_safety_passed {
            checks.push(PromotionReadinessCheck::passed(
                PromotionReadinessCheckKind::RuntimeSafety,
            ));
        } else {
            let reason = runtime_safety.rejection_reason().unwrap_or_else(|| {
                "runtime safety report rejected the artifact for live promotion".to_string()
            });
            rejection_reasons.push(reason.clone());
            checks.push(PromotionReadinessCheck::failed(
                PromotionReadinessCheckKind::RuntimeSafety,
                reason,
            ));
        }

        let live_contract_passed = match self
            .live_execution_contract
            .validate_provenance_at(&artifact.provenance, now)
        {
            Ok(()) => {
                checks.push(PromotionReadinessCheck::passed(
                    PromotionReadinessCheckKind::LiveExecutionContract,
                ));
                true
            }
            Err(err) => {
                let reason = err.to_string();
                rejection_reasons.push(reason.clone());
                checks.push(PromotionReadinessCheck::failed(
                    PromotionReadinessCheckKind::LiveExecutionContract,
                    reason,
                ));
                false
            }
        };

        let determinism_requirement_passed = !self.require_deterministic
            || matches!(
                artifact.provenance.determinism_policy,
                DeterminismPolicy::Deterministic { .. }
            );
        if !determinism_requirement_passed {
            let reason = ArtifactContractError::PromotionRejectedDeterminism {
                actual: artifact.provenance.determinism_policy.clone(),
            }
            .to_string();
            rejection_reasons.push(reason.clone());
            checks.push(PromotionReadinessCheck::failed(
                PromotionReadinessCheckKind::DeterminismRequirement,
                reason,
            ));
        } else {
            checks.push(PromotionReadinessCheck::passed(
                PromotionReadinessCheckKind::DeterminismRequirement,
            ));
        }

        PromotionReadinessReport {
            validation_evidence_complete,
            runtime_safety_passed,
            live_contract_passed,
            determinism_requirement_passed,
            ready: validation_evidence_complete
                && runtime_safety_passed
                && live_contract_passed
                && determinism_requirement_passed,
            rejection_reasons,
            runtime_safety,
            evidence_checks,
            checks,
        }
    }
}
