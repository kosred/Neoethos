use serde::{Deserialize, Serialize};

use super::{ArtifactContractError, ArtifactEnvelope, DeterminismPolicy, LiveExecutionContract};

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
        let required = [
            (
                "canonical_backtest_validation_hash",
                &self.canonical_backtest_validation_hash,
            ),
            (
                "walkforward_validation_hash",
                &self.walkforward_validation_hash,
            ),
            (
                "forward_test_validation_hash",
                &self.forward_test_validation_hash,
            ),
            (
                "live_execution_simulation_hash",
                &self.live_execution_simulation_hash,
            ),
            (
                "prop_firm_risk_validation_hash",
                &self.prop_firm_risk_validation_hash,
            ),
        ];
        for (field, value) in required {
            if value.trim().is_empty() {
                return Err(ArtifactContractError::MissingValidationEvidence(field));
            }
        }
        Ok(())
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
    pub live_contract_passed: bool,
    pub determinism_requirement_passed: bool,
    pub ready: bool,
    pub rejection_reasons: Vec<String>,
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
        evidence.validate()?;
        self.live_execution_contract
            .validate_provenance(&artifact.provenance)?;
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
        let mut rejection_reasons = Vec::new();

        let validation_evidence_complete = match evidence {
            Some(evidence) => match evidence.validate() {
                Ok(()) => true,
                Err(err) => {
                    rejection_reasons.push(err.to_string());
                    false
                }
            },
            None => {
                rejection_reasons.push("validation evidence manifest is missing".to_string());
                false
            }
        };

        let live_contract_passed = match self
            .live_execution_contract
            .validate_provenance(&artifact.provenance)
        {
            Ok(()) => true,
            Err(err) => {
                rejection_reasons.push(err.to_string());
                false
            }
        };

        let determinism_requirement_passed = !self.require_deterministic
            || matches!(
                artifact.provenance.determinism_policy,
                DeterminismPolicy::Deterministic { .. }
            );
        if !determinism_requirement_passed {
            rejection_reasons.push(
                ArtifactContractError::PromotionRejectedDeterminism {
                    actual: artifact.provenance.determinism_policy.clone(),
                }
                .to_string(),
            );
        }

        PromotionReadinessReport {
            validation_evidence_complete,
            live_contract_passed,
            determinism_requirement_passed,
            ready: validation_evidence_complete
                && live_contract_passed
                && determinism_requirement_passed,
            rejection_reasons,
        }
    }
}
