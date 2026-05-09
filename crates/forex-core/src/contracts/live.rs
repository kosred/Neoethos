use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ArtifactContractError, ArtifactProvenance, BackendKind, require_live_ready_provenance,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveExecutionContract {
    pub feature_schema_hash: String,
    pub timestamp_policy_hash: String,
    pub feature_availability_policy_hash: String,
    pub symbol_universe_hash: String,
    pub runtime_config_hash: String,
    pub risk_config_hash: String,
    pub required_backend_kind: Option<BackendKind>,
    pub max_artifact_age_seconds: Option<i64>,
    /// When `true`, the live bridge rejects any evidence record whose
    /// `walkforward_passed` flag is `false`. Defaults to `false` so
    /// upgrading a contract instance never unintentionally tightens the
    /// gate.
    pub require_walkforward_pass: bool,
    /// When `true`, the live bridge rejects any evidence record whose
    /// `cpcv_passed` flag is `false`.
    pub require_cpcv_pass: bool,
    /// When `true`, the live bridge rejects any evidence record whose
    /// `forward_test_passed` flag is `Some(false)` (and rejects missing
    /// forward-test evidence as `LiveRejectedMissingEvidence`).
    pub require_forward_test_pass: bool,
    /// When `true`, the live bridge rejects any evidence record whose
    /// `prop_firm_passed` flag is `Some(false)` (and rejects missing
    /// prop-firm evidence as `LiveRejectedMissingEvidence`).
    pub require_prop_firm_pass: bool,
    /// Required `runtime_model_hash` for the live execution simulation
    /// that produced the evidence. When `Some`, mismatched / missing
    /// evidence rejects the load.
    pub required_live_sim_runtime_model_hash: Option<String>,
}

/// Validation evidence accompanying a live-execution acceptance check.
/// Each gate is split into "required → required-pass" pairs so callers
/// can opt out per-gate; the `LiveExecutionContract` decides which gates
/// are enforced.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveValidationEvidence {
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    /// `Some(true)`/`Some(false)` records a forward-test outcome;
    /// `None` means "no forward-test was run" and the live bridge will
    /// reject the artifact when forward-test evidence is required.
    pub forward_test_passed: Option<bool>,
    pub prop_firm_passed: Option<bool>,
    /// Optional hash of the [`live_execution_simulation`] runtime model
    /// the evidence was derived from. Mirrors the hash on
    /// `LiveExecutionContract::required_live_sim_runtime_model_hash`.
    pub live_sim_runtime_model_hash: Option<String>,
}

impl LiveValidationEvidence {
    pub fn passed_all() -> Self {
        Self {
            walkforward_passed: true,
            cpcv_passed: true,
            forward_test_passed: Some(true),
            prop_firm_passed: Some(true),
            live_sim_runtime_model_hash: None,
        }
    }
}

impl LiveExecutionContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_schema_hash: impl Into<String>,
        timestamp_policy_hash: impl Into<String>,
        feature_availability_policy_hash: impl Into<String>,
        symbol_universe_hash: impl Into<String>,
        runtime_config_hash: impl Into<String>,
        risk_config_hash: impl Into<String>,
    ) -> Self {
        Self {
            feature_schema_hash: feature_schema_hash.into(),
            timestamp_policy_hash: timestamp_policy_hash.into(),
            feature_availability_policy_hash: feature_availability_policy_hash.into(),
            symbol_universe_hash: symbol_universe_hash.into(),
            runtime_config_hash: runtime_config_hash.into(),
            risk_config_hash: risk_config_hash.into(),
            required_backend_kind: None,
            max_artifact_age_seconds: None,
            require_walkforward_pass: false,
            require_cpcv_pass: false,
            require_forward_test_pass: false,
            require_prop_firm_pass: false,
            required_live_sim_runtime_model_hash: None,
        }
    }

    pub fn with_required_backend(mut self, backend: BackendKind) -> Self {
        self.required_backend_kind = Some(backend);
        self
    }

    pub fn with_max_artifact_age_seconds(mut self, max_artifact_age_seconds: i64) -> Self {
        assert!(
            max_artifact_age_seconds >= 0,
            "live artifact max age must be non-negative"
        );
        self.max_artifact_age_seconds = Some(max_artifact_age_seconds);
        self
    }

    pub fn with_required_walkforward_pass(mut self) -> Self {
        self.require_walkforward_pass = true;
        self
    }

    pub fn with_required_cpcv_pass(mut self) -> Self {
        self.require_cpcv_pass = true;
        self
    }

    pub fn with_required_forward_test_pass(mut self) -> Self {
        self.require_forward_test_pass = true;
        self
    }

    pub fn with_required_prop_firm_pass(mut self) -> Self {
        self.require_prop_firm_pass = true;
        self
    }

    pub fn with_required_live_sim_runtime_model_hash(mut self, hash: impl Into<String>) -> Self {
        self.required_live_sim_runtime_model_hash = Some(hash.into());
        self
    }

    /// Reject the evidence record when any gate that the contract marks
    /// required is missing or did not pass. Production callers chain
    /// this after `validate_provenance` so live execution refuses
    /// artifacts that survived the provenance / temporal checks but lack
    /// validation evidence.
    pub fn validate_evidence(
        &self,
        evidence: &LiveValidationEvidence,
    ) -> Result<(), ArtifactContractError> {
        if self.require_walkforward_pass && !evidence.walkforward_passed {
            return Err(ArtifactContractError::LiveRejectedFailedEvidenceGate {
                gate: "walkforward",
            });
        }
        if self.require_cpcv_pass && !evidence.cpcv_passed {
            return Err(ArtifactContractError::LiveRejectedFailedEvidenceGate { gate: "cpcv" });
        }
        if self.require_forward_test_pass {
            match evidence.forward_test_passed {
                Some(true) => {}
                Some(false) => {
                    return Err(ArtifactContractError::LiveRejectedFailedEvidenceGate {
                        gate: "forward_test",
                    });
                }
                None => {
                    return Err(ArtifactContractError::LiveRejectedMissingEvidence {
                        gate: "forward_test",
                    });
                }
            }
        }
        if self.require_prop_firm_pass {
            match evidence.prop_firm_passed {
                Some(true) => {}
                Some(false) => {
                    return Err(ArtifactContractError::LiveRejectedFailedEvidenceGate {
                        gate: "prop_firm",
                    });
                }
                None => {
                    return Err(ArtifactContractError::LiveRejectedMissingEvidence {
                        gate: "prop_firm",
                    });
                }
            }
        }
        if let Some(expected) = &self.required_live_sim_runtime_model_hash {
            match &evidence.live_sim_runtime_model_hash {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(ArtifactContractError::LiveRejectedMismatch {
                        field: "live_sim_runtime_model_hash",
                        actual: actual.clone(),
                        expected: expected.clone(),
                    });
                }
                None => {
                    return Err(ArtifactContractError::LiveRejectedMissingEvidence {
                        gate: "live_sim_runtime_model",
                    });
                }
            }
        }
        Ok(())
    }

    pub fn validate_provenance(
        &self,
        provenance: &ArtifactProvenance,
    ) -> Result<(), ArtifactContractError> {
        self.validate_provenance_at(provenance, Utc::now())
    }

    pub fn validate_provenance_at(
        &self,
        provenance: &ArtifactProvenance,
        now: DateTime<Utc>,
    ) -> Result<(), ArtifactContractError> {
        require_live_ready_provenance(provenance)?;
        require_live_field_match(
            "feature_schema_hash",
            &provenance.feature_schema_hash,
            &self.feature_schema_hash,
        )?;
        require_live_field_match(
            "timestamp_policy_hash",
            &provenance.timestamp_policy_hash,
            &self.timestamp_policy_hash,
        )?;
        require_live_field_match(
            "feature_availability_policy_hash",
            &provenance.feature_availability_policy_hash,
            &self.feature_availability_policy_hash,
        )?;
        require_live_field_match(
            "symbol_universe_hash",
            &provenance.symbol_universe_hash,
            &self.symbol_universe_hash,
        )?;
        require_live_field_match(
            "runtime_config_hash",
            &provenance.runtime_config_hash,
            &self.runtime_config_hash,
        )?;
        require_live_field_match(
            "risk_config_hash",
            &provenance.risk_config_hash,
            &self.risk_config_hash,
        )?;

        if let Some(required_backend) = self.required_backend_kind {
            if provenance.backend_kind != required_backend {
                return Err(ArtifactContractError::LiveRejectedMismatch {
                    field: "backend_kind",
                    actual: format!("{:?}", provenance.backend_kind),
                    expected: format!("{:?}", required_backend),
                });
            }
        }

        if let Some(max_age) = self.max_artifact_age_seconds {
            let age_seconds = now
                .signed_duration_since(provenance.created_at)
                .num_seconds()
                .max(0);
            if age_seconds > max_age {
                return Err(ArtifactContractError::LiveRejectedStaleArtifact {
                    age_seconds,
                    max_age_seconds: max_age,
                });
            }
        }

        Ok(())
    }
}

fn require_live_field_match(
    field: &'static str,
    actual: &str,
    expected: &str,
) -> Result<(), ArtifactContractError> {
    if actual != expected {
        return Err(ArtifactContractError::LiveRejectedMismatch {
            field,
            actual: actual.to_string(),
            expected: expected.to_string(),
        });
    }
    Ok(())
}
