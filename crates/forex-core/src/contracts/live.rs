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
