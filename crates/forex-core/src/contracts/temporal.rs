use serde::{Deserialize, Serialize};

use super::{
    ArtifactContractError, ArtifactProvenance, CandleTimestampPolicy, FeatureAvailabilityPolicy,
    MultiTimeframeAvailabilityPolicy, TimestampPolicy, TimestampUnit,
};

/// Canonical timestamp/feature availability boundary shared by training,
/// search, backtest, forward-test, and live execution.
///
/// The contract keeps the concrete timestamp and MTF/feature availability
/// policies together with the hashes that artifacts already persist in their
/// provenance. This prevents each subsystem from independently inferring
/// candle timestamp semantics, MTF availability, or embargo policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalFeatureContract {
    pub timestamp_policy: TimestampPolicy,
    pub feature_availability_policy: FeatureAvailabilityPolicy,
    pub label_policy_hash: String,
    pub walk_forward_policy_hash: String,
    pub live_readiness_policy_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalScopeHashes {
    pub temporal_contract_hash: String,
    pub timestamp_policy_hash: String,
    pub feature_availability_policy_hash: String,
    pub label_policy_hash: String,
}

impl TemporalScopeHashes {
    pub fn from_contract(contract: &TemporalFeatureContract) -> Self {
        Self {
            temporal_contract_hash: contract.temporal_contract_hash(),
            timestamp_policy_hash: contract.timestamp_policy_hash(),
            feature_availability_policy_hash: contract.feature_availability_policy_hash(),
            label_policy_hash: contract.label_policy_hash.clone(),
        }
    }

    pub fn validate_contract(
        &self,
        contract: &TemporalFeatureContract,
    ) -> Result<(), ArtifactContractError> {
        require_match(
            "temporal_contract_hash",
            &self.temporal_contract_hash,
            &contract.temporal_contract_hash(),
        )?;
        require_match(
            "timestamp_policy_hash",
            &self.timestamp_policy_hash,
            &contract.timestamp_policy_hash(),
        )?;
        require_match(
            "feature_availability_policy_hash",
            &self.feature_availability_policy_hash,
            &contract.feature_availability_policy_hash(),
        )?;
        require_match(
            "label_policy_hash",
            &self.label_policy_hash,
            &contract.label_policy_hash,
        )
    }
}

impl TemporalFeatureContract {
    pub fn new(
        timestamp_policy: TimestampPolicy,
        feature_availability_policy: FeatureAvailabilityPolicy,
        label_policy_hash: impl Into<String>,
        walk_forward_policy_hash: impl Into<String>,
        live_readiness_policy_hash: impl Into<String>,
    ) -> Result<Self, ArtifactContractError> {
        let contract = Self {
            timestamp_policy,
            feature_availability_policy,
            label_policy_hash: label_policy_hash.into(),
            walk_forward_policy_hash: walk_forward_policy_hash.into(),
            live_readiness_policy_hash: live_readiness_policy_hash.into(),
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Strict default for live-capable trading paths: millisecond timestamps,
    /// candle-open indexing, closed higher-timeframe bars only, and no lookahead.
    pub fn strict_live(
        timezone: impl Into<String>,
        alignment_policy_hash: impl Into<String>,
        label_policy_hash: impl Into<String>,
        walk_forward_policy_hash: impl Into<String>,
        live_readiness_policy_hash: impl Into<String>,
    ) -> Result<Self, ArtifactContractError> {
        Self::new(
            TimestampPolicy::new(
                TimestampUnit::Milliseconds,
                CandleTimestampPolicy::OpenTime,
                timezone,
            ),
            FeatureAvailabilityPolicy::strict_closed_mtf(alignment_policy_hash),
            label_policy_hash,
            walk_forward_policy_hash,
            live_readiness_policy_hash,
        )
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        require_non_empty("timestamp_policy.timezone", &self.timestamp_policy.timezone)?;
        require_non_empty(
            "feature_availability_policy.alignment_policy_hash",
            &self.feature_availability_policy.alignment_policy_hash,
        )?;
        require_non_empty("label_policy_hash", &self.label_policy_hash)?;
        require_non_empty("walk_forward_policy_hash", &self.walk_forward_policy_hash)?;
        require_non_empty(
            "live_readiness_policy_hash",
            &self.live_readiness_policy_hash,
        )?;

        if self.feature_availability_policy.allow_lookahead {
            return Err(ArtifactContractError::TemporalPolicyViolation {
                field: "feature_availability_policy.allow_lookahead",
                reason: "lookahead is forbidden for canonical trading artifacts".to_string(),
            });
        }
        if self.feature_availability_policy.multi_timeframe
            != MultiTimeframeAvailabilityPolicy::ClosedHigherTimeframeOnly
        {
            return Err(ArtifactContractError::TemporalPolicyViolation {
                field: "feature_availability_policy.multi_timeframe",
                reason: "higher-timeframe features must only become available after the higher candle closes".to_string(),
            });
        }
        Ok(())
    }

    pub fn timestamp_policy_hash(&self) -> String {
        stable_contract_hash(&self.timestamp_policy)
    }

    pub fn feature_availability_policy_hash(&self) -> String {
        stable_contract_hash(&self.feature_availability_policy)
    }

    pub fn temporal_contract_hash(&self) -> String {
        stable_contract_hash(self)
    }

    pub fn validate_provenance(
        &self,
        provenance: &ArtifactProvenance,
    ) -> Result<(), ArtifactContractError> {
        self.validate()?;
        require_match(
            "timestamp_policy_hash",
            &provenance.timestamp_policy_hash,
            &self.timestamp_policy_hash(),
        )?;
        require_match(
            "feature_availability_policy_hash",
            &provenance.feature_availability_policy_hash,
            &self.feature_availability_policy_hash(),
        )?;
        require_match(
            "label_policy_hash",
            &provenance.label_policy_hash,
            &self.label_policy_hash,
        )
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ArtifactContractError> {
    if value.trim().is_empty() {
        return Err(ArtifactContractError::TemporalPolicyViolation {
            field,
            reason: "value must not be empty".to_string(),
        });
    }
    Ok(())
}

fn require_match(
    field: &'static str,
    actual: &str,
    expected: &str,
) -> Result<(), ArtifactContractError> {
    if actual != expected {
        return Err(ArtifactContractError::TemporalPolicyMismatch {
            field,
            actual: actual.to_string(),
            expected: expected.to_string(),
        });
    }
    Ok(())
}

fn stable_contract_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("contract policy serialization must be stable");
    format!("fnv64:{:016x}", fnv1a64(&bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
