//! Immutable acquisition authority below the semantic broker-truth gate.
//!
//! This module freezes and hash-links the non-secret inputs used to acquire a
//! V2 broker bundle. It deliberately performs no signature/policy validation
//! and exposes no conversion to a financial capability or permit.

use std::collections::HashSet;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::contracts::{
    BrokerFinancialTruthContractErrorCodeV1, BrokerFinancialTruthContractErrorV1, EvidenceWindowV1,
    max_manifest_bytes, validate_sha256_hex,
};
use crate::contracts_v2::ReviewedQuoteReplayRuleIdentityV2;

pub const BROKER_TRUTH_ACQUISITION_AUTHORITY_SCHEMA_VERSION_V1: u16 = 1;

const ACQUISITION_AUTHORITY_HASH_DOMAIN_V1: &[u8] =
    b"neoethos.broker-truth-acquisition-authority.v1\0";
const MAX_ACQUISITION_ARTIFACT_NAME_BYTES_V1: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerTruthAcquisitionSemanticStatusV1 {
    UnvalidatedEvidenceOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerTruthAcquisitionPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrokerTruthAcquisitionArtifactRoleV1 {
    CanonicalSearchInputReceipt,
    CanonicalSearchArtifactScope,
    CanonicalRootVerificationReceipt,
    CanonicalScopeWindowBinding,
    CapturePlan,
    ReviewRecord,
    ProtocolEvidence,
    TrustRoot,
    QuoteSessionObservations { ordinal: u32 },
    ReviewedQuoteReplayRules { ordinal: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthAcquisitionArtifactV1 {
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: String,
    sha256: String,
    byte_len: u64,
}

impl BrokerTruthAcquisitionArtifactV1 {
    pub fn new(
        role: BrokerTruthAcquisitionArtifactRoleV1,
        relative_path: impl Into<String>,
        sha256: impl Into<String>,
        byte_len: u64,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let artifact = Self {
            role,
            relative_path: relative_path.into(),
            sha256: sha256.into(),
            byte_len,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn role(&self) -> BrokerTruthAcquisitionArtifactRoleV1 {
        self.role
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_acquisition_artifact_basename(&self.relative_path)?;
        validate_sha256_hex("acquisition artifact SHA-256", &self.sha256)?;
        if self.byte_len == 0 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                format!(
                    "acquisition artifact {} must contain at least one byte",
                    self.relative_path
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthReviewedSynchronizationBindingV1 {
    ordinal: u32,
    account_id: i64,
    symbol_id: i64,
    window: EvidenceWindowV1,
    review_identity: ReviewedQuoteReplayRuleIdentityV2,
    reviewed_rules_sha256: String,
}

impl BrokerTruthReviewedSynchronizationBindingV1 {
    pub fn new(
        ordinal: u32,
        account_id: i64,
        symbol_id: i64,
        window: EvidenceWindowV1,
        review_identity: ReviewedQuoteReplayRuleIdentityV2,
        reviewed_rules_sha256: impl Into<String>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let binding = Self {
            ordinal,
            account_id,
            symbol_id,
            window,
            review_identity,
            reviewed_rules_sha256: reviewed_rules_sha256.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub const fn window(&self) -> EvidenceWindowV1 {
        self.window
    }

    pub const fn review_identity(&self) -> &ReviewedQuoteReplayRuleIdentityV2 {
        &self.review_identity
    }

    pub fn reviewed_rules_sha256(&self) -> &str {
        &self.reviewed_rules_sha256
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.account_id <= 0 || self.symbol_id <= 0 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                "reviewed synchronization requires positive account and symbol ids",
            ));
        }
        self.window.validate()?;
        self.review_identity.validate_exact()?;
        validate_sha256_hex(
            "reviewed quote replay rules SHA-256",
            &self.reviewed_rules_sha256,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthAcquisitionAuthorityManifestV1 {
    schema_version: u16,
    semantic_status: BrokerTruthAcquisitionSemanticStatusV1,
    promotion_eligibility: BrokerTruthAcquisitionPromotionEligibilityV1,
    canonical_search_input_receipt_sha256: String,
    canonical_search_artifact_scope_sha256: String,
    canonical_root_verification_sha256: String,
    canonical_scope_window_binding_sha256: String,
    capture_plan_sha256: String,
    expected_trust_root_sha256: String,
    artifacts: Vec<BrokerTruthAcquisitionArtifactV1>,
    reviewed_synchronizations: Vec<BrokerTruthReviewedSynchronizationBindingV1>,
}

impl BrokerTruthAcquisitionAuthorityManifestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        canonical_search_input_receipt_sha256: impl Into<String>,
        canonical_search_artifact_scope_sha256: impl Into<String>,
        canonical_root_verification_sha256: impl Into<String>,
        canonical_scope_window_binding_sha256: impl Into<String>,
        capture_plan_sha256: impl Into<String>,
        expected_trust_root_sha256: impl Into<String>,
        artifacts: Vec<BrokerTruthAcquisitionArtifactV1>,
        reviewed_synchronizations: Vec<BrokerTruthReviewedSynchronizationBindingV1>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let manifest = Self {
            schema_version: BROKER_TRUTH_ACQUISITION_AUTHORITY_SCHEMA_VERSION_V1,
            semantic_status: BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly,
            promotion_eligibility:
                BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible,
            canonical_search_input_receipt_sha256: canonical_search_input_receipt_sha256.into(),
            canonical_search_artifact_scope_sha256: canonical_search_artifact_scope_sha256.into(),
            canonical_root_verification_sha256: canonical_root_verification_sha256.into(),
            canonical_scope_window_binding_sha256: canonical_scope_window_binding_sha256.into(),
            capture_plan_sha256: capture_plan_sha256.into(),
            expected_trust_root_sha256: expected_trust_root_sha256.into(),
            artifacts,
            reviewed_synchronizations,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn semantic_status(&self) -> BrokerTruthAcquisitionSemanticStatusV1 {
        self.semantic_status
    }

    pub const fn promotion_eligibility(&self) -> BrokerTruthAcquisitionPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub fn canonical_search_artifact_scope_sha256(&self) -> &str {
        &self.canonical_search_artifact_scope_sha256
    }

    pub fn canonical_root_verification_sha256(&self) -> &str {
        &self.canonical_root_verification_sha256
    }

    pub fn canonical_scope_window_binding_sha256(&self) -> &str {
        &self.canonical_scope_window_binding_sha256
    }

    pub fn capture_plan_sha256(&self) -> &str {
        &self.capture_plan_sha256
    }

    pub fn expected_trust_root_sha256(&self) -> &str {
        &self.expected_trust_root_sha256
    }

    pub fn artifacts(&self) -> &[BrokerTruthAcquisitionArtifactV1] {
        &self.artifacts
    }

    pub fn reviewed_synchronizations(&self) -> &[BrokerTruthReviewedSynchronizationBindingV1] {
        &self.reviewed_synchronizations
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot encode acquisition authority manifest: {error}"),
            )
        })
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        if bytes.len() as u64 > max_manifest_bytes() {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                "acquisition authority manifest exceeds the size limit",
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot decode acquisition authority manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn identity_sha256(&self) -> Result<String, BrokerFinancialTruthContractErrorV1> {
        let canonical = self.canonical_json_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(ACQUISITION_AUTHORITY_HASH_DOMAIN_V1);
        hasher.update(canonical);
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.schema_version != BROKER_TRUTH_ACQUISITION_AUTHORITY_SCHEMA_VERSION_V1 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::UnsupportedSchemaVersion,
                format!(
                    "unsupported acquisition authority schema version {}; expected {}",
                    self.schema_version, BROKER_TRUTH_ACQUISITION_AUTHORITY_SCHEMA_VERSION_V1
                ),
            ));
        }
        if self.semantic_status != BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
            || self.promotion_eligibility
                != BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                "acquisition authority must remain unvalidated and not promotion eligible",
            ));
        }
        for (label, digest) in [
            (
                "canonical search input receipt identity",
                self.canonical_search_input_receipt_sha256.as_str(),
            ),
            (
                "canonical search artifact scope identity",
                self.canonical_search_artifact_scope_sha256.as_str(),
            ),
            (
                "canonical root verification receipt",
                self.canonical_root_verification_sha256.as_str(),
            ),
            (
                "canonical scope window binding",
                self.canonical_scope_window_binding_sha256.as_str(),
            ),
            ("capture plan", self.capture_plan_sha256.as_str()),
            (
                "expected trust root",
                self.expected_trust_root_sha256.as_str(),
            ),
        ] {
            validate_sha256_hex(label, digest)?;
        }
        if self.reviewed_synchronizations.is_empty() {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "acquisition authority has no reviewed quote synchronization",
            ));
        }
        for (index, synchronization) in self.reviewed_synchronizations.iter().enumerate() {
            synchronization.validate()?;
            let expected_ordinal = u32::try_from(index).map_err(|_| {
                contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                    "reviewed synchronization count exceeds u32",
                )
            })?;
            if synchronization.ordinal != expected_ordinal {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                    "reviewed synchronization ordinals must start at zero and be contiguous",
                ));
            }
        }

        let mut expected_roles = vec![
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt,
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchArtifactScope,
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
        ];
        for synchronization in &self.reviewed_synchronizations {
            expected_roles.extend([
                BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations {
                    ordinal: synchronization.ordinal,
                },
                BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules {
                    ordinal: synchronization.ordinal,
                },
            ]);
        }
        let actual_roles = self
            .artifacts
            .iter()
            .map(BrokerTruthAcquisitionArtifactV1::role)
            .collect::<Vec<_>>();
        if actual_roles != expected_roles {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "acquisition artifacts do not exactly match the required ordered authority set",
            ));
        }

        let mut paths = HashSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !paths.insert(artifact.relative_path.as_str()) {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::DuplicateArtifact,
                    format!(
                        "duplicate acquisition artifact path {}",
                        artifact.relative_path
                    ),
                ));
            }
        }

        self.require_artifact_digest(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            &self.canonical_root_verification_sha256,
            "canonical root verification receipt",
        )?;
        self.require_artifact_digest(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            &self.canonical_scope_window_binding_sha256,
            "canonical scope window binding",
        )?;
        self.require_artifact_digest(
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            &self.capture_plan_sha256,
            "capture plan",
        )?;
        self.require_artifact_digest(
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
            &self.expected_trust_root_sha256,
            "trust root",
        )?;
        for synchronization in &self.reviewed_synchronizations {
            self.require_artifact_digest(
                BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
                synchronization.review_identity.review_record_sha256(),
                "review record",
            )?;
            self.require_artifact_digest(
                BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
                synchronization.review_identity.protocol_evidence_sha256(),
                "protocol evidence",
            )?;
            self.require_artifact_digest(
                BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations {
                    ordinal: synchronization.ordinal,
                },
                synchronization.review_identity.broker_observation_sha256(),
                "quote-session observations",
            )?;
            self.require_artifact_digest(
                BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules {
                    ordinal: synchronization.ordinal,
                },
                &synchronization.reviewed_rules_sha256,
                "reviewed quote replay rules",
            )?;
        }
        Ok(())
    }

    fn require_artifact_digest(
        &self,
        role: BrokerTruthAcquisitionArtifactRoleV1,
        expected_sha256: &str,
        label: &str,
    ) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        let Some(artifact) = self.artifacts.iter().find(|artifact| artifact.role == role) else {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                format!("acquisition authority is missing {label}"),
            ));
        };
        if artifact.sha256 != expected_sha256 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("{label} digest differs from its exact authority binding"),
            ));
        }
        Ok(())
    }
}

fn validate_acquisition_artifact_basename(
    value: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    let path = Path::new(value);
    let mut components = path.components();
    let exactly_one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if value.is_empty()
        || value.len() > MAX_ACQUISITION_ARTIFACT_NAME_BYTES_V1
        || !exactly_one_normal_component
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
    {
        return Err(contract_error(
            BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
            format!("acquisition artifact path {value:?} must be one safe lowercase basename"),
        ));
    }
    Ok(())
}

fn contract_error(
    code: BrokerFinancialTruthContractErrorCodeV1,
    detail: impl Into<String>,
) -> BrokerFinancialTruthContractErrorV1 {
    BrokerFinancialTruthContractErrorV1::new(code, detail)
}
