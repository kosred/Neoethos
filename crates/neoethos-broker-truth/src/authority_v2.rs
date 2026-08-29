//! Run-scoped semantic authority for one exact reviewed BFT2 capture.
//!
//! The content-addressed stores and the acquisition link prove integrity only.
//! This module is the sole mint for the move-only semantic authority after it
//! has reopened every linked input, matched the independent review identities,
//! and completed the full raw-versus-decoded BFT2 inspection.

use std::error::Error;
use std::fmt;

use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::Serialize;

use crate::acquisition_store_v1::{
    BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionStoreV1,
    VerifiedImmutableBrokerTruthAcquisitionAuthorityV1,
    VerifiedImmutableBrokerTruthAcquisitionLinkV1,
};
use crate::acquisition_v1::{
    BrokerTruthAcquisitionArtifactRoleV1, BrokerTruthAcquisitionPromotionEligibilityV1,
    BrokerTruthAcquisitionSemanticStatusV1, BrokerTruthReviewedSynchronizationBindingV1,
};
use crate::contracts::{EvidenceWindowV1, sha256_bytes, validate_sha256_hex};
use crate::contracts_v2::BrokerFinancialTruthBundleManifestV2;
use crate::semantic_v2::{
    BrokerFinancialTruthSemanticIngressErrorV2, UntrustedBrokerFinancialTruthIngressV2,
    inspect_untrusted_broker_financial_truth_bundle_v2,
};
use crate::store::BrokerFinancialTruthBundleStoreV1;

pub const BROKER_FINANCIAL_TRUTH_AUTHORITY_REFUSED_V2: &str =
    "BROKER_FINANCIAL_TRUTH_AUTHORITY_REFUSED_V2";

const MAX_AUTHORITY_ERROR_DETAIL_BYTES_V2: usize = 512;
const CLASS_DIGEST_DOMAIN_V2: &[u8] = b"neoethos.broker-financial-truth-class.v2\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthEvidenceClassV2 {
    PrimaryBidAsk,
    ConversionLegs,
    ExactSymbolAndAccountContracts,
    UnrealizedPnl,
    CloseDealReconciliation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthAuthoritySourceClassV2 {
    ResearchOnly,
}

impl BrokerFinancialTruthEvidenceClassV2 {
    const fn index(self) -> usize {
        match self {
            Self::PrimaryBidAsk => 0,
            Self::ConversionLegs => 1,
            Self::ExactSymbolAndAccountContracts => 2,
            Self::UnrealizedPnl => 3,
            Self::CloseDealReconciliation => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthAuthorityErrorCodeV2 {
    ReviewedEvidenceInvalid,
    ExactReopenFailed,
    RunIdentityMismatch,
    ScopeIdentityMismatch,
    WindowIdentityMismatch,
    TrustIdentityMismatch,
    ReviewIdentityMismatch,
    BrokerTruthManifestMismatch,
    ReviewedSynchronizationMismatch,
    PrimaryBidAskInvalid,
    ConversionLegsInvalid,
    ExactSymbolAndAccountContractsInvalid,
    UnrealizedPnlInvalid,
    CloseDealReconciliationInvalid,
    SemanticValidationFailed,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthAuthorityErrorV2 {
    code: BrokerFinancialTruthAuthorityErrorCodeV2,
    detail: String,
}

impl BrokerFinancialTruthAuthorityErrorV2 {
    pub const fn code(&self) -> BrokerFinancialTruthAuthorityErrorCodeV2 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerFinancialTruthAuthorityErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{BROKER_FINANCIAL_TRUTH_AUTHORITY_REFUSED_V2} code={:?} detail={}",
            self.code, self.detail
        )
    }
}

impl Error for BrokerFinancialTruthAuthorityErrorV2 {}

fn authority_error(
    code: BrokerFinancialTruthAuthorityErrorCodeV2,
    detail: impl Into<String>,
) -> BrokerFinancialTruthAuthorityErrorV2 {
    let detail = detail.into();
    let detail = if detail.len() <= MAX_AUTHORITY_ERROR_DETAIL_BYTES_V2 {
        detail
    } else {
        let mut end = MAX_AUTHORITY_ERROR_DETAIL_BYTES_V2;
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        detail[..end].to_owned()
    };
    BrokerFinancialTruthAuthorityErrorV2 { code, detail }
}

/// Independently supplied, checked review expectations for one exact run.
///
/// This value is evidence metadata only. It cannot create a capability or a
/// permit and is consumed by the validator so a review set cannot be silently
/// reused for another run.
#[derive(Debug)]
pub struct ReviewedBrokerFinancialTruthEvidenceV2 {
    canonical_run_identity_sha256: String,
    canonical_scope_identity_sha256: String,
    canonical_root_verification_sha256: String,
    canonical_scope_window_binding_sha256: String,
    capture_plan_sha256: String,
    trust_root_sha256: String,
    review_record_sha256: String,
    protocol_evidence_sha256: String,
    broker_truth_manifest_sha256: String,
    evidence_window: EvidenceWindowV1,
    reviewed_synchronizations: Vec<BrokerTruthReviewedSynchronizationBindingV1>,
}

impl ReviewedBrokerFinancialTruthEvidenceV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn checked_new(
        canonical_run_identity_sha256: impl Into<String>,
        canonical_scope_identity_sha256: impl Into<String>,
        canonical_root_verification_sha256: impl Into<String>,
        canonical_scope_window_binding_sha256: impl Into<String>,
        capture_plan_sha256: impl Into<String>,
        trust_root_sha256: impl Into<String>,
        review_record_sha256: impl Into<String>,
        protocol_evidence_sha256: impl Into<String>,
        broker_truth_manifest_sha256: impl Into<String>,
        evidence_window: EvidenceWindowV1,
        reviewed_synchronizations: Vec<BrokerTruthReviewedSynchronizationBindingV1>,
    ) -> Result<Self, BrokerFinancialTruthAuthorityErrorV2> {
        let evidence = Self {
            canonical_run_identity_sha256: canonical_run_identity_sha256.into(),
            canonical_scope_identity_sha256: canonical_scope_identity_sha256.into(),
            canonical_root_verification_sha256: canonical_root_verification_sha256.into(),
            canonical_scope_window_binding_sha256: canonical_scope_window_binding_sha256.into(),
            capture_plan_sha256: capture_plan_sha256.into(),
            trust_root_sha256: trust_root_sha256.into(),
            review_record_sha256: review_record_sha256.into(),
            protocol_evidence_sha256: protocol_evidence_sha256.into(),
            broker_truth_manifest_sha256: broker_truth_manifest_sha256.into(),
            evidence_window,
            reviewed_synchronizations,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthAuthorityErrorV2> {
        for (label, digest) in [
            (
                "canonical run identity",
                self.canonical_run_identity_sha256.as_str(),
            ),
            (
                "canonical scope identity",
                self.canonical_scope_identity_sha256.as_str(),
            ),
            (
                "canonical root verification",
                self.canonical_root_verification_sha256.as_str(),
            ),
            (
                "canonical scope-window binding",
                self.canonical_scope_window_binding_sha256.as_str(),
            ),
            ("capture plan", self.capture_plan_sha256.as_str()),
            ("trust root", self.trust_root_sha256.as_str()),
            ("review record", self.review_record_sha256.as_str()),
            ("protocol evidence", self.protocol_evidence_sha256.as_str()),
            ("BFT2 manifest", self.broker_truth_manifest_sha256.as_str()),
        ] {
            validate_sha256_hex(label, digest).map_err(|error| {
                authority_error(
                    BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                    error.to_string(),
                )
            })?;
        }
        EvidenceWindowV1::new(
            self.evidence_window.from_unix_ms_inclusive(),
            self.evidence_window.to_unix_ms_exclusive(),
        )
        .map_err(|error| {
            authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                error.to_string(),
            )
        })?;
        if self.reviewed_synchronizations.is_empty() {
            return Err(authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                "reviewed evidence has no synchronized Bid/Ask class",
            ));
        }
        for (index, synchronization) in self.reviewed_synchronizations.iter().enumerate() {
            let expected_ordinal = u32::try_from(index).map_err(|_| {
                authority_error(
                    BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                    "reviewed synchronization count exceeds u32",
                )
            })?;
            if synchronization.ordinal() != expected_ordinal
                || synchronization.window() != self.evidence_window
            {
                return Err(authority_error(
                    BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                    "reviewed synchronizations are not contiguous in the exact evidence window",
                ));
            }
            synchronization
                .review_identity()
                .validate_exact()
                .map_err(|error| {
                    authority_error(
                        BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid,
                        error.to_string(),
                    )
                })?;
        }
        Ok(())
    }
}

/// Move-only authority for one exact run, window, review set, and BFT2 bundle.
///
/// Deliberately no `Clone`, `Copy`, serialization, default, or public
/// constructor. Dropping it ends this run's authority; it never changes the
/// process-wide V1 gate.
#[derive(Debug)]
pub struct BrokerFinancialTruthAuthorityV2 {
    _verified_link: VerifiedImmutableBrokerTruthAcquisitionLinkV1,
    _verified_acquisition: VerifiedImmutableBrokerTruthAcquisitionAuthorityV1,
    _semantic_ingress: UntrustedBrokerFinancialTruthIngressV2,
    canonical_run_identity_sha256: String,
    canonical_scope_identity_sha256: String,
    canonical_root_verification_sha256: String,
    canonical_scope_window_binding_sha256: String,
    capture_plan_sha256: String,
    trust_root_sha256: String,
    review_record_sha256: String,
    protocol_evidence_sha256: String,
    broker_truth_manifest_sha256: String,
    evidence_window: EvidenceWindowV1,
    class_binding_sha256: [String; 5],
    reviewed_synchronization_count: usize,
    source_artifact_class: BrokerFinancialTruthAuthoritySourceClassV2,
    source_semantic_status: BrokerTruthAcquisitionSemanticStatusV1,
    source_promotion_eligibility: BrokerTruthAcquisitionPromotionEligibilityV1,
}

impl BrokerFinancialTruthAuthorityV2 {
    pub fn canonical_run_identity_sha256(&self) -> &str {
        &self.canonical_run_identity_sha256
    }

    pub fn canonical_scope_identity_sha256(&self) -> &str {
        &self.canonical_scope_identity_sha256
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

    pub fn trust_root_sha256(&self) -> &str {
        &self.trust_root_sha256
    }

    pub fn review_record_sha256(&self) -> &str {
        &self.review_record_sha256
    }

    pub fn protocol_evidence_sha256(&self) -> &str {
        &self.protocol_evidence_sha256
    }

    pub fn broker_truth_manifest_sha256(&self) -> &str {
        &self.broker_truth_manifest_sha256
    }

    pub const fn evidence_window(&self) -> EvidenceWindowV1 {
        self.evidence_window
    }

    pub fn evidence_class_binding_sha256(
        &self,
        class: BrokerFinancialTruthEvidenceClassV2,
    ) -> &str {
        &self.class_binding_sha256[class.index()]
    }

    pub const fn reviewed_synchronization_count(&self) -> usize {
        self.reviewed_synchronization_count
    }

    pub const fn source_artifact_class(&self) -> BrokerFinancialTruthAuthoritySourceClassV2 {
        self.source_artifact_class
    }

    pub const fn source_semantic_status(&self) -> BrokerTruthAcquisitionSemanticStatusV1 {
        self.source_semantic_status
    }

    pub const fn source_promotion_eligibility(
        &self,
    ) -> BrokerTruthAcquisitionPromotionEligibilityV1 {
        self.source_promotion_eligibility
    }
}

/// Exact-reopen and semantically validate one reviewed, content-addressed run.
pub fn validate_reviewed_broker_financial_truth_authority_v2(
    store: &BrokerTruthAcquisitionStoreV1,
    link_receipt: &BrokerTruthAcquisitionLinkReceiptV1,
    reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
) -> Result<BrokerFinancialTruthAuthorityV2, BrokerFinancialTruthAuthorityErrorV2> {
    reviewed.validate()?;
    let verified_link = store.open_link(link_receipt).map_err(|error| {
        authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::ExactReopenFailed,
            error.to_string(),
        )
    })?;
    let link_manifest = verified_link.manifest();
    let verified_acquisition = store
        .open_authority(link_manifest.authority_receipt())
        .map_err(|error| {
            authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ExactReopenFailed,
                error.to_string(),
            )
        })?;
    let acquisition = verified_acquisition.manifest();
    validate_acquisition_identities(acquisition, link_manifest.binding(), &reviewed)?;

    let verified_bundle = BrokerFinancialTruthBundleStoreV1::new(store.root())
        .open_exact_v2(
            link_manifest.broker_truth_receipt(),
            link_manifest.binding(),
        )
        .map_err(|error| {
            authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ExactReopenFailed,
                error.to_string(),
            )
        })?;
    if verified_bundle.receipt().manifest_sha256() != reviewed.broker_truth_manifest_sha256 {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::BrokerTruthManifestMismatch,
            "reopened BFT2 manifest differs from the independently reviewed manifest identity",
        ));
    }
    let manifest = verified_bundle.manifest();
    validate_reviewed_synchronizations(acquisition, manifest, &reviewed)?;
    let class_binding_sha256 = class_bindings(manifest)?;
    let semantic_ingress = inspect_untrusted_broker_financial_truth_bundle_v2(verified_bundle)
        .map_err(map_semantic_error)?;
    let reviewed_synchronization_count = reviewed.reviewed_synchronizations.len();
    let source_semantic_status = acquisition.semantic_status();
    let source_promotion_eligibility = acquisition.promotion_eligibility();

    Ok(BrokerFinancialTruthAuthorityV2 {
        _verified_link: verified_link,
        _verified_acquisition: verified_acquisition,
        _semantic_ingress: semantic_ingress,
        canonical_run_identity_sha256: reviewed.canonical_run_identity_sha256,
        canonical_scope_identity_sha256: reviewed.canonical_scope_identity_sha256,
        canonical_root_verification_sha256: reviewed.canonical_root_verification_sha256,
        canonical_scope_window_binding_sha256: reviewed.canonical_scope_window_binding_sha256,
        capture_plan_sha256: reviewed.capture_plan_sha256,
        trust_root_sha256: reviewed.trust_root_sha256,
        review_record_sha256: reviewed.review_record_sha256,
        protocol_evidence_sha256: reviewed.protocol_evidence_sha256,
        broker_truth_manifest_sha256: reviewed.broker_truth_manifest_sha256,
        evidence_window: reviewed.evidence_window,
        class_binding_sha256,
        reviewed_synchronization_count,
        source_artifact_class: BrokerFinancialTruthAuthoritySourceClassV2::ResearchOnly,
        source_semantic_status,
        source_promotion_eligibility,
    })
}

fn validate_acquisition_identities(
    acquisition: &crate::BrokerTruthAcquisitionAuthorityManifestV1,
    binding: &crate::BrokerFinancialTruthBindingV1,
    reviewed: &ReviewedBrokerFinancialTruthEvidenceV2,
) -> Result<(), BrokerFinancialTruthAuthorityErrorV2> {
    if acquisition.canonical_search_input_receipt_sha256() != reviewed.canonical_run_identity_sha256
        || binding.canonical_search_input_receipt_sha256() != reviewed.canonical_run_identity_sha256
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::RunIdentityMismatch,
            "acquisition, BFT2 binding, and reviewed run identity differ",
        ));
    }
    if acquisition.canonical_search_artifact_scope_sha256()
        != reviewed.canonical_scope_identity_sha256
        || acquisition.canonical_root_verification_sha256()
            != reviewed.canonical_root_verification_sha256
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::ScopeIdentityMismatch,
            "canonical scope or root-verification identity differs from reviewed evidence",
        ));
    }
    if acquisition.canonical_scope_window_binding_sha256()
        != reviewed.canonical_scope_window_binding_sha256
        || acquisition.capture_plan_sha256() != reviewed.capture_plan_sha256
        || binding.evaluated_window() != reviewed.evidence_window
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::WindowIdentityMismatch,
            "scope-window, capture-plan, or exact half-open window differs",
        ));
    }
    if acquisition.expected_trust_root_sha256() != reviewed.trust_root_sha256
        || artifact_digest(acquisition, BrokerTruthAcquisitionArtifactRoleV1::TrustRoot)
            != Some(reviewed.trust_root_sha256.as_str())
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::TrustIdentityMismatch,
            "trust-root identity differs from reviewed evidence",
        ));
    }
    if artifact_digest(
        acquisition,
        BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
    ) != Some(reviewed.review_record_sha256.as_str())
        || artifact_digest(
            acquisition,
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
        ) != Some(reviewed.protocol_evidence_sha256.as_str())
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::ReviewIdentityMismatch,
            "review-record or protocol-evidence identity differs",
        ));
    }
    Ok(())
}

fn artifact_digest(
    acquisition: &crate::BrokerTruthAcquisitionAuthorityManifestV1,
    role: BrokerTruthAcquisitionArtifactRoleV1,
) -> Option<&str> {
    acquisition
        .artifacts()
        .iter()
        .find(|artifact| artifact.role() == role)
        .map(crate::BrokerTruthAcquisitionArtifactV1::sha256)
}

fn validate_reviewed_synchronizations(
    acquisition: &crate::BrokerTruthAcquisitionAuthorityManifestV1,
    manifest: &BrokerFinancialTruthBundleManifestV2,
    reviewed: &ReviewedBrokerFinancialTruthEvidenceV2,
) -> Result<(), BrokerFinancialTruthAuthorityErrorV2> {
    let CanonicalDatasetScope::CTrader { account_id, .. } =
        manifest.binding().canonical_dataset_identity().scope()
    else {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedSynchronizationMismatch,
            "BFT2 binding is not cTrader",
        ));
    };
    let mut expected = Vec::new();
    expected.push(manifest.primary_quotes());
    for route in manifest.conversion_routes() {
        for leg in route.legs() {
            expected.push(leg.quotes());
        }
    }
    if acquisition.reviewed_synchronizations() != reviewed.reviewed_synchronizations
        || expected.len() != reviewed.reviewed_synchronizations.len()
    {
        return Err(authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedSynchronizationMismatch,
            "acquisition and reviewed synchronization sets are not exact matches",
        ));
    }
    for (ordinal, (quotes, synchronization)) in expected
        .into_iter()
        .zip(&reviewed.reviewed_synchronizations)
        .enumerate()
    {
        let ordinal = u32::try_from(ordinal).map_err(|_| {
            authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedSynchronizationMismatch,
                "BFT2 synchronization count exceeds u32",
            )
        })?;
        if synchronization.ordinal() != ordinal
            || synchronization.account_id() != *account_id
            || synchronization.symbol_id() != quotes.bid().symbol_id()
            || synchronization.window() != quotes.bid().requested_window()
            || synchronization.review_identity() != quotes.replay_rule().identity()
            || synchronization.reviewed_rules_sha256()
                != quotes.replay_rule().rules_decoded().sha256()
        {
            return Err(authority_error(
                BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedSynchronizationMismatch,
                "reviewed synchronization differs from exact primary or conversion evidence",
            ));
        }
    }
    Ok(())
}

fn class_bindings(
    manifest: &BrokerFinancialTruthBundleManifestV2,
) -> Result<[String; 5], BrokerFinancialTruthAuthorityErrorV2> {
    Ok([
        class_binding("primary_bid_ask", manifest.primary_quotes())?,
        class_binding("conversion_legs", &manifest.conversion_routes())?,
        class_binding(
            "exact_symbol_account_contracts",
            manifest.exact_symbol_contracts(),
        )?,
        class_binding("unrealized_pnl", manifest.broker_position_unrealized_pnl())?,
        class_binding(
            "close_deal_reconciliation",
            manifest.close_deal_reconciliation(),
        )?,
    ])
}

fn class_binding<T: Serialize + ?Sized>(
    label: &str,
    value: &T,
) -> Result<String, BrokerFinancialTruthAuthorityErrorV2> {
    let canonical = serde_json::to_vec(value).map_err(|error| {
        authority_error(
            BrokerFinancialTruthAuthorityErrorCodeV2::SemanticValidationFailed,
            format!("cannot bind {label} evidence class: {error}"),
        )
    })?;
    let mut bytes =
        Vec::with_capacity(CLASS_DIGEST_DOMAIN_V2.len() + label.len() + canonical.len());
    bytes.extend_from_slice(CLASS_DIGEST_DOMAIN_V2);
    bytes.extend_from_slice(label.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(sha256_bytes(&bytes))
}

fn map_semantic_error(
    error: BrokerFinancialTruthSemanticIngressErrorV2,
) -> BrokerFinancialTruthAuthorityErrorV2 {
    let artifact = error.artifact().unwrap_or_default();
    let code = if artifact.contains("bid") || artifact.contains("ask") {
        BrokerFinancialTruthAuthorityErrorCodeV2::PrimaryBidAskInvalid
    } else if artifact.contains("conversion") {
        BrokerFinancialTruthAuthorityErrorCodeV2::ConversionLegsInvalid
    } else if artifact.contains("symbol")
        || artifact.contains("asset")
        || artifact.contains("trader")
        || artifact.contains("contract")
    {
        BrokerFinancialTruthAuthorityErrorCodeV2::ExactSymbolAndAccountContractsInvalid
    } else if artifact.contains("pnl") {
        BrokerFinancialTruthAuthorityErrorCodeV2::UnrealizedPnlInvalid
    } else if artifact.contains("deal") || artifact.contains("reconcil") {
        BrokerFinancialTruthAuthorityErrorCodeV2::CloseDealReconciliationInvalid
    } else {
        BrokerFinancialTruthAuthorityErrorCodeV2::SemanticValidationFailed
    };
    authority_error(code, error.to_string())
}
