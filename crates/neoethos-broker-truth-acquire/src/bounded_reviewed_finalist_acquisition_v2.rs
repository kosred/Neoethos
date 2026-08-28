//! Bounded two-phase orchestration for one already locked finalist scope.
//!
//! Preparation performs the existing strict acquisition preflight and the
//! existing finalist coverage checks. Execution delegates to the production
//! same-session capture. Its result remains explicitly research-only,
//! unvalidated, and not promotion eligible; semantic authority is minted in
//! the leaf broker-truth crate only after independent review validation.

use std::error::Error;
use std::fmt;

use neoethos_broker_history::ProductionBrokerTruthCancellationV2;
use neoethos_broker_truth::{
    BrokerFinancialTruthBundleReceiptV2, BrokerTruthAcquisitionAuthorityReceiptV1,
    BrokerTruthAcquisitionLinkReceiptV1, LockedFinalistOosReplayScopeV1,
    ReviewedQuoteReplayRuleIdentityV2, VersionedLatencySlippagePolicyV1,
};
use neoethos_search::CanonicalSearchArtifactScopeV2;

use crate::finalist_quote_replay_acquisition_v1::{
    BrokerTruthPromotionEligibilityV1, BrokerTruthSemanticStatusV1,
    FinalistQuoteReplayAcquisitionInputV1, FinalistQuoteReplayAcquisitionOutcomeV1,
    FinalistQuoteReplayAcquisitionRequestV1, FinalistQuoteReplayArtifactClassV1,
    MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1, acquire_finalist_quote_replay_v1,
};
use crate::{BrokerTruthAcquisitionArgsV1, prepare_acquisition_v1};

pub const BOUNDED_REVIEWED_FINALIST_ACQUISITION_REFUSED_V2: &str =
    "BOUNDED_REVIEWED_FINALIST_ACQUISITION_REFUSED_V2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedReviewedFinalistAcquisitionErrorCodeV2 {
    PreflightFailed,
    FinalistRequestInvalid,
    CaptureFailed,
    OutcomeClassificationMismatch,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BoundedReviewedFinalistAcquisitionErrorV2 {
    code: BoundedReviewedFinalistAcquisitionErrorCodeV2,
    detail: &'static str,
}

impl BoundedReviewedFinalistAcquisitionErrorV2 {
    pub const fn code(&self) -> BoundedReviewedFinalistAcquisitionErrorCodeV2 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for BoundedReviewedFinalistAcquisitionErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{BOUNDED_REVIEWED_FINALIST_ACQUISITION_REFUSED_V2} code={:?} detail={}",
            self.code, self.detail
        )
    }
}

impl Error for BoundedReviewedFinalistAcquisitionErrorV2 {}

fn bounded_error(
    code: BoundedReviewedFinalistAcquisitionErrorCodeV2,
    detail: &'static str,
) -> BoundedReviewedFinalistAcquisitionErrorV2 {
    BoundedReviewedFinalistAcquisitionErrorV2 { code, detail }
}

/// Exact locked-finalist inputs that do not require broker credentials to
/// validate. The existing preflight supplies the prepared acquisition only
/// after all immutable canonical/review inputs have been opened exactly.
pub struct LockedFinalistBrokerTruthAcquisitionInputV2 {
    pub canonical_search_scope: CanonicalSearchArtifactScopeV2,
    pub canonical_search_input_receipt_sha256: String,
    pub canonical_signal_plan_sha256: String,
    pub portfolio_identity_sha256: String,
    pub search_config_hash: String,
    pub holdout_scope_identity_sha256: String,
    pub locked_finalist_scope: LockedFinalistOosReplayScopeV1,
    pub reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
    pub max_entry_wait_ms: i64,
    pub max_quote_staleness_ms: i64,
    pub max_exit_wait_ms: i64,
    pub latency_slippage: VersionedLatencySlippagePolicyV1,
}

/// Move-only prepared acquisition. Construction has completed both strict
/// preflight and bounded finalist coverage validation but has done no I/O to a
/// broker and minted no semantic authority.
pub struct PreparedBoundedReviewedFinalistAcquisitionV2 {
    request: FinalistQuoteReplayAcquisitionRequestV1,
}

/// Production capture result before semantic/trust review.
///
/// This wrapper deliberately exposes the exact immutable receipts and the
/// three evidence-only classifications, but no authority or permit conversion.
pub struct UnvalidatedLockedFinalistBrokerTruthEvidenceV2 {
    outcome: FinalistQuoteReplayAcquisitionOutcomeV1,
}

impl UnvalidatedLockedFinalistBrokerTruthEvidenceV2 {
    pub const fn authority_receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        self.outcome.authority_receipt()
    }

    pub const fn broker_truth_receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        self.outcome.broker_truth_receipt()
    }

    pub const fn acquisition_link_receipt(&self) -> &BrokerTruthAcquisitionLinkReceiptV1 {
        self.outcome.acquisition_link_receipt()
    }

    pub const fn artifact_class(&self) -> FinalistQuoteReplayArtifactClassV1 {
        self.outcome.artifact_class()
    }

    pub const fn semantic_status(&self) -> BrokerTruthSemanticStatusV1 {
        self.outcome.semantic_status()
    }

    pub const fn promotion_eligibility(&self) -> BrokerTruthPromotionEligibilityV1 {
        self.outcome.promotion_eligibility()
    }

    pub const fn outcome(&self) -> &FinalistQuoteReplayAcquisitionOutcomeV1 {
        &self.outcome
    }
}

/// Prepare exact immutable acquisition and locked-finalist coverage without
/// connecting to a broker. This is the fixture-friendly validation boundary.
pub fn prepare_bounded_reviewed_finalist_acquisition_v2(
    args: BrokerTruthAcquisitionArgsV1,
    input: LockedFinalistBrokerTruthAcquisitionInputV2,
) -> Result<PreparedBoundedReviewedFinalistAcquisitionV2, BoundedReviewedFinalistAcquisitionErrorV2>
{
    debug_assert!(MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1 > 0);
    let prepared_acquisition = prepare_acquisition_v1(args).map_err(|_| {
        bounded_error(
            BoundedReviewedFinalistAcquisitionErrorCodeV2::PreflightFailed,
            "strict broker-truth acquisition preflight failed",
        )
    })?;
    let request =
        FinalistQuoteReplayAcquisitionRequestV1::new(FinalistQuoteReplayAcquisitionInputV1 {
            prepared_acquisition,
            canonical_search_scope: input.canonical_search_scope,
            canonical_search_input_receipt_sha256: input.canonical_search_input_receipt_sha256,
            canonical_signal_plan_sha256: input.canonical_signal_plan_sha256,
            portfolio_identity_sha256: input.portfolio_identity_sha256,
            search_config_hash: input.search_config_hash,
            holdout_scope_identity_sha256: input.holdout_scope_identity_sha256,
            locked_finalist_scope: input.locked_finalist_scope,
            reviewed_replay_rule: input.reviewed_replay_rule,
            max_entry_wait_ms: input.max_entry_wait_ms,
            max_quote_staleness_ms: input.max_quote_staleness_ms,
            max_exit_wait_ms: input.max_exit_wait_ms,
            latency_slippage: input.latency_slippage,
        })
        .map_err(|_| {
            bounded_error(
                BoundedReviewedFinalistAcquisitionErrorCodeV2::FinalistRequestInvalid,
                "locked finalist scope, coverage, padding, or replay policy is invalid",
            )
        })?;
    Ok(PreparedBoundedReviewedFinalistAcquisitionV2 { request })
}

/// Execute one exact bounded same-session capture and keep its evidence
/// classification unvalidated until the separate reviewed validator succeeds.
pub fn execute_bounded_reviewed_finalist_acquisition_v2(
    prepared: PreparedBoundedReviewedFinalistAcquisitionV2,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<UnvalidatedLockedFinalistBrokerTruthEvidenceV2, BoundedReviewedFinalistAcquisitionErrorV2>
{
    let outcome =
        acquire_finalist_quote_replay_v1(prepared.request, cancellation).map_err(|_| {
            bounded_error(
                BoundedReviewedFinalistAcquisitionErrorCodeV2::CaptureFailed,
                "bounded same-session finalist capture failed",
            )
        })?;
    if outcome.artifact_class() != FinalistQuoteReplayArtifactClassV1::ResearchOnly
        || outcome.semantic_status() != BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly
        || outcome.promotion_eligibility()
            != BrokerTruthPromotionEligibilityV1::NotPromotionEligible
    {
        return Err(bounded_error(
            BoundedReviewedFinalistAcquisitionErrorCodeV2::OutcomeClassificationMismatch,
            "capture output did not retain its evidence-only classification",
        ));
    }
    Ok(UnvalidatedLockedFinalistBrokerTruthEvidenceV2 { outcome })
}
