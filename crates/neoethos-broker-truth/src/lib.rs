//! Immutable, versioned broker-financial evidence contracts below core/search/app.
//!
//! Storage integrity is not a permit. This crate deliberately keeps the
//! release gate closed until a later exact Vortex semantic validator proves
//! raw cTrader envelopes, decoded rows, synchronization and reconciliation for
//! the exact dataset/search/window binding.

#![forbid(unsafe_code)]

mod acquisition_store_v1;
mod acquisition_v1;
mod contracts;
mod contracts_v2;
mod execution_economics_v1;
mod execution_replay_v1;
mod gate;
mod semantic_v2;
mod store;

pub use acquisition_store_v1::{
    BROKER_TRUTH_ACQUISITION_AUTHORITY_ID_PREFIX_V1,
    BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1,
    BROKER_TRUTH_ACQUISITION_LINK_ID_PREFIX_V1, BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1,
    BROKER_TRUTH_ACQUISITION_LINK_SCHEMA_VERSION_V1, BrokerTruthAcquisitionArtifactSourceV1,
    BrokerTruthAcquisitionAuthorityReceiptV1, BrokerTruthAcquisitionLinkManifestV1,
    BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionStoreErrorCodeV1,
    BrokerTruthAcquisitionStoreErrorV1, BrokerTruthAcquisitionStoreV1,
    VerifiedImmutableBrokerTruthAcquisitionAuthorityV1,
    VerifiedImmutableBrokerTruthAcquisitionLinkV1,
};
pub use acquisition_v1::{
    BROKER_TRUTH_ACQUISITION_AUTHORITY_SCHEMA_VERSION_V1, BrokerTruthAcquisitionArtifactRoleV1,
    BrokerTruthAcquisitionArtifactV1, BrokerTruthAcquisitionAuthorityManifestV1,
    BrokerTruthAcquisitionPromotionEligibilityV1, BrokerTruthAcquisitionSemanticStatusV1,
    BrokerTruthReviewedSynchronizationBindingV1,
};

pub use contracts::{
    BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V1, BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V1,
    BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1, BrokerFinancialTruthBindingV1,
    BrokerFinancialTruthBundleManifestV1, BrokerFinancialTruthBundleReceiptV1,
    BrokerFinancialTruthContractErrorCodeV1, BrokerFinancialTruthContractErrorV1,
    BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1, ExactCapturedEvidencePairV1,
    ExactConversionLegEvidenceV1, ExactConversionRouteEvidenceV1, ExactQuoteSideEvidenceV1,
    ImmutableVortexArtifactV1, QuoteSideV1, SynchronizedBidAskEvidenceV1,
};
pub use contracts_v2::{
    BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V2, BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V2,
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthBundleReceiptV2,
    ExactBrokerRequestChunkV2, ExactBrokerRequestPageV2, ExactConversionLegEvidenceV2,
    ExactConversionRouteEvidenceV2, ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV2,
    ExactSymbolContractEvidenceV2, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2,
    ReviewedQuoteReplayRuleEvidenceV2, ReviewedQuoteReplayRuleIdentityV2,
    SynchronizedBidAskEvidenceV2,
};
pub use execution_economics_v1::{
    AccountMoneyV1, CausalQuoteToAccountConversionV1, EXECUTION_ECONOMICS_SCHEMA_VERSION_V1,
    ExecutionCommissionPolicyV1, ExecutionEconomicsArtifactClassV1, ExecutionEconomicsErrorCodeV1,
    ExecutionEconomicsErrorV1, ExecutionEconomicsPromotionEligibilityV1,
    ExecutionEconomicsResultV1, ExecutionSymbolContractV1, PnlConversionFeeV1,
    QuoteValidatedExecutionEconomicsLedgerV1, SignedSwapCashflowV1,
    build_quote_validated_execution_economics_v1,
};
pub use execution_replay_v1::{
    CanonicalBarSignalResearchDecisionV1, ClosedCanonicalBarTrailingThresholdV1,
    CompleteBidAskQuoteReplayEvidenceV1, CompleteQuoteSideCoverageV1, ExactHistoricalQuoteV1,
    ExactQuoteSourceOrdinalV1, ExactZeroRowQuoteWindowProofV1, LockedFinalistOosReplayScopeV1,
    QuoteValidatedEntryBookV1, QuoteValidatedPriceReferenceV1, QuoteValidatedResearchAuthorityV1,
    QuoteValidatedResearchExitReasonV1, QuoteValidatedResearchLedgerV1,
    QuoteValidatedResearchNonEntryReasonV1, QuoteValidatedResearchNonEntryV1,
    QuoteValidatedResearchPositionV1, QuoteValidatedResearchPromotionEligibilityV1,
    QuoteValidatedResearchReplayBindingV1, QuoteValidatedResearchReplayErrorCodeV1,
    QuoteValidatedResearchReplayErrorV1, QuoteValidatedResearchReplayPlanV1,
    QuoteValidatedResearchReplayPolicyV1, QuoteValidatedResearchReplayReceiptV1,
    ResearchPositionDirectionV1, ReviewedSameTimestampMergeRuleV1, SameTimestampCrossSideOrderV1,
    SealedHistoricalBidAskQuoteReplayEvidenceV1, SealedHistoricalQuoteValidatedResearchLedgerV1,
    VersionedLatencySlippagePolicyV1, open_sealed_historical_bid_ask_quote_replay_evidence_v1,
    replay_quote_validated_research_v1, replay_sealed_quote_validated_research_v1,
};
pub use gate::{
    BROKER_FINANCIAL_TRUTH_SCHEMA_VERSION_V1, BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1,
    BrokerFinancialOperationV1, BrokerFinancialTruthCapabilityV1, BrokerFinancialTruthErrorV1,
    BrokerFinancialTruthPermitV1, MissingBrokerFinancialEvidenceV1,
    current_broker_financial_truth_capability_v1,
};
pub use semantic_v2::{
    BROKER_FINANCIAL_TRUTH_SEMANTIC_INGRESS_REFUSED_V2,
    BrokerFinancialTruthSemanticIngressErrorCodeV2, BrokerFinancialTruthSemanticIngressErrorV2,
    UntrustedBrokerFinancialTruthIngressV2, inspect_untrusted_broker_financial_truth_bundle_v2,
};
pub use store::{
    BrokerFinancialTruthArtifactSourceV1, BrokerFinancialTruthBundleStoreV1,
    BrokerFinancialTruthStoreErrorCodeV1, BrokerFinancialTruthStoreErrorV1,
    VerifiedImmutableBrokerFinancialTruthBundleV1, VerifiedImmutableBrokerFinancialTruthBundleV2,
};
