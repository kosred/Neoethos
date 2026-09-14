use std::error::Error;
use std::fmt;

use neoethos_broker_history::{
    ProductionBrokerTruthCancellationV2, ProductionBrokerTruthCaptureOutcomeV2,
    ProductionBrokerTruthCaptureRequestV2, capture_production_broker_financial_truth_v2,
};
use neoethos_broker_truth::{
    BrokerFinancialTruthBundleReceiptV2, BrokerFinancialTruthBundleStoreV1,
    BrokerTruthAcquisitionAuthorityReceiptV1, BrokerTruthAcquisitionLinkReceiptV1,
    BrokerTruthAcquisitionStoreV1, CanonicalBarSignalResearchDecisionV1,
    ClosedCanonicalBarTrailingThresholdV1, LockedFinalistOosReplayScopeV1,
    MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2, QuoteValidatedResearchReplayBindingV1,
    QuoteValidatedResearchReplayPlanV1, QuoteValidatedResearchReplayPolicyV1,
    ReviewedBrokerFinancialTruthEvidenceV2, ReviewedQuoteReplayRuleIdentityV2,
    SealedHistoricalQuoteValidatedResearchLedgerV1, VersionedLatencySlippagePolicyV1,
    inspect_untrusted_broker_financial_truth_bundle_v2,
    into_sealed_historical_bid_ask_quote_replay_evidence_v2,
    replay_sealed_quote_validated_decision_sequence_v1,
    validate_reviewed_broker_financial_truth_authority_v2,
};
use neoethos_data::CanonicalDatasetScope;
use neoethos_search::{CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1};

use crate::{PreparedBrokerTruthAcquisitionV1, execute_prepared_acquisition_v1};

pub const MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1: i64 = 2 * 366 * 24 * 60 * 60 * 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalistQuoteReplayRestartPolicyV1 {
    RestartWholeBoundedCaptureOnce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalistQuoteReplayArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerTruthSemanticStatusV1 {
    UnvalidatedEvidenceOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerTruthPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalistQuoteReplayAcquisitionErrorCodeV1 {
    InvalidRequest,
    FinalistScopeMismatch,
    PaddingMismatch,
    WindowTooLarge,
    SameSessionCaptureRequired,
    IncompletePageCoverage,
    ZeroRowQuoteCoverage,
    ArtifactDigestMismatch,
    CoverageWindowMismatch,
    CaptureEvidenceInvalid,
    TwoPhaseManifestBindingMismatch,
    ReviewValidationFailed,
    QuoteReplayFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalistQuoteReplayAcquisitionErrorV1 {
    code: FinalistQuoteReplayAcquisitionErrorCodeV1,
    detail: &'static str,
}

impl FinalistQuoteReplayAcquisitionErrorV1 {
    pub const fn code(&self) -> FinalistQuoteReplayAcquisitionErrorCodeV1 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for FinalistQuoteReplayAcquisitionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.detail)
    }
}

impl Error for FinalistQuoteReplayAcquisitionErrorV1 {}

fn acquisition_error(
    code: FinalistQuoteReplayAcquisitionErrorCodeV1,
    detail: &'static str,
) -> FinalistQuoteReplayAcquisitionErrorV1 {
    FinalistQuoteReplayAcquisitionErrorV1 { code, detail }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

// Search config identity is not an artifact digest. The canonical Search
// producer currently emits the domain-tagged stable JSON identity below; raw
// SHA-256 remains accepted for already-persisted replay plans.
fn valid_search_config_identity(value: &str) -> bool {
    valid_sha256(value)
        || value.strip_prefix("fnv64:").is_some_and(|payload| {
            payload.len() == 16
                && payload
                    .as_bytes()
                    .iter()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        })
}

pub struct FinalistQuoteReplayAcquisitionInputV1 {
    pub prepared_acquisition: PreparedBrokerTruthAcquisitionV1,
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

pub struct FinalistQuoteReplayAcquisitionRequestV1 {
    prepared_acquisition: PreparedBrokerTruthAcquisitionV1,
    canonical_search_scope: CanonicalSearchArtifactScopeV2,
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    portfolio_identity_sha256: String,
    search_config_hash: String,
    holdout_scope_identity_sha256: String,
    locked_finalist_scope: LockedFinalistOosReplayScopeV1,
    reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
    replay_policy: QuoteValidatedResearchReplayPolicyV1,
    restart_policy: FinalistQuoteReplayRestartPolicyV1,
}

struct LockedReplayPolicyInputV1 {
    max_entry_wait_ms: i64,
    max_quote_staleness_ms: i64,
    max_exit_wait_ms: i64,
    latency_slippage: VersionedLatencySlippagePolicyV1,
    reviewed_same_timestamp_merge_rule:
        Option<neoethos_broker_truth::ReviewedSameTimestampMergeRuleV1>,
}

impl FinalistQuoteReplayAcquisitionRequestV1 {
    pub fn new(
        input: FinalistQuoteReplayAcquisitionInputV1,
    ) -> Result<Self, FinalistQuoteReplayAcquisitionErrorV1> {
        let policy_input = LockedReplayPolicyInputV1 {
            max_entry_wait_ms: input.max_entry_wait_ms,
            max_quote_staleness_ms: input.max_quote_staleness_ms,
            max_exit_wait_ms: input.max_exit_wait_ms,
            latency_slippage: input.latency_slippage,
            reviewed_same_timestamp_merge_rule: None,
        };
        let replay_policy = QuoteValidatedResearchReplayPolicyV1::new(
            policy_input.max_entry_wait_ms,
            policy_input.max_quote_staleness_ms,
            policy_input.max_exit_wait_ms,
            policy_input.latency_slippage,
            policy_input.reviewed_same_timestamp_merge_rule,
        )
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                "finalist replay policy is invalid",
            )
        })?;
        let request = Self {
            prepared_acquisition: input.prepared_acquisition,
            canonical_search_scope: input.canonical_search_scope,
            canonical_search_input_receipt_sha256: input.canonical_search_input_receipt_sha256,
            canonical_signal_plan_sha256: input.canonical_signal_plan_sha256,
            portfolio_identity_sha256: input.portfolio_identity_sha256,
            search_config_hash: input.search_config_hash,
            holdout_scope_identity_sha256: input.holdout_scope_identity_sha256,
            locked_finalist_scope: input.locked_finalist_scope,
            reviewed_replay_rule: input.reviewed_replay_rule,
            replay_policy,
            restart_policy: FinalistQuoteReplayRestartPolicyV1::RestartWholeBoundedCaptureOnce,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), FinalistQuoteReplayAcquisitionErrorV1> {
        self.canonical_search_scope.validate().map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "canonical finalist scope is invalid",
            )
        })?;
        if !matches!(
            self.canonical_search_scope.evaluated_window().role(),
            CanonicalSearchWindowRoleV1::Holdout | CanonicalSearchWindowRoleV1::ForwardTest
        ) {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "quote acquisition is restricted to a locked holdout or forward-test scope",
            ));
        }
        let recomputed_scope_identity =
            self.canonical_search_scope.identity_sha256().map_err(|_| {
                acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                    "canonical finalist scope identity cannot be recomputed",
                )
            })?;
        if self.canonical_search_scope.receipt_sha256()
            != self.canonical_search_input_receipt_sha256
            || recomputed_scope_identity != self.holdout_scope_identity_sha256
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "canonical finalist receipt or holdout scope identity changed",
            ));
        }
        for digest in [
            &self.canonical_search_input_receipt_sha256,
            &self.canonical_signal_plan_sha256,
            &self.portfolio_identity_sha256,
            &self.holdout_scope_identity_sha256,
        ] {
            if !valid_sha256(digest) {
                return Err(acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::ArtifactDigestMismatch,
                    "finalist acquisition contains an invalid SHA-256 identity",
                ));
            }
        }
        if !valid_search_config_identity(&self.search_config_hash) {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::ArtifactDigestMismatch,
                "finalist acquisition contains an invalid Search config identity",
            ));
        }

        let locked_evaluation_window = self.locked_finalist_scope.locked_evaluation_window();
        let required_quote_coverage_window =
            self.locked_finalist_scope.required_quote_coverage_window();
        let seed_padding_ms = self.locked_finalist_scope.seed_padding_ms();
        let exit_padding_ms = self.locked_finalist_scope.exit_padding_ms();
        let rebuilt_scope = LockedFinalistOosReplayScopeV1::new(
            locked_evaluation_window,
            seed_padding_ms,
            exit_padding_ms,
        )
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::PaddingMismatch,
                "locked finalist replay padding is invalid",
            )
        })?;
        if rebuilt_scope != self.locked_finalist_scope {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::PaddingMismatch,
                "locked finalist replay padding changed its required quote window",
            ));
        }
        let window_span = required_quote_coverage_window
            .to_unix_ms_exclusive()
            .checked_sub(required_quote_coverage_window.from_unix_ms_inclusive())
            .ok_or_else(|| {
                acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::CoverageWindowMismatch,
                    "required quote coverage window is reversed",
                )
            })?;
        if window_span > MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1 {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::WindowTooLarge,
                "bounded finalist quote-replay window exceeds its maximum span",
            ));
        }
        let evaluated = self.canonical_search_scope.evaluated_window();
        if locked_evaluation_window.from_unix_ms_inclusive() != evaluated.timestamp_start_ms()
            || locked_evaluation_window.to_unix_ms_exclusive() <= evaluated.timestamp_end_ms()
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "locked replay window does not exactly cover the canonical finalist timestamps",
            ));
        }
        let capture_request = self.prepared_acquisition.capture_request();
        let capture_binding = capture_request.binding();
        if self.prepared_acquisition.evidence_window() != required_quote_coverage_window
            || capture_request.window() != required_quote_coverage_window
            || capture_binding.evaluated_window() != required_quote_coverage_window
            || capture_binding.canonical_search_input_receipt_sha256()
                != self.canonical_search_input_receipt_sha256
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::CoverageWindowMismatch,
                "prepared capture is not bound to the exact padded finalist window",
            ));
        }
        let CanonicalDatasetScope::CTrader {
            account_id,
            symbol_id,
            ..
        } = capture_binding.canonical_dataset_identity().scope()
        else {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "finalist capture does not name an exact cTrader dataset scope",
            ));
        };
        if *account_id != self.prepared_acquisition.account_id()
            || *account_id != capture_request.account_id()
            || *symbol_id != capture_request.primary_instrument().symbol_id()
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::FinalistScopeMismatch,
                "finalist account or symbol differs from the prepared capture",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalistQuoteReplayAcquisitionOutcomeV1 {
    authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    broker_truth_receipt: BrokerFinancialTruthBundleReceiptV2,
    acquisition_link_receipt: BrokerTruthAcquisitionLinkReceiptV1,
    replay_binding: QuoteValidatedResearchReplayBindingV1,
    replay_policy: QuoteValidatedResearchReplayPolicyV1,
    artifact_class: FinalistQuoteReplayArtifactClassV1,
    semantic_status: BrokerTruthSemanticStatusV1,
    promotion_eligibility: BrokerTruthPromotionEligibilityV1,
    portfolio_identity_sha256: String,
    search_config_hash: String,
    holdout_scope_identity_sha256: String,
}

impl FinalistQuoteReplayAcquisitionOutcomeV1 {
    pub const fn authority_receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        &self.authority_receipt
    }

    pub const fn broker_truth_receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        &self.broker_truth_receipt
    }

    pub const fn acquisition_link_receipt(&self) -> &BrokerTruthAcquisitionLinkReceiptV1 {
        &self.acquisition_link_receipt
    }

    pub const fn replay_binding(&self) -> &QuoteValidatedResearchReplayBindingV1 {
        &self.replay_binding
    }

    pub const fn replay_policy(&self) -> &QuoteValidatedResearchReplayPolicyV1 {
        &self.replay_policy
    }

    pub const fn artifact_class(&self) -> FinalistQuoteReplayArtifactClassV1 {
        self.artifact_class
    }

    pub const fn semantic_status(&self) -> BrokerTruthSemanticStatusV1 {
        self.semantic_status
    }

    pub const fn promotion_eligibility(&self) -> BrokerTruthPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }

    pub fn search_config_hash(&self) -> &str {
        &self.search_config_hash
    }

    pub fn holdout_scope_identity_sha256(&self) -> &str {
        &self.holdout_scope_identity_sha256
    }

    /// Convert prepared canonical strategy signals into actual single-position
    /// quote replay outcomes, retaining decision-row/risk attribution. Review
    /// and open this exact captured snapshot once; never use the legacy mock
    /// replay or mint global trading authority. Account economics and the
    /// outer-holdout signal/risk identity binding remain caller responsibilities.
    pub fn replay_reviewed_signal_lane_v1(
        &self,
        store: &BrokerTruthAcquisitionStoreV1,
        reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
        lane: &neoethos_trader::data_replay::CanonicalSignalQuoteLaneV1<'_>,
    ) -> Result<
        Vec<neoethos_trader::data_replay::CanonicalSignalQuoteOutcomeV1>,
        FinalistQuoteReplayAcquisitionErrorV1,
    > {
        lane.validate(&self.replay_binding, &self.replay_policy).map_err(|_| acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
            "canonical strategy lane does not fit the captured symbol, window or fixed-stop exit policy",
        ))?;
        let evidence = self.open_reviewed_quote_snapshot_v1(store, reviewed)?;
        neoethos_trader::data_replay::replay_canonical_signal_quote_lane_v1(
            lane, &self.replay_binding, &self.replay_policy, &evidence,
        ).map_err(|_| acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::QuoteReplayFailed,
            "canonical strategy signals cannot be replayed with the exact captured quotes and exit policy",
        ))
    }

    /// Replay the exact pre-acquisition plan in independent strategy lanes and
    /// return a provenance-checked set for Discovery's common money consumer.
    /// Costs/sizing must be supplied explicitly for each closed quote ledger;
    /// this method neither infers broker economics nor calls a broker.
    pub fn replay_reviewed_locked_portfolio_v3<E, F>(
        &self,
        cpu: &neoethos_core::execution::BudgetedCpuScope<'_>,
        store: &BrokerTruthAcquisitionStoreV1,
        reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
        locked: &neoethos_search::LockedCanonicalSignalPlanV3<'_>,
        entry_inputs: E,
        execution_economics: F,
    ) -> Result<
        neoethos_search::LockedPortfolioOuterHoldoutReplaySetV3,
        FinalistQuoteReplayAcquisitionErrorV1,
    >
    where
        E: FnMut(
            &SealedHistoricalQuoteValidatedResearchLedgerV1,
            &neoethos_search::QuoteValidatedDecisionProvenanceV3,
        ) -> anyhow::Result<neoethos_search::QuoteEntryFinancialInputsV3>,
        F: FnMut(
            &SealedHistoricalQuoteValidatedResearchLedgerV1,
            &neoethos_search::QuoteEntrySizingEvidenceV3,
        )
            -> anyhow::Result<neoethos_broker_truth::QuoteValidatedExecutionEconomicsLedgerV1>,
    {
        cpu.require_current_pool().map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                "reviewed portfolio replay requires its exact active leased CPU pool",
            )
        })?;
        locked
            .validate_replay_binding(&self.replay_binding, &self.replay_policy)
            .map_err(|_| {
                acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                    "captured quotes do not match the prelocked canonical signal/risk plan",
                )
            })?;
        if self.portfolio_identity_sha256 != locked.portfolio_identity_sha256()
            || self.search_config_hash != locked.search_config_hash()
            || self.holdout_scope_identity_sha256 != locked.scope_identity_sha256()
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                "captured finalist portfolio/config/holdout differs from the prelocked producer",
            ));
        }
        let snapshot = self.open_reviewed_quote_snapshot_v1(store, reviewed)?;
        let lanes = neoethos_trader::data_replay::replay_locked_canonical_signal_portfolio_v3(
            cpu,
            locked,
            &self.replay_binding,
            &self.replay_policy,
            &snapshot,
        )
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::QuoteReplayFailed,
                "locked canonical strategy portfolio failed quote replay",
            )
        })?;
        neoethos_search::LockedPortfolioOuterHoldoutReplaySetV3::new(
            locked, &self.replay_binding, &self.replay_policy, &snapshot, lanes, entry_inputs, execution_economics,
        ).map_err(|_| acquisition_error(FinalistQuoteReplayAcquisitionErrorCodeV1::QuoteReplayFailed,
            "quote outcomes or execution economics do not match the complete locked signal plan"))
    }

    /// Open one exact independently reviewed snapshot for reuse across strategy
    /// lanes inside the caller's existing budgeted worker pool. This does not
    /// capture data, certify costs, or grant live/promotion authority.
    pub fn open_reviewed_quote_snapshot_v1(
        &self,
        store: &BrokerTruthAcquisitionStoreV1,
        reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
    ) -> Result<
        neoethos_broker_truth::SealedHistoricalBidAskQuoteReplayEvidenceV1,
        FinalistQuoteReplayAcquisitionErrorV1,
    > {
        let authority = validate_reviewed_broker_financial_truth_authority_v2(
            store,
            &self.acquisition_link_receipt,
            reviewed,
        )
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::ReviewValidationFailed,
                "captured finalist evidence failed independent reviewed V2 validation",
            )
        })?;
        into_sealed_historical_bid_ask_quote_replay_evidence_v2(authority, &self.replay_binding)
            .map_err(|_| {
                acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::TwoPhaseManifestBindingMismatch,
                    "reviewed quote records differ from the captured finalist replay binding",
                )
            })
    }

    /// Continue this exact capture through independent V2 review and the
    /// existing one-decision quote replay. The locked binding and fill policy
    /// come from capture preparation, not replacement caller settings.
    ///
    /// The caller still owns the canonical decision/trailing schedule. This
    /// does not implement a whole strategy position engine, certify financial
    /// cost assumptions, or make the outcome promotion eligible.
    pub fn replay_reviewed_decision_v1(
        &self,
        store: &BrokerTruthAcquisitionStoreV1,
        reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
        decision: CanonicalBarSignalResearchDecisionV1,
        trailing_thresholds: Vec<ClosedCanonicalBarTrailingThresholdV1>,
    ) -> Result<SealedHistoricalQuoteValidatedResearchLedgerV1, FinalistQuoteReplayAcquisitionErrorV1>
    {
        let mut ledgers = self.replay_reviewed_decision_sequence_v1(
            store,
            reviewed,
            vec![(decision, trailing_thresholds)],
        )?;
        ledgers.pop().ok_or_else(|| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::QuoteReplayFailed,
                "reviewed single decision produced no replay ledger",
            )
        })
    }

    /// Replay an ordered, non-overlapping decision lane from this capture.
    /// Validate every decision before IO, independently review the exact V2
    /// evidence once, then borrow the same sealed quotes for every position.
    /// Per-fill economics, bar-signal generation and promotion remain separate.
    pub fn replay_reviewed_decision_sequence_v1(
        &self,
        store: &BrokerTruthAcquisitionStoreV1,
        reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
        decisions: Vec<(
            CanonicalBarSignalResearchDecisionV1,
            Vec<ClosedCanonicalBarTrailingThresholdV1>,
        )>,
    ) -> Result<
        Vec<SealedHistoricalQuoteValidatedResearchLedgerV1>,
        FinalistQuoteReplayAcquisitionErrorV1,
    > {
        let plans = decisions
            .into_iter()
            .map(|(decision, trailing_thresholds)| {
                QuoteValidatedResearchReplayPlanV1::new(
                    self.replay_binding.clone(),
                    self.replay_policy.clone(),
                    vec![decision],
                    trailing_thresholds,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                    "canonical decision or trailing schedule does not fit the captured replay policy",
                )
            })?;
        QuoteValidatedResearchReplayPlanV1::validate_ordered_sequence(&plans).map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest,
                "captured decision sequence is empty, unordered or changes its binding or policy",
            )
        })?;
        let evidence = self.open_reviewed_quote_snapshot_v1(store, reviewed)?;
        replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence).map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::QuoteReplayFailed,
                "reviewed finalist quotes cannot replay the exact non-overlapping decision sequence",
            )
        })
    }
}

pub fn acquire_finalist_quote_replay_v1(
    request: FinalistQuoteReplayAcquisitionRequestV1,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<FinalistQuoteReplayAcquisitionOutcomeV1, FinalistQuoteReplayAcquisitionErrorV1> {
    request.validate()?;
    let restart_whole_bounded_capture = matches!(
        request.restart_policy,
        FinalistQuoteReplayRestartPolicyV1::RestartWholeBoundedCaptureOnce
    );
    if !restart_whole_bounded_capture {
        return Err(acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::SameSessionCaptureRequired,
            "finalist quotes require one whole same-session bounded capture",
        ));
    }

    let expected_window = request
        .locked_finalist_scope
        .required_quote_coverage_window();
    let maximum_request_span_ms = MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2;
    let validate_quote_side_coverage = |side: &neoethos_broker_truth::ExactQuoteSideEvidenceV2| {
        if side.requested_window() != expected_window {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::CoverageWindowMismatch,
                "captured quote side differs from the required finalist window",
            ));
        }
        let mut event_count = 0_u64;
        for chunk in side.request_chunks_newest_first() {
            let span = chunk
                .requested_window()
                .to_unix_ms_exclusive()
                .checked_sub(chunk.requested_window().from_unix_ms_inclusive())
                .ok_or_else(|| {
                    acquisition_error(
                        FinalistQuoteReplayAcquisitionErrorCodeV1::IncompletePageCoverage,
                        "captured quote chunk has a reversed window",
                    )
                })?;
            if span > maximum_request_span_ms {
                return Err(acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::IncompletePageCoverage,
                    "captured quote chunk exceeds the exact cTrader request span",
                ));
            }
            let pages = chunk.pages_newest_first();
            if pages.is_empty() || pages.last().is_some_and(|page| page.response_has_more()) {
                return Err(acquisition_error(
                    FinalistQuoteReplayAcquisitionErrorCodeV1::IncompletePageCoverage,
                    "captured quote chunk has incomplete page coverage",
                ));
            }
            for page in pages {
                event_count = event_count.checked_add(page.event_count()).ok_or_else(|| {
                    acquisition_error(
                        FinalistQuoteReplayAcquisitionErrorCodeV1::CaptureEvidenceInvalid,
                        "captured quote event count overflowed",
                    )
                })?;
            }
        }
        if side
            .request_chunks_newest_first()
            .iter()
            .flat_map(|chunk| chunk.pages_newest_first())
            .all(|page| page.event_count() == 0)
            || event_count == 0
        {
            return Err(acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::ZeroRowQuoteCoverage,
                "captured quote coverage contains zero rows",
            ));
        }
        Ok(())
    };
    let store_root = request.prepared_acquisition.store_root().to_path_buf();
    let expected_binding = request
        .prepared_acquisition
        .capture_request()
        .binding()
        .clone();
    let expected_account_id = request.prepared_acquisition.account_id();
    let expected_instrument = request
        .prepared_acquisition
        .capture_request()
        .primary_instrument()
        .clone();

    // This typed entry is the one production route. Its implementation owns
    // CTraderBrokerTruthAdapterV2 and capture_and_publish_broker_financial_truth_v2,
    // so no caller can splice pages from another session into this request.
    let production_capture_entry: fn(
        ProductionBrokerTruthCaptureRequestV2,
        &ProductionBrokerTruthCancellationV2,
    ) -> Result<
        ProductionBrokerTruthCaptureOutcomeV2,
        neoethos_broker_history::ProductionBrokerTruthCaptureErrorV2,
    > = capture_production_broker_financial_truth_v2;
    debug_assert_ne!(production_capture_entry as usize, 0);

    let captured = execute_prepared_acquisition_v1(request.prepared_acquisition, cancellation)
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::SameSessionCaptureRequired,
                "whole same-session finalist quote capture failed",
            )
        })?;
    let authority_receipt = captured.authority_receipt().clone();
    let broker_truth_receipt = captured.broker_truth_receipt().clone();
    let acquisition_store = BrokerTruthAcquisitionStoreV1::new(&store_root);
    acquisition_store
        .open_authority(&authority_receipt)
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::ArtifactDigestMismatch,
                "captured acquisition authority failed exact reopen",
            )
        })?;
    let verified_bundle = BrokerFinancialTruthBundleStoreV1::new(&store_root)
        .open_exact_v2(&broker_truth_receipt, &expected_binding)
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::ArtifactDigestMismatch,
                "captured broker-truth bundle failed exact reopen",
            )
        })?;
    let manifest = verified_bundle.manifest();
    if manifest.binding() != &expected_binding
        || manifest.binding().evaluated_window() != expected_window
    {
        return Err(acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::CoverageWindowMismatch,
            "captured broker-truth manifest changed the exact finalist binding",
        ));
    }
    validate_quote_side_coverage(manifest.primary_quotes().bid())?;
    validate_quote_side_coverage(manifest.primary_quotes().ask())?;
    inspect_untrusted_broker_financial_truth_bundle_v2(verified_bundle).map_err(|_| {
        acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::CaptureEvidenceInvalid,
            "captured broker-truth bundle failed structural evidence inspection",
        )
    })?;

    let actual_quote_evidence_manifest_sha256 = broker_truth_receipt.manifest_sha256().to_owned();
    let replay_binding = QuoteValidatedResearchReplayBindingV1::new(
        request.canonical_search_input_receipt_sha256,
        request.canonical_signal_plan_sha256,
        expected_account_id,
        expected_instrument.symbol_id(),
        expected_instrument.symbol_name(),
        request.locked_finalist_scope,
        request.reviewed_replay_rule,
        actual_quote_evidence_manifest_sha256,
    )
    .map_err(|_| {
        acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::TwoPhaseManifestBindingMismatch,
            "actual BFT2 manifest cannot form the exact finalist replay binding",
        )
    })?;
    let acquisition_link_receipt = acquisition_store
        .publish_link(&authority_receipt, &broker_truth_receipt, &expected_binding)
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::TwoPhaseManifestBindingMismatch,
                "captured finalist evidence link publication failed",
            )
        })?;
    let reopened_link = acquisition_store
        .open_link(&acquisition_link_receipt)
        .map_err(|_| {
            acquisition_error(
                FinalistQuoteReplayAcquisitionErrorCodeV1::TwoPhaseManifestBindingMismatch,
                "captured finalist evidence link failed exact reopen",
            )
        })?;
    if reopened_link.manifest().authority_receipt() != &authority_receipt
        || reopened_link.manifest().broker_truth_receipt() != &broker_truth_receipt
        || reopened_link.manifest().binding() != &expected_binding
        || captured.link_receipt() != &acquisition_link_receipt
    {
        return Err(acquisition_error(
            FinalistQuoteReplayAcquisitionErrorCodeV1::TwoPhaseManifestBindingMismatch,
            "reopened finalist evidence link differs from the captured two-phase identities",
        ));
    }

    Ok(FinalistQuoteReplayAcquisitionOutcomeV1 {
        authority_receipt,
        broker_truth_receipt,
        acquisition_link_receipt,
        replay_binding,
        replay_policy: request.replay_policy,
        artifact_class: FinalistQuoteReplayArtifactClassV1::ResearchOnly,
        semantic_status: BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly,
        promotion_eligibility: BrokerTruthPromotionEligibilityV1::NotPromotionEligible,
        portfolio_identity_sha256: request.portfolio_identity_sha256,
        search_config_hash: request.search_config_hash,
        holdout_scope_identity_sha256: request.holdout_scope_identity_sha256,
    })
}

#[cfg(test)]
#[path = "prelocked_portfolio_replay_tests.rs"]
mod prelocked_portfolio_replay_tests;

#[cfg(test)]
mod reviewed_replay_tests {
    use super::*;
    use neoethos_broker_truth::{
        BrokerTruthReviewedSynchronizationBindingV1, EvidenceWindowV1, ResearchPositionDirectionV1,
    };
    use serde_json::json;

    #[test]
    fn search_config_identity_accepts_canonical_search_and_persisted_formats_only() {
        assert!(valid_search_config_identity("fnv64:0123456789abcdef"));
        assert!(valid_search_config_identity(&"ab".repeat(32)));
        for invalid in [
            "",
            "0123456789abcdef",
            "fnv64:0123456789abc",
            "fnv64:0123456789abcdeg",
            "fnv64:0123456789ABCDEF",
            "fnv64:0123456789abcdef0",
            " fnv64:0123456789abcdef",
            "fnv64:0123456789abcdef ",
            "fnv64:UNHASHABLE-test",
        ] {
            assert!(
                !valid_search_config_identity(invalid),
                "non-canonical config identity was accepted: {invalid}"
            );
        }
    }
    fn unbacked_capture() -> (
        FinalistQuoteReplayAcquisitionOutcomeV1,
        ReviewedBrokerFinancialTruthEvidenceV2,
    ) {
        let digest = "ab".repeat(32);
        let window = EvidenceWindowV1::new(1_000, 4_000).expect("fixture coverage");
        let rule = ReviewedQuoteReplayRuleIdentityV2::new(&digest, &digest, &digest)
            .expect("well-formed review identifiers, not backed evidence");
        let synchronization = BrokerTruthReviewedSynchronizationBindingV1::new(
            0,
            7,
            42,
            window,
            rule.clone(),
            &digest,
        )
        .expect("fixture synchronization binding");
        let reviewed = ReviewedBrokerFinancialTruthEvidenceV2::checked_new(
            &digest,
            &digest,
            &digest,
            &digest,
            &digest,
            &digest,
            &digest,
            &digest,
            &digest,
            window,
            vec![synchronization],
        )
        .expect("well-formed independent review metadata");
        let authority_receipt = BrokerTruthAcquisitionAuthorityReceiptV1::from_json_bytes(
            &serde_json::to_vec(&json!({
                "authority_id": format!("bfta1-{digest}"), "manifest_sha256": digest,
            }))
            .expect("fixture authority receipt bytes"),
        )
        .expect("well-formed unbacked authority receipt");
        let broker_truth_receipt = BrokerFinancialTruthBundleReceiptV2::from_json_bytes(
            &serde_json::to_vec(&json!({
                "bundle_id": format!("bft2-{digest}"), "manifest_sha256": digest,
            }))
            .expect("fixture BFT2 receipt bytes"),
        )
        .expect("well-formed unbacked BFT2 receipt");
        let acquisition_link_receipt = BrokerTruthAcquisitionLinkReceiptV1::from_json_bytes(
            &serde_json::to_vec(&json!({
                "link_id": format!("bftl1-{digest}"), "manifest_sha256": digest,
            }))
            .expect("fixture link receipt bytes"),
        )
        .expect("well-formed unbacked link receipt");
        let scope = LockedFinalistOosReplayScopeV1::new(
            EvidenceWindowV1::new(2_000, 3_000).expect("fixture locked window"),
            1_000,
            1_000,
        )
        .expect("fixture padded scope");
        let replay_binding = QuoteValidatedResearchReplayBindingV1::new(
            &digest, &digest, 7, 42, "EURUSD", scope, rule, &digest,
        )
        .expect("fixture replay binding");
        let replay_policy = QuoteValidatedResearchReplayPolicyV1::new(
            100,
            500,
            100,
            VersionedLatencySlippagePolicyV1::new("unbacked-test-only", 0, 0, 0.0, 0.0001)
                .expect("fixture execution assumptions"),
            None,
        )
        .expect("fixture locked execution policy");
        (
            FinalistQuoteReplayAcquisitionOutcomeV1 {
                authority_receipt,
                broker_truth_receipt,
                acquisition_link_receipt,
                replay_binding,
                replay_policy,
                artifact_class: FinalistQuoteReplayArtifactClassV1::ResearchOnly,
                semantic_status: BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly,
                promotion_eligibility: BrokerTruthPromotionEligibilityV1::NotPromotionEligible,
                portfolio_identity_sha256: digest.clone(),
                search_config_hash: digest.clone(),
                holdout_scope_identity_sha256: digest,
            },
            reviewed,
        )
    }

    #[test]
    fn decision_outside_captured_scope_refuses_before_evidence_io() {
        let root = tempfile::tempdir().expect("isolated empty test directory");
        let absent_store = root.path().join("absent-store");
        let (capture, reviewed) = unbacked_capture();
        let error = capture
            .replay_reviewed_decision_v1(
                &BrokerTruthAcquisitionStoreV1::new(&absent_store),
                reviewed,
                CanonicalBarSignalResearchDecisionV1::new(
                    3_100,
                    3_200,
                    ResearchPositionDirectionV1::Long,
                    1.0,
                    1.5,
                )
                .expect("well-formed decision outside the locked capture"),
                Vec::new(),
            )
            .expect_err("invalid plan must fail before trying the absent immutable store");
        assert_eq!(
            error.code(),
            FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest
        );
        assert!(!absent_store.exists());
    }

    #[test]
    fn signal_lane_preflights_then_requires_real_immutable_review_evidence() {
        for invalid_scope in [true, false] {
            let root = tempfile::tempdir().expect("isolated empty lane test directory");
            let absent_store = root.path().join("absent-store");
            let (mut capture, reviewed) = unbacked_capture();
            let digest = "ab".repeat(32);
            capture.replay_binding = QuoteValidatedResearchReplayBindingV1::new(
                &digest,
                &digest,
                7,
                42,
                "EURUSD",
                LockedFinalistOosReplayScopeV1::new(
                    EvidenceWindowV1::new(60_000, 180_000).unwrap(),
                    1_000,
                    1_000,
                )
                .unwrap(),
                ReviewedQuoteReplayRuleIdentityV2::new(&digest, &digest, &digest).unwrap(),
                &digest,
            )
            .unwrap();
            let mut bars: Vec<_> = [60_000, 120_000]
                .into_iter()
                .map(|ts| neoethos_trader::LiveBar {
                    symbol: "EURUSD".to_owned(),
                    tf: "M1".to_owned(),
                    o: 1.0,
                    h: 1.001,
                    l: 0.999,
                    c: 1.0,
                    volume: 0.0,
                    ts,
                })
                .collect();
            if invalid_scope {
                bars[0].ts += 1;
            }
            let gene = neoethos_search::Gene {
                strategy_id: "unbacked-lane".to_owned(),
                sl_pips: 20.0,
                tp_pips: 40.0,
                ..Default::default()
            };
            let lane = neoethos_trader::data_replay::CanonicalSignalQuoteLaneV1 {
                bars: &bars,
                signals: &[1, 0],
                gene: &gene,
                pip_size: 0.0001,
                max_hold_bars: 1,
                trailing: None,
            };
            let error = capture
                .replay_reviewed_signal_lane_v1(
                    &BrokerTruthAcquisitionStoreV1::new(&absent_store),
                    reviewed,
                    &lane,
                )
                .expect_err(
                    "lane inputs or review metadata alone cannot manufacture execution evidence",
                );
            assert_eq!(
                error.code(),
                if invalid_scope {
                    FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest
                } else {
                    FinalistQuoteReplayAcquisitionErrorCodeV1::ReviewValidationFailed
                }
            );
            if invalid_scope {
                assert!(
                    !absent_store.exists(),
                    "invalid lane must refuse before evidence IO"
                );
            } else {
                // The existing store initializes its empty root when opening.
                // That is not capture/publication; no authority object exists.
                assert_eq!(
                    std::fs::read_dir(&absent_store).unwrap().count(),
                    0,
                    "review failure must leave the initialized store empty"
                );
            }
        }
    }

    #[test]
    fn review_metadata_without_immutable_evidence_never_reaches_replay() {
        let root = tempfile::tempdir().expect("isolated empty test directory");
        let (capture, reviewed) = unbacked_capture();
        let error = capture
            .replay_reviewed_decision_v1(
                &BrokerTruthAcquisitionStoreV1::new(root.path()),
                reviewed,
                CanonicalBarSignalResearchDecisionV1::new(
                    2_100,
                    2_200,
                    ResearchPositionDirectionV1::Long,
                    1.0,
                    1.5,
                )
                .expect("well-formed decision inside the locked capture"),
                Vec::new(),
            )
            .expect_err("metadata and valid-looking digests alone must never authorize replay");
        assert_eq!(
            error.code(),
            FinalistQuoteReplayAcquisitionErrorCodeV1::ReviewValidationFailed
        );
        assert_eq!(
            std::fs::read_dir(root.path())
                .expect("inspect only the test store")
                .count(),
            0,
            "the review consumer performs no capture or publication"
        );
    }

    #[test]
    fn every_sequence_decision_is_preflighted_before_evidence_io() {
        for times in [
            Vec::new(),
            vec![2_200, 2_200],
            vec![2_300, 2_200],
            vec![2_200, 3_200],
        ] {
            let root = tempfile::tempdir().expect("isolated empty test directory");
            let absent_store = root.path().join("absent-store");
            let (capture, reviewed) = unbacked_capture();
            let decisions = times
                .into_iter()
                .map(|at| {
                    (
                        CanonicalBarSignalResearchDecisionV1::new(
                            at - 100,
                            at,
                            ResearchPositionDirectionV1::Long,
                            1.0,
                            1.5,
                        )
                        .expect("individually well-formed decision"),
                        Vec::new(),
                    )
                })
                .collect();
            let error = capture
                .replay_reviewed_decision_sequence_v1(
                    &BrokerTruthAcquisitionStoreV1::new(&absent_store),
                    reviewed,
                    decisions,
                )
                .expect_err("all malformed plans must fail before the absent store is touched");
            assert_eq!(
                error.code(),
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest
            );
            assert!(!absent_store.exists());
            assert_eq!(
                std::fs::read_dir(root.path())
                    .expect("test directory")
                    .count(),
                0
            );
        }
    }
}
