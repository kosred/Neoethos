use std::{error::Error, fmt};

use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::{Deserialize, Serialize};

use crate::acquisition_store_v1::{
    BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionStoreV1,
};
use crate::contracts::{EvidenceWindowV1, QuoteSideV1, sha256_bytes, validate_sha256_hex};
use crate::contracts_v2::ReviewedQuoteReplayRuleIdentityV2;
use crate::semantic_v2::{
    StructurallyVerifiedQuoteSideReplayV2, inspect_untrusted_broker_financial_truth_bundle_v2,
};
use crate::store::BrokerFinancialTruthBundleStoreV1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedResearchReplayErrorCodeV1 {
    InvalidContract,
    InvalidSha256,
    InvalidWindow,
    InvalidQuote,
    InvalidQuoteOrder,
    InvalidPolicy,
    InvalidDecision,
    InvalidEncoding,
    MissingZeroRowWindowProof,
    IncompleteCoverage,
    ArtifactDigestMismatch,
    BindingMismatch,
    RequiredCoverageWindowMismatch,
    AmbiguousSameTimestampCrossSideOutcome,
    CrossedSynchronizedBook,
    ModeledEntryOutsideDecisionBounds,
    ExitReferenceUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuoteValidatedResearchReplayErrorV1 {
    code: QuoteValidatedResearchReplayErrorCodeV1,
    detail: String,
}

impl QuoteValidatedResearchReplayErrorV1 {
    fn new(code: QuoteValidatedResearchReplayErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> QuoteValidatedResearchReplayErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for QuoteValidatedResearchReplayErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "quote-validated research replay: {}",
            self.detail
        )
    }
}

impl Error for QuoteValidatedResearchReplayErrorV1 {}

fn replay_error(
    code: QuoteValidatedResearchReplayErrorCodeV1,
    detail: impl Into<String>,
) -> QuoteValidatedResearchReplayErrorV1 {
    QuoteValidatedResearchReplayErrorV1::new(code, detail)
}

fn validate_digest(label: &str, digest: &str) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
    validate_sha256_hex(label, digest).map_err(|error| {
        replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::InvalidSha256,
            error.to_string(),
        )
    })
}

fn hash_json<T: Serialize>(
    label: &str,
    value: &T,
) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::InvalidEncoding,
            format!("cannot encode {label}: {error}"),
        )
    })?;
    let mut domain_separated = format!("neoethos-{label}-v1\n").into_bytes();
    domain_separated.extend_from_slice(&encoded);
    Ok(sha256_bytes(&domain_separated))
}

fn validate_window(window: EvidenceWindowV1) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
    window.validate().map_err(|error| {
        replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::InvalidWindow,
            error.to_string(),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactQuoteSourceOrdinalV1 {
    request_chunk_index: u64,
    response_page_index: u64,
    row_index: u64,
}

impl ExactQuoteSourceOrdinalV1 {
    pub fn new(
        request_chunk_index: u64,
        response_page_index: u64,
        row_index: u64,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        Ok(Self {
            request_chunk_index,
            response_page_index,
            row_index,
        })
    }

    pub const fn request_chunk_index(self) -> u64 {
        self.request_chunk_index
    }

    pub const fn response_page_index(self) -> u64 {
        self.response_page_index
    }

    pub const fn row_index(self) -> u64 {
        self.row_index
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactHistoricalQuoteV1 {
    timestamp_unix_ms: i64,
    price: f64,
    source_ordinal: ExactQuoteSourceOrdinalV1,
}

impl ExactHistoricalQuoteV1 {
    pub fn new(
        timestamp_unix_ms: i64,
        price: f64,
        source_ordinal: ExactQuoteSourceOrdinalV1,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let quote = Self {
            timestamp_unix_ms,
            price,
            source_ordinal,
        };
        quote.validate()?;
        Ok(quote)
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.timestamp_unix_ms < 0 || !self.price.is_finite() || self.price <= 0.0 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidQuote,
                "historical quote timestamp and price must be finite positive values",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedFinalistOosReplayScopeV1 {
    locked_evaluation_window: EvidenceWindowV1,
    required_quote_coverage_window: EvidenceWindowV1,
    seed_padding_ms: i64,
    exit_padding_ms: i64,
}

impl LockedFinalistOosReplayScopeV1 {
    pub fn new(
        locked_evaluation_window: EvidenceWindowV1,
        seed_padding_ms: i64,
        exit_padding_ms: i64,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        validate_window(locked_evaluation_window)?;
        if seed_padding_ms < 0 || exit_padding_ms < 0 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidWindow,
                "seed and exit padding must be non-negative",
            ));
        }
        let from_unix_ms_inclusive = locked_evaluation_window
            .from_unix_ms_inclusive()
            .checked_sub(seed_padding_ms)
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidWindow,
                    "seed padding underflows the quote coverage window",
                )
            })?;
        let to_unix_ms_exclusive = locked_evaluation_window
            .to_unix_ms_exclusive()
            .checked_add(exit_padding_ms)
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidWindow,
                    "exit padding overflows the quote coverage window",
                )
            })?;
        let required_quote_coverage_window =
            EvidenceWindowV1::new(from_unix_ms_inclusive, to_unix_ms_exclusive).map_err(
                |error| {
                    replay_error(
                        QuoteValidatedResearchReplayErrorCodeV1::InvalidWindow,
                        error.to_string(),
                    )
                },
            )?;
        Ok(Self {
            locked_evaluation_window,
            required_quote_coverage_window,
            seed_padding_ms,
            exit_padding_ms,
        })
    }

    pub const fn locked_evaluation_window(self) -> EvidenceWindowV1 {
        self.locked_evaluation_window
    }

    pub const fn required_quote_coverage_window(self) -> EvidenceWindowV1 {
        self.required_quote_coverage_window
    }

    pub const fn seed_padding_ms(self) -> i64 {
        self.seed_padding_ms
    }

    pub const fn exit_padding_ms(self) -> i64 {
        self.exit_padding_ms
    }

    fn validate(self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        let recomputed = Self::new(
            self.locked_evaluation_window,
            self.seed_padding_ms,
            self.exit_padding_ms,
        )?;
        if recomputed.required_quote_coverage_window != self.required_quote_coverage_window {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::RequiredCoverageWindowMismatch,
                "stored quote coverage window does not match the locked window and padding",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteValidatedResearchReplayBindingV1 {
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    account_id: i64,
    symbol_id: i64,
    symbol_name: String,
    replay_scope: LockedFinalistOosReplayScopeV1,
    reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
    quote_evidence_manifest_sha256: String,
    identity_sha256: String,
}

#[derive(Serialize)]
struct ReplayBindingHashPayloadV1<'a> {
    canonical_search_input_receipt_sha256: &'a str,
    canonical_signal_plan_sha256: &'a str,
    account_id: i64,
    symbol_id: i64,
    symbol_name: &'a str,
    replay_scope: LockedFinalistOosReplayScopeV1,
    reviewed_replay_rule_identity_sha256: &'a str,
    quote_evidence_manifest_sha256: &'a str,
}

impl QuoteValidatedResearchReplayBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        canonical_search_input_receipt_sha256: impl Into<String>,
        canonical_signal_plan_sha256: impl Into<String>,
        account_id: i64,
        symbol_id: i64,
        symbol_name: impl Into<String>,
        replay_scope: LockedFinalistOosReplayScopeV1,
        reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
        quote_evidence_manifest_sha256: impl Into<String>,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let mut binding = Self {
            canonical_search_input_receipt_sha256: canonical_search_input_receipt_sha256.into(),
            canonical_signal_plan_sha256: canonical_signal_plan_sha256.into(),
            account_id,
            symbol_id,
            symbol_name: symbol_name.into(),
            replay_scope,
            reviewed_replay_rule,
            quote_evidence_manifest_sha256: quote_evidence_manifest_sha256.into(),
            identity_sha256: String::new(),
        };
        binding.validate_payload()?;
        binding.identity_sha256 = binding.recomputed_identity_sha256()?;
        Ok(binding)
    }

    fn hash_payload(&self) -> ReplayBindingHashPayloadV1<'_> {
        ReplayBindingHashPayloadV1 {
            canonical_search_input_receipt_sha256: &self.canonical_search_input_receipt_sha256,
            canonical_signal_plan_sha256: &self.canonical_signal_plan_sha256,
            account_id: self.account_id,
            symbol_id: self.symbol_id,
            symbol_name: &self.symbol_name,
            replay_scope: self.replay_scope,
            reviewed_replay_rule_identity_sha256: self.reviewed_replay_rule.identity_sha256(),
            quote_evidence_manifest_sha256: &self.quote_evidence_manifest_sha256,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
        hash_json("quote replay binding", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        validate_digest(
            "canonical search input receipt",
            &self.canonical_search_input_receipt_sha256,
        )?;
        validate_digest("canonical signal plan", &self.canonical_signal_plan_sha256)?;
        validate_digest(
            "quote evidence manifest",
            &self.quote_evidence_manifest_sha256,
        )?;
        if self.account_id <= 0 || self.symbol_id <= 0 || self.symbol_name.trim().is_empty() {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                "account, symbol id and symbol name must be explicit",
            ));
        }
        self.replay_scope.validate()?;
        self.reviewed_replay_rule.validate_exact().map_err(|error| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                error.to_string(),
            )
        })
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.validate_payload()?;
        validate_digest("quote replay binding identity", &self.identity_sha256)?;
        if self.recomputed_identity_sha256()? != self.identity_sha256 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                "quote replay binding identity does not match its exact fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactZeroRowQuoteWindowProofV1 {
    side: QuoteSideV1,
    account_id: i64,
    symbol_id: i64,
    requested_window: EvidenceWindowV1,
    raw_response_sha256: String,
    decoded_records_sha256: String,
    row_count: u64,
    response_has_more: bool,
}

impl ExactZeroRowQuoteWindowProofV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: QuoteSideV1,
        account_id: i64,
        symbol_id: i64,
        requested_window: EvidenceWindowV1,
        raw_response_sha256: impl Into<String>,
        decoded_records_sha256: impl Into<String>,
        row_count: u64,
        response_has_more: bool,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let proof = Self {
            side,
            account_id,
            symbol_id,
            requested_window,
            raw_response_sha256: raw_response_sha256.into(),
            decoded_records_sha256: decoded_records_sha256.into(),
            row_count,
            response_has_more,
        };
        proof.validate()?;
        Ok(proof)
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.response_has_more {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::IncompleteCoverage,
                "zero-row response still reports more broker history",
            ));
        }
        if self.row_count != 0 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                "zero-row proof has a non-zero row count",
            ));
        }
        if self.account_id <= 0 || self.symbol_id <= 0 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                "zero-row proof has no exact account or symbol",
            ));
        }
        validate_window(self.requested_window)?;
        validate_digest("zero-row raw response", &self.raw_response_sha256)?;
        validate_digest("zero-row decoded records", &self.decoded_records_sha256)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompleteQuoteSideCoverageV1 {
    side: QuoteSideV1,
    account_id: i64,
    symbol_id: i64,
    requested_window: EvidenceWindowV1,
    raw_response_sha256: String,
    decoded_records_sha256: String,
    quote_records: Vec<ExactHistoricalQuoteV1>,
    terminal_response_has_more: bool,
    zero_row_proof: Option<ExactZeroRowQuoteWindowProofV1>,
    content_sha256: String,
}

#[derive(Serialize)]
struct QuoteSideCoverageHashPayloadV1<'a> {
    side: QuoteSideV1,
    account_id: i64,
    symbol_id: i64,
    requested_window: EvidenceWindowV1,
    raw_response_sha256: &'a str,
    decoded_records_sha256: &'a str,
    quote_records: &'a [ExactHistoricalQuoteV1],
    terminal_response_has_more: bool,
    zero_row_proof: &'a Option<ExactZeroRowQuoteWindowProofV1>,
}

impl CompleteQuoteSideCoverageV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: QuoteSideV1,
        account_id: i64,
        symbol_id: i64,
        requested_window: EvidenceWindowV1,
        raw_response_sha256: impl Into<String>,
        decoded_records_sha256: impl Into<String>,
        quote_records: Vec<ExactHistoricalQuoteV1>,
        terminal_response_has_more: bool,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let mut coverage = Self {
            side,
            account_id,
            symbol_id,
            requested_window,
            raw_response_sha256: raw_response_sha256.into(),
            decoded_records_sha256: decoded_records_sha256.into(),
            quote_records,
            terminal_response_has_more,
            zero_row_proof: None,
            content_sha256: String::new(),
        };
        coverage.validate_payload()?;
        coverage.content_sha256 = coverage.recomputed_content_sha256()?;
        Ok(coverage)
    }

    pub fn empty(
        proof: ExactZeroRowQuoteWindowProofV1,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        proof.validate()?;
        let mut coverage = Self {
            side: proof.side,
            account_id: proof.account_id,
            symbol_id: proof.symbol_id,
            requested_window: proof.requested_window,
            raw_response_sha256: proof.raw_response_sha256.clone(),
            decoded_records_sha256: proof.decoded_records_sha256.clone(),
            quote_records: Vec::new(),
            terminal_response_has_more: proof.response_has_more,
            zero_row_proof: Some(proof),
            content_sha256: String::new(),
        };
        coverage.validate_payload()?;
        coverage.content_sha256 = coverage.recomputed_content_sha256()?;
        Ok(coverage)
    }

    pub fn event_count(&self) -> usize {
        self.quote_records.len()
    }

    pub const fn is_explicit_zero_row_window(&self) -> bool {
        self.zero_row_proof.is_some()
    }

    fn hash_payload(&self) -> QuoteSideCoverageHashPayloadV1<'_> {
        QuoteSideCoverageHashPayloadV1 {
            side: self.side,
            account_id: self.account_id,
            symbol_id: self.symbol_id,
            requested_window: self.requested_window,
            raw_response_sha256: &self.raw_response_sha256,
            decoded_records_sha256: &self.decoded_records_sha256,
            quote_records: &self.quote_records,
            terminal_response_has_more: self.terminal_response_has_more,
            zero_row_proof: &self.zero_row_proof,
        }
    }

    fn recomputed_content_sha256(&self) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
        hash_json("complete quote-side coverage", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.terminal_response_has_more {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::IncompleteCoverage,
                "terminal quote response still reports more history",
            ));
        }
        if self.account_id <= 0 || self.symbol_id <= 0 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                "quote coverage has no exact account or symbol",
            ));
        }
        validate_window(self.requested_window)?;
        validate_digest("raw quote response", &self.raw_response_sha256)?;
        validate_digest("decoded quote records", &self.decoded_records_sha256)?;

        if self.quote_records.is_empty() {
            let proof = self.zero_row_proof.as_ref().ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::MissingZeroRowWindowProof,
                    "empty quote coverage has no terminal zero-row broker proof",
                )
            })?;
            proof.validate()?;
            if proof.side != self.side
                || proof.account_id != self.account_id
                || proof.symbol_id != self.symbol_id
                || proof.requested_window != self.requested_window
                || proof.raw_response_sha256 != self.raw_response_sha256
                || proof.decoded_records_sha256 != self.decoded_records_sha256
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                    "zero-row proof does not match its quote-side coverage",
                ));
            }
        } else if self.zero_row_proof.is_some() {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidContract,
                "non-empty quote coverage cannot carry a zero-row proof",
            ));
        }

        for quote in &self.quote_records {
            quote.validate()?;
            if quote.timestamp_unix_ms < self.requested_window.from_unix_ms_inclusive()
                || quote.timestamp_unix_ms >= self.requested_window.to_unix_ms_exclusive()
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidQuote,
                    "quote falls outside its declared complete coverage window",
                ));
            }
        }
        for pair in self.quote_records.windows(2) {
            let previous = (&pair[0].timestamp_unix_ms, &pair[0].source_ordinal);
            let current = (&pair[1].timestamp_unix_ms, &pair[1].source_ordinal);
            if previous >= current {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidQuoteOrder,
                    "quote records are not strictly ordered by timestamp and source ordinal",
                ));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.validate_payload()?;
        validate_digest("quote-side content", &self.content_sha256)?;
        if self.recomputed_content_sha256()? != self.content_sha256 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch,
                "quote-side content does not match its sealed digest",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompleteBidAskQuoteReplayEvidenceV1 {
    binding: QuoteValidatedResearchReplayBindingV1,
    bid: CompleteQuoteSideCoverageV1,
    ask: CompleteQuoteSideCoverageV1,
    content_sha256: String,
}

#[derive(Serialize)]
struct BidAskEvidenceHashPayloadV1<'a> {
    binding_identity_sha256: &'a str,
    bid_content_sha256: &'a str,
    ask_content_sha256: &'a str,
}

impl CompleteBidAskQuoteReplayEvidenceV1 {
    pub fn new(
        binding: QuoteValidatedResearchReplayBindingV1,
        bid: CompleteQuoteSideCoverageV1,
        ask: CompleteQuoteSideCoverageV1,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let mut evidence = Self {
            binding,
            bid,
            ask,
            content_sha256: String::new(),
        };
        evidence.validate_payload()?;
        evidence.content_sha256 = evidence.recomputed_content_sha256()?;
        Ok(evidence)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let evidence: Self = serde_json::from_slice(bytes).map_err(|error| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidEncoding,
                format!("cannot decode complete Bid/Ask quote evidence: {error}"),
            )
        })?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, QuoteValidatedResearchReplayErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidEncoding,
                format!("cannot encode complete Bid/Ask quote evidence: {error}"),
            )
        })
    }

    fn hash_payload(&self) -> BidAskEvidenceHashPayloadV1<'_> {
        BidAskEvidenceHashPayloadV1 {
            binding_identity_sha256: &self.binding.identity_sha256,
            bid_content_sha256: &self.bid.content_sha256,
            ask_content_sha256: &self.ask.content_sha256,
        }
    }

    fn recomputed_content_sha256(&self) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
        hash_json("complete Bid/Ask quote evidence", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.binding.validate()?;
        self.bid.validate()?;
        self.ask.validate()?;
        if self.bid.side != QuoteSideV1::Bid || self.ask.side != QuoteSideV1::Ask {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                "complete quote evidence must contain exact Bid then Ask sides",
            ));
        }
        for coverage in [&self.bid, &self.ask] {
            if coverage.account_id != self.binding.account_id
                || coverage.symbol_id != self.binding.symbol_id
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                    "quote-side account or symbol differs from the replay binding",
                ));
            }
            if coverage.requested_window
                != self.binding.replay_scope.required_quote_coverage_window()
            {
                let code = if coverage.requested_window
                    == self.binding.replay_scope.locked_evaluation_window()
                {
                    QuoteValidatedResearchReplayErrorCodeV1::RequiredCoverageWindowMismatch
                } else {
                    QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch
                };
                return Err(replay_error(
                    code,
                    "quote-side coverage is not the complete locked window with padding",
                ));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.validate_payload()?;
        validate_digest("complete Bid/Ask quote evidence", &self.content_sha256)?;
        if self.recomputed_content_sha256()? != self.content_sha256 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch,
                "complete Bid/Ask quote evidence does not match its sealed digest",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchPositionDirectionV1 {
    Long,
    Short,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalBarSignalResearchDecisionV1 {
    signal_bar_open_unix_ms: i64,
    next_canonical_bar_open_unix_ms: i64,
    direction: ResearchPositionDirectionV1,
    stop_price: f64,
    target_price: f64,
}

impl CanonicalBarSignalResearchDecisionV1 {
    pub fn new(
        signal_bar_open_unix_ms: i64,
        next_canonical_bar_open_unix_ms: i64,
        direction: ResearchPositionDirectionV1,
        stop_price: f64,
        target_price: f64,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let decision = Self {
            signal_bar_open_unix_ms,
            next_canonical_bar_open_unix_ms,
            direction,
            stop_price,
            target_price,
        };
        decision.validate()?;
        Ok(decision)
    }

    pub const fn signal_bar_open_unix_ms(&self) -> i64 {
        self.signal_bar_open_unix_ms
    }

    pub const fn next_canonical_bar_open_unix_ms(&self) -> i64 {
        self.next_canonical_bar_open_unix_ms
    }

    pub const fn decision_at_unix_ms(&self) -> i64 {
        self.next_canonical_bar_open_unix_ms
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.signal_bar_open_unix_ms < 0
            || self.next_canonical_bar_open_unix_ms <= self.signal_bar_open_unix_ms
            || !self.stop_price.is_finite()
            || self.stop_price <= 0.0
            || !self.target_price.is_finite()
            || self.target_price <= 0.0
        {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                "canonical-bar decision has invalid time or price fields",
            ));
        }
        let thresholds_are_ordered = match self.direction {
            ResearchPositionDirectionV1::Long => self.stop_price < self.target_price,
            ResearchPositionDirectionV1::Short => self.target_price < self.stop_price,
        };
        if !thresholds_are_ordered {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                "stop and target are inconsistent with research position direction",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedCanonicalBarTrailingThresholdV1 {
    source_bar_open_unix_ms: i64,
    effective_at_next_bar_open_unix_ms: i64,
    direction: ResearchPositionDirectionV1,
    threshold_price: f64,
}

impl ClosedCanonicalBarTrailingThresholdV1 {
    pub fn new(
        source_bar_open_unix_ms: i64,
        effective_at_next_bar_open_unix_ms: i64,
        direction: ResearchPositionDirectionV1,
        threshold_price: f64,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let threshold = Self {
            source_bar_open_unix_ms,
            effective_at_next_bar_open_unix_ms,
            direction,
            threshold_price,
        };
        threshold.validate()?;
        Ok(threshold)
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.source_bar_open_unix_ms < 0
            || self.effective_at_next_bar_open_unix_ms <= self.source_bar_open_unix_ms
            || !self.threshold_price.is_finite()
            || self.threshold_price <= 0.0
        {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                "closed-bar trailing threshold has invalid causal fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedLatencySlippagePolicyV1 {
    policy_version: String,
    entry_latency_ms: i64,
    exit_latency_ms: i64,
    slippage_pips_per_fill: f64,
    pip_size: f64,
    identity_sha256: String,
}

#[derive(Serialize)]
struct LatencySlippageHashPayloadV1<'a> {
    policy_version: &'a str,
    entry_latency_ms: i64,
    exit_latency_ms: i64,
    slippage_pips_per_fill: f64,
    pip_size: f64,
}

impl VersionedLatencySlippagePolicyV1 {
    pub fn new(
        policy_version: impl Into<String>,
        entry_latency_ms: i64,
        exit_latency_ms: i64,
        slippage_pips_per_fill: f64,
        pip_size: f64,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let mut policy = Self {
            policy_version: policy_version.into(),
            entry_latency_ms,
            exit_latency_ms,
            slippage_pips_per_fill,
            pip_size,
            identity_sha256: String::new(),
        };
        policy.validate_payload()?;
        policy.identity_sha256 = policy.recomputed_identity_sha256()?;
        Ok(policy)
    }

    pub fn policy_version(&self) -> &str {
        &self.policy_version
    }

    pub const fn entry_latency_ms(&self) -> i64 {
        self.entry_latency_ms
    }

    pub const fn exit_latency_ms(&self) -> i64 {
        self.exit_latency_ms
    }

    pub const fn slippage_pips_per_fill(&self) -> f64 {
        self.slippage_pips_per_fill
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    fn hash_payload(&self) -> LatencySlippageHashPayloadV1<'_> {
        LatencySlippageHashPayloadV1 {
            policy_version: &self.policy_version,
            entry_latency_ms: self.entry_latency_ms,
            exit_latency_ms: self.exit_latency_ms,
            slippage_pips_per_fill: self.slippage_pips_per_fill,
            pip_size: self.pip_size,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
        hash_json("latency/slippage policy", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.policy_version.trim().is_empty()
            || self.entry_latency_ms < 0
            || self.exit_latency_ms < 0
            || !self.slippage_pips_per_fill.is_finite()
            || self.slippage_pips_per_fill < 0.0
            || !self.pip_size.is_finite()
            || self.pip_size <= 0.0
        {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                "versioned latency/slippage policy has invalid fields",
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.validate_payload()?;
        validate_digest("latency/slippage policy identity", &self.identity_sha256)?;
        if self.recomputed_identity_sha256()? != self.identity_sha256 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch,
                "latency/slippage policy identity does not match its fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SameTimestampCrossSideOrderV1 {
    BidBeforeAsk,
    AskBeforeBid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedSameTimestampMergeRuleV1 {
    reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
    side_order: SameTimestampCrossSideOrderV1,
}

impl ReviewedSameTimestampMergeRuleV1 {
    pub fn new(
        reviewed_replay_rule: ReviewedQuoteReplayRuleIdentityV2,
        side_order: SameTimestampCrossSideOrderV1,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        reviewed_replay_rule.validate_exact().map_err(|error| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                error.to_string(),
            )
        })?;
        Ok(Self {
            reviewed_replay_rule,
            side_order,
        })
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.reviewed_replay_rule.validate_exact().map_err(|error| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                error.to_string(),
            )
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteValidatedResearchReplayPolicyV1 {
    max_entry_wait_ms: i64,
    max_quote_staleness_ms: i64,
    max_exit_wait_ms: i64,
    latency_slippage: VersionedLatencySlippagePolicyV1,
    reviewed_same_timestamp_merge_rule: Option<ReviewedSameTimestampMergeRuleV1>,
    identity_sha256: String,
}

#[derive(Serialize)]
struct ReplayPolicyHashPayloadV1<'a> {
    max_entry_wait_ms: i64,
    max_quote_staleness_ms: i64,
    max_exit_wait_ms: i64,
    latency_slippage_identity_sha256: &'a str,
    reviewed_same_timestamp_merge_rule: &'a Option<ReviewedSameTimestampMergeRuleV1>,
}

impl QuoteValidatedResearchReplayPolicyV1 {
    pub fn new(
        max_entry_wait_ms: i64,
        max_quote_staleness_ms: i64,
        max_exit_wait_ms: i64,
        latency_slippage: VersionedLatencySlippagePolicyV1,
        reviewed_same_timestamp_merge_rule: Option<ReviewedSameTimestampMergeRuleV1>,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let mut policy = Self {
            max_entry_wait_ms,
            max_quote_staleness_ms,
            max_exit_wait_ms,
            latency_slippage,
            reviewed_same_timestamp_merge_rule,
            identity_sha256: String::new(),
        };
        policy.validate_payload()?;
        policy.identity_sha256 = policy.recomputed_identity_sha256()?;
        Ok(policy)
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    fn hash_payload(&self) -> ReplayPolicyHashPayloadV1<'_> {
        ReplayPolicyHashPayloadV1 {
            max_entry_wait_ms: self.max_entry_wait_ms,
            max_quote_staleness_ms: self.max_quote_staleness_ms,
            max_exit_wait_ms: self.max_exit_wait_ms,
            latency_slippage_identity_sha256: self.latency_slippage.identity_sha256(),
            reviewed_same_timestamp_merge_rule: &self.reviewed_same_timestamp_merge_rule,
        }
    }

    fn recomputed_identity_sha256(&self) -> Result<String, QuoteValidatedResearchReplayErrorV1> {
        hash_json("quote-validated replay policy", &self.hash_payload())
    }

    fn validate_payload(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        if self.max_entry_wait_ms <= 0
            || self.max_quote_staleness_ms < 0
            || self.max_exit_wait_ms <= 0
        {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                "wait limits must be positive and staleness must be non-negative",
            ));
        }
        self.latency_slippage.validate()?;
        if let Some(rule) = &self.reviewed_same_timestamp_merge_rule {
            rule.validate()?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.validate_payload()?;
        validate_digest("quote replay policy identity", &self.identity_sha256)?;
        if self.recomputed_identity_sha256()? != self.identity_sha256 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch,
                "quote replay policy identity does not match its exact fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteValidatedResearchReplayPlanV1 {
    binding: QuoteValidatedResearchReplayBindingV1,
    policy: QuoteValidatedResearchReplayPolicyV1,
    decisions: Vec<CanonicalBarSignalResearchDecisionV1>,
    trailing_thresholds: Vec<ClosedCanonicalBarTrailingThresholdV1>,
}

impl QuoteValidatedResearchReplayPlanV1 {
    pub fn new(
        binding: QuoteValidatedResearchReplayBindingV1,
        policy: QuoteValidatedResearchReplayPolicyV1,
        decisions: Vec<CanonicalBarSignalResearchDecisionV1>,
        trailing_thresholds: Vec<ClosedCanonicalBarTrailingThresholdV1>,
    ) -> Result<Self, QuoteValidatedResearchReplayErrorV1> {
        let plan = Self {
            binding,
            policy,
            decisions,
            trailing_thresholds,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
        self.binding.validate()?;
        self.policy.validate()?;
        if let Some(rule) = &self.policy.reviewed_same_timestamp_merge_rule
            && rule.reviewed_replay_rule.identity_sha256()
                != self.binding.reviewed_replay_rule.identity_sha256()
        {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                "same-timestamp merge rule is not bound to the reviewed replay rule",
            ));
        }
        if self.binding.replay_scope.seed_padding_ms() < self.policy.max_quote_staleness_ms {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::RequiredCoverageWindowMismatch,
                "seed padding is shorter than the synchronized-book staleness horizon",
            ));
        }
        let required_exit_padding = self
            .policy
            .latency_slippage
            .exit_latency_ms
            .checked_add(self.policy.max_exit_wait_ms)
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                    "exit latency and wait overflow the required padding",
                )
            })?;
        if self.binding.replay_scope.exit_padding_ms() < required_exit_padding {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::RequiredCoverageWindowMismatch,
                "exit padding is shorter than the modeled exit horizon",
            ));
        }
        if self.decisions.len() != 1 {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                "V1 quote replay requires exactly one canonical-bar decision",
            ));
        }
        let locked_window = self.binding.replay_scope.locked_evaluation_window();
        for decision in &self.decisions {
            decision.validate()?;
            if decision.signal_bar_open_unix_ms < locked_window.from_unix_ms_inclusive()
                || decision.next_canonical_bar_open_unix_ms >= locked_window.to_unix_ms_exclusive()
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                    "canonical-bar decision is outside the locked evaluation window",
                ));
            }
        }
        for pair in self.decisions.windows(2) {
            if pair[0].next_canonical_bar_open_unix_ms >= pair[1].next_canonical_bar_open_unix_ms {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                    "canonical-bar decisions are not strictly time ordered",
                ));
            }
        }
        let quote_window = self.binding.replay_scope.required_quote_coverage_window();
        let sole_decision = &self.decisions[0];
        for threshold in &self.trailing_thresholds {
            threshold.validate()?;
            if threshold.source_bar_open_unix_ms < locked_window.from_unix_ms_inclusive()
                || threshold.effective_at_next_bar_open_unix_ms
                    >= quote_window.to_unix_ms_exclusive()
                || threshold.direction != sole_decision.direction
                || threshold.effective_at_next_bar_open_unix_ms
                    < sole_decision.next_canonical_bar_open_unix_ms
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                    "closed-bar trailing threshold is outside the sole decision's causal ownership",
                ));
            }
        }
        for pair in self.trailing_thresholds.windows(2) {
            if pair[0].effective_at_next_bar_open_unix_ms
                >= pair[1].effective_at_next_bar_open_unix_ms
            {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                    "closed-bar trailing thresholds are not strictly time ordered",
                ));
            }
        }
        let mut latest_long_threshold: Option<f64> = None;
        let mut latest_short_threshold: Option<f64> = None;
        for threshold in &self.trailing_thresholds {
            let latest = match threshold.direction {
                ResearchPositionDirectionV1::Long => &mut latest_long_threshold,
                ResearchPositionDirectionV1::Short => &mut latest_short_threshold,
            };
            if let Some(previous) = *latest {
                let loosens = match threshold.direction {
                    ResearchPositionDirectionV1::Long => threshold.threshold_price < previous,
                    ResearchPositionDirectionV1::Short => threshold.threshold_price > previous,
                };
                if loosens {
                    return Err(replay_error(
                        QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision,
                        "closed-bar trailing schedule loosens a prior threshold",
                    ));
                }
            }
            *latest = Some(threshold.threshold_price);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedResearchExitReasonV1 {
    Stop,
    Target,
    TrailingStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedResearchNonEntryReasonV1 {
    NoEligibleQuoteWithinEntryWait,
    StaleSynchronizedBook,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedResearchPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteValidatedResearchAuthorityV1 {
    UnverifiedCallerSuppliedQuotes,
    HistoricalBidAskQuotesOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedPriceReferenceV1 {
    side: QuoteSideV1,
    timestamp_unix_ms: i64,
    price: f64,
}

impl QuoteValidatedPriceReferenceV1 {
    fn from_quote(side: QuoteSideV1, quote: &ExactHistoricalQuoteV1) -> Self {
        Self {
            side,
            timestamp_unix_ms: quote.timestamp_unix_ms,
            price: quote.price,
        }
    }

    pub const fn side(&self) -> QuoteSideV1 {
        self.side
    }

    pub const fn timestamp_unix_ms(&self) -> i64 {
        self.timestamp_unix_ms
    }

    pub const fn price(&self) -> f64 {
        self.price
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedEntryBookV1 {
    bid_reference: QuoteValidatedPriceReferenceV1,
    ask_reference: QuoteValidatedPriceReferenceV1,
}

impl QuoteValidatedEntryBookV1 {
    pub fn bid_reference(&self) -> &QuoteValidatedPriceReferenceV1 {
        &self.bid_reference
    }

    pub fn ask_reference(&self) -> &QuoteValidatedPriceReferenceV1 {
        &self.ask_reference
    }

    pub fn quoted_spread_price(&self) -> f64 {
        self.ask_reference.price - self.bid_reference.price
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedResearchPositionV1 {
    direction: ResearchPositionDirectionV1,
    entry_book: QuoteValidatedEntryBookV1,
    entry_reference: QuoteValidatedPriceReferenceV1,
    exit_reference: Option<QuoteValidatedPriceReferenceV1>,
    exit_reason: Option<QuoteValidatedResearchExitReasonV1>,
    modeled_entry_price: f64,
    modeled_exit_price: Option<f64>,
    slippage_pips_charged: f64,
    additional_spread_pips_charged: f64,
}

impl QuoteValidatedResearchPositionV1 {
    pub const fn direction(&self) -> ResearchPositionDirectionV1 {
        self.direction
    }

    pub fn entry_book(&self) -> &QuoteValidatedEntryBookV1 {
        &self.entry_book
    }

    pub fn entry_reference(&self) -> &QuoteValidatedPriceReferenceV1 {
        &self.entry_reference
    }

    pub fn exit_reference(&self) -> Option<&QuoteValidatedPriceReferenceV1> {
        self.exit_reference.as_ref()
    }

    pub const fn exit_reason(&self) -> Option<QuoteValidatedResearchExitReasonV1> {
        self.exit_reason
    }

    pub const fn modeled_entry_price(&self) -> f64 {
        self.modeled_entry_price
    }

    pub const fn modeled_exit_price(&self) -> Option<f64> {
        self.modeled_exit_price
    }

    pub const fn slippage_pips_charged(&self) -> f64 {
        self.slippage_pips_charged
    }

    pub const fn additional_spread_pips_charged(&self) -> f64 {
        self.additional_spread_pips_charged
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct QuoteValidatedResearchNonEntryV1 {
    decision_at_unix_ms: i64,
    deadline_unix_ms: i64,
    reason: QuoteValidatedResearchNonEntryReasonV1,
}

impl QuoteValidatedResearchNonEntryV1 {
    pub const fn decision_at_unix_ms(&self) -> i64 {
        self.decision_at_unix_ms
    }

    pub const fn reason(&self) -> QuoteValidatedResearchNonEntryReasonV1 {
        self.reason
    }

    pub const fn deadline_unix_ms(&self) -> i64 {
        self.deadline_unix_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct QuoteValidatedResearchReplayReceiptV1 {
    canonical_search_input_receipt_sha256: String,
    canonical_signal_plan_sha256: String,
    account_id: i64,
    symbol_id: i64,
    symbol_name: String,
    locked_evaluation_window: EvidenceWindowV1,
    required_quote_coverage_window: EvidenceWindowV1,
    seed_padding_ms: i64,
    exit_padding_ms: i64,
    reviewed_replay_rule_identity_sha256: String,
    quote_evidence_manifest_sha256: String,
    latency_slippage_policy_sha256: String,
    replay_policy_sha256: String,
    authority: QuoteValidatedResearchAuthorityV1,
    historical_acquisition_link_manifest_sha256: Option<String>,
    ledger_sha256: String,
}

impl QuoteValidatedResearchReplayReceiptV1 {
    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub fn canonical_signal_plan_sha256(&self) -> &str {
        &self.canonical_signal_plan_sha256
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub const fn locked_evaluation_window(&self) -> EvidenceWindowV1 {
        self.locked_evaluation_window
    }

    pub const fn required_quote_coverage_window(&self) -> EvidenceWindowV1 {
        self.required_quote_coverage_window
    }

    pub const fn seed_padding_ms(&self) -> i64 {
        self.seed_padding_ms
    }

    pub const fn exit_padding_ms(&self) -> i64 {
        self.exit_padding_ms
    }

    pub fn reviewed_replay_rule_identity_sha256(&self) -> &str {
        &self.reviewed_replay_rule_identity_sha256
    }

    pub fn quote_evidence_manifest_sha256(&self) -> &str {
        &self.quote_evidence_manifest_sha256
    }

    pub fn latency_slippage_policy_sha256(&self) -> &str {
        &self.latency_slippage_policy_sha256
    }

    pub fn replay_policy_sha256(&self) -> &str {
        &self.replay_policy_sha256
    }

    pub const fn authority(&self) -> QuoteValidatedResearchAuthorityV1 {
        self.authority
    }

    pub fn historical_acquisition_link_manifest_sha256(&self) -> Option<&str> {
        self.historical_acquisition_link_manifest_sha256.as_deref()
    }

    pub fn ledger_sha256(&self) -> &str {
        &self.ledger_sha256
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuoteValidatedResearchLedgerV1 {
    receipt: QuoteValidatedResearchReplayReceiptV1,
    positions: Vec<QuoteValidatedResearchPositionV1>,
    entry_unavailable: Vec<QuoteValidatedResearchNonEntryV1>,
    authority: QuoteValidatedResearchAuthorityV1,
    promotion_eligibility: QuoteValidatedResearchPromotionEligibilityV1,
}

impl QuoteValidatedResearchLedgerV1 {
    pub fn receipt(&self) -> &QuoteValidatedResearchReplayReceiptV1 {
        &self.receipt
    }

    pub fn positions(&self) -> &[QuoteValidatedResearchPositionV1] {
        &self.positions
    }

    pub fn entry_unavailable(&self) -> &[QuoteValidatedResearchNonEntryV1] {
        &self.entry_unavailable
    }

    pub const fn authority(&self) -> QuoteValidatedResearchAuthorityV1 {
        self.authority
    }

    pub const fn promotion_eligibility(&self) -> QuoteValidatedResearchPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub fn ledger_sha256(&self) -> &str {
        self.receipt.ledger_sha256()
    }
}

enum EntrySelectionV1 {
    Available(SelectedEntryBookV1),
    Unavailable(QuoteValidatedResearchNonEntryV1),
}

struct SelectedEntryBookV1 {
    entry_quote: ExactHistoricalQuoteV1,
    bid_quote: ExactHistoricalQuoteV1,
    ask_quote: ExactHistoricalQuoteV1,
}

fn side_precedes(
    first: QuoteSideV1,
    second: QuoteSideV1,
    order: SameTimestampCrossSideOrderV1,
) -> bool {
    match order {
        SameTimestampCrossSideOrderV1::BidBeforeAsk => {
            first == QuoteSideV1::Bid && second == QuoteSideV1::Ask
        }
        SameTimestampCrossSideOrderV1::AskBeforeBid => {
            first == QuoteSideV1::Ask && second == QuoteSideV1::Bid
        }
    }
}

fn latest_book_quote_at_entry<'a>(
    opposite_side: QuoteSideV1,
    entry_side: QuoteSideV1,
    quotes: &'a [ExactHistoricalQuoteV1],
    candidate_timestamp_unix_ms: i64,
    policy: &QuoteValidatedResearchReplayPolicyV1,
) -> Result<Option<&'a ExactHistoricalQuoteV1>, QuoteValidatedResearchReplayErrorV1> {
    let mut before = None;
    let mut same_timestamp = None;
    for quote in quotes {
        if quote.timestamp_unix_ms < candidate_timestamp_unix_ms {
            before = Some(quote);
        } else if quote.timestamp_unix_ms == candidate_timestamp_unix_ms {
            same_timestamp = Some(quote);
        } else {
            break;
        }
    }
    let Some(same_timestamp) = same_timestamp else {
        return Ok(before);
    };
    if let Some(rule) = &policy.reviewed_same_timestamp_merge_rule {
        if side_precedes(opposite_side, entry_side, rule.side_order) {
            return Ok(Some(same_timestamp));
        }
        return Ok(before);
    }
    Err(replay_error(
        QuoteValidatedResearchReplayErrorCodeV1::AmbiguousSameTimestampCrossSideOutcome,
        "unreviewed same-timestamp opposite-side update makes the entry book ambiguous",
    ))
}

fn select_entry(
    decision: &CanonicalBarSignalResearchDecisionV1,
    evidence: &CompleteBidAskQuoteReplayEvidenceV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
) -> Result<EntrySelectionV1, QuoteValidatedResearchReplayErrorV1> {
    let (entry_side, entry_quotes, opposite_side, opposite_quotes) = match decision.direction {
        ResearchPositionDirectionV1::Long => (
            QuoteSideV1::Ask,
            evidence.ask.quote_records.as_slice(),
            QuoteSideV1::Bid,
            evidence.bid.quote_records.as_slice(),
        ),
        ResearchPositionDirectionV1::Short => (
            QuoteSideV1::Bid,
            evidence.bid.quote_records.as_slice(),
            QuoteSideV1::Ask,
            evidence.ask.quote_records.as_slice(),
        ),
    };
    let eligible_at = decision
        .next_canonical_bar_open_unix_ms
        .checked_add(policy.latency_slippage.entry_latency_ms)
        .ok_or_else(|| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                "entry latency overflows the decision timestamp",
            )
        })?;
    let deadline = decision
        .next_canonical_bar_open_unix_ms
        .checked_add(policy.max_entry_wait_ms)
        .ok_or_else(|| {
            replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                "entry wait overflows the decision timestamp",
            )
        })?;
    let mut saw_stale_candidate = false;
    for candidate in entry_quotes {
        if candidate.timestamp_unix_ms < eligible_at {
            continue;
        }
        if candidate.timestamp_unix_ms > deadline {
            break;
        }
        let book_quote = latest_book_quote_at_entry(
            opposite_side,
            entry_side,
            opposite_quotes,
            candidate.timestamp_unix_ms,
            policy,
        )?;
        if let Some(book_quote) = book_quote
            && candidate.timestamp_unix_ms - book_quote.timestamp_unix_ms
                <= policy.max_quote_staleness_ms
        {
            let (bid_quote, ask_quote) = match decision.direction {
                ResearchPositionDirectionV1::Long => (book_quote.clone(), candidate.clone()),
                ResearchPositionDirectionV1::Short => (candidate.clone(), book_quote.clone()),
            };
            if bid_quote.price > ask_quote.price {
                return Err(replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::CrossedSynchronizedBook,
                    "synchronized entry evidence has Bid above Ask",
                ));
            }
            return Ok(EntrySelectionV1::Available(SelectedEntryBookV1 {
                entry_quote: candidate.clone(),
                bid_quote,
                ask_quote,
            }));
        }
        saw_stale_candidate = true;
    }
    let reason = if saw_stale_candidate {
        QuoteValidatedResearchNonEntryReasonV1::StaleSynchronizedBook
    } else {
        QuoteValidatedResearchNonEntryReasonV1::NoEligibleQuoteWithinEntryWait
    };
    Ok(EntrySelectionV1::Unavailable(
        QuoteValidatedResearchNonEntryV1 {
            decision_at_unix_ms: decision.next_canonical_bar_open_unix_ms,
            deadline_unix_ms: deadline,
            reason,
        },
    ))
}

fn effective_stop_price(
    decision: &CanonicalBarSignalResearchDecisionV1,
    thresholds: &[ClosedCanonicalBarTrailingThresholdV1],
    timestamp_unix_ms: i64,
) -> (f64, QuoteValidatedResearchExitReasonV1) {
    let mut stop_price = decision.stop_price;
    let mut reason = QuoteValidatedResearchExitReasonV1::Stop;
    for threshold in thresholds {
        if threshold.direction != decision.direction
            || threshold.effective_at_next_bar_open_unix_ms > timestamp_unix_ms
        {
            continue;
        }
        let tightens = match decision.direction {
            ResearchPositionDirectionV1::Long => threshold.threshold_price > stop_price,
            ResearchPositionDirectionV1::Short => threshold.threshold_price < stop_price,
        };
        if tightens {
            stop_price = threshold.threshold_price;
            reason = QuoteValidatedResearchExitReasonV1::TrailingStop;
        }
    }
    (stop_price, reason)
}

fn threshold_reason(
    decision: &CanonicalBarSignalResearchDecisionV1,
    thresholds: &[ClosedCanonicalBarTrailingThresholdV1],
    quote: &ExactHistoricalQuoteV1,
) -> Option<QuoteValidatedResearchExitReasonV1> {
    let (stop_price, stop_reason) =
        effective_stop_price(decision, thresholds, quote.timestamp_unix_ms);
    match decision.direction {
        ResearchPositionDirectionV1::Long => {
            if quote.price <= stop_price {
                Some(stop_reason)
            } else if quote.price >= decision.target_price {
                Some(QuoteValidatedResearchExitReasonV1::Target)
            } else {
                None
            }
        }
        ResearchPositionDirectionV1::Short => {
            if quote.price >= stop_price {
                Some(stop_reason)
            } else if quote.price <= decision.target_price {
                Some(QuoteValidatedResearchExitReasonV1::Target)
            } else {
                None
            }
        }
    }
}

fn same_timestamp_exit_occurs_after_entry(
    entry_side: QuoteSideV1,
    exit_side: QuoteSideV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
) -> Option<bool> {
    policy
        .reviewed_same_timestamp_merge_rule
        .as_ref()
        .map(|rule| side_precedes(entry_side, exit_side, rule.side_order))
}

fn select_exit(
    decision: &CanonicalBarSignalResearchDecisionV1,
    entry_quote: &ExactHistoricalQuoteV1,
    evidence: &CompleteBidAskQuoteReplayEvidenceV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    thresholds: &[ClosedCanonicalBarTrailingThresholdV1],
) -> Result<
    Option<(ExactHistoricalQuoteV1, QuoteValidatedResearchExitReasonV1)>,
    QuoteValidatedResearchReplayErrorV1,
> {
    let (entry_side, exit_side, exit_quotes) = match decision.direction {
        ResearchPositionDirectionV1::Long => (
            QuoteSideV1::Ask,
            QuoteSideV1::Bid,
            evidence.bid.quote_records.as_slice(),
        ),
        ResearchPositionDirectionV1::Short => (
            QuoteSideV1::Bid,
            QuoteSideV1::Ask,
            evidence.ask.quote_records.as_slice(),
        ),
    };
    for (trigger_index, trigger_quote) in exit_quotes.iter().enumerate() {
        if trigger_quote.timestamp_unix_ms < entry_quote.timestamp_unix_ms {
            continue;
        }
        let Some(reason) = threshold_reason(decision, thresholds, trigger_quote) else {
            continue;
        };
        if trigger_quote.timestamp_unix_ms == entry_quote.timestamp_unix_ms {
            match same_timestamp_exit_occurs_after_entry(entry_side, exit_side, policy) {
                Some(true) => {}
                Some(false) => continue,
                None => {
                    return Err(replay_error(
                        QuoteValidatedResearchReplayErrorCodeV1::AmbiguousSameTimestampCrossSideOutcome,
                        "same-timestamp side order changes the modeled position outcome",
                    ));
                }
            }
        }

        let eligible_at = trigger_quote
            .timestamp_unix_ms
            .checked_add(policy.latency_slippage.exit_latency_ms)
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                    "exit latency overflows the trigger timestamp",
                )
            })?;
        let deadline = trigger_quote
            .timestamp_unix_ms
            .checked_add(policy.max_exit_wait_ms)
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
                    "exit wait overflows the trigger timestamp",
                )
            })?;
        let reference = exit_quotes[trigger_index..]
            .iter()
            .find(|quote| {
                quote.timestamp_unix_ms >= eligible_at && quote.timestamp_unix_ms <= deadline
            })
            .cloned()
            .ok_or_else(|| {
                replay_error(
                    QuoteValidatedResearchReplayErrorCodeV1::ExitReferenceUnavailable,
                    "complete quote evidence has no exit reference within the explicit wait",
                )
            })?;
        return Ok(Some((reference, reason)));
    }
    Ok(None)
}

fn modeled_price(
    reference_price: f64,
    direction: ResearchPositionDirectionV1,
    is_entry: bool,
    assumptions: &VersionedLatencySlippagePolicyV1,
) -> Result<f64, QuoteValidatedResearchReplayErrorV1> {
    let adjustment = assumptions.slippage_pips_per_fill * assumptions.pip_size;
    let adverse_sign =
        match (direction, is_entry) {
            (ResearchPositionDirectionV1::Long, true)
            | (ResearchPositionDirectionV1::Short, false) => 1.0,
            (ResearchPositionDirectionV1::Short, true)
            | (ResearchPositionDirectionV1::Long, false) => -1.0,
        };
    let modeled = reference_price + adverse_sign * adjustment;
    if !modeled.is_finite() || modeled <= 0.0 {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::InvalidPolicy,
            "modeled latency/slippage price is not finite and positive",
        ));
    }
    Ok(modeled)
}

#[derive(Debug)]
pub struct SealedHistoricalBidAskQuoteReplayEvidenceV1 {
    evidence: CompleteBidAskQuoteReplayEvidenceV1,
    acquisition_link_manifest_sha256: String,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct SealedHistoricalQuoteValidatedResearchLedgerV1 {
    ledger: QuoteValidatedResearchLedgerV1,
}

impl SealedHistoricalQuoteValidatedResearchLedgerV1 {
    pub fn receipt(&self) -> &QuoteValidatedResearchReplayReceiptV1 {
        self.ledger.receipt()
    }

    pub fn positions(&self) -> &[QuoteValidatedResearchPositionV1] {
        self.ledger.positions()
    }

    pub fn entry_unavailable(&self) -> &[QuoteValidatedResearchNonEntryV1] {
        self.ledger.entry_unavailable()
    }

    pub const fn authority(&self) -> QuoteValidatedResearchAuthorityV1 {
        self.ledger.authority()
    }

    pub const fn promotion_eligibility(&self) -> QuoteValidatedResearchPromotionEligibilityV1 {
        self.ledger.promotion_eligibility()
    }

    pub fn ledger_sha256(&self) -> &str {
        self.ledger.ledger_sha256()
    }
}

fn sealed_ingress_error(
    stage: &str,
    error: impl fmt::Display,
) -> QuoteValidatedResearchReplayErrorV1 {
    replay_error(
        QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch,
        format!("{stage} refused exact immutable evidence: {error}"),
    )
}

fn validate_linked_replay_binding(
    linked_binding: &crate::contracts::BrokerFinancialTruthBindingV1,
    expected: &QuoteValidatedResearchReplayBindingV1,
) -> Result<(), QuoteValidatedResearchReplayErrorV1> {
    let CanonicalDatasetScope::CTrader {
        account_id,
        symbol_id,
        ..
    } = linked_binding.canonical_dataset_identity().scope()
    else {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "linked broker evidence is not bound to a cTrader dataset identity",
        ));
    };
    if linked_binding.canonical_search_input_receipt_sha256()
        != expected.canonical_search_input_receipt_sha256.as_str()
        || *account_id != expected.account_id
        || *symbol_id != expected.symbol_id
        || linked_binding.canonical_dataset_identity().symbol_name()
            != expected.symbol_name.as_str()
        || linked_binding.evaluated_window()
            != expected.replay_scope.required_quote_coverage_window()
    {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "linked broker evidence differs from the exact search/account/symbol/coverage binding",
        ));
    }
    Ok(())
}

fn complete_side_from_structural_ingress(
    side: StructurallyVerifiedQuoteSideReplayV2,
) -> Result<CompleteQuoteSideCoverageV1, QuoteValidatedResearchReplayErrorV1> {
    let quote_records = side
        .quote_records
        .into_iter()
        .map(|row| {
            ExactHistoricalQuoteV1::new(
                row.timestamp_unix_ms,
                row.price,
                ExactQuoteSourceOrdinalV1::new(
                    row.request_chunk_index,
                    row.response_page_index,
                    row.row_index,
                )?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    CompleteQuoteSideCoverageV1::new(
        side.side,
        side.account_id,
        side.symbol_id,
        side.requested_window,
        side.raw_response_sha256,
        side.decoded_records_sha256,
        quote_records,
        false,
    )
}

pub fn open_sealed_historical_bid_ask_quote_replay_evidence_v1(
    acquisition_store: &BrokerTruthAcquisitionStoreV1,
    link_receipt: &BrokerTruthAcquisitionLinkReceiptV1,
    expected_replay_binding: &QuoteValidatedResearchReplayBindingV1,
) -> Result<SealedHistoricalBidAskQuoteReplayEvidenceV1, QuoteValidatedResearchReplayErrorV1> {
    expected_replay_binding.validate()?;
    let verified_link = acquisition_store
        .open_link(link_receipt)
        .map_err(|error| sealed_ingress_error("acquisition link reopen", error))?;
    validate_linked_replay_binding(verified_link.manifest().binding(), expected_replay_binding)?;
    if verified_link
        .manifest()
        .broker_truth_receipt()
        .manifest_sha256()
        != expected_replay_binding
            .quote_evidence_manifest_sha256
            .as_str()
    {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "linked broker-truth manifest differs from the exact replay binding",
        ));
    }

    let authority_receipt = verified_link.manifest().authority_receipt().clone();
    let broker_truth_receipt = verified_link.manifest().broker_truth_receipt().clone();
    let broker_truth_binding = verified_link.manifest().binding().clone();
    let verified_authority = acquisition_store
        .open_authority(&authority_receipt)
        .map_err(|error| sealed_ingress_error("acquisition authority reopen", error))?;
    let verified_bundle = BrokerFinancialTruthBundleStoreV1::new(acquisition_store.root())
        .open_exact_v2(&broker_truth_receipt, &broker_truth_binding)
        .map_err(|error| sealed_ingress_error("broker-truth bundle reopen", error))?;
    let primary_replay_rule = verified_bundle.manifest().primary_quotes().replay_rule();
    if primary_replay_rule.identity() != &expected_replay_binding.reviewed_replay_rule
        || primary_replay_rule.observations_raw().sha256()
            != expected_replay_binding
                .reviewed_replay_rule
                .broker_observation_sha256()
    {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "broker-truth primary observations or replay rule differ from the reviewed replay binding",
        ));
    }
    let primary_reviewed_rules_sha256 = primary_replay_rule.rules_decoded().sha256().to_owned();
    let reviewed_matches = verified_authority
        .manifest()
        .reviewed_synchronizations()
        .iter()
        .filter(|synchronization| {
            synchronization.account_id() == expected_replay_binding.account_id
                && synchronization.symbol_id() == expected_replay_binding.symbol_id
                && synchronization.window()
                    == expected_replay_binding
                        .replay_scope
                        .required_quote_coverage_window()
                && synchronization.review_identity()
                    == &expected_replay_binding.reviewed_replay_rule
                && synchronization.reviewed_rules_sha256() == primary_reviewed_rules_sha256.as_str()
        })
        .count();
    if reviewed_matches != 1 {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "acquisition authority has no unique exact reviewed primary synchronization",
        ));
    }

    let semantic_ingress = inspect_untrusted_broker_financial_truth_bundle_v2(verified_bundle)
        .map_err(|error| sealed_ingress_error("V2 structural semantic ingress", error))?;
    let primary = semantic_ingress.into_primary_quote_replay();
    if primary.symbol_name.as_str() != expected_replay_binding.symbol_name.as_str()
        || primary.reviewed_replay_rule_identity_sha256.as_str()
            != expected_replay_binding
                .reviewed_replay_rule
                .identity_sha256()
        || primary.reviewed_rules_sha256 != primary_reviewed_rules_sha256
    {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "structurally decoded primary quotes differ from the sealed replay binding",
        ));
    }
    let evidence = CompleteBidAskQuoteReplayEvidenceV1::new(
        expected_replay_binding.clone(),
        complete_side_from_structural_ingress(primary.bid)?,
        complete_side_from_structural_ingress(primary.ask)?,
    )?;
    Ok(SealedHistoricalBidAskQuoteReplayEvidenceV1 {
        evidence,
        acquisition_link_manifest_sha256: link_receipt.manifest_sha256().to_owned(),
    })
}

#[derive(Serialize)]
struct LedgerHashPayloadV1<'a> {
    binding_identity_sha256: &'a str,
    quote_evidence_content_sha256: &'a str,
    replay_policy_sha256: &'a str,
    positions: &'a [QuoteValidatedResearchPositionV1],
    entry_unavailable: &'a [QuoteValidatedResearchNonEntryV1],
    authority: QuoteValidatedResearchAuthorityV1,
    historical_acquisition_link_manifest_sha256: Option<&'a str>,
    promotion_eligibility: QuoteValidatedResearchPromotionEligibilityV1,
}

pub fn replay_quote_validated_research_v1(
    plan: &QuoteValidatedResearchReplayPlanV1,
    evidence: CompleteBidAskQuoteReplayEvidenceV1,
) -> Result<QuoteValidatedResearchLedgerV1, QuoteValidatedResearchReplayErrorV1> {
    replay_with_authority_v1(
        plan,
        evidence,
        QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes,
        None,
    )
}

pub fn replay_sealed_quote_validated_research_v1(
    plan: &QuoteValidatedResearchReplayPlanV1,
    evidence: SealedHistoricalBidAskQuoteReplayEvidenceV1,
) -> Result<SealedHistoricalQuoteValidatedResearchLedgerV1, QuoteValidatedResearchReplayErrorV1> {
    let ledger = replay_with_authority_v1(
        plan,
        evidence.evidence,
        QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly,
        Some(evidence.acquisition_link_manifest_sha256),
    )?;
    Ok(SealedHistoricalQuoteValidatedResearchLedgerV1 { ledger })
}

fn replay_with_authority_v1(
    plan: &QuoteValidatedResearchReplayPlanV1,
    evidence: CompleteBidAskQuoteReplayEvidenceV1,
    authority: QuoteValidatedResearchAuthorityV1,
    historical_acquisition_link_manifest_sha256: Option<String>,
) -> Result<QuoteValidatedResearchLedgerV1, QuoteValidatedResearchReplayErrorV1> {
    plan.validate()?;
    evidence.validate()?;
    match (
        authority,
        historical_acquisition_link_manifest_sha256.as_deref(),
    ) {
        (QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly, Some(digest)) => {
            validate_digest("historical acquisition link manifest", digest)?;
        }
        (QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes, None) => {}
        _ => {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
                "quote replay authority differs from its sealed acquisition-link evidence",
            ));
        }
    }
    if plan.binding != evidence.binding {
        return Err(replay_error(
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "replay plan and complete quote evidence have different exact bindings",
        ));
    }

    let mut positions = Vec::new();
    let mut entry_unavailable = Vec::new();
    for decision in &plan.decisions {
        let selected_entry = match select_entry(decision, &evidence, &plan.policy)? {
            EntrySelectionV1::Available(entry) => entry,
            EntrySelectionV1::Unavailable(unavailable) => {
                entry_unavailable.push(unavailable);
                continue;
            }
        };
        let entry_side = match decision.direction {
            ResearchPositionDirectionV1::Long => QuoteSideV1::Ask,
            ResearchPositionDirectionV1::Short => QuoteSideV1::Bid,
        };
        let entry_book = QuoteValidatedEntryBookV1 {
            bid_reference: QuoteValidatedPriceReferenceV1::from_quote(
                QuoteSideV1::Bid,
                &selected_entry.bid_quote,
            ),
            ask_reference: QuoteValidatedPriceReferenceV1::from_quote(
                QuoteSideV1::Ask,
                &selected_entry.ask_quote,
            ),
        };
        let entry_quote = selected_entry.entry_quote;
        let entry_reference = QuoteValidatedPriceReferenceV1::from_quote(entry_side, &entry_quote);
        let modeled_entry_price = modeled_price(
            entry_reference.price,
            decision.direction,
            true,
            &plan.policy.latency_slippage,
        )?;
        let modeled_entry_within_bounds = match decision.direction {
            ResearchPositionDirectionV1::Long => {
                modeled_entry_price > decision.stop_price
                    && modeled_entry_price < decision.target_price
            }
            ResearchPositionDirectionV1::Short => {
                modeled_entry_price < decision.stop_price
                    && modeled_entry_price > decision.target_price
            }
        };
        if !modeled_entry_within_bounds {
            return Err(replay_error(
                QuoteValidatedResearchReplayErrorCodeV1::ModeledEntryOutsideDecisionBounds,
                "modeled entry is not strictly inside the decision stop-target interval",
            ));
        }
        let selected_exit = select_exit(
            decision,
            &entry_quote,
            &evidence,
            &plan.policy,
            &plan.trailing_thresholds,
        )?;
        let (exit_reference, exit_reason, modeled_exit_price, slippage_pips_charged) =
            if let Some((exit_quote, exit_reason)) = selected_exit {
                let exit_side = match decision.direction {
                    ResearchPositionDirectionV1::Long => QuoteSideV1::Bid,
                    ResearchPositionDirectionV1::Short => QuoteSideV1::Ask,
                };
                let exit_reference =
                    QuoteValidatedPriceReferenceV1::from_quote(exit_side, &exit_quote);
                let modeled_exit_price = modeled_price(
                    exit_reference.price,
                    decision.direction,
                    false,
                    &plan.policy.latency_slippage,
                )?;
                (
                    Some(exit_reference),
                    Some(exit_reason),
                    Some(modeled_exit_price),
                    plan.policy.latency_slippage.slippage_pips_per_fill * 2.0,
                )
            } else {
                (
                    None,
                    None,
                    None,
                    plan.policy.latency_slippage.slippage_pips_per_fill,
                )
            };
        positions.push(QuoteValidatedResearchPositionV1 {
            direction: decision.direction,
            entry_book,
            entry_reference,
            exit_reference,
            exit_reason,
            modeled_entry_price,
            modeled_exit_price,
            slippage_pips_charged,
            additional_spread_pips_charged: 0.0,
        });
    }

    let promotion_eligibility = QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible;
    let ledger_sha256 = hash_json(
        "quote-validated research ledger",
        &LedgerHashPayloadV1 {
            binding_identity_sha256: &plan.binding.identity_sha256,
            quote_evidence_content_sha256: &evidence.content_sha256,
            replay_policy_sha256: &plan.policy.identity_sha256,
            positions: &positions,
            entry_unavailable: &entry_unavailable,
            authority,
            historical_acquisition_link_manifest_sha256:
                historical_acquisition_link_manifest_sha256.as_deref(),
            promotion_eligibility,
        },
    )?;
    let receipt = QuoteValidatedResearchReplayReceiptV1 {
        canonical_search_input_receipt_sha256: plan
            .binding
            .canonical_search_input_receipt_sha256
            .clone(),
        canonical_signal_plan_sha256: plan.binding.canonical_signal_plan_sha256.clone(),
        account_id: plan.binding.account_id,
        symbol_id: plan.binding.symbol_id,
        symbol_name: plan.binding.symbol_name.clone(),
        locked_evaluation_window: plan.binding.replay_scope.locked_evaluation_window(),
        required_quote_coverage_window: plan.binding.replay_scope.required_quote_coverage_window(),
        seed_padding_ms: plan.binding.replay_scope.seed_padding_ms(),
        exit_padding_ms: plan.binding.replay_scope.exit_padding_ms(),
        reviewed_replay_rule_identity_sha256: plan
            .binding
            .reviewed_replay_rule
            .identity_sha256()
            .to_owned(),
        quote_evidence_manifest_sha256: plan.binding.quote_evidence_manifest_sha256.clone(),
        latency_slippage_policy_sha256: plan.policy.latency_slippage.identity_sha256.clone(),
        replay_policy_sha256: plan.policy.identity_sha256.clone(),
        authority,
        historical_acquisition_link_manifest_sha256,
        ledger_sha256,
    };
    Ok(QuoteValidatedResearchLedgerV1 {
        receipt,
        positions,
        entry_unavailable,
        authority,
        promotion_eligibility,
    })
}
