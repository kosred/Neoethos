use std::collections::HashSet;

use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::{Deserialize, Serialize};

use crate::contracts::{
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthContractErrorCodeV1,
    BrokerFinancialTruthContractErrorV1, BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1,
    ExactCapturedEvidencePairV1, ImmutableVortexArtifactV1, MAX_MANIFEST_BYTES, QuoteSideV1,
    sha256_bytes, validate_sha256_hex,
};

pub const BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V2: u16 = 2;
pub const BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V2: &str = "bft2-";
pub const MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2: i64 = 7 * 24 * 60 * 60 * 1_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBrokerRequestPageV2 {
    chunk_sequence: u64,
    page_sequence_in_chunk: u64,
    client_msg_id: String,
    requested_window: EvidenceWindowV1,
    first_event_unix_ms: Option<i64>,
    last_event_unix_ms: Option<i64>,
    event_count: u64,
    response_has_more: bool,
    max_rows: Option<u32>,
}

impl ExactBrokerRequestPageV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_sequence: u64,
        page_sequence_in_chunk: u64,
        client_msg_id: impl Into<String>,
        requested_window: EvidenceWindowV1,
        first_event_unix_ms: Option<i64>,
        last_event_unix_ms: Option<i64>,
        event_count: u64,
        response_has_more: bool,
        max_rows: Option<u32>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let page = Self {
            chunk_sequence,
            page_sequence_in_chunk,
            client_msg_id: client_msg_id.into(),
            requested_window,
            first_event_unix_ms,
            last_event_unix_ms,
            event_count,
            response_has_more,
            max_rows,
        };
        page.validate()?;
        Ok(page)
    }

    pub const fn chunk_sequence(&self) -> u64 {
        self.chunk_sequence
    }

    pub const fn page_sequence_in_chunk(&self) -> u64 {
        self.page_sequence_in_chunk
    }

    pub fn client_msg_id(&self) -> &str {
        &self.client_msg_id
    }

    pub const fn requested_window(&self) -> EvidenceWindowV1 {
        self.requested_window
    }

    pub const fn first_event_unix_ms(&self) -> Option<i64> {
        self.first_event_unix_ms
    }

    pub const fn last_event_unix_ms(&self) -> Option<i64> {
        self.last_event_unix_ms
    }

    pub const fn event_count(&self) -> u64 {
        self.event_count
    }

    pub const fn response_has_more(&self) -> bool {
        self.response_has_more
    }

    pub const fn max_rows(&self) -> Option<u32> {
        self.max_rows
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.requested_window.validate()?;
        if self.client_msg_id.trim().is_empty()
            || self.client_msg_id != self.client_msg_id.trim()
            || self.client_msg_id.len() > 160
            || self.client_msg_id.chars().any(char::is_control)
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                "exact broker request page has an invalid clientMsgId",
            ));
        }
        if self.max_rows == Some(0) {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                "exact broker request page maxRows must be positive when present",
            ));
        }
        match (
            self.event_count,
            self.first_event_unix_ms,
            self.last_event_unix_ms,
        ) {
            (0, None, None) if !self.response_has_more => {}
            (count, Some(first), Some(last))
                if count > 0
                    && first <= last
                    && first >= self.requested_window.from_unix_ms_inclusive()
                    && last < self.requested_window.to_unix_ms_exclusive() => {}
            _ => {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                    "exact broker response page has inconsistent count/bounds/hasMore",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBrokerRequestChunkV2 {
    chunk_sequence: u64,
    requested_window: EvidenceWindowV1,
    pages_newest_first: Vec<ExactBrokerRequestPageV2>,
}

impl ExactBrokerRequestChunkV2 {
    pub fn new(
        chunk_sequence: u64,
        requested_window: EvidenceWindowV1,
        pages_newest_first: Vec<ExactBrokerRequestPageV2>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let chunk = Self {
            chunk_sequence,
            requested_window,
            pages_newest_first,
        };
        chunk.validate_pages()?;
        Ok(chunk)
    }

    pub const fn chunk_sequence(&self) -> u64 {
        self.chunk_sequence
    }

    pub const fn requested_window(&self) -> EvidenceWindowV1 {
        self.requested_window
    }

    pub fn pages_newest_first(&self) -> &[ExactBrokerRequestPageV2] {
        &self.pages_newest_first
    }

    pub fn validate_quote_partition(
        complete_window: EvidenceWindowV1,
        chunks_newest_first: &[Self],
    ) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_partition(
            complete_window,
            chunks_newest_first,
            Some(MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2),
            PageContractV2::Quote,
        )
    }

    fn validate_deal_partition(
        complete_window: EvidenceWindowV1,
        chunks_newest_first: &[Self],
    ) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_partition(
            complete_window,
            chunks_newest_first,
            None,
            PageContractV2::Deal,
        )
    }

    fn validate_pages(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.requested_window.validate()?;
        if self.pages_newest_first.is_empty() {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "exact broker request chunk contains no retained response page",
            ));
        }
        let mut client_msg_ids = HashSet::new();
        for (page_index, page) in self.pages_newest_first.iter().enumerate() {
            page.validate()?;
            if page.chunk_sequence != self.chunk_sequence
                || page.page_sequence_in_chunk != page_index as u64
                || !client_msg_ids.insert(page.client_msg_id.as_str())
            {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                    "exact broker pages have a changed sequence or duplicate clientMsgId",
                ));
            }
            if page_index == 0 {
                if page.requested_window != self.requested_window {
                    return Err(contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                        "first page request does not equal its exact broker request chunk",
                    ));
                }
            } else {
                let newer = &self.pages_newest_first[page_index - 1];
                let boundary = newer.first_event_unix_ms.ok_or_else(|| {
                    contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                        "an empty response cannot establish an older exclusive page boundary",
                    )
                })?;
                if !newer.response_has_more
                    || page.requested_window.from_unix_ms_inclusive()
                        != self.requested_window.from_unix_ms_inclusive()
                    || page.requested_window.to_unix_ms_exclusive() != boundary
                    || page.last_event_unix_ms.is_some_and(|last| last >= boundary)
                {
                    return Err(contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                        "older page did not use the prior oldest event as an exclusive boundary",
                    ));
                }
            }
        }
        if self
            .pages_newest_first
            .last()
            .expect("non-empty exact pages")
            .response_has_more
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "terminal exact broker response page still reports hasMore=true",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PageContractV2 {
    Quote,
    Deal,
}

fn validate_partition(
    complete_window: EvidenceWindowV1,
    chunks_newest_first: &[ExactBrokerRequestChunkV2],
    maximum_span_ms: Option<i64>,
    page_contract: PageContractV2,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    complete_window.validate()?;
    if chunks_newest_first.is_empty() {
        return Err(contract_error(
            BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
            "exact broker request partition contains no chunks",
        ));
    }
    let mut client_msg_ids = HashSet::new();
    for (chunk_index, chunk) in chunks_newest_first.iter().enumerate() {
        chunk.validate_pages()?;
        if chunk.chunk_sequence != chunk_index as u64 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                "exact broker request chunks are not sequence ordered",
            ));
        }
        let span = chunk
            .requested_window
            .to_unix_ms_exclusive()
            .checked_sub(chunk.requested_window.from_unix_ms_inclusive())
            .ok_or_else(|| {
                contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                    "exact broker request chunk span overflowed",
                )
            })?;
        if maximum_span_ms.is_some_and(|maximum| span > maximum) {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                format!(
                    "cTrader tick request chunk span {span} exceeds {MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2} ms"
                ),
            ));
        }
        match page_contract {
            PageContractV2::Quote => {
                if chunk
                    .pages_newest_first
                    .iter()
                    .any(|page| page.max_rows.is_some() || page.event_count == 0)
                {
                    return Err(contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                        "quote pages must retain non-empty ticks and no DealList maxRows",
                    ));
                }
            }
            PageContractV2::Deal => {
                if chunk
                    .pages_newest_first
                    .iter()
                    .any(|page| page.max_rows.is_none())
                {
                    return Err(contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                        "DealList pages must retain the exact positive maxRows request",
                    ));
                }
            }
        }
        for page in &chunk.pages_newest_first {
            if !client_msg_ids.insert(page.client_msg_id.as_str()) {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::DuplicateArtifact,
                    "clientMsgId was reused across exact broker request chunks",
                ));
            }
        }
        if chunk_index == 0 {
            if chunk.requested_window.to_unix_ms_exclusive()
                != complete_window.to_unix_ms_exclusive()
            {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                    "newest exact request chunk does not end at the complete window boundary",
                ));
            }
        } else if chunk.requested_window.to_unix_ms_exclusive()
            != chunks_newest_first[chunk_index - 1]
                .requested_window
                .from_unix_ms_inclusive()
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                "exact request chunks contain a gap or overlap",
            ));
        }
    }
    if chunks_newest_first
        .last()
        .expect("non-empty exact chunks")
        .requested_window
        .from_unix_ms_inclusive()
        != complete_window.from_unix_ms_inclusive()
    {
        return Err(contract_error(
            BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
            "oldest exact request chunk does not begin at the complete window boundary",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactQuoteSideEvidenceV2 {
    side: QuoteSideV1,
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    quote_asset_id: i64,
    requested_window: EvidenceWindowV1,
    request_chunks_newest_first: Vec<ExactBrokerRequestChunkV2>,
    raw_pages: ImmutableVortexArtifactV1,
    decoded_ticks: ImmutableVortexArtifactV1,
}

impl ExactQuoteSideEvidenceV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: QuoteSideV1,
        symbol_id: i64,
        symbol_name: impl Into<String>,
        base_asset_id: i64,
        quote_asset_id: i64,
        requested_window: EvidenceWindowV1,
        request_chunks_newest_first: Vec<ExactBrokerRequestChunkV2>,
        raw_pages: ImmutableVortexArtifactV1,
        decoded_ticks: ImmutableVortexArtifactV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            side,
            symbol_id,
            symbol_name: symbol_name.into(),
            base_asset_id,
            quote_asset_id,
            requested_window,
            request_chunks_newest_first,
            raw_pages,
            decoded_ticks,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn side(&self) -> QuoteSideV1 {
        self.side
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub const fn base_asset_id(&self) -> i64 {
        self.base_asset_id
    }

    pub const fn quote_asset_id(&self) -> i64 {
        self.quote_asset_id
    }

    pub const fn requested_window(&self) -> EvidenceWindowV1 {
        self.requested_window
    }

    pub fn request_chunks_newest_first(&self) -> &[ExactBrokerRequestChunkV2] {
        &self.request_chunks_newest_first
    }

    pub const fn raw_pages(&self) -> &ImmutableVortexArtifactV1 {
        &self.raw_pages
    }

    pub const fn decoded_ticks(&self) -> &ImmutableVortexArtifactV1 {
        &self.decoded_ticks
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.symbol_id <= 0
            || self.symbol_name.trim().is_empty()
            || self.base_asset_id <= 0
            || self.quote_asset_id <= 0
            || self.base_asset_id == self.quote_asset_id
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                "V2 quote evidence has an invalid symbol or base/quote asset binding",
            ));
        }
        ExactBrokerRequestChunkV2::validate_quote_partition(
            self.requested_window,
            &self.request_chunks_newest_first,
        )?;
        require_schema(
            &self.raw_pages,
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
            "V2 raw tick request pages",
        )?;
        require_schema(
            &self.decoded_ticks,
            BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
            "V2 decoded ticks",
        )
    }

    fn artifacts(&self) -> [&ImmutableVortexArtifactV1; 2] {
        [&self.raw_pages, &self.decoded_ticks]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedQuoteReplayRuleIdentityV2 {
    identity_sha256: String,
    review_record_sha256: String,
    protocol_evidence_sha256: String,
    broker_observation_sha256: String,
}

impl ReviewedQuoteReplayRuleIdentityV2 {
    pub fn new(
        review_record_sha256: impl Into<String>,
        protocol_evidence_sha256: impl Into<String>,
        broker_observation_sha256: impl Into<String>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let review_record_sha256 = review_record_sha256.into();
        let protocol_evidence_sha256 = protocol_evidence_sha256.into();
        let broker_observation_sha256 = broker_observation_sha256.into();
        for (label, digest) in [
            ("replay-rule review record", review_record_sha256.as_str()),
            (
                "replay-rule protocol evidence",
                protocol_evidence_sha256.as_str(),
            ),
            (
                "replay-rule broker observation evidence",
                broker_observation_sha256.as_str(),
            ),
        ] {
            validate_sha256_hex(label, digest)?;
        }
        let identity_sha256 = sha256_bytes(
            format!(
                "broker-quote-replay-rule-v2\n{review_record_sha256}\n{protocol_evidence_sha256}\n{broker_observation_sha256}\n"
            )
            .as_bytes(),
        );
        Ok(Self {
            identity_sha256,
            review_record_sha256,
            protocol_evidence_sha256,
            broker_observation_sha256,
        })
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn review_record_sha256(&self) -> &str {
        &self.review_record_sha256
    }

    pub fn protocol_evidence_sha256(&self) -> &str {
        &self.protocol_evidence_sha256
    }

    pub fn broker_observation_sha256(&self) -> &str {
        &self.broker_observation_sha256
    }

    pub fn validate_exact(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.validate()
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        let recomputed = Self::new(
            self.review_record_sha256.clone(),
            self.protocol_evidence_sha256.clone(),
            self.broker_observation_sha256.clone(),
        )?;
        if recomputed.identity_sha256 != self.identity_sha256 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidSha256,
                "reviewed quote replay-rule identity does not match its exact inputs",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedQuoteReplayRuleEvidenceV2 {
    identity: ReviewedQuoteReplayRuleIdentityV2,
    observations_raw: ImmutableVortexArtifactV1,
    rules_decoded: ImmutableVortexArtifactV1,
}

impl ReviewedQuoteReplayRuleEvidenceV2 {
    pub fn new(
        identity: ReviewedQuoteReplayRuleIdentityV2,
        observations_raw: ImmutableVortexArtifactV1,
        rules_decoded: ImmutableVortexArtifactV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            identity,
            observations_raw,
            rules_decoded,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn identity(&self) -> &ReviewedQuoteReplayRuleIdentityV2 {
        &self.identity
    }

    pub const fn observations_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.observations_raw
    }

    pub const fn rules_decoded(&self) -> &ImmutableVortexArtifactV1 {
        &self.rules_decoded
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.identity.validate()?;
        require_schema(
            &self.observations_raw,
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
            "V2 raw quote-session observations",
        )?;
        require_schema(
            &self.rules_decoded,
            BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
            "V2 reviewed quote replay rules",
        )
    }

    fn artifacts(&self) -> [&ImmutableVortexArtifactV1; 2] {
        [&self.observations_raw, &self.rules_decoded]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynchronizedBidAskEvidenceV2 {
    bid: ExactQuoteSideEvidenceV2,
    ask: ExactQuoteSideEvidenceV2,
    replay_rule: ReviewedQuoteReplayRuleEvidenceV2,
}

impl SynchronizedBidAskEvidenceV2 {
    pub fn new(
        bid: ExactQuoteSideEvidenceV2,
        ask: ExactQuoteSideEvidenceV2,
        replay_rule: ReviewedQuoteReplayRuleEvidenceV2,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            bid,
            ask,
            replay_rule,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn bid(&self) -> &ExactQuoteSideEvidenceV2 {
        &self.bid
    }

    pub const fn ask(&self) -> &ExactQuoteSideEvidenceV2 {
        &self.ask
    }

    pub const fn replay_rule(&self) -> &ReviewedQuoteReplayRuleEvidenceV2 {
        &self.replay_rule
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.bid.validate()?;
        self.ask.validate()?;
        self.replay_rule.validate()?;
        if self.bid.side != QuoteSideV1::Bid
            || self.ask.side != QuoteSideV1::Ask
            || self.bid.symbol_id != self.ask.symbol_id
            || self.bid.symbol_name != self.ask.symbol_name
            || self.bid.base_asset_id != self.ask.base_asset_id
            || self.bid.quote_asset_id != self.ask.quote_asset_id
            || self.bid.requested_window != self.ask.requested_window
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                "V2 Bid/Ask evidence differs in side, symbol, assets, or exact window",
            ));
        }
        Ok(())
    }

    fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.bid
            .artifacts()
            .into_iter()
            .chain(self.ask.artifacts())
            .chain(self.replay_rule.artifacts())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactConversionLegEvidenceV2 {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    quotes: SynchronizedBidAskEvidenceV2,
}

impl ExactConversionLegEvidenceV2 {
    pub fn new(
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        quotes: SynchronizedBidAskEvidenceV2,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let leg = Self {
            from_asset_id,
            from_asset_name: from_asset_name.into(),
            to_asset_id,
            to_asset_name: to_asset_name.into(),
            quotes,
        };
        leg.validate()?;
        Ok(leg)
    }

    pub const fn from_asset_id(&self) -> i64 {
        self.from_asset_id
    }

    pub fn from_asset_name(&self) -> &str {
        &self.from_asset_name
    }

    pub const fn to_asset_id(&self) -> i64 {
        self.to_asset_id
    }

    pub fn to_asset_name(&self) -> &str {
        &self.to_asset_name
    }

    pub const fn quotes(&self) -> &SynchronizedBidAskEvidenceV2 {
        &self.quotes
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.from_asset_id <= 0
            || self.to_asset_id <= 0
            || self.from_asset_id == self.to_asset_id
            || self.from_asset_name.trim().is_empty()
            || self.to_asset_name.trim().is_empty()
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "V2 conversion leg has invalid exact assets",
            ));
        }
        self.quotes.validate()?;
        let base = self.quotes.bid.base_asset_id;
        let quote = self.quotes.bid.quote_asset_id;
        if !((base == self.from_asset_id && quote == self.to_asset_id)
            || (quote == self.from_asset_id && base == self.to_asset_id))
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "V2 conversion leg assets do not match its broker symbol",
            ));
        }
        Ok(())
    }

    fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.quotes.artifacts()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactConversionRouteEvidenceV2 {
    purpose: String,
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    legs: Vec<ExactConversionLegEvidenceV2>,
}

impl ExactConversionRouteEvidenceV2 {
    pub fn new(
        purpose: impl Into<String>,
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        legs: Vec<ExactConversionLegEvidenceV2>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let route = Self {
            purpose: purpose.into(),
            from_asset_id,
            from_asset_name: from_asset_name.into(),
            to_asset_id,
            to_asset_name: to_asset_name.into(),
            legs,
        };
        route.validate()?;
        Ok(route)
    }

    pub fn purpose(&self) -> &str {
        &self.purpose
    }

    pub const fn from_asset_id(&self) -> i64 {
        self.from_asset_id
    }

    pub fn from_asset_name(&self) -> &str {
        &self.from_asset_name
    }

    pub const fn to_asset_id(&self) -> i64 {
        self.to_asset_id
    }

    pub fn to_asset_name(&self) -> &str {
        &self.to_asset_name
    }

    pub fn legs(&self) -> &[ExactConversionLegEvidenceV2] {
        &self.legs
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if !matches!(
            self.purpose.as_str(),
            "primary_pnl_settlement" | "commission_settlement" | "margin_settlement"
        ) || self.from_asset_id <= 0
            || self.to_asset_id <= 0
            || self.from_asset_name.trim().is_empty()
            || self.to_asset_name.trim().is_empty()
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "V2 conversion route has an invalid purpose or asset binding",
            ));
        }
        if self.from_asset_id == self.to_asset_id {
            if self.from_asset_name != self.to_asset_name || !self.legs.is_empty() {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                    "V2 identity conversion must use the same exact asset and zero legs",
                ));
            }
            return Ok(());
        }
        if self.legs.is_empty() {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "V2 non-identity conversion route has no exact quote legs",
            ));
        }
        let mut expected = self.from_asset_id;
        let mut visited = HashSet::from([expected]);
        for leg in &self.legs {
            leg.validate()?;
            if leg.from_asset_id != expected || !visited.insert(leg.to_asset_id) {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                    "V2 conversion route is discontinuous or cyclic",
                ));
            }
            expected = leg.to_asset_id;
        }
        if expected != self.to_asset_id {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "V2 conversion route does not terminate at the exact destination",
            ));
        }
        Ok(())
    }

    fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.legs
            .iter()
            .flat_map(ExactConversionLegEvidenceV2::artifacts)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSymbolContractEvidenceV2 {
    light_symbol_responses_raw: ImmutableVortexArtifactV1,
    full_symbol_responses_raw: ImmutableVortexArtifactV1,
    account_asset_responses_raw: ImmutableVortexArtifactV1,
    trader_account_responses_raw: ImmutableVortexArtifactV1,
    contracts_decoded: ImmutableVortexArtifactV1,
}

impl ExactSymbolContractEvidenceV2 {
    pub fn new(
        light_symbol_responses_raw: ImmutableVortexArtifactV1,
        full_symbol_responses_raw: ImmutableVortexArtifactV1,
        account_asset_responses_raw: ImmutableVortexArtifactV1,
        trader_account_responses_raw: ImmutableVortexArtifactV1,
        contracts_decoded: ImmutableVortexArtifactV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            light_symbol_responses_raw,
            full_symbol_responses_raw,
            account_asset_responses_raw,
            trader_account_responses_raw,
            contracts_decoded,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn light_symbol_responses_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.light_symbol_responses_raw
    }

    pub const fn full_symbol_responses_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.full_symbol_responses_raw
    }

    pub const fn account_asset_responses_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.account_asset_responses_raw
    }

    pub const fn trader_account_responses_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.trader_account_responses_raw
    }

    pub const fn contracts_decoded(&self) -> &ImmutableVortexArtifactV1 {
        &self.contracts_decoded
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        for (artifact, schema, label) in [
            (
                &self.light_symbol_responses_raw,
                BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
                "raw ProtoOALightSymbol responses",
            ),
            (
                &self.full_symbol_responses_raw,
                BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
                "raw ProtoOASymbol responses",
            ),
            (
                &self.account_asset_responses_raw,
                BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
                "raw account asset responses",
            ),
            (
                &self.trader_account_responses_raw,
                BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
                "raw trader account responses",
            ),
            (
                &self.contracts_decoded,
                BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
                "decoded symbol/money contracts",
            ),
        ] {
            require_schema(artifact, schema, label)?;
        }
        Ok(())
    }

    fn artifacts(&self) -> [&ImmutableVortexArtifactV1; 5] {
        [
            &self.light_symbol_responses_raw,
            &self.full_symbol_responses_raw,
            &self.account_asset_responses_raw,
            &self.trader_account_responses_raw,
            &self.contracts_decoded,
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDealReconciliationEvidenceV2 {
    requested_window: EvidenceWindowV1,
    deal_request_chunk: ExactBrokerRequestChunkV2,
    reconcile_responses_raw: ImmutableVortexArtifactV1,
    deal_pages_raw: ImmutableVortexArtifactV1,
    reconciliation_decoded: ImmutableVortexArtifactV1,
}

impl ExactDealReconciliationEvidenceV2 {
    pub fn new(
        requested_window: EvidenceWindowV1,
        deal_request_chunk: ExactBrokerRequestChunkV2,
        reconcile_responses_raw: ImmutableVortexArtifactV1,
        deal_pages_raw: ImmutableVortexArtifactV1,
        reconciliation_decoded: ImmutableVortexArtifactV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            requested_window,
            deal_request_chunk,
            reconcile_responses_raw,
            deal_pages_raw,
            reconciliation_decoded,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn requested_window(&self) -> EvidenceWindowV1 {
        self.requested_window
    }

    pub const fn deal_request_chunk(&self) -> &ExactBrokerRequestChunkV2 {
        &self.deal_request_chunk
    }

    pub const fn reconcile_responses_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.reconcile_responses_raw
    }

    pub const fn deal_pages_raw(&self) -> &ImmutableVortexArtifactV1 {
        &self.deal_pages_raw
    }

    pub const fn reconciliation_decoded(&self) -> &ImmutableVortexArtifactV1 {
        &self.reconciliation_decoded
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        ExactBrokerRequestChunkV2::validate_deal_partition(
            self.requested_window,
            std::slice::from_ref(&self.deal_request_chunk),
        )?;
        require_schema(
            &self.reconcile_responses_raw,
            BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
            "raw reconcile responses",
        )?;
        require_schema(
            &self.deal_pages_raw,
            BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
            "raw paged DealList responses",
        )?;
        require_schema(
            &self.reconciliation_decoded,
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
            "decoded close/deal reconciliation",
        )
    }

    fn artifacts(&self) -> [&ImmutableVortexArtifactV1; 3] {
        [
            &self.reconcile_responses_raw,
            &self.deal_pages_raw,
            &self.reconciliation_decoded,
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerFinancialTruthBundleManifestV2 {
    schema_version: u16,
    binding: BrokerFinancialTruthBindingV1,
    primary_quotes: SynchronizedBidAskEvidenceV2,
    conversion_routes: Vec<ExactConversionRouteEvidenceV2>,
    exact_symbol_contracts: ExactSymbolContractEvidenceV2,
    broker_position_unrealized_pnl: ExactCapturedEvidencePairV1,
    close_deal_reconciliation: ExactDealReconciliationEvidenceV2,
}

impl BrokerFinancialTruthBundleManifestV2 {
    pub fn new(
        binding: BrokerFinancialTruthBindingV1,
        primary_quotes: SynchronizedBidAskEvidenceV2,
        conversion_routes: Vec<ExactConversionRouteEvidenceV2>,
        exact_symbol_contracts: ExactSymbolContractEvidenceV2,
        broker_position_unrealized_pnl: ExactCapturedEvidencePairV1,
        close_deal_reconciliation: ExactDealReconciliationEvidenceV2,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let manifest = Self {
            schema_version: BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V2,
            binding,
            primary_quotes,
            conversion_routes,
            exact_symbol_contracts,
            broker_position_unrealized_pnl,
            close_deal_reconciliation,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn binding(&self) -> &BrokerFinancialTruthBindingV1 {
        &self.binding
    }

    pub const fn primary_quotes(&self) -> &SynchronizedBidAskEvidenceV2 {
        &self.primary_quotes
    }

    pub fn conversion_routes(&self) -> &[ExactConversionRouteEvidenceV2] {
        &self.conversion_routes
    }

    pub const fn exact_symbol_contracts(&self) -> &ExactSymbolContractEvidenceV2 {
        &self.exact_symbol_contracts
    }

    pub const fn broker_position_unrealized_pnl(&self) -> &ExactCapturedEvidencePairV1 {
        &self.broker_position_unrealized_pnl
    }

    pub const fn close_deal_reconciliation(&self) -> &ExactDealReconciliationEvidenceV2 {
        &self.close_deal_reconciliation
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot encode V2 broker truth manifest: {error}"),
            )
        })
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("V2 broker truth manifest exceeds {MAX_MANIFEST_BYTES} bytes"),
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot decode V2 broker truth manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub(crate) fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.schema_version != BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V2 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::UnsupportedSchemaVersion,
                format!(
                    "unsupported broker truth V2 manifest schema {}",
                    self.schema_version
                ),
            ));
        }
        self.binding.validate()?;
        self.primary_quotes.validate()?;
        let CanonicalDatasetScope::CTrader { symbol_id, .. } =
            self.binding.canonical_dataset_identity().scope()
        else {
            unreachable!("binding validation requires cTrader")
        };
        if self.primary_quotes.bid.symbol_id != *symbol_id
            || self.primary_quotes.bid.symbol_name
                != self.binding.canonical_dataset_identity().symbol_name()
            || self.primary_quotes.bid.base_asset_id != self.binding.primary_base_asset_id()
            || self.primary_quotes.bid.quote_asset_id != self.binding.primary_quote_asset_id()
            || self.primary_quotes.bid.requested_window != self.binding.evaluated_window()
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                "V2 primary quotes differ from the exact dataset/binding/window",
            ));
        }
        for route in &self.conversion_routes {
            route.validate()?;
            for leg in &route.legs {
                if leg.quotes.bid.requested_window != self.binding.evaluated_window() {
                    return Err(contract_error(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                        "V2 conversion quotes differ from the exact evaluated window",
                    ));
                }
            }
        }
        let settlement = self
            .conversion_routes
            .iter()
            .filter(|route| route.purpose == "primary_pnl_settlement")
            .collect::<Vec<_>>();
        if settlement.len() != 1
            || settlement[0].from_asset_id != self.binding.primary_quote_asset_id()
            || settlement[0].to_asset_id != self.binding.account_asset_id()
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "V2 manifest requires exactly one quote-to-account PnL settlement route",
            ));
        }
        self.exact_symbol_contracts.validate()?;
        self.broker_position_unrealized_pnl.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2,
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2,
            "V2 broker position unrealized PnL",
        )?;
        self.close_deal_reconciliation.validate()?;
        if self.close_deal_reconciliation.requested_window != self.binding.evaluated_window() {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                "V2 DealList evidence differs from the exact evaluated window",
            ));
        }

        let mut paths = HashSet::new();
        for artifact in self.artifacts() {
            artifact.validate()?;
            if !paths.insert(artifact.relative_path()) {
                return Err(contract_error(
                    BrokerFinancialTruthContractErrorCodeV1::DuplicateArtifact,
                    format!(
                        "V2 artifact path {} appears more than once",
                        artifact.relative_path()
                    ),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.primary_quotes
            .artifacts()
            .into_iter()
            .chain(
                self.conversion_routes
                    .iter()
                    .flat_map(ExactConversionRouteEvidenceV2::artifacts),
            )
            .chain(self.exact_symbol_contracts.artifacts())
            .chain(self.broker_position_unrealized_pnl.artifacts())
            .chain(self.close_deal_reconciliation.artifacts())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerFinancialTruthBundleReceiptV2 {
    bundle_id: String,
    manifest_sha256: String,
}

impl BrokerFinancialTruthBundleReceiptV2 {
    pub(crate) fn from_manifest_sha256(
        manifest_sha256: String,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt = Self {
            bundle_id: format!("{BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V2}{manifest_sha256}"),
            manifest_sha256,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot decode V2 broker truth receipt: {error}"),
            )
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot encode V2 broker truth receipt: {error}"),
            )
        })
    }

    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub(crate) fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_sha256_hex("V2 manifest SHA-256", &self.manifest_sha256)?;
        if self.bundle_id
            != format!(
                "{BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V2}{}",
                self.manifest_sha256
            )
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                "V2 bundle id does not equal bft2- plus the exact manifest SHA-256",
            ));
        }
        Ok(())
    }
}

fn require_schema(
    artifact: &ImmutableVortexArtifactV1,
    expected: BrokerFinancialTruthVortexSchemaV1,
    label: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    artifact.validate()?;
    if artifact.schema() != expected {
        return Err(contract_error(
            BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
            format!("{label} has the wrong Vortex schema"),
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
