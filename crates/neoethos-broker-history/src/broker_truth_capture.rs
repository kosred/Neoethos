//! Fail-closed orchestration for one exact broker-financial evidence capture.
//!
//! The capture session is deliberately supplied by the existing authenticated
//! broker-history service. This module never opens another socket and never
//! treats a stored bundle as financial authority: it retains exact evidence
//! and returns only the leaf store's integrity receipt.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::path::Path;

use anyhow::Result;
use neoethos_broker_truth::{
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthBundleReceiptV1,
    BrokerFinancialTruthBundleReceiptV2, BrokerFinancialTruthBundleStoreV1,
    BrokerFinancialTruthContractErrorV1, EvidenceWindowV1, ExactBrokerRequestChunkV2,
    ExactBrokerRequestPageV2, QuoteSideV1, ReviewedQuoteReplayRuleIdentityV2,
};
use serde_json::Value;

use crate::broker_truth_vortex::{encode_broker_truth_capture_v1, encode_broker_truth_capture_v2};
use crate::ctrader_messages::{
    CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
};

const MAX_CAPTURE_LABEL_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthCaptureErrorCodeV1 {
    InvalidRequest,
    Cancelled,
    CaptureFailed,
    MissingQuotePages,
    InvalidQuotePage,
    TruncatedQuotePages,
    OverlappingQuotePages,
    OutOfOrderQuoteRows,
    MissingSynchronizationRules,
    MissingSymbolContracts,
    MissingUnrealizedPnl,
    MissingCloseDealReconciliation,
    EvidenceAccountMismatch,
    InvalidEvidenceRow,
    EncodingFailed,
    PublicationFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthCaptureErrorV1 {
    code: BrokerFinancialTruthCaptureErrorCodeV1,
    detail: String,
}

impl BrokerFinancialTruthCaptureErrorV1 {
    pub(crate) fn new(
        code: BrokerFinancialTruthCaptureErrorCodeV1,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> BrokerFinancialTruthCaptureErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerFinancialTruthCaptureErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "broker financial truth capture: {}", self.detail)
    }
}

impl Error for BrokerFinancialTruthCaptureErrorV1 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactQuoteInstrumentV1 {
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    base_asset_name: String,
    quote_asset_id: i64,
    quote_asset_name: String,
}

impl ExactQuoteInstrumentV1 {
    pub fn new(
        symbol_id: i64,
        symbol_name: impl Into<String>,
        base_asset_id: i64,
        base_asset_name: impl Into<String>,
        quote_asset_id: i64,
        quote_asset_name: impl Into<String>,
    ) -> Result<Self, BrokerFinancialTruthCaptureErrorV1> {
        let instrument = Self {
            symbol_id,
            symbol_name: symbol_name.into(),
            base_asset_id,
            base_asset_name: base_asset_name.into(),
            quote_asset_id,
            quote_asset_name: quote_asset_name.into(),
        };
        instrument.validate()?;
        Ok(instrument)
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

    pub fn base_asset_name(&self) -> &str {
        &self.base_asset_name
    }

    pub const fn quote_asset_id(&self) -> i64 {
        self.quote_asset_id
    }

    pub fn quote_asset_name(&self) -> &str {
        &self.quote_asset_name
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
        validate_id_label("quote symbol", self.symbol_id, &self.symbol_name)?;
        validate_id_label("base asset", self.base_asset_id, &self.base_asset_name)?;
        validate_id_label("quote asset", self.quote_asset_id, &self.quote_asset_name)?;
        if self.base_asset_id == self.quote_asset_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "quoted instrument has equal base and quote asset ids",
            ));
        }
        Ok(())
    }

    fn asset_name(&self, asset_id: i64) -> Option<&str> {
        if asset_id == self.base_asset_id {
            Some(&self.base_asset_name)
        } else if asset_id == self.quote_asset_id {
            Some(&self.quote_asset_name)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactConversionLegCaptureRequestV1 {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    instrument: ExactQuoteInstrumentV1,
}

impl ExactConversionLegCaptureRequestV1 {
    pub fn new(
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        instrument: ExactQuoteInstrumentV1,
    ) -> Result<Self, BrokerFinancialTruthCaptureErrorV1> {
        let leg = Self {
            from_asset_id,
            from_asset_name: from_asset_name.into(),
            to_asset_id,
            to_asset_name: to_asset_name.into(),
            instrument,
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

    pub const fn instrument(&self) -> &ExactQuoteInstrumentV1 {
        &self.instrument
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
        validate_id_label(
            "conversion source asset",
            self.from_asset_id,
            &self.from_asset_name,
        )?;
        validate_id_label(
            "conversion destination asset",
            self.to_asset_id,
            &self.to_asset_name,
        )?;
        if self.from_asset_id == self.to_asset_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "quoted conversion leg cannot convert an asset to itself",
            ));
        }
        self.instrument.validate()?;
        let from_name = self.instrument.asset_name(self.from_asset_id);
        let to_name = self.instrument.asset_name(self.to_asset_id);
        if from_name != Some(self.from_asset_name.as_str())
            || to_name != Some(self.to_asset_name.as_str())
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "conversion leg assets/names do not match its exact quoted instrument",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactConversionRouteCaptureRequestV1 {
    purpose: String,
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    legs: Vec<ExactConversionLegCaptureRequestV1>,
}

impl ExactConversionRouteCaptureRequestV1 {
    pub fn new(
        purpose: impl Into<String>,
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        legs: Vec<ExactConversionLegCaptureRequestV1>,
    ) -> Result<Self, BrokerFinancialTruthCaptureErrorV1> {
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

    pub fn legs(&self) -> &[ExactConversionLegCaptureRequestV1] {
        &self.legs
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
        if !matches!(
            self.purpose.as_str(),
            "primary_pnl_settlement" | "commission_settlement" | "margin_settlement"
        ) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                format!("unknown conversion route purpose {:?}", self.purpose),
            ));
        }
        validate_id_label(
            "route source asset",
            self.from_asset_id,
            &self.from_asset_name,
        )?;
        validate_id_label(
            "route destination asset",
            self.to_asset_id,
            &self.to_asset_name,
        )?;
        if self.from_asset_id == self.to_asset_id {
            if self.from_asset_name != self.to_asset_name || !self.legs.is_empty() {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                    "identity conversion route must carry the same exact asset and zero legs",
                ));
            }
            return Ok(());
        }
        if self.legs.is_empty() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "non-identity conversion route has no exact quoted legs",
            ));
        }
        let mut expected_id = self.from_asset_id;
        let mut expected_name = self.from_asset_name.as_str();
        let mut visited = HashSet::from([expected_id]);
        for leg in &self.legs {
            leg.validate()?;
            if leg.from_asset_id != expected_id
                || leg.from_asset_name != expected_name
                || !visited.insert(leg.to_asset_id)
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                    "conversion route is discontinuous, name-mismatched, or cyclic",
                ));
            }
            expected_id = leg.to_asset_id;
            expected_name = &leg.to_asset_name;
        }
        if expected_id != self.to_asset_id || expected_name != self.to_asset_name {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "conversion route does not terminate at the exact destination asset",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthCaptureRequestV1 {
    account_id: i64,
    binding: BrokerFinancialTruthBindingV1,
    primary_instrument: ExactQuoteInstrumentV1,
    conversion_routes: Vec<ExactConversionRouteCaptureRequestV1>,
}

impl BrokerFinancialTruthCaptureRequestV1 {
    pub fn new(
        account_id: i64,
        binding: BrokerFinancialTruthBindingV1,
        primary_instrument: ExactQuoteInstrumentV1,
        conversion_routes: Vec<ExactConversionRouteCaptureRequestV1>,
    ) -> Result<Self, BrokerFinancialTruthCaptureErrorV1> {
        let request = Self {
            account_id,
            binding,
            primary_instrument,
            conversion_routes,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn binding(&self) -> &BrokerFinancialTruthBindingV1 {
        &self.binding
    }

    pub const fn window(&self) -> EvidenceWindowV1 {
        self.binding.evaluated_window()
    }

    pub const fn primary_instrument(&self) -> &ExactQuoteInstrumentV1 {
        &self.primary_instrument
    }

    pub fn conversion_routes(&self) -> &[ExactConversionRouteCaptureRequestV1] {
        &self.conversion_routes
    }

    pub(crate) fn required_symbol_ids(&self) -> HashSet<i64> {
        std::iter::once(self.primary_instrument.symbol_id)
            .chain(
                self.conversion_routes
                    .iter()
                    .flat_map(|route| route.legs.iter().map(|leg| leg.instrument.symbol_id)),
            )
            .collect()
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
        if self.account_id <= 0 {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "capture account id must be positive",
            ));
        }
        self.primary_instrument.validate()?;
        if self.primary_instrument.symbol_name
            != self.binding.canonical_dataset_identity().symbol_name()
            || self.primary_instrument.base_asset_id != self.binding.primary_base_asset_id()
            || self.primary_instrument.base_asset_name != self.binding.primary_base_asset_name()
            || self.primary_instrument.quote_asset_id != self.binding.primary_quote_asset_id()
            || self.primary_instrument.quote_asset_name != self.binding.primary_quote_asset_name()
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "primary capture instrument does not match the exact bundle binding",
            ));
        }
        let mut purposes = HashSet::new();
        for route in &self.conversion_routes {
            route.validate()?;
            if route.to_asset_id != self.binding.account_asset_id()
                || route.to_asset_name != self.binding.account_asset_name()
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                    format!(
                        "conversion route {:?} does not terminate at the exact account asset",
                        route.purpose
                    ),
                ));
            }
            if !purposes.insert(route.purpose.as_str()) {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                    format!("duplicate conversion route purpose {:?}", route.purpose),
                ));
            }
        }
        let settlement = self
            .conversion_routes
            .iter()
            .filter(|route| route.purpose == "primary_pnl_settlement")
            .collect::<Vec<_>>();
        if settlement.len() != 1
            || settlement[0].from_asset_id != self.binding.primary_quote_asset_id()
            || settlement[0].from_asset_name != self.binding.primary_quote_asset_name()
            || settlement[0].to_asset_id != self.binding.account_asset_id()
            || settlement[0].to_asset_name != self.binding.account_asset_name()
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
                "capture requires exactly one quote-to-account primary PnL settlement route",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactQuoteCaptureRequestV1 {
    account_id: i64,
    instrument: ExactQuoteInstrumentV1,
    side: QuoteSideV1,
    window: EvidenceWindowV1,
}

impl ExactQuoteCaptureRequestV1 {
    fn new(
        account_id: i64,
        instrument: ExactQuoteInstrumentV1,
        side: QuoteSideV1,
        window: EvidenceWindowV1,
    ) -> Self {
        Self {
            account_id,
            instrument,
            side,
            window,
        }
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn instrument(&self) -> &ExactQuoteInstrumentV1 {
        &self.instrument
    }

    pub const fn side(&self) -> QuoteSideV1 {
        self.side
    }

    pub const fn window(&self) -> EvidenceWindowV1 {
        self.window
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactQuoteSynchronizationCaptureRequestV1 {
    account_id: i64,
    instrument: ExactQuoteInstrumentV1,
    window: EvidenceWindowV1,
}

impl ExactQuoteSynchronizationCaptureRequestV1 {
    fn new(account_id: i64, instrument: ExactQuoteInstrumentV1, window: EvidenceWindowV1) -> Self {
        Self {
            account_id,
            instrument,
            window,
        }
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn instrument(&self) -> &ExactQuoteInstrumentV1 {
        &self.instrument
    }

    pub const fn window(&self) -> EvidenceWindowV1 {
        self.window
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedTickV1 {
    pub(crate) timestamp_ms: i64,
    pub(crate) price: f64,
}

impl CapturedTickV1 {
    pub const fn new(timestamp_ms: i64, price: f64) -> Self {
        Self {
            timestamp_ms,
            price,
        }
    }

    pub const fn timestamp_ms(&self) -> i64 {
        self.timestamp_ms
    }

    pub const fn price(&self) -> f64 {
        self.price
    }
}

/// One untrusted cTrader tick response plus the exact request correlation that
/// produced it. Validation happens only inside the producer before encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedTickPageV1 {
    pub(crate) account_id: i64,
    pub(crate) symbol_id: i64,
    pub(crate) side: QuoteSideV1,
    pub(crate) client_msg_id: String,
    pub(crate) requested_window: EvidenceWindowV1,
    pub(crate) raw_response_json: String,
    pub(crate) ticks: Vec<CapturedTickV1>,
    pub(crate) has_more: bool,
}

impl CapturedTickPageV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        account_id: i64,
        symbol_id: i64,
        side: QuoteSideV1,
        client_msg_id: impl Into<String>,
        requested_window: EvidenceWindowV1,
        raw_response_json: impl Into<String>,
        ticks: Vec<CapturedTickV1>,
        has_more: bool,
    ) -> Self {
        Self {
            account_id,
            symbol_id,
            side,
            client_msg_id: client_msg_id.into(),
            requested_window,
            raw_response_json: raw_response_json.into(),
            ticks,
            has_more,
        }
    }

    pub fn replace_ticks_for_untrusted_capture(&mut self, ticks: Vec<CapturedTickV1>) {
        self.ticks = ticks;
    }

    pub fn set_has_more_for_untrusted_capture(&mut self, has_more: bool) {
        self.has_more = has_more;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedQuoteSideV1 {
    pages_newest_first: Vec<CapturedTickPageV1>,
}

impl CapturedQuoteSideV1 {
    pub fn new(pages_newest_first: Vec<CapturedTickPageV1>) -> Self {
        Self { pages_newest_first }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerEvidenceRowKindV1 {
    QuoteSessionObservation,
    QuoteReplayRule,
    SymbolResponse,
    SymbolContract,
    AccountAssetResponse,
    AccountAssetContract,
    TraderAccountResponse,
    TraderAccountContract,
    PositionUnrealizedPnlResponse,
    PositionUnrealizedPnl,
    OpenPositionReconcileResponse,
    DealResponse,
    CloseDealReconciliation,
}

impl BrokerEvidenceRowKindV1 {
    pub(crate) const fn code(&self) -> u8 {
        match self {
            Self::QuoteSessionObservation => 0,
            Self::QuoteReplayRule => 1,
            Self::SymbolResponse => 2,
            Self::SymbolContract => 3,
            Self::AccountAssetResponse => 4,
            Self::AccountAssetContract => 5,
            Self::TraderAccountResponse => 6,
            Self::TraderAccountContract => 7,
            Self::PositionUnrealizedPnlResponse => 8,
            Self::PositionUnrealizedPnl => 9,
            Self::OpenPositionReconcileResponse => 10,
            Self::DealResponse => 11,
            Self::CloseDealReconciliation => 12,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedBrokerEvidenceRowV1 {
    pub(crate) sequence: u64,
    pub(crate) account_id: i64,
    pub(crate) symbol_id: Option<i64>,
    pub(crate) quote_side: Option<QuoteSideV1>,
    pub(crate) kind: BrokerEvidenceRowKindV1,
    pub(crate) requested_window: Option<EvidenceWindowV1>,
    pub(crate) client_msg_id: String,
    pub(crate) payload_type: u32,
    pub(crate) payload_json: String,
}

impl CapturedBrokerEvidenceRowV1 {
    pub fn new(
        sequence: u64,
        account_id: i64,
        symbol_id: Option<i64>,
        quote_side: Option<QuoteSideV1>,
        kind: BrokerEvidenceRowKindV1,
        requested_window: Option<EvidenceWindowV1>,
        client_msg_id: impl Into<String>,
        payload_type: u32,
        payload_json: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            account_id,
            symbol_id,
            quote_side,
            kind,
            requested_window,
            client_msg_id: client_msg_id.into(),
            payload_type,
            payload_json: payload_json.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedBrokerEvidencePairV1 {
    raw_envelopes: Vec<CapturedBrokerEvidenceRowV1>,
    decoded_records: Vec<CapturedBrokerEvidenceRowV1>,
}

impl CapturedBrokerEvidencePairV1 {
    pub fn new(
        raw_envelopes: Vec<CapturedBrokerEvidenceRowV1>,
        decoded_records: Vec<CapturedBrokerEvidenceRowV1>,
    ) -> Self {
        Self {
            raw_envelopes,
            decoded_records,
        }
    }
}

/// Existing authenticated broker-history sessions implement this contract.
/// Implementations must issue every method over that same session/transport.
/// The producer validates and retains returned bytes but creates no capability.
pub trait ExactBrokerTruthCaptureSessionV1 {
    fn capture_quote_side(
        &mut self,
        request: &ExactQuoteCaptureRequestV1,
    ) -> Result<CapturedQuoteSideV1>;

    fn capture_quote_synchronization(
        &mut self,
        request: &ExactQuoteSynchronizationCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1>;

    fn capture_symbol_contracts(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1>;

    fn capture_position_unrealized_pnl(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1>;

    fn capture_close_deal_reconciliation(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1>;
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedTickRowV1 {
    pub(crate) page_sequence: u64,
    pub(crate) row_sequence: u64,
    pub(crate) timestamp_ms: i64,
    pub(crate) price: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedQuoteSideV1 {
    pub(crate) request: ExactQuoteCaptureRequestV1,
    pub(crate) pages_newest_first: Vec<CapturedTickPageV1>,
    pub(crate) ticks_ascending: Vec<ValidatedTickRowV1>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedEvidencePairV1 {
    pub(crate) raw_envelopes: Vec<CapturedBrokerEvidenceRowV1>,
    pub(crate) decoded_records: Vec<CapturedBrokerEvidenceRowV1>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedSynchronizedQuotesV1 {
    pub(crate) instrument: ExactQuoteInstrumentV1,
    pub(crate) bid: ValidatedQuoteSideV1,
    pub(crate) ask: ValidatedQuoteSideV1,
    pub(crate) synchronization: ValidatedEvidencePairV1,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedConversionLegV1 {
    pub(crate) request: ExactConversionLegCaptureRequestV1,
    pub(crate) quotes: ValidatedSynchronizedQuotesV1,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedConversionRouteV1 {
    pub(crate) request: ExactConversionRouteCaptureRequestV1,
    pub(crate) legs: Vec<ValidatedConversionLegV1>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedBrokerTruthCaptureV1 {
    pub(crate) primary_quotes: ValidatedSynchronizedQuotesV1,
    pub(crate) conversion_routes: Vec<ValidatedConversionRouteV1>,
    pub(crate) symbol_contracts: ValidatedEvidencePairV1,
    pub(crate) unrealized_pnl: ValidatedEvidencePairV1,
    pub(crate) close_deal_reconciliation: ValidatedEvidencePairV1,
}

pub fn capture_and_publish_broker_financial_truth_v1<S, F, B, P>(
    session: &mut S,
    request: &BrokerFinancialTruthCaptureRequestV1,
    capture_work_parent: impl AsRef<Path>,
    store: &BrokerFinancialTruthBundleStoreV1,
    is_cancelled: F,
    begin_publication: B,
) -> Result<BrokerFinancialTruthBundleReceiptV1, BrokerFinancialTruthCaptureErrorV1>
where
    S: ExactBrokerTruthCaptureSessionV1,
    F: Fn() -> bool,
    B: FnOnce() -> Result<P, BrokerFinancialTruthCaptureErrorV1>,
{
    request.validate()?;
    ensure_not_cancelled(&is_cancelled)?;

    let primary_quotes = capture_synchronized_quotes(
        session,
        request.account_id,
        request.primary_instrument.clone(),
        request.window(),
        &is_cancelled,
    )?;

    let mut conversion_routes = Vec::with_capacity(request.conversion_routes.len());
    for route in &request.conversion_routes {
        let mut legs = Vec::with_capacity(route.legs.len());
        for leg in &route.legs {
            let quotes = capture_synchronized_quotes(
                session,
                request.account_id,
                leg.instrument.clone(),
                request.window(),
                &is_cancelled,
            )?;
            legs.push(ValidatedConversionLegV1 {
                request: leg.clone(),
                quotes,
            });
        }
        conversion_routes.push(ValidatedConversionRouteV1 {
            request: route.clone(),
            legs,
        });
    }

    ensure_not_cancelled(&is_cancelled)?;
    let symbol_contracts = session
        .capture_symbol_contracts(request)
        .map_err(|error| capture_failed("exact symbol contracts", error))?;
    ensure_not_cancelled(&is_cancelled)?;
    let symbol_contracts = validate_evidence_pair(
        request.account_id,
        None,
        symbol_contracts,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
        "exact symbol contracts",
    )?;
    validate_required_symbol_contracts(&symbol_contracts, &request.required_symbol_ids())?;

    let unrealized_pnl = session
        .capture_position_unrealized_pnl(request)
        .map_err(|error| capture_failed("broker position unrealized PnL", error))?;
    ensure_not_cancelled(&is_cancelled)?;
    let unrealized_pnl = validate_evidence_pair(
        request.account_id,
        None,
        unrealized_pnl,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingUnrealizedPnl,
        "broker position unrealized PnL",
    )?;
    require_evidence_pair_kinds(
        &unrealized_pnl,
        &[BrokerEvidenceRowKindV1::PositionUnrealizedPnlResponse],
        &[BrokerEvidenceRowKindV1::PositionUnrealizedPnl],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingUnrealizedPnl,
        "broker position unrealized PnL",
    )?;

    let close_deal_reconciliation = session
        .capture_close_deal_reconciliation(request)
        .map_err(|error| capture_failed("close/deal reconciliation", error))?;
    ensure_not_cancelled(&is_cancelled)?;
    let close_deal_reconciliation = validate_evidence_pair(
        request.account_id,
        Some(request.window()),
        close_deal_reconciliation,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingCloseDealReconciliation,
        "close/deal reconciliation",
    )?;
    require_evidence_pair_kinds(
        &close_deal_reconciliation,
        &[
            BrokerEvidenceRowKindV1::OpenPositionReconcileResponse,
            BrokerEvidenceRowKindV1::DealResponse,
        ],
        &[BrokerEvidenceRowKindV1::CloseDealReconciliation],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingCloseDealReconciliation,
        "close/deal reconciliation",
    )?;

    let captured = ValidatedBrokerTruthCaptureV1 {
        primary_quotes,
        conversion_routes,
        symbol_contracts,
        unrealized_pnl,
        close_deal_reconciliation,
    };
    validate_unique_capture_raw_client_msg_ids(&captured)?;
    ensure_not_cancelled(&is_cancelled)?;
    let encoded = encode_broker_truth_capture_v1(request, &captured, capture_work_parent.as_ref())
        .map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EncodingFailed,
                format!("failed to encode exact Vortex evidence: {error}"),
            )
        })?;
    ensure_not_cancelled(&is_cancelled)?;
    // The caller must bridge this to the existing historical admission
    // registry's atomic `begin_publication` transition. Holding the returned
    // permit through `store.publish` closes the cancel-vs-publish race.
    let _publication_permit = begin_publication()?;
    store
        .publish(encoded.manifest(), encoded.sources())
        .map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::PublicationFailed,
                format!("immutable bundle publication failed: {error}"),
            )
        })
}

fn capture_synchronized_quotes<S, F>(
    session: &mut S,
    account_id: i64,
    instrument: ExactQuoteInstrumentV1,
    window: EvidenceWindowV1,
    is_cancelled: &F,
) -> Result<ValidatedSynchronizedQuotesV1, BrokerFinancialTruthCaptureErrorV1>
where
    S: ExactBrokerTruthCaptureSessionV1,
    F: Fn() -> bool,
{
    let bid_request =
        ExactQuoteCaptureRequestV1::new(account_id, instrument.clone(), QuoteSideV1::Bid, window);
    let ask_request =
        ExactQuoteCaptureRequestV1::new(account_id, instrument.clone(), QuoteSideV1::Ask, window);
    ensure_not_cancelled(is_cancelled)?;
    let bid = session
        .capture_quote_side(&bid_request)
        .map_err(|error| capture_failed("explicit Bid pages", error))?;
    ensure_not_cancelled(is_cancelled)?;
    let bid = validate_quote_side(&bid_request, bid)?;
    let ask = session
        .capture_quote_side(&ask_request)
        .map_err(|error| capture_failed("explicit Ask pages", error))?;
    ensure_not_cancelled(is_cancelled)?;
    let ask = validate_quote_side(&ask_request, ask)?;

    let synchronization_request =
        ExactQuoteSynchronizationCaptureRequestV1::new(account_id, instrument.clone(), window);
    let synchronization = session
        .capture_quote_synchronization(&synchronization_request)
        .map_err(|error| capture_failed("quote synchronization/replay rules", error))?;
    ensure_not_cancelled(is_cancelled)?;
    let synchronization = validate_evidence_pair(
        account_id,
        Some(window),
        synchronization,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
        "quote synchronization/replay rules",
    )?;
    require_evidence_pair_kinds(
        &synchronization,
        &[BrokerEvidenceRowKindV1::QuoteSessionObservation],
        &[BrokerEvidenceRowKindV1::QuoteReplayRule],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
        "quote synchronization/replay rules",
    )?;
    validate_quote_synchronization_sides(&synchronization)?;
    validate_quote_synchronization_symbol(&synchronization, instrument.symbol_id())?;

    Ok(ValidatedSynchronizedQuotesV1 {
        instrument,
        bid,
        ask,
        synchronization,
    })
}

fn validate_quote_side(
    request: &ExactQuoteCaptureRequestV1,
    capture: CapturedQuoteSideV1,
) -> Result<ValidatedQuoteSideV1, BrokerFinancialTruthCaptureErrorV1> {
    let pages = capture.pages_newest_first;
    if pages.is_empty() {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::MissingQuotePages,
            format!(
                "{} {:?} capture returned no pages",
                request.instrument.symbol_name, request.side
            ),
        ));
    }
    let mut client_msg_ids = HashSet::new();
    for (page_index, page) in pages.iter().enumerate() {
        if page.account_id != request.account_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
                format!(
                    "{} {:?} page {page_index} account {} differs from exact account {}",
                    request.instrument.symbol_name,
                    request.side,
                    page.account_id,
                    request.account_id
                ),
            ));
        }
        if page.symbol_id != request.instrument.symbol_id || page.side != request.side {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                format!(
                    "{} {:?} page {page_index} has a different symbol or explicit side",
                    request.instrument.symbol_name, request.side
                ),
            ));
        }
        if page.client_msg_id.trim().is_empty() || !client_msg_ids.insert(&page.client_msg_id) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                "quote page clientMsgId is empty or duplicated",
            ));
        }
        validate_raw_tick_envelope(page, request.account_id)?;
        if page.ticks.is_empty() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingQuotePages,
                format!(
                    "{} {:?} page {page_index} contains no decoded ticks",
                    request.instrument.symbol_name, request.side
                ),
            ));
        }
        let mut previous = None;
        for tick in &page.ticks {
            if tick.timestamp_ms < request.window.from_unix_ms_inclusive()
                || tick.timestamp_ms >= request.window.to_unix_ms_exclusive()
                || !tick.price.is_finite()
                || tick.price <= 0.0
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                    format!(
                        "{} {:?} page {page_index} has an out-of-window or invalid tick",
                        request.instrument.symbol_name, request.side
                    ),
                ));
            }
            if previous.is_some_and(|previous| tick.timestamp_ms <= previous) {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::OutOfOrderQuoteRows,
                    format!(
                        "{} {:?} page {page_index} is not strictly timestamp-ordered",
                        request.instrument.symbol_name, request.side
                    ),
                ));
            }
            previous = Some(tick.timestamp_ms);
        }

        if page_index == 0 {
            if page.requested_window != request.window {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                    "first quote page is not bound to the complete exact requested window",
                ));
            }
        } else {
            let newer = &pages[page_index - 1];
            let boundary = newer.ticks[0].timestamp_ms;
            if page.requested_window.from_unix_ms_inclusive()
                != request.window.from_unix_ms_inclusive()
                || page.requested_window.to_unix_ms_exclusive() != boundary
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                    "quote pagination request omitted or changed its exact prior-page boundary",
                ));
            }
            if page.ticks.last().expect("non-empty page").timestamp_ms >= boundary {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::OverlappingQuotePages,
                    format!(
                        "{} {:?} page {page_index} overlaps its newer page",
                        request.instrument.symbol_name, request.side
                    ),
                ));
            }
        }
        if page
            .ticks
            .iter()
            .any(|tick| tick.timestamp_ms >= page.requested_window.to_unix_ms_exclusive())
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::OverlappingQuotePages,
                "quote page contains rows at or beyond its retained exclusive boundary",
            ));
        }
        if page_index + 1 < pages.len() && !page.has_more {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::TruncatedQuotePages,
                "quote capture stopped even though a retained intermediate page said hasMore=false",
            ));
        }
    }
    if pages.last().expect("non-empty pages").has_more {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::TruncatedQuotePages,
            format!(
                "{} {:?} terminal page still reports hasMore=true",
                request.instrument.symbol_name, request.side
            ),
        ));
    }

    let mut ticks_ascending = Vec::new();
    for (page_index, page) in pages.iter().enumerate().rev() {
        for (row_index, tick) in page.ticks.iter().enumerate() {
            ticks_ascending.push(ValidatedTickRowV1 {
                page_sequence: page_index as u64,
                row_sequence: row_index as u64,
                timestamp_ms: tick.timestamp_ms,
                price: tick.price,
            });
        }
    }
    if ticks_ascending
        .windows(2)
        .any(|pair| pair[1].timestamp_ms <= pair[0].timestamp_ms)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::OverlappingQuotePages,
            "combined quote pages overlap or are not globally ordered",
        ));
    }
    Ok(ValidatedQuoteSideV1 {
        request: request.clone(),
        pages_newest_first: pages,
        ticks_ascending,
    })
}

fn validate_raw_tick_envelope(
    page: &CapturedTickPageV1,
    expected_account_id: i64,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    let envelope: Value = serde_json::from_str(&page.raw_response_json).map_err(|error| {
        capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
            format!("tick page raw envelope is not valid JSON: {error}"),
        )
    })?;
    if envelope.get("clientMsgId").and_then(Value::as_str) != Some(page.client_msg_id.as_str())
        || envelope.get("payloadType").and_then(Value::as_u64)
            != Some(u64::from(CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE))
        || envelope
            .pointer("/payload/ctidTraderAccountId")
            .and_then(Value::as_i64)
            != Some(expected_account_id)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
            "tick raw envelope does not match its exact clientMsgId/payloadType/account binding",
        ));
    }
    Ok(())
}

fn validate_evidence_pair(
    expected_account_id: i64,
    expected_window: Option<EvidenceWindowV1>,
    pair: CapturedBrokerEvidencePairV1,
    missing_code: BrokerFinancialTruthCaptureErrorCodeV1,
    label: &str,
) -> Result<ValidatedEvidencePairV1, BrokerFinancialTruthCaptureErrorV1> {
    if pair.raw_envelopes.is_empty() || pair.decoded_records.is_empty() {
        return Err(capture_error(
            missing_code,
            format!("{label} omitted its raw or decoded evidence rows"),
        ));
    }
    validate_evidence_rows(
        expected_account_id,
        expected_window,
        &pair.raw_envelopes,
        label,
        false,
    )?;
    validate_evidence_rows(
        expected_account_id,
        expected_window,
        &pair.decoded_records,
        label,
        true,
    )?;
    Ok(ValidatedEvidencePairV1 {
        raw_envelopes: pair.raw_envelopes,
        decoded_records: pair.decoded_records,
    })
}

fn validate_evidence_rows(
    expected_account_id: i64,
    expected_window: Option<EvidenceWindowV1>,
    rows: &[CapturedBrokerEvidenceRowV1],
    label: &str,
    require_canonical_json: bool,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    let mut client_msg_ids = HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        if row.account_id != expected_account_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
                format!(
                    "{label} row {index} account {} differs from exact account {expected_account_id}",
                    row.account_id
                ),
            ));
        }
        if row.requested_window != expected_window {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} changed or omitted its exact request window"),
            ));
        }
        let row_shape_valid = match row.kind {
            BrokerEvidenceRowKindV1::QuoteSessionObservation => {
                row.quote_side.is_some() && row.symbol_id.is_some_and(|symbol_id| symbol_id > 0)
            }
            BrokerEvidenceRowKindV1::QuoteReplayRule => {
                row.quote_side.is_none() && row.symbol_id.is_some_and(|symbol_id| symbol_id > 0)
            }
            BrokerEvidenceRowKindV1::SymbolResponse | BrokerEvidenceRowKindV1::SymbolContract => {
                row.quote_side.is_none() && row.symbol_id.is_some_and(|symbol_id| symbol_id > 0)
            }
            _ => row.quote_side.is_none(),
        };
        if !row_shape_valid {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} has an invalid symbol/quote-side shape"),
            ));
        }
        if row.sequence != index as u64
            || row.client_msg_id.trim().is_empty()
            || row.payload_type != expected_evidence_payload_type(row.kind)
            || !client_msg_ids.insert(row.client_msg_id.as_str())
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} rows have invalid sequence/clientMsgId/payloadType"),
            ));
        }
        let payload: Value = serde_json::from_str(&row.payload_json).map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} is not valid JSON: {error}"),
            )
        })?;
        if !payload.is_object() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} is not a JSON object"),
            ));
        }
        if require_canonical_json
            && serde_json::to_string(&payload).map_err(|error| {
                capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                    format!("cannot canonicalize {label} row {index}: {error}"),
                )
            })? != row.payload_json
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} decoded row {index} is not canonical JSON"),
            ));
        }
    }
    Ok(())
}

const fn expected_evidence_payload_type(kind: BrokerEvidenceRowKindV1) -> u32 {
    match kind {
        BrokerEvidenceRowKindV1::QuoteSessionObservation
        | BrokerEvidenceRowKindV1::QuoteReplayRule => {
            CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV1::SymbolResponse | BrokerEvidenceRowKindV1::SymbolContract => {
            CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV1::AccountAssetResponse
        | BrokerEvidenceRowKindV1::AccountAssetContract => {
            CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV1::TraderAccountResponse
        | BrokerEvidenceRowKindV1::TraderAccountContract => CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
        BrokerEvidenceRowKindV1::PositionUnrealizedPnlResponse
        | BrokerEvidenceRowKindV1::PositionUnrealizedPnl => {
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV1::OpenPositionReconcileResponse => {
            CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV1::DealResponse
        | BrokerEvidenceRowKindV1::CloseDealReconciliation => {
            CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE
        }
    }
}

fn validate_required_symbol_contracts(
    symbols: &ValidatedEvidencePairV1,
    required_symbol_ids: &HashSet<i64>,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    let captured = symbols
        .decoded_records
        .iter()
        .filter(|row| row.kind == BrokerEvidenceRowKindV1::SymbolContract)
        .filter_map(|row| row.symbol_id)
        .collect::<HashSet<_>>();
    let missing = required_symbol_ids
        .difference(&captured)
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
            format!("decoded exact symbol contracts omitted symbol ids {missing:?}"),
        ));
    }
    require_evidence_pair_kinds(
        symbols,
        &[
            BrokerEvidenceRowKindV1::SymbolResponse,
            BrokerEvidenceRowKindV1::AccountAssetResponse,
            BrokerEvidenceRowKindV1::TraderAccountResponse,
        ],
        &[
            BrokerEvidenceRowKindV1::AccountAssetContract,
            BrokerEvidenceRowKindV1::TraderAccountContract,
        ],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
        "exact symbol/account money contracts",
    )
}

fn validate_quote_synchronization_sides(
    synchronization: &ValidatedEvidencePairV1,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    for side in [QuoteSideV1::Bid, QuoteSideV1::Ask] {
        if !synchronization.raw_envelopes.iter().any(|row| {
            row.kind == BrokerEvidenceRowKindV1::QuoteSessionObservation
                && row.quote_side == Some(side)
        }) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
                format!("quote synchronization observations omitted explicit {side:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_quote_synchronization_symbol(
    synchronization: &ValidatedEvidencePairV1,
    expected_symbol_id: i64,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    if synchronization
        .raw_envelopes
        .iter()
        .chain(&synchronization.decoded_records)
        .any(|row| row.symbol_id != Some(expected_symbol_id))
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            format!(
                "quote synchronization evidence differs from exact symbol {expected_symbol_id}"
            ),
        ));
    }
    Ok(())
}

fn validate_unique_capture_raw_client_msg_ids(
    captured: &ValidatedBrokerTruthCaptureV1,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    let synchronized_quotes = std::iter::once(&captured.primary_quotes)
        .chain(
            captured
                .conversion_routes
                .iter()
                .flat_map(|route| route.legs.iter().map(|leg| &leg.quotes)),
        )
        .collect::<Vec<_>>();
    let mut client_msg_ids = HashSet::new();
    for quotes in &synchronized_quotes {
        for quote in [&quotes.bid, &quotes.ask] {
            for page in &quote.pages_newest_first {
                insert_unique_raw_client_msg_id(&mut client_msg_ids, &page.client_msg_id)?;
            }
        }
        for row in &quotes.synchronization.raw_envelopes {
            insert_unique_raw_client_msg_id(&mut client_msg_ids, &row.client_msg_id)?;
        }
    }
    for pair in [
        &captured.symbol_contracts,
        &captured.unrealized_pnl,
        &captured.close_deal_reconciliation,
    ] {
        for row in &pair.raw_envelopes {
            insert_unique_raw_client_msg_id(&mut client_msg_ids, &row.client_msg_id)?;
        }
    }
    Ok(())
}

fn insert_unique_raw_client_msg_id<'a>(
    client_msg_ids: &mut HashSet<&'a str>,
    client_msg_id: &'a str,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    if !client_msg_ids.insert(client_msg_id) {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            format!(
                "raw clientMsgId {client_msg_id:?} was reused across the single capture session"
            ),
        ));
    }
    Ok(())
}

fn require_evidence_pair_kinds(
    pair: &ValidatedEvidencePairV1,
    required_raw: &[BrokerEvidenceRowKindV1],
    required_decoded: &[BrokerEvidenceRowKindV1],
    missing_code: BrokerFinancialTruthCaptureErrorCodeV1,
    label: &str,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    for (rows, kind, representation) in required_raw
        .iter()
        .map(|kind| (&pair.raw_envelopes, kind, "raw"))
        .chain(
            required_decoded
                .iter()
                .map(|kind| (&pair.decoded_records, kind, "decoded")),
        )
    {
        if !rows.iter().any(|row| &row.kind == kind) {
            return Err(capture_error(
                missing_code,
                format!("{label} omitted required {representation} row kind {kind:?}"),
            ));
        }
    }
    Ok(())
}

fn ensure_not_cancelled(
    is_cancelled: &impl Fn() -> bool,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    if is_cancelled() {
        Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::Cancelled,
            "capture cancellation was observed before immutable publication",
        ))
    } else {
        Ok(())
    }
}

fn validate_id_label(
    field: &str,
    id: i64,
    label: &str,
) -> Result<(), BrokerFinancialTruthCaptureErrorV1> {
    if id <= 0
        || label.trim().is_empty()
        || label != label.trim()
        || label.len() > MAX_CAPTURE_LABEL_BYTES
        || label.chars().any(char::is_control)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidRequest,
            format!("{field} has an invalid id or exact label"),
        ));
    }
    Ok(())
}

fn capture_failed(label: &str, error: anyhow::Error) -> BrokerFinancialTruthCaptureErrorV1 {
    capture_error(
        BrokerFinancialTruthCaptureErrorCodeV1::CaptureFailed,
        format!("{label} capture failed: {error:#}"),
    )
}

fn capture_error(
    code: BrokerFinancialTruthCaptureErrorCodeV1,
    detail: impl Into<String>,
) -> BrokerFinancialTruthCaptureErrorV1 {
    BrokerFinancialTruthCaptureErrorV1::new(code, detail)
}

// ---------------------------------------------------------------------------
// Additive V2 capture surface.
//
// V1 remains readable for the already-published integrity checkpoint, but it
// cannot represent real cTrader request chunks, raw light-symbol authority or
// DealList pagination. New production capture must use this V2 surface.

pub type BrokerFinancialTruthCaptureErrorCodeV2 = BrokerFinancialTruthCaptureErrorCodeV1;
pub type BrokerFinancialTruthCaptureErrorV2 = BrokerFinancialTruthCaptureErrorV1;
pub type ExactQuoteInstrumentV2 = ExactQuoteInstrumentV1;
pub type ExactConversionLegCaptureRequestV2 = ExactConversionLegCaptureRequestV1;
pub type ExactConversionRouteCaptureRequestV2 = ExactConversionRouteCaptureRequestV1;
pub type BrokerFinancialTruthCaptureRequestV2 = BrokerFinancialTruthCaptureRequestV1;
pub type ExactQuoteCaptureRequestV2 = ExactQuoteCaptureRequestV1;
pub type ExactQuoteSynchronizationCaptureRequestV2 = ExactQuoteSynchronizationCaptureRequestV1;
pub type CapturedTickV2 = CapturedTickV1;

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedTickPageV2 {
    pub(crate) account_id: i64,
    pub(crate) symbol_id: i64,
    pub(crate) side: QuoteSideV1,
    pub(crate) chunk_sequence: u64,
    pub(crate) page_sequence_in_chunk: u64,
    pub(crate) client_msg_id: String,
    pub(crate) requested_chunk_window: EvidenceWindowV1,
    pub(crate) requested_page_window: EvidenceWindowV1,
    pub(crate) raw_response_json: String,
    pub(crate) ticks: Vec<CapturedTickV2>,
    pub(crate) has_more: bool,
}

impl CapturedTickPageV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        account_id: i64,
        symbol_id: i64,
        side: QuoteSideV1,
        chunk_sequence: u64,
        page_sequence_in_chunk: u64,
        client_msg_id: impl Into<String>,
        requested_chunk_window: EvidenceWindowV1,
        requested_page_window: EvidenceWindowV1,
        raw_response_json: impl Into<String>,
        ticks: Vec<CapturedTickV2>,
        has_more: bool,
    ) -> Self {
        Self {
            account_id,
            symbol_id,
            side,
            chunk_sequence,
            page_sequence_in_chunk,
            client_msg_id: client_msg_id.into(),
            requested_chunk_window,
            requested_page_window,
            raw_response_json: raw_response_json.into(),
            ticks,
            has_more,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedQuoteSideV2 {
    pages_newest_first: Vec<CapturedTickPageV2>,
}

impl CapturedQuoteSideV2 {
    pub fn new(pages_newest_first: Vec<CapturedTickPageV2>) -> Self {
        Self { pages_newest_first }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerEvidenceRowKindV2 {
    QuoteSessionObservation,
    QuoteReplayRule,
    LightSymbolResponse,
    LightSymbolContract,
    SymbolResponse,
    SymbolContract,
    AccountAssetResponse,
    AccountAssetContract,
    TraderAccountResponse,
    TraderAccountContract,
    PositionUnrealizedPnlResponse,
    PositionUnrealizedPnl,
    OpenPositionReconcileResponse,
    CloseDealReconciliation,
}

impl BrokerEvidenceRowKindV2 {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::QuoteSessionObservation => 0,
            Self::QuoteReplayRule => 1,
            Self::LightSymbolResponse => 2,
            Self::LightSymbolContract => 3,
            Self::SymbolResponse => 4,
            Self::SymbolContract => 5,
            Self::AccountAssetResponse => 6,
            Self::AccountAssetContract => 7,
            Self::TraderAccountResponse => 8,
            Self::TraderAccountContract => 9,
            Self::PositionUnrealizedPnlResponse => 10,
            Self::PositionUnrealizedPnl => 11,
            Self::OpenPositionReconcileResponse => 12,
            Self::CloseDealReconciliation => 13,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedBrokerEvidenceRowV2 {
    pub(crate) sequence: u64,
    pub(crate) account_id: i64,
    pub(crate) symbol_id: Option<i64>,
    pub(crate) quote_side: Option<QuoteSideV1>,
    pub(crate) kind: BrokerEvidenceRowKindV2,
    pub(crate) requested_window: Option<EvidenceWindowV1>,
    pub(crate) client_msg_id: String,
    pub(crate) payload_type: u32,
    pub(crate) payload_json: String,
}

impl CapturedBrokerEvidenceRowV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        account_id: i64,
        symbol_id: Option<i64>,
        quote_side: Option<QuoteSideV1>,
        kind: BrokerEvidenceRowKindV2,
        requested_window: Option<EvidenceWindowV1>,
        client_msg_id: impl Into<String>,
        payload_type: u32,
        payload_json: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            account_id,
            symbol_id,
            quote_side,
            kind,
            requested_window,
            client_msg_id: client_msg_id.into(),
            payload_type,
            payload_json: payload_json.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedBrokerEvidencePairV2 {
    raw_envelopes: Vec<CapturedBrokerEvidenceRowV2>,
    decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
}

impl CapturedBrokerEvidencePairV2 {
    pub fn new(
        raw_envelopes: Vec<CapturedBrokerEvidenceRowV2>,
        decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
    ) -> Self {
        Self {
            raw_envelopes,
            decoded_records,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedQuoteSynchronizationV2 {
    review_identity: ReviewedQuoteReplayRuleIdentityV2,
    evidence: CapturedBrokerEvidencePairV2,
}

impl CapturedQuoteSynchronizationV2 {
    pub fn new(
        review_identity: ReviewedQuoteReplayRuleIdentityV2,
        evidence: CapturedBrokerEvidencePairV2,
    ) -> Self {
        Self {
            review_identity,
            evidence,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedDealPageV2 {
    pub(crate) account_id: i64,
    pub(crate) page_sequence: u64,
    pub(crate) client_msg_id: String,
    pub(crate) requested_window: EvidenceWindowV1,
    pub(crate) max_rows: u32,
    pub(crate) raw_response_json: String,
    pub(crate) deal_execution_timestamps_ms: Vec<i64>,
    pub(crate) has_more: bool,
}

impl CapturedDealPageV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        account_id: i64,
        page_sequence: u64,
        client_msg_id: impl Into<String>,
        requested_window: EvidenceWindowV1,
        max_rows: u32,
        raw_response_json: impl Into<String>,
        deal_execution_timestamps_ms: Vec<i64>,
        has_more: bool,
    ) -> Self {
        Self {
            account_id,
            page_sequence,
            client_msg_id: client_msg_id.into(),
            requested_window,
            max_rows,
            raw_response_json: raw_response_json.into(),
            deal_execution_timestamps_ms,
            has_more,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedCloseDealReconciliationV2 {
    reconcile_raw: CapturedBrokerEvidenceRowV2,
    deal_pages_newest_first: Vec<CapturedDealPageV2>,
    decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
}

impl CapturedCloseDealReconciliationV2 {
    pub fn new(
        reconcile_raw: CapturedBrokerEvidenceRowV2,
        deal_pages_newest_first: Vec<CapturedDealPageV2>,
        decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
    ) -> Self {
        Self {
            reconcile_raw,
            deal_pages_newest_first,
            decoded_records,
        }
    }
}

pub trait ExactBrokerTruthCaptureSessionV2 {
    fn capture_quote_side(
        &mut self,
        request: &ExactQuoteCaptureRequestV2,
    ) -> Result<CapturedQuoteSideV2>;

    fn capture_quote_synchronization(
        &mut self,
        request: &ExactQuoteSynchronizationCaptureRequestV2,
    ) -> Result<CapturedQuoteSynchronizationV2>;

    fn capture_symbol_contracts(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2>;

    fn capture_position_unrealized_pnl(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedBrokerEvidencePairV2>;

    fn capture_close_deal_reconciliation(
        &mut self,
        request: &BrokerFinancialTruthCaptureRequestV2,
    ) -> Result<CapturedCloseDealReconciliationV2>;
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedTickRowV2 {
    pub(crate) chunk_sequence: u64,
    pub(crate) page_sequence_in_chunk: u64,
    pub(crate) row_sequence_in_page: u64,
    pub(crate) timestamp_ms: i64,
    pub(crate) price: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedQuoteSideV2 {
    pub(crate) request: ExactQuoteCaptureRequestV2,
    pub(crate) pages_newest_first: Vec<CapturedTickPageV2>,
    pub(crate) request_chunks_newest_first: Vec<ExactBrokerRequestChunkV2>,
    pub(crate) ticks_ascending: Vec<ValidatedTickRowV2>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedEvidencePairV2 {
    pub(crate) raw_envelopes: Vec<CapturedBrokerEvidenceRowV2>,
    pub(crate) decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedQuoteSynchronizationV2 {
    pub(crate) review_identity: ReviewedQuoteReplayRuleIdentityV2,
    pub(crate) evidence: ValidatedEvidencePairV2,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedSynchronizedQuotesV2 {
    pub(crate) bid: ValidatedQuoteSideV2,
    pub(crate) ask: ValidatedQuoteSideV2,
    pub(crate) synchronization: ValidatedQuoteSynchronizationV2,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedConversionLegV2 {
    pub(crate) request: ExactConversionLegCaptureRequestV2,
    pub(crate) quotes: ValidatedSynchronizedQuotesV2,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedConversionRouteV2 {
    pub(crate) request: ExactConversionRouteCaptureRequestV2,
    pub(crate) legs: Vec<ValidatedConversionLegV2>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedCloseDealReconciliationV2 {
    pub(crate) reconcile_raw: CapturedBrokerEvidenceRowV2,
    pub(crate) deal_pages_newest_first: Vec<CapturedDealPageV2>,
    pub(crate) deal_request_chunk: ExactBrokerRequestChunkV2,
    pub(crate) decoded_records: Vec<CapturedBrokerEvidenceRowV2>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedBrokerTruthCaptureV2 {
    pub(crate) primary_quotes: ValidatedSynchronizedQuotesV2,
    pub(crate) conversion_routes: Vec<ValidatedConversionRouteV2>,
    pub(crate) symbol_contracts: ValidatedEvidencePairV2,
    pub(crate) unrealized_pnl: ValidatedEvidencePairV2,
    pub(crate) close_deal_reconciliation: ValidatedCloseDealReconciliationV2,
}

pub fn capture_and_publish_broker_financial_truth_v2<S, F, B, P>(
    session: &mut S,
    request: &BrokerFinancialTruthCaptureRequestV2,
    capture_work_parent: impl AsRef<Path>,
    store: &BrokerFinancialTruthBundleStoreV1,
    is_cancelled: F,
    begin_publication: B,
) -> Result<BrokerFinancialTruthBundleReceiptV2, BrokerFinancialTruthCaptureErrorV2>
where
    S: ExactBrokerTruthCaptureSessionV2,
    F: Fn() -> bool,
    B: FnOnce() -> Result<P, BrokerFinancialTruthCaptureErrorV2>,
{
    request.validate()?;
    ensure_not_cancelled(&is_cancelled)?;
    let primary_quotes = capture_synchronized_quotes_v2(
        session,
        request.account_id,
        request.primary_instrument.clone(),
        request.window(),
        &is_cancelled,
    )?;

    let mut conversion_routes = Vec::with_capacity(request.conversion_routes.len());
    for route in &request.conversion_routes {
        let mut legs = Vec::with_capacity(route.legs.len());
        for leg in &route.legs {
            let quotes = capture_synchronized_quotes_v2(
                session,
                request.account_id,
                leg.instrument.clone(),
                request.window(),
                &is_cancelled,
            )?;
            legs.push(ValidatedConversionLegV2 {
                request: leg.clone(),
                quotes,
            });
        }
        conversion_routes.push(ValidatedConversionRouteV2 {
            request: route.clone(),
            legs,
        });
    }

    ensure_not_cancelled(&is_cancelled)?;
    let symbol_contracts = session
        .capture_symbol_contracts(request)
        .map_err(|error| capture_failed("V2 exact symbol contracts", error))?;
    let symbol_contracts = validate_evidence_pair_v2(
        request.account_id,
        None,
        symbol_contracts,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
        "V2 exact symbol contracts",
    )?;
    validate_required_symbol_contracts_v2(&symbol_contracts, &request.required_symbol_ids())?;

    ensure_not_cancelled(&is_cancelled)?;
    let unrealized_pnl = session
        .capture_position_unrealized_pnl(request)
        .map_err(|error| capture_failed("V2 broker position unrealized PnL", error))?;
    let unrealized_pnl = validate_evidence_pair_v2(
        request.account_id,
        None,
        unrealized_pnl,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingUnrealizedPnl,
        "V2 broker position unrealized PnL",
    )?;
    require_evidence_pair_kinds_v2(
        &unrealized_pnl,
        &[BrokerEvidenceRowKindV2::PositionUnrealizedPnlResponse],
        &[BrokerEvidenceRowKindV2::PositionUnrealizedPnl],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingUnrealizedPnl,
        "V2 broker position unrealized PnL",
    )?;
    reject_unexpected_evidence_kinds_v2(
        &unrealized_pnl,
        &[BrokerEvidenceRowKindV2::PositionUnrealizedPnlResponse],
        &[BrokerEvidenceRowKindV2::PositionUnrealizedPnl],
        "V2 broker position unrealized PnL",
    )?;

    ensure_not_cancelled(&is_cancelled)?;
    let close_deal_reconciliation = session
        .capture_close_deal_reconciliation(request)
        .map_err(|error| capture_failed("V2 close/deal reconciliation", error))?;
    let close_deal_reconciliation = validate_close_deal_reconciliation_v2(
        request.account_id,
        request.window(),
        close_deal_reconciliation,
    )?;

    let captured = ValidatedBrokerTruthCaptureV2 {
        primary_quotes,
        conversion_routes,
        symbol_contracts,
        unrealized_pnl,
        close_deal_reconciliation,
    };
    validate_unique_capture_raw_client_msg_ids_v2(&captured)?;
    ensure_not_cancelled(&is_cancelled)?;
    let encoded = encode_broker_truth_capture_v2(request, &captured, capture_work_parent.as_ref())
        .map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EncodingFailed,
                format!("failed to encode exact V2 Vortex evidence: {error}"),
            )
        })?;
    ensure_not_cancelled(&is_cancelled)?;
    let _publication_permit = begin_publication()?;
    store
        .publish_v2(encoded.manifest(), encoded.sources())
        .map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::PublicationFailed,
                format!("immutable V2 bundle publication failed: {error}"),
            )
        })
}

fn capture_synchronized_quotes_v2<S, F>(
    session: &mut S,
    account_id: i64,
    instrument: ExactQuoteInstrumentV2,
    window: EvidenceWindowV1,
    is_cancelled: &F,
) -> Result<ValidatedSynchronizedQuotesV2, BrokerFinancialTruthCaptureErrorV2>
where
    S: ExactBrokerTruthCaptureSessionV2,
    F: Fn() -> bool,
{
    let bid_request =
        ExactQuoteCaptureRequestV1::new(account_id, instrument.clone(), QuoteSideV1::Bid, window);
    let ask_request =
        ExactQuoteCaptureRequestV1::new(account_id, instrument.clone(), QuoteSideV1::Ask, window);
    ensure_not_cancelled(is_cancelled)?;
    let bid = session
        .capture_quote_side(&bid_request)
        .map_err(|error| capture_failed("V2 explicit Bid pages", error))?;
    let bid = validate_quote_side_v2(&bid_request, bid)?;
    ensure_not_cancelled(is_cancelled)?;
    let ask = session
        .capture_quote_side(&ask_request)
        .map_err(|error| capture_failed("V2 explicit Ask pages", error))?;
    let ask = validate_quote_side_v2(&ask_request, ask)?;

    let synchronization_request =
        ExactQuoteSynchronizationCaptureRequestV1::new(account_id, instrument.clone(), window);
    ensure_not_cancelled(is_cancelled)?;
    let synchronization = session
        .capture_quote_synchronization(&synchronization_request)
        .map_err(|error| capture_failed("V2 reviewed quote replay rules", error))?;
    synchronization
        .review_identity
        .validate_exact()
        .map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
                format!("invalid V2 replay-rule review identity: {error}"),
            )
        })?;
    let evidence = validate_evidence_pair_v2(
        account_id,
        Some(window),
        synchronization.evidence,
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
        "V2 reviewed quote replay rules",
    )?;
    require_evidence_pair_kinds_v2(
        &evidence,
        &[BrokerEvidenceRowKindV2::QuoteSessionObservation],
        &[BrokerEvidenceRowKindV2::QuoteReplayRule],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
        "V2 reviewed quote replay rules",
    )?;
    reject_unexpected_evidence_kinds_v2(
        &evidence,
        &[BrokerEvidenceRowKindV2::QuoteSessionObservation],
        &[BrokerEvidenceRowKindV2::QuoteReplayRule],
        "V2 reviewed quote replay rules",
    )?;
    validate_quote_synchronization_sides_v2(&evidence)?;
    validate_quote_synchronization_symbol_v2(&evidence, instrument.symbol_id())?;

    Ok(ValidatedSynchronizedQuotesV2 {
        bid,
        ask,
        synchronization: ValidatedQuoteSynchronizationV2 {
            review_identity: synchronization.review_identity,
            evidence,
        },
    })
}

fn validate_quote_side_v2(
    request: &ExactQuoteCaptureRequestV2,
    capture: CapturedQuoteSideV2,
) -> Result<ValidatedQuoteSideV2, BrokerFinancialTruthCaptureErrorV2> {
    let pages = capture.pages_newest_first;
    if pages.is_empty() {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::MissingQuotePages,
            "V2 quote capture returned no exact request pages",
        ));
    }
    let mut client_msg_ids = HashSet::new();
    let mut chunk_boundaries = Vec::<ExactBrokerRequestChunkV2>::new();
    let mut chunk_pages = Vec::<ExactBrokerRequestPageV2>::new();
    let mut current_chunk = 0_u64;
    let mut current_chunk_window = pages[0].requested_chunk_window;

    for page in &pages {
        if page.account_id != request.account_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
                "V2 quote page account differs from the exact request",
            ));
        }
        if page.symbol_id != request.instrument.symbol_id || page.side != request.side {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                "V2 quote page symbol or explicit side differs from the exact request",
            ));
        }
        if page.client_msg_id.trim().is_empty() || !client_msg_ids.insert(&page.client_msg_id) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                "V2 quote page clientMsgId is empty or duplicated",
            ));
        }
        validate_raw_tick_envelope_v2(page, request.account_id)?;
        if page.ticks.is_empty() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingQuotePages,
                "V2 quote page contains no decoded ticks",
            ));
        }
        if page.chunk_sequence < current_chunk
            || page.chunk_sequence > current_chunk.saturating_add(1)
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                "V2 quote chunks are not contiguous newest-first sequences",
            ));
        }
        if page.chunk_sequence != current_chunk {
            chunk_boundaries.push(
                ExactBrokerRequestChunkV2::new(
                    current_chunk,
                    current_chunk_window,
                    std::mem::take(&mut chunk_pages),
                )
                .map_err(contract_to_capture_v2)?,
            );
            current_chunk = page.chunk_sequence;
            current_chunk_window = page.requested_chunk_window;
        }
        if page.page_sequence_in_chunk != chunk_pages.len() as u64
            || page.requested_chunk_window != current_chunk_window
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                "V2 quote page sequence or exact chunk window changed",
            ));
        }
        let mut previous = None;
        for tick in &page.ticks {
            if tick.timestamp_ms < request.window.from_unix_ms_inclusive()
                || tick.timestamp_ms >= request.window.to_unix_ms_exclusive()
                || !tick.price.is_finite()
                || tick.price <= 0.0
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
                    "V2 quote page contains an out-of-window or invalid tick",
                ));
            }
            if previous.is_some_and(|previous| tick.timestamp_ms <= previous) {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::OutOfOrderQuoteRows,
                    "V2 decoded quote page is not strictly ascending",
                ));
            }
            previous = Some(tick.timestamp_ms);
        }
        chunk_pages.push(
            ExactBrokerRequestPageV2::new(
                page.chunk_sequence,
                page.page_sequence_in_chunk,
                page.client_msg_id.clone(),
                page.requested_page_window,
                page.ticks.first().map(|tick| tick.timestamp_ms),
                page.ticks.last().map(|tick| tick.timestamp_ms),
                page.ticks.len() as u64,
                page.has_more,
                None,
            )
            .map_err(contract_to_capture_v2)?,
        );
    }
    chunk_boundaries.push(
        ExactBrokerRequestChunkV2::new(current_chunk, current_chunk_window, chunk_pages)
            .map_err(contract_to_capture_v2)?,
    );
    ExactBrokerRequestChunkV2::validate_quote_partition(request.window, &chunk_boundaries)
        .map_err(contract_to_capture_v2)?;

    let mut ticks_ascending = Vec::new();
    for chunk in chunk_boundaries.iter().rev() {
        for page_boundary in chunk.pages_newest_first().iter().rev() {
            let page = pages
                .iter()
                .find(|page| {
                    page.chunk_sequence == page_boundary.chunk_sequence()
                        && page.page_sequence_in_chunk == page_boundary.page_sequence_in_chunk()
                })
                .expect("validated V2 page boundary has its captured page");
            for (row_index, tick) in page.ticks.iter().enumerate() {
                ticks_ascending.push(ValidatedTickRowV2 {
                    chunk_sequence: page.chunk_sequence,
                    page_sequence_in_chunk: page.page_sequence_in_chunk,
                    row_sequence_in_page: row_index as u64,
                    timestamp_ms: tick.timestamp_ms,
                    price: tick.price,
                });
            }
        }
    }
    if ticks_ascending
        .windows(2)
        .any(|pair| pair[1].timestamp_ms <= pair[0].timestamp_ms)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::OverlappingQuotePages,
            "V2 quote chunks/pages overlap or are not globally ordered",
        ));
    }

    Ok(ValidatedQuoteSideV2 {
        request: request.clone(),
        pages_newest_first: pages,
        request_chunks_newest_first: chunk_boundaries,
        ticks_ascending,
    })
}

fn validate_raw_tick_envelope_v2(
    page: &CapturedTickPageV2,
    expected_account_id: i64,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    let envelope: Value = serde_json::from_str(&page.raw_response_json).map_err(|error| {
        capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
            format!("V2 tick page raw envelope is not valid JSON: {error}"),
        )
    })?;
    if envelope.get("clientMsgId").and_then(Value::as_str) != Some(page.client_msg_id.as_str())
        || envelope.get("payloadType").and_then(Value::as_u64)
            != Some(u64::from(CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE))
        || envelope
            .pointer("/payload/ctidTraderAccountId")
            .and_then(Value::as_i64)
            != Some(expected_account_id)
        || envelope
            .pointer("/payload/hasMore")
            .and_then(Value::as_bool)
            != Some(page.has_more)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
            "V2 tick raw envelope differs from clientMsgId/payloadType/account/raw hasMore",
        ));
    }
    Ok(())
}

fn validate_evidence_pair_v2(
    expected_account_id: i64,
    expected_window: Option<EvidenceWindowV1>,
    pair: CapturedBrokerEvidencePairV2,
    missing_code: BrokerFinancialTruthCaptureErrorCodeV2,
    label: &str,
) -> Result<ValidatedEvidencePairV2, BrokerFinancialTruthCaptureErrorV2> {
    if pair.raw_envelopes.is_empty() || pair.decoded_records.is_empty() {
        return Err(capture_error(
            missing_code,
            format!("{label} omitted its raw or decoded V2 evidence rows"),
        ));
    }
    validate_evidence_rows_v2(
        expected_account_id,
        expected_window,
        &pair.raw_envelopes,
        label,
        false,
    )?;
    validate_evidence_rows_v2(
        expected_account_id,
        expected_window,
        &pair.decoded_records,
        label,
        true,
    )?;

    let raw_client_msg_ids = pair
        .raw_envelopes
        .iter()
        .map(|row| row.client_msg_id.as_str())
        .collect::<HashSet<_>>();
    if pair
        .decoded_records
        .iter()
        .any(|row| !raw_client_msg_ids.contains(row.client_msg_id.as_str()))
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            format!("{label} decoded row is not correlated to a retained raw clientMsgId"),
        ));
    }

    Ok(ValidatedEvidencePairV2 {
        raw_envelopes: pair.raw_envelopes,
        decoded_records: pair.decoded_records,
    })
}

fn validate_evidence_rows_v2(
    expected_account_id: i64,
    expected_window: Option<EvidenceWindowV1>,
    rows: &[CapturedBrokerEvidenceRowV2],
    label: &str,
    require_canonical_json: bool,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    let mut raw_client_msg_ids = HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        if row.account_id != expected_account_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
                format!(
                    "{label} row {index} account {} differs from exact account {expected_account_id}",
                    row.account_id
                ),
            ));
        }
        if row.requested_window != expected_window {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} changed or omitted its exact request window"),
            ));
        }
        let optional_symbol_is_valid = row.symbol_id.map_or(true, |symbol_id| symbol_id > 0);
        let row_shape_valid = optional_symbol_is_valid
            && match row.kind {
                BrokerEvidenceRowKindV2::QuoteSessionObservation => {
                    row.quote_side.is_some() && row.symbol_id.is_some()
                }
                BrokerEvidenceRowKindV2::QuoteReplayRule
                | BrokerEvidenceRowKindV2::LightSymbolContract
                | BrokerEvidenceRowKindV2::SymbolContract => {
                    row.quote_side.is_none() && row.symbol_id.is_some()
                }
                BrokerEvidenceRowKindV2::LightSymbolResponse
                | BrokerEvidenceRowKindV2::AccountAssetResponse
                | BrokerEvidenceRowKindV2::AccountAssetContract
                | BrokerEvidenceRowKindV2::TraderAccountResponse
                | BrokerEvidenceRowKindV2::TraderAccountContract => {
                    row.quote_side.is_none() && row.symbol_id.is_none()
                }
                BrokerEvidenceRowKindV2::SymbolResponse
                | BrokerEvidenceRowKindV2::PositionUnrealizedPnlResponse
                | BrokerEvidenceRowKindV2::PositionUnrealizedPnl
                | BrokerEvidenceRowKindV2::OpenPositionReconcileResponse
                | BrokerEvidenceRowKindV2::CloseDealReconciliation => row.quote_side.is_none(),
            };
        if !row_shape_valid {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} has an invalid symbol/quote-side shape"),
            ));
        }
        if row.sequence != index as u64
            || row.client_msg_id.trim().is_empty()
            || row.client_msg_id != row.client_msg_id.trim()
            || row.client_msg_id.chars().any(char::is_control)
            || row.payload_type != expected_evidence_payload_type_v2(row.kind)
            || (!require_canonical_json && !raw_client_msg_ids.insert(row.client_msg_id.as_str()))
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} rows have invalid sequence/clientMsgId/payloadType"),
            ));
        }
        let payload: Value = serde_json::from_str(&row.payload_json).map_err(|error| {
            capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} is not valid JSON: {error}"),
            )
        })?;
        if !payload.is_object() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} row {index} is not a JSON object"),
            ));
        }
        if require_canonical_json
            && serde_json::to_string(&payload).map_err(|error| {
                capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                    format!("cannot canonicalize {label} row {index}: {error}"),
                )
            })? != row.payload_json
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!("{label} decoded row {index} is not canonical JSON"),
            ));
        }
    }
    Ok(())
}

const fn expected_evidence_payload_type_v2(kind: BrokerEvidenceRowKindV2) -> u32 {
    match kind {
        BrokerEvidenceRowKindV2::QuoteSessionObservation
        | BrokerEvidenceRowKindV2::QuoteReplayRule => {
            CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::LightSymbolResponse
        | BrokerEvidenceRowKindV2::LightSymbolContract => {
            CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::SymbolResponse | BrokerEvidenceRowKindV2::SymbolContract => {
            CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::AccountAssetResponse
        | BrokerEvidenceRowKindV2::AccountAssetContract => {
            CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::TraderAccountResponse
        | BrokerEvidenceRowKindV2::TraderAccountContract => CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
        BrokerEvidenceRowKindV2::PositionUnrealizedPnlResponse
        | BrokerEvidenceRowKindV2::PositionUnrealizedPnl => {
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::OpenPositionReconcileResponse => {
            CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE
        }
        BrokerEvidenceRowKindV2::CloseDealReconciliation => {
            CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE
        }
    }
}

fn validate_required_symbol_contracts_v2(
    symbols: &ValidatedEvidencePairV2,
    required_symbol_ids: &HashSet<i64>,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    require_evidence_pair_kinds_v2(
        symbols,
        &[
            BrokerEvidenceRowKindV2::LightSymbolResponse,
            BrokerEvidenceRowKindV2::SymbolResponse,
            BrokerEvidenceRowKindV2::AccountAssetResponse,
            BrokerEvidenceRowKindV2::TraderAccountResponse,
        ],
        &[
            BrokerEvidenceRowKindV2::LightSymbolContract,
            BrokerEvidenceRowKindV2::SymbolContract,
            BrokerEvidenceRowKindV2::AccountAssetContract,
            BrokerEvidenceRowKindV2::TraderAccountContract,
        ],
        BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
        "V2 exact symbol/account money contracts",
    )?;
    reject_unexpected_evidence_kinds_v2(
        symbols,
        &[
            BrokerEvidenceRowKindV2::LightSymbolResponse,
            BrokerEvidenceRowKindV2::SymbolResponse,
            BrokerEvidenceRowKindV2::AccountAssetResponse,
            BrokerEvidenceRowKindV2::TraderAccountResponse,
        ],
        &[
            BrokerEvidenceRowKindV2::LightSymbolContract,
            BrokerEvidenceRowKindV2::SymbolContract,
            BrokerEvidenceRowKindV2::AccountAssetContract,
            BrokerEvidenceRowKindV2::TraderAccountContract,
        ],
        "V2 exact symbol/account money contracts",
    )?;

    for kind in [
        BrokerEvidenceRowKindV2::LightSymbolContract,
        BrokerEvidenceRowKindV2::SymbolContract,
    ] {
        let captured = symbols
            .decoded_records
            .iter()
            .filter(|row| row.kind == kind)
            .filter_map(|row| row.symbol_id)
            .collect::<HashSet<_>>();
        let missing = required_symbol_ids
            .difference(&captured)
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
                format!("decoded {kind:?} rows omitted required symbol ids {missing:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_quote_synchronization_sides_v2(
    synchronization: &ValidatedEvidencePairV2,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    for side in [QuoteSideV1::Bid, QuoteSideV1::Ask] {
        if !synchronization.raw_envelopes.iter().any(|row| {
            row.kind == BrokerEvidenceRowKindV2::QuoteSessionObservation
                && row.quote_side == Some(side)
        }) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::MissingSynchronizationRules,
                format!("V2 quote-session observations omitted explicit {side:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_quote_synchronization_symbol_v2(
    synchronization: &ValidatedEvidencePairV2,
    expected_symbol_id: i64,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    if synchronization
        .raw_envelopes
        .iter()
        .chain(&synchronization.decoded_records)
        .any(|row| row.symbol_id != Some(expected_symbol_id))
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            format!(
                "V2 quote synchronization evidence differs from exact symbol {expected_symbol_id}"
            ),
        ));
    }
    Ok(())
}

fn validate_close_deal_reconciliation_v2(
    expected_account_id: i64,
    expected_window: EvidenceWindowV1,
    capture: CapturedCloseDealReconciliationV2,
) -> Result<ValidatedCloseDealReconciliationV2, BrokerFinancialTruthCaptureErrorV2> {
    validate_evidence_rows_v2(
        expected_account_id,
        Some(expected_window),
        std::slice::from_ref(&capture.reconcile_raw),
        "V2 open-position reconcile response",
        false,
    )?;
    if capture.reconcile_raw.kind != BrokerEvidenceRowKindV2::OpenPositionReconcileResponse {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::MissingCloseDealReconciliation,
            "V2 close/deal evidence omitted its raw reconcile response",
        ));
    }
    if capture.deal_pages_newest_first.is_empty() || capture.decoded_records.is_empty() {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::MissingCloseDealReconciliation,
            "V2 close/deal evidence omitted exact DealList pages or decoded reconciliation",
        ));
    }

    let mut deal_boundaries = Vec::with_capacity(capture.deal_pages_newest_first.len());
    let mut raw_client_msg_ids = HashSet::new();
    let expected_max_rows = capture.deal_pages_newest_first[0].max_rows;
    for (page_index, page) in capture.deal_pages_newest_first.iter().enumerate() {
        if page.account_id != expected_account_id {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
                "V2 DealList page account differs from the exact request",
            ));
        }
        if page.page_sequence != page_index as u64
            || page.max_rows == 0
            || page.max_rows != expected_max_rows
            || page.client_msg_id.trim().is_empty()
            || !raw_client_msg_ids.insert(page.client_msg_id.as_str())
        {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                "V2 DealList pages changed sequence/maxRows or reused clientMsgId",
            ));
        }
        validate_raw_deal_envelope_v2(page, expected_account_id)?;
        let mut previous = None;
        for timestamp_ms in &page.deal_execution_timestamps_ms {
            if *timestamp_ms < page.requested_window.from_unix_ms_inclusive()
                || *timestamp_ms >= page.requested_window.to_unix_ms_exclusive()
                || previous.is_some_and(|previous| *timestamp_ms <= previous)
            {
                return Err(capture_error(
                    BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                    "V2 DealList page timestamps are out of bounds or not strictly ascending",
                ));
            }
            previous = Some(*timestamp_ms);
        }
        deal_boundaries.push(
            ExactBrokerRequestPageV2::new(
                0,
                page.page_sequence,
                page.client_msg_id.clone(),
                page.requested_window,
                page.deal_execution_timestamps_ms.first().copied(),
                page.deal_execution_timestamps_ms.last().copied(),
                page.deal_execution_timestamps_ms.len() as u64,
                page.has_more,
                Some(page.max_rows),
            )
            .map_err(contract_to_capture_v2)?,
        );
    }
    let deal_request_chunk = ExactBrokerRequestChunkV2::new(0, expected_window, deal_boundaries)
        .map_err(contract_to_capture_v2)?;

    validate_evidence_rows_v2(
        expected_account_id,
        Some(expected_window),
        &capture.decoded_records,
        "V2 decoded close/deal reconciliation",
        true,
    )?;
    if capture
        .decoded_records
        .iter()
        .any(|row| row.kind != BrokerEvidenceRowKindV2::CloseDealReconciliation)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            "V2 decoded close/deal artifact contains another evidence kind",
        ));
    }
    let known_raw_client_msg_ids = std::iter::once(capture.reconcile_raw.client_msg_id.as_str())
        .chain(
            capture
                .deal_pages_newest_first
                .iter()
                .map(|page| page.client_msg_id.as_str()),
        )
        .collect::<HashSet<_>>();
    if capture
        .decoded_records
        .iter()
        .any(|row| !known_raw_client_msg_ids.contains(row.client_msg_id.as_str()))
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            "V2 decoded close/deal row is not correlated to retained raw evidence",
        ));
    }

    Ok(ValidatedCloseDealReconciliationV2 {
        reconcile_raw: capture.reconcile_raw,
        deal_pages_newest_first: capture.deal_pages_newest_first,
        deal_request_chunk,
        decoded_records: capture.decoded_records,
    })
}

fn validate_raw_deal_envelope_v2(
    page: &CapturedDealPageV2,
    expected_account_id: i64,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    let envelope: Value = serde_json::from_str(&page.raw_response_json).map_err(|error| {
        capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            format!("V2 DealList raw envelope is not valid JSON: {error}"),
        )
    })?;
    if envelope.get("clientMsgId").and_then(Value::as_str) != Some(page.client_msg_id.as_str())
        || envelope.get("payloadType").and_then(Value::as_u64)
            != Some(u64::from(CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE))
        || envelope
            .pointer("/payload/ctidTraderAccountId")
            .and_then(Value::as_i64)
            != Some(expected_account_id)
        || envelope
            .pointer("/payload/hasMore")
            .and_then(Value::as_bool)
            != Some(page.has_more)
    {
        return Err(capture_error(
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
            "V2 DealList raw envelope differs from clientMsgId/payloadType/account/raw hasMore",
        ));
    }
    Ok(())
}

fn validate_unique_capture_raw_client_msg_ids_v2(
    captured: &ValidatedBrokerTruthCaptureV2,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    let synchronized_quotes = std::iter::once(&captured.primary_quotes)
        .chain(
            captured
                .conversion_routes
                .iter()
                .flat_map(|route| route.legs.iter().map(|leg| &leg.quotes)),
        )
        .collect::<Vec<_>>();
    let mut client_msg_ids = HashSet::new();
    for quotes in &synchronized_quotes {
        for quote in [&quotes.bid, &quotes.ask] {
            for page in &quote.pages_newest_first {
                insert_unique_raw_client_msg_id(&mut client_msg_ids, &page.client_msg_id)?;
            }
        }
        for row in &quotes.synchronization.evidence.raw_envelopes {
            insert_unique_raw_client_msg_id(&mut client_msg_ids, &row.client_msg_id)?;
        }
    }
    for pair in [&captured.symbol_contracts, &captured.unrealized_pnl] {
        for row in &pair.raw_envelopes {
            insert_unique_raw_client_msg_id(&mut client_msg_ids, &row.client_msg_id)?;
        }
    }
    insert_unique_raw_client_msg_id(
        &mut client_msg_ids,
        &captured
            .close_deal_reconciliation
            .reconcile_raw
            .client_msg_id,
    )?;
    for page in &captured.close_deal_reconciliation.deal_pages_newest_first {
        insert_unique_raw_client_msg_id(&mut client_msg_ids, &page.client_msg_id)?;
    }
    Ok(())
}

fn require_evidence_pair_kinds_v2(
    pair: &ValidatedEvidencePairV2,
    required_raw: &[BrokerEvidenceRowKindV2],
    required_decoded: &[BrokerEvidenceRowKindV2],
    missing_code: BrokerFinancialTruthCaptureErrorCodeV2,
    label: &str,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    for (rows, kind, representation) in required_raw
        .iter()
        .map(|kind| (&pair.raw_envelopes, kind, "raw"))
        .chain(
            required_decoded
                .iter()
                .map(|kind| (&pair.decoded_records, kind, "decoded")),
        )
    {
        if !rows.iter().any(|row| &row.kind == kind) {
            return Err(capture_error(
                missing_code,
                format!("{label} omitted required {representation} row kind {kind:?}"),
            ));
        }
    }
    Ok(())
}

fn reject_unexpected_evidence_kinds_v2(
    pair: &ValidatedEvidencePairV2,
    allowed_raw: &[BrokerEvidenceRowKindV2],
    allowed_decoded: &[BrokerEvidenceRowKindV2],
    label: &str,
) -> Result<(), BrokerFinancialTruthCaptureErrorV2> {
    for (rows, allowed, representation) in [
        (&pair.raw_envelopes, allowed_raw, "raw"),
        (&pair.decoded_records, allowed_decoded, "decoded"),
    ] {
        if let Some(row) = rows.iter().find(|row| !allowed.contains(&row.kind)) {
            return Err(capture_error(
                BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
                format!(
                    "{label} contains unexpected {representation} row kind {:?}",
                    row.kind
                ),
            ));
        }
    }
    Ok(())
}

fn contract_to_capture_v2(
    error: BrokerFinancialTruthContractErrorV1,
) -> BrokerFinancialTruthCaptureErrorV2 {
    capture_error(
        BrokerFinancialTruthCaptureErrorCodeV1::InvalidEvidenceRow,
        format!("invalid exact V2 broker request boundary: {error}"),
    )
}
