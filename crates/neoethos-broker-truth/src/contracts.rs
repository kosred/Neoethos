use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

use neoethos_dataset_contracts::{CanonicalDatasetIdentity, CanonicalDatasetScope};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

pub const BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V1: u16 = 1;
pub const BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V1: &str = "bft1-";
pub const BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1: &str = "broker-financial-truth.manifest.json";

pub(crate) const MAX_MANIFEST_BYTES: u64 = 1_048_576;
const MAX_ARTIFACT_NAME_BYTES: usize = 160;
const MAX_LABEL_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthContractErrorCodeV1 {
    UnsupportedSchemaVersion,
    InvalidDatasetIdentity,
    InvalidBinding,
    InvalidWindow,
    InvalidSha256,
    InvalidArtifact,
    InvalidQuoteEvidence,
    InvalidConversionRoute,
    MissingEvidence,
    DuplicateArtifact,
    InvalidManifest,
    InvalidReceipt,
    Io,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthContractErrorV1 {
    code: BrokerFinancialTruthContractErrorCodeV1,
    detail: String,
}

impl BrokerFinancialTruthContractErrorV1 {
    pub(crate) fn new(
        code: BrokerFinancialTruthContractErrorCodeV1,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> BrokerFinancialTruthContractErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerFinancialTruthContractErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "broker financial truth contract: {}",
            self.detail
        )
    }
}

impl Error for BrokerFinancialTruthContractErrorV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWindowV1 {
    from_unix_ms_inclusive: i64,
    to_unix_ms_exclusive: i64,
}

impl EvidenceWindowV1 {
    pub fn new(
        from_unix_ms_inclusive: i64,
        to_unix_ms_exclusive: i64,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let window = Self {
            from_unix_ms_inclusive,
            to_unix_ms_exclusive,
        };
        window.validate()?;
        Ok(window)
    }

    pub const fn from_unix_ms_inclusive(self) -> i64 {
        self.from_unix_ms_inclusive
    }

    pub const fn to_unix_ms_exclusive(self) -> i64 {
        self.to_unix_ms_exclusive
    }

    pub const fn covers(self, required: Self) -> bool {
        self.from_unix_ms_inclusive <= required.from_unix_ms_inclusive
            && self.to_unix_ms_exclusive >= required.to_unix_ms_exclusive
    }

    pub(crate) fn validate(self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.from_unix_ms_inclusive < 0
            || self.to_unix_ms_exclusive <= self.from_unix_ms_inclusive
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidWindow,
                format!(
                    "invalid half-open evidence window [{}, {})",
                    self.from_unix_ms_inclusive, self.to_unix_ms_exclusive
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthBindingV1 {
    canonical_dataset_identity: CanonicalDatasetIdentity,
    canonical_search_input_receipt_sha256: String,
    evaluated_window: EvidenceWindowV1,
    primary_base_asset_id: i64,
    primary_base_asset_name: String,
    primary_quote_asset_id: i64,
    primary_quote_asset_name: String,
    account_asset_id: i64,
    account_asset_name: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrokerFinancialTruthBindingSerdeV1 {
    canonical_dataset_identity: String,
    canonical_search_input_receipt_sha256: String,
    evaluated_window: EvidenceWindowV1,
    primary_base_asset_id: i64,
    primary_base_asset_name: String,
    primary_quote_asset_id: i64,
    primary_quote_asset_name: String,
    account_asset_id: i64,
    account_asset_name: String,
}

impl BrokerFinancialTruthBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        canonical_dataset_identity: &CanonicalDatasetIdentity,
        canonical_search_input_receipt_sha256: impl Into<String>,
        evaluated_window: EvidenceWindowV1,
        primary_base_asset_id: i64,
        primary_base_asset_name: impl Into<String>,
        primary_quote_asset_id: i64,
        primary_quote_asset_name: impl Into<String>,
        account_asset_id: i64,
        account_asset_name: impl Into<String>,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let binding = Self {
            canonical_dataset_identity: canonical_dataset_identity.clone(),
            canonical_search_input_receipt_sha256: canonical_search_input_receipt_sha256.into(),
            evaluated_window,
            primary_base_asset_id,
            primary_base_asset_name: primary_base_asset_name.into(),
            primary_quote_asset_id,
            primary_quote_asset_name: primary_quote_asset_name.into(),
            account_asset_id,
            account_asset_name: account_asset_name.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub const fn canonical_dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.canonical_dataset_identity
    }

    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub const fn evaluated_window(&self) -> EvidenceWindowV1 {
        self.evaluated_window
    }

    pub const fn primary_base_asset_id(&self) -> i64 {
        self.primary_base_asset_id
    }

    pub fn primary_base_asset_name(&self) -> &str {
        &self.primary_base_asset_name
    }

    pub const fn primary_quote_asset_id(&self) -> i64 {
        self.primary_quote_asset_id
    }

    pub fn primary_quote_asset_name(&self) -> &str {
        &self.primary_quote_asset_name
    }

    pub const fn account_asset_id(&self) -> i64 {
        self.account_asset_id
    }

    pub fn account_asset_name(&self) -> &str {
        &self.account_asset_name
    }

    pub(crate) fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        let CanonicalDatasetScope::CTrader {
            server,
            account_id,
            symbol_id,
            ..
        } = self.canonical_dataset_identity.scope()
        else {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidDatasetIdentity,
                "broker financial truth requires an exact cTrader dataset identity",
            ));
        };
        if server.trim().is_empty() || *account_id <= 0 || *symbol_id <= 0 {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidDatasetIdentity,
                "cTrader dataset identity has an empty server or non-positive account/symbol id",
            ));
        }
        self.evaluated_window.validate()?;
        validate_sha256_hex(
            "canonical search input receipt SHA-256",
            &self.canonical_search_input_receipt_sha256,
        )?;
        for (field, id, name) in [
            (
                "primary base asset",
                self.primary_base_asset_id,
                self.primary_base_asset_name.as_str(),
            ),
            (
                "primary quote asset",
                self.primary_quote_asset_id,
                self.primary_quote_asset_name.as_str(),
            ),
            (
                "account asset",
                self.account_asset_id,
                self.account_asset_name.as_str(),
            ),
        ] {
            validate_asset(field, id, name)?;
        }
        Ok(())
    }
}

impl Serialize for BrokerFinancialTruthBindingV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        BrokerFinancialTruthBindingSerdeV1 {
            canonical_dataset_identity: self.canonical_dataset_identity.to_path_component(),
            canonical_search_input_receipt_sha256: self
                .canonical_search_input_receipt_sha256
                .clone(),
            evaluated_window: self.evaluated_window,
            primary_base_asset_id: self.primary_base_asset_id,
            primary_base_asset_name: self.primary_base_asset_name.clone(),
            primary_quote_asset_id: self.primary_quote_asset_id,
            primary_quote_asset_name: self.primary_quote_asset_name.clone(),
            account_asset_id: self.account_asset_id,
            account_asset_name: self.account_asset_name.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BrokerFinancialTruthBindingV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = BrokerFinancialTruthBindingSerdeV1::deserialize(deserializer)?;
        let identity =
            CanonicalDatasetIdentity::from_path_component(&encoded.canonical_dataset_identity)
                .map_err(serde::de::Error::custom)?;
        Self::new(
            &identity,
            encoded.canonical_search_input_receipt_sha256,
            encoded.evaluated_window,
            encoded.primary_base_asset_id,
            encoded.primary_base_asset_name,
            encoded.primary_quote_asset_id,
            encoded.primary_quote_asset_name,
            encoded.account_asset_id,
            encoded.account_asset_name,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerFinancialTruthVortexSchemaV1 {
    CTraderTickPagesRawV1,
    CTraderTicksDecodedV1,
    CTraderQuoteSessionObservationsRawV1,
    CTraderQuoteReplayRulesDecodedV1,
    CTraderSymbolResponsesRawV1,
    CTraderSymbolContractsDecodedV1,
    CTraderUnrealizedPnlResponsesRawV1,
    CTraderUnrealizedPnlDecodedV1,
    CTraderDealResponsesRawV1,
    CTraderCloseDealReconciliationDecodedV1,
    CTraderTickRequestPagesRawV2,
    CTraderTicksDecodedV2,
    CTraderQuoteSessionObservationsRawV2,
    CTraderReviewedQuoteReplayRulesDecodedV2,
    CTraderLightSymbolResponsesRawV2,
    CTraderSymbolResponsesRawV2,
    CTraderAccountAssetResponsesRawV2,
    CTraderTraderAccountResponsesRawV2,
    CTraderSymbolMoneyContractsDecodedV2,
    CTraderUnrealizedPnlResponsesRawV2,
    CTraderUnrealizedPnlDecodedV2,
    CTraderReconcileResponsesRawV2,
    CTraderDealPagesRawV2,
    CTraderCloseDealReconciliationDecodedV2,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImmutableVortexArtifactV1 {
    relative_path: String,
    schema: BrokerFinancialTruthVortexSchemaV1,
    sha256: String,
    byte_len: u64,
    row_count: u64,
}

impl ImmutableVortexArtifactV1 {
    pub fn new(
        relative_path: impl Into<String>,
        schema: BrokerFinancialTruthVortexSchemaV1,
        sha256: impl Into<String>,
        byte_len: u64,
        row_count: u64,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let artifact = Self {
            relative_path: relative_path.into(),
            schema,
            sha256: sha256.into(),
            byte_len,
            row_count,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn from_file(
        relative_path: impl Into<String>,
        schema: BrokerFinancialTruthVortexSchemaV1,
        row_count: u64,
        source_path: &Path,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let metadata = source_path.metadata().map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::Io,
                format!("cannot inspect {}: {error}", source_path.display()),
            )
        })?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                format!(
                    "broker evidence source {} is not a non-empty regular file",
                    source_path.display()
                ),
            ));
        }
        let sha256 = sha256_file(source_path)?;
        Self::new(relative_path, schema, sha256, metadata.len(), row_count)
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub const fn schema(&self) -> BrokerFinancialTruthVortexSchemaV1 {
        self.schema
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub(crate) fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_relative_artifact_path(&self.relative_path)?;
        validate_sha256_hex("artifact SHA-256", &self.sha256)?;
        if self.byte_len == 0 || self.row_count == 0 {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
                format!(
                    "artifact {} must have non-zero byte_len and row_count",
                    self.relative_path
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCapturedEvidencePairV1 {
    raw_envelopes: ImmutableVortexArtifactV1,
    decoded_records: ImmutableVortexArtifactV1,
}

impl ExactCapturedEvidencePairV1 {
    pub const fn new(
        raw_envelopes: ImmutableVortexArtifactV1,
        decoded_records: ImmutableVortexArtifactV1,
    ) -> Self {
        Self {
            raw_envelopes,
            decoded_records,
        }
    }

    pub const fn raw_envelopes(&self) -> &ImmutableVortexArtifactV1 {
        &self.raw_envelopes
    }

    pub const fn decoded_records(&self) -> &ImmutableVortexArtifactV1 {
        &self.decoded_records
    }

    pub(crate) fn validate_schemas(
        &self,
        expected_raw: BrokerFinancialTruthVortexSchemaV1,
        expected_decoded: BrokerFinancialTruthVortexSchemaV1,
        evidence_name: &str,
    ) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.raw_envelopes.validate()?;
        self.decoded_records.validate()?;
        if self.raw_envelopes.schema != expected_raw
            || self.decoded_records.schema != expected_decoded
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                format!("{evidence_name} has the wrong raw/decoded Vortex schemas"),
            ));
        }
        Ok(())
    }

    pub(crate) fn artifacts(&self) -> [&ImmutableVortexArtifactV1; 2] {
        [&self.raw_envelopes, &self.decoded_records]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteSideV1 {
    Bid,
    Ask,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactQuoteSideEvidenceV1 {
    side: QuoteSideV1,
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    quote_asset_id: i64,
    requested_window: EvidenceWindowV1,
    returned_window: EvidenceWindowV1,
    capture: ExactCapturedEvidencePairV1,
}

impl ExactQuoteSideEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: QuoteSideV1,
        symbol_id: i64,
        symbol_name: impl Into<String>,
        base_asset_id: i64,
        quote_asset_id: i64,
        requested_window: EvidenceWindowV1,
        returned_window: EvidenceWindowV1,
        capture: ExactCapturedEvidencePairV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            side,
            symbol_id,
            symbol_name: symbol_name.into(),
            base_asset_id,
            quote_asset_id,
            requested_window,
            returned_window,
            capture,
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

    pub const fn returned_window(&self) -> EvidenceWindowV1 {
        self.returned_window
    }

    pub const fn capture(&self) -> &ExactCapturedEvidencePairV1 {
        &self.capture
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_id_and_label("quote symbol", self.symbol_id, &self.symbol_name)?;
        if self.base_asset_id <= 0
            || self.quote_asset_id <= 0
            || self.base_asset_id == self.quote_asset_id
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                format!(
                    "quote {} has invalid or equal base/quote asset ids",
                    self.symbol_name
                ),
            ));
        }
        self.requested_window.validate()?;
        self.returned_window.validate()?;
        if !self.returned_window.covers(self.requested_window) {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                format!(
                    "returned {} {:?} window does not cover the requested window",
                    self.symbol_name, self.side
                ),
            ));
        }
        self.capture.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderTickPagesRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV1,
            "quote capture",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynchronizedBidAskEvidenceV1 {
    bid: ExactQuoteSideEvidenceV1,
    ask: ExactQuoteSideEvidenceV1,
    synchronization_rules: ExactCapturedEvidencePairV1,
}

impl SynchronizedBidAskEvidenceV1 {
    pub fn new(
        bid: ExactQuoteSideEvidenceV1,
        ask: ExactQuoteSideEvidenceV1,
        synchronization_rules: ExactCapturedEvidencePairV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let evidence = Self {
            bid,
            ask,
            synchronization_rules,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn bid(&self) -> &ExactQuoteSideEvidenceV1 {
        &self.bid
    }

    pub const fn ask(&self) -> &ExactQuoteSideEvidenceV1 {
        &self.ask
    }

    pub const fn synchronization_rules(&self) -> &ExactCapturedEvidencePairV1 {
        &self.synchronization_rules
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        self.bid.validate()?;
        self.ask.validate()?;
        if self.bid.side != QuoteSideV1::Bid || self.ask.side != QuoteSideV1::Ask {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                "synchronized quote evidence must carry an explicit Bid artifact and Ask artifact",
            ));
        }
        if self.bid.symbol_id != self.ask.symbol_id
            || self.bid.symbol_name != self.ask.symbol_name
            || self.bid.base_asset_id != self.ask.base_asset_id
            || self.bid.quote_asset_id != self.ask.quote_asset_id
            || self.bid.requested_window != self.ask.requested_window
            || self.bid.returned_window != self.ask.returned_window
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidQuoteEvidence,
                "Bid and Ask evidence are not bound to the same symbol/assets/windows",
            ));
        }
        self.synchronization_rules.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteReplayRulesDecodedV1,
            "quote synchronization rule evidence",
        )
    }

    fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.bid
            .capture
            .artifacts()
            .into_iter()
            .chain(self.ask.capture.artifacts())
            .chain(self.synchronization_rules.artifacts())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactConversionLegEvidenceV1 {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    quotes: SynchronizedBidAskEvidenceV1,
}

impl ExactConversionLegEvidenceV1 {
    pub fn new(
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        quotes: SynchronizedBidAskEvidenceV1,
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

    pub const fn to_asset_id(&self) -> i64 {
        self.to_asset_id
    }

    pub const fn quotes(&self) -> &SynchronizedBidAskEvidenceV1 {
        &self.quotes
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_asset(
            "conversion from asset",
            self.from_asset_id,
            &self.from_asset_name,
        )?;
        validate_asset("conversion to asset", self.to_asset_id, &self.to_asset_name)?;
        if self.from_asset_id == self.to_asset_id {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "a quoted conversion leg cannot convert an asset to itself",
            ));
        }
        self.quotes.validate()?;
        let base = self.quotes.bid.base_asset_id;
        let quote = self.quotes.bid.quote_asset_id;
        if !((base == self.from_asset_id && quote == self.to_asset_id)
            || (quote == self.from_asset_id && base == self.to_asset_id))
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "conversion leg assets do not match its exact quoted symbol contract",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactConversionRouteEvidenceV1 {
    purpose: String,
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    legs: Vec<ExactConversionLegEvidenceV1>,
}

impl ExactConversionRouteEvidenceV1 {
    pub fn new(
        purpose: impl Into<String>,
        from_asset_id: i64,
        from_asset_name: impl Into<String>,
        to_asset_id: i64,
        to_asset_name: impl Into<String>,
        legs: Vec<ExactConversionLegEvidenceV1>,
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

    pub const fn to_asset_id(&self) -> i64 {
        self.to_asset_id
    }

    pub fn legs(&self) -> &[ExactConversionLegEvidenceV1] {
        &self.legs
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if !matches!(
            self.purpose.as_str(),
            "primary_pnl_settlement" | "commission_settlement" | "margin_settlement"
        ) {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                format!("unknown conversion route purpose {:?}", self.purpose),
            ));
        }
        validate_asset(
            "route from asset",
            self.from_asset_id,
            &self.from_asset_name,
        )?;
        validate_asset("route to asset", self.to_asset_id, &self.to_asset_name)?;
        if self.from_asset_id == self.to_asset_id {
            if self.from_asset_name != self.to_asset_name || !self.legs.is_empty() {
                return Err(BrokerFinancialTruthContractErrorV1::new(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                    "an identity conversion route must have the same exact asset and zero legs",
                ));
            }
            return Ok(());
        }
        if self.legs.is_empty() {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "a non-identity conversion route has no synchronized quote legs",
            ));
        }
        let mut expected_from = self.from_asset_id;
        let mut visited = HashSet::from([expected_from]);
        for leg in &self.legs {
            leg.validate()?;
            if leg.from_asset_id != expected_from || !visited.insert(leg.to_asset_id) {
                return Err(BrokerFinancialTruthContractErrorV1::new(
                    BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                    "conversion route is discontinuous or cyclic",
                ));
            }
            expected_from = leg.to_asset_id;
        }
        if expected_from != self.to_asset_id {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidConversionRoute,
                "conversion route does not terminate at the exact account asset",
            ));
        }
        Ok(())
    }

    fn artifacts(&self) -> Vec<&ImmutableVortexArtifactV1> {
        self.legs
            .iter()
            .flat_map(|leg| leg.quotes.artifacts())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerFinancialTruthBundleManifestV1 {
    schema_version: u16,
    binding: BrokerFinancialTruthBindingV1,
    primary_quotes: SynchronizedBidAskEvidenceV1,
    conversion_routes: Vec<ExactConversionRouteEvidenceV1>,
    exact_symbol_contracts: ExactCapturedEvidencePairV1,
    broker_position_unrealized_pnl: ExactCapturedEvidencePairV1,
    close_deal_reconciliation: ExactCapturedEvidencePairV1,
}

impl BrokerFinancialTruthBundleManifestV1 {
    pub fn new(
        binding: BrokerFinancialTruthBindingV1,
        primary_quotes: SynchronizedBidAskEvidenceV1,
        conversion_routes: Vec<ExactConversionRouteEvidenceV1>,
        exact_symbol_contracts: ExactCapturedEvidencePairV1,
        broker_position_unrealized_pnl: ExactCapturedEvidencePairV1,
        close_deal_reconciliation: ExactCapturedEvidencePairV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let manifest = Self {
            schema_version: BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V1,
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

    pub const fn primary_quotes(&self) -> &SynchronizedBidAskEvidenceV1 {
        &self.primary_quotes
    }

    pub fn conversion_routes(&self) -> &[ExactConversionRouteEvidenceV1] {
        &self.conversion_routes
    }

    pub const fn exact_symbol_contracts(&self) -> &ExactCapturedEvidencePairV1 {
        &self.exact_symbol_contracts
    }

    pub const fn broker_position_unrealized_pnl(&self) -> &ExactCapturedEvidencePairV1 {
        &self.broker_position_unrealized_pnl
    }

    pub const fn close_deal_reconciliation(&self) -> &ExactCapturedEvidencePairV1 {
        &self.close_deal_reconciliation
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot encode broker truth manifest: {error}"),
            )
        })
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("broker truth manifest exceeds {MAX_MANIFEST_BYTES} bytes"),
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot decode broker truth manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub(crate) fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.schema_version != BROKER_FINANCIAL_TRUTH_BUNDLE_SCHEMA_VERSION_V1 {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::UnsupportedSchemaVersion,
                format!(
                    "unsupported broker truth manifest schema {}",
                    self.schema_version
                ),
            ));
        }
        self.binding.validate()?;
        self.primary_quotes.validate()?;
        let identity = self.binding.canonical_dataset_identity();
        let CanonicalDatasetScope::CTrader {
            symbol_id,
            account_id: _,
            environment: _,
            server: _,
        } = identity.scope()
        else {
            unreachable!("binding validation requires cTrader")
        };
        if self.primary_quotes.bid.symbol_id != *symbol_id
            || self.primary_quotes.bid.symbol_name != identity.symbol_name()
            || self.primary_quotes.bid.base_asset_id != self.binding.primary_base_asset_id
            || self.primary_quotes.bid.quote_asset_id != self.binding.primary_quote_asset_id
            || self.primary_quotes.bid.requested_window != self.binding.evaluated_window
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                "primary Bid/Ask evidence is not bound to the exact dataset symbol/assets/window",
            ));
        }
        for route in &self.conversion_routes {
            route.validate()?;
            for leg in &route.legs {
                if leg.quotes.bid.requested_window != self.binding.evaluated_window {
                    return Err(BrokerFinancialTruthContractErrorV1::new(
                        BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
                        "conversion quote evidence is not bound to the evaluated window",
                    ));
                }
            }
        }
        let settlement_routes: Vec<_> = self
            .conversion_routes
            .iter()
            .filter(|route| route.purpose == "primary_pnl_settlement")
            .collect();
        if settlement_routes.len() != 1
            || settlement_routes[0].from_asset_id != self.binding.primary_quote_asset_id
            || settlement_routes[0].to_asset_id != self.binding.account_asset_id
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::MissingEvidence,
                "manifest must contain exactly one quote-to-account primary_pnl_settlement route",
            ));
        }
        self.exact_symbol_contracts.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolContractsDecodedV1,
            "exact ProtoOASymbol contracts",
        )?;
        self.broker_position_unrealized_pnl.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV1,
            "broker position unrealized PnL",
        )?;
        self.close_deal_reconciliation.validate_schemas(
            BrokerFinancialTruthVortexSchemaV1::CTraderDealResponsesRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV1,
            "close/deal reconciliation",
        )?;

        let mut artifact_paths = HashSet::new();
        for artifact in self.artifacts() {
            artifact.validate()?;
            if !artifact_paths.insert(artifact.relative_path.clone()) {
                return Err(BrokerFinancialTruthContractErrorV1::new(
                    BrokerFinancialTruthContractErrorCodeV1::DuplicateArtifact,
                    format!(
                        "artifact path {} appears more than once",
                        artifact.relative_path
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
                    .flat_map(ExactConversionRouteEvidenceV1::artifacts),
            )
            .chain(self.exact_symbol_contracts.artifacts())
            .chain(self.broker_position_unrealized_pnl.artifacts())
            .chain(self.close_deal_reconciliation.artifacts())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerFinancialTruthBundleReceiptV1 {
    bundle_id: String,
    manifest_sha256: String,
}

impl BrokerFinancialTruthBundleReceiptV1 {
    pub(crate) fn from_manifest_sha256(
        manifest_sha256: String,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt = Self {
            bundle_id: format!("{BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V1}{manifest_sha256}"),
            manifest_sha256,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt: Self = serde_json::from_slice(bytes).map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot decode broker truth receipt: {error}"),
            )
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot encode broker truth receipt: {error}"),
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
        validate_sha256_hex("manifest SHA-256", &self.manifest_sha256)?;
        if self.bundle_id
            != format!(
                "{BROKER_FINANCIAL_TRUTH_BUNDLE_ID_PREFIX_V1}{}",
                self.manifest_sha256
            )
        {
            return Err(BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                "bundle id does not equal bft1- plus the exact manifest SHA-256",
            ));
        }
        Ok(())
    }
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, BrokerFinancialTruthContractErrorV1> {
    let mut file = File::open(path).map_err(|error| {
        BrokerFinancialTruthContractErrorV1::new(
            BrokerFinancialTruthContractErrorCodeV1::Io,
            format!("cannot open {} for hashing: {error}", path.display()),
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            BrokerFinancialTruthContractErrorV1::new(
                BrokerFinancialTruthContractErrorCodeV1::Io,
                format!("cannot hash {}: {error}", path.display()),
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) const fn max_manifest_bytes() -> u64 {
    MAX_MANIFEST_BYTES
}

pub(crate) fn validate_sha256_hex(
    field: &str,
    value: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BrokerFinancialTruthContractErrorV1::new(
            BrokerFinancialTruthContractErrorCodeV1::InvalidSha256,
            format!("{field} must be exactly 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

fn validate_relative_artifact_path(value: &str) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    let path = Path::new(value);
    let mut components = path.components();
    let exactly_one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if value.is_empty()
        || value.len() > MAX_ARTIFACT_NAME_BYTES
        || !value.ends_with(".vortex")
        || !exactly_one_normal_component
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
    {
        return Err(BrokerFinancialTruthContractErrorV1::new(
            BrokerFinancialTruthContractErrorCodeV1::InvalidArtifact,
            format!("artifact path {value:?} must be one safe lowercase .vortex basename"),
        ));
    }
    Ok(())
}

fn validate_id_and_label(
    field: &str,
    id: i64,
    label: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    if id <= 0
        || label.trim().is_empty()
        || label.len() > MAX_LABEL_BYTES
        || label.chars().any(char::is_control)
    {
        return Err(BrokerFinancialTruthContractErrorV1::new(
            BrokerFinancialTruthContractErrorCodeV1::InvalidBinding,
            format!("{field} requires a positive id and non-empty bounded label"),
        ));
    }
    Ok(())
}

fn validate_asset(
    field: &str,
    id: i64,
    name: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    validate_id_and_label(field, id, name)
}
