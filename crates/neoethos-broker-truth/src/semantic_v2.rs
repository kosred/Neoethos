//! Sealed, untrusted V2 Vortex ingress.
//!
//! This module proves only that an integrity-reopened bundle can be decoded
//! under the exact V2 table layouts and that retained raw/decoded bookkeeping
//! is internally consistent. It exposes no decoded financial values publicly;
//! a crate-private primary-quote projection can feed only the sealed research
//! replay boundary. It constructs no capability or permit. A later validator
//! may add promotion authority only after a reviewed complete real-broker
//! fixture and immutable replay-review trust anchor exist.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::Deserialize;
use serde_json::Value;
use vortex_array::dtype::{DType, Nullability, PType, StructFields};
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ArrayRef, ToCanonical};
use vortex_file::OpenOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::{RuntimeSession, RuntimeSessionExt};
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::contracts::{
    BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1, ImmutableVortexArtifactV1, QuoteSideV1,
    sha256_file,
};
use crate::contracts_v2::{
    BrokerFinancialTruthBundleManifestV2, ExactDealReconciliationEvidenceV2,
    ExactQuoteSideEvidenceV2, ExactSymbolContractEvidenceV2, SynchronizedBidAskEvidenceV2,
};
use crate::store::VerifiedImmutableBrokerFinancialTruthBundleV2;

pub const BROKER_FINANCIAL_TRUTH_SEMANTIC_INGRESS_REFUSED_V2: &str =
    "BROKER_FINANCIAL_TRUTH_SEMANTIC_INGRESS_REFUSED_V2";

const CTRADER_TICK_RESPONSE: u32 = 2146;
const CTRADER_LIGHT_SYMBOL_RESPONSE: u32 = 2115;
const CTRADER_FULL_SYMBOL_RESPONSE: u32 = 2117;
const CTRADER_ASSET_RESPONSE: u32 = 2113;
const CTRADER_TRADER_RESPONSE: u32 = 2122;
const CTRADER_RECONCILE_RESPONSE: u32 = 2125;
const CTRADER_DEAL_RESPONSE: u32 = 2134;
const CTRADER_UNREALIZED_PNL_RESPONSE: u32 = 2188;
const MAX_MONEY_DIGITS_V2: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthSemanticIngressErrorCodeV2 {
    ArtifactChanged,
    ArtifactRowCountMismatch,
    VortexReadFailed,
    VortexSchemaMismatch,
    InvalidRawEnvelope,
    RawDecodedMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerFinancialTruthSemanticIngressErrorV2 {
    code: BrokerFinancialTruthSemanticIngressErrorCodeV2,
    artifact: Option<String>,
    detail: String,
}

impl BrokerFinancialTruthSemanticIngressErrorV2 {
    pub const fn code(&self) -> BrokerFinancialTruthSemanticIngressErrorCodeV2 {
        self.code
    }

    pub fn artifact(&self) -> Option<&str> {
        self.artifact.as_deref()
    }
}

impl fmt::Display for BrokerFinancialTruthSemanticIngressErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{BROKER_FINANCIAL_TRUTH_SEMANTIC_INGRESS_REFUSED_V2}"
        )?;
        if let Some(artifact) = &self.artifact {
            write!(formatter, " artifact={artifact}")?;
        }
        write!(formatter, " detail={}", self.detail)
    }
}

impl std::error::Error for BrokerFinancialTruthSemanticIngressErrorV2 {}

/// Opaque proof that every V2 table passed the untrusted structural ingress.
///
/// No decoded row or financial scalar is exposed outside this crate. The
/// crate-private primary-quote projection can only feed the sealed research
/// replay boundary. Holding this value does not satisfy the broker-financial
/// gate and cannot be converted into a permit.
#[derive(Debug)]
pub struct UntrustedBrokerFinancialTruthIngressV2 {
    verified_bundle: VerifiedImmutableBrokerFinancialTruthBundleV2,
    artifact_count: usize,
    primary_quote_replay: StructurallyVerifiedPrimaryBidAskQuoteReplayV2,
}

impl UntrustedBrokerFinancialTruthIngressV2 {
    pub const fn artifact_count(&self) -> usize {
        self.artifact_count
    }

    pub const fn bundle_schema_version(&self) -> u16 {
        self.verified_bundle.manifest().schema_version()
    }

    pub(crate) fn into_primary_quote_replay(
        self,
    ) -> StructurallyVerifiedPrimaryBidAskQuoteReplayV2 {
        self.primary_quote_replay
    }
}

#[derive(Debug)]
pub(crate) struct StructurallyVerifiedQuoteReplayRowV2 {
    pub(crate) request_chunk_index: u64,
    pub(crate) response_page_index: u64,
    pub(crate) row_index: u64,
    pub(crate) timestamp_unix_ms: i64,
    pub(crate) price: f64,
}

#[derive(Debug)]
pub(crate) struct StructurallyVerifiedQuoteSideReplayV2 {
    pub(crate) side: QuoteSideV1,
    pub(crate) account_id: i64,
    pub(crate) symbol_id: i64,
    pub(crate) requested_window: EvidenceWindowV1,
    pub(crate) raw_response_sha256: String,
    pub(crate) decoded_records_sha256: String,
    pub(crate) quote_records: Vec<StructurallyVerifiedQuoteReplayRowV2>,
}

#[derive(Debug)]
pub(crate) struct StructurallyVerifiedPrimaryBidAskQuoteReplayV2 {
    pub(crate) symbol_name: String,
    pub(crate) reviewed_replay_rule_identity_sha256: String,
    pub(crate) reviewed_rules_sha256: String,
    pub(crate) bid: StructurallyVerifiedQuoteSideReplayV2,
    pub(crate) ask: StructurallyVerifiedQuoteSideReplayV2,
}

/// Decode and cross-check one already integrity-reopened V2 bundle without
/// creating semantic authority.
pub fn inspect_untrusted_broker_financial_truth_bundle_v2(
    verified_bundle: VerifiedImmutableBrokerFinancialTruthBundleV2,
) -> Result<UntrustedBrokerFinancialTruthIngressV2, BrokerFinancialTruthSemanticIngressErrorV2> {
    let reader = VortexIngressReaderV2::new();
    let manifest = verified_bundle.manifest();
    let artifacts = manifest.artifacts();
    let mut tables = BTreeMap::new();
    for artifact in &artifacts {
        let table = read_and_decode_artifact(&reader, &verified_bundle, artifact)?;
        if tables
            .insert(artifact.relative_path().to_owned(), table)
            .is_some()
        {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                "duplicate artifact path reached semantic ingress",
            ));
        }
    }

    let exact_symbols = exact_symbol_bindings(manifest)?;
    let symbol_digits = validate_symbol_contracts(
        manifest.exact_symbol_contracts(),
        manifest,
        &tables,
        &exact_symbols,
    )?;
    validate_synchronized_quotes(
        manifest.primary_quotes(),
        binding_account_id(manifest),
        &tables,
        &symbol_digits,
    )?;
    for route in manifest.conversion_routes() {
        for leg in route.legs() {
            validate_synchronized_quotes(
                leg.quotes(),
                binding_account_id(manifest),
                &tables,
                &symbol_digits,
            )?;
        }
    }
    validate_unrealized_pnl(manifest, &tables)?;
    validate_close_deal_reconciliation(manifest.close_deal_reconciliation(), &tables)?;

    let artifact_count = artifacts.len();
    let primary_quote_replay = structurally_verified_primary_quote_replay(manifest, &tables)?;
    drop(artifacts);
    Ok(UntrustedBrokerFinancialTruthIngressV2 {
        verified_bundle,
        artifact_count,
        primary_quote_replay,
    })
}

fn structurally_verified_primary_quote_replay(
    manifest: &BrokerFinancialTruthBundleManifestV2,
    tables: &BTreeMap<String, ArtifactTableV2>,
) -> Result<
    StructurallyVerifiedPrimaryBidAskQuoteReplayV2,
    BrokerFinancialTruthSemanticIngressErrorV2,
> {
    let primary = manifest.primary_quotes();
    let account_id = binding_account_id(manifest);
    Ok(StructurallyVerifiedPrimaryBidAskQuoteReplayV2 {
        symbol_name: primary.bid().symbol_name().to_owned(),
        reviewed_replay_rule_identity_sha256: primary
            .replay_rule()
            .identity()
            .identity_sha256()
            .to_owned(),
        reviewed_rules_sha256: primary.replay_rule().rules_decoded().sha256().to_owned(),
        bid: structurally_verified_quote_side_replay(primary.bid(), account_id, tables)?,
        ask: structurally_verified_quote_side_replay(primary.ask(), account_id, tables)?,
    })
}

fn structurally_verified_quote_side_replay(
    quote: &ExactQuoteSideEvidenceV2,
    account_id: i64,
    tables: &BTreeMap<String, ArtifactTableV2>,
) -> Result<StructurallyVerifiedQuoteSideReplayV2, BrokerFinancialTruthSemanticIngressErrorV2> {
    let quote_records = decoded_tick_rows(tables, quote.decoded_ticks())?
        .iter()
        .map(|row| StructurallyVerifiedQuoteReplayRowV2 {
            request_chunk_index: row.chunk_sequence,
            response_page_index: row.page_sequence_in_chunk,
            row_index: row.row_sequence_in_page,
            timestamp_unix_ms: row.timestamp_ms,
            price: row.price,
        })
        .collect();
    Ok(StructurallyVerifiedQuoteSideReplayV2 {
        side: quote.side(),
        account_id,
        symbol_id: quote.symbol_id(),
        requested_window: quote.requested_window(),
        raw_response_sha256: quote.raw_pages().sha256().to_owned(),
        decoded_records_sha256: quote.decoded_ticks().sha256().to_owned(),
        quote_records,
    })
}

fn read_and_decode_artifact(
    reader: &VortexIngressReaderV2,
    bundle: &VerifiedImmutableBrokerFinancialTruthBundleV2,
    artifact: &ImmutableVortexArtifactV1,
) -> Result<ArtifactTableV2, BrokerFinancialTruthSemanticIngressErrorV2> {
    let path = bundle.artifact_path(artifact);
    let artifact_name = artifact.relative_path();
    let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
        ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            format!("cannot reopen exact artifact metadata: {error}"),
        )
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != artifact.byte_len()
    {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            "artifact type or byte length changed after integrity reopen",
        ));
    }
    let digest_before = sha256_file(&path).map_err(|error| {
        ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            format!("cannot hash artifact before Vortex read: {error}"),
        )
    })?;
    if digest_before != artifact.sha256() {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            "artifact digest changed after integrity reopen",
        ));
    }

    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader
            .read_array(&path)
            .and_then(|array| decode_artifact_table(artifact, &array))
    }));
    let table = match decoded {
        Ok(result) => result?,
        Err(payload) => {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
                Some(artifact_name),
                format!("Vortex parser panicked: {}", panic_message(payload)),
            ));
        }
    };

    let digest_after = sha256_file(&path).map_err(|error| {
        ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            format!("cannot hash artifact after Vortex read: {error}"),
        )
    })?;
    if digest_after != digest_before {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactChanged,
            Some(artifact_name),
            "artifact bytes changed while Vortex semantics were decoded",
        ));
    }
    Ok(table)
}

struct VortexIngressReaderV2 {
    runtime: CurrentThreadRuntime,
    session: VortexSession,
}

impl VortexIngressReaderV2 {
    fn new() -> Self {
        let runtime = CurrentThreadRuntime::new();
        let mut session = VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
            .with::<RuntimeSession>()
            .with_handle(runtime.handle());
        vortex_file::register_default_encodings(&mut session);
        Self { runtime, session }
    }

    fn read_array(
        &self,
        path: &Path,
    ) -> Result<ArrayRef, BrokerFinancialTruthSemanticIngressErrorV2> {
        let file = self
            .runtime
            .block_on(self.session.open_options().open_path(path))
            .map_err(|error| {
                ingress_error(
                    BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
                    path.file_name().and_then(|name| name.to_str()),
                    format!("cannot open Vortex footer/layout: {error}"),
                )
            })?;
        let stream = file
            .scan()
            .map_err(|error| {
                ingress_error(
                    BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
                    path.file_name().and_then(|name| name.to_str()),
                    format!("cannot scan Vortex layout: {error}"),
                )
            })?
            .into_array_stream()
            .map_err(|error| {
                ingress_error(
                    BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
                    path.file_name().and_then(|name| name.to_str()),
                    format!("cannot construct Vortex array stream: {error}"),
                )
            })?;
        self.runtime.block_on(stream.read_all()).map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
                path.file_name().and_then(|name| name.to_str()),
                format!("cannot materialize Vortex array: {error}"),
            )
        })
    }
}

enum ArtifactTableV2 {
    RawTickPages(Vec<RawTickPageRowV2>),
    DecodedTicks(Vec<DecodedTickRowV2>),
    Evidence(Vec<EvidenceRowV2>),
    RawDealPages(Vec<RawDealPageRowV2>),
}

#[derive(Clone)]
struct RawTickPageRowV2 {
    chunk_sequence: u64,
    page_sequence_in_chunk: u64,
    account_id: i64,
    symbol_id: i64,
    quote_side: u8,
    client_msg_id: String,
    chunk_from: i64,
    chunk_to: i64,
    page_from: i64,
    page_to: i64,
    first_tick: i64,
    last_tick: i64,
    decoded_count: u64,
    has_more: u8,
    raw_response_json: String,
}

#[derive(Clone)]
struct DecodedTickRowV2 {
    chunk_sequence: u64,
    page_sequence_in_chunk: u64,
    row_sequence_in_page: u64,
    account_id: i64,
    symbol_id: i64,
    quote_side: u8,
    timestamp_ms: i64,
    price: f64,
}

#[derive(Clone)]
struct EvidenceRowV2 {
    account_id: i64,
    evidence_kind: u8,
    symbol_id: Option<i64>,
    quote_side: Option<u8>,
    requested_window: Option<EvidenceWindowV1>,
    client_msg_id: String,
    payload_type: u32,
    payload: Value,
}

#[derive(Clone)]
struct RawDealPageRowV2 {
    chunk_sequence: u64,
    page_sequence_in_chunk: u64,
    account_id: i64,
    client_msg_id: String,
    chunk_from: i64,
    chunk_to: i64,
    page_from: i64,
    page_to: i64,
    max_rows: u32,
    has_events: u8,
    first_event: i64,
    last_event: i64,
    decoded_count: u64,
    has_more: u8,
    raw_response_json: String,
}

#[derive(Clone, Copy)]
enum ColumnTypeV2 {
    U8,
    U32,
    U64,
    I64,
    F64,
    Utf8,
}

impl ColumnTypeV2 {
    const fn dtype(self) -> DType {
        match self {
            Self::U8 => DType::Primitive(PType::U8, Nullability::NonNullable),
            Self::U32 => DType::Primitive(PType::U32, Nullability::NonNullable),
            Self::U64 => DType::Primitive(PType::U64, Nullability::NonNullable),
            Self::I64 => DType::Primitive(PType::I64, Nullability::NonNullable),
            Self::F64 => DType::Primitive(PType::F64, Nullability::NonNullable),
            Self::Utf8 => DType::Utf8(Nullability::NonNullable),
        }
    }
}

const RAW_TICK_FIELDS: [(&str, ColumnTypeV2); 15] = [
    ("chunk_sequence", ColumnTypeV2::U64),
    ("page_sequence_in_chunk", ColumnTypeV2::U64),
    ("account_id", ColumnTypeV2::I64),
    ("symbol_id", ColumnTypeV2::I64),
    ("quote_side", ColumnTypeV2::U8),
    ("client_msg_id", ColumnTypeV2::Utf8),
    ("chunk_from_unix_ms_inclusive", ColumnTypeV2::I64),
    ("chunk_to_unix_ms_exclusive", ColumnTypeV2::I64),
    ("page_from_unix_ms_inclusive", ColumnTypeV2::I64),
    ("page_to_unix_ms_exclusive", ColumnTypeV2::I64),
    ("first_tick_timestamp_ms", ColumnTypeV2::I64),
    ("last_tick_timestamp_ms", ColumnTypeV2::I64),
    ("decoded_tick_count", ColumnTypeV2::U64),
    ("has_more", ColumnTypeV2::U8),
    ("raw_response_json", ColumnTypeV2::Utf8),
];

const DECODED_TICK_FIELDS: [(&str, ColumnTypeV2); 8] = [
    ("chunk_sequence", ColumnTypeV2::U64),
    ("page_sequence_in_chunk", ColumnTypeV2::U64),
    ("row_sequence_in_page", ColumnTypeV2::U64),
    ("account_id", ColumnTypeV2::I64),
    ("symbol_id", ColumnTypeV2::I64),
    ("quote_side", ColumnTypeV2::U8),
    ("timestamp_ms", ColumnTypeV2::I64),
    ("price", ColumnTypeV2::F64),
];

const EVIDENCE_FIELDS: [(&str, ColumnTypeV2); 13] = [
    ("sequence", ColumnTypeV2::U64),
    ("account_id", ColumnTypeV2::I64),
    ("evidence_kind", ColumnTypeV2::U8),
    ("has_symbol_id", ColumnTypeV2::U8),
    ("symbol_id", ColumnTypeV2::I64),
    ("has_quote_side", ColumnTypeV2::U8),
    ("quote_side", ColumnTypeV2::U8),
    ("has_requested_window", ColumnTypeV2::U8),
    ("requested_from_unix_ms_inclusive", ColumnTypeV2::I64),
    ("requested_to_unix_ms_exclusive", ColumnTypeV2::I64),
    ("client_msg_id", ColumnTypeV2::Utf8),
    ("payload_type", ColumnTypeV2::U32),
    ("payload_json", ColumnTypeV2::Utf8),
];

const RAW_DEAL_FIELDS: [(&str, ColumnTypeV2); 15] = [
    ("chunk_sequence", ColumnTypeV2::U64),
    ("page_sequence_in_chunk", ColumnTypeV2::U64),
    ("account_id", ColumnTypeV2::I64),
    ("client_msg_id", ColumnTypeV2::Utf8),
    ("chunk_from_unix_ms_inclusive", ColumnTypeV2::I64),
    ("chunk_to_unix_ms_exclusive", ColumnTypeV2::I64),
    ("page_from_unix_ms_inclusive", ColumnTypeV2::I64),
    ("page_to_unix_ms_exclusive", ColumnTypeV2::I64),
    ("max_rows", ColumnTypeV2::U32),
    ("has_events", ColumnTypeV2::U8),
    ("first_deal_execution_timestamp_ms", ColumnTypeV2::I64),
    ("last_deal_execution_timestamp_ms", ColumnTypeV2::I64),
    ("decoded_deal_count", ColumnTypeV2::U64),
    ("has_more", ColumnTypeV2::U8),
    ("raw_response_json", ColumnTypeV2::Utf8),
];

fn decode_artifact_table(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
) -> Result<ArtifactTableV2, BrokerFinancialTruthSemanticIngressErrorV2> {
    if array.len() as u64 != artifact.row_count() {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactRowCountMismatch,
            Some(artifact.relative_path()),
            format!(
                "manifest rows={} Vortex rows={}",
                artifact.row_count(),
                array.len()
            ),
        ));
    }
    match artifact.schema() {
        BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2 => {
            validate_dtype(artifact, array, &RAW_TICK_FIELDS)?;
            Ok(ArtifactTableV2::RawTickPages(decode_raw_tick_rows(
                artifact, array,
            )?))
        }
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2 => {
            validate_dtype(artifact, array, &DECODED_TICK_FIELDS)?;
            Ok(ArtifactTableV2::DecodedTicks(decode_tick_rows(
                artifact, array,
            )?))
        }
        BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2 => {
            validate_dtype(artifact, array, &RAW_DEAL_FIELDS)?;
            Ok(ArtifactTableV2::RawDealPages(decode_deal_rows(
                artifact, array,
            )?))
        }
        schema if is_v2_evidence_schema(schema) => {
            validate_dtype(artifact, array, &EVIDENCE_FIELDS)?;
            Ok(ArtifactTableV2::Evidence(decode_evidence_rows(
                artifact, array,
            )?))
        }
        _ => Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
            Some(artifact.relative_path()),
            "semantic V2 ingress refuses legacy or unknown artifact schemas",
        )),
    }
}

const fn is_v2_evidence_schema(schema: BrokerFinancialTruthVortexSchemaV1) -> bool {
    matches!(
        schema,
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2
            | BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2
    )
}

fn validate_dtype(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    fields: &[(&str, ColumnTypeV2)],
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let expected = DType::Struct(
        StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.dtype()))),
        Nullability::NonNullable,
    );
    if array.dtype() != &expected {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
            Some(artifact.relative_path()),
            format!("expected {expected}, received {}", array.dtype()),
        ));
    }
    if !array.all_valid().map_err(|error| {
        ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
            Some(artifact.relative_path()),
            format!("cannot inspect root validity: {error}"),
        )
    })? {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
            Some(artifact.relative_path()),
            "root struct contains null rows",
        ));
    }
    let structure = array.to_struct();
    for (name, _) in fields {
        let field = structure.unmasked_field_by_name(name).map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                format!("cannot resolve exact field {name}: {error}"),
            )
        })?;
        if !field.all_valid().map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                format!("cannot inspect {name} validity: {error}"),
            )
        })? {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                format!("field {name} contains nulls"),
            ));
        }
    }
    Ok(())
}

fn decode_raw_tick_rows(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
) -> Result<Vec<RawTickPageRowV2>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let chunk_sequence = u64_column(artifact, array, "chunk_sequence")?;
    let page_sequence = u64_column(artifact, array, "page_sequence_in_chunk")?;
    let account_id = i64_column(artifact, array, "account_id")?;
    let symbol_id = i64_column(artifact, array, "symbol_id")?;
    let quote_side = u8_column(artifact, array, "quote_side")?;
    let client_msg_id = utf8_column(artifact, array, "client_msg_id")?;
    let chunk_from = i64_column(artifact, array, "chunk_from_unix_ms_inclusive")?;
    let chunk_to = i64_column(artifact, array, "chunk_to_unix_ms_exclusive")?;
    let page_from = i64_column(artifact, array, "page_from_unix_ms_inclusive")?;
    let page_to = i64_column(artifact, array, "page_to_unix_ms_exclusive")?;
    let first_tick = i64_column(artifact, array, "first_tick_timestamp_ms")?;
    let last_tick = i64_column(artifact, array, "last_tick_timestamp_ms")?;
    let decoded_count = u64_column(artifact, array, "decoded_tick_count")?;
    let has_more = u8_column(artifact, array, "has_more")?;
    let raw_response_json = utf8_column(artifact, array, "raw_response_json")?;
    let mut rows = Vec::with_capacity(array.len());
    for index in 0..array.len() {
        let row = RawTickPageRowV2 {
            chunk_sequence: chunk_sequence[index],
            page_sequence_in_chunk: page_sequence[index],
            account_id: account_id[index],
            symbol_id: symbol_id[index],
            quote_side: quote_side[index],
            client_msg_id: client_msg_id[index].clone(),
            chunk_from: chunk_from[index],
            chunk_to: chunk_to[index],
            page_from: page_from[index],
            page_to: page_to[index],
            first_tick: first_tick[index],
            last_tick: last_tick[index],
            decoded_count: decoded_count[index],
            has_more: has_more[index],
            raw_response_json: raw_response_json[index].clone(),
        };
        if row.account_id <= 0
            || row.symbol_id <= 0
            || row.quote_side > 1
            || row.client_msg_id.trim().is_empty()
            || row.client_msg_id != row.client_msg_id.trim()
            || row.chunk_from >= row.chunk_to
            || row.page_from >= row.page_to
            || row.page_from != row.chunk_from
            || row.page_to > row.chunk_to
            || row.first_tick > row.last_tick
            || row.first_tick < row.page_from
            || row.last_tick >= row.page_to
            || row.decoded_count == 0
            || row.has_more > 1
        {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
                Some(artifact.relative_path()),
                format!("raw tick-page row {index} has invalid exact metadata"),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn decode_tick_rows(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
) -> Result<Vec<DecodedTickRowV2>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let chunk_sequence = u64_column(artifact, array, "chunk_sequence")?;
    let page_sequence = u64_column(artifact, array, "page_sequence_in_chunk")?;
    let row_sequence = u64_column(artifact, array, "row_sequence_in_page")?;
    let account_id = i64_column(artifact, array, "account_id")?;
    let symbol_id = i64_column(artifact, array, "symbol_id")?;
    let quote_side = u8_column(artifact, array, "quote_side")?;
    let timestamp_ms = i64_column(artifact, array, "timestamp_ms")?;
    let price = f64_column(artifact, array, "price")?;
    let mut rows = Vec::with_capacity(array.len());
    for index in 0..array.len() {
        let row = DecodedTickRowV2 {
            chunk_sequence: chunk_sequence[index],
            page_sequence_in_chunk: page_sequence[index],
            row_sequence_in_page: row_sequence[index],
            account_id: account_id[index],
            symbol_id: symbol_id[index],
            quote_side: quote_side[index],
            timestamp_ms: timestamp_ms[index],
            price: price[index],
        };
        if row.account_id <= 0
            || row.symbol_id <= 0
            || row.quote_side > 1
            || row.timestamp_ms < 0
            || !row.price.is_finite()
            || row.price <= 0.0
        {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
                Some(artifact.relative_path()),
                format!("decoded tick row {index} has invalid values"),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn decode_evidence_rows(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
) -> Result<Vec<EvidenceRowV2>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let sequence = u64_column(artifact, array, "sequence")?;
    let account_id = i64_column(artifact, array, "account_id")?;
    let evidence_kind = u8_column(artifact, array, "evidence_kind")?;
    let has_symbol_id = u8_column(artifact, array, "has_symbol_id")?;
    let symbol_id = i64_column(artifact, array, "symbol_id")?;
    let has_quote_side = u8_column(artifact, array, "has_quote_side")?;
    let quote_side = u8_column(artifact, array, "quote_side")?;
    let has_window = u8_column(artifact, array, "has_requested_window")?;
    let window_from = i64_column(artifact, array, "requested_from_unix_ms_inclusive")?;
    let window_to = i64_column(artifact, array, "requested_to_unix_ms_exclusive")?;
    let client_msg_id = utf8_column(artifact, array, "client_msg_id")?;
    let payload_type = u32_column(artifact, array, "payload_type")?;
    let payload_json = utf8_column(artifact, array, "payload_json")?;
    let mut rows = Vec::with_capacity(array.len());
    for index in 0..array.len() {
        if sequence[index] != index as u64
            || account_id[index] <= 0
            || evidence_kind[index] > 13
            || has_symbol_id[index] > 1
            || has_quote_side[index] > 1
            || has_window[index] > 1
            || (has_symbol_id[index] == 0 && symbol_id[index] != 0)
            || (has_symbol_id[index] == 1 && symbol_id[index] <= 0)
            || (has_quote_side[index] == 0 && quote_side[index] != 0)
            || (has_quote_side[index] == 1 && quote_side[index] > 1)
            || (has_window[index] == 0 && (window_from[index] != 0 || window_to[index] != 0))
            || client_msg_id[index].trim().is_empty()
            || client_msg_id[index] != client_msg_id[index].trim()
            || payload_type[index] == 0
        {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
                Some(artifact.relative_path()),
                format!("evidence row {index} has invalid typed metadata"),
            ));
        }
        let requested_window = if has_window[index] == 1 {
            Some(
                EvidenceWindowV1::new(window_from[index], window_to[index]).map_err(|error| {
                    ingress_error(
                        BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
                        Some(artifact.relative_path()),
                        format!("evidence row {index} has invalid window: {error}"),
                    )
                })?,
            )
        } else {
            None
        };
        let payload: Value = serde_json::from_str(&payload_json[index]).map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::InvalidRawEnvelope,
                Some(artifact.relative_path()),
                format!("evidence row {index} payload is not JSON: {error}"),
            )
        })?;
        if !payload.is_object() {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::InvalidRawEnvelope,
                Some(artifact.relative_path()),
                format!("evidence row {index} payload is not an object"),
            ));
        }
        rows.push(EvidenceRowV2 {
            account_id: account_id[index],
            evidence_kind: evidence_kind[index],
            symbol_id: (has_symbol_id[index] == 1).then_some(symbol_id[index]),
            quote_side: (has_quote_side[index] == 1).then_some(quote_side[index]),
            requested_window,
            client_msg_id: client_msg_id[index].clone(),
            payload_type: payload_type[index],
            payload,
        });
    }
    Ok(rows)
}

fn decode_deal_rows(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
) -> Result<Vec<RawDealPageRowV2>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let chunk_sequence = u64_column(artifact, array, "chunk_sequence")?;
    let page_sequence = u64_column(artifact, array, "page_sequence_in_chunk")?;
    let account_id = i64_column(artifact, array, "account_id")?;
    let client_msg_id = utf8_column(artifact, array, "client_msg_id")?;
    let chunk_from = i64_column(artifact, array, "chunk_from_unix_ms_inclusive")?;
    let chunk_to = i64_column(artifact, array, "chunk_to_unix_ms_exclusive")?;
    let page_from = i64_column(artifact, array, "page_from_unix_ms_inclusive")?;
    let page_to = i64_column(artifact, array, "page_to_unix_ms_exclusive")?;
    let max_rows = u32_column(artifact, array, "max_rows")?;
    let has_events = u8_column(artifact, array, "has_events")?;
    let first_event = i64_column(artifact, array, "first_deal_execution_timestamp_ms")?;
    let last_event = i64_column(artifact, array, "last_deal_execution_timestamp_ms")?;
    let decoded_count = u64_column(artifact, array, "decoded_deal_count")?;
    let has_more = u8_column(artifact, array, "has_more")?;
    let raw_response_json = utf8_column(artifact, array, "raw_response_json")?;
    let mut rows = Vec::with_capacity(array.len());
    for index in 0..array.len() {
        let row = RawDealPageRowV2 {
            chunk_sequence: chunk_sequence[index],
            page_sequence_in_chunk: page_sequence[index],
            account_id: account_id[index],
            client_msg_id: client_msg_id[index].clone(),
            chunk_from: chunk_from[index],
            chunk_to: chunk_to[index],
            page_from: page_from[index],
            page_to: page_to[index],
            max_rows: max_rows[index],
            has_events: has_events[index],
            first_event: first_event[index],
            last_event: last_event[index],
            decoded_count: decoded_count[index],
            has_more: has_more[index],
            raw_response_json: raw_response_json[index].clone(),
        };
        let empty = row.decoded_count == 0;
        if row.chunk_sequence != 0
            || row.account_id <= 0
            || row.client_msg_id.trim().is_empty()
            || row.max_rows == 0
            || row.has_events > 1
            || row.has_more > 1
            || row.chunk_from >= row.chunk_to
            || row.page_from >= row.page_to
            || row.page_from != row.chunk_from
            || row.page_to > row.chunk_to
            || (empty && (row.has_events != 0 || row.first_event != 0 || row.last_event != 0))
            || (!empty
                && (row.has_events != 1
                    || row.first_event > row.last_event
                    || row.first_event < row.page_from
                    || row.last_event >= row.page_to))
        {
            return Err(ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
                Some(artifact.relative_path()),
                format!("raw DealList page row {index} has invalid metadata"),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn u8_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u8>, BrokerFinancialTruthSemanticIngressErrorV2> {
    Ok(primitive_field(artifact, array, name)?
        .as_slice::<u8>()
        .to_vec())
}

fn u32_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u32>, BrokerFinancialTruthSemanticIngressErrorV2> {
    Ok(primitive_field(artifact, array, name)?
        .as_slice::<u32>()
        .to_vec())
}

fn u64_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u64>, BrokerFinancialTruthSemanticIngressErrorV2> {
    Ok(primitive_field(artifact, array, name)?
        .as_slice::<u64>()
        .to_vec())
}

fn i64_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<i64>, BrokerFinancialTruthSemanticIngressErrorV2> {
    Ok(primitive_field(artifact, array, name)?
        .as_slice::<i64>()
        .to_vec())
}

fn f64_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<f64>, BrokerFinancialTruthSemanticIngressErrorV2> {
    Ok(primitive_field(artifact, array, name)?
        .as_slice::<f64>()
        .to_vec())
}

fn primitive_field(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<vortex_array::arrays::PrimitiveArray, BrokerFinancialTruthSemanticIngressErrorV2> {
    let field = array
        .to_struct()
        .unmasked_field_by_name(name)
        .map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                format!("cannot access primitive field {name}: {error}"),
            )
        })?
        .clone();
    Ok(field.to_primitive())
}

fn utf8_column(
    artifact: &ImmutableVortexArtifactV1,
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<String>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let field = array
        .to_struct()
        .unmasked_field_by_name(name)
        .map_err(|error| {
            ingress_error(
                BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                Some(artifact.relative_path()),
                format!("cannot access UTF-8 field {name}: {error}"),
            )
        })?
        .clone()
        .to_varbinview();
    (0..field.len())
        .map(|index| {
            let bytes = field.bytes_at(index);
            std::str::from_utf8(bytes.as_ref())
                .map(str::to_owned)
                .map_err(|error| {
                    ingress_error(
                        BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
                        Some(artifact.relative_path()),
                        format!("field {name} row {index} is not UTF-8: {error}"),
                    )
                })
        })
        .collect()
}

#[derive(Clone)]
struct ExactSymbolBindingV2 {
    symbol_name: String,
    base_asset_id: i64,
    quote_asset_id: i64,
}

fn binding_account_id(manifest: &BrokerFinancialTruthBundleManifestV2) -> i64 {
    let CanonicalDatasetScope::CTrader { account_id, .. } =
        manifest.binding().canonical_dataset_identity().scope()
    else {
        unreachable!("validated broker-truth binding is cTrader")
    };
    *account_id
}

fn exact_symbol_bindings(
    manifest: &BrokerFinancialTruthBundleManifestV2,
) -> Result<BTreeMap<i64, ExactSymbolBindingV2>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let mut symbols = BTreeMap::new();
    insert_exact_symbol(&mut symbols, manifest.primary_quotes().bid())?;
    insert_exact_symbol(&mut symbols, manifest.primary_quotes().ask())?;
    for route in manifest.conversion_routes() {
        for leg in route.legs() {
            insert_exact_symbol(&mut symbols, leg.quotes().bid())?;
            insert_exact_symbol(&mut symbols, leg.quotes().ask())?;
        }
    }
    Ok(symbols)
}

fn insert_exact_symbol(
    symbols: &mut BTreeMap<i64, ExactSymbolBindingV2>,
    quote: &ExactQuoteSideEvidenceV2,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let next = ExactSymbolBindingV2 {
        symbol_name: quote.symbol_name().to_owned(),
        base_asset_id: quote.base_asset_id(),
        quote_asset_id: quote.quote_asset_id(),
    };
    if let Some(previous) = symbols.insert(quote.symbol_id(), next.clone())
        && (previous.symbol_name != next.symbol_name
            || previous.base_asset_id != next.base_asset_id
            || previous.quote_asset_id != next.quote_asset_id)
    {
        return Err(ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
            None,
            format!(
                "symbol {} has conflicting exact quote bindings",
                quote.symbol_id()
            ),
        ));
    }
    Ok(())
}

fn validate_symbol_contracts(
    evidence: &ExactSymbolContractEvidenceV2,
    manifest: &BrokerFinancialTruthBundleManifestV2,
    tables: &BTreeMap<String, ArtifactTableV2>,
    exact_symbols: &BTreeMap<i64, ExactSymbolBindingV2>,
) -> Result<BTreeMap<i64, i32>, BrokerFinancialTruthSemanticIngressErrorV2> {
    let account_id = binding_account_id(manifest);
    let light = evidence_rows(tables, evidence.light_symbol_responses_raw())?;
    let full = evidence_rows(tables, evidence.full_symbol_responses_raw())?;
    let assets = evidence_rows(tables, evidence.account_asset_responses_raw())?;
    let trader = evidence_rows(tables, evidence.trader_account_responses_raw())?;
    let decoded = evidence_rows(tables, evidence.contracts_decoded())?;
    require_family_rows(light, 2, account_id, "raw light-symbol")?;
    require_family_rows(full, 4, account_id, "raw full-symbol")?;
    require_family_rows(assets, 6, account_id, "raw account-asset")?;
    require_family_rows(trader, 8, account_id, "raw trader-account")?;
    if light.len() != 1 || assets.len() != 1 || trader.len() != 1 {
        return Err(raw_decoded_error(
            "symbol contract ingress requires one exact light-symbol, asset, and trader response",
        ));
    }

    let mut raw_by_client = BTreeMap::new();
    for (rows, payload_type, label) in [
        (light, CTRADER_LIGHT_SYMBOL_RESPONSE, "light-symbol"),
        (full, CTRADER_FULL_SYMBOL_RESPONSE, "full-symbol"),
        (assets, CTRADER_ASSET_RESPONSE, "account-asset"),
        (trader, CTRADER_TRADER_RESPONSE, "trader-account"),
    ] {
        for row in rows {
            if row.payload_type != payload_type {
                return Err(raw_decoded_error(format!(
                    "{label} row has unexpected payload type {}",
                    row.payload_type
                )));
            }
            let payload = official_payload(row, label)?;
            if raw_by_client
                .insert(
                    (row.client_msg_id.clone(), row.payload_type),
                    (row, payload),
                )
                .is_some()
            {
                return Err(raw_decoded_error(format!(
                    "{label} response reuses one exact clientMsgId/payloadType"
                )));
            }
        }
    }

    let mut referenced = BTreeSet::new();
    let mut digits_by_symbol = BTreeMap::new();
    let mut decoded_kinds = BTreeSet::new();
    for row in decoded {
        if row.account_id != account_id || !matches!(row.evidence_kind, 3 | 5 | 7 | 9) {
            return Err(raw_decoded_error(
                "decoded symbol/money contract has wrong account or evidence kind",
            ));
        }
        let key = (row.client_msg_id.clone(), row.payload_type);
        let (raw, raw_payload) = raw_by_client.get(&key).ok_or_else(|| {
            raw_decoded_error(format!(
                "decoded symbol/money contract references absent raw response {:?}",
                row.client_msg_id
            ))
        })?;
        referenced.insert(key);
        if !decoded_kinds.insert((row.evidence_kind, row.symbol_id)) {
            return Err(raw_decoded_error(
                "decoded symbol/money contract duplicates one exact semantic row",
            ));
        }
        match row.evidence_kind {
            3 => validate_light_symbol_contract(row, raw, raw_payload, exact_symbols)?,
            5 => {
                let (symbol_id, digits) =
                    validate_full_symbol_contract(row, raw, raw_payload, exact_symbols)?;
                if digits_by_symbol.insert(symbol_id, digits).is_some() {
                    return Err(raw_decoded_error(format!(
                        "symbol {symbol_id} has duplicate full contract authority"
                    )));
                }
            }
            7 => validate_asset_contract(row, raw_payload, manifest)?,
            9 => validate_trader_contract(row, raw_payload, manifest)?,
            _ => unreachable!("kind guarded above"),
        }
    }
    if referenced.len() != raw_by_client.len()
        || exact_symbols
            .keys()
            .any(|symbol_id| !digits_by_symbol.contains_key(symbol_id))
    {
        return Err(raw_decoded_error(
            "raw symbol/account authority and decoded contracts are not one complete exact set",
        ));
    }
    Ok(digits_by_symbol)
}

fn validate_light_symbol_contract(
    decoded: &EvidenceRowV2,
    raw: &EvidenceRowV2,
    raw_payload: &Value,
    exact_symbols: &BTreeMap<i64, ExactSymbolBindingV2>,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    if decoded.payload_type != CTRADER_LIGHT_SYMBOL_RESPONSE || raw.evidence_kind != 2 {
        return Err(raw_decoded_error(
            "decoded light-symbol contract does not bind its raw response",
        ));
    }
    require_exact_fields(
        &decoded.payload,
        &["authority", "exactInstrument", "rawLightSymbol"],
        "decoded light-symbol contract",
    )?;
    if decoded.payload.get("authority").and_then(Value::as_str) != Some("ProtoOALightSymbol") {
        return Err(raw_decoded_error(
            "decoded light-symbol contract has changed authority label",
        ));
    }
    let symbol_id = decoded
        .symbol_id
        .ok_or_else(|| raw_decoded_error("decoded light-symbol contract omits symbol id"))?;
    let expected = exact_symbols
        .get(&symbol_id)
        .ok_or_else(|| raw_decoded_error("decoded light-symbol is not a required exact symbol"))?;
    let raw_symbols = required_array(raw_payload, "symbol", "raw light-symbol response")?;
    let raw_symbol = raw_symbols
        .iter()
        .find(|symbol| symbol.get("symbolId").and_then(Value::as_i64) == Some(symbol_id))
        .ok_or_else(|| raw_decoded_error("raw light-symbol response omits exact symbol"))?;
    if decoded.payload.get("rawLightSymbol") != Some(raw_symbol) {
        return Err(raw_decoded_error(
            "decoded light-symbol bytes differ from retained raw authority",
        ));
    }
    let instrument = decoded
        .payload
        .get("exactInstrument")
        .ok_or_else(|| raw_decoded_error("decoded light-symbol omits exact instrument"))?;
    require_exact_fields(
        instrument,
        &[
            "symbolId",
            "symbolName",
            "baseAssetId",
            "baseAssetName",
            "quoteAssetId",
            "quoteAssetName",
        ],
        "decoded exact instrument",
    )?;
    if instrument.get("symbolId").and_then(Value::as_i64) != Some(symbol_id)
        || instrument.get("symbolName").and_then(Value::as_str)
            != Some(expected.symbol_name.as_str())
        || instrument.get("baseAssetId").and_then(Value::as_i64) != Some(expected.base_asset_id)
        || instrument.get("quoteAssetId").and_then(Value::as_i64) != Some(expected.quote_asset_id)
        || raw_symbol.get("symbolName").and_then(Value::as_str)
            != Some(expected.symbol_name.as_str())
        || raw_symbol.get("baseAssetId").and_then(Value::as_i64) != Some(expected.base_asset_id)
        || raw_symbol.get("quoteAssetId").and_then(Value::as_i64) != Some(expected.quote_asset_id)
    {
        return Err(raw_decoded_error(
            "raw/decoded light-symbol binding differs from exact manifest symbol",
        ));
    }
    Ok(())
}

fn validate_full_symbol_contract(
    decoded: &EvidenceRowV2,
    raw: &EvidenceRowV2,
    raw_payload: &Value,
    exact_symbols: &BTreeMap<i64, ExactSymbolBindingV2>,
) -> Result<(i64, i32), BrokerFinancialTruthSemanticIngressErrorV2> {
    if decoded.payload_type != CTRADER_FULL_SYMBOL_RESPONSE
        || raw.evidence_kind != 4
        || decoded.symbol_id != raw.symbol_id
    {
        return Err(raw_decoded_error(
            "decoded full-symbol contract does not bind its raw response",
        ));
    }
    require_exact_fields(
        &decoded.payload,
        &["authority", "rawSymbol"],
        "decoded full-symbol contract",
    )?;
    if decoded.payload.get("authority").and_then(Value::as_str) != Some("ProtoOASymbol") {
        return Err(raw_decoded_error(
            "decoded full-symbol contract has changed authority label",
        ));
    }
    let symbol_id = decoded
        .symbol_id
        .ok_or_else(|| raw_decoded_error("decoded full-symbol contract omits symbol id"))?;
    if !exact_symbols.contains_key(&symbol_id) {
        return Err(raw_decoded_error(
            "decoded full-symbol is not required by exact quotes",
        ));
    }
    let raw_symbols = required_array(raw_payload, "symbol", "raw full-symbol response")?;
    if raw_symbols.len() != 1
        || raw_symbols[0].get("symbolId").and_then(Value::as_i64) != Some(symbol_id)
        || decoded.payload.get("rawSymbol") != Some(&raw_symbols[0])
    {
        return Err(raw_decoded_error(
            "decoded full-symbol bytes differ from retained exact raw response",
        ));
    }
    let digits = raw_symbols[0]
        .get("digits")
        .and_then(Value::as_i64)
        .and_then(|digits| i32::try_from(digits).ok())
        .filter(|digits| (0..=15).contains(digits))
        .ok_or_else(|| raw_decoded_error("raw ProtoOASymbol omits valid price digits"))?;
    Ok((symbol_id, digits))
}

fn validate_asset_contract(
    decoded: &EvidenceRowV2,
    raw_payload: &Value,
    manifest: &BrokerFinancialTruthBundleManifestV2,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    if decoded.payload_type != CTRADER_ASSET_RESPONSE || decoded.symbol_id.is_some() {
        return Err(raw_decoded_error(
            "decoded account-asset contract has wrong binding",
        ));
    }
    require_exact_fields(
        &decoded.payload,
        &["accountAssetId", "accountAssetName", "requiredRawAssets"],
        "decoded account-asset contract",
    )?;
    if decoded
        .payload
        .get("accountAssetId")
        .and_then(Value::as_i64)
        != Some(manifest.binding().account_asset_id())
        || decoded
            .payload
            .get("accountAssetName")
            .and_then(Value::as_str)
            != Some(manifest.binding().account_asset_name())
    {
        return Err(raw_decoded_error(
            "decoded account asset differs from exact bundle binding",
        ));
    }
    let raw_assets = required_array(raw_payload, "asset", "raw asset response")?;
    let decoded_assets = required_array(
        &decoded.payload,
        "requiredRawAssets",
        "decoded account-asset contract",
    )?;
    if decoded_assets
        .iter()
        .any(|asset| !raw_assets.contains(asset))
        || !decoded_assets.iter().any(|asset| {
            asset.get("assetId").and_then(Value::as_i64)
                == Some(manifest.binding().account_asset_id())
                && asset.get("name").and_then(Value::as_str)
                    == Some(manifest.binding().account_asset_name())
        })
    {
        return Err(raw_decoded_error(
            "decoded required assets differ from retained raw asset authority",
        ));
    }
    Ok(())
}

fn validate_trader_contract(
    decoded: &EvidenceRowV2,
    raw_payload: &Value,
    manifest: &BrokerFinancialTruthBundleManifestV2,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    if decoded.payload_type != CTRADER_TRADER_RESPONSE || decoded.symbol_id.is_some() {
        return Err(raw_decoded_error(
            "decoded trader contract has wrong binding",
        ));
    }
    require_exact_fields(
        &decoded.payload,
        &["accountAssetId", "rawTrader"],
        "decoded trader contract",
    )?;
    let raw_trader = raw_payload
        .get("trader")
        .filter(|value| value.is_object())
        .ok_or_else(|| raw_decoded_error("raw trader response omits trader object"))?;
    if decoded
        .payload
        .get("accountAssetId")
        .and_then(Value::as_i64)
        != Some(manifest.binding().account_asset_id())
        || decoded.payload.get("rawTrader") != Some(raw_trader)
        || raw_trader.get("depositAssetId").and_then(Value::as_i64)
            != Some(manifest.binding().account_asset_id())
        || raw_trader
            .get("moneyDigits")
            .and_then(Value::as_u64)
            .filter(|digits| *digits <= u64::from(MAX_MONEY_DIGITS_V2))
            .is_none()
        || raw_trader.get("balance").and_then(Value::as_i64).is_none()
    {
        return Err(raw_decoded_error(
            "decoded trader contract differs from raw deposit asset/money authority",
        ));
    }
    Ok(())
}

fn validate_synchronized_quotes(
    quotes: &SynchronizedBidAskEvidenceV2,
    account_id: i64,
    tables: &BTreeMap<String, ArtifactTableV2>,
    symbol_digits: &BTreeMap<i64, i32>,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    validate_quote_side(quotes.bid(), account_id, tables, symbol_digits)?;
    validate_quote_side(quotes.ask(), account_id, tables, symbol_digits)?;

    let raw = evidence_rows(tables, quotes.replay_rule().observations_raw())?;
    let decoded = evidence_rows(tables, quotes.replay_rule().rules_decoded())?;
    require_family_rows(raw, 0, account_id, "quote-session observation")?;
    require_family_rows(decoded, 1, account_id, "reviewed quote replay rule")?;
    let expected_symbol = quotes.bid().symbol_id();
    let expected_window = quotes.bid().requested_window();
    let sides = raw
        .iter()
        .map(|row| {
            if row.symbol_id != Some(expected_symbol)
                || row.requested_window != Some(expected_window)
                || row.quote_side.is_none()
            {
                return Err(raw_decoded_error(
                    "quote-session observation differs from exact symbol/window/side",
                ));
            }
            Ok(row.quote_side.expect("checked some"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if sides != BTreeSet::from([0, 1]) {
        return Err(raw_decoded_error(
            "quote-session observations do not contain exact Bid and Ask evidence",
        ));
    }
    let raw_links = raw
        .iter()
        .map(|row| ((row.client_msg_id.as_str(), row.payload_type), row))
        .collect::<BTreeMap<_, _>>();
    for row in decoded {
        if row.symbol_id != Some(expected_symbol)
            || row.requested_window != Some(expected_window)
            || row.quote_side.is_some()
            || !raw_links.contains_key(&(row.client_msg_id.as_str(), row.payload_type))
        {
            return Err(raw_decoded_error(
                "reviewed replay-rule row is not linked to exact raw observation",
            ));
        }
    }
    Ok(())
}

fn validate_quote_side(
    quote: &ExactQuoteSideEvidenceV2,
    account_id: i64,
    tables: &BTreeMap<String, ArtifactTableV2>,
    symbol_digits: &BTreeMap<i64, i32>,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let raw_rows = raw_tick_rows(tables, quote.raw_pages())?;
    let decoded_rows = decoded_tick_rows(tables, quote.decoded_ticks())?;
    let expected_side = quote_side_code(quote.side());
    let digits = *symbol_digits.get(&quote.symbol_id()).ok_or_else(|| {
        raw_decoded_error(format!(
            "quote symbol {} has no exact full-symbol digits",
            quote.symbol_id()
        ))
    })?;
    let pages = quote
        .request_chunks_newest_first()
        .iter()
        .flat_map(|chunk| chunk.pages_newest_first())
        .collect::<Vec<_>>();
    if raw_rows.len() != pages.len() {
        return Err(raw_decoded_error(
            "raw quote-page rows do not equal exact manifest pages",
        ));
    }
    let mut consumed_decoded = 0_usize;
    for (raw, page) in raw_rows.iter().zip(pages) {
        if raw.chunk_sequence != page.chunk_sequence()
            || raw.page_sequence_in_chunk != page.page_sequence_in_chunk()
            || raw.account_id != account_id
            || raw.symbol_id != quote.symbol_id()
            || raw.quote_side != expected_side
            || raw.client_msg_id != page.client_msg_id()
            || raw.chunk_from
                != quote
                    .request_chunks_newest_first()
                    .get(raw.chunk_sequence as usize)
                    .map(|chunk| chunk.requested_window().from_unix_ms_inclusive())
                    .unwrap_or(i64::MIN)
            || raw.chunk_to
                != quote
                    .request_chunks_newest_first()
                    .get(raw.chunk_sequence as usize)
                    .map(|chunk| chunk.requested_window().to_unix_ms_exclusive())
                    .unwrap_or(i64::MIN)
            || raw.page_from != page.requested_window().from_unix_ms_inclusive()
            || raw.page_to != page.requested_window().to_unix_ms_exclusive()
            || Some(raw.first_tick) != page.first_event_unix_ms()
            || Some(raw.last_tick) != page.last_event_unix_ms()
            || raw.decoded_count != page.event_count()
            || (raw.has_more == 1) != page.response_has_more()
        {
            return Err(raw_decoded_error(
                "raw quote-page table differs from exact request/page manifest",
            ));
        }
        let wire = parse_tick_envelope(raw)?;
        let reconstructed = decode_tick_wire(&wire.payload.tick_data, digits)?;
        if wire.client_msg_id != raw.client_msg_id
            || wire.payload_type != CTRADER_TICK_RESPONSE
            || wire.payload.account_id != account_id
            || wire.payload.has_more != (raw.has_more == 1)
            || reconstructed.len() as u64 != raw.decoded_count
            || reconstructed.first().map(|tick| tick.0) != Some(raw.first_tick)
            || reconstructed.last().map(|tick| tick.0) != Some(raw.last_tick)
        {
            return Err(raw_decoded_error(
                "raw cTrader tick envelope differs from retained page metadata",
            ));
        }
        let page_decoded = decoded_rows
            .iter()
            .filter(|row| {
                row.chunk_sequence == raw.chunk_sequence
                    && row.page_sequence_in_chunk == raw.page_sequence_in_chunk
            })
            .collect::<Vec<_>>();
        if page_decoded.len() != reconstructed.len() {
            return Err(raw_decoded_error(
                "decoded tick table count differs from re-decoded raw tick page",
            ));
        }
        for (row_index, (decoded, (timestamp, price))) in
            page_decoded.into_iter().zip(reconstructed).enumerate()
        {
            if decoded.row_sequence_in_page != row_index as u64
                || decoded.account_id != account_id
                || decoded.symbol_id != quote.symbol_id()
                || decoded.quote_side != expected_side
                || decoded.timestamp_ms != timestamp
                || decoded.price.to_bits() != price.to_bits()
            {
                return Err(raw_decoded_error(
                    "decoded tick row differs from exact raw cTrader delta replay",
                ));
            }
        }
        consumed_decoded += reconstructed_len(raw.decoded_count)?;
    }
    if consumed_decoded != decoded_rows.len()
        || decoded_rows
            .windows(2)
            .any(|pair| pair[1].timestamp_ms <= pair[0].timestamp_ms)
    {
        return Err(raw_decoded_error(
            "decoded quote rows contain unbound or non-ascending ticks",
        ));
    }
    Ok(())
}

fn reconstructed_len(count: u64) -> Result<usize, BrokerFinancialTruthSemanticIngressErrorV2> {
    usize::try_from(count)
        .map_err(|_| raw_decoded_error("decoded tick count does not fit process address space"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TickEnvelopeV2 {
    #[serde(rename = "clientMsgId")]
    client_msg_id: String,
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: TickPayloadV2,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TickPayloadV2 {
    #[serde(rename = "ctidTraderAccountId")]
    account_id: i64,
    #[serde(rename = "hasMore")]
    has_more: bool,
    #[serde(rename = "tickData")]
    tick_data: Vec<TickWireRowV2>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TickWireRowV2 {
    timestamp: i64,
    tick: i64,
}

fn parse_tick_envelope(
    raw: &RawTickPageRowV2,
) -> Result<TickEnvelopeV2, BrokerFinancialTruthSemanticIngressErrorV2> {
    serde_json::from_str(&raw.raw_response_json).map_err(|error| {
        ingress_error(
            BrokerFinancialTruthSemanticIngressErrorCodeV2::InvalidRawEnvelope,
            None,
            format!(
                "raw tick response {:?} is not the exact V2 envelope: {error}",
                raw.client_msg_id
            ),
        )
    })
}

fn decode_tick_wire(
    rows: &[TickWireRowV2],
    digits: i32,
) -> Result<Vec<(i64, f64)>, BrokerFinancialTruthSemanticIngressErrorV2> {
    if rows.is_empty() {
        return Err(raw_decoded_error("raw cTrader tick page contains no rows"));
    }
    let mut newest_first = Vec::with_capacity(rows.len());
    let mut previous_timestamp: Option<i64> = None;
    let mut previous_price: Option<i64> = None;
    for (index, row) in rows.iter().enumerate() {
        let timestamp = match previous_timestamp {
            Some(previous) => previous.checked_add(row.timestamp),
            None => Some(row.timestamp),
        }
        .ok_or_else(|| raw_decoded_error(format!("tick row {index} timestamp overflows")))?;
        let raw_price = match previous_price {
            Some(previous) => previous.checked_add(row.tick),
            None => Some(row.tick),
        }
        .ok_or_else(|| raw_decoded_error(format!("tick row {index} price overflows")))?;
        if timestamp < 0
            || raw_price <= 0
            || previous_timestamp.is_some_and(|previous| timestamp >= previous)
        {
            return Err(raw_decoded_error(format!(
                "tick row {index} violates exact newest-first delta contract"
            )));
        }
        let factor = 10_f64.powi(digits);
        let price = ((raw_price as f64 / 100_000.0) * factor).round() / factor;
        if !price.is_finite() || price <= 0.0 {
            return Err(raw_decoded_error(format!(
                "tick row {index} produces invalid exact price"
            )));
        }
        newest_first.push((timestamp, price));
        previous_timestamp = Some(timestamp);
        previous_price = Some(raw_price);
    }
    newest_first.reverse();
    Ok(newest_first)
}

fn validate_unrealized_pnl(
    manifest: &BrokerFinancialTruthBundleManifestV2,
    tables: &BTreeMap<String, ArtifactTableV2>,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let pair = manifest.broker_position_unrealized_pnl();
    let raw = evidence_rows(tables, pair.raw_envelopes())?;
    let decoded = evidence_rows(tables, pair.decoded_records())?;
    let account_id = binding_account_id(manifest);
    require_family_rows(raw, 10, account_id, "raw unrealized PnL")?;
    require_family_rows(decoded, 11, account_id, "decoded unrealized PnL")?;
    if raw.len() != 1 || decoded.len() != 1 {
        return Err(raw_decoded_error(
            "unrealized PnL ingress requires one raw and one decoded snapshot",
        ));
    }
    let raw = &raw[0];
    let decoded = &decoded[0];
    if raw.payload_type != CTRADER_UNREALIZED_PNL_RESPONSE
        || decoded.payload_type != CTRADER_UNREALIZED_PNL_RESPONSE
        || decoded.client_msg_id != raw.client_msg_id
        || raw.symbol_id.is_some()
        || decoded.symbol_id.is_some()
        || raw.requested_window.is_some()
        || decoded.requested_window.is_some()
    {
        return Err(raw_decoded_error(
            "unrealized PnL raw/decoded identity binding differs",
        ));
    }
    let raw_payload = official_payload(raw, "unrealized PnL")?;
    require_exact_fields(
        &decoded.payload,
        &["accountId", "moneyDigits", "positions"],
        "decoded unrealized PnL",
    )?;
    let raw_digits = raw_payload
        .get("moneyDigits")
        .and_then(Value::as_u64)
        .and_then(|digits| u32::try_from(digits).ok())
        .filter(|digits| *digits <= MAX_MONEY_DIGITS_V2)
        .ok_or_else(|| raw_decoded_error("raw unrealized PnL omits valid moneyDigits"))?;
    if decoded.payload.get("accountId").and_then(Value::as_i64) != Some(account_id)
        || decoded.payload.get("moneyDigits").and_then(Value::as_u64) != Some(u64::from(raw_digits))
    {
        return Err(raw_decoded_error(
            "decoded unrealized PnL account/moneyDigits differ from raw authority",
        ));
    }
    let raw_positions = raw_payload
        .get("positionUnrealizedPnL")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| raw_decoded_error("raw unrealized PnL positions are not an array"))
        })
        .transpose()?
        .map_or(&[][..], Vec::as_slice);
    let decoded_positions = required_array(&decoded.payload, "positions", "decoded PnL")?;
    if raw_positions.len() != decoded_positions.len() {
        return Err(raw_decoded_error(
            "raw/decoded unrealized PnL position counts differ",
        ));
    }
    let mut raw_by_position = BTreeMap::new();
    for position in raw_positions {
        require_exact_fields(
            position,
            &["positionId", "grossUnrealizedPnL", "netUnrealizedPnL"],
            "raw unrealized PnL position",
        )?;
        let position_id = position
            .get("positionId")
            .and_then(Value::as_i64)
            .filter(|id| *id > 0)
            .ok_or_else(|| raw_decoded_error("raw PnL position omits positive positionId"))?;
        let gross = position
            .get("grossUnrealizedPnL")
            .and_then(Value::as_i64)
            .ok_or_else(|| raw_decoded_error("raw PnL position omits gross integer"))?;
        let net = position
            .get("netUnrealizedPnL")
            .and_then(Value::as_i64)
            .ok_or_else(|| raw_decoded_error("raw PnL position omits net integer"))?;
        if raw_by_position.insert(position_id, (gross, net)).is_some() {
            return Err(raw_decoded_error(
                "raw unrealized PnL duplicates one positionId",
            ));
        }
    }
    let factor = 10_f64.powi(raw_digits as i32);
    let mut decoded_ids = BTreeSet::new();
    for position in decoded_positions {
        require_exact_fields(
            position,
            &["positionId", "grossUnrealizedPnL", "netUnrealizedPnL"],
            "decoded unrealized PnL position",
        )?;
        let position_id = position
            .get("positionId")
            .and_then(Value::as_i64)
            .ok_or_else(|| raw_decoded_error("decoded PnL position omits positionId"))?;
        let (raw_gross, raw_net) = raw_by_position
            .get(&position_id)
            .ok_or_else(|| raw_decoded_error("decoded PnL position has no raw authority"))?;
        let gross = position
            .get("grossUnrealizedPnL")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| raw_decoded_error("decoded PnL gross value is invalid"))?;
        let net = position
            .get("netUnrealizedPnL")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| raw_decoded_error("decoded PnL net value is invalid"))?;
        if !decoded_ids.insert(position_id)
            || gross.to_bits() != (*raw_gross as f64 / factor).to_bits()
            || net.to_bits() != (*raw_net as f64 / factor).to_bits()
        {
            return Err(raw_decoded_error(
                "decoded unrealized PnL differs from raw integer/moneyDigits replay",
            ));
        }
    }
    Ok(())
}

fn validate_close_deal_reconciliation(
    evidence: &ExactDealReconciliationEvidenceV2,
    tables: &BTreeMap<String, ArtifactTableV2>,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let reconcile = evidence_rows(tables, evidence.reconcile_responses_raw())?;
    let deal_rows = raw_deal_rows(tables, evidence.deal_pages_raw())?;
    let decoded = evidence_rows(tables, evidence.reconciliation_decoded())?;
    if reconcile.len() != 1 || reconcile[0].evidence_kind != 12 {
        return Err(raw_decoded_error(
            "close/deal evidence requires one exact raw reconcile response",
        ));
    }
    let account_id = reconcile[0].account_id;
    require_family_rows(reconcile, 12, account_id, "raw reconcile")?;
    require_family_rows(decoded, 13, account_id, "decoded close/deal")?;
    if reconcile[0].payload_type != CTRADER_RECONCILE_RESPONSE
        || reconcile[0].requested_window != Some(evidence.requested_window())
    {
        return Err(raw_decoded_error(
            "raw reconcile response differs from exact DealList window/account",
        ));
    }
    let reconcile_payload = official_payload(&reconcile[0], "reconcile")?;
    for field in ["position", "order"] {
        if reconcile_payload
            .get(field)
            .is_some_and(|value| !value.is_array())
        {
            return Err(raw_decoded_error(format!(
                "raw reconcile {field} is not an array"
            )));
        }
    }
    let manifest_pages = evidence.deal_request_chunk().pages_newest_first();
    if deal_rows.len() != manifest_pages.len() || decoded.len() != deal_rows.len() {
        return Err(raw_decoded_error(
            "DealList raw/decoded rows differ from exact paged manifest",
        ));
    }
    for ((raw, page), decoded) in deal_rows.iter().zip(manifest_pages).zip(decoded) {
        if raw.chunk_sequence != page.chunk_sequence()
            || raw.page_sequence_in_chunk != page.page_sequence_in_chunk()
            || raw.account_id != account_id
            || raw.client_msg_id != page.client_msg_id()
            || raw.chunk_from
                != evidence
                    .deal_request_chunk()
                    .requested_window()
                    .from_unix_ms_inclusive()
            || raw.chunk_to
                != evidence
                    .deal_request_chunk()
                    .requested_window()
                    .to_unix_ms_exclusive()
            || raw.page_from != page.requested_window().from_unix_ms_inclusive()
            || raw.page_to != page.requested_window().to_unix_ms_exclusive()
            || Some(raw.max_rows) != page.max_rows()
            || raw.decoded_count != page.event_count()
            || (raw.has_more == 1) != page.response_has_more()
            || raw.has_events != u8::from(page.event_count() > 0)
            || (page.event_count() > 0
                && (Some(raw.first_event) != page.first_event_unix_ms()
                    || Some(raw.last_event) != page.last_event_unix_ms()))
        {
            return Err(raw_decoded_error(
                "raw DealList table differs from exact request/page manifest",
            ));
        }
        let envelope = parse_official_envelope_json(
            &raw.raw_response_json,
            &raw.client_msg_id,
            CTRADER_DEAL_RESPONSE,
            raw.account_id,
            "DealList",
        )?;
        let has_more = envelope
            .get("hasMore")
            .and_then(Value::as_bool)
            .ok_or_else(|| raw_envelope_error("DealList payload omits boolean hasMore"))?;
        let raw_deals = match envelope.get("deal") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| raw_envelope_error("DealList deal field is not an array"))?
                .as_slice(),
            None if !has_more => &[],
            None => {
                return Err(raw_envelope_error(
                    "DealList hasMore payload omits deal rows",
                ));
            }
        };
        let timestamps = raw_deals
            .iter()
            .map(|deal| {
                deal.get("executionTimestamp")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| {
                        raw_envelope_error("DealList row omits integer executionTimestamp")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if has_more != (raw.has_more == 1)
            || timestamps.len() as u64 != raw.decoded_count
            || timestamps.windows(2).any(|pair| pair[1] <= pair[0])
            || timestamps
                .iter()
                .any(|timestamp| *timestamp < raw.page_from || *timestamp >= raw.page_to)
            || timestamps.first().copied().unwrap_or(0) != raw.first_event
            || timestamps.last().copied().unwrap_or(0) != raw.last_event
        {
            return Err(raw_decoded_error(
                "raw DealList envelope differs from retained page metadata",
            ));
        }
        if decoded.evidence_kind != 13
            || decoded.account_id != account_id
            || decoded.client_msg_id != raw.client_msg_id
            || decoded.payload_type != CTRADER_DEAL_RESPONSE
            || decoded.requested_window != Some(evidence.requested_window())
        {
            return Err(raw_decoded_error(
                "decoded close/deal row is not bound to exact raw DealList page",
            ));
        }
        require_exact_fields(
            &decoded.payload,
            &[
                "dealPageRequest",
                "rawDealPayload",
                "rawReconcilePayload",
                "returnProtectionOrders",
            ],
            "decoded close/deal row",
        )?;
        let request = decoded
            .payload
            .get("dealPageRequest")
            .ok_or_else(|| raw_decoded_error("decoded close/deal row omits request"))?;
        require_exact_fields(
            request,
            &["fromTimestamp", "toTimestamp", "maxRows"],
            "decoded DealList request",
        )?;
        if request.get("fromTimestamp").and_then(Value::as_i64) != Some(raw.page_from)
            || request.get("toTimestamp").and_then(Value::as_i64) != Some(raw.page_to)
            || request.get("maxRows").and_then(Value::as_u64) != Some(u64::from(raw.max_rows))
            || decoded.payload.get("rawDealPayload") != Some(&envelope)
            || decoded.payload.get("rawReconcilePayload") != Some(reconcile_payload)
            || decoded
                .payload
                .get("returnProtectionOrders")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(raw_decoded_error(
                "decoded close/deal row differs from raw page/reconcile/request evidence",
            ));
        }
    }
    Ok(())
}

fn official_payload<'a>(
    row: &'a EvidenceRowV2,
    label: &str,
) -> Result<&'a Value, BrokerFinancialTruthSemanticIngressErrorV2> {
    require_exact_fields(
        &row.payload,
        &["clientMsgId", "payloadType", "payload"],
        &format!("raw {label} envelope"),
    )?;
    if row.payload.get("clientMsgId").and_then(Value::as_str) != Some(row.client_msg_id.as_str())
        || row.payload.get("payloadType").and_then(Value::as_u64)
            != Some(u64::from(row.payload_type))
    {
        return Err(raw_envelope_error(format!(
            "raw {label} envelope identity differs from retained row"
        )));
    }
    let payload = row
        .payload
        .get("payload")
        .filter(|value| value.is_object())
        .ok_or_else(|| raw_envelope_error(format!("raw {label} payload is not an object")))?;
    if payload.get("ctidTraderAccountId").and_then(Value::as_i64) != Some(row.account_id) {
        return Err(raw_envelope_error(format!(
            "raw {label} account differs from retained row"
        )));
    }
    Ok(payload)
}

fn parse_official_envelope_json(
    json: &str,
    expected_client_msg_id: &str,
    expected_payload_type: u32,
    expected_account_id: i64,
    label: &str,
) -> Result<Value, BrokerFinancialTruthSemanticIngressErrorV2> {
    let envelope: Value = serde_json::from_str(json).map_err(|error| {
        raw_envelope_error(format!("raw {label} response is not JSON: {error}"))
    })?;
    require_exact_fields(
        &envelope,
        &["clientMsgId", "payloadType", "payload"],
        &format!("raw {label} envelope"),
    )?;
    if envelope.get("clientMsgId").and_then(Value::as_str) != Some(expected_client_msg_id)
        || envelope.get("payloadType").and_then(Value::as_u64)
            != Some(u64::from(expected_payload_type))
    {
        return Err(raw_envelope_error(format!(
            "raw {label} response identity mismatch"
        )));
    }
    let payload = envelope
        .get("payload")
        .filter(|value| value.is_object())
        .cloned()
        .ok_or_else(|| raw_envelope_error(format!("raw {label} payload is not an object")))?;
    if payload.get("ctidTraderAccountId").and_then(Value::as_i64) != Some(expected_account_id) {
        return Err(raw_envelope_error(format!(
            "raw {label} response account mismatch"
        )));
    }
    Ok(payload)
}

fn require_family_rows(
    rows: &[EvidenceRowV2],
    expected_kind: u8,
    account_id: i64,
    label: &str,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|row| row.evidence_kind != expected_kind || row.account_id != account_id)
    {
        return Err(raw_decoded_error(format!(
            "{label} table has missing or mismatched kind/account rows"
        )));
    }
    Ok(())
}

fn require_exact_fields(
    value: &Value,
    expected: &[&str],
    label: &str,
) -> Result<(), BrokerFinancialTruthSemanticIngressErrorV2> {
    let object = value
        .as_object()
        .ok_or_else(|| raw_decoded_error(format!("{label} is not an object")))?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(raw_decoded_error(format!(
            "{label} fields differ: expected={expected:?} actual={actual:?}"
        )));
    }
    Ok(())
}

fn required_array<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a Vec<Value>, BrokerFinancialTruthSemanticIngressErrorV2> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| raw_decoded_error(format!("{label} omits array field {field}")))
}

fn evidence_rows<'a>(
    tables: &'a BTreeMap<String, ArtifactTableV2>,
    artifact: &ImmutableVortexArtifactV1,
) -> Result<&'a [EvidenceRowV2], BrokerFinancialTruthSemanticIngressErrorV2> {
    match tables.get(artifact.relative_path()) {
        Some(ArtifactTableV2::Evidence(rows)) => Ok(rows),
        _ => Err(table_kind_error(artifact, "generic evidence")),
    }
}

fn raw_tick_rows<'a>(
    tables: &'a BTreeMap<String, ArtifactTableV2>,
    artifact: &ImmutableVortexArtifactV1,
) -> Result<&'a [RawTickPageRowV2], BrokerFinancialTruthSemanticIngressErrorV2> {
    match tables.get(artifact.relative_path()) {
        Some(ArtifactTableV2::RawTickPages(rows)) => Ok(rows),
        _ => Err(table_kind_error(artifact, "raw tick pages")),
    }
}

fn decoded_tick_rows<'a>(
    tables: &'a BTreeMap<String, ArtifactTableV2>,
    artifact: &ImmutableVortexArtifactV1,
) -> Result<&'a [DecodedTickRowV2], BrokerFinancialTruthSemanticIngressErrorV2> {
    match tables.get(artifact.relative_path()) {
        Some(ArtifactTableV2::DecodedTicks(rows)) => Ok(rows),
        _ => Err(table_kind_error(artifact, "decoded ticks")),
    }
}

fn raw_deal_rows<'a>(
    tables: &'a BTreeMap<String, ArtifactTableV2>,
    artifact: &ImmutableVortexArtifactV1,
) -> Result<&'a [RawDealPageRowV2], BrokerFinancialTruthSemanticIngressErrorV2> {
    match tables.get(artifact.relative_path()) {
        Some(ArtifactTableV2::RawDealPages(rows)) => Ok(rows),
        _ => Err(table_kind_error(artifact, "raw DealList pages")),
    }
}

fn table_kind_error(
    artifact: &ImmutableVortexArtifactV1,
    expected: &str,
) -> BrokerFinancialTruthSemanticIngressErrorV2 {
    ingress_error(
        BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
        Some(artifact.relative_path()),
        format!("artifact did not decode as expected {expected} table"),
    )
}

const fn quote_side_code(side: QuoteSideV1) -> u8 {
    match side {
        QuoteSideV1::Bid => 0,
        QuoteSideV1::Ask => 1,
    }
}

fn raw_envelope_error(detail: impl Into<String>) -> BrokerFinancialTruthSemanticIngressErrorV2 {
    ingress_error(
        BrokerFinancialTruthSemanticIngressErrorCodeV2::InvalidRawEnvelope,
        None,
        detail,
    )
}

fn raw_decoded_error(detail: impl Into<String>) -> BrokerFinancialTruthSemanticIngressErrorV2 {
    ingress_error(
        BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch,
        None,
        detail,
    )
}

fn ingress_error(
    code: BrokerFinancialTruthSemanticIngressErrorCodeV2,
    artifact: Option<&str>,
    detail: impl Into<String>,
) -> BrokerFinancialTruthSemanticIngressErrorV2 {
    BrokerFinancialTruthSemanticIngressErrorV2 {
        code,
        artifact: artifact.map(str::to_owned),
        detail: detail.into(),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}
