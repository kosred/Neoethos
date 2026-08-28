//! Evidence-only ingress for frozen reviewed cTrader quote synchronizations.
//!
//! This boundary verifies exact immutable file identities, decodes only the
//! fixed V2 Vortex layout, and constructs the existing concrete capture types.
//! It does not connect to a broker, publish a bundle, or authorize evaluation.

use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use neoethos_broker_truth::{
    BrokerFinancialTruthVortexSchemaV1, BrokerTruthReviewedSynchronizationBindingV1,
    EvidenceWindowV1, ImmutableVortexArtifactV1, QuoteSideV1,
};
use serde_json::Value;
use vortex_array::dtype::{DType, Nullability, PType, StructFields};
use vortex_array::{ArrayRef, ToCanonical};

use crate::broker_truth_capture::{
    BrokerEvidenceRowKindV2, CapturedBrokerEvidencePairV2, CapturedBrokerEvidenceRowV2,
    CapturedQuoteSynchronizationV2, ExactQuoteInstrumentV2,
};
use crate::broker_truth_ctrader::ReviewedCTraderQuoteSynchronizationV2;

const MAX_REVIEWED_SYNCHRONIZATION_FILE_BYTES_V2: u64 = 64 * 1024 * 1024;
const EXPECTED_OBSERVATION_ROWS_V2: usize = 2;
const EXPECTED_REPLAY_RULE_ROWS_V2: usize = 1;
const CTRADER_TICK_RESPONSE_PAYLOAD_TYPE_V2: u32 = 2_146;
const OBSERVATIONS_LOGICAL_NAME_V2: &str = "quote-session-observations.vortex";
const REPLAY_RULES_LOGICAL_NAME_V2: &str = "reviewed-quote-replay-rules.vortex";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2 {
    InvalidSource,
    UnsafePath,
    ArtifactDigestMismatch,
    VortexReadFailed,
    VortexSchemaMismatch,
    SynchronizationMismatch,
    RawDecodedMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewedCTraderQuoteSynchronizationIngressErrorV2 {
    code: ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2,
    detail: &'static str,
}

impl ReviewedCTraderQuoteSynchronizationIngressErrorV2 {
    pub const fn code(&self) -> ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for ReviewedCTraderQuoteSynchronizationIngressErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "reviewed cTrader quote synchronization ingress: {}",
            self.detail
        )
    }
}

impl Error for ReviewedCTraderQuoteSynchronizationIngressErrorV2 {}

fn ingress_error(
    code: ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2,
    detail: &'static str,
) -> ReviewedCTraderQuoteSynchronizationIngressErrorV2 {
    ReviewedCTraderQuoteSynchronizationIngressErrorV2 { code, detail }
}

#[derive(Clone, Debug)]
pub struct ReviewedCTraderQuoteSynchronizationSourceV2 {
    binding: BrokerTruthReviewedSynchronizationBindingV1,
    instrument: ExactQuoteInstrumentV2,
    observations_path: PathBuf,
    replay_rules_path: PathBuf,
}

impl ReviewedCTraderQuoteSynchronizationSourceV2 {
    pub fn new(
        binding: BrokerTruthReviewedSynchronizationBindingV1,
        instrument: ExactQuoteInstrumentV2,
        observations_path: impl Into<PathBuf>,
        replay_rules_path: impl Into<PathBuf>,
    ) -> Result<Self, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
        let observations_path = observations_path.into();
        let replay_rules_path = replay_rules_path.into();
        validate_explicit_evidence_file(&observations_path)?;
        validate_explicit_evidence_file(&replay_rules_path)?;
        if observations_path == replay_rules_path {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::InvalidSource,
                "reviewed synchronization requires two distinct evidence files",
            ));
        }
        binding.review_identity().validate_exact().map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization identity is invalid",
            )
        })?;
        if binding.symbol_id() != instrument.symbol_id() {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization instrument differs from its binding",
            ));
        }
        Ok(Self {
            binding,
            instrument,
            observations_path,
            replay_rules_path,
        })
    }
}

#[derive(Debug)]
pub struct LoadedReviewedCTraderQuoteSynchronizationV2 {
    ordinal: u32,
    account_id: i64,
    symbol_id: i64,
    window: EvidenceWindowV1,
    review_identity_sha256: String,
    raw_observation_count: usize,
    decoded_replay_rule_count: usize,
    synchronization: ReviewedCTraderQuoteSynchronizationV2,
}

impl LoadedReviewedCTraderQuoteSynchronizationV2 {
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

    pub fn review_identity_sha256(&self) -> &str {
        &self.review_identity_sha256
    }

    pub const fn raw_observation_count(&self) -> usize {
        self.raw_observation_count
    }

    pub const fn decoded_replay_rule_count(&self) -> usize {
        self.decoded_replay_rule_count
    }

    pub fn into_synchronization(self) -> ReviewedCTraderQuoteSynchronizationV2 {
        self.synchronization
    }
}

/// Decode explicit frozen reviewed-synchronization files in caller order.
pub fn load_reviewed_ctrader_quote_synchronizations_v2(
    sources: Vec<ReviewedCTraderQuoteSynchronizationSourceV2>,
) -> Result<
    Vec<LoadedReviewedCTraderQuoteSynchronizationV2>,
    ReviewedCTraderQuoteSynchronizationIngressErrorV2,
> {
    if sources.is_empty() {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::InvalidSource,
            "at least one reviewed synchronization source is required",
        ));
    }
    let mut used_paths = HashSet::with_capacity(sources.len() * 2);
    let mut loaded = Vec::with_capacity(sources.len());
    for (index, source) in sources.into_iter().enumerate() {
        let expected_ordinal = u32::try_from(index).map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization ordinal exceeds u32",
            )
        })?;
        if source.binding.ordinal() != expected_ordinal {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization sources are not contiguous in caller order",
            ));
        }
        if !used_paths.insert(source.observations_path.clone())
            || !used_paths.insert(source.replay_rules_path.clone())
        {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::InvalidSource,
                "reviewed synchronization evidence path was reused",
            ));
        }
        loaded.push(load_one_synchronization(source)?);
    }
    Ok(loaded)
}

fn load_one_synchronization(
    source: ReviewedCTraderQuoteSynchronizationSourceV2,
) -> Result<
    LoadedReviewedCTraderQuoteSynchronizationV2,
    ReviewedCTraderQuoteSynchronizationIngressErrorV2,
> {
    let observations = read_exact_evidence_rows(
        &source.observations_path,
        OBSERVATIONS_LOGICAL_NAME_V2,
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
        EXPECTED_OBSERVATION_ROWS_V2,
        source.binding.review_identity().broker_observation_sha256(),
    )?;
    let replay_rules = read_exact_evidence_rows(
        &source.replay_rules_path,
        REPLAY_RULES_LOGICAL_NAME_V2,
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
        EXPECTED_REPLAY_RULE_ROWS_V2,
        source.binding.reviewed_rules_sha256(),
    )?;

    validate_observation_rows(&source, &observations)?;
    validate_replay_rule_rows(&source, &observations, &replay_rules)?;

    let raw_envelopes = observations
        .into_iter()
        .map(|row| {
            let quote_side = match row.quote_side {
                Some(side) => Some(side),
                None => None,
            };
            CapturedBrokerEvidenceRowV2::new(
                row.sequence,
                row.account_id,
                row.symbol_id,
                quote_side,
                BrokerEvidenceRowKindV2::QuoteSessionObservation,
                row.requested_window,
                row.client_msg_id,
                row.payload_type,
                row.payload_json,
            )
        })
        .collect();
    let decoded_records = replay_rules
        .into_iter()
        .map(|row| {
            CapturedBrokerEvidenceRowV2::new(
                row.sequence,
                row.account_id,
                row.symbol_id,
                None,
                BrokerEvidenceRowKindV2::QuoteReplayRule,
                row.requested_window,
                row.client_msg_id,
                row.payload_type,
                row.payload_json,
            )
        })
        .collect();
    let ordinal = source.binding.ordinal();
    let account_id = source.binding.account_id();
    let symbol_id = source.binding.symbol_id();
    let window = source.binding.window();
    let review_identity = source.binding.review_identity().clone();
    let review_identity_sha256 = review_identity.identity_sha256().to_owned();
    let evidence = CapturedBrokerEvidencePairV2::new(raw_envelopes, decoded_records);
    let capture = CapturedQuoteSynchronizationV2::new(review_identity, evidence);
    let synchronization =
        ReviewedCTraderQuoteSynchronizationV2::new(account_id, source.instrument, window, capture)
            .map_err(|_| {
                ingress_error(
                    ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                    "concrete reviewed synchronization rejected decoded evidence",
                )
            })?;
    Ok(LoadedReviewedCTraderQuoteSynchronizationV2 {
        ordinal,
        account_id,
        symbol_id,
        window,
        review_identity_sha256,
        raw_observation_count: EXPECTED_OBSERVATION_ROWS_V2,
        decoded_replay_rule_count: EXPECTED_REPLAY_RULE_ROWS_V2,
        synchronization,
    })
}

fn validate_observation_rows(
    source: &ReviewedCTraderQuoteSynchronizationSourceV2,
    observations: &[DecodedEvidenceRowV2],
) -> Result<(), ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    if observations.len() != EXPECTED_OBSERVATION_ROWS_V2 {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
            "quote observation evidence must contain exactly two rows",
        ));
    }
    let expected_sides = [QuoteSideV1::Bid, QuoteSideV1::Ask];
    let mut client_msg_ids = HashSet::with_capacity(EXPECTED_OBSERVATION_ROWS_V2);
    for (index, row) in observations.iter().enumerate() {
        if row.sequence != index as u64
            || row.account_id != source.binding.account_id()
            || row.kind != 0
            || row.symbol_id != Some(source.binding.symbol_id())
            || row.quote_side != Some(expected_sides[index])
            || row.requested_window != Some(source.binding.window())
            || row.payload_type != CTRADER_TICK_RESPONSE_PAYLOAD_TYPE_V2
            || !client_msg_ids.insert(row.client_msg_id.as_str())
        {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "quote observation rows differ from exact reviewed synchronization",
            ));
        }
        if row.payload.get("clientMsgId").and_then(Value::as_str)
            != Some(row.client_msg_id.as_str())
            || row.payload.get("payloadType").and_then(Value::as_u64)
                != Some(u64::from(CTRADER_TICK_RESPONSE_PAYLOAD_TYPE_V2))
            || row
                .payload
                .pointer("/payload/ctidTraderAccountId")
                .and_then(Value::as_i64)
                != Some(source.binding.account_id())
        {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
                "quote observation row is detached from its retained raw envelope",
            ));
        }
    }
    Ok(())
}

fn validate_replay_rule_rows(
    source: &ReviewedCTraderQuoteSynchronizationSourceV2,
    observations: &[DecodedEvidenceRowV2],
    replay_rules: &[DecodedEvidenceRowV2],
) -> Result<(), ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    if replay_rules.len() != EXPECTED_REPLAY_RULE_ROWS_V2 {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
            "reviewed replay-rule evidence must contain exactly one row",
        ));
    }
    let row = &replay_rules[0];
    if row.sequence != 0
        || row.account_id != source.binding.account_id()
        || row.kind != 1
        || row.symbol_id != Some(source.binding.symbol_id())
        || row.quote_side.is_some()
        || row.requested_window != Some(source.binding.window())
        || row.payload_type != CTRADER_TICK_RESPONSE_PAYLOAD_TYPE_V2
    {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
            "reviewed replay-rule row differs from exact synchronization binding",
        ));
    }
    if !observations
        .iter()
        .any(|raw| raw.client_msg_id == row.client_msg_id)
    {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
            "reviewed replay-rule row is not linked to retained raw evidence",
        ));
    }
    let canonical = serde_json::to_string(&row.payload).map_err(|_| {
        ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
            "reviewed replay-rule JSON cannot be canonicalized",
        )
    })?;
    if canonical != row.payload_json {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
            "reviewed replay-rule JSON is not canonical",
        ));
    }
    Ok(())
}

fn read_exact_evidence_rows(
    path: &Path,
    logical_name: &str,
    schema: BrokerFinancialTruthVortexSchemaV1,
    expected_rows: usize,
    expected_sha256: &str,
) -> Result<Vec<DecodedEvidenceRowV2>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    validate_explicit_evidence_file(path)?;
    let digest_before = exact_file_sha256(path, logical_name, schema, expected_rows)?;
    if digest_before != expected_sha256 {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::ArtifactDigestMismatch,
            "reviewed synchronization evidence digest differs from its binding",
        ));
    }
    let array_result = neoethos_data::core::vortex_io::read_vortex_array(path).map_err(|_| {
        ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexReadFailed,
            "reviewed synchronization evidence is not readable Vortex",
        )
    });
    validate_explicit_evidence_file(path)?;
    let digest_after = exact_file_sha256(path, logical_name, schema, expected_rows)?;
    if digest_after != digest_before {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::ArtifactDigestMismatch,
            "reviewed synchronization evidence changed while being read",
        ));
    }
    let array = array_result?;
    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        validate_exact_evidence_dtype(&array)?;
        if array.len() != expected_rows {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization Vortex row count is not exact",
            ));
        }
        decode_evidence_rows(&array)
    }));
    match decoded {
        Ok(result) => result,
        Err(_) => Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexReadFailed,
            "reviewed synchronization Vortex decoding panicked",
        )),
    }
}

fn exact_file_sha256(
    path: &Path,
    logical_name: &str,
    schema: BrokerFinancialTruthVortexSchemaV1,
    expected_rows: usize,
) -> Result<String, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    ImmutableVortexArtifactV1::from_file(logical_name, schema, expected_rows as u64, path)
        .map(|artifact| artifact.sha256().to_owned())
        .map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::ArtifactDigestMismatch,
                "reviewed synchronization evidence cannot be hashed exactly",
            )
        })
}

#[derive(Clone)]
struct DecodedEvidenceRowV2 {
    sequence: u64,
    account_id: i64,
    kind: u8,
    symbol_id: Option<i64>,
    quote_side: Option<QuoteSideV1>,
    requested_window: Option<EvidenceWindowV1>,
    client_msg_id: String,
    payload_type: u32,
    payload_json: String,
    payload: Value,
}

#[derive(Clone, Copy)]
enum EvidenceColumnTypeV2 {
    U8,
    U32,
    U64,
    I64,
    Utf8,
}

impl EvidenceColumnTypeV2 {
    const fn dtype(self) -> DType {
        match self {
            Self::U8 => DType::Primitive(PType::U8, Nullability::NonNullable),
            Self::U32 => DType::Primitive(PType::U32, Nullability::NonNullable),
            Self::U64 => DType::Primitive(PType::U64, Nullability::NonNullable),
            Self::I64 => DType::Primitive(PType::I64, Nullability::NonNullable),
            Self::Utf8 => DType::Utf8(Nullability::NonNullable),
        }
    }
}

const EVIDENCE_FIELDS_V2: [(&str, EvidenceColumnTypeV2); 13] = [
    ("sequence", EvidenceColumnTypeV2::U64),
    ("account_id", EvidenceColumnTypeV2::I64),
    ("evidence_kind", EvidenceColumnTypeV2::U8),
    ("has_symbol_id", EvidenceColumnTypeV2::U8),
    ("symbol_id", EvidenceColumnTypeV2::I64),
    ("has_quote_side", EvidenceColumnTypeV2::U8),
    ("quote_side", EvidenceColumnTypeV2::U8),
    ("has_requested_window", EvidenceColumnTypeV2::U8),
    (
        "requested_from_unix_ms_inclusive",
        EvidenceColumnTypeV2::I64,
    ),
    ("requested_to_unix_ms_exclusive", EvidenceColumnTypeV2::I64),
    ("client_msg_id", EvidenceColumnTypeV2::Utf8),
    ("payload_type", EvidenceColumnTypeV2::U32),
    ("payload_json", EvidenceColumnTypeV2::Utf8),
];

fn validate_exact_evidence_dtype(
    array: &ArrayRef,
) -> Result<(), ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    let expected = DType::Struct(
        StructFields::from_iter(
            EVIDENCE_FIELDS_V2
                .iter()
                .map(|(name, field_type)| (*name, field_type.dtype())),
        ),
        Nullability::NonNullable,
    );
    if array.dtype() != &expected {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
            "reviewed synchronization Vortex dtype is not the exact V2 schema",
        ));
    }
    if !array.all_valid().map_err(|_| {
        ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
            "reviewed synchronization root validity cannot be inspected",
        )
    })? {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
            "reviewed synchronization root contains null rows",
        ));
    }
    let structure = array.to_struct();
    for (name, _) in EVIDENCE_FIELDS_V2 {
        let field = structure.unmasked_field_by_name(name).map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                "reviewed synchronization Vortex field is missing",
            )
        })?;
        if !field.all_valid().map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                "reviewed synchronization field validity cannot be inspected",
            )
        })? {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                "reviewed synchronization Vortex field contains nulls",
            ));
        }
    }
    Ok(())
}

fn decode_evidence_rows(
    array: &ArrayRef,
) -> Result<Vec<DecodedEvidenceRowV2>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    let sequence = u64_column(array, "sequence")?;
    let account_id = i64_column(array, "account_id")?;
    let evidence_kind = u8_column(array, "evidence_kind")?;
    let has_symbol_id = u8_column(array, "has_symbol_id")?;
    let symbol_id = i64_column(array, "symbol_id")?;
    let has_quote_side = u8_column(array, "has_quote_side")?;
    let quote_side = u8_column(array, "quote_side")?;
    let has_window = u8_column(array, "has_requested_window")?;
    let window_from = i64_column(array, "requested_from_unix_ms_inclusive")?;
    let window_to = i64_column(array, "requested_to_unix_ms_exclusive")?;
    let client_msg_id = utf8_column(array, "client_msg_id")?;
    let payload_type = u32_column(array, "payload_type")?;
    let payload_json = utf8_column(array, "payload_json")?;
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
            || client_msg_id[index].chars().any(char::is_control)
            || payload_type[index] == 0
        {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                "reviewed synchronization row has invalid exact metadata",
            ));
        }
        let requested_window = if has_window[index] == 1 {
            Some(
                EvidenceWindowV1::new(window_from[index], window_to[index]).map_err(|_| {
                    ingress_error(
                        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                        "reviewed synchronization row has an invalid half-open window",
                    )
                })?,
            )
        } else {
            None
        };
        let decoded_side = if has_quote_side[index] == 1 {
            Some(match quote_side[index] {
                0 => QuoteSideV1::Bid,
                1 => QuoteSideV1::Ask,
                _ => {
                    return Err(ingress_error(
                        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
                        "reviewed synchronization row has an invalid quote side",
                    ));
                }
            })
        } else {
            None
        };
        let payload: Value = serde_json::from_str(&payload_json[index]).map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
                "reviewed synchronization payload is not JSON",
            )
        })?;
        if !payload.is_object() {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
                "reviewed synchronization payload is not a JSON object",
            ));
        }
        rows.push(DecodedEvidenceRowV2 {
            sequence: sequence[index],
            account_id: account_id[index],
            kind: evidence_kind[index],
            symbol_id: if has_symbol_id[index] == 1 {
                Some(symbol_id[index])
            } else {
                None
            },
            quote_side: decoded_side,
            requested_window,
            client_msg_id: client_msg_id[index].clone(),
            payload_type: payload_type[index],
            payload_json: payload_json[index].clone(),
            payload,
        });
    }
    Ok(rows)
}

fn u8_column(
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u8>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    Ok(primitive_field(array, name)?.as_slice::<u8>().to_vec())
}

fn u32_column(
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u32>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    Ok(primitive_field(array, name)?.as_slice::<u32>().to_vec())
}

fn u64_column(
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<u64>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    Ok(primitive_field(array, name)?.as_slice::<u64>().to_vec())
}

fn i64_column(
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<i64>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    Ok(primitive_field(array, name)?.as_slice::<i64>().to_vec())
}

fn primitive_field(
    array: &ArrayRef,
    name: &str,
) -> Result<vortex_array::arrays::PrimitiveArray, ReviewedCTraderQuoteSynchronizationIngressErrorV2>
{
    let field = array
        .to_struct()
        .unmasked_field_by_name(name)
        .map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                "reviewed synchronization primitive field cannot be resolved",
            )
        })?
        .clone();
    Ok(field.to_primitive())
}

fn utf8_column(
    array: &ArrayRef,
    name: &str,
) -> Result<Vec<String>, ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    let field = array
        .to_struct()
        .unmasked_field_by_name(name)
        .map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                "reviewed synchronization UTF-8 field cannot be resolved",
            )
        })?
        .clone()
        .to_varbinview();
    (0..field.len())
        .map(|index| {
            let bytes = field.bytes_at(index);
            std::str::from_utf8(bytes.as_ref())
                .map(str::to_owned)
                .map_err(|_| {
                    ingress_error(
                        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
                        "reviewed synchronization string field is not UTF-8",
                    )
                })
        })
        .collect()
}

fn validate_explicit_evidence_file(
    path: &Path,
) -> Result<(), ReviewedCTraderQuoteSynchronizationIngressErrorV2> {
    if !path.is_absolute()
        || !path.components().all(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::Normal(_)
            )
        })
        || path
            .components()
            .any(|component| mutable_selector_component(component.as_os_str()))
    {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath,
            "reviewed synchronization requires an explicit immutable absolute path",
        ));
    }
    for entry in path.ancestors() {
        let metadata = fs::symlink_metadata(entry).map_err(|_| {
            ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath,
                "reviewed synchronization path cannot be inspected",
            )
        })?;
        if metadata_is_link_or_reparse(&metadata) {
            return Err(ingress_error(
                ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath,
                "reviewed synchronization path contains a link or reparse point",
            ));
        }
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath,
            "reviewed synchronization evidence file cannot be inspected",
        )
    })?;
    if !metadata.is_file()
        || metadata_is_link_or_reparse(&metadata)
        || metadata.len() == 0
        || metadata.len() > MAX_REVIEWED_SYNCHRONIZATION_FILE_BYTES_V2
    {
        return Err(ingress_error(
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath,
            "reviewed synchronization evidence is not a bounded regular file",
        ));
    }
    Ok(())
}

fn mutable_selector_component(component: &OsStr) -> bool {
    let Some(value) = component.to_str() else {
        return false;
    };
    ["current", "default", "latest"]
        .iter()
        .any(|selector| value.eq_ignore_ascii_case(selector))
}

fn metadata_is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}
