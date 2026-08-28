use crate::data_selection::CanonicalSearchArtifactScopeV2;
use crate::eval::{BacktestSettings, SessionSpreadProfile, SmcRow};
use neoethos_data::{FeatureFrame, Ohlcv};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;
use std::ops::Range;

pub const EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1: u16 = 1;

const PARENT_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.resident-parent.v1\0";
const VIEW_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.resident-view.v1\0";
const EVALUATION_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.resident-evaluation.v1\0";
const AUTHORITY_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.resident-authority.v1\0";
/// Bound the temporary dense projection used while hashing Vortex-backed or
/// in-memory features. At f64 plus typed validity this keeps the working set to
/// roughly 9 MiB instead of duplicating the complete search matrix.
const FEATURE_HASH_MAX_CELLS_V1: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactResidentDatasetAuthorityErrorCodeV1 {
    InvalidCanonicalScope,
    EmptyParent,
    ShapeMismatch,
    TimestampMismatch,
    InvalidView,
    AdaptiveShapeMismatch,
    DimensionOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExactResidentDatasetAuthorityErrorV1 {
    code: ExactResidentDatasetAuthorityErrorCodeV1,
    message: String,
}

impl ExactResidentDatasetAuthorityErrorV1 {
    pub const fn code(&self) -> ExactResidentDatasetAuthorityErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for ExactResidentDatasetAuthorityErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExactResidentDatasetAuthorityErrorV1 {}

fn error(
    code: ExactResidentDatasetAuthorityErrorCodeV1,
    message: impl Into<String>,
) -> ExactResidentDatasetAuthorityErrorV1 {
    ExactResidentDatasetAuthorityErrorV1 {
        code,
        message: message.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExactResidentDatasetViewV1 {
    Full { row_count: usize },
    ContiguousRange { start: usize, end: usize },
    OrderedIndices { indices: Vec<usize> },
}

impl ExactResidentDatasetViewV1 {
    pub fn row_count(&self) -> usize {
        match self {
            Self::Full { row_count } => *row_count,
            Self::ContiguousRange { start, end } => end - start,
            Self::OrderedIndices { indices } => indices.len(),
        }
    }

    pub fn contiguous_range(&self) -> Option<Range<usize>> {
        match self {
            Self::Full { .. } | Self::OrderedIndices { .. } => None,
            Self::ContiguousRange { start, end } => Some(*start..*end),
        }
    }

    pub fn ordered_indices(&self) -> Option<&[usize]> {
        match self {
            Self::OrderedIndices { indices } => Some(indices),
            Self::Full { .. } | Self::ContiguousRange { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExactResidentDatasetViewRequestV1<'a> {
    Full,
    ContiguousRange { start: usize, end: usize },
    OrderedIndices(&'a [usize]),
}

pub(crate) struct ExactResidentDatasetParentSealRequestV1<'a> {
    pub(crate) scope: &'a CanonicalSearchArtifactScopeV2,
    pub(crate) features: &'a FeatureFrame,
    pub(crate) ohlcv: &'a Ohlcv,
    pub(crate) smc_data: &'a [SmcRow],
}

/// Opaque parent identity computed once per exact discovery parent. View and
/// settings derivation cannot access or re-materialize the source arrays.
#[derive(Clone, Debug)]
pub(crate) struct SealedExactResidentDatasetParentV1 {
    canonical_scope_identity_sha256: String,
    parent_dataset_identity_sha256: String,
    parent_row_count: usize,
    feature_count: usize,
}

impl SealedExactResidentDatasetParentV1 {
    pub(crate) fn parent_dataset_identity_sha256(&self) -> &str {
        &self.parent_dataset_identity_sha256
    }

    pub(crate) fn canonical_scope_identity_sha256(&self) -> &str {
        &self.canonical_scope_identity_sha256
    }

    pub(crate) const fn parent_row_count(&self) -> usize {
        self.parent_row_count
    }

    pub(crate) const fn feature_count(&self) -> usize {
        self.feature_count
    }
}

pub(crate) struct ExactResidentDatasetAuthorityDeriveRequestV1<'a> {
    pub(crate) parent: &'a SealedExactResidentDatasetParentV1,
    pub(crate) settings: &'a BacktestSettings,
    pub(crate) view: ExactResidentDatasetViewRequestV1<'a>,
}

/// Opaque, immutable identity of the exact dataset and evaluation view a
/// resident accelerator session is allowed to consume.
///
/// There is intentionally no deserializer and no public constructor. Search
/// seals this value from the already-validated CanonicalSearch V2 scope and
/// concrete arrays; callers cannot promote a scalar cache key or supplied hash
/// into residency authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExactResidentDatasetAuthorityV1 {
    schema_version: u16,
    canonical_scope_identity_sha256: String,
    parent_dataset_identity_sha256: String,
    view_identity_sha256: String,
    evaluation_binding_sha256: String,
    identity_sha256: String,
    parent_row_count: usize,
    feature_count: usize,
    view: ExactResidentDatasetViewV1,
}

impl ExactResidentDatasetAuthorityV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn canonical_scope_identity_sha256(&self) -> &str {
        &self.canonical_scope_identity_sha256
    }

    pub fn parent_dataset_identity_sha256(&self) -> &str {
        &self.parent_dataset_identity_sha256
    }

    pub fn view_identity_sha256(&self) -> &str {
        &self.view_identity_sha256
    }

    pub fn evaluation_binding_sha256(&self) -> &str {
        &self.evaluation_binding_sha256
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn parent_row_count(&self) -> usize {
        self.parent_row_count
    }

    pub const fn feature_count(&self) -> usize {
        self.feature_count
    }

    pub const fn view(&self) -> &ExactResidentDatasetViewV1 {
        &self.view
    }
}

struct ExactHasherV1(Sha256);

impl ExactHasherV1 {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self(hasher)
    }

    fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.0.update(value);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.0.update([u8::from(value)]);
    }

    fn u8(&mut self, value: u8) {
        self.0.update([value]);
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.0.update(value.to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn finish(self) -> String {
        hex_lower(&self.0.finalize())
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn usize_to_u64(value: usize) -> Result<u64, ExactResidentDatasetAuthorityErrorV1> {
    u64::try_from(value).map_err(|_| {
        error(
            ExactResidentDatasetAuthorityErrorCodeV1::DimensionOverflow,
            "resident dataset dimension does not fit the versioned u64 identity",
        )
    })
}

fn hash_f64_slice(hasher: &mut ExactHasherV1, values: &[f64]) {
    hasher.u64(values.len() as u64);
    for value in values {
        hasher.f64(*value);
    }
}

fn hash_i64_slice(hasher: &mut ExactHasherV1, values: &[i64]) {
    hasher.u64(values.len() as u64);
    for value in values {
        hasher.i64(*value);
    }
}

fn hash_optional_session_spread(hasher: &mut ExactHasherV1, profile: Option<SessionSpreadProfile>) {
    match profile {
        None => hasher.u8(0),
        Some(profile) => {
            hasher.u8(1);
            hasher.f64(profile.asian_pips);
            hasher.f64(profile.overlap_pips);
            hasher.f64(profile.late_ny_pips);
        }
    }
}

fn hash_settings(hasher: &mut ExactHasherV1, settings: &BacktestSettings) {
    let BacktestSettings {
        sl_pips,
        tp_pips,
        max_hold_bars,
        min_hold_bars,
        max_trades_per_day,
        gap_threshold_ms,
        trailing_enabled,
        trailing_atr_multiplier,
        trailing_be_trigger_r,
        trailing_min_lock_pips,
        pip_value,
        spread_pips,
        commission_per_trade,
        pip_value_per_lot,
        kill_zones_enabled,
        session_spread_profile,
        swap_long_pips_per_day,
        swap_short_pips_per_day,
        pnl_conversion_fee_rate,
        risk_based_sizing,
        risk_per_trade_min,
        risk_per_trade_max,
        high_quality_confidence,
        adaptive_base_pips,
        adaptive_vol_mult,
        adaptive_rr,
    } = settings;

    hasher.f64(*sl_pips);
    hasher.f64(*tp_pips);
    hasher.u64(*max_hold_bars as u64);
    hasher.u64(*min_hold_bars as u64);
    hasher.u64(*max_trades_per_day as u64);
    hasher.i64(*gap_threshold_ms);
    hasher.bool(*trailing_enabled);
    hasher.f64(*trailing_atr_multiplier);
    hasher.f64(*trailing_be_trigger_r);
    hasher.f64(*trailing_min_lock_pips);
    hasher.f64(*pip_value);
    hasher.f64(*spread_pips);
    hasher.f64(*commission_per_trade);
    hasher.f64(*pip_value_per_lot);
    hasher.bool(*kill_zones_enabled);
    hash_optional_session_spread(hasher, *session_spread_profile);
    hasher.f64(*swap_long_pips_per_day);
    hasher.f64(*swap_short_pips_per_day);
    hasher.f64(*pnl_conversion_fee_rate);
    hasher.bool(*risk_based_sizing);
    hasher.f64(*risk_per_trade_min);
    hasher.f64(*risk_per_trade_max);
    hasher.f64(*high_quality_confidence);
    match adaptive_base_pips {
        None => hasher.u8(0),
        Some(values) => {
            hasher.u8(1);
            hash_f64_slice(hasher, values);
        }
    }
    hasher.f64(*adaptive_vol_mult);
    hasher.f64(*adaptive_rr);
}

fn exact_view(
    request: ExactResidentDatasetViewRequestV1<'_>,
    parent_rows: usize,
) -> Result<ExactResidentDatasetViewV1, ExactResidentDatasetAuthorityErrorV1> {
    match request {
        ExactResidentDatasetViewRequestV1::Full => Ok(ExactResidentDatasetViewV1::Full {
            row_count: parent_rows,
        }),
        ExactResidentDatasetViewRequestV1::ContiguousRange { start, end } => {
            if start >= end || end > parent_rows {
                return Err(error(
                    ExactResidentDatasetAuthorityErrorCodeV1::InvalidView,
                    "resident contiguous view is empty, reversed, or outside the exact parent",
                ));
            }
            Ok(ExactResidentDatasetViewV1::ContiguousRange { start, end })
        }
        ExactResidentDatasetViewRequestV1::OrderedIndices(indices) => {
            if indices.is_empty()
                || indices.iter().any(|index| *index >= parent_rows)
                || indices.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(error(
                    ExactResidentDatasetAuthorityErrorCodeV1::InvalidView,
                    "resident index view must be non-empty, in bounds, and strictly increasing",
                ));
            }
            Ok(ExactResidentDatasetViewV1::OrderedIndices {
                indices: indices.to_vec(),
            })
        }
    }
}

fn hash_view(
    parent_identity: &str,
    view: &ExactResidentDatasetViewV1,
) -> Result<String, ExactResidentDatasetAuthorityErrorV1> {
    let mut hasher = ExactHasherV1::new(VIEW_HASH_DOMAIN_V1);
    hasher.string(parent_identity);
    match view {
        ExactResidentDatasetViewV1::Full { row_count } => {
            hasher.u8(0);
            hasher.u64(usize_to_u64(*row_count)?);
        }
        ExactResidentDatasetViewV1::ContiguousRange { start, end } => {
            hasher.u8(1);
            hasher.u64(usize_to_u64(*start)?);
            hasher.u64(usize_to_u64(*end)?);
        }
        ExactResidentDatasetViewV1::OrderedIndices { indices } => {
            hasher.u8(2);
            hasher.u64(usize_to_u64(indices.len())?);
            for index in indices {
                hasher.u64(usize_to_u64(*index)?);
            }
        }
    }
    Ok(hasher.finish())
}

fn validate_shapes<'data>(
    request: &ExactResidentDatasetParentSealRequestV1<'data>,
) -> Result<&'data [i64], ExactResidentDatasetAuthorityErrorV1> {
    request.scope.validate().map_err(|source| {
        error(
            ExactResidentDatasetAuthorityErrorCodeV1::InvalidCanonicalScope,
            format!("canonical search scope is invalid: {source}"),
        )
    })?;

    let rows = request.features.n_samples();
    if rows == 0 || request.features.n_features() == 0 {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::EmptyParent,
            "resident parent requires non-empty rows and feature columns",
        ));
    }
    let Some(timestamps) = request.ohlcv.timestamp.as_deref() else {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::TimestampMismatch,
            "resident parent OHLCV has no canonical timestamps",
        ));
    };
    if request.features.timestamps.as_slice() != timestamps {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::TimestampMismatch,
            "resident feature and OHLCV timestamps differ",
        ));
    }
    if request.ohlcv.open.len() != rows
        || request.ohlcv.high.len() != rows
        || request.ohlcv.low.len() != rows
        || request.ohlcv.close.len() != rows
        || request
            .ohlcv
            .volume
            .as_ref()
            .is_some_and(|volume| volume.len() != rows)
        || request.smc_data.len() != rows
    {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::ShapeMismatch,
            "resident parent OHLCV, features, or SMC rows disagree",
        ));
    }

    let window = request.scope.evaluated_window();
    let scope_rows = window
        .row_end()
        .checked_sub(window.row_start())
        .ok_or_else(|| {
            error(
                ExactResidentDatasetAuthorityErrorCodeV1::InvalidCanonicalScope,
                "canonical search scope row range underflowed",
            )
        })?;
    if scope_rows != usize_to_u64(rows)?
        || window.timestamp_start_ms() != timestamps[0]
        || window.timestamp_end_ms() != timestamps[rows - 1]
    {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::TimestampMismatch,
            "canonical scope does not name the exact resident parent rows and timestamps",
        ));
    }
    Ok(timestamps)
}

fn hash_parent(
    request: &ExactResidentDatasetParentSealRequestV1<'_>,
    scope_identity: &str,
    timestamps: &[i64],
) -> Result<String, ExactResidentDatasetAuthorityErrorV1> {
    let rows = request.features.n_samples();
    let columns = request.features.n_features();

    let mut hasher = ExactHasherV1::new(PARENT_HASH_DOMAIN_V1);
    hasher.string(scope_identity);
    hasher.u64(usize_to_u64(rows)?);
    hash_i64_slice(&mut hasher, timestamps);
    hash_f64_slice(&mut hasher, &request.ohlcv.open);
    hash_f64_slice(&mut hasher, &request.ohlcv.high);
    hash_f64_slice(&mut hasher, &request.ohlcv.low);
    hash_f64_slice(&mut hasher, &request.ohlcv.close);
    match &request.ohlcv.volume {
        None => hasher.u8(0),
        Some(volume) => {
            hasher.u8(1);
            hash_f64_slice(&mut hasher, volume);
        }
    }

    hasher.u64(usize_to_u64(columns)?);
    for name in &request.features.names {
        hasher.string(name);
    }

    // Hash the exact same row-major `(f64 bits, validity code)` stream as the
    // former full dense projection, but bound temporary allocation by a fixed
    // cell budget. `dense_window` is the canonical projection API for all
    // FeatureFrame backings, including Vortex and nested views.
    let rows_per_chunk = (FEATURE_HASH_MAX_CELLS_V1 / columns).max(1);
    let mut start = 0usize;
    while start < rows {
        let end = start.saturating_add(rows_per_chunk).min(rows);
        let dense = request
            .features
            .dense_window(start, end)
            .map_err(|source| {
                error(
                    ExactResidentDatasetAuthorityErrorCodeV1::ShapeMismatch,
                    format!("materializing bounded resident feature window: {source}"),
                )
            })?;
        let chunk_rows = end - start;
        if dense.values.dim() != (chunk_rows, columns)
            || dense.validity.dim() != (chunk_rows, columns)
        {
            return Err(error(
                ExactResidentDatasetAuthorityErrorCodeV1::ShapeMismatch,
                "resident dense feature-window values/validity dimensions disagree",
            ));
        }
        for row in 0..chunk_rows {
            for column in 0..columns {
                hasher.f64(dense.values[(row, column)]);
                hasher.u8(dense.validity[(row, column)].code());
            }
        }
        start = end;
    }
    hasher.u64(usize_to_u64(request.smc_data.len())?);
    for row in request.smc_data {
        for value in row {
            hasher.u8(*value as u8);
        }
    }

    // Month/day values are uploaded independently by the native session. They
    // are deterministically derived here rather than accepted as caller data.
    let (months, days) = crate::genetic::month_day_indices(timestamps);
    hash_i64_slice(&mut hasher, &months);
    hash_i64_slice(&mut hasher, &days);
    Ok(hasher.finish())
}

pub(crate) fn seal_exact_resident_dataset_parent_v1(
    request: ExactResidentDatasetParentSealRequestV1<'_>,
) -> Result<SealedExactResidentDatasetParentV1, ExactResidentDatasetAuthorityErrorV1> {
    let timestamps = validate_shapes(&request)?;
    let canonical_scope_identity_sha256 = request.scope.identity_sha256().map_err(|source| {
        error(
            ExactResidentDatasetAuthorityErrorCodeV1::InvalidCanonicalScope,
            format!("hash canonical search scope: {source}"),
        )
    })?;
    let parent_dataset_identity_sha256 =
        hash_parent(&request, &canonical_scope_identity_sha256, timestamps)?;
    Ok(SealedExactResidentDatasetParentV1 {
        canonical_scope_identity_sha256,
        parent_dataset_identity_sha256,
        parent_row_count: request.features.n_samples(),
        feature_count: request.features.n_features(),
    })
}

pub(crate) fn derive_exact_resident_dataset_authority_v1(
    request: ExactResidentDatasetAuthorityDeriveRequestV1<'_>,
) -> Result<ExactResidentDatasetAuthorityV1, ExactResidentDatasetAuthorityErrorV1> {
    let view = exact_view(request.view, request.parent.parent_row_count)?;
    if request
        .settings
        .adaptive_base_pips
        .as_ref()
        .is_some_and(|values| values.len() != view.row_count())
    {
        return Err(error(
            ExactResidentDatasetAuthorityErrorCodeV1::AdaptiveShapeMismatch,
            "adaptive stop series must cover the exact resident evaluation view",
        ));
    }

    let view_identity_sha256 = hash_view(&request.parent.parent_dataset_identity_sha256, &view)?;

    let mut evaluation_hasher = ExactHasherV1::new(EVALUATION_HASH_DOMAIN_V1);
    evaluation_hasher.string(&view_identity_sha256);
    hash_settings(&mut evaluation_hasher, request.settings);
    let evaluation_binding_sha256 = evaluation_hasher.finish();

    let mut authority_hasher = ExactHasherV1::new(AUTHORITY_HASH_DOMAIN_V1);
    authority_hasher.u64(u64::from(
        EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1,
    ));
    authority_hasher.string(&request.parent.canonical_scope_identity_sha256);
    authority_hasher.string(&request.parent.parent_dataset_identity_sha256);
    authority_hasher.string(&view_identity_sha256);
    authority_hasher.string(&evaluation_binding_sha256);
    let identity_sha256 = authority_hasher.finish();

    Ok(ExactResidentDatasetAuthorityV1 {
        schema_version: EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1,
        canonical_scope_identity_sha256: request.parent.canonical_scope_identity_sha256.clone(),
        parent_dataset_identity_sha256: request.parent.parent_dataset_identity_sha256.clone(),
        view_identity_sha256,
        evaluation_binding_sha256,
        identity_sha256,
        parent_row_count: request.parent.parent_row_count,
        feature_count: request.parent.feature_count,
        view,
    })
}
