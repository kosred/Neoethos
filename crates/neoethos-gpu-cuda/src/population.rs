//! Safe Rust ownership wrapper around the persistent Prototype B CUDA session.
//!
//! Every argument is validated on the host before it crosses the C ABI, so an
//! invalid shape is a typed Rust error rather than undefined behaviour on the
//! device. A session owns exactly one native session; `Drop` destroys it on
//! every path, including error paths and unwinds.

use super::{
    DatasetHeader, GeneDescriptor, NeoPopulationCounters, NeoPopulationEvent,
    NeoPopulationMetricRow, NeoPopulationOutcome, NeoPopulationSettings, SMC_SLOTS,
    ScenarioDescriptor,
};
use neoethos_gpu_contracts::ABI_VERSION;
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;
#[cfg(feature = "cuda-device-fixtures")]
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

pub const STATUS_OK: i32 = 0;
pub const STATUS_UNSUPPORTED: i32 = -1;
pub const STATUS_NULL_SESSION: i32 = -30;
pub const STATUS_ABI_MISMATCH: i32 = -31;
pub const STATUS_INVALID_ARGUMENT: i32 = -32;
pub const STATUS_DEVICE_UNAVAILABLE: i32 = -33;
pub const STATUS_ALLOCATION_FAILED: i32 = -34;
pub const STATUS_TRANSFER_FAILED: i32 = -35;
pub const STATUS_LAUNCH_FAILED: i32 = -36;
pub const STATUS_EVENT_CAPACITY: i32 = -37;
pub const STATUS_MISSING_UPLOAD: i32 = -38;
pub const STATUS_READBACK_CAPACITY: i32 = -39;
pub const STATUS_SYNC_FAILED: i32 = -40;
pub const STATUS_UNKNOWN_EVENT: i32 = -41;
pub const STATUS_DATASET_REUPLOAD: i32 = -42;
pub const STATUS_WORKSPACE_MODE_MISMATCH: i32 = -43;
pub const STATUS_WORKSPACE_PLAN_MISMATCH: i32 = -44;
pub const STATUS_STRICT_RESIDENT_IN_FLIGHT: i32 = -45;
pub const STATUS_STRICT_RESIDENT_POISONED: i32 = -46;
pub const STATUS_ADAPTIVE_BASE_DEGENERATE: i32 = -47;
pub const STATUS_ASYNC_FREE_OUTCOME_UNKNOWN: i32 = -48;
pub const STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN: i32 = -49;

/// Trade slots the kernel reserves per candidate.
///
/// The outcome array is `population * MAX_TRADES_PER_CANDIDATE` records, so at
/// 72 bytes each this is ~590 KB per candidate and it is what the card runs out
/// of. A caller that does not know the number cannot know how many candidates
/// fit, and the session's own budget still sizes an event buffer that no longer
/// exists — so peak memory became a function of the requested population, which
/// is exactly what the never-OOM invariant forbids.
///
/// `trade_slots_match_the_kernel` keeps this equal to the kernel's own
/// constant. Two languages agreeing by convention is how the retry-smaller path
/// silently stopped working; this one is checked.
pub const MAX_TRADES_PER_CANDIDATE: u64 = 8192;
const POPULATION_METRIC_ROW_BYTES_V1: u64 = 104;
const POPULATION_SCENARIO_DEVICE_BYTES_V1: u64 = 56;
const POPULATION_F64_BYTES_V1: u64 = 8;
pub const RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1: &str = "neoethos.population.resident-adaptive-base.semantic-v1;view-local-full-or-contiguous;safe-log-floor=1e-12;safe-log=neoethos.quant.log.semantic-v3;sun-fdlibm-openlibm-e_log;positive-finite-binary64;commit=82e90aef0657289192efe77be89791c07dea0775;source-sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD;rounding=rn-no-fma;cpu-cuda-bit-tolerance=zero;real-log-accuracy=bounded-faithful-max-1ulp-reviewed-wide-domain;parkinson-window=50;horizon=5;tail-window=100;tail-alpha=0.975;q-index=2;tail-grid=view-origin;global-finite-median-replacement;degenerate=fail;zero-adaptive-h2d";

pub fn population_status_message(status: i32) -> &'static str {
    match status {
        STATUS_OK => "ok",
        STATUS_UNSUPPORTED => "this build has no CUDA runtime",
        STATUS_NULL_SESSION => "null native session",
        STATUS_ABI_MISMATCH => "native ABI version mismatch",
        STATUS_INVALID_ARGUMENT => "invalid native argument",
        STATUS_DEVICE_UNAVAILABLE => "CUDA device is unavailable",
        STATUS_ALLOCATION_FAILED => "device allocation failed",
        STATUS_TRANSFER_FAILED => "host/device transfer failed",
        STATUS_LAUNCH_FAILED => "kernel launch failed",
        STATUS_EVENT_CAPACITY => "emitted events exceeded the session capacity",
        STATUS_MISSING_UPLOAD => "a required upload or evaluation is missing",
        STATUS_READBACK_CAPACITY => "readback buffer is too small",
        STATUS_SYNC_FAILED => "stream or event synchronization failed",
        STATUS_UNKNOWN_EVENT => "event id does not belong to this session",
        STATUS_DATASET_REUPLOAD => "a session accepts exactly one logical dataset upload",
        STATUS_WORKSPACE_MODE_MISMATCH => {
            "population session workspace mode cannot change after first selection"
        }
        STATUS_WORKSPACE_PLAN_MISMATCH => {
            "resident population workspace does not match the sealed plan"
        }
        STATUS_STRICT_RESIDENT_IN_FLIGHT => {
            "strict resident GPU work has not been consumed by the next device stage"
        }
        STATUS_STRICT_RESIDENT_POISONED => {
            "strict resident GPU session is poisoned after an ambiguous or dropped launch"
        }
        STATUS_ADAPTIVE_BASE_DEGENERATE => {
            "resident adaptive-stop base is non-finite or has a non-positive median"
        }
        STATUS_ASYNC_FREE_OUTCOME_UNKNOWN => {
            "stream-ordered free outcome is unknown; pointer was retired and any allocation leak is deliberate"
        }
        STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN => {
            "stream-ordered allocation outcome is unknown; no device identity was published"
        }
        _ => "unknown native status",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaPopulationError {
    #[error("native CUDA population {operation} failed with status {status} ({message})")]
    Native {
        operation: &'static str,
        status: i32,
        message: &'static str,
    },
    #[error("native CUDA population runtime is unavailable in this build")]
    RuntimeUnavailable,
    #[error(
        "native CUDA population {operation} reported an unknown stream-ordered free outcome; the pointer identity is retired and a possible allocation leak is deliberate"
    )]
    AsyncFreeOutcomeUnknownDeliberateLeak { operation: &'static str },
    #[error(
        "native CUDA population {operation} reported an unknown stream-ordered allocation outcome; no device identity is available for reuse or cleanup"
    )]
    AsyncAllocationOutcomeUnknownDeliberateLeak { operation: &'static str },
    #[error("invalid Prototype B population input: {0}")]
    InvalidInput(String),
}

impl CudaPopulationError {
    pub(crate) fn native(operation: &'static str, status: i32) -> Self {
        if status == STATUS_UNSUPPORTED {
            return Self::RuntimeUnavailable;
        }
        if status == STATUS_ASYNC_FREE_OUTCOME_UNKNOWN {
            return Self::AsyncFreeOutcomeUnknownDeliberateLeak { operation };
        }
        if status == STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN {
            return Self::AsyncAllocationOutcomeUnknownDeliberateLeak { operation };
        }
        Self::Native {
            operation,
            status,
            message: population_status_message(status),
        }
    }

    pub fn is_runtime_unavailable(&self) -> bool {
        matches!(self, Self::RuntimeUnavailable)
    }

    /// Whether the card ran out of room, so the same work would succeed in
    /// smaller pieces.
    ///
    /// This matched only `STATUS_EVENT_CAPACITY` — the event buffer's own
    /// message. Once the reduce stopped materialising events that buffer was
    /// gone, and the allocation that now runs out first is the outcome array,
    /// which reports `STATUS_ALLOCATION_FAILED`. The caller kept asking about a
    /// buffer that no longer existed, so a plain out-of-memory read as a fault
    /// and the whole population went to the CPU instead of being halved.
    ///
    /// Measured cost of that gap: 770 500 of 778 205 validation items on the
    /// CPU, the card idle for 99 % of a run.
    pub fn is_capacity_exhausted(&self) -> bool {
        matches!(
            self,
            Self::Native {
                status: STATUS_EVENT_CAPACITY | STATUS_ALLOCATION_FAILED,
                ..
            }
        )
    }
}

fn invalid(detail: impl Into<String>) -> CudaPopulationError {
    CudaPopulationError::InvalidInput(detail.into())
}

/// Borrowed host dataset for exactly one logical upload.
#[derive(Debug, Clone, Copy)]
pub struct PopulationDatasetView<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    /// Canonical f64 feature-major `[feature][bar]` values.
    pub indicators: &'a [f64],
    pub feature_count: usize,
    pub months: &'a [i64],
    pub days: &'a [i64],
    pub timestamps: &'a [i64],
    /// Row-major `[bar][slot]` SMC contract, `SMC_SLOTS` per bar.
    pub smc_rows: &'a [i8],
    /// f64 on purpose: the canonical adaptive-at-entry stop distance is f64,
    /// and narrowing it would silently break exact parity.
    pub adaptive_base_pips: Option<&'a [f64]>,
}

impl PopulationDatasetView<'_> {
    fn validate(&self) -> Result<usize, CudaPopulationError> {
        let bars = self.close.len();
        if bars == 0 {
            return Err(invalid("dataset has no bars"));
        }
        if self.feature_count == 0 {
            return Err(invalid("dataset has no features"));
        }
        for (field, actual) in [
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("months", self.months.len()),
            ("days", self.days.len()),
            ("timestamps", self.timestamps.len()),
        ] {
            if actual != bars {
                return Err(invalid(format!(
                    "dataset {field} length {actual} does not match {bars} bars"
                )));
            }
        }
        if self.smc_rows.len() != bars * SMC_SLOTS {
            return Err(invalid(format!(
                "dataset smc_rows length {} does not match {bars} bars x {SMC_SLOTS} slots",
                self.smc_rows.len()
            )));
        }
        let expected = self
            .feature_count
            .checked_mul(bars)
            .ok_or_else(|| invalid("dataset indicator extent overflows usize"))?;
        if self.indicators.len() != expected {
            return Err(invalid(format!(
                "dataset indicators length {} does not match {expected}",
                self.indicators.len()
            )));
        }
        if self
            .adaptive_base_pips
            .is_some_and(|base| base.len() != bars)
        {
            return Err(invalid(format!(
                "adaptive base length {} does not match {bars} bars",
                self.adaptive_base_pips.map_or(0, <[f64]>::len)
            )));
        }
        for (field, values) in [
            ("close", self.close),
            ("high", self.high),
            ("low", self.low),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(invalid(format!("dataset {field}[{index}] is not finite")));
            }
        }
        Ok(bars)
    }
}

/// Immutable canonical parent buffers uploaded once for one native population
/// session. View-local adaptive settings and row selections deliberately do not
/// belong here: changing either must bind a view, never upload another parent.
#[derive(Debug, Clone)]
pub struct PopulationParentDatasetInputV1 {
    pub close: Arc<[f64]>,
    pub high: Arc<[f64]>,
    pub low: Arc<[f64]>,
    pub indicators_feature_major: Arc<[f64]>,
    pub feature_count: usize,
    pub months: Arc<[i64]>,
    pub days: Arc<[i64]>,
    pub timestamps: Arc<[i64]>,
    pub smc_rows: Arc<[i8]>,
}

#[derive(Debug, Clone)]
pub struct PopulationParentDatasetV1 {
    close: Arc<[f64]>,
    high: Arc<[f64]>,
    low: Arc<[f64]>,
    indicators_feature_major: Arc<[f64]>,
    feature_count: usize,
    months: Arc<[i64]>,
    days: Arc<[i64]>,
    timestamps: Arc<[i64]>,
    smc_rows: Arc<[i8]>,
}

impl PopulationParentDatasetV1 {
    pub fn new(input: PopulationParentDatasetInputV1) -> Result<Self, CudaPopulationError> {
        let PopulationParentDatasetInputV1 {
            close,
            high,
            low,
            indicators_feature_major,
            feature_count,
            months,
            days,
            timestamps,
            smc_rows,
        } = input;
        let rows = close.len();
        if rows == 0 {
            return Err(invalid("parent dataset has no rows"));
        }
        if feature_count == 0 {
            return Err(invalid("parent dataset has no features"));
        }
        for (field, actual) in [
            ("high", high.len()),
            ("low", low.len()),
            ("months", months.len()),
            ("days", days.len()),
            ("timestamps", timestamps.len()),
        ] {
            if actual != rows {
                return Err(invalid(format!(
                    "parent {field} length {actual} does not match {rows} rows"
                )));
            }
        }
        let indicator_count = feature_count
            .checked_mul(rows)
            .ok_or_else(|| invalid("parent feature extent overflows usize"))?;
        if indicators_feature_major.len() != indicator_count {
            return Err(invalid(format!(
                "parent indicators length {} does not match {indicator_count}",
                indicators_feature_major.len()
            )));
        }
        let smc_count = rows
            .checked_mul(SMC_SLOTS)
            .ok_or_else(|| invalid("parent SMC extent overflows usize"))?;
        if smc_rows.len() != smc_count {
            return Err(invalid(format!(
                "parent smc_rows length {} does not match {rows} rows x {SMC_SLOTS} slots",
                smc_rows.len()
            )));
        }
        for (field, values) in [
            ("close", close.as_ref()),
            ("high", high.as_ref()),
            ("low", low.as_ref()),
            ("indicators", indicators_feature_major.as_ref()),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(invalid(format!("parent {field}[{index}] is not finite")));
            }
        }
        if timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "parent canonical timestamps must be strictly increasing",
            ));
        }

        Ok(Self {
            close,
            high,
            low,
            indicators_feature_major,
            feature_count,
            months,
            days,
            timestamps,
            smc_rows,
        })
    }

    pub fn row_count(&self) -> usize {
        self.close.len()
    }

    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    pub fn indicators_feature_major(&self) -> &[f64] {
        &self.indicators_feature_major
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationViewKindV1 {
    Full,
    ContiguousRange,
    OrderedIndices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationTimestampModeV1 {
    Canonical,
    DisabledIndexDelta,
}

/// A view-local row mapping and optional adaptive stop series. Full and range
/// views are scalar descriptors; only an ordered view owns a compact u64 map.
#[derive(Debug, Clone)]
pub struct PopulationEvaluationViewV1 {
    parent_row_count: usize,
    kind: PopulationViewKindV1,
    range: Option<Range<usize>>,
    ordered_indices: Option<Arc<[u64]>>,
    timestamp_mode: PopulationTimestampModeV1,
    adaptive_base_pips: Option<Arc<[f64]>>,
}

impl PopulationEvaluationViewV1 {
    pub fn full(
        parent_row_count: usize,
        timestamp_mode: PopulationTimestampModeV1,
        adaptive_base_pips: Option<Arc<[f64]>>,
    ) -> Result<Self, CudaPopulationError> {
        Self::build(
            parent_row_count,
            PopulationViewKindV1::Full,
            Some(0..parent_row_count),
            None,
            timestamp_mode,
            adaptive_base_pips,
        )
    }

    pub fn contiguous_range(
        parent_row_count: usize,
        start: usize,
        end: usize,
        timestamp_mode: PopulationTimestampModeV1,
        adaptive_base_pips: Option<Arc<[f64]>>,
    ) -> Result<Self, CudaPopulationError> {
        if start >= end || end > parent_row_count {
            return Err(invalid(
                "contiguous population view is empty, reversed, or outside its parent",
            ));
        }
        Self::build(
            parent_row_count,
            PopulationViewKindV1::ContiguousRange,
            Some(start..end),
            None,
            timestamp_mode,
            adaptive_base_pips,
        )
    }

    pub fn ordered_indices(
        parent_row_count: usize,
        ordered_indices: Arc<[u64]>,
        timestamp_mode: PopulationTimestampModeV1,
        adaptive_base_pips: Option<Arc<[f64]>>,
    ) -> Result<Self, CudaPopulationError> {
        let parent_rows = u64::try_from(parent_row_count)
            .map_err(|_| invalid("parent row count does not fit the native u64 view contract"))?;
        if ordered_indices.is_empty()
            || ordered_indices.iter().any(|index| *index >= parent_rows)
            || ordered_indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "ordered population indices must be non-empty, in bounds, and strictly increasing",
            ));
        }
        Self::build(
            parent_row_count,
            PopulationViewKindV1::OrderedIndices,
            None,
            Some(ordered_indices),
            timestamp_mode,
            adaptive_base_pips,
        )
    }

    fn build(
        parent_row_count: usize,
        kind: PopulationViewKindV1,
        range: Option<Range<usize>>,
        ordered_indices: Option<Arc<[u64]>>,
        timestamp_mode: PopulationTimestampModeV1,
        adaptive_base_pips: Option<Arc<[f64]>>,
    ) -> Result<Self, CudaPopulationError> {
        if parent_row_count == 0 {
            return Err(invalid("population view has an empty parent"));
        }
        let row_count = ordered_indices.as_ref().map_or_else(
            || range.as_ref().map_or(0, |range| range.len()),
            |indices| indices.len(),
        );
        if adaptive_base_pips.as_ref().is_some_and(|values| {
            values.len() != row_count || values.iter().any(|value| !value.is_finite())
        }) {
            return Err(invalid(
                "adaptive population series must be finite and cover the exact view",
            ));
        }
        Ok(Self {
            parent_row_count,
            kind,
            range,
            ordered_indices,
            timestamp_mode,
            adaptive_base_pips,
        })
    }

    pub fn kind(&self) -> PopulationViewKindV1 {
        self.kind
    }

    pub fn row_count(&self) -> usize {
        self.ordered_indices.as_ref().map_or_else(
            || self.range.as_ref().map_or(0, |range| range.len()),
            |indices| indices.len(),
        )
    }

    pub fn range(&self) -> Option<Range<usize>> {
        self.range
            .clone()
            .filter(|_| matches!(self.kind, PopulationViewKindV1::ContiguousRange))
    }

    pub fn ordered_index_values(&self) -> Option<&[u64]> {
        self.ordered_indices.as_deref()
    }

    pub fn timestamp_mode(&self) -> PopulationTimestampModeV1 {
        self.timestamp_mode
    }

    pub fn adaptive_base_pips(&self) -> Option<&[f64]> {
        self.adaptive_base_pips.as_deref()
    }
}

/// Exact canonical recipe for producing one adaptive-stop base directly from
/// a resident V3 population parent.
///
/// The recipe is intentionally narrow: current production Search uses the
/// open-independent Parkinson estimator with a 50-bar volatility window and a
/// 100-return expected-shortfall tail. Full and contiguous Stage-1 views are
/// supported; ordered views fail closed until their view-local sequence has a
/// separately reviewed resident mapping contract.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentAdaptiveBaseRequestV1 {
    abi_version: u32,
    view_kind: u32,
    parent_row_count: u64,
    view_start: u64,
    view_row_count: u64,
    vol_window: u32,
    vol_horizon_bars: u32,
    tail_window: u32,
    tail_quantile_index: u32,
    tail_step: u64,
    tail_max_bars: u64,
    pip_size: f64,
    stop_k_vol: f64,
    stop_k_tail: f64,
    meta_label_min_dist: f64,
}

impl ResidentAdaptiveBaseRequestV1 {
    pub const VOL_WINDOW_V1: u32 = 50;
    pub const VOL_HORIZON_BARS_V1: u32 = 5;
    pub const TAIL_WINDOW_V1: u32 = 100;
    pub const TAIL_QUANTILE_INDEX_V1: u32 = 2;
    pub const STOP_K_VOL_V1: f64 = 1.0;
    pub const STOP_K_TAIL_V1: f64 = 1.25;
    pub const META_LABEL_MIN_DIST_V1: f64 = 0.0;
    pub const MIN_VIEW_ROWS_V1: usize = 101;

    pub fn checked_canonical_v1(
        view: &PopulationEvaluationViewV1,
        pip_size: f64,
        tail_step: usize,
        tail_max_bars: usize,
    ) -> Result<Self, CudaPopulationError> {
        if view.adaptive_base_pips().is_some() {
            return Err(invalid(
                "resident adaptive producer refuses a host adaptive-base slice",
            ));
        }
        if matches!(view.kind(), PopulationViewKindV1::OrderedIndices) {
            return Err(invalid(
                "resident adaptive producer V1 supports only full/contiguous views",
            ));
        }
        let view_row_count = view.row_count();
        if view_row_count < Self::MIN_VIEW_ROWS_V1 {
            return Err(invalid(format!(
                "resident adaptive producer needs at least {} view rows, got {view_row_count}",
                Self::MIN_VIEW_ROWS_V1
            )));
        }
        if !(pip_size.is_finite() && pip_size > 0.0) {
            return Err(invalid(
                "resident adaptive producer requires a positive finite pip size",
            ));
        }
        if tail_step == 0 {
            return Err(invalid(
                "resident adaptive producer requires a non-zero tail step",
            ));
        }
        if tail_max_bars > 0 && view_row_count > tail_max_bars {
            return Err(invalid(format!(
                "resident adaptive view has {view_row_count} rows, exceeding tail cap {tail_max_bars}",
            )));
        }
        let (view_kind, view_start) = match view.kind() {
            PopulationViewKindV1::Full => (0, 0),
            PopulationViewKindV1::ContiguousRange => (
                1,
                view.range()
                    .ok_or_else(|| invalid("resident adaptive range lost its exact start"))?
                    .start,
            ),
            PopulationViewKindV1::OrderedIndices => {
                return Err(invalid(
                    "resident adaptive producer V1 supports only full/contiguous views",
                ));
            }
        };
        Ok(Self {
            abi_version: ABI_VERSION,
            view_kind,
            parent_row_count: u64::try_from(view.parent_row_count).map_err(|_| {
                invalid("resident adaptive parent row count does not fit the native ABI")
            })?,
            view_start: u64::try_from(view_start)
                .map_err(|_| invalid("resident adaptive view start does not fit the native ABI"))?,
            view_row_count: u64::try_from(view_row_count).map_err(|_| {
                invalid("resident adaptive view row count does not fit the native ABI")
            })?,
            vol_window: Self::VOL_WINDOW_V1,
            vol_horizon_bars: Self::VOL_HORIZON_BARS_V1,
            tail_window: Self::TAIL_WINDOW_V1,
            tail_quantile_index: Self::TAIL_QUANTILE_INDEX_V1,
            tail_step: u64::try_from(tail_step)
                .map_err(|_| invalid("resident adaptive tail step does not fit the native ABI"))?,
            tail_max_bars: u64::try_from(tail_max_bars)
                .map_err(|_| invalid("resident adaptive tail cap does not fit the native ABI"))?,
            pip_size,
            stop_k_vol: Self::STOP_K_VOL_V1,
            stop_k_tail: Self::STOP_K_TAIL_V1,
            meta_label_min_dist: Self::META_LABEL_MIN_DIST_V1,
        })
    }

    pub const fn parent_row_count(self) -> u64 {
        self.parent_row_count
    }

    pub const fn view_start(self) -> u64 {
        self.view_start
    }

    pub const fn view_row_count(self) -> u64 {
        self.view_row_count
    }

    pub const fn tail_step(self) -> u64 {
        self.tail_step
    }

    pub const fn tail_max_bars(self) -> u64 {
        self.tail_max_bars
    }

    pub const fn pip_size(self) -> f64 {
        self.pip_size
    }

    pub fn identity_sha256(self) -> [u8; 32] {
        hash_resident_adaptive_base_request_v1(self)
    }
}

/// Device-resident proof that one exact view has a canonical adaptive base
/// queued before every later population read on the admitted stream.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentAdaptiveBaseViewTokenV1 {
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    request_identity_sha256: [u8; 32],
    token_identity_sha256: [u8; 32],
}

/// Copyable, non-authorizing receipt evidence for a resident adaptive view.
/// Authorization requires the original non-Clone token to be passed directly
/// to a purpose-bound validator while it is borrowed from its owning session.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentAdaptiveBaseViewTokenIdentityV1 {
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    request_identity_sha256: [u8; 32],
    token_identity_sha256: [u8; 32],
}

#[cfg(feature = "cuda")]
impl ResidentAdaptiveBaseViewTokenIdentityV1 {
    pub const fn resident_session_identity_sha256(self) -> [u8; 32] {
        self.resident_session_identity_sha256
    }

    pub const fn view_identity_sha256(self) -> [u8; 32] {
        self.view_identity_sha256
    }

    pub const fn request_identity_sha256(self) -> [u8; 32] {
        self.request_identity_sha256
    }

    pub const fn token_identity_sha256(self) -> [u8; 32] {
        self.token_identity_sha256
    }
}

impl ResidentAdaptiveBaseViewTokenV1 {
    pub const fn resident_session_identity_sha256(&self) -> [u8; 32] {
        self.resident_session_identity_sha256
    }

    pub const fn view_identity_sha256(&self) -> [u8; 32] {
        self.view_identity_sha256
    }

    pub const fn request_identity_sha256(&self) -> [u8; 32] {
        self.request_identity_sha256
    }

    pub const fn token_identity_sha256(&self) -> [u8; 32] {
        self.token_identity_sha256
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn identity_facts_v1(&self) -> ResidentAdaptiveBaseViewTokenIdentityV1 {
        ResidentAdaptiveBaseViewTokenIdentityV1 {
            resident_session_identity_sha256: self.resident_session_identity_sha256,
            view_identity_sha256: self.view_identity_sha256,
            request_identity_sha256: self.request_identity_sha256,
            token_identity_sha256: self.token_identity_sha256,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PopulationResidencyCountersV1 {
    parent_upload_count: u64,
    parent_upload_bytes: u64,
    view_binding_count: u64,
    full_binding_count: u64,
    range_binding_count: u64,
    ordered_binding_count: u64,
    ordered_index_upload_bytes: u64,
    adaptive_upload_bytes: u64,
    stream_creation_count: u64,
    explicit_synchronization_count: u64,
    metric_rows_readback_count: u64,
    metric_rows_readback_rows: u64,
    metric_rows_readback_bytes: u64,
    diagnostic_readback_count: u64,
    diagnostic_readback_rows: u64,
    diagnostic_readback_bytes: u64,
    accepted_trade_total_readback_count: u64,
    accepted_trade_total_readback_bytes: u64,
}

impl PopulationResidencyCountersV1 {
    pub const fn parent_upload_count(self) -> u64 {
        self.parent_upload_count
    }
    pub const fn parent_upload_bytes(self) -> u64 {
        self.parent_upload_bytes
    }
    pub const fn view_binding_count(self) -> u64 {
        self.view_binding_count
    }
    pub const fn full_binding_count(self) -> u64 {
        self.full_binding_count
    }
    pub const fn range_binding_count(self) -> u64 {
        self.range_binding_count
    }
    pub const fn ordered_binding_count(self) -> u64 {
        self.ordered_binding_count
    }
    pub const fn ordered_index_upload_bytes(self) -> u64 {
        self.ordered_index_upload_bytes
    }
    pub const fn adaptive_upload_bytes(self) -> u64 {
        self.adaptive_upload_bytes
    }
    pub const fn stream_creation_count(self) -> u64 {
        self.stream_creation_count
    }
    pub const fn explicit_synchronization_count(self) -> u64 {
        self.explicit_synchronization_count
    }
    pub const fn metric_rows_readback_count(self) -> u64 {
        self.metric_rows_readback_count
    }
    pub const fn metric_rows_readback_rows(self) -> u64 {
        self.metric_rows_readback_rows
    }
    pub const fn metric_rows_readback_bytes(self) -> u64 {
        self.metric_rows_readback_bytes
    }
    pub const fn diagnostic_readback_count(self) -> u64 {
        self.diagnostic_readback_count
    }
    pub const fn diagnostic_readback_rows(self) -> u64 {
        self.diagnostic_readback_rows
    }
    pub const fn diagnostic_readback_bytes(self) -> u64 {
        self.diagnostic_readback_bytes
    }
    pub const fn accepted_trade_total_readback_count(self) -> u64 {
        self.accepted_trade_total_readback_count
    }
    pub const fn accepted_trade_total_readback_bytes(self) -> u64 {
        self.accepted_trade_total_readback_bytes
    }
}

/// Exact physical CUDA device selected by one native population session.
/// Fields are private so callers cannot construct hardware evidence; the value
/// can only be read from the already-created native session.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPopulationDeviceIdentityV1 {
    selected_device_ordinal: u32,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    total_global_memory_bytes: u64,
    pci_domain_id: i32,
    pci_bus_id: i32,
    pci_device_id: i32,
    uuid: [u8; 16],
    name: [u8; 256],
}

impl Default for CudaPopulationDeviceIdentityV1 {
    fn default() -> Self {
        Self {
            selected_device_ordinal: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            multiprocessor_count: 0,
            total_global_memory_bytes: 0,
            pci_domain_id: 0,
            pci_bus_id: 0,
            pci_device_id: 0,
            uuid: [0; 16],
            name: [0; 256],
        }
    }
}

impl CudaPopulationDeviceIdentityV1 {
    pub const fn selected_device_ordinal(self) -> u32 {
        self.selected_device_ordinal
    }

    pub const fn compute_capability_major(self) -> u32 {
        self.compute_capability_major
    }

    pub const fn compute_capability_minor(self) -> u32 {
        self.compute_capability_minor
    }

    pub const fn multiprocessor_count(self) -> u32 {
        self.multiprocessor_count
    }

    pub const fn total_global_memory_bytes(self) -> u64 {
        self.total_global_memory_bytes
    }

    pub const fn pci_domain_id(self) -> i32 {
        self.pci_domain_id
    }

    pub const fn pci_bus_id(self) -> i32 {
        self.pci_bus_id
    }

    pub const fn pci_device_id(self) -> i32 {
        self.pci_device_id
    }

    pub const fn uuid(&self) -> &[u8; 16] {
        &self.uuid
    }

    pub fn name_bytes(&self) -> &[u8] {
        let length = self
            .name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(self.name.len());
        &self.name[..length]
    }
}

/// Borrowed canonical gene batch. CSR and SMC signal values stay f64 through
/// the native upload and device arithmetic.
#[derive(Debug, Clone, Copy)]
pub struct PopulationGeneView<'a> {
    pub descriptors: &'a [GeneDescriptor],
    pub offsets: &'a [i32],
    pub indices: &'a [i32],
    pub weights: &'a [f64],
    pub stop_pips: &'a [f64],
    pub target_pips: &'a [f64],
    pub stop_vol_multipliers: &'a [f64],
    /// Row-major `[candidate][slot]` gene SMC flags.
    pub smc_flags: &'a [i8],
    pub smc_weights: &'a [f64; SMC_SLOTS],
    pub gate_threshold: f64,
    pub smc_gate_disabled: bool,
}

impl PopulationGeneView<'_> {
    fn validate(&self, feature_count: usize) -> Result<usize, CudaPopulationError> {
        let population = self.descriptors.len();
        if population == 0 {
            return Err(invalid("gene batch is empty"));
        }
        for (field, actual) in [
            ("stop_pips", self.stop_pips.len()),
            ("target_pips", self.target_pips.len()),
            ("stop_vol_multipliers", self.stop_vol_multipliers.len()),
        ] {
            if actual != population {
                return Err(invalid(format!(
                    "gene {field} length {actual} does not match population {population}"
                )));
            }
        }
        if self.smc_flags.len() != population * SMC_SLOTS {
            return Err(invalid(format!(
                "gene smc_flags length {} does not match {population} x {SMC_SLOTS}",
                self.smc_flags.len()
            )));
        }
        if self.offsets.len() != population + 1 {
            return Err(invalid(format!(
                "gene offsets length {} does not match population {population} + 1",
                self.offsets.len()
            )));
        }
        if self.indices.len() != self.weights.len() {
            return Err(invalid("gene indices and weights lengths differ"));
        }
        if self.offsets.first().copied() != Some(0)
            || self.offsets.last().copied() != Some(self.indices.len() as i32)
            || self
                .offsets
                .windows(2)
                .any(|window| window[0] < 0 || window[0] > window[1])
        {
            return Err(invalid("gene CSR offsets are not monotonic and complete"));
        }
        if let Some((position, index)) = self
            .indices
            .iter()
            .copied()
            .enumerate()
            .find(|(_, index)| *index < 0 || *index as usize >= feature_count)
        {
            return Err(invalid(format!(
                "gene term {position} references feature {index}, outside 0..{feature_count}"
            )));
        }
        Ok(population)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PopulationDiagnostics {
    pub events: Vec<NeoPopulationEvent>,
    pub outcomes: Vec<NeoPopulationOutcome>,
}

/// Checked device-memory extent for the strict metrics-only population mode.
/// Fields are private so callers cannot mint a plan independently of session
/// extents; the resident handle exposes the plan only after native parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PopulationMetricsOnlyPlanV1 {
    scenario_count: u64,
    month_capacity: u64,
    metric_rows_bytes: u64,
    monthly_pnls_bytes: u64,
    month_start_equities_bytes: u64,
    scenario_descriptor_bytes: u64,
    total_device_bytes: u64,
    outcome_bytes: u64,
    accepted_trade_total_bytes: u64,
}

impl PopulationMetricsOnlyPlanV1 {
    pub fn checked_from_session_extents_v1(
        scenario_count: usize,
        month_capacity: u32,
    ) -> Result<Self, CudaPopulationError> {
        let scenario_count = u64::try_from(scenario_count)
            .map_err(|_| invalid("scenario count does not fit the metrics-only u64 plan"))?;
        let month_capacity = u64::from(month_capacity);
        if scenario_count == 0 || month_capacity == 0 {
            return Err(invalid(
                "metrics-only plan requires non-zero scenarios and month capacity",
            ));
        }
        let metric_rows_bytes = scenario_count
            .checked_mul(POPULATION_METRIC_ROW_BYTES_V1)
            .ok_or_else(|| invalid("metrics-only metric-row bytes overflow u64"))?;
        let monthly_elements = scenario_count
            .checked_mul(month_capacity)
            .ok_or_else(|| invalid("metrics-only monthly element count overflows u64"))?;
        let monthly_pnls_bytes = monthly_elements
            .checked_mul(POPULATION_F64_BYTES_V1)
            .ok_or_else(|| invalid("metrics-only monthly PnL bytes overflow u64"))?;
        let month_start_equities_bytes = monthly_elements
            .checked_mul(POPULATION_F64_BYTES_V1)
            .ok_or_else(|| invalid("metrics-only month-start bytes overflow u64"))?;
        let scenario_descriptor_bytes = scenario_count
            .checked_mul(POPULATION_SCENARIO_DEVICE_BYTES_V1)
            .ok_or_else(|| invalid("metrics-only scenario descriptor bytes overflow u64"))?;
        let total_device_bytes = metric_rows_bytes
            .checked_add(monthly_pnls_bytes)
            .and_then(|total| total.checked_add(month_start_equities_bytes))
            .and_then(|total| total.checked_add(scenario_descriptor_bytes))
            .ok_or_else(|| invalid("metrics-only total device bytes overflow u64"))?;
        Ok(Self {
            scenario_count,
            month_capacity,
            metric_rows_bytes,
            monthly_pnls_bytes,
            month_start_equities_bytes,
            scenario_descriptor_bytes,
            total_device_bytes,
            outcome_bytes: 0,
            accepted_trade_total_bytes: 0,
        })
    }

    pub const fn scenario_count(self) -> u64 {
        self.scenario_count
    }

    pub const fn month_capacity(self) -> u64 {
        self.month_capacity
    }

    pub const fn metric_rows_bytes(self) -> u64 {
        self.metric_rows_bytes
    }

    pub const fn monthly_pnls_bytes(self) -> u64 {
        self.monthly_pnls_bytes
    }

    pub const fn month_start_equities_bytes(self) -> u64 {
        self.month_start_equities_bytes
    }

    pub const fn scenario_descriptor_bytes(self) -> u64 {
        self.scenario_descriptor_bytes
    }

    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }

    pub const fn outcome_bytes(self) -> u64 {
        self.outcome_bytes
    }

    pub const fn accepted_trade_total_bytes(self) -> u64 {
        self.accepted_trade_total_bytes
    }
}

/// Checked allocation plan for the immutable parent owned by a strict V1
/// population session.
///
/// This is allocation memory, not upload traffic. In addition to the copied
/// parent arrays, the native session always reserves one full-parent
/// `view_indices`, `adaptive_base_pips`, and `gap_flags` array. Keeping those
/// three arrays in this plan prevents a caller from reproducing the older
/// `(8 * features + 68) * rows` undercharge; the exact native allocation is
/// `(8 * features + 76) * rows` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PopulationParentDevicePlanV1 {
    parent_rows: u64,
    feature_count: u64,
    copied_parent_bytes: u64,
    view_indices_bytes: u64,
    adaptive_base_pips_bytes: u64,
    gap_flags_bytes: u64,
    total_device_bytes: u64,
}

impl PopulationParentDevicePlanV1 {
    pub fn checked_from_parent_extents_v1(
        parent_rows: usize,
        feature_count: usize,
    ) -> Result<Self, CudaPopulationError> {
        let parent_rows = u64::try_from(parent_rows)
            .map_err(|_| invalid("parent rows do not fit the strict device plan"))?;
        let feature_count = u64::try_from(feature_count)
            .map_err(|_| invalid("feature count does not fit the strict device plan"))?;
        if parent_rows == 0 || feature_count == 0 {
            return Err(invalid(
                "strict parent device plan requires non-zero rows and features",
            ));
        }

        // close/high/low + months/days/timestamps + SMC rows + indicators.
        let copied_fixed_per_row = 3u64
            .checked_mul(POPULATION_F64_BYTES_V1)
            .and_then(|bytes| bytes.checked_add(3 * 8))
            .and_then(|bytes| bytes.checked_add(SMC_SLOTS as u64))
            .ok_or_else(|| invalid("strict parent fixed bytes overflow u64"))?;
        let indicator_bytes_per_row = feature_count
            .checked_mul(POPULATION_F64_BYTES_V1)
            .ok_or_else(|| invalid("strict parent indicator bytes overflow u64"))?;
        let copied_parent_bytes = parent_rows
            .checked_mul(
                copied_fixed_per_row
                    .checked_add(indicator_bytes_per_row)
                    .ok_or_else(|| invalid("strict parent copied bytes overflow u64"))?,
            )
            .ok_or_else(|| invalid("strict parent copied bytes overflow u64"))?;
        let view_indices_bytes = parent_rows
            .checked_mul(8)
            .ok_or_else(|| invalid("strict parent view-index bytes overflow u64"))?;
        let adaptive_base_pips_bytes = parent_rows
            .checked_mul(POPULATION_F64_BYTES_V1)
            .ok_or_else(|| invalid("strict parent adaptive bytes overflow u64"))?;
        let gap_flags_bytes = parent_rows;
        let total_device_bytes = copied_parent_bytes
            .checked_add(view_indices_bytes)
            .and_then(|total| total.checked_add(adaptive_base_pips_bytes))
            .and_then(|total| total.checked_add(gap_flags_bytes))
            .ok_or_else(|| invalid("strict parent total device bytes overflow u64"))?;

        Ok(Self {
            parent_rows,
            feature_count,
            copied_parent_bytes,
            view_indices_bytes,
            adaptive_base_pips_bytes,
            gap_flags_bytes,
            total_device_bytes,
        })
    }

    pub const fn parent_rows(self) -> u64 {
        self.parent_rows
    }

    pub const fn feature_count(self) -> u64 {
        self.feature_count
    }

    pub const fn copied_parent_bytes(self) -> u64 {
        self.copied_parent_bytes
    }

    pub const fn view_indices_bytes(self) -> u64 {
        self.view_indices_bytes
    }

    pub const fn adaptive_base_pips_bytes(self) -> u64 {
        self.adaptive_base_pips_bytes
    }

    pub const fn gap_flags_bytes(self) -> u64 {
        self.gap_flags_bytes
    }

    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }
}

/// Checked allocation plan for one unsplittable native gene upload.
///
/// Scenario splitting cannot reduce this allocation. The native layout is
/// exactly `63 * population + 12 * terms + 92` bytes: candidate IDs, CSR
/// offsets/indices/weights, five f64 scalar arrays, 11 SMC flags per gene, and
/// the one 11-f64 SMC weight vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PopulationGeneStorePlanV1 {
    candidate_count: u64,
    term_count: u64,
    total_device_bytes: u64,
}

impl PopulationGeneStorePlanV1 {
    pub fn checked_from_gene_extents_v1(
        candidate_count: usize,
        term_count: usize,
    ) -> Result<Self, CudaPopulationError> {
        let candidate_count = u64::try_from(candidate_count)
            .map_err(|_| invalid("gene count does not fit the strict device plan"))?;
        let term_count = u64::try_from(term_count)
            .map_err(|_| invalid("gene term count does not fit the strict device plan"))?;
        if candidate_count == 0 {
            return Err(invalid(
                "strict gene-store plan requires a non-zero candidate count",
            ));
        }
        let candidate_bytes = candidate_count
            .checked_mul(63)
            .ok_or_else(|| invalid("strict gene candidate bytes overflow u64"))?;
        let term_bytes = term_count
            .checked_mul(12)
            .ok_or_else(|| invalid("strict gene term bytes overflow u64"))?;
        let total_device_bytes = candidate_bytes
            .checked_add(term_bytes)
            .and_then(|total| total.checked_add(92))
            .ok_or_else(|| invalid("strict gene total device bytes overflow u64"))?;
        Ok(Self {
            candidate_count,
            term_count,
            total_device_bytes,
        })
    }

    pub const fn candidate_count(self) -> u64 {
        self.candidate_count
    }

    pub const fn term_count(self) -> u64 {
        self.term_count
    }

    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }
}

#[repr(C)]
struct RawDatasetView {
    header: DatasetHeader,
    close: *const f64,
    high: *const f64,
    low: *const f64,
    indicators: *const f64,
    months: *const i64,
    days: *const i64,
    timestamps: *const i64,
    smc_rows: *const i8,
    adaptive_base_pips: *const f64,
    adaptive_base_pips_len: usize,
}

#[repr(C)]
struct RawParentDatasetV1 {
    header: DatasetHeader,
    close: *const f64,
    high: *const f64,
    low: *const f64,
    indicators_feature_major: *const f64,
    months: *const i64,
    days: *const i64,
    timestamps: *const i64,
    smc_rows: *const i8,
}

/// Crate-private immediate FFI view of one already-sealed V3 resident store.
/// Every pointer is derived from a gpu-cuda-owned allocation and is retained by
/// `ResidentFeatureStoreImportV3`; no caller can construct this descriptor.
#[repr(C)]
#[cfg(feature = "cuda")]
pub(crate) struct RawResidentFeatureStoreBindV3 {
    pub(crate) abi_version: u32,
    pub(crate) selected_device_ordinal: u32,
    pub(crate) row_count: u64,
    pub(crate) feature_count: u32,
    pub(crate) smc_slots: u32,
    pub(crate) compute_capability_major: u16,
    pub(crate) compute_capability_minor: u16,
    pub(crate) reserved: u32,
    pub(crate) packed_validity_bytes: u64,
    pub(crate) close: *const f64,
    pub(crate) high: *const f64,
    pub(crate) low: *const f64,
    pub(crate) indicators_bar_major: *const f64,
    pub(crate) indicators_validity_u4: *const u8,
    pub(crate) months: *const i64,
    pub(crate) days: *const i64,
    pub(crate) timestamps: *const i64,
    pub(crate) smc_rows: *const i8,
    pub(crate) admitted_primary_context: *mut c_void,
    pub(crate) admitted_run_stream: *mut c_void,
    pub(crate) ready_event: *mut c_void,
    pub(crate) device_uuid: [u8; 16],
    pub(crate) admission_identity_sha256: [u8; 32],
    pub(crate) canonical_content_merkle: [u8; 32],
    pub(crate) allocator_context_reserve_bytes: u64,
    pub(crate) run_stream_process_token_v3: [u8; 32],
}

#[repr(C)]
struct RawEvaluationViewV1 {
    abi_version: u32,
    view_kind: u32,
    parent_row_count: u64,
    range_start: u64,
    row_count: u64,
    ordered_indices: *const u64,
    ordered_index_count: usize,
    timestamp_mode: u32,
    adaptive_base_pips: *const f64,
    adaptive_base_pips_len: usize,
}

#[repr(C)]
struct RawGeneView {
    descriptors: *const GeneDescriptor,
    count: usize,
    offsets: *const i32,
    indices: *const i32,
    weights: *const f64,
    term_count: usize,
    stop_pips: *const f64,
    target_pips: *const f64,
    stop_vol_multipliers: *const f64,
    smc_flags: *const i8,
    smc_weights: *const f64,
    gate_threshold: f64,
    smc_gate_disabled: u32,
}

#[repr(C)]
struct RawScenarioView {
    descriptors: *const ScenarioDescriptor,
    count: usize,
}

#[repr(C)]
struct RawReadback {
    rows: *mut NeoPopulationMetricRow,
    capacity: usize,
    written: *mut usize,
}

#[repr(C)]
struct RawDiagnosticReadback {
    events: *mut NeoPopulationEvent,
    outcomes: *mut NeoPopulationOutcome,
    capacity: usize,
    written: *mut usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RawResidentPopulationMetricsHandleV1 {
    abi_version: u32,
    reserved: u32,
    event_id: u64,
    scenario_count: u64,
    month_capacity: u64,
    metric_rows_bytes: u64,
    monthly_pnls_bytes: u64,
    month_start_equities_bytes: u64,
    scenario_descriptor_bytes: u64,
    total_device_bytes: u64,
    outcome_bytes: u64,
    accepted_trade_total_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentScoringPopulationSourceV2 {
    abi_version: u32,
    selected_cuda_ordinal: u32,
    admitted_run_stream: *mut c_void,
    metrics_ready_event: *mut c_void,
    scoring_ready_event: *mut c_void,
    receipt_token: *const c_void,
    population_lifetime_owner: *mut c_void,
    metric_rows_device: *const NeoPopulationMetricRow,
    expected_scenario_ids_device: *const u64,
    logical_population_count: u64,
    feature_count: u64,
    max_terms_per_gene: u32,
    reserved: u32,
    full_discovery_reserve_bytes: u64,
}

impl Default for RawResidentScoringPopulationSourceV2 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            selected_cuda_ordinal: 0,
            admitted_run_stream: std::ptr::null_mut(),
            metrics_ready_event: std::ptr::null_mut(),
            scoring_ready_event: std::ptr::null_mut(),
            receipt_token: std::ptr::null(),
            population_lifetime_owner: std::ptr::null_mut(),
            metric_rows_device: std::ptr::null(),
            expected_scenario_ids_device: std::ptr::null(),
            logical_population_count: 0,
            feature_count: 0,
            max_terms_per_gene: 0,
            reserved: 0,
            full_discovery_reserve_bytes: 0,
        }
    }
}

const _: [(); 96] = [(); std::mem::size_of::<RawResidentScoringPopulationSourceV2>()];

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RawTerminalCompactPopulationResultV1 {
    abi_version: u32,
    reserved: u32,
    event_id: u64,
    scenario_count: u64,
    metric_row: NeoPopulationMetricRow,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_rows: u64,
    terminal_readback_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RawHostPopulationMetricsResultV1 {
    abi_version: u32,
    reserved: u32,
    event_id: u64,
    scenario_count: u64,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_rows: u64,
    terminal_readback_bytes: u64,
}

fn validate_exact_resident_receipt_v1(
    receipt: &RawResidentPopulationMetricsHandleV1,
    plan: PopulationMetricsOnlyPlanV1,
) -> Result<(), CudaPopulationError> {
    let exact = receipt.abi_version == ABI_VERSION
        && receipt.reserved == 0
        && receipt.event_id != 0
        && receipt.scenario_count == plan.scenario_count()
        && receipt.month_capacity == plan.month_capacity()
        && receipt.metric_rows_bytes == plan.metric_rows_bytes()
        && receipt.monthly_pnls_bytes == plan.monthly_pnls_bytes()
        && receipt.month_start_equities_bytes == plan.month_start_equities_bytes()
        && receipt.scenario_descriptor_bytes == plan.scenario_descriptor_bytes()
        && receipt.total_device_bytes == plan.total_device_bytes()
        && receipt.outcome_bytes == plan.outcome_bytes()
        && receipt.accepted_trade_total_bytes == plan.accepted_trade_total_bytes();
    if !exact {
        return Err(invalid(format!(
            "native resident metrics receipt does not match the exact checked plan: \
             receipt={receipt:?}, plan={plan:?}"
        )));
    }
    Ok(())
}

fn hash_length_v1(hasher: &mut Sha256, length: usize) {
    hasher.update((length as u128).to_le_bytes());
}

fn hash_f64_v1(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

#[cfg(feature = "cuda")]
fn hash_resident_population_session_identity_v3(
    resident: &RawResidentFeatureStoreBindV3,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.resident-session.v3");
    hasher.update(resident.selected_device_ordinal.to_le_bytes());
    hasher.update(resident.row_count.to_le_bytes());
    hasher.update(resident.feature_count.to_le_bytes());
    hasher.update(resident.smc_slots.to_le_bytes());
    hasher.update(resident.compute_capability_major.to_le_bytes());
    hasher.update(resident.compute_capability_minor.to_le_bytes());
    hasher.update(resident.device_uuid);
    hasher.update(resident.admission_identity_sha256);
    hasher.update(resident.canonical_content_merkle);
    hasher.update(resident.allocator_context_reserve_bytes.to_le_bytes());
    hasher.update(resident.run_stream_process_token_v3);
    hasher.finalize().into()
}

#[cfg(feature = "cuda")]
fn hash_native_build_identity_v3(resident: &RawResidentFeatureStoreBindV3) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.native-build-binding.v3");
    // The one-shot admission identity already seals the exact gpu-cuda artifact,
    // NVCC/SASS target, driver/context API versions and math authority. Device
    // identity is repeated here so this build binding cannot be detached from
    // the card on which the native session was admitted.
    hasher.update(resident.admission_identity_sha256);
    hasher.update(resident.device_uuid);
    hasher.update(resident.compute_capability_major.to_le_bytes());
    hasher.update(resident.compute_capability_minor.to_le_bytes());
    hasher.finalize().into()
}

fn hash_population_view_identity_v1(view: &PopulationEvaluationViewV1) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.view.v1");
    hash_length_v1(&mut hasher, view.parent_row_count);
    hasher.update([match view.kind {
        PopulationViewKindV1::Full => 0,
        PopulationViewKindV1::ContiguousRange => 1,
        PopulationViewKindV1::OrderedIndices => 2,
    }]);
    if let Some(range) = &view.range {
        hasher.update([1]);
        hash_length_v1(&mut hasher, range.start);
        hash_length_v1(&mut hasher, range.end);
    } else {
        hasher.update([0]);
    }
    if let Some(indices) = &view.ordered_indices {
        hasher.update([1]);
        hash_length_v1(&mut hasher, indices.len());
        for index in indices.iter() {
            hasher.update(index.to_le_bytes());
        }
    } else {
        hasher.update([0]);
    }
    hasher.update([match view.timestamp_mode {
        PopulationTimestampModeV1::Canonical => 0,
        PopulationTimestampModeV1::DisabledIndexDelta => 1,
    }]);
    if let Some(values) = &view.adaptive_base_pips {
        hasher.update([1]);
        hash_length_v1(&mut hasher, values.len());
        for value in values.iter().copied() {
            hash_f64_v1(&mut hasher, value);
        }
    } else {
        hasher.update([0]);
    }
    hasher.finalize().into()
}

fn hash_resident_adaptive_base_request_v1(request: ResidentAdaptiveBaseRequestV1) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.resident-adaptive-base-request.v1");
    hasher.update(RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1.as_bytes());
    for value in [
        request.abi_version,
        request.view_kind,
        request.vol_window,
        request.vol_horizon_bars,
        request.tail_window,
        request.tail_quantile_index,
    ] {
        hasher.update(value.to_le_bytes());
    }
    for value in [
        request.parent_row_count,
        request.view_start,
        request.view_row_count,
        request.tail_step,
        request.tail_max_bars,
    ] {
        hasher.update(value.to_le_bytes());
    }
    for value in [
        request.pip_size,
        request.stop_k_vol,
        request.stop_k_tail,
        request.meta_label_min_dist,
    ] {
        hash_f64_v1(&mut hasher, value);
    }
    hasher.finalize().into()
}

#[cfg(feature = "cuda")]
fn hash_resident_adaptive_view_token_v1(
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    request_identity_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.resident-adaptive-view-token.v1");
    hasher.update(resident_session_identity_sha256);
    hasher.update(view_identity_sha256);
    hasher.update(request_identity_sha256);
    hasher.finalize().into()
}

#[cfg(feature = "cuda")]
fn hash_resident_adaptive_population_view_identity_v1(
    base_view_identity_sha256: [u8; 32],
    request_identity_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.resident-adaptive-view.v1");
    hasher.update(base_view_identity_sha256);
    hasher.update(request_identity_sha256);
    hasher.finalize().into()
}

fn hash_population_gene_batch_identity_v1(genes: &PopulationGeneView<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.gene-batch.v1");
    hash_length_v1(&mut hasher, genes.descriptors.len());
    for descriptor in genes.descriptors {
        hasher.update(descriptor.candidate_id.to_le_bytes());
        hasher.update(descriptor.term_offset.to_le_bytes());
        hasher.update(descriptor.term_count.to_le_bytes());
        hash_f64_v1(&mut hasher, descriptor.long_threshold);
        hash_f64_v1(&mut hasher, descriptor.short_threshold);
        hasher.update(descriptor.stop_ticks.to_le_bytes());
        hasher.update(descriptor.target_ticks.to_le_bytes());
        hash_f64_v1(&mut hasher, descriptor.stop_vol_multiplier);
        hasher.update(descriptor.flags.to_le_bytes());
        hasher.update(descriptor.reserved.to_le_bytes());
    }
    hash_length_v1(&mut hasher, genes.offsets.len());
    for value in genes.offsets {
        hasher.update(value.to_le_bytes());
    }
    hash_length_v1(&mut hasher, genes.indices.len());
    for value in genes.indices {
        hasher.update(value.to_le_bytes());
    }
    hash_length_v1(&mut hasher, genes.weights.len());
    for value in genes.weights.iter().copied() {
        hash_f64_v1(&mut hasher, value);
    }
    for values in [
        genes.stop_pips,
        genes.target_pips,
        genes.stop_vol_multipliers,
    ] {
        hash_length_v1(&mut hasher, values.len());
        for value in values.iter().copied() {
            hash_f64_v1(&mut hasher, value);
        }
    }
    hash_length_v1(&mut hasher, genes.smc_flags.len());
    for value in genes.smc_flags {
        hasher.update(value.to_le_bytes());
    }
    for value in genes.smc_weights.iter().copied() {
        hash_f64_v1(&mut hasher, value);
    }
    hash_f64_v1(&mut hasher, genes.gate_threshold);
    hasher.update([u8::from(genes.smc_gate_disabled)]);
    hasher.finalize().into()
}

fn hash_population_scenario_batch_identity_v1(scenarios: &[ScenarioDescriptor]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.scenario-batch.v1");
    hash_length_v1(&mut hasher, scenarios.len());
    for scenario in scenarios {
        hasher.update(scenario.base_candidate_id.to_le_bytes());
        hasher.update(scenario.scenario_id.to_le_bytes());
        hasher.update(scenario.rng_counter.to_le_bytes());
        hasher.update(scenario.window_offset.to_le_bytes());
        hasher.update(scenario.window_len.to_le_bytes());
        hasher.update(scenario.scenario_type.to_le_bytes());
        hasher.update(scenario.spread_ticks.to_le_bytes());
        hasher.update(scenario.slippage_ticks.to_le_bytes());
        hasher.update(scenario.commission_micros.to_le_bytes());
        hasher.update(scenario.perturbation_offset.to_le_bytes());
        hasher.update(scenario.perturbation_count.to_le_bytes());
        hasher.update(scenario.reserved.to_le_bytes());
    }
    hasher.finalize().into()
}

fn hash_population_settings_identity_v1(settings: &NeoPopulationSettings) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.settings.v1");
    for value in [
        settings.abi_version,
        settings.flags,
        settings.max_hold_bars,
        settings.min_hold_bars,
        settings.max_trades_per_day,
        settings.month_capacity,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update(settings.gap_threshold_ms.to_le_bytes());
    for value in [
        settings.initial_equity,
        settings.pip_value,
        settings.spread_pips,
        settings.commission_per_trade,
        settings.pip_value_per_lot,
        settings.swap_long_pips_per_day,
        settings.swap_short_pips_per_day,
        settings.pnl_conversion_fee_rate,
        settings.risk_per_trade_min,
        settings.risk_per_trade_max,
        settings.high_quality_confidence,
        settings.adaptive_rr,
    ] {
        hash_f64_v1(&mut hasher, value);
    }
    hasher.update(settings.trailing_enabled.to_le_bytes());
    hasher.update(settings._trailing_pad.to_le_bytes());
    for value in [
        settings.trailing_atr_multiplier,
        settings.trailing_be_trigger_r,
        settings.trailing_min_lock_pips,
        settings.spread_pips_asian,
        settings.spread_pips_overlap,
        settings.spread_pips_late_ny,
    ] {
        hash_f64_v1(&mut hasher, value);
    }
    hasher.finalize().into()
}

fn validate_terminal_compact_result_v1(
    raw: &RawTerminalCompactPopulationResultV1,
    expected_event_id: u64,
    expected_candidate_id: u64,
    expected_scenario_id: u64,
) -> Result<(), CudaPopulationError> {
    let exact = raw.abi_version == ABI_VERSION
        && raw.reserved == 0
        && raw.event_id == expected_event_id
        && raw.scenario_count == 1
        && raw.metric_row.candidate_id == expected_candidate_id
        && raw.metric_row.scenario_id == expected_scenario_id
        && raw.metric_row.values.iter().all(|value| value.is_finite())
        && raw.terminal_synchronization_count == 1
        && raw.terminal_readback_count == 1
        && raw.terminal_readback_rows == 1
        && raw.terminal_readback_bytes == POPULATION_METRIC_ROW_BYTES_V1;
    if !exact {
        return Err(invalid(format!(
            "native terminal compact result violated its sealed one-row contract: {raw:?}"
        )));
    }
    Ok(())
}

fn hash_terminal_compact_result_receipt_v1(
    raw: &RawTerminalCompactPopulationResultV1,
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    gene_batch_identity_sha256: [u8; 32],
    scenario_batch_identity_sha256: [u8; 32],
    settings_identity_sha256: [u8; 32],
    native_build_identity_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.terminal-compact-result.v1");
    hasher.update(resident_session_identity_sha256);
    hasher.update(view_identity_sha256);
    hasher.update(gene_batch_identity_sha256);
    hasher.update(scenario_batch_identity_sha256);
    hasher.update(settings_identity_sha256);
    hasher.update(native_build_identity_sha256);
    hasher.update(raw.event_id.to_le_bytes());
    hasher.update(raw.scenario_count.to_le_bytes());
    hasher.update(raw.metric_row.candidate_id.to_le_bytes());
    hasher.update(raw.metric_row.scenario_id.to_le_bytes());
    for value in raw.metric_row.values {
        hash_f64_v1(&mut hasher, value);
    }
    hasher.update(raw.terminal_synchronization_count.to_le_bytes());
    hasher.update(raw.terminal_readback_count.to_le_bytes());
    hasher.update(raw.terminal_readback_rows.to_le_bytes());
    hasher.update(raw.terminal_readback_bytes.to_le_bytes());
    hasher.finalize().into()
}

fn validate_host_population_metrics_result_v1(
    raw: &RawHostPopulationMetricsResultV1,
    written: usize,
    rows: &[NeoPopulationMetricRow],
    expected_event_id: u64,
    plan: PopulationMetricsOnlyPlanV1,
    expected_identities: &[(u64, u64)],
) -> Result<(), CudaPopulationError> {
    let expected_rows = usize::try_from(plan.scenario_count())
        .map_err(|_| invalid("metrics-only scenario count does not fit host usize"))?;
    let header_is_exact = raw.abi_version == ABI_VERSION
        && raw.reserved == 0
        && raw.event_id == expected_event_id
        && raw.scenario_count == plan.scenario_count()
        && raw.terminal_synchronization_count == 1
        && raw.terminal_readback_count == 1
        && raw.terminal_readback_rows == plan.scenario_count()
        && raw.terminal_readback_bytes == plan.metric_rows_bytes()
        && written == expected_rows
        && rows.len() == expected_rows
        && expected_identities.len() == expected_rows;
    if !header_is_exact {
        return Err(invalid(format!(
            "native host metrics result violated its sealed transfer contract: \
             result={raw:?}, written={written}, rows={}, expected_rows={expected_rows}",
            rows.len()
        )));
    }
    for (index, (row, expected)) in rows.iter().zip(expected_identities).enumerate() {
        if (row.candidate_id, row.scenario_id) != *expected
            || row.values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(format!(
                "host metric row {index} violated uploaded scenario order/identity or finiteness: \
                 got=({}, {}), expected={expected:?}",
                row.candidate_id, row.scenario_id
            )));
        }
    }
    Ok(())
}

fn hash_host_population_metrics_receipt_v1(
    raw: &RawHostPopulationMetricsResultV1,
    rows: &[NeoPopulationMetricRow],
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    gene_batch_identity_sha256: [u8; 32],
    scenario_batch_identity_sha256: [u8; 32],
    settings_identity_sha256: [u8; 32],
    native_build_identity_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.population.host-metrics-result.v1");
    hasher.update(resident_session_identity_sha256);
    hasher.update(view_identity_sha256);
    hasher.update(gene_batch_identity_sha256);
    hasher.update(scenario_batch_identity_sha256);
    hasher.update(settings_identity_sha256);
    hasher.update(native_build_identity_sha256);
    hasher.update(raw.event_id.to_le_bytes());
    hasher.update(raw.scenario_count.to_le_bytes());
    for row in rows {
        hasher.update(row.candidate_id.to_le_bytes());
        hasher.update(row.scenario_id.to_le_bytes());
        for value in row.values {
            hash_f64_v1(&mut hasher, value);
        }
    }
    hasher.update(raw.terminal_synchronization_count.to_le_bytes());
    hasher.update(raw.terminal_readback_count.to_le_bytes());
    hasher.update(raw.terminal_readback_rows.to_le_bytes());
    hasher.update(raw.terminal_readback_bytes.to_le_bytes());
    hasher.finalize().into()
}

fn strict_enqueue_failure_is_known_prelaunch_v1(status: i32) -> bool {
    matches!(
        status,
        STATUS_UNSUPPORTED
            | STATUS_NULL_SESSION
            | STATUS_ABI_MISMATCH
            | STATUS_INVALID_ARGUMENT
            | STATUS_DEVICE_UNAVAILABLE
            | STATUS_ALLOCATION_FAILED
            | STATUS_MISSING_UPLOAD
            | STATUS_DATASET_REUPLOAD
            | STATUS_WORKSPACE_MODE_MISMATCH
            | STATUS_WORKSPACE_PLAN_MISMATCH
    )
}

unsafe extern "C" {
    fn neoethos_gpu_cuda_population_create(
        abi_version: u32,
        device: i32,
        max_events: usize,
        status: *mut i32,
    ) -> *mut c_void;
    fn neoethos_gpu_cuda_population_upload_dataset(
        session: *mut c_void,
        dataset: *const RawDatasetView,
    ) -> i32;
    fn neoethos_gpu_cuda_population_upload_parent_v1(
        session: *mut c_void,
        parent: *const RawParentDatasetV1,
    ) -> i32;
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_bind_resident_feature_store_v3(
        resident: *const RawResidentFeatureStoreBindV3,
        status: *mut i32,
    ) -> *mut c_void;
    fn neoethos_gpu_cuda_population_bind_view_v1(
        session: *mut c_void,
        view: *const RawEvaluationViewV1,
    ) -> i32;
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1(
        session: *mut c_void,
        view: *const RawEvaluationViewV1,
        request: *const ResidentAdaptiveBaseRequestV1,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn neoethos_gpu_cuda_population_copy_resident_adaptive_base_fixture_v1(
        session: *mut c_void,
        host_values: *mut f64,
        value_count: usize,
    ) -> i32;
    fn neoethos_gpu_cuda_population_read_residency_counters_v1(
        session: *mut c_void,
        counters: *mut PopulationResidencyCountersV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_read_device_identity_v1(
        session: *mut c_void,
        identity: *mut CudaPopulationDeviceIdentityV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_upload_genes(
        session: *mut c_void,
        genes: *const RawGeneView,
    ) -> i32;
    fn neoethos_gpu_cuda_population_upload_scenarios(
        session: *mut c_void,
        scenarios: *const RawScenarioView,
    ) -> i32;
    #[allow(dead_code)] // Reached by the crate-private resident Search owner.
    fn neoethos_gpu_cuda_population_upload_resident_scenarios_v2(
        session: *mut c_void,
        scenarios: *const RawScenarioView,
        planned_population: u64,
    ) -> i32;
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(
        session: *mut c_void,
        genes: *const crate::resident_search_v2::RawResidentGenerationGeneViewV2,
        settings: *const NeoPopulationSettings,
        resident_metrics: *mut RawResidentPopulationMetricsHandleV1,
        counters: *mut NeoPopulationCounters,
    ) -> i32;
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_export_resident_scoring_source_v2(
        session: *mut c_void,
        resident_metrics: *const RawResidentPopulationMetricsHandleV1,
        expected_population: u64,
        expected_feature_count: u64,
        expected_max_terms: u32,
        source: *mut RawResidentScoringPopulationSourceV2,
    ) -> i32;
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_finish_resident_scoring_source_v2(
        session: *mut c_void,
        resident_metrics: *const RawResidentPopulationMetricsHandleV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(
        session: *mut c_void,
        settings: *const NeoPopulationSettings,
        resident_metrics: *mut RawResidentPopulationMetricsHandleV1,
        counters: *mut NeoPopulationCounters,
    ) -> i32;
    fn neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(
        session: *mut c_void,
        resident_metrics: *const RawResidentPopulationMetricsHandleV1,
        compact_result: *mut RawTerminalCompactPopulationResultV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_consume_host_metrics_v1(
        session: *mut c_void,
        resident_metrics: *const RawResidentPopulationMetricsHandleV1,
        readback: *mut RawReadback,
        result: *mut RawHostPopulationMetricsResultV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
        session: *mut c_void,
        resident_metrics: *const RawResidentPopulationMetricsHandleV1,
    ) -> i32;
    fn neoethos_gpu_cuda_population_b_evaluate(
        session: *mut c_void,
        settings: *const NeoPopulationSettings,
        event_id: *mut u64,
        counters: *mut NeoPopulationCounters,
    ) -> i32;
    fn neoethos_gpu_cuda_population_wait(session: *mut c_void, event_id: u64) -> i32;
    fn neoethos_gpu_cuda_population_read_metrics(
        session: *mut c_void,
        readback: *mut RawReadback,
    ) -> i32;
    fn neoethos_gpu_cuda_population_read_diagnostics(
        session: *mut c_void,
        readback: *mut RawDiagnosticReadback,
    ) -> i32;
    #[allow(dead_code)] // Compatibility ABI remains covered by the signature test below.
    fn neoethos_gpu_cuda_population_destroy(session: *mut c_void);
    #[cfg(feature = "cuda")]
    fn neoethos_gpu_cuda_population_destroy_terminal_checked_v2(session: *mut c_void) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrictResidentSessionStateV1 {
    StrictIdle,
    InFlight,
    Poisoned,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PopulationSessionDropPolicyV3 {
    DestroyWhenIdle,
    LeakUntilResidentConsumerEvent,
}

#[cfg(feature = "cuda-device-fixtures")]
static TERMINAL_SEARCH_SESSION_DESTROY_COUNT_V2: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "cuda-device-fixtures")]
pub(crate) fn terminal_search_session_destroy_count_fixture_v2() -> u64 {
    TERMINAL_SEARCH_SESSION_DESTROY_COUNT_V2.load(Ordering::SeqCst)
}

/// Owns exactly one native CUDA population session.
#[derive(Debug)]
pub struct PopulationSession {
    handle: *mut c_void,
    device: i32,
    max_events: usize,
    bars: usize,
    feature_count: usize,
    population: usize,
    /// Threads the walk launches and metric rows it returns.
    ///
    /// Equal to `population` for a caller that uploads one scenario per gene —
    /// which is what makes that case identical to the pre-scenario engine — and
    /// larger for a screen that asks for many treatments of the same genes.
    scenario_count: usize,
    emitted_events: usize,
    dataset_uploaded: bool,
    genes_uploaded: bool,
    scenarios_uploaded: bool,
    pending_event: Option<u64>,
    metrics_ready: bool,
    strict_resident_state: StrictResidentSessionStateV1,
    #[cfg(feature = "cuda")]
    drop_policy_v3: PopulationSessionDropPolicyV3,
    parent_source_v1: Option<PopulationParentDatasetV1>,
    resident_parent_shape_v3: Option<(usize, usize)>,
    bound_view_source_v1: Option<PopulationEvaluationViewV1>,
    resident_session_identity_sha256: Option<[u8; 32]>,
    native_build_identity_sha256: Option<[u8; 32]>,
    view_identity_sha256: Option<[u8; 32]>,
    resident_adaptive_base_view_token_v1: Option<ResidentAdaptiveBaseViewTokenV1>,
    gene_batch_identity_sha256: Option<[u8; 32]>,
    scenario_batch_identity_sha256: Option<[u8; 32]>,
    uploaded_candidate_ids: Vec<u64>,
    expected_scenario_identities: Arc<[(u64, u64)]>,
    terminal_scenario_identity: Option<(u64, u64)>,
}

/// Opaque evidence for the one bounded host-visible metric row that terminates
/// the current one-scenario strict resident V1 seam. Every authority field is
/// derived from the already-sealed session and exact uploaded inputs.
#[derive(Debug, PartialEq)]
pub struct TerminalCompactPopulationResultReceiptV1 {
    metric_row: NeoPopulationMetricRow,
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    gene_batch_identity_sha256: [u8; 32],
    scenario_batch_identity_sha256: [u8; 32],
    settings_identity_sha256: [u8; 32],
    native_build_identity_sha256: [u8; 32],
    event_id: u64,
    scenario_count: u64,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_rows: u64,
    terminal_readback_bytes: u64,
    receipt_identity_sha256: [u8; 32],
}

impl TerminalCompactPopulationResultReceiptV1 {
    pub const fn metric_row(&self) -> &NeoPopulationMetricRow {
        &self.metric_row
    }

    pub const fn resident_session_identity_sha256(&self) -> [u8; 32] {
        self.resident_session_identity_sha256
    }

    pub const fn view_identity_sha256(&self) -> [u8; 32] {
        self.view_identity_sha256
    }

    pub const fn gene_batch_identity_sha256(&self) -> [u8; 32] {
        self.gene_batch_identity_sha256
    }

    pub const fn scenario_batch_identity_sha256(&self) -> [u8; 32] {
        self.scenario_batch_identity_sha256
    }

    pub const fn settings_identity_sha256(&self) -> [u8; 32] {
        self.settings_identity_sha256
    }

    pub const fn native_build_identity_sha256(&self) -> [u8; 32] {
        self.native_build_identity_sha256
    }

    pub const fn event_id(&self) -> u64 {
        self.event_id
    }

    pub const fn scenario_count(&self) -> u64 {
        self.scenario_count
    }

    pub const fn terminal_synchronization_count(&self) -> u64 {
        self.terminal_synchronization_count
    }

    pub const fn terminal_readback_count(&self) -> u64 {
        self.terminal_readback_count
    }

    pub const fn terminal_readback_rows(&self) -> u64 {
        self.terminal_readback_rows
    }

    pub const fn terminal_readback_bytes(&self) -> u64 {
        self.terminal_readback_bytes
    }

    pub const fn receipt_identity_sha256(&self) -> [u8; 32] {
        self.receipt_identity_sha256
    }
}

/// Sealed host result for one strict metrics-only population launch. It owns
/// exactly the rows transferred by the single terminal D2H and binds them to
/// the run/session/view/gene/scenario/settings/build identities.
#[derive(Debug, PartialEq)]
pub struct HostPopulationMetricsReceiptV1 {
    metric_rows: Vec<NeoPopulationMetricRow>,
    counters: NeoPopulationCounters,
    resident_session_identity_sha256: [u8; 32],
    view_identity_sha256: [u8; 32],
    gene_batch_identity_sha256: [u8; 32],
    scenario_batch_identity_sha256: [u8; 32],
    settings_identity_sha256: [u8; 32],
    native_build_identity_sha256: [u8; 32],
    event_id: u64,
    scenario_count: u64,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_rows: u64,
    terminal_readback_bytes: u64,
    receipt_identity_sha256: [u8; 32],
}

impl HostPopulationMetricsReceiptV1 {
    pub const fn counters(&self) -> NeoPopulationCounters {
        self.counters
    }

    pub fn metric_rows(&self) -> &[NeoPopulationMetricRow] {
        &self.metric_rows
    }

    pub fn into_metric_rows(self) -> Vec<NeoPopulationMetricRow> {
        self.metric_rows
    }

    pub const fn event_id(&self) -> u64 {
        self.event_id
    }

    pub const fn scenario_count(&self) -> u64 {
        self.scenario_count
    }

    pub const fn terminal_synchronization_count(&self) -> u64 {
        self.terminal_synchronization_count
    }

    pub const fn terminal_readback_count(&self) -> u64 {
        self.terminal_readback_count
    }

    pub const fn terminal_readback_rows(&self) -> u64 {
        self.terminal_readback_rows
    }

    pub const fn terminal_readback_bytes(&self) -> u64 {
        self.terminal_readback_bytes
    }

    pub const fn receipt_identity_sha256(&self) -> [u8; 32] {
        self.receipt_identity_sha256
    }
}

#[must_use = "resident GPU metrics must be consumed by the next device stage"]
pub struct ResidentPopulationMetricsV1<'session> {
    session: &'session mut PopulationSession,
    receipt: Box<RawResidentPopulationMetricsHandleV1>,
    plan: PopulationMetricsOnlyPlanV1,
    resident_session_identity_sha256: Option<[u8; 32]>,
    view_identity_sha256: Option<[u8; 32]>,
    gene_batch_identity_sha256: Option<[u8; 32]>,
    scenario_batch_identity_sha256: Option<[u8; 32]>,
    settings_identity_sha256: [u8; 32],
    native_build_identity_sha256: Option<[u8; 32]>,
    terminal_scenario_identity: Option<(u64, u64)>,
    expected_scenario_identities: Arc<[(u64, u64)]>,
    counters: NeoPopulationCounters,
    consumed: bool,
}

/// Move-only population receipt retained across the asynchronous Search
/// completion boundary. Unlike the borrowed metrics facade, this owner carries
/// the complete session and cannot make it reusable before terminal proof.
#[cfg(feature = "cuda")]
pub(crate) struct ResidentSearchPopulationCompletionLeaseV2 {
    session: Option<PopulationSession>,
    receipt: Box<RawResidentPopulationMetricsHandleV1>,
    raw: RawResidentScoringPopulationSourceV2,
    #[cfg(feature = "cuda-device-fixtures")]
    counters: NeoPopulationCounters,
    consumed: bool,
}

#[cfg(feature = "cuda")]
impl ResidentSearchPopulationCompletionLeaseV2 {
    pub(crate) const fn raw_source_v2(&self) -> &RawResidentScoringPopulationSourceV2 {
        &self.raw
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) const fn counters_fixture_v2(&self) -> NeoPopulationCounters {
        self.counters
    }

    pub(crate) fn finish_device_consume_v2(
        mut self,
    ) -> Result<PopulationSession, CudaPopulationError> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("resident Search completion lease lost its session"))?;
        // SAFETY: the exact boxed receipt and its native session remain owned
        // here until the terminal completion event has been proven Ready.
        let status = unsafe {
            neoethos_gpu_cuda_population_finish_resident_scoring_source_v2(
                session.handle,
                self.receipt.as_ref(),
            )
        };
        if status != STATUS_OK {
            session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(CudaPopulationError::native(
                "finish_resident_scoring_source_v2",
                status,
            ));
        }
        session.strict_resident_state = StrictResidentSessionStateV1::StrictIdle;
        session.pending_event = None;
        session.metrics_ready = false;
        #[cfg(feature = "cuda")]
        session.authorize_resident_session_destroy_v3();
        self.consumed = true;
        self.session
            .take()
            .ok_or_else(|| invalid("resident Search completion lease lost its session"))
    }

    pub(crate) fn poison_without_reuse_v2(&mut self) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
        // SAFETY: abandonment receives the exact stable receipt. Native keeps
        // the session poisoned, so no in-flight state can be reused.
        let _ = unsafe {
            neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                session.handle,
                self.receipt.as_ref(),
            )
        };
        self.consumed = true;
    }
}

#[cfg(feature = "cuda")]
impl Drop for ResidentSearchPopulationCompletionLeaseV2 {
    fn drop(&mut self) {
        if !self.consumed {
            self.poison_without_reuse_v2();
        }
    }
}

impl<'session> ResidentPopulationMetricsV1<'session> {
    pub const fn plan(&self) -> PopulationMetricsOnlyPlanV1 {
        self.plan
    }

    pub fn selected_device_ordinal(&self) -> i32 {
        self.session.device()
    }

    pub fn resident_device_bytes(&self) -> u64 {
        self.receipt.total_device_bytes
    }

    pub fn consume_terminal_compact_result_v1(
        mut self,
    ) -> Result<TerminalCompactPopulationResultReceiptV1, CudaPopulationError> {
        if self.plan.scenario_count() != 1 {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(invalid(
                "terminal compact-result V1 requires exactly one resident scenario",
            ));
        }
        let (
            Some(resident_session_identity_sha256),
            Some(view_identity_sha256),
            Some(gene_batch_identity_sha256),
            Some(scenario_batch_identity_sha256),
            Some(native_build_identity_sha256),
            Some((expected_candidate_id, expected_scenario_id)),
        ) = (
            self.resident_session_identity_sha256,
            self.view_identity_sha256,
            self.gene_batch_identity_sha256,
            self.scenario_batch_identity_sha256,
            self.native_build_identity_sha256,
            self.terminal_scenario_identity,
        )
        else {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(invalid(
                "terminal compact-result V1 requires sealed resident session, view, gene, scenario and build identities",
            ));
        };
        let mut raw = RawTerminalCompactPopulationResultV1::default();
        // SAFETY: both fixed-width values are live for the duration of the call;
        // the native function synchronizes the already-recorded event and
        // copies exactly one metric row before returning.
        let status = unsafe {
            neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(
                self.session.handle,
                self.receipt.as_ref(),
                &mut raw,
            )
        };
        if status != STATUS_OK {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(CudaPopulationError::native(
                "consume_terminal_compact_result_v1",
                status,
            ));
        }
        if let Err(error) = validate_terminal_compact_result_v1(
            &raw,
            self.receipt.event_id,
            expected_candidate_id,
            expected_scenario_id,
        ) {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(error);
        }
        self.session.strict_resident_state = StrictResidentSessionStateV1::StrictIdle;
        self.session.pending_event = None;
        self.session.metrics_ready = false;
        self.consumed = true;
        let receipt_identity_sha256 = hash_terminal_compact_result_receipt_v1(
            &raw,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            self.settings_identity_sha256,
            native_build_identity_sha256,
        );
        Ok(TerminalCompactPopulationResultReceiptV1 {
            metric_row: raw.metric_row,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            settings_identity_sha256: self.settings_identity_sha256,
            native_build_identity_sha256,
            event_id: raw.event_id,
            scenario_count: raw.scenario_count,
            terminal_synchronization_count: raw.terminal_synchronization_count,
            terminal_readback_count: raw.terminal_readback_count,
            terminal_readback_rows: raw.terminal_readback_rows,
            terminal_readback_bytes: raw.terminal_readback_bytes,
            receipt_identity_sha256,
        })
    }

    /// Terminate one strict metrics-only launch with exactly one event wait and
    /// one bounded D2H containing every metric row in uploaded scenario order.
    pub fn consume_host_metrics_v1(
        mut self,
    ) -> Result<HostPopulationMetricsReceiptV1, CudaPopulationError> {
        let (
            Some(resident_session_identity_sha256),
            Some(view_identity_sha256),
            Some(gene_batch_identity_sha256),
            Some(scenario_batch_identity_sha256),
            Some(native_build_identity_sha256),
        ) = (
            self.resident_session_identity_sha256,
            self.view_identity_sha256,
            self.gene_batch_identity_sha256,
            self.scenario_batch_identity_sha256,
            self.native_build_identity_sha256,
        )
        else {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(invalid(
                "host metrics V1 requires sealed resident session, view, gene, scenario and build identities",
            ));
        };
        let row_count = usize::try_from(self.plan.scenario_count())
            .map_err(|_| invalid("metrics-only row count does not fit host usize"))?;
        if self.expected_scenario_identities.len() != row_count {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(invalid(
                "host metrics V1 lost the exact uploaded scenario identity order",
            ));
        }

        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count)
            .map_err(|_| invalid("host metric row allocation failed for checked plan"))?;
        rows.resize(row_count, NeoPopulationMetricRow::default());
        let mut written = 0usize;
        let mut readback = RawReadback {
            rows: rows.as_mut_ptr(),
            capacity: rows.len(),
            written: &mut written,
        };
        let mut raw = RawHostPopulationMetricsResultV1::default();
        // SAFETY: the boxed receipt has retained one stable address since the
        // enqueue; the output slice covers the exact checked plan and both
        // fixed-width out-parameters remain live for the complete call.
        let status = unsafe {
            neoethos_gpu_cuda_population_consume_host_metrics_v1(
                self.session.handle,
                self.receipt.as_ref(),
                &mut readback,
                &mut raw,
            )
        };
        if status != STATUS_OK {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(CudaPopulationError::native(
                "consume_host_metrics_v1",
                status,
            ));
        }
        if let Err(error) = validate_host_population_metrics_result_v1(
            &raw,
            written,
            &rows,
            self.receipt.event_id,
            self.plan,
            &self.expected_scenario_identities,
        ) {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(error);
        }

        let mut counters = self.counters;
        counters.synchronization_events = counters
            .synchronization_events
            .checked_add(raw.terminal_synchronization_count)
            .ok_or_else(|| invalid("population synchronization counter overflow"))?;
        counters.full_readback_bytes = counters
            .full_readback_bytes
            .checked_add(raw.terminal_readback_bytes)
            .ok_or_else(|| invalid("population metric readback byte counter overflow"))?;
        let receipt_identity_sha256 = hash_host_population_metrics_receipt_v1(
            &raw,
            &rows,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            self.settings_identity_sha256,
            native_build_identity_sha256,
        );

        self.session.strict_resident_state = StrictResidentSessionStateV1::StrictIdle;
        self.session.pending_event = None;
        self.session.metrics_ready = false;
        self.consumed = true;
        Ok(HostPopulationMetricsReceiptV1 {
            metric_rows: rows,
            counters,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            settings_identity_sha256: self.settings_identity_sha256,
            native_build_identity_sha256,
            event_id: raw.event_id,
            scenario_count: raw.scenario_count,
            terminal_synchronization_count: raw.terminal_synchronization_count,
            terminal_readback_count: raw.terminal_readback_count,
            terminal_readback_rows: raw.terminal_readback_rows,
            terminal_readback_bytes: raw.terminal_readback_bytes,
            receipt_identity_sha256,
        })
    }
}

impl Drop for ResidentPopulationMetricsV1<'_> {
    fn drop(&mut self) {
        if !self.consumed {
            self.session.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            // SAFETY: the session and boxed receipt are still live. Drop is the
            // final owner of this in-flight token, so native execution must be
            // poisoned rather than silently becoming reusable.
            let _ = unsafe {
                neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                    self.session.handle,
                    self.receipt.as_ref(),
                )
            };
        }
    }
}

impl PopulationSession {
    fn require_strict_idle_v1(&self, operation: &'static str) -> Result<(), CudaPopulationError> {
        match self.strict_resident_state {
            StrictResidentSessionStateV1::StrictIdle => Ok(()),
            StrictResidentSessionStateV1::InFlight => Err(CudaPopulationError::native(
                operation,
                STATUS_STRICT_RESIDENT_IN_FLIGHT,
            )),
            StrictResidentSessionStateV1::Poisoned => Err(CudaPopulationError::native(
                operation,
                STATUS_STRICT_RESIDENT_POISONED,
            )),
        }
    }

    #[allow(dead_code)] // First bounded Search ownership seam; no public raw handle.
    pub(crate) fn admit_resident_search_owner_v2(
        &mut self,
        expected_feature_count: usize,
    ) -> Result<*mut c_void, CudaPopulationError> {
        self.require_strict_idle_v1("begin_resident_search_v2")?;
        if self.handle.is_null() {
            return Err(invalid("resident Search V2 requires one live CUDA session"));
        }
        if !self.dataset_uploaded || self.feature_count != expected_feature_count {
            return Err(invalid(
                "resident Search V2 requires the exact uploaded dataset feature extent",
            ));
        }
        if self.genes_uploaded || self.scenarios_uploaded {
            return Err(invalid(
                "resident Search V2 admission must precede compatibility gene/scenario uploads",
            ));
        }
        Ok(self.handle)
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn resident_search_native_handle_v2(&self) -> *mut c_void {
        self.handle
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn poison_resident_search_owner_v2(&mut self) {
        self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
    }

    pub fn create(device: i32, max_events: usize) -> Result<Self, CudaPopulationError> {
        if max_events == 0 {
            return Err(invalid("max_events must be non-zero"));
        }
        if device < 0 {
            return Err(invalid("device index must be non-negative"));
        }
        let mut status = STATUS_OK;
        // SAFETY: `status` is a valid out-parameter; the native side either
        // returns a live session or a null pointer plus a typed status.
        let handle = unsafe {
            neoethos_gpu_cuda_population_create(ABI_VERSION, device, max_events, &mut status)
        };
        if handle.is_null() {
            return Err(CudaPopulationError::native("create", status));
        }
        Ok(Self {
            handle,
            device,
            max_events,
            bars: 0,
            feature_count: 0,
            population: 0,
            scenario_count: 0,
            emitted_events: 0,
            dataset_uploaded: false,
            genes_uploaded: false,
            scenarios_uploaded: false,
            pending_event: None,
            metrics_ready: false,
            strict_resident_state: StrictResidentSessionStateV1::StrictIdle,
            #[cfg(feature = "cuda")]
            drop_policy_v3: PopulationSessionDropPolicyV3::DestroyWhenIdle,
            parent_source_v1: None,
            resident_parent_shape_v3: None,
            bound_view_source_v1: None,
            resident_session_identity_sha256: None,
            native_build_identity_sha256: None,
            view_identity_sha256: None,
            resident_adaptive_base_view_token_v1: None,
            gene_batch_identity_sha256: None,
            scenario_batch_identity_sha256: None,
            uploaded_candidate_ids: Vec::new(),
            expected_scenario_identities: Arc::from([]),
            terminal_scenario_identity: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn bind_resident_feature_store_v3(
        resident: RawResidentFeatureStoreBindV3,
    ) -> Result<Self, CudaPopulationError> {
        let device = i32::try_from(resident.selected_device_ordinal)
            .map_err(|_| invalid("resident device ordinal does not fit the native i32 ABI"))?;
        let rows = usize::try_from(resident.row_count)
            .map_err(|_| invalid("resident row count does not fit usize"))?;
        let feature_count = usize::try_from(resident.feature_count)
            .map_err(|_| invalid("resident feature count does not fit usize"))?;
        if rows == 0 || feature_count == 0 || resident.smc_slots != SMC_SLOTS as u32 {
            return Err(invalid(
                "resident population bind requires non-zero rows/features and exact SMC slots",
            ));
        }
        let resident_session_identity_sha256 =
            hash_resident_population_session_identity_v3(&resident);
        let native_build_identity_sha256 = hash_native_build_identity_v3(&resident);
        let mut status = STATUS_OK;
        // SAFETY: the opaque resident import retains every allocation, event,
        // context and stream represented by this immediate descriptor. Native
        // code retains only device pointers while the Rust wrapper retains the
        // import for the complete session lifetime.
        let handle = unsafe {
            neoethos_gpu_cuda_population_bind_resident_feature_store_v3(&resident, &mut status)
        };
        if handle.is_null() {
            return Err(CudaPopulationError::native(
                "bind_resident_feature_store_v3",
                status,
            ));
        }
        Ok(Self {
            handle,
            device,
            // The native population event-capacity parameter is vestigial and
            // the V3 bind does not fabricate a compatibility-only value.
            max_events: 0,
            bars: 0,
            feature_count,
            population: 0,
            scenario_count: 0,
            emitted_events: 0,
            dataset_uploaded: true,
            genes_uploaded: false,
            scenarios_uploaded: false,
            pending_event: None,
            metrics_ready: false,
            strict_resident_state: StrictResidentSessionStateV1::StrictIdle,
            drop_policy_v3: PopulationSessionDropPolicyV3::LeakUntilResidentConsumerEvent,
            parent_source_v1: None,
            resident_parent_shape_v3: Some((rows, feature_count)),
            bound_view_source_v1: None,
            resident_session_identity_sha256: Some(resident_session_identity_sha256),
            native_build_identity_sha256: Some(native_build_identity_sha256),
            view_identity_sha256: None,
            resident_adaptive_base_view_token_v1: None,
            gene_batch_identity_sha256: None,
            scenario_batch_identity_sha256: None,
            uploaded_candidate_ids: Vec::new(),
            expected_scenario_identities: Arc::from([]),
            terminal_scenario_identity: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn arm_resident_session_leak_only_v3(&mut self) {
        self.drop_policy_v3 = PopulationSessionDropPolicyV3::LeakUntilResidentConsumerEvent;
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn authorize_resident_session_destroy_v3(&mut self) {
        self.drop_policy_v3 = PopulationSessionDropPolicyV3::DestroyWhenIdle;
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn destroy_terminal_proven_resident_search_v2(
        mut self,
    ) -> Result<(), CudaPopulationError> {
        if self.handle.is_null()
            || self.strict_resident_state != StrictResidentSessionStateV1::StrictIdle
            || self.drop_policy_v3 != PopulationSessionDropPolicyV3::DestroyWhenIdle
        {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(invalid(
                "resident Search session destruction requires exact terminal proof",
            ));
        }
        // SAFETY: the completion event was Ready, the metric lease was consumed,
        // and generation/scoring owners were explicitly released first. Native
        // acknowledges every owned free/event destroy before deleting the
        // session; a failure retains a poisoned tombstone instead.
        let status =
            unsafe { neoethos_gpu_cuda_population_destroy_terminal_checked_v2(self.handle) };
        if status != STATUS_OK {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            self.handle = std::ptr::null_mut();
            return Err(CudaPopulationError::native(
                "neoethos_gpu_cuda_population_destroy_terminal_checked_v2",
                status,
            ));
        }
        self.handle = std::ptr::null_mut();
        #[cfg(feature = "cuda-device-fixtures")]
        TERMINAL_SEARCH_SESSION_DESTROY_COUNT_V2.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn take_for_resident_consumer_lease_v3(&mut self) -> Self {
        std::mem::replace(self, Self::detached_resident_v3())
    }

    #[cfg(feature = "cuda")]
    fn detached_resident_v3() -> Self {
        Self {
            handle: std::ptr::null_mut(),
            device: -1,
            max_events: 0,
            bars: 0,
            feature_count: 0,
            population: 0,
            scenario_count: 0,
            emitted_events: 0,
            dataset_uploaded: false,
            genes_uploaded: false,
            scenarios_uploaded: false,
            pending_event: None,
            metrics_ready: false,
            strict_resident_state: StrictResidentSessionStateV1::Poisoned,
            drop_policy_v3: PopulationSessionDropPolicyV3::LeakUntilResidentConsumerEvent,
            parent_source_v1: None,
            resident_parent_shape_v3: None,
            bound_view_source_v1: None,
            resident_session_identity_sha256: None,
            native_build_identity_sha256: None,
            view_identity_sha256: None,
            resident_adaptive_base_view_token_v1: None,
            gene_batch_identity_sha256: None,
            scenario_batch_identity_sha256: None,
            uploaded_candidate_ids: Vec::new(),
            expected_scenario_identities: Arc::from([]),
            terminal_scenario_identity: None,
        }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn max_events(&self) -> usize {
        self.max_events
    }

    pub fn population(&self) -> usize {
        self.population
    }

    /// Scenarios in the last upload — the number of metric rows an evaluate
    /// will produce.
    pub fn scenario_count(&self) -> usize {
        self.scenario_count
    }

    pub fn bars(&self) -> usize {
        self.bars
    }

    pub fn emitted_events(&self) -> usize {
        self.emitted_events
    }

    pub fn upload_dataset(
        &mut self,
        dataset: PopulationDatasetView<'_>,
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("upload_dataset")?;
        // compatibility-only V0 upload. Production Search uses the sealed V1
        // parent plus explicit view binding below; this function remains until
        // real-card parity authorizes removal of the old ABI.
        if self.dataset_uploaded {
            return Err(CudaPopulationError::native(
                "upload_dataset",
                STATUS_DATASET_REUPLOAD,
            ));
        }
        let bars = dataset.validate()?;
        let header = DatasetHeader {
            abi_version: ABI_VERSION,
            row_count: bars as u64,
            feature_count: dataset.feature_count as u32,
            ..DatasetHeader::default()
        };
        let raw = RawDatasetView {
            header,
            close: dataset.close.as_ptr(),
            high: dataset.high.as_ptr(),
            low: dataset.low.as_ptr(),
            indicators: dataset.indicators.as_ptr(),
            months: dataset.months.as_ptr(),
            days: dataset.days.as_ptr(),
            timestamps: dataset.timestamps.as_ptr(),
            smc_rows: dataset.smc_rows.as_ptr(),
            adaptive_base_pips: dataset
                .adaptive_base_pips
                .map_or(std::ptr::null(), <[f64]>::as_ptr),
            adaptive_base_pips_len: dataset.adaptive_base_pips.map_or(0, <[f64]>::len),
        };
        // SAFETY: every pointer borrows a host slice that outlives this call and
        // whose length was validated above.
        let status = unsafe { neoethos_gpu_cuda_population_upload_dataset(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_dataset", status));
        }
        self.bars = bars;
        self.feature_count = dataset.feature_count;
        self.dataset_uploaded = true;
        Ok(())
    }

    pub fn upload_parent_dataset_v1(
        &mut self,
        parent: PopulationParentDatasetV1,
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("upload_parent_dataset_v1")?;
        if self.dataset_uploaded || self.parent_source_v1.is_some() {
            return Err(CudaPopulationError::native(
                "upload_parent_dataset_v1",
                STATUS_DATASET_REUPLOAD,
            ));
        }
        let rows = parent.row_count();
        let row_count = u64::try_from(rows)
            .map_err(|_| invalid("parent row count does not fit the native u64 contract"))?;
        let feature_count = u32::try_from(parent.feature_count)
            .map_err(|_| invalid("parent feature count does not fit the native u32 contract"))?;
        // Retain every asynchronous H2D source before entering native code. If
        // the native call fails after partially enqueueing transfers, the
        // buffers stay alive until this session is destroyed; retry is refused
        // above rather than reusing a partially populated device parent.
        self.parent_source_v1 = Some(parent);
        let parent = self
            .parent_source_v1
            .as_ref()
            .ok_or_else(|| invalid("retained parent dataset is missing"))?;
        let header = DatasetHeader {
            abi_version: ABI_VERSION,
            row_count,
            feature_count,
            ..DatasetHeader::default()
        };
        let raw = RawParentDatasetV1 {
            header,
            close: parent.close.as_ptr(),
            high: parent.high.as_ptr(),
            low: parent.low.as_ptr(),
            indicators_feature_major: parent.indicators_feature_major.as_ptr(),
            months: parent.months.as_ptr(),
            days: parent.days.as_ptr(),
            timestamps: parent.timestamps.as_ptr(),
            smc_rows: parent.smc_rows.as_ptr(),
        };
        // SAFETY: validation occurred in the only public constructor. The Arc
        // allocations are retained in `parent_source_v1` for the lifetime of
        // every asynchronous transfer submitted by the native session.
        let status = unsafe { neoethos_gpu_cuda_population_upload_parent_v1(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native(
                "upload_parent_dataset_v1",
                status,
            ));
        }
        self.bars = rows;
        self.feature_count = parent.feature_count;
        self.dataset_uploaded = true;
        Ok(())
    }

    pub fn bind_evaluation_view_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("bind_evaluation_view_v1")?;
        let resident_parent_rows = self.resident_parent_shape_v3.map(|(rows, _)| rows);
        let parent_rows = self
            .parent_source_v1
            .as_ref()
            .map(PopulationParentDatasetV1::row_count)
            .or(resident_parent_rows)
            .ok_or_else(|| {
                CudaPopulationError::native("bind_evaluation_view_v1", STATUS_MISSING_UPLOAD)
            })?;
        if view.parent_row_count != parent_rows {
            return Err(invalid(format!(
                "population view parent has {} rows; uploaded parent has {}",
                view.parent_row_count, parent_rows
            )));
        }
        let view_identity_sha256 = hash_population_view_identity_v1(&view);
        let parent_row_count = u64::try_from(view.parent_row_count)
            .map_err(|_| invalid("view parent row count does not fit the native u64 contract"))?;
        let row_count = u64::try_from(view.row_count())
            .map_err(|_| invalid("view row count does not fit the native u64 contract"))?;
        let (view_kind, range_start) = match view.kind {
            PopulationViewKindV1::Full => (0, 0),
            PopulationViewKindV1::ContiguousRange => {
                let start = view
                    .range
                    .as_ref()
                    .ok_or_else(|| invalid("validated range view lost its range"))?
                    .start;
                let start = u64::try_from(start).map_err(|_| {
                    invalid("view range start does not fit the native u64 contract")
                })?;
                (1, start)
            }
            PopulationViewKindV1::OrderedIndices => (2, 0),
        };
        let raw = RawEvaluationViewV1 {
            abi_version: ABI_VERSION,
            view_kind,
            parent_row_count,
            range_start,
            row_count,
            ordered_indices: view
                .ordered_indices
                .as_deref()
                .map_or(std::ptr::null(), <[u64]>::as_ptr),
            ordered_index_count: view.ordered_indices.as_deref().map_or(0, <[u64]>::len),
            timestamp_mode: match view.timestamp_mode {
                PopulationTimestampModeV1::Canonical => 0,
                PopulationTimestampModeV1::DisabledIndexDelta => 1,
            },
            adaptive_base_pips: view
                .adaptive_base_pips
                .as_deref()
                .map_or(std::ptr::null(), <[f64]>::as_ptr),
            adaptive_base_pips_len: view.adaptive_base_pips.as_deref().map_or(0, <[f64]>::len),
        };
        // SAFETY: every pointer comes from a validated Arc retained below until
        // the next bind, which native code refuses while work is incomplete.
        let status = unsafe { neoethos_gpu_cuda_population_bind_view_v1(self.handle, &raw) };
        if status != STATUS_OK {
            // See the parent upload above: an error may follow accepted async
            // copies, so retain this view's host buffers until session teardown.
            self.bound_view_source_v1 = Some(view);
            return Err(CudaPopulationError::native(
                "bind_evaluation_view_v1",
                status,
            ));
        }
        self.bars = view.row_count();
        self.bound_view_source_v1 = Some(view);
        self.view_identity_sha256 = Some(view_identity_sha256);
        self.resident_adaptive_base_view_token_v1 = None;
        self.scenarios_uploaded = false;
        self.scenario_count = 0;
        self.scenario_batch_identity_sha256 = None;
        self.expected_scenario_identities = Arc::from([]);
        self.terminal_scenario_identity = None;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn bind_evaluation_view_with_resident_adaptive_base_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
        request: ResidentAdaptiveBaseRequestV1,
    ) -> Result<&ResidentAdaptiveBaseViewTokenV1, CudaPopulationError> {
        self.require_strict_idle_v1("bind_evaluation_view_with_resident_adaptive_base_v1")?;
        let (parent_rows, _) = self.resident_parent_shape_v3.ok_or_else(|| {
            invalid("resident adaptive producer requires one borrowed resident V3 parent")
        })?;
        if self.parent_source_v1.is_some() || view.parent_row_count != parent_rows {
            return Err(invalid(
                "resident adaptive view does not match the exact borrowed parent",
            ));
        }
        let tail_step = usize::try_from(request.tail_step)
            .map_err(|_| invalid("resident adaptive tail step does not fit this process"))?;
        let tail_max_bars = usize::try_from(request.tail_max_bars)
            .map_err(|_| invalid("resident adaptive tail cap does not fit this process"))?;
        let expected = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
            &view,
            request.pip_size,
            tail_step,
            tail_max_bars,
        )?;
        if request != expected {
            return Err(invalid(
                "resident adaptive recipe drifted from the canonical V1 formula/view",
            ));
        }

        let parent_row_count = u64::try_from(view.parent_row_count)
            .map_err(|_| invalid("view parent row count does not fit the native u64 contract"))?;
        let row_count = u64::try_from(view.row_count())
            .map_err(|_| invalid("view row count does not fit the native u64 contract"))?;
        let (view_kind, range_start) = match view.kind {
            PopulationViewKindV1::Full => (0, 0),
            PopulationViewKindV1::ContiguousRange => {
                let start = view
                    .range
                    .as_ref()
                    .ok_or_else(|| invalid("validated range view lost its range"))?
                    .start;
                let start = u64::try_from(start).map_err(|_| {
                    invalid("view range start does not fit the native u64 contract")
                })?;
                (1, start)
            }
            PopulationViewKindV1::OrderedIndices => {
                return Err(invalid(
                    "resident adaptive producer V1 refuses ordered views",
                ));
            }
        };
        let raw = RawEvaluationViewV1 {
            abi_version: ABI_VERSION,
            view_kind,
            parent_row_count,
            range_start,
            row_count,
            ordered_indices: std::ptr::null(),
            ordered_index_count: 0,
            timestamp_mode: match view.timestamp_mode {
                PopulationTimestampModeV1::Canonical => 0,
                PopulationTimestampModeV1::DisabledIndexDelta => 1,
            },
            adaptive_base_pips: std::ptr::null(),
            adaptive_base_pips_len: 0,
        };
        let resident_session_identity_sha256 = self
            .resident_session_identity_sha256
            .ok_or_else(|| invalid("resident adaptive producer lacks its session identity"))?;
        let request_identity_sha256 = request.identity_sha256();
        let base_view_identity_sha256 = hash_population_view_identity_v1(&view);
        let view_identity_sha256 = hash_resident_adaptive_population_view_identity_v1(
            base_view_identity_sha256,
            request_identity_sha256,
        );

        // SAFETY: this descriptor carries no host price/base pointer. Native
        // code reads the resident parent and writes its retained adaptive output
        // on the already-admitted stream; `view` only contributes scalar bounds.
        let status = unsafe {
            neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1(self.handle, &raw, &request)
        };
        if status != STATUS_OK {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            return Err(CudaPopulationError::native(
                "bind_evaluation_view_with_resident_adaptive_base_v1",
                status,
            ));
        }

        let token_identity_sha256 = hash_resident_adaptive_view_token_v1(
            resident_session_identity_sha256,
            view_identity_sha256,
            request_identity_sha256,
        );
        self.bars = view.row_count();
        self.bound_view_source_v1 = Some(view);
        self.view_identity_sha256 = Some(view_identity_sha256);
        self.resident_adaptive_base_view_token_v1 = Some(ResidentAdaptiveBaseViewTokenV1 {
            resident_session_identity_sha256,
            view_identity_sha256,
            request_identity_sha256,
            token_identity_sha256,
        });
        self.scenarios_uploaded = false;
        self.scenario_count = 0;
        self.scenario_batch_identity_sha256 = None;
        self.expected_scenario_identities = Arc::from([]);
        self.terminal_scenario_identity = None;
        self.metrics_ready = false;
        self.pending_event = None;
        self.resident_adaptive_base_view_token_v1
            .as_ref()
            .ok_or_else(|| invalid("resident adaptive producer lost its typed view token"))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn arm_resident_adaptive_validator_guard_v1(
        &mut self,
    ) -> Result<&ResidentAdaptiveBaseViewTokenV1, CudaPopulationError> {
        self.require_strict_idle_v1("arm_resident_adaptive_validator_guard_v1")?;
        if self.resident_adaptive_base_view_token_v1.is_none() {
            return Err(invalid(
                "resident adaptive validator guard requires the current bound token",
            ));
        }
        // Poison first: if the external validator unwinds and its caller catches
        // that panic, every later upload remains fail-closed. Only the explicit
        // success transition below can restore StrictIdle.
        self.poison_resident_search_owner_v2();
        self.resident_adaptive_base_view_token_v1
            .as_ref()
            .ok_or_else(|| invalid("resident adaptive validator guard lost its current token"))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn accept_resident_adaptive_validator_guard_v1(
        &mut self,
    ) -> Result<(), CudaPopulationError> {
        if self.strict_resident_state != StrictResidentSessionStateV1::Poisoned
            || self.resident_adaptive_base_view_token_v1.is_none()
        {
            return Err(invalid(
                "resident adaptive validator success transition lacks its guarded token",
            ));
        }
        self.strict_resident_state = StrictResidentSessionStateV1::StrictIdle;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn poison_after_resident_adaptive_validator_rejection_v1(&mut self) {
        self.resident_adaptive_base_view_token_v1 = None;
        self.bound_view_source_v1 = None;
        self.view_identity_sha256 = None;
        self.bars = 0;
        self.poison_resident_search_owner_v2();
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn copy_resident_adaptive_base_fixture_v1(
        &mut self,
    ) -> Result<Vec<f64>, CudaPopulationError> {
        self.require_strict_idle_v1("copy_resident_adaptive_base_fixture_v1")?;
        if self.resident_adaptive_base_view_token_v1.is_none() || self.bars == 0 {
            return Err(invalid(
                "adaptive fixture readback requires one exact resident adaptive view token",
            ));
        }
        let mut values = vec![0.0_f64; self.bars];
        // SAFETY: the vector has exactly `bars` initialized f64 slots and the
        // fixture-only native boundary synchronizes the admitted session stream
        // before returning. Production builds do not export this D2H boundary.
        let status = unsafe {
            neoethos_gpu_cuda_population_copy_resident_adaptive_base_fixture_v1(
                self.handle,
                values.as_mut_ptr(),
                values.len(),
            )
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native(
                "copy_resident_adaptive_base_fixture_v1",
                status,
            ));
        }
        Ok(values)
    }

    pub fn read_residency_counters_v1(
        &self,
    ) -> Result<PopulationResidencyCountersV1, CudaPopulationError> {
        self.require_strict_idle_v1("read_residency_counters_v1")?;
        let mut counters = PopulationResidencyCountersV1::default();
        // SAFETY: `counters` is a live repr(C) out-parameter and the session is
        // exclusively borrowed for the duration of the call.
        let status = unsafe {
            neoethos_gpu_cuda_population_read_residency_counters_v1(self.handle, &mut counters)
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native(
                "read_residency_counters_v1",
                status,
            ));
        }
        Ok(counters)
    }

    pub fn read_device_identity_v1(
        &self,
    ) -> Result<CudaPopulationDeviceIdentityV1, CudaPopulationError> {
        self.require_strict_idle_v1("read_device_identity_v1")?;
        let mut identity = CudaPopulationDeviceIdentityV1::default();
        // SAFETY: `identity` is a live repr(C) out-parameter and the session is
        // immutably borrowed for the duration of the read.
        let status = unsafe {
            neoethos_gpu_cuda_population_read_device_identity_v1(self.handle, &mut identity)
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native(
                "read_device_identity_v1",
                status,
            ));
        }
        Ok(identity)
    }

    pub fn upload_genes(
        &mut self,
        genes: PopulationGeneView<'_>,
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("upload_genes")?;
        if !self.dataset_uploaded {
            return Err(CudaPopulationError::native(
                "upload_genes",
                STATUS_MISSING_UPLOAD,
            ));
        }
        let population = genes.validate(self.feature_count)?;
        let gene_batch_identity_sha256 = hash_population_gene_batch_identity_v1(&genes);
        let uploaded_candidate_ids = genes
            .descriptors
            .iter()
            .map(|descriptor| descriptor.candidate_id)
            .collect::<Vec<_>>();
        let raw = RawGeneView {
            descriptors: genes.descriptors.as_ptr(),
            count: population,
            offsets: genes.offsets.as_ptr(),
            indices: genes.indices.as_ptr(),
            weights: genes.weights.as_ptr(),
            term_count: genes.indices.len(),
            stop_pips: genes.stop_pips.as_ptr(),
            target_pips: genes.target_pips.as_ptr(),
            stop_vol_multipliers: genes.stop_vol_multipliers.as_ptr(),
            smc_flags: genes.smc_flags.as_ptr(),
            smc_weights: genes.smc_weights.as_ptr(),
            gate_threshold: genes.gate_threshold,
            smc_gate_disabled: u32::from(genes.smc_gate_disabled),
        };
        // SAFETY: as above; the native side copies before returning.
        let status = unsafe { neoethos_gpu_cuda_population_upload_genes(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_genes", status));
        }
        self.population = population;
        self.genes_uploaded = true;
        self.gene_batch_identity_sha256 = Some(gene_batch_identity_sha256);
        self.uploaded_candidate_ids = uploaded_candidate_ids;
        self.scenarios_uploaded = false;
        self.scenario_count = 0;
        self.scenario_batch_identity_sha256 = None;
        self.expected_scenario_identities = Arc::from([]);
        self.terminal_scenario_identity = None;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    /// Upload the work list.
    ///
    /// The count is NO LONGER required to equal the population. That equality is
    /// why a screen wanting 101 treatments of one gene had to clone the gene 101
    /// times; each descriptor now names its own gene, window, costs and
    /// perturbation counter, so 174 genes and 17 574 scenarios go up together.
    ///
    /// What is still required is that every scenario names a gene that exists —
    /// checked here rather than only natively, because an out-of-range index is
    /// an out-of-bounds device read of thresholds and CSR offsets that would
    /// still produce a metric row. The native side checks it again, including
    /// the window bounds it alone knows the series length for.
    pub fn upload_scenarios(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("upload_scenarios")?;
        if !self.genes_uploaded {
            return Err(CudaPopulationError::native(
                "upload_scenarios",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if scenarios.is_empty() {
            return Err(invalid("scenario array is empty"));
        }
        if let Some((index, scenario)) = scenarios
            .iter()
            .enumerate()
            .find(|(_, scenario)| scenario.base_candidate_id as usize >= self.population)
        {
            return Err(invalid(format!(
                "scenario {index} names gene {} outside the population of {}",
                scenario.base_candidate_id, self.population
            )));
        }
        let scenario_batch_identity_sha256 = hash_population_scenario_batch_identity_v1(scenarios);
        let expected_scenario_identities = scenarios
            .iter()
            .map(|scenario| {
                (
                    self.uploaded_candidate_ids[scenario.base_candidate_id as usize],
                    scenario.scenario_id,
                )
            })
            .collect::<Arc<[(u64, u64)]>>();
        let terminal_scenario_identity = (scenarios.len() == 1).then(|| {
            let scenario = scenarios[0];
            (
                self.uploaded_candidate_ids[scenario.base_candidate_id as usize],
                scenario.scenario_id,
            )
        });
        let raw = RawScenarioView {
            descriptors: scenarios.as_ptr(),
            count: scenarios.len(),
        };
        // SAFETY: as above.
        let status = unsafe { neoethos_gpu_cuda_population_upload_scenarios(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_scenarios", status));
        }
        self.scenario_count = scenarios.len();
        self.scenarios_uploaded = true;
        self.scenario_batch_identity_sha256 = Some(scenario_batch_identity_sha256);
        self.expected_scenario_identities = expected_scenario_identities;
        self.terminal_scenario_identity = terminal_scenario_identity;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    #[allow(dead_code)] // Reached by the crate-private resident Search owner.
    pub(crate) fn upload_resident_scenarios_v2(
        &mut self,
        scenarios: &[ScenarioDescriptor],
        planned_population: u64,
        generation_index: u64,
        gene_batch_identity_sha256: [u8; 32],
    ) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("upload_resident_scenarios_v2")?;
        if self.genes_uploaded || self.scenarios_uploaded {
            return Err(invalid(
                "resident scenario admission requires one fresh generation-owned session",
            ));
        }
        let population = usize::try_from(planned_population)
            .map_err(|_| invalid("resident population does not fit host usize"))?;
        if population == 0 || population > i32::MAX as usize || generation_index > u32::MAX as u64 {
            return Err(invalid(
                "resident population/generation is outside the V2 evaluator ABI",
            ));
        }
        if scenarios.is_empty() {
            return Err(invalid("resident scenario array is empty"));
        }
        if let Some((index, scenario)) = scenarios.iter().enumerate().find(|(_, scenario)| {
            usize::try_from(scenario.base_candidate_id)
                .map_or(true, |candidate| candidate >= population)
        }) {
            return Err(invalid(format!(
                "resident scenario {index} names gene {} outside the population of {population}",
                scenario.base_candidate_id
            )));
        }
        let identity_prefix = generation_index << 32;
        let uploaded_candidate_ids = (0..population)
            .map(|candidate| identity_prefix ^ candidate as u64)
            .collect::<Vec<_>>();
        let expected_scenario_identities = scenarios
            .iter()
            .map(|scenario| {
                (
                    uploaded_candidate_ids[scenario.base_candidate_id as usize],
                    scenario.scenario_id,
                )
            })
            .collect::<Arc<[(u64, u64)]>>();
        let terminal_scenario_identity = (scenarios.len() == 1).then(|| {
            let scenario = scenarios[0];
            (
                uploaded_candidate_ids[scenario.base_candidate_id as usize],
                scenario.scenario_id,
            )
        });
        let scenario_batch_identity_sha256 = hash_population_scenario_batch_identity_v1(scenarios);
        let raw = RawScenarioView {
            descriptors: scenarios.as_ptr(),
            count: scenarios.len(),
        };
        // SAFETY: native copies the exact checked descriptor slice on the
        // admitted stream and resolves genes only from its resident V2 run.
        let status = unsafe {
            neoethos_gpu_cuda_population_upload_resident_scenarios_v2(
                self.handle,
                &raw,
                planned_population,
            )
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native(
                "upload_resident_scenarios_v2",
                status,
            ));
        }
        self.population = population;
        self.scenario_count = scenarios.len();
        self.scenarios_uploaded = true;
        self.gene_batch_identity_sha256 = Some(gene_batch_identity_sha256);
        self.scenario_batch_identity_sha256 = Some(scenario_batch_identity_sha256);
        self.uploaded_candidate_ids = uploaded_candidate_ids;
        self.expected_scenario_identities = expected_scenario_identities;
        self.terminal_scenario_identity = terminal_scenario_identity;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    /// Enqueue the production metrics-only population walk on this session's
    /// existing native stream. The returned owner keeps the session borrowed;
    /// only a later resident GPU stage may consume its private event dependency.
    pub fn enqueue_metrics_only_v1(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentPopulationMetricsV1<'_>, CudaPopulationError> {
        self.require_strict_idle_v1("enqueue_metrics_only_v1")?;
        if !self.scenarios_uploaded {
            return Err(CudaPopulationError::native(
                "enqueue_metrics_only_v1",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if settings.month_capacity == 0 || settings.month_capacity > i32::MAX as u32 {
            return Err(invalid(
                "month_capacity must be non-zero and fit the native signed extent",
            ));
        }
        let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            self.scenario_count,
            settings.month_capacity,
        )?;
        let settings_identity_sha256 = hash_population_settings_identity_v1(settings);
        // Box before FFI so native can seal this stable address as the private
        // consumer token. Event ids alone are session-local and can collide.
        let mut receipt = Box::new(RawResidentPopulationMetricsHandleV1::default());
        let mut counters = NeoPopulationCounters::default();
        // SAFETY: settings and the fixed-width receipt are live POD values.
        // Native retains only the stable boxed receipt address as a private
        // token until consume/abandon; the receipt value and session outlive it.
        let status = unsafe {
            neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(
                self.handle,
                settings,
                receipt.as_mut(),
                &mut counters,
            )
        };
        if status != STATUS_OK {
            if !strict_enqueue_failure_is_known_prelaunch_v1(status) {
                self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            }
            return Err(CudaPopulationError::native(
                "enqueue_metrics_only_v1",
                status,
            ));
        }
        self.strict_resident_state = StrictResidentSessionStateV1::InFlight;
        self.emitted_events = 0;
        self.pending_event = Some(receipt.event_id);
        self.metrics_ready = false;
        if let Err(error) = validate_exact_resident_receipt_v1(receipt.as_ref(), plan) {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            // SAFETY: native accepted this exact boxed receipt and is now
            // in-flight; a rejected receipt must poison that native owner.
            let _ = unsafe {
                neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                    self.handle,
                    receipt.as_ref(),
                )
            };
            return Err(error);
        }
        let resident_session_identity_sha256 = self.resident_session_identity_sha256;
        let view_identity_sha256 = self.view_identity_sha256;
        let gene_batch_identity_sha256 = self.gene_batch_identity_sha256;
        let scenario_batch_identity_sha256 = self.scenario_batch_identity_sha256;
        let native_build_identity_sha256 = self.native_build_identity_sha256;
        let terminal_scenario_identity = self.terminal_scenario_identity;
        let expected_scenario_identities = Arc::clone(&self.expected_scenario_identities);
        Ok(ResidentPopulationMetricsV1 {
            session: self,
            receipt,
            plan,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            settings_identity_sha256,
            native_build_identity_sha256,
            terminal_scenario_identity,
            expected_scenario_identities,
            counters,
            consumed: false,
        })
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn enqueue_resident_gene_metrics_fixture_v2(
        &mut self,
        genes: &crate::resident_search_v2::RawResidentGenerationGeneViewV2,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentPopulationMetricsV1<'_>, CudaPopulationError> {
        self.require_strict_idle_v1("enqueue_resident_gene_metrics_fixture_v2")?;
        if !self.scenarios_uploaded || self.gene_batch_identity_sha256.is_none() {
            return Err(CudaPopulationError::native(
                "enqueue_resident_gene_metrics_fixture_v2",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if settings.month_capacity == 0 || settings.month_capacity > i32::MAX as u32 {
            return Err(invalid(
                "month_capacity must be non-zero and fit the native signed extent",
            ));
        }
        let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            self.scenario_count,
            settings.month_capacity,
        )?;
        let settings_identity_sha256 = hash_population_settings_identity_v1(settings);
        let mut receipt = Box::new(RawResidentPopulationMetricsHandleV1::default());
        let mut counters = NeoPopulationCounters::default();
        // SAFETY: the feature-gated real-card fixture owns the Search gene view;
        // native retains only this boxed metrics receipt until terminal consume.
        let status = unsafe {
            neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(
                self.handle,
                genes,
                settings,
                receipt.as_mut(),
                &mut counters,
            )
        };
        if status != STATUS_OK {
            if !strict_enqueue_failure_is_known_prelaunch_v1(status) {
                self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            }
            return Err(CudaPopulationError::native(
                "enqueue_resident_gene_metrics_fixture_v2",
                status,
            ));
        }
        self.strict_resident_state = StrictResidentSessionStateV1::InFlight;
        self.emitted_events = 0;
        self.pending_event = Some(receipt.event_id);
        self.metrics_ready = false;
        if let Err(error) = validate_exact_resident_receipt_v1(receipt.as_ref(), plan) {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            // SAFETY: native accepted this exact boxed receipt; rejection of
            // its fixed-width plan poisons that same native owner.
            let _ = unsafe {
                neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                    self.handle,
                    receipt.as_ref(),
                )
            };
            return Err(error);
        }
        let resident_session_identity_sha256 = self.resident_session_identity_sha256;
        let view_identity_sha256 = self.view_identity_sha256;
        let gene_batch_identity_sha256 = self.gene_batch_identity_sha256;
        let scenario_batch_identity_sha256 = self.scenario_batch_identity_sha256;
        let native_build_identity_sha256 = self.native_build_identity_sha256;
        let terminal_scenario_identity = self.terminal_scenario_identity;
        let expected_scenario_identities = Arc::clone(&self.expected_scenario_identities);
        Ok(ResidentPopulationMetricsV1 {
            session: self,
            receipt,
            plan,
            resident_session_identity_sha256,
            view_identity_sha256,
            gene_batch_identity_sha256,
            scenario_batch_identity_sha256,
            settings_identity_sha256,
            native_build_identity_sha256,
            terminal_scenario_identity,
            expected_scenario_identities,
            counters,
            consumed: false,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)] // Consumed by the move-only resident Search pending owner.
    pub(crate) fn enqueue_resident_gene_metrics_owned_v2(
        mut self,
        genes: &crate::resident_search_v2::RawResidentGenerationGeneViewV2,
        settings: &NeoPopulationSettings,
        logical_population_count: u64,
        retained_evaluation_capacity: u64,
        expected_feature_count: u64,
        expected_max_terms: u32,
        expected_full_discovery_reserve_bytes: u64,
    ) -> Result<ResidentSearchPopulationCompletionLeaseV2, CudaPopulationError> {
        self.require_strict_idle_v1("enqueue_resident_gene_metrics_owned_v2")?;
        if !self.scenarios_uploaded
            || self.gene_batch_identity_sha256.is_none()
            || logical_population_count == 0
            || retained_evaluation_capacity != logical_population_count
            || self.scenario_count as u64 != logical_population_count
            || self.population as u64 != logical_population_count
            || expected_full_discovery_reserve_bytes == 0
        {
            return Err(invalid(
                "owned resident scoring requires one immutable full-population chunk",
            ));
        }
        if settings.month_capacity == 0 || settings.month_capacity > i32::MAX as u32 {
            return Err(invalid(
                "month_capacity must be non-zero and fit the native signed extent",
            ));
        }
        let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            self.scenario_count,
            settings.month_capacity,
        )?;
        let mut receipt = Box::new(RawResidentPopulationMetricsHandleV1::default());
        let mut counters = NeoPopulationCounters::default();
        // SAFETY: `self` is moved into the returned lease, so the native session
        // and stable boxed receipt outlive every queued evaluator/scoring read.
        let status = unsafe {
            neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(
                self.handle,
                genes,
                settings,
                receipt.as_mut(),
                &mut counters,
            )
        };
        if status != STATUS_OK {
            if !strict_enqueue_failure_is_known_prelaunch_v1(status) {
                self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            }
            return Err(CudaPopulationError::native(
                "enqueue_resident_gene_metrics_owned_v2",
                status,
            ));
        }
        self.strict_resident_state = StrictResidentSessionStateV1::InFlight;
        self.emitted_events = 0;
        self.pending_event = Some(receipt.event_id);
        self.metrics_ready = false;
        if let Err(error) = validate_exact_resident_receipt_v1(receipt.as_ref(), plan) {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            // SAFETY: native accepted this exact receipt and must be abandoned
            // before the failed owner leaves scope.
            let _ = unsafe {
                neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                    self.handle,
                    receipt.as_ref(),
                )
            };
            return Err(error);
        }
        let mut raw = RawResidentScoringPopulationSourceV2::default();
        // SAFETY: the source is minted from the same stable receipt retained by
        // the owning completion lease below.
        let status = unsafe {
            neoethos_gpu_cuda_population_export_resident_scoring_source_v2(
                self.handle,
                receipt.as_ref(),
                logical_population_count,
                expected_feature_count,
                expected_max_terms,
                &mut raw,
            )
        };
        if status != STATUS_OK
            || raw.abi_version != 2
            || raw.reserved != 0
            || raw.admitted_run_stream.is_null()
            || raw.metrics_ready_event.is_null()
            || raw.scoring_ready_event.is_null()
            || raw.receipt_token.is_null()
            || raw.population_lifetime_owner.is_null()
            || raw.population_lifetime_owner != self.handle
            || std::ptr::eq(
                raw.population_lifetime_owner.cast_const(),
                raw.receipt_token,
            )
            || raw.metric_rows_device.is_null()
            || raw.expected_scenario_ids_device.is_null()
            || raw.logical_population_count != logical_population_count
            || raw.feature_count != expected_feature_count
            || raw.max_terms_per_gene != expected_max_terms
            || raw.full_discovery_reserve_bytes != expected_full_discovery_reserve_bytes
        {
            self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
            let _ = unsafe {
                neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
                    self.handle,
                    receipt.as_ref(),
                )
            };
            return Err(if status == STATUS_OK {
                invalid("native owned resident scoring source identity mismatch")
            } else {
                CudaPopulationError::native("export_resident_scoring_source_v2", status)
            });
        }
        Ok(ResidentSearchPopulationCompletionLeaseV2 {
            session: Some(self),
            receipt,
            raw,
            #[cfg(feature = "cuda-device-fixtures")]
            counters,
            consumed: false,
        })
    }

    /// Compatibility/DeviceParityOnly. This allocates the legacy diagnostic
    /// outcome workspace and returns a host-visible event id.
    pub fn evaluate(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<(u64, NeoPopulationCounters), CudaPopulationError> {
        self.require_strict_idle_v1("evaluate")?;
        if !self.scenarios_uploaded {
            return Err(CudaPopulationError::native(
                "evaluate",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if settings.month_capacity == 0 {
            return Err(invalid("month_capacity must be non-zero"));
        }
        let mut event_id = 0_u64;
        let mut counters = NeoPopulationCounters::default();
        // SAFETY: settings is a live POD reference; both out-parameters are valid.
        let status = unsafe {
            neoethos_gpu_cuda_population_b_evaluate(
                self.handle,
                settings,
                &mut event_id,
                &mut counters,
            )
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("evaluate", status));
        }
        self.emitted_events = counters.event_count as usize;
        self.pending_event = Some(event_id);
        self.metrics_ready = false;
        Ok((event_id, counters))
    }

    /// Compatibility/DeviceParityOnly host synchronization.
    pub fn wait(&mut self, event_id: u64) -> Result<(), CudaPopulationError> {
        self.require_strict_idle_v1("wait")?;
        if self.pending_event != Some(event_id) {
            return Err(CudaPopulationError::native("wait", STATUS_UNKNOWN_EVENT));
        }
        // SAFETY: the handle is live for the lifetime of `self`.
        let status = unsafe { neoethos_gpu_cuda_population_wait(self.handle, event_id) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("wait", status));
        }
        self.metrics_ready = true;
        Ok(())
    }

    /// Compatibility/DeviceParityOnly full metric-row D2H readback.
    pub fn read_metrics(&mut self) -> Result<Vec<NeoPopulationMetricRow>, CudaPopulationError> {
        self.require_strict_idle_v1("read_metrics")?;
        if !self.metrics_ready {
            return Err(CudaPopulationError::native(
                "read_metrics",
                STATUS_MISSING_UPLOAD,
            ));
        }
        // One row per SCENARIO. Sizing this by the population was correct only
        // while the two were forced equal; a 174-gene, 17 574-scenario launch
        // would have been refused for readback capacity, which is the honest
        // failure but not the useful one.
        let mut rows = vec![NeoPopulationMetricRow::default(); self.scenario_count];
        let mut written = 0_usize;
        let mut raw = RawReadback {
            rows: rows.as_mut_ptr(),
            capacity: rows.len(),
            written: &mut written,
        };
        // SAFETY: `rows` covers `capacity` elements and does not alias inputs.
        let status = unsafe { neoethos_gpu_cuda_population_read_metrics(self.handle, &mut raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("read_metrics", status));
        }
        if written != self.scenario_count {
            return Err(invalid(format!(
                "native metric readback wrote {written} rows, expected {}",
                self.scenario_count
            )));
        }
        Ok(rows)
    }

    /// Diagnostic-only outcome readback for the FIRST `scenarios` scenarios.
    ///
    /// The outcome array is scenario-major with [`MAX_TRADES_PER_CANDIDATE`]
    /// slots each, so a prefix of it is exactly "the trades of the first N
    /// scenarios" — which is what a parity investigation wants, and the whole
    /// array is not.
    ///
    /// Never call it inside a timed benchmark repetition: it is a device-to-host
    /// copy that exists to localize a parity failure.
    pub fn read_diagnostics_for(
        &mut self,
        scenarios: usize,
    ) -> Result<PopulationDiagnostics, CudaPopulationError> {
        self.require_strict_idle_v1("read_diagnostics_for")?;
        if !self.metrics_ready {
            return Err(CudaPopulationError::native(
                "read_diagnostics",
                STATUS_MISSING_UPLOAD,
            ));
        }
        let wanted = (scenarios as u64)
            .saturating_mul(MAX_TRADES_PER_CANDIDATE)
            .min(self.emitted_events as u64) as usize;
        let mut outcomes = vec![NeoPopulationOutcome::default(); wanted];
        let mut written = 0_usize;
        let mut raw = RawDiagnosticReadback {
            // NULL, deliberately. There is no event stream — the reduce opens
            // positions from the signal — and the native side used to MEMSET a
            // buffer this size to zero to carry nothing: at the ~20 000-scenario
            // launches the new sizing approves that is 163.8 M records, 9.2 GB
            // of host RAM allocated and zeroed for no content, next to 11.8 GB
            // of outcomes. On the rented box where this is actually run, that is
            // most of the machine.
            events: std::ptr::null_mut(),
            outcomes: outcomes.as_mut_ptr(),
            capacity: wanted,
            written: &mut written,
        };
        // SAFETY: `outcomes` covers `capacity` elements and does not alias; the
        // native side accepts a null `events` and skips it.
        let status =
            unsafe { neoethos_gpu_cuda_population_read_diagnostics(self.handle, &mut raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("read_diagnostics", status));
        }
        if written != wanted {
            return Err(invalid(format!(
                "native diagnostic readback wrote {written} outcomes, expected {wanted}"
            )));
        }
        Ok(PopulationDiagnostics {
            // Always empty: nothing emits events. Kept in the struct so the
            // diagnostic contract does not change shape under callers.
            events: Vec::new(),
            outcomes,
        })
    }

    /// Every recorded outcome, refused above a host-RAM budget.
    ///
    /// `emitted_events` is `scenario_count * MAX_TRADES_PER_CANDIDATE`, which
    /// grew by ~100x when the scenario became the unit of work. This used to
    /// allocate two vectors of that length unconditionally.
    /// Compatibility/DeviceParityOnly full diagnostic D2H readback.
    pub fn read_diagnostics(&mut self) -> Result<PopulationDiagnostics, CudaPopulationError> {
        self.require_strict_idle_v1("read_diagnostics")?;
        const MAX_DIAGNOSTIC_BYTES: usize = 1 << 30;
        let bytes = self
            .emitted_events
            .saturating_mul(std::mem::size_of::<NeoPopulationOutcome>());
        if bytes > MAX_DIAGNOSTIC_BYTES {
            return Err(invalid(format!(
                "a full diagnostic readback of {} outcomes wants {bytes} B of host RAM; ask for \
                 a range with `read_diagnostics_for(scenarios)` instead",
                self.emitted_events
            )));
        }
        let scenarios = self.scenario_count;
        self.read_diagnostics_for(scenarios)
    }
}

impl Drop for PopulationSession {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        let resident_drop_requires_leak =
            self.drop_policy_v3 == PopulationSessionDropPolicyV3::LeakUntilResidentConsumerEvent;
        #[cfg(not(feature = "cuda"))]
        let resident_drop_requires_leak = false;
        if resident_drop_requires_leak
            || matches!(
                self.strict_resident_state,
                StrictResidentSessionStateV1::InFlight | StrictResidentSessionStateV1::Poisoned
            )
        {
            // Fail closed: native destruction uses synchronous CUDA frees. Until
            // a future same-stream stage atomically consumes the resident event,
            // leaking this run-owned session is safer than freeing storage still
            // reachable by in-flight kernels or a borrowed V3 parent.
            self.handle = std::ptr::null_mut();
            return;
        }
        if !self.handle.is_null() {
            // SAFETY: the handle was produced by `create`. The checked native
            // destroy deletes it only after every owned resource acknowledges
            // release; otherwise it remains a fail-closed native tombstone.
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = neoethos_gpu_cuda_population_destroy_terminal_checked_v2(self.handle);
            }
            #[cfg(not(feature = "cuda"))]
            unsafe {
                neoethos_gpu_cuda_population_destroy(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_source(source: &str) -> String {
        source.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn abi_v4_contract_matches_version_layouts_and_signatures() {
        use core::mem::{align_of, offset_of, size_of};
        use neoethos_gpu_contracts::device::{
            BufferRef, HandleToken, Metrics, PropFirmState, TradeOutcome,
        };

        macro_rules! layout {
            ($ty:ty, $size:expr, $align:expr) => {
                assert_eq!(size_of::<$ty>(), $size, "sizeof({})", stringify!($ty));
                assert_eq!(align_of::<$ty>(), $align, "alignof({})", stringify!($ty));
            };
        }

        assert_eq!(ABI_VERSION, 4);
        layout!(BufferRef, 16, 8);
        assert_eq!(offset_of!(BufferRef, len), 8);
        layout!(HandleToken, 32, 8);
        assert_eq!(offset_of!(HandleToken, generation), 16);
        assert_eq!(offset_of!(HandleToken, reserved), 28);
        layout!(DatasetHeader, 152, 8);
        assert_eq!(offset_of!(DatasetHeader, timestamps), 24);
        assert_eq!(offset_of!(DatasetHeader, features), 104);
        assert_eq!(offset_of!(DatasetHeader, days), 136);
        layout!(GeneDescriptor, 72, 8);
        assert_eq!(offset_of!(GeneDescriptor, long_threshold), 16);
        assert_eq!(offset_of!(GeneDescriptor, short_threshold), 24);
        assert_eq!(offset_of!(GeneDescriptor, stop_ticks), 32);
        assert_eq!(offset_of!(GeneDescriptor, target_ticks), 40);
        assert_eq!(offset_of!(GeneDescriptor, stop_vol_multiplier), 48);
        assert_eq!(offset_of!(GeneDescriptor, flags), 56);
        assert_eq!(offset_of!(GeneDescriptor, reserved), 64);
        layout!(ScenarioDescriptor, 72, 8);
        assert_eq!(offset_of!(ScenarioDescriptor, window_offset), 24);
        assert_eq!(offset_of!(ScenarioDescriptor, commission_micros), 48);
        assert_eq!(offset_of!(ScenarioDescriptor, reserved), 64);
        layout!(TradeOutcome, 56, 8);
        assert_eq!(offset_of!(TradeOutcome, pnl_micros), 32);
        assert_eq!(offset_of!(TradeOutcome, reserved), 48);
        layout!(Metrics, 80, 8);
        assert_eq!(offset_of!(Metrics, net_profit), 16);
        assert_eq!(offset_of!(Metrics, trade_count), 56);
        assert_eq!(offset_of!(Metrics, flags), 72);
        layout!(PropFirmState, 56, 8);
        assert_eq!(offset_of!(PropFirmState, trading_days), 48);
        layout!(crate::CudaFirstHitEvent, 32, 8);
        assert_eq!(offset_of!(crate::CudaFirstHitEvent, stop_price), 16);
        assert_eq!(offset_of!(crate::CudaFirstHitEvent, target_price), 24);
        layout!(crate::CudaFirstHitResult, 8, 4);
        assert_eq!(offset_of!(crate::CudaFirstHitResult, exit_reason), 4);
        layout!(NeoPopulationSettings, 184, 8);
        assert_eq!(offset_of!(NeoPopulationSettings, gap_threshold_ms), 24);
        assert_eq!(offset_of!(NeoPopulationSettings, adaptive_rr), 120);
        assert_eq!(offset_of!(NeoPopulationSettings, trailing_enabled), 128);
        assert_eq!(offset_of!(NeoPopulationSettings, spread_pips_asian), 160);
        assert_eq!(offset_of!(NeoPopulationSettings, spread_pips_late_ny), 176);
        layout!(NeoPopulationEvent, 56, 8);
        assert_eq!(offset_of!(NeoPopulationEvent, direction), 24);
        assert_eq!(offset_of!(NeoPopulationEvent, stop_price), 32);
        assert_eq!(offset_of!(NeoPopulationEvent, entry_price), 48);
        layout!(NeoPopulationOutcome, 72, 8);
        assert_eq!(offset_of!(NeoPopulationOutcome, exit_bar), 16);
        assert_eq!(offset_of!(NeoPopulationOutcome, entry_bar), 24);
        assert_eq!(offset_of!(NeoPopulationOutcome, mfe), 32);
        assert_eq!(offset_of!(NeoPopulationOutcome, exit_price), 48);
        assert_eq!(offset_of!(NeoPopulationOutcome, r_multiple), 64);
        layout!(NeoPopulationMetricRow, 104, 8);
        assert_eq!(offset_of!(NeoPopulationMetricRow, values), 16);
        layout!(NeoPopulationCounters, 96, 8);
        assert_eq!(offset_of!(NeoPopulationCounters, reserved), 72);
        layout!(RawDatasetView, 232, 8);
        assert_eq!(offset_of!(RawDatasetView, close), 152);
        assert_eq!(offset_of!(RawDatasetView, indicators), 176);
        assert_eq!(offset_of!(RawDatasetView, smc_rows), 208);
        assert_eq!(offset_of!(RawDatasetView, adaptive_base_pips_len), 224);
        layout!(RawParentDatasetV1, 216, 8);
        assert_eq!(
            offset_of!(RawParentDatasetV1, indicators_feature_major),
            176
        );
        assert_eq!(offset_of!(RawParentDatasetV1, smc_rows), 208);
        #[cfg(feature = "cuda")]
        {
            layout!(RawResidentFeatureStoreBindV3, 256, 8);
            assert_eq!(offset_of!(RawResidentFeatureStoreBindV3, row_count), 8);
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, compute_capability_major),
                24
            );
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, packed_validity_bytes),
                32
            );
            assert_eq!(offset_of!(RawResidentFeatureStoreBindV3, close), 40);
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, indicators_bar_major),
                64
            );
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, indicators_validity_u4),
                72
            );
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, admitted_primary_context),
                112
            );
            assert_eq!(offset_of!(RawResidentFeatureStoreBindV3, device_uuid), 136);
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, admission_identity_sha256),
                152
            );
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, canonical_content_merkle),
                184
            );
            assert_eq!(
                offset_of!(
                    RawResidentFeatureStoreBindV3,
                    allocator_context_reserve_bytes
                ),
                216
            );
            assert_eq!(
                offset_of!(RawResidentFeatureStoreBindV3, run_stream_process_token_v3),
                224
            );
        }
        layout!(RawEvaluationViewV1, 72, 8);
        assert_eq!(offset_of!(RawEvaluationViewV1, ordered_indices), 32);
        assert_eq!(offset_of!(RawEvaluationViewV1, adaptive_base_pips), 56);
        layout!(ResidentAdaptiveBaseRequestV1, 96, 8);
        assert_eq!(
            offset_of!(ResidentAdaptiveBaseRequestV1, parent_row_count),
            8
        );
        assert_eq!(
            offset_of!(ResidentAdaptiveBaseRequestV1, view_row_count),
            24
        );
        assert_eq!(offset_of!(ResidentAdaptiveBaseRequestV1, vol_window), 32);
        assert_eq!(offset_of!(ResidentAdaptiveBaseRequestV1, tail_step), 48);
        assert_eq!(offset_of!(ResidentAdaptiveBaseRequestV1, pip_size), 64);
        assert_eq!(
            offset_of!(ResidentAdaptiveBaseRequestV1, meta_label_min_dist),
            88
        );
        layout!(PopulationResidencyCountersV1, 144, 8);
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, metric_rows_readback_count),
            80
        );
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, metric_rows_readback_rows),
            88
        );
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, metric_rows_readback_bytes),
            96
        );
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, diagnostic_readback_count),
            104
        );
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, diagnostic_readback_rows),
            112
        );
        assert_eq!(
            offset_of!(PopulationResidencyCountersV1, diagnostic_readback_bytes),
            120
        );
        assert_eq!(
            offset_of!(
                PopulationResidencyCountersV1,
                accepted_trade_total_readback_count
            ),
            128
        );
        assert_eq!(
            offset_of!(
                PopulationResidencyCountersV1,
                accepted_trade_total_readback_bytes
            ),
            136
        );
        layout!(CudaPopulationDeviceIdentityV1, 312, 8);
        assert_eq!(
            offset_of!(CudaPopulationDeviceIdentityV1, total_global_memory_bytes),
            16
        );
        assert_eq!(offset_of!(CudaPopulationDeviceIdentityV1, uuid), 36);
        assert_eq!(offset_of!(CudaPopulationDeviceIdentityV1, name), 52);
        layout!(RawGeneView, 104, 8);
        assert_eq!(offset_of!(RawGeneView, count), 8);
        assert_eq!(offset_of!(RawGeneView, weights), 32);
        assert_eq!(offset_of!(RawGeneView, stop_pips), 48);
        assert_eq!(offset_of!(RawGeneView, smc_flags), 72);
        assert_eq!(offset_of!(RawGeneView, gate_threshold), 88);
        assert_eq!(offset_of!(RawGeneView, smc_gate_disabled), 96);
        layout!(RawScenarioView, 16, 8);
        assert_eq!(offset_of!(RawScenarioView, count), 8);
        layout!(RawReadback, 24, 8);
        assert_eq!(offset_of!(RawReadback, written), 16);
        layout!(RawDiagnosticReadback, 32, 8);
        assert_eq!(offset_of!(RawDiagnosticReadback, outcomes), 8);
        assert_eq!(offset_of!(RawDiagnosticReadback, written), 24);
        layout!(RawResidentPopulationMetricsHandleV1, 88, 8);
        assert_eq!(
            offset_of!(RawResidentPopulationMetricsHandleV1, event_id),
            8
        );
        assert_eq!(
            offset_of!(RawResidentPopulationMetricsHandleV1, total_device_bytes),
            64
        );
        layout!(RawResidentScoringPopulationSourceV2, 96, 8);
        assert_eq!(
            offset_of!(
                RawResidentScoringPopulationSourceV2,
                population_lifetime_owner
            ),
            40
        );
        assert_eq!(
            offset_of!(
                RawResidentScoringPopulationSourceV2,
                full_discovery_reserve_bytes
            ),
            88
        );
        let default_scoring_source = RawResidentScoringPopulationSourceV2::default();
        assert!(default_scoring_source.receipt_token.is_null());
        assert!(default_scoring_source.population_lifetime_owner.is_null());
        layout!(RawTerminalCompactPopulationResultV1, 160, 8);
        assert_eq!(
            offset_of!(RawTerminalCompactPopulationResultV1, metric_row),
            24
        );
        assert_eq!(
            offset_of!(
                RawTerminalCompactPopulationResultV1,
                terminal_readback_bytes
            ),
            152
        );

        let _: unsafe extern "C" fn() -> u32 = crate::neoethos_gpu_cuda_abi_version;
        let _: unsafe extern "C" fn() -> i32 = crate::neoethos_gpu_cuda_runtime_available;
        let _: unsafe extern "C" fn() -> i32 = crate::neoethos_gpu_cuda_device_count;
        let _: unsafe extern "C" fn(i32) -> u64 = crate::neoethos_gpu_cuda_device_free_memory;
        let _: unsafe extern "C" fn(*const u32, *mut u32, usize) -> i32 =
            crate::neoethos_gpu_cuda_smoke;
        let _: unsafe extern "C" fn(
            *const f64,
            *const f64,
            usize,
            *const crate::CudaFirstHitEvent,
            *mut crate::CudaFirstHitResult,
            usize,
        ) -> i32 = crate::neoethos_gpu_cuda_warp_first_hit;
        let _: unsafe extern "C" fn(u32, i32, usize, *mut i32) -> *mut c_void =
            neoethos_gpu_cuda_population_create;
        let _: unsafe extern "C" fn(*mut c_void, *const RawDatasetView) -> i32 =
            neoethos_gpu_cuda_population_upload_dataset;
        let _: unsafe extern "C" fn(*mut c_void, *const RawParentDatasetV1) -> i32 =
            neoethos_gpu_cuda_population_upload_parent_v1;
        #[cfg(feature = "cuda")]
        let _: unsafe extern "C" fn(
            *const RawResidentFeatureStoreBindV3,
            *mut i32,
        ) -> *mut c_void = neoethos_gpu_cuda_population_bind_resident_feature_store_v3;
        let _: unsafe extern "C" fn(*mut c_void, *const RawEvaluationViewV1) -> i32 =
            neoethos_gpu_cuda_population_bind_view_v1;
        #[cfg(feature = "cuda")]
        let _: unsafe extern "C" fn(
            *mut c_void,
            *const RawEvaluationViewV1,
            *const ResidentAdaptiveBaseRequestV1,
        ) -> i32 = neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1;
        let _: unsafe extern "C" fn(*mut c_void, *mut PopulationResidencyCountersV1) -> i32 =
            neoethos_gpu_cuda_population_read_residency_counters_v1;
        let _: unsafe extern "C" fn(*mut c_void, *mut CudaPopulationDeviceIdentityV1) -> i32 =
            neoethos_gpu_cuda_population_read_device_identity_v1;
        let _: unsafe extern "C" fn(*mut c_void, *const RawGeneView) -> i32 =
            neoethos_gpu_cuda_population_upload_genes;
        let _: unsafe extern "C" fn(*mut c_void, *const RawScenarioView) -> i32 =
            neoethos_gpu_cuda_population_upload_scenarios;
        let _: unsafe extern "C" fn(
            *mut c_void,
            *const NeoPopulationSettings,
            *mut RawResidentPopulationMetricsHandleV1,
            *mut NeoPopulationCounters,
        ) -> i32 = neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1;
        let _: unsafe extern "C" fn(
            *mut c_void,
            *const RawResidentPopulationMetricsHandleV1,
            *mut RawTerminalCompactPopulationResultV1,
        ) -> i32 = neoethos_gpu_cuda_population_consume_terminal_compact_result_v1;
        let _: unsafe extern "C" fn(
            *mut c_void,
            *const NeoPopulationSettings,
            *mut u64,
            *mut NeoPopulationCounters,
        ) -> i32 = neoethos_gpu_cuda_population_b_evaluate;
        let _: unsafe extern "C" fn(*mut c_void, u64) -> i32 = neoethos_gpu_cuda_population_wait;
        let _: unsafe extern "C" fn(*mut c_void, *mut RawReadback) -> i32 =
            neoethos_gpu_cuda_population_read_metrics;
        let _: unsafe extern "C" fn(*mut c_void, *mut RawDiagnosticReadback) -> i32 =
            neoethos_gpu_cuda_population_read_diagnostics;
        let _: unsafe extern "C" fn(*mut c_void) = neoethos_gpu_cuda_population_destroy;

        const HEADER: &str = include_str!("../native/neoethos_gpu_cuda.h");
        const ASSERTS: &str = include_str!("../native/layout_asserts.cpp");
        let mut missing = Vec::new();
        for required in [
            "#define NEOETHOS_GPU_ABI_VERSION 4u",
            "double long_threshold;",
            "double short_threshold;",
            "double stop_vol_multiplier;",
        ] {
            if !HEADER.contains(required) {
                missing.push(format!("header: {required}"));
            }
        }
        for required in [
            "static_assert(sizeof(NeoGeneDescriptor) == 72);",
            "static_assert(alignof(NeoGeneDescriptor) == 8);",
            "static_assert(offsetof(NeoGeneDescriptor, long_threshold) == 16);",
            "static_assert(offsetof(NeoGeneDescriptor, short_threshold) == 24);",
            "static_assert(offsetof(NeoGeneDescriptor, stop_ticks) == 32);",
            "static_assert(offsetof(NeoGeneDescriptor, target_ticks) == 40);",
            "static_assert(offsetof(NeoGeneDescriptor, stop_vol_multiplier) == 48);",
            "static_assert(offsetof(NeoGeneDescriptor, flags) == 56);",
            "static_assert(offsetof(NeoGeneDescriptor, reserved) == 64);",
            "static_assert(sizeof(NeoPopulationDatasetView) == 232);",
            "static_assert(sizeof(NeoPopulationParentDatasetV1) == 216);",
            "static_assert(sizeof(NeoPopulationResidentFeatureStoreV3) == 256);",
            "static_assert(sizeof(NeoPopulationEvaluationViewV1) == 72);",
            "static_assert(sizeof(NeoResidentAdaptiveBaseRequestV1) == 96);",
            "static_assert(sizeof(NeoPopulationResidencyCountersV1) == 144);",
            "static_assert(sizeof(NeoPopulationDeviceIdentityV1) == 312);",
            "static_assert(sizeof(NeoPopulationGeneView) == 104);",
            "static_assert(sizeof(NeoPopulationScenarioView) == 16);",
            "static_assert(sizeof(NeoPopulationReadback) == 24);",
            "static_assert(sizeof(NeoPopulationDiagnosticReadback) == 32);",
        ] {
            if !ASSERTS.contains(required) {
                missing.push(format!("native layout proof: {required}"));
            }
        }
        let native_version = crate::native_abi_version();
        if native_version != ABI_VERSION {
            missing.push(format!(
                "native ABI version {native_version}, Rust ABI version {ABI_VERSION}"
            ));
        }
        assert!(missing.is_empty(), "ABI v4 drift:\n{}", missing.join("\n"));
    }

    #[test]
    fn abi_v4_contract_preserves_threshold_f64_distinctions() {
        let lower = f64::from_bits(1.0_f64.to_bits() - 1);
        let upper = f64::from_bits(1.0_f64.to_bits() + 1);
        assert_ne!(lower, upper);
        assert_eq!(lower as f32, upper as f32);
        let score = f64::from(1.0_f32);
        assert!(score >= lower);
        assert!(score < upper);

        const KERNEL: &str = include_str!("../native/prototype_b_population.cu");
        let kernel = normalize_source(KERNEL);
        for required in [
            "const double* long_thresholds;",
            "const double* short_thresholds;",
            "double long_threshold;",
            "double short_threshold;",
            "double gap;",
            "double* long_thresholds = nullptr;",
            "double* short_thresholds = nullptr;",
            "staging->long_thresholds = new (std::nothrow) double[population];",
            "staging->short_thresholds = new (std::nothrow) double[population];",
            "plan.long_threshold *= perturb_factor(",
            "plan.short_threshold *= perturb_factor(",
            "double gap = fabs(plan.long_threshold - plan.short_threshold);",
            "const double margin =",
            "double* confidence_out",
            "double signal_confidence_here = 0.0;",
        ] {
            assert!(
                kernel.contains(required),
                "missing f64 threshold path: {required}"
            );
        }
        for forbidden in [
            "const float* long_thresholds;",
            "const float* short_thresholds;",
            "float* long_thresholds = nullptr;",
            "float* short_thresholds = nullptr;",
            "float gap = fabsf(plan.long_threshold - plan.short_threshold);",
            "static_cast<float>( static_cast<double>(plan.long_threshold)",
            "static_cast<float>( static_cast<double>(plan.short_threshold)",
        ] {
            assert!(
                !kernel.contains(forbidden),
                "threshold path still narrows through `{forbidden}`"
            );
        }
    }

    #[test]
    fn population_f64_contract_preserves_feature_and_csr_precision() {
        use core::mem::{offset_of, size_of};

        let lower = f64::from_bits(1.0_f64.to_bits() - 1);
        let upper = f64::from_bits(1.0_f64.to_bits() + 1);
        assert_ne!(lower, upper);
        assert_eq!(lower as f32, upper as f32);

        let long_threshold = 1.0_f64;
        assert!(
            lower * 1.0 < long_threshold,
            "the lower f64 feature must not signal long"
        );
        assert!(
            upper * 1.0 > long_threshold,
            "the upper f64 feature must signal long"
        );
        assert!(
            1.0 * lower < long_threshold,
            "the lower f64 CSR weight must not signal long"
        );
        assert!(
            1.0 * upper > long_threshold,
            "the upper f64 CSR weight must signal long"
        );

        let indicators = [];
        let dataset = PopulationDatasetView {
            close: &[],
            high: &[],
            low: &[],
            indicators: &indicators,
            feature_count: 0,
            months: &[],
            days: &[],
            timestamps: &[],
            smc_rows: &[],
            adaptive_base_pips: None,
        };
        assert_eq!(core::any::type_name_of_val(&dataset.indicators), "&[f64]");

        let weights = [];
        let smc_weights = [0.0; SMC_SLOTS];
        let genes = PopulationGeneView {
            descriptors: &[],
            offsets: &[],
            indices: &[],
            weights: &weights,
            stop_pips: &[],
            target_pips: &[],
            stop_vol_multipliers: &[],
            smc_flags: &[],
            smc_weights: &smc_weights,
            gate_threshold: 0.0,
            smc_gate_disabled: false,
        };
        assert_eq!(core::any::type_name_of_val(&genes.weights), "&[f64]");

        assert_eq!(size_of::<RawGeneView>(), 104);
        assert_eq!(offset_of!(RawGeneView, gate_threshold), 88);
        assert_eq!(offset_of!(RawGeneView, smc_gate_disabled), 96);

        const HEADER: &str = include_str!("../native/neoethos_gpu_cuda.h");
        const ASSERTS: &str = include_str!("../native/layout_asserts.cpp");
        const KERNEL: &str = include_str!("../native/prototype_b_population.cu");
        let header = normalize_source(HEADER);
        let asserts = normalize_source(ASSERTS);
        let kernel = normalize_source(KERNEL);

        for required in ["const double* indicators;", "const double* weights;"] {
            assert!(
                header.contains(required),
                "missing ABI4 f64 population field: {required}"
            );
        }
        for required in [
            "static_assert(sizeof(NeoPopulationGeneView) == 104);",
            "static_assert(offsetof(NeoPopulationGeneView, gate_threshold) == 88);",
            "static_assert(offsetof(NeoPopulationGeneView, smc_gate_disabled) == 96);",
        ] {
            assert!(
                asserts.contains(required),
                "missing ABI4 population layout proof: {required}"
            );
        }
        for required in [
            "const double* indicators_bar_major;",
            "const double* weights;",
            "__shared__ double tile[kTransposeTile][kTransposeTile + 1];",
            "double combined = 0.0;",
            "const double weight = genes.weights[term] * perturb_factor(",
            "double* indicators_bar_major = nullptr;",
            "double* gene_weights = nullptr;",
            "features * bars * sizeof(double)",
            "terms * sizeof(double)",
        ] {
            assert!(
                kernel.contains(required),
                "missing f64 population data/math path: {required}"
            );
        }
        for forbidden in [
            "const float* indicators_bar_major;",
            "const float* weights;",
            "__shared__ float tile[kTransposeTile][kTransposeTile + 1];",
            "float combined = 0.0f;",
            "const float weight = static_cast<float>(",
            "float* indicators_bar_major = nullptr;",
            "float* gene_weights = nullptr;",
            "features * bars * sizeof(float)",
            "terms * sizeof(float)",
        ] {
            assert!(
                !kernel.contains(forbidden),
                "population feature/CSR path still narrows through `{forbidden}`"
            );
        }
    }

    #[test]
    fn population_f64_contract_preserves_smc_weight_and_gate_precision() {
        let lower = f64::from_bits(1.0_f64.to_bits() - 1);
        let upper = f64::from_bits(1.0_f64.to_bits() + 1);
        assert_ne!(lower, upper);
        assert_eq!(lower as f32, upper as f32);

        let gate = 1.0_f64;
        assert!(lower < gate, "the lower f64 SMC score must fail the gate");
        assert!(upper > gate, "the upper f64 SMC score must pass the gate");
        let score = 1.0_f64;
        assert!(score >= lower, "the lower f64 gate must pass");
        assert!(score < upper, "the upper f64 gate must fail");

        let weights = [];
        let smc_weights = [0.0; SMC_SLOTS];
        let genes = PopulationGeneView {
            descriptors: &[],
            offsets: &[],
            indices: &[],
            weights: &weights,
            stop_pips: &[],
            target_pips: &[],
            stop_vol_multipliers: &[],
            smc_flags: &[],
            smc_weights: &smc_weights,
            gate_threshold: 0.0,
            smc_gate_disabled: false,
        };
        assert_eq!(
            core::any::type_name_of_val(&genes.smc_weights),
            "&[f64; 11]"
        );
        assert_eq!(core::any::type_name_of_val(&genes.gate_threshold), "f64");

        const HEADER: &str = include_str!("../native/neoethos_gpu_cuda.h");
        const KERNEL: &str = include_str!("../native/prototype_b_population.cu");
        let header = normalize_source(HEADER);
        let kernel = normalize_source(KERNEL);

        for required in ["const double* smc_weights;", "double gate_threshold;"] {
            assert!(
                header.contains(required),
                "missing ABI4 f64 SMC field: {required}"
            );
        }
        for required in [
            "const double* smc_weights;",
            "double gate_threshold;",
            "double active_sum;",
            "double gate;",
            "double active_sum = 0.0;",
            "plan.gate = fmin(genes.gate_threshold, active_sum);",
            "double score = 0.0;",
            "double* smc_weights = nullptr;",
            "double gate_threshold = 0.0;",
            "kSmcSlots * sizeof(double)",
        ] {
            assert!(
                kernel.contains(required),
                "missing f64 SMC data/math path: {required}"
            );
        }
        for forbidden in [
            "const float* smc_weights;",
            "float gate_threshold;",
            "float active_sum;",
            "float gate;",
            "float active_sum = 0.0f;",
            "fminf(genes.gate_threshold, active_sum)",
            "float score = 0.0f;",
            "float* smc_weights = nullptr;",
            "float gate_threshold = 0.0f;",
            "kSmcSlots * sizeof(float)",
        ] {
            assert!(
                !kernel.contains(forbidden),
                "population SMC path still narrows through `{forbidden}`"
            );
        }
    }

    fn dataset_slices() -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i8>) {
        (
            vec![1.0; 4],
            vec![0.5; 8],
            vec![0; 4],
            vec![0; 4 * SMC_SLOTS],
        )
    }

    #[test]
    fn zero_capacity_and_negative_device_are_rejected_before_ffi() {
        assert!(matches!(
            PopulationSession::create(0, 0),
            Err(CudaPopulationError::InvalidInput(_))
        ));
        assert!(matches!(
            PopulationSession::create(-1, 8),
            Err(CudaPopulationError::InvalidInput(_))
        ));
    }

    #[test]
    fn metrics_only_default_month_plan_is_exactly_4000_bytes_per_scenario() {
        let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(1, 240)
            .expect("one scenario with 240 months has a checked metrics-only plan");
        assert_eq!(plan.metric_rows_bytes(), 104);
        assert_eq!(plan.monthly_pnls_bytes(), 1_920);
        assert_eq!(plan.month_start_equities_bytes(), 1_920);
        assert_eq!(plan.scenario_descriptor_bytes(), 56);
        assert_eq!(plan.total_device_bytes(), 4_000);
        assert_eq!(plan.outcome_bytes(), 0);
        assert_eq!(plan.accepted_trade_total_bytes(), 0);
    }

    #[test]
    fn trade_slots_match_the_kernel() {
        // Read from the source the kernel actually compiles, so the two cannot
        // drift. A mismatch means the host budgets for a different array than
        // the device allocates — undersize and it splits for no reason,
        // oversize and it runs the card out of memory.
        const KERNEL: &str = include_str!("../native/prototype_b_population.cu");
        let declaration = KERNEL
            .lines()
            .find(|line| line.contains("constexpr unsigned long long kMaxTradesPerCandidate"))
            .expect("the kernel declares its trade slots");
        let value: u64 = declaration
            .rsplit('=')
            .next()
            .and_then(|tail| {
                tail.trim()
                    .trim_end_matches(&[';', 'u', 'l'][..])
                    .parse()
                    .ok()
            })
            .unwrap_or_else(|| panic!("cannot read the slot count from: {declaration}"));
        assert_eq!(
            value, MAX_TRADES_PER_CANDIDATE,
            "the kernel reserves {value} trade slots per candidate but the host budgets for {MAX_TRADES_PER_CANDIDATE}"
        );
    }

    #[test]
    fn out_of_memory_counts_as_capacity_so_the_caller_retries_smaller() {
        // The whole point of the flag is "this would fit in pieces". A device
        // allocation failure is exactly that, and treating it as a fault sent
        // 99 % of a validation run to the CPU.
        let oom = CudaPopulationError::native("evaluate", STATUS_ALLOCATION_FAILED);
        assert!(oom.is_capacity_exhausted());
        let events = CudaPopulationError::native("evaluate", STATUS_EVENT_CAPACITY);
        assert!(events.is_capacity_exhausted());

        // A launch failure is a real fault: halving the work would hide it
        // behind a slower failure rather than fixing anything.
        let launch = CudaPopulationError::native("evaluate", STATUS_LAUNCH_FAILED);
        assert!(!launch.is_capacity_exhausted());
        let abi = CudaPopulationError::native("evaluate", STATUS_ABI_MISMATCH);
        assert!(!abi.is_capacity_exhausted());
    }

    #[test]
    fn status_messages_cover_every_documented_code() {
        for status in [
            STATUS_UNSUPPORTED,
            STATUS_NULL_SESSION,
            STATUS_ABI_MISMATCH,
            STATUS_INVALID_ARGUMENT,
            STATUS_DEVICE_UNAVAILABLE,
            STATUS_ALLOCATION_FAILED,
            STATUS_TRANSFER_FAILED,
            STATUS_LAUNCH_FAILED,
            STATUS_EVENT_CAPACITY,
            STATUS_MISSING_UPLOAD,
            STATUS_READBACK_CAPACITY,
            STATUS_SYNC_FAILED,
            STATUS_UNKNOWN_EVENT,
            STATUS_DATASET_REUPLOAD,
            STATUS_WORKSPACE_MODE_MISMATCH,
            STATUS_WORKSPACE_PLAN_MISMATCH,
            STATUS_STRICT_RESIDENT_IN_FLIGHT,
            STATUS_STRICT_RESIDENT_POISONED,
            STATUS_ADAPTIVE_BASE_DEGENERATE,
            STATUS_ASYNC_FREE_OUTCOME_UNKNOWN,
            STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN,
        ] {
            assert_ne!(population_status_message(status), "unknown native status");
        }
        assert_eq!(population_status_message(-999), "unknown native status");
    }

    #[test]
    fn async_resource_outcomes_are_typed_and_never_capacity_retries() {
        let free = CudaPopulationError::native(
            "release_resident_search",
            STATUS_ASYNC_FREE_OUTCOME_UNKNOWN,
        );
        assert!(!free.is_capacity_exhausted());
        assert!(matches!(
            free,
            CudaPopulationError::AsyncFreeOutcomeUnknownDeliberateLeak {
                operation: "release_resident_search"
            }
        ));

        let allocation = CudaPopulationError::native(
            "create_resident_search",
            STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN,
        );
        assert!(!allocation.is_capacity_exhausted());
        assert!(matches!(
            allocation,
            CudaPopulationError::AsyncAllocationOutcomeUnknownDeliberateLeak {
                operation: "create_resident_search"
            }
        ));
    }

    #[test]
    fn dataset_shape_errors_are_typed_not_undefined_behaviour() {
        let (prices, indicators, calendar, smc) = dataset_slices();
        let view = PopulationDatasetView {
            close: &prices,
            high: &prices,
            low: &prices[..3],
            indicators: &indicators,
            feature_count: 2,
            months: &calendar,
            days: &calendar,
            timestamps: &calendar,
            smc_rows: &smc,
            adaptive_base_pips: None,
        };
        let error = view.validate().unwrap_err();
        assert!(
            matches!(error, CudaPopulationError::InvalidInput(ref detail) if detail.contains("low"))
        );
    }

    #[test]
    fn non_finite_prices_are_rejected_before_ffi() {
        let (mut prices, indicators, calendar, smc) = dataset_slices();
        prices[2] = f64::NAN;
        let view = PopulationDatasetView {
            close: &prices,
            high: &prices,
            low: &prices,
            indicators: &indicators,
            feature_count: 2,
            months: &calendar,
            days: &calendar,
            timestamps: &calendar,
            smc_rows: &smc,
            adaptive_base_pips: None,
        };
        assert!(
            matches!(view.validate(), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("close[2]"))
        );
    }

    #[test]
    fn gene_csr_and_feature_bounds_are_validated_before_ffi() {
        let descriptors = vec![GeneDescriptor::default(); 2];
        let smc_flags = vec![0_i8; 2 * SMC_SLOTS];
        let smc_weights = [0.0_f64; SMC_SLOTS];
        let genes = PopulationGeneView {
            descriptors: &descriptors,
            offsets: &[0, 1, 2],
            indices: &[0, 7],
            weights: &[1.0, 1.0],
            stop_pips: &[10.0, 10.0],
            target_pips: &[20.0, 20.0],
            stop_vol_multipliers: &[0.0, 0.0],
            smc_flags: &smc_flags,
            smc_weights: &smc_weights,
            gate_threshold: 0.0,
            smc_gate_disabled: false,
        };
        assert!(
            matches!(genes.validate(4), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("feature 7"))
        );

        let broken = PopulationGeneView {
            offsets: &[0, 2, 1],
            ..genes
        };
        assert!(
            matches!(broken.validate(8), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("CSR"))
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_reports_runtime_unavailable_for_population_sessions() {
        match PopulationSession::create(0, 1024) {
            Err(error) => assert!(
                error.is_runtime_unavailable(),
                "a CUDA-free build must report an unavailable runtime, got {error}"
            ),
            Ok(_) => panic!("a CUDA-free build must not create a native population session"),
        }
    }
}
