//! Native variable-width Higher-Timeframe Alignment semantic-v3.
//!
//! Direct-timeframe parent feature batches move into this executor together
//! with their resident OHLCV/timestamp source. Alignment copies only device
//! pointers and compact offsets to CUDA. Feature values and logical validity
//! never make a host round trip. The executor stays move-only and releases all
//! retained parents only after its caller has retired every downstream pack.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cust::context::{Context, CurrentContext};
use cust::memory::{
    AsyncCopyDestination, DeviceBuffer, DeviceCopy, DeviceSlice, GpuBuffer, LockedBuffer,
};
use cust::stream::Stream;
use cust::sys::CUstream;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3,
};
use sha2::{Digest, Sha256};

use crate::resident_classic_ta_v3::{
    ResidentClassicTaExecutorV3, ResidentClassicTaPreDeviceMemoryReceiptV4,
    ResidentClassicTaRecipeV3,
};
use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentF64FeatureBatchV3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3, ResidentParentDatasetSourceV3, ResidentProducerReadyEventV3,
};
use crate::resident_footprint_v2::{
    ResidentFootprintRuntimeReceiptV2, launch_resident_footprint_v2,
};
use crate::resident_quant_v3::{
    ResidentQuantLaunchAuthorityV3, ResidentQuantRuntimeReceiptV3, launch_resident_quant_v3,
};
use crate::resident_regime_v3::{ResidentRegimeRuntimeReceiptV3, launch_resident_regime_v3};
use crate::resident_session_v2::{
    ResidentSessionLaunchAuthorityV2, ResidentSessionRuntimeReceiptV2, launch_resident_session_v2,
};
use crate::resident_smc_v3::ResidentSmcMaterializationV3;

const SHA256_BYTES: usize = 32;
const MAX_NATIVE_BATCH_COLUMNS_V3: usize = 64;
const POINTER_FIELDS_PER_COLUMN_V3: usize = 4;
const LOGICAL_VALIDITY_BYTES_PER_CELL_V3: usize = 1;
const F64_BYTES_V3: usize = std::mem::size_of::<f64>();
const U64_BYTES_V3: usize = std::mem::size_of::<u64>();

pub const RESIDENT_HTF_SEMANTIC_VERSION_V3: u32 = 3;
pub const RESIDENT_HTF_IMPLEMENTATION_ID_V3: &str =
    "neoethos.cuda.resident-higher-timeframe-alignment.semantic-v3";
pub const RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3: &str = "neoethos.higher-timeframe-alignment.cpu-oracle.semantic-v3;direct-source-only;selected-parent-order;cpu-producer-order;fixed-open-plus-period-v1;calendar-next-direct-bar-open-v1;forward-fill=true;fixed-max-age=2x-period;logical-validity-preserved;zero-feature-d2h";
pub const RESIDENT_HTF_CANONICAL_QNAN_BITS_V3: u64 = 0x7ff8_0000_0000_0000;
const RESIDENT_HTF_LOGICAL_VALIDITY_SCHEMA_V3: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";
const CANONICAL_CPU_PRODUCER_ORDER_V3: [ResidentFeatureProducerV3; 6] = [
    ResidentFeatureProducerV3::Smc,
    ResidentFeatureProducerV3::ClassicTa,
    ResidentFeatureProducerV3::Quant,
    ResidentFeatureProducerV3::Session,
    ResidentFeatureProducerV3::Regime,
    ResidentFeatureProducerV3::Footprint,
];

#[repr(u32)]
#[derive(Debug, PartialEq, Eq)]
pub enum ResidentHigherTimeframeAvailabilityRuleV3 {
    FixedOpenPlusPeriod = 1,
    NextDirectBarOpen = 2,
}

impl ResidentHigherTimeframeAvailabilityRuleV3 {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FixedOpenPlusPeriod => "fixed_open_plus_period_v1",
            Self::NextDirectBarOpen => "next_direct_bar_open_v1",
        }
    }

    fn abi_tag(&self) -> u32 {
        match self {
            Self::FixedOpenPlusPeriod => 1,
            Self::NextDirectBarOpen => 2,
        }
    }
}

#[repr(C)]
struct NeoResidentHigherTimeframeParentSegmentV3 {
    first_column: u32,
    column_count: u32,
    availability_rule: u32,
    reserved: u32,
    parent_row_count: u64,
    fixed_period_ms: i64,
    max_age_ms: i64,
    parent_open_ms: *const i64,
}

#[repr(C)]
struct NeoResidentHigherTimeframeLaunchV3 {
    abi_version: u32,
    semantic_version: u32,
    feature_column_count: u32,
    parent_segment_count: u32,
    base_row_count: u64,
    base_open_ms: *const i64,
    source_value_buffers_device: *const *const f64,
    source_validity_buffers_device: *const *const u8,
    source_value_offsets_device: *const u64,
    source_validity_offsets_device: *const u64,
    feature_values: *mut f64,
    feature_validity_u8: *mut u8,
    parent_segments_host: *const NeoResidentHigherTimeframeParentSegmentV3,
}

const _: [(); 48] = [(); std::mem::size_of::<NeoResidentHigherTimeframeParentSegmentV3>()];
const _: [(); 88] = [(); std::mem::size_of::<NeoResidentHigherTimeframeLaunchV3>()];

unsafe extern "C" {
    fn neoethos_resident_higher_timeframe_alignment_f64_v3(
        launch: *const NeoResidentHigherTimeframeLaunchV3,
        stream: CUstream,
    ) -> i32;
}

/// Move-only proof that Rust, ABI, CUDA and the current CPU oracle were all in
/// the implementation source closure. Creating this does not advertise the
/// capability in the production manifest.
#[must_use = "HTF-v3 source closure must move into one launch authority"]
#[derive(Debug)]
pub struct SealedResidentHigherTimeframeSourceClosureV3 {
    implementation_sha256: [u8; SHA256_BYTES],
}

impl SealedResidentHigherTimeframeSourceClosureV3 {
    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }
}

pub fn seal_resident_higher_timeframe_source_closure_v3()
-> SealedResidentHigherTimeframeSourceClosureV3 {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-htf.f64.semantic-v3\0");
    implementation.update(include_bytes!("resident_higher_timeframe_alignment_v3.rs"));
    implementation.update(include_bytes!(
        "../native/resident_higher_timeframe_alignment_v3_abi.cuh"
    ));
    implementation.update(include_bytes!(
        "../native/resident_higher_timeframe_alignment_v3.cu"
    ));
    implementation.update(include_bytes!("../../neoethos-data/src/core/features.rs"));
    implementation.update(include_bytes!("../../neoethos-data/src/lib.rs"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs"
    ));
    implementation.update(RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3.as_bytes());
    SealedResidentHigherTimeframeSourceClosureV3 {
        implementation_sha256: implementation.finalize().into(),
    }
}

pub fn resident_higher_timeframe_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let closure = seal_resident_higher_timeframe_source_closure_v3();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::HigherTimeframeAlignment,
        RESIDENT_HTF_IMPLEMENTATION_ID_V3,
        closure.implementation_sha256(),
        RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3,
    )
    .map_err(Into::into)
}

#[derive(Debug)]
pub struct ResidentHigherTimeframeRouteAuthorityV3 {
    source_feature_name: String,
    source_producer: ResidentFeatureProducerV3,
    source_route_receipt_sha256: [u8; SHA256_BYTES],
    output_route_receipt_sha256: [u8; SHA256_BYTES],
}

impl ResidentHigherTimeframeRouteAuthorityV3 {
    pub fn seal(
        source_feature_name: impl Into<String>,
        source_producer: ResidentFeatureProducerV3,
        source_route_receipt_sha256: [u8; SHA256_BYTES],
        output_route_receipt_sha256: [u8; SHA256_BYTES],
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let source_feature_name = source_feature_name.into();
        if source_feature_name.trim().is_empty()
            || !CANONICAL_CPU_PRODUCER_ORDER_V3.contains(&source_producer)
            || source_route_receipt_sha256 == [0; SHA256_BYTES]
            || output_route_receipt_sha256 == [0; SHA256_BYTES]
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF route authority is incomplete".into(),
            ));
        }
        Ok(Self {
            source_feature_name,
            source_producer,
            source_route_receipt_sha256,
            output_route_receipt_sha256,
        })
    }
}

/// Data-sealed description of one retained direct parent. It carries no raw
/// pointer and cannot substitute for the actual moved native carrier.
#[derive(Debug)]
pub struct ResidentHigherTimeframeParentAuthorityV3 {
    timeframe: String,
    parent_row_count: usize,
    availability_rule: ResidentHigherTimeframeAvailabilityRuleV3,
    fixed_period_ms: i64,
    max_age_ms: i64,
    forward_fill: bool,
    source_binding_sha256: [u8; SHA256_BYTES],
    parent_store_identity_sha256: [u8; SHA256_BYTES],
    retained_parent_device_bytes: usize,
    parent_context_process_token: [u8; SHA256_BYTES],
    parent_stream_process_token: [u8; SHA256_BYTES],
    routes: Vec<ResidentHigherTimeframeRouteAuthorityV3>,
}

impl ResidentHigherTimeframeParentAuthorityV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        timeframe: impl Into<String>,
        parent_row_count: usize,
        availability_rule: ResidentHigherTimeframeAvailabilityRuleV3,
        fixed_period_ms: i64,
        max_age_ms: i64,
        source_binding_sha256: [u8; SHA256_BYTES],
        parent_store_identity_sha256: [u8; SHA256_BYTES],
        retained_parent_device_bytes: usize,
        parent_context_process_token: [u8; SHA256_BYTES],
        parent_stream_process_token: [u8; SHA256_BYTES],
        routes: Vec<ResidentHigherTimeframeRouteAuthorityV3>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let timeframe = timeframe.into();
        if timeframe.trim().is_empty()
            || parent_row_count == 0
            || retained_parent_device_bytes == 0
            || routes.is_empty()
            || source_binding_sha256 == [0; SHA256_BYTES]
            || parent_store_identity_sha256 == [0; SHA256_BYTES]
            || parent_context_process_token == [0; SHA256_BYTES]
            || parent_stream_process_token == [0; SHA256_BYTES]
            || parent_context_process_token == parent_stream_process_token
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF parent authority is incomplete".into(),
            ));
        }
        match &availability_rule {
            ResidentHigherTimeframeAvailabilityRuleV3::FixedOpenPlusPeriod => {
                if fixed_period_ms <= 0 || max_age_ms != fixed_period_ms.saturating_mul(2) {
                    return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "fixed HTF availability must bind period and saturated 2x max-age".into(),
                    ));
                }
            }
            ResidentHigherTimeframeAvailabilityRuleV3::NextDirectBarOpen => {
                if fixed_period_ms != 0 || max_age_ms != -1 {
                    return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "calendar HTF availability must bind the observed next direct open".into(),
                    ));
                }
            }
        }
        validate_cpu_producer_order_v3(&routes)?;
        Ok(Self {
            timeframe,
            parent_row_count,
            availability_rule,
            fixed_period_ms,
            max_age_ms,
            forward_fill: true,
            source_binding_sha256,
            parent_store_identity_sha256,
            retained_parent_device_bytes,
            parent_context_process_token,
            parent_stream_process_token,
            routes,
        })
    }
}

fn validate_cpu_producer_order_v3(
    routes: &[ResidentHigherTimeframeRouteAuthorityV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let mut cursor = 0;
    let mut names = BTreeSet::new();
    for expected in CANONICAL_CPU_PRODUCER_ORDER_V3 {
        let start = cursor;
        while cursor < routes.len() && routes[cursor].source_producer == expected {
            if !names.insert(routes[cursor].source_feature_name.as_str()) {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                    "resident HTF repeats parent route `{}`",
                    routes[cursor].source_feature_name
                )));
            }
            cursor += 1;
        }
        if cursor == start {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident HTF parent omits canonical producer {}",
                expected.as_str()
            )));
        }
    }
    if cursor != routes.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident HTF parent route order has a noncanonical tail".into(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub struct ResidentHigherTimeframeLaunchAuthorityV3 {
    base_row_count: usize,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
    selected_parent_order: String,
    canonical_cpu_producer_order: String,
    base_context_process_token: [u8; SHA256_BYTES],
    base_stream_process_token: [u8; SHA256_BYTES],
}

impl ResidentHigherTimeframeLaunchAuthorityV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        base_row_count: usize,
        input_identity_sha256: [u8; SHA256_BYTES],
        semantic_source_sha256: [u8; SHA256_BYTES],
        selected_parent_order: impl Into<String>,
        canonical_cpu_producer_order: impl Into<String>,
        base_context_process_token: [u8; SHA256_BYTES],
        base_stream_process_token: [u8; SHA256_BYTES],
        closure: SealedResidentHigherTimeframeSourceClosureV3,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let selected_parent_order = selected_parent_order.into();
        let canonical_cpu_producer_order = canonical_cpu_producer_order.into();
        if base_row_count == 0
            || input_identity_sha256 == [0; SHA256_BYTES]
            || semantic_source_sha256 == [0; SHA256_BYTES]
            || closure.implementation_sha256() == [0; SHA256_BYTES]
            || selected_parent_order.trim().is_empty()
            || canonical_cpu_producer_order.trim().is_empty()
            || base_context_process_token == [0; SHA256_BYTES]
            || base_stream_process_token == [0; SHA256_BYTES]
            || base_context_process_token == base_stream_process_token
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF launch authority is incomplete".into(),
            ));
        }
        Ok(Self {
            base_row_count,
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256: closure.implementation_sha256(),
            selected_parent_order,
            canonical_cpu_producer_order,
            base_context_process_token,
            base_stream_process_token,
        })
    }
}

/// Complete direct-parent launch description. Data seals the producer-owned
/// authorities and globally admitted bindings; gpu-cuda keeps every concrete
/// feature batch opaque while it executes them on one admitted stream.
#[must_use = "the direct-parent launch plan must move into one HTF capture"]
#[derive(Debug)]
pub struct ResidentHigherTimeframeDirectParentLaunchPlanV3 {
    classic_recipe: ResidentClassicTaRecipeV3,
    classic_bindings: Vec<ResidentFeatureColumnBindingV3>,
    classic_pre_device_memory_receipt: ResidentClassicTaPreDeviceMemoryReceiptV4,
    quant_bindings: Vec<ResidentFeatureColumnBindingV3>,
    quant_launch_authority: ResidentQuantLaunchAuthorityV3,
    session_bindings: Vec<ResidentFeatureColumnBindingV3>,
    session_launch_authority: ResidentSessionLaunchAuthorityV2,
    regime_bindings: Vec<ResidentFeatureColumnBindingV3>,
    regime_scale_anchor: f64,
    footprint_bindings: Vec<ResidentFeatureColumnBindingV3>,
}

impl ResidentHigherTimeframeDirectParentLaunchPlanV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        classic_recipe: ResidentClassicTaRecipeV3,
        classic_bindings: Vec<ResidentFeatureColumnBindingV3>,
        classic_pre_device_memory_receipt: ResidentClassicTaPreDeviceMemoryReceiptV4,
        quant_bindings: Vec<ResidentFeatureColumnBindingV3>,
        quant_launch_authority: ResidentQuantLaunchAuthorityV3,
        session_bindings: Vec<ResidentFeatureColumnBindingV3>,
        session_launch_authority: ResidentSessionLaunchAuthorityV2,
        regime_bindings: Vec<ResidentFeatureColumnBindingV3>,
        regime_scale_anchor: f64,
        footprint_bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if [
            classic_bindings.as_slice(),
            quant_bindings.as_slice(),
            session_bindings.as_slice(),
            regime_bindings.as_slice(),
            footprint_bindings.as_slice(),
        ]
        .iter()
        .any(|bindings| bindings.is_empty())
            || !regime_scale_anchor.is_finite()
            || regime_scale_anchor <= 0.0
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF direct-parent launch plan is incomplete".into(),
            ));
        }
        Ok(Self {
            classic_recipe,
            classic_bindings,
            classic_pre_device_memory_receipt,
            quant_bindings,
            quant_launch_authority,
            session_bindings,
            session_launch_authority,
            regime_bindings,
            regime_scale_anchor,
            footprint_bindings,
        })
    }
}

/// Immutable host descriptor cloned only from one already-live resident batch.
/// It contains route identity, never a value/validity array or raw pointer.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentHigherTimeframeCapturedRouteV3 {
    producer: ResidentFeatureProducerV3,
    binding: ResidentFeatureColumnBindingV3,
}

impl ResidentHigherTimeframeCapturedRouteV3 {
    pub const fn producer(&self) -> ResidentFeatureProducerV3 {
        self.producer
    }

    pub const fn binding(&self) -> &ResidentFeatureColumnBindingV3 {
        &self.binding
    }
}

#[derive(Debug)]
struct ResidentHigherTimeframeTaggedBatchV3 {
    producer: ResidentFeatureProducerV3,
    batch: Option<Box<dyn ResidentF64FeatureBatchV3>>,
}

impl ResidentHigherTimeframeTaggedBatchV3 {
    fn into_batch(mut self) -> Box<dyn ResidentF64FeatureBatchV3> {
        self.batch
            .take()
            .expect("live HTF tagged batch retains its native owner")
    }
}

impl Drop for ResidentHigherTimeframeTaggedBatchV3 {
    fn drop(&mut self) {
        if let Some(batch) = self.batch.take() {
            std::mem::forget(batch);
        }
    }
}

/// Opaque, move-only direct-parent capture. The resident parent and every
/// feature-major producer batch stay owned here until Data supplies the exact
/// route/availability authority and consumes the capture into native HTF.
#[must_use = "the captured direct parent must move into native HTF alignment"]
#[derive(Debug)]
pub struct PendingResidentHigherTimeframeDirectParentCaptureV3 {
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    producer_batches: Option<Vec<ResidentHigherTimeframeTaggedBatchV3>>,
    route_descriptors: Vec<ResidentHigherTimeframeCapturedRouteV3>,
    feature_names: BTreeSet<String>,
    present_producers: BTreeSet<ResidentFeatureProducerV3>,
    previous_producer_rank: Option<usize>,
    rows: usize,
    device_ordinal: u32,
    context_process_token: [u8; SHA256_BYTES],
    stream_process_token: [u8; SHA256_BYTES],
    retained_device_bytes: usize,
    quant_runtime_receipt: Option<ResidentQuantRuntimeReceiptV3>,
    session_runtime_receipt: Option<ResidentSessionRuntimeReceiptV2>,
    regime_runtime_receipt: Option<ResidentRegimeRuntimeReceiptV3>,
    footprint_runtime_receipt: Option<ResidentFootprintRuntimeReceiptV2>,
}

impl PendingResidentHigherTimeframeDirectParentCaptureV3 {
    fn new(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        parent_source: Box<dyn ResidentParentDatasetSourceV3>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let rows = parent_source.rows();
        let device_ordinal = parent_source.device_ordinal();
        let retained_device_bytes = parent_source.retained_device_bytes();
        let capture = Self {
            parent_source: Some(parent_source),
            producer_batches: Some(Vec::new()),
            route_descriptors: Vec::new(),
            feature_names: BTreeSet::new(),
            present_producers: BTreeSet::new(),
            previous_producer_rank: None,
            rows,
            device_ordinal,
            context_process_token: run_device.device_identity().primary_context_process_token(),
            stream_process_token: run_device.run_stream_process_token_v3(),
            retained_device_bytes,
            quant_runtime_receipt: None,
            session_runtime_receipt: None,
            regime_runtime_receipt: None,
            footprint_runtime_receipt: None,
        };
        let parent = capture
            .parent_source
            .as_deref()
            .expect("new HTF capture retains its parent source");
        if rows == 0
            || retained_device_bytes == 0
            || parent.timestamps().len() != rows
            || device_ordinal != run_device.device_identity().ordinal()
            || parent.producer_context().as_raw()
                != run_device
                    .primary_context_for_resident_producer_v3()
                    .as_raw()
            || parent.producer_stream().as_inner()
                != run_device.run_stream_for_resident_producer_v3().as_inner()
            || parent.producer_stream().as_inner().is_null()
            || capture.context_process_token == [0; SHA256_BYTES]
            || capture.stream_process_token == [0; SHA256_BYTES]
            || capture.context_process_token == capture.stream_process_token
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF captured parent shape/context/stream is invalid".into(),
            ));
        }
        Ok(capture)
    }

    fn parent_source(&self) -> &dyn ResidentParentDatasetSourceV3 {
        self.parent_source
            .as_deref()
            .expect("live HTF capture retains its parent source")
    }

    fn push_batch(
        &mut self,
        producer: ResidentFeatureProducerV3,
        batch: Box<dyn ResidentF64FeatureBatchV3>,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let producer_rank = CANONICAL_CPU_PRODUCER_ORDER_V3
            .iter()
            .position(|candidate| *candidate == producer)
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident HTF captured a noncanonical producer".into(),
                )
            })?;
        let bindings = batch.column_bindings();
        let expected_first_ordinal = self.route_descriptors.len();
        let mut batch_feature_names = BTreeSet::new();
        let valid = !bindings.is_empty()
            && bindings.len() <= MAX_NATIVE_BATCH_COLUMNS_V3
            && batch.rows() == self.rows
            && batch.device_ordinal() == self.device_ordinal
            && batch.producer_context().as_raw()
                == self.parent_source().producer_context().as_raw()
            && batch.producer_stream().as_inner()
                == self.parent_source().producer_stream().as_inner()
            && self
                .previous_producer_rank
                .is_none_or(|previous| producer_rank >= previous)
            && bindings.iter().enumerate().all(|(local_column, binding)| {
                expected_first_ordinal
                    .checked_add(local_column)
                    .is_some_and(|ordinal| binding.ordinal == ordinal)
                    && !binding.feature_name.trim().is_empty()
                    && binding.canonical_parameter_tuple_sha256 != [0; SHA256_BYTES]
                    && binding.route_receipt_sha256 != [0; SHA256_BYTES]
                    && !self.feature_names.contains(binding.feature_name.as_str())
                    && batch_feature_names.insert(binding.feature_name.as_str())
            });
        if !valid {
            std::mem::forget(batch);
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF captured batch route/shape/order drifted".into(),
            ));
        }
        let Some(next_retained_device_bytes) = self
            .retained_device_bytes
            .checked_add(batch.retained_device_bytes())
            .and_then(|bytes| bytes.checked_add(batch.retained_scratch_bytes()))
        else {
            std::mem::forget(batch);
            return Err(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF captured retained bytes",
            ));
        };
        for binding in bindings {
            self.feature_names.insert(binding.feature_name.clone());
            self.route_descriptors
                .push(ResidentHigherTimeframeCapturedRouteV3 {
                    producer,
                    binding: binding.clone(),
                });
        }
        self.present_producers.insert(producer);
        self.previous_producer_rank = Some(producer_rank);
        self.retained_device_bytes = next_retained_device_bytes;
        self.producer_batches
            .as_mut()
            .expect("live HTF capture retains its producer batches")
            .push(ResidentHigherTimeframeTaggedBatchV3 {
                producer,
                batch: Some(batch),
            });
        Ok(())
    }

    fn validate_complete(&self) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let mut route_cursor = 0_usize;
        let tagged_batches_are_exact = self.producer_batches.as_deref().is_some_and(|batches| {
            !batches.is_empty()
                && batches.iter().all(|tagged| {
                    let Some(batch) = tagged.batch.as_deref() else {
                        return false;
                    };
                    let next_route_cursor =
                        match route_cursor.checked_add(batch.column_bindings().len()) {
                            Some(next) => next,
                            None => return false,
                        };
                    let exact = self
                        .route_descriptors
                        .get(route_cursor..next_route_cursor)
                        .is_some_and(|routes| {
                            routes.iter().all(|route| route.producer == tagged.producer)
                        });
                    route_cursor = next_route_cursor;
                    exact
                })
                && route_cursor == self.route_descriptors.len()
        });
        if self.route_descriptors.is_empty()
            || self.present_producers
                != CANONICAL_CPU_PRODUCER_ORDER_V3
                    .into_iter()
                    .collect::<BTreeSet<_>>()
            || !tagged_batches_are_exact
            || self.quant_runtime_receipt.is_none()
            || self.session_runtime_receipt.is_none()
            || self.regime_runtime_receipt.is_none()
            || self.footprint_runtime_receipt.is_none()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF direct-parent capture omitted a canonical producer".into(),
            ));
        }
        Ok(())
    }

    pub fn route_descriptors(&self) -> &[ResidentHigherTimeframeCapturedRouteV3] {
        &self.route_descriptors
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    pub const fn retained_device_bytes(&self) -> usize {
        self.retained_device_bytes
    }

    pub const fn context_process_token(&self) -> [u8; SHA256_BYTES] {
        self.context_process_token
    }

    pub const fn stream_process_token(&self) -> [u8; SHA256_BYTES] {
        self.stream_process_token
    }

    pub fn quant_runtime_receipt(&self) -> &ResidentQuantRuntimeReceiptV3 {
        self.quant_runtime_receipt
            .as_ref()
            .expect("complete HTF capture retains its Quant-v3 receipt")
    }

    pub fn session_runtime_receipt(&self) -> &ResidentSessionRuntimeReceiptV2 {
        self.session_runtime_receipt
            .as_ref()
            .expect("complete HTF capture retains its Session-v2 receipt")
    }

    pub fn regime_runtime_receipt(&self) -> &ResidentRegimeRuntimeReceiptV3 {
        self.regime_runtime_receipt
            .as_ref()
            .expect("complete HTF capture retains its Regime-v3 receipt")
    }

    pub fn footprint_runtime_receipt(&self) -> &ResidentFootprintRuntimeReceiptV2 {
        self.footprint_runtime_receipt
            .as_ref()
            .expect("complete HTF capture retains its Footprint-v2 receipt")
    }

    pub fn into_direct_parent(
        mut self,
        authority: ResidentHigherTimeframeParentAuthorityV3,
    ) -> Result<ResidentHigherTimeframeDirectParentV3, ResidentFeatureStoreCudaErrorV3> {
        self.validate_complete()?;
        let parent_source = self
            .parent_source
            .take()
            .expect("live HTF capture retains its parent source");
        let producer_batches = self
            .producer_batches
            .take()
            .expect("live HTF capture retains its producer batches");
        let feature_batches = producer_batches
            .into_iter()
            .map(ResidentHigherTimeframeTaggedBatchV3::into_batch)
            .collect();
        ResidentHigherTimeframeDirectParentV3::seal(authority, parent_source, feature_batches)
    }
}

/// Launch and move all six existing resident producer families into one
/// feature-major direct-parent carrier. This path deliberately never packs a
/// batch and never reads feature or validity arrays back to the host.
pub fn capture_resident_higher_timeframe_direct_parent_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    smc_materialization: ResidentSmcMaterializationV3,
    plan: ResidentHigherTimeframeDirectParentLaunchPlanV3,
) -> Result<PendingResidentHigherTimeframeDirectParentCaptureV3, ResidentFeatureStoreCudaErrorV3> {
    let ResidentHigherTimeframeDirectParentLaunchPlanV3 {
        classic_recipe,
        classic_bindings,
        classic_pre_device_memory_receipt,
        quant_bindings,
        quant_launch_authority,
        session_bindings,
        session_launch_authority,
        regime_bindings,
        regime_scale_anchor,
        footprint_bindings,
    } = plan;
    let (parent_source, smc_batch) = smc_materialization.into_higher_timeframe_parent_parts_v3()?;
    let mut capture =
        match PendingResidentHigherTimeframeDirectParentCaptureV3::new(run_device, parent_source) {
            Ok(capture) => capture,
            Err(error) => {
                std::mem::forget(smc_batch);
                return Err(error);
            }
        };
    capture.push_batch(ResidentFeatureProducerV3::Smc, smc_batch)?;

    let mut classic_executor = ResidentClassicTaExecutorV3::new_v4(
        run_device,
        capture.parent_source(),
        classic_recipe,
        classic_bindings,
        classic_pre_device_memory_receipt,
    )
    .map_err(classic_capture_error_v3)?;
    let shared_derived_input_bytes = classic_executor.retained_derived_input_bytes();
    while let Some(mut classic_batch) = classic_executor
        .next_pending_batch_v3()
        .map_err(classic_capture_error_v3)?
    {
        if let Err(error) =
            classic_batch.detach_shared_derived_input_charge_v3(shared_derived_input_bytes)
        {
            std::mem::forget(classic_batch);
            return Err(classic_capture_error_v3(error));
        }
        capture.push_batch(
            ResidentFeatureProducerV3::ClassicTa,
            Box::new(classic_batch),
        )?;
    }
    drop(classic_executor);

    let quant_batch = launch_resident_quant_v3(
        run_device,
        capture.parent_source(),
        quant_bindings,
        quant_launch_authority,
    )?;
    let quant_runtime_receipt = quant_batch.receipt().clone();
    capture.push_batch(ResidentFeatureProducerV3::Quant, Box::new(quant_batch))?;
    capture.quant_runtime_receipt = Some(quant_runtime_receipt);

    let session_batch = launch_resident_session_v2(
        run_device,
        capture.parent_source(),
        session_bindings,
        session_launch_authority,
    )?;
    let session_runtime_receipt = session_batch.receipt().clone();
    capture.push_batch(ResidentFeatureProducerV3::Session, Box::new(session_batch))?;
    capture.session_runtime_receipt = Some(session_runtime_receipt);

    let regime_batch = launch_resident_regime_v3(
        run_device,
        capture.parent_source(),
        regime_bindings,
        regime_scale_anchor,
    )?;
    let regime_runtime_receipt = regime_batch.receipt().clone();
    capture.push_batch(ResidentFeatureProducerV3::Regime, Box::new(regime_batch))?;
    capture.regime_runtime_receipt = Some(regime_runtime_receipt);

    let footprint_batch =
        launch_resident_footprint_v2(run_device, capture.parent_source(), footprint_bindings)?;
    let footprint_runtime_receipt = footprint_batch.receipt().clone();
    capture.push_batch(
        ResidentFeatureProducerV3::Footprint,
        Box::new(footprint_batch),
    )?;
    capture.footprint_runtime_receipt = Some(footprint_runtime_receipt);
    capture.validate_complete()?;
    Ok(capture)
}

fn classic_capture_error_v3(
    error: crate::resident_classic_ta_v3::ResidentClassicTaExecutorErrorV3,
) -> ResidentFeatureStoreCudaErrorV3 {
    ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
        "resident HTF Classic direct-parent capture failed: {error}"
    ))
}

impl Drop for PendingResidentHigherTimeframeDirectParentCaptureV3 {
    fn drop(&mut self) {
        if let Some(producer_batches) = self.producer_batches.take() {
            std::mem::forget(producer_batches);
        }
        if let Some(parent_source) = self.parent_source.take() {
            std::mem::forget(parent_source);
        }
    }
}

/// Actual direct-timeframe resident owner. The source and every producer batch
/// move in together and are released together only by executor finalization.
#[derive(Debug)]
struct ResidentHigherTimeframeDirectParentConstructionGuardV3 {
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    feature_batches: Option<Vec<Box<dyn ResidentF64FeatureBatchV3>>>,
}

impl ResidentHigherTimeframeDirectParentConstructionGuardV3 {
    fn new(
        parent_source: Box<dyn ResidentParentDatasetSourceV3>,
        feature_batches: Vec<Box<dyn ResidentF64FeatureBatchV3>>,
    ) -> Self {
        Self {
            parent_source: Some(parent_source),
            feature_batches: Some(feature_batches),
        }
    }

    fn parent_source(&self) -> &dyn ResidentParentDatasetSourceV3 {
        self.parent_source
            .as_deref()
            .expect("live HTF construction guard retains its parent source")
    }

    fn feature_batches(&self) -> &[Box<dyn ResidentF64FeatureBatchV3>] {
        self.feature_batches
            .as_deref()
            .expect("live HTF construction guard retains its feature batches")
    }

    fn disarm(
        mut self,
    ) -> (
        Box<dyn ResidentParentDatasetSourceV3>,
        Vec<Box<dyn ResidentF64FeatureBatchV3>>,
    ) {
        let parent_source = self
            .parent_source
            .take()
            .expect("live HTF construction guard retains its parent source");
        let feature_batches = self
            .feature_batches
            .take()
            .expect("live HTF construction guard retains its feature batches");
        (parent_source, feature_batches)
    }
}

impl Drop for ResidentHigherTimeframeDirectParentConstructionGuardV3 {
    fn drop(&mut self) {
        if let Some(feature_batches) = self.feature_batches.take() {
            std::mem::forget(feature_batches);
        }
        if let Some(parent_source) = self.parent_source.take() {
            std::mem::forget(parent_source);
        }
    }
}

#[must_use = "direct HTF parent carrier must move into resident alignment"]
#[derive(Debug)]
pub struct ResidentHigherTimeframeDirectParentV3 {
    authority: ResidentHigherTimeframeParentAuthorityV3,
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    feature_batches: Option<Vec<Box<dyn ResidentF64FeatureBatchV3>>>,
}

impl ResidentHigherTimeframeDirectParentV3 {
    pub fn seal(
        authority: ResidentHigherTimeframeParentAuthorityV3,
        parent_source: Box<dyn ResidentParentDatasetSourceV3>,
        feature_batches: Vec<Box<dyn ResidentF64FeatureBatchV3>>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let guard = ResidentHigherTimeframeDirectParentConstructionGuardV3::new(
            parent_source,
            feature_batches,
        );
        if guard.feature_batches().is_empty()
            || guard.parent_source().rows() != authority.parent_row_count
            || guard.parent_source().timestamps().len() != authority.parent_row_count
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF direct parent shape is incomplete".into(),
            ));
        }
        let parent_context = guard.parent_source().producer_context().as_raw();
        let parent_stream = guard.parent_source().producer_stream().as_inner();
        let parent_device = guard.parent_source().device_ordinal();
        if parent_stream.is_null() {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        let mut route_cursor = 0;
        let mut retained_parent_device_bytes = guard.parent_source().retained_device_bytes();
        for batch in guard.feature_batches() {
            let columns = batch.column_bindings().len();
            if columns == 0
                || columns > MAX_NATIVE_BATCH_COLUMNS_V3
                || batch.rows() != authority.parent_row_count
                || batch.device_ordinal() != parent_device
                || batch.producer_context().as_raw() != parent_context
                || batch.producer_stream().as_inner() != parent_stream
            {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident HTF direct parent batch identity or shape drifted".into(),
                ));
            }
            retained_parent_device_bytes = retained_parent_device_bytes
                .checked_add(batch.retained_device_bytes())
                .and_then(|bytes| bytes.checked_add(batch.retained_scratch_bytes()))
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF retained parent device bytes",
                ))?;
            for binding in batch.column_bindings() {
                let route = authority.routes.get(route_cursor).ok_or_else(|| {
                    ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "resident HTF direct parent emitted more routes than its authority".into(),
                    )
                })?;
                if binding.feature_name != route.source_feature_name
                    || binding.route_receipt_sha256 != route.source_route_receipt_sha256
                {
                    return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "resident HTF direct parent route binding drifted".into(),
                    ));
                }
                route_cursor += 1;
            }
            for local_column in 0..columns {
                validate_source_column_extent_v3(
                    batch.as_ref(),
                    local_column,
                    authority.parent_row_count,
                )?;
            }
        }
        if route_cursor != authority.routes.len()
            || retained_parent_device_bytes != authority.retained_parent_device_bytes
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF direct parent route or retained-byte receipt drifted".into(),
            ));
        }
        let (parent_source, feature_batches) = guard.disarm();
        Ok(Self {
            authority,
            parent_source: Some(parent_source),
            feature_batches: Some(feature_batches),
        })
    }

    fn parent_source(&self) -> &dyn ResidentParentDatasetSourceV3 {
        self.parent_source
            .as_deref()
            .expect("live HTF carrier retains its direct parent source")
    }

    fn feature_batches(&self) -> &[Box<dyn ResidentF64FeatureBatchV3>] {
        self.feature_batches
            .as_deref()
            .expect("live HTF carrier retains its direct parent feature batches")
    }

    fn enqueue_nonblocking_release(
        mut self,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let mut batches = self
            .feature_batches
            .take()
            .expect("live HTF carrier retains feature batches");
        while let Some(batch) = batches.pop() {
            if let Err(error) = batch.enqueue_nonblocking_release(release_stream) {
                std::mem::forget(batches);
                if let Some(parent_source) = self.parent_source.take() {
                    std::mem::forget(parent_source);
                }
                return Err(error);
            }
        }
        self.parent_source
            .take()
            .expect("live HTF carrier retains parent source")
            .enqueue_nonblocking_release(release_stream)
    }
}

impl Drop for ResidentHigherTimeframeDirectParentV3 {
    fn drop(&mut self) {
        if let Some(feature_batches) = self.feature_batches.take() {
            std::mem::forget(feature_batches);
        }
        if let Some(parent_source) = self.parent_source.take() {
            std::mem::forget(parent_source);
        }
    }
}

fn validate_source_column_extent_v3(
    batch: &dyn ResidentF64FeatureBatchV3,
    column: usize,
    rows: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let value_end = batch.value_offset(column).checked_add(rows).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF source value extent"),
    )?;
    let validity_end = batch.validity_offset(column).checked_add(rows).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF source validity extent"),
    )?;
    if value_end > batch.value_buffer(column).len()
        || validity_end > batch.validity_buffer(column).len()
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident HTF source column extent is shorter than its parent rows".into(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct StreamOrderedHigherTimeframeBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedHigherTimeframeBufferV3<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let buffer = unsafe { DeviceBuffer::uninitialized_async(len, stream.as_ref())? };
        Ok(Self {
            buffer: Some(buffer),
            context,
            stream,
        })
    }

    fn is_owned_by_stream(&self, stream: &Stream) -> bool {
        !stream.as_inner().is_null() && self.stream.as_inner() == stream.as_inner()
    }
}

impl<T: DeviceCopy> Deref for StreamOrderedHigherTimeframeBufferV3<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live HTF owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedHigherTimeframeBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live HTF owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedHigherTimeframeBufferV3<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(self.stream.as_ref());
        } else {
            std::mem::forget(buffer);
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeFeatureBatchV3 {
    feature_values: StreamOrderedHigherTimeframeBufferV3<f64>,
    feature_validity_u8: StreamOrderedHigherTimeframeBufferV3<u8>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    _pointer_host: LockedBuffer<u64>,
    retained_feature_device_bytes: usize,
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentHigherTimeframeFeatureBatchV3 {
    fn column_bindings(&self) -> &[ResidentFeatureColumnBindingV3] {
        &self.bindings
    }

    fn value_buffer(&self, _column: usize) -> &DeviceBuffer<f64> {
        &self.feature_values
    }

    fn validity_buffer(&self, _column: usize) -> &DeviceBuffer<u8> {
        &self.feature_validity_u8
    }

    fn value_offset(&self, column: usize) -> usize {
        column * self.rows
    }

    fn validity_offset(&self, column: usize) -> usize {
        column * self.rows
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    fn producer_context(&self) -> &Context {
        self.context.as_ref()
    }

    fn producer_stream(&self) -> &Stream {
        self.stream.as_ref()
    }

    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3 {
        &self.ready_event
    }

    fn retained_device_bytes(&self) -> usize {
        self.retained_feature_device_bytes
    }

    fn retained_scratch_bytes(&self) -> usize {
        0
    }

    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.feature_values.is_owned_by_stream(release_stream)
            || !self.feature_validity_u8.is_owned_by_stream(release_stream)
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentHigherTimeframeRuntimeReceiptV3 {
    semantic_version: u32,
    base_row_count: usize,
    parent_count: usize,
    parent_feature_column_count: usize,
    retained_feature_device_bytes: usize,
    retained_parent_device_bytes: usize,
    scratch_device_bytes: usize,
    pointer_table_device_bytes: usize,
    pointer_table_h2d_bytes: usize,
    isolated_pointer_schema_metadata_bytes: usize,
    parent_feature_h2d_bytes: usize,
    feature_value_d2h_bytes: usize,
    feature_validity_d2h_bytes: usize,
    /// Number of synchronous C-ABI batch submissions. This is the frozen Data
    /// owner's `native_launch_count` authority.
    native_launch_count: usize,
    /// Exact CUDA kernels submitted by those ABI batches (one per contiguous
    /// parent-clock segment, including batches that cross parent boundaries).
    native_kernel_launch_count: usize,
    producer_ready_event_count: usize,
    producer_ready_event_synchronize_count: usize,
    host_synchronize_count: usize,
    logical_validity_schema: &'static str,
    logical_validity_codes: [u8; 10],
    canonical_qnan_bits: u64,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
    selected_parent_order: String,
    canonical_cpu_producer_order: String,
    base_context_process_token: [u8; SHA256_BYTES],
    base_stream_process_token: [u8; SHA256_BYTES],
}

impl ResidentHigherTimeframeRuntimeReceiptV3 {
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }

    pub const fn base_row_count(&self) -> usize {
        self.base_row_count
    }

    pub const fn parent_count(&self) -> usize {
        self.parent_count
    }

    pub const fn parent_feature_column_count(&self) -> usize {
        self.parent_feature_column_count
    }

    pub const fn retained_feature_device_bytes(&self) -> usize {
        self.retained_feature_device_bytes
    }

    pub const fn retained_parent_device_bytes(&self) -> usize {
        self.retained_parent_device_bytes
    }

    pub const fn scratch_device_bytes(&self) -> usize {
        self.scratch_device_bytes
    }

    pub const fn pointer_table_device_bytes(&self) -> usize {
        self.pointer_table_device_bytes
    }

    pub const fn pointer_table_h2d_bytes(&self) -> usize {
        self.pointer_table_h2d_bytes
    }

    pub const fn isolated_pointer_schema_metadata_bytes(&self) -> usize {
        self.isolated_pointer_schema_metadata_bytes
    }

    pub const fn parent_feature_h2d_bytes(&self) -> usize {
        self.parent_feature_h2d_bytes
    }

    pub const fn feature_value_d2h_bytes(&self) -> usize {
        self.feature_value_d2h_bytes
    }

    pub const fn feature_validity_d2h_bytes(&self) -> usize {
        self.feature_validity_d2h_bytes
    }

    pub const fn native_launch_count(&self) -> usize {
        self.native_launch_count
    }

    pub const fn native_kernel_launch_count(&self) -> usize {
        self.native_kernel_launch_count
    }

    pub const fn producer_ready_event_count(&self) -> usize {
        self.producer_ready_event_count
    }

    pub const fn producer_ready_event_synchronize_count(&self) -> usize {
        self.producer_ready_event_synchronize_count
    }

    pub const fn host_synchronize_count(&self) -> usize {
        self.host_synchronize_count
    }

    pub const fn logical_validity_schema(&self) -> &'static str {
        self.logical_validity_schema
    }

    pub const fn logical_validity_codes(&self) -> [u8; 10] {
        self.logical_validity_codes
    }

    pub const fn canonical_qnan_bits(&self) -> u64 {
        self.canonical_qnan_bits
    }

    pub const fn input_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.input_identity_sha256
    }

    pub const fn semantic_source_sha256(&self) -> [u8; SHA256_BYTES] {
        self.semantic_source_sha256
    }

    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }

    pub fn selected_parent_order(&self) -> &str {
        &self.selected_parent_order
    }

    pub fn canonical_cpu_producer_order(&self) -> &str {
        &self.canonical_cpu_producer_order
    }

    pub const fn base_context_process_token(&self) -> [u8; SHA256_BYTES] {
        self.base_context_process_token
    }

    pub const fn base_stream_process_token(&self) -> [u8; SHA256_BYTES] {
        self.base_stream_process_token
    }
}

/// Sequential-batch native owner. Callers must append and event-retire each
/// returned batch before requesting the next one, matching the store assembler
/// contract already used by resident Classic TA.
#[derive(Debug)]
struct ResidentHigherTimeframeColumnSourceV3 {
    parent_index: usize,
    batch_index: usize,
    local_column: usize,
}

#[must_use = "HTF executor must be finished after every pack retires"]
#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeExecutorV3 {
    parents: Option<Vec<ResidentHigherTimeframeDirectParentV3>>,
    column_sources: Vec<ResidentHigherTimeframeColumnSourceV3>,
    output_bindings: Vec<Option<ResidentFeatureColumnBindingV3>>,
    authority: Option<ResidentHigherTimeframeLaunchAuthorityV3>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    device_ordinal: u32,
    // The crate-private assembler retains its boxed base parent throughout the
    // executor loop; no external constructor can detach this pointer from that
    // owner. Every direct higher parent is owned above instead of borrowed.
    base_open_ms: *const i64,
    base_row_count: usize,
    pointer_table: StreamOrderedHigherTimeframeBufferV3<u64>,
    next_output_binding: usize,
    parent_feature_column_count: usize,
    retained_feature_device_bytes: usize,
    retained_parent_device_bytes: usize,
    pointer_table_device_bytes: usize,
    pointer_table_h2d_bytes: usize,
    isolated_pointer_schema_metadata_bytes: usize,
    native_launch_count: usize,
    native_kernel_launch_count: usize,
    producer_ready_event_count: usize,
}

impl ResidentHigherTimeframeExecutorV3 {
    pub(crate) fn new(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        base_parent: &dyn ResidentParentDatasetSourceV3,
        parents: Vec<ResidentHigherTimeframeDirectParentV3>,
        output_bindings: Vec<ResidentFeatureColumnBindingV3>,
        authority: ResidentHigherTimeframeLaunchAuthorityV3,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if parents.is_empty()
            || output_bindings.is_empty()
            || base_parent.rows() != authority.base_row_count
            || base_parent.timestamps().len() != authority.base_row_count
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF base/parent/output shape is incomplete".into(),
            ));
        }
        let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
        let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
        let device_ordinal = run_device.device_identity().ordinal();
        CurrentContext::set_current(context.as_ref())?;
        let base_context_process_token =
            run_device.device_identity().primary_context_process_token();
        let base_stream_process_token = run_device.run_stream_process_token_v3();
        if stream.as_inner().is_null()
            || base_parent.device_ordinal() != device_ordinal
            || base_parent.producer_context().as_raw() != context.as_raw()
            || base_parent.producer_stream().as_inner() != stream.as_inner()
            || authority.base_context_process_token != base_context_process_token
            || authority.base_stream_process_token != base_stream_process_token
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        base_parent.producer_ready_event().wait_before_read(
            context.as_ref(),
            stream.as_ref(),
            device_ordinal,
        )?;

        let selected_parent_order = parents
            .iter()
            .map(|parent| parent.authority.timeframe.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let canonical_cpu_producer_order = CANONICAL_CPU_PRODUCER_ORDER_V3
            .iter()
            .map(|producer| producer.as_str())
            .collect::<Vec<_>>()
            .join(",");
        if authority.selected_parent_order != selected_parent_order
            || authority.canonical_cpu_producer_order != canonical_cpu_producer_order
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF parent or CPU producer order drifted".into(),
            ));
        }

        let mut parent_feature_column_count = 0usize;
        let mut retained_parent_device_bytes = 0usize;
        let mut expected_outputs = Vec::new();
        let mut column_sources = Vec::new();
        for (parent_index, parent) in parents.iter().enumerate() {
            if parent.parent_source().device_ordinal() != device_ordinal
                || parent.parent_source().producer_context().as_raw() != context.as_raw()
                || parent.parent_source().producer_stream().as_inner() != stream.as_inner()
                || parent.authority.parent_context_process_token != base_context_process_token
                || parent.authority.parent_stream_process_token != base_stream_process_token
                || parent.authority.source_binding_sha256 == [0; SHA256_BYTES]
                || parent.authority.parent_store_identity_sha256 == [0; SHA256_BYTES]
                || !parent.authority.forward_fill
            {
                return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
            }
            parent
                .parent_source()
                .producer_ready_event()
                .wait_before_read(context.as_ref(), stream.as_ref(), device_ordinal)?;
            retained_parent_device_bytes = retained_parent_device_bytes
                .checked_add(parent.authority.retained_parent_device_bytes)
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF all retained parent bytes",
                ))?;
            for (batch_index, batch) in parent.feature_batches().iter().enumerate() {
                column_sources.extend((0..batch.column_bindings().len()).map(|local_column| {
                    ResidentHigherTimeframeColumnSourceV3 {
                        parent_index,
                        batch_index,
                        local_column,
                    }
                }));
            }
            parent_feature_column_count = parent_feature_column_count
                .checked_add(parent.authority.routes.len())
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF parent feature columns",
                ))?;
            expected_outputs.extend(parent.authority.routes.iter().map(|route| {
                (
                    format!(
                        "{}_{}",
                        parent.authority.timeframe, route.source_feature_name
                    ),
                    route.output_route_receipt_sha256,
                )
            }));
        }
        if expected_outputs.len() != output_bindings.len()
            || parent_feature_column_count != output_bindings.len()
            || column_sources.len() != output_bindings.len()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF admitted output census differs from live parent bindings".into(),
            ));
        }
        let first_ordinal = output_bindings[0].ordinal;
        for (index, (binding, (expected_name, expected_receipt))) in
            output_bindings.iter().zip(&expected_outputs).enumerate()
        {
            let expected_ordinal = first_ordinal.checked_add(index).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF admitted global ordinal",
                ),
            )?;
            if binding.ordinal != expected_ordinal
                || binding.feature_name != *expected_name
                || binding.route_receipt_sha256 != *expected_receipt
                || binding.canonical_parameter_tuple_sha256 == [0; SHA256_BYTES]
            {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident HTF output route binding differs from admission".into(),
                ));
            }
        }
        let max_batch_columns = output_bindings.len().min(MAX_NATIVE_BATCH_COLUMNS_V3);
        let mut max_isolated_pointer_schema_metadata_bytes = 0usize;
        for batch in output_bindings.chunks(MAX_NATIVE_BATCH_COLUMNS_V3) {
            let pointer_bytes = batch
                .len()
                .checked_mul(POINTER_FIELDS_PER_COLUMN_V3)
                .and_then(|entries| entries.checked_mul(U64_BYTES_V3))
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF pointer-table bytes",
                ))?;
            let name_offset_bytes = batch
                .len()
                .checked_add(1)
                .and_then(|count| count.checked_mul(U64_BYTES_V3))
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF route-name offset bytes",
                ))?;
            let name_bytes = batch.iter().try_fold(0usize, |sum, binding| {
                sum.checked_add(binding.feature_name.len()).ok_or(
                    ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                        "resident HTF route-name bytes",
                    ),
                )
            })?;
            max_isolated_pointer_schema_metadata_bytes = max_isolated_pointer_schema_metadata_bytes
                .max(
                    pointer_bytes
                        .checked_add(name_offset_bytes)
                        .and_then(|bytes| bytes.checked_add(name_bytes))
                        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                            "resident HTF isolated pointer/schema bytes",
                        ))?,
                );
        }
        let pointer_table_entries = max_batch_columns
            .checked_mul(POINTER_FIELDS_PER_COLUMN_V3)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF pointer-table entries",
            ))?;
        let pointer_table_device_bytes = pointer_table_entries.checked_mul(U64_BYTES_V3).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF pointer-table device bytes",
            ),
        )?;
        let pointer_table = StreamOrderedHigherTimeframeBufferV3::uninitialized_async(
            pointer_table_entries,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?;

        Ok(Self {
            parents: Some(parents),
            column_sources,
            output_bindings: output_bindings.into_iter().map(Some).collect(),
            authority: Some(authority),
            context,
            stream,
            device_ordinal,
            base_open_ms: base_parent.timestamps().as_device_ptr().as_ptr(),
            base_row_count: base_parent.rows(),
            pointer_table,
            next_output_binding: 0,
            parent_feature_column_count,
            retained_feature_device_bytes: 0,
            retained_parent_device_bytes,
            pointer_table_device_bytes,
            pointer_table_h2d_bytes: 0,
            isolated_pointer_schema_metadata_bytes: max_isolated_pointer_schema_metadata_bytes,
            native_launch_count: 0,
            native_kernel_launch_count: 0,
            producer_ready_event_count: 0,
        })
    }

    /// Exact direct-parent graph that remains live while every aligned output
    /// batch is launched, packed, and event-retired. The feature-store
    /// assembler consumes this only as runtime peak-memory evidence; ownership
    /// stays with this executor until [`Self::finish_v3`].
    pub(crate) const fn retained_parent_device_bytes(&self) -> usize {
        self.retained_parent_device_bytes
    }

    pub(crate) fn next_pending_batch_v3(
        &mut self,
    ) -> Result<Option<ResidentHigherTimeframeFeatureBatchV3>, ResidentFeatureStoreCudaErrorV3>
    {
        if self.next_output_binding == self.parent_feature_column_count {
            return Ok(None);
        }
        let binding_end = self
            .next_output_binding
            .checked_add(MAX_NATIVE_BATCH_COLUMNS_V3)
            .map(|end| end.min(self.parent_feature_column_count))
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF global batch end",
            ))?;
        let columns = binding_end.checked_sub(self.next_output_binding).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF global batch width"),
        )?;
        let pointer_entries = columns.checked_mul(POINTER_FIELDS_PER_COLUMN_V3).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF active pointer table",
            ),
        )?;
        let mut value_addresses = Vec::with_capacity(columns);
        let mut validity_addresses = Vec::with_capacity(columns);
        let mut value_offsets = Vec::with_capacity(columns);
        let mut validity_offsets = Vec::with_capacity(columns);
        let mut parent_segments = Vec::<NeoResidentHigherTimeframeParentSegmentV3>::new();
        let mut waited_source_batches = BTreeSet::new();
        let mut last_parent_index = None;
        {
            let parents = self
                .parents
                .as_ref()
                .expect("live HTF executor retains parents");
            let sources = self
                .column_sources
                .get(self.next_output_binding..binding_end)
                .ok_or_else(|| {
                    ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "resident HTF source span exceeds live flattened parents".into(),
                    )
                })?;
            for (local_output_column, source) in sources.iter().enumerate() {
                let parent = parents.get(source.parent_index).ok_or_else(|| {
                    ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "resident HTF source parent index drifted".into(),
                    )
                })?;
                let source_batch = parent
                    .feature_batches()
                    .get(source.batch_index)
                    .ok_or_else(|| {
                        ResidentFeatureStoreCudaErrorV3::InvalidInput(
                            "resident HTF source batch index drifted".into(),
                        )
                    })?;
                if source.local_column >= source_batch.column_bindings().len() {
                    return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "resident HTF source local column drifted".into(),
                    ));
                }
                if waited_source_batches.insert((source.parent_index, source.batch_index)) {
                    source_batch.producer_ready_event().wait_before_read(
                        self.context.as_ref(),
                        self.stream.as_ref(),
                        self.device_ordinal,
                    )?;
                }
                value_addresses.push(
                    source_batch
                        .value_buffer(source.local_column)
                        .as_device_ptr()
                        .as_raw(),
                );
                validity_addresses.push(
                    source_batch
                        .validity_buffer(source.local_column)
                        .as_device_ptr()
                        .as_raw(),
                );
                value_offsets.push(
                    u64::try_from(source_batch.value_offset(source.local_column)).map_err(
                        |_| {
                            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                                "resident HTF source value offset ABI",
                            )
                        },
                    )?,
                );
                validity_offsets.push(
                    u64::try_from(source_batch.validity_offset(source.local_column)).map_err(
                        |_| {
                            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                                "resident HTF source validity offset ABI",
                            )
                        },
                    )?,
                );
                if last_parent_index == Some(source.parent_index) {
                    let segment = parent_segments
                        .last_mut()
                        .expect("matching parent segment must exist");
                    segment.column_count = segment.column_count.checked_add(1).ok_or(
                        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                            "resident HTF parent segment width",
                        ),
                    )?;
                } else {
                    parent_segments.push(NeoResidentHigherTimeframeParentSegmentV3 {
                        first_column: u32::try_from(local_output_column).map_err(|_| {
                            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                                "resident HTF parent segment first-column ABI",
                            )
                        })?,
                        column_count: 1,
                        availability_rule: parent.authority.availability_rule.abi_tag(),
                        reserved: 0,
                        parent_row_count: u64::try_from(parent.authority.parent_row_count)
                            .map_err(|_| {
                                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                                    "resident HTF parent-row ABI",
                                )
                            })?,
                        fixed_period_ms: parent.authority.fixed_period_ms,
                        max_age_ms: parent.authority.max_age_ms,
                        parent_open_ms: parent
                            .parent_source()
                            .timestamps()
                            .as_device_ptr()
                            .as_ptr(),
                    });
                    last_parent_index = Some(source.parent_index);
                }
            }
        }
        let mut pointer_table = Vec::with_capacity(pointer_entries);
        pointer_table.extend(value_addresses);
        pointer_table.extend(validity_addresses);
        pointer_table.extend(value_offsets);
        pointer_table.extend(validity_offsets);
        if pointer_table.len() != pointer_entries || parent_segments.is_empty() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF pointer or parent-segment census drifted".into(),
            ));
        }

        let mut bindings = Vec::with_capacity(columns);
        for binding in self
            .output_bindings
            .get_mut(self.next_output_binding..binding_end)
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident HTF output binding span exceeds admission".into(),
                )
            })?
        {
            bindings.push(binding.take().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident HTF output binding was consumed twice".into(),
                )
            })?);
        }

        let cells = self.base_row_count.checked_mul(columns).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF output cells"),
        )?;
        let retained_feature_device_bytes = cells
            .checked_mul(F64_BYTES_V3 + LOGICAL_VALIDITY_BYTES_PER_CELL_V3)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF output value/validity bytes",
            ))?;
        let feature_values = StreamOrderedHigherTimeframeBufferV3::<f64>::uninitialized_async(
            cells,
            Arc::clone(&self.context),
            Arc::clone(&self.stream),
        )?;
        let feature_validity_u8 = StreamOrderedHigherTimeframeBufferV3::<u8>::uninitialized_async(
            cells,
            Arc::clone(&self.context),
            Arc::clone(&self.stream),
        )?;
        let columns_twice =
            columns
                .checked_mul(2)
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF pointer-table value-offset start",
                ))?;
        let columns_thrice =
            columns
                .checked_mul(3)
                .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "resident HTF pointer-table validity-offset start",
                ))?;
        let raw_pointer_table = self.pointer_table.as_device_ptr().as_ptr();
        let source_value_buffers_device = raw_pointer_table.cast::<*const f64>();
        let source_validity_buffers_device =
            unsafe { raw_pointer_table.add(columns) }.cast::<*const u8>();
        let source_value_offsets_device = unsafe { raw_pointer_table.add(columns_twice) };
        let source_validity_offsets_device = unsafe { raw_pointer_table.add(columns_thrice) };
        let feature_column_count = u32::try_from(columns).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF feature-column ABI")
        })?;
        let parent_segment_count = u32::try_from(parent_segments.len()).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF parent-segment ABI")
        })?;
        let base_row_count = u64::try_from(self.base_row_count).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident HTF base-row ABI")
        })?;
        let pointer_h2d_bytes = pointer_table.len().checked_mul(U64_BYTES_V3).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF pointer-table current H2D bytes",
            ),
        )?;
        let next_pointer_table_h2d_bytes = self
            .pointer_table_h2d_bytes
            .checked_add(pointer_h2d_bytes)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF pointer-table H2D bytes",
            ))?;
        let next_native_launch_count = self.native_launch_count.checked_add(1).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF native ABI launch count",
            ),
        )?;
        let next_native_kernel_launch_count = self
            .native_kernel_launch_count
            .checked_add(parent_segments.len())
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF native kernel launch count",
            ))?;
        let next_producer_ready_event_count = self
            .producer_ready_event_count
            .checked_add(1)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF producer-ready event count",
            ))?;
        let next_retained_feature_device_bytes = self
            .retained_feature_device_bytes
            .checked_add(retained_feature_device_bytes)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident HTF all retained output bytes",
            ))?;

        let pointer_host = LockedBuffer::from_slice(&pointer_table)?;
        let mut active_pointer_table = unsafe {
            DeviceSlice::from_raw_parts_mut(self.pointer_table.as_device_ptr(), pointer_table.len())
        };
        if let Err(error) =
            unsafe { active_pointer_table.async_copy_from(&pointer_host, self.stream.as_ref()) }
        {
            std::mem::forget(pointer_host);
            return Err(error.into());
        }
        let native_launch = NeoResidentHigherTimeframeLaunchV3 {
            abi_version: 3,
            semantic_version: RESIDENT_HTF_SEMANTIC_VERSION_V3,
            feature_column_count,
            parent_segment_count,
            base_row_count,
            base_open_ms: self.base_open_ms,
            source_value_buffers_device,
            source_validity_buffers_device,
            source_value_offsets_device,
            source_validity_offsets_device,
            feature_values: feature_values.as_device_ptr().as_mut_ptr(),
            feature_validity_u8: feature_validity_u8.as_device_ptr().as_mut_ptr(),
            parent_segments_host: parent_segments.as_ptr(),
        };
        let status = unsafe {
            neoethos_resident_higher_timeframe_alignment_f64_v3(
                &native_launch,
                self.stream.as_inner(),
            )
        };
        if status != 0 {
            std::mem::forget(pointer_host);
            return Err(ResidentFeatureStoreCudaErrorV3::Native {
                operation: "neoethos_resident_higher_timeframe_alignment_f64_v3",
                status,
            });
        }
        let ready_event = match ResidentProducerReadyEventV3::record(
            self.context.as_ref(),
            self.stream.as_ref(),
            self.device_ordinal,
        ) {
            Ok(event) => event,
            Err(error) => {
                std::mem::forget(pointer_host);
                return Err(error);
            }
        };
        self.next_output_binding = binding_end;
        self.pointer_table_h2d_bytes = next_pointer_table_h2d_bytes;
        self.native_launch_count = next_native_launch_count;
        self.native_kernel_launch_count = next_native_kernel_launch_count;
        self.producer_ready_event_count = next_producer_ready_event_count;
        self.retained_feature_device_bytes = next_retained_feature_device_bytes;
        Ok(Some(ResidentHigherTimeframeFeatureBatchV3 {
            feature_values,
            feature_validity_u8,
            bindings,
            rows: self.base_row_count,
            device_ordinal: self.device_ordinal,
            context: Arc::clone(&self.context),
            stream: Arc::clone(&self.stream),
            ready_event,
            _pointer_host: pointer_host,
            retained_feature_device_bytes,
        }))
    }

    pub(crate) fn finish_v3(
        mut self,
    ) -> Result<ResidentHigherTimeframeRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        let expected_native_launch_count = self
            .parent_feature_column_count
            .div_ceil(MAX_NATIVE_BATCH_COLUMNS_V3);
        if self.next_output_binding != self.parent_feature_column_count
            || self.output_bindings.iter().any(Option::is_some)
            || self.native_launch_count == 0
            || self.native_launch_count != expected_native_launch_count
            || self.native_kernel_launch_count < self.native_launch_count
            || self.native_launch_count != self.producer_ready_event_count
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF executor finished before every admitted route launched".into(),
            ));
        }
        let authority = self
            .authority
            .take()
            .expect("live HTF executor retains launch authority");
        let parent_count = self
            .parents
            .as_ref()
            .expect("live HTF executor retains parents")
            .len();
        let parents = self
            .parents
            .take()
            .expect("live HTF executor retains parents");
        for parent in parents {
            parent.enqueue_nonblocking_release(self.stream.as_ref())?;
        }
        Ok(ResidentHigherTimeframeRuntimeReceiptV3 {
            semantic_version: RESIDENT_HTF_SEMANTIC_VERSION_V3,
            base_row_count: self.base_row_count,
            parent_count,
            parent_feature_column_count: self.parent_feature_column_count,
            retained_feature_device_bytes: self.retained_feature_device_bytes,
            retained_parent_device_bytes: self.retained_parent_device_bytes,
            scratch_device_bytes: 0,
            pointer_table_device_bytes: self.pointer_table_device_bytes,
            pointer_table_h2d_bytes: self.pointer_table_h2d_bytes,
            isolated_pointer_schema_metadata_bytes: self.isolated_pointer_schema_metadata_bytes,
            parent_feature_h2d_bytes: 0,
            feature_value_d2h_bytes: 0,
            feature_validity_d2h_bytes: 0,
            native_launch_count: self.native_launch_count,
            native_kernel_launch_count: self.native_kernel_launch_count,
            producer_ready_event_count: self.producer_ready_event_count,
            producer_ready_event_synchronize_count: 0,
            host_synchronize_count: 0,
            logical_validity_schema: RESIDENT_HTF_LOGICAL_VALIDITY_SCHEMA_V3,
            logical_validity_codes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            canonical_qnan_bits: 0x7ff8_0000_0000_0000,
            input_identity_sha256: authority.input_identity_sha256,
            semantic_source_sha256: authority.semantic_source_sha256,
            implementation_sha256: authority.implementation_sha256,
            selected_parent_order: authority.selected_parent_order,
            canonical_cpu_producer_order: authority.canonical_cpu_producer_order,
            base_context_process_token: authority.base_context_process_token,
            base_stream_process_token: authority.base_stream_process_token,
        })
    }
}

impl Drop for ResidentHigherTimeframeExecutorV3 {
    fn drop(&mut self) {
        if let Some(parents) = self.parents.take() {
            // An unfinished executor may have queued parent reads without the
            // matching downstream pack retirement. Leak rather than free live
            // allocations or insert a host wait on an error path.
            std::mem::forget(parents);
        }
    }
}
