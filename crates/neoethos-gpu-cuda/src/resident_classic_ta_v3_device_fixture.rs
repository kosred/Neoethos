//! Required-card evidence for the opaque resident Classic TA lifecycle.
//!
//! This module is compiled only by the explicit `cuda-device-fixtures`
//! feature. It exposes no context, stream, event, allocation or raw-pointer
//! authority. The bounded readbacks below are test parity evidence and are
//! never reachable from the production resident-store route.

use super::*;
use crate::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_device_v3;
use crate::resident_feature_store_v3::{
    RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3, ResidentFeatureStoreAssemblerV3,
};
use crate::{SMC_SLOTS, acquire_discovery_run_device_admission_v1};
use cust::memory::CopyDestination;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentParentDatasetLayoutV4, ResidentWorkingSetRequestV3,
};
use sha2::{Digest, Sha256};
use std::error::Error;

type FixtureErrorV3 = Box<dyn Error + Send + Sync + 'static>;
type FixtureResultV3<T> = Result<T, FixtureErrorV3>;

const COMPUTE_FAILURE_VALIDITY_CODE_V3: u8 = 8; // FeatureCellValidity::ComputeFailure
const WARMUP_VALIDITY_CODE_V3: u8 = 1; // FeatureCellValidity::Warmup
const VALID_VALIDITY_CODE_V3: u8 = 0;
const NATIVE_NONFINITE_VALIDITY_SENTINEL_V3: u8 = 0xff;
const NATIVE_NONFINITE_ERROR_V3: u32 = 3;
const SYNTHETIC_PACK_WIDTHS_V3: [usize; 6] = [1_usize, 31, 32, 33, 63, 64];

#[derive(Debug)]
pub struct ResidentClassicTaExpectedColumnV3 {
    pub feature_name: String,
    pub expected_value_bits: Vec<u64>,
    pub expected_validity_codes: Vec<u8>,
}

#[derive(Debug)]
pub struct ResidentClassicTaDeviceFixtureRequestV3 {
    pub recipe: ResidentClassicTaRecipeV3,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub expected_columns: Vec<ResidentClassicTaExpectedColumnV3>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentClassicTaDeviceFixtureReceiptV3 {
    pub reviewed_routeable_output_count: usize,
    pub natural_launch_count: usize,
    pub natural_launch_widths: Vec<usize>,
    pub synthetic_pack_widths: [usize; 6],
    pub parent_upload_count: u64,
    pub parent_reupload_count: u64,
    pub second_context_count: u64,
    pub second_stream_count: u64,
    pub value_d2h_bytes: u64,
    pub validity_d2h_bytes: u64,
    pub control_plane_d2h_bytes: u64,
    pub bounded_test_parity_d2h_bytes: u64,
    pub changed_final_feature_bit_observed: bool,
    pub launched_all_nan_compute_failure_observed: bool,
    pub canonical_placeholder_warmup_observed: bool,
    pub output_infinity_refused: bool,
}

fn fixture_error(message: impl Into<String>) -> FixtureErrorV3 {
    std::io::Error::other(message.into()).into()
}

fn require_fixture(condition: bool, message: impl Into<String>) -> FixtureResultV3<()> {
    if condition {
        Ok(())
    } else {
        Err(fixture_error(message))
    }
}

fn hash_f64_bits(values: &[f64]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

fn hash_i64(values: &[i64]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

fn hash_i8(values: &[i8]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Debug)]
struct FixtureResidentParentV3 {
    open: Option<ResidentClassicTaPinnedCopyV3<f64>>,
    high: Option<ResidentClassicTaPinnedCopyV3<f64>>,
    low: Option<ResidentClassicTaPinnedCopyV3<f64>>,
    close: Option<ResidentClassicTaPinnedCopyV3<f64>>,
    volume: Option<ResidentClassicTaPinnedCopyV3<f64>>,
    timestamps: Option<ResidentClassicTaPinnedCopyV3<i64>>,
    months: Option<ResidentClassicTaPinnedCopyV3<i64>>,
    days: Option<ResidentClassicTaPinnedCopyV3<i64>>,
    smc_rows: Option<ResidentClassicTaPinnedCopyV3<i8>>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    layout: ResidentParentDatasetLayoutV4,
    retained_device_bytes: usize,
}

impl FixtureResidentParentV3 {
    fn upload_once(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        request: &ResidentClassicTaDeviceFixtureRequestV3,
    ) -> FixtureResultV3<Self> {
        let rows = request.close.len();
        require_fixture(rows > 0, "fixture parent is empty")?;
        for (name, len) in [
            ("open", request.open.len()),
            ("high", request.high.len()),
            ("low", request.low.len()),
            ("volume", request.volume.len()),
            ("timestamps", request.timestamps.len()),
        ] {
            require_fixture(
                len == rows,
                format!("fixture {name} extent differs from close"),
            )?;
        }
        let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
        let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
        let device_ordinal = run_device.device_identity().ordinal();
        CurrentContext::set_current(context.as_ref())?;
        let months = vec![0_i64; rows];
        let days = vec![0_i64; rows];
        let smc_rows = vec![
            0_i8;
            rows.checked_mul(SMC_SLOTS).ok_or_else(|| {
                fixture_error("fixture SMC parent extent overflow")
            })?
        ];
        let layout = ResidentParentDatasetLayoutV4::new(
            rows,
            hash_f64_bits(&request.open),
            hash_f64_bits(&request.high),
            hash_f64_bits(&request.low),
            hash_f64_bits(&request.close),
            hash_f64_bits(&request.volume),
            hash_i64(&request.timestamps),
            hash_i64(&months),
            hash_i64(&days),
            hash_i8(&smc_rows),
        )?;
        let open = ResidentClassicTaPinnedCopyV3::copy_async(&request.open, &context, &stream)?;
        let high = ResidentClassicTaPinnedCopyV3::copy_async(&request.high, &context, &stream)?;
        let low = ResidentClassicTaPinnedCopyV3::copy_async(&request.low, &context, &stream)?;
        let close = ResidentClassicTaPinnedCopyV3::copy_async(&request.close, &context, &stream)?;
        let volume = ResidentClassicTaPinnedCopyV3::copy_async(&request.volume, &context, &stream)?;
        let timestamps =
            ResidentClassicTaPinnedCopyV3::copy_async(&request.timestamps, &context, &stream)?;
        let months = ResidentClassicTaPinnedCopyV3::copy_async(&months, &context, &stream)?;
        let days = ResidentClassicTaPinnedCopyV3::copy_async(&days, &context, &stream)?;
        let smc_rows = ResidentClassicTaPinnedCopyV3::copy_async(&smc_rows, &context, &stream)?;
        let ready_event = ResidentProducerReadyEventV3::record(&context, &stream, device_ordinal)?;
        let retained_device_bytes = rows
            .checked_mul(
                5 * std::mem::size_of::<f64>()
                    + 3 * std::mem::size_of::<i64>()
                    + SMC_SLOTS * std::mem::size_of::<i8>(),
            )
            .ok_or_else(|| fixture_error("fixture parent retained-byte overflow"))?;
        Ok(Self {
            open: Some(open),
            high: Some(high),
            low: Some(low),
            close: Some(close),
            volume: Some(volume),
            timestamps: Some(timestamps),
            months: Some(months),
            days: Some(days),
            smc_rows: Some(smc_rows),
            rows,
            device_ordinal,
            context,
            stream,
            ready_event,
            layout,
            retained_device_bytes,
        })
    }

    fn copy<T: DeviceCopy>(slot: &Option<ResidentClassicTaPinnedCopyV3<T>>) -> &DeviceBuffer<T> {
        slot.as_ref()
            .expect("fixture parent retains every uploaded array")
            .device()
    }

    fn release_copy<T: DeviceCopy>(
        slot: &mut Option<ResidentClassicTaPinnedCopyV3<T>>,
        stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if let Some(copy) = slot.take() {
            copy.enqueue_release(stream)?;
        }
        Ok(())
    }
}

unsafe impl ResidentParentDatasetSourceV3 for FixtureResidentParentV3 {
    fn open(&self) -> &DeviceBuffer<f64> {
        Self::copy(&self.open)
    }
    fn close(&self) -> &DeviceBuffer<f64> {
        Self::copy(&self.close)
    }
    fn high(&self) -> &DeviceBuffer<f64> {
        Self::copy(&self.high)
    }
    fn low(&self) -> &DeviceBuffer<f64> {
        Self::copy(&self.low)
    }
    fn volume(&self) -> &DeviceBuffer<f64> {
        Self::copy(&self.volume)
    }
    fn timestamps(&self) -> &DeviceBuffer<i64> {
        Self::copy(&self.timestamps)
    }
    fn months(&self) -> &DeviceBuffer<i64> {
        Self::copy(&self.months)
    }
    fn days(&self) -> &DeviceBuffer<i64> {
        Self::copy(&self.days)
    }
    fn smc_rows(&self) -> &DeviceBuffer<i8> {
        Self::copy(&self.smc_rows)
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
        self.retained_device_bytes
    }
    fn parent_dataset_layout(&self) -> &ResidentParentDatasetLayoutV4 {
        &self.layout
    }
    fn enqueue_nonblocking_release(
        mut self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if release_stream.as_inner().is_null()
            || release_stream.as_inner() != self.stream.as_inner()
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        Self::release_copy(&mut self.open, release_stream)?;
        Self::release_copy(&mut self.high, release_stream)?;
        Self::release_copy(&mut self.low, release_stream)?;
        Self::release_copy(&mut self.close, release_stream)?;
        Self::release_copy(&mut self.volume, release_stream)?;
        Self::release_copy(&mut self.timestamps, release_stream)?;
        Self::release_copy(&mut self.months, release_stream)?;
        Self::release_copy(&mut self.days, release_stream)?;
        Self::release_copy(&mut self.smc_rows, release_stream)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
struct FixtureReadbackCountersV3 {
    value_d2h_bytes: u64,
    validity_d2h_bytes: u64,
    control_plane_d2h_bytes: u64,
    natural_launch_widths: Vec<usize>,
}

#[derive(Debug)]
struct FixtureParityMismatchV3 {
    feature_name: String,
    value_mismatch_count: usize,
    validity_mismatch_count: usize,
    first_value_mismatch_row: Option<usize>,
    first_expected_bits: u64,
    first_observed_bits: u64,
    first_value_expected_validity: u8,
    first_value_observed_validity: u8,
    first_validity_mismatch_row: Option<usize>,
    first_expected_validity: u8,
    first_observed_validity: u8,
}

fn checked_readback_bytes(cells: usize, cell_bytes: usize) -> FixtureResultV3<u64> {
    let bytes = cells
        .checked_mul(cell_bytes)
        .ok_or_else(|| fixture_error("fixture readback byte overflow"))?;
    u64::try_from(bytes).map_err(|_| fixture_error("fixture readback exceeds u64"))
}

fn read_and_compare_natural_batch(
    batch: &PendingResidentClassicTaBatchV3,
    expected: &[ResidentClassicTaExpectedColumnV3],
    counters: &mut FixtureReadbackCountersV3,
    mismatches: &mut Vec<FixtureParityMismatchV3>,
) -> FixtureResultV3<()> {
    batch.stream.synchronize()?;
    let columns = batch.bindings.len();
    counters.natural_launch_widths.push(columns);
    let mut validity_error = [0_u32; 1];
    batch
        .validity_device_error
        .as_ref()
        .expect("live batch retains validity error")
        .copy_to(&mut validity_error)?;
    require_fixture(
        validity_error[0] == 0,
        "natural Classic TA validity kernel failed",
    )?;
    counters.control_plane_d2h_bytes += std::mem::size_of::<u32>() as u64;
    let mut validity = vec![0_u8; batch.rows * columns];
    batch
        .validity_u8
        .as_ref()
        .expect("live batch retains validity bytes")
        .copy_to(&mut validity)?;
    counters.validity_d2h_bytes += checked_readback_bytes(validity.len(), 1)?;
    for (column, binding) in batch.bindings.iter().enumerate() {
        let authority = expected.get(binding.ordinal).ok_or_else(|| {
            fixture_error(format!(
                "GPU emitted unplanned destination {}",
                binding.ordinal
            ))
        })?;
        require_fixture(
            authority.feature_name == binding.feature_name,
            format!("feature name mismatch at destination {}", binding.ordinal),
        )?;
        let mut values = vec![0.0_f64; batch.rows];
        batch
            .output_owner
            .as_ref()
            .expect("live batch retains output owner")
            .value_buffer(column)
            .copy_to(&mut values)?;
        let observed_bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        require_fixture(
            observed_bits.len() == authority.expected_value_bits.len(),
            format!("f64 extent mismatch for {}", binding.feature_name),
        )?;
        let begin = column * batch.rows;
        let end = begin + batch.rows;
        let first_value_mismatch_row = observed_bits
            .iter()
            .zip(&authority.expected_value_bits)
            .position(|(observed, expected)| observed != expected);
        let value_mismatch_count = observed_bits
            .iter()
            .zip(&authority.expected_value_bits)
            .filter(|(observed, expected)| observed != expected)
            .count();
        let observed_validity = &validity[begin..end];
        let first_validity_mismatch_row = observed_validity
            .iter()
            .zip(&authority.expected_validity_codes)
            .position(|(observed, expected)| observed != expected);
        let validity_mismatch_count = observed_validity
            .iter()
            .zip(&authority.expected_validity_codes)
            .filter(|(observed, expected)| observed != expected)
            .count();
        if value_mismatch_count != 0 || validity_mismatch_count != 0 {
            let value_row = first_value_mismatch_row.unwrap_or(0);
            let validity_row = first_validity_mismatch_row.unwrap_or(0);
            mismatches.push(FixtureParityMismatchV3 {
                feature_name: binding.feature_name.clone(),
                value_mismatch_count,
                validity_mismatch_count,
                first_value_mismatch_row,
                first_expected_bits: authority.expected_value_bits[value_row],
                first_observed_bits: observed_bits[value_row],
                first_value_expected_validity: authority.expected_validity_codes[value_row],
                first_value_observed_validity: observed_validity[value_row],
                first_validity_mismatch_row,
                first_expected_validity: authority.expected_validity_codes[validity_row],
                first_observed_validity: observed_validity[validity_row],
            });
        }
        counters.value_d2h_bytes += checked_readback_bytes(values.len(), 8)?;
    }
    Ok(())
}

fn fixture_hash(seed: u8) -> [u8; SHA256_BYTES] {
    [seed; SHA256_BYTES]
}

fn fixture_route(
    destination: usize,
    name: impl Into<String>,
    output_id: impl Into<String>,
    stage: ResidentClassicTaStageV3,
    swept_period: Option<u64>,
) -> FixtureResultV3<ResidentClassicTaOutputRouteV3> {
    Ok(ResidentClassicTaOutputRouteV3::new(
        destination,
        name,
        output_id,
        stage,
        swept_period,
        fixture_hash(0x31),
        fixture_hash(0x71),
    )?)
}

fn fixture_recipe(
    rows: usize,
    launch: ResidentClassicTaLaunchRecipeV3,
) -> FixtureResultV3<ResidentClassicTaRecipeV3> {
    Ok(ResidentClassicTaRecipeV3::seal(
        rows,
        rows,
        1,
        fixture_hash(0x55),
        vec![launch],
    )?)
}

fn usize_parameter(key: &str, value: usize) -> FixtureResultV3<ResidentClassicTaParameterV3> {
    Ok(ResidentClassicTaParameterV3::new(
        key,
        ResidentClassicTaParameterValueV3::Usize(u64::try_from(value)?),
    )?)
}

fn execute_one_diagnostic_batch(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &FixtureResidentParentV3,
    recipe: ResidentClassicTaRecipeV3,
    counters: &mut FixtureReadbackCountersV3,
) -> FixtureResultV3<(Vec<Vec<u64>>, Vec<Vec<u8>>)> {
    let mut executor = ResidentClassicTaExecutorV3::new(run_device, parent, recipe)?;
    let batch = executor
        .next_pending_batch_v3()?
        .ok_or_else(|| fixture_error("diagnostic recipe emitted no batch"))?;
    require_fixture(
        executor.next_pending_batch_v3()?.is_none(),
        "single diagnostic recipe emitted multiple batches",
    )?;
    batch.stream.synchronize()?;
    let mut validity_error = [0_u32; 1];
    batch
        .validity_device_error
        .as_ref()
        .expect("diagnostic batch retains validity error")
        .copy_to(&mut validity_error)?;
    require_fixture(validity_error[0] == 0, "diagnostic validity launch failed")?;
    counters.control_plane_d2h_bytes += std::mem::size_of::<u32>() as u64;
    let mut validity = vec![0_u8; batch.rows * batch.bindings.len()];
    batch
        .validity_u8
        .as_ref()
        .expect("diagnostic batch retains validity")
        .copy_to(&mut validity)?;
    counters.validity_d2h_bytes += checked_readback_bytes(validity.len(), 1)?;
    let mut value_bits = Vec::with_capacity(batch.bindings.len());
    let mut validity_codes = Vec::with_capacity(batch.bindings.len());
    for column in 0..batch.bindings.len() {
        let mut values = vec![0.0_f64; batch.rows];
        batch
            .output_owner
            .as_ref()
            .expect("diagnostic batch retains output")
            .value_buffer(column)
            .copy_to(&mut values)?;
        counters.value_d2h_bytes += checked_readback_bytes(values.len(), 8)?;
        value_bits.push(values.iter().map(|value| value.to_bits()).collect());
        validity_codes.push(validity[column * batch.rows..(column + 1) * batch.rows].to_vec());
    }
    Box::new(batch).enqueue_nonblocking_release(parent.stream.as_ref())?;
    Ok((value_bits, validity_codes))
}

fn verify_compute_failure_and_warmup(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &FixtureResidentParentV3,
    counters: &mut FixtureReadbackCountersV3,
) -> FixtureResultV3<(bool, bool)> {
    let rows = parent.rows;
    let compute_failure = ResidentClassicTaLaunchRecipeV3::new(
        "alligator",
        "alligator_outputs_f64",
        ResidentClassicTaInputV3::Hl2,
        ResidentClassicTaFirstValidRuleV3::NamedRouteOwned,
        vec![
            usize_parameter("jaw_period", rows)?,
            usize_parameter("jaw_offset", rows)?,
            usize_parameter("teeth_period", 1)?,
            usize_parameter("teeth_offset", 0)?,
            usize_parameter("lips_period", 1)?,
            usize_parameter("lips_offset", 0)?,
        ],
        vec![fixture_route(
            0,
            "fixture_launched_all_nan",
            "jaw",
            ResidentClassicTaStageV3::Base,
            None,
        )?],
        COMPUTE_FAILURE_VALIDITY_CODE_V3,
    )?;
    let (failure_values, failure_validity) = execute_one_diagnostic_batch(
        run_device,
        parent,
        fixture_recipe(rows, compute_failure)?,
        counters,
    )?;
    let failure_observed = failure_values[0]
        .iter()
        .all(|bits| f64::from_bits(*bits).is_nan())
        && failure_validity[0]
            .iter()
            .all(|code| *code == COMPUTE_FAILURE_VALIDITY_CODE_V3);

    let sma = vector_ta::indicators::dispatch::f64_kernel_for("sma")
        .ok_or_else(|| fixture_error("fixture lost canonical SMA f64 route"))?;
    let warmup = ResidentClassicTaLaunchRecipeV3::new(
        "sma",
        sma.kernel.entry_point(),
        ResidentClassicTaInputV3::Close,
        ResidentClassicTaFirstValidRuleV3::AllInputsNonNan,
        vec![usize_parameter("cuda_period", rows)?],
        vec![fixture_route(
            0,
            "fixture_canonical_placeholder",
            "value",
            ResidentClassicTaStageV3::Historical,
            Some(u64::try_from(rows)?),
        )?],
        WARMUP_VALIDITY_CODE_V3,
    )?;
    let (warmup_values, warmup_validity) =
        execute_one_diagnostic_batch(run_device, parent, fixture_recipe(rows, warmup)?, counters)?;
    let warmup_observed = warmup_values[0]
        .iter()
        .all(|bits| f64::from_bits(*bits).is_nan())
        && warmup_validity[0]
            .iter()
            .all(|code| *code == WARMUP_VALIDITY_CODE_V3);
    Ok((failure_observed, warmup_observed))
}

fn verify_changed_final_feature_bit(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &FixtureResidentParentV3,
    final_close_bits: u64,
    counters: &mut FixtureReadbackCountersV3,
) -> FixtureResultV3<bool> {
    let sma = vector_ta::indicators::dispatch::f64_kernel_for("sma")
        .ok_or_else(|| fixture_error("fixture lost canonical SMA f64 route"))?;
    let launch = ResidentClassicTaLaunchRecipeV3::new(
        "sma",
        sma.kernel.entry_point(),
        ResidentClassicTaInputV3::Close,
        ResidentClassicTaFirstValidRuleV3::AllInputsNonNan,
        vec![usize_parameter("cuda_period", 1)?],
        vec![fixture_route(
            0,
            "fixture_sma_1_final_bit",
            "value",
            ResidentClassicTaStageV3::Base,
            None,
        )?],
        COMPUTE_FAILURE_VALIDITY_CODE_V3,
    )?;
    let (values, validity) = execute_one_diagnostic_batch(
        run_device,
        parent,
        fixture_recipe(parent.rows, launch)?,
        counters,
    )?;
    Ok(values[0].last().copied() == Some(final_close_bits)
        && validity[0].last().copied() == Some(VALID_VALIDITY_CODE_V3))
}

fn verify_output_infinity_refusal(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &FixtureResidentParentV3,
    counters: &mut FixtureReadbackCountersV3,
) -> FixtureResultV3<bool> {
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let values = vec![f64::INFINITY; parent.rows];
    let mut copy = ResidentClassicTaPinnedCopyV3::copy_async(&values, &context, &stream)?;
    let host = copy
        .host
        .take()
        .expect("fixture infinity copy retains host");
    let device = copy
        .device
        .take()
        .expect("fixture infinity copy retains device");
    let launch = ResidentClassicTaLaunchRecipeV3::new(
        "fixture_infinity",
        "fixture_infinity_not_dispatched",
        ResidentClassicTaInputV3::Close,
        ResidentClassicTaFirstValidRuleV3::AllInputsFinite,
        Vec::new(),
        vec![fixture_route(
            0,
            "fixture_infinity",
            "value",
            ResidentClassicTaStageV3::Base,
            None,
        )?],
        COMPUTE_FAILURE_VALIDITY_CODE_V3,
    )?;
    let bindings = launch
        .outputs()
        .iter()
        .map(|route| ResidentFeatureColumnBindingV3 {
            ordinal: route.destination_column(),
            feature_name: route.feature_name().to_owned(),
            canonical_parameter_tuple_sha256: route.canonical_parameter_tuple_sha256(),
            route_receipt_sha256: route.route_receipt_sha256(),
        })
        .collect::<Vec<_>>();
    let batch = PendingResidentClassicTaBatchV3::launch_validity(
        &launch,
        bindings,
        ResidentClassicTaOutputOwnerV3::Warmup(vec![device]),
        &context,
        &stream,
        run_device.device_identity().ordinal(),
        parent.rows,
    )?;
    stream.synchronize()?;
    drop(host);
    let mut error = [0_u32; 1];
    batch
        .validity_device_error
        .as_ref()
        .expect("infinity batch retains error")
        .copy_to(&mut error)?;
    let mut validity = vec![0_u8; parent.rows];
    batch
        .validity_u8
        .as_ref()
        .expect("infinity batch retains validity")
        .copy_to(&mut validity)?;
    counters.control_plane_d2h_bytes += std::mem::size_of::<u32>() as u64;
    counters.validity_d2h_bytes += checked_readback_bytes(validity.len(), 1)?;
    let refused = error[0] == NATIVE_NONFINITE_ERROR_V3
        && validity
            .iter()
            .all(|code| *code == NATIVE_NONFINITE_VALIDITY_SENTINEL_V3);
    Box::new(batch).enqueue_nonblocking_release(stream.as_ref())?;
    Ok(refused)
}

fn synthetic_bindings() -> Vec<ResidentFeatureColumnBindingV3> {
    let mut ordinal = 0_usize;
    let mut bindings = Vec::new();
    for width in SYNTHETIC_PACK_WIDTHS_V3 {
        for column in 0..width {
            bindings.push(ResidentFeatureColumnBindingV3 {
                ordinal,
                feature_name: format!("fixture_pack_{width}_{column}"),
                canonical_parameter_tuple_sha256: fixture_hash(0x41),
                route_receipt_sha256: fixture_hash(0x81),
            });
            ordinal += 1;
        }
    }
    bindings
}

fn synthetic_working_set(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &FixtureResidentParentV3,
    bindings: &[ResidentFeatureColumnBindingV3],
) -> FixtureResultV3<neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetBoundV3> {
    let max_width = *SYNTHETIC_PACK_WIDTHS_V3
        .iter()
        .max()
        .expect("synthetic widths are nonempty");
    let max_live_producer_bytes = parent
        .rows
        .checked_mul(max_width)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>() + 1))
        .ok_or_else(|| fixture_error("synthetic producer bytes overflow"))?;
    let max_live_producer_scratch_bytes = max_width
        .checked_mul(3 * std::mem::size_of::<u64>() + std::mem::size_of::<u8>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>()))
        .ok_or_else(|| fixture_error("synthetic producer scratch overflow"))?;
    let max_pointer_bytes = max_width
        .checked_mul(4 * std::mem::size_of::<u64>())
        .ok_or_else(|| fixture_error("synthetic pointer table overflow"))?;
    let name_offset_bytes = bindings
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(|| fixture_error("synthetic name offsets overflow"))?;
    let name_bytes = bindings.iter().try_fold(0_usize, |total, binding| {
        total
            .checked_add(binding.feature_name.len())
            .ok_or_else(|| fixture_error("synthetic names overflow"))
    })?;
    Ok(ResidentWorkingSetRequestV3 {
        row_count: parent.rows,
        column_count: bindings.len(),
        max_live_producer_bytes: u64::try_from(max_live_producer_bytes)?,
        max_live_producer_scratch_bytes: u64::try_from(max_live_producer_scratch_bytes)?,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        pointer_and_schema_metadata_bytes: u64::try_from(
            max_pointer_bytes
                .checked_add(name_offset_bytes)
                .and_then(|bytes| bytes.checked_add(name_bytes))
                .ok_or_else(|| fixture_error("synthetic metadata bytes overflow"))?,
        )?,
        device_free_bytes_snapshot: run_device.phase_one_free_bytes_snapshot(),
        allocator_context_reserve_bytes: run_device.allocator_context_reserve_bytes(),
        reserve_policy_id: RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3.to_owned(),
    }
    .seal()?)
}

fn synthetic_batch(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    device_ordinal: u32,
    rows: usize,
    bindings: &[ResidentFeatureColumnBindingV3],
) -> FixtureResultV3<(PendingResidentClassicTaBatchV3, Vec<LockedBuffer<f64>>)> {
    let mut hosts = Vec::with_capacity(bindings.len());
    let mut devices = Vec::with_capacity(bindings.len());
    for (column, _) in bindings.iter().enumerate() {
        let values = (0..rows)
            .map(|row| 1.0 + column as f64 * 0.01 + row as f64 * 0.000_001)
            .collect::<Vec<_>>();
        let mut copy = ResidentClassicTaPinnedCopyV3::copy_async(&values, &context, &stream)?;
        hosts.push(copy.host.take().expect("synthetic copy retains host"));
        devices.push(copy.device.take().expect("synthetic copy retains device"));
    }
    let routes = bindings
        .iter()
        .map(|binding| {
            ResidentClassicTaOutputRouteV3::new(
                binding.ordinal,
                binding.feature_name.clone(),
                "value",
                ResidentClassicTaStageV3::Base,
                None,
                binding.canonical_parameter_tuple_sha256,
                binding.route_receipt_sha256,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let launch = ResidentClassicTaLaunchRecipeV3::new(
        "fixture_pack_boundary",
        "fixture_pack_boundary_not_dispatched",
        ResidentClassicTaInputV3::Close,
        ResidentClassicTaFirstValidRuleV3::AllInputsFinite,
        Vec::new(),
        routes,
        COMPUTE_FAILURE_VALIDITY_CODE_V3,
    )?;
    let batch = PendingResidentClassicTaBatchV3::launch_validity(
        &launch,
        bindings.to_vec(),
        ResidentClassicTaOutputOwnerV3::Warmup(devices),
        context,
        stream,
        device_ordinal,
        rows,
    )?;
    Ok((batch, hosts))
}

fn run_synthetic_pack_boundaries(
    run_device: GpuOnlyRunDeviceAdmissionV3,
    parent: FixtureResidentParentV3,
) -> FixtureResultV3<u64> {
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let device_ordinal = run_device.device_identity().ordinal();
    let rows = parent.rows;
    let bindings = synthetic_bindings();
    let working_set = synthetic_working_set(&run_device, &parent, &bindings)?;
    let mut assembler = ResidentFeatureStoreAssemblerV3::new(
        run_device,
        bindings.clone(),
        Box::new(parent),
        &working_set,
    )?;
    let mut cursor = 0_usize;
    for width in SYNTHETIC_PACK_WIDTHS_V3 {
        let end = cursor + width;
        let admitted = &bindings[cursor..end];
        let (batch, hosts) = synthetic_batch(&context, &stream, device_ordinal, rows, admitted)?;
        assembler.append_batch(Box::new(batch))?;
        while !assembler.try_retire_completed_batch()? {
            std::thread::yield_now();
        }
        drop(hosts);
        cursor = end;
    }
    let owner = assembler.seal()?;
    let hashes = loop {
        match owner.compact_hashes_if_ready() {
            Ok(hashes) => break hashes,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    };
    let layout = owner.layout_evidence(&hashes);
    require_fixture(
        layout.producer_batch_count == SYNTHETIC_PACK_WIDTHS_V3.len(),
        "pack batch count drift",
    )?;
    require_fixture(
        layout.full_feature_major_staging_bytes == 0,
        "fixture reached full staging",
    )?;
    let control_plane = u64::try_from(layout.compact_control_plane_d2h_bytes)?;
    drop(owner);
    stream.synchronize()?;
    drop(context);
    Ok(control_plane)
}

/// Run required-card parity over the actual opaque executor. The full recipe
/// is the explicit reviewed routeable ALL-order subset through Halftrend
/// supplied by Data's test-only frozen planner. No global-ALL claim is made
/// here.
pub fn run_resident_classic_ta_v3_device_fixture(
    request: ResidentClassicTaDeviceFixtureRequestV3,
) -> std::result::Result<
    ResidentClassicTaDeviceFixtureReceiptV3,
    Box<dyn Error + Send + Sync + 'static>,
> {
    require_fixture(
        std::env::var_os("NEOETHOS_REQUIRE_GPU").is_some(),
        "required-card Classic TA fixture refuses to skip without NEOETHOS_REQUIRE_GPU",
    )?;
    require_fixture(
        request.recipe.rows() == request.close.len(),
        "fixture recipe/parent row mismatch",
    )?;
    require_fixture(
        request.recipe.output_count() == request.expected_columns.len(),
        "fixture expected-column count differs from recipe",
    )?;

    let admission = acquire_discovery_run_device_admission_v1()?;
    let probes = admission.probe_counters();
    require_fixture(
        probes.physical_inventory_probe_count() == 1
            && probes.cuda_enumeration_count() == 1
            && probes.primary_context_acquisition_count() == 1
            && probes.run_stream_creation_count() == 1,
        "required-card fixture did not retain exactly one probe/context/stream authority",
    )?;
    let run_device = seal_test_full_discovery_run_device_v3(admission, 64 * 1024 * 1024, 1024)?;
    let parent_upload_count = 1_u64;
    let parent = FixtureResidentParentV3::upload_once(&run_device, &request)?;
    let mut counters = FixtureReadbackCountersV3::default();

    let mut executor =
        ResidentClassicTaExecutorV3::new(&run_device, &parent, request.recipe.clone())?;
    let mut natural_launch_count = 0_usize;
    let mut mismatches = Vec::new();
    while let Some(batch) = executor.next_pending_batch_v3()? {
        read_and_compare_natural_batch(
            &batch,
            &request.expected_columns,
            &mut counters,
            &mut mismatches,
        )?;
        Box::new(batch).enqueue_nonblocking_release(parent.stream.as_ref())?;
        natural_launch_count += 1;
    }
    drop(executor);
    require_fixture(
        natural_launch_count == request.recipe.launches().len(),
        "opaque executor did not traverse every admitted launch",
    )?;
    if !mismatches.is_empty() {
        let mut report = format!(
            "Classic TA parity mismatch census: {} feature(s)",
            mismatches.len()
        );
        for mismatch in mismatches {
            report.push_str(&format!(
                "\n- {}: value_mismatches={}, validity_mismatches={}",
                mismatch.feature_name,
                mismatch.value_mismatch_count,
                mismatch.validity_mismatch_count
            ));
            if let Some(row) = mismatch.first_value_mismatch_row {
                report.push_str(&format!(
                    "; first_value_row={row}, expected_bits={:#018x}, observed_bits={:#018x}, expected_validity={}, observed_validity={}",
                    mismatch.first_expected_bits,
                    mismatch.first_observed_bits,
                    mismatch.first_value_expected_validity,
                    mismatch.first_value_observed_validity
                ));
            }
            if let Some(row) = mismatch.first_validity_mismatch_row {
                report.push_str(&format!(
                    "; first_validity_row={row}, expected_validity={}, observed_validity={}",
                    mismatch.first_expected_validity, mismatch.first_observed_validity
                ));
            }
        }
        return Err(fixture_error(report));
    }

    let (launched_all_nan_compute_failure_observed, canonical_placeholder_warmup_observed) =
        verify_compute_failure_and_warmup(&run_device, &parent, &mut counters)?;
    let changed_final_feature_bit_observed = verify_changed_final_feature_bit(
        &run_device,
        &parent,
        request
            .close
            .last()
            .ok_or_else(|| fixture_error("fixture close is empty"))?
            .to_bits(),
        &mut counters,
    )?;
    let output_infinity_refused =
        verify_output_infinity_refusal(&run_device, &parent, &mut counters)?;

    let synthetic_control_plane = run_synthetic_pack_boundaries(run_device, parent)?;
    counters.control_plane_d2h_bytes = counters
        .control_plane_d2h_bytes
        .checked_add(synthetic_control_plane)
        .ok_or_else(|| fixture_error("control-plane D2H accounting overflow"))?;
    let bounded_test_parity_d2h_bytes = counters
        .value_d2h_bytes
        .checked_add(counters.validity_d2h_bytes)
        .and_then(|bytes| bytes.checked_add(counters.control_plane_d2h_bytes))
        .ok_or_else(|| fixture_error("bounded parity D2H accounting overflow"))?;

    Ok(ResidentClassicTaDeviceFixtureReceiptV3 {
        reviewed_routeable_output_count: request.expected_columns.len(),
        natural_launch_count,
        natural_launch_widths: counters.natural_launch_widths,
        synthetic_pack_widths: SYNTHETIC_PACK_WIDTHS_V3,
        parent_upload_count,
        parent_reupload_count: 0,
        second_context_count: 0,
        second_stream_count: 0,
        value_d2h_bytes: counters.value_d2h_bytes,
        validity_d2h_bytes: counters.validity_d2h_bytes,
        control_plane_d2h_bytes: counters.control_plane_d2h_bytes,
        bounded_test_parity_d2h_bytes,
        changed_final_feature_bit_observed,
        launched_all_nan_compute_failure_observed,
        canonical_placeholder_warmup_observed,
        output_infinity_refused,
    })
}
