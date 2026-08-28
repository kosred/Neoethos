//! Required-card exact-bit fixture for resident Quant-v3.
//!
//! Production never downloads a feature payload. This feature-gated harness
//! performs one bounded test-only parity D2H after the producer-ready stream
//! completes, solely to compare all 63 value bits and validity codes.

#![cfg(feature = "cuda-device-fixtures")]

use super::*;
use crate::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_device_v3;
use crate::{SMC_SLOTS, acquire_discovery_run_device_admission_v1};
use cust::memory::{AsyncCopyDestination, CopyDestination, LockedBuffer};
use neoethos_gpu_contracts::resident_feature_store_v3::ResidentParentDatasetLayoutV4;
use sha2::{Digest, Sha256};
use std::error::Error;

type FixtureErrorV3 = Box<dyn Error + Send + Sync + 'static>;
type FixtureResultV3<T> = Result<T, FixtureErrorV3>;

#[derive(Debug)]
pub struct ResidentQuantDeviceFixtureOutputV3 {
    pub values: Vec<f64>,
    pub validity_u8: Vec<u8>,
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
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

fn hash_i64(values: &[i64]) -> [u8; SHA256_BYTES] {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().into()
}

fn hash_i8(values: &[i8]) -> [u8; SHA256_BYTES] {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().into()
}

#[derive(Debug)]
struct FixtureCopyV3<T: DeviceCopy> {
    host: Option<LockedBuffer<T>>,
    device: Option<StreamOrderedQuantBufferV3<T>>,
}

impl<T: DeviceCopy> FixtureCopyV3<T> {
    fn copy_async(
        source: &[T],
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> FixtureResultV3<Self> {
        let host = LockedBuffer::from_slice(source)?;
        let mut device = StreamOrderedQuantBufferV3::<T>::uninitialized_async(
            source.len(),
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        if let Err(error) = unsafe { device.async_copy_from(&host, stream.as_ref()) } {
            std::mem::forget(host);
            std::mem::forget(device);
            return Err(error.into());
        }
        Ok(Self {
            host: Some(host),
            device: Some(device),
        })
    }

    fn device(&self) -> &DeviceBuffer<T> {
        self.device
            .as_ref()
            .expect("fixture retains its device copy")
    }

    fn enqueue_release(
        mut self,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self
            .device
            .as_ref()
            .is_some_and(|device| device.is_owned_by_stream(release_stream))
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self.host.take());
        drop(self.device.take());
        Ok(())
    }
}

impl<T: DeviceCopy> Drop for FixtureCopyV3<T> {
    fn drop(&mut self) {
        if let Some(host) = self.host.take() {
            std::mem::forget(host);
        }
    }
}

#[derive(Debug)]
struct FixtureParentV3 {
    open: Option<FixtureCopyV3<f64>>,
    high: Option<FixtureCopyV3<f64>>,
    low: Option<FixtureCopyV3<f64>>,
    close: Option<FixtureCopyV3<f64>>,
    volume: Option<FixtureCopyV3<f64>>,
    timestamps: Option<FixtureCopyV3<i64>>,
    months: Option<FixtureCopyV3<i64>>,
    days: Option<FixtureCopyV3<i64>>,
    smc_rows: Option<FixtureCopyV3<i8>>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    layout: ResidentParentDatasetLayoutV4,
    retained_device_bytes: usize,
}

impl FixtureParentV3 {
    #[allow(clippy::too_many_arguments)]
    fn upload_once(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        timestamps: &[i64],
    ) -> FixtureResultV3<Self> {
        let rows = close.len();
        require_fixture(rows > 0, "Quant fixture parent is empty")?;
        for (name, len) in [
            ("open", open.len()),
            ("high", high.len()),
            ("low", low.len()),
            ("volume", volume.len()),
            ("timestamps", timestamps.len()),
        ] {
            require_fixture(len == rows, format!("Quant fixture {name} extent mismatch"))?;
        }
        let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
        let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
        let device_ordinal = run_device.device_identity().ordinal();
        CurrentContext::set_current(context.as_ref())?;
        let months = vec![0_i64; rows];
        let days = vec![0_i64; rows];
        let smc_rows = vec![
            0_i8;
            rows.checked_mul(SMC_SLOTS)
                .ok_or_else(|| fixture_error("Quant fixture SMC extent overflow"))?
        ];
        let layout = ResidentParentDatasetLayoutV4::new(
            rows,
            hash_f64_bits(open),
            hash_f64_bits(high),
            hash_f64_bits(low),
            hash_f64_bits(close),
            hash_f64_bits(volume),
            hash_i64(timestamps),
            hash_i64(&months),
            hash_i64(&days),
            hash_i8(&smc_rows),
        )?;
        let open = FixtureCopyV3::copy_async(open, &context, &stream)?;
        let high = FixtureCopyV3::copy_async(high, &context, &stream)?;
        let low = FixtureCopyV3::copy_async(low, &context, &stream)?;
        let close = FixtureCopyV3::copy_async(close, &context, &stream)?;
        let volume = FixtureCopyV3::copy_async(volume, &context, &stream)?;
        let timestamps = FixtureCopyV3::copy_async(timestamps, &context, &stream)?;
        let months = FixtureCopyV3::copy_async(&months, &context, &stream)?;
        let days = FixtureCopyV3::copy_async(&days, &context, &stream)?;
        let smc_rows = FixtureCopyV3::copy_async(&smc_rows, &context, &stream)?;
        let ready_event = ResidentProducerReadyEventV3::record(
            context.as_ref(),
            stream.as_ref(),
            device_ordinal,
        )?;
        let retained_device_bytes = rows
            .checked_mul(
                5 * std::mem::size_of::<f64>()
                    + 3 * std::mem::size_of::<i64>()
                    + SMC_SLOTS * std::mem::size_of::<i8>(),
            )
            .ok_or_else(|| fixture_error("Quant fixture parent bytes overflow"))?;
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

    fn copy<T: DeviceCopy>(slot: &Option<FixtureCopyV3<T>>) -> &DeviceBuffer<T> {
        slot.as_ref()
            .expect("Quant fixture parent is live")
            .device()
    }

    fn release_copy<T: DeviceCopy>(
        slot: &mut Option<FixtureCopyV3<T>>,
        stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if let Some(copy) = slot.take() {
            copy.enqueue_release(stream)?;
        }
        Ok(())
    }
}

unsafe impl ResidentParentDatasetSourceV3 for FixtureParentV3 {
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

/// Performs the sole feature-value readback allowed by the required-card test.
#[allow(clippy::too_many_arguments)]
pub fn run_resident_quant_v3_device_fixture(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    timeframe_millis: u64,
) -> FixtureResultV3<ResidentQuantDeviceFixtureOutputV3> {
    require_fixture(
        std::env::var_os("NEOETHOS_REQUIRE_GPU").is_some(),
        "required-card Quant fixture refuses to skip without NEOETHOS_REQUIRE_GPU",
    )?;
    require_fixture(
        timeframe_millis > 0
            && UTC_DAY_MILLIS_V3 % timeframe_millis == 0
            && ASIAN_SESSION_MILLIS_V3 % timeframe_millis == 0,
        "Quant fixture timeframe is not a typed UTC/Asian divisor",
    )?;
    let bars_per_utc_day = UTC_DAY_MILLIS_V3 / timeframe_millis;
    let bars_per_asian_session = ASIAN_SESSION_MILLIS_V3 / timeframe_millis;
    require_fixture(
        bars_per_asian_session >= 12,
        "Quant fixture requires at least twelve Asian-session bars",
    )?;

    let row_count = u64::try_from(close.len())
        .map_err(|_| fixture_error("Quant fixture row count exceeds u64"))?;
    let admission = acquire_discovery_run_device_admission_v1()?;
    let run_device =
        seal_test_full_discovery_run_device_v3(admission, 256 * 1024 * 1024, row_count)?;
    let parent =
        FixtureParentV3::upload_once(&run_device, open, high, low, close, volume, timestamps)?;
    let bindings = RESIDENT_QUANT_COLUMN_NAMES_V3
        .iter()
        .enumerate()
        .map(|(ordinal, name)| ResidentFeatureColumnBindingV3 {
            ordinal,
            feature_name: (*name).to_owned(),
            canonical_parameter_tuple_sha256: [1; SHA256_BYTES],
            route_receipt_sha256: [2; SHA256_BYTES],
        })
        .collect::<Vec<_>>();
    let authority = ResidentQuantLaunchAuthorityV3::seal(
        close.len(),
        timeframe_millis,
        bars_per_asian_session,
        bars_per_utc_day,
        bars_per_utc_day * 5,
        252,
        bars_per_utc_day * 252,
        [3; SHA256_BYTES],
        [4; SHA256_BYTES],
        seal_resident_quant_migration_closure_v3(),
    )?;
    let batch = launch_resident_quant_v3(&run_device, &parent, bindings, authority)?;
    parent.stream.synchronize()?;
    let cells = close
        .len()
        .checked_mul(RESIDENT_QUANT_COLUMN_NAMES_V3.len())
        .ok_or_else(|| fixture_error("Quant fixture output extent overflow"))?;
    let mut values = vec![0.0_f64; cells];
    let mut validity_u8 = vec![0_u8; cells];
    batch.feature_values.copy_to(&mut values)?;
    batch.feature_validity_u8.copy_to(&mut validity_u8)?;
    Box::new(batch).enqueue_nonblocking_release(parent.stream.as_ref())?;
    Box::new(parent)
        .enqueue_nonblocking_release(run_device.run_stream_for_resident_producer_v3().as_ref())?;
    Ok(ResidentQuantDeviceFixtureOutputV3 {
        values,
        validity_u8,
    })
}

/// Repeats the production ABI over one retained parent/output set. It performs
/// zero feature D2H so `nsys stats` can report the single-lane kernel
/// distribution without parity-readback or allocation noise.
#[allow(clippy::too_many_arguments)]
pub fn run_resident_quant_v3_device_perf_fixture(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    timeframe_millis: u64,
    repetitions: usize,
) -> FixtureResultV3<()> {
    require_fixture(
        std::env::var_os("NEOETHOS_REQUIRE_GPU").is_some(),
        "required-card Quant perf fixture refuses to skip without NEOETHOS_REQUIRE_GPU",
    )?;
    require_fixture(
        (1..=10_000).contains(&repetitions),
        "Quant perf repetitions must be in 1..=10_000",
    )?;
    require_fixture(
        timeframe_millis > 0
            && UTC_DAY_MILLIS_V3 % timeframe_millis == 0
            && ASIAN_SESSION_MILLIS_V3 % timeframe_millis == 0,
        "Quant perf timeframe is not a typed UTC/Asian divisor",
    )?;
    let bars_per_utc_day = UTC_DAY_MILLIS_V3 / timeframe_millis;
    let bars_per_asian_session = ASIAN_SESSION_MILLIS_V3 / timeframe_millis;
    require_fixture(
        bars_per_asian_session >= 12,
        "Quant perf fixture requires at least twelve Asian-session bars",
    )?;

    let row_count = u64::try_from(close.len())
        .map_err(|_| fixture_error("Quant fixture row count exceeds u64"))?;
    let admission = acquire_discovery_run_device_admission_v1()?;
    let run_device =
        seal_test_full_discovery_run_device_v3(admission, 256 * 1024 * 1024, row_count)?;
    let parent =
        FixtureParentV3::upload_once(&run_device, open, high, low, close, volume, timestamps)?;
    let context = Arc::clone(&parent.context);
    let stream = Arc::clone(&parent.stream);
    parent.producer_ready_event().wait_before_read(
        context.as_ref(),
        stream.as_ref(),
        parent.device_ordinal,
    )?;
    let cells = close
        .len()
        .checked_mul(RESIDENT_QUANT_COLUMN_NAMES_V3.len())
        .ok_or_else(|| fixture_error("Quant perf output extent overflow"))?;
    let feature_values = StreamOrderedQuantBufferV3::<f64>::uninitialized_async(
        cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let feature_validity_u8 = StreamOrderedQuantBufferV3::<u8>::uninitialized_async(
        cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let native_launch = NeoResidentQuantLaunchV3 {
        abi_version: 3,
        semantic_version: RESIDENT_QUANT_SEMANTIC_VERSION_V3,
        feature_column_count: RESIDENT_QUANT_COLUMN_NAMES_V3.len() as u32,
        reserved: 0,
        row_count,
        timeframe_millis,
        bars_per_asian_session,
        bars_per_utc_day,
        bars_per_trading_week: bars_per_utc_day * 5,
        trading_sessions_per_year: 252,
        annualization_periods_per_year: bars_per_utc_day * 252,
        open: parent.open().as_device_ptr().as_ptr(),
        high: parent.high().as_device_ptr().as_ptr(),
        low: parent.low().as_device_ptr().as_ptr(),
        close: parent.close().as_device_ptr().as_ptr(),
        volume: parent.volume().as_device_ptr().as_ptr(),
        timestamps: parent.timestamps().as_device_ptr().as_ptr(),
        feature_values: feature_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: feature_validity_u8.as_device_ptr().as_mut_ptr(),
    };
    for _ in 0..repetitions {
        // SAFETY: one retained context/stream owns every exact checked extent
        // for all repetitions, and every launch fully overwrites both outputs.
        let status = unsafe { neoethos_resident_quant_f64_v3(&native_launch, stream.as_inner()) };
        if status != 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::Native {
                operation: "neoethos_resident_quant_f64_v3 performance fixture",
                status,
            }
            .into());
        }
    }
    stream.synchronize()?;
    drop(feature_values);
    drop(feature_validity_u8);
    Box::new(parent).enqueue_nonblocking_release(stream.as_ref())?;
    stream.synchronize()?;
    Ok(())
}
