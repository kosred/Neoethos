//! Test-only Session-v2 parity readback. Production has no feature D2H path.

#![cfg(feature = "cuda-device-fixtures")]

use cust::memory::{CopyDestination, DeviceBuffer, GpuBuffer};
use cust::stream::{Stream, StreamFlags};
use cust::sys::CUstream;

use crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NeoResidentSessionLaunchV2 {
    abi_version: u32,
    semantic_version: u32,
    feature_column_count: u32,
    reserved: u32,
    row_count: u64,
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    volume: *const f64,
    timestamps_ms: *const i64,
    feature_values: *mut f64,
    feature_validity_u8: *mut u8,
}

const _: () = {
    assert!(std::mem::size_of::<NeoResidentSessionLaunchV2>() == 88);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, timestamps_ms) == 64);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, feature_validity_u8) == 80);
};

unsafe extern "C" {
    fn neoethos_resident_session_f64_v2(
        launch: *const NeoResidentSessionLaunchV2,
        stream: CUstream,
    ) -> i32;
}

#[derive(Debug)]
pub struct ResidentSessionV2DeviceFixtureOutput {
    pub values: Vec<f64>,
    pub validity_u8: Vec<u8>,
}

fn validate_fixture_extents(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps_ms: &[i64],
) -> Result<(usize, usize), ResidentFeatureStoreCudaErrorV3> {
    let rows = close.len();
    if rows == 0
        || [
            open.len(),
            high.len(),
            low.len(),
            volume.len(),
            timestamps_ms.len(),
        ]
        .into_iter()
        .any(|len| len != rows)
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "Session-v2 device fixture extent mismatch".into(),
        ));
    }
    let cells = rows
        .checked_mul(23)
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "Session-v2 fixture cells",
        ))?;
    Ok((rows, cells))
}

#[allow(clippy::too_many_arguments)]
pub fn run_resident_session_v2_device_fixture(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps_ms: &[i64],
) -> Result<ResidentSessionV2DeviceFixtureOutput, ResidentFeatureStoreCudaErrorV3> {
    let (rows, cells) = validate_fixture_extents(open, high, low, close, volume, timestamps_ms)?;
    let _context = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let device_open = DeviceBuffer::from_slice(open)?;
    let device_high = DeviceBuffer::from_slice(high)?;
    let device_low = DeviceBuffer::from_slice(low)?;
    let device_close = DeviceBuffer::from_slice(close)?;
    let device_volume = DeviceBuffer::from_slice(volume)?;
    let device_timestamps = DeviceBuffer::from_slice(timestamps_ms)?;
    // SAFETY: exact checked fixture extents initialize every cell in one
    // kernel before the explicit test-only parity D2H below.
    let device_values = unsafe { DeviceBuffer::<f64>::uninitialized(cells)? };
    // SAFETY: same full-write authority as the value allocation.
    let device_validity = unsafe { DeviceBuffer::<u8>::uninitialized(cells)? };
    let launch = NeoResidentSessionLaunchV2 {
        abi_version: 2,
        semantic_version: 2,
        feature_column_count: 23,
        reserved: 0,
        row_count: rows as u64,
        open: device_open.as_device_ptr().as_ptr(),
        high: device_high.as_device_ptr().as_ptr(),
        low: device_low.as_device_ptr().as_ptr(),
        close: device_close.as_device_ptr().as_ptr(),
        volume: device_volume.as_device_ptr().as_ptr(),
        timestamps_ms: device_timestamps.as_device_ptr().as_ptr(),
        feature_values: device_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: device_validity.as_device_ptr().as_mut_ptr(),
    };
    // SAFETY: every pointer belongs to the current fixture context and remains
    // live through the synchronized test-only readback.
    let status = unsafe { neoethos_resident_session_f64_v2(&launch, stream.as_inner()) };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_session_f64_v2 fixture",
            status,
        });
    }
    stream.synchronize()?;
    let mut values = vec![0.0; cells];
    let mut validity_u8 = vec![0_u8; cells];
    device_values.copy_to(&mut values)?;
    device_validity.copy_to(&mut validity_u8)?;
    Ok(ResidentSessionV2DeviceFixtureOutput {
        values,
        validity_u8,
    })
}

/// Repeats the exact production ABI on one context/stream and one retained
/// input/output set. It deliberately performs zero feature D2H so `nsys stats`
/// can gate kernel p95 without transfer or allocation noise.
#[allow(clippy::too_many_arguments)]
pub fn run_resident_session_v2_device_perf_fixture(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps_ms: &[i64],
    repetitions: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let (rows, cells) = validate_fixture_extents(open, high, low, close, volume, timestamps_ms)?;
    if !(1..=10_000).contains(&repetitions) {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "Session-v2 perf repetitions must be in 1..=10_000".into(),
        ));
    }
    let _context = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let device_open = DeviceBuffer::from_slice(open)?;
    let device_high = DeviceBuffer::from_slice(high)?;
    let device_low = DeviceBuffer::from_slice(low)?;
    let device_close = DeviceBuffer::from_slice(close)?;
    let device_volume = DeviceBuffer::from_slice(volume)?;
    let device_timestamps = DeviceBuffer::from_slice(timestamps_ms)?;
    // SAFETY: each repeated launch fully overwrites both exact allocations.
    let device_values = unsafe { DeviceBuffer::<f64>::uninitialized(cells)? };
    // SAFETY: each repeated launch fully overwrites both exact allocations.
    let device_validity = unsafe { DeviceBuffer::<u8>::uninitialized(cells)? };
    let launch = NeoResidentSessionLaunchV2 {
        abi_version: 2,
        semantic_version: 2,
        feature_column_count: 23,
        reserved: 0,
        row_count: rows as u64,
        open: device_open.as_device_ptr().as_ptr(),
        high: device_high.as_device_ptr().as_ptr(),
        low: device_low.as_device_ptr().as_ptr(),
        close: device_close.as_device_ptr().as_ptr(),
        volume: device_volume.as_device_ptr().as_ptr(),
        timestamps_ms: device_timestamps.as_device_ptr().as_ptr(),
        feature_values: device_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: device_validity.as_device_ptr().as_mut_ptr(),
    };
    for _ in 0..repetitions {
        // SAFETY: one retained context/stream and exact buffers outlive every
        // ordered launch; no host pointer or feature D2H enters the loop.
        let status = unsafe { neoethos_resident_session_f64_v2(&launch, stream.as_inner()) };
        if status != 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::Native {
                operation: "neoethos_resident_session_f64_v2 perf fixture",
                status,
            });
        }
    }
    stream.synchronize()?;
    Ok(())
}
