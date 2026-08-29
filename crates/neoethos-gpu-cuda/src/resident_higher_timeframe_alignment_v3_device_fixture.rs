//! Test-only resident HTF-v3 CUDA parity readback.
//!
//! Production has no aligned feature/validity D2H path. This fixture exercises
//! two parent clocks in one global batch, all ten logical validity codes, the
//! exact fixed stale boundary, and calendar tail hiding on a real device.

#![cfg(feature = "cuda-device-fixtures")]

use cust::memory::{CopyDestination, DeviceBuffer, GpuBuffer};
use cust::stream::{Stream, StreamFlags};
use cust::sys::CUstream;

use crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3;

#[cfg(test)]
const QNAN_BITS_V3: u64 = 0x7ff8_0000_0000_0000;
const BASE_ROWS_V3: usize = 11;
const FIXED_COLUMNS_V3: usize = 10;
const CALENDAR_COLUMNS_V3: usize = 2;
const FEATURE_COLUMNS_V3: usize = FIXED_COLUMNS_V3 + CALENDAR_COLUMNS_V3;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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

const _: () = {
    assert!(std::mem::size_of::<NeoResidentHigherTimeframeParentSegmentV3>() == 48);
    assert!(std::mem::size_of::<NeoResidentHigherTimeframeLaunchV3>() == 88);
    assert!(std::mem::offset_of!(NeoResidentHigherTimeframeLaunchV3, base_open_ms) == 24);
    assert!(std::mem::offset_of!(NeoResidentHigherTimeframeLaunchV3, parent_segments_host) == 80);
};

unsafe extern "C" {
    fn neoethos_resident_higher_timeframe_alignment_f64_v3(
        launch: *const NeoResidentHigherTimeframeLaunchV3,
        stream: CUstream,
    ) -> i32;
}

#[derive(Debug)]
pub struct ResidentHigherTimeframeV3DeviceFixtureOutput {
    pub values: Vec<f64>,
    pub validity_u8: Vec<u8>,
}

pub fn run_resident_higher_timeframe_v3_device_fixture()
-> Result<ResidentHigherTimeframeV3DeviceFixtureOutput, ResidentFeatureStoreCudaErrorV3> {
    let _context = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let base_open_ms = [0_i64, 5, 10, 15, 16, 20, 21, 23, 46, 47, 100];
    let fixed_parent_open_ms = [0_i64, 5];
    let calendar_parent_open_ms = [0_i64, 23, 47];
    let device_base_open_ms = DeviceBuffer::from_slice(&base_open_ms)?;
    let device_fixed_parent_open_ms = DeviceBuffer::from_slice(&fixed_parent_open_ms)?;
    let device_calendar_parent_open_ms = DeviceBuffer::from_slice(&calendar_parent_open_ms)?;

    let mut source_values = Vec::with_capacity(FEATURE_COLUMNS_V3);
    let mut source_validity = Vec::with_capacity(FEATURE_COLUMNS_V3);
    for validity_code in 0_u8..=9 {
        let column = f64::from(validity_code);
        source_values.push(DeviceBuffer::from_slice(&[
            1_000.0 + column * 10.0,
            1_001.0 + column * 10.0,
        ])?);
        source_validity.push(DeviceBuffer::from_slice(&[validity_code, validity_code])?);
    }
    source_values.push(DeviceBuffer::from_slice(&[2_000.0, 2_001.0, 2_002.0])?);
    source_validity.push(DeviceBuffer::from_slice(&[0_u8, 3, 0])?);
    source_values.push(DeviceBuffer::from_slice(&[3_000.0, 3_001.0, 3_002.0])?);
    source_validity.push(DeviceBuffer::from_slice(&[9_u8, 0, 0])?);

    let mut pointer_table = Vec::with_capacity(FEATURE_COLUMNS_V3 * 4);
    pointer_table.extend(
        source_values
            .iter()
            .map(|buffer| buffer.as_device_ptr().as_raw()),
    );
    pointer_table.extend(
        source_validity
            .iter()
            .map(|buffer| buffer.as_device_ptr().as_raw()),
    );
    pointer_table.extend(std::iter::repeat_n(0_u64, FEATURE_COLUMNS_V3));
    pointer_table.extend(std::iter::repeat_n(0_u64, FEATURE_COLUMNS_V3));
    let device_pointer_table = DeviceBuffer::from_slice(&pointer_table)?;
    let pointer_base = device_pointer_table.as_device_ptr().as_ptr();
    let cells = BASE_ROWS_V3.checked_mul(FEATURE_COLUMNS_V3).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("HTF-v3 fixture cells"),
    )?;
    // SAFETY: the fixture launch covers every feature-major cell exactly once.
    let device_values = unsafe { DeviceBuffer::<f64>::uninitialized(cells)? };
    // SAFETY: the same launch covers every logical validity cell exactly once.
    let device_validity = unsafe { DeviceBuffer::<u8>::uninitialized(cells)? };
    let parent_segments = [
        NeoResidentHigherTimeframeParentSegmentV3 {
            first_column: 0,
            column_count: FIXED_COLUMNS_V3 as u32,
            availability_rule: 1,
            reserved: 0,
            parent_row_count: fixed_parent_open_ms.len() as u64,
            fixed_period_ms: 5,
            max_age_ms: 10,
            parent_open_ms: device_fixed_parent_open_ms.as_device_ptr().as_ptr(),
        },
        NeoResidentHigherTimeframeParentSegmentV3 {
            first_column: FIXED_COLUMNS_V3 as u32,
            column_count: CALENDAR_COLUMNS_V3 as u32,
            availability_rule: 2,
            reserved: 0,
            parent_row_count: calendar_parent_open_ms.len() as u64,
            fixed_period_ms: 0,
            max_age_ms: -1,
            parent_open_ms: device_calendar_parent_open_ms.as_device_ptr().as_ptr(),
        },
    ];
    let launch = NeoResidentHigherTimeframeLaunchV3 {
        abi_version: 3,
        semantic_version: 3,
        feature_column_count: FEATURE_COLUMNS_V3 as u32,
        parent_segment_count: parent_segments.len() as u32,
        base_row_count: BASE_ROWS_V3 as u64,
        base_open_ms: device_base_open_ms.as_device_ptr().as_ptr(),
        source_value_buffers_device: pointer_base.cast::<*const f64>(),
        source_validity_buffers_device: unsafe { pointer_base.add(FEATURE_COLUMNS_V3) }
            .cast::<*const u8>(),
        source_value_offsets_device: unsafe { pointer_base.add(FEATURE_COLUMNS_V3 * 2) },
        source_validity_offsets_device: unsafe { pointer_base.add(FEATURE_COLUMNS_V3 * 3) },
        feature_values: device_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: device_validity.as_device_ptr().as_mut_ptr(),
        parent_segments_host: parent_segments.as_ptr(),
    };
    // SAFETY: all device pointers belong to this context and remain live until
    // the explicit test-only synchronization/readback below. Parent segment
    // descriptors are consumed synchronously by the C ABI wrapper.
    let status =
        unsafe { neoethos_resident_higher_timeframe_alignment_f64_v3(&launch, stream.as_inner()) };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_higher_timeframe_alignment_f64_v3 fixture",
            status,
        });
    }
    stream.synchronize()?;
    let mut values = vec![0.0; cells];
    let mut validity_u8 = vec![0_u8; cells];
    device_values.copy_to(&mut values)?;
    device_validity.copy_to(&mut validity_u8)?;
    Ok(ResidentHigherTimeframeV3DeviceFixtureOutput {
        values,
        validity_u8,
    })
}

#[test]
fn resident_higher_timeframe_v3_device_route_value_and_validity_parity() {
    let output = run_resident_higher_timeframe_v3_device_fixture()
        .expect("resident HTF-v3 fixture requires a real CUDA device");
    for validity_code in 0_u8..=9 {
        let column = usize::from(validity_code);
        let start = column * BASE_ROWS_V3;
        let end = start + BASE_ROWS_V3;
        assert_eq!(
            &output.validity_u8[start..end],
            &[
                9,
                validity_code,
                validity_code,
                validity_code,
                validity_code,
                validity_code,
                4,
                4,
                4,
                4,
                4,
            ]
        );
        if validity_code == 0 {
            assert_eq!(output.values[start].to_bits(), QNAN_BITS_V3);
            assert_eq!(output.values[start + 1].to_bits(), 1_000.0_f64.to_bits());
            for row in 2..=5 {
                assert_eq!(output.values[start + row].to_bits(), 1_001.0_f64.to_bits());
            }
            for row in 6..BASE_ROWS_V3 {
                assert_eq!(output.values[start + row].to_bits(), QNAN_BITS_V3);
            }
        } else {
            assert!(
                output.values[start..end]
                    .iter()
                    .all(|value| value.to_bits() == QNAN_BITS_V3)
            );
        }
    }

    let calendar_a = FIXED_COLUMNS_V3 * BASE_ROWS_V3;
    assert_eq!(
        &output.validity_u8[calendar_a..calendar_a + BASE_ROWS_V3],
        &[9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 3]
    );
    assert_eq!(
        output.values[calendar_a + 7].to_bits(),
        2_000.0_f64.to_bits()
    );
    assert_eq!(
        output.values[calendar_a + 8].to_bits(),
        2_000.0_f64.to_bits()
    );
    assert!(
        !output.values[calendar_a..calendar_a + BASE_ROWS_V3]
            .iter()
            .any(|value| value.to_bits() == 2_002.0_f64.to_bits())
    );

    let calendar_b = (FIXED_COLUMNS_V3 + 1) * BASE_ROWS_V3;
    assert_eq!(
        &output.validity_u8[calendar_b..calendar_b + BASE_ROWS_V3],
        &[9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0]
    );
    assert_eq!(
        output.values[calendar_b + 9].to_bits(),
        3_001.0_f64.to_bits()
    );
    assert_eq!(
        output.values[calendar_b + 10].to_bits(),
        3_001.0_f64.to_bits()
    );
    assert!(
        !output.values[calendar_b..calendar_b + BASE_ROWS_V3]
            .iter()
            .any(|value| value.to_bits() == 3_002.0_f64.to_bits())
    );
}
