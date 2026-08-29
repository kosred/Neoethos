//! Bounded, feature-gated readback for exact device-fixture parity only.
//!
//! Production keeps the resident feature/value payload on the card. This
//! module is compiled only by `cuda-device-fixtures` and is the sole place
//! where an integrated fixture may download the final bar-major buffers.

#![cfg(feature = "cuda-device-fixtures")]

use cust::memory::{CopyDestination, DeviceBuffer};

use crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3;

#[derive(Debug)]
pub struct ResidentFeatureStoreDeviceReadbackV3 {
    pub values: Vec<f64>,
    pub validity_u8: Vec<u8>,
    pub rows: usize,
    pub columns: usize,
}

pub(crate) fn copy_bar_major_for_device_fixture_v3(
    values: &DeviceBuffer<f64>,
    validity_u4: &DeviceBuffer<u8>,
    rows: usize,
    columns: usize,
) -> Result<ResidentFeatureStoreDeviceReadbackV3, ResidentFeatureStoreCudaErrorV3> {
    let cells = rows.checked_mul(columns).ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "device-fixture bar-major cell extent overflowed".into(),
        )
    })?;
    if values.len() != cells || validity_u4.len() < cells.div_ceil(2) {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "device-fixture bar-major allocation extent drifted".into(),
        ));
    }

    let mut host_values = vec![0.0_f64; cells];
    let mut host_validity_u4 = vec![0_u8; validity_u4.len()];
    values.copy_to(&mut host_values)?;
    validity_u4.copy_to(&mut host_validity_u4)?;
    let validity_u8 = (0..cells)
        .map(|cell| {
            let packed = host_validity_u4[cell / 2];
            if cell % 2 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            }
        })
        .collect();

    Ok(ResidentFeatureStoreDeviceReadbackV3 {
        values: host_values,
        validity_u8,
        rows,
        columns,
    })
}
