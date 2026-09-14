//! Manual kernel-only CUDA check. It calls the linked production v3 entry;
//! it does not fabricate a Data/Search admission or provide whole-pipeline proof.

use std::io::Write as _;
use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cust::context::{Context, CurrentContext};
use cust::device::Device;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy, GpuBuffer};
use cust::stream::{Stream, StreamFlags};
use cust::sys::CUstream;
use neoethos_gpu_contracts::normalization_v3::resident_normalization_fit_metadata_sha256_v3;
use neoethos_gpu_contracts::resident_feature_store_v3::pack_logical_validity_u4_v3;
use neoethos_gpu_cuda::resident_robust_normalization_v2::ResidentRobustNormalizationPlanV2;

use super::search_normalization_column_mode_v3;
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::normalization::{RobustNormalizationFitF64, normalize_search_feature_column_f64};

unsafe extern "C" {
    fn neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
        values: *mut f64,
        validity: *mut u8,
        validity_bytes: usize,
        rows: usize,
        columns: usize,
        start: usize,
        end: usize,
        padded_rows: usize,
        scratch: *mut u64,
        scratch_slots: usize,
        fits: *mut u64,
        fit_words: usize,
        modes: *const u8,
        mode_count: usize,
        control: *mut u32,
        stream: CUstream,
    ) -> i32;
}

const ROWS: usize = 100;
const COLUMNS: usize = 69;
const TRAINING_END: usize = 80;

fn fixture_columns() -> Result<Vec<FeatureColumnF64>> {
    (0..COLUMNS)
        .map(|column| {
            let name = match column {
                0 => "smc_eqh".into(),
                1 => "smc_eql".into(),
                2 => "smc_ob".into(),
                3 => "smc_trend_bias".into(),
                64 => "H1_smc_fvg".into(),
                65 => "M5_smc_eqh".into(),
                66 => "M15_smc_trend_bias".into(),
                _ => format!("continuous_{column}"),
            };
            let mut values = Vec::with_capacity(ROWS);
            let mut validity = Vec::with_capacity(ROWS);
            for row in 0..ROWS {
                let reason = if row % 17 == 0 {
                    FeatureCellValidity::Gap
                } else if row == 3 {
                    FeatureCellValidity::Warmup
                } else {
                    FeatureCellValidity::Valid
                };
                let value = match column {
                    0 | 65 => 1.0,
                    1 | 67 => 0.0,
                    2 | 64 => match row % 3 {
                        0 => -0.0,
                        1 => 1.0,
                        _ => -1.0,
                    },
                    3 | 66 => row as f64 - 50.0,
                    68 => {
                        if row % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    _ if row >= TRAINING_END => 1_000_000.0,
                    _ => (row % 7) as f64 - 3.0,
                };
                values.push(if reason.is_valid() { value } else { f64::NAN });
                validity.push(reason);
            }
            FeatureColumnF64::new(name, values, validity)
        })
        .collect()
}

/// Test allocations retain the real stream/context if a completion or free
/// fails. No cleanup retry may destroy an owner still covering accepted work.
struct Buffers {
    words: Vec<DeviceBuffer<u64>>,
    control: Vec<DeviceBuffer<u32>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    pending: bool,
}

fn free_checked<T: DeviceCopy>(buffer: DeviceBuffer<T>) -> Result<()> {
    DeviceBuffer::drop(buffer).map_err(|(error, retained)| {
        std::mem::forget(retained);
        anyhow::anyhow!("CUDA fixture buffer release failed and was retained: {error}")
    })
}

impl Buffers {
    fn close(&mut self) -> Result<()> {
        if self.pending {
            let words = std::mem::take(&mut self.words);
            let control = std::mem::take(&mut self.control);
            std::mem::forget((
                words,
                control,
                Arc::clone(&self.context),
                Arc::clone(&self.stream),
            ));
            anyhow::bail!("CUDA fixture completion is ambiguous; owners retained");
        }
        let mut failure = None;
        for buffer in self.words.drain(..) {
            if let Err(error) = free_checked(buffer) {
                failure.get_or_insert(error);
            }
        }
        for buffer in self.control.drain(..) {
            if let Err(error) = free_checked(buffer) {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            std::mem::forget((Arc::clone(&self.context), Arc::clone(&self.stream)));
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            let _ = writeln!(std::io::stderr().lock(), "{error:#}");
        }
    }
}

fn run_case(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    raw: &[FeatureColumnF64],
    expected_control: u32,
) -> Result<()> {
    let modes = raw
        .iter()
        .map(|column| search_normalization_column_mode_v3(&column.name))
        .collect::<Vec<_>>();
    let plan =
        ResidentRobustNormalizationPlanV2::preflight(ROWS, raw.len(), 0..TRAINING_END, true)?
            .with_column_modes_v3(modes.clone())?;
    let mut expected = raw.to_vec();
    let cpu_fits = expected
        .iter_mut()
        .map(|column| normalize_search_feature_column_f64(column, 0..TRAINING_END))
        .collect::<Result<Vec<_>>>();
    ensure!(
        cpu_fits.is_err() == (expected_control != 0),
        "CPU fixture rejection expectation differs"
    );
    let mut value_bits = Vec::with_capacity(ROWS * raw.len());
    let mut logical = Vec::with_capacity(value_bits.capacity());
    for row in 0..ROWS {
        for column in raw {
            value_bits.push(column.values[row].to_bits());
            logical.push(column.validity[row].code());
        }
    }
    let mut packed = pack_logical_validity_u4_v3(&logical)?;
    packed.resize(plan.packed_validity_allocated_bytes(), 0);
    let packed_words = packed
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut device = Buffers {
        words: Vec::new(),
        control: Vec::new(),
        context: Arc::clone(context),
        stream: Arc::clone(stream),
        pending: false,
    };
    device.words.push(DeviceBuffer::from_slice(&value_bits)?);
    device.words.push(DeviceBuffer::from_slice(
        &vec![0u64; plan.normalization_scratch_slots()],
    )?);
    device.words.push(DeviceBuffer::from_slice(&vec![
        0u64;
        plan.fit_metadata_words()
    ])?);
    device
        .control
        .push(DeviceBuffer::from_slice(&packed_words)?);
    device.control.push(DeviceBuffer::from_slice(&[0u32])?);
    device.pending = true;
    let status = unsafe {
        neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
            device.words[0].as_device_ptr().as_mut_ptr().cast(),
            device.control[0].as_device_ptr().as_mut_ptr().cast(),
            plan.packed_validity_allocated_bytes(),
            ROWS,
            raw.len(),
            0,
            TRAINING_END,
            plan.padded_training_rows(),
            device.words[1].as_device_ptr().as_mut_ptr(),
            plan.normalization_scratch_slots(),
            device.words[2].as_device_ptr().as_mut_ptr(),
            plan.fit_metadata_words(),
            modes.as_ptr().cast(),
            modes.len(),
            device.control[1].as_device_ptr().as_mut_ptr(),
            stream.as_inner(),
        )
    };
    // Even a launch error may follow accepted earlier kernels.
    stream
        .synchronize()
        .context("CUDA fixture completion failed")?;
    device.pending = false;
    ensure!(
        status == 0,
        "production normalizer rejected kernel invocation: {status}"
    );
    let mut control = [0u32];
    device.control[1].copy_to(&mut control)?;
    if expected_control != 0 {
        ensure!(
            control[0] == expected_control,
            "semantic rejection produced {}, expected {expected_control}",
            control[0]
        );
    } else {
        ensure!(
            control[0] == 0,
            "unexpected native normalization fault {}",
            control[0]
        );
        device.words[0].copy_to(&mut value_bits)?;
        let mut actual_validity = vec![0u32; packed_words.len()];
        device.control[0].copy_to(&mut actual_validity)?;
        let actual_validity = actual_validity
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let expected_logical = (0..ROWS)
            .flat_map(|row| {
                expected
                    .iter()
                    .map(move |column| column.validity[row].code())
            })
            .collect::<Vec<_>>();
        let mut expected_packed = pack_logical_validity_u4_v3(&expected_logical)?;
        expected_packed.resize(plan.packed_validity_allocated_bytes(), 0);
        ensure!(
            actual_validity == expected_packed,
            "packed validity or aligned padding differs"
        );
        let mut actual_fits = vec![0u64; plan.fit_metadata_words()];
        device.words[2].copy_to(&mut actual_fits)?;
        let mut digest_words = [0u64; 4];
        device.words[1].index(0..4).copy_to(&mut digest_words)?;
        ensure!(
            digest_words
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
                == resident_normalization_fit_metadata_sha256_v3(&actual_fits),
            "native fit digest mismatch"
        );
        for row in 0..ROWS {
            for (column, expected) in expected.iter().enumerate() {
                let cell = row * raw.len() + column;
                ensure!(
                    value_bits[cell] == expected.values[row].to_bits(),
                    "value bits differ at ({row},{column})"
                );
                let code = (actual_validity[cell / 2] >> ((cell & 1) * 4)) & 15;
                ensure!(
                    code == expected.validity[row].code(),
                    "validity differs at ({row},{column})"
                );
            }
        }
        for (actual, fit) in actual_fits.chunks_exact(6).zip(cpu_fits?) {
            let RobustNormalizationFitF64 {
                training_rows,
                median,
                scale,
                valid_training_cells,
                degenerate,
            } = fit;
            ensure!(
                actual
                    == [
                        training_rows.start as u64,
                        training_rows.end as u64,
                        median.to_bits(),
                        scale.to_bits(),
                        valid_training_cells as u64,
                        u64::from(degenerate)
                    ],
                "actual native fit words differ from CPU Search policy"
            );
        }
    }
    device.close()?;
    Ok(())
}

#[test]
#[ignore = "manual actual CUDA kernel check; requires NEOETHOS_RUN_CUDA_DEVICE_TESTS=1 and a real NVIDIA device"]
fn actual_cuda_search_normalization_policy3_mixed69_and_semantic_faults() -> Result<()> {
    ensure!(
        std::env::var("NEOETHOS_RUN_CUDA_DEVICE_TESTS")
            .ok()
            .as_deref()
            == Some("1"),
        "explicit real-device opt-in required; absence is not a passing skip"
    );
    let ordinal = std::env::var("NEOETHOS_CUDA_DEVICE_ORDINAL")
        .unwrap_or_else(|_| "0".into())
        .parse::<u32>()?;
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(ordinal)?;
    let context = Arc::new(Context::new(device)?);
    CurrentContext::set_current(context.as_ref())?;
    let stream = Arc::new(Stream::new(StreamFlags::NON_BLOCKING, None)?);
    eprintln!(
        "CUDA kernel-only normalization fixture: ordinal={ordinal}, name={}, rows={ROWS}, columns={COLUMNS}",
        device.name()?
    );
    let outcome = (|| {
        let raw = fixture_columns()?;
        run_case(&context, &stream, &raw, 0)?;
        for (column, value) in [(0, 0.25), (2, 2.0), (3, f64::INFINITY)] {
            let mut invalid = raw.clone();
            invalid[column].values[99] = value;
            invalid[column].validity[99] = FeatureCellValidity::Valid;
            run_case(&context, &stream, &invalid, 1 << 3)?;
        }
        let mut invalid = raw;
        for row in 0..TRAINING_END {
            invalid[4].values[row] = f64::NAN;
            invalid[4].validity[row] = FeatureCellValidity::Gap;
        }
        run_case(&context, &stream, &invalid, 1 << 1)
    })();
    let stream_cleanup = Arc::try_unwrap(stream)
        .map_err(|retained| {
            std::mem::forget(retained);
            anyhow::anyhow!("CUDA fixture stream is quarantined")
        })
        .and_then(|stream| {
            Stream::drop(stream).map_err(|(e, retained)| {
                std::mem::forget(retained);
                std::mem::forget(Arc::clone(&context));
                anyhow::anyhow!("CUDA stream cleanup: {e}")
            })
        });
    let context_cleanup = Arc::try_unwrap(context)
        .map_err(|retained| {
            std::mem::forget(retained);
            anyhow::anyhow!("CUDA fixture context is quarantined")
        })
        .and_then(|context| {
            Context::drop(context).map_err(|(e, retained)| {
                std::mem::forget(retained);
                anyhow::anyhow!("CUDA context cleanup: {e}")
            })
        });
    outcome?;
    stream_cleanup?;
    context_cleanup?;
    Ok(())
}
