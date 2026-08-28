use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current directory"))
}

fn source(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end = source[start..]
        .find(end)
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start..end]
}

#[test]
fn gpu_only_cuda_surface_has_no_host_compute_then_upload_escape_hatch() {
    let wrappers = [
        (
            "src/cuda/mass_wrapper.rs",
            [
                "host_fallback::record",
                "mass_many_series_one_param_time_major_dev",
                "mass_with_kernel(&input, Kernel::Scalar)",
            ],
        ),
        (
            "src/cuda/net_myrsi_wrapper.rs",
            [
                "host_fallback::record",
                "net_myrsi_many_series_one_param_time_major_dev",
                "net_myrsi_with_kernel(",
            ],
        ),
        (
            "src/cuda/rvi_wrapper.rs",
            [
                "host_fallback::record",
                "rvi_batch_with_kernel",
                "Kernel::ScalarBatch",
            ],
        ),
        (
            "src/cuda/vosc_wrapper.rs",
            [
                "host_fallback::record",
                "vosc_many_series_one_param_time_major_dev",
                "vosc_with_kernel(&input, Kernel::Scalar)",
            ],
        ),
    ];

    let mut violations = Vec::new();
    for (path, forbidden) in wrappers {
        let contents = source(path);
        for token in forbidden {
            if contents.contains(token) {
                violations.push(format!("{path} contains `{token}`"));
            }
        }
    }

    let cuda_mod = source("src/cuda/mod.rs");
    if cuda_mod.contains("pub mod host_fallback;") {
        violations.push("src/cuda/mod.rs exports `host_fallback`".to_string());
    }
    if crate_root().join("src/cuda/host_fallback.rs").exists() {
        violations.push("src/cuda/host_fallback.rs still exists".to_string());
    }

    assert!(
        violations.is_empty(),
        "GpuOnly must not compute indicator results on the host and upload them:\n{}",
        violations.join("\n")
    );
}

#[test]
fn rvi_host_input_route_delegates_to_the_device_batch_implementation() {
    let rvi = source("src/cuda/rvi_wrapper.rs");
    let host_input_route = section(
        &rvi,
        "pub fn rvi_batch_dev(",
        "pub fn rvi_batch_dev_from_device_prices(",
    );

    assert!(
        host_input_route.contains("self.rvi_batch_dev_from_device_prices"),
        "rvi_batch_dev must upload the input and delegate to the native device implementation"
    );
    for forbidden in [
        "host_fallback",
        "rvi_batch_with_kernel",
        "Kernel::Scalar",
        "CPU fallback",
    ] {
        assert!(
            !host_input_route.contains(forbidden),
            "rvi_batch_dev contains forbidden host-compute token `{forbidden}`"
        );
    }
}

#[cfg(feature = "cuda-build-native")]
#[test]
fn r5_rvi_small_shape_host_input_executes_native_device_batch_without_fallback() {
    use crate::cuda::CudaRvi;
    use crate::indicators::rvi::{RviBatchRange, rvi_batch_with_kernel};
    use crate::utilities::enums::Kernel;
    use cust::memory::CopyDestination;

    const LEN: usize = 4_096;
    const FIRST_VALID: usize = 32;
    // Keep the established dispatch-level CUDA-versus-CPU RVI parity contract.
    const CUDA_PARITY_TOLERANCE: f32 = 5e-2;

    let mut input = vec![f32::NAN; LEN];
    for (index, value) in input.iter_mut().enumerate().skip(FIRST_VALID) {
        let x = index as f32;
        *value = (x * 0.017).sin() + 0.35 * (x * 0.0061).cos() + x * 0.0002;
    }
    let sweep = RviBatchRange {
        period: (10, 10, 0),
        ma_len: (14, 14, 0),
        matype: (1, 1, 0),
        devtype: (0, 0, 0),
    };

    let cpu_input: Vec<f64> = input.iter().map(|&value| value as f64).collect();
    let cpu = rvi_batch_with_kernel(&cpu_input, &sweep, Kernel::ScalarBatch)
        .expect("Rust CPU RVI reference must compute");
    assert!(
        cpu.rows
            .checked_mul(cpu.cols)
            .expect("RVI output shape must not overflow")
            < 2_000_000,
        "the R5 proof must stay below the removed host-fallback threshold"
    );

    let cuda = CudaRvi::new(0).expect(
        "R5 requires a real CUDA device and the exact native RVI module; skipping is forbidden",
    );
    let (device, device_combos) = cuda
        .rvi_batch_dev(&input, &sweep)
        .expect("small-shape RVI must execute the native device batch path");

    assert_eq!(device.rows, cpu.rows);
    assert_eq!(device.cols, cpu.cols);
    assert_eq!(device_combos.len(), cpu.combos.len());
    assert_eq!(device_combos.len(), 1);
    assert_eq!(device_combos[0].period, Some(10));
    assert_eq!(device_combos[0].ma_len, Some(14));
    assert_eq!(device_combos[0].matype, Some(1));
    assert_eq!(device_combos[0].devtype, Some(0));

    let mut device_values = vec![0.0f32; device.len()];
    device
        .buf
        .copy_to(&mut device_values)
        .expect("native RVI device output readback must succeed");
    assert_eq!(device_values.len(), cpu.values.len());

    let mut finite_values_compared = 0usize;
    for (index, (&expected, &actual)) in cpu.values.iter().zip(&device_values).enumerate() {
        if expected.is_nan() && actual.is_nan() {
            continue;
        }
        assert_eq!(
            expected.is_nan(),
            actual.is_nan(),
            "RVI NaN warmup parity mismatch at output index {index}: cpu={expected} cuda={actual}"
        );
        let absolute_error = (expected as f32 - actual).abs();
        assert!(
            absolute_error < CUDA_PARITY_TOLERANCE,
            "RVI parity mismatch at output index {index}: cpu={expected} cuda={actual} \
             abs_error={absolute_error} tolerance={CUDA_PARITY_TOLERANCE}"
        );
        finite_values_compared += 1;
    }
    assert!(
        finite_values_compared > LEN / 2,
        "R5 must compare a substantial finite RVI output tail"
    );
}
