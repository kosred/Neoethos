use std::fs;
use std::path::{Path, PathBuf};

const CHAN_GOLUB_LEVEQUE_1983: &str = "https://doi.org/10.1080/00031305.1983.10483115";
const NEUMAIER_1974: &str = "https://doi.org/10.1002/zamm.19740540106";
const RTX_PERIOD_9_STABLE_BITS_V2: u64 = 0x3efa_abdb_f868_38c1;

fn workspace_root() -> PathBuf {
    let here = Path::new(file!());
    here.parent()
        .and_then(Path::parent)
        .expect("contract lives under vector-ta/tests")
        .to_path_buf()
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace_root().join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
}

#[derive(Default)]
struct NeumaierOracleV2 {
    sum: f64,
    correction: f64,
}

impl NeumaierOracleV2 {
    fn add(&mut self, value: f64) {
        let updated = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - updated) + value;
        } else {
            self.correction += (value - updated) + self.sum;
        }
        self.sum = updated;
    }

    fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

fn floor_power_of_two_input_scale_oracle_v2(value: f64) -> f64 {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if exponent != 0 {
        return f64::from_bits(exponent << 52);
    }
    let fraction = bits & ((1_u64 << 52) - 1);
    f64::from_bits(1_u64 << (63 - fraction.leading_zeros() as u64))
}

fn scaled_two_pass_population_deviation_v2(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mut max_abs_input = 0.0f64;
    for &value in values {
        assert!(value.is_finite());
        max_abs_input = max_abs_input.max(value.abs());
    }
    if max_abs_input == 0.0 {
        return 0.0;
    }

    let scale = floor_power_of_two_input_scale_oracle_v2(max_abs_input);
    let anchor = values[0] / scale;
    let mut shifted = NeumaierOracleV2::default();
    for &value in values {
        let normalized_value = value / scale;
        let delta = normalized_value - anchor;
        shifted.add(delta);
    }

    let mean_delta = shifted.value() / n;
    let mut normalized_squares = NeumaierOracleV2::default();
    for &value in values {
        let normalized_value = value / scale;
        let centered = (normalized_value - anchor) - mean_delta;
        normalized_squares.add(centered.mul_add(centered, 0.0));
    }
    scale * (normalized_squares.value() / n).sqrt()
}

#[test]
fn independent_corrected_two_pass_oracle_pins_the_rtx_truth_bits() {
    let values = [
        0x3ff1_335e_310d_bf05,
        0x3ff1_3317_9f58_f63b,
        0x3ff1_3342_4cab_aa3d,
        0x3ff1_330f_a73c_f04b,
        0x3ff1_334d_3466_391a,
        0x3ff1_332d_6ece_13f5,
        0x3ff1_335a_34ff_bc0d,
        0x3ff1_3324_6a42_93f9,
        0x3ff1_333f_5d0d_2150,
    ]
    .map(f64::from_bits);

    assert_eq!(
        scaled_two_pass_population_deviation_v2(&values).to_bits(),
        RTX_PERIOD_9_STABLE_BITS_V2
    );
    assert!(CHAN_GOLUB_LEVEQUE_1983.starts_with("https://doi.org/"));
    assert!(NEUMAIER_1974.starts_with("https://doi.org/"));
}

#[test]
fn global_input_scaling_precedes_subtraction_of_finite_extremes() {
    let values = [f64::MAX, -f64::MAX];

    assert_eq!(
        scaled_two_pass_population_deviation_v2(&values).to_bits(),
        f64::MAX.to_bits(),
        "opposite finite extremes must not overflow before scaling"
    );
}

#[test]
fn rust_f64_deviation_uses_one_anchored_scaled_two_pass_authority() {
    let rust = source("src/indicators/deviation.rs");
    let production = rust
        .split("#[cfg(test)]")
        .next()
        .expect("deviation production source precedes tests");

    assert!(production.contains("struct NeumaierSumF64V2"));
    assert!(production.contains("fn floor_power_of_two_input_scale_v2"));
    assert!(production.contains("fn stable_population_deviation_window_v2"));
    assert!(production.contains(
        "deviation_population_f64_global_pow2_anchored_neumaier_two_pass_fma_sqrt_rn_v2"
    ));
    assert!(production.contains("let normalized_value = value / scale;"));
    assert!(production.contains("let delta = normalized_value - anchor;"));
    assert!(production.contains("shifted_sum.add(delta);"));
    assert!(production.contains("max_abs_input = max_abs_input.max(value.abs());"));
    assert!(production.contains("let centered = (normalized_value - anchor) - mean_delta;"));
    assert!(production.contains("centered.mul_add(centered, 0.0)"));
    assert!(!production.contains("struct StablePopulationVarianceV1"));
    assert!(!production.contains("slides_since_rebase"));
    assert!(!production.contains("var.abs() / (scale.max(1e-30)) < 1e-10"));
}

#[test]
fn strict_cuda_f64_deviation_transcribes_the_same_authority() {
    let cuda = source("kernels/cuda/deviation_kernel.cu");
    let marker = cuda
        .find("// S3 f64 LANE — deviation")
        .expect("strict-f64 marker must remain present");
    let strict = &cuda[marker..];

    assert!(strict.contains("neo_s3_neumaier_add_v2"));
    assert!(strict.contains("neo_s3_floor_power_of_two_input_scale_v2"));
    assert!(strict.contains("neo_s3_stable_population_deviation_window_v2"));
    assert!(strict.contains("const double normalized_value = __ddiv_rn(value, scale);"));
    assert!(strict.contains("const double delta = __dsub_rn(normalized_value, anchor);"));
    assert!(strict.contains("__fma_rn(centered, centered, 0.0)"));
    assert!(strict.contains("__dsqrt_rn(__ddiv_rn("));
    assert!(strict.contains("__dadd_rn("));
    assert!(strict.contains("__dsub_rn("));
    assert!(strict.contains("__ddiv_rn("));
    assert!(strict.contains("__dmul_rn("));
    assert!(strict.contains("const int output_index ="));
    assert!(!strict.contains("neo_s3_dev_var"));
    assert!(!strict.contains("const double scale = fabs(sumsq / n);"));
    assert!(!strict.contains("slides_since_rebase"));
}

#[test]
fn strict_cuda_wrapper_parallelizes_independent_deviation_windows() {
    let wrapper = source("src/cuda/neoethos_f64_wrapper.rs");
    let launch = wrapper
        .split("fn launch_chunk(")
        .nth(1)
        .expect("launch_chunk must remain present");

    assert!(launch.contains("if kernel == F64Kernel::Deviation"));
    assert!(launch.contains("GridSize::xy("));
    assert!(launch.contains("((cols as u32) + BLOCK_X - 1) / BLOCK_X"));
    assert!(launch.contains("rows as u32"));
}

#[test]
fn f32_deviation_remains_a_separate_precision_contract() {
    let cuda = source("kernels/cuda/deviation_kernel.cu");
    assert!(cuda.contains("extern \"C\" __global__ void deviation_build_prefix_f32"));
    assert!(cuda.contains("extern \"C\" __global__ void deviation_batch_f32"));
    assert!(cuda.contains("extern \"C\" __global__ void deviation_many_series_one_param_f32"));
}
