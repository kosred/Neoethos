use std::fs;
use std::path::{Path, PathBuf};

const EHLERS_WINDOWING_PRIMARY_INDEX: &str =
    "https://technical.traders.com/archive/display2.asp?mo=SEP&yr=2021";
const EHLERS_AUTHOR_CODE: &str =
    "https://traders.com/Documentation/FEEDbk_docs/2021/12/TradersTips.html";
const EHMA_F64_AUTHORITY_V2: &str = "ehma_hann_f64_msun_ddangle_symmetric_pow2_anchored_dot2_v2";
const RTX_REVIEWED_ROW_13_BITS_V2: u64 = 0x3ff1_3338_cd76_5d61;
const CANONICAL_QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;

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
struct IndependentDot2 {
    sum: f64,
    correction: f64,
}

impl IndependentDot2 {
    fn add_product(&mut self, left: f64, right: f64) {
        let product = left * right;
        let product_error = left.mul_add(right, -product);
        let updated = self.sum + product;
        let recovered = updated - self.sum;
        let addition_error = (self.sum - (updated - recovered)) + (product - recovered);
        self.sum = updated;
        self.correction += product_error + addition_error;
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

fn independent_hann_weights(period: usize) -> (Vec<f64>, f64) {
    let mut weights = vec![0.0; period];
    for k in 1..=((period + 1) / 2) {
        let half_angle = std::f64::consts::PI * (k as f64 / (period as f64 + 1.0));
        let sine = half_angle.sin();
        let weight = 2.0 * (sine * sine);
        weights[k - 1] = weight;
        weights[period - k] = weight;
    }
    let mut coefficient = IndependentDot2::default();
    for &weight in &weights {
        coefficient.add_product(1.0, weight);
    }
    (weights, coefficient.value())
}

fn floor_power_of_two_scale(value: f64) -> f64 {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if exponent != 0 {
        return f64::from_bits(exponent << 52);
    }
    let fraction = bits & ((1_u64 << 52) - 1);
    f64::from_bits(1_u64 << (63 - fraction.leading_zeros() as u64))
}

fn independent_ehlers_hann_mean(values: &[f64]) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::from_bits(CANONICAL_QNAN_BITS);
    }
    let (weights, coefficient) = independent_hann_weights(values.len());
    let max_abs = values
        .iter()
        .fold(0.0_f64, |current, value| current.max(value.abs()));
    if max_abs == 0.0 {
        return 0.0;
    }
    let scale = floor_power_of_two_scale(max_abs);
    let anchor = values[0] / scale;
    let mut shifted = IndependentDot2::default();
    for (&value, &weight) in values.iter().zip(&weights) {
        shifted.add_product(value / scale - anchor, weight);
    }
    scale * (anchor + shifted.value() / coefficient)
}

fn reviewed_routeable_row_13() -> [f64; 14] {
    [
        0x3ff1_335e_310d_bf05,
        0x3ff1_3317_9f58_f63b,
        0x3ff1_3342_4cab_aa3d,
        0x3ff1_330f_a73c_f04b,
        0x3ff1_334d_3466_391a,
        0x3ff1_332d_6ece_13f5,
        0x3ff1_335a_34ff_bc0d,
        0x3ff1_3324_6a42_93f9,
        0x3ff1_333f_5d0d_2150,
        0x3ff1_3319_4cd8_1fe7,
        0x3ff1_334c_5da6_a444,
        0x3ff1_3366_4401_b790,
        0x3ff1_331f_b24c_eec6,
        0x3ff1_334a_5f9f_a2c8,
    ]
    .map(f64::from_bits)
}

#[test]
fn independent_creator_formula_pins_the_rtx_truth_bit() {
    let value = independent_ehlers_hann_mean(&reviewed_routeable_row_13());
    assert_eq!(value.to_bits(), RTX_REVIEWED_ROW_13_BITS_V2);
    assert!(EHLERS_WINDOWING_PRIMARY_INDEX.starts_with("https://technical.traders.com/"));
    assert!(EHLERS_AUTHOR_CODE.starts_with("https://traders.com/"));
}

#[test]
fn independent_oracle_preserves_constants_large_offsets_and_gap_identity() {
    for period in [1, 2, 3, 14, 31, 64, 127, 256, 512] {
        let constant = f64::from_bits(0x5f30_0000_0000_0000);
        let values = vec![constant; period];
        assert_eq!(
            independent_ehlers_hann_mean(&values).to_bits(),
            constant.to_bits()
        );
    }

    let mut offset = [f64::from_bits(0x42b0_0000_0000_0000); 14];
    for (index, value) in offset.iter_mut().enumerate() {
        *value += (index as f64 - 7.0) * 0.000_244_140_625;
    }
    assert!(independent_ehlers_hann_mean(&offset).is_finite());

    offset[6] = f64::NAN;
    assert_eq!(
        independent_ehlers_hann_mean(&offset).to_bits(),
        CANONICAL_QNAN_BITS
    );
}

#[test]
fn rust_f64_ehma_uses_one_deterministic_authority() {
    let rust = source("src/indicators/moving_averages/ehma.rs");
    let production = rust
        .split("#[cfg(test)]")
        .next()
        .expect("EHMA production source precedes tests");

    for required in [
        EHMA_F64_AUTHORITY_V2,
        "fn ehma_msun_k_cos_v2",
        "fn ehma_msun_k_sin_v2",
        "fn ehma_reduce_pio2_v2",
        "fn ehma_half_angle_v2",
        "fn build_hann_weights_v2",
        "struct EhmaDot2V2",
        "fn floor_power_of_two_scale_v2",
        "fn ehma_stable_window_indexed_v2",
        "for &weight in weights.iter()",
    ] {
        assert!(
            production.contains(required),
            "missing CPU authority token `{required}`"
        );
    }
    assert!(!production.contains("fn build_hann_weights_rec"));
    assert!(!production.contains("omega.sin_cos()"));
    assert!(!production.contains("_mm256_fmadd_pd"));
    assert!(!production.contains("_mm512_fmadd_pd"));
    assert!(!production.contains("sum_x:"));
    assert!(!production.contains("z_re:"));
    assert!(!production.contains("z_im:"));
    assert!(!production.contains("cos_wp:"));
    assert!(!production.contains("for &weight in &weights"));
}

#[test]
fn strict_cuda_f64_ehma_transcribes_the_same_authority() {
    let cuda = source("kernels/cuda/moving_averages/ehma_kernel.cu");
    let marker = cuda
        .find("S4 f64 LANE")
        .expect("EHMA strict-f64 marker must remain present");
    let strict = &cuda[marker..];

    for required in [
        EHMA_F64_AUTHORITY_V2,
        "neo_ehma_msun_k_cos_v2",
        "neo_ehma_msun_k_sin_v2",
        "neo_ehma_reduce_pio2_v2",
        "neo_ehma_half_angle_v2",
        "neo_ehma_build_weights_v2",
        "neo_ehma_dot2_add_product_v2",
        "neo_ehma_floor_power_of_two_scale_v2",
        "__fma_rn",
        "__ddiv_rn",
    ] {
        assert!(
            strict.contains(required),
            "missing CUDA authority token `{required}`"
        );
    }
    assert!(!strict.contains("sincos("));
    assert!(!strict.contains("double cm ="));
    assert!(!strict.contains("double sm ="));
}

#[test]
fn f32_ehma_remains_a_separate_precision_contract() {
    let cuda = source("kernels/cuda/moving_averages/ehma_kernel.cu");
    let marker = cuda
        .find("S4 f64 LANE")
        .expect("EHMA strict-f64 marker must remain present");
    let f32 = &cuda[..marker];

    for entry in [
        "ehma_batch_f32",
        "ehma_multi_series_one_param_f32",
        "ehma_batch_tiled_f32_2x_tile128",
        "ehma_ms1p_tiled_f32_tx128_ty2",
    ] {
        assert!(f32.contains(entry), "f32 EHMA entry `{entry}` disappeared");
    }
}

#[test]
fn strict_wrapper_keeps_the_reviewed_period_bound() {
    let wrapper = source("src/cuda/neoethos_f64_wrapper.rs");
    assert!(wrapper.contains("const EHMA_MAX_PERIOD: usize = 512;"));
    assert!(wrapper.contains("F64Kernel::Ehma => Some(EHMA_MAX_PERIOD)"));
}
