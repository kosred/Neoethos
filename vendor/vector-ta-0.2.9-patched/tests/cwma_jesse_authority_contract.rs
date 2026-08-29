use std::fs;
use std::path::{Path, PathBuf};

const CREATOR_COMMIT: &str =
    "https://github.com/jesse-ai/jesse/commit/2f24de176d62e10d38f435e74590bad451815d6d";
const CREATOR_SOURCE: &str = "https://raw.githubusercontent.com/jesse-ai/jesse/2f24de176d62e10d38f435e74590bad451815d6d/jesse/indicators/cwma.py";

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

fn reviewed_routeable_subset_close_v3() -> Vec<f64> {
    (0..4_096)
        .map(|row| {
            let drift = row as f64 * 0.000_000_7;
            let wave = match row % 11 {
                0 => 0.000_041,
                1 => -0.000_027,
                2 => 0.000_013,
                3 => -0.000_036,
                4 => 0.000_022,
                5 => -0.000_009,
                6 => 0.000_033,
                7 => -0.000_019,
                8 => 0.000_006,
                9 => -0.000_031,
                _ => 0.000_017,
            };
            1.075 + drift + wave
        })
        .collect()
}

fn creator_weights_and_norm(period: usize) -> (Vec<f64>, f64) {
    let mut weights = Vec::with_capacity(period - 1);
    let mut norm = 0.0;
    for offset in 0..period - 1 {
        let base = (period - offset) as f64;
        let weight = base * base * base;
        weights.push(weight);
        norm += weight;
    }
    (weights, norm)
}

fn creator_value(data: &[f64], row: usize, period: usize) -> f64 {
    let (weights, norm) = creator_weights_and_norm(period);
    let mut sum = 0.0;
    for (offset, weight) in weights.iter().copied().enumerate() {
        let term = data[row - offset] * weight;
        sum += term;
    }
    sum / norm
}

fn superseded_avx2_tree(data: &[f64], row: usize, period: usize) -> f64 {
    let (weights, norm) = creator_weights_and_norm(period);
    let mut lanes = [0.0; 4];
    let vector_blocks = weights.len() / 4;
    for block in 0..vector_blocks {
        for lane in 0..4 {
            let offset = block * 4 + (3 - lane);
            lanes[lane] = data[row - offset].mul_add(weights[offset], lanes[lane]);
        }
    }
    let mut tail = 0.0;
    for (offset, weight) in weights.iter().copied().enumerate().skip(vector_blocks * 4) {
        tail = data[row - offset].mul_add(weight, tail);
    }
    (((lanes[2] + lanes[0]) + (lanes[3] + lanes[1])) + tail) * (1.0 / norm)
}

#[test]
fn gate_220_fixture_proves_creator_order_not_cpu_avx_tree() {
    let close = reviewed_routeable_subset_close_v3();
    let creator = creator_value(&close, 18, 14);
    let superseded = superseded_avx2_tree(&close, 18, 14);

    assert_eq!(creator.to_bits(), 0x3ff1_333f_5fc7_4bcd);
    assert_eq!(superseded.to_bits(), 0x3ff1_333f_5fc7_4bcc);
    assert_ne!(creator.to_bits(), superseded.to_bits());
    assert!(CREATOR_COMMIT.ends_with("2f24de176d62e10d38f435e74590bad451815d6d"));
}

#[test]
fn cpu_scalar_batch_stream_and_simd_pin_creator_operation_order() {
    let cpu = source("src/indicators/moving_averages/cwma.rs");

    assert!(cpu.contains(CREATOR_SOURCE));
    assert!(cpu.contains("fn cwma_creator_exact_value_v1("));
    assert!(cpu.contains("let term = data[row - offset] * weights[offset];"));
    assert!(cpu.contains("sum += term;"));
    assert!(cpu.contains("sum / norm"));
    assert!(cpu.contains("self.sum_weighted() / self.norm"));
    assert!(cpu.contains("_mm256_mul_pd"));
    assert!(cpu.contains("_mm256_add_pd"));
    assert!(cpu.contains("_mm256_div_pd"));
    assert!(cpu.contains("_mm512_mul_pd"));
    assert!(cpu.contains("_mm512_add_pd"));
    assert!(cpu.contains("_mm512_div_pd"));
    assert!(!cpu.contains("_mm256_fmadd_pd"));
    assert!(!cpu.contains("_mm512_fmadd_pd"));
    assert!(!cpu.contains("sum * inv_norm"));
    assert!(!cpu.contains("self.sum_weighted() * self.inv_norm"));
}

#[test]
fn strict_f64_cuda_route_pins_creator_operation_order() {
    let cuda = source("kernels/cuda/moving_averages/cwma_kernel.cu");
    let f64_lane = cuda
        .split("S4 f64 LANE")
        .nth(1)
        .expect("CWMA f64 lane marker must remain present");

    assert!(f64_lane.contains(CREATOR_SOURCE));
    assert!(f64_lane.contains("__dmul_rn"));
    assert!(f64_lane.contains("__dadd_rn"));
    assert!(f64_lane.contains("__ddiv_rn"));
    assert!(!f64_lane.contains("fma("));
    assert!(!f64_lane.contains("inv_norm"));
    assert!(!f64_lane.contains("double s0"));
}
