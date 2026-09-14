use std::fs;
use std::path::{Path, PathBuf};

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

#[test]
fn natr_cuda_routes_pin_talib_lookback_period_one_and_zero_rules() {
    let kernel = source("kernels/cuda/natr_kernel.cu");
    let production_f64 = source("kernels/cuda/neoethos_f64_kernels.cu");
    let production_natr = production_f64
        .split("// NATR — reference")
        .nth(1)
        .and_then(|tail| tail.split("// ADXR — reference").next())
        .expect("production f64 NATR section");

    for source in [kernel.as_str(), production_natr] {
        assert!(source.contains("first_valid + period"));
        assert!(source.contains("period <= 1"));
        assert!(source.contains("== 0.0"));
        assert!(!source.contains("NATR_TA_EPSILON"));
        assert!(!source.contains("first_valid + period - 1"));
        assert!(!source.contains("fma(tr - atr"));
        assert!(!source.contains("__fmaf_rn(alpha"));
    }
}

#[test]
fn nvi_cuda_routes_carry_zero_and_nonfinite_updates() {
    let kernel = source("kernels/cuda/nvi_kernel.cu");
    let wrapper = source("src/cuda/nvi_wrapper.rs");

    assert!(kernel.contains("prev_close != 0.0"));
    assert!(kernel.contains("isfinite(candidate)"));
    assert!(!wrapper.contains("try_launch_batch_scan"));
    assert!(!kernel.contains("nvi_scan_blocks_f32"));
    assert!(!kernel.contains("nvi_scan_block_products_f64"));
    assert!(!kernel.contains("nvi_apply_block_products_f32"));
}

#[test]
fn nvi_zero_lookback_accepts_one_seed_bar_on_every_route() {
    let cpu = source("src/indicators/nvi.rs");
    let wrapper = source("src/cuda/nvi_wrapper.rs");
    let kernel = source("kernels/cuda/nvi_kernel.cu");

    assert!(!cpu.contains("len() - first < 2"));
    assert!(!cpu.contains("cols - first < 2"));
    assert!(!wrapper.contains("need >= 2 after first valid"));
    assert!(!kernel.contains("(n - first_valid) < 2"));
}
