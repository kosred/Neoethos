use std::fs;
use std::path::{Path, PathBuf};

const PRIMARY_AUTHORITY: &str =
    "https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_OBV.c";

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

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing start token: {start}"));
    let tail = &source[start_index..];
    let end_index = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing end token after {start}: {end}"));
    &tail[..end_index]
}

fn source_from<'a>(source: &'a str, start: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing start token: {start}"));
    &source[start_index..]
}

#[test]
fn primary_authority_is_the_official_talib_obv_implementation() {
    assert_eq!(
        PRIMARY_AUTHORITY,
        "https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_OBV.c"
    );
}

#[test]
fn cpu_scalar_stream_and_batch_pin_lookback_zero_and_first_volume_seed() {
    let cpu = source("src/indicators/obv.rs");
    let avx = source_between(&cpu, "pub unsafe fn obv_avx2", "pub unsafe fn obv_avx512");

    assert!(cpu.contains("let mut prev_obv = volume[first_valid];"));
    assert!(cpu.contains("*out.get_unchecked_mut(first_valid) = prev_obv;"));
    assert!(cpu.contains("self.prev_obv = volume;"));
    assert!(cpu.contains("return Some(volume);"));
    assert!(!cpu.contains("v.mul_add(s, prev_obv)"));
    assert!(!cpu.contains("volume.mul_add(s, self.prev_obv)"));
    assert!(avx.contains("obv_scalar(close, volume, first_valid, out)"));
    assert!(!avx.contains("_mm_add_pd"));
}

#[test]
fn cuda_f32_routes_are_sequential_and_seed_from_first_volume() {
    let wrapper = source("src/cuda/obv_wrapper.rs");
    let kernel = source("kernels/cuda/obv_kernel.cu");
    let batch = source_between(
        &kernel,
        "void obv_batch_f32_serial_ref",
        "void obv_many_series_one_param_time_major_f32",
    );
    let many_series = source_between(
        &kernel,
        "void obv_many_series_one_param_time_major_f32",
        "Native f64 batch route",
    );

    for retired in [
        "obv_batch_f32_pass1_tilescan",
        "obv_batch_f32_pass2_scan_block_sums",
        "obv_batch_f32_pass3_add_offsets",
    ] {
        assert!(
            !wrapper.contains(retired),
            "wrapper still routes through {retired}"
        );
        assert!(!kernel.contains(retired), "kernel still ships {retired}");
    }
    assert!(batch.contains("double prev_obv = (double)volume[fv];"));
    assert!(many_series.contains("double prev_obv = (double)volume_tm[idx0];"));
    for route in [batch, many_series] {
        assert!(route.contains("prev_obv += v;"));
        assert!(route.contains("prev_obv -= v;"));
    }
    assert!(!kernel.contains("FPair"));
    assert!(!kernel.contains("fma("));
}

#[test]
fn both_active_f64_cuda_symbols_pin_talib_order_and_seed() {
    let dedicated = source("kernels/cuda/obv_kernel.cu");
    let shared = source("kernels/cuda/neoethos_f64_kernels.cu");
    let dedicated_obv = source_from(&dedicated, "void obv_neo_batch_f64");
    let shared_obv = source_between(
        &shared,
        "extern \"C\" __global__ void neoethos_obv_batch_f64",
        "// VWAP",
    );

    for route in [dedicated_obv, shared_obv] {
        assert!(route.contains("double prev_obv = volume[first_valid];"));
        assert!(route.contains("prev_obv += v;"));
        assert!(route.contains("prev_obv -= v;"));
        assert!(!route.contains("fma("));
    }
}
