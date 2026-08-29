use std::fs;
use std::path::{Path, PathBuf};

const CREATOR_SOURCE: &str = "https://pine-facade.tradingview.com/pine-facade/get/PUB%3B1763d63e649c4be4baf7fe86bee776b8/last";

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
fn cpu_and_cuda_sources_pin_creator_pine_warmup_and_na_semantics() {
    let cpu = source("src/indicators/adaptive_momentum_oscillator.rs");
    let cuda = source("kernels/cuda/adaptive_momentum_oscillator_kernel.cu");
    let wrapper = source("src/cuda/neoethos_f64_wrapper.rs");

    assert!(cpu.contains(CREATOR_SOURCE));
    assert!(cpu.contains("let mut max_momentum: f64 = 0.0;"));
    assert!(cpu.contains("let mut selected_delta = 0.0;"));
    assert!(cpu.contains("if self.count == self.length"));
    assert!(!cpu.contains("self.count >= self.length"));
    assert!(!cpu.contains("if !past.is_finite()"));

    assert!(cuda.contains("double max_momentum = 0.0;"));
    assert!(cuda.contains("double selected_delta = 0.0;"));
    assert!(cuda.contains("if (isnan(max_momentum) || isnan(absolute_momentum))"));
    assert!(!cuda.contains("raw_count >= length"));
    assert!(!cuda.contains("raw_count >= L"));

    assert!(wrapper.contains("if valid < max_smoothing_length"));
    assert!(
        wrapper.contains("not enough valid data: needed={max_smoothing_length}, valid={valid}")
    );
    assert!(!wrapper.contains("max_needed = max_needed.max(length.checked_add(smoothing_length)"));
    assert!(!wrapper.contains("length+smoothing_length overflow"));
}
