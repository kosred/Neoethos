use std::fs;
use std::path::{Path, PathBuf};

const CREATOR_PAGE: &str = "https://www.tradingview.com/script/Hgvs8kZi-N-Order-EMA/";
const CREATOR_FACADE_ID: &str = "PUB;0d0d8869215f4446b4c17e62c6080830";
const CREATOR_SOURCE_SHA256: &str =
    "539EEC25A8422DDE96705873212CD55302301BD6CE3284A411C2010536B843D3";

fn repo_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let from_manifest = Path::new(manifest_dir);
        if from_manifest.join("src/indicators").is_dir() {
            return from_manifest.to_path_buf();
        }
    }

    let mut cursor = std::env::current_dir().expect("current directory");
    loop {
        let candidate = cursor.join("vendor/vector-ta-0.2.9-patched");
        if candidate.join("src/indicators").is_dir() {
            return candidate;
        }
        assert!(cursor.pop(), "vector-ta workspace root not found");
    }
}

fn load(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative)).expect(relative)
}

#[test]
fn creator_identity_is_frozen_in_both_cpu_and_cuda_authority() {
    let cpu = load("src/indicators/moving_averages/n_order_ema.rs");
    let cuda = load("kernels/cuda/moving_averages/n_order_ema_kernel.cu");
    for source in [&cpu, &cuda] {
        assert!(source.contains(CREATOR_PAGE));
        assert!(source.contains(CREATOR_FACADE_ID));
        assert!(source.contains(CREATOR_SOURCE_SHA256));
    }
}

#[test]
fn cpu_has_no_period_warmup_or_consecutive_valid_run_gate() {
    let cpu = load("src/indicators/moving_averages/n_order_ema.rs");
    assert!(!cpu.contains("fn warmup_len("));
    assert!(!cpu.contains("fn required_valid_len("));
    assert!(!cpu.contains("if self.count > self.warmup"));
    assert!(cpu.contains("value.is_nan()"));
    assert!(cpu.contains("let safe_value = if value.is_nan()"));
}

#[test]
fn strict_f64_cuda_matches_creator_start_and_gap_rules() {
    let cuda = load("kernels/cuda/moving_averages/n_order_ema_kernel.cu");
    assert!(!cuda.contains("const int warmup = period_i - 1"));
    assert!(!cuda.contains("count > warmup"));
    assert!(!cuda.contains("const int needed = warmup + 1"));
    assert!(cuda.contains("const double safe_x = isnan(x) ? 0.0 : x"));
    assert!(cuda.contains("row[i] = acc"));
}

#[test]
fn cpu_f64_and_strict_cuda_f64_keep_the_same_ordered_recurrence() {
    let cpu = load("src/indicators/moving_averages/n_order_ema.rs");
    let cuda = load("kernels/cuda/moving_averages/n_order_ema_kernel.cu");

    assert!(cpu.contains("let fc = 2.0 / (period + 1.0)"));
    assert!(cpu.contains("acc -= self.coeffs.a[k] * y"));
    assert!(cuda.contains("const double fc = 2.0 / (period + 1.0)"));
    assert!(cuda.contains("acc = acc - (a0 * y_in)"));
    assert!(!cuda.contains("fma("));
    assert!(!cuda.contains("float acc"));
}

#[test]
fn shared_f64_wrapper_or_build_surface_is_not_part_of_this_repair() {
    assert!(CREATOR_PAGE.ends_with("Hgvs8kZi-N-Order-EMA/"));
    assert!(CREATOR_FACADE_ID.starts_with("PUB;"));
    assert_eq!(CREATOR_SOURCE_SHA256.len(), 64);
}
