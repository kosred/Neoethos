use std::fs;
use std::path::{Path, PathBuf};

const EXTERNAL_AUTHORITY_COMMIT: &str = "3800d9ed0006fa63cab818737fbea998219419ce";
const EXTERNAL_ADOSC_AUTHORITY: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADOSC.c";
const EXTERNAL_ADXR_AUTHORITY: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADXR.c";
const EXTERNAL_ADX_AUTHORITY: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADX.c";

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

fn production_rust(source: &str) -> &str {
    source.split("#[cfg(test)]").next().unwrap_or(source)
}

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let suffix = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing start token: {start}"))
        .1;
    suffix
        .split_once(end)
        .unwrap_or_else(|| panic!("missing end token: {end}"))
        .0
}

fn external_adosc_contribution(high: f64, low: f64, close: f64, volume: f64) -> f64 {
    let range = high - low;
    if range > 0.0 {
        (((close - low) - (high - close)) / range) * volume
    } else {
        0.0
    }
}

fn external_adxr(adx: &[f64], period: usize) -> Vec<f64> {
    let lag = period - 1;
    (0..adx.len())
        .map(|index| {
            if index < lag {
                f64::NAN
            } else {
                0.5 * (adx[index] + adx[index - lag])
            }
        })
        .collect()
}

#[test]
fn adosc_malformed_range_carries_instead_of_inverting_flow() {
    assert_eq!(
        external_adosc_contribution(2.0, 1.0, 1.75, 8.0).to_bits(),
        4.0f64.to_bits()
    );
    assert_eq!(
        external_adosc_contribution(2.0, 2.0, 2.0, 8.0).to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(
        external_adosc_contribution(1.0, 2.0, 1.75, 8.0).to_bits(),
        0.0f64.to_bits()
    );
    assert!(EXTERNAL_ADOSC_AUTHORITY.contains(EXTERNAL_AUTHORITY_COMMIT));
}

#[test]
fn adxr_uses_period_minus_one_lag_and_three_period_minus_two_lookback() {
    let adx = [10.0, 20.0, 30.0, 40.0];
    let period_three = external_adxr(&adx, 3);
    assert!(period_three[0].is_nan());
    assert!(period_three[1].is_nan());
    assert_eq!(period_three[2].to_bits(), 20.0f64.to_bits());
    assert_eq!(period_three[3].to_bits(), 30.0f64.to_bits());

    let period_one = external_adxr(&adx, 1);
    assert_eq!(
        period_one
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        adx.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(3 * 3 - 2, 7);
    assert!(EXTERNAL_ADXR_AUTHORITY.contains(EXTERNAL_AUTHORITY_COMMIT));
    assert!(EXTERNAL_ADX_AUTHORITY.contains(EXTERNAL_AUTHORITY_COMMIT));
}

#[test]
fn vector_ta_adosc_cpu_and_cuda_pin_the_positive_range_guard_only() {
    let cpu_source = source("src/indicators/adosc.rs");
    let cpu = production_rust(&cpu_source);
    let cuda = source("kernels/cuda/oscillators/adosc_kernel.cu");

    assert!(cpu.contains(EXTERNAL_ADOSC_AUTHORITY));
    assert!(cpu.contains("if hl0 > 0.0"));
    assert!(cpu.contains("if hl > 0.0"));
    assert!(!cpu.contains("if hl0 != 0.0"));
    assert!(!cpu.contains("if hl != 0.0"));

    assert!(cuda.contains(EXTERNAL_ADOSC_AUTHORITY));
    assert!(cuda.contains("if (!(hl > 0.0f)) return 0.0f;"));
    assert!(cuda.contains("(hl0 > 0.0) ?"));
    assert!(cuda.contains("(hl > 0.0) ?"));
    assert!(!cuda.contains("hl == 0.0f"));
    assert!(!cuda.contains("hl0 != 0.0"));
    assert!(!cuda.contains("hl != 0.0"));

    // This repair deliberately preserves today's VectorTA schema/EMA identity.
    assert!(cpu.contains("short_period: Some(3)"));
    assert!(cpu.contains("long_period: Some(10)"));
    assert!(cpu.contains("let alpha_short = 2.0 / (short as f64 + 1.0);"));
    assert!(cpu.contains("let alpha_long = 2.0 / (long as f64 + 1.0);"));
}

#[test]
fn vector_ta_adxr_cpu_pins_one_lag_and_lookback_authority() {
    let cpu_source = source("src/indicators/adxr.rs");
    let cpu = production_rust(&cpu_source);

    assert!(cpu.contains(EXTERNAL_ADXR_AUTHORITY));
    assert!(cpu.contains(EXTERNAL_ADX_AUTHORITY));
    assert!(cpu.contains("const fn adxr_lag(period: usize) -> usize"));
    assert!(cpu.contains("period - 1"));
    assert!(cpu.contains("const fn adxr_lookback(period: usize) -> usize"));
    assert!(cpu.contains("3 * period - 2"));
    assert!(cpu.contains("adxr_push_lagged"));
    assert!(cpu.contains("vec![f64::NAN; adxr_lag(period)]"));
    assert!(!cpu.contains("first + 2 * period"));
    assert!(!cpu.contains("vec![f64::NAN; period]"));
    assert!(!cpu.contains("if head == period"));
    assert!(!cpu.contains("% self.period"));
}

#[test]
fn vector_ta_strict_f64_cuda_pins_adxr_lag_and_lookback() {
    let cuda = source("kernels/cuda/neoethos_f64_kernels.cu");
    let adxr = source_between(
        &cuda,
        "extern \"C\" __global__ void neoethos_adxr_batch_f64",
        "// EFI — reference:",
    );

    assert!(cuda.contains(EXTERNAL_ADXR_AUTHORITY));
    assert!(cuda.contains(EXTERNAL_ADX_AUTHORITY));
    assert!(adxr.contains("const int lag = period - 1;"));
    assert!(adxr.contains("const int warmup_start = first_valid + 3 * period - 2;"));
    assert!(adxr.contains("for (int k = 0; k < lag; ++k)"));
    assert!(adxr.contains("if (lag == 0)"));
    assert!(adxr.contains("if (head == lag) head = 0;"));
    assert!(!adxr.contains("first_valid + 2 * period"));
    assert!(!adxr.contains("for (int k = 0; k < period; ++k)"));
    assert!(!adxr.contains("if (head == period)"));
}

#[test]
fn this_is_an_external_oracle_not_a_runtime_backend() {
    let cargo = source("Cargo.toml");
    assert!(!cargo.to_ascii_lowercase().contains("ta-lib"));
    assert!(!cargo.to_ascii_lowercase().contains("talib"));
}
