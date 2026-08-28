use std::fs;
use std::path::{Path, PathBuf};

const TA_LIB_COMMIT: &str = "3800d9ed0006fa63cab818737fbea998219419ce";
const TA_LIB_AD: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_AD.c";
const TA_LIB_AROON: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_AROON.c";
const TA_LIB_AROONOSC: &str = "https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_AROONOSC.c";

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

fn talib_ad_bar(high: f64, low: f64, close: f64, volume: f64) -> f64 {
    let range = high - low;
    if range > 0.0 {
        (((close - low) - (high - close)) / range) * volume
    } else {
        0.0
    }
}

fn talib_latest_extreme_at_row(
    high: &[f64],
    low: &[f64],
    row: usize,
    period: usize,
) -> (f64, f64, f64) {
    let start = row - period;
    let mut highest_idx = start;
    let mut lowest_idx = start;
    let mut highest = high[start];
    let mut lowest = low[start];
    for index in (start + 1)..=row {
        if high[index] >= highest {
            highest = high[index];
            highest_idx = index;
        }
        if low[index] <= lowest {
            lowest = low[index];
            lowest_idx = index;
        }
    }
    let factor = 100.0 / period as f64;
    let up = factor * (period - (row - highest_idx)) as f64;
    let down = factor * (period - (row - lowest_idx)) as f64;
    (up, down, factor * (highest_idx as f64 - lowest_idx as f64))
}

#[test]
fn talib_ad_carries_zero_and_negative_ranges() {
    let positive = talib_ad_bar(2.0, 1.0, 1.75, 8.0);
    let zero = talib_ad_bar(2.0, 2.0, 2.0, 8.0);
    let inverted = talib_ad_bar(1.0, 2.0, 1.75, 8.0);

    assert_eq!(positive.to_bits(), 4.0f64.to_bits());
    assert_eq!(zero.to_bits(), 0.0f64.to_bits());
    assert_eq!(inverted.to_bits(), 0.0f64.to_bits());
    assert!(TA_LIB_AD.contains(TA_LIB_COMMIT));
}

#[test]
fn talib_aroon_uses_latest_tie_and_period_plus_one_bar_window() {
    let high = [10.0, 10.0, 9.0];
    let low = [5.0, 6.0, 5.0];
    let (up, down, oscillator) = talib_latest_extreme_at_row(&high, &low, 2, 2);

    assert_eq!(up.to_bits(), 50.0f64.to_bits());
    assert_eq!(down.to_bits(), 100.0f64.to_bits());
    assert_eq!(oscillator.to_bits(), (-50.0f64).to_bits());
    assert!(TA_LIB_AROON.contains(TA_LIB_COMMIT));
    assert!(TA_LIB_AROONOSC.contains(TA_LIB_COMMIT));
}

#[test]
fn cpu_and_cuda_sources_pin_talib_ad_range_authority() {
    let cpu = source("src/indicators/ad.rs");
    let cuda = source("kernels/cuda/ad_kernel.cu");

    assert!(cpu.contains(TA_LIB_AD));
    assert!(cpu.contains("if hl > 0.0"));
    assert!(cpu.contains("_CMP_GT_OQ"));
    assert!(!cpu.contains("if hl != 0.0"));
    assert!(!cpu.contains("_CMP_NEQ_OQ"));
    assert!(!cpu.contains("_mm512_cmpneq_pd_mask"));

    assert!(cuda.contains(TA_LIB_AD));
    assert!(cuda.contains("if (!(hl > 0.0f)) return 0.0f;"));
    assert!(cuda.contains("if (!(hl > 0.0)) return 0.0;"));
    assert!(cuda.contains("if (hl > 0.0)"));
    assert!(!cuda.contains("if (hl != 0.0)"));
    assert!(!cuda.contains("if (hl == 0.0"));
}

#[test]
fn cpu_sources_pin_talib_latest_tie_semantics() {
    let aroon = source("src/indicators/aroon.rs");
    let oscillator = source("src/indicators/aroonosc.rs");

    assert!(aroon.contains(TA_LIB_AROON));
    assert!(aroon.contains("value >= current"));
    assert!(aroon.contains("value <= current"));
    assert!(aroon.contains("scale_100 * ((length - dist) as f64)"));
    assert!(aroon.contains("if high >= v"));
    assert!(aroon.contains("if low <= v"));
    assert!(!aroon.contains("if hv > max"));
    assert!(!aroon.contains("if lv < min"));
    assert!(!aroon.contains("if high > v"));
    assert!(!aroon.contains("if low < v"));
    assert!(!aroon.contains("mul_add(scale, 100.0)"));

    assert!(oscillator.contains(TA_LIB_AROONOSC));
    assert!(oscillator.contains("if hv >= max"));
    assert!(oscillator.contains("if lv <= min"));
    assert!(oscillator.contains("if last_val <= v_hi"));
    assert!(oscillator.contains("if last_val >= v_lo"));
    assert!(!oscillator.contains("if hv > max"));
    assert!(!oscillator.contains("if lv < min"));
    assert!(!oscillator.contains("if last_val < v_hi"));
    assert!(!oscillator.contains("if last_val > v_lo"));
}

#[test]
fn cuda_sources_pin_talib_latest_tie_semantics() {
    let aroon = source("kernels/cuda/aroon_kernel.cu");
    let oscillator = source("kernels/cuda/oscillators/aroonosc_kernel.cu");

    assert!(aroon.contains(TA_LIB_AROON));
    assert!(aroon.contains("if (hv >= mx)"));
    assert!(aroon.contains("if (lv <= mn)"));
    assert!(aroon.contains("scale_100 * (double)(length - dist_hi)"));
    assert!(aroon.contains("if (dq_max_val[last_slot] <= h)"));
    assert!(aroon.contains("if (dq_min_val[last_slot] >= l)"));
    assert!(!aroon.contains("if (hv > mx)"));
    assert!(!aroon.contains("if (lv < mn)"));
    assert!(!aroon.contains("fmaf(-(float)dist"));
    assert!(!aroon.contains("100.0 - dist_hi"));

    assert!(oscillator.contains(TA_LIB_AROONOSC));
    assert!(oscillator.contains("max_latest_update"));
    assert!(oscillator.contains("min_latest_update"));
    assert!(oscillator.contains("i > best_i"));
    assert!(oscillator.contains("if (hv >= mx)"));
    assert!(oscillator.contains("if (lv <= mn)"));
    assert!(!oscillator.contains("EARLIEST index"));
    assert!(!oscillator.contains("max_earliest_update"));
    assert!(!oscillator.contains("min_earliest_update"));
    assert!(!oscillator.contains("if (hv > mx)"));
    assert!(!oscillator.contains("if (lv < mn)"));
}
