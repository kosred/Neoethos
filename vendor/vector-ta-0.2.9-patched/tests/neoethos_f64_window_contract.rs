const ATR_PERCENTILE_CUDA: &str = include_str!("../kernels/cuda/atr_percentile_kernel.cu");
const CCI_CYCLE_CUDA: &str = include_str!("../kernels/cuda/oscillators/cci_cycle_kernel.cu");
const GARMAN_KLASS_CUDA: &str = include_str!("../kernels/cuda/garman_klass_volatility_kernel.cu");

fn strict_entry<'a>(source: &'a str, symbol: &str) -> &'a str {
    source
        .split_once(symbol)
        .unwrap_or_else(|| panic!("missing strict f64 symbol {symbol}"))
        .1
}

#[test]
fn atr_percentile_strict_f64_consumes_the_versioned_window_anchor() {
    let body = strict_entry(ATR_PERCENTILE_CUDA, "atr_percentile_neo_batch_f64");

    assert!(body.contains("const int anchor = periods[combo];"));
    assert!(body.contains("if (anchor <= 0) return;"));
    assert!(body.contains("const int AL = anchor_atr_length_v1(anchor);"));
    assert!(body.contains("const int PL = anchor;"));
    assert!(body.contains("PL > len - AL"));
    assert!(!body.contains("PL > len - AL + 1"));
    assert!(body.contains("anchor_atr_length_v1"));
    assert!(ATR_PERCENTILE_CUDA.contains("const int quotient = anchor / 5;"));
    assert!(ATR_PERCENTILE_CUDA.contains("const int remainder = anchor % 5;"));
    assert!(
        ATR_PERCENTILE_CUDA.contains("const int scaled = quotient + (remainder >= 3 ? 1 : 0);")
    );
    assert!(!ATR_PERCENTILE_CUDA.contains("(anchor + 2) / 5"));
    assert!(
        body.find("if (anchor <= 0) return;")
            < body.find("const int AL = anchor_atr_length_v1(anchor);")
    );
    assert!(!body.contains("(void)periods"));
    assert!(!body.contains("ATRP_NEO_ATR_LEN"));
    assert!(!body.contains("ATRP_NEO_PCT_LEN"));
}

#[test]
fn cci_cycle_strict_f64_consumes_length_and_preserves_the_v9_schedule() {
    let body = strict_entry(CCI_CYCLE_CUDA, "cci_cycle_neo_batch_f64");

    assert!(body.contains("const int length = periods[combo];"));
    assert!(CCI_CYCLE_CUDA.contains("#define NEO_CCICYC_MAX_LENGTH 200"));
    assert!(CCI_CYCLE_CUDA.contains("#define NEO_CCICYC_CLASSIC_SEMANTIC_VERSION 9"));
    assert!(body.contains("const int half = length / 2;"));
    assert!(body.contains("const int slot = segment_bars % length;"));
    assert!(body.contains("ema_short = ema_short_seed / (double)half;"));
    assert!(body.contains("ema_long = ema_long_seed / (double)length;"));
    assert!(body.contains("rma = rma_seed / (double)rma_length;"));
    assert!(body.contains("if (!isfinite(close))"));
    assert!(body.contains("segment_bars = 0;"));
    assert!(body.contains("previous_f1 = 0.0;"));
    assert!(body.contains("previous_pff = 0.0;"));
    assert!(body.contains("o[i] = NEO_F64_NAN;"));
    assert!(body.contains("o[i] = pff;"));
    assert!(!body.contains("use_fused"));
    assert!(!body.contains("(void)periods"));
    assert!(!body.contains("NEO_CCICYC_LENGTH 10"));
}

#[test]
fn garman_klass_strict_f64_consumes_each_requested_lookback() {
    let body = strict_entry(GARMAN_KLASS_CUDA, "garman_klass_volatility_neo_batch_f64");

    assert!(body.contains("const int LB = periods[combo];"));
    assert!(body.contains("for (int i = n - 1; i >= warmup; --i)"));
    assert!(body.contains("prefix_sum_ws"));
    assert!(body.contains("int invalid_count = 0;"));
    assert!(body.contains("if (invalid_count == 0)"));
    assert!(body.contains("invalid_count -= 1;"));
    assert!(body.contains("invalid_count += 1;"));
    assert!(
        !body.contains("for (int j = ws; j <= i; ++j)"),
        "GK validity must roll in O(N), not rescan every lookback window"
    );
    assert!(!body.contains("(void)periods"));
    assert!(!body.contains("NEO_GK_LOOKBACK"));
}
