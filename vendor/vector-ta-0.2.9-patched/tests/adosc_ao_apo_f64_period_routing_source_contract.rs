const ADOSC_CUDA: &[u8] = include_bytes!("../kernels/cuda/oscillators/adosc_kernel.cu");
const AO_CUDA: &[u8] = include_bytes!("../kernels/cuda/oscillators/ao_kernel.cu");
const APO_CUDA: &[u8] = include_bytes!("../kernels/cuda/moving_averages/apo_kernel.cu");

fn function_body<'a>(source: &'a str, symbol: &str) -> &'a str {
    let symbol_start = source
        .find(symbol)
        .unwrap_or_else(|| panic!("missing CUDA symbol {symbol}"));
    let open = source[symbol_start..]
        .find('{')
        .map(|offset| symbol_start + offset)
        .unwrap_or_else(|| panic!("missing body for CUDA symbol {symbol}"));

    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for CUDA symbol {symbol}");
}

#[test]
fn adosc_f64_routes_each_swept_anchor_into_the_vector_ta_3_to_10_tuple() {
    let source = String::from_utf8_lossy(ADOSC_CUDA);
    let body = function_body(&source, "adosc_neo_batch_f64");

    assert!(body.contains("const int long_period = periods[combo];"));
    assert!(
        body.contains("(3LL * (long long)long_period + 5LL) / 10LL"),
        "ADOSC must use integer half-up scaling of short=3 against long=10"
    );
    assert!(body.contains("const double alpha_short = 2.0 / ((double)short_period + 1.0);"));
    assert!(body.contains("const double alpha_long = 2.0 / ((double)long_period + 1.0);"));
    assert!(!body.contains("(void)periods"));

    // VectorTA's own f64 contract is live from bar zero: both EMAs are seeded
    // from the first cumulative A/D value, so the first output is exact +0.
    assert!(body.contains("o[0] = short_ema - long_ema;"));
}

#[test]
fn ao_f64_routes_each_swept_anchor_and_uses_the_scaled_long_for_warmup() {
    let source = String::from_utf8_lossy(AO_CUDA);
    let body = function_body(&source, "neoethos_ao_batch_f64");

    assert!(body.contains("const int longp = periods[r];"));
    assert!(
        body.contains("(5LL * (long long)longp + 17LL) / 34LL"),
        "AO must use integer half-up scaling of short=5 against long=34"
    );
    assert!(!body.contains("(void)periods"));
    assert!(body.contains("const int warm = first_valid + longp - 1;"));
    assert!(body.contains("row[i] = fma(short_sum, inv_s, -(long_sum * inv_l));"));
}

#[test]
fn apo_f64_routes_each_swept_anchor_and_preserves_vector_ta_first_bar_seeding() {
    let source = String::from_utf8_lossy(APO_CUDA);
    let body = function_body(&source, "neoethos_apo_batch_f64");

    assert!(body.contains("const int long_p = periods[r];"));
    assert!(
        body.contains("(10LL * (long long)long_p + 10LL) / 20LL"),
        "APO must use integer half-up scaling of short=10 against long=20"
    );
    assert!(!body.contains("(void)periods"));
    assert!(body.contains("for (int i = 0; i < first_valid; ++i)"));
    assert!(body.contains("row[first_valid] = 0.0;"));
    assert!(body.contains("se = alpha_s * p0 + oma_s * se;"));
    assert!(
        !body.contains("fma("),
        "VectorTA APO uses three-rounding a*b+c updates"
    );
}

#[test]
fn audited_sweep_anchors_expand_to_the_same_integer_tuples_as_the_cpu_plan() {
    fn half_up(default: i64, anchor: i64, swept: i64) -> i64 {
        ((default * swept + anchor / 2) / anchor).max(1)
    }

    let swept = [7, 21, 50, 100, 200];

    // The unsuffixed/base pass sends the largest registry default as its ABI
    // anchor, so the same reconstruction must be the identity at that anchor.
    assert_eq!((half_up(3, 10, 10), 10), (3, 10));
    assert_eq!((half_up(5, 34, 34), 34), (5, 34));
    assert_eq!((half_up(10, 20, 20), 20), (10, 20));

    let adosc: Vec<(i64, i64)> = swept
        .iter()
        .map(|&long| (half_up(3, 10, long), long))
        .collect();
    let ao: Vec<(i64, i64)> = swept
        .iter()
        .map(|&long| (half_up(5, 34, long), long))
        .collect();
    let apo: Vec<(i64, i64)> = swept
        .iter()
        .map(|&long| (half_up(10, 20, long), long))
        .collect();

    assert_eq!(adosc, [(2, 7), (6, 21), (15, 50), (30, 100), (60, 200)]);
    assert_eq!(ao, [(1, 7), (3, 21), (7, 50), (15, 100), (29, 200)]);
    assert_eq!(apo, [(4, 7), (11, 21), (25, 50), (50, 100), (100, 200)]);
}
