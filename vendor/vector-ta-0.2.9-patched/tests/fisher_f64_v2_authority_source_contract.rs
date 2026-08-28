use std::fs;
use std::path::{Path, PathBuf};

const FISHER_F32_PREFIX_CANONICAL_LF_FNV1A64: u64 = 0xcc7b_ed74_87bf_dbe0;
const OPENLIBM_COMMIT: &str = "82e90aef0657289192efe77be89791c07dea0775";
const OPENLIBM_E_LOG_URL: &str = "https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/src/e_log.c";
const OPENLIBM_LICENSE_URL: &str = "https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/LICENSE.md";
const OPENLIBM_E_LOG_SHA256: &str =
    "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD";

fn manifest_dir() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current directory must be readable"))
}

fn read(relative: impl AsRef<Path>) -> String {
    fs::read_to_string(manifest_dir().join(relative))
        .expect("the reviewed Fisher source must be readable")
}

fn read_bytes(relative: impl AsRef<Path>) -> Vec<u8> {
    fs::read(manifest_dir().join(relative)).expect("the reviewed Fisher bytes must be readable")
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let tail = &source[from..];
    let to = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing section end after {start}: {end}"));
    &tail[..to]
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes(block[offset..offset + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    state.iter().map(|word| format!("{word:08X}")).collect()
}

fn assert_ordered(source: &str, tokens: &[&str]) {
    let mut cursor = 0usize;
    for token in tokens {
        let found = source[cursor..]
            .find(token)
            .unwrap_or_else(|| panic!("missing ordered token after byte {cursor}: {token}"));
        cursor += found + token.len();
    }
}

#[test]
fn host_f64_v2_is_one_bounded_faithful_segmented_authority() {
    let source = read("src/indicators/fisher.rs");
    let production = source
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("Fisher production section must exist");

    for token in [
        "const FISHER_QNAN_BITS_F64_V2: u64 = 0x7ff8_0000_0000_0000;",
        "const FISHER_CUDA_F64_MAX_PERIOD_V2: usize = 1024;",
        "fn fisher_log_f64_v2(",
        "fn fisher_transition_f64_v2(",
        "fn fisher_f64_into_v2(",
        "fn fisher_first_finite_midpoint_v2(",
        "fn fisher_admit_shape_v2(",
        "fn fisher_admit_period_v2(",
        "fn fisher_admit_finite_tail_v2(",
        "fn fisher_admit_host_v2(",
        "fn fisher_grid_shape_v2(",
        "fn expand_grid_checked_v2(",
        "fn fisher_raw_into_is_admitted_v2(",
        "fn fisher_stream_period_is_admitted_v2(",
        "fn reset_finite_segment_v2(",
        "0.67f64.mul_add(",
        "0.5f64.mul_add(",
        "const FISHER_RANGE_FLOOR_F64_V2: f64 = 0.001;",
    ] {
        assert!(production.contains(token), "missing host v2 token: {token}");
    }

    assert!(
        !production.contains(".ln()"),
        "native platform log must not remain in the host f64 production route"
    );
    assert!(
        !production.contains("0.67 * val1 +"),
        "plain recurrence must not remain behind an AVX/batch label"
    );

    let scalar = section(
        production,
        "pub fn fisher_scalar_into(",
        "pub fn fisher_avx512_into(",
    );
    assert!(scalar.contains("fisher_f64_into_v2("));
    let labelled_simd = section(
        production,
        "pub fn fisher_avx512_into(",
        "pub fn fisher_into_slice(",
    );
    assert_eq!(labelled_simd.matches("fisher_scalar_into(").count(), 2);

    let batch = section(
        production,
        "fn fisher_batch_inner(",
        "pub struct FisherStream",
    );
    assert!(batch.contains("fisher_f64_into_v2("));
    assert!(!batch.contains("fisher_row_scalar_direct("));
    assert!(!batch.contains("fisher_row_scalar_from_hl("));
    let private_into = section(
        production,
        "fn fisher_batch_inner_into(",
        "pub struct FisherStream",
    );
    assert_ordered(
        private_into,
        &[
            "fisher_admit_shape_v2(high, low)?",
            "fisher_grid_shape_v2(sweep, data_len)?",
            "fisher_admit_period_v2(grid.max_period, data_len)?",
            "fisher_first_finite_midpoint_v2(high, low)",
            "fisher_admit_finite_tail_v2(data_len, first, grid.max_period)?",
            "let expected = rows.checked_mul(cols)",
            "if fisher_out.len() != expected || signal_out.len() != expected",
            "expand_grid_checked_v2(sweep, grid)?",
        ],
    );

    let stream = &production[production
        .find("impl FisherStream")
        .expect("FisherStream implementation must exist")..];
    assert!(stream.contains("reset_finite_segment_v2("));
    assert!(stream.contains("fisher_transition_f64_v2("));
    assert!(stream.contains("if !high.is_finite() || !low.is_finite()"));
}

#[test]
fn public_row_labels_and_every_host_entry_use_prework_admission() {
    let source = read("src/indicators/fisher.rs");
    let production = source
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("Fisher production section must exist");

    for signature in [
        "pub fn fisher_row_avx2_direct(\n    high: &[f64],\n    low: &[f64],\n    first: usize,\n    period: usize,\n    out_fish: &mut [f64],\n    out_signal: &mut [f64],\n)",
        "pub fn fisher_row_avx512_direct(\n    high: &[f64],\n    low: &[f64],\n    first: usize,\n    period: usize,\n    out_fish: &mut [f64],\n    out_signal: &mut [f64],\n)",
    ] {
        assert!(
            production.contains(signature),
            "missing exact public signature"
        );
    }
    let avx2 = section(
        production,
        "pub fn fisher_row_avx2_direct(",
        "pub fn fisher_row_avx512_direct(",
    );
    let avx512 = section(
        production,
        "pub fn fisher_row_avx512_direct(",
        "pub struct FisherStream",
    );
    assert!(avx2.contains("fisher_f64_into_v2(high, low, period, first,"));
    assert!(avx512.contains("fisher_f64_into_v2(high, low, period, first,"));

    let admission = section(
        production,
        "fn fisher_admit_host_v2(",
        "fn fisher_raw_into_is_admitted_v2(",
    );
    assert_ordered(
        admission,
        &[
            "fisher_admit_shape_v2(high, low)?",
            "fisher_admit_period_v2(period, data_len)?",
            "fisher_first_finite_midpoint_v2(high, low)",
            "fisher_admit_finite_tail_v2(data_len, first, period)?",
        ],
    );

    let raw = section(
        production,
        "fn fisher_f64_into_v2(",
        "impl<'a> FisherInput<'a>",
    );
    assert_ordered(
        raw,
        &[
            "if !fisher_raw_into_is_admitted_v2(",
            "return;",
            "fisher_out.fill(fisher_qnan_f64_v2())",
            "VecDeque::with_capacity(period + 1)",
        ],
    );

    let batch = section(
        production,
        "fn fisher_batch_inner(",
        "pub struct FisherStream",
    );
    assert!(!batch.contains("high.len().min(low.len())"));
    assert_ordered(
        batch,
        &[
            "fisher_admit_shape_v2(high, low)?",
            "fisher_grid_shape_v2(sweep, data_len)?",
            "fisher_admit_period_v2(grid.max_period, data_len)?",
            "fisher_first_finite_midpoint_v2(high, low)",
            "fisher_admit_finite_tail_v2(data_len, first, grid.max_period)?",
            "expand_grid_checked_v2(sweep, grid)?",
            "make_uninit_matrix(rows, cols)",
        ],
    );
    assert!(batch.contains("checked_mul(cols)"));
    assert!(production.contains("checked_add(1)"));
    assert!(production.contains("checked_add(step)"));
    assert!(production.contains("checked_sub(step)"));
    assert!(production.contains("count > max_count"));

    let stream_ctor = section(
        production,
        "pub fn try_new(params: FisherParams)",
        "pub fn update(",
    );
    assert!(stream_ctor.contains("fisher_stream_period_is_admitted_v2(period)"));
    assert!(!stream_ctor.contains("VecDeque::with_capacity(period"));
}

#[test]
fn strict_cuda_f64_v2_preserves_f32_and_uses_openlibm_rn_deques() {
    let bytes = read_bytes("kernels/cuda/oscillators/fisher_kernel.cu");
    assert!(
        !bytes.contains(&b'\r'),
        "the dedicated CUDA source must be canonical LF, never mixed-EOL"
    );
    let marker = b"// NeoEthos f64 lane";
    let marker_at = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("f64 marker must exist");
    assert_eq!(
        fnv1a64(&bytes[..marker_at]),
        FISHER_F32_PREFIX_CANONICAL_LF_FNV1A64,
        "the canonical-LF pre-existing f32 prefix changed"
    );

    let source = String::from_utf8(bytes).expect("CUDA source must be UTF-8");
    let f64_source = &source[marker_at..];
    for token in [
        "#define NEO_FISHER_F64_MAX_PERIOD 1024",
        "struct NeoFisherDequeF64V2",
        "fisher_log_f64_v2(",
        "fisher_transition_f64_v2(",
        "fisher_row_deque_f64_v2(",
        "extern __shared__ int fisher_deque_storage[];",
        "const int combo = (int)blockIdx.x;",
        "if (threadIdx.x != 0) return;",
        "if (period <= 0 || period > NEO_FISHER_F64_MAX_PERIOD) return;",
        "__dadd_rn(",
        "__dsub_rn(",
        "__dmul_rn(",
        "__ddiv_rn(",
        "__fma_rn(",
    ] {
        assert!(f64_source.contains(token), "missing CUDA v2 token: {token}");
    }
    assert!(
        !f64_source.contains(" log("),
        "libdevice log must not remain in the strict f64 suffix"
    );
    assert!(
        !f64_source.contains("fisher_row_f64("),
        "no exported ABI may retain the superseded direct body"
    );
    assert!(
        !f64_source.contains("fisher_row_direct_f64_v2("),
        "strict f64 must reject periods above 1024, never enter a quadratic fallback"
    );
    assert!(!f64_source.contains("O(N*period)"));
    assert!(!f64_source.contains("min(max_period,1024)"));
    assert_eq!(f64_source.matches("fisher_row_f64_v2(").count(), 5);
    assert_eq!(
        f64_source
            .matches("extern __shared__ int fisher_deque_storage[];")
            .count(),
        4
    );
    assert_eq!(
        f64_source
            .matches("const int combo = (int)blockIdx.x;")
            .count(),
        4
    );

    for symbol in [
        "void neoethos_fisher_f64(",
        "void neoethos_fisher_signal_f64(",
        "void fisher_outputs_f64(",
        "void neoethos_fisher_batch_f64(",
    ] {
        assert_eq!(
            f64_source.matches(symbol).count(),
            1,
            "preserved ABI symbol must appear exactly once: {symbol}"
        );
    }
}

#[test]
fn immutable_openlibm_e_log_receipt_pins_source_license_constants_and_order() {
    let relative = format!("tests/fixtures/openlibm/e_log-{OPENLIBM_COMMIT}.c");
    let bytes = read_bytes(relative);
    assert_eq!(sha256_hex(&bytes), OPENLIBM_E_LOG_SHA256);
    let oracle = String::from_utf8(bytes).expect("OpenLibm e_log.c must be UTF-8");
    for license_token in [
        "Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.",
        "Permission to use, copy, modify, and distribute this",
        "software is freely granted, provided that this notice",
        "is preserved.",
    ] {
        assert!(oracle.contains(license_token));
    }

    let receipt = read(format!(
        "tests/fixtures/openlibm/e_log-{OPENLIBM_COMMIT}.receipt.txt"
    ));
    for token in [
        OPENLIBM_COMMIT,
        OPENLIBM_E_LOG_URL,
        OPENLIBM_LICENSE_URL,
        OPENLIBM_E_LOG_SHA256,
        "Sun fdlibm notice embedded in e_log.c",
    ] {
        assert!(
            receipt.contains(token),
            "missing immutable receipt token: {token}"
        );
    }

    let host = read("src/indicators/fisher.rs");
    let cuda = read("kernels/cuda/oscillators/fisher_kernel.cu");
    for source in [&host, &cuda] {
        for token in [
            OPENLIBM_COMMIT,
            OPENLIBM_E_LOG_URL,
            OPENLIBM_LICENSE_URL,
            OPENLIBM_E_LOG_SHA256,
        ] {
            assert!(
                source.contains(token),
                "authority source lost receipt: {token}"
            );
        }
    }

    let constants = [
        (1.801_439_850_948_198_4e16f64, 0x4350_0000_0000_0000u64),
        (6.931_471_803_691_238e-1, 0x3fe6_2e42_fee0_0000),
        (1.908_214_929_270_587_7e-10, 0x3dea_39ef_3579_3c76),
        (6.666_666_666_666_735e-1, 0x3fe5_5555_5555_5593),
        (3.999_999_999_940_942e-1, 0x3fd9_9999_9997_fa04),
        (2.857_142_874_366_239e-1, 0x3fd2_4924_9422_9359),
        (2.222_219_843_214_978_4e-1, 0x3fcc_71c5_1d8e_78af),
        (1.818_357_216_161_805e-1, 0x3fc7_4664_96cb_03de),
        (1.531_383_769_920_937_3e-1, 0x3fc3_9a09_d078_c69f),
        (1.479_819_860_511_658_6e-1, 0x3fc2_f112_df3e_5244),
    ];
    for (value, bits) in constants {
        assert_eq!(value.to_bits(), bits);
    }
    for token in [
        "const FISHER_LOG_TWO54_F64_V2: f64 = 1.801_439_850_948_198_400_00e16;",
        "const FISHER_LOG_LN2_HI_F64_V2: f64 = 6.931_471_803_691_238_164_90e-1;",
        "const FISHER_LOG_LN2_LO_F64_V2: f64 = 1.908_214_929_270_587_700_02e-10;",
        "const FISHER_LOG_LG1_F64_V2: f64 = 6.666_666_666_666_735_130e-1;",
        "const FISHER_LOG_LG2_F64_V2: f64 = 3.999_999_999_940_941_908e-1;",
        "const FISHER_LOG_LG3_F64_V2: f64 = 2.857_142_874_366_239_149e-1;",
        "const FISHER_LOG_LG4_F64_V2: f64 = 2.222_219_843_214_978_396e-1;",
        "const FISHER_LOG_LG5_F64_V2: f64 = 1.818_357_216_161_805_012e-1;",
        "const FISHER_LOG_LG6_F64_V2: f64 = 1.531_383_769_920_937_332e-1;",
        "const FISHER_LOG_LG7_F64_V2: f64 = 1.479_819_860_511_658_591e-1;",
    ] {
        assert!(
            host.contains(token),
            "host constant/order receipt drifted: {token}"
        );
    }
    for token in [
        "const double TWO54 = 1.80143985094819840000e+16;",
        "const double LN2_HI = 6.93147180369123816490e-01;",
        "const double LN2_LO = 1.90821492927058770002e-10;",
        "const double LG1 = 6.666666666666735130e-01;",
        "const double LG2 = 3.999999999940941908e-01;",
        "const double LG3 = 2.857142874366239149e-01;",
        "const double LG4 = 2.222219843214978396e-01;",
        "const double LG5 = 1.818357216161805012e-01;",
        "const double LG6 = 1.531383769920937332e-01;",
        "const double LG7 = 1.479819860511658591e-01;",
    ] {
        assert!(
            cuda.contains(token),
            "CUDA constant/order receipt drifted: {token}"
        );
    }

    let host_log = section(&host, "fn fisher_log_f64_v2(", "fn fisher_midpoint_f64_v2(");
    assert_ordered(
        host_log,
        &[
            "if high < 0x0010_0000",
            "value *= FISHER_LOG_TWO54_F64_V2",
            "exponent += (high >> 20) - 1023",
            "let normalize = (high + 0x0009_5f64) & 0x0010_0000",
            "let fraction = value - 1.0",
            "if (0x000f_ffff & (2 + high)) < 3",
            "let scaled = fraction / (2.0 + fraction)",
            "let square = scaled * scaled",
            "let fourth = square * square",
            "let remainder = odd + even",
            "let result = if selector > 0",
        ],
    );
    let cuda_log = section(
        &cuda,
        "fisher_log_f64_v2(double value, double* output)",
        "fisher_midpoint_f64_v2(",
    );
    assert_ordered(
        cuda_log,
        &[
            "if (high < 0x00100000)",
            "value = fisher_mul_rn_f64_v2(value, TWO54)",
            "exponent += (high >> 20) - 1023",
            "const int normalize = (high + 0x00095f64) & 0x00100000",
            "const double fraction = fisher_sub_rn_f64_v2(value, 1.0)",
            "if ((0x000fffff & (2 + high)) < 3)",
            "const double scaled = fisher_div_rn_f64_v2(",
            "const double square = fisher_mul_rn_f64_v2(scaled, scaled)",
            "const double fourth = fisher_mul_rn_f64_v2(square, square)",
            "const double remainder = fisher_add_rn_f64_v2(odd, even)",
            "if (selector > 0)",
        ],
    );
}

#[test]
fn standalone_contract_registration_preserves_both_hce_targets() {
    let manifest = read("Cargo.toml");
    for block in [
        "[[test]]\nname = \"half_causal_estimator_f64_v2_source_contract\"\npath = \"tests/half_causal_estimator_f64_v2_source_contract.rs\"",
        "[[test]]\nname = \"half_causal_estimator_f64_v2_direct_oracle\"\npath = \"tests/half_causal_estimator_f64_v2_direct_oracle.rs\"",
        "[[test]]\nname = \"fisher_f64_v2_authority_source_contract\"\npath = \"tests/fisher_f64_v2_authority_source_contract.rs\"",
    ] {
        assert_eq!(manifest.matches(block).count(), 1, "manifest block drifted");
    }
}

#[test]
fn bounded_faithful_claim_names_fixture_and_cancellation_limits() {
    let host = read("src/indicators/fisher.rs");
    let cuda = read("kernels/cuda/oscillators/fisher_kernel.cu");
    for source in [&host, &cuda] {
        assert!(source.contains("FISHER_F64_V2_FIXTURE_MAX_ULP=2"));
        assert!(source.contains("FISHER_F64_V2_FIXTURE_MAX_ABS=8.881784197001252e-16"));
        assert!(source.contains("FISHER_F64_V2_ADVERSARIAL_MAX_ABS=1.7763568394002505e-15"));
        assert!(source.contains("not a universal RN or ULP guarantee"));
        assert!(source.contains("24,195 primary cells"));
        assert!(source.contains("28 above one ULP"));
    }
    assert!(host.contains("receipt measured 2.32x"));
    assert!(cuda.contains("RTX 1M-row x 250-period receipt"));
    assert!(cuda.contains("report zero local-array spill"));
}

#[test]
fn strict_shared_wrapper_closes_fisher_bound_and_deque_launches_before_work() {
    let wrapper = read("src/cuda/neoethos_f64_wrapper.rs");
    assert!(wrapper.contains("pub const FISHER_F64_MAX_PERIOD: usize = 1024;"));
    assert!(wrapper.contains("F64Kernel::Fisher => Some(FISHER_F64_MAX_PERIOD),"));

    let helper = section(
        &wrapper,
        "fn fisher_shared_bytes_for_max_period_v2(",
        "fn fisher_shared_bytes_for_periods_v2(",
    );
    assert_ordered(
        helper,
        &[
            "if max_period == 0",
            "if max_period > FISHER_F64_MAX_PERIOD",
            "CudaF64IndicatorError::PeriodTooLarge",
            ".checked_add(1)",
            ".checked_mul(2)",
            ".checked_mul(std::mem::size_of::<i32>())",
            "u32::try_from(bytes)",
        ],
    );

    let all_outputs = section(
        &wrapper,
        "pub fn fisher_all_outputs(",
        "/// Launch the canonical FBEO",
    );
    assert_ordered(
        all_outputs,
        &[
            "if period == 0",
            "if period > FISHER_F64_MAX_PERIOD",
            "if period > cols",
            "let fisher_shared_bytes =",
            "let mut period_values = Vec::with_capacity(rows);",
            "let d_periods = DeviceBuffer::from_slice(&period_values)?;",
            "let module = self.module_for(F64Kernel::Fisher)?;",
        ],
    );
    for token in [
        "let grid = GridSize::x(rows_u32);",
        "let block = BlockSize::x(32);",
        "launch!(function<<<grid, block, fisher_shared_bytes, stream>>>",
    ] {
        assert!(
            all_outputs.contains(token),
            "full-pair launch drifted: {token}"
        );
    }

    let launch_chunk = section(&wrapper, "fn launch_chunk(", "#[cfg(test)]");
    assert_ordered(
        launch_chunk,
        &[
            "fisher_shared_bytes: u32,",
            "if kernel == F64Kernel::Fisher",
            "F64Inputs::HighLow { high, low }",
            "let grid = GridSize::x(rows_u32);",
            "let block = BlockSize::x(32);",
            "launch!(func<<<grid, block, fisher_shared_bytes, stream>>>",
            "return Ok(None);",
            "if kernel.is_sequential() && kernel != F64Kernel::Cci",
        ],
    );
}

#[test]
fn shared_dispatch_and_data_use_finite_midpoint_authority() {
    let dispatch = read("src/indicators/dispatch/cuda_f64.rs");
    let data = read("../../crates/neoethos-data/src/core/gpu_indicators.rs");
    let resident = read("../../crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");

    assert!(dispatch.contains("HighLowMidpointFinite,"));
    let fisher_row = section(
        &dispatch,
        "indicator_id: \"fisher\"",
        "indicator_id: \"safezonestop\"",
    );
    assert!(fisher_row.contains("input: F64InputKind::HighLow"));
    assert!(fisher_row.contains("first_valid: F64FirstValidRule::HighLowMidpointFinite"));
    assert!(dispatch.contains("(\"fisher\", F64FirstValidRule::HighLowMidpointFinite),"));
    assert!(dispatch.contains("fn fisher_v2_declares_finite_midpoint_admission()"));

    assert!(
        data.contains("F64FirstValidRule::HighLowMidpointFinite => self.first_valid_hl2_finite,")
    );
    let fisher_resident = section(
        &data,
        "pub fn compute_fisher_outputs_device(",
        "/// Launch FBEO's canonical",
    );
    assert!(
        fisher_resident
            .contains(".fisher_all_outputs(high_low, self.first_valid_hl2_finite, periods)")
    );
    assert!(data.contains("fn fisher_v2_finite_midpoint_admission_source_is_closed()"));

    let resident_mapping = section(
        &resident,
        "fn primary_first_valid(",
        "fn primary_descriptor(",
    );
    assert!(resident_mapping.contains("F64FirstValidRule::HighLowMidpointFinite"));
    assert!(resident_mapping.contains("ResidentClassicTaFirstValidRuleV3::AllInputsFinite"));
}
