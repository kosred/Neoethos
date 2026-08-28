const HOST: &str = include_str!("../src/indicators/half_causal_estimator.rs");
const STABLE_MATH: &str = include_str!("../src/indicators/half_causal_estimator_stable_math.rs");
const CUDA: &str = include_str!("../kernels/cuda/half_causal_estimator_kernel.cu");
const STRICT_WRAPPER: &str = include_str!("../src/cuda/neoethos_f64_wrapper.rs");
const GENERIC_WRAPPER: &str = include_str!("../src/cuda/half_causal_estimator_wrapper.rs");
const HPC_TA: &str = include_str!("../../../crates/neoethos-data/src/core/hpc_ta.rs");
const CREATOR_RECEIPT: &str =
    include_str!("../audit_receipts/half_causal_estimator/script24_receipt.toml");
const CREATOR_RAW: &str = include_str!(
    "../audit_receipts/half_causal_estimator/tradingview_pine_facade_script24_raw.json"
);
const CREATOR_RECEIPT_BYTES: &[u8] =
    include_bytes!("../audit_receipts/half_causal_estimator/script24_receipt.toml");
const CREATOR_RAW_BYTES: &[u8] = include_bytes!(
    "../audit_receipts/half_causal_estimator/tradingview_pine_facade_script24_raw.json"
);

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

fn decode_json_string_field(json: &[u8], field: &str) -> Vec<u8> {
    let marker = format!("\"{field}\":\"");
    let start = json
        .windows(marker.len())
        .position(|window| window == marker.as_bytes())
        .map(|index| index + marker.len())
        .unwrap_or_else(|| panic!("missing JSON string field `{field}`"));
    let mut decoded = Vec::new();
    let mut index = start;
    while index < json.len() {
        match json[index] {
            b'"' => return decoded,
            b'\\' => {
                index += 1;
                match json[index] {
                    b'"' | b'\\' | b'/' => decoded.push(json[index]),
                    b'b' => decoded.push(8),
                    b'f' => decoded.push(12),
                    b'n' => decoded.push(b'\n'),
                    b'r' => decoded.push(b'\r'),
                    b't' => decoded.push(b'\t'),
                    other => panic!("unsupported JSON escape: {other}"),
                }
            }
            byte => decoded.push(byte),
        }
        index += 1;
    }
    panic!("unterminated JSON string field `{field}`")
}

#[test]
fn host_uses_one_stable_creator_aligned_math_authority() {
    assert!(HOST.contains("half_causal_estimator_stable_math.rs"));
    assert!(HOST.contains(
        "half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl;public-retained-budget-64mib/v1"
    ));
    assert!(!HOST.contains(
        "half-causal-estimator-f64-v2-creator-pine6-script24-chronological-welford-population-neumaier-oN-tod-state"
    ));
    assert!(HOST.contains("StablePopulationMoments"));
    assert!(HOST.contains("NeumaierSum"));
    assert!(!HOST.contains("sum_sq"));
    assert!(!HOST.contains("avg.abs() <= f64::EPSILON"));

    assert!(STABLE_MATH.contains("let next_count = self.count + 1"));
    assert!(STABLE_MATH.contains("let delta = value - self.mean"));
    assert!(STABLE_MATH.contains("let delta_after_mean = value - self.mean"));
    assert!(STABLE_MATH.contains("self.m2 += delta * delta_after_mean"));
    assert!(STABLE_MATH.contains("if mean == 0.0"));
    assert!(STABLE_MATH.contains("let scaled = value * confidence"));
    assert!(STABLE_MATH.contains("let term = scaled * coefficient"));
    assert!(STABLE_MATH.contains("self.sum + self.correction"));
}

#[test]
fn creator_receipt_freezes_the_raw_script24_oracle_without_making_it_runtime_code() {
    for required in [
        "VectorTA only",
        "PUB%3B28b6b0520c9b45c597b96d7644327a89/last",
        "original_script_version = \"24.0\"",
        "03F2046E0D55E956F77304C1CB557223C892A32656C07204746AE4140BA8A837",
        "4B7FD8AEC6B333A4ECE967D7CFA6D957357CE436CB098E96EB1EB8A1480A8080",
    ] {
        assert!(CREATOR_RECEIPT.contains(required));
    }
    for required in [
        "\"originalScriptVersion\":\"24.0\"",
        "method init_make_window",
        "method maintain_window",
        "session.isfirstbar or session.isfirstbar_regular",
        "expected_value_maintain_window",
    ] {
        assert!(CREATOR_RAW.contains(required));
    }

    assert_eq!(CREATOR_RECEIPT_BYTES.len(), 2130);
    assert_eq!(CREATOR_RECEIPT_BYTES.last(), Some(&b'\n'));
    assert_eq!(
        CREATOR_RECEIPT_BYTES
            .iter()
            .filter(|byte| **byte == b'\n')
            .count(),
        41
    );
    assert_eq!(
        sha256_hex(CREATOR_RECEIPT_BYTES),
        "18D24B85AA160B571BDE2BB6D023046C7403EE309F9C841694C51A1F8B90650F"
    );
    assert_eq!(CREATOR_RAW_BYTES.len(), 21006);
    assert_eq!(&CREATOR_RAW_BYTES[21004..], b"}\n");
    assert_eq!(
        sha256_hex(CREATOR_RAW_BYTES),
        "D371BB32D723C17997EA210E230597FFFD1AD876C7A537DA3DFCD272EC4582AD"
    );
    assert_eq!(
        sha256_hex(&CREATOR_RAW_BYTES[..21005]),
        "03F2046E0D55E956F77304C1CB557223C892A32656C07204746AE4140BA8A837"
    );
    let decoded = decode_json_string_field(&CREATOR_RAW_BYTES[..21005], "source");
    assert_eq!(decoded.len(), 19463);
    assert_eq!(
        decoded.windows(2).filter(|pair| *pair == b"\r\n").count(),
        520
    );
    assert_eq!(
        decoded
            .iter()
            .enumerate()
            .filter(|(index, byte)| {
                **byte == b'\n' && (*index == 0 || decoded[*index - 1] != b'\r')
            })
            .count(),
        0
    );
    assert_eq!(
        sha256_hex(&decoded),
        "4B7FD8AEC6B333A4ECE967D7CFA6D957357CE436CB098E96EB1EB8A1480A8080"
    );
}

#[test]
fn host_uses_cached_creator_windows_and_explicit_candle_session_slots() {
    for required in [
        "struct FutureWindowCache",
        "struct ExpectedWindowCache",
        "fn initialize(",
        "fn maintain(",
        "window_key",
        "session_start: bool",
        "PreparedSlots::Explicit",
        "session_starts",
        "half_causal_estimator_batch_prepared",
    ] {
        assert!(HOST.contains(required), "host omitted `{required}`");
    }
    assert!(!HOST.contains("fn collect_future_into("));

    let batch_builder = HOST
        .split("impl HalfCausalEstimatorBatchBuilder")
        .nth(1)
        .expect("candle batch builder exists");
    let candle_batch = batch_builder
        .split("pub fn apply_candles(")
        .nth(1)
        .expect("candle batch apply route exists");
    assert!(candle_batch.contains("&prepared.slots"));
    assert!(!candle_batch.contains("half_causal_estimator_batch_with_kernel(&prepared.values"));
}

#[test]
fn host_validation_and_series_semantics_are_fail_closed_before_allocation() {
    for required in [
        "slots_per_day > 1440",
        "1440 % slots_per_day != 0",
        "data_period > 0",
        "checked_add(1)",
        "extra_smoothing > 2",
        "checked_mul(2)",
        "try_reserve_exact",
        "checked_sweep_cardinality",
        "effective_data_period_for_frame",
        "InvalidExtraSmoothing",
        "AllocationFailed",
        "ArithmeticOverflow",
        "CandleFieldLengthMismatch",
        "validate_candle_source_lengths",
        "validate_timestamp",
    ] {
        assert!(
            HOST.contains(required),
            "host omitted validation `{required}`"
        );
    }
    let wma = HOST
        .split("impl FillWmaState")
        .nth(1)
        .expect("WMA state exists");
    let advance = wma
        .find("self.values.push_front(value)")
        .expect("WMA advances");
    let missing = wma
        .find("if !value.is_finite()")
        .expect("WMA returns a missing output for a missing bar");
    assert!(advance < missing);
    assert!(wma.contains("unwrap_or(first)"));
    assert!(HOST.contains("filter(|value| value.is_finite()).unwrap_or(close)"));

    let preparation = HOST
        .split("fn prepare_source_and_slots")
        .nth(1)
        .expect("candle preparation exists");
    let field_admission = preparation
        .find("validate_candle_source_lengths(candles, source)")
        .expect("candle field lengths are admitted");
    let timestamp_admission = preparation
        .find("validate_timestamp(timestamp)")
        .expect("every candle timestamp is admitted");
    let slot_admission = preparation
        .find("slots_per_day > 1440 || 1440 % slots_per_day != 0")
        .expect("UTC-day slots are admitted");
    let slot_allocation = preparation
        .find("try_vec_with_capacity")
        .expect("explicit slot allocation exists");
    let source_conversion = preparation
        .find("source_from_candles(candles, source, slots_per_day)")
        .expect("candle source conversion exists");
    assert!(field_admission < slot_allocation);
    assert!(timestamp_admission < slot_allocation);
    assert!(slot_admission < slot_allocation);
    assert!(slot_admission < source_conversion);

    let direct = HOST
        .split("pub fn half_causal_estimator_with_kernel")
        .nth(1)
        .expect("direct host route exists");
    assert!(
        direct
            .find("resolve_and_prepare(input)")
            .expect("direct host admission exists")
            < direct
                .find("try_alloc_f64")
                .expect("direct host output allocation exists")
    );
    assert!(
        direct
            .find("validate_frame_public_cpu_retained_budget_v1")
            .expect("direct retained-memory admission exists")
            < direct
                .find("try_alloc_f64")
                .expect("direct host output allocation exists")
    );
    let batch = HOST
        .split("fn half_causal_estimator_batch_prepared_inner")
        .nth(1)
        .expect("prepared batch core exists");
    assert!(
        batch
            .find("resolve_grid_for_frame")
            .expect("every batch combo is pre-resolved")
            < batch
                .find("try_alloc_f64")
                .expect("batch output allocation exists")
    );
    assert!(
        batch
            .find("validate_frame_public_cpu_retained_budget_v1")
            .expect("batch retained-memory admission exists")
            < batch
                .find("try_alloc_f64")
                .expect("batch output allocation exists")
    );
}

#[test]
fn host_accepts_creator_unbounded_d0_and_checks_every_host_allocation_shape() {
    assert!(!HOST.contains("data_period == 0 ||"));
    assert!(HOST.contains("data_period > 0"));
    assert!(HOST.contains("data_period: (0, 0, 0)"));
    assert!(HOST.contains("unbounded_data_period_direct_batch_stream_match_through_holes"));
    assert!(HOST.contains("finite_frame_effective_data_period_avoids_a_false_public_d_cap"));
    assert!(HOST.contains("hostile_window_and_sweep_shapes_fail_typed_before_allocation"));
    assert!(HOST.contains("checked_sweep_cardinality(sweep)?"));
    assert!(HOST.contains(".checked_mul(cols)"));
    assert!(HOST.contains("try_reserve_exact(elements)"));
    assert!(HOST.contains(
        "pub const HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1: usize = 64 * 1024 * 1024;"
    ));
    assert!(HOST.contains("PublicRetainedMemoryBudgetExceeded"));
    assert!(HOST.contains("public_cpu_retained_bytes_v1"));
    assert!(HOST.contains("validate_public_cpu_retained_budget_v1"));
    assert!(HOST.contains("huge_public_contexts_fail_typed_or_short_frame_skips_allocation"));
    assert!(HOST.contains("registry_anchor_21_is_not_base_20_at_the_creator_readiness_boundary"));
    let context_constructor = HOST
        .split("fn try_new(params: ResolvedParams)")
        .nth(1)
        .expect("HCE retained context constructor exists")
        .split("fn update(")
        .next()
        .expect("context constructor has a bounded source slice");
    assert!(
        context_constructor
            .find("validate_public_cpu_retained_budget_v1")
            .expect("retained-memory budget admission exists")
            < context_constructor
                .find("TimeOfDayStore::try_new")
                .expect("first retained Vec-backed state allocation exists")
    );
    let compute_row = HOST
        .split("fn compute_row(")
        .nth(1)
        .expect("host row core exists")
        .split("pub fn half_causal_estimator(")
        .next()
        .expect("host row core has a bounded source slice");
    assert!(
        compute_row
            .find("frame_can_become_ready")
            .expect("short finite frames skip impossible retained state")
            < compute_row
                .find("HalfCausalEstimatorContext::try_new")
                .expect("row retained context allocation exists")
    );
    let impossible_frame = compute_row
        .split("if !frame_can_become_ready")
        .nth(1)
        .expect("short-frame retained allocation guard exists")
        .split("let mut ctx")
        .next()
        .expect("short-frame guard has a bounded source slice");
    assert!(impossible_frame.contains("estimate_out.fill(f64::NAN)"));
    assert!(impossible_frame.contains("expected_value_out.fill(f64::NAN)"));
    let f64_axis = HOST
        .split("fn axis_len_f64")
        .nth(1)
        .expect("f64 sweep-axis length helper exists")
        .split("fn expand_axis_f64")
        .next()
        .expect("f64 sweep-axis helper has a bounded source slice");
    assert!(f64_axis.contains(".checked_add(1)"));
    assert!(f64_axis.contains("SweepCardinalityOverflow"));
    assert!(HOST.contains("18_446_744_073_709_551_616.0"));
}

#[test]
fn strict_cuda_is_on_stateful_on_linear_work_not_reverse_history_scans() {
    assert!(CUDA.contains("NEO_HCE_MAX_SLOTS_PER_DAY 1440"));
    assert!(CUDA.contains("NEO_HCE_MAX_DATA_PERIOD 50"));
    assert!(CUDA.contains("NEO_HCE_MAX_FILTER_LENGTH 200"));
    assert!(CUDA.contains("NEO_HCE_MAX_FUTURE_LEN 199"));
    assert!(CUDA.contains("NEO_HCE_MAX_WINDOW_SIZE 399"));
    assert!(CUDA.contains("neo_hce_resolve_registry_anchor"));
    assert!(CUDA.contains("case 21: *data_period = 5; *filter_length = 21; return true;"));
    assert!(!CUDA.contains("Anchor 21 is deliberately excluded"));
    assert!(!CUDA.contains("Anchor 21 is intentionally excluded"));
    assert!(CUDA.contains("neo_hce_welford_add"));
    assert!(CUDA.contains("neo_hce_neumaier_add"));
    assert!(CUDA.contains("tod_values_scratch"));
    assert!(CUDA.contains("tod_counts_scratch"));
    assert!(CUDA.contains("tod_next_scratch"));
    assert!(!CUDA.contains("__device__ bool neo_hce_bucket("));
    assert!(CUDA.contains("blockIdx.x != 0 || threadIdx.x != 0"));
    assert!(CUDA.contains("for (int combo = 0; combo < n_combos; ++combo)"));
    assert!(CUDA.contains("const int anchor = periods[combo]"));
    assert!(CUDA.contains("const int data_period"));
    assert!(CUDA.contains("const int filter_length"));
    assert!(!CUDA.contains("(void)periods"));
    assert!(!CUDA.contains("KERNEL_GAUSSIAN"));
    assert!(!CUDA.contains("KERNEL_EPANECHNIKOV"));
    assert!(!CUDA.contains("KERNEL_TRIANGULAR"));
    assert!(!CUDA.contains("KERNEL_SINC"));
    assert!(CUDA.contains("if (mean != 0.0)"));
    assert!(CUDA.contains("if (isfinite(volume[i]))"));
    assert!(!CUDA.contains("NEO_HCE_F64_EPSILON"));
    for required in [
        "neo_hce_initialize_future_window",
        "neo_hce_maintain_future_window",
        "future_window_key",
        "neo_hce_utc_day",
        "session_start",
        "neo_hce_validate_all_timestamps",
        "NEO_HCE_CHRONO_MIN_TIMESTAMP_MS",
        "NEO_HCE_CHRONO_MAX_TIMESTAMP_MS",
    ] {
        assert!(CUDA.contains(required), "strict CUDA omitted `{required}`");
    }
    let strict = CUDA
        .split("half_causal_estimator_neo_batch_f64(")
        .nth(1)
        .expect("strict HCE-v2 CUDA entry exists");
    assert!(!strict.contains("while (found < NEO_HCE_FUTURE_LEN)"));
    assert!(!strict.contains("slot <= prev_slot"));
    let timestamp_admission = strict
        .find("neo_hce_validate_all_timestamps(timestamps, n)")
        .expect("strict timestamps are admitted as a whole input");
    let first_finite_scratch_write = strict
        .find("tod_counts_scratch[slot] = 0")
        .expect("strict TOD scratch initialization exists");
    let first_finite_output_write = strict
        .find("estimate;")
        .expect("strict finite estimate output exists");
    assert!(timestamp_admission < first_finite_scratch_write);
    assert!(timestamp_admission < first_finite_output_write);
}

#[test]
fn strict_cuda_registry_ratio_rows_are_typed_and_independent() {
    for mapping in [
        "case 7: *data_period = 2; *filter_length = 7;",
        "case 20: *data_period = 5; *filter_length = 20;",
        "case 21: *data_period = 5; *filter_length = 21;",
        "case 50: *data_period = 13; *filter_length = 50;",
        "case 100: *data_period = 25; *filter_length = 100;",
        "case 200: *data_period = 50; *filter_length = 200;",
    ] {
        assert!(CUDA.contains(mapping), "missing strict mapping `{mapping}`");
    }
    let entry = CUDA
        .split("half_causal_estimator_neo_batch_f64(")
        .nth(1)
        .expect("strict HCE-v2 CUDA entry exists");
    let combo = entry
        .find("for (int combo = 0; combo < n_combos; ++combo)")
        .expect("serial per-combo loop exists");
    let reset = entry[combo..]
        .find("tod_counts_scratch[slot] = 0")
        .expect("TOD state resets inside each combo");
    let estimate = entry[combo..]
        .find("row[i] = estimate")
        .expect("each combo writes only its own row");
    assert!(reset < estimate);
    assert!(
        !entry
            .contains("for (int combo = 0; combo < n_combos; ++combo) {\n                    out[")
    );
}

#[test]
fn host_and_cuda_compute_before_current_finite_tod_insert() {
    let context = HOST
        .split("impl HalfCausalEstimatorContext")
        .nth(1)
        .expect("host HCE context exists");
    let host_update = context
        .split("fn update(")
        .nth(1)
        .expect("host HCE context update exists");
    let host_compute = host_update
        .find("self.compute_estimate_window()")
        .expect("host estimate uses the historical TOD state");
    let host_insert = host_update
        .find("self.store.add(slot, value)")
        .expect("host inserts the current finite value");
    assert!(host_compute < host_insert);

    let cuda_entry = CUDA
        .split("half_causal_estimator_neo_batch_f64(")
        .nth(1)
        .expect("strict HCE-v2 CUDA entry exists");
    let cuda_compute = cuda_entry
        .find("neo_hce_add_weighted(")
        .expect("CUDA computes the estimate");
    let cuda_insert = cuda_entry
        .find("neo_hce_insert_finite(")
        .expect("CUDA inserts the current finite value");
    assert!(cuda_compute < cuda_insert);
}

#[test]
fn strict_sync_and_resident_routes_retain_typed_hce_scratch() {
    assert!(STRICT_WRAPPER.contains("HceStableScratchV2"));
    assert!(STRICT_WRAPPER.contains("launch_hce_stable_v2"));
    assert!(STRICT_WRAPPER.contains("scratch_i32"));
    assert!(STRICT_WRAPPER.contains("F64Kernel::HalfCausalEstimator"));
    assert!(STRICT_WRAPPER.contains("HCE_V2_SCRATCH_F64_ELEMS"));
    assert!(STRICT_WRAPPER.contains("HCE_V2_SCRATCH_I32_ELEMS"));
    assert!(STRICT_WRAPPER.contains("HCE_V2_MAX_DATA_PERIOD: usize = 50"));
    assert!(STRICT_WRAPPER.contains("HCE_V2_MAX_ANCHOR: usize = 200"));
    assert!(STRICT_WRAPPER.contains("DeviceBuffer::from_slice(periods)"));
    assert!(STRICT_WRAPPER.contains("LockedBuffer::from_slice(periods)"));
    assert!(!STRICT_WRAPPER.contains("periods[..1]"));
    assert!(STRICT_WRAPPER.contains("validate_hce_v2_preallocation"));
    assert!(STRICT_WRAPPER.contains("resolve_hce_registry_ratio_v2"));
    for mapping in [
        "7 => Some((2, 7))",
        "20 => Some((5, 20))",
        "21 => Some((5, 21))",
        "50 => Some((13, 50))",
        "100 => Some((25, 100))",
        "200 => Some((50, 200))",
    ] {
        assert!(STRICT_WRAPPER.contains(mapping));
    }
    assert!(STRICT_WRAPPER.contains("expected one of 7,20,21,50,100,200"));
    assert!(!STRICT_WRAPPER.contains("21 is deliberately excluded"));
    let exclusions = HPC_TA
        .split("pub const SWEEP_POINT_EXCLUSIONS")
        .nth(1)
        .expect("production sweep exclusions exist")
        .split("pub fn sweep_point_exclusion")
        .next()
        .expect("sweep exclusions have a bounded source slice");
    assert!(!exclusions.contains("half_causal_estimator"));
    assert!(HPC_TA.contains("(\"half_causal_estimator\", 21)"));
    assert!(HPC_TA.contains("assert_eq!(hce_21_value(\"data_period\"), 5)"));
    assert!(HPC_TA.contains("assert_eq!(hce_21_value(\"filter_length\"), 21)"));
    let sync = STRICT_WRAPPER
        .split("pub fn sweep(")
        .nth(1)
        .expect("sync strict sweep exists");
    let sync_admission = sync
        .find("validate_hce_v2_preallocation")
        .expect("sync HCE preallocation admission exists");
    let sync_allocation = sync
        .find("DeviceBuffer::<f64>::uninitialized(output_elems)")
        .expect("sync output allocation exists");
    assert!(sync_admission < sync_allocation);
    let resident = STRICT_WRAPPER
        .split("pub fn sweep_resident_v3(")
        .nth(1)
        .expect("resident strict sweep exists");
    let resident_admission = resident
        .find("validate_hce_v2_preallocation")
        .expect("resident HCE preallocation admission exists");
    let resident_allocation = resident
        .find("DeviceBuffer::<f64>::uninitialized_async(output_elements")
        .expect("resident output allocation exists");
    assert!(resident_admission < resident_allocation);
    for required in [
        "timestamps.len() != cols",
        "price.len() != cols",
        "volume.len() != cols",
        "timestamps.device_id() != self.device_id()",
        "price.device_id() != self.device_id()",
        "volume.device_id() != self.device_id()",
        "i32::try_from(cols)",
        "i32::try_from(rows)",
        "i32::try_from(first_valid)",
    ] {
        assert!(STRICT_WRAPPER.contains(required));
    }
    let invariant = STRICT_WRAPPER
        .split("pub fn is_period_invariant(self) -> bool")
        .nth(1)
        .expect("period-invariant classifier exists")
        .split("pub fn indicator_id")
        .next()
        .unwrap();
    assert!(!invariant.contains("F64Kernel::HalfCausalEstimator"));
    let generic_chunk = STRICT_WRAPPER
        .split("fn launch_chunk(")
        .nth(1)
        .expect("generic strict chunk launcher exists");
    assert!(generic_chunk.contains("HCE-v2 must use launch_hce_stable_v2"));
}

#[test]
fn generic_cuda_cannot_silently_claim_a_second_hce_v2_result() {
    let aligned = GENERIC_WRAPPER.contains("HceGenericStableScratchV2")
        && CUDA.contains("half_causal_estimator_batch_f64_v2");
    let unavailable = GENERIC_WRAPPER.contains("StableAuthorityV2Unavailable")
        && GENERIC_WRAPPER.contains("half-causal-estimator-f64-v2");
    assert!(aligned || unavailable);

    if unavailable {
        let constructor = GENERIC_WRAPPER
            .split("pub fn new(device_id: usize)")
            .nth(1)
            .expect("generic HCE constructor exists");
        let constructor_refusal = constructor
            .find("StableAuthorityV2Unavailable")
            .expect("generic constructor refuses before CUDA initialization");
        let cuda_initialization = constructor
            .find("cust::init")
            .expect("legacy CUDA initialization remains behind the refusal");
        assert!(constructor_refusal < cuda_initialization);

        let batch = GENERIC_WRAPPER
            .split("pub fn batch_dev(")
            .nth(1)
            .expect("generic HCE batch route exists");
        let refusal = batch
            .find("StableAuthorityV2Unavailable")
            .expect("generic route refuses the superseded authority");
        let allocation = batch
            .find("DeviceBuffer::<f64>")
            .expect("legacy allocation remains behind the refusal");
        assert!(refusal < allocation);
    }
}

#[test]
fn hce_v2_remains_f64_only() {
    let joined = [HOST, STABLE_MATH, CUDA, STRICT_WRAPPER, GENERIC_WRAPPER].concat();
    assert!(!joined.contains("half_causal_estimator_batch_f32"));
    assert!(!joined.contains("half_causal_estimator_neo_batch_f32"));
}
