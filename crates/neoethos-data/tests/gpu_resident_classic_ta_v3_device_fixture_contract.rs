use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-data must live under the workspace root")
        .to_path_buf()
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(workspace_root().join(path)).unwrap_or_default()
}

#[test]
fn classic_ta_device_fixture_is_test_only_and_keeps_runtime_authority_opaque() {
    let gpu_manifest = read("crates/neoethos-gpu-cuda/Cargo.toml");
    let data_manifest = read("crates/neoethos-data/Cargo.toml");
    let hpc = read("crates/neoethos-data/src/core/hpc_ta.rs");
    let classic = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3_device_fixture.rs");

    assert!(gpu_manifest.contains("cuda-device-fixtures = [\"cuda\"]"));
    assert!(data_manifest.contains("gpu-cuda-device-fixtures"));
    assert!(hpc.contains("mod gpu_resident_classic_ta_v3_device_tests;"));
    assert!(classic.contains("#[cfg(feature = \"cuda-device-fixtures\")]"));
    assert!(classic.contains("#[path = \"resident_classic_ta_v3_device_fixture.rs\"]"));
    assert!(classic.contains("mod resident_classic_ta_v3_device_fixture;"));
    assert!(fixture.contains("run_resident_classic_ta_v3_device_fixture"));
    for forbidden in [
        "pub fn primary_context",
        "pub fn run_stream",
        "pub fn raw_pointer",
        "Context::new",
        "Stream::new",
        "upload_ohlcv_f64",
        "download_primary_output_f64",
        "download_named_outputs_f64",
        "compute_cpu_batch",
        "fallback",
    ] {
        assert!(
            !fixture.contains(forbidden),
            "test seam exposed or reintroduced forbidden `{forbidden}`"
        );
    }
}

#[test]
fn fixture_uses_an_explicit_reviewed_subset_and_keeps_every_gap_visible() {
    let data_fixture =
        read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3_device_tests.rs");
    let hpc = read("crates/neoethos-data/src/core/hpc_ta.rs");

    for token in [
        "reviewed routeable ALL-order subset through Halftrend",
        "halftrend",
        "KNOWN_UNROUTEABLE_BASE_FAMILIES_THROUGH_HALFTREND_V3",
        "REVIEWED_HISTORICAL_IDS_THROUGH_HALFTREND_V3",
        "full_through_halftrend_graph_retains_exact_45_fail_closed_contracts",
        "Gate139",
        "before the first CUDA context/launch",
        "expected_full_graph_gap_columns_v3",
        "gaps.len() == 45",
        "classic_indicator_id_for_column",
        "excluded_historical_ids",
        "prepare_classic_ta_device_fixture_plan_v3",
        "reviewed_routeable_subset_fixture_v3",
        "compute_classic_ta_device_fixture_cpu_oracle_v3",
        "preflight_resident_classic_ta_v3",
        "run_resident_classic_ta_v3_device_fixture",
        "map_err(anyhow::Error::from_boxed)",
        "FeatureCellValidity::code",
        "to_bits",
    ] {
        assert!(
            data_fixture.contains(token) || hpc.contains(token),
            "reviewed-subset fixture omitted `{token}`"
        );
    }
    for excluded_family in [
        "dec_osc",
        "decycler",
        "demand_index",
        "donchian_channel_width",
        "dti",
        "dynamic_momentum_index",
        "ehlers_adaptive_cg",
        "ehlers_adaptive_cyber_cycle",
        "ehlers_detrending_filter",
        "ehlers_pma",
        "ehlers_simple_cycle_indicator",
        "fractal_dimension_index",
        "fvg_positioning_average",
        "gmma_oscillator",
        "goertzel_cycle_composite_wave",
    ] {
        assert!(
            data_fixture.contains(excluded_family),
            "reviewed-subset fixture lost explicit debt family `{excluded_family}`"
        );
    }
    assert!(data_fixture.contains("&reviewed_historical_ids"));
    assert!(data_fixture.contains("preflight_exact_classic_cuda_plan(&plan)"));
    assert!(!data_fixture.contains("currently routeable ALL-order prefix ending at Halftrend"));
    assert!(!data_fixture.contains("currently_routeable_prefix_fixture_v3"));
    assert!(!data_fixture.contains("global ALL"));
    assert!(!hpc.contains("fallback_to_cpu"));
}

#[test]
fn actual_executor_parity_and_boundary_cases_share_one_carrier_and_parent_upload() {
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3_device_fixture.rs");

    for token in [
        "acquire_discovery_run_device_admission_v1",
        "seal_test_full_discovery_run_device_v3",
        "ResidentClassicTaExecutorV3::new",
        "next_pending_batch_v3",
        "producer_ready_event",
        "enqueue_nonblocking_release",
        "physical_inventory_probe_count",
        "primary_context_acquisition_count",
        "run_stream_creation_count",
        "parent_upload_count",
        "natural_launch_count",
        "expected_value_bits",
        "expected_validity_codes",
        "first_value_mismatch_row",
        "expected_bits={:#018x}",
        "observed_bits={:#018x}",
        "expected_validity={}",
        "observed_validity={}",
        "[1_usize, 31, 32, 33, 63, 64]",
        "ComputeFailure",
        "Warmup",
        "0xff",
        "changed_final_feature_bit_observed",
    ] {
        assert!(fixture.contains(token), "device fixture omitted `{token}`");
    }
    assert!(!fixture.contains("Context::new"));
    assert!(!fixture.contains("Stream::new"));
}

#[test]
fn natural_fixture_reports_every_feature_mismatch_from_one_device_traversal() {
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3_device_fixture.rs");
    let comparison = fixture
        .split_once("fn read_and_compare_natural_batch(")
        .and_then(|(_, tail)| tail.split_once("fn fixture_hash("))
        .map(|(body, _)| body)
        .expect("natural-batch comparison must remain source-visible");

    for token in [
        "struct FixtureParityMismatchV3",
        "mismatches: &mut Vec<FixtureParityMismatchV3>",
        "value_mismatch_count",
        "validity_mismatch_count",
        "first_value_mismatch_row",
        "first_validity_mismatch_row",
        "Classic TA parity mismatch census:",
        "natural_launch_count == request.recipe.launches().len()",
    ] {
        assert!(
            fixture.contains(token),
            "one-run mismatch census omitted `{token}`"
        );
    }
    assert!(
        !comparison
            .contains("return Err(fixture_error(format!(\n                \"f64 bit mismatch"),
        "natural comparison still aborts at the first feature mismatch"
    );
}

#[test]
fn bounded_test_readback_and_exact_sanitizer_filter_are_pinned() {
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3_device_fixture.rs");
    let data_fixture =
        read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3_device_tests.rs");

    for token in [
        "bounded_test_parity_d2h_bytes",
        "value_d2h_bytes",
        "validity_d2h_bytes",
        "control_plane_d2h_bytes",
        "second_context_count: 0",
        "second_stream_count: 0",
        "parent_reupload_count: 0",
        "reviewed_routeable_output_count",
        "resident_classic_ta_v3_reviewed_routeable_subset_through_halftrend_is_exact_and_leak_free",
    ] {
        assert!(
            fixture.contains(token) || data_fixture.contains(token),
            "bounded device evidence omitted `{token}`"
        );
    }
}

#[test]
fn adaptive_momentum_all_output_kernel_matches_cpu_reciprocal_then_multiply_order() {
    let cpu = read("vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/linreg.rs");
    let cuda =
        read("vendor/vector-ta-0.2.9-patched/kernels/cuda/adaptive_momentum_oscillator_kernel.cu");
    let helper = cuda
        .split_once("static __device__ inline double amo_linreg_from_ring(")
        .and_then(|(_, tail)| tail.split_once("extern \"C\" __global__ void"))
        .map(|(body, _)| body)
        .expect("Adaptive Momentum all-output CUDA helper must remain source-visible");

    assert!(
        cpu.contains("let bd = 1.0 / (pf * self.x2_sum - self.x_sum * self.x_sum);")
            && cpu.contains("let b = (pf * xy_sum - self.x_sum * y_sum) * bd;"),
        "CPU LinRegStream reciprocal-then-multiply authority drifted"
    );
    for token in [
        "double bd = 1.0 / denom;",
        "double b = (period_f * xy_sum - x_sum * y_sum) * bd;",
        "Gate157",
        "0xbec5c12413ce0b68",
        "0xbec5c12413ce0b60",
    ] {
        assert!(
            helper.contains(token),
            "Adaptive Momentum all-output CUDA helper omitted exact token `{token}`"
        );
    }
    assert!(
        !helper.contains("(period_f * xy_sum - x_sum * y_sum) / denom"),
        "Adaptive Momentum reintroduced direct slope division instead of the CPU operation order"
    );
}

#[test]
fn adaptive_schaff_resident_recipe_preserves_canonical_hlc_input_authority() {
    let registry = read("vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let seed = registry
        .split_once("id: \"adaptive_schaff_trend_cycle\"")
        .and_then(|(_, tail)| tail.split_once("},\n"))
        .map(|(entry, _)| entry)
        .expect("ASTC must be a canonical registry entry");
    assert!(
        seed.contains("input_kind: IndicatorInputKind::Ohlc"),
        "ASTC registry must retain the canonical OHLC input authority"
    );

    let cpu = read("vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let cpu_dispatch = cpu
        .split_once("fn compute_adaptive_schaff_trend_cycle_batch(")
        .and_then(|(_, tail)| tail.split_once("fn compute_ehlers_detrending_filter_batch("))
        .map(|(body, _)| body)
        .expect("ASTC CPU batch dispatcher must remain source-visible");
    assert!(
        cpu_dispatch.contains("extract_ohlc_input(\"adaptive_schaff_trend_cycle\", req.data)"),
        "ASTC CPU dispatch must extract high, low and close from canonical OHLC input"
    );

    let cuda = read("vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let cuda_spec = cuda
        .split_once("indicator_id: \"adaptive_schaff_trend_cycle\"")
        .and_then(|(_, tail)| tail.split_once("},"))
        .map(|(entry, _)| entry)
        .expect("ASTC must have an exact f64 CUDA route spec");
    assert!(
        cuda_spec.contains("input: F64InputKind::Hlc"),
        "ASTC f64 CUDA route must bind high, low and close"
    );

    let data = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");
    let descriptor = data
        .split_once("ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {")
        .and_then(|(_, tail)| {
            tail.split_once("ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {")
        })
        .map(|(body, _)| body)
        .expect("Data resident projection must describe the ASTC launch");
    assert!(
        descriptor.contains("ResidentClassicTaInputV3::Hlc"),
        "Data resident projection must not downgrade ASTC HLC authority to close-only metadata"
    );

    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let dispatcher = runtime
        .split_once("\"adaptive_schaff_trend_cycle\" => {")
        .and_then(|(_, tail)| tail.split_once("\"adjustable_ma_alternating_extremities\" => {"))
        .map(|(body, _)| body)
        .expect("resident ASTC dispatcher must exist");
    assert!(
        dispatcher.contains("require_named_input_v3(launch, ResidentClassicTaInputV3::Hlc)?;"),
        "resident ASTC dispatcher must require the same exact HLC recipe authority"
    );
}

#[test]
fn adjustable_ma_cuda_uses_cpu_owned_exact_weights_and_noncontracted_accumulation() {
    let cpu = read(
        "vendor/vector-ta-0.2.9-patched/src/indicators/adjustable_ma_alternating_extremities.rs",
    );
    assert!(
        cpu.contains("pub(crate) fn adjustable_ma_alternating_extremities_exact_weights(")
            && cpu
                .matches("adjustable_ma_alternating_extremities_exact_weights(")
                .count()
                >= 2,
        "CPU calculation and CUDA upload must share one exact normalized-weight authority"
    );

    let wrapper = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = wrapper
        .split_once("pub fn adjustable_ma_alternating_extremities_all_outputs(")
        .and_then(|(_, tail)| tail.split_once("pub fn "))
        .map(|(body, _)| body)
        .expect("Adjustable MA all-output wrapper must remain source-visible");
    for token in [
        "adjustable_ma_alternating_extremities_exact_weights(",
        "exact_weight_rows",
        "weight_stride",
        "coefficient_bytes",
        "d_exact_weights",
        "_parameter_f64",
    ] {
        assert!(
            resident.contains(token),
            "Adjustable MA resident wrapper omitted exact coefficient-lifecycle token `{token}`"
        );
    }

    let cuda = read(
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/adjustable_ma_alternating_extremities_kernel.cu",
    );
    let all_outputs = cuda
        .split_once("extern \"C\" __global__ void adjustable_ma_alternating_extremities_batch_f64(")
        .and_then(|(_, tail)| tail.split_once("// NEOETHOS f64 LANE"))
        .map(|(body, _)| body)
        .expect("Adjustable MA all-output CUDA kernel must remain source-visible");
    for token in [
        "const double* __restrict__ normalized_weights",
        "int weight_stride",
        "const double* row_weights",
        "__dadd_rn(ma_acc, __dmul_rn(close[i - j], row_weights[j]))",
        "Gate172",
        "0x3ff1335952bbb637",
        "0x3ff1335952bbb636",
    ] {
        assert!(
            all_outputs.contains(token),
            "Adjustable MA all-output CUDA kernel omitted exact-bit token `{token}`"
        );
    }
    for forbidden in ["raw_weight(", "pow(", "sin("] {
        assert!(
            !all_outputs.contains(forbidden),
            "Adjustable MA all-output CUDA kernel still rebuilds CPU-owned weights via `{forbidden}`"
        );
    }
}

#[test]
fn alma_cuda_uses_the_cpu_exact_weight_row_instead_of_libdevice_exp() {
    let cpu = read("vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/alma.rs");
    assert!(
        cpu.contains("pub(crate) fn alma_exact_weight_row(")
            && cpu.matches("alma_exact_weight_row(").count() >= 2,
        "CPU ALMA calculation and CUDA upload must share one exact Gaussian-weight authority"
    );
    for token in [
        "ALMA_TRADINGVIEW_NONFLOORED_SEMANTICS_V1",
        "let m = offset * (period - 1) as f64;",
        "let s = period as f64 / sigma;",
        "let weight = (-(diff * diff) / s2).exp();",
        "Kernel::Auto => Kernel::Scalar",
        "let kernel = Kernel::Scalar;",
    ] {
        assert!(
            cpu.contains(token),
            "CPU ALMA canonical scalar authority omitted published-formula token `{token}`"
        );
    }
    assert!(
        !cpu.contains("detect_best_kernel") && !cpu.contains("detect_best_batch_kernel"),
        "CPU ALMA Auto semantics still depend on the host SIMD/FMA feature set"
    );

    let wrapper = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    for token in [
        "alma_exact_weight_row",
        "alma_coefficient_stride_v3",
        "prepare_alma_exact_coefficient_rows_v3",
        "parameter_f64_host",
        "parameter_f64",
    ] {
        assert!(
            wrapper.contains(token),
            "resident ALMA wrapper omitted exact coefficient-lifecycle token `{token}`"
        );
    }

    let cuda = read("vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/alma_kernel.cu");
    let f64_lane = cuda
        .split_once("extern \"C\" __global__ void neoethos_alma_batch_f64(")
        .map(|(_, body)| body)
        .expect("ALMA f64 CUDA kernel must remain source-visible");
    for token in [
        "const double* __restrict__ exact_coefficients",
        "int coefficient_stride",
        "const double* weights",
        "Gate182",
        "0x3ff1333f3da1ceaa",
        "0x3ff1333f3da1ceab",
        "hardware-dependent AVX2/FMA oracle",
        "canonical non-FMA scalar result",
    ] {
        assert!(
            f64_lane.contains(token),
            "ALMA f64 CUDA kernel omitted exact-bit token `{token}`"
        );
    }
    assert!(
        !f64_lane.contains("exp("),
        "ALMA f64 CUDA kernel still rebuilds CPU-owned weights with libdevice exp"
    );
}

#[test]
fn avsl_resident_recipe_binds_the_actual_production_entry_point() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");
    let descriptor = data
        .split_once("ResolvedClassicCudaLaunch::Avsl {")
        .and_then(|(_, tail)| tail.split_once("ResolvedClassicCudaLaunch::Bandpass {"))
        .map(|(body, _)| body)
        .expect("Data resident projection must describe the AVSL launch");

    let wrapper = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let production = wrapper
        .split_once("pub fn avsl_production_output(")
        .and_then(|(_, tail)| tail.split_once("pub fn "))
        .map(|(body, _)| body)
        .expect("AVSL production wrapper must remain source-visible");

    for source in [descriptor, production] {
        assert!(
            source.contains("avsl_production_f64"),
            "AVSL recipe and observed runtime manifest must bind the same production entry point"
        );
    }
    assert!(
        !descriptor.contains("avsl_outputs_f64"),
        "Data AVSL recipe retained the stale entry-point identity rejected by the device run"
    );
}

#[test]
fn chop_uses_the_published_formula_with_one_platform_independent_log10_authority() {
    let cpu = read("vendor/vector-ta-0.2.9-patched/src/indicators/chop.rs");
    for token in [
        "CHOP_TRADINGVIEW_LOG10_SEMANTICS_V1",
        "https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/",
        "pub(crate) fn chop_log10_positive_exact_v1(",
        "fn chop_value_from_ratio_exact_v1(",
        "let ratio = rolling_sum_atr / range;",
        "chop_value_from_ratio_exact_v1(ratio, scalar, log10_period)",
        "denominator <= 49",
    ] {
        assert!(
            cpu.contains(token),
            "CPU CHOP authority omitted published deterministic token `{token}`"
        );
    }
    for forbidden in [
        "rolling_sum_atr.log10() - range.log10()",
        "(ratio - 1.0).ln_1p()",
        "scale_ln * ratio.ln()",
    ] {
        assert!(
            !cpu.contains(forbidden),
            "CPU CHOP retained platform-dependent algebra `{forbidden}`"
        );
    }

    let cuda = read("vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/chop_kernel.cu");
    let f64_lane = cuda
        .split_once("// S1 f64 LANE  --  chop")
        .map(|(_, body)| body)
        .expect("CHOP f64 CUDA lane must remain source-visible");
    for token in [
        "neoethos_chop_log10_cpu_exact_v1",
        "neoethos_chop_value_from_ratio_exact_v1",
        "denominator <= 49U",
        "const double ratio = rolling_sum_atr / range;",
        "CHOP_TRADINGVIEW_LOG10_SEMANTICS_V1",
        "if (value == INFINITY) return INFINITY;",
        "Gate196",
        "0x40569935e3af9c88",
        "0x40569935e3af9c87",
    ] {
        assert!(
            f64_lane.contains(token),
            "CUDA CHOP authority omitted published deterministic token `{token}`"
        );
    }
    for forbidden in [
        "scalar / log(14.0)",
        "scale_ln * log1p(",
        "scale_ln * log(ratio)",
        "log10(rolling_sum_atr) - log10(range)",
        "log10((double)period)",
        "if (isinf(value))",
    ] {
        assert!(
            !f64_lane.contains(forbidden),
            "CUDA CHOP still calls platform libdevice through `{forbidden}`"
        );
    }
}

#[test]
fn cora_wave_uses_redk_weights_from_one_cpu_owned_coefficient_authority() {
    let cpu = read("vendor/vector-ta-0.2.9-patched/src/indicators/cora_wave.rs");
    for token in [
        "CORA_WAVE_REDK_COMPOUND_RATIO_SEMANTICS_V1",
        "CORA_WAVE_REDK_OPEN_SOURCE_V3",
        "https://www.tradingview.com/script/NgLjvBWA-RedK-Compound-Ratio-Moving-Average-CoRa-Wave/",
        "https://pine-facade.tradingview.com/pine-facade/get/PUB%3BpOgLeMa347Zkml8worsHeXu8bvuxiHrh/last",
        "pub(crate) fn cora_wave_exact_weight_row(",
        "fn cora_wave_roll_forward_exact_v1(",
        "scalar_and_batch_share_direct_short_input_smoothing_bits",
        "let without_old = (-a_old).mul_add(x_old, previous * inv_r);",
        "w_last.mul_add(x_new, without_old)",
        "let mut w = start_wt * base;",
        "weights.push(w);",
        "w *= base;",
    ] {
        assert!(
            cpu.contains(token),
            "CPU CoRa Wave omitted exact compound-ratio authority token `{token}`"
        );
    }
    for forbidden in [
        "S = (S * inv_R) - a_old * x_old + w_last * x_new;",
        "self.S = (self.S * self.inv_R) - self.a_old * x_old + self.w_last * x_new;",
    ] {
        assert!(
            !cpu.contains(forbidden),
            "CPU CoRa Wave still leaves recurrence contraction to the compiler via `{forbidden}`"
        );
    }
    let batch_row = cpu
        .split_once("unsafe fn cora_wave_row_scalar_with_weights(")
        .and_then(|(_, tail)| tail.split_once("\n#[cfg(test)]"))
        .map(|(body, _)| body)
        .expect("production CoRa Wave batch-row body must remain source-visible");
    assert_eq!(
        batch_row.matches("if n < 100_000 {").count(),
        2,
        "short production batches must use direct WMA accumulation in both period branches"
    );

    let wrapper = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    for token in [
        "cora_wave_exact_weight_row",
        "prepare_cora_wave_exact_coefficient_rows_v3",
        "F64Kernel::CoraWave => cora_wave_coefficient_stride_v3(periods).map(Some)",
        "resident CoRa Wave chunk retains its exact coefficient buffer",
    ] {
        assert!(
            wrapper.contains(token),
            "resident CoRa Wave launch omitted CPU-owned coefficient token `{token}`"
        );
    }

    let cuda =
        read("vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/cora_wave_kernel.cu");
    let f64_lane = cuda
        .split_once("// NEOETHOS f64 LANE  --  closer 4, round 3")
        .map(|(_, body)| body)
        .expect("CoRa Wave f64 CUDA lane must remain source-visible");
    for token in [
        "const double* __restrict__ exact_coefficients",
        "const int coefficient_stride",
        "const double* __restrict__ weights =",
        "const double inv_wsum = weights[coefficient_stride - 1];",
        "Gate203",
        "0x3ff1333a931db83d",
        "0x3ff1333a931db83e",
        "Gate208",
        "0x3ff1334448be5cc8",
        "0x3ff1334448be5cc7",
        "const double without_old = fma(-a_old, x_old, previous * inv_r);",
        "return fma(w_last, x_new, without_old);",
    ] {
        assert!(
            f64_lane.contains(token),
            "CoRa Wave CUDA lane omitted exact coefficient token `{token}`"
        );
    }
    for forbidden in [
        "const double r = pow(",
        "double wj = start_wt * base;",
        "wj *= base;",
        "S = (S * inv_R) - a_old * x_old + w_last * x_new;",
    ] {
        assert!(
            !f64_lane.contains(forbidden),
            "CoRa Wave CUDA lane still regenerates CPU-owned coefficients via `{forbidden}`"
        );
    }
}
