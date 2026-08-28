use std::{fs, path::PathBuf};

fn source(path: &str) -> String {
    let path = option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
        .join(path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has no opening brace");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).expect("unbalanced closing brace");
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function `{signature}` has no closing brace");
}

#[cfg(feature = "gpu-cuda")]
fn has_f64_resident_output_route(indicator_id: &str, output_id: &str) -> bool {
    vector_ta::indicators::dispatch::has_f64_resident_output_route(indicator_id, output_id)
}

#[cfg(not(feature = "gpu-cuda"))]
fn named_resident_route_is_declared(capability: &str, indicator_id: &str, output_id: &str) -> bool {
    let indicator = format!("\"{indicator_id}\"");
    let output = format!("\"{output_id}\"");

    for (indicator_offset, _) in capability.match_indices(&indicator) {
        let Some(group_open) = capability[..indicator_offset].rfind('(') else {
            continue;
        };
        let mut depth = 0usize;
        for (relative_offset, byte) in capability.as_bytes()[group_open..]
            .iter()
            .copied()
            .enumerate()
        {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    depth = depth.checked_sub(1).expect("unbalanced route group");
                    if depth == 0 {
                        let group = &capability[group_open..group_open + relative_offset + 1];
                        if group.contains(&output) {
                            return true;
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    false
}

#[cfg(not(feature = "gpu-cuda"))]
fn has_f64_resident_output_route(indicator_id: &str, output_id: &str) -> bool {
    let dispatch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let capability = function_body(&dispatch, "pub fn has_f64_resident_output_route(");
    if named_resident_route_is_declared(capability, indicator_id, output_id) {
        return true;
    }

    let kernel_table_end = dispatch
        .find("pub fn f64_kernel_for(")
        .expect("missing f64 kernel lookup");
    let kernel_row = format!("indicator_id: \"{indicator_id}\",");
    if !dispatch[..kernel_table_end].contains(&kernel_row) {
        return false;
    }

    let primary = function_body(&dispatch, "pub fn primary_output_id(");
    let explicit_prefix = format!("\"{indicator_id}\" => Some(");
    if primary.contains(&explicit_prefix) {
        let explicit_route = format!("\"{indicator_id}\" => Some(\"{output_id}\")");
        return primary.contains(&explicit_route);
    }

    vector_ta::indicators::registry::get_indicator(indicator_id)
        .and_then(|info| info.outputs.first())
        .is_some_and(|output| output.id == output_id)
}

#[test]
fn gpu_only_enters_one_exact_cuda_executor_instead_of_the_cpu_body() {
    let hpc = source("src/core/hpc_ta.rs");
    let wrapper = function_body(&hpc, "pub fn compute_classic_ta_columns_sized_report(");
    let entry = function_body(
        &hpc,
        "pub fn compute_classic_ta_columns_sized_report_with_run_plan(",
    );

    assert!(
        wrapper.contains("prepare_classic_ta_run_plan")
            && wrapper.contains("compute_classic_ta_columns_sized_report_with_run_plan")
            && entry.contains("IndicatorComputePolicy::GpuOnly")
            && entry.contains("execute_gpu_only_classic_plan"),
        "GpuOnly must return through the named exact CUDA executor"
    );
    assert!(
        !entry.contains("GpuOnly preflight rejected before any CPU or CUDA work"),
        "the current unconditional rejection is not a production CUDA route"
    );
}

#[test]
fn one_admission_decision_precedes_both_cuda_and_cpu_allocation() {
    let hpc = source("src/core/hpc_ta.rs");
    let prepare = function_body(&hpc, "pub fn prepare_classic_ta_run_plan(");
    let wrapper = function_body(&hpc, "pub fn compute_classic_ta_columns_sized_report(");
    let entry = function_body(
        &hpc,
        "pub fn compute_classic_ta_columns_sized_report_with_run_plan(",
    );
    assert_eq!(
        prepare.matches("build_classic_ta_admission_plan").count(),
        1,
        "production must probe RAM/admit the vocabulary exactly once"
    );
    assert_eq!(
        wrapper.matches("prepare_classic_ta_run_plan").count(),
        1,
        "the sized entrypoint must capture exactly one run-wide admission plan"
    );
    assert!(
        !entry.contains("build_classic_ta_admission_plan"),
        "execution through a frozen run plan must not probe/admit again"
    );
    let admission = prepare
        .find("build_classic_ta_admission_plan")
        .expect("missing shared admission decision");
    let cuda_preflight = prepare
        .find("build_exact_classic_cuda_plan")
        .expect("missing exact CUDA preflight plan");
    assert!(
        admission < cuda_preflight,
        "admission must be finalized before exact CUDA preflight"
    );
    let frozen_admission = entry
        .find("let admission = run_plan.admission.clone()")
        .expect("execution did not consume the frozen admission");
    let cuda = entry
        .find("build_exact_classic_cuda_plan")
        .expect("missing exact CUDA execution plan");
    let cpu_allocation = entry
        .find("Candles::new")
        .expect("missing CPU Candles boundary");
    assert!(
        frozen_admission < cuda && frozen_admission < cpu_allocation,
        "admission must be finalized before either execution lane allocates"
    );
}

#[test]
fn exact_cuda_executor_preflights_before_one_engine_and_has_no_cpu_escape() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let body = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );

    let preflight = body
        .find("resolve_gpu_only_classic_plan")
        .expect("executor must preflight the exact admitted plan");
    let engine = body
        .find("GpuIndicatorEngine::new")
        .expect("executor must construct the real CUDA engine");
    assert!(
        preflight < engine,
        "every unsupported output must be rejected before the first CUDA context/launch"
    );
    assert_eq!(
        body.matches("GpuIndicatorEngine::new").count(),
        1,
        "one frame must own exactly one resident CUDA engine/session"
    );
    for required in [
        "compute_primary_device",
        "download_primary_output_f64",
        "engine.synchronize",
        "ClassicTaComputation",
        "admission.admission_ledger",
        "admission.execution_report",
    ] {
        assert!(
            body.contains(required),
            "strict CUDA executor is only structural; missing real lifecycle step `{required}`"
        );
    }

    for forbidden in [
        "compute_cpu",
        "cpu_multi_period_all",
        "cpu_multi_period_columns",
        "Kernel::Auto",
        "IndicatorComputePolicy::CpuOnly",
        "Fallback",
    ] {
        assert!(
            !body.contains(forbidden),
            "strict CUDA executor contains forbidden CPU/fallback route `{forbidden}`"
        );
    }
}

#[test]
fn exact_plan_is_built_from_admission_and_the_installed_working_set() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let body = function_body(
        &implementation,
        "pub(crate) fn build_exact_classic_cuda_plan(",
    );

    for required in [
        "admitted_indicator_ids",
        "historical",
        "extended_groups",
        "planned_output_count",
        "append_indicator_nodes",
    ] {
        assert!(
            body.contains(required),
            "exact CUDA plan does not consume production planning fact `{required}`"
        );
    }
    assert!(
        implementation.contains("output_ids_for(indicator_id)"),
        "typed output expansion must use the canonical production output resolver"
    );
}

#[test]
fn first_named_family_uses_one_exact_resident_all_output_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let tuple = function_body(
        &implementation,
        "fn resolve_absolute_strength_index_oscillator_parameters(",
    );
    for required in [
        "get_indicator",
        "ema_length",
        "signal_length",
        "ParamValueStatic::Int",
        "overrides",
    ] {
        assert!(
            tuple.contains(required),
            "ASI production routing does not prove exact parameter fact `{required}`"
        );
    }

    assert!(
        implementation.contains("AbsoluteStrengthIndexOscillator {")
            && implementation.contains("ASI_OUTPUT_IDS")
            && implementation.contains("group_resolved_classic_cuda_launches"),
        "ASI outputs must preflight as one typed canonical family, not three unrelated primaries"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_absolute_strength_index_oscillator_outputs_device")
            .count(),
        1,
        "the canonical ASI family must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("download_named_outputs_f64"),
        "ASI output matrices must remain resident until the existing f64 feature boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let download = function_body(&gpu, "pub fn download_named_outputs_f64(");
    assert!(
        download.contains("download_matrix_f64")
            && download.contains("output.output_id")
            && download.contains("result.rows")
            && download.contains("result.cols"),
        "named output materialization must validate identity and exact resident matrix shape"
    );
}

#[test]
fn adaptive_bounds_cpu_dispatch_uses_canonical_registry_output_ids_only() {
    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let field = function_body(&cpu, "fn adaptive_bounds_rsi_field(");
    for canonical in ["lower", "middle", "upper"] {
        assert!(
            field.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Adaptive Bounds RSI CPU dispatch is missing canonical registry output `{canonical}`"
        );
    }
    for retired in ["lower_bound", "mid", "upper_bound"] {
        assert!(
            !field.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "unversioned retired Adaptive Bounds RSI output alias `{retired}` is still accepted"
        );
    }

    let ledger = source("src/core/indicator_ledger.rs");
    assert!(
        !ledger.contains("EXPECTED_NON_PRODUCING_OUTPUTS")
            && !ledger.contains("expected_non_producing_output"),
        "the repaired output-level exception mechanism and its stale tests must be deleted"
    );

    let generator = source("../../vendor/vector-ta-0.2.9-patched/src/bin/generate_references.rs");
    for (canonical, retired) in [
        ("\"lower\":", "\"lower_bound\":"),
        ("\"middle\":", "\"mid\":"),
        ("\"upper\":", "\"upper_bound\":"),
    ] {
        assert!(generator.contains(canonical));
        assert!(
            !generator.contains(retired),
            "reference generator still publishes retired output identity `{retired}`"
        );
    }
}

#[test]
fn adaptive_bounds_uses_one_exact_typed_resident_launch_for_admitted_outputs() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_adaptive_bounds_rsi_parameters(",
    );
    for required in [
        "get_indicator",
        "rsi_length",
        "alpha",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "overrides",
        "to_bits",
    ] {
        assert!(
            parameters.contains(required),
            "Adaptive Bounds RSI route does not prove exact parameter fact `{required}`"
        );
    }

    assert!(
        implementation.contains("AdaptiveBoundsRsi {")
            && implementation.contains("ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS")
            && implementation.contains("ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS"),
        "Adaptive Bounds RSI needs distinct typed production and full-kernel output contracts"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adaptive_bounds_rsi_outputs_device")
            .count(),
        1,
        "Adaptive Bounds RSI must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("AdaptiveBoundsRsiParams")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("planned_output_ids"),
        "the typed route must build exact params and download only admitted named outputs"
    );
}

#[test]
fn adaptive_macd_cpu_dispatch_uses_canonical_registry_output_ids_only() {
    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn adaptive_macd_field(");
    for canonical in ["macd", "signal", "hist"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Adaptive MACD CPU dispatch is missing canonical registry output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "the unversioned non-registry Adaptive MACD output alias `value` is still accepted"
    );
}

#[test]
fn adaptive_macd_uses_one_exact_typed_resident_launch_per_parameter_point() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_adaptive_macd_parameters(");
    for required in [
        "get_indicator",
        "length",
        "fast_period",
        "slow_period",
        "signal_period",
        "ParamValueStatic::Int",
        "overrides",
    ] {
        assert!(
            parameters.contains(required),
            "Adaptive MACD route does not prove exact parameter fact `{required}`"
        );
    }

    assert!(
        implementation.contains("AdaptiveMacd {")
            && implementation.contains("ADAPTIVE_MACD_OUTPUT_IDS")
            && implementation.contains("ADAPTIVE_MACD_PARAMETER_KEYS"),
        "Adaptive MACD outputs and four-parameter ABI must form one typed canonical family"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adaptive_macd_outputs_device")
            .count(),
        1,
        "Adaptive MACD must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("download_named_outputs_f64")
            && executor.contains("ADAPTIVE_MACD_OUTPUT_IDS"),
        "Adaptive MACD matrices must remain resident until the canonical f64 feature boundary"
    );
}

#[test]
fn adaptive_momentum_cpu_dispatch_uses_canonical_registry_output_ids_only() {
    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn adaptive_momentum_oscillator_field(");
    for canonical in ["amo", "ama"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Adaptive Momentum Oscillator CPU dispatch is missing canonical registry output \
             `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "the unversioned non-registry Adaptive Momentum Oscillator output alias `value` is still \
         accepted"
    );
}

#[test]
fn adaptive_momentum_uses_one_exact_typed_resident_launch_per_parameter_point() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_adaptive_momentum_oscillator_parameters(",
    );
    for required in [
        "get_indicator",
        "length",
        "smoothing_length",
        "output",
        "ParamValueStatic::Int",
        "ParamValueStatic::EnumString",
        "overrides",
    ] {
        assert!(
            parameters.contains(required),
            "Adaptive Momentum Oscillator route does not prove exact parameter fact `{required}`"
        );
    }

    assert!(
        implementation.contains("AdaptiveMomentumOscillator {")
            && implementation.contains("ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS")
            && implementation.contains("ADAPTIVE_MOMENTUM_OSCILLATOR_PARAMETER_KEYS"),
        "Adaptive Momentum Oscillator outputs and parameter ABI must form one typed family"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adaptive_momentum_oscillator_outputs_device")
            .count(),
        1,
        "Adaptive Momentum Oscillator must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("download_named_outputs_f64")
            && executor.contains("ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS"),
        "Adaptive Momentum Oscillator matrices must remain resident until the f64 boundary"
    );
}

#[test]
fn adaptive_schaff_registry_and_cpu_dispatch_use_one_canonical_contract() {
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let parameters = registry
        .split_once("const PARAM_ADAPTIVE_SCHAFF_TREND_CYCLE:")
        .map(|(_, tail)| tail.split_once("];").map_or(tail, |(item, _)| item))
        .expect("Adaptive Schaff Trend Cycle needs an exact registry parameter schema");
    let mut cursor = 0usize;
    for (key, default) in [
        ("adaptive_length", "ParamValueStatic::Int(55)"),
        ("stc_length", "ParamValueStatic::Int(12)"),
        ("smoothing_factor", "ParamValueStatic::Float(0.45)"),
        ("fast_length", "ParamValueStatic::Int(26)"),
        ("slow_length", "ParamValueStatic::Int(50)"),
    ] {
        let key_offset = parameters[cursor..]
            .find(&format!("key: \"{key}\""))
            .unwrap_or_else(|| panic!("registry omitted canonical ASTC parameter `{key}`"));
        cursor += key_offset;
        let default_offset = parameters[cursor..]
            .find(default)
            .unwrap_or_else(|| panic!("registry changed the ASTC default for `{key}`"));
        cursor += default_offset + default.len();
    }

    let outputs = registry
        .split_once("const OUTPUTS_ADAPTIVE_SCHAFF_TREND_CYCLE:")
        .map(|(_, tail)| tail.split_once("];").map_or(tail, |(item, _)| item))
        .expect("ASTC needs a canonical registry output schema");
    let stc = outputs
        .find("id: \"stc\"")
        .expect("ASTC registry omitted canonical output `stc`");
    let histogram = outputs
        .find("OUTPUT_HISTOGRAM")
        .expect("ASTC registry omitted canonical output `histogram`");
    assert!(
        stc < histogram,
        "ASTC registry output order must be [stc, histogram]"
    );
    assert_eq!(
        outputs.matches("IndicatorOutputInfo").count(),
        2,
        "ASTC registry declared an unexpected inline output"
    );
    assert_eq!(
        outputs.matches("OUTPUT_HISTOGRAM").count(),
        1,
        "ASTC registry declared an unexpected named output"
    );
    let seed = registry
        .split_once("id: \"adaptive_schaff_trend_cycle\"")
        .map(|(_, tail)| tail.split_once("},\n").map_or(tail, |(item, _)| item))
        .expect("ASTC must be a canonical registry entry");
    assert!(
        seed.contains("outputs: OUTPUTS_ADAPTIVE_SCHAFF_TREND_CYCLE")
            && seed.contains("params: PARAM_ADAPTIVE_SCHAFF_TREND_CYCLE"),
        "ASTC registry entry is not wired to its exact output and parameter schemas"
    );

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn adaptive_schaff_trend_cycle_field(");
    for canonical in ["stc", "histogram"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "ASTC CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    for retired in ["value", "hist"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "unversioned retired ASTC output alias `{retired}` is still accepted"
        );
    }
}

#[test]
fn adaptive_schaff_uses_one_exact_typed_resident_launch_per_parameter_point() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_adaptive_schaff_trend_cycle_parameters(",
    );
    for required in [
        "get_indicator",
        "adaptive_length",
        "stc_length",
        "smoothing_factor",
        "fast_length",
        "slow_length",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "overrides",
        "to_bits",
    ] {
        assert!(
            parameters.contains(required),
            "ASTC route does not prove exact parameter fact `{required}`"
        );
    }

    assert!(
        implementation.contains("AdaptiveSchaffTrendCycle {")
            && implementation.contains("ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS")
            && implementation.contains("ADAPTIVE_SCHAFF_TREND_CYCLE_PARAMETER_KEYS"),
        "ASTC outputs and five-parameter ABI must form one typed canonical family"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adaptive_schaff_trend_cycle_outputs_device")
            .count(),
        1,
        "ASTC must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("AdaptiveSchaffTrendCycleParams")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS"),
        "ASTC matrices must remain resident until the canonical f64 boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_adaptive_schaff_trend_cycle_outputs_device(",
    );
    assert!(
        bridge.contains("adaptive_schaff_trend_cycle_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.adaptive_schaff_valid_suffix_len"),
        "ASTC GPU bridge must use the shared resident HLC upload and its exact CPU validity rule"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn adaptive_schaff_trend_cycle_all_outputs(");
    for required in [
        "adaptive_schaff_trend_cycle_batch_f64",
        "F64Kernel::AdaptiveSchaffTrendCycle",
        "stc",
        "histogram",
        "_parameter_i32",
        "_parameter_f64",
        "_scratch_f64",
        "_scratch_i32",
        "queue_stride",
        "i32::MAX",
    ] {
        assert!(
            resident.contains(required),
            "ASTC resident wrapper is missing exact lifecycle fact `{required}`"
        );
    }
    assert!(
        wrapper.contains("ohlcv: CudaDeviceOhlcvF64Ref"),
        "ASTC resident wrapper must borrow the existing f64 OHLCV upload"
    );
    assert!(
        !resident.contains("adaptive_schaff_trend_cycle_neo_batch_f64"),
        "production ASTC must not use the pinned one-output research entry point"
    );
}

#[test]
fn adjustable_ma_cpu_dispatch_uses_registry_ids_without_value_alias() {
    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn adjustable_ma_alternating_extremities_field(");
    for canonical in [
        "ma",
        "upper",
        "lower",
        "extremity",
        "state",
        "changed",
        "smoothed_open",
        "smoothed_high",
        "smoothed_low",
        "smoothed_close",
    ] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Adjustable MA CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "unversioned retired Adjustable MA `value` alias is still accepted"
    );

    let ledger = source("src/core/indicator_ledger.rs");
    assert!(
        ledger.contains("\"adjustable_ma_alternating_extremities\"")
            && ledger.contains("Some(\"smoothed_close\")")
            && ledger.contains("structurally identical"),
        "the reviewed ma/smoothed_close production exclusion must remain authoritative"
    );
}

#[test]
fn adjustable_ma_uses_one_full_resident_launch_for_nine_admitted_outputs() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_adjustable_ma_alternating_extremities_parameters(",
    );
    for required in [
        "get_indicator",
        "length",
        "mult",
        "alpha",
        "beta",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "overrides",
        "to_bits",
    ] {
        assert!(
            parameters.contains(required),
            "Adjustable MA typed resolver is missing `{required}`"
        );
    }
    assert!(
        implementation.contains("AdjustableMaAlternatingExtremities {")
            && implementation.contains("ADJUSTABLE_MA_FULL_OUTPUT_IDS")
            && implementation.contains("ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS")
            && implementation.contains("ADJUSTABLE_MA_PARAMETER_KEYS"),
        "Adjustable MA needs one exact typed full-kernel/admitted-schema contract"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adjustable_ma_alternating_extremities_outputs_device")
            .count(),
        1,
        "Adjustable MA must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ADJUSTABLE_MA_FULL_OUTPUT_IDS")
            && executor.contains("ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "the full ten-output ABI must be checked while only nine planned IDs materialize"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_adjustable_ma_alternating_extremities_outputs_device(",
    );
    assert!(
        bridge.contains("adjustable_ma_alternating_extremities_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.max_consecutive_finite_hlc"),
        "Adjustable MA bridge must use resident HLC and the exact CPU run admission rule"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn adjustable_ma_alternating_extremities_all_outputs(",
    );
    assert!(
        resident.contains("adjustable_ma_alternating_extremities_batch_f64")
            && resident.contains("F64Kernel::AdjustableMaAlternatingExtremities")
            && resident.contains("const OUTPUT_IDS: [&str; 10]")
            && resident.contains("_parameter_i32")
            && resident.contains("_parameter_f64"),
        "Adjustable MA full resident kernel/lifecycle ABI is incomplete"
    );
}

#[test]
fn bulls_v_bears_registry_cpu_and_exclusions_match_canonical_schema() {
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let outputs = registry
        .split_once("const OUTPUTS_BULLS_V_BEARS:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("Bulls v Bears needs a canonical registry output schema");
    let mut previous = None;
    for token in [
        "OUTPUT_VALUE_F64",
        "OUTPUT_BULL",
        "OUTPUT_BEAR",
        "OUTPUT_MA",
        "OUTPUT_UPPER",
        "OUTPUT_LOWER",
        "OUTPUT_BULLISH_SIGNAL",
        "OUTPUT_BEARISH_SIGNAL",
        "OUTPUT_ZERO_CROSS_UP",
        "OUTPUT_ZERO_CROSS_DOWN",
    ] {
        let position = outputs
            .find(token)
            .unwrap_or_else(|| panic!("Bulls v Bears registry omitted `{token}`"));
        assert!(
            previous.is_none_or(|earlier| earlier < position),
            "Bulls v Bears registry output order drifted at `{token}`"
        );
        previous = Some(position);
    }

    let params = registry
        .split_once("const PARAM_BULLS_V_BEARS:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("Bulls v Bears needs an exact registry parameter schema");
    let mut previous = None;
    for key in [
        "period",
        "ma_type",
        "calculation_method",
        "normalized_bars_back",
        "raw_rolling_period",
        "raw_threshold_percentile",
        "threshold_level",
    ] {
        let token = format!("key: \"{key}\"");
        let position = params
            .find(&token)
            .unwrap_or_else(|| panic!("Bulls v Bears registry omitted `{key}`"));
        assert!(
            previous.is_none_or(|earlier| earlier < position),
            "Bulls v Bears registry parameter order drifted at `{key}`"
        );
        previous = Some(position);
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_bulls_v_bears_batch(");
    for canonical in [
        "value",
        "bull",
        "bear",
        "ma",
        "upper",
        "lower",
        "bullish_signal",
        "bearish_signal",
        "zero_cross_up",
        "zero_cross_down",
    ] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Bulls v Bears CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        10,
        "Bulls v Bears CPU dispatch retained a non-registry output alias"
    );

    let ledger = source("src/core/indicator_ledger.rs");
    let exclusions = ledger
        .split_once("pub const PRODUCTION_OUTPUT_EXCLUSIONS:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("production exclusions must remain an explicit schema contract");
    assert_eq!(
        exclusions.matches("\"bulls_v_bears\"").count(),
        3,
        "Bulls v Bears must keep exactly its three reviewed output exclusions"
    );
    for excluded in ["ma", "upper", "lower"] {
        assert!(
            exclusions.contains(&format!("Some(\"{excluded}\")")),
            "Bulls v Bears reviewed exclusion `{excluded}` disappeared"
        );
    }
}

#[test]
fn bulls_v_bears_uses_one_full_resident_launch_for_seven_admitted_outputs() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_bulls_v_bears_parameters(");
    for required in [
        "get_indicator",
        "period",
        "ma_type",
        "calculation_method",
        "normalized_bars_back",
        "raw_rolling_period",
        "raw_threshold_percentile",
        "threshold_level",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ParamValueStatic::EnumString",
        "overrides",
        "to_bits",
    ] {
        assert!(
            parameters.contains(required),
            "Bulls v Bears typed resolver is missing `{required}`"
        );
    }
    assert!(
        implementation.contains("BullsVBears {")
            && implementation.contains("BULLS_V_BEARS_FULL_OUTPUT_IDS")
            && implementation.contains("BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS")
            && implementation.contains("BULLS_V_BEARS_PARAMETER_KEYS"),
        "Bulls v Bears needs one exact typed full-kernel/admitted-schema contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_bulls_v_bears_outputs_device")
            .count(),
        1,
        "Bulls v Bears must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("BULLS_V_BEARS_FULL_OUTPUT_IDS")
            && executor.contains("BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "the full ten-output ABI must be checked while only seven planned IDs materialize"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_bulls_v_bears_outputs_device(");
    assert!(
        bridge.contains("bulls_v_bears_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.max_consecutive_finite_hlc"),
        "Bulls v Bears bridge must use resident HLC and exact finite-run metadata"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn bulls_v_bears_all_outputs(");
    assert!(
        resident.contains("bulls_v_bears_batch_f64")
            && resident.contains("F64Kernel::BullsVBears")
            && resident.contains("const OUTPUT_IDS: [&str; 10]")
            && resident.contains("_parameter_i32")
            && resident.contains("_parameter_f64"),
        "Bulls v Bears full resident kernel/lifecycle ABI is incomplete"
    );
}

#[test]
fn alligator_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let runtime = vector_ta::indicators::registry::get_indicator("alligator")
        .expect("Alligator needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["jaw", "teeth", "lips"],
        "Alligator runtime registry output identity/order drifted"
    );

    let parameters = registry
        .split_once("const PARAM_ALLIGATOR:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("Alligator needs an exact registry parameter schema");
    let mut cursor = 0usize;
    for (key, default) in [
        ("jaw_period", "ParamValueStatic::Int(13)"),
        ("jaw_offset", "ParamValueStatic::Int(8)"),
        ("teeth_period", "ParamValueStatic::Int(8)"),
        ("teeth_offset", "ParamValueStatic::Int(5)"),
        ("lips_period", "ParamValueStatic::Int(5)"),
        ("lips_offset", "ParamValueStatic::Int(3)"),
    ] {
        let key_offset = parameters[cursor..]
            .find(&format!("key: \"{key}\""))
            .unwrap_or_else(|| panic!("Alligator registry omitted `{key}`"));
        cursor += key_offset;
        let default_offset = parameters[cursor..]
            .find(default)
            .unwrap_or_else(|| panic!("Alligator registry changed the default for `{key}`"));
        cursor += default_offset + default.len();
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_alligator_batch(");
    for canonical in ["jaw", "teeth", "lips"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Alligator CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "unversioned retired Alligator `value` alias is still accepted"
    );
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        3,
        "Alligator CPU dispatch retained a non-registry output alias"
    );
}

#[test]
fn alligator_uses_one_exact_three_output_resident_launch_per_parameter_tuple() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_alligator_parameters(");
    for required in [
        "get_indicator",
        "jaw_period",
        "jaw_offset",
        "teeth_period",
        "teeth_offset",
        "lips_period",
        "lips_offset",
        "ParamValueStatic::Int",
        "overrides",
    ] {
        assert!(
            parameters.contains(required),
            "Alligator typed resolver is missing `{required}`"
        );
    }
    assert!(
        implementation.contains("Alligator {")
            && implementation.contains("ALLIGATOR_OUTPUT_IDS")
            && implementation.contains("ALLIGATOR_PARAMETER_KEYS")
            && implementation.contains("ALLIGATOR_SWEEP_PARAMETER_KEYS"),
        "Alligator needs one exact typed output/default/ratio-sweep contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_alligator_outputs_device").count(),
        1,
        "Alligator must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ALLIGATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "all three Alligator matrices must stay resident until the f64 feature boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_alligator_outputs_device(");
    assert!(
        bridge.contains("alligator_all_outputs")
            && bridge.contains("self.hl2.as_view_f64()")
            && bridge.contains("self.first_valid_hl2"),
        "Alligator bridge must use the shared resident hl2 upload and exact CPU start index"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn alligator_all_outputs(");
    for required in [
        "alligator_outputs_f64",
        "F64Kernel::Alligator",
        "const OUTPUT_IDS: [&str; 3]",
        "jaw_period",
        "jaw_offset",
        "teeth_period",
        "teeth_offset",
        "lips_period",
        "lips_offset",
        "_parameter_i32",
    ] {
        assert!(
            resident.contains(required),
            "Alligator resident wrapper is missing exact ABI fact `{required}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/alligator_kernel.cu",
    );
    let full = function_body(
        &kernel,
        "extern \"C\" __global__ void alligator_outputs_f64(",
    );
    for required in [
        "jaw_periods",
        "jaw_offsets",
        "teeth_periods",
        "teeth_offsets",
        "lips_periods",
        "lips_offsets",
        "out_jaw",
        "out_teeth",
        "out_lips",
    ] {
        assert!(
            full.contains(required),
            "Alligator full kernel is missing exact ABI fact `{required}`"
        );
    }
}

#[test]
fn alphatrend_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let runtime = vector_ta::indicators::registry::get_indicator("alphatrend")
        .expect("AlphaTrend needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["k1", "k2"],
        "AlphaTrend runtime registry output identity/order drifted"
    );

    let parameters = registry
        .split_once("const PARAM_ALPHATREND:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("AlphaTrend needs an exact registry parameter schema");
    let mut cursor = 0usize;
    for (key, default) in [
        ("coeff", "ParamValueStatic::Float(1.0)"),
        ("period", "ParamValueStatic::Int(14)"),
        ("no_volume", "ParamValueStatic::Bool(false)"),
    ] {
        let key_offset = parameters[cursor..]
            .find(&format!("key: \"{key}\""))
            .unwrap_or_else(|| panic!("AlphaTrend registry omitted `{key}`"));
        cursor += key_offset;
        let default_offset = parameters[cursor..]
            .find(default)
            .unwrap_or_else(|| panic!("AlphaTrend changed the default for `{key}`"));
        cursor += default_offset + default.len();
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn alphatrend_field(");
    for canonical in ["k1", "k2"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "AlphaTrend CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "unversioned retired AlphaTrend `value` alias is still accepted"
    );
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "AlphaTrend CPU dispatch retained a non-registry output alias"
    );
}

#[test]
fn alphatrend_uses_one_exact_two_output_resident_launch_per_parameter_tuple() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_alphatrend_parameters(");
    for required in [
        "get_indicator",
        "coeff",
        "period",
        "no_volume",
        "ParamValueStatic::Float",
        "ParamValueStatic::Int",
        "ParamValueStatic::Bool",
        "overrides",
    ] {
        assert!(
            parameters.contains(required),
            "AlphaTrend typed resolver is missing `{required}`"
        );
    }
    assert!(
        implementation.contains("AlphaTrend {")
            && implementation.contains("ALPHATREND_OUTPUT_IDS")
            && implementation.contains("ALPHATREND_PARAMETER_KEYS")
            && implementation.contains("ALPHATREND_SWEEP_PARAMETER_KEYS"),
        "AlphaTrend needs one exact typed output/default/period-sweep contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_alphatrend_outputs_device")
            .count(),
        1,
        "AlphaTrend must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ALPHATREND_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "both AlphaTrend matrices must stay resident until the f64 feature boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_alphatrend_outputs_device(");
    assert!(
        bridge.contains("alphatrend_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_close"),
        "AlphaTrend bridge must reuse resident OHLCV and the exact close-only start index"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn alphatrend_all_outputs(");
    for required in [
        "alphatrend_outputs_f64",
        "F64Kernel::Alphatrend",
        "const OUTPUT_IDS: [&str; 2]",
        "coeff",
        "period",
        "no_volume",
        "_parameter_i32",
    ] {
        assert!(
            resident.contains(required),
            "AlphaTrend resident wrapper is missing exact ABI fact `{required}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/alphatrend_kernel.cu");
    let primary = function_body(
        &kernel,
        "extern \"C\" __global__\nvoid alphatrend_neo_batch_f64(",
    );
    let full = function_body(
        &kernel,
        "extern \"C\" __global__\nvoid alphatrend_outputs_f64(",
    );
    assert!(
        primary.contains("alphatrend_row_f64") && full.contains("alphatrend_row_f64"),
        "primary k1 and full k1+k2 entry points must share one exact f64 state authority"
    );
    assert!(
        full.contains("out_k1") && full.contains("out_k2"),
        "AlphaTrend full kernel must expose exactly k1 and k2"
    );
}

#[test]
fn acosc_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    let runtime = vector_ta::indicators::registry::get_indicator("acosc")
        .expect("ACOSC needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["osc", "change"],
        "ACOSC runtime registry output identity/order drifted"
    );
    assert!(
        runtime.params.is_empty(),
        "ACOSC is one exact no-parameter tuple"
    );

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn acosc_field(");
    for canonical in ["osc", "change"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "ACOSC CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "unversioned retired ACOSC `value` alias is still accepted"
    );
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "ACOSC CPU dispatch retained a non-registry output alias"
    );
}

#[test]
fn adaptive_bandpass_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime =
        vector_ta::indicators::registry::get_indicator("adaptive_bandpass_trigger_oscillator")
            .expect("Adaptive Bandpass needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["in_phase", "lead"],
        "Adaptive Bandpass runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["delta", "alpha"],
        "Adaptive Bandpass runtime parameter identity/order drifted"
    );
    for (parameter, (default, minimum, maximum)) in runtime.params.iter().zip([
        (0.1_f64, 0.0000001_f64, 0.9999999_f64),
        (0.07_f64, 0.0000001_f64, 0.9999999_f64),
    ]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        let Some(ParamValueStatic::Float(actual_default)) = parameter.default else {
            panic!("{} must carry one exact f64 default", parameter.key);
        };
        assert_eq!(actual_default.to_bits(), default.to_bits());
        assert_eq!(parameter.min.map(f64::to_bits), Some(minimum.to_bits()));
        assert_eq!(parameter.max.map(f64::to_bits), Some(maximum.to_bits()));
        assert!(minimum > 0.0 && maximum < 1.0 && default > 0.0 && default < 1.0);
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(
        &cpu,
        "fn compute_adaptive_bandpass_trigger_oscillator_batch(",
    );
    for canonical in ["in_phase", "lead"] {
        assert!(
            dispatch.contains(&format!("\"{canonical}\" =>")),
            "Adaptive Bandpass CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("\"value\""),
        "unversioned Adaptive Bandpass `value` alias is accepted"
    );
}

#[test]
fn adaptive_bandpass_uses_one_exact_typed_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_adaptive_bandpass_trigger_oscillator_parameters(",
    );
    for required in [
        "get_indicator",
        "delta",
        "alpha",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Float",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "to_bits",
    ] {
        assert!(
            parameters.contains(required),
            "Adaptive Bandpass route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS: [&str; 2] = [\"in_phase\", \"lead\"]",
        ) && implementation.contains("AdaptiveBandpassTriggerOscillator {")
            && implementation.contains("ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS"),
        "Adaptive Bandpass needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_adaptive_bandpass_trigger_oscillator_outputs_device")
            .count(),
        1,
        "Adaptive Bandpass must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("f64::from_bits(delta_bits)")
            && executor.contains("f64::from_bits(alpha_bits)"),
        "both Adaptive Bandpass matrices must stay resident through the exact f64 tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_adaptive_bandpass_trigger_oscillator_outputs_device(",
    );
    assert!(
        bridge.contains("adaptive_bandpass_trigger_oscillator_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.finite_close_count"),
        "Adaptive Bandpass bridge must borrow the shared resident close upload"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn adaptive_bandpass_trigger_oscillator_all_outputs(",
    );
    for required in [
        "adaptive_bandpass_trigger_oscillator_batch_f64",
        "F64Kernel::AdaptiveBandpassTriggerOscillator",
        "in_phase",
        "lead",
        "d_deltas",
        "d_alphas",
        "_parameter_f64",
    ] {
        assert!(
            resident.contains(required),
            "Adaptive Bandpass resident wrapper is missing exact ABI fact `{required}`"
        );
    }
}

#[test]
fn acosc_uses_one_exact_two_output_resident_state_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const ACOSC_OUTPUT_IDS: [&str; 2] = [\"osc\", \"change\"]")
            && implementation.contains("Acosc {")
            && implementation.contains("ClassicCudaResolvedRoute::Acosc"),
        "ACOSC needs one exact typed no-parameter two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_acosc_outputs_device").count(),
        1,
        "ACOSC must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ACOSC_OUTPUT_IDS") && executor.contains("download_named_outputs_f64"),
        "both ACOSC matrices must stay resident until the f64 feature boundary"
    );
    assert!(
        executor.contains("ClassicCudaResolvedRoute::Acosc =>"),
        "the primary receipt matcher must explicitly reject an impossible typed ACOSC route"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_acosc_outputs_device(");
    assert!(
        bridge.contains("acosc_all_outputs") && bridge.contains("self.ohlcv.as_view()"),
        "ACOSC bridge must reuse the resident high/low frame"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let sequential = function_body(&wrapper, "pub fn is_sequential(self) -> bool");
    assert!(
        sequential.contains("F64Kernel::Acosc"),
        "ACOSC's rolling state must be classified as sequential"
    );
    assert!(
        !wrapper.contains("acosc_outputs_bar_parallel_grid")
            && !wrapper.contains("acosc_outputs_launch_capacity"),
        "ACOSC retained stale bar-parallel occupancy assumptions"
    );
    let resident = function_body(&wrapper, "pub fn acosc_all_outputs(");
    for required in [
        "acosc_outputs_f64",
        "F64Kernel::Acosc",
        "const OUTPUT_IDS: [&str; 2]",
        "GridSize::x(1)",
        "BlockSize::x(1)",
    ] {
        assert!(
            resident.contains(required),
            "ACOSC resident wrapper is missing exact sequential ABI fact `{required}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/acosc_kernel.cu");
    let state = function_body(
        &kernel,
        "static __device__ __forceinline__ void acosc_row_f64(",
    );
    let mut cursor = 0usize;
    for operation in [
        "if (!isfinite(high_value) || !isfinite(low_value))",
        "const double median = (high_value + low_value) * 0.5",
        "if (state.median_count < 5)",
        "state.median_fast_sum += median",
        "state.median_fast_sum += median - state.median_fast[state.median_fast_index]",
        "if (state.median_count < 34)",
        "state.median_slow_sum += median",
        "state.median_count += 1",
        "state.median_slow_sum += median - state.median_slow[state.median_slow_index]",
        "if (state.median_count < 34)",
        "state.median_fast_sum / 5.0 - state.median_slow_sum / 34.0",
        "if (!isfinite(ao))",
        "if (state.ao_count < 5)",
        "state.ao_signal_sum += ao",
        "state.ao_count += 1",
        "if (state.ao_count < 5)",
        "state.ao_signal_sum += ao - state.ao_signal[state.ao_signal_index]",
        "ac = ao - state.ao_signal_sum / 5.0",
        "if (state.has_previous_ac) change = ac - state.previous_ac",
        "state.previous_ac = ac",
        "state.has_previous_ac = true",
    ] {
        let offset = state[cursor..]
            .find(operation)
            .unwrap_or_else(|| panic!("ACOSC exact state omitted or reordered `{operation}`"));
        cursor += offset + operation.len();
    }
    let primary = function_body(&kernel, "void acosc_neo_batch_f64(");
    let full = function_body(&kernel, "void acosc_outputs_f64(");
    assert!(
        kernel.contains("acosc_row_f64")
            && primary.contains("acosc_row_f64")
            && full.contains("acosc_row_f64"),
        "primary osc and full osc+change entry points must share one exact f64 state authority"
    );
    assert!(
        !kernel.contains("acosc_value_at_f64"),
        "fixed-window ACOSC recomputation changed the CPU rolling-state arithmetic order"
    );
    assert!(
        full.contains("out_osc") && full.contains("out_change"),
        "ACOSC full kernel must expose exactly osc and change"
    );
}

#[test]
fn andean_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("andean_oscillator")
        .expect("Andean Oscillator needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["bull", "bear", "signal"],
        "Andean Oscillator runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["length", "signal_length"],
        "Andean Oscillator runtime parameter identity/order drifted"
    );
    for (parameter, expected_default) in runtime.params.iter().zip([50_i64, 9_i64]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(
            parameter.default,
            Some(ParamValueStatic::Int(expected_default))
        );
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max, None);
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_andean_oscillator_batch(");
    for canonical in ["bull", "bear", "signal"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Andean Oscillator CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "unversioned retired Andean Oscillator `value` alias is accepted"
    );
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        3,
        "Andean Oscillator CPU dispatch retained a non-registry output alias"
    );
}

#[test]
fn andean_uses_one_shared_resident_three_output_launch_without_the_standalone_wrapper() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_andean_oscillator_parameters(");
    for required in [
        "get_indicator",
        "ANDEAN_OSCILLATOR_OUTPUT_IDS",
        "ANDEAN_OSCILLATOR_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "ParamValueStatic::Int",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"length\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Andean Oscillator route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const ANDEAN_OSCILLATOR_OUTPUT_IDS: [&str; 3] = [\"bull\", \"bear\", \"signal\"]",
        ) && implementation.contains("AndeanOscillator {")
            && implementation.contains("ANDEAN_OSCILLATOR_PARAMETER_KEYS"),
        "Andean Oscillator needs one exact typed three-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_andean_oscillator_outputs_device")
            .count(),
        1,
        "Andean Oscillator must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ANDEAN_OSCILLATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(length, signal_length)]"),
        "all three Andean matrices must stay resident through the exact tuple"
    );
    for forbidden in ["CudaAndeanOscillator", ".batch_dev("] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered the standalone Andean route `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_andean_oscillator_outputs_device(");
    assert!(
        bridge.contains("andean_oscillator_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_open_close_finite"),
        "Andean Oscillator bridge must borrow the shared resident open/close frame"
    );
    for forbidden in ["CudaAndeanOscillator", ".batch_dev("] {
        assert!(
            !bridge.contains(forbidden),
            "production bridge entered the standalone Andean route `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn andean_oscillator_all_outputs(");
    for required in [
        "andean_oscillator_batch_f64",
        "F64Kernel::AndeanOscillator",
        "const OUTPUT_IDS: [&str; 3]",
        "ohlcv.open()",
        "ohlcv.close()",
        "d_lengths",
        "d_signal_lengths",
        "_parameter_i32",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Andean resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "Andean bull/bear/signal must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "d_open = DeviceBuffer::from_slice",
        "d_close = DeviceBuffer::from_slice",
        ".synchronize()",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Andean route retained standalone context/upload/sync `{forbidden}`"
        );
    }

    let dispatch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let capability = function_body(&dispatch, "pub fn has_f64_resident_output_route(");
    assert!(
        capability.contains("andean_oscillator")
            && capability.contains("bear")
            && capability.contains("signal"),
        "Andean bear/signal are not advertised by the exact resident capability"
    );
}

#[test]
fn aroon_registry_and_dispatchers_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("aroon")
        .expect("Aroon needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["up", "down"],
        "Aroon runtime registry output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["length"],
        "Aroon runtime parameter identity/order drifted"
    );
    let length = &runtime.params[0];
    assert!(matches!(length.kind, IndicatorParamKind::Int));
    assert!(!length.required);
    assert_eq!(length.default, Some(ParamValueStatic::Int(14)));
    assert_eq!(length.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(length.max, None);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let cpu_dispatch = function_body(&cpu, "fn compute_aroon_batch(");
    for canonical in ["up", "down"] {
        assert!(
            cpu_dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Aroon CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    for retired in ["value", "aroon_up", "aroon_down"] {
        assert!(
            !cpu_dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "Aroon CPU dispatch retained retired output alias `{retired}`"
        );
    }
    assert_eq!(
        cpu_dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "Aroon CPU dispatch retained a non-registry output alias"
    );

    let cuda = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda.rs");
    let device_dispatch = function_body(&cuda, "fn compute_aroon_cuda_device(");
    for canonical in ["up", "down"] {
        assert!(
            device_dispatch.contains(&format!("id == \"{canonical}\"")),
            "Aroon device dispatcher is missing canonical output `{canonical}`"
        );
    }
    for retired in ["first", "second"] {
        assert!(
            !device_dispatch.contains(&format!("id == \"{retired}\"")),
            "Aroon device dispatcher retained retired output alias `{retired}`"
        );
    }

    let host = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_non_ma_generated.rs",
    );
    let host_dispatch = function_body(&host, "\"aroon\" => Some((|| {");
    assert!(
        host_dispatch.contains("let fallback_outputs: &[&str] = &[]"),
        "Aroon host CUDA dispatcher must resolve only canonical registry outputs"
    );
    for retired in ["first", "second"] {
        assert!(
            !host_dispatch.contains(&format!("\"{retired}\"")),
            "Aroon host CUDA dispatcher retained retired output alias `{retired}`"
        );
    }
}

#[test]
fn aroon_uses_one_shared_resident_two_output_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_aroon_parameters(");
    for required in [
        "get_indicator",
        "AROON_OUTPUT_IDS",
        "AROON_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "ParamValueStatic::Int",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"length\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Aroon route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const AROON_OUTPUT_IDS: [&str; 2] = [\"up\", \"down\"]")
            && implementation.contains("Aroon {")
            && implementation.contains("AROON_PARAMETER_KEYS"),
        "Aroon needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_aroon_outputs_device").count(),
        1,
        "Aroon must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("AROON_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[length]"),
        "both Aroon matrices must stay resident through the exact length tuple"
    );
    for forbidden in ["CudaAroon", ".aroon_batch_dev", "F64Kernel::Aroon,"] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered a standalone/replay Aroon route `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_aroon_outputs_device(");
    assert!(
        bridge.contains("aroon_all_outputs") && bridge.contains("self.ohlcv.as_view()"),
        "Aroon bridge must borrow the shared resident high/low frame"
    );
    for forbidden in ["CudaAroon", ".aroon_batch_dev", "compute_batch_f64"] {
        assert!(
            !bridge.contains(forbidden),
            "production bridge entered a standalone/replay Aroon route `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn aroon_all_outputs(");
    for required in [
        "aroon_outputs_f64",
        "F64Kernel::Aroon",
        "const OUTPUT_IDS: [&str; 2]",
        "ohlcv.high()",
        "ohlcv.low()",
        "d_lengths",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Aroon resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "Aroon up/down must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        ".synchronize()",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Aroon route retained standalone context/upload/sync `{forbidden}`"
        );
    }

    let sequential = function_body(&wrapper, "pub fn is_sequential(self) -> bool");
    assert!(
        sequential.contains("F64Kernel::Aroon"),
        "Aroon's rolling extreme state must remain classified as sequential"
    );
    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant
            .lines()
            .any(|line| line.trim() == "| F64Kernel::Aroon"),
        "Aroon length sweep was incorrectly classified as period-invariant"
    );

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/aroon_kernel.cu");
    let primary = function_body(&kernel, "void aroon_neo_batch_f64(");
    let full = function_body(&kernel, "void aroon_outputs_f64(");
    assert!(
        kernel.contains("aroon_row_f64")
            && primary.contains("aroon_row_f64")
            && full.contains("aroon_row_f64"),
        "primary up and full up/down entry points must share one exact f64 state authority"
    );
    assert!(
        full.contains("out_up") && full.contains("out_down"),
        "Aroon full kernel must expose exactly up and down"
    );

    let dispatch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let capability = function_body(&dispatch, "pub fn has_f64_resident_output_route(");
    assert!(
        capability.contains("(\"aroon\", \"down\")"),
        "Aroon down is not advertised by the exact resident capability"
    );
}

#[test]
fn aso_registry_and_dispatchers_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("aso")
        .expect("ASO needs a canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["bulls", "bears"],
        "ASO runtime registry output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "mode"],
        "ASO runtime parameter identity/order drifted"
    );
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(10)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    let mode = &runtime.params[1];
    assert!(matches!(mode.kind, IndicatorParamKind::Int));
    assert!(!mode.required);
    assert_eq!(mode.default, Some(ParamValueStatic::Int(0)));
    assert_eq!(mode.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert_eq!(mode.max.map(f64::to_bits), Some(2.0_f64.to_bits()));
    assert_eq!(mode.step.map(f64::to_bits), Some(1.0_f64.to_bits()));

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let cpu_dispatch = function_body(&cpu, "fn compute_aso_batch(");
    for canonical in ["bulls", "bears"] {
        assert!(
            cpu_dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "ASO CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        cpu_dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "ASO CPU dispatch retained a non-registry output alias"
    );
    for retired in ["value", "bull", "bear", "output_0", "output_1"] {
        assert!(
            !cpu_dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "ASO CPU dispatch retained retired output alias `{retired}`"
        );
    }

    let cuda = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda.rs");
    let device_dispatch = function_body(&cuda, "fn compute_aso_cuda_device(");
    for canonical in ["bulls", "bears"] {
        assert!(
            device_dispatch.contains(&format!("id == \"{canonical}\"")),
            "ASO device dispatcher is missing canonical output `{canonical}`"
        );
    }
    for retired in ["value", "bull", "bear"] {
        assert!(
            !device_dispatch.contains(&format!("id == \"{retired}\"")),
            "ASO device dispatcher retained retired output alias `{retired}`"
        );
    }

    let host = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_non_ma_generated.rs",
    );
    let host_dispatch = function_body(&host, "\"aso\" => Some((|| {");
    assert!(
        host_dispatch.contains("let fallback_outputs: &[&str] = &[]"),
        "ASO host CUDA dispatcher must resolve only canonical registry outputs"
    );
    for retired in ["output_0", "output_1"] {
        assert!(
            !host_dispatch.contains(&format!("\"{retired}\"")),
            "ASO host CUDA dispatcher retained retired output alias `{retired}`"
        );
    }
}

#[test]
fn aso_uses_one_shared_resident_two_output_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_aso_parameters(");
    for required in [
        "get_indicator",
        "ASO_OUTPUT_IDS",
        "ASO_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "ParamValueStatic::Int",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys.as_slice() != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "ASO route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const ASO_OUTPUT_IDS: [&str; 2] = [\"bulls\", \"bears\"]")
            && implementation.contains("Aso {")
            && implementation.contains("ASO_PARAMETER_KEYS"),
        "ASO needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_aso_outputs_device").count(),
        1,
        "ASO must have exactly one all-output launch site"
    );
    assert!(
        executor.contains("ASO_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(period, mode)]"),
        "both ASO matrices must stay resident through the exact parameter tuple"
    );
    for forbidden in ["CudaAso", ".aso_batch_dev", "F64Kernel::Aso,"] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered a standalone/replay ASO route `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_aso_outputs_device(");
    assert!(
        bridge.contains("aso_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_close"),
        "ASO bridge must borrow the shared resident OHLC frame and exact close-first index"
    );
    for forbidden in ["CudaAso", ".aso_batch_dev", "compute_batch_f64"] {
        assert!(
            !bridge.contains(forbidden),
            "production bridge entered a standalone/replay ASO route `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn aso_all_outputs(");
    for required in [
        "neoethos_aso_outputs_f64",
        "F64Kernel::Aso",
        "const OUTPUT_IDS: [&str; 2]",
        "ohlcv.open()",
        "ohlcv.high()",
        "ohlcv.low()",
        "ohlcv.close()",
        "d_periods",
        "d_modes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "ASO resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "ASO bulls/bears must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(open",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident ASO route retained standalone context/upload/sync `{forbidden}`"
        );
    }

    let sequential = function_body(&wrapper, "pub fn is_sequential(self) -> bool");
    assert!(
        sequential.contains("F64Kernel::Aso"),
        "ASO's running mean must remain classified as sequential"
    );
    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant
            .lines()
            .any(|line| line.trim() == "| F64Kernel::Aso"),
        "ASO period sweep was incorrectly classified as period-invariant"
    );

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/aso_kernel.cu");
    let primary = function_body(&kernel, "void neoethos_aso_batch_f64(");
    let full = function_body(&kernel, "void neoethos_aso_outputs_f64(");
    assert!(
        kernel.contains("neo_s3_aso_row_f64")
            && primary.contains("neo_s3_aso_row_f64")
            && full.contains("neo_s3_aso_row_f64"),
        "primary bulls and full bulls/bears entry points must share one exact f64 state authority"
    );
    for required in ["periods", "modes", "out_bulls", "out_bears"] {
        assert!(
            full.contains(required),
            "ASO full kernel is missing exact ABI fact `{required}`"
        );
    }

    let dispatch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let capability = function_body(&dispatch, "pub fn has_f64_resident_output_route(");
    assert!(
        capability.contains("(\"aso\", \"bears\")"),
        "ASO bears is not advertised by the exact resident capability"
    );
}

#[test]
fn autocorrelation_indicator_registry_and_cpu_dispatch_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let runtime = vector_ta::indicators::registry::get_indicator("autocorrelation_indicator")
        .expect("ACI needs a canonical runtime registry entry");
    assert!(matches!(runtime.input_kind, IndicatorInputKind::Slice));
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["filtered", "correlation"],
        "ACI runtime registry output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["length", "lag", "use_test_signal"],
        "ACI runtime parameter identity/order drifted"
    );
    for (index, expected_default) in [20_i64, 1_i64].into_iter().enumerate() {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(
            parameter.default,
            Some(ParamValueStatic::Int(expected_default))
        );
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max, None);
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    }
    let use_test_signal = &runtime.params[2];
    assert!(matches!(use_test_signal.kind, IndicatorParamKind::Bool));
    assert!(!use_test_signal.required);
    assert_eq!(use_test_signal.default, Some(ParamValueStatic::Bool(false)));
    assert_eq!(use_test_signal.min, None);
    assert_eq!(use_test_signal.max, None);
    assert_eq!(use_test_signal.step, None);
    assert_eq!(use_test_signal.enum_values, ["true", "false"]);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_autocorrelation_indicator_batch(");
    for canonical in ["filtered", "correlation"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "ACI CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "ACI CPU dispatch retained a non-registry output alias"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "ACI CPU dispatch retained retired `value` alias"
    );
}

#[test]
fn autocorrelation_indicator_uses_one_shared_resident_selected_correlation_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_autocorrelation_indicator_parameters(",
    );
    for required in [
        "get_indicator",
        "AUTOCORRELATION_INDICATOR_OUTPUT_IDS",
        "AUTOCORRELATION_INDICATOR_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Bool",
        "ParamValueStatic::Int",
        "ParamValueStatic::Bool",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"length\"]",
    ] {
        assert!(
            parameters.contains(required),
            "ACI route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const AUTOCORRELATION_INDICATOR_OUTPUT_IDS: [&str; 2] = [\"filtered\", \"correlation\"]"
        ) && implementation.contains("AutocorrelationIndicator {")
            && implementation.contains("AUTOCORRELATION_INDICATOR_PARAMETER_KEYS"),
        "ACI needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_autocorrelation_indicator_outputs_device")
            .count(),
        1,
        "ACI must have exactly one selected-correlation launch site"
    );
    assert!(
        executor.contains("AUTOCORRELATION_INDICATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(length, lag, use_test_signal)]"),
        "both ACI matrices must stay resident through the exact parameter tuple"
    );
    for forbidden in [
        "CudaAutocorrelationIndicator",
        ".batch_dev",
        "autocorrelation_indicator_batch_f64",
        "compute_batch_f64",
    ] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered a standalone/replay ACI route `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_autocorrelation_indicator_outputs_device(",
    );
    assert!(
        bridge.contains("autocorrelation_indicator_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()"),
        "ACI bridge must borrow the shared resident close series"
    );
    for forbidden in [
        "CudaAutocorrelationIndicator",
        ".batch_dev",
        "autocorrelation_indicator_batch_f64",
        "compute_batch_f64",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "production bridge entered a standalone/replay ACI route `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn autocorrelation_indicator_all_outputs(");
    for required in [
        "autocorrelation_indicator_outputs_f64",
        "F64Kernel::AutocorrelationIndicator",
        "const OUTPUT_IDS: [&str; 2]",
        "d_lengths",
        "d_lags",
        "d_use_test_signals",
        "scratch_prefix",
        "scratch_prefix_sq",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "ACI resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "ACI filtered/correlation pair must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        ".synchronize()",
        "autocorrelation_indicator_batch_f64",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident ACI route retained standalone context/upload/sync/all-lag path `{forbidden}`"
        );
    }

    let sequential = function_body(&wrapper, "pub fn is_sequential(self) -> bool");
    assert!(
        sequential.contains("F64Kernel::AutocorrelationIndicator"),
        "ACI's smoother and rolling correlation must remain sequential"
    );
    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant
            .lines()
            .any(|line| line.trim() == "| F64Kernel::AutocorrelationIndicator"),
        "ACI length sweep was incorrectly classified as period-invariant"
    );

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/autocorrelation_indicator_kernel.cu",
    );
    let primary = function_body(&kernel, "void autocorrelation_indicator_neo_batch_f64(");
    let full = function_body(&kernel, "void autocorrelation_indicator_outputs_f64(");
    assert!(
        kernel.contains("neo_aci_filter_row_f64")
            && primary.contains("periods[combo]")
            && primary.contains("neo_aci_filter_row_f64")
            && full.contains("neo_aci_filter_row_f64"),
        "primary must consume its length and both entries must share one exact f64 filter authority"
    );
    assert!(
        kernel.contains("void autocorrelation_indicator_batch_f64("),
        "ACI standalone full-correlation kernel ABI was removed"
    );
    for required in [
        "lengths",
        "lags",
        "use_test_signals",
        "out_filtered",
        "out_correlation",
        "scratch_prefix",
        "scratch_prefix_sq",
    ] {
        assert!(
            full.contains(required),
            "ACI selected-correlation kernel is missing exact ABI fact `{required}`"
        );
    }

    let dispatch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let capability = function_body(&dispatch, "pub fn has_f64_resident_output_route(");
    assert!(
        capability.contains("(\"autocorrelation_indicator\", \"correlation\")"),
        "ACI correlation is not advertised by the exact resident capability"
    );

    let scalar =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/autocorrelation_indicator.rs");
    for public_contract in [
        "pub struct AutocorrelationIndicatorOutput",
        "pub enum AutocorrelationIndicatorOutputField",
        "pub struct AutocorrelationIndicatorParams",
        "pub fn autocorrelation_indicator_output_into_slice(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "ACI scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/autocorrelation_indicator_wrapper.rs",
    );
    for public_contract in [
        "pub struct CudaAutocorrelationIndicator",
        "pub struct CudaAutocorrelationIndicatorBatchResult",
        "pub fn batch_dev(",
        "pub fn synchronize(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "ACI standalone/public ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn avsl_registry_and_cpu_dispatch_use_only_the_exact_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let runtime = vector_ta::indicators::registry::get_indicator("avsl")
        .expect("AVSL needs one canonical runtime registry entry");
    assert!(matches!(runtime.input_kind, IndicatorInputKind::Ohlcv));
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["value"],
        "AVSL runtime registry output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["fast_period", "slow_period", "multiplier"],
        "AVSL runtime parameter identity/order drifted"
    );
    for (index, expected_default) in [12_i64, 26_i64].into_iter().enumerate() {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(
            parameter.default,
            Some(ParamValueStatic::Int(expected_default))
        );
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max, None);
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    }
    let multiplier = &runtime.params[2];
    assert!(matches!(multiplier.kind, IndicatorParamKind::Float));
    assert!(!multiplier.required);
    assert_eq!(multiplier.default, Some(ParamValueStatic::Float(2.0)));
    assert_eq!(multiplier.min.map(f64::to_bits), Some(0.1_f64.to_bits()));
    assert_eq!(multiplier.max, None);
    assert_eq!(multiplier.step.map(f64::to_bits), Some(0.1_f64.to_bits()));

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_avsl_batch(");
    assert!(
        dispatch.contains("expect_value_output(\"avsl\", output_id)"),
        "AVSL CPU dispatch must accept only canonical `value`"
    );
    for forbidden in [
        "eq_ignore_ascii_case(\"values\")",
        "eq_ignore_ascii_case(\"avsl\")",
        "eq_ignore_ascii_case(\"output\")",
    ] {
        assert!(
            !dispatch.contains(forbidden),
            "AVSL CPU dispatch retained non-registry output alias `{forbidden}`"
        );
    }
}

#[test]
fn avsl_uses_one_exact_parameterized_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_avsl_parameters(");
    for required in [
        "get_indicator",
        "AVSL_OUTPUT_IDS",
        "AVSL_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"fast_period\", \"slow_period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "AVSL route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const AVSL_OUTPUT_IDS: [&str; 1] = [\"value\"]")
            && implementation
                .contains("const AVSL_REQUESTED_OUTPUT_IDS: [Option<&str>; 1] = [None]")
            && implementation.contains("Avsl {")
            && implementation.contains("AVSL_PARAMETER_KEYS"),
        "AVSL needs distinct exact runtime-request and CUDA-output identities"
    );
    assert!(
        parameters.contains("AVSL_REQUESTED_OUTPUT_IDS"),
        "AVSL parameter preflight must preserve the canonical single-output None receipt"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_avsl_outputs_device").count(),
        1,
        "AVSL must have exactly one parameterized production launch site"
    );
    assert!(
        executor.contains("AVSL_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(fast_period, slow_period, f64::from_bits(multiplier_bits))]"),
        "AVSL value must stay resident through the exact parameter tuple"
    );
    for forbidden in [
        "CudaAvsl",
        ".avsl_batch_dev",
        "avsl_batch_f32",
        "compute_batch_f64",
    ] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered a standalone/f32/replay AVSL route `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_avsl_outputs_device(");
    assert!(
        bridge.contains("avsl_production_output")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.ohlcv.low.as_view_f64()")
            && bridge.contains("self.ohlcv.volume.as_view_f64()")
            && bridge.contains("self.first_valid_avsl"),
        "AVSL bridge must borrow the one resident close/low/volume frame and exact first-valid"
    );
    for forbidden in [
        "CudaAvsl",
        ".avsl_batch_dev",
        "avsl_batch_f32",
        "compute_batch_f64",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "production bridge entered a standalone/f32/replay AVSL route `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn avsl_production_output(");
    for required in [
        "avsl_production_f64",
        "F64Kernel::Avsl",
        "const OUTPUT_IDS: [&str; 1]",
        "d_fast_periods",
        "d_slow_periods",
        "d_multipliers",
        "scratch_final_sma",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "AVSL resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "AVSL parameter rows must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(close",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(volume",
        ".synchronize()",
        "avsl_batch_f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident AVSL route retained standalone context/upload/sync/f32 path `{forbidden}`"
        );
    }

    let sequential = function_body(&wrapper, "pub fn is_sequential(self) -> bool");
    assert!(
        sequential.contains("F64Kernel::Avsl"),
        "AVSL's rolling state must remain sequential"
    );
    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant
            .lines()
            .any(|line| line.trim() == "| F64Kernel::Avsl"),
        "AVSL slow-anchor sweep was incorrectly classified as period-invariant"
    );

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/avsl_kernel.cu");
    let primary = function_body(&kernel, "void avsl_neo_batch_f64(");
    let production = function_body(&kernel, "void avsl_production_f64(");
    assert!(
        kernel.contains("neo_avsl_row_f64")
            && kernel.contains("neo_avsl_fast_from_slow")
            && kernel.contains("const int len_v = isnan(m) ? 0 : (int)m")
            && primary.contains("periods[combo]")
            && primary.contains("neo_avsl_fast_from_slow")
            && primary.contains("fc >= 0 && fl >= 0 && fv >= 0")
            && primary.contains("neo_avsl_row_f64")
            && production.contains("neo_avsl_row_f64"),
        "primary must require all three independent scans, consume slow anchor, and both entries \
         must share one exact f64 row authority"
    );
    for required in [
        "fast_periods",
        "slow_periods",
        "multipliers",
        "first_valid",
        "scratch_final_sma",
        "out_value",
    ] {
        assert!(
            production.contains(required),
            "AVSL production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/avsl.rs");
    for public_contract in [
        "pub struct AvslOutput",
        "pub struct AvslParams",
        "pub struct AvslInput<'a>",
        "pub fn avsl_into_slice(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "AVSL scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/avsl_wrapper.rs");
    for public_contract in [
        "pub struct CudaAvsl",
        "pub fn avsl_batch_dev(",
        "pub fn avsl_batch_dev_from_device_inputs(",
        "pub fn avsl_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "AVSL standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_bandpass_registry_cpu_and_scalar_use_the_exact_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("bandpass")
        .expect("Bandpass needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["bp", "bp_normalized", "signal", "trigger"],
        "Bandpass runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "bandwidth"],
        "Bandpass runtime parameter identity/order drifted"
    );
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(20)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));

    let bandwidth = &runtime.params[1];
    assert!(matches!(bandwidth.kind, IndicatorParamKind::Float));
    assert!(!bandwidth.required);
    assert_eq!(bandwidth.default, Some(ParamValueStatic::Float(0.3)));
    assert_eq!(
        bandwidth.min.map(f64::to_bits),
        Some(f64::from_bits(1).to_bits())
    );
    assert_eq!(bandwidth.max.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(bandwidth.step, None);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_bandpass_batch(");
    for canonical in ["bp", "bp_normalized", "signal", "trigger"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Bandpass CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    for retired in ["value", "normalized"] {
        let retired_dispatch = format!("eq_ignore_ascii_case(\"{retired}\")");
        assert!(
            !dispatch.contains(retired_dispatch.as_str()),
            "Bandpass CPU dispatch retained retired output alias `{retired}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/bandpass.rs");
    let prepare = function_body(&scalar, "fn bandpass_prepare<'a>(");
    assert!(
        prepare.contains("!bandwidth.is_finite()")
            && prepare.contains("bandwidth <= 0.0")
            && prepare.contains("bandwidth > 1.0"),
        "Bandpass must reject zero/non-finite/out-of-domain bandwidth directly"
    );
}

#[test]
fn classic_bandpass_uses_one_exact_four_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_bandpass_parameters(");
    for required in [
        "get_indicator",
        "BANDPASS_OUTPUT_IDS",
        "BANDPASS_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys.as_slice() != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Bandpass route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const BANDPASS_OUTPUT_IDS: [&str; 4] = [\"bp\", \"bp_normalized\", \"signal\", \"trigger\"]"
        ) && implementation.contains("Bandpass {")
            && implementation.contains("BANDPASS_PARAMETER_KEYS"),
        "Bandpass needs one exact typed four-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_bandpass_outputs_device").count(),
        1,
        "Bandpass must have exactly one four-output production launch site"
    );
    assert!(
        executor.contains("BANDPASS_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(period, f64::from_bits(bandwidth_bits))]"),
        "all four Bandpass matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_bandpass_outputs_device(");
    assert!(
        bridge.contains("bandpass_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close_finite"),
        "Bandpass bridge must borrow the resident close upload and exact finite start"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn bandpass_all_outputs(");
    for required in [
        "bandpass_production_f64",
        "F64Kernel::Bandpass",
        "const OUTPUT_IDS: [&str; 4]",
        "d_periods",
        "d_bandwidths",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Bandpass resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Bandpass outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        ".synchronize()",
        "bandpass_batch_from_hp_f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Bandpass route retained standalone/upload/sync/f32 path `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/bandpass_kernel.cu");
    let primary = function_body(&kernel, "void bandpass_neo_batch_f64(");
    let production = function_body(&kernel, "void bandpass_production_f64(");
    assert!(
        kernel.contains("neo_bandpass_row_f64")
            && primary.contains("neo_bandpass_row_f64")
            && production.contains("neo_bandpass_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "periods",
        "bandwidths",
        "first_valid",
        "out_bp",
        "out_bp_normalized",
        "out_signal",
        "out_trigger",
    ] {
        assert!(
            production.contains(required),
            "Bandpass production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/bandpass.rs");
    for public_contract in [
        "pub struct BandPassOutput",
        "pub struct BandPassParams",
        "pub struct BandPassInput<'a>",
        "pub fn bandpass_into(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Bandpass scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/bandpass_wrapper.rs");
    for public_contract in [
        "pub struct CudaBandpass",
        "pub fn bandpass_batch_dev(",
        "pub fn bandpass_batch_dev_from_device_prices(",
        "pub fn bandpass_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Bandpass standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_bollinger_bands_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("bollinger_bands")
        .expect("Bollinger Bands needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["upper", "middle", "lower"],
        "Bollinger Bands runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "devup", "devdn"],
        "Bollinger Bands runtime parameter identity/order drifted"
    );

    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(20)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    for (parameter, key) in runtime.params[1..].iter().zip(["devup", "devdn"]) {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Float(2.0)));
        assert_eq!(parameter.min, None);
        assert_eq!(parameter.max, None);
        assert_eq!(parameter.step, None);
        assert!(parameter.enum_values.is_empty());
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_bollinger_batch(");
    for canonical in ["upper", "middle", "lower"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Bollinger Bands CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Bollinger Bands CPU dispatch retained the retired `value` output alias"
    );
    for hidden_default in [
        "get_enum_param(\"bollinger_bands\", params, \"matype\", \"sma\")",
        "get_usize_param(\"bollinger_bands\", params, \"devtype\", 0)",
    ] {
        assert!(
            dispatch.contains(hidden_default),
            "Bollinger Bands CPU dispatch drifted from hidden default `{hidden_default}`"
        );
    }
}

#[test]
fn classic_bollinger_bands_uses_one_exact_three_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_bollinger_bands_parameters(");
    for required in [
        "get_indicator",
        "BOLLINGER_BANDS_OUTPUT_IDS",
        "BOLLINGER_BANDS_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys.as_slice() != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Bollinger Bands route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const BOLLINGER_BANDS_OUTPUT_IDS: [&str; 3] = [\"upper\", \"middle\", \"lower\"]"
        ) && implementation.contains("BollingerBands {")
            && implementation.contains("BOLLINGER_BANDS_PARAMETER_KEYS"),
        "Bollinger Bands needs one exact typed three-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_bollinger_bands_outputs_device")
            .count(),
        1,
        "Bollinger Bands must have exactly one three-output production launch site"
    );
    assert!(
        executor.contains("BOLLINGER_BANDS_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("let parameter_tuples")
            && executor.contains("f64::from_bits(devup_bits)")
            && executor.contains("f64::from_bits(devdn_bits)"),
        "all three Bollinger Bands matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_bollinger_bands_outputs_device(");
    assert!(
        bridge.contains("bollinger_bands_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close"),
        "Bollinger Bands bridge must borrow the resident close upload and exact start"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn bollinger_bands_all_outputs(");
    for required in [
        "bollinger_bands_production_f64",
        "F64Kernel::BollingerBands",
        "const OUTPUT_IDS: [&str; 3]",
        "d_periods",
        "d_devups",
        "d_devdns",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Bollinger Bands resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Bollinger Bands outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        ".synchronize()",
        "bollinger_bands_batch_dev",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Bollinger Bands route retained standalone/upload/sync/f32 path `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/bollinger_bands_kernel.cu");
    let primary = function_body(&kernel, "void bollinger_bands_neo_batch_f64(");
    let production = function_body(&kernel, "void bollinger_bands_production_f64(");
    assert!(
        kernel.contains("neo_bollinger_bands_row_f64")
            && primary.contains("neo_bollinger_bands_row_f64")
            && production.contains("neo_bollinger_bands_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "periods",
        "devups",
        "devdns",
        "first_valid",
        "out_upper",
        "out_middle",
        "out_lower",
    ] {
        assert!(
            production.contains(required),
            "Bollinger Bands production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/bollinger_bands.rs");
    for public_contract in [
        "pub struct BollingerBandsOutput",
        "pub struct BollingerBandsParams",
        "pub struct BollingerBandsInput<'a>",
        "pub fn bollinger_bands(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Bollinger Bands scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone =
        source("../../vendor/vector-ta-0.2.9-patched/src/cuda/bollinger_bands_wrapper.rs");
    for public_contract in [
        "pub struct CudaBollingerBands",
        "pub fn bollinger_bands_batch_dev(",
        "pub fn bollinger_bands_batch_from_device_ptr(",
        "pub fn bollinger_bands_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Bollinger Bands standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn buff_averages_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("buff_averages")
        .expect("Buff Averages needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["fast", "slow"],
        "Buff Averages runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["fast_period", "slow_period", "output"],
        "Buff Averages runtime parameter identity/order drifted"
    );
    for (parameter, key, default) in runtime.params[..2]
        .iter()
        .zip(["fast_period", "slow_period"])
        .zip([5_i64, 20_i64])
        .map(|((parameter, key), default)| (parameter, key, default))
    {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max, None);
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let output = &runtime.params[2];
    assert!(matches!(output.kind, IndicatorParamKind::EnumString));
    assert!(!output.required);
    assert_eq!(output.default, Some(ParamValueStatic::EnumString("fast")));
    assert_eq!(output.enum_values, ["fast", "slow"]);

    let cpu =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ma_batch.rs");
    let dispatch = function_body(&cpu, "pub fn ma_batch_with_kernel_and_typed_params<'a>(");
    for canonical in ["\"fast\" => out.fast", "\"slow\" => out.slow"] {
        assert!(
            dispatch.contains(canonical),
            "Buff Averages CPU dispatch is missing canonical branch `{canonical}`"
        );
    }
    for retired in ["fast_buff", "slow_buff"] {
        assert!(
            !dispatch.contains(retired),
            "Buff Averages CPU dispatch retained retired alias `{retired}`"
        );
    }

    let cpu_batch =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let period = function_body(&cpu_batch, "fn ma_period_for_combo(");
    for required in [
        "buff_averages",
        "slow_period",
        "ParamValueStatic::Int",
        "usize::try_from",
    ] {
        assert!(
            period.contains(required),
            "Buff Averages CPU production dispatch does not prove default authority `{required}`"
        );
    }
}

#[test]
fn buff_averages_uses_one_exact_two_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_buff_averages_parameters(");
    for required in [
        "get_indicator",
        "BUFF_AVERAGES_OUTPUT_IDS",
        "BUFF_AVERAGES_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::EnumString",
        "ParamValueStatic::Int",
        "ParamValueStatic::EnumString",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"fast_period\", \"slow_period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Buff Averages route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const BUFF_AVERAGES_OUTPUT_IDS: [&str; 2] = [\"fast\", \"slow\"]")
            && implementation.contains("BuffAverages {")
            && implementation.contains("BUFF_AVERAGES_PARAMETER_KEYS"),
        "Buff Averages needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_buff_averages_outputs_device")
            .count(),
        1,
        "Buff Averages must have exactly one two-output production launch site"
    );
    assert!(
        executor.contains("BUFF_AVERAGES_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("let parameter_tuples"),
        "both Buff Averages matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_buff_averages_outputs_device(");
    assert!(
        bridge.contains("buff_averages_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.ohlcv.volume.as_view_f64()")
            && bridge.contains("self.first_valid_close"),
        "Buff Averages bridge must borrow the resident close/volume uploads and exact start"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn buff_averages_all_outputs(");
    for required in [
        "buff_averages_production_f64",
        "F64Kernel::BuffAverages",
        "const OUTPUT_IDS: [&str; 2]",
        "d_fast_periods",
        "d_slow_periods",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Buff Averages resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "both Buff Averages outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        "DeviceBuffer::from_slice(volumes",
        ".synchronize()",
        "buff_averages_batch_dev",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Buff Averages route retained standalone/upload/sync/f32 path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/buff_averages_kernel.cu",
    );
    let primary = function_body(&kernel, "void buff_averages_neo_batch_f64(");
    let production = function_body(&kernel, "void buff_averages_production_f64(");
    assert!(
        kernel.contains("buff_averages_neo_row_f64")
            && primary.contains("buff_averages_neo_row_f64")
            && production.contains("buff_averages_neo_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "fast_periods",
        "slow_periods",
        "first_valid",
        "out_fast",
        "out_slow",
    ] {
        assert!(
            production.contains(required),
            "Buff Averages production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/buff_averages.rs",
    );
    for public_contract in [
        "pub struct BuffAveragesOutput",
        "pub struct BuffAveragesParams",
        "pub struct BuffAveragesInput<'a>",
        "pub fn buff_averages(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Buff Averages scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/moving_averages/buff_averages_wrapper.rs",
    );
    for public_contract in [
        "pub struct CudaBuffAverages",
        "pub fn buff_averages_batch_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Buff Averages standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn candle_strength_oscillator_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("candle_strength_oscillator")
        .expect("Candle Strength Oscillator needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        [
            "strength",
            "highs",
            "lows",
            "mid",
            "long_signal",
            "short_signal",
        ],
        "Candle Strength Oscillator runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "atr_enabled", "atr_length", "mode"],
        "Candle Strength Oscillator runtime parameter identity/order drifted"
    );

    for (parameter, key) in runtime.params[..3]
        .iter()
        .zip(["period", "atr_enabled", "atr_length"])
    {
        assert_eq!(parameter.key, key);
        assert!(!parameter.required);
        assert!(parameter.enum_values.is_empty());
    }
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert_eq!(period.default, Some(ParamValueStatic::Int(50)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));

    let atr_enabled = &runtime.params[1];
    assert!(matches!(atr_enabled.kind, IndicatorParamKind::Bool));
    assert_eq!(atr_enabled.default, Some(ParamValueStatic::Bool(false)));
    assert_eq!(atr_enabled.min, None);
    assert_eq!(atr_enabled.max, None);
    assert_eq!(atr_enabled.step, None);

    let atr_length = &runtime.params[2];
    assert!(matches!(atr_length.kind, IndicatorParamKind::Int));
    assert_eq!(atr_length.default, Some(ParamValueStatic::Int(50)));
    assert_eq!(atr_length.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(atr_length.max, None);
    assert_eq!(atr_length.step.map(f64::to_bits), Some(1.0_f64.to_bits()));

    let mode = &runtime.params[3];
    assert!(matches!(mode.kind, IndicatorParamKind::EnumString));
    assert!(!mode.required);
    assert_eq!(
        mode.default,
        Some(ParamValueStatic::EnumString("bollinger"))
    );
    assert_eq!(mode.min, None);
    assert_eq!(mode.max, None);
    assert_eq!(mode.step, None);
    assert_eq!(mode.enum_values, ["bollinger", "donchian"]);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_candle_strength_oscillator_batch(");
    for canonical in [
        "strength",
        "highs",
        "lows",
        "mid",
        "long_signal",
        "short_signal",
    ] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Candle Strength Oscillator CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Candle Strength Oscillator CPU dispatch retained retired `value` output alias"
    );
    for canonical_mode in ["bollinger", "donchian"] {
        assert!(
            dispatch.contains(&format!("mode.eq_ignore_ascii_case(\"{canonical_mode}\")")),
            "Candle Strength Oscillator CPU dispatch is missing canonical mode `{canonical_mode}`"
        );
    }
    for retired_mode in ["bb", "dc"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired_mode}\")")),
            "Candle Strength Oscillator CPU dispatch retained retired mode `{retired_mode}`"
        );
    }
}

#[test]
fn candle_strength_oscillator_uses_one_exact_six_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_candle_strength_oscillator_parameters(",
    );
    for required in [
        "get_indicator",
        "CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS",
        "CANDLE_STRENGTH_OSCILLATOR_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Bool",
        "IndicatorParamKind::EnumString",
        "ParamValueStatic::Int",
        "ParamValueStatic::Bool",
        "ParamValueStatic::EnumString",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Candle Strength Oscillator route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS: [&str; 6] = [")
            && implementation.contains("CandleStrengthOscillator {")
            && implementation.contains("ClassicCandleStrengthMode"),
        "Candle Strength Oscillator needs one exact typed six-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_candle_strength_oscillator_outputs_device")
            .count(),
        1,
        "Candle Strength Oscillator must have exactly one six-output production launch site"
    );
    assert!(
        executor.contains("CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("CandleStrengthOscillatorParams"),
        "all six Candle Strength Oscillator matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_candle_strength_oscillator_outputs_device(",
    );
    assert!(
        bridge.contains("candle_strength_oscillator_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.max_consecutive_finite_ohlc"),
        "Candle Strength Oscillator bridge must borrow resident OHLC and exact run metadata"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn candle_strength_oscillator_all_outputs(");
    for required in [
        "candle_strength_oscillator_batch_f64",
        "F64Kernel::CandleStrengthOscillator",
        "const OUTPUT_IDS: [&str; 6]",
        "d_periods",
        "d_atr_lengths",
        "d_full",
        "d_half",
        "d_sqrt",
        "d_level",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Candle Strength Oscillator resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Candle Strength Oscillator outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(open",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
        "CudaCandleStrengthOscillator",
        "batch_dev(",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Candle Strength Oscillator route retained standalone/upload/sync path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/candle_strength_oscillator_kernel.cu",
    );
    let primary = function_body(&kernel, "void candle_strength_oscillator_neo_batch_f64(");
    let production = function_body(&kernel, "void candle_strength_oscillator_batch_f64(");
    assert!(
        kernel.contains("candle_strength_oscillator_row_f64")
            && primary.contains("candle_strength_oscillator_row_f64")
            && production.contains("candle_strength_oscillator_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "periods",
        "atr_lengths",
        "atr_enabled",
        "mode",
        "full_scratch",
        "half_scratch",
        "sqrt_scratch",
        "level_scratch",
        "out_strength",
        "out_highs",
        "out_lows",
        "out_mid",
        "out_long_signal",
        "out_short_signal",
    ] {
        assert!(
            production.contains(required),
            "Candle Strength Oscillator production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/candle_strength_oscillator.rs");
    for public_contract in [
        "pub struct CandleStrengthOscillatorOutput",
        "pub struct CandleStrengthOscillatorParams",
        "pub struct CandleStrengthOscillatorInput<'a>",
        "pub fn candle_strength_oscillator_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Candle Strength Oscillator scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/candle_strength_oscillator_wrapper.rs",
    );
    for public_contract in [
        "pub struct CudaCandleStrengthOscillator",
        "pub fn batch_dev(",
        "pub fn synchronize(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Candle Strength Oscillator standalone/public API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn chandelier_exit_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("chandelier_exit")
        .expect("Chandelier Exit needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["long_stop", "short_stop"],
        "Chandelier Exit runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "mult", "use_close"],
        "Chandelier Exit runtime parameter identity/order drifted"
    );

    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(22)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let mult = &runtime.params[1];
    assert!(matches!(mult.kind, IndicatorParamKind::Float));
    assert!(!mult.required);
    assert_eq!(mult.default, Some(ParamValueStatic::Float(3.0)));
    assert_eq!(mult.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert_eq!(mult.max, None);
    assert_eq!(mult.step, None);
    assert!(mult.enum_values.is_empty());

    let use_close = &runtime.params[2];
    assert!(matches!(use_close.kind, IndicatorParamKind::Bool));
    assert!(!use_close.required);
    assert_eq!(use_close.default, Some(ParamValueStatic::Bool(true)));
    assert_eq!(use_close.min, None);
    assert_eq!(use_close.max, None);
    assert_eq!(use_close.step, None);
    assert_eq!(use_close.enum_values, ["true", "false"]);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_chandelier_exit_batch(");
    for canonical in ["long_stop", "short_stop"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Chandelier Exit CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Chandelier Exit CPU dispatch retained retired `value` output alias"
    );
}

#[test]
fn chandelier_exit_uses_one_exact_pair_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_chandelier_exit_parameters(");
    for required in [
        "get_indicator",
        "CHANDELIER_EXIT_OUTPUT_IDS",
        "CHANDELIER_EXIT_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "IndicatorParamKind::Bool",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ParamValueStatic::Bool",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Chandelier Exit route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const CHANDELIER_EXIT_OUTPUT_IDS: [&str; 2] =")
            && implementation.contains("ChandelierExit {")
            && implementation.contains("mult_bits"),
        "Chandelier Exit needs one exact typed two-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_chandelier_exit_outputs_device")
            .count(),
        1,
        "Chandelier Exit must have exactly one pair production launch site"
    );
    assert!(
        executor.contains("CHANDELIER_EXIT_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("ChandelierExitParams"),
        "both Chandelier Exit matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_chandelier_exit_outputs_device(");
    assert!(
        bridge.contains("chandelier_exit_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_close")
            && bridge.contains("self.first_valid_hlc"),
        "Chandelier Exit bridge must borrow resident HLC and both exact first-valid facts"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn chandelier_exit_all_outputs(");
    for required in [
        "chandelier_exit_outputs_f64",
        "F64Kernel::ChandelierExit",
        "const OUTPUT_IDS: [&str; 2]",
        "d_periods",
        "d_mults",
        "d_max_deque",
        "d_min_deque",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Chandelier Exit resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "both Chandelier Exit outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
        "CudaChandelierExit",
        "chandelier_exit_batch_dev(",
        "chandelier_exit_batch_from_device_dev(",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Chandelier Exit route retained standalone/upload/sync path `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/chandelier_exit_kernel.cu");
    let primary = function_body(&kernel, "void neoethos_chandelier_exit_batch_f64(");
    let production = function_body(&kernel, "void chandelier_exit_outputs_f64(");
    assert!(
        kernel.contains("chandelier_exit_row_f64")
            && primary.contains("chandelier_exit_row_f64")
            && production.contains("chandelier_exit_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "periods",
        "mults",
        "use_close",
        "max_deque_scratch",
        "min_deque_scratch",
        "out_long_stop",
        "out_short_stop",
    ] {
        assert!(
            production.contains(required),
            "Chandelier Exit production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/chandelier_exit.rs");
    for public_contract in [
        "pub struct ChandelierExitOutput",
        "pub struct ChandelierExitParams",
        "pub struct ChandelierExitInput<'a>",
        "pub fn chandelier_exit_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Chandelier Exit scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone =
        source("../../vendor/vector-ta-0.2.9-patched/src/cuda/chandelier_exit_wrapper.rs");
    for public_contract in [
        "pub struct CudaChandelierExit",
        "pub fn stream(",
        "pub fn chandelier_exit_batch_dev(",
        "pub fn chandelier_exit_batch_from_device_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Chandelier Exit standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn cksp_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("cksp")
        .expect("CKSP needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["long_values", "short_values"],
        "CKSP runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["p", "x", "q"],
        "CKSP runtime parameter identity/order drifted"
    );

    let p = &runtime.params[0];
    assert!(matches!(p.kind, IndicatorParamKind::Int));
    assert!(!p.required);
    assert_eq!(p.default, Some(ParamValueStatic::Int(10)));
    assert_eq!(p.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(p.max, None);
    assert_eq!(p.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(p.enum_values.is_empty());

    let x = &runtime.params[1];
    assert!(matches!(x.kind, IndicatorParamKind::Float));
    assert!(!x.required);
    assert_eq!(x.default, Some(ParamValueStatic::Float(1.0)));
    assert_eq!(x.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert_eq!(x.max, None);
    assert_eq!(x.step, None);
    assert!(x.enum_values.is_empty());

    let q = &runtime.params[2];
    assert!(matches!(q.kind, IndicatorParamKind::Int));
    assert!(!q.required);
    assert_eq!(q.default, Some(ParamValueStatic::Int(9)));
    assert_eq!(q.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(q.max, None);
    assert_eq!(q.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(q.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_cksp_batch(");
    for canonical in ["long_values", "short_values"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "CKSP CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    for retired in ["long", "short", "value"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "CKSP CPU dispatch retained retired output alias `{retired}`"
        );
    }
}

#[test]
fn cksp_uses_one_exact_default_pair_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_cksp_parameters(");
    for required in [
        "get_indicator",
        "CKSP_OUTPUT_IDS",
        "CKSP_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Int",
        "ParamValueStatic::Float",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "default-only",
    ] {
        assert!(
            parameters.contains(required),
            "CKSP route does not prove exact default-only fact `{required}`"
        );
    }
    assert!(
        implementation.contains("const CKSP_OUTPUT_IDS: [&str; 2] =")
            && implementation.contains("Cksp {")
            && implementation.contains("x_bits"),
        "CKSP needs one exact typed two-output default contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_cksp_outputs_device").count(),
        1,
        "CKSP must have exactly one pair production launch site"
    );
    assert!(
        executor.contains("CKSP_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("CkspParams"),
        "both CKSP matrices must stay resident through the exact default tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_cksp_outputs_device(");
    assert!(
        bridge.contains("cksp_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_close"),
        "CKSP bridge must borrow resident HLC and the exact close first-valid fact"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn cksp_all_outputs(");
    for required in [
        "cksp_outputs_f64",
        "F64Kernel::Cksp",
        "const OUTPUT_IDS: [&str; 2]",
        "d_p_values",
        "d_x_values",
        "d_q_values",
        "d_h_index",
        "d_l_index",
        "d_long_index",
        "d_short_index",
        "d_long_value",
        "d_short_value",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "CKSP resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "both CKSP outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
        "CudaCksp",
        "cksp_batch_dev(",
        "cksp_batch_dev_from_device_inputs(",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident CKSP route retained standalone/upload/sync path `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/cksp_kernel.cu");
    let primary = function_body(&kernel, "void cksp_neo_batch_f64(");
    let production = function_body(&kernel, "void cksp_outputs_f64(");
    assert!(
        kernel.contains("cksp_row_f64")
            && primary.contains("cksp_row_f64")
            && production.contains("cksp_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "p_values",
        "x_values",
        "q_values",
        "h_index_scratch",
        "l_index_scratch",
        "long_index_scratch",
        "short_index_scratch",
        "long_value_scratch",
        "short_value_scratch",
        "out_long_values",
        "out_short_values",
    ] {
        assert!(
            production.contains(required),
            "CKSP production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/cksp.rs");
    for public_contract in [
        "pub struct CkspOutput",
        "pub struct CkspParams",
        "pub struct CkspInput<'a>",
        "pub fn cksp_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "CKSP scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/cksp_wrapper.rs");
    for public_contract in [
        "pub struct CudaCksp",
        "pub fn cksp_batch_dev(",
        "pub fn cksp_batch_dev_from_device_inputs(",
        "pub fn cksp_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "CKSP standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn coppock_registry_and_cpu_use_only_the_exact_canonical_contract() {
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let params = registry
        .split_once("const PARAM_COPPOCK:")
        .expect("missing canonical Coppock parameter table")
        .1
        .split_once("];\n")
        .expect("unterminated Coppock parameter table")
        .0;
    let mut cursor = 0usize;
    for (key, default) in [
        ("short_roc_period", "11"),
        ("long_roc_period", "14"),
        ("ma_period", "10"),
    ] {
        let relative = params[cursor..]
            .find(&format!("key: \"{key}\""))
            .unwrap_or_else(|| panic!("Coppock registry lost ordered key `{key}`"));
        cursor += relative;
        let declaration = &params[cursor..params.len().min(cursor + 420)];
        for required in [
            "kind: IndicatorParamKind::Int",
            "required: false",
            &format!("default: Some(ParamValueStatic::Int({default}))"),
            "min: Some(1.0)",
            "max: None",
            "step: Some(1.0)",
            "enum_values: EMPTY_ENUM_VALUES",
        ] {
            assert!(
                declaration.contains(required),
                "Coppock `{key}` lost canonical registry fact `{required}`"
            );
        }
    }
    let seed = registry
        .split_once("id: \"coppock\"")
        .expect("Coppock registry seed disappeared")
        .1;
    assert!(
        seed.contains("outputs: OUTPUTS_VALUE_F64") && seed.contains("params: PARAM_COPPOCK"),
        "Coppock must retain exactly the sole canonical value output and parameter table"
    );

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_coppock_batch(");
    for required in [
        "expect_value_output",
        "short_roc_period",
        "long_roc_period",
        "ma_period",
        "Some(\"wma\".to_string())",
        "coppock_with_kernel",
    ] {
        assert!(
            dispatch.contains(required),
            "Coppock CPU dispatch lost exact canonical fact `{required}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"values\")"),
        "Coppock CPU dispatch retained the retired unversioned `values` alias"
    );

    let output_normalization = function_body(&cpu, "fn normalize_output_token(");
    assert!(
        output_normalization.contains("is_ascii_alphanumeric")
            && output_normalization.contains("to_ascii_lowercase"),
        "authoritative punctuation/case normalization was removed"
    );
    assert!(
        !output_normalization.contains("normalized == \"values\"")
            && !output_normalization.contains("\"value\".to_string()"),
        "global CPU output normalization still aliases canonical `values` and `value` identities"
    );
}

#[test]
fn coppock_uses_one_exact_dynamic_tuple_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_coppock_parameters(");
    for required in [
        "COPPOCK_OUTPUT_ID",
        "COPPOCK_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "ParamValueStatic::Int(11)",
        "ParamValueStatic::Int(14)",
        "ParamValueStatic::Int(10)",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "short_roc_period",
        "long_roc_period",
        "ma_period",
    ] {
        assert!(
            parameters.contains(required),
            "Coppock resolver is missing exact tuple/schema fact `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_coppock_output_device").count(),
        1,
        "the typed Coppock route must enter one resident engine bridge"
    );
    for forbidden in [
        "CudaCoppock",
        ".coppock_batch_dev",
        "coppock_batch_f32",
        "compute_cpu_batch",
    ] {
        assert!(
            !executor.contains(forbidden),
            "production executor entered a standalone/f32/CPU Coppock path `{forbidden}`"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_coppock_output_device(");
    assert!(
        bridge.contains("coppock_production_output")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close"),
        "Coppock bridge must borrow the one resident close upload and exact first-valid fact"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn coppock_production_output(");
    for required in [
        "coppock_production_f64",
        "F64Kernel::Coppock",
        "const OUTPUT_IDS: [&str; 1]",
        "d_short_roc_periods",
        "d_long_roc_periods",
        "d_ma_periods",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Coppock resident wrapper is missing shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Coppock tuples must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
        "CudaCoppock",
        "coppock_batch_dev(",
        "coppock_batch_f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Coppock route retained standalone/upload/sync/f32 path `{forbidden}`"
        );
    }

    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant
            .lines()
            .any(|line| line.trim() == "| F64Kernel::Coppock"),
        "Coppock ratio sweep was incorrectly left period-invariant"
    );

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/coppock_kernel.cu");
    let primary = function_body(&kernel, "void coppock_neo_batch_f64(");
    let production = function_body(&kernel, "void coppock_production_f64(");
    assert!(
        kernel.contains("coppock_row_f64")
            && kernel.contains("coppock_scaled_window_f64")
            && primary.contains("periods[combo]")
            && primary.contains("coppock_scaled_window_f64")
            && primary.contains("coppock_row_f64")
            && production.contains("coppock_row_f64"),
        "primary must consume its ratio anchor and both entries must share exact row authority"
    );
    for required in [
        "short_roc_periods",
        "long_roc_periods",
        "ma_periods",
        "first_valid",
        "out_value",
    ] {
        assert!(
            production.contains(required),
            "Coppock production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/coppock.rs");
    for public_contract in [
        "pub struct CoppockOutput",
        "pub struct CoppockParams",
        "pub struct CoppockInput<'a>",
        "pub fn coppock_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Coppock scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/coppock_wrapper.rs");
    for public_contract in [
        "pub struct CudaCoppock",
        "pub fn coppock_batch_dev(",
        "pub fn coppock_batch_dev_from_device_prices(",
        "pub fn coppock_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Coppock standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_correlation_cycle_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("correlation_cycle")
        .expect("Correlation Cycle needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["real", "imag", "angle", "state"],
        "Correlation Cycle runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "threshold"],
        "Correlation Cycle runtime parameter identity/order drifted"
    );

    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(20)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let threshold = &runtime.params[1];
    assert!(matches!(threshold.kind, IndicatorParamKind::Float));
    assert!(!threshold.required);
    assert_eq!(threshold.default, Some(ParamValueStatic::Float(9.0)));
    assert_eq!(threshold.min, None);
    assert_eq!(threshold.max, None);
    assert_eq!(threshold.step, None);
    assert!(threshold.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_correlation_cycle_batch(");
    for canonical in ["real", "imag", "angle", "state"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Correlation Cycle CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Correlation Cycle CPU dispatch retained the retired `value` alias"
    );
}

#[test]
fn classic_correlation_cycle_uses_one_exact_four_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_correlation_cycle_parameters(");
    for required in [
        "get_indicator",
        "CORRELATION_CYCLE_OUTPUT_IDS",
        "CORRELATION_CYCLE_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Int(20)",
        "9.0_f64.to_bits()",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "override_keys != [\"period\"]",
    ] {
        assert!(
            parameters.contains(required),
            "Correlation Cycle route does not prove exact parameter fact `{required}`"
        );
    }
    assert!(
        implementation.contains(
            "const CORRELATION_CYCLE_OUTPUT_IDS: [&str; 4] = [\"real\", \"imag\", \"angle\", \"state\"]"
        ) && implementation.contains("CorrelationCycle {")
            && implementation.contains("CORRELATION_CYCLE_PARAMETER_KEYS"),
        "Correlation Cycle needs one exact typed four-output contract"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_correlation_cycle_outputs_device")
            .count(),
        1,
        "Correlation Cycle must have exactly one four-output production launch site"
    );
    assert!(
        executor.contains("CORRELATION_CYCLE_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(period, f64::from_bits(threshold_bits))]"),
        "all four Correlation Cycle matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_correlation_cycle_outputs_device(");
    assert!(
        bridge.contains("correlation_cycle_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close"),
        "Correlation Cycle bridge must borrow the resident close upload and exact non-NaN start"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn correlation_cycle_all_outputs(");
    for required in [
        "correlation_cycle_outputs_f64",
        "F64Kernel::CorrelationCycle",
        "const OUTPUT_IDS: [&str; 4]",
        "d_periods",
        "d_thresholds",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Correlation Cycle resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Correlation Cycle outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        ".synchronize()",
        "CudaCorrelationCycle",
        "correlation_cycle_batch_dev(",
        "correlation_cycle_batch_f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Correlation Cycle route retained standalone/upload/sync/f32 path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
    );
    let primary = function_body(&kernel, "void neoethos_correlation_cycle_batch_f64(");
    let production = function_body(&kernel, "void correlation_cycle_outputs_f64(");
    assert!(
        kernel.contains("neo_s3_correlation_cycle_row_f64")
            && primary.contains("neo_s3_correlation_cycle_row_f64")
            && production.contains("neo_s3_correlation_cycle_row_f64"),
        "primary and production entries must share one exact f64 row authority"
    );
    for required in [
        "periods",
        "thresholds",
        "first_valid",
        "out_real",
        "out_imag",
        "out_angle",
        "out_state",
    ] {
        assert!(
            production.contains(required),
            "Correlation Cycle production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/correlation_cycle.rs");
    for public_contract in [
        "pub struct CorrelationCycleOutput",
        "pub struct CorrelationCycleParams",
        "pub struct CorrelationCycleInput<'a>",
        "pub fn correlation_cycle_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Correlation Cycle scalar/public API drifted at `{public_contract}`"
        );
    }
    let standalone = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/moving_averages/correlation_cycle_wrapper.rs",
    );
    for public_contract in [
        "pub struct CudaCorrelationCycle",
        "pub fn correlation_cycle_batch_dev(",
        "pub fn correlation_cycle_batch_dev_from_device_prices(",
        "pub fn correlation_cycle_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Correlation Cycle standalone/public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_correlation_cycle_uses_one_reviewed_deterministic_f64_math_authority() {
    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/correlation_cycle.rs");
    for required in [
        "const CORRELATION_CYCLE_HALF_PI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18)",
        "const CORRELATION_CYCLE_TWO_PI: f64 = f64::from_bits(0x4019_21fb_5444_2d18)",
        "const CORRELATION_CYCLE_RADIANS_TO_DEGREES: f64 =",
        "f64::from_bits(0x404c_a5dc_1a63_c1f8)",
        "f64::from_bits(0x3fd5_5555_5555_550d)",
        "fn correlation_cycle_ms_k_sin(",
        "fn correlation_cycle_ms_k_cos(",
        "fn correlation_cycle_reduce_pio2(",
        "fn correlation_cycle_deterministic_sin_cos(",
        "fn correlation_cycle_deterministic_weight(",
        "fn correlation_cycle_deterministic_atan(",
        "fn correlation_cycle_deterministic_angle(",
    ] {
        assert!(
            scalar.contains(required),
            "Correlation Cycle CPU authority is missing deterministic math fact `{required}`"
        );
    }
    for forbidden in [".sin_cos()", ".atan()", ".to_degrees()", "f64::asin("] {
        assert!(
            !scalar.contains(forbidden),
            "Correlation Cycle CPU retained platform transcendental `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
    );
    let f64_lane = kernel
        .split("// S3 f64 LANE")
        .nth(1)
        .expect("Correlation Cycle CUDA source lost its f64 lane marker");
    for required in [
        "0x1.921fb54442d18p+0",
        "0x1.921fb54442d18p+2",
        "0x1.ca5dc1a63c1f8p+5",
        "0x1.555555555550dp-2",
        "neo_s3_cc_ms_k_sin",
        "neo_s3_cc_ms_k_cos",
        "neo_s3_cc_reduce_pio2",
        "neo_s3_cc_deterministic_sin_cos",
        "neo_s3_cc_deterministic_weight",
        "neo_s3_cc_deterministic_atan",
        "neo_s3_cc_deterministic_angle",
    ] {
        assert!(
            f64_lane.contains(required),
            "Correlation Cycle CUDA authority is missing deterministic math fact `{required}`"
        );
    }
    for forbidden in [
        "return cos(",
        "return -sin(",
        " = atan(",
        "asin(1.0)",
        "to_degrees(",
    ] {
        assert!(
            !f64_lane.contains(forbidden),
            "Correlation Cycle f64 CUDA lane retained libdevice/platform trig `{forbidden}`"
        );
    }

    let row = function_body(&kernel, "void neo_s3_correlation_cycle_row_f64(");
    assert!(
        row.contains("neo_s3_cc_deterministic_angle")
            && row.contains("neo_s3_cc_deterministic_weight"),
        "the shared f64 row must consume the reviewed deterministic authority directly"
    );

    let cpu_weight = function_body(&scalar, "fn correlation_cycle_deterministic_weight(");
    let cuda_weight = function_body(&kernel, "void neo_s3_cc_deterministic_weight(");
    assert!(
        cpu_weight.contains("j & !3usize")
            && cpu_weight.contains("angle += w")
            && cuda_weight.contains("j & ~3")
            && cuda_weight.contains("angle += w"),
        "CPU and CUDA must construct every four-wide weight angle in the same order"
    );

    let cpu_reduce = function_body(&scalar, "fn correlation_cycle_reduce_pio2(");
    let cuda_reduce = function_body(&kernel, "int neo_s3_cc_reduce_pio2(");
    for (cpu_fact, cuda_fact) in [
        ("0x3fe4_5f30_6dc9_c883", "0x1.45f306dc9c883p-1"),
        ("0x4338_0000_0000_0000", "0x1.8p+52"),
        ("ex - ey > 16", "ex - ey > 16"),
        ("ex - ey > 49", "ex - ey > 49"),
        ("(r - y0) - w", "(r - y0) - w"),
    ] {
        assert!(
            cpu_reduce.contains(cpu_fact) && cuda_reduce.contains(cuda_fact),
            "CPU/CUDA pi/2 reduction drifted at `{cpu_fact}` / `{cuda_fact}`"
        );
    }

    let cpu_atan = function_body(&scalar, "fn correlation_cycle_deterministic_atan(");
    let cuda_atan = function_body(&kernel, "double neo_s3_cc_deterministic_atan(");
    for (cpu_fact, cuda_fact) in [
        ("0x3fdc_0000", "0x3fdc0000U"),
        ("0x3ff3_0000", "0x3ff30000U"),
        ("0x3fe6_0000", "0x3fe60000U"),
        ("0x4003_8000", "0x40038000U"),
        ("return x - x * (s1 + s2)", "return x - x * (s1 + s2)"),
        (
            "atan_hi[index] - ((x * (s1 + s2) - atan_lo[index]) - x)",
            "- ((x * (s1 + s2) - atan_lo[reduction]) - x)",
        ),
    ] {
        assert!(
            cpu_atan.contains(cpu_fact) && cuda_atan.contains(cuda_fact),
            "CPU/CUDA atan reduction drifted at `{cpu_fact}` / `{cuda_fact}`"
        );
    }
}

#[test]
fn classic_cvi_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("cvi")
        .expect("CVI needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["value"],
        "CVI runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period"],
        "CVI runtime parameter identity/order drifted"
    );

    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(10)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(period.max, None);
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_cvi_batch(");
    assert!(
        dispatch.contains("expect_value_output(\"cvi\", output_id)?"),
        "CVI CPU dispatch must require the sole canonical `value` output"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"values\")"),
        "CVI CPU dispatch retained the retired struct-field `values` alias"
    );
}

#[test]
fn classic_cvi_uses_the_bounded_exact_resident_f64_route() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const CVI_ID: &str = \"cvi\"")
            && implementation.contains("CVI_PARAMETER_KEYS")
            && implementation.contains("resolve_cvi_parameters("),
        "CVI needs an exact typed registry/parameter contract before launch"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let max_period = function_body(&wrapper, "pub fn max_period(self) -> Option<usize>");
    assert!(
        max_period.contains("F64Kernel::Cvi => Some(CVI_MAX_PERIOD)"),
        "CVI's compiled 512-entry ring must be a named fail-closed host bound"
    );
    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant.contains("F64Kernel::Cvi"),
        "CVI must consume every admitted period instead of replaying its default"
    );

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/cvi_kernel.cu");
    let production = function_body(&kernel, "void cvi_batch_f64(");
    for required in [
        "const double alpha = 2.0 / (static_cast<double>(period) + 1.0)",
        "val += (range - val) * alpha",
        "100.0 * (val - old) / old",
        "CVI_MAX_PERIOD_F64",
    ] {
        assert!(
            production.contains(required),
            "CVI f64 route is missing scalar-order fact `{required}`"
        );
    }
    for forbidden in ["__fma", "__fmaf", "float"] {
        assert!(
            !production.contains(forbidden),
            "CVI f64 production route retained narrowed/fused operation `{forbidden}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_cvi_output_device").count(),
        1,
        "CVI production must enter its exact retained-parameter bridge once"
    );
    assert!(
        !executor.contains("compute_primary_device(CVI_ID"),
        "CVI production retained the generic sweep's per-chunk synchronization"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_cvi_output_device(");
    for required in [
        "CudaDeviceHighLowF64Ref::new(",
        "self.ohlcv.high.as_view_f64()",
        "self.ohlcv.low.as_view_f64()",
        "self.first_valid_high_low",
        ".cvi_production_output(",
    ] {
        assert!(
            bridge.contains(required),
            "CVI's shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaCvi",
        "CudaRuntime::new",
        ".synchronize()",
        "upload",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "CVI bridge would cross a forbidden host/standalone path `{forbidden}`"
        );
    }

    let resident = function_body(&wrapper, "pub fn cvi_production_output(");
    for required in [
        "CVI_MAX_PERIOD",
        "CudaF64IndicatorError::PeriodTooLarge",
        "checked_mul(",
        "mem_get_info()",
        "self.module_for(F64Kernel::Cvi)",
        "get_function(ENTRY_POINT)",
        "CudaDeviceMatrixF64::from_buffer(",
        "_parameter_i32: vec![d_periods]",
    ] {
        assert!(
            resident.contains(required),
            "CVI's retained-parameter resident launch is missing `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "CVI must launch the whole admitted period matrix exactly once"
    );
    for forbidden in [
        ".synchronize()",
        "Context::new",
        "CudaRuntime::new",
        "HostF64",
        "compute_cpu",
        "CudaCvi",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "CVI resident launch retained forbidden production path `{forbidden}`"
        );
    }
}

#[test]
fn classic_cyberpunk_value_trend_analyzer_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("cyberpunk_value_trend_analyzer")
        .expect("Cyberpunk Value Trend Analyzer needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        [
            "value_trend",
            "value_trend_lag",
            "deviation_index",
            "overbought_signal",
            "buy_signal",
            "sell_signal",
        ],
        "Cyberpunk Value Trend Analyzer runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["entry_level", "exit_level"],
        "Cyberpunk Value Trend Analyzer runtime parameter identity/order drifted"
    );
    for (parameter, default) in runtime.params.iter().zip([30_i64, 75]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max.map(f64::to_bits), Some(100.0_f64.to_bits()));
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_cyberpunk_value_trend_analyzer_batch(");
    for canonical in [
        "value_trend",
        "value_trend_lag",
        "deviation_index",
        "overbought_signal",
        "buy_signal",
        "sell_signal",
    ] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Cyberpunk Value Trend Analyzer CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    for retired in ["value", "lag", "overbought"] {
        let retired_dispatch = format!("eq_ignore_ascii_case(\"{retired}\")");
        assert!(
            !dispatch.contains(retired_dispatch.as_str()),
            "Cyberpunk Value Trend Analyzer CPU dispatch retained retired alias `{retired}`"
        );
    }
}

#[test]
fn classic_cyberpunk_value_trend_analyzer_uses_one_exact_six_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_cyberpunk_value_trend_analyzer_parameters(",
    );
    for required in [
        "get_indicator",
        "CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS",
        "CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS",
        "IndicatorParamKind::Int",
        "ParamValueStatic::Int(30)",
        "ParamValueStatic::Int(75)",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "has no canonical period sweep",
    ] {
        assert!(
            parameters.contains(required),
            "Cyberpunk Value Trend Analyzer route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_cyberpunk_value_trend_analyzer_outputs_device")
            .count(),
        1,
        "Cyberpunk Value Trend Analyzer must have exactly one six-output production launch site"
    );
    assert!(
        executor.contains("CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[(entry_level, exit_level)]"),
        "all six Cyberpunk Value Trend Analyzer matrices must stay resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_cyberpunk_value_trend_analyzer_outputs_device(",
    );
    assert!(
        bridge.contains("cyberpunk_value_trend_analyzer_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.max_consecutive_finite_ohlc"),
        "Cyberpunk Value Trend Analyzer bridge must borrow the resident OHLC upload and exact valid-run receipt"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn cyberpunk_value_trend_analyzer_all_outputs(",
    );
    for required in [
        "cyberpunk_value_trend_analyzer_batch_f64",
        "F64Kernel::CyberpunkValueTrendAnalyzer",
        "const OUTPUT_IDS: [&str; 6]",
        "d_entry_levels",
        "d_exit_levels",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Cyberpunk Value Trend Analyzer resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Cyberpunk Value Trend Analyzer outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(open",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Cyberpunk Value Trend Analyzer route retained forbidden path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/cyberpunk_value_trend_analyzer_kernel.cu",
    );
    let production = function_body(&kernel, "void cyberpunk_value_trend_analyzer_batch_f64(");
    for required in [
        "entry_levels",
        "exit_levels",
        "out_value_trend",
        "out_value_trend_lag",
        "out_deviation_index",
        "out_overbought_signal",
        "out_buy_signal",
        "out_sell_signal",
    ] {
        assert!(
            production.contains(required),
            "Cyberpunk Value Trend Analyzer production kernel is missing exact ABI fact `{required}`"
        );
    }

    let scalar = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/cyberpunk_value_trend_analyzer.rs",
    );
    for public_contract in [
        "pub struct CyberpunkValueTrendAnalyzerOutput",
        "pub struct CyberpunkValueTrendAnalyzerParams",
        "pub struct CyberpunkValueTrendAnalyzerInput<'a>",
        "pub fn cyberpunk_value_trend_analyzer_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Cyberpunk Value Trend Analyzer scalar/public API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_cycle_channel_oscillator_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("cycle_channel_oscillator")
        .expect("Cycle Channel Oscillator needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["fast", "slow"],
        "Cycle Channel Oscillator runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        [
            "source",
            "short_cycle_length",
            "medium_cycle_length",
            "short_multiplier",
            "medium_multiplier",
        ],
        "Cycle Channel Oscillator runtime parameter identity/order drifted"
    );

    let source_parameter = &runtime.params[0];
    assert!(matches!(
        source_parameter.kind,
        IndicatorParamKind::EnumString
    ));
    assert!(!source_parameter.required);
    assert_eq!(
        source_parameter.default,
        Some(ParamValueStatic::EnumString("close"))
    );
    assert!(source_parameter.min.is_none());
    assert!(source_parameter.max.is_none());
    assert!(source_parameter.step.is_none());
    assert_eq!(
        source_parameter.enum_values,
        [
            "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"
        ]
    );

    for (parameter, default) in runtime.params[1..3].iter().zip([10_i64, 30]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(2.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    for (parameter, default) in runtime.params[3..].iter().zip([1.0_f64, 3.0]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Float(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(0.1_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_cycle_channel_oscillator_batch(");
    for canonical in ["fast", "slow"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Cycle Channel Oscillator CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Cycle Channel Oscillator CPU dispatch retained retired `value` alias"
    );
}

#[test]
fn classic_cycle_channel_oscillator_uses_one_exact_two_output_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_cycle_channel_oscillator_parameters(",
    );
    for required in [
        "get_indicator",
        "CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS",
        "CYCLE_CHANNEL_OSCILLATOR_PARAMETER_KEYS",
        "ParamValueStatic::EnumString(\"close\")",
        "let default_windows = [10usize, 30usize]",
        "Some(ParamValueStatic::Int(default as i64))",
        "expected_multiplier_defaults = [1.0_f64, 3.0_f64]",
        "Some(ParamValueStatic::Float(value))",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "sweep_point_exclusion",
    ] {
        assert!(
            parameters.contains(required),
            "Cycle Channel Oscillator route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_cycle_channel_oscillator_outputs_device")
            .count(),
        1,
        "Cycle Channel Oscillator must have exactly one two-output production launch site"
    );
    assert!(
        executor.contains("CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("short_multiplier_bits")
            && executor.contains("medium_multiplier_bits"),
        "both Cycle Channel Oscillator matrices must remain resident through each exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_cycle_channel_oscillator_outputs_device(",
    );
    assert!(
        bridge.contains("cycle_channel_oscillator_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_hlc_finite"),
        "Cycle Channel Oscillator bridge must borrow the resident default-close HLC view"
    );
    for forbidden in ["compute_cpu", "HostF64", "upload", ".synchronize()"] {
        assert!(
            !bridge.contains(forbidden),
            "Cycle Channel Oscillator bridge retained forbidden production path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn cycle_channel_oscillator_all_outputs(");
    for required in [
        "cycle_channel_oscillator_batch_f64",
        "F64Kernel::CycleChannelOscillator",
        "const OUTPUT_IDS: [&str; 2]",
        "d_short_cycle_lengths",
        "d_medium_cycle_lengths",
        "d_short_multipliers",
        "d_medium_multipliers",
        "d_short_history",
        "d_medium_history",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Cycle Channel Oscillator resident wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Cycle Channel Oscillator tuples/outputs must use one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(source",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Cycle Channel Oscillator route retained forbidden path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/cycle_channel_oscillator_kernel.cu",
    );
    let signature_start = kernel
        .find("void cycle_channel_oscillator_batch_f64(")
        .expect("Cycle Channel Oscillator production entry point is missing");
    let signature_end = kernel[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .expect("Cycle Channel Oscillator production entry point has no body");
    let production_signature = &kernel[signature_start..signature_end];
    for required in [
        "const double* source",
        "const double* high",
        "const double* low",
        "const double* close",
        "short_cycle_lengths",
        "medium_cycle_lengths",
        "short_multipliers",
        "medium_multipliers",
        "out_fast",
        "out_slow",
        "short_history",
        "medium_history",
    ] {
        assert!(
            production_signature.contains(required),
            "Cycle Channel Oscillator production kernel is missing exact ABI fact `{required}`"
        );
    }
    let production = function_body(&kernel, "void cycle_channel_oscillator_batch_f64(");
    for required in [
        "short_cycle_lengths[row]",
        "medium_cycle_lengths[row]",
        "short_multipliers[row]",
        "medium_multipliers[row]",
        "row_fast[i]",
        "row_slow[i]",
        "row_short_history[valid_count]",
        "row_medium_history[valid_count]",
    ] {
        assert!(
            production.contains(required),
            "Cycle Channel Oscillator production body does not consume ABI fact `{required}`"
        );
    }

    let scalar =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/cycle_channel_oscillator.rs");
    for public_contract in [
        "pub struct CycleChannelOscillatorOutput",
        "pub struct CycleChannelOscillatorParams",
        "pub struct CycleChannelOscillatorInput<'a>",
        "pub fn cycle_channel_oscillator_with_kernel(",
    ] {
        assert!(
            scalar.contains(public_contract),
            "Cycle Channel Oscillator scalar/public API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_daily_factor_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("daily_factor")
        .expect("Daily Factor needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["value", "ema", "signal"],
        "Daily Factor runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["threshold_level"],
        "Daily Factor runtime parameter identity/order drifted"
    );
    let parameter = &runtime.params[0];
    assert!(matches!(parameter.kind, IndicatorParamKind::Float));
    assert!(!parameter.required);
    assert_eq!(parameter.default, Some(ParamValueStatic::Float(0.35_f64)));
    assert_eq!(parameter.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert_eq!(parameter.max.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(parameter.step.map(f64::to_bits), Some(0.01_f64.to_bits()));
    assert!(parameter.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_daily_factor_batch(");
    for canonical in ["value", "ema", "signal"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Daily Factor CPU dispatch is missing canonical output `{canonical}`"
        );
    }

    let ledger = source("src/core/indicator_ledger.rs");
    let exclusions = ledger
        .split_once("pub const PRODUCTION_OUTPUT_EXCLUSIONS:")
        .map(|(_, tail)| tail.split_once("];\n").map_or(tail, |(item, _)| item))
        .expect("production exclusions must remain an explicit schema contract");
    assert_eq!(
        exclusions.matches("\"daily_factor\"").count(),
        1,
        "Daily Factor must keep exactly its one reviewed output exclusion"
    );
    assert!(
        exclusions.contains("Some(\"ema\")")
            && exclusions.contains("fixed-period EMA(14)")
            && exclusions.contains("standalone EMA feature"),
        "the reviewed fixed EMA(14) production exclusion must remain authoritative"
    );
}

#[test]
fn classic_daily_factor_uses_one_full_three_output_launch_for_two_admitted_outputs() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_daily_factor_parameters(");
    for required in [
        "get_indicator",
        "DAILY_FACTOR_FULL_OUTPUT_IDS",
        "DAILY_FACTOR_PRODUCTION_OUTPUT_IDS",
        "DAILY_FACTOR_PARAMETER_KEYS",
        "IndicatorParamKind::Float",
        "ParamValueStatic::Float(0.35)",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "has no canonical period sweep",
    ] {
        assert!(
            parameters.contains(required),
            "Daily Factor route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_daily_factor_outputs_device")
            .count(),
        1,
        "Daily Factor must have exactly one full three-output launch site"
    );
    assert!(
        executor.contains("DAILY_FACTOR_FULL_OUTPUT_IDS")
            && executor.contains("DAILY_FACTOR_PRODUCTION_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("[f64::from_bits(threshold_level_bits)]"),
        "the full three-output ABI must remain resident while only two admitted IDs materialize"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_daily_factor_outputs_device(");
    assert!(
        bridge.contains("daily_factor_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_ohlc4_finite"),
        "Daily Factor bridge must borrow the resident OHLC upload and exact first-valid receipt"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn daily_factor_all_outputs(");
    for required in [
        "daily_factor_batch_f64",
        "F64Kernel::DailyFactor",
        "const OUTPUT_IDS: [&str; 3]",
        "d_threshold_levels",
        "_parameter_f64: vec![d_threshold_levels]",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Daily Factor resident wrapper is missing exact shared-session ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all Daily Factor outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(open",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDailyFactor",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Daily Factor route retained forbidden path `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/daily_factor_kernel.cu");
    let production = function_body(&kernel, "void daily_factor_batch_f64(");
    for required in [
        "threshold_levels[combo_idx]",
        "out_value",
        "out_ema",
        "out_signal",
        "prev_ema + alpha * (c - prev_ema)",
        "fabs(prev_open - prev_close) / range",
    ] {
        assert!(
            production.contains(required),
            "Daily Factor production kernel is missing exact formula/ABI fact `{required}`"
        );
    }

    let standalone =
        source("../../vendor/vector-ta-0.2.9-patched/src/cuda/daily_factor_wrapper.rs");
    for public_contract in [
        "pub struct CudaDailyFactor",
        "pub struct CudaDailyFactorBatchResult",
        "pub fn batch_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Daily Factor standalone/public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_damiani_volatmeter_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("damiani_volatmeter")
        .expect("Damiani Volatmeter needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["vol", "anti"],
        "Damiani Volatmeter runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["vis_atr", "vis_std", "sed_atr", "sed_std", "threshold"],
        "Damiani Volatmeter runtime parameter identity/order drifted"
    );
    for (parameter, expected_default) in runtime.params[..4].iter().zip([13_i64, 20, 40, 100]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(
            parameter.default,
            Some(ParamValueStatic::Int(expected_default))
        );
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let threshold = &runtime.params[4];
    assert!(matches!(threshold.kind, IndicatorParamKind::Float));
    assert!(!threshold.required);
    assert_eq!(threshold.default, Some(ParamValueStatic::Float(1.4_f64)));
    assert!(threshold.min.is_none());
    assert!(threshold.max.is_none());
    assert!(threshold.step.is_none());
    assert!(threshold.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_damiani_volatmeter_batch(");
    for canonical in ["vol", "anti"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Damiani Volatmeter CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case").count(),
        2,
        "Damiani Volatmeter CPU dispatch retained a non-registry output alias"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Damiani Volatmeter `value` alias remains accepted"
    );
}

#[test]
fn classic_damiani_volatmeter_uses_one_dynamic_pair_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_damiani_volatmeter_parameters(");
    for required in [
        "get_indicator",
        "DAMIANI_VOLATMETER_OUTPUT_IDS",
        "DAMIANI_VOLATMETER_PARAMETER_KEYS",
        "let default_windows = [13_usize, 20, 40, 100]",
        "ParamValueStatic::Int(default as i64)",
        "Some(ParamValueStatic::Float(value))",
        "1.4_f64.to_bits()",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
    ] {
        assert!(
            parameters.contains(required),
            "Damiani Volatmeter route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_damiani_volatmeter_outputs_device")
            .count(),
        1,
        "Damiani Volatmeter must have exactly one two-output production launch site"
    );
    assert!(
        executor.contains("DAMIANI_VOLATMETER_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64")
            && executor.contains("f64::from_bits(threshold_bits)"),
        "both Damiani Volatmeter matrices must remain resident through the exact tuple"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_damiani_volatmeter_outputs_device(");
    assert!(
        bridge.contains("damiani_volatmeter_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close"),
        "Damiani Volatmeter bridge must borrow the resident close upload and exact receipt"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn damiani_volatmeter_all_outputs(");
    for required in [
        "damiani_volatmeter_outputs_f64",
        "F64Kernel::DamianiVolatmeter",
        "const OUTPUT_IDS: [&str; 2]",
        "d_vis_atrs",
        "d_vis_stds",
        "d_sed_atrs",
        "d_sed_stds",
        "d_thresholds",
        "d_ring_vis",
        "d_ring_sed",
        "_scratch_f64: vec![d_ring_vis, d_ring_sed]",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Damiani Volatmeter resident wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "both Damiani Volatmeter outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(prices",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDamianiVolatmeter",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Damiani Volatmeter route retained forbidden path `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/damiani_volatmeter_kernel.cu");
    let production = function_body(&kernel, "void damiani_volatmeter_outputs_f64(");
    for required in [
        "vis_atrs[combo]",
        "vis_stds[combo]",
        "sed_atrs[combo]",
        "sed_stds[combo]",
        "thresholds[combo]",
        "ring_vis_scratch",
        "ring_sed_scratch",
        "out_vol",
        "out_anti",
        "damiani_volatmeter_row_f64",
    ] {
        assert!(
            production.contains(required),
            "Damiani Volatmeter production kernel is missing exact formula/ABI fact `{required}`"
        );
    }
    let primary = function_body(&kernel, "void damiani_volatmeter_neo_batch_f64(");
    assert!(
        primary.contains("damiani_volatmeter_row_f64"),
        "the preserved primary ABI bypasses the shared exact f64 row authority"
    );
    let shared = function_body(&kernel, "void damiani_volatmeter_row_f64(");
    for required in [
        "vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3)",
        "anti[i] = threshold - ratio",
        "sum_sq_vis_std = sum_sq_vis_std - (old_v * old_v) + (val * val)",
        "sum_sq_sed_std = sum_sq_sed_std - (old_s * old_s) + (val * val)",
        "atr_vis_val = ((vis_atr_f - 1.0) * atr_vis_val + tr) / vis_atr_f",
        "atr_sed_val = ((sed_atr_f - 1.0) * atr_sed_val + tr) / sed_atr_f",
    ] {
        assert!(
            shared.contains(required),
            "Damiani Volatmeter shared f64 row authority is missing exact operation `{required}`"
        );
    }

    let standalone =
        source("../../vendor/vector-ta-0.2.9-patched/src/cuda/damiani_volatmeter_wrapper.rs");
    for public_contract in [
        "pub struct CudaDamianiVolatmeter",
        "pub fn damiani_volatmeter_batch_dev_from_device_prices(",
        "pub fn damiani_volatmeter_batch_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Damiani Volatmeter standalone/public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_di_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("di")
        .expect("DI needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["plus", "minus"],
        "DI registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 1);
    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(14)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_di_batch(");
    for canonical in ["plus", "minus"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "DI CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        2,
        "DI CPU dispatch retained an unversioned output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired DI `value` alias remains accepted"
    );
}

#[test]
fn classic_di_uses_one_dynamic_pair_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_di_parameters(");
    for required in [
        "get_indicator",
        "DI_OUTPUT_IDS",
        "DI_PARAMETER_KEYS",
        "ParamValueStatic::Int(14)",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "positive_usize_parameter",
    ] {
        assert!(
            parameters.contains(required),
            "DI route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_di_outputs_device").count(),
        1,
        "DI must have exactly one two-output production launch site"
    );
    assert!(
        executor.contains("DI_OUTPUT_IDS") && executor.contains("download_named_outputs_f64"),
        "both DI matrices must stay resident until the named FeatureFrame boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_di_outputs_device(");
    assert!(
        bridge.contains("di_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_hlc"),
        "DI bridge must borrow the resident HLC upload and exact first-valid receipt"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn di_all_outputs(");
    for required in [
        "di_outputs_f64",
        "F64Kernel::Di",
        "const OUTPUT_IDS: [&str; 2]",
        "d_periods",
        "ohlcv.high()",
        "ohlcv.low()",
        "ohlcv.close()",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident DI wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "plus/minus must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDi",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident DI route retained forbidden path `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/di_kernel.cu");
    let production = function_body(&kernel, "void di_outputs_f64(");
    for required in ["periods[combo]", "out_plus", "out_minus", "di_row_f64"] {
        assert!(
            production.contains(required),
            "DI production kernel is missing exact pair fact `{required}`"
        );
    }
    let primary = function_body(&kernel, "void neoethos_di_batch_f64(");
    assert!(
        primary.contains("di_row_f64"),
        "the preserved DI primary ABI bypasses the shared exact f64 row authority"
    );
    let shared = function_body(&kernel, "void di_row_f64(");
    for required in [
        "if (dp > dm && dp > 0.0)",
        "if (dm > dp && dm > 0.0)",
        "cur_plus = fma(cur_plus, keep, inc_p)",
        "cur_minus = fma(cur_minus, keep, inc_m)",
        "cur_tr = fma(cur_tr, keep, tr)",
        "(cur_tr == 0.0) ? 0.0 : (100.0 / cur_tr)",
    ] {
        assert!(
            shared.contains(required),
            "DI shared f64 row authority is missing exact operation `{required}`"
        );
    }

    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/di_wrapper.rs");
    for public_contract in [
        "pub struct CudaDi",
        "pub fn di_batch_dev(",
        "pub fn di_batch_dev_from_device_inputs(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "DI standalone/public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_didi_index_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    let runtime = vector_ta::indicators::registry::get_indicator("didi_index")
        .expect("Didi Index needs one canonical runtime registry entry");
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["short", "long", "crossover", "crossunder"],
        "Didi Index registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 3);
    for (parameter, key, default) in runtime
        .params
        .iter()
        .zip(["short_length", "medium_length", "long_length"])
        .zip([3_i64, 8, 20])
        .map(|((parameter, key), default)| (parameter, key, default))
    {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_didi_index_batch(");
    for canonical in ["short", "long", "crossover", "crossunder"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Didi Index CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        4,
        "Didi Index CPU dispatch retained an unversioned output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Didi Index `value` alias remains accepted"
    );
    assert!(
        !cpu.contains("output_id.unwrap_or(\"short\")"),
        "a Didi Index fast path still invents an ambiguous default output"
    );
}

#[test]
fn classic_didi_index_replaces_divergent_full_math_with_one_resident_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(&implementation, "fn resolve_didi_index_parameters(");
    for required in [
        "get_indicator",
        "DIDI_INDEX_OUTPUT_IDS",
        "DIDI_INDEX_PARAMETER_KEYS",
        "const DEFAULTS: [i64; 3] = [3, 8, 20]",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "positive_usize_parameter",
    ] {
        assert!(
            parameters.contains(required),
            "Didi Index route does not prove exact parameter fact `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_didi_index_outputs_device")
            .count(),
        1,
        "Didi Index must have exactly one four-output production launch site"
    );
    assert!(
        executor.contains("DIDI_INDEX_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "all Didi Index matrices must stay resident until the named FeatureFrame boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_didi_index_outputs_device(");
    assert!(
        bridge.contains("didi_index_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.finite_close_count"),
        "Didi Index bridge must borrow the resident close upload and exact CPU validity receipt"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn didi_index_all_outputs(");
    for required in [
        "didi_index_batch_f64",
        "F64Kernel::DidiIndex",
        "const OUTPUT_IDS: [&str; 4]",
        "d_short_lengths",
        "d_medium_lengths",
        "d_long_lengths",
        "d_short_rings",
        "d_medium_rings",
        "d_long_rings",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Didi Index wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "four Didi Index outputs must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(data",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDidiIndex",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Didi Index route retained forbidden path `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/didi_index_kernel.cu");
    let full = function_body(&kernel, "void didi_index_batch_f64(");
    for required in [
        "short_lengths[row]",
        "medium_lengths[row]",
        "long_lengths[row]",
        "short_rings",
        "medium_rings",
        "long_rings",
        "didi_index_row_f64",
    ] {
        assert!(
            full.contains(required),
            "Didi Index full kernel is missing exact dynamic fact `{required}`"
        );
    }
    assert!(
        !full.contains("for (int j ="),
        "retired from-scratch window summation remains in the full kernel"
    );
    let primary = function_body(&kernel, "void didi_index_neo_batch_f64(");
    assert!(
        primary.contains("didi_index_row_f64"),
        "the preserved Didi Index primary ABI bypasses the shared exact f64 row authority"
    );
    let shared = function_body(&kernel, "void didi_index_row_f64(");
    for required in [
        "s_sum += value",
        "s_sum += value - old",
        "m_sum += value",
        "m_sum += value - old",
        "l_sum += value",
        "l_sum += value - old",
        "short_ma / medium_ma",
        "long_ma / medium_ma",
        "short_value > long_value && prev_short <= prev_long",
        "short_value < long_value && prev_short >= prev_long",
    ] {
        assert!(
            shared.contains(required),
            "Didi Index shared f64 row authority is missing exact operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void didi_index_row_f64(").count(),
        1,
        "Didi Index retained more than one full arithmetic authority"
    );
    assert!(
        !kernel.contains("for (int j ="),
        "the divergent full-kernel resummation implementation was not removed"
    );

    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/didi_index_wrapper.rs");
    for public_contract in [
        "pub struct CudaDidiIndex",
        "pub fn batch_dev(",
        "pub struct CudaDidiIndexBatchResult",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Didi Index standalone/public CUDA API drifted at `{public_contract}`"
        );
    }
    let standalone_batch = function_body(&standalone, "pub fn batch_dev(");
    for required in ["d_short_rings", "d_medium_rings", "d_long_rings"] {
        assert!(
            standalone_batch.contains(required),
            "standalone Didi Index internals did not adopt exact shared scratch `{required}`"
        );
    }
}

#[test]
fn classic_directional_imbalance_index_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let runtime = vector_ta::indicators::registry::get_indicator("directional_imbalance_index")
        .expect("Directional Imbalance Index needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::HighLow);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["up", "down", "bulls", "bears", "upper", "lower"],
        "Directional Imbalance Index registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 2);
    for ((parameter, key), default) in runtime
        .params
        .iter()
        .zip(["length", "period"])
        .zip([10_i64, 70])
    {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let ledger = source("src/core/indicator_ledger.rs");
    let unregistered_start = ledger
        .find("pub const UNREGISTERED_MULTI_OUTPUTS:")
        .expect("unregistered output authority disappeared");
    let unregistered_tail = &ledger[unregistered_start..];
    let unregistered_end = unregistered_tail
        .find("];")
        .expect("unregistered output authority is unterminated");
    let unregistered = &unregistered_tail[..unregistered_end];
    assert!(
        !unregistered.contains("directional_imbalance_index"),
        "a second unregistered output schema survived the canonical registry repair"
    );

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_directional_imbalance_index_batch(");
    for canonical in ["up", "down", "bulls", "bears", "upper", "lower"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Directional Imbalance Index CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        6,
        "Directional Imbalance Index CPU dispatch retained an unversioned output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Directional Imbalance Index `value` alias remains accepted"
    );
}

#[test]
fn classic_directional_imbalance_index_uses_one_shared_resident_six_output_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    let parameters = function_body(
        &implementation,
        "fn resolve_directional_imbalance_index_parameters(",
    );
    for required in [
        "get_indicator",
        "DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS",
        "DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS",
        "const DEFAULTS: [i64; 2] = [10, 70]",
        "ClassicCudaParameters::Defaults",
        "ClassicCudaParameters::Swept",
        "positive_usize_parameter",
    ] {
        assert!(
            parameters.contains(required),
            "Directional Imbalance Index route does not prove `{required}`"
        );
    }

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_directional_imbalance_index_outputs_device")
            .count(),
        1,
        "Directional Imbalance Index must have one six-output production launch site"
    );
    assert!(
        executor.contains("DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS")
            && executor.contains("download_named_outputs_f64"),
        "all six matrices must stay resident to the named FeatureFrame boundary"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_directional_imbalance_index_outputs_device(",
    );
    assert!(
        bridge.contains("directional_imbalance_index_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_high_low_finite"),
        "Directional Imbalance Index bridge must borrow resident high/low and exact validity"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn directional_imbalance_index_all_outputs(");
    for required in [
        "directional_imbalance_index_batch_f64",
        "F64Kernel::DirectionalImbalanceIndex",
        "const OUTPUT_IDS: [&str; 6]",
        "d_lengths",
        "d_periods",
        "d_high_ring",
        "d_low_ring",
        "d_up_hits",
        "d_down_hits",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Directional Imbalance Index wrapper is missing `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "six Directional Imbalance Index outputs must come from one launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDirectionalImbalanceIndex",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Directional Imbalance Index route retained `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/directional_imbalance_index_kernel.cu",
    );
    let full = function_body(&kernel, "void directional_imbalance_index_batch_f64(");
    assert!(
        full.contains("directional_imbalance_index_row_f64") && !full.contains("for (int i ="),
        "dynamic six-output ABI must delegate rather than retain divergent row arithmetic"
    );
    let primary = function_body(&kernel, "void directional_imbalance_index_neo_batch_f64(");
    assert!(
        primary.contains("directional_imbalance_index_row_f64")
            && !primary.contains("for (int i ="),
        "preserved primary ABI must delegate to the same exact row authority"
    );
    let shared = function_body(&kernel, "void directional_imbalance_index_row_f64(");
    for required in [
        "!isfinite(h) || !isfinite(l)",
        "window_cap = length + 1",
        "h == upper",
        "l == lower",
        "up_sum -= row_up_hits[hit_head]",
        "up_sum += up_hit",
        "up_sum / total",
        "down_sum / total",
    ] {
        assert!(
            shared.contains(required),
            "shared Directional Imbalance Index row is missing `{required}`"
        );
    }
    assert_eq!(
        kernel
            .matches("void directional_imbalance_index_row_f64(")
            .count(),
        1,
        "Directional Imbalance Index retained multiple row authorities"
    );

    let standalone = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/directional_imbalance_index_wrapper.rs",
    );
    for public_contract in [
        "pub struct CudaDirectionalImbalanceIndex",
        "pub fn batch_dev(",
        "pub struct CudaDirectionalImbalanceIndexBatchResult",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Directional Imbalance Index public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_disparity_index_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry.matches("id: \"disparity_index\"").count(),
        1,
        "Disparity Index must have exactly one canonical registry seed"
    );

    let runtime = vector_ta::indicators::registry::get_indicator("disparity_index")
        .expect("Disparity Index needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["value"],
        "Disparity Index runtime output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        [
            "ema_period",
            "lookback_period",
            "smoothing_period",
            "smoothing_type",
        ],
        "Disparity Index runtime parameter identity/order drifted"
    );
    for (parameter, default) in runtime.params[..3].iter().zip([14_i64, 14, 9]) {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let smoothing_type = &runtime.params[3];
    assert!(matches!(
        smoothing_type.kind,
        IndicatorParamKind::EnumString
    ));
    assert!(!smoothing_type.required);
    assert_eq!(
        smoothing_type.default,
        Some(ParamValueStatic::EnumString("ema"))
    );
    assert!(smoothing_type.min.is_none());
    assert!(smoothing_type.max.is_none());
    assert!(smoothing_type.step.is_none());
    assert_eq!(smoothing_type.enum_values, ["ema", "sma"]);
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_disparity_index_batch(");
    assert!(
        dispatch.contains("expect_value_output(\"disparity_index\", output_id)?"),
        "Disparity Index CPU dispatch must require the sole canonical `value` output"
    );
    for retired in [
        "eq_ignore_ascii_case(\"values\")",
        "eq_ignore_ascii_case(\"disparity\")",
    ] {
        assert!(
            !dispatch.contains(retired),
            "Disparity Index CPU dispatch retained `{retired}`"
        );
    }
}

#[test]
fn classic_disparity_index_uses_one_shared_resident_exact_f64_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const DISPARITY_INDEX_ID: &str = \"disparity_index\"")
            && implementation.contains("DISPARITY_INDEX_PARAMETER_KEYS")
            && implementation.contains("resolve_disparity_index_parameters("),
        "Disparity Index needs one exact typed registry/parameter contract before launch"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_disparity_index_output_device")
            .count(),
        1,
        "Disparity Index production must enter its retained-parameter bridge once"
    );
    assert!(
        !executor.contains("compute_primary_device(DISPARITY_INDEX_ID"),
        "Disparity Index production retained a fixed-default primary replay"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_disparity_index_output_device(");
    for required in [
        "self.ohlcv.close.as_view_f64()",
        "self.max_consecutive_finite_close",
        ".disparity_index_production_output(",
    ] {
        assert!(
            bridge.contains(required),
            "Disparity Index shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaDisparityIndex",
        "CudaRuntime::new",
        ".synchronize()",
        "upload",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Disparity Index bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn disparity_index_production_output(");
    for required in [
        "disparity_index_batch_f64",
        "F64Kernel::DisparityIndex",
        "d_ema_periods",
        "d_lookback_periods",
        "d_smoothing_periods",
        "d_smoothing_flags",
        "d_disparity_buffer",
        "d_sma_buffer",
        "max_consecutive_finite_close",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Disparity Index wrapper is missing `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "every admitted Disparity Index tuple must come from one retained launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        "DeviceBuffer::from_slice(data",
        ".synchronize()",
        "compute_cpu",
        "HostF64",
        "CudaDisparityIndex",
        "f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Disparity Index route retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/disparity_index_kernel.cu");
    let full = function_body(&kernel, "void disparity_index_batch_f64(");
    assert!(
        full.contains("disparity_index_row_f64") && !full.contains("for (int i ="),
        "dynamic full ABI must delegate rather than retain divergent row arithmetic"
    );
    let primary = function_body(&kernel, "void disparity_index_neo_batch_f64(");
    assert!(
        primary.contains("disparity_index_row_f64") && !primary.contains("for (int i ="),
        "preserved primary ABI must delegate to the same exact row authority"
    );
    let shared = function_body(&kernel, "void disparity_index_row_f64(");
    for required in [
        "fma(ema, ema_beta, ema_alpha * value)",
        "fmax(high, window_value)",
        "fmin(low, window_value)",
        "!(high > low)",
        "fma(smoothed, smoothing_beta, smoothing_alpha * scaled)",
        "sma_sum += scaled - old",
    ] {
        assert!(
            shared.contains(required),
            "shared Disparity Index row is missing `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void disparity_index_row_f64(").count(),
        1,
        "Disparity Index retained multiple row authorities"
    );

    let standalone =
        source("../../vendor/vector-ta-0.2.9-patched/src/cuda/disparity_index_wrapper.rs");
    for public_contract in [
        "pub struct CudaDisparityIndex",
        "pub struct CudaDisparityIndexBatchResult",
        "pub fn batch_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Disparity Index public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_dm_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry.matches("id: \"dm\"").count(),
        1,
        "DM must have exactly one canonical registry seed"
    );

    let runtime = vector_ta::indicators::registry::get_indicator("dm")
        .expect("DM needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::HighLow);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["plus", "minus"],
        "DM registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 1);
    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(14)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_dm_batch(");
    for canonical in ["plus", "minus"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "DM CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        2,
        "DM CPU dispatch retained an unversioned output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired DM `value` alias remains accepted"
    );
}

#[test]
fn classic_dm_uses_one_shared_resident_exact_f64_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const DM_ID: &str = \"dm\"")
            && implementation.contains("DM_OUTPUT_IDS")
            && implementation.contains("DM_PARAMETER_KEYS")
            && implementation.contains("resolve_dm_parameters("),
        "DM needs one exact typed registry/parameter contract before launch"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_dm_outputs_device").count(),
        1,
        "DM production must enter exactly one full-pair bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(DM_ID"),
        "DM production retained a plus-only primary replay"
    );

    let cuda_f64 =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    assert!(
        cuda_f64.contains("| (\"dm\", \"minus\")"),
        "DM minus must be admitted as a named resident f64 output"
    );

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_dm_outputs_device(");
    for required in [
        ".dm_all_outputs(",
        "CudaDeviceHighLowF64Ref::new(",
        "self.ohlcv.high.as_view_f64()",
        "self.ohlcv.low.as_view_f64()",
        "self.first_valid_high_low",
    ] {
        assert!(
            bridge.contains(required),
            "DM shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaDm",
        "CudaRuntime::new",
        "upload",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "DM bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn dm_all_outputs(");
    for required in [
        "dm_batch_f64",
        "F64Kernel::Dm",
        "const OUTPUT_IDS: [&str; 2]",
        "d_periods",
        "high_low.high()",
        "high_low.low()",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident DM wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "plus/minus must be emitted by one exact CUDA launch"
    );
    for forbidden in ["Context::new", "Stream::new", ".synchronize()", "upload"] {
        assert!(
            !resident.contains(forbidden),
            "resident DM wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/dm_kernel.cu");
    for entry_point in [
        "void dm_batch_f64(",
        "void dm_many_series_one_param_time_major_f64(",
        "void dm_neo_batch_f64(",
    ] {
        let entry = function_body(&kernel, entry_point);
        assert!(
            entry.contains("dm_row_f64(") && !entry.contains("for (int"),
            "preserved DM ABI `{entry_point}` must delegate without duplicate row arithmetic"
        );
    }
    let shared = function_body(&kernel, "void dm_row_f64(");
    for required in [
        "const double diff_p = hi - prev_high",
        "const double diff_m = prev_low - lo",
        "if (diff_p > 0.0 && diff_p > diff_m)",
        "else if (diff_m > 0.0 && diff_m > diff_p)",
        "sum_plus += diff_p",
        "sum_minus += diff_m",
        "sum_plus = sum_plus - (sum_plus * inv_p) + plus_value",
        "sum_minus = sum_minus - (sum_minus * inv_p) + minus_value",
    ] {
        assert!(
            shared.contains(required),
            "shared DM row authority is missing exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void dm_row_f64(").count(),
        1,
        "DM retained multiple f64 row authorities"
    );
    for retired in ["dm_step_f64", "CompSum_f64", "CompEMA_f64"] {
        assert!(
            !kernel.contains(retired),
            "DM retained divergent compensated helper `{retired}`"
        );
    }

    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/dm_wrapper.rs");
    for public_contract in [
        "pub struct CudaDm",
        "pub fn dm_batch_dev(",
        "pub fn dm_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "DM public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_donchian_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry.matches("id: \"donchian\"").count(),
        1,
        "Donchian must have exactly one canonical registry seed"
    );

    let runtime = vector_ta::indicators::registry::get_indicator("donchian")
        .expect("Donchian needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::HighLow);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["upper", "middle", "lower"],
        "Donchian registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 1);
    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(20)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_donchian_batch(");
    for canonical in ["upper", "middle", "lower"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Donchian CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        3,
        "Donchian CPU dispatch retained an unversioned output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Donchian `value` alias remains accepted"
    );
}

#[test]
fn classic_donchian_uses_one_shared_resident_exact_f64_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const DONCHIAN_ID: &str = \"donchian\"")
            && implementation.contains("DONCHIAN_OUTPUT_IDS")
            && implementation.contains("DONCHIAN_PARAMETER_KEYS")
            && implementation.contains("resolve_donchian_parameters("),
        "Donchian needs one exact typed registry/parameter contract before launch"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_donchian_outputs_device").count(),
        1,
        "Donchian production must enter exactly one full triple-output bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(DONCHIAN_ID"),
        "Donchian production retained an upper-only primary replay"
    );

    let cuda_f64 =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    for named in ["middle", "lower"] {
        assert!(
            cuda_f64.contains(&format!("| (\"donchian\", \"{named}\")")),
            "Donchian `{named}` must be admitted as a named resident f64 output"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_donchian_outputs_device(");
    for required in [
        ".donchian_all_outputs(",
        "CudaDeviceHighLowF64Ref::new(",
        "self.ohlcv.high.as_view_f64()",
        "self.ohlcv.low.as_view_f64()",
        "self.first_valid_high_low",
    ] {
        assert!(
            bridge.contains(required),
            "Donchian shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaDonchian",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Donchian bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn donchian_all_outputs(");
    for required in [
        "donchian_all_outputs_batch_f64",
        "F64Kernel::Donchian",
        "const OUTPUT_IDS: [&str; 3]",
        "d_periods",
        "high_low.high()",
        "high_low.low()",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Donchian wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "upper/middle/lower must be emitted by one exact CUDA launch"
    );
    for forbidden in ["Context::new", "Stream::new", ".synchronize()", "upload"] {
        assert!(
            !resident.contains(forbidden),
            "resident Donchian wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/donchian_kernel.cu");
    for entry_point in [
        "void donchian_batch_f64(",
        "void donchian_all_outputs_batch_f64(",
    ] {
        let entry = function_body(&kernel, entry_point);
        assert!(
            entry.contains("donchian_row_f64(") && !entry.contains("for (int"),
            "Donchian ABI `{entry_point}` must delegate without duplicate row arithmetic"
        );
    }
    let shared = function_body(&kernel, "void donchian_row_f64(");
    for required in [
        "if (period <= 32)",
        "if (isnan(h) || isnan(l))",
        "const bool ok = isfinite(h) && isfinite(l)",
        "const int suffix_end =",
        "const int prefix_start =",
        "suffix_max > prefix_max ? suffix_max : prefix_max",
        "suffix_min < prefix_min ? suffix_min : prefix_min",
        "fma(maxv - minv, 0.5, minv)",
    ] {
        assert!(
            shared.contains(required),
            "shared Donchian row authority is missing exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void donchian_row_f64(").count(),
        1,
        "Donchian retained multiple f64 row authorities"
    );

    let standalone = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/donchian_wrapper.rs");
    for public_contract in [
        "pub struct CudaDonchian",
        "pub fn donchian_batch_dev(",
        "pub fn donchian_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            standalone.contains(public_contract),
            "Donchian public CUDA API drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_dual_ulcer_index_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry.matches("id: \"dual_ulcer_index\"").count(),
        1,
        "Dual Ulcer Index must have exactly one canonical registry seed"
    );

    let runtime = vector_ta::indicators::registry::get_indicator("dual_ulcer_index")
        .expect("Dual Ulcer Index needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["long_ulcer", "short_ulcer", "threshold"],
        "Dual Ulcer Index registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 3);

    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(5)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let auto_threshold = &runtime.params[1];
    assert_eq!(auto_threshold.key, "auto_threshold");
    assert!(matches!(auto_threshold.kind, IndicatorParamKind::Bool));
    assert!(!auto_threshold.required);
    assert_eq!(auto_threshold.default, Some(ParamValueStatic::Bool(true)));
    assert!(auto_threshold.min.is_none());
    assert!(auto_threshold.max.is_none());
    assert!(auto_threshold.step.is_none());
    assert!(auto_threshold.enum_values.is_empty());

    let threshold = &runtime.params[2];
    assert_eq!(threshold.key, "threshold");
    assert!(matches!(threshold.kind, IndicatorParamKind::Float));
    assert!(!threshold.required);
    assert_eq!(threshold.default, Some(ParamValueStatic::Float(0.1_f64)));
    assert_eq!(threshold.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert!(threshold.max.is_none());
    assert_eq!(threshold.step.map(f64::to_bits), Some(0.1_f64.to_bits()));
    assert!(threshold.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_dual_ulcer_index_batch(");
    for canonical in ["long_ulcer", "short_ulcer", "threshold"] {
        let canonical_dispatch = format!("eq_ignore_ascii_case(\"{canonical}\")");
        assert!(
            dispatch.contains(canonical_dispatch.as_str()),
            "Dual Ulcer Index CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        3,
        "Dual Ulcer Index CPU dispatch retained an unversioned output spelling"
    );
    for retired in ["value", "uulcer", "dulcer"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "retired Dual Ulcer Index `{retired}` alias remains accepted"
        );
    }
}

#[test]
fn classic_dual_ulcer_index_uses_one_shared_resident_exact_f64_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const DUAL_ULCER_INDEX_ID: &str = \"dual_ulcer_index\"")
            && implementation.contains("DUAL_ULCER_INDEX_OUTPUT_IDS")
            && implementation.contains("DUAL_ULCER_INDEX_PARAMETER_KEYS")
            && implementation.contains("resolve_dual_ulcer_index_parameters("),
        "Dual Ulcer Index needs one exact typed registry/parameter contract before launch"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_dual_ulcer_index_outputs_device")
            .count(),
        1,
        "Dual Ulcer Index production must enter exactly one full triple-output bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(DUAL_ULCER_INDEX_ID"),
        "Dual Ulcer Index production retained a long-only primary replay"
    );

    let cuda_f64 =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    for named in ["short_ulcer", "threshold"] {
        assert!(
            cuda_f64.contains(&format!("| (\"dual_ulcer_index\", \"{named}\")")),
            "Dual Ulcer Index `{named}` must be admitted as a named resident f64 output"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_dual_ulcer_index_outputs_device(");
    for required in [
        ".dual_ulcer_index_all_outputs(",
        "self.ohlcv.close.as_view_f64()",
        "self.max_consecutive_valid_dual_ulcer_close",
    ] {
        assert!(
            bridge.contains(required),
            "Dual Ulcer Index shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaDualUlcerIndex",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Dual Ulcer Index bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn dual_ulcer_index_all_outputs(");
    for required in [
        "dual_ulcer_index_all_outputs_f64",
        "F64Kernel::DualUlcerIndex",
        "const OUTPUT_IDS: [&str; 3]",
        "d_periods",
        "d_auto_thresholds",
        "d_thresholds",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Dual Ulcer Index wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "long/short/threshold must be emitted by one exact CUDA launch"
    );
    for forbidden in ["Context::new", "Stream::new", ".synchronize()", "upload"] {
        assert!(
            !resident.contains(forbidden),
            "resident Dual Ulcer Index wrapper retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/dual_ulcer_index_kernel.cu");
    for entry_point in [
        "void dual_ulcer_index_all_outputs_f64(",
        "void dual_ulcer_index_neo_batch_f64(",
    ] {
        let entry = function_body(&kernel, entry_point);
        assert!(
            entry.contains("dual_ulcer_index_row_f64("),
            "Dual Ulcer Index ABI `{entry_point}` bypasses the shared row authority"
        );
    }
    let shared = function_body(&kernel, "void dual_ulcer_index_row_f64(");
    for required in [
        "if (!neo_dui_valid(close))",
        "close_count = 0",
        "sq_count = 0",
        "long_sq_sum = 0.0",
        "short_sq_sum = 0.0",
        "long_sq_sum -= leaving_long_sq",
        "short_sq_sum -= leaving_short_sq",
        "long_sq_sum += long_sq",
        "short_sq_sum += short_sq",
        "sqrt(long_sq_sum) / denom",
        "sqrt(short_sq_sum) / denom",
        "diff_sum += diff",
        "diff_count += 1",
    ] {
        assert!(
            shared.contains(required),
            "shared Dual Ulcer Index row authority is missing exact operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void dual_ulcer_index_row_f64(").count(),
        1,
        "Dual Ulcer Index retained multiple production f64 row authorities"
    );

    for retired_production_entry in [
        "dual_ulcer_index_build_squares_f64",
        "dual_ulcer_index_finalize_f64",
    ] {
        assert!(
            !resident.contains(retired_production_entry),
            "production still invokes standalone two-pass ABI `{retired_production_entry}`"
        );
        assert!(
            kernel.contains(retired_production_entry),
            "public two-pass ABI `{retired_production_entry}` was not preserved"
        );
    }
}

#[test]
fn classic_dvdiqqe_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry.matches("id: \"dvdiqqe\"").count(),
        1,
        "DVDIQQE must have exactly one canonical registry seed"
    );

    let runtime = vector_ta::indicators::registry::get_indicator("dvdiqqe")
        .expect("DVDIQQE needs one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["dvdi", "fast_tl", "slow_tl", "center_line"],
        "DVDIQQE registry output identities/order drifted"
    );
    assert_eq!(runtime.params.len(), 7);

    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(13)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());

    let smoothing = &runtime.params[1];
    assert_eq!(smoothing.key, "smoothing_period");
    assert!(matches!(smoothing.kind, IndicatorParamKind::Int));
    assert!(!smoothing.required);
    assert_eq!(smoothing.default, Some(ParamValueStatic::Int(6)));
    assert_eq!(smoothing.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(smoothing.max.is_none());
    assert_eq!(smoothing.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(smoothing.enum_values.is_empty());

    for (parameter, key, default) in [
        (&runtime.params[2], "fast_multiplier", 2.618_f64),
        (&runtime.params[3], "slow_multiplier", 4.236_f64),
    ] {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Float(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert!(parameter.step.is_none());
        assert!(parameter.enum_values.is_empty());
    }

    for (parameter, key, default) in [
        (&runtime.params[4], "volume_type", "default"),
        (&runtime.params[5], "center_type", "dynamic"),
    ] {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::EnumString));
        assert!(!parameter.required);
        assert_eq!(
            parameter.default,
            Some(ParamValueStatic::EnumString(default))
        );
        assert!(parameter.min.is_none());
        assert!(parameter.max.is_none());
        assert!(parameter.step.is_none());
        assert!(parameter.enum_values.is_empty());
    }

    let tick = &runtime.params[6];
    assert_eq!(tick.key, "tick_size");
    assert!(matches!(tick.kind, IndicatorParamKind::Float));
    assert!(!tick.required);
    assert_eq!(tick.default, Some(ParamValueStatic::Float(0.01_f64)));
    assert_eq!(tick.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert!(tick.max.is_none());
    assert!(tick.step.is_none());
    assert!(tick.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_dvdiqqe_batch(");
    for canonical in ["dvdi", "fast_tl", "slow_tl", "center_line"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "DVDIQQE CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        4,
        "DVDIQQE CPU dispatch retained an unversioned output spelling"
    );
    for retired in ["value", "fast", "slow", "center"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "retired DVDIQQE `{retired}` alias remains accepted"
        );
    }
}

#[test]
fn classic_dvdiqqe_uses_one_shared_resident_exact_f64_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    assert!(
        implementation.contains("const DVDIQQE_ID: &str = \"dvdiqqe\"")
            && implementation.contains("DVDIQQE_OUTPUT_IDS")
            && implementation.contains("DVDIQQE_PARAMETER_KEYS")
            && implementation.contains("resolve_dvdiqqe_parameters("),
        "DVDIQQE needs one exact typed registry/parameter contract before launch"
    );

    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_dvdiqqe_outputs_device").count(),
        1,
        "DVDIQQE production must enter exactly one full four-output bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(DVDIQQE_ID"),
        "DVDIQQE production retained a dvdi-only primary replay"
    );

    for named in ["fast_tl", "slow_tl", "center_line"] {
        assert!(
            has_f64_resident_output_route("dvdiqqe", named),
            "DVDIQQE `{named}` must be admitted as a named resident f64 output"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_dvdiqqe_outputs_device(");
    for required in [
        ".dvdiqqe_all_outputs(",
        "self.ohlcv.as_view()",
        "self.first_valid_close_finite",
    ] {
        assert!(
            bridge.contains(required),
            "DVDIQQE shared-session bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaDvdiqqe",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "DVDIQQE bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn dvdiqqe_all_outputs(");
    for required in [
        "dvdiqqe_all_outputs_f64",
        "F64Kernel::Dvdiqqe",
        "const OUTPUT_IDS: [&str; 4]",
        "d_periods",
        "d_smoothing_periods",
        "d_fast_multipliers",
        "d_slow_multipliers",
        "d_use_tick_only",
        "d_dynamic_center",
        "d_tick_sizes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident DVDIQQE wrapper is missing exact ABI fact `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "dvdi/fast/slow/center must be emitted by one exact CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "upload",
        "CudaDvdiqqe",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident DVDIQQE wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/dvdiqqe_kernel.cu");
    for entry_point in [
        "void dvdiqqe_all_outputs_f64(",
        "void dvdiqqe_neo_batch_f64(",
    ] {
        let entry = function_body(&kernel, entry_point);
        assert!(
            entry.contains("dvdiqqe_row_f64("),
            "DVDIQQE ABI `{entry_point}` bypasses the shared row authority"
        );
    }
    let shared = function_body(&kernel, "void dvdiqqe_row_f64(");
    for required in [
        "neo_dvdi_ema_step",
        "pvi_prev += d_close",
        "nvi_prev -= d_close",
        "const double dvdi = pd - nd",
        "fabs(dvdi - prev_dvdi)",
        "smooth_range * fast_multiplier",
        "smooth_range * slow_multiplier",
        "center_sum += dvdi",
        "center_count += 1.0",
    ] {
        assert!(
            shared.contains(required),
            "shared DVDIQQE row authority is missing exact operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void dvdiqqe_row_f64(").count(),
        1,
        "DVDIQQE retained multiple production f64 row authorities"
    );

    let public_f32 = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/dvdiqqe_wrapper.rs");
    for public_contract in [
        "pub struct CudaDvdiqqe",
        "pub fn dvdiqqe_batch_dev(",
        "pub fn dvdiqqe_batch_dev_from_device_inputs(",
        "pub fn dvdiqqe_many_series_one_param_time_major_dev(",
    ] {
        assert!(
            public_f32.contains(public_contract),
            "DVDIQQE public f32 ABI drifted at `{public_contract}`"
        );
    }
}

#[test]
fn classic_ehlers_autocorrelation_periodogram_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    let runtime =
        vector_ta::indicators::registry::get_indicator("ehlers_autocorrelation_periodogram")
            .expect("Ehlers Autocorrelation Periodogram needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["dominant_cycle", "normalized_power"],
        "Ehlers Autocorrelation Periodogram output identity/order drifted"
    );
    assert_eq!(runtime.params.len(), 4);

    for (index, key, default, minimum) in [
        (0, "min_period", 8, 3.0_f64),
        (1, "max_period", 48, 4.0_f64),
        (2, "avg_length", 3, 0.0_f64),
    ] {
        let parameter = &runtime.params[index];
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(minimum.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let enhance = &runtime.params[3];
    assert_eq!(enhance.key, "enhance");
    assert!(matches!(enhance.kind, IndicatorParamKind::Bool));
    assert!(!enhance.required);
    assert_eq!(enhance.default, Some(ParamValueStatic::Bool(true)));
    assert!(enhance.min.is_none());
    assert!(enhance.max.is_none());
    assert!(enhance.step.is_none());
    assert_eq!(enhance.enum_values, ["true", "false"]);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_ehlers_autocorrelation_periodogram_batch(");
    for canonical in ["dominant_cycle", "normalized_power"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "CPU dispatch is missing canonical Ehlers Autocorrelation Periodogram output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        2,
        "CPU dispatch retained an unversioned Ehlers Autocorrelation Periodogram output spelling"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "CPU dispatch retained the retired Ehlers Autocorrelation Periodogram `value` alias"
    );
}

#[test]
fn classic_ehlers_autocorrelation_periodogram_uses_one_exact_resident_pair_launch() {
    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EHLERS_AUTOCORRELATION_PERIODOGRAM_ID",
        "EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS",
        "EHLERS_AUTOCORRELATION_PERIODOGRAM_PARAMETER_KEYS",
        "resolve_ehlers_autocorrelation_periodogram_parameters(",
    ] {
        assert!(
            implementation.contains(required),
            "typed Ehlers Autocorrelation Periodogram plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_ehlers_autocorrelation_periodogram_outputs_device")
            .count(),
        1,
        "all admitted Ehlers Autocorrelation Periodogram receipts need one full-pair bridge"
    );

    for output_id in ["dominant_cycle", "normalized_power"] {
        assert!(
            has_f64_resident_output_route("ehlers_autocorrelation_periodogram", output_id,),
            "Ehlers Autocorrelation Periodogram `{output_id}` is missing its resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_ehlers_autocorrelation_periodogram_outputs_device(",
    );
    assert!(
        bridge.contains("ehlers_autocorrelation_periodogram_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()"),
        "the production bridge must borrow the resident close series and full-pair route"
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEhlersAutocorrelationPeriodogram",
        "CudaRuntime::new",
        ".synchronize()",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Ehlers Autocorrelation Periodogram bridge retained `{forbidden}`"
        );
    }

    let scalar = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/ehlers_autocorrelation_periodogram.rs",
    );
    let constructor = function_body(&scalar, "fn new_resolved(");
    assert!(
        scalar.contains("ehlers_autocorrelation_periodogram_exact_coefficients(")
            && constructor.contains("build_exact_coefficients("),
        "CPU and CUDA coefficient uploads must share the CPU-owned immutable table authority"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn ehlers_autocorrelation_periodogram_all_outputs(",
    );
    for required in [
        "ehlers_autocorrelation_periodogram_outputs_f64",
        "ehlers_autocorrelation_periodogram_exact_coefficients",
        "parameter_bytes",
        "coefficient_bytes",
        "trig_table_bytes",
        "scratch_bytes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Ehlers Autocorrelation Periodogram wrapper is missing `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "dominant_cycle and normalized_power must come from one CUDA launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEhlersAutocorrelationPeriodogram",
        "ehlers_autocorrelation_periodogram_batch_f64",
        "ehlers_autocorrelation_periodogram_neo_batch_f64",
    ] {
        assert!(
            !resident.contains(forbidden),
            "production wrapper retained standalone/legacy path `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
    );
    let production = function_body(
        &kernel,
        "void ehlers_autocorrelation_periodogram_outputs_f64(",
    );
    let row = function_body(
        &kernel,
        "void ehlers_autocorrelation_periodogram_exact_row_f64(",
    );
    assert!(
        production.contains("ehlers_autocorrelation_periodogram_exact_row_f64("),
        "the production entry must use one shared exact per-row authority"
    );
    for exact_cpu_order in [
        "if (avg_length == 3)",
        "avg3_sx = avg3_x0 + avg3_x1 + avg3_x2",
        "avg3_sxx = avg3_x0 * avg3_x0 + avg3_x1 * avg3_x1 + avg3_x2 * avg3_x2",
        "cos_table[trig_base + n]",
        "sin_table[trig_base + n]",
    ] {
        assert!(
            row.contains(exact_cpu_order),
            "shared row lost exact CPU operation `{exact_cpu_order}`"
        );
    }
    for forbidden in [" cos(", " sin(", " exp(", " pow("] {
        assert!(
            !row.contains(forbidden),
            "production row retained device transcendental `{forbidden}` instead of uploaded CPU bits"
        );
    }
    for preserved_abi in [
        "void ehlers_autocorrelation_periodogram_batch_f64(",
        "void ehlers_autocorrelation_periodogram_neo_batch_f64(",
    ] {
        assert!(
            kernel.contains(preserved_abi),
            "existing public CUDA ABI `{preserved_abi}` was not preserved"
        );
    }
}

#[test]
fn classic_ehlers_data_sampling_rsi_registry_cpu_and_ledger_use_only_canonical_contracts() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "ehlers_data_sampling_relative_strength_indicator";
    let registry = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    assert_eq!(
        registry
            .matches("id: \"ehlers_data_sampling_relative_strength_indicator\"")
            .count(),
        1,
        "EDSRSI must have exactly one canonical registry seed"
    );
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("EDSRSI requires one canonical runtime registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["ds_rsi", "original_rsi", "signal"]
    );
    assert_eq!(runtime.params.len(), 1);
    let length = &runtime.params[0];
    assert_eq!(length.key, "length");
    assert!(matches!(length.kind, IndicatorParamKind::Int));
    assert!(!length.required);
    assert_eq!(length.default, Some(ParamValueStatic::Int(14)));
    assert_eq!(length.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(length.max.is_none());
    assert_eq!(length.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(length.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(
        &cpu,
        "fn compute_ehlers_data_sampling_relative_strength_indicator_batch(",
    );
    for canonical in ["ds_rsi", "original_rsi", "signal"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "EDSRSI CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        3,
        "EDSRSI CPU dispatch retained an unversioned output spelling"
    );
    for retired in ["data_sampling_rsi", "orig_rsi", "value"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "retired EDSRSI `{retired}` alias remains accepted"
        );
    }

    let ledger = source("src/core/indicator_ledger.rs");
    let unregistered = ledger
        .split_once("pub const UNREGISTERED_MULTI_OUTPUTS:")
        .expect("unregistered output authority missing")
        .1
        .split_once("];\n")
        .expect("unregistered output authority is unterminated")
        .0;
    assert!(
        !unregistered.contains(ID),
        "canonical EDSRSI registry and unregistered override must not coexist"
    );
    assert!(
        ledger.contains("canonical_base_and_length_sweeps - 2")
            && ledger.contains("canonical_base_and_length_sweeps, 12"),
        "EDSRSI +10 dataset/search identity migration must remain explicit"
    );
    assert!(
        ledger.contains("original_rsi is the unmodified RSI auxiliary already emitted"),
        "reviewed original_rsi production exclusion was lost"
    );
}

#[test]
fn classic_ehlers_data_sampling_rsi_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "ehlers_data_sampling_relative_strength_indicator";
    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EHLERS_DATA_SAMPLING_RSI_ID",
        "EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS",
        "EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS",
        "EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS",
        "resolve_ehlers_data_sampling_rsi_length(",
    ] {
        assert!(
            implementation.contains(required),
            "EDSRSI typed planner is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_ehlers_data_sampling_rsi_outputs_device")
            .count(),
        1,
        "EDSRSI production must enter exactly one full-output resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(EHLERS_DATA_SAMPLING_RSI_ID"),
        "EDSRSI production retained a primary replay"
    );

    for named in ["original_rsi", "signal"] {
        assert!(
            has_f64_resident_output_route(ID, named),
            "EDSRSI `{named}` lacks an exact resident f64 capability"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_ehlers_data_sampling_rsi_outputs_device(",
    );
    for required in [
        ".ehlers_data_sampling_relative_strength_indicator_all_outputs(",
        "self.ohlcv.as_view()",
    ] {
        assert!(
            bridge.contains(required),
            "EDSRSI bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "EDSRSI bridge retained forbidden path `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn ehlers_data_sampling_relative_strength_indicator_all_outputs(",
    );
    for required in [
        "ehlers_data_sampling_relative_strength_indicator_batch_f64",
        "F64Kernel::EhlersDataSamplingRelativeStrengthIndicator",
        "const OUTPUT_IDS: [&str; 3]",
        "d_lengths",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident EDSRSI wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in ["Context::new", "Stream::new", ".synchronize()", "upload"] {
        assert!(
            !resident.contains(forbidden),
            "resident EDSRSI wrapper retained `{forbidden}`"
        );
    }

    let wrapper_source = source(
        "../../vendor/vector-ta-0.2.9-patched/src/cuda/ehlers_data_sampling_relative_strength_indicator_wrapper.rs",
    );
    assert!(
        wrapper_source.contains("ehlers_data_sampling_relative_strength_indicator_batch_f64"),
        "standalone public EDSRSI ABI was not preserved"
    );
    assert!(
        !resident.contains("CudaEhlersDataSamplingRelativeStrengthIndicator"),
        "production resident route invokes the standalone second-context wrapper"
    );

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_data_sampling_relative_strength_indicator_kernel.cu",
    );
    for entry_point in [
        "void ehlers_data_sampling_relative_strength_indicator_batch_f64(",
        "void ehlers_data_sampling_relative_strength_indicator_neo_batch_f64(",
    ] {
        let entry = function_body(&kernel, entry_point);
        assert!(
            entry.contains("edsrsi_compute_rsi_row("),
            "EDSRSI ABI `{entry_point}` bypasses the shared exact row authority"
        );
    }
    let row = function_body(&kernel, "void edsrsi_compute_rsi_row(");
    for exact_cpu_order in [
        "fma(avg_gain, beta, inv_period * g1)",
        "fma(avg_loss, beta, inv_period * l1)",
        "fma(avg_gain, beta, inv_period * g2)",
        "fma(avg_loss, beta, inv_period * l2)",
        "fma(avg_gain, beta, inv_period * g)",
        "fma(avg_loss, beta, inv_period * l)",
    ] {
        assert!(
            row.contains(exact_cpu_order),
            "shared EDSRSI row lost `{exact_cpu_order}`"
        );
    }
    let primary = function_body(
        &kernel,
        "void ehlers_data_sampling_relative_strength_indicator_neo_batch_f64(",
    );
    assert!(primary.contains("periods[row_idx]"));
    assert!(!primary.contains("NEO_EDSRSI_LENGTH"));
    assert!(!kernel.contains("#define NEO_EDSRSI_LENGTH"));
}

#[test]
fn classic_ehlers_linear_extrapolation_predictor_registry_and_cpu_are_canonical() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "ehlers_linear_extrapolation_predictor";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("ELEP needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["prediction", "filter", "state", "go_long", "go_short"]
    );
    assert_eq!(runtime.params.len(), 5);
    for (index, key, default, minimum, maximum, step) in [
        (
            0,
            "high_pass_length",
            125,
            Some(1.0_f64),
            None,
            Some(1.0_f64),
        ),
        (1, "low_pass_length", 12, Some(1.0_f64), None, Some(1.0_f64)),
        (
            3,
            "bars_forward",
            5,
            Some(0.0_f64),
            Some(10.0_f64),
            Some(1.0_f64),
        ),
    ] {
        let parameter = &runtime.params[index];
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), minimum.map(f64::to_bits));
        assert_eq!(parameter.max.map(f64::to_bits), maximum.map(f64::to_bits));
        assert_eq!(parameter.step.map(f64::to_bits), step.map(f64::to_bits));
        assert!(parameter.enum_values.is_empty());
    }
    let gain = &runtime.params[2];
    assert_eq!(gain.key, "gain");
    assert!(matches!(gain.kind, IndicatorParamKind::Float));
    assert_eq!(gain.default, Some(ParamValueStatic::Float(0.7)));
    assert!(gain.min.is_none() && gain.max.is_none() && gain.step.is_none());
    let signal_mode = &runtime.params[4];
    assert_eq!(signal_mode.key, "signal_mode");
    assert!(matches!(signal_mode.kind, IndicatorParamKind::EnumString));
    assert_eq!(
        signal_mode.default,
        Some(ParamValueStatic::EnumString("predict_filter_crosses"))
    );
    assert_eq!(
        signal_mode.enum_values,
        [
            "predict_filter_crosses",
            "predict_middle_crosses",
            "filter_middle_crosses",
        ]
    );
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(
        &cpu,
        "fn compute_ehlers_linear_extrapolation_predictor_batch(",
    );
    for canonical in ["prediction", "filter", "state", "go_long", "go_short"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "CPU dispatch is missing canonical ELEP `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        5,
        "ELEP CPU dispatch retained an unversioned output spelling"
    );
    assert!(!dispatch.contains("eq_ignore_ascii_case(\"value\")"));
}

#[test]
fn classic_ehlers_linear_extrapolation_predictor_uses_one_exact_resident_quint_launch() {
    const ID: &str = "ehlers_linear_extrapolation_predictor";
    const OUTPUT_IDS: [&str; 5] = ["prediction", "filter", "state", "go_long", "go_short"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID",
        "EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS",
        "EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_PARAMETER_KEYS",
        "resolve_ehlers_linear_extrapolation_predictor_parameters(",
    ] {
        assert!(
            implementation.contains(required),
            "typed ELEP plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_ehlers_linear_extrapolation_predictor_outputs_device")
            .count(),
        1,
        "all five ELEP receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "ELEP `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_ehlers_linear_extrapolation_predictor_outputs_device(",
    );
    assert!(
        bridge.contains("ehlers_linear_extrapolation_predictor_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEhlersLinearExtrapolationPredictor",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "ELEP bridge retained `{forbidden}`"
        );
    }

    let scalar = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/ehlers_linear_extrapolation_predictor.rs",
    );
    let resolved = function_body(&scalar, "fn resolve_params(");
    let specialized = function_body(&scalar, "fn row_from_slice_lowpass12(");
    let stream = function_body(&scalar, "fn new_resolved(");
    assert!(
        scalar.contains("ehlers_linear_extrapolation_predictor_exact_coefficients(")
            && resolved.contains("hann_weights(")
            && specialized.contains("params.hann_weights")
            && stream.contains("params.hann_weights.clone()"),
        "CPU rows and CUDA upload must share one CPU-owned coefficient/Hann authority"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn ehlers_linear_extrapolation_predictor_all_outputs(",
    );
    for required in [
        "ehlers_linear_extrapolation_predictor_outputs_f64",
        "ehlers_linear_extrapolation_predictor_exact_coefficients",
        "parameter_bytes",
        "coefficient_bytes",
        "hann_weight_bytes",
        "scratch_bytes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "ELEP wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEhlersLinearExtrapolationPredictor",
        "ehlers_linear_extrapolation_predictor_batch_f64",
        "ehlers_linear_extrapolation_predictor_neo_batch_f64",
    ] {
        assert!(
            !resident.contains(forbidden),
            "ELEP wrapper retained `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_linear_extrapolation_predictor_kernel.cu",
    );
    let production = function_body(
        &kernel,
        "void ehlers_linear_extrapolation_predictor_outputs_f64(",
    );
    assert!(production.contains("ehlers_linear_extrapolation_predictor_row_f64("));
    for forbidden in [
        " exp(",
        " cos(",
        "ehlers_linear_extrapolation_predictor_batch_f64(",
        "ehlers_linear_extrapolation_predictor_neo_batch_f64(",
    ] {
        assert!(
            !production.contains(forbidden),
            "production ELEP entry retained `{forbidden}`"
        );
    }
    for preserved_abi in [
        "void ehlers_linear_extrapolation_predictor_batch_f64(",
        "void ehlers_linear_extrapolation_predictor_neo_batch_f64(",
    ] {
        let entry = function_body(&kernel, preserved_abi);
        assert!(entry.contains("ehlers_linear_extrapolation_predictor_row_f64("));
    }
    assert_eq!(
        kernel
            .matches("void ehlers_linear_extrapolation_predictor_row_f64(")
            .count(),
        1,
        "ELEP retained multiple complete row arithmetic authorities"
    );
}

#[test]
fn classic_ehlers_undersampled_double_moving_average_registry_and_cpu_are_canonical() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "ehlers_undersampled_double_moving_average";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("EUDMA needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["fast", "slow"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["fast_length", "slow_length", "sample_length", "output"]
    );
    for (index, default) in [(0, 6), (1, 12), (2, 5)] {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert_eq!(parameter.max.map(f64::to_bits), Some(4096.0_f64.to_bits()));
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let output = &runtime.params[3];
    assert!(matches!(output.kind, IndicatorParamKind::EnumString));
    assert_eq!(output.default, Some(ParamValueStatic::EnumString("fast")));
    assert_eq!(output.enum_values, ["fast", "slow"]);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ma_batch.rs");
    assert!(cpu.contains(
        "if ma_type.eq_ignore_ascii_case(\"ehlers_undersampled_double_moving_average\")"
    ));
    assert!(cpu.contains("\"fast\" => out.fast_values"));
    assert!(cpu.contains("\"slow\" => out.slow_values"));
    assert!(!cpu.contains("\"fast\" | \"value\" => out.fast_values"));
}

#[test]
fn classic_ehlers_undersampled_double_moving_average_uses_one_exact_resident_pair_launch() {
    const ID: &str = "ehlers_undersampled_double_moving_average";
    const OUTPUT_IDS: [&str; 2] = ["fast", "slow"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID",
        "EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS",
        "EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_PARAMETER_KEYS",
        "resolve_ehlers_undersampled_double_moving_average_parameters(",
    ] {
        assert!(
            implementation.contains(required),
            "typed EUDMA plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_ehlers_undersampled_double_moving_average_outputs_device")
            .count(),
        1,
        "both EUDMA receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "EUDMA `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_ehlers_undersampled_double_moving_average_outputs_device(",
    );
    assert!(
        bridge.contains("ehlers_undersampled_double_moving_average_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEhlersUndersampledDoubleMovingAverage",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "EUDMA bridge retained `{forbidden}`"
        );
    }

    let scalar = source(
        "../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ehlers_undersampled_double_moving_average.rs",
    );
    assert!(
        scalar.contains("ehlers_undersampled_double_moving_average_exact_hann_payload(")
            && scalar.contains("HannFilterState::new(fast_length)")
            && scalar.contains("HannFilterState::new(slow_length)"),
        "CPU rows and CUDA upload must share the exact CPU-owned Hann authority"
    );

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn ehlers_undersampled_double_moving_average_all_outputs(",
    );
    for required in [
        "ehlers_undersampled_double_moving_average_outputs_f64",
        "ehlers_undersampled_double_moving_average_exact_hann_payload",
        "parameter_bytes",
        "hann_weight_bytes",
        "scratch_bytes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "EUDMA wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEhlersUndersampledDoubleMovingAverage",
        "ehlers_undersampled_double_moving_average_neo_batch_f64",
    ] {
        assert!(
            !resident.contains(forbidden),
            "EUDMA wrapper retained `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/ehlers_undersampled_double_moving_average_kernel.cu",
    );
    let production = function_body(
        &kernel,
        "void ehlers_undersampled_double_moving_average_outputs_f64(",
    );
    assert!(production.contains("ehlers_undersampled_double_moving_average_row_f64("));
    for forbidden in [
        " cos(",
        "ehlers_undersampled_double_moving_average_neo_batch_f64(",
    ] {
        assert!(
            !production.contains(forbidden),
            "production EUDMA retained `{forbidden}`"
        );
    }
    let primary = function_body(
        &kernel,
        "void ehlers_undersampled_double_moving_average_neo_batch_f64(",
    );
    assert!(primary.contains("ehlers_undersampled_double_moving_average_row_f64("));
    assert_eq!(
        kernel
            .matches("void ehlers_undersampled_double_moving_average_row_f64(")
            .count(),
        1,
        "EUDMA retained multiple complete row arithmetic authorities"
    );
}

#[test]
fn classic_ema_deviation_corrected_t3_registry_cpu_and_artifact_boundary_are_canonical() {
    use neoethos_data::core::feature_registry::{
        ProductionFeatureProducerId, production_feature_producer_manifest_v1,
    };
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "ema_deviation_corrected_t3";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("EDCT3 needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["corrected", "t3"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "hot", "t3_mode", "output"]
    );
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(10)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    let hot = &runtime.params[1];
    assert!(matches!(hot.kind, IndicatorParamKind::Float));
    assert!(!hot.required);
    assert_eq!(hot.default, Some(ParamValueStatic::Float(0.7)));
    assert_eq!(hot.min.map(f64::to_bits), Some((-16.0_f64).to_bits()));
    assert_eq!(hot.max.map(f64::to_bits), Some(16.0_f64.to_bits()));
    assert_eq!(hot.step.map(f64::to_bits), Some(0.01_f64.to_bits()));
    let mode = &runtime.params[2];
    assert!(matches!(mode.kind, IndicatorParamKind::Int));
    assert!(!mode.required);
    assert_eq!(mode.default, Some(ParamValueStatic::Int(0)));
    assert_eq!(mode.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert_eq!(mode.max.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert_eq!(mode.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    let output = &runtime.params[3];
    assert!(matches!(output.kind, IndicatorParamKind::EnumString));
    assert_eq!(
        output.default,
        Some(ParamValueStatic::EnumString("corrected"))
    );
    assert_eq!(output.enum_values, ["corrected", "t3"]);
    assert!(runtime.capabilities.supports_cuda_single);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu =
        source("../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ma_batch.rs");
    let dispatch = function_body(
        &cpu,
        "if ma_type.eq_ignore_ascii_case(\"ema_deviation_corrected_t3\")",
    );
    assert!(dispatch.contains("\"corrected\" => out.corrected"));
    assert!(dispatch.contains("\"t3\" => out.t3"));
    assert!(!dispatch.contains("\"corrected\" | \"value\""));

    let implementation = source("src/core/classic_cuda_plan.rs");
    let resolver = function_body(
        &implementation,
        "fn resolve_ema_deviation_corrected_t3_parameters(",
    );
    assert!(resolver.contains("Some(ParamValueStatic::Int(10))"));
    for forbidden in [
        "ParamValueStatic::Int(14)",
        "legacy_period",
        "legacy_default",
        "cached_default",
        "current_default",
    ] {
        assert!(
            !resolver.contains(forbidden),
            "EDCT3 resolver retained a silent old-artifact path `{forbidden}`"
        );
    }

    let classic_manifest = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::ClassicVectorTa)
        .expect("classic/vector-ta producer manifest row");
    assert!(
        classic_manifest.semantic_version() >= 5,
        "period-10 EDCT3 values must not reuse period-14 classic artifacts"
    );
    let semantic_paths = classic_manifest
        .semantic_sources()
        .entries()
        .iter()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    assert!(
        semantic_paths.contains(&"vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs")
            && semantic_paths.contains(
                &"vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/registry.rs"
            ),
        "the content-addressed classic generation must bind both EDCT3 registry authorities"
    );
}

#[test]
fn classic_ema_deviation_corrected_t3_uses_one_shared_resident_exact_f64_pair_launch() {
    const ID: &str = "ema_deviation_corrected_t3";
    const OUTPUT_IDS: [&str; 2] = ["corrected", "t3"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EMA_DEVIATION_CORRECTED_T3_ID",
        "EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS",
        "EMA_DEVIATION_CORRECTED_T3_PARAMETER_KEYS",
        "resolve_ema_deviation_corrected_t3_parameters(",
    ] {
        assert!(
            implementation.contains(required),
            "typed EDCT3 plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_ema_deviation_corrected_t3_outputs_device")
            .count(),
        1,
        "both EDCT3 receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "EDCT3 `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_ema_deviation_corrected_t3_outputs_device(",
    );
    assert!(
        bridge.contains("ema_deviation_corrected_t3_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEmaDeviationCorrectedT3",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "EDCT3 bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn ema_deviation_corrected_t3_all_outputs(");
    for required in [
        "ema_deviation_corrected_t3_outputs_f64",
        "parameter_bytes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "EDCT3 wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEmaDeviationCorrectedT3",
        "ema_deviation_corrected_t3_neo_batch_f64",
    ] {
        assert!(
            !resident.contains(forbidden),
            "EDCT3 wrapper retained `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/ema_deviation_corrected_t3_kernel.cu",
    );
    let production = function_body(&kernel, "void ema_deviation_corrected_t3_outputs_f64(");
    assert!(production.contains("ema_deviation_corrected_t3_row_f64("));
    assert!(!production.contains("ema_deviation_corrected_t3_neo_batch_f64("));
    let primary = function_body(&kernel, "void ema_deviation_corrected_t3_neo_batch_f64(");
    assert!(primary.contains("ema_deviation_corrected_t3_row_f64("));
    assert_eq!(
        kernel
            .matches("void ema_deviation_corrected_t3_row_f64(")
            .count(),
        1,
        "EDCT3 retained multiple complete row arithmetic authorities"
    );
}

#[test]
fn classic_emd_registry_cpu_and_exact_coefficient_authority_are_canonical() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "emd";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("EMD needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::HighLow);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["upperband", "middleband", "lowerband"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "delta", "fraction"]
    );
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(20)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    for (index, default) in [(1, 0.5_f64), (2, 0.1_f64)] {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Float(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert!(parameter.max.is_none() && parameter.step.is_none());
        assert!(parameter.enum_values.is_empty());
    }
    assert!(
        !runtime.capabilities.supports_cuda_single,
        "supplemental EMD has no public CUDA-single dispatcher"
    );
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_emd_batch(");
    for canonical in ["upperband", "middleband", "lowerband"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "EMD CPU dispatcher lost canonical `{canonical}`"
        );
    }
    for retired in ["upper", "middle", "lower", "value"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "EMD CPU dispatcher retained retired `{retired}` alias"
        );
    }

    let scalar = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/emd.rs");
    let exact = function_body(&scalar, "pub(crate) fn emd_exact_coefficients(");
    for required in [
        ".cos()",
        ".sqrt()",
        "half_one_minus_alpha",
        "beta_times_one_plus_alpha",
        "inv_up_low",
        "inv_mid",
    ] {
        assert!(
            exact.contains(required),
            "EMD exact CPU coefficient authority is missing `{required}`"
        );
    }
    let high_low = function_body(&scalar, "pub unsafe fn emd_scalar_into(");
    assert!(high_low.contains("emd_exact_coefficients("));
    assert!(!high_low.contains(".cos()") && !high_low.contains(".sqrt()"));
}

#[test]
fn classic_emd_uses_one_parameter_only_resident_exact_f64_triple_launch() {
    const ID: &str = "emd";
    const OUTPUT_IDS: [&str; 3] = ["upperband", "middleband", "lowerband"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EMD_ID",
        "EMD_OUTPUT_IDS",
        "EMD_PARAMETER_KEYS",
        "resolve_emd_parameters(",
        "ResolvedClassicCudaLaunch::Emd",
    ] {
        assert!(
            implementation.contains(required),
            "typed EMD plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_emd_outputs_device").count(),
        1,
        "all three EMD receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "EMD `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_emd_outputs_device(");
    assert!(
        bridge.contains("emd_all_outputs")
            && bridge.contains("CudaDeviceHighLowF64Ref::new(")
            && bridge.contains("self.ohlcv.high.as_view_f64()")
            && bridge.contains("self.ohlcv.low.as_view_f64()")
            && bridge.contains("self.first_valid_high_low")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEmd",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
        "emd_batch_f64",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "EMD bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn emd_all_outputs(");
    for required in [
        "emd_outputs_f64",
        "emd_exact_coefficients(",
        "parameter_bytes",
        "scratch_bytes",
        "self.session.stream()",
        "d_sp_rings",
        "d_sv_rings",
        "d_bp_rings",
    ] {
        assert!(
            resident.contains(required),
            "EMD wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "CudaEmd",
        "emd_batch_f64",
        "emd_batch_f32",
    ] {
        assert!(
            !resident.contains(forbidden),
            "production EMD wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/emd_kernel.cu");
    for preserved in [
        "void emd_batch_f32(",
        "void emd_many_series_one_param_time_major_f32(",
        "void emd_batch_f64(",
    ] {
        assert!(
            kernel.contains(preserved),
            "EMD compatibility ABI lost `{preserved}`"
        );
    }
    let production = function_body(&kernel, "void emd_outputs_f64(");
    assert!(production.contains("emd_row_f64("));
    assert!(!production.contains("emd_batch_f64("));
    assert!(!production.contains("cos(") && !production.contains("sqrt("));
    let primary = function_body(&kernel, "void emd_batch_f64(");
    assert!(primary.contains("emd_row_f64("));
    assert_eq!(
        kernel.matches("void emd_row_f64(").count(),
        1,
        "EMD retained multiple complete f64 row arithmetic authorities"
    );
}

#[test]
fn classic_emd_trend_registry_cpu_and_artifact_identity_are_canonical() {
    use neoethos_data::core::feature_registry::{
        ProductionFeatureProducerId, production_feature_producer_manifest_v1,
    };
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "emd_trend";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("EMD Trend needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["direction", "average", "upper", "lower"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["source", "avg_type", "length", "mult"]
    );
    assert_eq!(
        runtime.params[0].default,
        Some(ParamValueStatic::EnumString("close"))
    );
    assert_eq!(
        runtime.params[1].default,
        Some(ParamValueStatic::EnumString("SMA"))
    );
    let length = &runtime.params[2];
    assert!(matches!(length.kind, IndicatorParamKind::Int));
    assert!(!length.required);
    assert_eq!(length.default, Some(ParamValueStatic::Int(28)));
    assert_eq!(length.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(length.max.is_none());
    assert_eq!(length.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(length.enum_values.is_empty());
    let mult = &runtime.params[3];
    assert!(matches!(mult.kind, IndicatorParamKind::Float));
    assert!(!mult.required);
    assert_eq!(mult.default, Some(ParamValueStatic::Float(1.0)));
    assert_eq!(mult.min.map(f64::to_bits), Some(0.05_f64.to_bits()));
    assert!(mult.max.is_none() && mult.step.is_none());
    assert!(mult.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_emd_trend_batch(");
    for canonical in ["direction", "average", "upper", "lower"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "EMD Trend CPU dispatcher lost canonical `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "EMD Trend CPU dispatcher retained the anonymous `value` alias"
    );

    let classic_manifest = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::ClassicVectorTa)
        .expect("classic/vector-ta producer manifest row");
    assert!(
        classic_manifest.semantic_version() >= 6,
        "the 1-to-24 EMD Trend identity migration must not reuse old Classic artifacts"
    );

    let implementation = source("src/core/classic_cuda_plan.rs");
    let resolver = function_body(&implementation, "fn resolve_emd_trend_parameters(");
    for forbidden in [
        "legacy_average",
        "legacy_output",
        "legacy_default",
        "cached_default",
        "current_default",
    ] {
        assert!(
            !resolver.contains(forbidden),
            "EMD Trend resolver retained a silent old-artifact path `{forbidden}`"
        );
    }
}

#[test]
fn classic_emd_trend_uses_one_shared_resident_exact_f64_quad_launch() {
    const ID: &str = "emd_trend";
    const OUTPUT_IDS: [&str; 4] = ["direction", "average", "upper", "lower"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EMD_TREND_ID",
        "EMD_TREND_OUTPUT_IDS",
        "EMD_TREND_PARAMETER_KEYS",
        "resolve_emd_trend_parameters(",
        "ResolvedClassicCudaLaunch::EmdTrend",
    ] {
        assert!(
            implementation.contains(required),
            "typed EMD Trend plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_emd_trend_outputs_device").count(),
        1,
        "all four EMD Trend receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "EMD Trend `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_emd_trend_outputs_device(");
    assert!(
        bridge.contains("emd_trend_all_outputs")
            && bridge.contains("self.ohlcv.close.as_view_f64()")
            && bridge.contains("self.first_valid_close")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEmdTrend",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
        "emd_trend_batch_f64",
        "emd_trend_neo_batch_f64",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "EMD Trend bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn emd_trend_all_outputs(");
    for required in [
        "emd_trend_outputs_f64",
        "parameter_bytes",
        "scratch_bytes",
        "self.session.stream()",
        "d_lengths",
        "d_mults",
        "d_sma_rings",
    ] {
        assert!(
            resident.contains(required),
            "EMD Trend wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEmdTrend",
        "emd_trend_batch_f64",
        "emd_trend_neo_batch_f64",
        "compute_cpu",
    ] {
        assert!(
            !resident.contains(forbidden),
            "production EMD Trend wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/emd_trend_kernel.cu");
    for preserved in ["void emd_trend_batch_f64(", "void emd_trend_neo_batch_f64("] {
        assert!(
            kernel.contains(preserved),
            "EMD Trend compatibility ABI lost `{preserved}`"
        );
    }
    let production = function_body(&kernel, "void emd_trend_outputs_f64(");
    assert!(production.contains("emd_trend_row_f64("));
    assert!(!production.contains("emd_trend_batch_f64("));
    assert!(!production.contains("emd_trend_neo_batch_f64("));
    let primary = function_body(&kernel, "void emd_trend_neo_batch_f64(");
    assert!(primary.contains("emd_trend_row_f64("));
    let row = function_body(&kernel, "void emd_trend_row_f64(");
    for required in [
        "emd_trend_compensated_add_f64(",
        "fma(",
        "src[i] > upper",
        "src[i] < lower",
    ] {
        assert!(
            row.contains(required),
            "shared EMD Trend row lost exact arithmetic token `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void emd_trend_row_f64(").count(),
        1,
        "EMD Trend retained multiple complete f64 row arithmetic authorities"
    );
}

#[test]
fn classic_eri_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "eri";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("ERI needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["bull", "bear"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["period", "ma_type"]
    );
    let period = &runtime.params[0];
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(13)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    let ma_type = &runtime.params[1];
    assert!(matches!(ma_type.kind, IndicatorParamKind::EnumString));
    assert!(!ma_type.required);
    assert_eq!(ma_type.default, Some(ParamValueStatic::EnumString("ema")));
    assert!(ma_type.min.is_none() && ma_type.max.is_none() && ma_type.step.is_none());
    assert!(ma_type.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_eri_batch(");
    for canonical in ["bull", "bear"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "ERI CPU dispatcher lost canonical `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "ERI CPU dispatcher retained the retired `value` alias"
    );
}

#[test]
fn classic_eri_uses_one_shared_resident_exact_f64_pair_launch() {
    const ID: &str = "eri";
    const OUTPUT_IDS: [&str; 2] = ["bull", "bear"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const ERI_ID",
        "ERI_OUTPUT_IDS",
        "ERI_PARAMETER_KEYS",
        "resolve_eri_parameters(",
        "ResolvedClassicCudaLaunch::Eri",
    ] {
        assert!(
            implementation.contains(required),
            "typed ERI plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_eri_outputs_device").count(),
        1,
        "both ERI receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "ERI `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_eri_outputs_device(");
    assert!(
        bridge.contains("eri_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.first_valid_hlc")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEri",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
        "eri_batch_f64",
        "eri_neo_batch_f64",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "ERI bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn eri_all_outputs(");
    for required in [
        "eri_outputs_f64",
        "parameter_bytes",
        "self.session.stream()",
        "d_periods",
    ] {
        assert!(
            resident.contains(required),
            "ERI wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "CudaEri",
        "eri_batch_f64",
        "eri_neo_batch_f64",
        "compute_cpu",
    ] {
        assert!(
            !resident.contains(forbidden),
            "production ERI wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/eri_kernel.cu");
    for preserved in [
        "void eri_batch_f64(",
        "void eri_many_series_one_param_time_major_f64(",
        "void eri_one_series_many_params_time_major_f64(",
        "void eri_neo_batch_f64(",
    ] {
        assert!(
            kernel.contains(preserved),
            "ERI compatibility ABI lost `{preserved}`"
        );
    }
    let production = function_body(&kernel, "void eri_outputs_f64(");
    assert!(production.contains("eri_row_f64("));
    assert!(!production.contains("eri_batch_f64("));
    assert!(!production.contains("eri_neo_batch_f64("));
    let primary = function_body(&kernel, "void eri_neo_batch_f64(");
    assert!(primary.contains("eri_row_f64("));
    let row = function_body(&kernel, "void eri_row_f64(");
    for required in [
        "sum += close[first_valid + i]",
        "ema = alpha * close[i] + beta * ema",
        "high[i] - ema",
        "low[i] - ema",
    ] {
        assert!(
            row.contains(required),
            "shared ERI row lost exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void eri_row_f64(").count(),
        1,
        "ERI retained multiple complete f64 row arithmetic authorities"
    );
}

#[test]
fn classic_evasive_supertrend_registry_cpu_and_artifact_identity_are_canonical() {
    use neoethos_data::core::feature_registry::{
        ProductionFeatureProducerId, production_feature_producer_manifest_v1,
    };
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "evasive_supertrend";
    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("Evasive Supertrend needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        ["band", "state", "noisy", "changed"]
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        [
            "atr_length",
            "base_multiplier",
            "noise_threshold",
            "expansion_alpha",
        ]
    );
    let atr_length = &runtime.params[0];
    assert!(matches!(atr_length.kind, IndicatorParamKind::Int));
    assert!(!atr_length.required);
    assert_eq!(atr_length.default, Some(ParamValueStatic::Int(10)));
    assert_eq!(atr_length.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(atr_length.max.is_none());
    assert_eq!(atr_length.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(atr_length.enum_values.is_empty());
    for (index, default, min) in [
        (1, 3.0_f64, 0.1_f64),
        (2, 1.0_f64, 0.1_f64),
        (3, 0.5_f64, 0.0_f64),
    ] {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Float));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Float(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(min.to_bits()));
        assert!(parameter.max.is_none() && parameter.step.is_none());
        assert!(parameter.enum_values.is_empty());
    }
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_evasive_supertrend_batch(");
    for canonical in ["band", "state", "noisy", "changed"] {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Evasive Supertrend CPU dispatcher lost canonical `{canonical}`"
        );
    }
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "Evasive Supertrend CPU dispatcher retained the anonymous `value` alias"
    );

    let classic_manifest = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::ClassicVectorTa)
        .expect("classic/vector-ta producer manifest row");
    assert!(
        classic_manifest.semantic_version() >= 7,
        "the 1-to-24 Evasive Supertrend identity migration must not reuse old Classic artifacts"
    );

    let implementation = source("src/core/classic_cuda_plan.rs");
    let resolver = function_body(&implementation, "fn resolve_evasive_supertrend_parameters(");
    for forbidden in [
        "legacy_band",
        "legacy_output",
        "legacy_default",
        "cached_default",
        "current_default",
    ] {
        assert!(
            !resolver.contains(forbidden),
            "Evasive Supertrend resolver retained a silent old-artifact path `{forbidden}`"
        );
    }
}

#[test]
fn classic_evasive_supertrend_uses_one_shared_resident_exact_f64_quad_launch() {
    const ID: &str = "evasive_supertrend";
    const OUTPUT_IDS: [&str; 4] = ["band", "state", "noisy", "changed"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const EVASIVE_SUPERTREND_ID",
        "EVASIVE_SUPERTREND_OUTPUT_IDS",
        "EVASIVE_SUPERTREND_PARAMETER_KEYS",
        "resolve_evasive_supertrend_parameters(",
        "ResolvedClassicCudaLaunch::EvasiveSupertrend",
    ] {
        assert!(
            implementation.contains(required),
            "typed Evasive Supertrend plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_evasive_supertrend_outputs_device")
            .count(),
        1,
        "all four Evasive Supertrend receipts need one full-output resident bridge"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "Evasive Supertrend `{output_id}` is missing its named resident route"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_evasive_supertrend_outputs_device(");
    assert!(
        bridge.contains("evasive_supertrend_all_outputs")
            && bridge.contains("self.ohlcv.as_view()")
            && bridge.contains("self.max_consecutive_finite_ohlc")
    );
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaEvasiveSuperTrend",
        "CudaRuntime::new",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Evasive Supertrend bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn evasive_supertrend_all_outputs(");
    for required in [
        "evasive_supertrend_batch_f64",
        "parameter_bytes",
        "self.session.stream()",
        "d_atr_lengths",
        "d_base_multipliers",
        "d_noise_thresholds",
        "d_expansion_alphas",
    ] {
        assert!(
            resident.contains(required),
            "Evasive Supertrend wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(open",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
        "CudaEvasiveSuperTrend",
        "compute_cpu",
    ] {
        assert!(
            !resident.contains(forbidden),
            "production Evasive Supertrend wrapper retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/evasive_supertrend_kernel.cu");
    for preserved in [
        "void evasive_supertrend_batch_f64(",
        "void evasive_supertrend_neo_batch_f64(",
    ] {
        assert!(
            kernel.contains(preserved),
            "Evasive Supertrend compatibility ABI lost `{preserved}`"
        );
    }
    let production = function_body(&kernel, "void evasive_supertrend_batch_f64(");
    assert!(production.contains("evasive_supertrend_row_f64("));
    let primary = function_body(&kernel, "void evasive_supertrend_neo_batch_f64(");
    assert!(primary.contains("evasive_supertrend_row_f64("));
    assert!(primary.contains("periods[combo]"));
    let row = function_body(&kernel, "void evasive_supertrend_row_f64(");
    for required in [
        "tr_sum += tr",
        "atr = ((atr * (period_f64 - 1.0)) + tr) / period_f64",
        "(high_value + low_value) * 0.5",
        "fabs(close_value - prev_band) < atr * noise_threshold",
    ] {
        assert!(
            row.contains(required),
            "shared Evasive Supertrend row lost exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void evasive_supertrend_row_f64(").count(),
        1,
        "Evasive Supertrend retained multiple complete f64 row arithmetic authorities"
    );
}

#[test]
fn classic_fibonacci_entry_bands_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic,
    };

    const ID: &str = "fibonacci_entry_bands";
    const FULL_OUTPUT_IDS: [&str; 18] = [
        "middle",
        "trend",
        "upper_0618",
        "upper_1000",
        "upper_1618",
        "upper_2618",
        "lower_0618",
        "lower_1000",
        "lower_1618",
        "lower_2618",
        "tp_long_band",
        "tp_short_band",
        "go_long",
        "go_short",
        "rejection_long",
        "rejection_short",
        "long_bounce",
        "short_bounce",
    ];
    const PARAMETER_KEYS: [&str; 5] = [
        "source",
        "length",
        "atr_length",
        "use_atr",
        "tp_aggressiveness",
    ];

    let runtime = vector_ta::indicators::registry::get_indicator(ID)
        .expect("Fibonacci Entry Bands needs one canonical registry entry");
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        FULL_OUTPUT_IDS,
        "Fibonacci Entry Bands registry output identity/order drifted"
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        PARAMETER_KEYS,
        "Fibonacci Entry Bands registry parameter identity/order drifted"
    );

    let source_parameter = &runtime.params[0];
    assert!(matches!(
        source_parameter.kind,
        IndicatorParamKind::EnumString
    ));
    assert_eq!(
        source_parameter.default,
        Some(ParamValueStatic::EnumString("hlc3"))
    );
    assert_eq!(
        source_parameter.enum_values,
        [
            "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"
        ]
    );

    for (parameter, key, default) in [
        (&runtime.params[1], "length", 21_i64),
        (&runtime.params[2], "atr_length", 14_i64),
    ] {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }

    let use_atr = &runtime.params[3];
    assert!(matches!(use_atr.kind, IndicatorParamKind::Bool));
    assert_eq!(use_atr.default, Some(ParamValueStatic::Bool(true)));
    assert_eq!(use_atr.enum_values, ["true", "false"]);

    let tp = &runtime.params[4];
    assert!(matches!(tp.kind, IndicatorParamKind::EnumString));
    assert_eq!(tp.default, Some(ParamValueStatic::EnumString("low")));
    assert_eq!(tp.enum_values, ["low", "medium", "high"]);
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_fibonacci_entry_bands_batch(");
    for canonical in FULL_OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Fibonacci Entry Bands CPU dispatch is missing canonical output `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        FULL_OUTPUT_IDS.len(),
        "Fibonacci Entry Bands CPU dispatch retained an unversioned output spelling"
    );
    for retired in ["basis", "long_entry", "short_entry", "value"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "retired Fibonacci Entry Bands `{retired}` alias remains accepted"
        );
    }

    let ledger = source("src/core/indicator_ledger.rs");
    assert_eq!(
        ledger.matches("\"fibonacci_entry_bands\"").count(),
        2,
        "Fibonacci Entry Bands must retain exactly the two reviewed duplicate-output exclusions"
    );
    for excluded in ["tp_long_band", "tp_short_band"] {
        assert!(
            ledger.contains(&format!("Some(\"{excluded}\")")),
            "reviewed Fibonacci Entry Bands exclusion `{excluded}` disappeared"
        );
    }
}

#[test]
fn classic_fibonacci_entry_bands_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "fibonacci_entry_bands";
    const FULL_OUTPUT_IDS: [&str; 18] = [
        "middle",
        "trend",
        "upper_0618",
        "upper_1000",
        "upper_1618",
        "upper_2618",
        "lower_0618",
        "lower_1000",
        "lower_1618",
        "lower_2618",
        "tp_long_band",
        "tp_short_band",
        "go_long",
        "go_short",
        "rejection_long",
        "rejection_short",
        "long_bounce",
        "short_bounce",
    ];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const FIBONACCI_ENTRY_BANDS_ID",
        "FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS",
        "FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS",
        "FIBONACCI_ENTRY_BANDS_PARAMETER_KEYS",
        "resolve_fibonacci_entry_bands_parameters(",
        "ResolvedClassicCudaLaunch::FibonacciEntryBands",
    ] {
        assert!(
            implementation.contains(required),
            "typed Fibonacci Entry Bands plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_fibonacci_entry_bands_outputs_device")
            .count(),
        1,
        "all admitted Fibonacci Entry Bands receipts need one full-output resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(FIBONACCI_ENTRY_BANDS_ID"),
        "production retained a primary replay beside the full resident launch"
    );
    for output_id in FULL_OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "Fibonacci Entry Bands `{output_id}` is missing its resident f64 capability"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_fibonacci_entry_bands_outputs_device(");
    for required in [
        ".fibonacci_entry_bands_all_outputs(",
        "self.ohlcv.as_view()",
        "self.max_consecutive_finite_hlc",
        "self.max_consecutive_finite_ohlc",
    ] {
        assert!(
            bridge.contains(required),
            "Fibonacci Entry Bands bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaFibonacciEntryBands",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Fibonacci Entry Bands bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn fibonacci_entry_bands_all_outputs(");
    for required in [
        "fibonacci_entry_bands_batch_f64",
        "const OUTPUT_IDS: [&str; 18]",
        "\"middle\"",
        "d_lengths",
        "d_atr_lengths",
        "d_stdev_scratch",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Fibonacci Entry Bands wrapper is missing `{required}`"
        );
    }
    assert_eq!(
        resident.matches("launch!(").count(),
        1,
        "all eighteen outputs must come from one resident launch"
    );
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(open",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Fibonacci Entry Bands wrapper retained `{forbidden}`"
        );
    }

    let invariant = function_body(&wrapper, "pub fn is_period_invariant(self) -> bool");
    assert!(
        !invariant.contains("F64Kernel::FibonacciEntryBands"),
        "Fibonacci Entry Bands still discards the canonical length anchor"
    );

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/fibonacci_entry_bands_kernel.cu");
    for entry_point in [
        "void fibonacci_entry_bands_batch_f64(",
        "void fibonacci_entry_bands_neo_batch_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("fibonacci_entry_bands_row_f64("),
            "preserved ABI `{entry_point}` bypasses the shared f64 row authority"
        );
    }
    let primary = function_body(&kernel, "void fibonacci_entry_bands_neo_batch_f64(");
    assert!(
        primary.contains("periods[combo]") && !primary.contains("(void)periods"),
        "preserved primary ABI does not consume the requested length"
    );
    let row = function_body(&kernel, "void fibonacci_entry_bands_row_f64(");
    for required in [
        "ema1 + ema_alpha * (source - ema1)",
        "ema2 + ema_alpha * (ema1 - ema2)",
        "atr_sum += tr",
        "static_cast<double>(atr_length - 1) * atr_value",
        "stdev_state.update(source, &volatility)",
        "basis + volatility * MULT1",
        "basis - volatility * MULT4",
        "crossunder(c, tp_long_band",
        "crossover(c, tp_short_band",
    ] {
        assert!(
            row.contains(required),
            "shared Fibonacci Entry Bands row lost exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel
            .matches("void fibonacci_entry_bands_row_f64(")
            .count(),
        1,
        "Fibonacci Entry Bands retained multiple complete f64 row authorities"
    );
}

#[test]
fn classic_fibonacci_trailing_stop_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic, list_indicators};

    const ID: &str = "fibonacci_trailing_stop";
    const OUTPUT_IDS: [&str; 4] = ["trailing_stop", "long_stop", "short_stop", "direction"];
    const PARAMETER_KEYS: [&str; 4] = ["left_bars", "right_bars", "level", "trigger"];

    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "Fibonacci Trailing Stop must be registered exactly once"
    );
    let runtime = &matches[0];
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        PARAMETER_KEYS
    );
    for (parameter, key, default) in [
        (&runtime.params[0], "left_bars", 20_i64),
        (&runtime.params[1], "right_bars", 1_i64),
    ] {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let level = &runtime.params[2];
    assert!(matches!(level.kind, IndicatorParamKind::Float));
    assert_eq!(level.default, Some(ParamValueStatic::Float(-0.382)));
    assert!(level.min.is_none() && level.max.is_none());
    assert_eq!(level.step.map(f64::to_bits), Some(0.001_f64.to_bits()));
    assert_eq!(
        level.notes,
        Some("Finite Fibonacci extension/retracement factor")
    );
    let trigger = &runtime.params[3];
    assert!(matches!(trigger.kind, IndicatorParamKind::EnumString));
    assert_eq!(trigger.default, Some(ParamValueStatic::EnumString("close")));
    assert_eq!(trigger.enum_values, ["close", "wick"]);
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_fibonacci_trailing_stop_batch(");
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Fibonacci Trailing Stop CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "Fibonacci Trailing Stop CPU dispatch retained an unversioned output alias"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Fibonacci Trailing Stop `value` alias remains accepted"
    );
}

#[test]
fn classic_fibonacci_trailing_stop_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "fibonacci_trailing_stop";
    const OUTPUT_IDS: [&str; 4] = ["trailing_stop", "long_stop", "short_stop", "direction"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const FIBONACCI_TRAILING_STOP_ID",
        "FIBONACCI_TRAILING_STOP_OUTPUT_IDS",
        "FIBONACCI_TRAILING_STOP_PARAMETER_KEYS",
        "resolve_fibonacci_trailing_stop_parameters(",
        "ResolvedClassicCudaLaunch::FibonacciTrailingStop",
    ] {
        assert!(
            implementation.contains(required),
            "typed Fibonacci Trailing Stop plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_fibonacci_trailing_stop_outputs_device")
            .count(),
        1,
        "all four receipts need one full resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(FIBONACCI_TRAILING_STOP_ID"),
        "production retained a primary replay beside the full resident launch"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "Fibonacci Trailing Stop `{output_id}` is missing its resident f64 capability"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_fibonacci_trailing_stop_outputs_device(",
    );
    for required in [
        ".fibonacci_trailing_stop_all_outputs(",
        "self.ohlcv.as_view()",
        "self.max_consecutive_finite_hlc",
    ] {
        assert!(
            bridge.contains(required),
            "resident bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaFibonacciTrailingStop",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "resident bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn fibonacci_trailing_stop_all_outputs(");
    for required in [
        "fibonacci_trailing_stop_batch_f64",
        "const OUTPUT_IDS: [&str; 4]",
        "d_left_bars",
        "d_right_bars",
        "d_levels",
        "d_trigger_modes",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident wrapper retained `{forbidden}`"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/fibonacci_trailing_stop_kernel.cu",
    );
    for entry_point in [
        "void fibonacci_trailing_stop_batch_f64(",
        "void fibonacci_trailing_stop_neo_batch_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("fibonacci_trailing_stop_row_f64("),
            "preserved ABI `{entry_point}` bypasses the shared f64 row authority"
        );
    }
    let row = function_body(&kernel, "void fibonacci_trailing_stop_row_f64(");
    for required in [
        "confirmed_pivot_high_at(high, len, i, left, right",
        "confirmed_pivot_low_at(low, len, i, left, right",
        "double max_value = fmax(p0, p1)",
        "double min_value = fmin(p0, p1)",
        "st = (max_value + min_value) * 0.5",
        "max_value += dif * level",
        "min_value -= dif * level",
        "st = fmin(st, max_level)",
        "st = fmax(st, min_level)",
    ] {
        assert!(
            row.contains(required),
            "shared row lost exact operation `{required}`"
        );
    }
    assert_eq!(
        kernel
            .matches("void fibonacci_trailing_stop_row_f64(")
            .count(),
        1,
        "multiple complete Fibonacci Trailing Stop f64 authorities remain"
    );
    assert!(!kernel.contains("output_id == \"value\""));
}

#[test]
fn classic_fisher_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic, list_indicators};

    const ID: &str = "fisher";
    const OUTPUT_IDS: [&str; 2] = ["fisher", "signal"];

    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "Fisher must be registered exactly once");
    let runtime = &matches[0];
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(runtime.params.len(), 1);
    let period = &runtime.params[0];
    assert_eq!(period.key, "period");
    assert!(matches!(period.kind, IndicatorParamKind::Int));
    assert!(!period.required);
    assert_eq!(period.default, Some(ParamValueStatic::Int(9)));
    assert_eq!(period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.max.is_none());
    assert_eq!(period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(period.enum_values.is_empty());
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_fisher_batch(");
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Fisher CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "Fisher CPU dispatch retained an unversioned output alias"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired Fisher `value` alias remains accepted"
    );
}

#[test]
fn classic_fisher_uses_one_shared_resident_exact_f64_pair_launch() {
    const ID: &str = "fisher";
    const OUTPUT_IDS: [&str; 2] = ["fisher", "signal"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const FISHER_ID",
        "FISHER_OUTPUT_IDS",
        "FISHER_PARAMETER_KEYS",
        "resolve_fisher_parameters(",
        "ResolvedClassicCudaLaunch::Fisher",
    ] {
        assert!(
            implementation.contains(required),
            "typed Fisher plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_fisher_outputs_device").count(),
        1,
        "both Fisher receipts need one full resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(FISHER_ID"),
        "production retained a primary replay beside the full pair launch"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "Fisher `{output_id}` is missing its resident f64 capability"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_fisher_outputs_device(");
    for required in [
        ".fisher_all_outputs(",
        "CudaDeviceHighLowF64Ref::new(",
        "self.first_valid_hl2_finite",
    ] {
        assert!(
            bridge.contains(required),
            "resident Fisher bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaFisher",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "resident Fisher bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn fisher_all_outputs(");
    for required in [
        "fisher_outputs_f64",
        "const OUTPUT_IDS: [&str; 2]",
        "d_periods",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident Fisher wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident Fisher wrapper retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/fisher_kernel.cu");
    for entry_point in [
        "void neoethos_fisher_f64(",
        "void neoethos_fisher_signal_f64(",
        "void neoethos_fisher_batch_f64(",
        "void fisher_outputs_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("fisher_row_f64_v2("),
            "preserved/full ABI `{entry_point}` bypasses the shared f64 row authority"
        );
    }
    let row = function_body(&kernel, "void fisher_row_f64_v2(");
    for required in [
        "const double qnan = fisher_qnan_f64_v2()",
        "__syncthreads()",
        "period > NEO_FISHER_F64_MAX_PERIOD",
        "fisher_row_deque_f64_v2(",
    ] {
        assert!(
            row.contains(required),
            "shared Fisher row lost v2 authority token `{required}`"
        );
    }
    let deque = function_body(
        &kernel,
        "__device__ __forceinline__ void fisher_row_deque_f64_v2(",
    );
    for required in [
        "fisher_midpoint_f64_v2(",
        "fisher_admit_midpoint_f64_v2(",
        "fisher_transition_f64_v2(",
        "fisher_reset_deques_f64_v2(",
    ] {
        assert!(
            deque.contains(required),
            "shared Fisher deque lost v2 operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void fisher_row_f64_v2(").count(),
        1,
        "multiple complete Fisher f64 row authorities remain"
    );
    assert!(!kernel.contains("output_id \"value\""));
}

#[test]
fn classic_forward_backward_exponential_oscillator_registry_and_cpu_are_canonical_only() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic, list_indicators,
    };

    const ID: &str = "forward_backward_exponential_oscillator";
    const OUTPUT_IDS: [&str; 3] = ["forward_backward", "backward", "histogram"];
    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "FBEO must be registered exactly once");
    let runtime = &matches[0];
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(runtime.params.len(), 2);
    for (parameter, key, default) in [
        (&runtime.params[0], "length", 20),
        (&runtime.params[1], "smooth", 10),
    ] {
        assert_eq!(parameter.key, key);
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(
        &cpu,
        "fn compute_forward_backward_exponential_oscillator_batch(",
    );
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "FBEO CPU dispatch is missing canonical output {canonical}"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "FBEO CPU dispatch retained an unversioned output alias"
    );
    for retired in ["value", "fb", "bwrd", "bw", "hist"] {
        assert!(
            !dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")),
            "retired FBEO alias remains accepted: {retired}"
        );
    }
}

#[test]
fn classic_forward_backward_exponential_oscillator_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "forward_backward_exponential_oscillator";
    const OUTPUT_IDS: [&str; 3] = ["forward_backward", "backward", "histogram"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID",
        "FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS",
        "FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_PARAMETER_KEYS",
        "resolve_forward_backward_exponential_oscillator_parameters(",
        "ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator",
    ] {
        assert!(
            implementation.contains(required),
            "typed FBEO plan is missing {required}"
        );
    }
    let resolver = function_body(
        &implementation,
        "fn resolve_forward_backward_exponential_oscillator_parameters(",
    );
    assert!(
        resolver.contains("for parameter in &info.params {"),
        "FBEO registry validation must borrow parameters from shared registry info"
    );
    assert!(
        !resolver.contains("for parameter in info.params {"),
        "FBEO registry validation must not move parameters out of shared registry info"
    );
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_forward_backward_exponential_oscillator_outputs_device")
            .count(),
        1,
        "all three FBEO receipts need one full resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID"),
        "production retained a primary replay beside the full triple launch"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "FBEO output is missing its resident f64 capability: {output_id}"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(
        &gpu,
        "pub fn compute_forward_backward_exponential_oscillator_outputs_device(",
    );
    for required in [
        ".forward_backward_exponential_oscillator_all_outputs(",
        "self.ohlcv.close.as_view_f64()",
        "self.finite_close_count",
    ] {
        assert!(
            bridge.contains(required),
            "resident FBEO bridge is missing {required}"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaForwardBackwardExponentialOscillator",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "resident FBEO bridge retained {forbidden}"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(
        &wrapper,
        "pub fn forward_backward_exponential_oscillator_all_outputs(",
    );
    for required in [
        "forward_backward_exponential_oscillator_batch_f64",
        "const OUTPUT_IDS: [&str; 3]",
        "d_lengths",
        "d_smooths",
        "d_ema1_buffer",
        "d_diff_buffer",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident FBEO wrapper is missing {required}"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
        "DeviceBuffer::from_slice(data",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident FBEO wrapper retained {forbidden}"
        );
    }

    let kernel = source(
        "../../vendor/vector-ta-0.2.9-patched/kernels/cuda/forward_backward_exponential_oscillator_kernel.cu",
    );
    for entry_point in [
        "void forward_backward_exponential_oscillator_batch_f64(",
        "void forward_backward_exponential_oscillator_neo_batch_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("forward_backward_exponential_oscillator_row_f64("),
            "preserved/full ABI bypasses the shared f64 row authority: {entry_point}"
        );
    }
    let row = function_body(
        &kernel,
        "void forward_backward_exponential_oscillator_row_f64(",
    );
    for required in [
        "alpha * value + (1.0 - alpha) * ema1_state",
        "ema2 += alpha * (window_value - ema2)",
        "diff_sum += diff",
        "diff_abs_sum += fabs(diff)",
        "diff_sum -= removed",
        "diff_abs_sum -= fabs(removed)",
        "(forward_backward_value - backward_value) * 0.25 + 50.0",
    ] {
        assert!(
            row.contains(required),
            "shared FBEO row lost exact scalar operation {required}"
        );
    }
    assert!(
        row.find("diff_sum += diff") < row.find("diff_sum -= removed"),
        "FBEO must preserve CPU RollingDiffWindow add-before-remove ordering"
    );
    assert_eq!(
        kernel
            .matches("void forward_backward_exponential_oscillator_row_f64(")
            .count(),
        1,
        "multiple complete FBEO f64 row authorities remain"
    );
    for retired in [
        "output_id \"value\"",
        "output_id \"fb\"",
        "output_id \"hist\"",
    ] {
        assert!(
            !kernel.contains(retired),
            "kernel retained retired alias comment {retired}"
        );
    }
}

#[test]
fn classic_fvg_trailing_stop_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic, list_indicators};

    const ID: &str = "fvg_trailing_stop";
    const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "FVG Trailing Stop must be registered exactly once"
    );
    let runtime = &matches[0];
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        [
            "unmitigated_fvg_lookback",
            "smoothing_length",
            "reset_on_cross",
        ]
    );
    for (parameter, default) in [(&runtime.params[0], 5), (&runtime.params[1], 9)] {
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }
    let reset = &runtime.params[2];
    assert!(matches!(reset.kind, IndicatorParamKind::Bool));
    assert!(!reset.required);
    assert_eq!(reset.default, Some(ParamValueStatic::Bool(false)));
    assert!(reset.min.is_none() && reset.max.is_none() && reset.step.is_none());
    assert_eq!(reset.enum_values, ["true", "false"]);
    assert!(runtime.capabilities.supports_cpu_batch);
    assert!(runtime.capabilities.supports_cuda_batch);
    assert!(runtime.capabilities.supports_cuda_vram);

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_fvg_trailing_stop_batch(");
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "FVG Trailing Stop CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "FVG Trailing Stop CPU dispatch retained an unversioned output alias"
    );
    assert!(
        !dispatch.contains("eq_ignore_ascii_case(\"value\")"),
        "retired FVG Trailing Stop `value` alias remains accepted"
    );
}

#[test]
fn classic_fvg_trailing_stop_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "fvg_trailing_stop";
    const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const FVG_TRAILING_STOP_ID",
        "FVG_TRAILING_STOP_OUTPUT_IDS",
        "FVG_TRAILING_STOP_PARAMETER_KEYS",
        "resolve_fvg_trailing_stop_parameters(",
        "ResolvedClassicCudaLaunch::FvgTrailingStop",
    ] {
        assert!(
            implementation.contains(required),
            "typed FVG Trailing Stop plan is missing `{required}`"
        );
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor
            .matches("compute_fvg_trailing_stop_outputs_device")
            .count(),
        1,
        "all four FVG Trailing Stop receipts need one full resident bridge"
    );
    assert!(
        !executor.contains("compute_primary_device(FVG_TRAILING_STOP_ID"),
        "production retained a primary replay beside the full four-output launch"
    );
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "FVG Trailing Stop output is missing its resident f64 capability: {output_id}"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_fvg_trailing_stop_outputs_device(");
    for required in [
        ".fvg_trailing_stop_all_outputs(",
        "self.ohlcv.as_view()",
        "self.first_valid_hlc",
    ] {
        assert!(
            bridge.contains(required),
            "resident FVG Trailing Stop bridge is missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaFvgTrailingStop",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "resident FVG Trailing Stop bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn fvg_trailing_stop_all_outputs(");
    for required in [
        "fvg_trailing_stop_outputs_f64",
        "const OUTPUT_IDS: [&str; 4]",
        "d_lookbacks",
        "d_smoothing_lengths",
        "d_reset_on_cross",
        "d_bull_gaps",
        "d_bear_gaps",
        "d_bull_rings",
        "d_bear_rings",
        "d_bull_ring_nan",
        "d_bear_ring_nan",
        "cols - first_valid_hlc < needed",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "resident FVG Trailing Stop wrapper is missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(high",
        "DeviceBuffer::from_slice(low",
        "DeviceBuffer::from_slice(close",
    ] {
        assert!(
            !resident.contains(forbidden),
            "resident FVG Trailing Stop wrapper retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/fvg_trailing_stop_kernel.cu");
    for entry_point in [
        "void fvg_trailing_stop_outputs_f64(",
        "void fvg_trailing_stop_neo_batch_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("fvg_trailing_stop_row_f64("),
            "preserved/full ABI bypasses the shared f64 row authority: {entry_point}"
        );
    }
    let row = function_body(&kernel, "void fvg_trailing_stop_row_f64(");
    for required in [
        "bull_acc += v",
        "bear_acc += v",
        "bull_sum -= bull_ring_vals[idx]",
        "bull_sum += x_bull",
        "bear_sum -= bear_ring_vals[idx]",
        "bear_sum += x_bear",
        "fmax(bull_disp, ts_val)",
        "fmin(bear_disp, ts_val)",
        "upper_row[i] = bear_disp",
        "lower_row[i] = bull_disp",
        "upper_ts_row[i] = ts_nz",
        "lower_ts_row[i] = ts_nz",
    ] {
        assert!(
            row.contains(required),
            "shared FVG Trailing Stop row lost exact scalar operation `{required}`"
        );
    }
    assert_eq!(
        kernel.matches("void fvg_trailing_stop_row_f64(").count(),
        1,
        "multiple complete FVG Trailing Stop f64 row authorities remain"
    );
    for retired in [
        "PERIOD-INVARIANT",
        "five identical rows",
        "output_id == \"value\"",
    ] {
        assert!(
            !kernel.contains(retired),
            "kernel retained superseded FVG Trailing Stop contract `{retired}`"
        );
    }
}

#[test]
fn classic_gatorosc_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic, list_indicators,
    };

    const ID: &str = "gatorosc";
    const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_change", "lower_change"];
    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "Gator Oscillator must be registered once");
    let runtime = &matches[0];
    assert_eq!(runtime.input_kind, IndicatorInputKind::Slice);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        [
            "jaws_length",
            "jaws_shift",
            "teeth_length",
            "teeth_shift",
            "lips_length",
            "lips_shift",
        ]
    );
    for (index, (default, minimum)) in [
        (13, 1.0_f64),
        (8, 0.0),
        (8, 1.0),
        (5, 0.0),
        (5, 1.0),
        (3, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let parameter = &runtime.params[index];
        assert!(matches!(parameter.kind, IndicatorParamKind::Int));
        assert!(!parameter.required);
        assert_eq!(parameter.default, Some(ParamValueStatic::Int(default)));
        assert_eq!(parameter.min.map(f64::to_bits), Some(minimum.to_bits()));
        assert!(parameter.max.is_none());
        assert_eq!(parameter.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
        assert!(parameter.enum_values.is_empty());
    }

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_gatorosc_batch(");
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "Gator Oscillator CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "Gator Oscillator CPU dispatch retained an unversioned output alias"
    );
    assert!(!dispatch.contains("eq_ignore_ascii_case(\"value\")"));
}

#[test]
fn classic_gatorosc_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "gatorosc";
    const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_change", "lower_change"];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const GATOROSC_ID",
        "GATOROSC_OUTPUT_IDS",
        "GATOROSC_PARAMETER_KEYS",
        "GATOROSC_SWEEP_PARAMETER_KEYS",
        "resolve_gatorosc_parameters(",
        "ResolvedClassicCudaLaunch::Gatorosc",
    ] {
        assert!(implementation.contains(required), "missing `{required}`");
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_gatorosc_outputs_device").count(),
        1,
        "all four Gator receipts need one resident full launch"
    );
    assert!(!executor.contains("compute_primary_device(GATOROSC_ID"));
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "Gator output lacks a resident f64 route: {output_id}"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_gatorosc_outputs_device(");
    for required in [
        ".gatorosc_all_outputs(",
        "self.ohlcv.close.as_view_f64()",
        "self.first_valid_close",
    ] {
        assert!(
            bridge.contains(required),
            "Gator bridge missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaGatorOsc",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "Gator bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn gatorosc_all_outputs(");
    for required in [
        "gatorosc_outputs_f64",
        "const OUTPUT_IDS: [&str; 4]",
        "d_jaws_lengths",
        "d_jaws_shifts",
        "d_teeth_lengths",
        "d_teeth_shifts",
        "d_lips_lengths",
        "d_lips_shifts",
        "d_ring_scratch",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "Gator wrapper missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
    ] {
        assert!(
            !resident.contains(forbidden),
            "Gator wrapper retained `{forbidden}`"
        );
    }

    let kernel =
        source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/gatorosc_kernel.cu");
    for entry_point in [
        "void neoethos_gatorosc_batch_f64(",
        "void gatorosc_outputs_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("gatorosc_row_f64("),
            "Gator ABI bypasses the shared f64 row: {entry_point}"
        );
    }
    let row = function_body(&kernel, "void gatorosc_row_f64(");
    for required in [
        "jema = fma(jma, jema, ja * x)",
        "tema = fma(tma, tema, ta * x)",
        "lema = fma(lma, lema, la * x)",
        "const double u = fabs(jring[jj] - tring[tt])",
        "const double l = -fabs(tring[tt] - lring[ll])",
        "upper_change[i] = u - u_prev",
        "lower_change[i] = -(l - l_prev)",
    ] {
        assert!(row.contains(required), "shared Gator row lost `{required}`");
    }
    assert_eq!(kernel.matches("void gatorosc_row_f64(").count(), 1);
    for retired in [
        "PERIOD-INVARIANT",
        "maps \"value\"",
        "one matrix",
        "NEO_S1_GATOR_MAX_SHIFT",
    ] {
        assert!(
            !kernel.contains(retired),
            "Gator kernel retained `{retired}`"
        );
    }
}

#[test]
fn classic_halftrend_registry_and_cpu_use_only_the_canonical_contract() {
    use vector_ta::indicators::registry::{
        IndicatorInputKind, IndicatorParamKind, ParamValueStatic, list_indicators,
    };

    const ID: &str = "halftrend";
    const OUTPUT_IDS: [&str; 6] = [
        "halftrend",
        "trend",
        "atr_high",
        "atr_low",
        "buy_signal",
        "sell_signal",
    ];
    let matches = list_indicators()
        .into_iter()
        .filter(|indicator| indicator.id == ID)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "HalfTrend must be registered once");
    let runtime = &matches[0];
    assert_eq!(runtime.input_kind, IndicatorInputKind::Ohlc);
    assert_eq!(
        runtime
            .outputs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>(),
        OUTPUT_IDS
    );
    assert_eq!(
        runtime
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>(),
        ["amplitude", "channel_deviation", "atr_period"]
    );
    let amplitude = &runtime.params[0];
    assert!(matches!(amplitude.kind, IndicatorParamKind::Int));
    assert!(!amplitude.required);
    assert_eq!(amplitude.default, Some(ParamValueStatic::Int(2)));
    assert_eq!(amplitude.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(amplitude.max.is_none());
    assert_eq!(amplitude.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(amplitude.enum_values.is_empty());

    let deviation = &runtime.params[1];
    assert!(matches!(deviation.kind, IndicatorParamKind::Float));
    assert!(!deviation.required);
    assert_eq!(deviation.default, Some(ParamValueStatic::Float(2.0)));
    assert_eq!(deviation.min.map(f64::to_bits), Some(0.0_f64.to_bits()));
    assert!(deviation.max.is_none());
    assert!(deviation.step.is_none());
    assert!(deviation.enum_values.is_empty());

    let atr_period = &runtime.params[2];
    assert!(matches!(atr_period.kind, IndicatorParamKind::Int));
    assert!(!atr_period.required);
    assert_eq!(atr_period.default, Some(ParamValueStatic::Int(100)));
    assert_eq!(atr_period.min.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(atr_period.max.is_none());
    assert_eq!(atr_period.step.map(f64::to_bits), Some(1.0_f64.to_bits()));
    assert!(atr_period.enum_values.is_empty());

    let cpu = source("../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs");
    let dispatch = function_body(&cpu, "fn compute_halftrend_batch(");
    for canonical in OUTPUT_IDS {
        assert!(
            dispatch.contains(&format!("eq_ignore_ascii_case(\"{canonical}\")")),
            "HalfTrend CPU dispatch is missing `{canonical}`"
        );
    }
    assert_eq!(
        dispatch.matches("eq_ignore_ascii_case(").count(),
        OUTPUT_IDS.len(),
        "HalfTrend CPU dispatch retained an unversioned output alias"
    );
    for retired in ["value", "buy", "sell"] {
        assert!(!dispatch.contains(&format!("eq_ignore_ascii_case(\"{retired}\")")));
    }
}

#[test]
fn classic_halftrend_uses_one_shared_resident_exact_f64_launch() {
    const ID: &str = "halftrend";
    const OUTPUT_IDS: [&str; 6] = [
        "halftrend",
        "trend",
        "atr_high",
        "atr_low",
        "buy_signal",
        "sell_signal",
    ];

    let implementation = source("src/core/classic_cuda_plan.rs");
    for required in [
        "const HALFTREND_ID",
        "HALFTREND_OUTPUT_IDS",
        "HALFTREND_PARAMETER_KEYS",
        "resolve_halftrend_parameters(",
        "ResolvedClassicCudaLaunch::Halftrend",
    ] {
        assert!(implementation.contains(required), "missing `{required}`");
    }
    let executor = function_body(
        &implementation,
        "pub(crate) fn execute_gpu_only_classic_plan(",
    );
    assert_eq!(
        executor.matches("compute_halftrend_outputs_device").count(),
        1,
        "all six HalfTrend receipts need one resident full launch"
    );
    assert!(!executor.contains("compute_primary_device(HALFTREND_ID"));
    for output_id in OUTPUT_IDS {
        assert!(
            has_f64_resident_output_route(ID, output_id),
            "HalfTrend output lacks a resident f64 route: {output_id}"
        );
    }

    let gpu = source("src/core/gpu_indicators.rs");
    let bridge = function_body(&gpu, "pub fn compute_halftrend_outputs_device(");
    for required in [
        ".halftrend_all_outputs(",
        "self.ohlcv.high.as_view_f64()",
        "self.ohlcv.low.as_view_f64()",
        "self.ohlcv.close.as_view_f64()",
    ] {
        assert!(
            bridge.contains(required),
            "HalfTrend bridge missing `{required}`"
        );
    }
    for forbidden in [
        "HostF64",
        "compute_cpu",
        "CudaHalfTrend",
        "CudaRuntime::new",
        "upload",
        ".synchronize()",
        "compute_primary_device",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "HalfTrend bridge retained `{forbidden}`"
        );
    }

    let wrapper = source("../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let resident = function_body(&wrapper, "pub fn halftrend_all_outputs(");
    for required in [
        "halftrend_outputs_f64",
        "const OUTPUT_IDS: [&str; 6]",
        "d_amplitudes",
        "d_channel_deviations",
        "d_atr_periods",
        "self.session.stream()",
    ] {
        assert!(
            resident.contains(required),
            "HalfTrend wrapper missing `{required}`"
        );
    }
    assert_eq!(resident.matches("launch!(").count(), 1);
    for forbidden in [
        "Context::new",
        "Stream::new",
        ".synchronize()",
        "DeviceBuffer::from_slice(prices",
    ] {
        assert!(
            !resident.contains(forbidden),
            "HalfTrend wrapper retained `{forbidden}`"
        );
    }

    let kernel = source("../../vendor/vector-ta-0.2.9-patched/kernels/cuda/halftrend_kernel.cu");
    for entry_point in [
        "void halftrend_neo_batch_f64(",
        "void halftrend_outputs_f64(",
    ] {
        let body = function_body(&kernel, entry_point);
        assert!(
            body.contains("halftrend_row_f64("),
            "HalfTrend ABI bypasses the shared exact row: {entry_point}"
        );
    }
    let row = function_body(&kernel, "void halftrend_row_f64(");
    for required in [
        "const bool classic_default =",
        "halftrend_compensated_add_f64(&sum_high, &correction_high",
        "halftrend_compensated_add_f64(&sum_low, &correction_low",
        "rma = fma(-alpha, rma, rma) + alpha * tr",
        "rma += alpha * (tr - rma)",
        "const double dev = classic_default ? fma(a, ch_half, 0.0) : a * ch_half",
        "halftrend_row[i] = up",
        "trend_row[i] = 0.0",
        "atr_high_row[i] = up + dev",
        "atr_low_row[i] = up - dev",
        "buy_signal_row[i] = up - atr2",
        "sell_signal_row[i] = down + atr2",
    ] {
        assert!(
            row.contains(required),
            "shared HalfTrend row lost exact CPU operation `{required}`"
        );
    }
    assert_eq!(kernel.matches("void halftrend_row_f64(").count(), 1);
    for retired in [
        "PERIOD-INVARIANT",
        "five identical",
        "maps \"value\"",
        "output_id == \"value\"",
    ] {
        assert!(
            !kernel.contains(retired),
            "HalfTrend kernel retained `{retired}`"
        );
    }
}
