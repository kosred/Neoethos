use std::fs;
use std::path::PathBuf;

fn search_source(relative: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join("src").join(relative))
        .unwrap_or_else(|error| panic!("could not read search source {relative}: {error}"))
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature `{signature}`"));
    let body_start = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function must have a body");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function `{signature}`");
}

#[test]
fn cuda_search_uses_one_f64_lane_and_routes_native_b_explicitly() {
    let identity = search_source("engine_identity.rs");
    assert!(
        !identity.contains("CubeclF32"),
        "the superseded f32 search engine must not remain an active identity"
    );
    assert!(
        identity
            .contains("PrototypeBReadiness::NotCompiledIn => Ok(PopulationEvalEngine::CubeclF64)"),
        "non-native GPU builds must resolve to the canonical f64 CubeCL lane"
    );

    let backend = search_source("backend.rs");
    let gpu_impl_start = backend
        .find("#[cfg(feature = \"gpu\")]\nfn evaluate_gpu_required_population(")
        .expect("GPU-required implementation must exist");
    let dispatch = function_body(
        &backend[gpu_impl_start..],
        "fn evaluate_gpu_required_population(",
    );
    assert!(
        dispatch.contains("PopulationEvalEngine::CudaNativeF64 =>"),
        "strict CUDA dispatch must branch on the resolved engine"
    );
    assert!(
        dispatch.contains("try_evaluate_population_b("),
        "the native-B branch must call Prototype B directly"
    );
    assert!(
        dispatch.contains("PopulationEvalEngine::CubeclF64 =>"),
        "the CubeCL branch must be explicit and f64"
    );

    let cubecl = search_source("cubecl_eval.rs");
    let cubecl_entry = function_body(&cubecl, "pub(crate) fn try_evaluate_population_cuda(");
    for stale in [
        "ArrayView2<'_, f32>",
        "gene_weights: &[f32]",
        "long_thr: &[f32]",
        "short_thr: &[f32]",
        "gate_threshold: f32",
        "smc_weights: &[f32;",
        "gpu_f64_backtest_enabled",
        "PopulationEvalEngine::CubeclF32",
    ] {
        assert!(
            !cubecl_entry.contains(stale),
            "CubeCL population entry still exposes stale precision surface `{stale}`"
        );
    }
}

#[test]
fn cuda_parity_fixtures_and_cubecl_device_gate_cannot_hide_f32_or_native_b_skip() {
    let eval = search_source("eval.rs");
    let parity_start = eval
        .find("mod gpu_cpu_parity_tests")
        .expect("GPU parity module must exist");
    let parity = &eval[parity_start..];
    for stale in ["Vec<f32>", "Array2::<f32>", "0.0f32", "as f32"] {
        assert!(
            !parity.contains(stale),
            "GPU parity fixtures still construct stale f32 input via `{stale}`"
        );
    }
    assert!(
        !parity.contains("gpu_fallback::require_gpu"),
        "device tests must not consult the retired ambient REQUIRE_GPU path"
    );
    for test in [
        "gpu_population_eval_matches_cpu",
        "gpu_cpu_prop_firm_ftmo_matches",
        "gpu_population_eval_matches_cpu_adaptive_stops",
        "gpu_population_eval_matches_cpu_heavy_rows",
        "gpu_montecarlo_batch_matches_cpu",
        "gpu_cpcv_gathered_fold_matches_cpu",
        "gpu_walkforward_split_matches_cpu",
    ] {
        let body = function_body(parity, &format!("fn {test}()"));
        assert!(
            body.contains("real_cuda_search_test_enabled"),
            "{test} must use the explicit real-card gate"
        );
    }

    let cubecl_test = function_body(&eval, "fn gpu_cubecl_trailing_stop_matches_cpu()");
    assert!(
        !cubecl_test.contains("prototype_b_available"),
        "a direct CubeCL test must not skip merely because Prototype B is available"
    );
    assert!(
        cubecl_test.contains("NEOETHOS_RUN_CUDA_SEARCH_TESTS"),
        "the real-device test needs an explicit test-only hardware gate"
    );
}

#[test]
fn direct_fused_and_prototype_a_gates_fail_loud_once_the_real_card_run_is_requested() {
    let cubecl = search_source("cubecl_eval.rs");
    let fused = function_body(
        &cubecl,
        "fn fused_path_is_byte_identical_to_windowed_path()",
    );
    assert!(
        fused.contains("NEOETHOS_RUN_CUDA_SEARCH_TESTS"),
        "the fused device gate must require the explicit real-card run switch"
    );
    assert!(
        !fused.contains("Err(_) => return"),
        "the fused device gate must not pass after a requested GPU client fails"
    );

    let prototype_a = search_source("gpu_native/prototype_a.rs");
    let direct = function_body(
        &prototype_a,
        "fn direct_prototype_a_engine_is_resident_and_matches_cpu_fixture()",
    );
    assert!(
        direct.contains("NEOETHOS_RUN_CUDA_SEARCH_TESTS"),
        "Prototype A must require the explicit real-card run switch"
    );
    assert!(
        !direct.contains("is_known_no_adapter_error")
            && direct.contains("Prototype A engine creation failed"),
        "requested Prototype A hardware proof must fail instead of skipping a missing adapter"
    );
    assert!(
        direct.contains("evaluate_test_oracle") && !direct.contains("fixture.evaluate("),
        "the direct Prototype A test must build its CPU reference through the test oracle, not \
         cross production broker admission before launching the GPU"
    );

    let prototype_a_engine = search_source("gpu_native/prototype_a_engine.rs");
    let scenario_validation = function_body(
        &prototype_a_engine,
        "fn validate_supported_scenarios(&self)",
    );
    assert!(
        scenario_validation.contains("NO_TICK_OVERRIDE")
            && scenario_validation.contains("NO_MICRO_OVERRIDE")
            && scenario_validation.contains("SCENARIO_BASE"),
        "Prototype A must accept the canonical base-scenario sentinels instead of stale zero-cost \
         descriptors"
    );
}

#[test]
fn every_native_b_parity_case_is_explicitly_gated_and_cannot_turn_device_errors_into_success() {
    let eval = search_source("eval.rs");
    let module_start = eval
        .find("mod trailing_parity_tests")
        .expect("native-B trailing parity module must exist");
    let module = &eval[module_start..];
    for test in [
        "gpu_matches_cpu_with_a_trailing_stop",
        "uniform_buckets_are_a_scalar_by_another_name",
        "gpu_matches_cpu_with_a_session_spread_profile",
    ] {
        let body = function_body(module, &format!("fn {test}()"));
        assert!(
            body.contains("real_native_cuda_search_test_enabled"),
            "{test} must use the explicit native real-card gate"
        );
        assert!(
            !body.contains("skipping") && !body.contains("no usable device"),
            "{test} must fail after a requested native device call errors"
        );
    }
}

#[test]
fn superseded_native_b_event_oracle_test_and_counter_stay_removed() {
    let prototype_b = search_source("gpu_native/prototype_b_engine.rs");
    assert!(
        !prototype_b.contains("fn cuda_population_metrics_match_the_canonical_oracle()"),
        "the retired event-pipeline oracle must not remain as an ignored native-B authority"
    );
    assert!(
        !prototype_b.contains("pub fn emitted_events(&self)"),
        "native B must not expose the retired event-buffer count as a production diagnostic"
    );

    let eval = search_source("eval.rs");
    assert!(
        eval.contains("fn gpu_matches_cpu_with_a_trailing_stop()"),
        "the walk-based CPU/native-B P&L parity replacement must remain present"
    );
}

#[test]
fn native_b_residency_is_scoped_and_released_after_every_outer_run() {
    let adapter = search_source("gpu_native/prototype_b_population_eval.rs");
    assert!(
        adapter.contains("struct NativePopulationResidencyScope"),
        "the process-wide native-B cache needs an explicit RAII lifetime"
    );
    let scope = function_body(&adapter, "fn native_population_residency_scope()");
    assert!(
        scope.contains("active_residency_scopes"),
        "nested population evaluations must share one outer residency lifetime"
    );
    let drop_impl = function_body(&adapter, "fn drop(&mut self)");
    assert!(
        drop_impl.contains("*slot = None"),
        "the last residency scope must drop the native PopulationSession"
    );
    let scenarios = function_body(&adapter, "pub(crate) fn try_evaluate_scenarios_b(");
    assert!(
        scenarios.contains("native_population_residency_scope()"),
        "a direct native-B evaluation must own a fallback scope instead of leaking a static cache"
    );

    let discovery = search_source("discovery.rs");
    for signature in [
        "pub fn run_discovery_cycle_with_holdout_and_progress<F>(",
        "pub fn run_discovery_cycle_with_progress<F>(",
    ] {
        let body = function_body(&discovery, signature);
        assert!(
            body.contains("native_population_residency_scope()"),
            "{signature} must retain native-B residency for the whole run and release it on exit"
        );
    }
}
