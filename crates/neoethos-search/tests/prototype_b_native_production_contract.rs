const CARGO_TOML: &str = include_str!("../Cargo.toml");
const EVAL_RS: &str = include_str!("../src/eval.rs");

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source boundary `{start}`"));
    let rest = &source[start..];
    let end = rest
        .find(end)
        .unwrap_or_else(|| panic!("missing source boundary `{end}` after `{start}`"));
    &rest[..end]
}

#[test]
fn prototype_b_native_production_contract_keeps_standalone_feature_cubecl_free() {
    let feature = CARGO_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("gpu-b-native ="))
        .expect("gpu-b-native feature must remain declared");

    assert!(feature.contains("gpu-b-adapter"));
    assert!(feature.contains("neoethos-gpu-cuda/cuda"));
    assert!(
        !feature.contains("\"gpu\""),
        "standalone native CUDA must not pull the generic CubeCL feature: {feature}"
    );
    assert!(
        !feature.to_ascii_lowercase().contains("cubecl"),
        "standalone native CUDA must not depend on CubeCL: {feature}"
    );
}

#[test]
fn prototype_b_native_production_contract_routes_both_canonical_callers_directly() {
    for required in [
        "fn evaluate_population_b_native_only(",
        "evaluate_population_b_native_only(&inputs, \"population_eval\")",
        "fn validation_backtest_population_native_only(",
        "evaluate_population_b_native_only(&inputs, \"validation_eval\")",
        "#[cfg(all(feature = \"gpu-b-native\", not(feature = \"gpu\")))]",
        "prototype_b_population_eval::submission_ceiling",
    ] {
        assert!(
            EVAL_RS.contains(required),
            "standalone native production route is missing `{required}`"
        );
    }

    let route = source_between(
        EVAL_RS,
        "fn evaluate_population_b_native_only(",
        "pub fn evaluate_population_core(",
    );
    assert!(route.contains("prototype_b_population_eval::try_evaluate_population_b"));
    assert!(route.contains("require_exact_native_population_rows"));
    for forbidden in [
        "cubecl",
        "try_evaluate_population_cuda",
        "gpu_fallback",
        "catch_unwind",
        "RecomputeOnCpu",
    ] {
        assert!(
            !route.contains(forbidden),
            "native-only production route must not contain `{forbidden}`"
        );
    }
}

#[test]
fn prototype_b_native_production_contract_has_no_native_cpu_error_recompute() {
    let validation = source_between(
        EVAL_RS,
        "fn validation_backtest_population_native_only(",
        "// ── Scenarios:",
    );
    assert!(validation.contains("validation_backtest_population_cpu(inputs)"));
    assert!(
        validation.contains("panic!(\"Prototype B native validation failed: {error}\")"),
        "device errors and wrong shapes must terminate the native validation lane"
    );
    assert!(
        !validation.contains("Ok(Err")
            && !validation.contains("FallbackDecision")
            && !validation.contains("RecomputeOnCpu"),
        "native validation must use CPU only when no card/explicit CPU is selected, never after a device fault"
    );
}
