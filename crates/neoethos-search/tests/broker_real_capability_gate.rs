use ndarray::Array2;
use neoethos_search::eval::{
    PopulationEvalInputs, SessionSpreadProfile, validation_backtest_scenarios_cpu,
};
use neoethos_search::gpu_native::population_fixture::TinyPopulationFixture;
use neoethos_search::gpu_native::prototype_population::{
    PropFirmRequirement, PrototypeBcRequirements,
};
use neoethos_search::gpu_native::prototype_population_oracle::evaluate_population_oracle;
use neoethos_search::{BacktestSettings, evaluate_population_core};

const BROKER_TRUTH_UNAVAILABLE: &str = "BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1";

#[test]
fn historical_financial_evaluation_fails_before_an_empty_population_can_short_circuit() {
    let indicators = Array2::<f64>::zeros((0, 0));
    let weights = [0.0_f64; 11];
    let mut settings = BacktestSettings::default();
    settings.spread_pips = 1.5;
    settings.commission_per_trade = 14.0;
    settings.session_spread_profile = Some(SessionSpreadProfile {
        asian_pips: 2.4,
        overlap_pips: 0.8,
        late_ny_pips: 1.4,
    });

    let result = evaluate_population_core(PopulationEvalInputs {
        close: &[],
        high: &[],
        low: &[],
        indicators: indicators.view(),
        gene_offsets: &[],
        gene_indices: &[],
        gene_weights: &[],
        long_thr: &[],
        short_thr: &[],
        month_idx: &[],
        day_idx: &[],
        timestamps: &[],
        sl_pips: &[],
        tp_pips: &[],
        stop_vol_mult: &[],
        smc_data: &[],
        gene_smc_flags: &[],
        gate_threshold: 0.0,
        weights: &weights,
        settings: &settings,
    });

    let error = match result {
        Ok(_) => panic!(
            "historical evaluation executed without synchronized Bid/Ask, conversion legs, an exact ProtoOASymbol contract, and broker deal truth"
        ),
        Err(error) => error,
    };
    assert!(
        error.contains(BROKER_TRUTH_UNAVAILABLE),
        "the refusal must be typed and versioned, got: {error}"
    );
}

#[test]
fn scenario_lane_fails_before_an_empty_work_list_can_bypass_the_gate() {
    let indicators = Array2::<f64>::zeros((0, 0));
    let weights = [0.0_f64; 11];
    let settings = BacktestSettings::default();

    let result = validation_backtest_scenarios_cpu(
        PopulationEvalInputs {
            close: &[],
            high: &[],
            low: &[],
            indicators: indicators.view(),
            gene_offsets: &[],
            gene_indices: &[],
            gene_weights: &[],
            long_thr: &[],
            short_thr: &[],
            month_idx: &[],
            day_idx: &[],
            timestamps: &[],
            sl_pips: &[],
            tp_pips: &[],
            stop_vol_mult: &[],
            smc_data: &[],
            gene_smc_flags: &[],
            gate_threshold: 0.0,
            weights: &weights,
            settings: &settings,
        },
        &[],
    );

    let error = match result {
        Ok(_) => panic!("an empty scenario list bypassed the broker-truth boundary"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains(BROKER_TRUTH_UNAVAILABLE),
        "the scenario refusal must be typed and versioned, got: {error:#}"
    );
}

#[test]
fn gpu_population_oracle_refuses_before_synthetic_cost_arithmetic() {
    let workload = TinyPopulationFixture::new(2, 128, 4)
        .population_workload(PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        })
        .expect("build structural benchmark workload");

    let error = evaluate_population_oracle(&workload)
        .expect_err("the public GPU benchmark oracle must not price synthetic trades");
    assert!(
        error.to_string().contains(BROKER_TRUTH_UNAVAILABLE),
        "GPU benchmark arithmetic bypassed the broker-truth boundary: {error}"
    );
}

#[test]
fn discovery_config_refuses_before_resolving_flat_or_session_costs() {
    let mut settings = neoethos_core::Settings::default();
    settings.risk.backtest_spread_pips = 1.5;
    settings.risk.backtest_spread_pips_asian = Some(2.4);
    settings.risk.backtest_spread_pips_overlap = Some(0.8);
    settings.risk.backtest_spread_pips_late_ny = Some(1.4);
    settings.risk.commission_per_lot = 7.0;

    let result = neoethos_search::DiscoveryConfig::try_from_settings(&settings);
    let error = match result {
        Ok(_) => panic!("discovery configuration resolved heuristic financial inputs"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains(BROKER_TRUTH_UNAVAILABLE),
        "discovery configuration must stop before spread/commission arithmetic: {error:#}"
    );
}

#[test]
fn direct_evaluation_config_resolution_is_also_fail_closed() {
    let result = neoethos_search::DiscoveryConfig::default().try_evaluation_config(None);
    let error = match result {
        Ok(_) => panic!("direct financial config resolution bypassed broker truth"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains(BROKER_TRUTH_UNAVAILABLE),
        "direct config refusal must be typed and versioned: {error:#}"
    );
}

fn assert_gate_precedes(
    source_name: &str,
    source: &str,
    scope_marker: &str,
    arithmetic_marker: &str,
) {
    let scope = source
        .find(scope_marker)
        .unwrap_or_else(|| panic!("{source_name} is missing scope marker {scope_marker:?}"));
    let scoped = &source[scope..];
    let gate = scoped
        .find("current_broker_financial_truth_capability_v1")
        .unwrap_or_else(|| panic!("{source_name} has no broker-truth gate after {scope_marker:?}"));
    let arithmetic = scoped.find(arithmetic_marker).unwrap_or_else(|| {
        panic!("{source_name} is missing arithmetic marker {arithmetic_marker:?}")
    });
    assert!(
        gate < arithmetic,
        "{source_name} reaches {arithmetic_marker:?} before its broker-truth gate"
    );
}

#[test]
fn unchecked_discovery_settings_adapter_is_crate_private_and_unused_by_production() {
    let discovery = include_str!("../src/discovery.rs");
    let discovery_config_impl = discovery
        .split_once("impl DiscoveryConfig {")
        .map(|(_, body)| body)
        .expect("DiscoveryConfig implementation must exist");
    assert!(
        discovery_config_impl.contains("pub(crate) fn from_settings("),
        "the settings adapter that resolves legacy cost fields must not be public"
    );
    assert!(
        !discovery_config_impl.contains("pub fn from_settings("),
        "the unchecked DiscoveryConfig settings adapter is still public"
    );

    for (name, source) in [
        (
            "neoethos-cli/main.rs",
            include_str!("../../neoethos-cli/src/main.rs"),
        ),
        (
            "neoethos-app/main.rs",
            include_str!("../../neoethos-app/src/main.rs"),
        ),
        (
            "neoethos-app/engines_control.rs",
            include_str!("../../neoethos-app/src/server/engines_control.rs"),
        ),
        (
            "neoethos-app/validation.rs",
            include_str!("../../neoethos-app/src/app_services/validation.rs"),
        ),
        (
            "neoethos-autoresearch/runner.rs",
            include_str!("../../neoethos-autoresearch/src/runner.rs"),
        ),
        (
            "gpu_discovery_probe.rs",
            include_str!("../examples/gpu_discovery_probe.rs"),
        ),
    ] {
        assert!(
            !source.contains("DiscoveryConfig::from_settings("),
            "{name} bypasses the checked settings adapter"
        );
    }
}

#[test]
fn every_externally_reachable_search_finance_path_is_checked() {
    let eval = include_str!("../src/eval.rs");
    let backend = include_str!("../src/backend.rs");
    let validation = include_str!("../src/validation.rs");
    let models = include_str!("../../neoethos-models/src/training_orchestrator.rs");
    let benchmark = include_str!("../src/gpu_native/benchmark.rs");
    let population_oracle = include_str!("../src/gpu_native/prototype_population_oracle.rs");
    let autoresearch_runner = include_str!("../../neoethos-autoresearch/src/runner.rs");

    for raw_api in [
        "fast_evaluate_strategy_core",
        "simulate_trades_core",
        "validation_backtest_population",
        "validation_backtest_population_cpu",
    ] {
        assert!(
            eval.contains(&format!("pub(crate) fn {raw_api}(")),
            "unchecked financial primitive {raw_api} is still public"
        );
    }

    assert_gate_precedes(
        "backend.rs",
        backend,
        "pub fn evaluate_population_core_with_backend_and_audit(",
        "backend.validate()",
    );
    assert_gate_precedes(
        "validation.rs",
        validation,
        "pub fn embargoed_walkforward_backtest(",
        "let n = close.len()",
    );
    assert_gate_precedes(
        "training_orchestrator.rs",
        models,
        "fn derive_labels(&self, ohlcv: &Ohlcv, symbol: &str)",
        "let n = ohlcv.close.len()",
    );
    assert_gate_precedes(
        "benchmark.rs",
        benchmark,
        "pub fn execute_population_benchmark<E>(",
        "let coverage_summary = eligibility.coverage()",
    );
    assert_gate_precedes(
        "prototype_population_oracle.rs",
        population_oracle,
        "pub fn evaluate_population_oracle(",
        "evaluate_population_oracle_unchecked_test_oracle(workload)",
    );
    assert_gate_precedes(
        "neoethos-autoresearch/runner.rs",
        autoresearch_runner,
        "pub fn run_with_executor(",
        "let started = Instant::now()",
    );
    for raw_api in [
        "population_settings",
        "population_settings_for_dataset",
        "validate_population_oracle_workload",
        "emit_population_events",
        "validate_population_events",
        "resolve_population_outcomes",
        "reduce_population_outcomes",
    ] {
        assert!(
            population_oracle.contains(&format!("pub(crate) fn {raw_api}(")),
            "unchecked GPU financial primitive {raw_api} is still public"
        );
    }

    for (name, source) in [(
        "autoresearch streaming",
        include_str!("../../neoethos-autoresearch/src/runner/streaming.rs"),
    )] {
        for forbidden in [
            "simulate_trades_core(",
            "validation_backtest_population(",
            "validation_backtest_population_cpu(",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} calls unchecked financial primitive {forbidden}"
            );
        }
    }

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for obsolete in [
        "examples/stop_regime_divergence.rs",
        "examples/stage_stop_agreement.rs",
        "examples/gpu_eval_bench.rs",
    ] {
        assert!(
            !manifest_dir.join(obsolete).exists(),
            "obsolete heuristic-finance example is still shipped: {obsolete}"
        );
    }

    let gpu_runbook = include_str!("../../../docs/gpu-rental-runbook.md");
    assert!(
        !gpu_runbook.contains("--example gpu_eval_bench"),
        "the active GPU runbook still advertises the removed heuristic-finance benchmark"
    );
}

#[test]
fn legacy_default_pip_size_is_not_an_external_finance_bypass() {
    let strategy_gene = include_str!("../src/genetic/strategy_gene.rs");
    let genetic_mod = include_str!("../src/genetic/mod.rs");
    let search_lib = include_str!("../src/lib.rs");
    let discovery = include_str!("../src/discovery.rs");
    let eval = include_str!("../src/eval.rs");

    assert!(
        strategy_gene.contains("pub(crate) fn default_pip_size("),
        "the symbol-name pip heuristic must be crate-private"
    );
    assert!(
        !genetic_mod.contains("default_pip_size,"),
        "the genetic module still re-exports the symbol-name pip heuristic"
    );
    assert!(
        !search_lib.contains("default_pip_size,"),
        "neoethos-search still exposes the symbol-name pip heuristic"
    );
    for private_finance_helper in [
        "pub(crate) fn round_trip_commission_per_lot(",
        "pub(crate) fn infer_market_cost_profile(",
        "pub(crate) fn for_symbol(",
    ] {
        assert!(
            strategy_gene.contains(private_finance_helper),
            "unchecked finance helper is still externally callable: {private_finance_helper}"
        );
    }
    assert!(
        discovery.contains("pub(crate) fn evaluation_config("),
        "unchecked DiscoveryConfig financial resolution is still public"
    );
    assert_gate_precedes(
        "discovery.rs",
        discovery,
        "pub fn try_evaluation_config(",
        "self.evaluation_config(price_hint)",
    );
    assert!(
        eval.contains("pub(crate) fn spread_pips_at("),
        "the active session-spread helper must stay internal to the gated evaluator"
    );
    for deleted_dead_helper in ["fn for_symbol(", "fn spread_pips_for_bar("] {
        assert!(
            !eval.contains(deleted_dead_helper),
            "superseded dead financial helper is still shipped: {deleted_dead_helper}"
        );
    }

    for (name, source) in [
        (
            "neoethos-trader/data_replay.rs",
            include_str!("../../neoethos-trader/src/data_replay.rs"),
        ),
        (
            "neoethos-models/training_orchestrator.rs",
            include_str!("../../neoethos-models/src/training_orchestrator.rs"),
        ),
        (
            "neoethos-app/live_parity.rs",
            include_str!("../../neoethos-app/src/app_services/live_parity.rs"),
        ),
    ] {
        assert!(
            !source.contains("neoethos_search::default_pip_size("),
            "{name} still calls the symbol-name pip heuristic"
        );
    }

    for (name, source) in [
        (
            "neoethos-autoresearch/runner.rs",
            include_str!("../../neoethos-autoresearch/src/runner.rs"),
        ),
        (
            "neoethos-autoresearch/runner/streaming.rs",
            include_str!("../../neoethos-autoresearch/src/runner/streaming.rs"),
        ),
        (
            "neoethos-autoresearch/proposer.rs",
            include_str!("../../neoethos-autoresearch/src/proposer.rs"),
        ),
        (
            "neoethos-autoresearch/proposal.rs",
            include_str!("../../neoethos-autoresearch/src/proposal.rs"),
        ),
    ] {
        assert!(
            !source.contains(".evaluation_config("),
            "{name} calls unchecked financial config resolution"
        );
    }
}
