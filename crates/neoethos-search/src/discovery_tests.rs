// GROUP F remediation 2026-05-25: the synthetic 10-bar alternating
// signal + ramp generators were retired in favour of the canonical
// real-data fixture in `neoethos_data::test_fixtures`. The fixture
// is a 100-bar EURUSD M1 sample seeded from a real cTrader Open API
// capture, which gives every test more realistic warm-up (longest
// indicator window is Hurst-100) and uniform behaviour across the
// workspace. See task #224.
use super::*;

// Task #66's ENV_VAR_TEST_LOCK / env_var_test_lock() helper was removed in the
// 2026-06-03 config-consolidation: the discovery tests no longer mutate
// process-global NEOETHOS_BOT_DISCOVERY_* env vars (mode + runtime knobs + the
// prop-firm gate are all config-driven now), so there is nothing to serialise.

use crate::FilteringConfig;

/// GROUP F: route the discovery tests through the canonical EURUSD
/// M1 fixture from `neoethos_data::test_fixtures` instead of the
/// 10-bar synthetic ramp. The fixture's 100-bar window satisfies
/// every indicator warm-up the discovery pipeline runs.
fn sample_feature_frame() -> FeatureFrame {
    neoethos_data::test_fixtures::ctrader_sample_feature_frame()
}

fn sample_ohlcv() -> Ohlcv {
    neoethos_data::test_fixtures::ctrader_sample_ohlcv()
}

fn sample_run_input<'a>(
    features: &'a FeatureFrame,
    ohlcv: &'a Ohlcv,
) -> CanonicalSearchRunInputV2<'a> {
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, features)
        .expect("canonical search test receipt");
    CanonicalSearchRunInputV2::new_for_test_values(receipt, features, ohlcv)
        .expect("receipt-bound canonical search test input")
}

fn sample_search_input_receipt() -> CanonicalSearchInputReceiptV2 {
    let features = sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
        .expect("canonical search test receipt")
}

fn sample_discovery_selection_scope() -> CanonicalSearchArtifactScopeV2 {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    CanonicalSearchArtifactScopeV2::from_run_input(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        &input,
    )
    .expect("canonical full discovery test scope")
}

fn sample_split_search_scopes() -> (
    CanonicalSearchInputReceiptV2,
    CanonicalSearchArtifactScopeV2,
    CanonicalSearchArtifactScopeV2,
) {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let windows = CanonicalDiscoveryRunInputs::with_holdout(&input)
        .expect("canonical fixture must produce exact 80/20 scopes");
    let selection_scope = windows.selection().scope().clone();
    let holdout_scope = windows
        .holdout()
        .expect("canonical fixture must contain a holdout")
        .scope()
        .clone();
    (input.receipt().clone(), selection_scope, holdout_scope)
}

fn sample_split_search_values() -> (
    CanonicalSearchInputReceiptV2,
    CanonicalSearchArtifactScopeV2,
    CanonicalSearchArtifactScopeV2,
    FeatureFrame,
    Ohlcv,
) {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let windows = CanonicalDiscoveryRunInputs::with_holdout(&input)
        .expect("canonical fixture must produce exact 80/20 values");
    let selection_scope = windows.selection().scope().clone();
    let holdout = windows
        .holdout()
        .expect("canonical fixture must contain holdout values");
    (
        input.receipt().clone(),
        selection_scope,
        holdout.scope().clone(),
        holdout.features().clone(),
        holdout.ohlcv().clone(),
    )
}

#[test]
fn holdout_split_stores_exact_contiguous_80_20_scopes_and_values() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .expect("canonical fixture timestamps");

    assert_eq!(input.ohlcv().len(), 100, "fixture must remain 100 rows");

    let windows = CanonicalDiscoveryRunInputs::with_holdout(&input)
        .expect("100 rows must produce an exact 80/20 split");
    let selection = windows.selection();
    let holdout = windows
        .holdout()
        .expect("split discovery input must store holdout evidence");
    let selection_window = selection.scope().evaluated_window();
    let holdout_window = holdout.scope().evaluated_window();

    assert_eq!(
        selection_window.role(),
        CanonicalSearchWindowRoleV1::InSample
    );
    assert_eq!(
        (selection_window.row_start(), selection_window.row_end()),
        (0, 80)
    );
    assert_eq!(selection_window.timestamp_start_ms(), timestamps[0]);
    assert_eq!(selection_window.timestamp_end_ms(), timestamps[79]);
    assert_eq!(holdout_window.role(), CanonicalSearchWindowRoleV1::Holdout);
    assert_eq!(
        (holdout_window.row_start(), holdout_window.row_end()),
        (80, 100)
    );
    assert_eq!(holdout_window.timestamp_start_ms(), timestamps[80]);
    assert_eq!(holdout_window.timestamp_end_ms(), timestamps[99]);

    assert_eq!(selection.scope().receipt(), input.receipt());
    assert_eq!(holdout.scope().receipt(), input.receipt());
    assert_eq!(selection.features().timestamps, features.timestamps[0..80]);
    assert_eq!(
        selection.ohlcv().timestamp.as_deref(),
        Some(&timestamps[0..80])
    );
    assert_eq!(holdout.features().timestamps, features.timestamps[80..100]);
    assert_eq!(
        holdout.ohlcv().timestamp.as_deref(),
        Some(&timestamps[80..100])
    );
}

#[test]
fn holdout_free_input_stores_exact_full_discovery_scope() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .expect("canonical fixture timestamps");

    let windows = CanonicalDiscoveryRunInputs::entire(&input)
        .expect("validated canonical input must produce a full discovery scope");
    let selection = windows.selection();
    let window = selection.scope().evaluated_window();

    assert!(windows.holdout().is_none());
    assert_eq!(window.role(), CanonicalSearchWindowRoleV1::DiscoveryInput);
    assert_eq!((window.row_start(), window.row_end()), (0, 100));
    assert_eq!(window.timestamp_start_ms(), timestamps[0]);
    assert_eq!(window.timestamp_end_ms(), timestamps[99]);
    assert_eq!(selection.scope().receipt(), input.receipt());
    assert_eq!(selection.features().timestamps, features.timestamps);
    assert_eq!(selection.ohlcv().timestamp.as_deref(), Some(timestamps));
}

#[test]
fn holdout_scope_pair_refuses_swapped_roles() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let wrong_selection = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::Holdout,
        &input,
        0..80,
    )
    .expect("valid range");
    let wrong_holdout = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::InSample,
        &input,
        80..100,
    )
    .expect("valid range");

    let error = validate_discovery_scope_pair(&input, &wrong_selection, Some(&wrong_holdout))
        .expect_err("swapped selection/holdout roles must be refused");

    assert!(
        error.to_string().contains("role"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn holdout_scope_pair_refuses_gap_and_overlap() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let holdout = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::Holdout,
        &input,
        80..100,
    )
    .expect("valid holdout range");

    for (name, selection_range) in [("gap", 0..79), ("overlap", 0..81)] {
        let selection = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::InSample,
            &input,
            selection_range,
        )
        .expect("valid selection range");
        let error = validate_discovery_scope_pair(&input, &selection, Some(&holdout))
            .expect_err("non-contiguous selection/holdout scopes must be refused");
        assert!(
            error.to_string().contains("contiguous"),
            "{name} produced unexpected error: {error:#}"
        );
    }
}

#[test]
fn holdout_constructor_refuses_empty_or_too_short_windows() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);

    let empty_selection = CanonicalDiscoveryRunInputs::with_holdout_at(&input, 0)
        .expect_err("empty in-sample window must be refused");
    assert!(
        empty_selection.to_string().contains("empty"),
        "unexpected error: {empty_selection:#}"
    );

    let too_short = CanonicalDiscoveryRunInputs::with_holdout_at(&input, 63)
        .expect_err("fewer than 64 in-sample rows must be refused");
    assert!(
        too_short.to_string().contains("at least 64"),
        "unexpected error: {too_short:#}"
    );

    let empty_holdout = CanonicalDiscoveryRunInputs::with_holdout_at(&input, 100)
        .expect_err("missing holdout suffix must be refused");
    assert!(
        empty_holdout.to_string().contains("holdout")
            && empty_holdout.to_string().contains("empty"),
        "unexpected error: {empty_holdout:#}"
    );
}

fn profitable_gene(strategy_id: &str) -> Gene {
    Gene {
        strategy_id: strategy_id.to_string(),
        indices: vec![0],
        weights: vec![1.0],
        long_threshold: 0.5,
        short_threshold: -0.5,
        fitness: 150.0,
        sharpe_ratio: 1.4,
        win_rate: 0.61,
        max_drawdown: 0.04,
        profit_factor: 1.3,
        trades_count: 10,
        consistency: 0.8,
        ..Gene::default()
    }
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("forex-discovery-{name}-{unique}.json"))
}

#[test]
fn empty_portfolio_is_an_explicit_error() {
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: Vec::new(),
        candidates: vec![Gene::default()],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let err = ensure_non_empty_portfolio(&result, "EURUSD M1")
        .expect_err("expected empty discovery portfolio to fail");
    let msg = err.to_string();
    // F-343: the message is now an actionable diagnosis. With no funnel
    // profile captured it still names the context + candidate count.
    assert!(msg.contains("no strategies"), "unexpected error: {msg}");
    assert!(msg.contains("EURUSD M1"), "unexpected error: {msg}");
    assert!(msg.contains("1 candidate"), "unexpected error: {msg}");
}

#[test]
fn non_empty_portfolio_is_accepted() {
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![Gene::default()],
        candidates: vec![Gene::default()],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    ensure_non_empty_portfolio(&result, "EURUSD M1").expect("expected non-empty portfolio to pass");
}

#[test]
fn candidate_truncation_honors_small_explicit_limits() {
    assert_eq!(candidate_truncation_limit(2, 500), 2);
    assert_eq!(candidate_truncation_limit(0, 500), 500);
    assert_eq!(candidate_truncation_limit(500, 2), 2);
    assert_eq!(candidate_truncation_limit(5, 0), 0);
}

#[test]
fn finalize_candidates_emits_selection_milestones_before_broker_truth_refusal() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let config = DiscoveryConfig {
        candidate_count: 2,
        portfolio_size: 2,
        corr_threshold: 0.9,
        min_trades_per_day: 1.0,
        filtering: FilteringConfig {
            min_profit: 1.0,
            min_trades: 1.0,
            min_sharpe: 0.1,
            min_win_rate: 0.5,
            min_profit_factor: 1.01,
            max_dd: 0.2,
            anomaly_guard: false,
            elite_mode: false,
            ..FilteringConfig::default()
        },
        ..DiscoveryConfig::default()
    };
    // `profitable_gene` is also a compact artifact fixture and intentionally
    // uses broad +/-0.5 thresholds. The canonical EURUSD sample's first
    // feature is `close_minus_open`, whose real magnitude is much smaller, so
    // those artifact-only thresholds produce no signals at all. Pin this
    // finalization test to two otherwise-identical genes that actually cross
    // the canonical feature around zero; their identical signals then exercise
    // the intended correlation-pruning milestone.
    let mut alpha_1 = profitable_gene("alpha-1");
    alpha_1.long_threshold = 0.0;
    alpha_1.short_threshold = 0.0;
    let mut alpha_2 = profitable_gene("alpha-2");
    alpha_2.long_threshold = 0.0;
    alpha_2.short_threshold = 0.0;
    let candidates = vec![alpha_1, alpha_2];
    let signal_config = config.evaluation_config_with_smc_gate(ohlcv.close.last().copied(), 0.75);
    let candidate_signals = candidates
        .iter()
        .map(|gene| {
            signals_for_gene_full(&features, &ohlcv, gene, &signal_config)
                .expect("milestone fixture signal synthesis")
        })
        .collect::<Vec<_>>();
    assert!(
        candidate_signals
            .iter()
            .all(|signals| signals.iter().any(|signal| *signal != 0)),
        "milestone genes must reach the min-trades screen"
    );
    assert_eq!(
        candidate_signals[0], candidate_signals[1],
        "the second milestone gene must be rejected by correlation"
    );
    let mut progress_events = Vec::new();
    let input = sample_run_input(&features, &ohlcv);
    let selection_scope = CanonicalSearchArtifactScopeV2::from_run_input(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        &input,
    )
    .expect("canonical full discovery test scope");
    let strict_device_admission = crate::acquire_strict_discovery_device_admission_v1()
        .expect("real strict device admission for milestone fixture");
    let population_execution_run =
        crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            strict_device_admission,
            &selection_scope,
            &features,
            &ohlcv,
        )
        .expect("seal milestone fixture population evidence");

    let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M1");
    let error = finalize_candidates_with_progress(
        candidates,
        &features,
        &ohlcv,
        input.receipt(),
        &selection_scope,
        None,
        "fnv64:0123456789abcdef",
        &config,
        0.75,
        features.names.clone(),
        &population_execution_run,
        &mut funnel,
        |event| progress_events.push(event),
    )
    .expect_err("financial validation must require exact broker evidence");
    assert!(
        error
            .to_string()
            .contains(neoethos_core::BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1),
        "unexpected finalization error: {error:#}"
    );
    assert!(progress_events.iter().any(|event| matches!(
        event,
        DiscoveryProgress::CandidatesRanked { candidate_count, truncated_to }
            if *candidate_count == 2 && *truncated_to == 2
    )));
    assert!(progress_events.iter().any(|event| matches!(
        event,
        DiscoveryProgress::CandidatesFiltered { passed_filters, evaluated_candidates, min_trades_required }
            if *passed_filters == 2 && *evaluated_candidates == 2 && *min_trades_required == 1
    )));
    assert!(progress_events.iter().any(|event| matches!(
        event,
        DiscoveryProgress::PortfolioSelected { portfolio_size, rejected_by_correlation, target_portfolio }
            if *portfolio_size == 1 && *rejected_by_correlation == 1 && *target_portfolio == 2
    )));
    assert!(
        !progress_events
            .iter()
            .any(|event| matches!(event, DiscoveryProgress::Completed { .. })),
        "broker-truth refusal must not emit a successful completion milestone"
    );
}

#[test]
fn portfolio_export_requires_validation_gates() {
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    let path = temp_path("portfolio-gates");

    let err = save_portfolio_json(&path, &result)
        .expect_err("portfolio export must fail before validation gates pass");
    assert!(err.to_string().contains("walkforward_passed"));
    assert!(!path.exists());
}

#[test]
fn portfolio_export_blocked_when_only_prop_firm_window_passed() {
    // MANDATORY-OOS regression guard (operator directive 2026-06-30): the
    // prop-firm window is an ADDITIONAL requirement, never a bypass. A
    // portfolio that cleared the window but NOT walkforward+CPCV (the exact
    // shape of the AUDUSD 20-straight-losses incident) must never export.
    let mut result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    result.validation_gates.prop_firm_window_passed = true;
    result.validation_gates.prop_firm_window_count = 50;
    result.validation_gates.prop_firm_window_pass_rate = 0.72;
    let path = temp_path("portfolio-prop-firm-export");

    let err = save_portfolio_json(&path, &result)
        .expect_err("prop-firm window alone must NOT unlock export (mandatory OOS)");
    assert!(err.to_string().contains("walkforward_passed"));
    assert!(!path.exists());
}

#[test]
fn prop_firm_gate_config_overrides_populate_discovery_config() {
    // Config-driven gate params: set models.discovery_runtime.prop_firm_gate
    // (carried on DiscoveryConfig.prop_firm_gate_params) instead of the retired
    // NEOETHOS_BOT_DISCOVERY_PROP_FIRM_* env vars. No env, no lock needed.
    let cfg = DiscoveryConfig {
        prop_firm_gate_params: neoethos_core::config::PropFirmGateConfig {
            pass_rate: 0.42,
            n_windows: 17,
            window_days: 21,
            profit_target_pct: Some(0.08),
            ..Default::default()
        },
        ..Default::default()
    }
    .apply_mode_overrides();
    let pf = cfg
        .prop_firm_gate
        .expect("default mode is PropFirm — gate must be auto-enabled");
    assert_eq!(pf.n_windows, 17);
    assert_eq!(pf.window_days, 21);
    assert!((pf.pass_rate - 0.42).abs() < 1e-9);
    assert!(pf.rules.require_profit_target);
    assert!((pf.rules.min_profit_target_pct - 0.08).abs() < 1e-9);
}

#[test]
fn prop_firm_gate_auto_enables_with_default_config() {
    // The whole point: a default config (zero overrides) still produces a
    // smart, ready-to-run prop-firm config — the FTMO baseline. No env vars
    // involved any more; the gate params come from
    // models.discovery_runtime.prop_firm_gate (all defaults here).
    let cfg = DiscoveryConfig::default().apply_mode_overrides();
    let pf = cfg.prop_firm_gate.expect("default = PropFirm mode");
    // FTMO baseline: 5%/10%/10%/5 days, 60-day window
    assert_eq!(pf.window_days, 60);
    assert_eq!(pf.n_windows, 0); // sentinel — auto-tuned at runtime
    assert!((pf.pass_rate - 0.0).abs() < 1e-9); // ranking-only by default
    // Task #66 follow-up — these constants come from
    // `PropFirmConstraints::FTMO_STANDARD` which is declared as `f32`
    // (per the prop_firm.rs domain module). Casting through `as f64`
    // introduces ~1.5e-9 rounding for values like 0.10 that aren't
    // exactly representable in f32. The previous 1e-9 tolerance
    // happened to pass for 0.05 (~7e-10 error) but failed for 0.10
    // (~1.5e-9 error). 1e-6 is well within "FTMO didn't change the
    // rules on us" semantics and survives the f32 round-trip.
    assert!((pf.rules.max_daily_loss_pct - 0.05).abs() < 1e-6);
    assert!((pf.rules.max_overall_drawdown_pct - 0.10).abs() < 1e-6);
    // 2026-06-06 RE-CALIBRATED: the discovery default per-window profit target is now the
    // operator's bar (8%/60-day window = >=4%/month), NOT the full FTMO 10% — see
    // derive_prop_firm_gate. (max_daily_loss / max_dd stay at the FTMO 5%/10% guards.)
    assert!((pf.rules.min_profit_target_pct - 0.08).abs() < 1e-6);
    assert!(pf.rules.require_profit_target);
    // Permissive filter floors should be applied automatically.
    assert!(!cfg.filtering.anomaly_guard);
    assert!(cfg.filtering.min_sharpe < 0.0);
}

#[test]
fn prop_firm_gate_disabled_in_strict_mode() {
    // Config-driven mode: select the regime via the DiscoveryConfig.mode
    // field (models.discovery_mode = "strict") instead of the retired
    // NEOETHOS_BOT_DISCOVERY_MODE env var.
    let cfg = DiscoveryConfig {
        mode: DiscoveryMode::Strict,
        ..Default::default()
    }
    .apply_mode_overrides();
    assert!(
        cfg.prop_firm_gate.is_none(),
        "strict mode must NOT auto-enable the prop-firm gate"
    );
    // Production filter floors stay intact.
    assert!(cfg.filtering.anomaly_guard);
}

#[test]
fn auto_tune_n_windows_scales_with_history() {
    // Empty / degenerate input falls back to a usable default.
    assert_eq!(auto_tune_n_windows(&[], 60), 50);
    assert_eq!(auto_tune_n_windows(&[1, 2, 3], 0), 50);

    // A two-year history with 60-day windows: 730/60 ≈ 12 spans → 36
    // windows, but the floor pushes us to 20 minimum.
    let day_ms: i64 = 86_400_000;
    let two_years: Vec<i64> = (0..730).map(|d| d * day_ms).collect();
    assert_eq!(auto_tune_n_windows(&two_years, 60), 36);

    // A five-year history → 30 spans × 3 = 90 windows.
    let five_years: Vec<i64> = (0..1_825).map(|d| d * day_ms).collect();
    assert_eq!(auto_tune_n_windows(&five_years, 60), 90);

    // A twenty-year history → would compute to 360 but caps at 200.
    let twenty_years: Vec<i64> = (0..7_300).map(|d| d * day_ms).collect();
    assert_eq!(auto_tune_n_windows(&twenty_years, 60), 200);
}

#[test]
fn holdout_portfolio_export_uses_effective_names_and_stored_selection_scope() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let input = sample_run_input(&features, &ohlcv);
    let windows = CanonicalDiscoveryRunInputs::with_holdout(&input)
        .expect("canonical fixture must produce exact holdout scopes");
    let selection_scope = windows.selection().scope().clone();
    let holdout_scope = windows
        .holdout()
        .expect("split must contain holdout scope")
        .scope()
        .clone();
    let mut result = DiscoveryResult {
        search_input_receipt: input.receipt().clone(),
        selection_scope: selection_scope.clone(),
        holdout_scope: Some(holdout_scope.clone()),
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["filtered_signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    result.validation_gates.walkforward_passed = true;
    result.validation_gates.cpcv_passed = true;
    let path = temp_path("portfolio-export");

    save_portfolio_json(&path, &result)
        .expect("portfolio export should pass once validation gates are true");
    let exported = std::fs::read_to_string(&path).expect("portfolio export should exist");
    assert!(exported.contains("filtered_signal"));
    let envelope: CanonicalSearchArtifactEnvelopeV2<Vec<serde_json::Value>> =
        CanonicalSearchArtifactEnvelopeV2::from_json_bytes(exported.as_bytes())
            .expect("portfolio export must be a strict receipt-bound envelope");
    assert_eq!(envelope.artifact_kind(), "neoethos.search-portfolio.v1");
    assert_eq!(envelope.search_config_hash(), result.search_config_hash);
    assert_eq!(
        result
            .selection_scope()
            .expect("valid stored selection scope"),
        &selection_scope
    );
    assert_eq!(
        result.holdout_scope().expect("valid stored holdout scope"),
        Some(&holdout_scope)
    );
    assert_eq!(
        envelope.scope(),
        result
            .selection_scope()
            .expect("valid stored selection scope")
    );
    assert_eq!(
        envelope.scope().evaluated_window().role(),
        CanonicalSearchWindowRoleV1::InSample
    );
    envelope
        .scope()
        .validate_against_receipt(&result.search_input_receipt)
        .expect("portfolio envelope must retain the exact discovery receipt");

    let mut swapped = result.clone();
    swapped.selection_scope = holdout_scope;
    swapped.holdout_scope = Some(selection_scope);
    let invalid_path = temp_path("portfolio-export-swapped-holdout-scopes");
    let error = save_portfolio_json(&invalid_path, &swapped)
        .expect_err("writer must refuse public literals with swapped stored scope roles");
    assert!(
        error.to_string().contains("role"),
        "unexpected swapped-scope error: {error:#}"
    );
    assert!(!invalid_path.exists());

    let _ = std::fs::remove_file(path);
}

#[test]
fn discovery_profile_exports_validation_gate_status() {
    let mut result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: vec![profitable_gene("alpha-1")],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    result.validation_gates.walkforward_passed = true;
    result.validation_gates.cpcv_passed = true;
    result.validation_gates.canonical_backtest_artifacts = 1;
    result.validation_gates.walkforward_validation_artifacts = 1;
    result.validation_gates.cpcv_fold_count = 3;
    result.validation_gates.cpcv_profitable_fold_ratio = 1.0;

    let profile = build_discovery_profile(&DiscoveryConfig::default(), &result);

    assert!(profile.walkforward_passed);
    assert!(profile.cpcv_passed);
    assert_eq!(profile.canonical_backtest_artifacts_observed, 1);
    assert_eq!(profile.walkforward_validation_artifacts_observed, 1);
    assert_eq!(profile.cpcv_fold_count, 3);
    assert_eq!(profile.cpcv_profitable_fold_ratio, 1.0);
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("forex-discovery-{name}-{unique}"))
}

fn sample_canonical_backtest_artifact(gene: &Gene) -> CanonicalBacktestArtifactFile {
    CanonicalBacktestArtifactFile::new(
        sample_discovery_selection_scope(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        gene,
        BacktestMetrics::from_metric_array([0.0; 11]),
    )
    .expect("strict canonical fixture")
}

fn sample_walkforward_summary() -> WalkforwardSummary {
    WalkforwardSummary {
        walk_forward_splits: 1,
        avg_pnl: 1.0,
        avg_win_rate: 0.5,
        avg_max_dd: 0.1,
        avg_max_consec_losses: 0.0,
        avg_daily_min_dd: 0.0,
        avg_max_daily_loss: 0.0,
        any_daily_loss_breach: false,
        any_consistency_violation: false,
        any_trade_limit_violation: false,
        all_min_trading_days_ok: true,
        splits: Vec::new(),
    }
}

const STRICT_VALIDATION_SEARCH_CONFIG_HASH: &str = "fnv64:0123456789abcdef";

fn strict_forward_test_summary(
    net_profit: f64,
    trade_count: usize,
) -> crate::validation::ForwardTestSummary {
    let mut metrics = [0.0_f64; 11];
    metrics[0] = net_profit;
    metrics[8] = trade_count as f64;
    crate::validation::ForwardTestSummary {
        bars: 20,
        metrics: BacktestMetrics::from_metric_array(metrics),
        span_days: 1.0,
    }
}

fn strict_prop_firm_summary(
    all_rules_passed: bool,
) -> crate::validation::PropFirmRiskValidationSummary {
    crate::validation::PropFirmRiskValidationSummary {
        rules: PropFirmRiskRules::default(),
        trades_observed: 1,
        trading_days_observed: 1,
        max_daily_loss_pct_observed: 0.0,
        max_overall_drawdown_pct_observed: 0.0,
        largest_profit_share_observed: 0.0,
        max_trades_per_day_observed: 1,
        net_return_pct: 0.01,
        daily_loss_breach: false,
        overall_drawdown_breach: false,
        consistency_violation: false,
        trade_limit_violation: false,
        min_trading_days_ok: true,
        profit_target_met: true,
        all_rules_passed,
    }
}

#[test]
fn validation_v2_envelopes_bind_selection_holdout_config_and_exact_gene() {
    let (_receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let gene = profitable_gene("strict-alpha");

    let canonical = CanonicalBacktestArtifactFile::new(
        selection_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        BacktestMetrics::from_metric_array([0.0; 11]),
    )
    .expect("canonical v2 envelope");
    let walkforward = WalkforwardValidationArtifactFile::new(
        selection_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        sample_walkforward_summary(),
    )
    .expect("walkforward v2 envelope");
    let forward = ForwardTestValidationArtifactFile::new(
        holdout_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        strict_forward_test_summary(5.0, 1),
    )
    .expect("forward-test v2 envelope");
    let prop = PropFirmRiskValidationArtifactFile::new(
        holdout_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        strict_prop_firm_summary(true),
    )
    .expect("prop-firm v2 envelope");

    for artifact in [canonical.scope(), walkforward.scope()] {
        assert_eq!(artifact, &selection_scope);
    }
    for artifact in [forward.scope(), prop.scope()] {
        assert_eq!(artifact, &holdout_scope);
    }
    for config_hash in [
        canonical.search_config_hash(),
        walkforward.search_config_hash(),
        forward.search_config_hash(),
        prop.search_config_hash(),
    ] {
        assert_eq!(config_hash, STRICT_VALIDATION_SEARCH_CONFIG_HASH);
    }

    let expected_gene_hash = crate::artifact_io::stable_json_hash(&gene).expect("exact gene hash");
    for identity in [
        canonical.strategy_identity(),
        walkforward.strategy_identity(),
        forward.strategy_identity(),
        prop.strategy_identity(),
    ] {
        assert_eq!(identity.strategy_id(), gene.strategy_id);
        assert_eq!(identity.exact_gene_hash(), expected_gene_hash);
    }

    canonical
        .validate_against(
            &selection_scope,
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &gene,
        )
        .expect("canonical exact authority must validate");
    walkforward
        .validate_against(
            &selection_scope,
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &gene,
        )
        .expect("walkforward exact authority must validate");
    forward
        .validate_against(&holdout_scope, STRICT_VALIDATION_SEARCH_CONFIG_HASH, &gene)
        .expect("forward-test exact authority must validate");
    prop.validate_against(&holdout_scope, STRICT_VALIDATION_SEARCH_CONFIG_HASH, &gene)
        .expect("prop-firm exact authority must validate");
}

#[test]
fn validation_v2_refuses_role_config_and_strategy_substitution() {
    let (_receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let gene = profitable_gene("strict-alpha");
    let other_gene = profitable_gene("strict-beta");
    let artifact = CanonicalBacktestArtifactFile::new(
        selection_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        BacktestMetrics::from_metric_array([0.0; 11]),
    )
    .expect("canonical v2 envelope");

    assert!(
        CanonicalBacktestArtifactFile::new(
            holdout_scope.clone(),
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &gene,
            BacktestMetrics::from_metric_array([0.0; 11]),
        )
        .is_err(),
        "canonical evidence must never accept a holdout role"
    );
    assert!(
        ForwardTestValidationArtifactFile::new(
            selection_scope.clone(),
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &gene,
            strict_forward_test_summary(5.0, 1),
        )
        .is_err(),
        "forward evidence must never accept a selection role"
    );
    assert!(
        artifact
            .validate_against(&holdout_scope, STRICT_VALIDATION_SEARCH_CONFIG_HASH, &gene,)
            .is_err(),
        "scope substitution must fail"
    );
    assert!(
        artifact
            .validate_against(&selection_scope, "fnv64:fedcba9876543210", &gene)
            .is_err(),
        "search-config substitution must fail"
    );
    assert!(
        artifact
            .validate_against(
                &selection_scope,
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                &other_gene,
            )
            .is_err(),
        "strategy substitution must fail"
    );
}

#[test]
fn validation_v2_wire_rejects_unknown_fields_and_legacy_weak_scope() {
    let (_receipt, selection_scope, _holdout_scope) = sample_split_search_scopes();
    let gene = profitable_gene("strict-alpha");
    let artifact = CanonicalBacktestArtifactFile::new(
        selection_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &gene,
        BacktestMetrics::from_metric_array([0.0; 11]),
    )
    .expect("canonical v2 envelope");
    let bytes = artifact.to_json_bytes().expect("strict v2 bytes");
    let text = std::str::from_utf8(&bytes).expect("JSON must be UTF-8");
    assert!(text.contains("search_config_hash"));
    assert!(text.contains("exact_gene_hash"));
    assert!(!text.contains("dataset_hash"));
    assert!(!text.contains("evaluation_config_hash"));
    assert!(!text.contains("temporal_scope"));

    let mut unknown_outer: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse v2 envelope fixture");
    unknown_outer["legacy_dataset_hash"] = serde_json::json!("weak");
    let unknown_outer = serde_json::to_vec(&unknown_outer).expect("serialize unknown outer");
    assert!(CanonicalBacktestArtifactFile::from_json_bytes(&unknown_outer).is_err());

    let mut unknown_payload: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse v2 envelope fixture");
    unknown_payload["payload"]["legacy_eval_hash"] = serde_json::json!("weak");
    let unknown_payload = serde_json::to_vec(&unknown_payload).expect("serialize unknown payload");
    assert!(CanonicalBacktestArtifactFile::from_json_bytes(&unknown_payload).is_err());

    let legacy = serde_json::json!({
        "artifact_kind": "canonical_strategy_backtest_artifact",
        "artifact_schema_version": 1,
        "scope": {
            "dataset_hash": "weak-dataset",
            "evaluation_config_hash": "weak-config",
            "strategy_hash": "weak-strategy",
            "temporal_scope": {
                "temporal_contract_hash": "t",
                "timestamp_policy_hash": "ts",
                "feature_availability_policy_hash": "fa",
                "label_policy_hash": "lp"
            }
        },
        "metrics": BacktestMetrics::from_metric_array([0.0; 11])
    });
    let error = CanonicalBacktestArtifactFile::from_json_bytes(
        &serde_json::to_vec(&legacy).expect("serialize legacy fixture"),
    )
    .expect_err("legacy weak v1 must fail closed");
    let message = error.to_string();
    assert!(message.contains("legacy") || message.contains("version 1"));
    assert!(
        message.contains("regenerate"),
        "unexpected error: {message}"
    );
}

fn strict_split_validation_result(portfolio: Vec<Gene>) -> DiscoveryResult {
    let (receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let canonical_backtest_artifacts = portfolio
        .iter()
        .map(|gene| {
            CanonicalBacktestArtifactFile::new(
                selection_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                BacktestMetrics::from_metric_array([0.0; 11]),
            )
            .expect("canonical v2 fixture")
        })
        .collect();
    let walkforward_validation_artifacts = portfolio
        .iter()
        .map(|gene| {
            WalkforwardValidationArtifactFile::new(
                selection_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                sample_walkforward_summary(),
            )
            .expect("walkforward v2 fixture")
        })
        .collect();
    let forward_test_validation_artifacts = portfolio
        .iter()
        .map(|gene| {
            ForwardTestValidationArtifactFile::new(
                holdout_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                strict_forward_test_summary(5.0, 1),
            )
            .expect("forward-test v2 fixture")
        })
        .collect();
    let prop_firm_validation_artifacts = portfolio
        .iter()
        .map(|gene| {
            PropFirmRiskValidationArtifactFile::new(
                holdout_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                strict_prop_firm_summary(true),
            )
            .expect("prop-firm v2 fixture")
        })
        .collect();
    let mut validation_gates = DiscoveryValidationGates::pending();
    validation_gates.walkforward_passed = true;
    validation_gates.cpcv_passed = true;
    DiscoveryResult {
        search_input_receipt: receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope),
        search_config_hash: STRICT_VALIDATION_SEARCH_CONFIG_HASH.to_owned(),
        cost_band_by_strategy: Vec::new(),
        portfolio,
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_owned()],
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        forward_test_validation_artifacts,
        prop_firm_validation_artifacts,
        funnel_profile: None,
        effective_smc_gate_threshold: f64::NAN,
    }
}

fn snapshot_tree_bytes(
    root: &std::path::Path,
) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    fn visit(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) {
        let mut entries = std::fs::read_dir(dir)
            .expect("snapshot directory must be readable")
            .map(|entry| entry.expect("snapshot entry must be readable"))
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata =
                std::fs::symlink_metadata(&path).expect("snapshot entry metadata must be readable");
            assert!(
                !metadata.file_type().is_symlink(),
                "an authoritative snapshot must never contain a symlink: {}",
                path.display()
            );
            if metadata.is_dir() {
                visit(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root)
                        .expect("snapshot entry must remain below root")
                        .to_path_buf(),
                    std::fs::read(&path).expect("snapshot member bytes must be readable"),
                );
            }
        }
    }

    let mut files = std::collections::BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[test]
fn validation_snapshot_commits_strict_content_addressed_generation_and_reuses_it_byte_for_byte() {
    let root = temp_dir("strict-validation-snapshot-idempotent");
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let result = strict_split_validation_result(vec![beta.clone(), alpha.clone()]);

    let first = crate::save_discovery_validation_snapshot(&root, &result)
        .expect("the complete result must commit one strict snapshot");
    assert!(
        first.generation_id().starts_with("fnv64-"),
        "the immutable generation leaf must be content-addressed"
    );
    let current_path = root.join("CURRENT.json");
    let current_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&current_path).expect("CURRENT must be committed last"),
    )
    .expect("CURRENT must be strict JSON");
    let current = current_json
        .as_object()
        .expect("CURRENT must be one small object");
    assert_eq!(
        current
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        ["generation_id", "manifest_hash", "schema_version"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        "CURRENT may select a generation but must not duplicate manifest authority"
    );

    let loaded = crate::load_discovery_validation_snapshot(&root)
        .expect("the committed snapshot must validate on read");
    loaded
        .validate_against(&result)
        .expect("reader must revalidate exact scopes/config/genes/evidence");
    assert_eq!(loaded.pointer(), &first);
    assert_eq!(loaded.manifest().scope(), result.selection_scope().unwrap());
    assert_eq!(
        loaded.manifest().search_config_hash(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH
    );
    assert_eq!(
        loaded.manifest().payload().holdout_scope(),
        result.holdout_scope().unwrap().unwrap()
    );
    let strategies = loaded.manifest().payload().strategies();
    assert_eq!(strategies.len(), 2);
    let strategy_hashes = strategies
        .iter()
        .map(|entry| entry.strategy_identity().exact_gene_hash())
        .collect::<Vec<_>>();
    assert!(strategy_hashes.windows(2).all(|pair| pair[0] < pair[1]));
    for entry in strategies {
        entry
            .strategy_identity()
            .validate_against(entry.gene())
            .expect("the manifest must carry each exact full gene");
        assert_eq!(entry.members().len(), 4, "one exact member per kind");
    }

    let manifest_path = loaded.generation_dir().join("manifest.json");
    let manifest_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path).expect("generation manifest must exist"),
    )
    .expect("generation manifest must be JSON");
    assert!(
        manifest_json
            .get("payload")
            .and_then(|payload| payload.get("manifest_hash"))
            .is_none(),
        "the manifest hash must live only in CURRENT and cannot hash itself"
    );
    assert_eq!(
        crate::artifact_io::stable_json_hash(&manifest_json).expect("manifest hash"),
        first.manifest_hash()
    );

    let before = snapshot_tree_bytes(&root);
    let second = crate::save_discovery_validation_snapshot(&root, &result)
        .expect("an exact existing immutable generation must be verified and reused");
    assert_eq!(second, first);
    assert_eq!(
        snapshot_tree_bytes(&root),
        before,
        "idempotent reuse is byte exact"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validation_snapshot_new_current_has_no_stale_members_from_the_previous_generation() {
    let root = temp_dir("strict-validation-snapshot-replacement");
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let first_result = strict_split_validation_result(vec![alpha.clone(), beta]);
    let second_result = strict_split_validation_result(vec![alpha]);

    let first =
        crate::save_discovery_validation_snapshot(&root, &first_result).expect("first generation");
    let second = crate::save_discovery_validation_snapshot(&root, &second_result)
        .expect("replacement generation");
    assert_ne!(first.generation_id(), second.generation_id());
    assert!(
        root.join("generations")
            .join(first.generation_id())
            .is_dir(),
        "old immutable generations may remain for audit but are not current"
    );

    let loaded = crate::load_discovery_validation_snapshot(&root)
        .expect("CURRENT must select only the replacement generation");
    assert_eq!(loaded.pointer(), &second);
    loaded
        .validate_against(&second_result)
        .expect("replacement must validate exactly");
    assert_eq!(loaded.manifest().payload().strategies().len(), 1);
    assert_eq!(
        loaded.manifest().payload().all_members().len(),
        5,
        "four per-strategy artifacts plus one promotion summary"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validation_snapshot_failures_before_current_swap_preserve_previous_authority() {
    let root = temp_dir("strict-validation-snapshot-fault");
    let original_result = strict_split_validation_result(vec![profitable_gene("strict-alpha")]);
    let replacement_result = strict_split_validation_result(vec![profitable_gene("strict-beta")]);
    let original = crate::save_discovery_validation_snapshot(&root, &original_result)
        .expect("original generation");
    let current_path = root.join("CURRENT.json");
    let current_before = std::fs::read(&current_path).expect("original CURRENT");

    for fault in [
        crate::validation_snapshot::ValidationSnapshotTestFault::AfterFirstMemberWrite,
        crate::validation_snapshot::ValidationSnapshotTestFault::BeforeCurrentSwap,
    ] {
        let error = crate::validation_snapshot::save_discovery_validation_snapshot_with_test_fault(
            &root,
            &replacement_result,
            fault,
        )
        .expect_err("an injected staging/commit fault must fail loudly");
        assert!(error.to_string().contains("injected"), "{error:#}");
        assert_eq!(
            std::fs::read(&current_path).expect("CURRENT must remain readable"),
            current_before,
            "CURRENT is swapped last and must remain byte-identical on failure"
        );
        let loaded = crate::load_discovery_validation_snapshot(&root)
            .expect("the previous generation must remain authoritative");
        assert_eq!(loaded.pointer(), &original);
        loaded.validate_against(&original_result).unwrap();
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validation_snapshot_existing_content_address_is_verified_never_overwritten() {
    let root = temp_dir("strict-validation-snapshot-no-overwrite");
    let old_result = strict_split_validation_result(vec![profitable_gene("strict-alpha")]);
    let current_result = strict_split_validation_result(vec![profitable_gene("strict-beta")]);

    let old_ref =
        crate::save_discovery_validation_snapshot(&root, &old_result).expect("old generation");
    let old_loaded = crate::load_discovery_validation_snapshot(&root).unwrap();
    let relative_member = old_loaded.manifest().payload().strategies()[0].members()[0]
        .relative_path()
        .to_path_buf();
    let current_ref = crate::save_discovery_validation_snapshot(&root, &current_result)
        .expect("current generation");
    let tampered_path = root
        .join("generations")
        .join(old_ref.generation_id())
        .join(relative_member);
    let mut tampered_bytes = std::fs::read(&tampered_path).unwrap();
    tampered_bytes.extend_from_slice(b" ");
    std::fs::write(&tampered_path, &tampered_bytes).unwrap();

    let error = crate::save_discovery_validation_snapshot(&root, &old_result)
        .expect_err("an existing content address with different bytes must refuse");
    assert!(
        error.to_string().contains("existing immutable generation"),
        "{error:#}"
    );
    assert_eq!(
        std::fs::read(&tampered_path).unwrap(),
        tampered_bytes,
        "the writer must never repair by overwriting an immutable generation"
    );
    let loaded = crate::load_discovery_validation_snapshot(&root)
        .expect("failed reuse must leave the prior CURRENT authoritative");
    assert_eq!(loaded.pointer(), &current_ref);
    loaded.validate_against(&current_result).unwrap();

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validation_snapshot_reader_rejects_extra_members_unknown_current_fields_and_unsafe_paths() {
    for unsafe_path in [
        std::path::Path::new("../escape.json"),
        std::path::Path::new("nested/../../escape.json"),
        std::path::Path::new("."),
        std::path::Path::new(""),
        std::path::Path::new("/absolute.json"),
        std::path::Path::new(r"C:\\absolute.json"),
    ] {
        assert!(
            crate::validation_snapshot::validate_snapshot_relative_path(unsafe_path).is_err(),
            "unsafe snapshot path must refuse: {}",
            unsafe_path.display()
        );
    }
    crate::validation_snapshot::validate_snapshot_relative_path(std::path::Path::new(
        "canonical/fnv64-0123456789abcdef.json",
    ))
    .expect("normal safe relative member path");

    let extra_root = temp_dir("strict-validation-snapshot-extra-member");
    let result = strict_split_validation_result(vec![profitable_gene("strict-alpha")]);
    crate::save_discovery_validation_snapshot(&extra_root, &result).unwrap();
    let loaded = crate::load_discovery_validation_snapshot(&extra_root).unwrap();
    std::fs::write(loaded.generation_dir().join("unexpected.json"), b"{}\n").unwrap();
    let extra_error = crate::load_discovery_validation_snapshot(&extra_root)
        .expect_err("unlisted generation members must fail closed");
    assert!(extra_error.to_string().contains("extra"), "{extra_error:#}");

    let missing_root = temp_dir("strict-validation-snapshot-missing-member");
    crate::save_discovery_validation_snapshot(&missing_root, &result).unwrap();
    let loaded = crate::load_discovery_validation_snapshot(&missing_root).unwrap();
    let missing_member = loaded.manifest().payload().strategies()[0].members()[0]
        .relative_path()
        .to_path_buf();
    std::fs::remove_file(loaded.generation_dir().join(missing_member)).unwrap();
    let missing_error = crate::load_discovery_validation_snapshot(&missing_root)
        .expect_err("a missing listed generation member must fail closed");
    assert!(
        missing_error.to_string().contains("missing"),
        "{missing_error:#}"
    );

    let pointer_root = temp_dir("strict-validation-snapshot-current-unknown");
    crate::save_discovery_validation_snapshot(&pointer_root, &result).unwrap();
    let current_path = pointer_root.join("CURRENT.json");
    let mut current: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&current_path).unwrap()).unwrap();
    current
        .as_object_mut()
        .unwrap()
        .insert("legacy_fallback".to_owned(), serde_json::Value::Bool(true));
    std::fs::write(&current_path, serde_json::to_vec_pretty(&current).unwrap()).unwrap();
    let pointer_error = crate::load_discovery_validation_snapshot(&pointer_root)
        .expect_err("CURRENT must deny unknown fields");
    assert!(
        pointer_error.to_string().contains("unknown field"),
        "{pointer_error:#}"
    );

    let _ = std::fs::remove_dir_all(extra_root);
    let _ = std::fs::remove_dir_all(missing_root);
    let _ = std::fs::remove_dir_all(pointer_root);
}

#[cfg(unix)]
#[test]
fn validation_snapshot_member_resolution_refuses_symlink_traversal() {
    use std::os::unix::fs::symlink;

    let root = temp_dir("strict-validation-snapshot-symlink");
    let generation = root.join("generation");
    std::fs::create_dir_all(&generation).unwrap();
    let outside = root.join("outside.json");
    std::fs::write(&outside, b"{}\n").unwrap();
    symlink(&outside, generation.join("member.json")).unwrap();

    let error = crate::validation_snapshot::resolve_snapshot_member_without_symlinks(
        &generation,
        std::path::Path::new("member.json"),
    )
    .expect_err("snapshot members may not traverse a symlink");
    assert!(error.to_string().contains("symlink"), "{error:#}");

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn orchestration_and_cli_publish_only_complete_snapshots_including_every_streaming_batch() {
    let orchestration = include_str!("orchestration.rs");
    assert_eq!(
        orchestration
            .match_indices("save_discovery_validation_snapshot(")
            .count(),
        1,
        "batch orchestration must have one canonical snapshot writer"
    );
    for superseded in [
        "save_canonical_backtest_artifacts(",
        "save_walkforward_validation_artifacts(",
        "save_forward_test_validation_artifacts(",
        "save_prop_firm_validation_artifacts(",
        "save_promotion_summary_json(",
    ] {
        assert!(
            !orchestration.contains(superseded),
            "orchestration must not publish parallel authority via {superseded}"
        );
    }

    let cli = include_str!("../../neoethos-cli/src/main.rs");
    let streaming_loop = cli
        .find("for (cursor, batch_result) in &extra")
        .expect("streaming extra-batch loop");
    let run_level_publication = cli[streaming_loop..]
        .find("StreamingRunPortfolio {")
        .map(|offset| streaming_loop + offset)
        .expect("run-level streaming publication");
    let streaming_persistence = &cli[streaming_loop..run_level_publication];
    assert!(
        streaming_persistence.contains("save_discovery_validation_snapshot"),
        "every extra surviving batch needs its own strict snapshot before run-level publication"
    );
    assert!(
        cli.contains("StreamingPromotionAuthorityV1::PerBatchLocalOnly"),
        "canonical index remapping changes exact gene hashes, so the run artifact must name the batch-local-only authority boundary"
    );
    assert!(
        cli.match_indices("save_discovery_validation_snapshot")
            .count()
            >= 2,
        "CLI must snapshot both the primary and every extra result"
    );
}

#[test]
fn validation_set_refuses_extra_strategy_before_hash_pass_or_write() {
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let mut result = strict_split_validation_result(vec![alpha]);
    result.canonical_backtest_artifacts.push(
        CanonicalBacktestArtifactFile::new(
            result.selection_scope().expect("selection scope").clone(),
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &beta,
            BacktestMetrics::from_metric_array([0.0; 11]),
        )
        .expect("extra canonical fixture"),
    );

    let hash_error = discovery_per_kind_evidence_hashes(&result)
        .expect_err("hashing must validate exact final-strategy coverage first");
    assert!(hash_error.to_string().contains("extra"), "{hash_error:#}");

    let evidence_error = live_validation_evidence_from_discovery(&result)
        .expect_err("pass/fail aggregation must validate exact bindings first");
    assert!(
        evidence_error.to_string().contains("extra"),
        "{evidence_error:#}"
    );

    let dir = temp_dir("strict-extra-strategy");
    let write_error = save_canonical_backtest_artifacts(&dir, &result)
        .expect_err("writer must validate the whole set before writing");
    assert!(write_error.to_string().contains("extra"), "{write_error:#}");
    assert!(
        !dir.exists(),
        "invalid evidence must create no output directory"
    );
}

#[test]
fn validation_hashes_are_independent_of_parallel_artifact_completion_order() {
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let ordered = strict_split_validation_result(vec![alpha, beta]);
    let mut reversed = ordered.clone();
    reversed.canonical_backtest_artifacts.reverse();
    reversed.walkforward_validation_artifacts.reverse();
    reversed.forward_test_validation_artifacts.reverse();
    reversed.prop_firm_validation_artifacts.reverse();

    let ordered_hashes =
        discovery_per_kind_evidence_hashes(&ordered).expect("ordered exact evidence hashes");
    let reversed_hashes =
        discovery_per_kind_evidence_hashes(&reversed).expect("reversed exact evidence hashes");
    assert_eq!(ordered_hashes, reversed_hashes);
}

#[test]
fn final_portfolio_prunes_selection_artifacts_by_exact_strategy_identity() {
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let result = strict_split_validation_result(vec![alpha.clone(), beta]);
    let mut canonical = result.canonical_backtest_artifacts;
    let mut walkforward = result.walkforward_validation_artifacts;

    retain_selection_validation_artifacts_for_final_portfolio(
        std::slice::from_ref(&alpha),
        &mut canonical,
        &mut walkforward,
    )
    .expect("selection artifacts must prune to the final portfolio");

    assert_eq!(canonical.len(), 1);
    assert_eq!(walkforward.len(), 1);
    assert_eq!(
        canonical[0].strategy_identity().strategy_id(),
        alpha.strategy_id
    );
    assert_eq!(
        walkforward[0].strategy_identity().strategy_id(),
        alpha.strategy_id
    );
}

#[test]
fn complete_promotion_evidence_requires_one_artifact_per_kind_and_strategy() {
    let alpha = profitable_gene("strict-alpha");
    let mut result = strict_split_validation_result(vec![alpha]);
    result.prop_firm_validation_artifacts.clear();

    let error = result
        .validate_complete_promotion_evidence()
        .expect_err("missing prop evidence must fail closed for promotion");
    assert!(error.to_string().contains("prop_firm"), "{error:#}");
    assert!(error.to_string().contains("missing"), "{error:#}");
}

#[test]
fn promotion_summary_v3_binds_exact_composite_scopes_and_validated_hashes() {
    let alpha = profitable_gene("strict-alpha");
    let result = strict_split_validation_result(vec![alpha.clone()]);
    let expected_hashes =
        discovery_per_kind_evidence_hashes(&result).expect("validated evidence hashes");
    let path = temp_path("strict-promotion-summary-v3");

    save_promotion_summary_json(&path, &result).expect("strict composite promotion summary");
    let bytes = std::fs::read(&path).expect("read promotion summary v3");
    let envelope =
        CanonicalSearchArtifactEnvelopeV2::<PromotionSummaryAuthorityPayloadV3>::from_json_bytes(
            &bytes,
        )
        .expect("strict promotion-summary v3 envelope");
    envelope
        .validate_against(
            PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
            STRICT_VALIDATION_SEARCH_CONFIG_HASH,
            &result.search_input_receipt,
            result
                .selection_scope()
                .expect("selection scope")
                .evaluated_window(),
        )
        .expect("outer promotion authority must bind exact selection scope");
    assert_eq!(
        envelope.payload().holdout_scope(),
        result
            .holdout_scope()
            .expect("valid result scopes")
            .expect("promotion requires holdout")
    );
    assert_eq!(
        envelope.payload().validation_evidence_hashes(),
        &expected_hashes
    );
    assert_eq!(envelope.payload().strategy_evidence().len(), 1);
    assert_eq!(
        envelope.payload().strategy_evidence()[0]
            .strategy_identity()
            .strategy_id(),
        alpha.strategy_id
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn promotion_summary_v3_validates_everything_before_creating_a_file() {
    let alpha = profitable_gene("strict-alpha");
    let mut result = strict_split_validation_result(vec![alpha]);
    result.forward_test_validation_artifacts.clear();
    let path = temp_path("strict-promotion-summary-missing-forward");

    let error = save_promotion_summary_json(&path, &result)
        .expect_err("incomplete composite evidence must fail closed");
    assert!(error.to_string().contains("forward_test"), "{error:#}");
    assert!(
        !path.exists(),
        "invalid promotion authority must not be written"
    );
}

fn sample_walkforward_validation_artifact(gene: &Gene) -> WalkforwardValidationArtifactFile {
    WalkforwardValidationArtifactFile::new(
        sample_discovery_selection_scope(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        gene,
        sample_walkforward_summary(),
    )
    .expect("strict walk-forward fixture")
}

#[test]
fn save_canonical_backtest_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("canonical-backtests");
    let alpha_1 = profitable_gene("alpha-1");
    let alpha_2 = profitable_gene("alpha-2");
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![alpha_1.clone(), alpha_2.clone()],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: vec![
            sample_canonical_backtest_artifact(&alpha_1),
            sample_canonical_backtest_artifact(&alpha_2),
        ],
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let written = save_canonical_backtest_artifacts(&dir, &result)
        .expect("canonical backtest artifacts should persist");
    assert_eq!(written, 2);

    let entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("backtest dir should exist")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    assert_eq!(entries.len(), 2);
    for entry in &entries {
        let payload = std::fs::read_to_string(entry.path()).expect("artifact readable");
        assert!(payload.contains(crate::validation::CANONICAL_BACKTEST_ARTIFACT_KIND));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_walkforward_validation_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("walkforward-validations");
    let alpha = profitable_gene("alpha-1");
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![alpha.clone()],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: vec![sample_walkforward_validation_artifact(&alpha)],
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let written = save_walkforward_validation_artifacts(&dir, &result)
        .expect("walk-forward validation artifacts should persist");
    assert_eq!(written, 1);

    let entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("walkforward dir should exist")
        .filter_map(|entry| entry.ok())
        .collect();
    assert_eq!(entries.len(), 1);
    let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
    assert!(payload.contains(crate::validation::WALKFORWARD_VALIDATION_ARTIFACT_KIND));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_canonical_backtest_artifacts_skips_when_empty() {
    let dir = temp_dir("canonical-backtests-empty");
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: Vec::new(),
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let written = save_canonical_backtest_artifacts(&dir, &result)
        .expect("empty canonical backtest list should be a no-op");
    assert_eq!(written, 0);
    assert!(!dir.exists());
}

#[test]
fn artifact_filename_strips_invalid_characters() {
    let name = artifact_filename_for_strategy_hash("fnv64:abc123", 0);
    assert!(!name.contains(':'));
    assert!(name.ends_with(".json"));
    assert!(name.contains("abc123"));
}

#[test]
fn discovery_runtime_overrides_defaults_match_legacy_env_defaults() {
    let defaults = DiscoveryRuntimeOverrides::default();
    // 240, not the legacy 50. The three copies of this number had drifted
    // (code 50 / root yaml 240 / desktop yaml 50) and the run artifact did not
    // record which one it had used, so "the indicator pool" meant two
    // different searches. All three now agree; see
    // `docs/pending-edits-forbidden-territory.md` §2.
    assert_eq!(defaults.prefilter_top_k, 240);
    assert!((defaults.prefilter_insample_frac - 0.80).abs() < 1e-9);
    assert_eq!(defaults.prefilter_min_per_timeframe, 6);
    assert!((defaults.funnel_stage1_pct - 0.25).abs() < 1e-9);
}

#[test]
fn discovery_runtime_overrides_clamp_invalid_values() {
    let overrides = DiscoveryRuntimeOverrides {
        prefilter_top_k: 0,
        prefilter_insample_frac: f64::NAN,
        prefilter_min_per_timeframe: 6,
        funnel_stage1_pct: 5.0,
        stage1_window: Stage1Window::Earliest,
        // Tests opt-out of the 10y minimum: synthetic fixtures don't carry
        // 10 years of bars. The pre-flight check honours min_history_years == 0
        // as the explicit "skip" sentinel (see ensure_sufficient_history).
        min_history_years: 0,
    };
    // Stale-test fix (2026-07-02): the insample-frac fallback moved 0.80 → 0.70
    // in the resolver; the assertions now track the CURRENT fallback.
    assert!((overrides.resolved_prefilter_insample_frac() - 0.70).abs() < 1e-9);
    assert!((overrides.resolved_funnel_stage1_pct() - 1.0).abs() < 1e-9);

    let too_small = DiscoveryRuntimeOverrides {
        prefilter_top_k: 0,
        prefilter_insample_frac: 0.0,
        prefilter_min_per_timeframe: 6,
        funnel_stage1_pct: 0.0001,
        stage1_window: Stage1Window::Earliest,
        min_history_years: 0,
    };
    assert!((too_small.resolved_prefilter_insample_frac() - 0.70).abs() < 1e-9);
    assert!((too_small.resolved_funnel_stage1_pct() - 0.01).abs() < 1e-9);
}

#[test]
fn default_discovery_config_does_not_read_environment() {
    // Sanity guard: the default config should be deterministic regardless
    // of the legacy env vars set by other test runners.
    let cfg = DiscoveryConfig::default();
    assert_eq!(
        cfg.runtime_overrides,
        DiscoveryRuntimeOverrides::default(),
        "default DiscoveryConfig must not pick up legacy env overrides"
    );
}

#[test]
fn discovery_profile_exports_runtime_override_resolution() {
    let mut config = DiscoveryConfig::default();
    config.runtime_overrides = DiscoveryRuntimeOverrides {
        prefilter_top_k: 17,
        prefilter_insample_frac: 0.6,
        prefilter_min_per_timeframe: 6,
        funnel_stage1_pct: 0.5,
        stage1_window: Stage1Window::Earliest,
        // Tests opt-out of the 10y minimum (synthetic fixtures, no real data).
        min_history_years: 0,
    };
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let profile = build_discovery_profile(&config, &result);
    assert_eq!(profile.prefilter_top_k, 17);
    assert!((profile.prefilter_insample_frac - 0.6).abs() < 1e-9);
    assert_eq!(profile.prefilter_min_per_timeframe, 6);
    assert!((profile.funnel_stage1_pct - 0.5).abs() < 1e-9);
}

#[test]
fn timeframe_group_classifies_multitimeframe_prefixes() {
    // Higher-TF columns are emitted as "{TF}_{indicator}" by
    // prepare_multitimeframe_features_with_options.
    assert_eq!(timeframe_group("H1_rsi_14"), Some("H1"));
    assert_eq!(timeframe_group("H4_ema_20"), Some("H4"));
    assert_eq!(timeframe_group("M15_macd_signal"), Some("M15"));
    assert_eq!(timeframe_group("D1_atr"), Some("D1"));
    assert_eq!(timeframe_group("MN1_close"), Some("MN1"));
    // Base-TF + regime columns are unprefixed → no group.
    assert_eq!(timeframe_group("rsi_14"), None);
    assert_eq!(timeframe_group("macd_signal"), None);
    assert_eq!(timeframe_group("ema_20"), None);
    assert_eq!(timeframe_group("regime_wilder_adx_14_v3"), None);
    // Uppercase base heads that are NOT timeframe labels must not match.
    assert_eq!(timeframe_group("MA_20"), None); // letters then non-digit
    assert_eq!(timeframe_group("MACD_x"), None); // 4 chars, too long
}

#[test]
fn prefilter_per_timeframe_quota_rescues_multitimeframe_features() {
    // The correlation prefilter ranks by |corr| with the BASE TF's 1-bar
    // forward return. Higher-TF columns are near-constant across base bars →
    // ~0 correlation → the global top-K discards them ALL. This test proves
    // the per-TF quota (min_per_tf > 0) force-keeps each higher-TF group, while
    // min_per_tf == 0 reproduces the legacy base-only behaviour.
    let n = 60usize;
    // Close series whose 1-bar returns alternate sign deterministically.
    let mut close = vec![100.0f64; n];
    for i in 1..n {
        let dir = if (i - 1) % 2 == 0 { 1.0 } else { -1.0 };
        close[i] = close[i - 1] * (1.0 + 0.01 * dir);
    }
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(n);
    let ohlcv = Ohlcv {
        timestamp: Some(timestamps.clone()),
        open: close.clone(),
        high: close.clone(),
        low: close.clone(),
        close: close.clone(),
        volume: Some(vec![1.0; n]),
    };

    let names = vec![
        "base_a".to_string(),
        "base_b".to_string(),
        "base_c".to_string(),
        "H1_x".to_string(),
        "H1_y".to_string(),
        "H4_z".to_string(),
    ];
    // base_* track the alternating return sign (high |corr|); H*_* are slowly
    // rising near-constant columns (~0 |corr| vs the zero-mean alternation).
    let data = ndarray::Array2::from_shape_fn((n, names.len()), |(i, j)| {
        let sign = if i % 2 == 0 { 1.0f64 } else { -1.0f64 };
        match j {
            0 | 1 | 2 => sign,
            _ => 1000.0 + (i as f64) * 0.001,
        }
    });
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
        timestamps, names, data,
    )
    .expect("valid f64 quota fixture");

    // This fixture's bars are a pure ±1% alternation with no intrabar range, so
    // nothing ever reaches a 2-ATR barrier and the first-passage label is
    // degenerate. The prefilter's degenerate-label guard therefore falls back to
    // the 1-bar forward return, which is exactly the target this test was
    // written against — so the quota assertions below still measure the quota
    // and nothing else. The first-passage target has its own tests.
    let spec = |top_k: usize, min_per_tf: usize| PrefilterSpec {
        top_k,
        insample_frac: 1.0,
        min_per_tf,
        max_hold_bars: 8,
        atr_period: 14,
        sl_atr_mult: 1.0,
        rr: 2.0,
        round_trip_cost_px: 0.0,
        // No CPCV: this fixture is 200-odd rows and the point of the test is the
        // per-timeframe quota, not the fold refit.
        cpcv: None,
    };

    // Legacy (no quota): top-3 by |corr| are the 3 base columns; no HTF.
    let (legacy, _) =
        prefilter_features(&frame, &ohlcv, &spec(3, 0)).expect("legacy prefilter succeeds");
    assert!(
        !legacy.names.iter().any(|n| timeframe_group(n).is_some()),
        "legacy prefilter should keep only base features, got {:?}",
        legacy.names
    );

    // With quota: each present higher-TF group gets at least 1 representative.
    let (quota, _) =
        prefilter_features(&frame, &ohlcv, &spec(3, 1)).expect("quota prefilter succeeds");
    assert!(
        quota.names.iter().any(|n| n.starts_with("H1_")),
        "quota prefilter must keep an H1_ feature, got {:?}",
        quota.names
    );
    assert!(
        quota.names.iter().any(|n| n.starts_with("H4_")),
        "quota prefilter must keep an H4_ feature, got {:?}",
        quota.names
    );
    // The base top-K survivors are preserved (additive, no regression).
    assert!(
        quota
            .names
            .iter()
            .filter(|n| n.starts_with("base_"))
            .count()
            >= 3
    );
}

/// A trending series with real intrabar range must produce DECIDED
/// first-passage labels — otherwise the prefilter's new target carries no
/// information and the degenerate-label guard would silently take the run back
/// to the 1-bar forward return.
#[test]
fn first_passage_labels_decide_on_a_trending_series_and_are_fully_counted() {
    let n = 600usize;
    let mut close = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    for i in 0..n {
        // Slow uptrend with a small oscillation, and a real high/low range so
        // the ATR is non-degenerate and the upper barrier is reachable.
        let c = 1.1000 + (i as f64) * 0.00005 + 0.0002 * ((i as f64) * 0.3).sin();
        close.push(c);
        high.push(c + 0.0004);
        low.push(c - 0.0004);
    }
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(n);
    let ohlcv = Ohlcv {
        timestamp: Some(timestamps.clone()),
        open: close.clone(),
        high,
        low,
        close,
        volume: Some(vec![1.0; n]),
    };
    let spec = PrefilterSpec {
        top_k: 4,
        insample_frac: 0.8,
        min_per_tf: 0,
        max_hold_bars: 40,
        atr_period: 14,
        sl_atr_mult: 1.0,
        rr: 2.0,
        round_trip_cost_px: 0.0,
        cpcv: None,
    };
    let (labels, census) = first_passage_labels(&ohlcv, &spec);
    assert_eq!(labels.long.len(), n);
    assert_eq!(labels.short.len(), n);
    // Each direction's four buckets, plus the shared undefined count, must cover
    // every bar exactly once. An uncounted bar is a silent drop.
    let counted_long = census.label_up
        + census.label_down
        + census.label_vertical
        + census.label_ambiguous
        + census.label_undefined;
    let counted_short = census.label_short_win
        + census.label_short_loss
        + census.label_vertical_short
        + census.label_ambiguous_short
        + census.label_undefined;
    assert_eq!(
        counted_long, n,
        "long census does not cover every bar ({census:?})"
    );
    assert_eq!(
        counted_short, n,
        "short census does not cover every bar ({census:?})"
    );
    assert!(
        census.label_up > 0,
        "an uptrending series with a reachable 2-ATR target must produce upper-barrier hits, \
         got {census:?}"
    );
    // The SHORT label is a different question and must be answered separately —
    // the defect this replaced ranked features by a long-only label while the GA
    // trades both directions. On an uptrend the short trade mostly stops out.
    assert!(
        census.label_short_loss > 0,
        "the short direction must be labelled at all, got {census:?}"
    );
    assert!(
        census.label_short_loss > census.label_short_win,
        "on an uptrend a short's stop should be reached more often than its target, got \
         {census:?}"
    );
}

/// The cost is charged so that a LOSS costs exactly one stop distance NET.
///
/// The barrier that used to sit at `entry - stop - cost` made a loss rarer than
/// the cost model implies; it belongs at `entry - stop + cost`, i.e. BOTH of a
/// long's barriers move up by the cost, and both of a short's move down.
#[test]
fn the_round_trip_cost_moves_both_barriers_the_same_way() {
    // Flat series: only the cost decides where the barriers sit, so a series
    // that drifts by exactly one stop-plus-cost must resolve, and one that
    // drifts by one stop-minus-cost must not.
    let n = 400usize;
    let close: Vec<f64> = (0..n).map(|i| 1.1000 - (i as f64) * 0.000_01).collect();
    let ohlcv = Ohlcv {
        timestamp: Some((0..n as i64).collect()),
        open: close.clone(),
        high: close.iter().map(|c| c + 0.000_05).collect(),
        low: close.iter().map(|c| c - 0.000_05).collect(),
        close,
        volume: Some(vec![1.0; n]),
    };
    let spec = |cost: f64| PrefilterSpec {
        top_k: 4,
        insample_frac: 0.8,
        min_per_tf: 0,
        max_hold_bars: 60,
        atr_period: 14,
        sl_atr_mult: 1.0,
        rr: 2.0,
        round_trip_cost_px: cost,
        cpcv: None,
    };
    let (_, free) = first_passage_labels(&ohlcv, &spec(0.0));
    let (_, charged) = first_passage_labels(&ohlcv, &spec(0.0005));
    // Charging cost pulls a long's STOP up (closer to a falling price), so on a
    // downtrend the long stops out at least as often as it did for free.
    assert!(
        charged.label_down >= free.label_down,
        "charging the round trip must not make a long's stop HARDER to reach: free={free:?} \
         charged={charged:?}"
    );
}

/// The refit must actually produce several fit windows when CPCV is configured,
/// and exactly one (the legacy prefix) when it is not. This is the difference
/// between a contaminated CPCV number and an honest one.
#[test]
fn prefilter_refits_inside_cpcv_folds_and_falls_back_to_a_single_prefix() {
    let base = PrefilterSpec {
        top_k: 4,
        insample_frac: 0.8,
        min_per_tf: 0,
        max_hold_bars: 40,
        atr_period: 14,
        sl_atr_mult: 1.0,
        rr: 2.0,
        round_trip_cost_px: 0.0,
        cpcv: None,
    };
    let (prefix_windows, available) = prefilter_fit_windows(10_000, &base);
    assert_eq!(
        prefix_windows.len(),
        1,
        "no CPCV configured means the OLD behaviour: one leading prefix, fit once"
    );
    assert_eq!(available, 0);

    let with_cpcv = PrefilterSpec {
        cpcv: Some((8, 2, 0.01, 0.01, 0)),
        ..base
    };
    let (fold_windows, available) = prefilter_fit_windows(10_000, &with_cpcv);
    assert!(
        fold_windows.len() > 1,
        "the ranking must be refit inside SEVERAL fold train sets, got {}",
        fold_windows.len()
    );
    assert!(
        fold_windows.len() <= PREFILTER_MAX_REFIT_FOLDS,
        "the refit is capped at {PREFILTER_MAX_REFIT_FOLDS} folds, got {}",
        fold_windows.len()
    );
    assert_eq!(
        available, 28,
        "C(8,2) = 28 folds are available; the run must report how many of them it used"
    );
    for window in &fold_windows {
        assert!(!window.is_empty(), "an empty fit window must not be kept");
        assert!(window.iter().all(|&i| i < 10_000));
    }
}

/// The whole point of the f64 pairwise rewrite: a column whose leading rows are
/// NaN must be RANKED on its finite rows, not scored 0.0 and swept aside by a
/// stable-sort tie-break. Here the NaN-prefixed column is the ONLY informative
/// one, so the old code would have discarded it.
#[test]
fn a_nan_prefixed_column_is_ranked_on_its_finite_rows_not_scored_zero() {
    let n = 800usize;
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(n);
    let mut close = Vec::with_capacity(n);
    for i in 0..n {
        close.push(1.1000 + (i as f64) * 0.00005 + 0.0002 * ((i as f64) * 0.3).sin());
    }
    let ohlcv = Ohlcv {
        timestamp: Some(timestamps.clone()),
        open: close.clone(),
        high: close.iter().map(|c| c + 0.0004).collect(),
        low: close.iter().map(|c| c - 0.0004).collect(),
        close: close.clone(),
        volume: Some(vec![1.0; n]),
    };
    let spec = PrefilterSpec {
        top_k: 1,
        insample_frac: 1.0,
        min_per_tf: 0,
        max_hold_bars: 40,
        atr_period: 14,
        sl_atr_mult: 1.0,
        rr: 2.0,
        round_trip_cost_px: 0.0,
        cpcv: None,
    };
    let (labels, _) = first_passage_labels(&ohlcv, &spec);

    // Column 0: the LONG label itself, with the leading-NaN prefix every aligned
    // higher-timeframe column carries. Column 1: pure noise.
    //
    // `.long` is named explicitly because the label is no longer one series. It
    // used to be a single long-only vector, which ranked features by what
    // predicted a one-ATR DECLINE while the GA trades both directions; it is now
    // one series per direction and a column is scored against both, keeping
    // whichever it predicts better. This test copies the long lane, so it still
    // asserts exactly what it always did — a column that IS the target must
    // survive the prefilter — and it can no longer pass by accident on a lane it
    // did not mean to name.
    let names = vec!["H1_informative".to_string(), "base_noise".to_string()];
    let data = ndarray::Array2::from_shape_fn((n, 2), |(i, j)| match j {
        0 => {
            if i < 60 {
                f64::NAN
            } else {
                labels.long[i]
            }
        }
        _ => ((i % 7) as f64) - 3.0,
    });
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
        timestamps, names, data,
    )
    .expect("valid f64 non-finite fixture");

    let (kept, census) = prefilter_features(&frame, &ohlcv, &spec).expect("prefilter succeeds");
    assert!(
        !census.label_fell_back_to_forward_return,
        "the fixture must exercise the first-passage target, not the fallback ({census:?})"
    );
    assert!(
        kept.names.iter().any(|n| n == "H1_informative"),
        "the NaN-prefixed informative column must survive the prefilter; the pre-2026-08-09 \
         f32 code scored it exactly 0.0 and dropped it. kept = {:?}, census = {census:?}",
        kept.names
    );
    assert!(
        census.columns_with_nonfinite_rows >= 1,
        "the skipped rows must be COUNTED, not hidden ({census:?})"
    );
}

#[test]
fn compute_discovery_forward_test_artifacts_returns_empty_for_empty_portfolio() {
    let config = DiscoveryConfig::default();
    let (_, _, holdout_scope, features, ohlcv) = sample_split_search_values();
    let artifacts = compute_discovery_forward_test_artifacts(
        &[],
        &features.names,
        &features,
        &ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
    )
    .expect("empty portfolio should produce zero artifacts");
    assert!(artifacts.is_empty());
}

#[test]
fn compute_discovery_forward_test_artifacts_rejects_tails_missing_features() {
    let config = DiscoveryConfig::default();
    let portfolio = vec![profitable_gene("alpha-1")];
    let (_, _, holdout_scope, mut tail_features, tail_ohlcv) = sample_split_search_values();
    tail_features.names = vec!["unrelated_feature".to_string()];
    let err = compute_discovery_forward_test_artifacts(
        &portfolio,
        &["signal".to_string()],
        &tail_features,
        &tail_ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
    )
    .expect_err("tail without the effective feature must be rejected");
    assert!(err.to_string().contains("missing feature 'signal'"));
}

#[test]
fn compute_discovery_forward_test_artifacts_produces_one_artifact_per_strategy() {
    let mut config = DiscoveryConfig::default();
    config.runtime_overrides.prefilter_top_k = 0;
    let portfolio = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
    let (_, _, holdout_scope, features, ohlcv) = sample_split_search_values();
    let artifacts = compute_discovery_forward_test_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
    )
    .expect("forward-test artifacts should build for in-band tail");
    assert_eq!(artifacts.len(), portfolio.len());
    for artifact in &artifacts {
        assert!(artifact.summary().bars > 0);
        assert!(!artifact.strategy_identity().exact_gene_hash().is_empty());
        assert_eq!(artifact.scope(), &holdout_scope);
    }
}

#[test]
fn save_forward_test_validation_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("forward-test-validations");
    let config = DiscoveryConfig::default();
    let portfolio = vec![profitable_gene("alpha-1")];
    let (receipt, selection_scope, holdout_scope, features, ohlcv) = sample_split_search_values();
    let artifacts = compute_discovery_forward_test_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
    )
    .expect("forward-test artifacts should build");

    let result = DiscoveryResult {
        search_input_receipt: receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope),
        search_config_hash: STRICT_VALIDATION_SEARCH_CONFIG_HASH.to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio,
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: features.names.clone(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: artifacts,
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let written = save_forward_test_validation_artifacts(&dir, &result)
        .expect("forward-test artifacts should persist");
    assert_eq!(written, 1);

    let entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("forward-test dir should exist")
        .filter_map(|entry| entry.ok())
        .collect();
    assert_eq!(entries.len(), 1);
    let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
    assert!(payload.contains(crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn discovery_profile_exports_forward_test_artifact_count() {
    let config = DiscoveryConfig::default();
    let (receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let gene = profitable_gene("alpha-1");
    let summary = crate::validation::ForwardTestSummary {
        bars: 20,
        metrics: BacktestMetrics::from_metric_array([0.0; 11]),
        span_days: 0.0,
    };
    let mut result = DiscoveryResult {
        search_input_receipt: receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope.clone()),
        search_config_hash: STRICT_VALIDATION_SEARCH_CONFIG_HASH.to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![gene.clone()],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: vec![
            ForwardTestValidationArtifactFile::new(
                holdout_scope,
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                &gene,
                summary,
            )
            .expect("strict forward-test fixture"),
        ],
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    result.validation_gates.walkforward_passed = true;
    result.validation_gates.cpcv_passed = true;

    let profile = build_discovery_profile(&config, &result);
    assert_eq!(profile.forward_test_validation_artifacts_observed, 1);
}

fn forward_test_artifact_with_metrics(
    gene: &Gene,
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    net_profit: f64,
    trade_count: usize,
) -> ForwardTestValidationArtifactFile {
    let mut metrics_array = [0.0_f64; 11];
    metrics_array[0] = net_profit; // net_profit
    metrics_array[8] = trade_count as f64; // trade_count
    let summary = crate::validation::ForwardTestSummary {
        bars: 20,
        metrics: BacktestMetrics::from_metric_array(metrics_array),
        span_days: 0.0,
    };
    ForwardTestValidationArtifactFile::new(
        holdout_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        gene,
        summary,
    )
    .expect("strict forward-test fixture")
}

fn empty_discovery_result_with_gates(
    walkforward_passed: bool,
    cpcv_passed: bool,
) -> DiscoveryResult {
    let mut gates = DiscoveryValidationGates::pending();
    gates.walkforward_passed = walkforward_passed;
    gates.cpcv_passed = cpcv_passed;
    DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: Vec::new(),
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: gates,
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    }
}

#[test]
fn evidence_bridge_mirrors_discovery_validation_gates_with_no_forward_test_artifacts() {
    let result = empty_discovery_result_with_gates(true, true);
    let error = live_validation_evidence_from_discovery(&result)
        .expect_err("live evidence must reject a missing final portfolio/evidence set");
    assert!(error.to_string().contains("missing"), "{error:#}");
}

#[test]
fn evidence_bridge_marks_forward_test_passed_when_every_artifact_is_profitable() {
    let result = strict_split_validation_result(vec![
        profitable_gene("strict-alpha"),
        profitable_gene("strict-beta"),
    ]);
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert_eq!(evidence.forward_test_passed, Some(true));
}

#[test]
fn evidence_bridge_marks_forward_test_failed_when_any_artifact_is_unprofitable() {
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let mut result = strict_split_validation_result(vec![alpha, beta.clone()]);
    let holdout_scope = result
        .holdout_scope
        .as_ref()
        .expect("holdout scope")
        .clone();
    result.forward_test_validation_artifacts[1] =
        forward_test_artifact_with_metrics(&beta, &holdout_scope, -10.0, 2);
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert_eq!(evidence.forward_test_passed, Some(false));
}

#[test]
fn evidence_bridge_marks_forward_test_failed_when_artifact_has_zero_trades() {
    let alpha = profitable_gene("strict-alpha");
    let mut result = strict_split_validation_result(vec![alpha.clone()]);
    let holdout_scope = result
        .holdout_scope
        .as_ref()
        .expect("holdout scope")
        .clone();
    result.forward_test_validation_artifacts[0] =
        forward_test_artifact_with_metrics(&alpha, &holdout_scope, 5.0, 0);
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert_eq!(evidence.forward_test_passed, Some(false));
}

#[test]
fn evidence_bridge_propagates_failed_walkforward_and_cpcv() {
    let mut result = strict_split_validation_result(vec![profitable_gene("strict-alpha")]);
    result.validation_gates.walkforward_passed = false;
    result.validation_gates.cpcv_passed = false;
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert!(!evidence.walkforward_passed);
    assert!(!evidence.cpcv_passed);
}

#[test]
fn evidence_bridge_marks_prop_firm_passed_when_every_artifact_passes() {
    let result = strict_split_validation_result(vec![
        profitable_gene("strict-alpha"),
        profitable_gene("strict-beta"),
    ]);
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert_eq!(evidence.prop_firm_passed, Some(true));
}

#[test]
fn evidence_bridge_marks_prop_firm_failed_when_any_artifact_fails() {
    let alpha = profitable_gene("strict-alpha");
    let beta = profitable_gene("strict-beta");
    let mut result = strict_split_validation_result(vec![alpha, beta.clone()]);
    let holdout_scope = result
        .holdout_scope
        .as_ref()
        .expect("holdout scope")
        .clone();
    result.prop_firm_validation_artifacts[1] = PropFirmRiskValidationArtifactFile::new(
        holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &beta,
        strict_prop_firm_summary(false),
    )
    .expect("strict failing prop-firm fixture");
    let evidence = live_validation_evidence_from_discovery(&result)
        .expect("complete exact evidence must aggregate");
    assert_eq!(evidence.prop_firm_passed, Some(false));
}

#[test]
fn compute_discovery_prop_firm_artifacts_returns_empty_for_empty_portfolio() {
    let config = DiscoveryConfig::default();
    let (_, _, holdout_scope, features, ohlcv) = sample_split_search_values();
    let artifacts = compute_discovery_prop_firm_artifacts(
        &[],
        &features.names,
        &features,
        &ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
        PropFirmRiskRules::default(),
    )
    .expect("empty portfolio should produce zero artifacts");
    assert!(artifacts.is_empty());
}

#[test]
fn compute_discovery_prop_firm_artifacts_rejects_tails_missing_features() {
    let config = DiscoveryConfig::default();
    let portfolio = vec![profitable_gene("alpha-1")];
    let (_, _, holdout_scope, mut tail_features, tail_ohlcv) = sample_split_search_values();
    tail_features.names = vec!["unrelated_feature".to_string()];
    let err = compute_discovery_prop_firm_artifacts(
        &portfolio,
        &["signal".to_string()],
        &tail_features,
        &tail_ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
        PropFirmRiskRules::default(),
    )
    .expect_err("tail without the effective feature must be rejected");
    assert!(err.to_string().contains("missing feature 'signal'"));
}

#[test]
fn compute_discovery_prop_firm_artifacts_produces_one_artifact_per_strategy() {
    let mut config = DiscoveryConfig::default();
    config.runtime_overrides.prefilter_top_k = 0;
    let portfolio = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
    let (_, _, holdout_scope, features, ohlcv) = sample_split_search_values();
    let artifacts = compute_discovery_prop_firm_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &holdout_scope,
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &config,
        PropFirmRiskRules::default(),
    )
    .expect("prop-firm artifacts should build");
    assert_eq!(artifacts.len(), portfolio.len());
    for artifact in &artifacts {
        assert!(!artifact.strategy_identity().exact_gene_hash().is_empty());
        assert_eq!(artifact.scope(), &holdout_scope);
    }
}

#[test]
fn save_prop_firm_validation_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("prop-firm-validations");
    let (receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let alpha = profitable_gene("alpha-1");
    let artifact = PropFirmRiskValidationArtifactFile::new(
        holdout_scope.clone(),
        STRICT_VALIDATION_SEARCH_CONFIG_HASH,
        &alpha,
        strict_prop_firm_summary(true),
    )
    .expect("strict prop-firm fixture");
    let result = DiscoveryResult {
        search_input_receipt: receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope),
        search_config_hash: STRICT_VALIDATION_SEARCH_CONFIG_HASH.to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![alpha],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: vec![artifact],
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let written = save_prop_firm_validation_artifacts(&dir, &result)
        .expect("prop-firm artifacts should persist");
    assert_eq!(written, 1);

    let entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("prop-firm dir should exist")
        .filter_map(|entry| entry.ok())
        .collect();
    assert_eq!(entries.len(), 1);
    let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
    assert!(payload.contains(crate::validation::PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND));

    let _ = std::fs::remove_dir_all(&dir);
}

fn populated_discovery_result(
    canonical_count: usize,
    walkforward_count: usize,
    forward_test_count: usize,
    prop_firm_count: usize,
) -> DiscoveryResult {
    let (receipt, selection_scope, holdout_scope) = sample_split_search_scopes();
    let strategy_count = canonical_count
        .max(walkforward_count)
        .max(forward_test_count)
        .max(prop_firm_count)
        .max(1);
    let portfolio = (0..strategy_count)
        .map(|idx| profitable_gene(&format!("strict-{idx}")))
        .collect::<Vec<_>>();
    let canonical_backtest_artifacts = portfolio
        .iter()
        .take(canonical_count)
        .map(|gene| {
            CanonicalBacktestArtifactFile::new(
                selection_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                BacktestMetrics::from_metric_array([0.0; 11]),
            )
            .expect("strict canonical fixture")
        })
        .collect();
    let walkforward_validation_artifacts = portfolio
        .iter()
        .take(walkforward_count)
        .map(|gene| {
            WalkforwardValidationArtifactFile::new(
                selection_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                sample_walkforward_summary(),
            )
            .expect("strict walk-forward fixture")
        })
        .collect();
    let forward_test_validation_artifacts = portfolio
        .iter()
        .take(forward_test_count)
        .map(|gene| forward_test_artifact_with_metrics(gene, &holdout_scope, 1.0, 1))
        .collect();
    let prop_firm_validation_artifacts = portfolio
        .iter()
        .take(prop_firm_count)
        .map(|gene| {
            PropFirmRiskValidationArtifactFile::new(
                holdout_scope.clone(),
                STRICT_VALIDATION_SEARCH_CONFIG_HASH,
                gene,
                strict_prop_firm_summary(true),
            )
            .expect("strict prop-firm fixture")
        })
        .collect();
    DiscoveryResult {
        search_input_receipt: receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope),
        search_config_hash: STRICT_VALIDATION_SEARCH_CONFIG_HASH.to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio,
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        forward_test_validation_artifacts,
        prop_firm_validation_artifacts,
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    }
}

#[test]
fn discovery_validation_evidence_manifest_rejects_missing_live_sim_evidence() {
    let result = populated_discovery_result(1, 1, 1, 1);
    let err = discovery_validation_evidence_manifest(&result)
        .expect_err("manifest must surface missing live-sim evidence");
    assert!(err.to_string().contains("live_execution_simulation_hash"));
}

#[test]
fn discovery_validation_evidence_manifest_rejects_missing_walkforward_evidence() {
    let result = populated_discovery_result(1, 0, 1, 1);
    let err = discovery_validation_evidence_manifest(&result)
        .expect_err("manifest must surface missing walkforward evidence");
    assert!(err.to_string().contains("walkforward_validation_hash"));
}

#[test]
fn discovery_per_kind_evidence_hashes_returns_some_only_for_present_kinds() {
    let result = populated_discovery_result(1, 0, 1, 1);
    let hashes = discovery_per_kind_evidence_hashes(&result)
        .expect("per-kind hash extraction should succeed");
    assert!(hashes.canonical_backtest.is_some());
    assert!(hashes.walkforward.is_none());
    assert!(hashes.forward_test.is_some());
    assert!(hashes.prop_firm.is_some());
    assert!(hashes.live_execution_simulation.is_none());
}

#[test]
fn discovery_per_kind_evidence_hashes_returns_none_for_empty_result() {
    let result = populated_discovery_result(0, 0, 0, 0);
    let hashes = discovery_per_kind_evidence_hashes(&result)
        .expect("per-kind hash extraction should succeed");
    assert!(hashes.canonical_backtest.is_none());
    assert!(hashes.walkforward.is_none());
    assert!(hashes.forward_test.is_none());
    assert!(hashes.prop_firm.is_none());
    assert!(hashes.live_execution_simulation.is_none());
}

#[test]
fn lossy_manifest_accepts_complete_producer_side_evidence() {
    let result = populated_discovery_result(1, 1, 1, 1);
    let manifest = discovery_validation_evidence_manifest_excluding_live_sim(&result)
        .expect("lossy manifest should accept complete producer-side evidence");
    assert!(
        manifest
            .live_execution_simulation_hash
            .starts_with("deferred:")
    );
}

#[test]
fn lossy_manifest_still_rejects_missing_producer_side_evidence() {
    let result = populated_discovery_result(1, 0, 1, 1);
    let err = discovery_validation_evidence_manifest_excluding_live_sim(&result)
        .expect_err("lossy manifest must still reject missing walk-forward");
    assert!(err.to_string().contains("walkforward_validation_hash"));
}

#[test]
fn all_producer_kinds_present_ignores_live_sim() {
    let hashes = DiscoveryPerKindEvidenceHashes {
        canonical_backtest: Some("h1".into()),
        walkforward: Some("h2".into()),
        forward_test: Some("h3".into()),
        prop_firm: Some("h4".into()),
        live_execution_simulation: None,
    };
    assert!(hashes.all_producer_kinds_present());
    assert!(!hashes.all_present());
}

#[test]
fn full_validation_chain_with_complete_producer_evidence_passes_lossy_manifest() {
    // Build a result with all four producer-side artifact kinds populated.
    let result = populated_discovery_result(2, 2, 2, 2);

    // 1. Per-kind hashes know which kinds are present.
    let hashes = discovery_per_kind_evidence_hashes(&result)
        .expect("per-kind hash extraction should succeed");
    assert!(hashes.canonical_backtest.is_some());
    assert!(hashes.walkforward.is_some());
    assert!(hashes.forward_test.is_some());
    assert!(hashes.prop_firm.is_some());
    assert!(hashes.live_execution_simulation.is_none());
    assert!(hashes.all_producer_kinds_present());
    assert!(!hashes.all_present()); // live-sim missing keeps full check off

    // 2. Strict manifest rejects on missing live-sim.
    let strict_err = discovery_validation_evidence_manifest(&result)
        .expect_err("strict manifest must reject when live-sim hash is empty");
    assert!(strict_err.to_string().contains("live_execution_simulation"));

    // 3. Lossy manifest accepts the same result.
    let lossy = discovery_validation_evidence_manifest_excluding_live_sim(&result)
        .expect("lossy manifest accepts complete producer-side evidence");
    assert!(
        lossy
            .live_execution_simulation_hash
            .starts_with("deferred:")
    );

    // 4. Evidence bridge surfaces the producer-side outcomes.
    let mut result_for_evidence = result.clone();
    result_for_evidence.validation_gates.walkforward_passed = true;
    result_for_evidence.validation_gates.cpcv_passed = true;
    let evidence = live_validation_evidence_from_discovery(&result_for_evidence)
        .expect("complete exact producer evidence must aggregate");
    assert!(evidence.walkforward_passed);
    assert!(evidence.cpcv_passed);
    assert_eq!(evidence.forward_test_passed, Some(true));
    assert_eq!(evidence.prop_firm_passed, Some(true));
    assert!(evidence.live_sim_runtime_model_hash.is_none());

    // 5. Profile carries the same data without re-deriving anything.
    let profile = build_discovery_profile(&DiscoveryConfig::default(), &result_for_evidence);
    // The Phase 49 prop-firm count IS sourced from the artifact
    // vector directly (not from validation_gates), so it should
    // reflect the constructed fixture.
    assert_eq!(profile.prop_firm_validation_artifacts_observed, 2);
    assert_eq!(profile.forward_test_validation_artifacts_observed, 2);
    assert!(!profile.validation_evidence_complete); // live-sim still missing
    assert!(
        profile
            .validation_evidence_missing_kinds
            .iter()
            .any(|k| k == "live_execution_simulation")
    );
    // Producer-side completeness is true (all four kinds present).
    assert!(
        profile
            .validation_evidence_hashes
            .all_producer_kinds_present()
    );
}

#[test]
fn discovery_run_profile_records_typed_determinism_policy() {
    // The OnceLock-installed determinism policy may carry whatever
    // any earlier test in this process installed, so we assert only
    // that the profile carries one of the three legal variants —
    // every one of which is serializable, which is the property the
    // promotion-readiness runbook documents.
    let config = DiscoveryConfig::default();
    let result = populated_discovery_result(0, 0, 0, 0);
    let profile = build_discovery_profile(&config, &result);
    match profile.determinism_policy {
        DeterminismPolicy::Deterministic { seed: _ }
        | DeterminismPolicy::BestEffort
        | DeterminismPolicy::NonDeterministicAllowed => {}
    }
}

/// A legacy profile cannot mint engine evidence from process-global state.
#[test]
fn discovery_run_profile_leaves_engine_identity_empty_without_a_run_receipt() {
    let profile = build_discovery_profile(
        &DiscoveryConfig::default(),
        &populated_discovery_result(0, 0, 0, 0),
    );
    assert!(profile.population_eval_engines.is_empty());
    let json = serde_json::to_string(&profile).unwrap();
    assert!(json.contains("\"population_eval_engines\":[]"), "{json}");
}

#[test]
fn discovery_run_profile_persists_the_full_run_scoped_execution_receipt_v2() {
    let mut result = populated_discovery_result(0, 0, 0, 0);
    let run = crate::population_engine_run_receipt_v1::begin_population_engine_run_v1(
        &result.selection_scope,
    )
    .unwrap();
    run.record_successful_population(crate::engine_identity::PopulationEvalEngine::Cpu, 3, 3)
        .unwrap();
    let engine_receipt_v1 = run.finish().unwrap();
    let receipt_v2 =
        crate::population_execution_run_receipt_v2::seal_exact_population_execution_run_receipt_v2(
            engine_receipt_v1.clone(),
            None,
        )
        .unwrap();
    result
        .funnel_profile
        .as_mut()
        .unwrap()
        .attach_population_execution_run_receipt_v2(receipt_v2.clone())
        .unwrap();

    let profile = build_discovery_profile(&DiscoveryConfig::default(), &result);
    assert_eq!(
        profile.population_eval_engines,
        vec![crate::engine_identity::PopulationEvalEngine::Cpu]
    );
    assert_eq!(
        profile.population_execution_run_receipt_v2.as_ref(),
        Some(&receipt_v2)
    );
    let json = serde_json::to_string(&profile).unwrap();
    assert!(json.contains(receipt_v2.identity_sha256()), "{json}");
    assert!(
        json.contains(engine_receipt_v1.canonical_scope_identity_sha256()),
        "{json}"
    );
}

#[test]
fn discovery_run_profile_exposes_validation_evidence_hashes_and_missing_kinds() {
    let config = DiscoveryConfig::default();
    let result = populated_discovery_result(1, 0, 1, 1);
    let profile = build_discovery_profile(&config, &result);
    assert!(
        profile
            .validation_evidence_hashes
            .canonical_backtest
            .is_some()
    );
    assert!(profile.validation_evidence_hashes.walkforward.is_none());
    assert!(profile.validation_evidence_hashes.forward_test.is_some());
    assert!(profile.validation_evidence_hashes.prop_firm.is_some());
    assert!(
        profile
            .validation_evidence_hashes
            .live_execution_simulation
            .is_none()
    );
    assert!(!profile.validation_evidence_complete);
    assert!(
        profile
            .validation_evidence_missing_kinds
            .iter()
            .any(|k| k == "walkforward")
    );
    assert!(
        profile
            .validation_evidence_missing_kinds
            .iter()
            .any(|k| k == "live_execution_simulation")
    );
    assert_eq!(profile.prop_firm_validation_artifacts_observed, 1);
}

// ─── F-304: pre-flight bail tests (2026-05-28) ────────────────────
//
// `run_discovery_cycle_with_progress` must fail loud BEFORE spinning
// up the GA when `evaluation_symbol` or `evaluation_account_currency`
// is empty. The previous behaviour was to silently propagate the
// empty strings into the cost-model NaN-sentinel guard which made
// every GA candidate produce zero-trade metrics that the sanitizer
// scrubbed to 0.0 — operator's "no trades found" with no clue why.

fn valid_discovery_config() -> DiscoveryConfig {
    DiscoveryConfig {
        timeframe_label: "M1".to_string(),
        evaluation_symbol: "EURUSD".to_string(),
        evaluation_account_currency: "USD".to_string(),
        evaluation_spread_pips: 1.0,
        evaluation_commission_per_trade: 7.0,
        population: 10,
        generations: 1,
        candidate_count: 10,
        portfolio_size: 5,
        ..DiscoveryConfig::default()
    }
}

fn assert_broker_truth_precedes_legacy_config_math(error: &anyhow::Error) {
    let message = format!("{error:#}");
    assert!(
        message.contains(neoethos_core::BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1),
        "discovery reached legacy config/cost handling before broker truth: {message}"
    );
}

#[test]
fn run_discovery_cycle_bails_on_empty_evaluation_symbol() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let mut cfg = valid_discovery_config();
    cfg.evaluation_symbol = String::new();
    let input = sample_run_input(&features, &ohlcv);
    let err = run_discovery_cycle(&input, &cfg).expect_err("empty symbol must bail");
    assert_broker_truth_precedes_legacy_config_math(&err);
}

#[test]
fn run_discovery_cycle_bails_on_empty_account_currency() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let mut cfg = valid_discovery_config();
    cfg.evaluation_account_currency = String::new();
    let input = sample_run_input(&features, &ohlcv);
    let err = run_discovery_cycle(&input, &cfg).expect_err("empty account_currency must bail");
    assert_broker_truth_precedes_legacy_config_math(&err);
}

#[test]
fn run_discovery_cycle_bails_on_nan_spread() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let mut cfg = valid_discovery_config();
    cfg.evaluation_spread_pips = f64::NAN;
    let input = sample_run_input(&features, &ohlcv);
    let err = run_discovery_cycle(&input, &cfg).expect_err("NaN spread must bail");
    assert_broker_truth_precedes_legacy_config_math(&err);
}

#[test]
fn run_discovery_cycle_bails_on_nan_commission() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let mut cfg = valid_discovery_config();
    cfg.evaluation_commission_per_trade = f64::NAN;
    let input = sample_run_input(&features, &ohlcv);
    let err = run_discovery_cycle(&input, &cfg).expect_err("NaN commission must bail");
    assert_broker_truth_precedes_legacy_config_math(&err);
}

#[test]
fn run_discovery_cycle_bails_on_whitespace_only_currency() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let mut cfg = valid_discovery_config();
    cfg.evaluation_account_currency = "   ".to_string();
    let input = sample_run_input(&features, &ohlcv);
    let err = run_discovery_cycle(&input, &cfg).expect_err("whitespace-only currency must bail");
    assert_broker_truth_precedes_legacy_config_math(&err);
}

#[test]
fn from_settings_propagates_account_currency() {
    // F-304: regression guard — verify that
    // `DiscoveryConfig::from_settings` now pulls `account_currency` from
    // SystemConfig instead of hardcoding `String::new()`. Without this
    // fix, every settings-derived config tripped the pre-flight bail.
    let mut settings = neoethos_core::Settings::default();
    settings.system.symbol = "GBPJPY".to_string();
    settings.system.account_currency = "GBP".to_string();
    settings.risk.backtest_spread_pips = 1.5;
    settings.risk.commission_per_lot = 7.0;
    let cfg = DiscoveryConfig::from_settings(&settings);
    assert_eq!(cfg.evaluation_symbol, "GBPJPY");
    assert_eq!(cfg.evaluation_account_currency, "GBP");
    assert!(cfg.evaluation_spread_pips.is_finite());
    assert!(cfg.evaluation_commission_per_trade.is_finite());
}

// ─── F-305 PropFirm gate scaling tests (2026-05-28) ───────────────

#[test]
fn min_trades_per_month_scale_intra_day_unchanged() {
    // Intra-day TFs keep operator's value at 1.0× — plenty of bars,
    // 15 trades/month is fine.
    for timeframe in ["M1", "M2", "M3", "M4", "M5", "M10", "M15"] {
        assert_eq!(min_trades_per_month_scale_for_tf(timeframe), 1.0);
    }
    assert_eq!(
        min_trades_per_month_scale_for_tf("H12"),
        1.0,
        "H12 preserves the old conservative fallback until policy review"
    );
}

#[test]
fn min_trades_per_month_scale_drops_for_higher_tfs() {
    // The whole point: higher TFs have fewer bars, so a tight floor
    // mechanically rejects sane swing strategies.
    let m30 = min_trades_per_month_scale_for_tf("M30");
    let h1 = min_trades_per_month_scale_for_tf("H1");
    let h4 = min_trades_per_month_scale_for_tf("H4");
    let d1 = min_trades_per_month_scale_for_tf("D1");
    let w1 = min_trades_per_month_scale_for_tf("W1");
    let mn1 = min_trades_per_month_scale_for_tf("MN1");
    // Monotone-decreasing in bar density
    assert!(m30 < 1.0, "M30 should be < 1.0");
    assert!(h1 < m30, "H1 < M30");
    assert!(h4 < h1, "H4 < H1");
    assert!(d1 < h4, "D1 < H4");
    assert!(w1 < d1, "W1 < D1");
    assert!(mn1 < w1, "MN1 < W1");
    // Sanity: for operator's default 15 trades/month, D1 must produce
    // a sane floor (e.g. ≤ 3 trades/month so realistic swing
    // strategies aren't auto-rejected).
    assert!(
        15.0 * d1 <= 3.0,
        "D1 floor at base=15 must be ≤ 3, got {}",
        15.0 * d1
    );
}

#[test]
fn min_trades_per_month_scale_case_insensitive() {
    assert_eq!(
        min_trades_per_month_scale_for_tf("d1"),
        min_trades_per_month_scale_for_tf("D1")
    );
    assert_eq!(
        min_trades_per_month_scale_for_tf("h4"),
        min_trades_per_month_scale_for_tf("H4")
    );
}

#[test]
fn min_trades_per_month_scale_unknown_tf_is_conservative() {
    // Unknown TFs default to 1.0 — don't silently relax thresholds
    // for inputs we don't understand.
    assert_eq!(min_trades_per_month_scale_for_tf(""), 1.0);
    assert_eq!(min_trades_per_month_scale_for_tf("H2"), 1.0); // non-canonical
    assert_eq!(min_trades_per_month_scale_for_tf("XYZ"), 1.0);
}

#[test]
fn annual_bar_estimate_covers_every_official_timeframe_without_private_aliases() {
    use neoethos_core::CanonicalTimeframe as T;

    for timeframe in T::ALL {
        assert!(
            approx_bars_per_year(timeframe.as_str()) > 0,
            "official timeframe {timeframe} has no annual estimate"
        );
    }
    assert_eq!(approx_bars_per_year("M2"), 220 * 24 * 30);
    assert_eq!(approx_bars_per_year("M4"), 220 * 24 * 15);
    assert_eq!(approx_bars_per_year("M10"), 220 * 24 * 6);
    assert_eq!(approx_bars_per_year("H2"), 0);
}

#[test]
fn propfirm_mode_scales_min_trades_per_month_for_d1() {
    // End-to-end: PropFirm mode + D1 should produce a clearly-lower
    // min_trades_per_month than the operator's raw config value.
    //
    // Note: env-var test lock not needed here — we read the mode
    // via `resolve_discovery_mode()` which is process-global, but
    // the default with no env is PropFirm anyway. Tests that mutate
    // NEOETHOS_BOT_DISCOVERY_MODE must use ENV_VAR_TEST_LOCK; we don't.
    let mut cfg = DiscoveryConfig::default();
    cfg.evaluation_symbol = "EURUSD".to_string();
    cfg.evaluation_account_currency = "USD".to_string();
    cfg.evaluation_spread_pips = 1.0;
    cfg.evaluation_commission_per_trade = 7.0;
    cfg.timeframe_label = "D1".to_string();
    cfg.filtering.min_trades_per_month = 15.0;
    cfg.filtering.opportunistic_min_trades_per_month = 10.0;

    let cfg = cfg.apply_mode_overrides();
    // PropFirm mode is the default; D1 scale = 0.13 → 15 × 0.13 = 1.95
    // (clamped to ≥ 0.5).
    assert!(
        cfg.filtering.min_trades_per_month < 5.0,
        "expected D1 PropFirm min_trades_per_month < 5.0, got {}",
        cfg.filtering.min_trades_per_month
    );
    assert!(
        cfg.filtering.min_trades_per_month >= 0.5,
        "expected floor of 0.5, got {}",
        cfg.filtering.min_trades_per_month
    );
}

#[test]
fn propfirm_mode_leaves_m1_min_trades_per_month_unchanged() {
    // On M1, scale = 1.0 → operator's value passes through unchanged.
    let mut cfg = DiscoveryConfig::default();
    cfg.evaluation_symbol = "EURUSD".to_string();
    cfg.evaluation_account_currency = "USD".to_string();
    cfg.evaluation_spread_pips = 1.0;
    cfg.evaluation_commission_per_trade = 7.0;
    cfg.timeframe_label = "M1".to_string();
    cfg.filtering.min_trades_per_month = 15.0;

    let cfg = cfg.apply_mode_overrides();
    assert_eq!(cfg.filtering.min_trades_per_month, 15.0);
}

#[test]
fn discovery_runtime_from_settings_default_matches_struct_default() {
    // Behaviour lock: with config at its defaults,
    // `DiscoveryRuntimeOverrides::from_settings` reproduces `default()` exactly.
    //
    // 2026-08-10: `from_env()` — the six-name `NEOETHOS_BOT_PREFILTER_*` /
    // `_FUNNEL_*` / `_MIN_HISTORY_YEARS` reader this test used to compare
    // against — is DELETED. It had zero production callers and was kept "for
    // reference", which is another way of saying there were two ways to set the
    // same knob and only one of them was visible. `from_settings` is now the
    // only constructor that reads operator input.
    let s = neoethos_core::Settings::default();
    assert_eq!(
        DiscoveryRuntimeOverrides::from_settings(&s),
        DiscoveryRuntimeOverrides::default(),
    );
}

// ── F-343 (#14): actionable empty-portfolio diagnosis ────────────────

#[test]
fn empty_portfolio_diagnosis_names_bottleneck_and_remedy() {
    use crate::funnel_profile::{FunnelProfile, FunnelStage};

    let mut funnel = FunnelProfile::new("EURUSD", "M1");
    // Quality screen is the bottleneck: 412 in, 0 out.
    let mut quality = FunnelStage::new("passed_quality");
    quality.record(412, 0);
    quality.top_reasons = vec![
        ("low_sharpe".to_string(), 210),
        ("low_profit_factor".to_string(), 150),
    ];
    funnel.stages = vec![FunnelStage::passthrough("passed_min_trades", 412), quality];
    funnel.bottleneck_stage = "passed_quality".to_string();

    let msg = describe_empty_portfolio_funnel(&funnel);
    assert!(msg.contains("passed_quality"), "names the stage: {msg}");
    assert!(msg.contains("low_sharpe×210"), "surfaces reasons: {msg}");
    assert!(
        msg.contains("Sharpe") || msg.contains("win-rate"),
        "gives a remedy: {msg}"
    );
}

#[test]
fn empty_portfolio_diagnosis_falls_back_when_no_bottleneck_set() {
    use crate::funnel_profile::{FunnelProfile, FunnelStage};

    let mut funnel = FunnelProfile::new("GBPUSD", "H1");
    let mut base = FunnelStage::new("passed_base_filter");
    base.record(80, 0); // most-rejecting stage, bottleneck_stage left empty
    funnel.stages = vec![FunnelStage::passthrough("data_loaded", 80), base];
    funnel.bottleneck_stage = String::new();

    let msg = describe_empty_portfolio_funnel(&funnel);
    assert!(
        msg.contains("passed_base_filter"),
        "infers bottleneck: {msg}"
    );
    assert!(msg.contains("max-drawdown") || msg.contains("min-profit"));
}

/// Reference implementation of the tie-corrected Spearman midrank, written
/// the naive O(n²) way (rescan the slice for every element). The shipping
/// `spearman_corr_i8` computes the same quantity from a 256-bucket
/// histogram in O(n); these two must agree exactly.
fn spearman_corr_i8_naive(a: &[i8], b: &[i8]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let n = a.len();
    let rank_of = |vals: &[i8], v: i8| -> f64 {
        let count = vals.iter().filter(|&&x| x == v).count() as f64;
        let before = vals.iter().filter(|&&x| x < v).count() as f64;
        before + (count + 1.0) / 2.0
    };
    let ranks_a: Vec<f64> = a.iter().map(|&v| rank_of(a, v)).collect();
    let ranks_b: Vec<f64> = b.iter().map(|&v| rank_of(b, v)).collect();
    let mean_a: f64 = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = ranks_b.iter().sum::<f64>() / n as f64;
    let (mut num, mut denom_a, mut denom_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let da = ranks_a[i] - mean_a;
        let db = ranks_b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a == 0.0 || denom_b == 0.0 {
        return None;
    }
    let correlation = num / (denom_a.sqrt() * denom_b.sqrt());
    correlation.is_finite().then_some(correlation)
}

#[test]
fn spearman_histogram_matches_the_naive_rank_scan() {
    // Deterministic pseudo-random trit signals (the real shape: -1/0/+1),
    // plus perfectly-(anti)correlated and full-i8-range edge cases.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = 4096;
    let a: Vec<i8> = (0..n).map(|_| (next() % 3) as i8 - 1).collect();
    let b: Vec<i8> = (0..n).map(|_| (next() % 3) as i8 - 1).collect();
    let inverted: Vec<i8> = a.iter().map(|v| -v).collect();
    let constant = vec![0_i8; n];
    // Full i8 range too — the histogram must handle every bucket, not just trits.
    let wide_a: Vec<i8> = (0..n).map(|i| (i % 256) as i16 as i8).collect();
    let wide_b: Vec<i8> = (0..n).map(|i| ((i * 7) % 256) as i16 as i8).collect();

    for (label, x, y) in [
        ("independent", &a, &b),
        ("self", &a, &a),
        ("inverted", &a, &inverted),
        ("wide range", &wide_a, &wide_b),
    ] {
        let fast = spearman_corr_i8(x, y).expect("defined production Spearman");
        let naive = spearman_corr_i8_naive(x, y).expect("defined naive Spearman");
        assert!(
            (fast - naive).abs() < 1e-12,
            "{label}: fast={fast} naive={naive}"
        );
    }
    // Sanity anchors on top of the equivalence check.
    assert!((spearman_corr_i8(&a, &a).unwrap() - 1.0).abs() < 1e-12);
    assert!((spearman_corr_i8(&a, &inverted).unwrap() + 1.0).abs() < 1e-12);
    assert_eq!(
        spearman_corr_i8(&a, &constant),
        Err(CorrelationUndefinedV1::ConstantInput)
    );

    // SciPy's published tie fixture anchors average-rank semantics rather than
    // relying on the pre-existing CPU implementation as the authority.
    let tied_a = [1_i8, 2, 3, 4, 5];
    let tied_b = [5_i8, 6, 7, 8, 7];
    assert!((spearman_corr_i8(&tied_a, &tied_b).unwrap() - 0.820_782_681_668_123_3).abs() < 1e-15);
}

#[test]
fn portfolio_correlation_gate_rejects_undefined_inputs_and_threshold_equality() {
    let defined = [-1_i8, 0, 1, -1, 0, 1];
    let constant = [0_i8; 6];
    assert_eq!(
        pearson_corr_i8(&defined, &constant),
        Err(CorrelationUndefinedV1::ConstantInput)
    );
    assert_eq!(
        pearson_corr_i8(&defined, &defined[..5]),
        Err(CorrelationUndefinedV1::LengthMismatch)
    );
    assert_eq!(
        spearman_corr_i8(&defined[..1], &defined[..1]),
        Err(CorrelationUndefinedV1::InsufficientPairedObservations)
    );
    assert_eq!(
        pairwise_portfolio_correlation_decision_v1(&defined, &defined, 1.0),
        PortfolioCorrelationDecisionV1::RejectThreshold
    );
    assert_eq!(
        pairwise_portfolio_correlation_decision_v1(&defined, &constant, 0.9),
        PortfolioCorrelationDecisionV1::RejectUndefined(CorrelationUndefinedV1::ConstantInput)
    );
    assert!(!portfolio_signal_is_correlation_rankable_v1(&constant));

    let near_constant_sum_squares = (0.5 * SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1).powi(2);
    assert_eq!(
        classify_centered_correlation_input_v1(1.0, near_constant_sum_squares),
        Err(CorrelationUndefinedV1::NearConstantInput)
    );
    assert_eq!(
        finish_correlation_v1(f64::NAN, 1.0, 1.0),
        Err(CorrelationUndefinedV1::NonFiniteResult)
    );
}

/// The operator's risk band must reach the BACKTEST, not just live sizing.
/// Before 2026-07-21 `discovery_backtest_settings` fell through to
/// `BacktestSettings::default()` for these two fields, so discovery always
/// sized at 0.5%..3% no matter what `config.yaml` said — while the Discovery
/// pre-flight told the operator the value applied to "this search". Risky mode
/// could therefore never search at the aggressive size it exists for.
#[test]
fn operator_risk_band_reaches_the_discovery_backtest() {
    let mut settings = neoethos_core::Settings::default();
    settings.risk.min_risk_per_trade = 0.05;
    settings.risk.max_risk_per_trade = 0.30;

    let config = DiscoveryConfig::from_settings(&settings);
    assert!((config.risk_per_trade_min - 0.05).abs() < 1e-12);
    assert!((config.risk_per_trade_max - 0.30).abs() < 1e-12);

    let gene = Gene {
        sl_pips: 20.0,
        tp_pips: 40.0,
        ..Default::default()
    };
    let settings_out = PopulationTemplateResolver::new(&config, Some(1.25)).template(&gene);
    assert!(
        (settings_out.risk_per_trade_max - 0.30).abs() < 1e-12,
        "the backtest must size at the operator's 30%, got {}",
        settings_out.risk_per_trade_max
    );
    assert!((settings_out.risk_per_trade_min - 0.05).abs() < 1e-12);
}

/// SLICE-2 GUARD (2026-08-08). The raw builder `discovery_backtest_settings`
/// leaves the adaptive-stop fields at fixed defaults; 9 of its 13 former call
/// sites — including the quality screen — therefore backtested adaptive genes
/// on their unused fixed pips while GA scoring ran `sl = stop_vol_mult ×
/// base[i]` (measured on one signal: 30 331 vs 1 727 trades, 17.6×). Every
/// call now routes through `GeneEvalSettingsResolver` (serial per-gene eval)
/// or `PopulationTemplateResolver` (templates for the population helpers).
/// The function is private to the `discovery` module, so the compiler already
/// blocks the rest of the crate; this test blocks NEW call sites inside
/// discovery.rs / discovery_tests.rs themselves.
#[test]
fn discovery_backtest_settings_has_no_callers_outside_the_resolvers() {
    // Built from two halves so this test's own source can never match itself.
    let needle = format!("{}{}", "discovery_backtest_", "settings(");
    let discovery_src = include_str!("discovery.rs");
    let call_sites = discovery_src.matches(needle.as_str()).count();
    assert_eq!(
        call_sites, 3,
        "expected exactly 3 occurrences of `{needle}` in discovery.rs \
         (1 definition + 1 in PopulationTemplateResolver::template + 1 in \
         GeneEvalSettingsResolver::settings_for_gene), found {call_sites}. \
         A new direct call bypasses the ONE resolver and reintroduces the \
         fixed-stop divergence: route serial per-gene evaluation through \
         GeneEvalSettingsResolver::settings_for_gene and population-helper \
         templates through PopulationTemplateResolver::template."
    );
    let tests_src = include_str!("discovery_tests.rs");
    assert_eq!(
        tests_src.matches(needle.as_str()).count(),
        0,
        "discovery_tests.rs must call the resolvers, not the raw builder"
    );
}

/// The resolver installs the SCORED stop regime: an adaptive gene gets its own
/// `stop_vol_mult`, the shared slice base series and the reward:risk; a fixed
/// gene keeps the scalar-pips path untouched. This is the property every
/// selection-bearing serial stage now inherits by construction.
#[test]
fn gene_eval_settings_resolver_installs_the_scored_stop_regime() {
    let mut config = DiscoveryConfig::from_settings(&neoethos_core::Settings::default());
    config.evaluation_symbol = "EURUSD".to_string();

    // Deterministic bars with real high/low spread so the base estimator has
    // volatility to measure (mirrors search_engine's synthetic_hlc).
    let n = 200usize;
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    for i in 0..n {
        let c = 1.10 + 0.002 * ((i as f64) * 0.1).sin();
        let amp = 0.001 + 0.001 * (((i as f64) * 0.05).cos().abs());
        close.push(c);
        high.push(c + amp);
        low.push(c - amp);
    }

    let adaptive_gene = Gene {
        sl_pips: 13.0,
        tp_pips: 26.0,
        stop_vol_mult: 1.75,
        ..Default::default()
    };
    let fixed_gene = Gene {
        sl_pips: 13.0,
        tp_pips: 26.0,
        stop_vol_mult: 0.0,
        ..Default::default()
    };

    let resolver = GeneEvalSettingsResolver::for_slice(
        &config,
        [&adaptive_gene, &fixed_gene],
        &high,
        &low,
        &close,
    )
    .expect("resolver builds on a 200-bar slice");

    let adaptive = resolver.settings_for_gene(&adaptive_gene);
    assert_eq!(
        adaptive.adaptive_vol_mult, 1.75,
        "the gene's scored stop_vol_mult must reach the serial backtest"
    );
    let base = adaptive
        .adaptive_base_pips
        .as_ref()
        .expect("adaptive gene gets the slice base series");
    assert_eq!(base.len(), n, "base series indexed to the resolver's slice");
    assert!(base.iter().all(|&b| b.is_finite() && b > 0.0));
    assert!(adaptive.adaptive_rr > 0.0);

    let fixed = resolver.settings_for_gene(&fixed_gene);
    assert_eq!(fixed.adaptive_vol_mult, 0.0);
    assert!(
        fixed.adaptive_base_pips.is_none(),
        "a fixed gene keeps the byte-identical scalar-pips path"
    );
    assert_eq!(fixed.sl_pips, 13.0);
    assert_eq!(fixed.tp_pips, 26.0);
}

#[test]
fn risk_band_is_clamped_and_ordered() {
    // A min above max would otherwise size every trade at the floor.
    let mut settings = neoethos_core::Settings::default();
    settings.risk.min_risk_per_trade = 0.40;
    settings.risk.max_risk_per_trade = 0.10;
    let config = DiscoveryConfig::from_settings(&settings);
    assert!(
        config.risk_per_trade_max >= config.risk_per_trade_min,
        "max {} must not fall below min {}",
        config.risk_per_trade_max,
        config.risk_per_trade_min
    );

    // Absurd values are bounded to [0, 100%].
    let mut settings = neoethos_core::Settings::default();
    settings.risk.min_risk_per_trade = -1.0;
    settings.risk.max_risk_per_trade = 12.0;
    let config = DiscoveryConfig::from_settings(&settings);
    assert_eq!(config.risk_per_trade_min, 0.0);
    assert_eq!(config.risk_per_trade_max, 1.0);

    // A bare default keeps the historical band so nothing else shifts.
    let d = DiscoveryConfig::default();
    assert!((d.risk_per_trade_min - 0.005).abs() < 1e-12);
    assert!((d.risk_per_trade_max - 0.03).abs() < 1e-12);
}

/// Risky and Prop-firm must NOT share one sizing knob. Before 2026-07-21 they
/// did: flipping `system.trading_mode` silently carried the other mode's risk
/// into the search, so a 30% risky band made every prop-firm candidate break
/// the firm's daily-loss rule on its first loss — the search returned nothing
/// and the screen gave no reason.
#[test]
fn each_mode_keeps_its_own_risk_band() {
    let mut settings = neoethos_core::Settings::default();
    settings.risk.min_risk_per_trade = 0.0;
    settings.risk.max_risk_per_trade = 0.03; // shared fallback
    settings.risk.risky_max_risk_per_trade = Some(0.30);
    settings.risk.prop_firm_max_risk_per_trade = Some(0.01);

    let risky = DiscoveryConfig {
        mode: DiscoveryMode::Risky,
        ..DiscoveryConfig::from_settings(&settings)
    }
    .apply_mode_overrides();
    assert!(
        (risky.risk_per_trade_max - 0.30).abs() < 1e-12,
        "risky must size at 30%, got {}",
        risky.risk_per_trade_max
    );

    let prop = DiscoveryConfig {
        mode: DiscoveryMode::PropFirm,
        ..DiscoveryConfig::from_settings(&settings)
    }
    .apply_mode_overrides();
    assert!(
        (prop.risk_per_trade_max - 0.01).abs() < 1e-12,
        "prop-firm must size at 1%, got {}",
        prop.risk_per_trade_max
    );

    // The one that matters most: the same settings produce DIFFERENT sizing.
    assert!(risky.risk_per_trade_max > prop.risk_per_trade_max * 10.0);
}

#[test]
fn unset_mode_band_inherits_the_shared_one() {
    // Behaviour when NO per-mode band is set must be pure inheritance of the
    // shared band. The Risky 30% ceiling is now a default (operator decision
    // 2026-08-09), so clear the per-mode bands explicitly to exercise the
    // inheritance path rather than relying on the default being None.
    let mut settings = neoethos_core::Settings::default();
    settings.risk.min_risk_per_trade = 0.002;
    settings.risk.max_risk_per_trade = 0.05;
    settings.risk.risky_min_risk_per_trade = None;
    settings.risk.risky_max_risk_per_trade = None;
    settings.risk.prop_firm_min_risk_per_trade = None;
    settings.risk.prop_firm_max_risk_per_trade = None;

    for mode in [
        DiscoveryMode::Risky,
        DiscoveryMode::PropFirm,
        DiscoveryMode::Strict,
    ] {
        let c = DiscoveryConfig {
            mode,
            ..DiscoveryConfig::from_settings(&settings)
        }
        .apply_mode_overrides();
        assert!((c.risk_per_trade_min - 0.002).abs() < 1e-12, "{mode:?}");
        assert!((c.risk_per_trade_max - 0.05).abs() < 1e-12, "{mode:?}");
    }
}

#[test]
fn mode_band_rejects_nonsense_and_orders_itself() {
    let mut settings = neoethos_core::Settings::default();
    settings.risk.max_risk_per_trade = 0.03;

    // A zero / negative / non-finite max means "not set" -> inherit.
    for bad in [Some(0.0), Some(-0.5), Some(f64::NAN)] {
        settings.risk.risky_max_risk_per_trade = bad;
        let c = DiscoveryConfig {
            mode: DiscoveryMode::Risky,
            ..DiscoveryConfig::from_settings(&settings)
        }
        .apply_mode_overrides();
        assert!((c.risk_per_trade_max - 0.03).abs() < 1e-12, "bad={bad:?}");
    }

    // min above max cannot invert the band, and nothing exceeds 100%.
    settings.risk.risky_min_risk_per_trade = Some(0.9);
    settings.risk.risky_max_risk_per_trade = Some(0.2);
    let c = DiscoveryConfig {
        mode: DiscoveryMode::Risky,
        ..DiscoveryConfig::from_settings(&settings)
    }
    .apply_mode_overrides();
    assert!(c.risk_per_trade_max >= c.risk_per_trade_min);
    assert!(c.risk_per_trade_max <= 1.0 && c.risk_per_trade_min >= 0.0);

    settings.risk.risky_min_risk_per_trade = None;
    settings.risk.risky_max_risk_per_trade = Some(9.0);
    let c = DiscoveryConfig {
        mode: DiscoveryMode::Risky,
        ..DiscoveryConfig::from_settings(&settings)
    }
    .apply_mode_overrides();
    assert_eq!(c.risk_per_trade_max, 1.0, "capped at 100% of the account");
}

#[test]
fn a_wiped_out_account_is_rejected_however_well_it_scores() {
    use crate::quality::empty_metrics;
    // The candidate that motivated this gate: 4 917 trades, profit factor 2.85,
    // graded EXCELLENT — on an account whose equity curve bottomed at
    // -30 596 EUR. A drawdown past 100 % is not a bad score, it is a state that
    // cannot exist, so no other metric may compensate for it.
    let mut ruined = empty_metrics("gene_332618_18");
    ruined.profit_factor = 2.85;
    ruined.win_rate = 0.38;
    ruined.trades_per_month = 40.0;
    ruined.positive_months = 100;
    ruined.avg_monthly_return_pct = 65.0;
    ruined.max_drawdown_pct = 4.031; // 403.1 % — a fraction, despite the name
    assert!(!super::survived_the_backtest(&ruined));

    // Exactly total loss is still ruin.
    ruined.max_drawdown_pct = 1.0;
    assert!(!super::survived_the_backtest(&ruined));

    // A severe but survivable drawdown stays eligible: this gate decides
    // possibility, not quality.
    ruined.max_drawdown_pct = 0.438; // the 43.8 % seen on a real survivor
    assert!(super::survived_the_backtest(&ruined));
}

#[test]
fn risky_mode_keeps_the_operators_activity_floor() {
    // This used to be pinned to 0.001 for risky mode, so
    // `models.prop_search_val_min_trades_per_day` was set, resolved, passed in —
    // and discarded — in the one mode actually used. A strategy trading twice a
    // decade cannot compound a small balance to a large one, so the floor has to
    // reach the search.
    let mut config = DiscoveryConfig::default();
    config.min_trades_per_day = 1.0;
    config.mode = DiscoveryMode::Risky;
    let risky = config.apply_mode_overrides();
    assert!(
        (risky.min_trades_per_day - 1.0).abs() < 1e-9,
        "risky rewrote the activity floor to {}",
        risky.min_trades_per_day
    );

    // The quality floors around it are still deliberately loosened — this test
    // must not be read as risky having become strict.
    assert!(risky.filtering.min_profit_factor <= 0.0);
    assert!(risky.filtering.min_win_rate <= 0.0);

    // One trade a day over a year of weekdays is roughly a year of weekdays'
    // worth of in-market bars, not one trade in total.
    let year_of_weekdays: Vec<i64> = (0..365)
        .map(|d| (1_700_000_000_i64 + d * 86_400) * 1000)
        .collect();
    let required = super::min_trades_required(&year_of_weekdays, 1.0, year_of_weekdays.len());
    assert!(
        required > 200 && required < 366,
        "expected ~one per weekday, got {required}"
    );
}

/// A candidate that makes money after costs — the baseline every shape test
/// below starts from, because since 2026-08-09 nothing reaches a shape check
/// until it has cleared the expectancy floor.
fn profitable_metrics(id: &str) -> crate::quality::StrategyMetrics {
    let mut m = crate::quality::empty_metrics(id);
    m.total_trades = 500;
    m.profit_per_trade = 12.0;
    m.net_expectancy_stderr = 3.0;
    m.net_expectancy_t_stat = 4.0;
    m
}

/// THE GUARD AGAINST THE REWARD HACK RETURNING.
///
/// The trailing stop became searchable in the same change that added this test.
/// Under a payoff-floor objective that is a free win for the GA: widening the
/// trail raises the payoff ratio without touching the money. Measured on real
/// EURUSD bars while sweeping exactly that knob —
///
///   trail multiplier 1.0 → payoff 0.91, expectancy -4.15 pips/trade
///   trail multiplier 3.0 → payoff 2.53, expectancy -4.18 pips/trade
///
/// — the payoff moved by a factor of 2.8 and the expectancy did not move at all.
/// On a driftless price, exit geometry redistributes the (win-rate, payoff)
/// split and their product stays pinned at minus the cost.
///
/// So a 2.0 payoff floor ACCEPTS the second row and REJECTS the first, and the
/// second row empties the account marginally faster. If this test ever fails,
/// the payoff floor has become sufficient on its own again and the search is
/// once more selecting for a number that is not money.
#[test]
fn a_high_payoff_money_loser_is_refused_by_name() {
    use super::{TargetProfile, TargetProfileRejection};
    use crate::quality::empty_metrics;

    let profile = TargetProfile {
        min_payoff_ratio: 2.0,
        ..TargetProfile::default()
    };

    // The measured reward hack, in the analyzer's own units. Payoff 2.53 clears
    // the 2.0 floor outright; the average trade still loses money after costs.
    let mut hacked = empty_metrics("trail_mult_3");
    hacked.total_trades = 4_000;
    hacked.payoff_ratio = 2.53;
    hacked.win_rate = 0.24;
    hacked.profit_per_trade = -4.18; // pips, net of spread + commission
    hacked.net_expectancy_stderr = 0.35;
    hacked.net_expectancy_t_stat = -11.9;

    assert_eq!(
        profile.evaluate(&hacked),
        Err(TargetProfileRejection::NegativeNetExpectancy),
        "a payoff of {} must not survive an expectancy of {} per trade",
        hacked.payoff_ratio,
        hacked.profit_per_trade
    );
    assert!(!profile.accepts(&hacked));

    // And the refusal is NOT something a configuration can switch off. There is
    // no `min_net_expectancy_per_trade` that admits a money-loser: the field is
    // a floor, `0.0` already means "strictly positive", and the loader clamps
    // negatives away. Assert on the type directly so a future edit that adds an
    // `if self.min_net_expectancy_per_trade > 0.0` guard — turning it into an
    // opt-in preference like its neighbours — fails here.
    let all_gates_off = TargetProfile::default();
    assert_eq!(
        all_gates_off.evaluate(&hacked),
        Err(TargetProfileRejection::NegativeNetExpectancy),
        "an empty target profile must still refuse a money-loser"
    );

    // The mirror case, which is the whole reason the payoff floor is demoted
    // rather than deleted: the SAME strategy family at trail multiplier 1.0 has
    // a payoff of 0.91 and the same losing expectancy. Both must be refused, and
    // for the same named reason — the payoff ratio must not be what decides.
    let mut unhacked = hacked.clone();
    unhacked.payoff_ratio = 0.91;
    unhacked.win_rate = 0.52;
    unhacked.profit_per_trade = -4.15;
    assert_eq!(
        profile.evaluate(&unhacked),
        Err(TargetProfileRejection::NegativeNetExpectancy)
    );

    // A money-MAKER with a modest payoff survives the expectancy floor and is
    // then refused by the payoff preference, by that name. That is what
    // "secondary filter" means: it can only narrow, never admit.
    let mut modest = profitable_metrics("modest_but_profitable");
    modest.payoff_ratio = 1.10;
    modest.win_rate = 0.62;
    assert_eq!(
        profile.evaluate(&modest),
        Err(TargetProfileRejection::PayoffTooLow)
    );
    assert!(
        TargetProfile::default().accepts(&modest),
        "with no shape preference configured, a profitable candidate survives"
    );
}

/// Positive expectancy is necessary, not sufficient, once a significance bar is
/// configured — and the bar must be opt-in, because choosing it is an operator
/// decision rather than a correctness bound.
#[test]
fn the_expectancy_significance_bar_is_opt_in_and_binds_when_set() {
    use super::{TargetProfile, TargetProfileRejection};

    // +20 per trade over 30 trades with a per-trade sd of ~197: the standard
    // error is 36, so the point estimate is well inside its own noise.
    let mut noisy = profitable_metrics("thirty_trades");
    noisy.total_trades = 30;
    noisy.profit_per_trade = 20.0;
    noisy.net_expectancy_stderr = 36.0;
    noisy.net_expectancy_t_stat = 20.0 / 36.0;

    // Sign only (the default): it survives, and the run reports the standard
    // error next to the number so the operator can see what it is worth.
    assert!(TargetProfile::default().accepts(&noisy));

    let strict = TargetProfile {
        min_expectancy_t_stat: 2.0,
        ..TargetProfile::default()
    };
    assert_eq!(
        strict.evaluate(&noisy),
        Err(TargetProfileRejection::ExpectancyNotSignificant)
    );

    // The same expectancy measured over enough trades clears it.
    let mut solid = noisy.clone();
    solid.total_trades = 5_000;
    solid.net_expectancy_stderr = 2.8;
    solid.net_expectancy_t_stat = 20.0 / 2.8;
    assert!(strict.accepts(&solid));
}

/// A candidate that never traded reports an expectancy of exactly 0.0. Nothing
/// traded is not an edge, and the strict `>` in the gate is what says so.
#[test]
fn a_candidate_with_no_trades_is_not_an_edge() {
    use super::{TargetProfile, TargetProfileRejection};
    use crate::quality::empty_metrics;

    assert_eq!(
        TargetProfile::default().evaluate(&empty_metrics("silent")),
        Err(TargetProfileRejection::NegativeNetExpectancy)
    );
}

#[test]
fn the_target_profile_separates_win_rate_from_payoff() {
    use super::TargetProfile;

    // The operator's stated target: 57-65 % of trades win, each winner worth
    // about 2.2 losers.
    let profile = TargetProfile {
        min_win_rate: 0.57,
        min_payoff_ratio: 2.2,
        max_in_market: 0.35,
        ..TargetProfile::default()
    };

    // Every fixture below starts PROFITABLE. Before 2026-08-09 these fixtures
    // were built from `empty_metrics`, i.e. zero expectancy, and still passed —
    // because the profile had no opinion about money at all. That is exactly the
    // hole this file now guards.
    let mut wanted = profitable_metrics("wanted");
    wanted.win_rate = 0.60;
    wanted.payoff_ratio = 2.21;
    wanted.in_market_pct = 0.20;
    assert!(profile.accepts(&wanted));

    // A trend follower: excellent payoff, far too few winners. A perfectly good
    // system — just not this one, which is the whole reason the two are stated
    // separately instead of through profit factor.
    let mut trend = wanted.clone();
    trend.win_rate = 0.32;
    trend.payoff_ratio = 5.0;
    assert!(!profile.accepts(&trend));

    // A scalper: wins constantly, gives it all back on the losers.
    let mut scalper = wanted.clone();
    scalper.win_rate = 0.82;
    scalper.payoff_ratio = 0.4;
    assert!(!profile.accepts(&scalper));

    // In the market 78 % of the time — the figure a real GPU run reported —
    // is not selecting entries, whatever the other numbers say.
    let mut always_in = wanted.clone();
    always_in.in_market_pct = 0.78;
    assert!(!profile.accepts(&always_in));

    // Unmeasurable exposure must not be read as "never in the market", or the
    // candidates with no exit times sail through the one gate meant for them.
    let mut unknown_exposure = wanted.clone();
    unknown_exposure.in_market_pct = 0.0;
    assert!(profile.accepts(&unknown_exposure));

    // An empty profile states no SHAPE preference, so both survive — but note
    // what changed: "empty" is no longer "vacuous". Both of these are profitable
    // fixtures. Strip the profit and the empty profile refuses them, which is
    // the one opinion the profile is not allowed to drop.
    assert!(TargetProfile::default().accepts(&trend));
    assert!(TargetProfile::default().accepts(&scalper));

    let mut broke = trend.clone();
    broke.profit_per_trade = -0.01;
    assert!(
        !TargetProfile::default().accepts(&broke),
        "no configuration admits a candidate that loses money on the average trade"
    );
}

/// The report must name the mode the engine actually runs.
///
/// `neoethos-core`'s `resolved_config::resolve_discovery_mode` and
/// `neoethos_search::discovery::resolve_discovery_mode` are two functions with
/// the SAME NAME in two crates, and until 2026-08-04 they read different
/// inputs: the engine read `system.trading_mode` + `models.discovery_mode`, the
/// display read only `models.discovery_mode` and could never return "risky".
/// Every Risky run — the mode the operator actually uses — was reported as
/// `prop_firm`, and `models.discovery_mode = "legacy"` (which the engine
/// accepts as Strict) was reported as `prop_firm` too.
///
/// This test lives in neoethos-search because search depends on core and can
/// therefore see both sides. That is exactly why the two drifted: neither
/// crate could check it alone.
#[test]
fn display_mode_matches_the_engine_mode() {
    // (trading_mode, discovery_mode) -> the mode string the report must print.
    let cases: &[(&str, &str, &str, DiscoveryMode)] = &[
        (
            "prop_firm",
            "prop_firm",
            "prop_firm",
            DiscoveryMode::PropFirm,
        ),
        ("risky", "prop_firm", "risky", DiscoveryMode::Risky),
        ("growth", "prop_firm", "risky", DiscoveryMode::Risky),
        // The escape hatch wins over the master switch, in both vocabularies.
        ("risky", "strict", "strict", DiscoveryMode::Strict),
        ("prop_firm", "strict", "strict", DiscoveryMode::Strict),
        ("risky", "legacy", "strict", DiscoveryMode::Strict),
        // Unknown trading modes fall back to prop_firm on both sides.
        ("", "prop_firm", "prop_firm", DiscoveryMode::PropFirm),
        (
            "nonsense",
            "prop_firm",
            "prop_firm",
            DiscoveryMode::PropFirm,
        ),
    ];

    for (trading_mode, discovery_mode, expected_label, expected_engine) in cases {
        let mut settings = neoethos_core::Settings::default();
        settings.system.trading_mode = (*trading_mode).to_string();
        settings.models.discovery_mode = (*discovery_mode).to_string();

        let engine = DiscoveryConfig::from_settings(&settings);
        assert_eq!(
            engine.mode, *expected_engine,
            "engine mode changed for trading_mode={trading_mode:?} \
             discovery_mode={discovery_mode:?}"
        );

        let displayed = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);
        assert_eq!(
            displayed.search.mode, *expected_label,
            "the report says mode {:?} but the engine runs {:?} \
             (trading_mode={trading_mode:?}, discovery_mode={discovery_mode:?}). \
             Change both resolvers or neither.",
            displayed.search.mode, engine.mode
        );
    }
}

/// `models.discovery_mode` is RESTRICTED, not merged into `system.trading_mode`.
///
/// It reaches `Strict`, which `trading_mode` structurally cannot express, so
/// merging the two would delete a regime. What it must NOT do is silently
/// accept `risky` / `prop_firm` — the two values the CLI TUI has been offering
/// — and decide nothing with them, which is what "the knob I set changed
/// nothing" looked like from the operator's chair.
///
/// This locks the accepted set at exactly `strict | legacy`. Everything else
/// returns `None`, which is what makes the fall-through to `system.trading_mode`
/// nameable in the log instead of invisible.
#[test]
fn discovery_mode_accepts_only_strict_and_legacy() {
    for accepted in ["strict", "legacy", "  STRICT  ", "Legacy"] {
        assert_eq!(
            discovery_mode_from_config(accepted),
            Some(DiscoveryMode::Strict),
            "models.discovery_mode = {accepted:?} must select the Strict pipeline"
        );
    }
    for no_op in [
        "",
        "  ",
        "risky",
        "prop_firm",
        "growth",
        "permissive",
        "nonsense",
    ] {
        assert_eq!(
            discovery_mode_from_config(no_op),
            None,
            "models.discovery_mode = {no_op:?} decides NOTHING and must report so, not \
             masquerade as a selected regime"
        );
    }

    // The fall-through itself: a no-op value leaves `system.trading_mode` in
    // charge, in both directions.
    assert_eq!(
        resolve_discovery_mode("risky", "prop_firm"),
        DiscoveryMode::Risky,
        "discovery_mode='prop_firm' must not override trading_mode='risky'"
    );
    assert_eq!(
        resolve_discovery_mode("prop_firm", "risky"),
        DiscoveryMode::PropFirm,
        "discovery_mode='risky' is a no-op — the regime comes from trading_mode"
    );
}

/// The duplicate knobs resolve to the winner this wave documented, and the
/// loser is genuinely ignored.
///
/// Three knobs exist twice under different section names. This test sets the
/// LOSING copy to a value that would be unmistakable if it ever bound, and
/// asserts the winner's number came through. Without it, "which copy wins" is a
/// claim in a comment; with it, changing the precedence fails a test that names
/// the money at stake.
#[test]
fn duplicate_knobs_resolve_to_the_documented_winner() {
    let mut settings = neoethos_core::Settings::default();

    // Winners.
    settings.system.symbol = "EURUSD".to_string();
    settings.system.account_currency = "GBP".to_string();
    settings.risk.backtest_spread_pips = 1.25;
    settings.risk.slippage_pips = 0.25;
    settings.risk.commission_per_lot = 7.0;
    settings.risk.commission_per_lot_is_per_side = false;

    // Losers — deliberately absurd, so binding them would be obvious.
    settings.models.eval_runtime.symbol = Some("XAUUSD".to_string());
    settings.models.eval_runtime.account_currency = Some("JPY".to_string());
    settings.models.eval_runtime.spread_pips = Some(99.0);
    settings.models.eval_runtime.commission_per_trade = Some(999.0);

    let cfg = DiscoveryConfig::from_settings(&settings);

    assert_eq!(
        cfg.evaluation_symbol, "EURUSD",
        "system.symbol wins; models.eval_runtime.symbol must not reach discovery"
    );
    assert_eq!(
        cfg.evaluation_account_currency, "GBP",
        "system.account_currency wins; a wrong currency silently rescales every result"
    );
    assert!(
        (cfg.evaluation_spread_pips - 1.75).abs() < 1e-9,
        "risk.backtest_spread_pips + two risk.slippage_pips fill assumptions wins \
         (expected 1.75, got {}); \
         models.eval_runtime.spread_pips is what the Settings screen calls \
         cost.spread_pips and it must not bind in discovery",
        cfg.evaluation_spread_pips
    );
    assert!(
        cfg.evaluation_commission_per_trade < 100.0,
        "risk.commission_per_lot wins (got {}); models.eval_runtime.commission_per_trade \
         must not bind in discovery",
        cfg.evaluation_commission_per_trade
    );
}

/// The sensitivity ("higher commission") stress pass can never charge LESS than
/// the run it stresses.
///
/// `models.prop_search_sensitivity_commission_per_lot` is assigned straight into
/// `BacktestSettings::commission_per_trade`, which every evaluator subtracts
/// exactly once per closed trade — the same contract as the baseline. It was the
/// one commission input that skipped `round_trip_commission_per_lot`, so at the
/// shipped defaults (7.0 here, 7.0 per side → 14.0 round trip on the baseline)
/// the stress scenario cost half the baseline and every candidate passed it.
#[test]
fn sensitivity_commission_is_round_trip_and_never_below_the_baseline() {
    let mut settings = neoethos_core::Settings::default();
    settings.risk.commission_per_lot = 7.0;
    settings.risk.commission_per_lot_is_per_side = true;

    // The shipped state: the same quote on both knobs.
    settings.models.prop_search_sensitivity_commission_per_lot = 7.0;
    let cfg = DiscoveryConfig::from_settings(&settings);
    assert!(
        cfg.sensitivity_commission_per_lot >= cfg.evaluation_commission_per_trade,
        "the stress pass charged {} while the baseline charged {} — a cheaper \
         stress test passes everything",
        cfg.sensitivity_commission_per_lot,
        cfg.evaluation_commission_per_trade
    );

    // A genuinely harsher quote survives the conversion and is NOT clamped down.
    settings.models.prop_search_sensitivity_commission_per_lot = 20.0;
    let cfg = DiscoveryConfig::from_settings(&settings);
    assert!(
        (cfg.sensitivity_commission_per_lot - 40.0).abs() < 1e-9,
        "a per-side quote of 20.0 is a 40.0 round trip, got {}",
        cfg.sensitivity_commission_per_lot
    );

    // With a round-trip quote the flag must not double anything.
    settings.risk.commission_per_lot_is_per_side = false;
    settings.models.prop_search_sensitivity_commission_per_lot = 20.0;
    let cfg = DiscoveryConfig::from_settings(&settings);
    assert!(
        (cfg.sensitivity_commission_per_lot - 20.0).abs() < 1e-9,
        "a round-trip quote must pass through unchanged, got {}",
        cfg.sensitivity_commission_per_lot
    );
}

/// `models.discovery_runtime` is the ONLY operator input for the discovery
/// runtime knobs, and an out-of-range value keeps the default.
///
/// The six `NEOETHOS_BOT_PREFILTER_*` / `_FUNNEL_*` / `_MIN_HISTORY_YEARS`
/// names were deleted on 2026-08-10. `prefilter_top_k` is the exact key
/// `shipped_config_matches_defaults.rs` exists to protect — at 50 the base
/// feature set collapses from 217 columns to roughly 64, with the SMC, session
/// and footprint families dying first — so a second, invisible way to set it
/// was the highest-cost env var in the crate.
#[test]
fn discovery_runtime_reads_config_and_keeps_the_default_on_garbage() {
    let mut settings = neoethos_core::Settings::default();
    settings.models.discovery_runtime.prefilter_top_k = 240;
    settings.models.discovery_runtime.prefilter_insample_frac = f64::NAN;
    settings.models.discovery_runtime.funnel_stage1_pct = 5.0;
    settings.models.discovery_runtime.stage1_window = "sideways".to_string();

    let resolved = DiscoveryRuntimeOverrides::from_settings(&settings);
    let default = DiscoveryRuntimeOverrides::default();

    assert_eq!(resolved.prefilter_top_k, 240);
    assert_eq!(
        resolved.prefilter_insample_frac, default.prefilter_insample_frac,
        "a non-finite fraction must keep the default, and say so"
    );
    assert!(
        (resolved.funnel_stage1_pct - 1.0).abs() < 1e-9,
        "an out-of-range funnel fraction is clamped to 1.0, not accepted"
    );
    assert_eq!(
        resolved.stage1_window, default.stage1_window,
        "an unrecognised stage1_window keeps the OOS-safe default"
    );
}

/// The report must describe the engine it reports on — for EVERY mode, and by
/// reading what the report ACTUALLY prints.
///
/// Supersedes the 2026-08-03 version of this test, which asserted against
/// display literals RETYPED into the test body:
///
/// ```ignore
/// let displayed_max_drawdown = 0.15_f64;   // <- a third copy, not the display
/// assert_eq!(enforced.max_dd, displayed_max_drawdown, ...);
/// ```
///
/// That guards one edge of a triangle. Measured 2026-08-04: with
/// `resolved_config.rs`'s floors edited to 0.20 / 0.9 / 0.99 / 9.9, the old
/// test still reported `ok` — it could not fail for display drift, which is
/// the only drift it was written to catch. It now calls
/// `ResolvedConfig::from_settings` and compares the real thing.
///
/// `min_fitness_score` and `min_trades` are still excluded: both are
/// config-driven on the display side rather than mode-derived.
#[test]
fn display_floors_match_the_enforced_ones() {
    for (trading_mode, discovery_mode) in [
        ("prop_firm", "prop_firm"),
        ("risky", "prop_firm"),
        ("prop_firm", "strict"),
    ] {
        let mut settings = neoethos_core::Settings::default();
        settings.system.trading_mode = trading_mode.to_string();
        settings.models.discovery_mode = discovery_mode.to_string();

        // What the engine will enforce: `from_settings` applies the mode
        // overrides, so this is the post-override floor set.
        let enforced = DiscoveryConfig::from_settings(&settings).filtering;
        // What `neoethos-cli config` and the Settings UI will print.
        let shown =
            neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings).filters;

        let ctx = format!("trading_mode={trading_mode:?} discovery_mode={discovery_mode:?}");
        assert_eq!(
            shown.max_drawdown, enforced.max_dd,
            "[{ctx}] report shows max drawdown {} but the engine enforces {}. \
             Change both or neither.",
            shown.max_drawdown, enforced.max_dd
        );
        assert_eq!(
            shown.min_sharpe, enforced.min_sharpe,
            "[{ctx}] report shows min sharpe {} but the engine enforces {}. \
             Change both or neither.",
            shown.min_sharpe, enforced.min_sharpe
        );
        assert_eq!(
            shown.min_win_rate, enforced.min_win_rate,
            "[{ctx}] report shows min win rate {} but the engine enforces {}. \
             Change both or neither.",
            shown.min_win_rate, enforced.min_win_rate
        );
        assert_eq!(
            shown.min_profit_factor, enforced.min_profit_factor,
            "[{ctx}] report shows min profit factor {} but the engine enforces {}. \
             Change both or neither.",
            shown.min_profit_factor, enforced.min_profit_factor
        );
    }
}

#[test]
fn choosing_a_mode_changes_what_the_mode_says_it_changes() {
    // `apply_mode_overrides` was called from tests and nowhere else, so every
    // real run used the struct defaults regardless of `trading_mode`: max_dd
    // 0.15, min_win_rate 0.50, min_profit_factor 1.20. Risky's 0.60 cap and
    // PropFirm's 0.50 existed and never took effect, and the search found
    // nothing in any mode — 1 713 of 2 211 candidates rejected against a
    // drawdown cap the selected mode had raised.
    let mut settings = neoethos_core::Settings::default();

    settings.system.trading_mode = "risky".to_string();
    settings.models.discovery_mode = "prop_firm".to_string();
    let risky = DiscoveryConfig::from_settings(&settings);
    assert_eq!(risky.mode, DiscoveryMode::Risky);
    assert!(
        risky.filtering.max_dd > 0.5,
        "risky kept the default cap: {}",
        risky.filtering.max_dd
    );

    settings.system.trading_mode = "prop_firm".to_string();
    let prop = DiscoveryConfig::from_settings(&settings);
    assert_eq!(prop.mode, DiscoveryMode::PropFirm);
    assert!(
        prop.filtering.max_dd > 0.3,
        "prop_firm kept the default cap: {}",
        prop.filtering.max_dd
    );

    // Strict is the one mode whose floors ARE the defaults, so it is the
    // control: if this loosened too, the overrides would be firing for
    // everyone rather than per mode.
    settings.models.discovery_mode = "strict".to_string();
    let strict = DiscoveryConfig::from_settings(&settings);
    assert_eq!(strict.mode, DiscoveryMode::Strict);
    assert!(
        strict.filtering.max_dd <= 0.2,
        "strict should stay tight: {}",
        strict.filtering.max_dd
    );
}

/// The "nonzero_signals" stage builds the SMC gate arrays once for the whole
/// candidate pool instead of once per candidate. Pins that the pool-wide build
/// changes nothing the funnel reads: same survivors, same order, same signal
/// vectors, same "fired at all" count as screening each candidate on its own.
///
/// The fixture is 100 bars, so this pins arithmetic only. The stage exists as
/// a hoist because the rebuild is a full-series cost repeated per candidate —
/// that is not observable at this size and is not claimed here.
#[test]
fn signal_count_screen_matches_screening_each_candidate_alone() {
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let eval_config =
        EvaluationConfig::for_symbol("EURUSD", "USD", ohlcv.close.last().copied(), None, None);

    let base = Gene {
        indices: vec![0, 1],
        weights: vec![1.0, 0.5],
        long_threshold: 0.4,
        short_threshold: -0.4,
        ..Gene::default()
    };
    let candidates: Vec<(usize, Gene)> = vec![
        (
            7,
            Gene {
                strategy_id: "no-flags".to_string(),
                ..base.clone()
            },
        ),
        (
            2,
            Gene {
                strategy_id: "ob".to_string(),
                use_ob: true,
                ..base.clone()
            },
        ),
        (
            11,
            Gene {
                strategy_id: "structure".to_string(),
                use_bos: true,
                use_choch: true,
                ..base.clone()
            },
        ),
        // Never crosses its long threshold: exercises the zero-signal branch
        // the funnel reports as "zero_signals_after_smc_gate".
        (
            3,
            Gene {
                strategy_id: "silent".to_string(),
                long_threshold: 1e9,
                short_threshold: -1e9,
                ..base.clone()
            },
        ),
        (
            5,
            Gene {
                strategy_id: "all-flags".to_string(),
                use_ob: true,
                use_fvg: true,
                use_liq_sweep: true,
                mtf_confirmation: true,
                use_premium_discount: true,
                use_inducement: true,
                use_bos: true,
                use_choch: true,
                use_eqh: true,
                use_eql: true,
                use_displacement: true,
                ..base
            },
        ),
    ];
    let min_trades = 3usize;

    let expected: Vec<(usize, Gene, Vec<i8>)> = candidates
        .iter()
        .filter_map(|(idx, gene)| {
            let sig = signals_for_gene_full(&features, &ohlcv, gene, &eval_config)
                .expect("valid test signal inputs");
            let firing = sig.iter().filter(|v| **v != 0).count();
            (firing >= min_trades).then(|| (*idx, gene.clone(), sig))
        })
        .collect();
    let expected_nonzero = candidates
        .iter()
        .filter(|(_, gene)| {
            signals_for_gene_full(&features, &ohlcv, gene, &eval_config)
                .expect("valid test signal inputs")
                .iter()
                .any(|v| *v != 0)
        })
        .count();

    // Both branches of the funnel's two counters must be exercised, otherwise
    // the comparison below cannot catch a regression in either.
    assert!(
        !expected.is_empty() && expected.len() < candidates.len(),
        "fixture must both keep and drop candidates (kept {} of {})",
        expected.len(),
        candidates.len()
    );
    assert!(expected_nonzero > 0 && expected_nonzero < candidates.len());

    let (survivors, nonzero) =
        screen_candidates_by_signal_count(&features, &ohlcv, candidates, &eval_config, min_trades)
            .expect("candidate screen succeeds");
    assert_eq!(nonzero, expected_nonzero, "'fired at all' count");
    assert_eq!(survivors.len(), expected.len(), "survivor count");
    for (got, want) in survivors.iter().zip(expected.iter()) {
        assert_eq!(got.0, want.0, "candidate index (order must be preserved)");
        assert_eq!(got.1, want.1, "gene");
        assert_eq!(got.2, want.2, "signal vector for candidate {}", want.0);
    }
}

/// Pins that the screen builds the SMC gate arrays ONCE for the whole pool.
///
/// A reviewer moved the build back inside the `filter_map` — strictly worse
/// than the code before the hoist, since it also pays the row-pack per
/// candidate — and all 379 tests stayed green. The saving could be handed back
/// with no signal at all, which makes this the only assertion that defends it.
#[test]
fn the_screen_builds_the_gate_arrays_once_for_the_whole_pool() {
    use crate::genetic::search_engine::SMC_GATE_BUILD_CALLS;

    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let eval_config =
        EvaluationConfig::for_symbol("EURUSD", "USD", ohlcv.close.last().copied(), None, None);
    let base = Gene {
        indices: vec![0, 1],
        weights: vec![1.0, 0.5],
        long_threshold: 0.4,
        short_threshold: -0.4,
        ..Gene::default()
    };
    let genes: Vec<(usize, Gene)> = (0..6)
        .map(|i| {
            (
                i,
                Gene {
                    strategy_id: format!("g{i}"),
                    ..base.clone()
                },
            )
        })
        .collect();

    // One rayon worker, so every build the screen triggers is counted on the
    // thread doing the counting. With the default pool the work is stolen
    // across threads and a thread-local counter sees a fraction of the truth —
    // which is the same class of mistake as the process-global it replaced.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-worker pool");
    let built = pool.install(|| {
        let before = SMC_GATE_BUILD_CALLS.with(|c| c.get());
        let _ = super::screen_candidates_by_signal_count(&features, &ohlcv, genes, &eval_config, 0);
        SMC_GATE_BUILD_CALLS.with(|c| c.get()) - before
    });

    assert_eq!(
        built, 1,
        "screening six candidates built the gate arrays {built} times — the whole point          of the hoist is that it is one, whatever the pool size"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 2026-08-04 — "one computation, several call sites, each passing something
// different, while a comment claims they agree".
//
// `DiscoveryConfig` has TWO production constructors:
//
//   * `DiscoveryConfig::from_settings(&settings)` — the normal path.
//   * `DiscoveryConfig::default()` — the fallback taken when the settings
//     file cannot be read. `neoethos_app::server::engines_control` does
//     exactly this:
//         Some(settings) => DiscoveryConfig::from_settings(settings),
//         None           => DiscoveryConfig::default(),
//     and `neoethos_app::main`'s headless loop constructs one directly.
//
// `default()` hand-writes ~50 literals and its comments assert they match
// the config side ("these defaults match ModelsConfig::default", "Search-
// memory ledger defaults mirror DiscoveryLedgerConfig::default"). Nothing
// checked that claim. A hand-copied literal that drifts from the config
// default does not fail to compile, does not warn, and produces a run that
// looks completely normal — the filter floors drifted five points exactly
// this way.
//
// These tests make the claim executable. The first compares the whole
// `Debug` rendering rather than a hand-maintained field list, because a
// hand-maintained list would need the same discipline it is meant to
// enforce: add a field to `DiscoveryConfig`, forget to wire it into
// `from_settings`, and this fails.
// ─────────────────────────────────────────────────────────────────────────

/// Fields `default()` deliberately leaves as "unset" sentinels which
/// `from_settings` fills from the operator's config. These are the
/// documented, intentional differences — everything else must agree.
fn discovery_config_intentional_divergences() -> &'static [&'static str] {
    &[
        // GROUP C sentinels: `default()` uses empty / NaN so a config that
        // skipped `for_symbol` cannot silently backtest EURUSD/USD.
        "symbol",
        "timeframe_label",
        "evaluation_symbol",
        "evaluation_account_currency",
        "evaluation_spread_pips",
        "evaluation_commission_per_trade",
    ]
}

/// The fields on which the two production constructors are KNOWN to
/// disagree, as measured on 2026-08-04. This list is a **defect record,
/// not a specification** — every entry is a parameter whose value depends
/// on whether `config.yaml` could be read, with nothing in the resulting
/// artifact saying which branch ran.
///
/// The two that change reported NUMBERS rather than just search effort:
///   * `initial_balance` — 100 000 vs 10 000. It is the denominator of
///     every drawdown-% and PnL-% the run reports, so the same trades
///     produce different headline figures on the two branches.
///   * `higher_timeframes` — empty vs eleven. The fallback searches
///     single-timeframe while the configured path is multi-resolution.
///
/// The test below fails if this list GROWS. Shrinking it (by making the
/// two constructors agree, or by deleting `Default` and forcing the
/// fallback to fail loud) is the fix; the list is here so the debt is
/// counted rather than rediscovered.
fn discovery_config_known_default_vs_settings_divergences() -> &'static [&'static str] {
    &[
        "population",
        "generations",
        "max_indicators",
        "portfolio_size",
        "max_hours",
        "walkforward_splits",
        "cpcv_max_rows",
        "initial_balance",
        "risk_per_trade_min",
        "higher_timeframes",
        // `filtering.*` — the opportunistic-candidate lane is entirely OFF
        // on the fallback branch and entirely ON on the configured one.
        "opportunistic_enabled",
        "use_opportunistic_candidates",
        "opportunistic_min_positive_months",
        "opportunistic_min_trades_per_month",
        "opportunistic_min_trade_return_pct",
        "opportunistic_max_dd",
    ]
}

/// Replace every named field's rendered VALUE with a placeholder, keeping
/// the field itself in place. Block-aware: a value that opens `[`, `{` or
/// `(` is collapsed together with all of its nested lines, so a field that
/// renders across many lines on one side and one line on the other cannot
/// desynchronise the comparison that follows.
fn redact_fields(rendered: &str, fields: &[&str]) -> String {
    let lines: Vec<&str> = rendered.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
        let indent = line.len() - line.trim_start().len();
        let name = line.trim().split(':').next().unwrap_or("").trim();
        if line.contains(':') && fields.contains(&name) {
            out.push(format!("{}{}: <redacted>", " ".repeat(indent), name));
            let opens = line.trim_end().ends_with('[')
                || line.trim_end().ends_with('{')
                || line.trim_end().ends_with('(');
            idx += 1;
            if opens {
                // Consume until the closer sitting at the field's own indent.
                while idx < lines.len() {
                    let cur = lines[idx];
                    let cur_indent = cur.len() - cur.trim_start().len();
                    idx += 1;
                    if cur_indent == indent {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(line.trim_end().to_string());
        idx += 1;
    }
    out.join("\n")
}

#[test]
fn discovery_config_default_vs_from_settings_divergence_does_not_grow() {
    // Apples to apples: `from_settings` ENDS with `.apply_mode_overrides()`,
    // so the bare `default()` gets exactly one application too. Comparing
    // raw `default()` against `from_settings` would compare zero
    // applications against one and report a difference that is not drift.
    let settings = neoethos_core::Settings::default();

    let mut redacted: Vec<&str> = Vec::new();
    redacted.extend_from_slice(discovery_config_intentional_divergences());
    redacted.extend_from_slice(discovery_config_known_default_vs_settings_divergences());

    let from_defaults = redact_fields(
        &format!("{:#?}", DiscoveryConfig::from_settings(&settings)),
        &redacted,
    );
    let bare_default = redact_fields(
        &format!("{:#?}", DiscoveryConfig::default().apply_mode_overrides()),
        &redacted,
    );

    assert_eq!(
        bare_default, from_defaults,
        "A NEW field diverges between DiscoveryConfig::default() and \
         DiscoveryConfig::from_settings(&Settings::default()).\n\
         \n\
         Both are PRODUCTION constructors. `engines_control` falls back to `default()` \
         whenever config.yaml fails to load, so a discovery run searches with different \
         parameters depending on whether the file could be read — and the artifact does not \
         record which branch ran. Either wire the new field into `from_settings`, or add it \
         to `discovery_config_known_default_vs_settings_divergences` with a reason."
    );
}

/// The redaction list must stay honest: every field named in it has to
/// ACTUALLY diverge. A stale entry would silently widen the hole the test
/// above is guarding.
#[test]
fn every_known_divergence_is_still_a_real_divergence() {
    let settings = neoethos_core::Settings::default();
    let from_defaults = format!("{:#?}", DiscoveryConfig::from_settings(&settings));
    let bare_default = format!("{:#?}", DiscoveryConfig::default().apply_mode_overrides());

    for field in discovery_config_known_default_vs_settings_divergences() {
        let a = redact_fields(&bare_default, &[field]);
        let b = redact_fields(&from_defaults, &[field]);
        // Redacting a field that genuinely differs must remove a difference;
        // if the renderings were already equal on it, the entry is stale.
        assert_ne!(
            (bare_default.clone(), from_defaults.clone()),
            (a, b),
            "`{field}` is listed as a known divergence but the two constructors agree on it \
             — remove it from discovery_config_known_default_vs_settings_divergences"
        );
    }
}

/// `apply_mode_overrides` MULTIPLIES `min_trades_per_month` by a
/// per-timeframe scale factor. Multiplication is not idempotent, and
/// `DiscoveryConfig::from_settings` already ends with
/// `.apply_mode_overrides()` — so every caller that applies it again
/// squares the scale factor and searches with a floor several times
/// looser than the one the code's own comment documents
/// ("H4: 0.20× (15 → 3/month)").
///
/// The counts differ per entry point, traced 2026-08-04:
///
/// * **UI Discovery — 3 applications.**
///   `engines_control::start_discovery` builds the config with
///   `DiscoveryConfig::from_settings(settings)` (1st, inside
///   `from_settings`), applies `config = config.apply_mode_overrides()`
///   before assembling the request (2nd), and the resulting
///   `DiscoveryRequest.config` reaches
///   `app_services::discovery`'s
///   `search_request.config.clone().apply_mode_overrides()` (3rd).
/// * **CLI — 2 applications.** `from_settings` (1st) then the
///   `DiscoveryConfig { .., ..defaults.clone() }.apply_mode_overrides()`
///   in `neoethos-cli` (2nd).
/// * **The code comment describes 1.**
///
/// On H4 with the operator's `prop_search_val_min_trades_per_month: 15`
/// that is 3.0 documented, 0.6 on the CLI, 0.5 (the `.max(0.5)` clamp)
/// on the UI — three answers to one knob, none of them announced.
/// Intra-day timeframes have scale 1.0 and are unaffected; see
/// `intraday_timeframes_are_immune_to_the_repeated_mode_override`.
///
/// This test states the arithmetic. It does not assert which call count
/// is correct — that is the operator's call — it asserts that the count
/// CHANGES THE ANSWER, which is what makes "how many times did this run"
/// a silent correctness question instead of a no-op.
#[test]
fn apply_mode_overrides_is_not_idempotent_for_the_timeframe_scaled_floor() {
    // The operator's own config.yaml sets
    // `models.prop_search_val_min_trades_per_month: 15`.
    let mut settings = neoethos_core::Settings::default();
    settings.models.prop_search_val_min_trades_per_month = 15;
    settings.models.discovery_mode = "prop_firm".to_string();
    settings.system.trading_mode = "prop_firm".to_string();

    // H4: the documented factor is 0.20 → 15 becomes 3 trades/month.
    let mut once = DiscoveryConfig::from_settings(&settings);
    once.timeframe_label = "H4".to_string();
    let once = once.apply_mode_overrides();
    let twice = once.clone().apply_mode_overrides();
    let thrice = twice.clone().apply_mode_overrides();

    assert_eq!(
        once.filtering.min_trades_per_month, 3.0,
        "one application must reproduce the documented H4 factor (15 × 0.20)"
    );
    assert!(
        (twice.filtering.min_trades_per_month - 0.6).abs() < 1e-9,
        "a second application squares the factor: got {}",
        twice.filtering.min_trades_per_month
    );
    assert!(
        (thrice.filtering.min_trades_per_month - 0.5).abs() < 1e-9,
        "a third cubes it and lands on the .max(0.5) clamp: got {}",
        thrice.filtering.min_trades_per_month
    );
    assert_ne!(
        once.filtering.min_trades_per_month, twice.filtering.min_trades_per_month,
        "if this ever becomes equal the hazard is gone and this test can go"
    );
}

/// The intra-day timeframes the operator actually runs are UNAFFECTED,
/// because their scale factor is exactly 1.0 and the block is guarded by
/// `scale < 1.0`. Pinning this keeps the blast radius of the hazard above
/// honest and stops it being over-claimed.
#[test]
fn intraday_timeframes_are_immune_to_the_repeated_mode_override() {
    let mut settings = neoethos_core::Settings::default();
    settings.models.prop_search_val_min_trades_per_month = 15;
    settings.models.discovery_mode = "prop_firm".to_string();
    settings.system.trading_mode = "prop_firm".to_string();

    for tf in ["M1", "M3", "M5", "M15"] {
        let mut cfg = DiscoveryConfig::from_settings(&settings);
        cfg.timeframe_label = tf.to_string();
        let once = cfg.apply_mode_overrides();
        let twice = once.clone().apply_mode_overrides();
        assert_eq!(
            once.filtering.min_trades_per_month, twice.filtering.min_trades_per_month,
            "{tf} has scale 1.0, so repeated application must be a no-op"
        );
    }
}

#[test]
fn discovery_ledger_defaults_are_not_a_hand_copy_that_can_drift() {
    // `DiscoveryConfig::default()` writes these three literals under the
    // comment "Search-memory ledger defaults mirror DiscoveryLedgerConfig::
    // default". Make the mirror executable.
    let cfg = DiscoveryConfig::default();
    let ledger = neoethos_core::config::DiscoveryLedgerConfig::default();
    assert_eq!(cfg.discovery_ledger_enabled, ledger.enabled);
    assert_eq!(cfg.discovery_ledger_cache_dir, ledger.cache_dir);
    assert_eq!(cfg.discovery_ledger_archive_top_n, ledger.archive_top_n);
}

#[test]
fn walkforward_export_defaults_are_not_a_hand_copy_that_can_drift() {
    // Same for the two the comment claims "match ModelsConfig::default".
    let cfg = DiscoveryConfig::default();
    let models = neoethos_core::config::ModelsConfig::default();
    assert_eq!(
        cfg.require_walkforward_for_export,
        models.require_walkforward_for_export
    );
    assert_eq!(cfg.prop_firm_min_pass_rate, models.prop_firm_min_pass_rate);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLICE 5 (2026-08-08): the env-knob census — "can a run name its own
// arithmetic?"
//
// Scans every `.rs` file in this crate for env-var names, then requires each
// discovered name to be CLASSIFIED: either it is recorded in the serialized
// `DiscoveryRunProfile` (with a JSON pointer this test VERIFIES resolves), or
// it is explicitly declared diagnostic-only with a written justification.
// A new knob that skips the profile fails this test with instructions.
// NEOETHOS_GPU_F64 (once chose the GPU kernel precision without authority) is
// the proven failure mode this ratchet exists to prevent. It is now retired:
// CubeCL search arithmetic is unconditionally f64.
// ═══════════════════════════════════════════════════════════════════════════

/// How a discovered env knob is accounted for.
enum KnobClass {
    /// The knob's resolved value (or raw ambient value, for GPU env
    /// overrides whose resolvers are cfg-gated) appears at this JSON
    /// pointer in the serialized `DiscoveryRunProfile`.
    Profile(&'static str),
    /// The knob cannot change what the search selects — logging/timing only.
    /// The string is the justification, kept next to the exemption (read by
    /// humans reviewing the exemption, not by the assertions).
    DiagnosticOnly(#[allow(dead_code)] &'static str),
    // DELETED 2026-08-10: a third variant, `Retired(&str)`. Its doc claimed
    // "the assertion below requires every name classified here to be present in
    // RETIRED_ENV_VARS or RETIRED_SEARCH_ENV_VARS" — and no such assertion was
    // ever written, nor was the variant ever constructed. It was an exemption
    // route with an imaginary guard, i.e. the exact shape this census exists to
    // catch, sitting inside the census. A retired knob whose config successor
    // is recorded takes a `Profile` row (see NEOETHOS_FEATURE_CUBE_MODE); one
    // that truly decides nothing takes `DiagnosticOnly` with its justification.
}

fn collect_env_knob_names_in_crate_sources() -> std::collections::BTreeSet<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let entries = std::fs::read_dir(dir).expect("crate src dir must be readable");
        for entry in entries {
            let path = entry.expect("dir entry must be readable").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    // Tokens that do not carry the NEOETHOS_ prefix but are real env knobs
    // read by this crate. Extend when a new foreign-prefix knob appears.
    const FOREIGN_PREFIX_KNOBS: [&str; 2] = ["RAYON_NUM_THREADS", "FOREX_TRAIN_PRECISION"];

    let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src_root, &mut files);
    assert!(
        files.len() > 20,
        "source walk found only {} files — the census is scanning the wrong directory",
        files.len()
    );

    let mut names = std::collections::BTreeSet::new();
    // Built via concat so the needle itself does not end up in the census
    // (it would register as a stray prefix fragment otherwise).
    let needle = concat!("NEO", "ETHOS_");
    for file in files {
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("census cannot read {}: {e}", file.display()));
        for (start, _) in text.match_indices(needle) {
            let tail = &text[start..];
            let token: String = tail
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            // Fragments like `NEOETHOS_BOT_` (doc-comment prefixes) end with
            // an underscore and are not knob names.
            if !token.ends_with('_') {
                names.insert(token);
            }
        }
        for foreign in FOREIGN_PREFIX_KNOBS {
            if text.contains(foreign) {
                names.insert(foreign.to_string());
            }
        }
    }
    names
}

#[test]
fn every_env_knob_is_classified_and_recorded_in_the_run_profile() {
    use KnobClass::{DiagnosticOnly, Profile};

    // The complete classification. ORDER: alphabetical within each group.
    // To add a knob: give it a profile field (preferred) and point at it, or
    // justify why it cannot change selection.
    let table: &[(&str, KnobClass)] = &[
        // ── GA selection knobs (genetic::runtime_overrides) ──
        (
            "NEOETHOS_BOT_DISABLE_SMC_GATE",
            Profile("/execution/genetic_search/smc_gate/disable_gate"),
        ),
        (
            "NEOETHOS_BOT_NOVELTY_WEIGHT",
            Profile("/execution/genetic_search/novelty_weight"),
        ),
        (
            "NEOETHOS_BOT_PROP_ARCHIVE_CAP",
            Profile("/execution/genetic_search/archive_cap_override"),
        ),
        (
            "NEOETHOS_BOT_PROP_ARCHIVE_MIN_NET",
            Profile("/execution/genetic_search/archive_scoring/min_net"),
        ),
        (
            "NEOETHOS_BOT_PROP_ARCHIVE_MIN_PF",
            Profile("/execution/genetic_search/archive_scoring/min_pf"),
        ),
        (
            "NEOETHOS_BOT_PROP_ARCHIVE_MIN_SHARPE",
            Profile("/execution/genetic_search/archive_scoring/min_sharpe"),
        ),
        (
            "NEOETHOS_BOT_PROP_ARCHIVE_MODE",
            Profile("/execution/genetic_search/archive_scoring/mode"),
        ),
        (
            "NEOETHOS_BOT_PROP_CONVERGENCE_GENS",
            Profile("/execution/genetic_search/convergence_patience"),
        ),
        (
            "NEOETHOS_BOT_PROP_CONVERGENCE_MIN_ELAPSED_FRAC",
            Profile("/execution/genetic_search/convergence_min_elapsed_fraction"),
        ),
        (
            "NEOETHOS_BOT_PROP_ELITE_FRACTION",
            Profile("/execution/genetic_search/selection/survivor_fraction"),
        ),
        (
            "NEOETHOS_BOT_PROP_MIN_IMPROVEMENT",
            Profile("/execution/genetic_search/min_improvement"),
        ),
        (
            "NEOETHOS_BOT_PROP_PARENT_SELECTION",
            Profile("/execution/genetic_search/selection/parent"),
        ),
        (
            "NEOETHOS_BOT_PROP_RANDOM_IMMIGRANTS",
            Profile("/execution/genetic_search/selection/immigrant_ratio"),
        ),
        (
            "NEOETHOS_BOT_PROP_SEEN_RETRY",
            Profile("/execution/genetic_search/seen_retry_attempts"),
        ),
        (
            "NEOETHOS_BOT_PROP_SELECTION_TEMPERATURE",
            Profile("/execution/genetic_search/selection/temperature"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_GATE_CURVE",
            Profile("/execution/genetic_search/smc_gate/curve"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_GATE_END",
            Profile("/execution/genetic_search/smc_gate/end"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_GATE_STAGNATION_STEP",
            Profile("/execution/genetic_search/smc_gate/stagnation_step"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_GATE_START",
            Profile("/execution/genetic_search/smc_gate/start"),
        ),
        (
            "NEOETHOS_BOT_PROP_STAGNATION_GENS",
            Profile("/execution/genetic_search/stagnation_patience"),
        ),
        (
            "NEOETHOS_BOT_PROP_SURVIVOR_FRACTION",
            Profile("/execution/genetic_search/selection/survivor_fraction"),
        ),
        (
            "NEOETHOS_BOT_PROP_SURVIVOR_SELECTION",
            Profile("/execution/genetic_search/selection/survivor"),
        ),
        (
            "NEOETHOS_BOT_PROP_TOURNAMENT_SIZE",
            Profile("/execution/genetic_search/tournament_size_override"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_SEED",
            Profile("/execution/genetic_search/seed"),
        ),
        // ── Evaluation cost profile + SMC weights ──
        (
            "NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY",
            Profile("/execution/strategy_eval/cost_profile/account_currency"),
        ),
        (
            "NEOETHOS_BOT_PROP_COMMISSION",
            Profile("/execution/strategy_eval/cost_profile/commission_per_trade"),
        ),
        (
            "NEOETHOS_BOT_PROP_PIP_VALUE",
            Profile("/execution/strategy_eval/cost_profile/pip_value"),
        ),
        (
            "NEOETHOS_BOT_PROP_PIP_VALUE_PER_LOT",
            Profile("/execution/strategy_eval/cost_profile/pip_value_per_lot"),
        ),
        (
            "NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE",
            Profile("/execution/strategy_eval/cost_profile/quote_to_account_rate"),
        ),
        (
            "NEOETHOS_BOT_PROP_SPREAD_PIPS",
            Profile("/execution/strategy_eval/cost_profile/spread_pips"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_GATE",
            Profile("/execution/strategy_eval/smc_weights/gate_threshold"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_BOS",
            Profile("/execution/strategy_eval/smc_weights/w_bos"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_CHOCH",
            Profile("/execution/strategy_eval/smc_weights/w_choch"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_DISPLACEMENT",
            Profile("/execution/strategy_eval/smc_weights/w_displacement"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_EQH",
            Profile("/execution/strategy_eval/smc_weights/w_eqh"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_EQL",
            Profile("/execution/strategy_eval/smc_weights/w_eql"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_FVG",
            Profile("/execution/strategy_eval/smc_weights/w_fvg"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_INDUCEMENT",
            Profile("/execution/strategy_eval/smc_weights/w_inducement"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_LIQ",
            Profile("/execution/strategy_eval/smc_weights/w_liq"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_MTF",
            Profile("/execution/strategy_eval/smc_weights/w_mtf"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_OB",
            Profile("/execution/strategy_eval/smc_weights/w_ob"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_W_PREMIUM",
            Profile("/execution/strategy_eval/smc_weights/w_premium"),
        ),
        (
            "NEOETHOS_BOT_PROP_SYMBOL",
            Profile("/execution/strategy_eval/cost_profile/symbol"),
        ),
        (
            "NEOETHOS_BOT_REJECT_PIP_FALLBACK",
            Profile("/execution/strategy_eval/cost_profile/reject_pip_fallback"),
        ),
        // ── Backtest arithmetic + threads ──
        (
            "NEOETHOS_BOT_BACKTEST_INITIAL_EQUITY",
            Profile("/execution/backtest/initial_equity"),
        ),
        (
            "NEOETHOS_BOT_BACKTEST_MAX_MONTH_BUCKETS",
            Profile("/execution/backtest/month_capacity"),
        ),
        (
            "NEOETHOS_BOT_RUST_THREADS",
            Profile("/execution/backtest/rayon_threads"),
        ),
        (
            "RAYON_NUM_THREADS",
            Profile("/execution/backtest/rayon_threads"),
        ),
        // ── Quality screen ──
        (
            "NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH",
            Profile("/execution/quality/min_trades_per_month"),
        ),
        (
            "NEOETHOS_BOT_TRADING_DAYS_PER_MONTH",
            Profile("/execution/quality/trading_days_per_month"),
        ),
        // ── Seen-signature memory (CROSS-RUN state) ──
        (
            "NEOETHOS_BOT_PROP_SEEN_FILE",
            Profile("/execution/seen_memory/file_path"),
        ),
        (
            "NEOETHOS_BOT_PROP_SEEN_FLUSH_EVERY",
            Profile("/execution/seen_memory/flush_every"),
        ),
        (
            "NEOETHOS_BOT_PROP_SEEN_LOAD_MAX",
            Profile("/execution/seen_memory/load_max"),
        ),
        (
            "NEOETHOS_BOT_PROP_SEEN_MAX_ENTRIES",
            Profile("/execution/seen_memory/max_entries"),
        ),
        // ── SMC gene-injection probabilities ──
        // ENABLE_P is the umbrella default for every p_* field and
        // FORCE_ENABLED=false zeroes force_ratio+min_flags; the profile
        // records the RESOLVED per-field values, so both umbrellas are
        // covered by the fields they feed.
        (
            "NEOETHOS_BOT_PROP_SMC_ENABLE_P",
            Profile("/execution/smc_search/p_ob"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_FORCE_ENABLED",
            Profile("/execution/smc_search/force_ratio"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_FORCE_RATIO",
            Profile("/execution/smc_search/force_ratio"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_MIN_FLAGS",
            Profile("/execution/smc_search/min_flags"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_BOS",
            Profile("/execution/smc_search/p_bos"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_CHOCH",
            Profile("/execution/smc_search/p_choch"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_DISPLACEMENT",
            Profile("/execution/smc_search/p_displacement"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_EQH",
            Profile("/execution/smc_search/p_eqh"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_EQL",
            Profile("/execution/smc_search/p_eql"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_FVG",
            Profile("/execution/smc_search/p_fvg"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_INDUCEMENT",
            Profile("/execution/smc_search/p_inducement"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_LIQ",
            Profile("/execution/smc_search/p_liq"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_MTF",
            Profile("/execution/smc_search/p_mtf"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_OB",
            Profile("/execution/smc_search/p_ob"),
        ),
        (
            "NEOETHOS_BOT_PROP_SMC_P_PREMIUM",
            Profile("/execution/smc_search/p_premium"),
        ),
        // ── Adaptive stops ──
        (
            "NEOETHOS_ADAPTIVE_STOPS",
            Profile("/execution/adaptive_stops_enabled"),
        ),
        (
            "NEOETHOS_ADAPTIVE_STOP_RR",
            Profile("/execution/adaptive_stops_rr"),
        ),
        // ── Feature cube (neoethos-data) ──
        // The RESOLVED value of `models.data_runtime.feature_cube_mode`, which
        // replaced the env var. Recorded, not exempted: the RAM and disk
        // assemblies are bit-identical BY TEST, not by construction, so the
        // artifact must say which one built the cube a run searched over.
        (
            "NEOETHOS_FEATURE_CUBE_MODE",
            Profile("/execution/feature_cube_mode"),
        ),
        // ── GPU lane ──
        (
            "NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL",
            Profile("/execution/gpu/cuda_backtest_kernel_enabled"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS",
            Profile("/execution/gpu/cuda_backtest_kernel_units"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE",
            Profile("/execution/gpu/cuda_device_id"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICES",
            Profile("/execution/gpu/multi_cuda_devices_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL",
            Profile("/execution/gpu/cuda_eval_kernel_enabled"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS",
            Profile("/execution/gpu/cuda_eval_kernel_units"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_PRECISION",
            Profile("/execution/gpu/cuda_precision"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE",
            Profile("/execution/gpu/wgpu_device_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICES",
            Profile("/execution/gpu/multi_wgpu_devices_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB",
            Profile("/execution/gpu/gpu_buffer_mb_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB",
            Profile("/execution/gpu/host_budget_mb_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_USE_IGPU",
            Profile("/execution/gpu/use_igpu_env"),
        ),
        (
            "NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB",
            Profile("/execution/gpu/vram_budget_mb_env"),
        ),
        (
            "NEOETHOS_BOT_TRAIN_PRECISION",
            Profile("/execution/gpu/cuda_precision"),
        ),
        (
            "FOREX_TRAIN_PRECISION",
            Profile("/execution/gpu/cuda_precision"),
        ),
        (
            "NEOETHOS_GPU_F64",
            DiagnosticOnly("retired and ignored; CubeCL search arithmetic is unconditionally f64"),
        ),
        (
            "NEOETHOS_GPU_FUSED_EVAL",
            Profile("/execution/gpu/fused_eval_decision"),
        ),
        (
            "NEOETHOS_REQUIRE_GPU",
            Profile("/execution/gpu/require_gpu_env"),
        ),
        (
            "NEOETHOS_RUN_CUDA_SEARCH_TESTS",
            DiagnosticOnly(
                "compiled only into real-device test gates; it cannot alter a production search",
            ),
        ),
        (
            "NEOETHOS_BOT_SEARCH_VRAM_LOG",
            DiagnosticOnly(
                "emits VRAM telemetry lines only; never touches a kernel input, \
                 launch dimension, or fallback decision",
            ),
        ),
        (
            "NEOETHOS_GPU_TIMING",
            DiagnosticOnly(
                "accumulates per-phase Durations into a thread-local and logs them; \
                 the gpu_timing module runs the wrapped work byte-identically when off \
                 and never touches kernel inputs when on",
            ),
        ),
        // ── Discovery config knobs (legacy env names; config-driven now, the
        //    profile records the RESOLVED config value) ──
        (
            "NEOETHOS_BOT_DISCOVERY_MIN_TRADES_PER_DAY",
            Profile("/min_trades_per_day"),
        ),
        ("NEOETHOS_BOT_DISCOVERY_MODE", Profile("/mode")),
        ("NEOETHOS_BOT_DISCOVERY_PERMISSIVE", Profile("/mode")),
        (
            "NEOETHOS_BOT_DISCOVERY_PROP_FIRM_GATE",
            Profile("/prop_firm_gate_params"),
        ),
        (
            "NEOETHOS_BOT_FUNNEL_STAGE1_PCT",
            Profile("/funnel_stage1_pct"),
        ),
        (
            "NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW",
            Profile("/stage1_window"),
        ),
        (
            "NEOETHOS_BOT_MIN_HISTORY_YEARS",
            Profile("/min_history_years"),
        ),
        (
            "NEOETHOS_BOT_PREFILTER_INSAMPLE",
            Profile("/prefilter_insample_frac"),
        ),
        (
            "NEOETHOS_BOT_PREFILTER_MIN_PER_TF",
            Profile("/prefilter_min_per_timeframe"),
        ),
        ("NEOETHOS_BOT_PREFILTER_TOP_K", Profile("/prefilter_top_k")),
        (
            "NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS",
            Profile("/adaptive_thresholds"),
        ),
    ];

    // 1. The profile the classification is checked against: default config +
    //    empty result. Field EXISTENCE is what is asserted (a null value
    //    resolves fine), so the fixture's emptiness does not weaken the test.
    let empty_result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: Vec::new(),
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,
        effective_smc_gate_threshold: f64::NAN,
    };
    let profile = build_discovery_profile(&DiscoveryConfig::default(), &empty_result);
    let json = serde_json::to_value(&profile).expect("run profile must serialize");

    // 2. Every classified-as-recorded knob must actually resolve in the JSON.
    for (name, class) in table {
        if let KnobClass::Profile(pointer) = class {
            assert!(
                json.pointer(pointer).is_some(),
                "census table maps env knob {name} to profile pointer {pointer}, \
                 but the serialized DiscoveryRunProfile has no such field — \
                 the table is lying; fix the pointer or add the field"
            );
        }
    }

    // 3. Every env name in the sources must be classified…
    let discovered = collect_env_knob_names_in_crate_sources();
    let classified: std::collections::BTreeSet<&str> =
        table.iter().map(|(name, _)| *name).collect();
    let unclassified: Vec<&String> = discovered
        .iter()
        .filter(|name| !classified.contains(name.as_str()))
        .collect();
    assert!(
        unclassified.is_empty(),
        "New env knob(s) found in neoethos-search sources that are NOT \
         accounted for in the discovery run profile: {unclassified:?}.\n\
         A knob that can change what the search selects but is absent from \
         the profile makes runs unreproducible-by-inspection (the \
         NEOETHOS_GPU_F64 failure mode). Fix: capture the knob's RESOLVED \
         value in ExecutionEnvironmentProfile (crates/neoethos-search/src/\
         execution_profile.rs) or the DiscoveryRunProfile, then add a \
         (name, Profile(\"/json/pointer\")) row to the census table in this \
         test. Only if the knob provably cannot alter selection (pure \
         logging), add a DiagnosticOnly row with the justification."
    );

    // 4. …and every classified name must still exist in the sources, so the
    //    table cannot accrete stale rows that hide future collisions.
    let stale: Vec<&&str> = classified
        .iter()
        .filter(|name| !discovered.contains(**name))
        .collect();
    assert!(
        stale.is_empty(),
        "census table rows with no matching source occurrence (knob was \
         removed or renamed — delete or update the row): {stale:?}"
    );
}

#[test]
fn identical_configs_produce_identical_profile_json_apart_from_ambient_state() {
    // The serialized profile itself must be deterministic for a fixed config
    // + environment — otherwise diffing two run profiles (the whole point of
    // slice 5) reports phantom differences. This guards, e.g., HashMap
    // iteration order leaking into the JSON (max_rows_by_timeframe is a
    // BTreeMap in the profile for exactly this reason).
    let mut config = DiscoveryConfig::default();
    config.max_rows_by_timeframe.extend([
        ("M1".to_string(), 100),
        ("M5".to_string(), 200),
        ("H1".to_string(), 50),
    ]);
    let result = DiscoveryResult {
        search_input_receipt: sample_search_input_receipt(),
        selection_scope: sample_discovery_selection_scope(),
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: Vec::new(),
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,
        effective_smc_gate_threshold: f64::NAN,
    };
    let mut a = build_discovery_profile(&config, &result);
    let mut b = build_discovery_profile(&config, &result);
    // Engine observations are intentionally process-global ambient telemetry
    // and other parallel tests may add a bit between these two snapshots. The
    // dedicated engine-profile test below verifies that field. Remove only
    // that documented ambient input before checking deterministic encoding.
    a.population_eval_engines.clear();
    b.population_eval_engines.clear();
    let a = serde_json::to_string(&a).expect("profile must serialize");
    let b = serde_json::to_string(&b).expect("profile must serialize");
    assert_eq!(
        a, b,
        "two profile builds from the same config+environment serialized \
         differently — the profile artifact itself is non-deterministic"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// MEASUREMENT SLICE (2026-08-09): the eight named base-quality criteria.
//
// `rejected_base_quality` was ONE counter standing in for at least eight
// independent gates, so a run could report "174 screened, 0 survived" with
// nobody able to say which condition did it — and the answer, in that run, was
// a single one: the payoff floor. These tests pin the attribution: every
// criterion is reachable, and the split does not change which candidates
// survive.
// ───────────────────────────────────────────────────────────────────────────

/// Metrics that clear every base-quality criterion under `strict_only_filters`.
fn base_quality_passing_metrics() -> StrategyMetrics {
    let mut m = crate::quality::empty_metrics("gene_ok");
    m.max_drawdown_pct = 0.10;
    m.win_rate = 0.55;
    m.payoff_ratio = 2.5;
    m.in_market_pct = 0.10;
    m.positive_months = 12;
    m.trades_per_month = 20.0;
    m.avg_monthly_return_pct = 0.10;
    m.avg_win_pct = 0.02; // → avg_trade_return_pct 2.0
    // The net-expectancy criterion is UNCONDITIONAL — `TargetProfile::default()`
    // means "must be strictly positive", not "no preference". A fixture that
    // left this at 0.0 described a candidate that does not make money, which is
    // no longer a passing candidate on any lane. Added when the objective moved
    // from a payoff floor to cost-charged net expectancy per trade.
    m.profit_per_trade = 4.0;
    m.net_expectancy_stderr = 1.0;
    m.net_expectancy_t_stat = 4.0;
    m
}

/// Strict floors the metrics above clear, with the opportunistic lane BOTH
/// switched off AND floored so high it could not rescue anything even if it
/// were open. That isolates the strict criteria for attribution.
fn strict_only_filters() -> FilteringConfig {
    FilteringConfig {
        min_positive_months: 3,
        min_trades_per_month: 5.0,
        min_monthly_return_pct: 0.01,
        opportunistic_enabled: false,
        use_opportunistic_candidates: false,
        opportunistic_min_positive_months: 3,
        opportunistic_min_trades_per_month: 5.0,
        opportunistic_min_trade_return_pct: 5.0,
        opportunistic_max_dd: 0.50,
        ..FilteringConfig::default()
    }
}

#[test]
fn base_quality_attribution_names_each_of_the_ten_criteria() {
    let filters = strict_only_filters();
    let no_profile = TargetProfile::default();

    // Baseline: passes, on the strict lane (not the opportunistic one).
    assert_eq!(
        classify_base_quality(&base_quality_passing_metrics(), &no_profile, &filters),
        Ok(false)
    );

    // 1. Total loss beats everything else, including a profile it satisfies.
    let mut wiped = base_quality_passing_metrics();
    wiped.max_drawdown_pct = 1.0;
    assert_eq!(
        classify_base_quality(&wiped, &no_profile, &filters),
        Err(BaseQualityReject::AccountWiped)
    );

    // 2/3/4/5/6. The five TargetProfile criteria, each on its own.
    //
    // The classifier DELEGATES to `TargetProfile::evaluate` rather than
    // restating its criteria, so these cases also pin that the delegation is
    // complete: a criterion added to the profile and not mapped here is a
    // non-exhaustive-match compile error, not a silent hole in the census.
    let profile = TargetProfile {
        min_net_expectancy_per_trade: 0.0,
        min_expectancy_t_stat: 0.0,
        // 0.50, BELOW the fixture's 0.55. It was 0.60, which the fixture fails,
        // so every case after the win-rate one short-circuited on win rate and
        // the payoff/in-market criteria were never actually exercised. Each
        // criterion here must be reachable with only its own field perturbed.
        min_win_rate: 0.50,
        min_payoff_ratio: 2.0,
        max_in_market: 0.20,
    };

    // THE primary criterion, and the only unconditional one: a money-loser is
    // refused before any shape question is asked, even under a default profile
    // that expresses no other preference. Payoff 2.53 at expectancy -4.18 pips
    // is the measured case this exists to refuse.
    let mut loses_money = base_quality_passing_metrics();
    loses_money.profit_per_trade = -4.18;
    loses_money.payoff_ratio = 2.53;
    assert_eq!(
        classify_base_quality(&loses_money, &no_profile, &filters),
        Err(BaseQualityReject::ProfileNetExpectancy)
    );

    // Positive, but inside its own sampling noise. Opt-in, and it binds when set.
    let mut noisy = base_quality_passing_metrics();
    noisy.net_expectancy_t_stat = 0.4;
    let significance_profile = TargetProfile {
        min_expectancy_t_stat: 2.0,
        ..TargetProfile::default()
    };
    assert_eq!(
        classify_base_quality(&noisy, &significance_profile, &filters),
        Err(BaseQualityReject::ProfileExpectancySignificance)
    );

    let mut low_wr = base_quality_passing_metrics();
    low_wr.win_rate = 0.40;
    assert_eq!(
        classify_base_quality(&low_wr, &profile, &filters),
        Err(BaseQualityReject::ProfileWinRate)
    );

    // The gate that decided the 0-of-174 run all by itself: a realised payoff
    // of 1.08 (the best cell measured anywhere in the real-bar grid) against a
    // configured floor of 2.0.
    let mut low_payoff = base_quality_passing_metrics();
    low_payoff.payoff_ratio = 1.08;
    assert_eq!(
        classify_base_quality(&low_payoff, &profile, &filters),
        Err(BaseQualityReject::ProfilePayoffRatio)
    );

    let mut always_in = base_quality_passing_metrics();
    always_in.in_market_pct = 0.90;
    assert_eq!(
        classify_base_quality(&always_in, &profile, &filters),
        Err(BaseQualityReject::ProfileInMarket)
    );

    // 8/9/10. The strict-quality criteria, with the opportunistic lane unable to
    // rescue any of them.
    let mut few_months = base_quality_passing_metrics();
    few_months.positive_months = 1;
    assert_eq!(
        classify_base_quality(&few_months, &no_profile, &filters),
        Err(BaseQualityReject::PositiveMonths)
    );

    let mut thin = base_quality_passing_metrics();
    thin.trades_per_month = 1.0;
    assert_eq!(
        classify_base_quality(&thin, &no_profile, &filters),
        Err(BaseQualityReject::TradesPerMonth)
    );

    let mut flat = base_quality_passing_metrics();
    flat.avg_monthly_return_pct = 0.0001;
    assert_eq!(
        classify_base_quality(&flat, &no_profile, &filters),
        Err(BaseQualityReject::MonthlyReturn)
    );
}

#[test]
fn a_candidate_killed_by_the_opportunistic_switch_is_not_reported_as_a_metric_failure() {
    // "N candidates were killed by a config switch" and "N candidates missed a
    // measurement" call for opposite decisions. The old single counter said
    // neither.
    let mut filters = strict_only_filters();
    filters.min_monthly_return_pct = 0.50; // strict lane unreachable
    filters.opportunistic_min_positive_months = 1;
    filters.opportunistic_min_trades_per_month = 1.0;
    filters.opportunistic_min_trade_return_pct = 0.0;
    filters.opportunistic_max_dd = 0.90;

    let metrics = base_quality_passing_metrics();
    let no_profile = TargetProfile::default();

    // Lane closed → attributed to the switch, not to a metric.
    assert_eq!(
        classify_base_quality(&metrics, &no_profile, &filters),
        Err(BaseQualityReject::OpportunisticLaneClosed)
    );

    // Lane open → the same candidate survives, on the opportunistic lane.
    filters.opportunistic_enabled = true;
    filters.use_opportunistic_candidates = true;
    assert_eq!(
        classify_base_quality(&metrics, &no_profile, &filters),
        Ok(true)
    );
}

#[test]
fn attribution_reproduces_the_original_pass_fail_decision_exactly() {
    // The split is INSTRUMENTATION: it must not change which candidates
    // survive. Sweep the grid and assert the classifier agrees with the
    // original `profile_ok && (strict || opportunistic)` expression on every
    // cell, and that the lane it reports matches too.
    let profile = TargetProfile {
        min_net_expectancy_per_trade: 0.0,
        min_expectancy_t_stat: 0.0,
        min_win_rate: 0.35,
        min_payoff_ratio: 1.5,
        max_in_market: 0.50,
    };
    let mut checked = 0usize;
    for lane_open in [false, true] {
        for dd in [0.05_f64, 1.0] {
            for wr in [0.20_f64, 0.60] {
                for payoff in [0.9_f64, 2.4] {
                    for in_market in [0.10_f64, 0.80] {
                        for months in [0_usize, 12] {
                            for tpm in [0.5_f64, 20.0] {
                                for ret in [0.0001_f64, 0.10] {
                                    let mut m = base_quality_passing_metrics();
                                    m.max_drawdown_pct = dd;
                                    m.win_rate = wr;
                                    m.payoff_ratio = payoff;
                                    m.in_market_pct = in_market;
                                    m.positive_months = months;
                                    m.trades_per_month = tpm;
                                    m.avg_monthly_return_pct = ret;

                                    let mut f = strict_only_filters();
                                    f.opportunistic_enabled = lane_open;
                                    f.use_opportunistic_candidates = lane_open;
                                    f.opportunistic_min_positive_months = 1;
                                    f.opportunistic_min_trades_per_month = 1.0;
                                    f.opportunistic_min_trade_return_pct = 0.0;
                                    f.opportunistic_max_dd = 0.90;

                                    let profile_ok = profile.accepts(&m);
                                    let strict = profile_ok && passes_strict_quality(&m, &f);
                                    let opportunistic = profile_ok
                                        && !strict
                                        && passes_opportunistic_quality(&m, &f);
                                    let legacy_survives = strict || opportunistic;

                                    let verdict = classify_base_quality(&m, &profile, &f);
                                    assert_eq!(
                                        verdict.is_ok(),
                                        legacy_survives,
                                        "attribution changed the survival decision at \
                                         dd={dd} wr={wr} payoff={payoff} in_market={in_market} \
                                         months={months} tpm={tpm} ret={ret} open={lane_open}"
                                    );
                                    if let Ok(lane) = verdict {
                                        assert_eq!(lane, opportunistic);
                                    }
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 256);
}

#[test]
fn the_funnel_keeps_every_reject_reason_the_quality_screen_records() {
    // The quality screen records six gate-level reasons plus the eight named
    // base-quality criteria — fourteen. A 10-entry cap would have silently
    // dropped four, and the dropped ones would be the smallest counts, i.e.
    // exactly the rare causes a funnel exists to surface.
    let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M5");
    funnel.record_stage("passed_quality", 100, 0);
    for i in 0..14 {
        funnel.add_reject_reason("passed_quality", format!("reason_{i}"), i + 1);
    }
    let stage = funnel
        .stages
        .iter()
        .find(|s| s.name == "passed_quality")
        .expect("passed_quality stage");
    assert_eq!(
        stage.top_reasons.len(),
        14,
        "the funnel truncated reject reasons — a silent drop on a diagnostics path"
    );
    assert!(crate::funnel_profile::MAX_REJECT_REASONS_PER_STAGE >= 14);
}

/// A cost band that sits BELOW the cost the run already charges cannot fail
/// anybody: the edges replace the whole cost, cost is monotone, so every
/// screened survivor clears both edges by construction and the census reads
/// clean on every run. That is worse than no census, because a reader takes it
/// as evidence.
///
/// The shipped numbers are exactly this case, which is why the arithmetic is
/// pinned here rather than left as prose: spread 1.5 + slippage 0.5 +
/// commission 14 USD/lot ÷ 10 USD/pip = 3.4 pips, against edges 1.6 / 2.4.
#[test]
fn a_cost_band_below_the_charged_cost_cannot_discriminate() {
    let baseline = crate::run_identity::cost_pips_round_trip(2.0, 14.0, 10.0);
    assert!(
        (baseline - 3.4).abs() < 1e-12,
        "shipped baseline is 3.4 pips"
    );

    assert!(
        !crate::discovery::cost_band_discriminates(Some((1.6, 2.4)), baseline),
        "the shipped band is entirely below the shipped baseline and must be refused"
    );
    // Bracketing the baseline is what makes the band able to say anything.
    assert!(crate::discovery::cost_band_discriminates(
        Some((2.8, 4.4)),
        baseline
    ));
    // Exactly equal is still not discriminating: a candidate that cleared the
    // baseline clears an identical cost trivially.
    assert!(!crate::discovery::cost_band_discriminates(
        Some((1.6, 3.4)),
        baseline
    ));
    // No band configured is not "discriminating"; it is unmeasured.
    assert!(!crate::discovery::cost_band_discriminates(None, baseline));
    // A non-finite baseline must never be read as a pass.
    assert!(!crate::discovery::cost_band_discriminates(
        Some((2.8, 4.4)),
        f64::NAN
    ));
}

/// Audit #75/#217 — the weekend kill zones are ONE knob, and the search reads it.
///
/// Until 2026-08-10 `discovery_backtest_settings` hardcoded
/// `kill_zones_enabled: true` while only the live loop consulted
/// `risk.kill_zones_enabled`. That made the knob one-sided: turning it off could
/// move live AWAY from every validated backtest and never toward it. This
/// asserts BOTH ends — the config field is sourced from `Settings`, and the
/// settings template every discovery lane flows through carries it — so the
/// literal cannot come back without a red test.
#[test]
fn the_kill_zone_switch_reaches_the_discovery_backtest() {
    let mut settings = neoethos_core::Settings::default();
    // The shipped value, and the value the live loop defaults to.
    assert!(
        settings.risk.kill_zones_enabled,
        "the shipped default must stay ON — this test's whole point is that both \
         sides move together, not that they are both true"
    );

    let gene = profitable_gene("kill-zone-wiring");

    // Through the RESOLVER, never the raw builder: the SLICE-2 guard
    // (`discovery_backtest_settings_has_no_callers_outside_the_resolvers`)
    // permits exactly three occurrences of the private builder and none in
    // this file. `PopulationTemplateResolver::template` is the gene-independent
    // template every population lane takes, and it is the one that carries the
    // weekend policy, so asserting on it asserts on what actually runs.
    let template = |config: &crate::discovery::DiscoveryConfig| {
        crate::discovery::PopulationTemplateResolver::new(config, None).template(&gene)
    };

    let on = crate::discovery::DiscoveryConfig::from_settings(&settings);
    assert!(on.kill_zones_enabled, "from_settings must carry the field");
    assert!(
        template(&on).kill_zones_enabled,
        "the ON setting must reach the settings template"
    );

    settings.risk.kill_zones_enabled = false;
    let off = crate::discovery::DiscoveryConfig::from_settings(&settings);
    assert!(
        !off.kill_zones_enabled,
        "flipping risk.kill_zones_enabled must flip the discovery config"
    );
    assert!(
        !template(&off).kill_zones_enabled,
        "the OFF setting must reach the settings template — a hardcoded `true` here is \
         exactly the defect this test exists for"
    );

    // And the two runs must be TELLABLE APART afterwards through the one
    // canonical search-config authority every validation envelope carries.
    let hash_on = crate::run_identity::config_hash_for(
        &on,
        on.evaluation_config(None).pip_value_per_lot,
        false,
    )
    .expect("search config hash (on)");
    let hash_off = crate::run_identity::config_hash_for(
        &off,
        off.evaluation_config(None).pip_value_per_lot,
        false,
    )
    .expect("search config hash (off)");
    assert_ne!(
        hash_on, hash_off,
        "two runs under opposite weekend policies must not hash identically"
    );
}
