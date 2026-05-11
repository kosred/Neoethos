use super::*;


use crate::FilteringConfig;
use ndarray::Array2;

fn sample_feature_frame() -> FeatureFrame {
    let start = 1_704_067_200_000_i64;
    FeatureFrame {
        timestamps: (0..10).map(|idx| start + idx * 60_000).collect(),
        names: vec!["signal".to_string()],
        data: Array2::from_shape_vec(
            (10, 1),
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        )
        .expect("feature frame shape should be valid"),
    }
}

fn sample_ohlcv() -> Ohlcv {
    let start = 1_704_067_200_000_i64;
    let close: Vec<f64> = vec![
        1.1000, 1.1010, 1.1020, 1.1015, 1.1030, 1.1025, 1.1040, 1.1035, 1.1050, 1.1045,
    ];
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if idx == 0 {
                *value - 0.0005
            } else {
                close[idx - 1]
            }
        })
        .collect();
    let high: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| open.max(*close) + 0.0004)
        .collect();
    let low: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| open.min(*close) - 0.0004)
        .collect();
    let volume: Vec<f64> = (0..10).map(|idx| 1000.0 + (idx as f64 * 25.0)).collect();

    Ohlcv {
        timestamp: Some((0..10).map(|idx| start + idx * 60_000).collect()),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
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
    };

    let err = ensure_non_empty_portfolio(&result, "EURUSD M1")
        .expect_err("expected empty discovery portfolio to fail");
    let msg = err.to_string();
    assert!(msg.contains("empty portfolio"), "unexpected error: {msg}");
    assert!(msg.contains("candidates=1"), "unexpected error: {msg}");
}

#[test]
fn non_empty_portfolio_is_accepted() {
    let result = DiscoveryResult {
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
    };

    ensure_non_empty_portfolio(&result, "EURUSD M1")
        .expect("expected non-empty portfolio to pass");
}

#[test]
fn candidate_truncation_honors_small_explicit_limits() {
    assert_eq!(candidate_truncation_limit(2, 500), 2);
    assert_eq!(candidate_truncation_limit(0, 500), 500);
    assert_eq!(candidate_truncation_limit(500, 2), 2);
    assert_eq!(candidate_truncation_limit(5, 0), 0);
}

#[test]
fn finalize_candidates_with_progress_emits_filter_and_portfolio_milestones() {
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
    let candidates = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
    let mut progress_events = Vec::new();

    let result = finalize_candidates_with_progress(
        candidates,
        &features,
        &ohlcv,
        &config,
        features.names.clone(),
        |event| progress_events.push(event),
    )
    .expect("candidate finalization should succeed");

    assert_eq!(result.candidates.len(), 2);
    assert_eq!(result.portfolio.len(), 1);
    assert_eq!(
        result.canonical_backtest_artifacts.len(),
        result.portfolio.len()
    );
    assert_eq!(
        result.walkforward_validation_artifacts.len(),
        result.portfolio.len()
    );
    assert_eq!(
        result.validation_gates.canonical_backtest_artifacts,
        result.portfolio.len()
    );
    assert!(result.validation_gates.temporal_contract_hash.is_some());
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
    assert!(progress_events.iter().any(|event| matches!(
        event,
        DiscoveryProgress::Completed { candidate_count, filtered_count, portfolio_size }
            if *candidate_count == 2 && *filtered_count == 2 && *portfolio_size == 1
    )));
}

#[test]
fn portfolio_export_requires_validation_gates() {
    let result = DiscoveryResult {
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
    };
    let path = temp_path("portfolio-gates");

    let err = save_portfolio_json(&path, &result)
        .expect_err("portfolio export must fail before validation gates pass");
    assert!(err.to_string().contains("walkforward_passed"));
    assert!(!path.exists());
}

#[test]
fn portfolio_export_succeeds_when_prop_firm_window_passed_even_without_walkforward() {
    let mut result = DiscoveryResult {
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
    };
    // The prop-firm window-pass gate is the canonical export path
    // when active; walkforward and CPCV are intentionally unset.
    result.validation_gates.prop_firm_window_passed = true;
    result.validation_gates.prop_firm_window_count = 50;
    result.validation_gates.prop_firm_window_pass_rate = 0.72;
    let path = temp_path("portfolio-prop-firm-export");

    save_portfolio_json(&path, &result)
        .expect("portfolio export should pass when prop-firm gate is the active path");
    assert!(path.exists());
    let _ = std::fs::remove_file(path);
}

#[test]
fn prop_firm_gate_env_overrides_populate_discovery_config() {
    // SAFETY: tests that read process-wide env may race. This group
    // sets and unsets its own keys; cargo test runs lib tests in a
    // single process so we keep the surface narrow and balanced.
    // We also clear the mode flag at the start to avoid bleed from
    // sibling tests.
    unsafe {
        std::env::remove_var("FOREX_BOT_DISCOVERY_MODE");
        std::env::set_var("FOREX_BOT_DISCOVERY_PROP_FIRM_PASS_RATE", "0.42");
        std::env::set_var("FOREX_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS", "17");
        std::env::set_var("FOREX_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS", "21");
        std::env::set_var(
            "FOREX_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT",
            "0.08",
        );
    }
    let cfg = DiscoveryConfig::default().with_env_runtime_overrides();
    unsafe {
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_PASS_RATE");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT");
    }
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
fn prop_firm_gate_auto_enables_with_no_env_at_all() {
    // The whole point: zero env vars should still produce a smart,
    // ready-to-run prop-firm config.
    unsafe {
        std::env::remove_var("FOREX_BOT_DISCOVERY_MODE");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PERMISSIVE");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_PASS_RATE");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_MAX_DAILY_LOSS_PCT");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_MAX_DD_PCT");
        std::env::remove_var("FOREX_BOT_DISCOVERY_PROP_FIRM_MIN_TRADING_DAYS");
    }
    let cfg = DiscoveryConfig::default().with_env_runtime_overrides();
    let pf = cfg.prop_firm_gate.expect("default = PropFirm mode");
    // FTMO baseline: 5%/10%/10%/5 days, 60-day window
    assert_eq!(pf.window_days, 60);
    assert_eq!(pf.n_windows, 0); // sentinel — auto-tuned at runtime
    assert!((pf.pass_rate - 0.0).abs() < 1e-9); // ranking-only by default
    assert!((pf.rules.max_daily_loss_pct - 0.05).abs() < 1e-9);
    assert!((pf.rules.max_overall_drawdown_pct - 0.10).abs() < 1e-9);
    assert!((pf.rules.min_profit_target_pct - 0.10).abs() < 1e-9);
    assert!(pf.rules.require_profit_target);
    // Permissive filter floors should be applied automatically.
    assert!(!cfg.filtering.anomaly_guard);
    assert!(cfg.filtering.min_sharpe < 0.0);
}

#[test]
fn prop_firm_gate_disabled_in_strict_mode() {
    unsafe {
        std::env::set_var("FOREX_BOT_DISCOVERY_MODE", "strict");
    }
    let cfg = DiscoveryConfig::default().with_env_runtime_overrides();
    unsafe {
        std::env::remove_var("FOREX_BOT_DISCOVERY_MODE");
    }
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
fn portfolio_export_uses_effective_names_after_validation_gates_pass() {
    let mut result = DiscoveryResult {
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
    };
    result.validation_gates.walkforward_passed = true;
    result.validation_gates.cpcv_passed = true;
    let path = temp_path("portfolio-export");

    save_portfolio_json(&path, &result)
        .expect("portfolio export should pass once validation gates are true");
    let exported = std::fs::read_to_string(&path).expect("portfolio export should exist");
    assert!(exported.contains("filtered_signal"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn discovery_profile_exports_validation_gate_status() {
    let mut result = DiscoveryResult {
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

fn sample_temporal_contract() -> TemporalFeatureContract {
    discovery_temporal_contract(&DiscoveryConfig::default(), &["signal".to_string()])
        .expect("temporal contract for default discovery config")
}

fn sample_canonical_backtest_artifact(strategy_hash: &str) -> CanonicalBacktestArtifactFile {
    let contract = sample_temporal_contract();
    let scope = CanonicalBacktestScope::new("dataset", "evaluation", strategy_hash, &contract);
    CanonicalBacktestArtifactFile::new(scope, BacktestMetrics::from_metric_array([0.0; 11]))
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

fn sample_walkforward_validation_artifact(
    strategy_hash: &str,
) -> WalkforwardValidationArtifactFile {
    let contract = sample_temporal_contract();
    let scope = WalkforwardValidationScope::for_strategy(
        "dataset",
        "evaluation",
        strategy_hash,
        &contract,
    );
    WalkforwardValidationArtifactFile::new(scope, sample_walkforward_summary())
}

#[test]
fn save_canonical_backtest_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("canonical-backtests");
    let result = DiscoveryResult {
        portfolio: vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: vec![
            sample_canonical_backtest_artifact("fnv64:0123456789abcdef"),
            sample_canonical_backtest_artifact("fnv64:fedcba9876543210"),
        ],
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
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
    let result = DiscoveryResult {
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: vec![sample_walkforward_validation_artifact(
            "fnv64:0011223344556677",
        )],
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
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
    assert_eq!(defaults.prefilter_top_k, 50);
    assert!((defaults.prefilter_insample_frac - 0.70).abs() < 1e-9);
    assert!((defaults.funnel_stage1_pct - 0.25).abs() < 1e-9);
}

#[test]
fn discovery_runtime_overrides_clamp_invalid_values() {
    let overrides = DiscoveryRuntimeOverrides {
        prefilter_top_k: 0,
        prefilter_insample_frac: f64::NAN,
        funnel_stage1_pct: 5.0,
        stage1_window: Stage1Window::Earliest,
    };
    assert!((overrides.resolved_prefilter_insample_frac() - 0.70).abs() < 1e-9);
    assert!((overrides.resolved_funnel_stage1_pct() - 1.0).abs() < 1e-9);

    let too_small = DiscoveryRuntimeOverrides {
        prefilter_top_k: 0,
        prefilter_insample_frac: 0.0,
        funnel_stage1_pct: 0.0001,
        stage1_window: Stage1Window::Earliest,
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
        funnel_stage1_pct: 0.5,
        stage1_window: Stage1Window::Earliest,
    };
    let result = DiscoveryResult {
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
    };

    let profile = build_discovery_profile(&config, &result);
    assert_eq!(profile.prefilter_top_k, 17);
    assert!((profile.prefilter_insample_frac - 0.6).abs() < 1e-9);
    assert!((profile.funnel_stage1_pct - 0.5).abs() < 1e-9);
}

#[test]
fn compute_discovery_forward_test_artifacts_returns_empty_for_empty_portfolio() {
    let config = DiscoveryConfig::default();
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let artifacts = compute_discovery_forward_test_artifacts(
        &[],
        &features.names,
        &features,
        &ohlcv,
        &config,
    )
    .expect("empty portfolio should produce zero artifacts");
    assert!(artifacts.is_empty());
}

#[test]
fn compute_discovery_forward_test_artifacts_rejects_tails_missing_features() {
    let config = DiscoveryConfig::default();
    let portfolio = vec![profitable_gene("alpha-1")];
    let mut tail_features = sample_feature_frame();
    tail_features.names = vec!["unrelated_feature".to_string()];
    let err = compute_discovery_forward_test_artifacts(
        &portfolio,
        &["signal".to_string()],
        &tail_features,
        &sample_ohlcv(),
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
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let artifacts = compute_discovery_forward_test_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &config,
    )
    .expect("forward-test artifacts should build for in-band tail");
    assert_eq!(artifacts.len(), portfolio.len());
    for artifact in &artifacts {
        assert_eq!(
            artifact.artifact_kind,
            crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND
        );
        assert!(artifact.summary.bars > 0);
        assert!(!artifact.scope.strategy_hash.is_empty());
    }
}

#[test]
fn save_forward_test_validation_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("forward-test-validations");
    let config = DiscoveryConfig::default();
    let portfolio = vec![profitable_gene("alpha-1")];
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let artifacts = compute_discovery_forward_test_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &config,
    )
    .expect("forward-test artifacts should build");

    let result = DiscoveryResult {
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
    let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
        .expect("temporal contract for default discovery config");
    let scope = ForwardTestValidationScope::new("dataset", "eval", "strategy", &temporal);
    let summary = crate::validation::ForwardTestSummary {
        bars: 5,
        metrics: BacktestMetrics::from_metric_array([0.0; 11]),
        span_days: 0.0,
    };
    let mut result = DiscoveryResult {
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: vec![ForwardTestValidationArtifactFile::new(
            scope, summary,
        )],
        prop_firm_validation_artifacts: Vec::new(),
    };
    result.validation_gates.walkforward_passed = true;
    result.validation_gates.cpcv_passed = true;

    let profile = build_discovery_profile(&config, &result);
    assert_eq!(profile.forward_test_validation_artifacts_observed, 1);
}

fn forward_test_artifact_with_metrics(
    strategy_hash: &str,
    net_profit: f64,
    trade_count: usize,
) -> ForwardTestValidationArtifactFile {
    let config = DiscoveryConfig::default();
    let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
        .expect("temporal contract for default discovery config");
    let scope = ForwardTestValidationScope::new("dataset", "eval", strategy_hash, &temporal);
    let mut metrics_array = [0.0_f64; 11];
    metrics_array[0] = net_profit; // net_profit
    metrics_array[8] = trade_count as f64; // trade_count
    let summary = crate::validation::ForwardTestSummary {
        bars: 5,
        metrics: BacktestMetrics::from_metric_array(metrics_array),
        span_days: 0.0,
    };
    ForwardTestValidationArtifactFile::new(scope, summary)
}

fn empty_discovery_result_with_gates(
    walkforward_passed: bool,
    cpcv_passed: bool,
) -> DiscoveryResult {
    let mut gates = DiscoveryValidationGates::pending();
    gates.walkforward_passed = walkforward_passed;
    gates.cpcv_passed = cpcv_passed;
    DiscoveryResult {
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
    }
}

#[test]
fn evidence_bridge_mirrors_discovery_validation_gates_with_no_forward_test_artifacts() {
    let result = empty_discovery_result_with_gates(true, true);
    let evidence = live_validation_evidence_from_discovery(&result);
    assert!(evidence.walkforward_passed);
    assert!(evidence.cpcv_passed);
    assert_eq!(evidence.forward_test_passed, None);
    assert_eq!(evidence.prop_firm_passed, None);
    assert!(evidence.live_sim_runtime_model_hash.is_none());
}

#[test]
fn evidence_bridge_marks_forward_test_passed_when_every_artifact_is_profitable() {
    let mut result = empty_discovery_result_with_gates(true, true);
    result.forward_test_validation_artifacts = vec![
        forward_test_artifact_with_metrics("fnv64:abc", 25.0, 3),
        forward_test_artifact_with_metrics("fnv64:def", 10.0, 1),
    ];
    let evidence = live_validation_evidence_from_discovery(&result);
    assert_eq!(evidence.forward_test_passed, Some(true));
}

#[test]
fn evidence_bridge_marks_forward_test_failed_when_any_artifact_is_unprofitable() {
    let mut result = empty_discovery_result_with_gates(true, true);
    result.forward_test_validation_artifacts = vec![
        forward_test_artifact_with_metrics("fnv64:abc", 25.0, 3),
        forward_test_artifact_with_metrics("fnv64:def", -10.0, 2),
    ];
    let evidence = live_validation_evidence_from_discovery(&result);
    assert_eq!(evidence.forward_test_passed, Some(false));
}

#[test]
fn evidence_bridge_marks_forward_test_failed_when_artifact_has_zero_trades() {
    let mut result = empty_discovery_result_with_gates(true, true);
    result.forward_test_validation_artifacts =
        vec![forward_test_artifact_with_metrics("fnv64:abc", 5.0, 0)];
    let evidence = live_validation_evidence_from_discovery(&result);
    assert_eq!(evidence.forward_test_passed, Some(false));
}

#[test]
fn evidence_bridge_propagates_failed_walkforward_and_cpcv() {
    let result = empty_discovery_result_with_gates(false, false);
    let evidence = live_validation_evidence_from_discovery(&result);
    assert!(!evidence.walkforward_passed);
    assert!(!evidence.cpcv_passed);
}

fn prop_firm_artifact_with_pass_flag(
    strategy_hash: &str,
    all_rules_passed: bool,
) -> PropFirmRiskValidationArtifactFile {
    let config = DiscoveryConfig::default();
    let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
        .expect("temporal contract for default discovery config");
    let rules = PropFirmRiskRules::default();
    let scope =
        PropFirmRiskValidationScope::new("dataset", "eval", strategy_hash, &rules, &temporal)
            .expect("scope construction should succeed");
    let summary = crate::validation::PropFirmRiskValidationSummary {
        rules,
        trades_observed: 0,
        trading_days_observed: 0,
        max_daily_loss_pct_observed: 0.0,
        max_overall_drawdown_pct_observed: 0.0,
        largest_profit_share_observed: 0.0,
        max_trades_per_day_observed: 0,
        net_return_pct: 0.0,
        daily_loss_breach: false,
        overall_drawdown_breach: false,
        consistency_violation: false,
        trade_limit_violation: false,
        min_trading_days_ok: true,
        profit_target_met: true,
        all_rules_passed,
    };
    PropFirmRiskValidationArtifactFile::new(scope, summary)
}

#[test]
fn evidence_bridge_marks_prop_firm_passed_when_every_artifact_passes() {
    let mut result = empty_discovery_result_with_gates(true, true);
    result.prop_firm_validation_artifacts = vec![
        prop_firm_artifact_with_pass_flag("fnv64:abc", true),
        prop_firm_artifact_with_pass_flag("fnv64:def", true),
    ];
    let evidence = live_validation_evidence_from_discovery(&result);
    assert_eq!(evidence.prop_firm_passed, Some(true));
}

#[test]
fn evidence_bridge_marks_prop_firm_failed_when_any_artifact_fails() {
    let mut result = empty_discovery_result_with_gates(true, true);
    result.prop_firm_validation_artifacts = vec![
        prop_firm_artifact_with_pass_flag("fnv64:abc", true),
        prop_firm_artifact_with_pass_flag("fnv64:def", false),
    ];
    let evidence = live_validation_evidence_from_discovery(&result);
    assert_eq!(evidence.prop_firm_passed, Some(false));
}

#[test]
fn compute_discovery_prop_firm_artifacts_returns_empty_for_empty_portfolio() {
    let config = DiscoveryConfig::default();
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let artifacts = compute_discovery_prop_firm_artifacts(
        &[],
        &features.names,
        &features,
        &ohlcv,
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
    let mut tail_features = sample_feature_frame();
    tail_features.names = vec!["unrelated_feature".to_string()];
    let err = compute_discovery_prop_firm_artifacts(
        &portfolio,
        &["signal".to_string()],
        &tail_features,
        &sample_ohlcv(),
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
    let features = sample_feature_frame();
    let ohlcv = sample_ohlcv();
    let artifacts = compute_discovery_prop_firm_artifacts(
        &portfolio,
        &features.names,
        &features,
        &ohlcv,
        &config,
        PropFirmRiskRules::default(),
    )
    .expect("prop-firm artifacts should build");
    assert_eq!(artifacts.len(), portfolio.len());
    for artifact in &artifacts {
        assert_eq!(
            artifact.artifact_kind,
            crate::validation::PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND
        );
        assert!(!artifact.scope.strategy_hash.is_empty());
    }
}

#[test]
fn save_prop_firm_validation_artifacts_writes_one_file_per_strategy() {
    let dir = temp_dir("prop-firm-validations");
    let result = DiscoveryResult {
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: vec![prop_firm_artifact_with_pass_flag(
            "fnv64:abc",
            true,
        )],
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
    DiscoveryResult {
        portfolio: vec![profitable_gene("alpha-1")],
        candidates: Vec::new(),
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: vec!["signal".to_string()],
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: (0..canonical_count)
            .map(|idx| sample_canonical_backtest_artifact(&format!("canonical-{idx}")))
            .collect(),
        walkforward_validation_artifacts: (0..walkforward_count)
            .map(|idx| sample_walkforward_validation_artifact(&format!("walkforward-{idx}")))
            .collect(),
        forward_test_validation_artifacts: (0..forward_test_count)
            .map(|idx| forward_test_artifact_with_metrics(&format!("forward-{idx}"), 1.0, 1))
            .collect(),
        prop_firm_validation_artifacts: (0..prop_firm_count)
            .map(|idx| prop_firm_artifact_with_pass_flag(&format!("prop-{idx}"), true))
            .collect(),
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
    let result = populated_discovery_result(2, 1, 1, 2);

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
    let evidence = live_validation_evidence_from_discovery(&result_for_evidence);
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
    assert_eq!(profile.forward_test_validation_artifacts_observed, 1);
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
