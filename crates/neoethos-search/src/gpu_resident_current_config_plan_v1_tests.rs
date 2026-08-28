use std::path::PathBuf;

use neoethos_core::Settings;

use super::*;
use crate::discovery::{PropFirmGateOverrides, resolve_prefilter_financial_geometry_v1};
use crate::validation::PropFirmRiskRules;

const REQUIRED_RATE_V1: u64 = 223_106_667;

#[derive(Clone, Copy)]
struct AdmissionFixtureV1 {
    selected_device_ordinal: u32,
    identities: [[u8; 32]; 8],
    measured_rate: u64,
    phase_one_free_bytes: u64,
    allocator_context_reserve_bytes: u64,
    required_workspace_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    full_discovery_reserve_bytes: u64,
}

impl Default for AdmissionFixtureV1 {
    fn default() -> Self {
        Self {
            selected_device_ordinal: 0,
            identities: [
                [0x11; 32], [0x12; 32], [0x13; 32], [0x14; 32], [0x15; 32], [0x22; 32], [0x33; 32],
                [0x44; 32],
            ],
            measured_rate: 300_000_000,
            phase_one_free_bytes: 8_000_000_000,
            allocator_context_reserve_bytes: 1_000_000_000,
            required_workspace_bytes: 2_500_000_000,
            trim_prefilter_reserved_bytes: 300_000_000,
            full_discovery_reserve_bytes: 3_000_000_000,
        }
    }
}

impl AdmissionFixtureV1 {
    fn seal(self) -> CurrentConfigResidentSearchAdmissionFactsV1 {
        CurrentConfigResidentSearchAdmissionFactsV1::test_fixture_v1(
            self.selected_device_ordinal,
            self.identities[0],
            self.identities[1],
            self.identities[2],
            self.identities[3],
            self.identities[4],
            self.identities[5],
            self.identities[6],
            self.identities[7],
            self.measured_rate,
            self.phase_one_free_bytes,
            self.allocator_context_reserve_bytes,
            self.required_workspace_bytes,
            self.trim_prefilter_reserved_bytes,
            self.full_discovery_reserve_bytes,
        )
    }
}

fn repo_config() -> Settings {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("config.yaml");
    Settings::from_yaml(&path).expect("the shipped headless config must pass the production loader")
}

fn config_and_runtime() -> (DiscoveryConfig, GeneticSearchRuntimeOverrides) {
    let settings = repo_config();
    (
        DiscoveryConfig::from_settings(&settings),
        GeneticSearchRuntimeOverrides::from_settings(&settings),
    )
}

fn seal(
    config: &DiscoveryConfig,
    runtime: &GeneticSearchRuntimeOverrides,
    admission: AdmissionFixtureV1,
) -> Result<SealedCurrentConfigResidentSearchPlanV1, CurrentConfigResidentSearchPlanErrorV1> {
    seal_current_config_resident_search_plan_v1(config, runtime, 1_000_000, 500, admission.seal())
}

fn changed_f64(value: f64) -> f64 {
    f64::from_bits(value.to_bits() ^ 1)
}

fn config_digest(config: &DiscoveryConfig) -> [u8; 32] {
    canonical_discovery_config_digest_v1(config).expect("fixture config must encode")
}

fn assert_config_digest_changes(
    baseline: &DiscoveryConfig,
    field: &str,
    mutate: impl FnOnce(&mut DiscoveryConfig),
) {
    let baseline_digest = config_digest(baseline);
    let mut changed = baseline.clone();
    mutate(&mut changed);
    assert_ne!(
        baseline_digest,
        config_digest(&changed),
        "canonical digest ignored DiscoveryConfig::{field}"
    );
}

macro_rules! assert_config_mutations {
    ($baseline:expr; $($field:literal => $mutate:expr),+ $(,)?) => {
        $(assert_config_digest_changes($baseline, $field, $mutate);)+
    };
}

#[test]
fn shipped_headless_config_seals_exact_current_config_requirements() {
    let (config, runtime) = config_and_runtime();
    let plan = seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap();

    assert_eq!(plan.population(), 200);
    assert_eq!(plan.maximum_generations(), 20_000);
    assert_eq!(plan.maximum_runtime_millis(), 3_600_000);
    assert_eq!(plan.maximum_terms_per_gene(), 16);
    assert_eq!(plan.parent_row_range(), 0..1_000_000);
    assert_eq!(plan.prefilter_fit_row_range(), 0..800_000);
    assert_eq!(plan.outer_holdout_row_range(), 800_000..1_000_000);
    assert_eq!(plan.prefilter_top_k(), 240);
    assert_eq!(plan.prefilter_min_per_timeframe(), 6);
    assert_eq!(plan.immutable_base_scenario_count(), 200);
    assert_eq!(plan.novelty_weight().to_bits(), 0.2_f64.to_bits());
    assert_eq!(plan.novelty_neighbors(), 15);
    assert_eq!(plan.permanent_archive_capacity(), 50_000);
    assert_eq!(plan.archive_min_net().to_bits(), 0.0_f64.to_bits());
    assert_eq!(plan.maximum_archive_knn_distance_count(), 200_796_000_000);
    assert_eq!(plan.gene_signature_word_count(), 4);
    assert_eq!(
        plan.maximum_archive_knn_popcount_word_count(),
        803_184_000_000
    );
    assert_eq!(
        plan.required_archive_knn_popcount_words_per_second(),
        REQUIRED_RATE_V1
    );
    assert!(plan.archive_knn_budget_admitted());
    assert_eq!(plan.trim_prefilter_reserved_bytes(), 300_000_000);
    assert_eq!(plan.required_workspace_bytes(), 2_500_000_000);
    assert_eq!(plan.full_discovery_reserve_bytes(), 3_000_000_000);
    assert_ne!(plan.plan_identity_sha256(), [0; 32]);
}

#[test]
fn novelty_and_calibration_are_explicit_fail_closed_identity_inputs() {
    let (config, runtime) = config_and_runtime();
    let baseline = seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap();

    let mut changed_k = runtime.clone();
    changed_k.novelty_neighbors = 14;
    let changed_k = seal(&config, &changed_k, AdmissionFixtureV1::default()).unwrap();
    assert_ne!(
        baseline.plan_identity_sha256(),
        changed_k.plan_identity_sha256()
    );

    let mut changed_calibration = AdmissionFixtureV1::default();
    changed_calibration.identities[7] = [0x45; 32];
    let changed_calibration = seal(&config, &runtime, changed_calibration).unwrap();
    assert_ne!(
        baseline.plan_identity_sha256(),
        changed_calibration.plan_identity_sha256()
    );

    let mut under_budget = AdmissionFixtureV1::default();
    under_budget.measured_rate = REQUIRED_RATE_V1 - 1;
    assert_eq!(
        seal(&config, &runtime, under_budget).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::ArchiveKnnBudgetExceeded
    );
}

#[test]
fn invalid_novelty_or_admission_fails_before_native_allocation() {
    let (config, baseline) = config_and_runtime();
    for invalid_neighbors in [0, config.population] {
        let mut runtime = baseline.clone();
        runtime.novelty_neighbors = invalid_neighbors;
        assert_eq!(
            seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
            CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyNeighbors
        );
    }
    for invalid_weight in [-0.0, f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut runtime = baseline.clone();
        runtime.novelty_weight = invalid_weight;
        assert_eq!(
            seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
            CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyWeight
        );
    }

    let mut invalid_identity = AdmissionFixtureV1::default();
    invalid_identity.identities[3] = [0; 32];
    assert_eq!(
        seal(&config, &baseline, invalid_identity).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InvalidAdmissionFacts
    );
    let invalid_reserve = AdmissionFixtureV1 {
        trim_prefilter_reserved_bytes: 3_000_000_001,
        ..AdmissionFixtureV1::default()
    };
    assert_eq!(
        seal(&config, &baseline, invalid_reserve).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InvalidAdmissionFacts
    );
}

#[test]
fn generic_symbol_or_account_geometry_fails_before_native_allocation() {
    let (baseline, runtime) = config_and_runtime();
    assert_eq!(baseline.evaluation_symbol, "EURUSD");
    assert_eq!(baseline.evaluation_account_currency, "GBP");

    let mut changed_symbol = baseline.clone();
    changed_symbol.evaluation_symbol = "GBPUSD".to_owned();
    assert_eq!(
        seal(&changed_symbol, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics
    );

    let mut changed_account = baseline.clone();
    changed_account.evaluation_account_currency = "USD".to_owned();
    assert_eq!(
        seal(&changed_account, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics
    );
}

#[test]
fn trim_and_archive_arithmetic_refuses_short_or_unrepresentable_runs() {
    let (mut config, runtime) = config_and_runtime();
    assert_eq!(
        seal_current_config_resident_search_plan_v1(
            &config,
            &runtime,
            79,
            500,
            AdmissionFixtureV1::default().seal(),
        )
        .unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InsufficientRows
    );

    config.generations = usize::MAX;
    assert_eq!(
        seal_current_config_resident_search_plan_v1(
            &config,
            &runtime,
            10_000,
            usize::MAX,
            AdmissionFixtureV1::default().seal(),
        )
        .unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow
    );
}

#[test]
fn canonical_digest_binds_market_session_and_cost_fields() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "timeframe_label" => |c| c.timeframe_label.push('x'),
        "evaluation_symbol" => |c| c.evaluation_symbol.push('x'),
        "evaluation_account_currency" => |c| c.evaluation_account_currency.push('x'),
        "evaluation_spread_pips" => |c| c.evaluation_spread_pips = changed_f64(c.evaluation_spread_pips),
        "evaluation_commission_per_trade" => |c| c.evaluation_commission_per_trade = changed_f64(c.evaluation_commission_per_trade),
        "session_spread_pips" => |c| c.session_spread_pips = Some([0.11, 0.22, 0.33]),
        "cost_band_pips" => |c| c.cost_band_pips = Some((0.44, 0.55)),
        "swap_long_pips_per_day" => |c| c.swap_long_pips_per_day = changed_f64(c.swap_long_pips_per_day),
        "swap_short_pips_per_day" => |c| c.swap_short_pips_per_day = changed_f64(c.swap_short_pips_per_day),
        "kill_zones_enabled" => |c| c.kill_zones_enabled = !c.kill_zones_enabled,
    );
}

#[test]
fn canonical_digest_binds_shape_targets_and_direct_timeframe_identity() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "population" => |c| c.population += 1,
        "population_auto" => |c| c.population_auto = !c.population_auto,
        "generations" => |c| c.generations += 1,
        "max_indicators" => |c| c.max_indicators += 1,
        "candidate_count" => |c| c.candidate_count += 1,
        "portfolio_size" => |c| c.portfolio_size += 1,
        "max_rows" => |c| c.max_rows += 1,
        "max_rows_by_timeframe" => |c| { c.max_rows_by_timeframe.insert("__DIGEST__".to_owned(), 7); },
        "max_hours" => |c| c.max_hours = changed_f64(c.max_hours),
        "corr_threshold" => |c| c.corr_threshold = changed_f64(c.corr_threshold),
        "min_trades_per_day" => |c| c.min_trades_per_day = changed_f64(c.min_trades_per_day),
        "higher_timeframes" => |c| c.higher_timeframes.push("__DIRECT_IDENTITY__".to_owned()),
    );
    assert_config_mutations!(&config;
        "target_profile.min_net_expectancy_per_trade" => |c| c.target_profile.min_net_expectancy_per_trade = changed_f64(c.target_profile.min_net_expectancy_per_trade),
        "target_profile.min_expectancy_t_stat" => |c| c.target_profile.min_expectancy_t_stat = changed_f64(c.target_profile.min_expectancy_t_stat),
        "target_profile.min_win_rate" => |c| c.target_profile.min_win_rate = changed_f64(c.target_profile.min_win_rate),
        "target_profile.min_payoff_ratio" => |c| c.target_profile.min_payoff_ratio = changed_f64(c.target_profile.min_payoff_ratio),
        "target_profile.max_in_market" => |c| c.target_profile.max_in_market = changed_f64(c.target_profile.max_in_market),
    );

    let mut ordered = config.clone();
    ordered.higher_timeframes = vec!["H1".to_owned(), "H4".to_owned()];
    let mut reversed = ordered.clone();
    reversed.higher_timeframes.reverse();
    assert_ne!(config_digest(&ordered), config_digest(&reversed));

    let mut first_map = config.clone();
    first_map.max_rows_by_timeframe.clear();
    first_map.max_rows_by_timeframe.insert("H1".to_owned(), 10);
    first_map.max_rows_by_timeframe.insert("H4".to_owned(), 20);
    let mut reverse_map = config.clone();
    reverse_map.max_rows_by_timeframe.clear();
    reverse_map
        .max_rows_by_timeframe
        .insert("H4".to_owned(), 20);
    reverse_map
        .max_rows_by_timeframe
        .insert("H1".to_owned(), 10);
    assert_eq!(config_digest(&first_map), config_digest(&reverse_map));
}

#[test]
fn canonical_digest_binds_all_filter_fields() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "filtering.max_dd" => |c| c.filtering.max_dd = changed_f64(c.filtering.max_dd),
        "filtering.min_profit" => |c| c.filtering.min_profit = changed_f64(c.filtering.min_profit),
        "filtering.min_trades" => |c| c.filtering.min_trades = changed_f64(c.filtering.min_trades),
        "filtering.min_sharpe" => |c| c.filtering.min_sharpe = changed_f64(c.filtering.min_sharpe),
        "filtering.min_win_rate" => |c| c.filtering.min_win_rate = changed_f64(c.filtering.min_win_rate),
        "filtering.min_profit_factor" => |c| c.filtering.min_profit_factor = changed_f64(c.filtering.min_profit_factor),
        "filtering.min_positive_months" => |c| c.filtering.min_positive_months += 1,
        "filtering.min_trades_per_month" => |c| c.filtering.min_trades_per_month = changed_f64(c.filtering.min_trades_per_month),
        "filtering.min_monthly_return_pct" => |c| c.filtering.min_monthly_return_pct = changed_f64(c.filtering.min_monthly_return_pct),
        "filtering.log_trades" => |c| c.filtering.log_trades = !c.filtering.log_trades,
        "filtering.trade_log_max" => |c| c.filtering.trade_log_max += 1,
        "filtering.opportunistic_enabled" => |c| c.filtering.opportunistic_enabled = !c.filtering.opportunistic_enabled,
        "filtering.use_opportunistic_candidates" => |c| c.filtering.use_opportunistic_candidates = !c.filtering.use_opportunistic_candidates,
        "filtering.opportunistic_min_positive_months" => |c| c.filtering.opportunistic_min_positive_months += 1,
        "filtering.opportunistic_min_trades_per_month" => |c| c.filtering.opportunistic_min_trades_per_month = changed_f64(c.filtering.opportunistic_min_trades_per_month),
        "filtering.opportunistic_min_trade_return_pct" => |c| c.filtering.opportunistic_min_trade_return_pct = changed_f64(c.filtering.opportunistic_min_trade_return_pct),
        "filtering.opportunistic_max_dd" => |c| c.filtering.opportunistic_max_dd = changed_f64(c.filtering.opportunistic_max_dd),
        "filtering.anomaly_guard" => |c| c.filtering.anomaly_guard = !c.filtering.anomaly_guard,
        "filtering.elite_mode" => |c| c.filtering.elite_mode = !c.filtering.elite_mode,
    );
}

#[test]
fn canonical_digest_binds_walkforward_cpcv_and_risk_fields() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "walkforward_splits" => |c| c.walkforward_splits += 1,
        "embargo_minutes" => |c| c.embargo_minutes += 1,
        "enable_cpcv" => |c| c.enable_cpcv = !c.enable_cpcv,
        "cpcv_n_splits" => |c| c.cpcv_n_splits += 1,
        "cpcv_n_test_groups" => |c| c.cpcv_n_test_groups += 1,
        "cpcv_embargo_pct" => |c| c.cpcv_embargo_pct = changed_f64(c.cpcv_embargo_pct),
        "cpcv_purge_pct" => |c| c.cpcv_purge_pct = changed_f64(c.cpcv_purge_pct),
        "cpcv_min_phi" => |c| c.cpcv_min_phi = changed_f64(c.cpcv_min_phi),
        "cpcv_max_rows" => |c| c.cpcv_max_rows += 1,
        "max_pbo" => |c| c.max_pbo = changed_f64(c.max_pbo),
        "initial_balance" => |c| c.initial_balance = changed_f64(c.initial_balance),
        "risk_per_trade_min" => |c| c.risk_per_trade_min = changed_f64(c.risk_per_trade_min),
        "risk_per_trade_max" => |c| c.risk_per_trade_max = changed_f64(c.risk_per_trade_max),
        "risky_risk_band" => |c| c.risky_risk_band = Some((0.011, 0.022)),
        "prop_firm_risk_band" => |c| c.prop_firm_risk_band = Some((0.003, 0.004)),
        "max_regime_loss_pct" => |c| c.max_regime_loss_pct = changed_f64(c.max_regime_loss_pct),
    );
}

#[test]
fn canonical_digest_binds_funnel_and_runtime_overrides() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "runtime_overrides.prefilter_top_k" => |c| c.runtime_overrides.prefilter_top_k += 1,
        "runtime_overrides.prefilter_insample_frac" => |c| c.runtime_overrides.prefilter_insample_frac = changed_f64(c.runtime_overrides.prefilter_insample_frac),
        "runtime_overrides.prefilter_min_per_timeframe" => |c| c.runtime_overrides.prefilter_min_per_timeframe += 1,
        "runtime_overrides.funnel_stage1_pct" => |c| c.runtime_overrides.funnel_stage1_pct = changed_f64(c.runtime_overrides.funnel_stage1_pct),
        "runtime_overrides.stage1_window" => |c| c.runtime_overrides.stage1_window = match c.runtime_overrides.stage1_window { crate::discovery::Stage1Window::MostRecent => crate::discovery::Stage1Window::Earliest, crate::discovery::Stage1Window::Earliest => crate::discovery::Stage1Window::MostRecent },
        "runtime_overrides.min_history_years" => |c| c.runtime_overrides.min_history_years += 1,
    );
}

#[test]
fn canonical_digest_binds_prop_firm_rules_and_gate_parameters() {
    let (mut config, _) = config_and_runtime();
    config.prop_firm_gate = Some(PropFirmGateOverrides {
        rules: PropFirmRiskRules::default(),
        n_windows: 17,
        window_days: 31,
        pass_rate: 0.61,
    });
    assert_config_mutations!(&config;
        "prop_firm_gate option" => |c| c.prop_firm_gate = None,
        "prop_firm_gate.rules.max_daily_loss_pct" => |c| { let g = c.prop_firm_gate.as_mut().unwrap(); g.rules.max_daily_loss_pct = changed_f64(g.rules.max_daily_loss_pct); },
        "prop_firm_gate.rules.max_overall_drawdown_pct" => |c| { let g = c.prop_firm_gate.as_mut().unwrap(); g.rules.max_overall_drawdown_pct = changed_f64(g.rules.max_overall_drawdown_pct); },
        "prop_firm_gate.rules.max_profit_consistency_ratio" => |c| { let g = c.prop_firm_gate.as_mut().unwrap(); g.rules.max_profit_consistency_ratio = changed_f64(g.rules.max_profit_consistency_ratio); },
        "prop_firm_gate.rules.min_trading_days" => |c| c.prop_firm_gate.as_mut().unwrap().rules.min_trading_days += 1,
        "prop_firm_gate.rules.max_trades_per_day" => |c| c.prop_firm_gate.as_mut().unwrap().rules.max_trades_per_day += 1,
        "prop_firm_gate.rules.require_profit_target" => |c| { let value = &mut c.prop_firm_gate.as_mut().unwrap().rules.require_profit_target; *value = !*value; },
        "prop_firm_gate.rules.min_profit_target_pct" => |c| { let g = c.prop_firm_gate.as_mut().unwrap(); g.rules.min_profit_target_pct = changed_f64(g.rules.min_profit_target_pct); },
        "prop_firm_gate.n_windows" => |c| c.prop_firm_gate.as_mut().unwrap().n_windows += 1,
        "prop_firm_gate.window_days" => |c| c.prop_firm_gate.as_mut().unwrap().window_days += 1,
        "prop_firm_gate.pass_rate" => |c| { let g = c.prop_firm_gate.as_mut().unwrap(); g.pass_rate = changed_f64(g.pass_rate); },
        "prop_firm_gate_params.max_daily_loss_pct" => |c| c.prop_firm_gate_params.max_daily_loss_pct = Some(0.051),
        "prop_firm_gate_params.max_overall_drawdown_pct" => |c| c.prop_firm_gate_params.max_overall_drawdown_pct = Some(0.101),
        "prop_firm_gate_params.profit_target_pct" => |c| c.prop_firm_gate_params.profit_target_pct = Some(0.091),
        "prop_firm_gate_params.min_trading_days" => |c| c.prop_firm_gate_params.min_trading_days = Some(7),
        "prop_firm_gate_params.window_days" => |c| c.prop_firm_gate_params.window_days += 1,
        "prop_firm_gate_params.n_windows" => |c| c.prop_firm_gate_params.n_windows += 1,
        "prop_firm_gate_params.pass_rate" => |c| c.prop_firm_gate_params.pass_rate = changed_f64(c.prop_firm_gate_params.pass_rate),
    );
}

#[test]
fn canonical_digest_binds_mc_adaptive_risky_export_and_ledger_fields() {
    let (config, _) = config_and_runtime();
    assert_config_mutations!(&config;
        "mc_runs" => |c| c.mc_runs += 1,
        "mc_min_profitable" => |c| c.mc_min_profitable += 1,
        "sensitivity_spread_pips" => |c| c.sensitivity_spread_pips = changed_f64(c.sensitivity_spread_pips),
        "sensitivity_commission_per_lot" => |c| c.sensitivity_commission_per_lot = changed_f64(c.sensitivity_commission_per_lot),
        "adaptive_thresholds" => |c| c.adaptive_thresholds = !c.adaptive_thresholds,
        "mode" => |c| c.mode = match c.mode { DiscoveryMode::Strict => DiscoveryMode::PropFirm, DiscoveryMode::PropFirm => DiscoveryMode::Risky, DiscoveryMode::Risky => DiscoveryMode::Strict },
        "risky_start_balance" => |c| c.risky_start_balance = changed_f64(c.risky_start_balance),
        "risky_target_balance" => |c| c.risky_target_balance = changed_f64(c.risky_target_balance),
        "risky_horizon_days" => |c| c.risky_horizon_days = changed_f64(c.risky_horizon_days),
        "require_walkforward_for_export" => |c| c.require_walkforward_for_export = !c.require_walkforward_for_export,
        "prop_firm_min_pass_rate" => |c| c.prop_firm_min_pass_rate = changed_f64(c.prop_firm_min_pass_rate),
        "discovery_ledger_enabled" => |c| c.discovery_ledger_enabled = !c.discovery_ledger_enabled,
        "discovery_ledger_cache_dir" => |c| c.discovery_ledger_cache_dir.push('x'),
        "discovery_ledger_archive_top_n" => |c| c.discovery_ledger_archive_top_n += 1,
    );
}

#[test]
fn eurusd_gbp_prefilter_geometry_ignores_optional_last_close_exactly() {
    let (config, _) = config_and_runtime();
    assert_eq!(config.evaluation_symbol, "EURUSD");
    assert_eq!(config.evaluation_account_currency, "GBP");
    let absent = resolve_prefilter_financial_geometry_v1(&config, None);
    let present = resolve_prefilter_financial_geometry_v1(&config, Some(1.173_456_789));
    assert_eq!(absent.max_hold_bars, present.max_hold_bars);
    assert_eq!(
        absent.stop_atr_multiplier.to_bits(),
        present.stop_atr_multiplier.to_bits()
    );
    assert_eq!(
        absent.reward_risk_ratio.to_bits(),
        present.reward_risk_ratio.to_bits()
    );
    assert_eq!(
        absent.round_trip_cost_price.to_bits(),
        present.round_trip_cost_price.to_bits()
    );
}
