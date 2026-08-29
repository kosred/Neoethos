use std::path::PathBuf;

use neoethos_core::Settings;
use sha2::{Digest as _, Sha256 as IndependentSha256V2};

use super::*;
use crate::discovery::{PropFirmGateOverrides, resolve_prefilter_financial_geometry_v1};
use crate::validation::PropFirmRiskRules;

const REQUIRED_RATE_V1: u64 = 223_106_667;
const CURRENT_CONFIG_SLICE2_PLAN_IDENTITY_KNOWN_SHA256_V2: [u8; 32] = [
    0x7d, 0x09, 0x56, 0xe0, 0xdd, 0x7a, 0x24, 0x00, 0xc8, 0xbc, 0xd4, 0xfa, 0xda, 0x51, 0xef, 0xa6,
    0xe1, 0xd1, 0x79, 0x23, 0xbb, 0x39, 0x98, 0xbf, 0xe5, 0xd1, 0x7a, 0x72, 0xbd, 0x2b, 0x1f, 0xb3,
];

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

fn current_config_slice2_facts_v2() -> CurrentConfigResidentSearchSlice2PlanFactsV2 {
    CurrentConfigResidentSearchSlice2PlanFactsV2 {
        population: 200,
        maximum_generations: 20_000,
        maximum_runtime_millis: 3_600_000,
        maximum_terms_per_gene: 16,
        gene_signature_word_count: 4,
        novelty_weight_bits: 0.2_f64.to_bits(),
        novelty_neighbors: 15,
        permanent_archive_capacity: 50_000,
        calibration_active_count: 50_000,
        maximum_jaccard_union: 32,
        maximum_jaccard_cross_product: 1_024,
        maximum_archive_knn_distance_count: 200_796_000_000,
        maximum_archive_knn_popcount_word_count: 803_184_000_000,
        required_archive_knn_distance_items_per_second: 55_776_667,
        required_archive_knn_popcount_words_per_second: 223_106_667,
        layout_alignment_bytes: 256,
        archive_gene_scalars_bytes: 3_600_128,
        archive_term_indices_bytes: 6_400_000,
        archive_term_weights_bytes: 6_400_000,
        archive_metric_rows_bytes: 5_200_128,
        archive_signatures_bytes: 1_600_000,
        archive_hashes_bytes: 400_128,
        current_population_signatures_bytes: 6_400,
        novelty_scores_bytes: 1_792,
        exact_top_k_keys_bytes: 96_000,
        admission_flags_bytes: 1_024,
        admission_offsets_bytes: 1_792,
        archive_control_and_seal_bytes: 256,
        control_subtotal_bytes: 3_072,
        slice2_replacement_subtotal_bytes: 23_707_648,
        replaced_v1_scoring_bytes: 8_448,
        slice2_net_additional_bytes: 23_699_200,
        current_source_kind_wire: 0,
        archive_source_kind_wire: 1,
        current_ordinal_exclusive_end: 200,
        archive_ordinal_exclusive_end: 50_000,
        binary64_operation_sequence_wire: 1,
        binary64_math_mode_wire: 1,
        binary64_tolerance_policy_wire: 1,
        binary64_absolute_tolerance_bits: 0x3cd0_0000_0000_0000,
        binary64_relative_tolerance_bits: 0x3cf0_0000_0000_0000,
        binary64_max_ulp_distance: 4,
        novelty_semantics_identity_sha256: [0x51; 32],
        archive_capacity_identity_sha256: [0x52; 32],
        calibration_active_count_identity_sha256: [0x53; 32],
        layout_identity_sha256: [0x54; 32],
        calibration_identity_sha256: [0x55; 32],
        source_kind_encoding_identity_sha256: [0x56; 32],
        current_ordinal_domain_identity_sha256: [0x57; 32],
        archive_ordinal_domain_identity_sha256: [0x58; 32],
        tie_order_identity_sha256: [0x59; 32],
        binary64_operation_sequence_identity_sha256: [0x5a; 32],
        binary64_math_mode_identity_sha256: [0x5b; 32],
        binary64_tolerance_identity_sha256: [0x5c; 32],
    }
}

fn current_config_slice2_base_v1() -> SealedCurrentConfigResidentSearchPlanV1 {
    let (config, runtime) = config_and_runtime();
    seal(&config, &runtime, AdmissionFixtureV1::default())
        .expect("the already-GREEN current-config V1 plan must seal")
}

fn independent_current_config_slice2_identity_oracle_v2(
    base_v1: &SealedCurrentConfigResidentSearchPlanV1,
    facts: &CurrentConfigResidentSearchSlice2PlanFactsV2,
) -> [u8; 32] {
    let mut hash = IndependentSha256V2::new();
    hash.update(CURRENT_CONFIG_RESIDENT_SEARCH_SLICE2_PLAN_SEMANTICS_V2.as_bytes());
    hash.update(base_v1.plan_identity_sha256());
    hash.update(facts.population.to_le_bytes());
    hash.update(facts.maximum_generations.to_le_bytes());
    hash.update(facts.maximum_runtime_millis.to_le_bytes());
    hash.update(facts.maximum_terms_per_gene.to_le_bytes());
    hash.update(facts.gene_signature_word_count.to_le_bytes());
    hash.update(facts.novelty_weight_bits.to_le_bytes());
    hash.update(facts.novelty_neighbors.to_le_bytes());
    hash.update(facts.permanent_archive_capacity.to_le_bytes());
    hash.update(facts.calibration_active_count.to_le_bytes());
    hash.update(facts.maximum_jaccard_union.to_le_bytes());
    hash.update(facts.maximum_jaccard_cross_product.to_le_bytes());
    hash.update(facts.maximum_archive_knn_distance_count.to_le_bytes());
    hash.update(facts.maximum_archive_knn_popcount_word_count.to_le_bytes());
    hash.update(
        facts
            .required_archive_knn_distance_items_per_second
            .to_le_bytes(),
    );
    hash.update(
        facts
            .required_archive_knn_popcount_words_per_second
            .to_le_bytes(),
    );
    hash.update(facts.layout_alignment_bytes.to_le_bytes());
    hash.update(facts.archive_gene_scalars_bytes.to_le_bytes());
    hash.update(facts.archive_term_indices_bytes.to_le_bytes());
    hash.update(facts.archive_term_weights_bytes.to_le_bytes());
    hash.update(facts.archive_metric_rows_bytes.to_le_bytes());
    hash.update(facts.archive_signatures_bytes.to_le_bytes());
    hash.update(facts.archive_hashes_bytes.to_le_bytes());
    hash.update(facts.current_population_signatures_bytes.to_le_bytes());
    hash.update(facts.novelty_scores_bytes.to_le_bytes());
    hash.update(facts.exact_top_k_keys_bytes.to_le_bytes());
    hash.update(facts.admission_flags_bytes.to_le_bytes());
    hash.update(facts.admission_offsets_bytes.to_le_bytes());
    hash.update(facts.archive_control_and_seal_bytes.to_le_bytes());
    hash.update(facts.control_subtotal_bytes.to_le_bytes());
    hash.update(facts.slice2_replacement_subtotal_bytes.to_le_bytes());
    hash.update(facts.replaced_v1_scoring_bytes.to_le_bytes());
    hash.update(facts.slice2_net_additional_bytes.to_le_bytes());
    hash.update([facts.current_source_kind_wire]);
    hash.update([facts.archive_source_kind_wire]);
    hash.update(facts.current_ordinal_exclusive_end.to_le_bytes());
    hash.update(facts.archive_ordinal_exclusive_end.to_le_bytes());
    hash.update([facts.binary64_operation_sequence_wire]);
    hash.update([facts.binary64_math_mode_wire]);
    hash.update([facts.binary64_tolerance_policy_wire]);
    hash.update(facts.binary64_absolute_tolerance_bits.to_le_bytes());
    hash.update(facts.binary64_relative_tolerance_bits.to_le_bytes());
    hash.update(facts.binary64_max_ulp_distance.to_le_bytes());
    hash.update(facts.novelty_semantics_identity_sha256);
    hash.update(facts.archive_capacity_identity_sha256);
    hash.update(facts.calibration_active_count_identity_sha256);
    hash.update(facts.layout_identity_sha256);
    hash.update(facts.calibration_identity_sha256);
    hash.update(facts.source_kind_encoding_identity_sha256);
    hash.update(facts.current_ordinal_domain_identity_sha256);
    hash.update(facts.archive_ordinal_domain_identity_sha256);
    hash.update(facts.tie_order_identity_sha256);
    hash.update(facts.binary64_operation_sequence_identity_sha256);
    hash.update(facts.binary64_math_mode_identity_sha256);
    hash.update(facts.binary64_tolerance_identity_sha256);
    hash.finalize().into()
}

fn seal_current_config_slice2_v2(
    facts_v2: CurrentConfigResidentSearchSlice2PlanFactsV2,
) -> Result<
    SealedCurrentConfigResidentSearchSlice2PlanV2,
    CurrentConfigResidentSearchSlice2PlanErrorV2,
> {
    seal_current_config_resident_search_slice2_plan_v2(current_config_slice2_base_v1(), facts_v2)
}

fn require_current_config_slice2_v2(
    facts_v2: CurrentConfigResidentSearchSlice2PlanFactsV2,
) -> SealedCurrentConfigResidentSearchSlice2PlanV2 {
    match seal_current_config_slice2_v2(facts_v2) {
        Ok(plan) => plan,
        Err(error) => {
            panic!("Slice2 R1 valid current-config seal failed with exact RED: {error:?}")
        }
    }
}

fn assert_slice2_facts_rejected_before_allocation(
    label: &str,
    facts_v2: CurrentConfigResidentSearchSlice2PlanFactsV2,
    expected_error: CurrentConfigResidentSearchSlice2PlanErrorV2,
) {
    match seal_current_config_slice2_v2(facts_v2) {
        Ok(_) => panic!(
            "{label}: invalid Slice2 facts yielded the sealed authority required for allocation"
        ),
        Err(error) => assert_eq!(
            error, expected_error,
            "{label}: invalid Slice2 facts returned the wrong fail-closed error"
        ),
    }
}

#[test]
fn slice2_current_config_facts_layout_and_binary64_contract_are_exact() {
    let expected = current_config_slice2_facts_v2();
    let base_v1 = current_config_slice2_base_v1();
    let identity_oracle = independent_current_config_slice2_identity_oracle_v2(&base_v1, &expected);
    assert_eq!(
        identity_oracle,
        CURRENT_CONFIG_SLICE2_PLAN_IDENTITY_KNOWN_SHA256_V2
    );
    let plan = require_current_config_slice2_v2(expected);
    let facts = plan.facts_v2();

    assert_eq!(
        CURRENT_CONFIG_RESIDENT_SEARCH_SLICE2_PLAN_SEMANTICS_V2,
        "neoethos.current-config-resident-search-slice2-plan.v2"
    );
    assert_eq!(
        plan.identity_receipt_v2().identity_sha256(),
        identity_oracle
    );
    assert_eq!(facts, &expected);
    assert_eq!(facts.population, 200);
    assert_eq!(facts.maximum_generations, 20_000);
    assert_eq!(facts.maximum_runtime_millis, 3_600_000);
    assert_eq!(facts.maximum_terms_per_gene, 16);
    assert_eq!(facts.gene_signature_word_count, 4);
    assert_eq!(facts.novelty_weight_bits, 0.2_f64.to_bits());
    assert_eq!(facts.novelty_neighbors, 15);
    assert_eq!(facts.permanent_archive_capacity, 50_000);
    assert_eq!(facts.calibration_active_count, 50_000);

    let maximum_union = facts
        .maximum_terms_per_gene
        .checked_mul(2)
        .expect("current-config union bound must fit");
    assert_eq!(maximum_union, u64::from(facts.maximum_jaccard_union));
    assert_eq!(
        maximum_union.checked_mul(maximum_union),
        Some(facts.maximum_jaccard_cross_product)
    );
    let neighbors_per_candidate = facts
        .permanent_archive_capacity
        .checked_add(facts.population - 1)
        .expect("current-config neighbor bound must fit");
    let distances_per_generation = facts
        .population
        .checked_mul(neighbors_per_candidate)
        .expect("current-config distance bound must fit");
    assert_eq!(distances_per_generation, 10_039_800);
    assert_eq!(
        distances_per_generation.checked_mul(facts.maximum_generations),
        Some(facts.maximum_archive_knn_distance_count)
    );
    assert_eq!(
        facts
            .maximum_archive_knn_distance_count
            .checked_mul(facts.gene_signature_word_count),
        Some(facts.maximum_archive_knn_popcount_word_count)
    );
    assert_eq!(
        facts
            .maximum_archive_knn_distance_count
            .checked_mul(1_000)
            .and_then(|scaled| scaled.checked_add(facts.maximum_runtime_millis - 1))
            .map(|rounded| rounded / facts.maximum_runtime_millis),
        Some(facts.required_archive_knn_distance_items_per_second)
    );
    assert_eq!(
        facts
            .maximum_archive_knn_popcount_word_count
            .checked_mul(1_000)
            .and_then(|scaled| scaled.checked_add(facts.maximum_runtime_millis - 1))
            .map(|rounded| rounded / facts.maximum_runtime_millis),
        Some(facts.required_archive_knn_popcount_words_per_second)
    );

    let replacement_components = [
        facts.archive_gene_scalars_bytes,
        facts.archive_term_indices_bytes,
        facts.archive_term_weights_bytes,
        facts.archive_metric_rows_bytes,
        facts.archive_signatures_bytes,
        facts.archive_hashes_bytes,
        facts.current_population_signatures_bytes,
        facts.novelty_scores_bytes,
        facts.exact_top_k_keys_bytes,
        facts.admission_flags_bytes,
        facts.admission_offsets_bytes,
        facts.archive_control_and_seal_bytes,
    ];
    assert_eq!(facts.layout_alignment_bytes, 256);
    assert_eq!(
        replacement_components
            .into_iter()
            .try_fold(0_u64, u64::checked_add),
        Some(facts.slice2_replacement_subtotal_bytes)
    );
    assert_eq!(
        facts
            .admission_flags_bytes
            .checked_add(facts.admission_offsets_bytes)
            .and_then(|value| value.checked_add(facts.archive_control_and_seal_bytes)),
        Some(facts.control_subtotal_bytes)
    );
    assert_eq!(facts.slice2_replacement_subtotal_bytes, 23_707_648);
    assert_eq!(facts.replaced_v1_scoring_bytes, 8_448);
    assert_eq!(
        facts
            .slice2_replacement_subtotal_bytes
            .checked_sub(facts.replaced_v1_scoring_bytes),
        Some(facts.slice2_net_additional_bytes)
    );

    assert_eq!(facts.current_source_kind_wire, 0);
    assert_eq!(facts.archive_source_kind_wire, 1);
    assert_eq!(facts.current_ordinal_exclusive_end, facts.population);
    assert_eq!(
        facts.archive_ordinal_exclusive_end,
        facts.permanent_archive_capacity
    );
    assert_eq!(facts.binary64_operation_sequence_wire, 1);
    assert_eq!(facts.binary64_math_mode_wire, 1);
    assert_eq!(facts.binary64_tolerance_policy_wire, 1);
    assert_eq!(
        facts.binary64_absolute_tolerance_bits,
        2.0_f64.powi(-50).to_bits()
    );
    assert_eq!(
        facts.binary64_relative_tolerance_bits,
        2.0_f64.powi(-48).to_bits()
    );
    assert_eq!(facts.binary64_max_ulp_distance, 4);
}

#[test]
fn slice2_identity_inputs_change_run_identity_and_reject_stale_receipts_independently() {
    let baseline_facts = current_config_slice2_facts_v2();
    let baseline_plan = require_current_config_slice2_v2(baseline_facts);
    assert_eq!(
        baseline_plan.validate_identity_receipt_v2(baseline_plan.identity_receipt_v2()),
        Ok(())
    );

    macro_rules! assert_identity_change {
        ($label:literal, $field:ident, $replacement:expr) => {{
            let mut changed_facts = baseline_facts;
            changed_facts.$field = $replacement;
            let changed_plan = require_current_config_slice2_v2(changed_facts);
            assert_ne!(
                baseline_plan.identity_receipt_v2(),
                changed_plan.identity_receipt_v2(),
                "{} identity did not alter the Slice2 run identity",
                $label
            );
            assert_eq!(
                changed_plan.validate_identity_receipt_v2(baseline_plan.identity_receipt_v2()),
                Err(CurrentConfigResidentSearchSlice2PlanErrorV2::IdentityReceiptMismatch),
                "{} mutation accepted the old Slice2 identity receipt",
                $label
            );
        }};
    }

    assert_identity_change!("novelty", novelty_semantics_identity_sha256, [0x81; 32]);
    assert_identity_change!(
        "archive capacity",
        archive_capacity_identity_sha256,
        [0x82; 32]
    );
    assert_identity_change!(
        "calibration active count",
        calibration_active_count_identity_sha256,
        [0x83; 32]
    );
    assert_identity_change!("layout", layout_identity_sha256, [0x84; 32]);
    assert_identity_change!("calibration", calibration_identity_sha256, [0x85; 32]);
    assert_identity_change!(
        "source kind",
        source_kind_encoding_identity_sha256,
        [0x86; 32]
    );
    assert_identity_change!(
        "current ordinal domain",
        current_ordinal_domain_identity_sha256,
        [0x87; 32]
    );
    assert_identity_change!(
        "archive ordinal domain",
        archive_ordinal_domain_identity_sha256,
        [0x88; 32]
    );
    assert_identity_change!("tie order", tie_order_identity_sha256, [0x89; 32]);
    assert_identity_change!(
        "binary64 operation sequence",
        binary64_operation_sequence_identity_sha256,
        [0x8a; 32]
    );
    assert_identity_change!(
        "binary64 math mode",
        binary64_math_mode_identity_sha256,
        [0x8b; 32]
    );
    assert_identity_change!(
        "binary64 tolerance",
        binary64_tolerance_identity_sha256,
        [0x8c; 32]
    );
}

#[test]
fn slice2_checked_overflow_fails_before_allocation() {
    let baseline = current_config_slice2_facts_v2();
    let _ = require_current_config_slice2_v2(baseline);

    let mut neighbor_overflow = baseline;
    neighbor_overflow.permanent_archive_capacity = u64::MAX;
    assert_slice2_facts_rejected_before_allocation(
        "neighbor extent overflow",
        neighbor_overflow,
        CurrentConfigResidentSearchSlice2PlanErrorV2::ArithmeticOverflow,
    );

    let mut rational_overflow = baseline;
    rational_overflow.maximum_terms_per_gene = u64::MAX;
    rational_overflow.maximum_jaccard_union = u32::MAX;
    rational_overflow.maximum_jaccard_cross_product = u64::MAX;
    assert_slice2_facts_rejected_before_allocation(
        "rational cross-product overflow",
        rational_overflow,
        CurrentConfigResidentSearchSlice2PlanErrorV2::ArithmeticOverflow,
    );

    let mut work_overflow = baseline;
    work_overflow.maximum_archive_knn_distance_count = u64::MAX;
    work_overflow.maximum_archive_knn_popcount_word_count = u64::MAX;
    assert_slice2_facts_rejected_before_allocation(
        "work-rate scaling overflow",
        work_overflow,
        CurrentConfigResidentSearchSlice2PlanErrorV2::ArithmeticOverflow,
    );

    let mut layout_overflow = baseline;
    layout_overflow.archive_gene_scalars_bytes = u64::MAX;
    layout_overflow.slice2_replacement_subtotal_bytes = u64::MAX;
    assert_slice2_facts_rejected_before_allocation(
        "layout subtotal overflow",
        layout_overflow,
        CurrentConfigResidentSearchSlice2PlanErrorV2::ArithmeticOverflow,
    );
}

#[test]
fn slice2_every_current_config_numeric_extent_drift_fails_before_allocation() {
    let baseline = current_config_slice2_facts_v2();
    let _ = require_current_config_slice2_v2(baseline);

    macro_rules! assert_extent_drift {
        ($label:literal, $field:ident, $replacement:expr) => {{
            let mut changed = baseline;
            changed.$field = $replacement;
            assert_slice2_facts_rejected_before_allocation(
                $label,
                changed,
                CurrentConfigResidentSearchSlice2PlanErrorV2::CurrentConfigExtentMismatch,
            );
        }};
    }

    assert_extent_drift!("population", population, 201);
    assert_extent_drift!("maximum generations", maximum_generations, 20_001);
    assert_extent_drift!("maximum runtime", maximum_runtime_millis, 3_600_001);
    assert_extent_drift!("maximum terms", maximum_terms_per_gene, 17);
    assert_extent_drift!("signature words", gene_signature_word_count, 5);
    assert_extent_drift!("novelty weight", novelty_weight_bits, 0.25_f64.to_bits());
    assert_extent_drift!("novelty neighbors", novelty_neighbors, 14);
    assert_extent_drift!("archive capacity", permanent_archive_capacity, 49_999);
    assert_extent_drift!("calibration active count", calibration_active_count, 49_999);
    assert_extent_drift!("maximum union", maximum_jaccard_union, 31);
    assert_extent_drift!(
        "maximum cross product",
        maximum_jaccard_cross_product,
        1_023
    );
    assert_extent_drift!(
        "maximum distance count",
        maximum_archive_knn_distance_count,
        200_795_999_999
    );
    assert_extent_drift!(
        "maximum popcount word count",
        maximum_archive_knn_popcount_word_count,
        803_183_999_999
    );
    assert_extent_drift!(
        "required distance rate",
        required_archive_knn_distance_items_per_second,
        55_776_666
    );
    assert_extent_drift!(
        "required popcount rate",
        required_archive_knn_popcount_words_per_second,
        223_106_666
    );
    assert_extent_drift!("layout alignment", layout_alignment_bytes, 128);
    assert_extent_drift!(
        "archive gene scalars",
        archive_gene_scalars_bytes,
        3_600_127
    );
    assert_extent_drift!(
        "archive term indices",
        archive_term_indices_bytes,
        6_399_999
    );
    assert_extent_drift!(
        "archive term weights",
        archive_term_weights_bytes,
        6_399_999
    );
    assert_extent_drift!("archive metric rows", archive_metric_rows_bytes, 5_200_127);
    assert_extent_drift!("archive signatures", archive_signatures_bytes, 1_599_999);
    assert_extent_drift!("archive hashes", archive_hashes_bytes, 400_127);
    assert_extent_drift!(
        "current population signatures",
        current_population_signatures_bytes,
        6_399
    );
    assert_extent_drift!("novelty scores", novelty_scores_bytes, 1_791);
    assert_extent_drift!("exact top-k keys", exact_top_k_keys_bytes, 95_999);
    assert_extent_drift!("admission flags", admission_flags_bytes, 1_023);
    assert_extent_drift!("admission offsets", admission_offsets_bytes, 1_791);
    assert_extent_drift!(
        "archive control and seal",
        archive_control_and_seal_bytes,
        255
    );
    assert_extent_drift!("control subtotal", control_subtotal_bytes, 3_071);
    assert_extent_drift!(
        "replacement subtotal",
        slice2_replacement_subtotal_bytes,
        23_707_647
    );
    assert_extent_drift!("replaced V1 scoring", replaced_v1_scoring_bytes, 8_447);
    assert_extent_drift!(
        "net additional bytes",
        slice2_net_additional_bytes,
        23_699_199
    );
    assert_extent_drift!("current source wire", current_source_kind_wire, 1);
    assert_extent_drift!("archive source wire", archive_source_kind_wire, 0);
    assert_extent_drift!("current ordinal domain", current_ordinal_exclusive_end, 199);
    assert_extent_drift!(
        "archive ordinal domain",
        archive_ordinal_exclusive_end,
        49_999
    );
    assert_extent_drift!(
        "binary64 operation sequence",
        binary64_operation_sequence_wire,
        2
    );
    assert_extent_drift!("binary64 math mode", binary64_math_mode_wire, 2);
    assert_extent_drift!(
        "binary64 tolerance policy",
        binary64_tolerance_policy_wire,
        2
    );
    assert_extent_drift!(
        "binary64 absolute tolerance",
        binary64_absolute_tolerance_bits,
        0x3cd0_0000_0000_0001
    );
    assert_extent_drift!(
        "binary64 relative tolerance",
        binary64_relative_tolerance_bits,
        0x3cf0_0000_0000_0001
    );
    assert_extent_drift!("binary64 ULP tolerance", binary64_max_ulp_distance, 5);
}
