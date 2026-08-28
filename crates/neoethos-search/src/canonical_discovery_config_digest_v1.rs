//! Versioned, deterministic identity for every shipped Search-semantic field.
//!
//! This deliberately does not use serde text. Field names, scalar kinds,
//! option presence, collection lengths, ordered vector positions, sorted map
//! keys, and raw IEEE-754 bits are all part of the wire identity. The
//! exhaustive destructuring below is also a compile-time ratchet: adding a
//! configuration field requires an explicit encoder decision.

use neoethos_core::config::PropFirmGateConfig;
use sha2::{Digest, Sha256};

use crate::discovery::{
    DiscoveryConfig, DiscoveryMode, DiscoveryRuntimeOverrides, PropFirmGateOverrides, Stage1Window,
    TargetProfile,
};
use crate::genetic::FilteringConfig;
use crate::validation::PropFirmRiskRules;

pub(crate) const CANONICAL_DISCOVERY_CONFIG_DIGEST_SCHEMA_V1: &str =
    "neoethos.discovery-config.canonical-typed.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CanonicalDiscoveryConfigDigestErrorV1 {
    ExtentOverflow,
}

pub(crate) fn canonical_discovery_config_digest_v1(
    config: &DiscoveryConfig,
) -> Result<[u8; 32], CanonicalDiscoveryConfigDigestErrorV1> {
    let DiscoveryConfig {
        timeframe_label,
        evaluation_symbol,
        evaluation_account_currency,
        evaluation_spread_pips,
        evaluation_commission_per_trade,
        session_spread_pips,
        cost_band_pips,
        swap_long_pips_per_day,
        swap_short_pips_per_day,
        kill_zones_enabled,
        population,
        population_auto,
        generations,
        max_indicators,
        candidate_count,
        portfolio_size,
        max_rows,
        max_rows_by_timeframe,
        max_hours,
        corr_threshold,
        min_trades_per_day,
        target_profile,
        walkforward_splits,
        embargo_minutes,
        enable_cpcv,
        cpcv_n_splits,
        cpcv_n_test_groups,
        cpcv_embargo_pct,
        cpcv_purge_pct,
        cpcv_min_phi,
        cpcv_max_rows,
        max_pbo,
        filtering,
        initial_balance,
        risk_per_trade_min,
        risk_per_trade_max,
        risky_risk_band,
        prop_firm_risk_band,
        max_regime_loss_pct,
        higher_timeframes,
        runtime_overrides,
        prop_firm_gate,
        mc_runs,
        mc_min_profitable,
        sensitivity_spread_pips,
        sensitivity_commission_per_lot,
        adaptive_thresholds,
        mode,
        prop_firm_gate_params,
        risky_start_balance,
        risky_target_balance,
        risky_horizon_days,
        require_walkforward_for_export,
        prop_firm_min_pass_rate,
        discovery_ledger_enabled,
        discovery_ledger_cache_dir,
        discovery_ledger_archive_top_n,
    } = config;

    let mut encoder = CanonicalDigestEncoderV1::new();
    encoder.string("timeframe_label", timeframe_label)?;
    encoder.string("evaluation_symbol", evaluation_symbol)?;
    encoder.string("evaluation_account_currency", evaluation_account_currency)?;
    encoder.f64("evaluation_spread_pips", *evaluation_spread_pips)?;
    encoder.f64(
        "evaluation_commission_per_trade",
        *evaluation_commission_per_trade,
    )?;
    encoder.option_f64_array_3("session_spread_pips", session_spread_pips)?;
    encoder.option_f64_pair("cost_band_pips", cost_band_pips)?;
    encoder.f64("swap_long_pips_per_day", *swap_long_pips_per_day)?;
    encoder.f64("swap_short_pips_per_day", *swap_short_pips_per_day)?;
    encoder.boolean("kill_zones_enabled", *kill_zones_enabled)?;
    encoder.usize("population", *population)?;
    encoder.boolean("population_auto", *population_auto)?;
    encoder.usize("generations", *generations)?;
    encoder.usize("max_indicators", *max_indicators)?;
    encoder.usize("candidate_count", *candidate_count)?;
    encoder.usize("portfolio_size", *portfolio_size)?;
    encoder.usize("max_rows", *max_rows)?;
    encoder.sorted_string_usize_map("max_rows_by_timeframe", max_rows_by_timeframe)?;
    encoder.f64("max_hours", *max_hours)?;
    encoder.f64("corr_threshold", *corr_threshold)?;
    encoder.f64("min_trades_per_day", *min_trades_per_day)?;
    encode_target_profile_v1(&mut encoder, target_profile)?;
    encoder.usize("walkforward_splits", *walkforward_splits)?;
    encoder.usize("embargo_minutes", *embargo_minutes)?;
    encoder.boolean("enable_cpcv", *enable_cpcv)?;
    encoder.usize("cpcv_n_splits", *cpcv_n_splits)?;
    encoder.usize("cpcv_n_test_groups", *cpcv_n_test_groups)?;
    encoder.f64("cpcv_embargo_pct", *cpcv_embargo_pct)?;
    encoder.f64("cpcv_purge_pct", *cpcv_purge_pct)?;
    encoder.f64("cpcv_min_phi", *cpcv_min_phi)?;
    encoder.usize("cpcv_max_rows", *cpcv_max_rows)?;
    encoder.f64("max_pbo", *max_pbo)?;
    encode_filtering_v1(&mut encoder, filtering)?;
    encoder.f64("initial_balance", *initial_balance)?;
    encoder.f64("risk_per_trade_min", *risk_per_trade_min)?;
    encoder.f64("risk_per_trade_max", *risk_per_trade_max)?;
    encoder.option_f64_pair("risky_risk_band", risky_risk_band)?;
    encoder.option_f64_pair("prop_firm_risk_band", prop_firm_risk_band)?;
    encoder.f64("max_regime_loss_pct", *max_regime_loss_pct)?;
    encoder.ordered_strings("higher_timeframes", higher_timeframes)?;
    encode_runtime_overrides_v1(&mut encoder, runtime_overrides)?;
    encode_optional_prop_firm_gate_v1(&mut encoder, prop_firm_gate.as_ref())?;
    encoder.u32("mc_runs", *mc_runs)?;
    encoder.u32("mc_min_profitable", *mc_min_profitable)?;
    encoder.f64("sensitivity_spread_pips", *sensitivity_spread_pips)?;
    encoder.f64(
        "sensitivity_commission_per_lot",
        *sensitivity_commission_per_lot,
    )?;
    encoder.boolean("adaptive_thresholds", *adaptive_thresholds)?;
    encoder.enumeration("mode", discovery_mode_wire_v1(*mode))?;
    encode_prop_firm_gate_params_v1(&mut encoder, prop_firm_gate_params)?;
    encoder.f64("risky_start_balance", *risky_start_balance)?;
    encoder.f64("risky_target_balance", *risky_target_balance)?;
    encoder.f64("risky_horizon_days", *risky_horizon_days)?;
    encoder.boolean(
        "require_walkforward_for_export",
        *require_walkforward_for_export,
    )?;
    encoder.f64("prop_firm_min_pass_rate", *prop_firm_min_pass_rate)?;
    encoder.boolean("discovery_ledger_enabled", *discovery_ledger_enabled)?;
    encoder.string("discovery_ledger_cache_dir", discovery_ledger_cache_dir)?;
    encoder.usize(
        "discovery_ledger_archive_top_n",
        *discovery_ledger_archive_top_n,
    )?;
    Ok(encoder.finish())
}

fn encode_target_profile_v1(
    encoder: &mut CanonicalDigestEncoderV1,
    target: &TargetProfile,
) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
    let TargetProfile {
        min_net_expectancy_per_trade,
        min_expectancy_t_stat,
        min_win_rate,
        min_payoff_ratio,
        max_in_market,
    } = target;
    encoder.record_start("target_profile")?;
    encoder.f64(
        "min_net_expectancy_per_trade",
        *min_net_expectancy_per_trade,
    )?;
    encoder.f64("min_expectancy_t_stat", *min_expectancy_t_stat)?;
    encoder.f64("min_win_rate", *min_win_rate)?;
    encoder.f64("min_payoff_ratio", *min_payoff_ratio)?;
    encoder.f64("max_in_market", *max_in_market)?;
    encoder.record_end("target_profile")
}

fn encode_filtering_v1(
    encoder: &mut CanonicalDigestEncoderV1,
    filtering: &FilteringConfig,
) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
    let FilteringConfig {
        max_dd,
        min_profit,
        min_trades,
        min_sharpe,
        min_win_rate,
        min_profit_factor,
        min_positive_months,
        min_trades_per_month,
        min_monthly_return_pct,
        log_trades,
        trade_log_max,
        opportunistic_enabled,
        use_opportunistic_candidates,
        opportunistic_min_positive_months,
        opportunistic_min_trades_per_month,
        opportunistic_min_trade_return_pct,
        opportunistic_max_dd,
        anomaly_guard,
        elite_mode,
    } = filtering;
    encoder.record_start("filtering")?;
    encoder.f64("max_dd", *max_dd)?;
    encoder.f64("min_profit", *min_profit)?;
    encoder.f64("min_trades", *min_trades)?;
    encoder.f64("min_sharpe", *min_sharpe)?;
    encoder.f64("min_win_rate", *min_win_rate)?;
    encoder.f64("min_profit_factor", *min_profit_factor)?;
    encoder.usize("min_positive_months", *min_positive_months)?;
    encoder.f64("min_trades_per_month", *min_trades_per_month)?;
    encoder.f64("min_monthly_return_pct", *min_monthly_return_pct)?;
    encoder.boolean("log_trades", *log_trades)?;
    encoder.usize("trade_log_max", *trade_log_max)?;
    encoder.boolean("opportunistic_enabled", *opportunistic_enabled)?;
    encoder.boolean(
        "use_opportunistic_candidates",
        *use_opportunistic_candidates,
    )?;
    encoder.usize(
        "opportunistic_min_positive_months",
        *opportunistic_min_positive_months,
    )?;
    encoder.f64(
        "opportunistic_min_trades_per_month",
        *opportunistic_min_trades_per_month,
    )?;
    encoder.f64(
        "opportunistic_min_trade_return_pct",
        *opportunistic_min_trade_return_pct,
    )?;
    encoder.f64("opportunistic_max_dd", *opportunistic_max_dd)?;
    encoder.boolean("anomaly_guard", *anomaly_guard)?;
    encoder.boolean("elite_mode", *elite_mode)?;
    encoder.record_end("filtering")
}

fn encode_runtime_overrides_v1(
    encoder: &mut CanonicalDigestEncoderV1,
    runtime: &DiscoveryRuntimeOverrides,
) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
    let DiscoveryRuntimeOverrides {
        prefilter_top_k,
        prefilter_insample_frac,
        prefilter_min_per_timeframe,
        funnel_stage1_pct,
        stage1_window,
        min_history_years,
    } = runtime;
    encoder.record_start("runtime_overrides")?;
    encoder.usize("prefilter_top_k", *prefilter_top_k)?;
    encoder.f64("prefilter_insample_frac", *prefilter_insample_frac)?;
    encoder.usize("prefilter_min_per_timeframe", *prefilter_min_per_timeframe)?;
    encoder.f64("funnel_stage1_pct", *funnel_stage1_pct)?;
    encoder.enumeration("stage1_window", stage1_window_wire_v1(*stage1_window))?;
    encoder.u32("min_history_years", *min_history_years)?;
    encoder.record_end("runtime_overrides")
}

fn encode_optional_prop_firm_gate_v1(
    encoder: &mut CanonicalDigestEncoderV1,
    gate: Option<&PropFirmGateOverrides>,
) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
    encoder.option_start("prop_firm_gate", gate.is_some())?;
    if let Some(gate) = gate {
        let PropFirmGateOverrides {
            rules,
            n_windows,
            window_days,
            pass_rate,
        } = gate;
        let PropFirmRiskRules {
            max_daily_loss_pct,
            max_overall_drawdown_pct,
            max_profit_consistency_ratio,
            min_trading_days,
            max_trades_per_day,
            require_profit_target,
            min_profit_target_pct,
        } = rules;
        encoder.record_start("rules")?;
        encoder.f64("max_daily_loss_pct", *max_daily_loss_pct)?;
        encoder.f64("max_overall_drawdown_pct", *max_overall_drawdown_pct)?;
        encoder.f64(
            "max_profit_consistency_ratio",
            *max_profit_consistency_ratio,
        )?;
        encoder.usize("min_trading_days", *min_trading_days)?;
        encoder.usize("max_trades_per_day", *max_trades_per_day)?;
        encoder.boolean("require_profit_target", *require_profit_target)?;
        encoder.f64("min_profit_target_pct", *min_profit_target_pct)?;
        encoder.record_end("rules")?;
        encoder.usize("n_windows", *n_windows)?;
        encoder.usize("window_days", *window_days)?;
        encoder.f64("pass_rate", *pass_rate)?;
    }
    encoder.option_end("prop_firm_gate")
}

fn encode_prop_firm_gate_params_v1(
    encoder: &mut CanonicalDigestEncoderV1,
    params: &PropFirmGateConfig,
) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
    let PropFirmGateConfig {
        max_daily_loss_pct,
        max_overall_drawdown_pct,
        profit_target_pct,
        min_trading_days,
        window_days,
        n_windows,
        pass_rate,
    } = params;
    encoder.record_start("prop_firm_gate_params")?;
    encoder.option_f64("max_daily_loss_pct", max_daily_loss_pct)?;
    encoder.option_f64("max_overall_drawdown_pct", max_overall_drawdown_pct)?;
    encoder.option_f64("profit_target_pct", profit_target_pct)?;
    encoder.option_usize("min_trading_days", min_trading_days)?;
    encoder.usize("window_days", *window_days)?;
    encoder.usize("n_windows", *n_windows)?;
    encoder.f64("pass_rate", *pass_rate)?;
    encoder.record_end("prop_firm_gate_params")
}

struct CanonicalDigestEncoderV1 {
    hash: Sha256,
}

impl CanonicalDigestEncoderV1 {
    fn new() -> Self {
        let mut hash = Sha256::new();
        hash.update(CANONICAL_DISCOVERY_CONFIG_DIGEST_SCHEMA_V1.as_bytes());
        Self { hash }
    }

    fn finish(self) -> [u8; 32] {
        self.hash.finalize().into()
    }

    fn name(&mut self, value: &str) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        let length = u64::try_from(value.len())
            .map_err(|_| CanonicalDiscoveryConfigDigestErrorV1::ExtentOverflow)?;
        self.hash.update(length.to_le_bytes());
        self.hash.update(value.as_bytes());
        Ok(())
    }

    fn marker(
        &mut self,
        name: &str,
        marker: u8,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.name(name)?;
        self.hash.update([marker]);
        Ok(())
    }

    fn record_start(&mut self, name: &str) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0xA0)
    }

    fn record_end(&mut self, name: &str) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0xAF)
    }

    fn option_start(
        &mut self,
        name: &str,
        present: bool,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x90)?;
        self.hash.update([u8::from(present)]);
        Ok(())
    }

    fn option_end(&mut self, name: &str) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x9F)
    }

    fn boolean(
        &mut self,
        name: &str,
        value: bool,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x01)?;
        self.hash.update([u8::from(value)]);
        Ok(())
    }

    fn enumeration(
        &mut self,
        name: &str,
        value: u8,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x02)?;
        self.hash.update([value]);
        Ok(())
    }

    fn u32(&mut self, name: &str, value: u32) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x03)?;
        self.hash.update(value.to_le_bytes());
        Ok(())
    }

    fn usize(
        &mut self,
        name: &str,
        value: usize,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        let value = u64::try_from(value)
            .map_err(|_| CanonicalDiscoveryConfigDigestErrorV1::ExtentOverflow)?;
        self.marker(name, 0x04)?;
        self.hash.update(value.to_le_bytes());
        Ok(())
    }

    fn f64(&mut self, name: &str, value: f64) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x05)?;
        self.hash.update(value.to_bits().to_le_bytes());
        Ok(())
    }

    fn string(
        &mut self,
        name: &str,
        value: &str,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x06)?;
        self.name(value)
    }

    fn option_f64(
        &mut self,
        name: &str,
        value: &Option<f64>,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.option_start(name, value.is_some())?;
        if let Some(value) = value {
            self.f64("some", *value)?;
        }
        self.option_end(name)
    }

    fn option_usize(
        &mut self,
        name: &str,
        value: &Option<usize>,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.option_start(name, value.is_some())?;
        if let Some(value) = value {
            self.usize("some", *value)?;
        }
        self.option_end(name)
    }

    fn option_f64_pair(
        &mut self,
        name: &str,
        value: &Option<(f64, f64)>,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.option_start(name, value.is_some())?;
        if let Some((first, second)) = value {
            self.f64("first", *first)?;
            self.f64("second", *second)?;
        }
        self.option_end(name)
    }

    fn option_f64_array_3(
        &mut self,
        name: &str,
        value: &Option<[f64; 3]>,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.option_start(name, value.is_some())?;
        if let Some(values) = value {
            self.usize("length", values.len())?;
            for (index, value) in values.iter().enumerate() {
                self.f64(&format!("item-{index}"), *value)?;
            }
        }
        self.option_end(name)
    }

    fn ordered_strings(
        &mut self,
        name: &str,
        values: &[String],
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x70)?;
        self.usize("length", values.len())?;
        for (index, value) in values.iter().enumerate() {
            self.string(&format!("item-{index}"), value)?;
        }
        Ok(())
    }

    fn sorted_string_usize_map(
        &mut self,
        name: &str,
        values: &std::collections::HashMap<String, usize>,
    ) -> Result<(), CanonicalDiscoveryConfigDigestErrorV1> {
        self.marker(name, 0x71)?;
        self.usize("length", values.len())?;
        let mut entries = values.iter().collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
        for (index, (key, value)) in entries.into_iter().enumerate() {
            self.string(&format!("key-{index}"), key)?;
            self.usize(&format!("value-{index}"), *value)?;
        }
        Ok(())
    }
}

const fn stage1_window_wire_v1(window: Stage1Window) -> u8 {
    match window {
        Stage1Window::MostRecent => 1,
        Stage1Window::Earliest => 2,
    }
}

const fn discovery_mode_wire_v1(mode: DiscoveryMode) -> u8 {
    match mode {
        DiscoveryMode::Strict => 1,
        DiscoveryMode::PropFirm => 2,
        DiscoveryMode::Risky => 3,
    }
}
