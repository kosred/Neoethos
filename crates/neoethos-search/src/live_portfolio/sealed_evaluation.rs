//! Complete immutable EvaluationConfig wire snapshot. The explicit struct
//! literal on replay ensures newly added evaluator fields cannot silently use
//! process defaults. Research costs remain assumptions, not quote authority.
use super::LiveTradingPolicyV1;
use crate::genetic::EvaluationConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SealedEvaluationPolicyV1 {
    schema_version: u16,
    smc_gate_disabled: bool,
    adaptive_stops_v1: crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    symbol: String,
    account_currency: String,
    initial_equity: f64,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    trailing_min_lock_pips: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
    kill_zones_enabled: bool,
    session_spread_pips: Option<[f64; 3]>,
    risk_per_trade_min: f64,
    risk_per_trade_max: f64,
    high_quality_confidence: f64,
    smc_gate_threshold: f64,
    smc_weight_ob: f64,
    smc_weight_fvg: f64,
    smc_weight_liq: f64,
    smc_weight_mtf: f64,
    smc_weight_premium: f64,
    smc_weight_inducement: f64,
    smc_weight_bos: f64,
    smc_weight_choch: f64,
    smc_weight_eqh: f64,
    smc_weight_eql: f64,
    smc_weight_displacement: f64,
    growth_objective: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    growth_goal: Option<crate::scoring::RiskyGrowthGoal>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluation_and_stops() -> (
        EvaluationConfig,
        crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) {
        let mut evaluation =
            EvaluationConfig::for_symbol("EURUSD", "USD", Some(1.1), Some(1.2), Some(7.0));
        evaluation.initial_equity = 10_000.0;
        evaluation.growth_objective = true;
        let mut settings = crate::stop_target::StopTargetSettings::default();
        settings.vol_estimator = "parkinson".to_owned();
        settings.atr_stop_multiplier = 1.5;
        let stops =
            crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(settings, true, 2.0)
                .unwrap();
        (evaluation, stops)
    }

    #[test]
    fn optional_growth_goal_preserves_legacy_wire_and_actual_simulation_capital() {
        let (mut evaluation, stops) = evaluation_and_stops();
        let legacy = SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).unwrap();
        let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
        assert!(
            !serde_json::to_value(&legacy)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("growth_goal")
        );
        let reopened: SealedEvaluationPolicyV1 = serde_json::from_slice(&legacy_bytes).unwrap();
        reopened.validate().unwrap();
        assert_eq!(reopened.to_evaluation().growth_goal, None);
        assert_eq!(serde_json::to_vec(&reopened).unwrap(), legacy_bytes);

        let goal = crate::scoring::RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        };
        evaluation.growth_goal = Some(goal);
        let snapshot =
            SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).unwrap();
        let bytes = serde_json::to_vec(&snapshot).unwrap();
        assert_ne!(bytes, legacy_bytes);
        let reopened: SealedEvaluationPolicyV1 = serde_json::from_slice(&bytes).unwrap();
        reopened.validate().unwrap();
        assert_eq!(reopened.to_evaluation().growth_goal, Some(goal));
        assert_eq!(reopened.to_evaluation().initial_equity, 10_000.0);
        assert_eq!(serde_json::to_vec(&reopened).unwrap(), bytes);
        evaluation.growth_goal.as_mut().unwrap().horizon_days = 90.0;
        let changed =
            SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).unwrap();
        assert_ne!(serde_json::to_vec(&changed).unwrap(), bytes);
    }

    #[test]
    fn optional_growth_goal_rejects_invalid_or_non_growth_policy() {
        let (mut evaluation, stops) = evaluation_and_stops();
        let goal = crate::scoring::RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        };
        for invalid in [
            crate::scoring::RiskyGrowthGoal {
                start_balance: 0.0,
                ..goal
            },
            crate::scoring::RiskyGrowthGoal {
                target_balance: 100.0,
                ..goal
            },
            crate::scoring::RiskyGrowthGoal {
                horizon_days: f64::NAN,
                ..goal
            },
        ] {
            evaluation.growth_goal = Some(invalid);
            assert!(SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).is_err());
        }
        evaluation.growth_goal = Some(goal);
        let valid = SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).unwrap();
        let mut tampered = serde_json::to_value(valid).unwrap();
        tampered["growth_goal"]["horizon_days"] = 0.0.into();
        let tampered: SealedEvaluationPolicyV1 = serde_json::from_value(tampered).unwrap();
        assert!(tampered.validate().is_err());
        evaluation.growth_objective = false;
        assert!(SealedEvaluationPolicyV1::from_evaluation(&evaluation, false, &stops).is_err());
    }
}

impl SealedEvaluationPolicyV1 {
    pub(super) fn from_evaluation(
        value: &EvaluationConfig,
        smc_gate_disabled: bool,
        adaptive_stops: &crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) -> anyhow::Result<Self> {
        let snapshot = Self {
            schema_version: 1,
            smc_gate_disabled,
            adaptive_stops_v1: adaptive_stops.clone(),
            symbol: value.symbol.clone(),
            account_currency: value.account_currency.clone(),
            initial_equity: value.initial_equity,
            max_hold_bars: value.max_hold_bars,
            trailing_enabled: value.trailing_enabled,
            trailing_atr_multiplier: value.trailing_atr_multiplier,
            trailing_be_trigger_r: value.trailing_be_trigger_r,
            trailing_min_lock_pips: value.trailing_min_lock_pips,
            pip_value: value.pip_value,
            spread_pips: value.spread_pips,
            commission_per_trade: value.commission_per_trade,
            pip_value_per_lot: value.pip_value_per_lot,
            swap_long_pips_per_day: value.swap_long_pips_per_day,
            swap_short_pips_per_day: value.swap_short_pips_per_day,
            pnl_conversion_fee_rate: value.pnl_conversion_fee_rate,
            kill_zones_enabled: value.kill_zones_enabled,
            session_spread_pips: value.session_spread_pips,
            risk_per_trade_min: value.risk_per_trade_min,
            risk_per_trade_max: value.risk_per_trade_max,
            high_quality_confidence: value.high_quality_confidence,
            smc_gate_threshold: value.smc_gate_threshold,
            smc_weight_ob: value.smc_weight_ob,
            smc_weight_fvg: value.smc_weight_fvg,
            smc_weight_liq: value.smc_weight_liq,
            smc_weight_mtf: value.smc_weight_mtf,
            smc_weight_premium: value.smc_weight_premium,
            smc_weight_inducement: value.smc_weight_inducement,
            smc_weight_bos: value.smc_weight_bos,
            smc_weight_choch: value.smc_weight_choch,
            smc_weight_eqh: value.smc_weight_eqh,
            smc_weight_eql: value.smc_weight_eql,
            smc_weight_displacement: value.smc_weight_displacement,
            growth_objective: value.growth_objective,
            growth_goal: value.growth_goal,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub(super) fn to_evaluation(&self) -> EvaluationConfig {
        EvaluationConfig {
            symbol: self.symbol.clone(),
            account_currency: self.account_currency.clone(),
            initial_equity: self.initial_equity,
            max_hold_bars: self.max_hold_bars,
            trailing_enabled: self.trailing_enabled,
            trailing_atr_multiplier: self.trailing_atr_multiplier,
            trailing_be_trigger_r: self.trailing_be_trigger_r,
            trailing_min_lock_pips: self.trailing_min_lock_pips,
            pip_value: self.pip_value,
            spread_pips: self.spread_pips,
            commission_per_trade: self.commission_per_trade,
            pip_value_per_lot: self.pip_value_per_lot,
            swap_long_pips_per_day: self.swap_long_pips_per_day,
            swap_short_pips_per_day: self.swap_short_pips_per_day,
            pnl_conversion_fee_rate: self.pnl_conversion_fee_rate,
            kill_zones_enabled: self.kill_zones_enabled,
            session_spread_pips: self.session_spread_pips,
            risk_per_trade_min: self.risk_per_trade_min,
            risk_per_trade_max: self.risk_per_trade_max,
            high_quality_confidence: self.high_quality_confidence,
            smc_gate_threshold: self.smc_gate_threshold,
            smc_weight_ob: self.smc_weight_ob,
            smc_weight_fvg: self.smc_weight_fvg,
            smc_weight_liq: self.smc_weight_liq,
            smc_weight_mtf: self.smc_weight_mtf,
            smc_weight_premium: self.smc_weight_premium,
            smc_weight_inducement: self.smc_weight_inducement,
            smc_weight_bos: self.smc_weight_bos,
            smc_weight_choch: self.smc_weight_choch,
            smc_weight_eqh: self.smc_weight_eqh,
            smc_weight_eql: self.smc_weight_eql,
            smc_weight_displacement: self.smc_weight_displacement,
            growth_objective: self.growth_objective,
            growth_goal: self.growth_goal,
        }
    }

    pub(super) fn smc_gate_disabled(&self) -> bool {
        self.smc_gate_disabled
    }
    pub(super) fn symbol(&self) -> &str {
        &self.symbol
    }

    pub(super) fn adaptive_stops(&self) -> &crate::stop_target::ResolvedAdaptiveStopsPolicyV1 {
        &self.adaptive_stops_v1
    }

    pub(super) fn set_final_gate(&mut self, threshold: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            threshold.is_finite() && threshold >= 0.0,
            "final effective SMC gate is not finite and non-negative"
        );
        self.smc_gate_threshold = threshold;
        self.validate()
    }

    pub(super) fn validate_against_policy(
        &self,
        policy: &LiveTradingPolicyV1,
    ) -> anyhow::Result<()> {
        self.validate()?;
        anyhow::ensure!(
            self.trailing_enabled == policy.trailing_enabled
                && self.trailing_atr_multiplier == policy.trailing_stop_multiplier
                && self.trailing_be_trigger_r == policy.trailing_be_trigger_r
                && self.trailing_min_lock_pips == policy.trailing_min_lock_pips
                && self.kill_zones_enabled == policy.kill_zones_enabled
                && self.spread_pips == policy.baseline_spread_pips
                && self.session_spread_pips == policy.session_spread_pips,
            "sealed evaluation policy disagrees with the persisted exit/session policy"
        );
        Ok(())
    }

    fn validate(&self) -> anyhow::Result<()> {
        self.adaptive_stops_v1.validate()?;
        anyhow::ensure!(
            self.schema_version == 1,
            "unsupported sealed evaluation schema"
        );
        if let Some(goal) = self.growth_goal {
            anyhow::ensure!(
                self.growth_objective,
                "sealed growth goal requires the growth objective"
            );
            goal.validate().map_err(anyhow::Error::msg)?;
        }
        anyhow::ensure!(
            !self.symbol.trim().is_empty()
                && self.symbol.trim() == self.symbol
                && !self.symbol.chars().any(char::is_control)
                && !self.account_currency.trim().is_empty()
                && self.account_currency.trim() == self.account_currency
                && !self.account_currency.chars().any(char::is_control),
            "sealed evaluation has an invalid symbol or account currency"
        );
        for (name, value) in [
            ("initial_equity", self.initial_equity),
            ("trailing_atr_multiplier", self.trailing_atr_multiplier),
            ("trailing_be_trigger_r", self.trailing_be_trigger_r),
            ("trailing_min_lock_pips", self.trailing_min_lock_pips),
            ("pip_value", self.pip_value),
            ("spread_pips", self.spread_pips),
            ("commission_per_trade", self.commission_per_trade),
            ("pip_value_per_lot", self.pip_value_per_lot),
            ("swap_long_pips_per_day", self.swap_long_pips_per_day),
            ("swap_short_pips_per_day", self.swap_short_pips_per_day),
            ("pnl_conversion_fee_rate", self.pnl_conversion_fee_rate),
            ("risk_per_trade_min", self.risk_per_trade_min),
            ("risk_per_trade_max", self.risk_per_trade_max),
            ("high_quality_confidence", self.high_quality_confidence),
            ("smc_gate_threshold", self.smc_gate_threshold),
            ("smc_weight_ob", self.smc_weight_ob),
            ("smc_weight_fvg", self.smc_weight_fvg),
            ("smc_weight_liq", self.smc_weight_liq),
            ("smc_weight_mtf", self.smc_weight_mtf),
            ("smc_weight_premium", self.smc_weight_premium),
            ("smc_weight_inducement", self.smc_weight_inducement),
            ("smc_weight_bos", self.smc_weight_bos),
            ("smc_weight_choch", self.smc_weight_choch),
            ("smc_weight_eqh", self.smc_weight_eqh),
            ("smc_weight_eql", self.smc_weight_eql),
            ("smc_weight_displacement", self.smc_weight_displacement),
        ] {
            anyhow::ensure!(value.is_finite(), "sealed evaluation {name} is not finite");
        }
        anyhow::ensure!(
            self.initial_equity > 0.0 && self.pip_value > 0.0 && self.pip_value_per_lot > 0.0,
            "sealed evaluation needs positive equity, pip size and pip value"
        );
        anyhow::ensure!(
            self.risk_per_trade_min >= 0.0
                && self.risk_per_trade_max >= self.risk_per_trade_min
                && self.risk_per_trade_max <= 1.0
                && self.high_quality_confidence > 0.0
                && self.high_quality_confidence <= 1.0,
            "sealed evaluation has invalid risk/confidence geometry"
        );
        anyhow::ensure!(
            self.spread_pips >= 0.0
                && self.commission_per_trade >= 0.0
                && (0.0..=1.0).contains(&self.pnl_conversion_fee_rate)
                && self.smc_gate_threshold >= 0.0
                && self.trailing_atr_multiplier >= 0.0
                && self.trailing_be_trigger_r >= 0.0
                && self.trailing_min_lock_pips >= 0.0,
            "sealed evaluation has negative cost/gate/exit geometry"
        );
        anyhow::ensure!(
            !self.trailing_enabled
                || (self.trailing_atr_multiplier > 0.0 && self.trailing_be_trigger_r > 0.0),
            "sealed evaluation enables trailing without positive geometry"
        );
        if let Some(curve) = self.session_spread_pips {
            anyhow::ensure!(
                curve.iter().all(|value| value.is_finite() && *value >= 0.0),
                "sealed evaluation has an invalid session spread curve"
            );
        }
        Ok(())
    }
}
