//! Rolling five-UTC-weekday research objective, opt-in for new runs.
//! Entries come from the same cost-charged CPU execution as the eleven metrics.
use chrono::{Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};

pub const WEEKLY_CADENCE_CONTRACT_SCHEMA_V1: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeeklyCadenceCalendarV1 {
    /// Observed Sunday FX entries cover the following Monday if it is inside
    /// the evaluated interval. Sunday and Monday never cover two separate days.
    UtcWeekdaysSundayToMonday,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeeklyCadenceContractV1 {
    pub schema_version: u16,
    pub calendar: WeeklyCadenceCalendarV1,
    pub evaluation_start_ms: i64,
    pub evaluation_end_ms: i64,
    pub trading_days_per_week: u64,
    pub trading_days_per_month: f64,
    pub min_trades_per_week: f64,
}
impl WeeklyCadenceContractV1 {
    pub fn new(start: i64, end: i64, trading_days_per_month: f64) -> anyhow::Result<Self> {
        let value = Self {
            schema_version: WEEKLY_CADENCE_CONTRACT_SCHEMA_V1,
            calendar: WeeklyCadenceCalendarV1::UtcWeekdaysSundayToMonday,
            evaluation_start_ms: start,
            evaluation_end_ms: end,
            trading_days_per_week: 5,
            trading_days_per_month,
            min_trades_per_week: 1.0,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == 1
                && self.trading_days_per_week == 5
                && self.min_trades_per_week == 1.0,
            "unsupported rolling weekly contract"
        );
        anyhow::ensure!(
            self.evaluation_start_ms >= 0 && self.evaluation_end_ms > self.evaluation_start_ms,
            "weekly interval must be ordered and nonnegative"
        );
        anyhow::ensure!(
            self.trading_days_per_month.is_finite() && self.trading_days_per_month >= 1.0,
            "invalid trading month convention"
        );
        Ok(())
    }
}

/// Additional active floors remain binding, including stricter monthly activity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeeklySearchLimitsV1 {
    pub initial_equity: f64,
    /// Resolved account-risk band; cash risk supplies the zero-floor expectancy scale.
    pub per_trade_risk_band: (f64, f64),
    pub max_drawdown: f64,
    pub max_daily_drawdown: f64,
    pub min_trades_per_month: f64,
    pub min_net_profit: f64,
    pub min_net_expectancy: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
}
impl WeeklySearchLimitsV1 {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.initial_equity.is_finite() && self.initial_equity > 0.0,
            "weekly objective requires positive initial equity"
        );
        let (risk_min, risk_max) = self.per_trade_risk_band;
        anyhow::ensure!(
            risk_min.is_finite()
                && risk_max.is_finite()
                && risk_min >= 0.0
                && risk_max > 0.0
                && risk_max >= risk_min
                && risk_max <= 1.0,
            "weekly objective requires a valid account-risk band"
        );
        anyhow::ensure!(
            [self.max_drawdown, self.max_daily_drawdown]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0),
            "weekly objective requires explicit positive drawdown limits"
        );
        anyhow::ensure!(
            [
                self.min_trades_per_month,
                self.min_net_profit,
                self.min_net_expectancy,
                self.min_win_rate,
                self.min_profit_factor
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
                && self.min_win_rate <= 1.0,
            "weekly objective contains invalid acceptance floors"
        );
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeeklySearchObjectiveV1 {
    pub cadence: WeeklyCadenceContractV1,
    pub limits: WeeklySearchLimitsV1,
}
impl WeeklySearchObjectiveV1 {
    /// One context check for generation evaluation and frozen-policy replay.
    pub fn checked_for_evaluation(
        self,
        equity: f64,
        growth_objective: bool,
        risk_band: (f64, f64),
    ) -> anyhow::Result<Self> {
        self.validate()?;
        anyhow::ensure!(
            !growth_objective
                && self.limits.initial_equity == equity
                && self.limits.per_trade_risk_band == risk_band,
            "weekly objective must match its PropFirm simulation equity"
        );
        Ok(self)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.cadence.validate()?;
        self.limits.validate()?;
        anyhow::ensure!(
            crate::quality::evaluation_weekdays(
                self.cadence.evaluation_start_ms,
                self.cadence.evaluation_end_ms
            )? >= 5,
            "weekly search needs at least five UTC weekdays of evidence"
        );
        Ok(())
    }
    /// Feasible fitness is positive. Infeasible fitness is minus the sum of
    /// dimensionless deficits; exact ties retain the original objective in the
    /// GA caller. This feasibility layer adds no trade-count bonus after the
    /// floors; the existing objective still supplies its sample-confidence terms.
    pub fn fitness(&self, m: &[f64; 11], report: &WeeklyRollingCadenceReportV1, old: f64) -> f64 {
        if m.iter().any(|v| !v.is_finite()) || !old.is_finite() {
            return f64::NEG_INFINITY;
        }
        let l = self.limits;
        let deficit = |observed: f64, floor: f64, scale: f64| (floor - observed).max(0.0) / scale;
        let expectancy_scale = if l.min_net_expectancy > 0.0 {
            l.min_net_expectancy
        } else {
            l.initial_equity * l.per_trade_risk_band.1
        };
        // The operator's minimum is one trade per rolling five trading days:
        // cadence is the PRIMARY feasibility tier. A weekly-valid but losing
        // candidate is still infeasible (negative fitness), but it must outrank
        // a sparse historically-profitable genome so evolution repairs cadence
        // before optimising net/risk inside the cadence-valid tier.
        let cadence_ratio = report.violation / report.trading_days.saturating_sub(4).max(1) as f64;
        let deficits = deficit(m[0], l.min_net_profit, l.initial_equity)
            + deficit(m[6], l.min_net_expectancy, expectancy_scale)
            + deficit(l.max_drawdown, m[3], l.max_drawdown)
            + deficit(l.max_daily_drawdown, m[10], l.max_daily_drawdown)
            + deficit(
                report.trades_per_month,
                l.min_trades_per_month,
                l.min_trades_per_month.max(1.0),
            )
            + deficit(m[4], l.min_win_rate, 1.0)
            + deficit(m[5], l.min_profit_factor, l.min_profit_factor.max(1.0))
            // Losses are graded even when the configured floor is zero: a
            // cadence-valid losing genome must rank below a smaller loss and a
            // high-frequency losing churn must never look like progress.
            + (-m[0]).max(0.0) / l.initial_equity
            + (-m[6]).max(0.0) / expectancy_scale;
        let feasible = cadence_ratio == 0.0
            && report.feasible()
            && deficits == 0.0
            && m[0] > 0.0
            && m[6] > 0.0
            && m[3] < l.max_drawdown
            && m[10] < l.max_daily_drawdown;
        if feasible {
            1.0 + old.atan() / std::f64::consts::PI
        } else {
            -(1_000.0 * (1.0 + cadence_ratio) + deficits)
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeeklyRollingCadenceReportV1 {
    pub trading_days: u64,
    pub trades: usize,
    /// All weekend fills retain costs/counts; only in-interval Sunday->Monday
    /// mappings can cover a weekday. This is a sealed research calendar.
    pub sunday_entries_credited: usize,
    /// Total actual Saturday/Sunday fills, including credited Sunday entries.
    pub weekend_entries: usize,
    pub trades_per_month: f64,
    pub longest_silent_trading_days: u64,
    pub violating_windows: u64,
    pub violation: f64,
    pub clustered: bool,
}
impl WeeklyRollingCadenceReportV1 {
    pub fn feasible(&self) -> bool {
        self.violating_windows == 0 && self.trading_days >= 5
    }
}
pub fn evaluate_rolling_weekly_cadence(
    contract: &WeeklyCadenceContractV1,
    entries: &[i64],
) -> anyhow::Result<WeeklyRollingCadenceReportV1> {
    contract.validate()?;
    let date = |ms| {
        Utc.timestamp_millis_opt(ms)
            .single()
            .map(|ts| ts.date_naive())
            .ok_or_else(|| anyhow::anyhow!("timestamp outside UTC calendar"))
    };
    let start = date(contract.evaluation_start_ms)?;
    let end = date(contract.evaluation_end_ms)?;
    let mut days = Vec::new();
    let mut day = start;
    while day <= end {
        if day.weekday().num_days_from_monday() < 5 {
            days.push(day);
        }
        if day == end {
            break;
        }
        day = day
            .succ_opt()
            .ok_or_else(|| anyhow::anyhow!("UTC calendar overflow"))?;
    }
    anyhow::ensure!(!days.is_empty(), "weekly interval contains no UTC weekday");
    let mut presence = vec![false; days.len()];
    let mut weekend_entries = 0;
    let mut sunday_entries_credited = 0;
    for &ms in entries {
        anyhow::ensure!(
            ms >= contract.evaluation_start_ms && ms <= contract.evaluation_end_ms,
            "trade entry outside weekly evaluation interval"
        );
        let mut day = date(ms)?;
        let weekday = day.weekday().num_days_from_monday();
        if weekday >= 5 {
            weekend_entries += 1;
            if weekday == 5 {
                continue;
            }
            day = day
                .succ_opt()
                .ok_or_else(|| anyhow::anyhow!("UTC calendar overflow"))?;
            if day > end {
                continue;
            }
            sunday_entries_credited += 1;
        }
        presence[days
            .binary_search(&day)
            .expect("in-range weekday was enumerated")] = true;
    }
    let mut silent = 0u64;
    let mut longest = 0u64;
    let mut violating_windows = 0u64;
    for present in &presence {
        silent = if *present { 0 } else { silent + 1 };
        longest = longest.max(silent);
        if silent >= 5 {
            violating_windows += 1;
        }
    }
    let weekdays = crate::quality::evaluation_weekdays(
        contract.evaluation_start_ms,
        contract.evaluation_end_ms,
    )?;
    let trades_per_month = entries.len() as f64 * contract.trading_days_per_month / weekdays as f64;
    Ok(WeeklyRollingCadenceReportV1 {
        trading_days: days.len() as u64,
        trades: entries.len(),
        weekend_entries,
        sunday_entries_credited,
        trades_per_month,
        longest_silent_trading_days: longest,
        violating_windows,
        violation: violating_windows as f64,
        clustered: violating_windows > 0
            && entries.len().saturating_sub(weekend_entries) + sunday_entries_credited
                >= days.len().div_ceil(5),
    })
}

#[cfg(test)]
pub(crate) fn test_weekly_objective() -> WeeklySearchObjectiveV1 {
    WeeklySearchObjectiveV1 {
        cadence: WeeklyCadenceContractV1::new(1_767_571_200_000, 1_769_731_200_000, 21.0).unwrap(),
        limits: WeeklySearchLimitsV1 {
            initial_equity: 10_000.0,
            per_trade_risk_band: (0.005, 0.03),
            max_drawdown: 0.1,
            max_daily_drawdown: 0.05,
            min_trades_per_month: 0.0,
            min_net_profit: 0.0,
            min_net_expectancy: 0.0,
            min_win_rate: 0.0,
            min_profit_factor: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ms(day: i64) -> i64 {
        Utc.with_ymd_and_hms(2026, 1, 5, 0, 0, 0)
            .unwrap()
            .timestamp_millis()
            + day * 86_400_000
    }
    fn report(entries: &[i64], end: i64) -> WeeklyRollingCadenceReportV1 {
        evaluate_rolling_weekly_cadence(
            &WeeklyCadenceContractV1::new(ms(0), ms(end), 21.0).unwrap(),
            &entries.iter().map(|d| ms(*d)).collect::<Vec<_>>(),
        )
        .unwrap()
    }
    #[test]
    fn rolling_weekly_endpoint_oracle_includes_all_inactive_days() {
        let endpoints = report(&[0, 11], 11);
        assert_eq!(
            (
                endpoints.trading_days,
                endpoints.longest_silent_trading_days,
                endpoints.violating_windows
            ),
            (10, 8, 4)
        );
        assert!(!endpoints.feasible());
        let regular = report(&[0, 7], 11);
        assert_eq!(regular.longest_silent_trading_days, 4);
        assert!(regular.feasible());
        assert_eq!(
            regular.trades_per_month.to_bits(),
            endpoints.trades_per_month.to_bits()
        );
        assert_eq!(report(&[], 11).violating_windows, 6);
        assert_eq!(report(&[0], 11).violating_windows, 5);
    }
    #[test]
    fn rolling_weekly_clusters_and_longer_gaps_are_worse_without_churn_bonus() {
        let regular = report(&[0, 7, 14, 21], 25);
        let clustered = report(&[0, 0, 0, 0], 25);
        assert!(regular.feasible());
        assert!(!clustered.feasible() && clustered.clustered);
        let mut objective = test_weekly_objective();
        objective.cadence = WeeklyCadenceContractV1::new(ms(0), ms(25), 21.0).unwrap();
        let good = [
            50.0, 1.0, 10_050.0, 0.01, 0.6, 1.2, 12.5, 0.5, 4.0, 0.5, 0.005,
        ];
        let fitness = objective.fitness(&good, &regular, 0.1);
        assert!(fitness > 0.0);
        assert!(objective.fitness(&good, &clustered, 0.1) < 0.0);
        let mut losing = good;
        losing[0] = -50.0;
        losing[6] = -12.5;
        let mut worse = losing;
        worse[0] = -500.0;
        worse[6] = -125.0;
        assert!(
            objective.fitness(&worse, &regular, 0.1) < objective.fitness(&losing, &regular, 0.1)
        );
        let mut churn = regular.clone();
        churn.trades *= 100;
        churn.trades_per_month *= 100.0;
        assert_eq!(
            objective.fitness(&good, &regular, 0.1),
            objective.fitness(&good, &churn, 0.1)
        );
        let short_gap = report(&[0, 7, 14], 25);
        let long_gap = report(&[0, 7], 25);
        assert!(
            objective.fitness(&good, &long_gap, 0.1) < objective.fitness(&good, &short_gap, 0.1)
        );
    }
    #[test]
    fn rolling_weekly_sunday_maps_to_monday_and_saturday_cannot_cover_a_weekday() {
        let r = report(&[5, 5], 11);
        assert_eq!(r.weekend_entries, 2);
        assert_eq!(r.trades, 2);
        assert_eq!(r.violating_windows, 6);
        let sunday = report(&[6], 11);
        let sunday_and_monday = report(&[6, 7], 11);
        assert_eq!(sunday.sunday_entries_credited, 1);
        assert_eq!(sunday.violating_windows, 1);
        assert_eq!(
            sunday.violating_windows,
            sunday_and_monday.violating_windows
        );
        let c = WeeklyCadenceContractV1::new(ms(5), ms(6), 21.0).unwrap();
        assert!(evaluate_rolling_weekly_cadence(&c, &[]).is_err());
        assert!(
            evaluate_rolling_weekly_cadence(
                &WeeklyCadenceContractV1::new(ms(0), ms(11), 21.0).unwrap(),
                &[ms(12)]
            )
            .is_err()
        );
        let mut o = test_weekly_objective();
        o.cadence = WeeklyCadenceContractV1::new(ms(0), ms(2), 21.0).unwrap();
        assert!(o.validate().is_err());
        o.cadence.min_trades_per_week = 2.0;
        assert!(o.validate().is_err());
    }
    #[test]
    fn rolling_weekly_reverses_the_actual_one_trade_scoring_refuge() {
        // Algebraic counterexample in the actual legacy scorer, not a market replay.
        let one = [
            50.0, 0.343, 10_050.0, 0.001, 1.0, 10.0, 50.0, 0.0, 1.0, 0.1, 0.001,
        ];
        let active = [
            1_000.0,
            1.0,
            11_000.0,
            0.02,
            0.6,
            1.5,
            1_000.0 / 2_400.0,
            0.0,
            2_400.0,
            0.5,
            0.01,
        ];
        let old_one = crate::scoring::ga_fitness(&one);
        let old_active = crate::scoring::ga_fitness(&active);
        assert!(old_one > old_active);
        let objective = test_weekly_objective();
        let refuge = report(&[0], 25);
        let entries = (0..2_400)
            .map(|i| [0, 7, 14, 21][i % 4])
            .collect::<Vec<_>>();
        let regular = report(&entries, 25);
        assert!(objective.fitness(&active, &regular, old_active) > 0.0);
        assert!(objective.fitness(&one, &refuge, old_one) < 0.0);
    }

    #[test]
    fn rolling_weekly_expectancy_deficit_has_its_own_scale_and_no_zero_step() {
        let r = report(&[0, 7, 14, 21], 25);
        let mut o = test_weekly_objective();
        o.limits.min_net_expectancy = 0.5;
        let mut m = [
            50.0, 1.0, 10_050.0, 0.01, 0.6, 1.2, 0.49, 0.5, 4.0, 0.5, 0.005,
        ];
        assert!((o.fitness(&m, &r, 0.0) + 1_000.02).abs() < 1e-12);
        m[6] = -1e-9;
        let below = o.fitness(&m, &r, 0.0);
        m[6] = 1e-9;
        let above = o.fitness(&m, &r, 0.0);
        assert!((above - below).abs() < 1e-7);
        m[6] = 1.0;
        m[3] = o.limits.max_drawdown;
        assert!(
            o.fitness(&m, &r, 0.0) <= 0.0,
            "current risk owner rejects equality"
        );
        m[3] = 0.01;
        m[0] = 0.0;
        assert!(
            o.fitness(&m, &r, 0.0) <= 0.0,
            "zero net never receives feasible tier"
        );
    }

    #[test]
    fn rolling_weekly_cadence_is_the_primary_infeasibility_tier() {
        let objective = test_weekly_objective();
        let regular = report(&[0, 7, 14, 21], 25);
        let sparse = report(&[0], 25);
        let good = [
            50.0, 1.0, 10_050.0, 0.01, 0.6, 1.2, 12.5, 0.5, 4.0, 0.5, 0.005,
        ];
        let mut weekly_losing = good;
        weekly_losing[0] = -50.0;
        weekly_losing[6] = -5.0;
        let feasible = objective.fitness(&good, &regular, 0.0);
        let losing = objective.fitness(&weekly_losing, &regular, 0.0);
        let sparse_profit = objective.fitness(&good, &sparse, 0.0);
        assert!(feasible > 0.0, "weekly-valid positive net is feasible");
        assert!(losing < 0.0, "weekly-valid losing churn is never feasible");
        assert!(sparse_profit < 0.0);
        assert!(
            losing > sparse_profit,
            "cadence is primary: weekly-valid losing outranks sparse profit"
        );
        let mut heavier_loss = weekly_losing;
        heavier_loss[0] = -500.0;
        heavier_loss[6] = -50.0;
        assert!(objective.fitness(&heavier_loss, &regular, 0.0) < losing);
    }

    #[test]
    fn rolling_weekly_honors_stricter_monthly_and_risk_limits() {
        let r = report(&[0, 7, 14, 21], 25);
        let mut o = test_weekly_objective();
        let good = [
            50.0, 1.0, 10_050.0, 0.01, 0.6, 1.2, 12.5, 0.5, 4.0, 0.5, 0.005,
        ];
        assert!(o.fitness(&good, &r, 0.0) > 0.0);
        o.limits.min_trades_per_month = 20.0;
        assert!(o.fitness(&good, &r, 0.0) < 0.0);
        o.limits.min_trades_per_month = 0.0;
        let mut risky = good;
        risky[3] = 0.2;
        let mut riskier = risky;
        riskier[3] = 0.4;
        assert!(o.fitness(&riskier, &r, 0.0) < o.fitness(&risky, &r, 0.0));
        risky[1] = f64::NEG_INFINITY;
        assert_eq!(o.fitness(&risky, &r, 0.0), f64::NEG_INFINITY);
    }
}
