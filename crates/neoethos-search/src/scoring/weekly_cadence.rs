//! Versioned weekly-cadence feasibility (2026-10-21 operator requirement).
//!
//! The new requirement is at least ONE trade in every five trading days.
//! An average trades/month count cannot certify that: a strategy may cluster
//! trades in one week and stay silent for years while its monthly average looks
//! acceptable. This module evaluates the full evaluation interval, counts
//! inclusive UTC weekdays with the same convention as the production quality
//! analyzer, and turns every empty ISO week into a violation that grows with the
//! length of the silent run. It is research-only ranking evidence: it does not
//! replace the quality gate, the frozen historical daily-profile thresholds, or
//! any risk/cost gate.
use chrono::{Datelike, TimeZone, Utc};

/// Frozen version stamp for the weekly-cadence contract.
pub const WEEKLY_CADENCE_CONTRACT_SCHEMA_V1: u16 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct WeeklyCadenceContractV1 {
    pub schema_version: u16,
    pub evaluation_start_ms: i64,
    pub evaluation_end_ms: i64,
    /// Trading days in a weekly window (five: Mon-Fri).
    pub trading_days_per_week: u64,
    /// Month convention shared with the quality analyzer (21.0 default).
    pub trading_days_per_month: f64,
    pub min_trades_per_week: f64,
}

impl WeeklyCadenceContractV1 {
    pub fn new(
        evaluation_start_ms: i64,
        evaluation_end_ms: i64,
        trading_days_per_month: f64,
    ) -> anyhow::Result<Self> {
        let contract = Self {
            schema_version: WEEKLY_CADENCE_CONTRACT_SCHEMA_V1,
            evaluation_start_ms,
            evaluation_end_ms,
            trading_days_per_week: 5,
            trading_days_per_month,
            min_trades_per_week: 1.0,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == WEEKLY_CADENCE_CONTRACT_SCHEMA_V1,
            "unsupported weekly-cadence contract schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.evaluation_start_ms >= 0 && self.evaluation_end_ms > self.evaluation_start_ms,
            "weekly-cadence interval must be ordered and nonnegative"
        );
        anyhow::ensure!(
            self.trading_days_per_week == 5,
            "weekly-cadence windows are five trading days"
        );
        anyhow::ensure!(
            self.trading_days_per_month.is_finite() && self.trading_days_per_month >= 1.0,
            "weekly-cadence trading-days-per-month must be finite and at least one"
        );
        anyhow::ensure!(
            self.min_trades_per_week.is_finite() && self.min_trades_per_week >= 1.0,
            "weekly-cadence minimum is at least one trade per week"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeeklyCadenceReportV1 {
    pub expected_weeks: u64,
    pub weeks_with_trades: u64,
    pub trades: usize,
    pub inclusive_weekdays: u64,
    pub trades_per_month: f64,
    pub longest_silent_weeks: u64,
    /// Nonnegative; zero only when every expected week has at least the minimum.
    pub violation: f64,
    pub clustered: bool,
}

impl WeeklyCadenceReportV1 {
    pub fn feasible(&self) -> bool {
        self.violation == 0.0
    }
}

fn iso_week_key(ms: i64) -> anyhow::Result<(i32, u32)> {
    let ts = Utc
        .timestamp_millis_opt(ms)
        .single()
        .ok_or_else(|| anyhow::anyhow!("timestamp outside the UTC calendar: {ms}"))?;
    let iso = ts.iso_week();
    Ok((iso.year(), iso.week()))
}

fn inclusive_weekdays(contract: &WeeklyCadenceContractV1) -> anyhow::Result<u64> {
    let start = Utc
        .timestamp_millis_opt(contract.evaluation_start_ms)
        .single()
        .ok_or_else(|| anyhow::anyhow!("weekly-cadence start outside the UTC calendar"))?
        .date_naive();
    let end = Utc
        .timestamp_millis_opt(contract.evaluation_end_ms)
        .single()
        .ok_or_else(|| anyhow::anyhow!("weekly-cadence end outside the UTC calendar"))?
        .date_naive();
    let days = (end - start).num_days() as u64 + 1;
    let first = u64::from(start.weekday().num_days_from_monday());
    Ok((days / 7) * 5
        + (0..days % 7)
            .filter(|offset| (first + offset) % 7 < 5)
            .count() as u64)
}

/// Evaluate explicit trade ENTRY timestamps against the weekly contract.
pub fn evaluate_weekly_cadence(
    contract: &WeeklyCadenceContractV1,
    trade_entry_timestamps_ms: &[i64],
) -> anyhow::Result<WeeklyCadenceReportV1> {
    contract.validate()?;
    let mut per_week: std::collections::BTreeMap<(i32, u32), usize> =
        std::collections::BTreeMap::new();
    for &ms in trade_entry_timestamps_ms {
        anyhow::ensure!(
            ms >= contract.evaluation_start_ms && ms <= contract.evaluation_end_ms,
            "trade entry {ms} lies outside the weekly evaluation interval"
        );
        *per_week.entry(iso_week_key(ms)?).or_insert(0) += 1;
    }
    // Expected ISO weeks are exactly the weeks that contain at least one weekDAY
    // of the interval; a partial week still needs its minimum.
    let mut expected: Vec<(i32, u32)> = Vec::new();
    let mut cursor = contract.evaluation_start_ms;
    while cursor <= contract.evaluation_end_ms {
        let key = iso_week_key(cursor)?;
        if expected.last().copied() != Some(key) {
            expected.push(key);
        }
        cursor += 86_400_000;
    }
    let mut violation = 0.0_f64;
    let mut silent_run = 0u64;
    let mut longest_silent_weeks = 0u64;
    let mut weeks_with_trades = 0u64;
    for key in &expected {
        let count = per_week.get(key).copied().unwrap_or(0);
        if count > 0 {
            weeks_with_trades += 1;
            silent_run = 0;
        } else {
            silent_run += 1;
            longest_silent_weeks = longest_silent_weeks.max(silent_run);
            // Strictly increasing in the length of the silent run.
            violation += (silent_run as f64) * ((silent_run as f64) + 1.0) / 2.0;
        }
    }
    let inclusive_weekdays = inclusive_weekdays(contract)?;
    let months = inclusive_weekdays as f64 / contract.trading_days_per_month.max(1.0);
    let trades_per_month = if months > 0.0 {
        trade_entry_timestamps_ms.len() as f64 / months
    } else {
        0.0
    };
    let clustered = !expected.is_empty()
        && weeks_with_trades < expected.len() as u64
        && trade_entry_timestamps_ms.len() as u64 >= expected.len() as u64;
    Ok(WeeklyCadenceReportV1 {
        expected_weeks: expected.len() as u64,
        weeks_with_trades,
        trades: trade_entry_timestamps_ms.len(),
        inclusive_weekdays,
        trades_per_month,
        longest_silent_weeks,
        violation,
        clustered,
    })
}

/// Feasible candidates rank above every infeasible one; an infeasible candidate
/// is graded by its normalized constraint violations so evolution still has an
/// ordering instead of an all-infeasible dead population. Losing churn never
/// earns the feasible tier.
pub fn weekly_feasibility_score(
    net_after_costs: f64,
    risk_ok: bool,
    report: &WeeklyCadenceReportV1,
) -> f64 {
    let net_ok = net_after_costs.is_finite() && net_after_costs > 0.0;
    if report.feasible() && net_ok && risk_ok {
        return 1_000_000.0 + net_after_costs.min(1_000.0);
    }
    let cadence = report.violation;
    let net_penalty = if net_ok { 0.0 } else { 1.0 };
    let risk_penalty = if risk_ok { 0.0 } else { 1.0 };
    -(1.0 + cadence + net_penalty + risk_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(day: i64) -> i64 {
        1_600_041_600_000 + day * 86_400_000
    }

    fn contract(weeks: i64) -> WeeklyCadenceContractV1 {
        WeeklyCadenceContractV1::new(ms(0), ms(weeks * 7 - 1), 21.0).expect("weekly cadence contract")
    }

    #[test]
    fn one_trade_every_week_is_feasible_and_an_average_does_not_certify_it() {
        let c = contract(40);
        let regular: Vec<i64> = (0..40).map(|w| ms(w * 7 + 1)).collect();
        let report = evaluate_weekly_cadence(&c, &regular).expect("regular cadence");
        assert!(report.feasible(), "regular weekly trades must be feasible: {report:?}");
        // The same trade COUNT clustered in one week: the monthly average is
        // identical, but the weekly requirement is violated.
        let clustered: Vec<i64> = (0..40).map(|i| ms(i % 5)).collect();
        let clustered_report = evaluate_weekly_cadence(&c, &clustered).expect("clustered cadence");
        assert!(!clustered_report.feasible());
        assert!(clustered_report.violation > report.violation);
        assert!(clustered_report.clustered);
        assert!(weekly_feasibility_score(100.0, true, &clustered_report)
            < weekly_feasibility_score(100.0, true, &report));
    }

    #[test]
    fn one_trade_refuge_and_losing_churn_never_reach_the_feasible_tier() {
        let c = contract(40);
        let regular: Vec<i64> = (0..40).map(|w| ms(w * 7 + 1)).collect();
        let regular_report = evaluate_weekly_cadence(&c, &regular).unwrap();
        let refuge = evaluate_weekly_cadence(&c, &[ms(3)]).unwrap();
        assert!(!refuge.feasible());
        assert_eq!(refuge.longest_silent_weeks, 39);
        assert!(weekly_feasibility_score(1_000.0, true, &refuge)
            < weekly_feasibility_score(1.0, true, &regular_report));
        // Weekly-regular but net-negative churn is infeasible regardless of cadence.
        assert!(weekly_feasibility_score(-50.0, true, &regular_report)
            < weekly_feasibility_score(1.0, true, &regular_report));
    }

    #[test]
    fn longer_silent_runs_are_monotonically_worse() {
        let c = contract(40);
        let mut previous = -1.0;
        for silent_weeks in 1..=4 {
            // Trade on week 0, stay silent for `silent_weeks` weeks, then regular.
            let mut trades: Vec<i64> = vec![ms(0)];
            trades.extend(((silent_weeks + 1)..40).map(|w| ms(w * 7)));
            let report = evaluate_weekly_cadence(&c, &trades).unwrap();
            assert!(
                report.violation > previous,
                "violation must increase with the silent run: {report:?}"
            );
            previous = report.violation;
        }
        assert!(previous > 0.0);
    }
}