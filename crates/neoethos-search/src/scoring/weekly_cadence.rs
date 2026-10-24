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
/// Version of the SCORING semantics (fitness arithmetic and ranking) bound into
/// the wire objective. 0 is reserved for historical objectives serialized before
/// this binding existed; they stay readable but are explicitly unbound and are
/// refused as current equivalence/replay.
pub const WEEKLY_SCORING_SEMANTICS_VERSION_V1: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeeklySearchObjectiveV1 {
    pub cadence: WeeklyCadenceContractV1,
    pub limits: WeeklySearchLimitsV1,
    /// 0 = legacy/unbound; new runs always bind the current scoring semantics.
    #[serde(default)]
    pub scoring_semantics_version: u16,
}
impl WeeklySearchObjectiveV1 {
    /// New objectives must bind the current scoring semantics explicitly.
    pub fn current(cadence: WeeklyCadenceContractV1, limits: WeeklySearchLimitsV1) -> Self {
        Self {
            cadence,
            limits,
            scoring_semantics_version: WEEKLY_SCORING_SEMANTICS_VERSION_V1,
        }
    }

    pub fn is_current_scoring_semantics(&self) -> bool {
        self.scoring_semantics_version == WEEKLY_SCORING_SEMANTICS_VERSION_V1
    }

    /// Wire validation accepts the historical unbound form; current-equivalence
    /// callers must use `validate_current`.
    pub fn validate_current(&self) -> anyhow::Result<()> {
        self.validate()?;
        anyhow::ensure!(
            self.is_current_scoring_semantics(),
            "legacy weekly objective has no scoring-semantics binding (version {}); it remains inspectable but is not admissible as current equivalence or replay",
            self.scoring_semantics_version
        );
        Ok(())
    }

    /// One context check for generation evaluation and frozen-policy replay.
    pub fn checked_for_evaluation(
        self,
        equity: f64,
        growth_objective: bool,
        risk_band: (f64, f64),
    ) -> anyhow::Result<Self> {
        self.validate_current()?;
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
            self.scoring_semantics_version <= WEEKLY_SCORING_SEMANTICS_VERSION_V1,
            "unsupported weekly scoring semantics version {}",
            self.scoring_semantics_version
        );
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
        // cadence is the PRIMARY feasibility tier and EACH violated window is a
        // full tier. A weekly-valid but losing candidate is still infeasible
        // (negative fitness), but it must outrank any candidate with even one
        // violated window so evolution closes the last gaps instead of trading
        // tiny deficit improvements against remaining inactivity.
        let cadence_tier = report.violating_windows as f64;
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
        let feasible = cadence_tier == 0.0
            && report.feasible()
            && deficits == 0.0
            && m[0] > 0.0
            && m[6] > 0.0
            && m[3] < l.max_drawdown
            && m[10] < l.max_daily_drawdown;
        if feasible {
            1.0 + old.atan() / std::f64::consts::PI
        } else {
            -(1_000.0 * (1.0 + cadence_tier) + deficits)
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

/// Research adaptation of nondominated sorting and crowding to the weekly
/// feasibility problem. This is not stock constrained-NSGA-II and it does not
/// claim to reproduce it or to guarantee a profitable genome. It follows the
/// published nondominated-ranking/crowding idea (Deb et al., IEEE TEC 2002,
/// doi:10.1109/4235.996017) only as an explicit small-population selection aid.
pub const WEEKLY_SELECTION_AXES_V1: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WeeklySelectionRankV1 {
    pub feasible: bool,
    /// Intentional per-candidate invalid observation; never accepted.
    pub invalid: bool,
    pub front_rank: u32,
    pub crowding: f64,
    pub bounded_distance: f64,
    pub fitness: f64,
    /// Full violation vector used for input permutation checks and unique-vector
    /// diversity; never a missing daily-DD zero because callers supply all 11
    /// metrics plus the same-execution report.
    pub axes: [f64; WEEKLY_SELECTION_AXES_V1],
}

impl WeeklySelectionRankV1 {
    /// Feasible first, then better fronts, then wider crowding, then the
    /// bounded smooth distance, then the original objective. Used by the
    /// bounded weekly archive only; no caller may relabel an infeasible rank.
    /// The SINGLE ordering owner. Greater always means better: feasible first,
    /// then better front, wider crowding, lower smooth distance, higher
    /// original financial fitness. Order/archive/survivor code must delegate
    /// here instead of mirroring comparator directions.
    pub fn preference_cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.invalid, other.invalid) {
            (false, true) => return std::cmp::Ordering::Greater,
            (true, false) => return std::cmp::Ordering::Less,
            (true, true) => return std::cmp::Ordering::Equal,
            (false, false) => {}
        }
        match (self.feasible, other.feasible) {
            (true, false) => return std::cmp::Ordering::Greater,
            (false, true) => return std::cmp::Ordering::Less,
            _ => {}
        }
        other
            .front_rank
            .cmp(&self.front_rank)
            .then_with(|| self.crowding.total_cmp(&other.crowding))
            .then_with(|| other.bounded_distance.total_cmp(&self.bounded_distance))
            .then_with(|| self.fitness.total_cmp(&other.fitness))
    }

    pub fn retention_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.preference_cmp(other)
    }
}

impl Eq for WeeklySelectionRankV1 {}

fn weekly_dominates(
    a: &[f64; WEEKLY_SELECTION_AXES_V1],
    b: &[f64; WEEKLY_SELECTION_AXES_V1],
) -> bool {
    let mut strictly_better = false;
    for axis in 0..WEEKLY_SELECTION_AXES_V1 {
        if a[axis] > b[axis] {
            return false;
        }
        if a[axis] < b[axis] {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Full observed metrics plus the same-execution cadence report. Non-finite
/// inputs fail closed; no missing daily drawdown is replaced by zero.
fn validate_weekly_report(report: &WeeklyRollingCadenceReportV1) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.trading_days >= 5
            && report.violating_windows <= report.trading_days
            && report.violation.is_finite()
            && report.trades_per_month.is_finite()
            && report.trades_per_month >= 0.0,
        "malformed weekly cadence report"
    );
    Ok(())
}

pub fn weekly_violation_axes(
    objective: &WeeklySearchObjectiveV1,
    metrics: &[f64; 11],
    report: &WeeklyRollingCadenceReportV1,
) -> anyhow::Result<[f64; WEEKLY_SELECTION_AXES_V1]> {
    objective.validate_current()?;
    anyhow::ensure!(
        metrics.iter().all(|value| value.is_finite())
            && report.violation.is_finite()
            && report.trades_per_month.is_finite()
            && report.trading_days > 4,
        "weekly selection requires finite observed metrics and a finite cadence report"
    );
    let limits = &objective.limits;
    let expectancy_scale = if limits.min_net_expectancy > 0.0 {
        limits.min_net_expectancy
    } else {
        limits.initial_equity * limits.per_trade_risk_band.1
    };
    Ok([
        report.violating_windows as f64,
        (limits.min_net_profit - metrics[0]).max(0.0) / limits.initial_equity,
        (limits.min_net_expectancy - metrics[6]).max(0.0) / expectancy_scale,
        (metrics[3] - limits.max_drawdown).max(0.0) / limits.max_drawdown,
        (metrics[10] - limits.max_daily_drawdown).max(0.0) / limits.max_daily_drawdown,
        (limits.min_trades_per_month - report.trades_per_month).max(0.0)
            / limits.min_trades_per_month.max(1.0),
        (limits.min_win_rate - metrics[4]).max(0.0),
        (limits.min_profit_factor - metrics[5]).max(0.0) / limits.min_profit_factor.max(1.0),
    ])
}

pub fn weekly_axes_are_feasible(
    objective: &WeeklySearchObjectiveV1,
    metrics: &[f64; 11],
    report: &WeeklyRollingCadenceReportV1,
    axes: &[f64; WEEKLY_SELECTION_AXES_V1],
) -> bool {
    axes.iter().all(|axis| *axis == 0.0)
        && report.feasible()
        && metrics[0] > 0.0
        && metrics[6] > 0.0
        && metrics[3] < objective.limits.max_drawdown
        && metrics[10] < objective.limits.max_daily_drawdown
}

/// Deterministic O(N^2)-time / O(N)-memory nondominated ranking with
/// per-front crowding. The explicit bound is the population, never the history.
pub fn weekly_selection_ranks(
    objective: &WeeklySearchObjectiveV1,
    metrics: &[[f64; 11]],
    reports: &[WeeklyRollingCadenceReportV1],
    fitness: &[f64],
) -> anyhow::Result<Vec<WeeklySelectionRankV1>> {
    anyhow::ensure!(
        metrics.len() == reports.len() && reports.len() == fitness.len(),
        "weekly selection requires aligned genes, metrics and same-execution cadence reports"
    );
    let n = metrics.len();
    let mut ranks = vec![
        WeeklySelectionRankV1 {
            feasible: false,
            invalid: false,
            front_rank: 0,
            crowding: 0.0,
            bounded_distance: 0.0,
            fitness: 0.0,
            axes: [0.0; WEEKLY_SELECTION_AXES_V1],
        };
        n
    ];
    let mut feasible = vec![false; n];
    for i in 0..n {
        validate_weekly_report(&reports[i])?;
        if !metrics[i].iter().all(|value| value.is_finite()) || !fitness[i].is_finite() {
            ranks[i].invalid = true;
            ranks[i].front_rank = u32::MAX;
            ranks[i].bounded_distance = f64::INFINITY;
            ranks[i].fitness = f64::NEG_INFINITY;
            ranks[i].axes = [f64::INFINITY; WEEKLY_SELECTION_AXES_V1];
            continue;
        }
        let axes = weekly_violation_axes(objective, &metrics[i], &reports[i])?;
        feasible[i] = weekly_axes_are_feasible(objective, &metrics[i], &reports[i], &axes);
        ranks[i].axes = axes;
        ranks[i].feasible = feasible[i];
        ranks[i].bounded_distance = axes.iter().map(|x| x / (1.0 + x)).sum();
        ranks[i].fitness = fitness[i];
    }
    let infeasible: Vec<usize> = (0..n).filter(|i| !feasible[*i]).collect();
    if infeasible.is_empty() {
        return Ok(ranks);
    }
    let m = infeasible.len();
    let mut dominated_by = vec![0usize; n];
    for &candidate in &infeasible {
        for &other in &infeasible {
            if other != candidate && weekly_dominates(&ranks[other].axes, &ranks[candidate].axes) {
                dominated_by[candidate] += 1;
            }
        }
    }
    let mut removed = vec![false; n];
    let mut front_rank = 1u32;
    loop {
        let front: Vec<usize> = infeasible
            .iter()
            .copied()
            .filter(|i| !removed[*i] && dominated_by[*i] == 0)
            .collect();
        if front.is_empty() {
            anyhow::ensure!(
                front_rank > 1 && removed.iter().filter(|v| **v).count() >= m,
                "weekly nondominated front unexpectedly empty"
            );
            break;
        }
        let positions: std::collections::HashMap<usize, usize> = front
            .iter()
            .enumerate()
            .map(|(position, index)| (*index, position))
            .collect();
        let mut crowding = vec![0.0; front.len()];
        if front.len() > 1 {
            for axis in 0..WEEKLY_SELECTION_AXES_V1 {
                let mut order = front.clone();
                order.sort_by(|&a, &b| {
                    ranks[a].axes[axis]
                        .total_cmp(&ranks[b].axes[axis])
                        .then_with(|| a.cmp(&b))
                });
                let span = ranks[order[order.len() - 1]].axes[axis] - ranks[order[0]].axes[axis];
                if !(span > 0.0) {
                    continue;
                }
                crowding[positions[&order[0]]] += f64::INFINITY;
                crowding[positions[&order[order.len() - 1]]] += f64::INFINITY;
                for pos in 1..order.len() - 1 {
                    let gap = ranks[order[pos + 1]].axes[axis] - ranks[order[pos - 1]].axes[axis];
                    crowding[positions[&order[pos]]] += gap / span;
                }
            }
        }
        for &index in &front {
            ranks[index].front_rank = front_rank;
            ranks[index].crowding = crowding[positions[&index]];
        }
        for &index in &front {
            removed[index] = true;
        }
        for &index in &front {
            for &other in &infeasible {
                if !removed[other] && weekly_dominates(&ranks[index].axes, &ranks[other].axes) {
                    dominated_by[other] = dominated_by[other].saturating_sub(1);
                }
            }
        }
        front_rank = front_rank
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("weekly front rank overflow"))?;
    }
    Ok(ranks)
}

pub fn weekly_selection_order(ranks: &[WeeklySelectionRankV1]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ranks.len()).collect();
    order.sort_by(|&a, &b| ranks[b].preference_cmp(&ranks[a]).then_with(|| a.cmp(&b)));
    order
}

/// Bounded scalar for existing tournament/parent selectors. It preserves the
/// order above; it is not a feasibility label and cannot erase a front because
/// survivors use the front-aware allocator below.
pub fn weekly_selection_scores(ranks: &[WeeklySelectionRankV1]) -> Vec<f64> {
    let order = weekly_selection_order(ranks);
    let mut scores = vec![0.0; ranks.len()];
    let total = order.len().max(1) as f64;
    for (position, index) in order.into_iter().enumerate() {
        scores[index] = 1.0 - position as f64 / total;
    }
    scores
}

/// NSGA-II-style environmental selection: feasible first, then front by front,
/// crowding-truncated. It retains complementary infeasible paths while capacity
/// remains; repeated identical vectors have zero crowding and cannot flood it.
pub fn weekly_select_survivors(ranks: &[WeeklySelectionRankV1], requested: usize) -> Vec<usize> {
    let requested = requested.min(ranks.len());
    if requested == 0 {
        return Vec::new();
    }
    let mut selected: Vec<usize> = Vec::with_capacity(requested);
    let mut seen_vectors: std::collections::HashSet<[u64; WEEKLY_SELECTION_AXES_V1]> =
        std::collections::HashSet::new();
    let vector_key = |rank: &WeeklySelectionRankV1| {
        let mut key = [0u64; WEEKLY_SELECTION_AXES_V1];
        for (slot, value) in rank.axes.iter().enumerate() {
            key[slot] = if *value == 0.0 { 0 } else { value.to_bits() };
        }
        key
    };
    // Pass 1: one representative per unique violation vector, feasible first
    // and then front by front. This is what stops a quiet-vector flood from
    // consuming every survivor slot.
    let mut feasible: Vec<usize> = (0..ranks.len())
        .filter(|i| !ranks[*i].invalid && ranks[*i].feasible)
        .collect();
    feasible.sort_by(|&a, &b| ranks[b].preference_cmp(&ranks[a]).then_with(|| a.cmp(&b)));
    for index in feasible {
        if selected.len() == requested {
            return selected;
        }
        // Every feasible candidate is retained in financial-objective order;
        // unique-vector diversity applies only to the infeasible exploration
        // section below.
        selected.push(index);
    }
    let max_front = ranks.iter().map(|rank| rank.front_rank).max().unwrap_or(0);
    let mut skipped: Vec<usize> = Vec::new();
    for front in 1..=max_front {
        let mut members: Vec<usize> = (0..ranks.len())
            .filter(|i| !ranks[*i].invalid && !ranks[*i].feasible && ranks[*i].front_rank == front)
            .collect();
        members.sort_by(|&a, &b| ranks[b].preference_cmp(&ranks[a]).then_with(|| a.cmp(&b)));
        for index in members {
            if selected.len() == requested {
                return selected;
            }
            if seen_vectors.insert(vector_key(&ranks[index])) {
                selected.push(index);
            } else {
                skipped.push(index);
            }
        }
    }
    // Pass 2: only after every complementary vector has a representative may
    // remaining capacity hold exact-duplicate genomes for reproducibility.
    for index in skipped {
        if selected.len() == requested {
            break;
        }
        selected.push(index);
    }
    selected
}

/// Shared synthetic cadence report for selection/archive/GA tests. These
/// window counts are hand-built, not replay-generated.
#[cfg(test)]
pub(crate) fn synthetic_weekly_report(windows: u64, trades: usize) -> WeeklyRollingCadenceReportV1 {
    WeeklyRollingCadenceReportV1 {
        trading_days: 2_219,
        trades,
        sunday_entries_credited: 0,
        weekend_entries: 0,
        trades_per_month: 21.0 * trades as f64 / 2_219.0,
        longest_silent_trading_days: if windows == 0 { 4 } else { 5 },
        violating_windows: windows,
        violation: windows as f64,
        clustered: windows > 0,
    }
}

/// Shared synthetic metric row for selection/archive/GA tests. Not a replay
/// observation and not a profitability claim; report windows are synthetic too.
#[cfg(test)]
pub(crate) fn synthetic_weekly_metrics(
    net: f64,
    drawdown: f64,
    daily_drawdown: f64,
    trades: usize,
) -> [f64; 11] {
    [
        net,
        1.0,
        10_000.0,
        drawdown,
        0.6,
        1.2,
        net / trades.max(1) as f64,
        0.0,
        trades as f64,
        0.0,
        daily_drawdown,
    ]
}

#[cfg(test)]
pub(crate) fn test_weekly_objective() -> WeeklySearchObjectiveV1 {
    WeeklySearchObjectiveV1::current(
        WeeklyCadenceContractV1::new(1_767_571_200_000, 1_769_731_200_000, 21.0).unwrap(),
        WeeklySearchLimitsV1 {
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
    )
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
        // One remaining violated window is a full tier above a cadence-valid
        // candidate with any deficits: closing the last gap is primary.
        let one_window = report(&[0, 7, 14], 25);
        assert!(one_window.violating_windows > 0);
        assert!(
            objective.fitness(&good, &one_window, 0.0)
                < objective.fitness(&heavier_loss, &regular, 0.0)
        );
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

    #[test]
    fn weekly_scoring_semantics_legacy_wire_is_inspectable_but_unbound() {
        let current = test_weekly_objective();
        assert_eq!(
            current.scoring_semantics_version,
            WEEKLY_SCORING_SEMANTICS_VERSION_V1
        );
        current.validate_current().unwrap();
        let good_report = report(&[0, 7, 14, 21], 25);
        let metrics = [
            50.0, 1.0, 10_050.0, 0.01, 0.6, 1.2, 12.5, 0.5, 4.0, 0.5, 0.005,
        ];
        assert!(current.fitness(&metrics, &good_report, 0.0) > 0.0);

        let mut version_one = current;
        version_one.scoring_semantics_version = 1;
        version_one.validate().unwrap();
        assert!(version_one.validate_current().is_err());
        let mut legacy = current;
        legacy.scoring_semantics_version = 0;
        legacy.validate().unwrap();
        assert!(legacy.validate_current().is_err());
        assert!(
            legacy
                .checked_for_evaluation(10_000.0, false, (0.005, 0.03))
                .is_err()
        );
        let mut value = serde_json::to_value(legacy).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("scoring_semantics_version");
        let decoded: WeeklySearchObjectiveV1 = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.scoring_semantics_version, 0);
        decoded.validate().unwrap();
        assert!(decoded.validate_current().is_err());

        let mut stale = current;
        stale.scoring_semantics_version = WEEKLY_SCORING_SEMANTICS_VERSION_V1 + 1;
        assert!(stale.validate().is_err());
    }
    // Synthetic selection-only fixtures, not reports reconstructed from entry
    // timestamps. The test objective's 3% risk ceiling is not the v4 run's 1%.
    fn selection_report(windows: u64, trades: usize) -> WeeklyRollingCadenceReportV1 {
        WeeklyRollingCadenceReportV1 {
            trading_days: 2219,
            trades,
            sunday_entries_credited: 0,
            weekend_entries: 0,
            trades_per_month: 21.0 * trades as f64 / 2219.0,
            longest_silent_trading_days: if windows == 0 { 4 } else { 5 },
            violating_windows: windows,
            violation: windows as f64,
            clustered: windows > 0,
        }
    }

    fn selection_metrics(
        net: f64,
        drawdown: f64,
        daily_drawdown: f64,
        _expectancy: f64,
        trades: usize,
    ) -> [f64; 11] {
        [
            net,
            if net >= 0.0 { 1.0 } else { -1.0 },
            10_000.0 + net,
            drawdown,
            0.6,
            1.2,
            net / trades.max(1) as f64,
            0.0,
            trades as f64,
            0.0,
            daily_drawdown,
        ]
    }

    fn selection_fixture() -> (
        WeeklySearchObjectiveV1,
        Vec<[f64; 11]>,
        Vec<WeeklyRollingCadenceReportV1>,
        Vec<f64>,
        [usize; 5],
    ) {
        let objective = test_weekly_objective();
        let metrics = vec![
            selection_metrics(100.0, 0.09, 0.02, 1.0, 445), // B one-gap, joint dimensions
            selection_metrics(50.0, 0.001, 0.001, 50.0, 1), // R one-trade refuge
            selection_metrics(100.0, 0.30, 0.02, 1.0, 445), // D one gap but bad DD
            selection_metrics(-9_900.0, 0.99, 0.049, -2.5, 5_000), // C zero-window near ruin
            selection_metrics(-9_999.0, 0.999, 0.049, -4.0, 5_000), // C2 worse
        ];
        let reports = vec![
            selection_report(1, 445),
            selection_report(2_214, 1),
            selection_report(1, 445),
            selection_report(0, 5_000),
            selection_report(0, 5_000),
        ];
        let fitness = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let indices = [0, 1, 2, 3, 4];
        (objective, metrics, reports, fitness, indices)
    }

    #[test]
    fn weekly_vector_fronts_retain_complementary_infeasible_paths() {
        let (objective, metrics, reports, fitness, i) = selection_fixture();
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        assert_eq!(ranks[i[0]].front_rank, 1, "B is nondominated");
        assert_eq!(ranks[i[3]].front_rank, 1, "C is nondominated with B");
        assert_eq!(ranks[i[1]].front_rank, 2, "R is a complementary path");
        assert_eq!(ranks[i[2]].front_rank, 2, "D is a complementary path");
        assert_eq!(ranks[i[4]].front_rank, 2, "C2 is in the later front");
        assert!(ranks[i[3]].retention_cmp(&ranks[i[4]]) == std::cmp::Ordering::Greater);

        let order = weekly_selection_order(&ranks);
        assert_eq!(order[0], i[0], "bounded distance prefers jointly closer B");
        let survivors = weekly_select_survivors(&ranks, 2);
        let retained: std::collections::BTreeSet<usize> = survivors.into_iter().collect();
        assert_eq!(retained, std::collections::BTreeSet::from([i[0], i[3]]));
    }

    #[test]
    fn weekly_vector_selection_rejects_near_ruin_and_losing_churn_as_feasible() {
        let (objective, mut metrics, mut reports, mut fitness, i) = selection_fixture();
        // Add a genuinely feasible active candidate and a zero-window losing churn.
        metrics.push(selection_metrics(100.0, 0.09, 0.02, 1.0, 445));
        reports.push(selection_report(0, 445));
        fitness.push(0.5);
        let feasible_index = metrics.len() - 1;
        metrics.push(selection_metrics(-100.0, 0.05, 0.02, -1.0, 5_000));
        reports.push(selection_report(0, 5_000));
        fitness.push(-0.5);
        let churn_index = metrics.len() - 1;
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        assert!(ranks[feasible_index].feasible);
        assert!(
            !ranks[churn_index].feasible,
            "zero windows cannot rescue losses"
        );
        assert!(!ranks[i[3]].feasible, "near-ruin C is never feasible");
        let survivors = weekly_select_survivors(&ranks, 2);
        assert_eq!(survivors[0], feasible_index);
        assert!(!survivors.contains(&i[3]));

        let mut order = weekly_selection_order(&ranks);
        assert_eq!(order.remove(0), feasible_index);
    }

    #[test]
    fn weekly_vector_diversity_survives_repeated_identical_vectors() {
        let (objective, base_metrics, base_reports, _, i) = selection_fixture();
        let mut metrics = vec![base_metrics[i[1]], base_metrics[i[2]], base_metrics[i[3]]];
        let mut reports = vec![
            base_reports[i[1]].clone(),
            base_reports[i[2]].clone(),
            base_reports[i[3]].clone(),
        ];
        for _ in 0..20 {
            metrics.push(base_metrics[i[1]]);
            reports.push(base_reports[i[1]].clone());
        }
        let fitness = vec![-1.0; metrics.len()];
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        let survivors = weekly_select_survivors(&ranks, 3);
        assert_eq!(survivors.len(), 3);
        assert!(survivors.contains(&0), "one R representative remains");
        assert!(survivors.contains(&1), "D remains a complementary path");
        assert!(survivors.contains(&2), "C remains a complementary path");
        assert_eq!(
            survivors.iter().filter(|index| **index > 2).count(),
            0,
            "a flood of identical R vectors cannot take the complementary slots"
        );
    }

    #[test]
    fn weekly_vector_same_front_ties_use_fitness_and_crowding_consistently() {
        let (mut objective, mut metrics, mut reports, _, i) = selection_fixture();
        objective.limits.min_net_profit = 0.0;
        metrics.truncate(2);
        reports.truncate(2);
        metrics[0] = selection_metrics(100.0, 0.09, 0.02, 1.0, 445);
        reports[0] = selection_report(0, 445);
        metrics[1] = selection_metrics(200.0, 0.09, 0.02, 1.0, 445);
        reports[1] = selection_report(0, 445);
        let fitness = vec![1.0, 5.0];
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        assert!(ranks[0].feasible && ranks[1].feasible);
        assert_eq!(ranks[1].front_rank, 0);
        assert_eq!(
            ranks[1].preference_cmp(&ranks[0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(weekly_selection_order(&ranks)[0], 1);
        assert_eq!(
            ranks[1].retention_cmp(&ranks[0]),
            std::cmp::Ordering::Greater
        );
        let _ = i;
    }

    #[test]
    fn weekly_vector_ranking_is_input_permutation_invariant() {
        let (objective, metrics, reports, fitness, _) = selection_fixture();
        let original = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        let permutation = [4usize, 0, 3, 1, 2];
        let permuted_metrics: Vec<_> = permutation.iter().map(|i| metrics[*i]).collect();
        let permuted_reports: Vec<_> = permutation.iter().map(|i| reports[*i].clone()).collect();
        let permuted_fitness: Vec<_> = permutation.iter().map(|i| fitness[*i]).collect();
        let shuffled = weekly_selection_ranks(
            &objective,
            &permuted_metrics,
            &permuted_reports,
            &permuted_fitness,
        )
        .unwrap();
        for (new_index, old_index) in permutation.iter().enumerate() {
            assert_eq!(
                shuffled[new_index].front_rank,
                original[*old_index].front_rank
            );
            assert_eq!(
                shuffled[new_index].crowding.to_bits(),
                original[*old_index].crowding.to_bits()
            );
            assert_eq!(shuffled[new_index].feasible, original[*old_index].feasible);
        }
    }

    #[test]
    fn weekly_vector_reverse_chain_peels_one_front_at_a_time() {
        let objective = test_weekly_objective();
        let n = 60usize;
        let metrics: Vec<[f64; 11]> = (0..n)
            .map(|_| selection_metrics(-100.0, 0.05, 0.02, -1.0, 5_000))
            .collect();
        let reports: Vec<_> = (0..n)
            .map(|windows| selection_report(windows as u64, 5_000))
            .collect();
        let fitness = vec![-1.0; n];
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        let mut fronts: Vec<u32> = ranks.iter().map(|rank| rank.front_rank).collect();
        fronts.sort_unstable();
        assert_eq!(fronts, (1..=n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn weekly_vector_empty_observations_keep_empty_selection() {
        let objective = test_weekly_objective();
        let ranks = weekly_selection_ranks(&objective, &[], &[], &[]).unwrap();
        assert!(ranks.is_empty());
        assert!(weekly_selection_order(&ranks).is_empty());
        for requested in [0, 1, usize::MAX] {
            assert!(weekly_select_survivors(&ranks, requested).is_empty());
        }
    }

    #[test]
    fn weekly_vector_all_feasible_preserves_cardinality_and_financial_order() {
        let objective = test_weekly_objective();
        let fitness = [1.0, 5.0, 3.0];
        let expected = [vec![0], vec![1, 0], vec![1, 2, 0]];
        for count in 1..=3 {
            let metrics: Vec<_> = (0..count)
                .map(|_| selection_metrics(100.0, 0.09, 0.02, 1.0, 445))
                .collect();
            let reports = vec![selection_report(0, 445); count];
            let ranks =
                weekly_selection_ranks(&objective, &metrics, &reports, &fitness[..count]).unwrap();
            assert_eq!(ranks.len(), count);
            assert!(ranks.iter().all(|rank| rank.feasible));
            assert!(ranks.iter().all(|rank| rank.axes == [0.0; 8]));
            assert_eq!(weekly_selection_order(&ranks), expected[count - 1]);
            for requested in 0..=count + 1 {
                let survivors = weekly_select_survivors(&ranks, requested);
                assert_eq!(survivors.len(), requested.min(count));
                assert_eq!(survivors, expected[count - 1][..requested.min(count)]);
            }
        }
    }

    #[test]
    fn weekly_vector_feasible_candidates_fill_capacity_before_infeasible() {
        let objective = test_weekly_objective();
        let metrics = vec![selection_metrics(100.0, 0.09, 0.02, 1.0, 445); 3];
        let reports = vec![
            selection_report(1, 445),
            selection_report(0, 445),
            selection_report(0, 445),
        ];
        let ranks =
            weekly_selection_ranks(&objective, &metrics, &reports, &[100.0, 1.0, 5.0]).unwrap();
        assert!(!ranks[0].feasible);
        assert!(ranks[1].feasible && ranks[2].feasible);
        assert_eq!(weekly_select_survivors(&ranks, 2), vec![2, 1]);
        assert_eq!(weekly_select_survivors(&ranks, 3), vec![2, 1, 0]);
    }

    #[test]
    fn weekly_vector_exact_drawdown_boundaries_remain_infeasible() {
        let objective = test_weekly_objective();
        let limits = objective.limits;
        let metrics = vec![
            selection_metrics(100.0, limits.max_drawdown, 0.02, 1.0, 445),
            selection_metrics(100.0, 0.09, limits.max_daily_drawdown, 1.0, 445),
            selection_metrics(
                100.0,
                limits.max_drawdown - f64::EPSILON,
                limits.max_daily_drawdown - f64::EPSILON,
                1.0,
                445,
            ),
        ];
        let reports = vec![selection_report(0, 445); 3];
        let ranks =
            weekly_selection_ranks(&objective, &metrics, &reports, &[100.0, 200.0, -5.0]).unwrap();
        assert!(ranks.iter().all(|rank| rank.axes == [0.0; 8]));
        assert!(
            !ranks[0].feasible,
            "overall drawdown must be strictly below its limit"
        );
        assert!(
            !ranks[1].feasible,
            "daily drawdown must be strictly below its limit"
        );
        assert!(ranks[2].feasible);
        assert_eq!(weekly_select_survivors(&ranks, 1), vec![2]);
        let survivors = weekly_select_survivors(&ranks, 3);
        assert_eq!(survivors.len(), 3);
        assert_eq!(survivors[0], 2);
        assert_eq!(
            survivors
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1, 2])
        );
    }

    #[test]
    fn weekly_vector_requested_capacity_does_not_drop_duplicate_slots() {
        let (objective, mut metrics, mut reports, mut fitness, i) = selection_fixture();
        for _ in 0..3 {
            metrics.push(metrics[i[1]]);
            reports.push(reports[i[1]].clone());
            fitness.push(-2.0);
        }
        let ranks = weekly_selection_ranks(&objective, &metrics, &reports, &fitness).unwrap();
        for requested in [0, 1, 3, ranks.len(), ranks.len() + 1] {
            let survivors = weekly_select_survivors(&ranks, requested);
            let distinct: std::collections::BTreeSet<_> = survivors.iter().copied().collect();
            assert_eq!(survivors.len(), requested.min(ranks.len()));
            assert_eq!(
                distinct.len(),
                survivors.len(),
                "no input slot may be selected twice"
            );
            assert!(distinct.iter().all(|index| *index < ranks.len()));
            if requested >= ranks.len() {
                assert_eq!(distinct, (0..ranks.len()).collect());
            }
        }
    }

    #[test]
    fn weekly_vector_nonpositive_financial_results_remain_infeasible() {
        let objective = test_weekly_objective();
        let metrics = vec![
            selection_metrics(0.0, 0.09, 0.02, 0.0, 445),
            selection_metrics(-100.0, 0.09, 0.02, -1.0, 445),
            selection_metrics(100.0, 0.09, 0.02, 1.0, 445),
        ];
        let reports = vec![selection_report(0, 445); 3];
        let ranks =
            weekly_selection_ranks(&objective, &metrics, &reports, &[100.0, 200.0, -5.0]).unwrap();
        assert!(
            !ranks[0].feasible,
            "zero net and expectancy are not acceptable"
        );
        assert!(!ranks[1].feasible, "active losses are not acceptable");
        assert!(ranks[2].feasible);
        assert_eq!(weekly_select_survivors(&ranks, 1), vec![2]);
    }

    #[test]
    fn weekly_vector_selection_marks_nonfinite_metrics_invalid_not_accepted() {
        let (objective, metrics, reports, fitness, _) = selection_fixture();
        let mut bad = metrics;
        bad[0][0] = f64::NAN;
        let ranks = weekly_selection_ranks(&objective, &bad, &reports, &fitness).unwrap();
        assert!(ranks[0].invalid);
        assert!(!ranks[0].feasible);
        let survivors = weekly_select_survivors(&ranks, ranks.len());
        assert!(!survivors.contains(&0));
    }
}
