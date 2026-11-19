//! `ga_fitness_monthly_upside_v1` — an asymmetric, monthly-upside GA fitness
//! variant (RESEARCH / OPT-IN; not the default objective).
//!
//! ## Why this exists (neo-671 reward-shape experiment)
//!
//! The measured selection experiment recorded on ledger neo-671 ranked V7
//! ladder3 pools (same entries, different exit rungs) under five reward
//! shapes. The mean/Sharpe-family rewards (`ga_fitness`, expectancy, Sharpe)
//! selected early-lock rungs with 76-79% win rates but worst months of
//! -13% to -129%; only the asymmetric reward (median monthly upside plus a
//! tail-loss-only penalty) selected adaptive rungs with sane worst months of
//! -1.7% / -2.1%. A mean rewards the magnitude of a bad month in exactly the
//! same way it rewards the magnitude of a good one, so a candidate with a
//! catastrophic month can outrank a steady one with a slightly lower mean.
//!
//! This scorer breaks that symmetry WITHOUT touching the canonical formulas:
//!
//! ```text
//! score = clip(median(monthly) * 100, -10, +200)
//!       + upside_bonus                       // positive-bias, never negative
//!       + tail_penalty                       // non-zero ONLY below the tail threshold
//!       + win_rate_floor_bonus
//!
//! upside_bonus  = min(max(best_month_pct - median_pct, 0) * 0.25, 50)
//! tail_penalty  = (2.0 + worst_month_pct) * 1.0   iff worst_month_pct < -2.0
//!               = 0.0                             otherwise
//! win_rate_bonus = +5.0 iff win_rate >= 0.50 else 0.0
//! ```
//!
//! Consequences, all deliberate:
//!
//! * **Small variance is not punished.** A flat month series scores exactly
//!   its median (clipped) — there is no variance term, no drawdown term, no
//!   Sharpe term. The canonical scorers keep those terms; this variant does
//!   not inherit them.
//! * **Larger profit is rewarded.** The upside bonus is one-sided: a best
//!   month above the typical month adds credit (capped), while a best month
//!   that merely matches the median adds nothing — it can never subtract.
//! * **Only the tail is punished.** A worst month of -1.9% costs nothing;
//!   -2.1% costs 0.1 point; -13% costs 11; -129% costs 127. The penalty
//!   scales linearly with the depth below the 2% threshold, so a single
//!   catastrophic month can never be mean-compensated by many good ones.
//! * **The median, not the mean.** One -129% month does not move the median
//!   at all, while it drags any mean down (or, in the experiment's
//!   mean-inflated overtrading rungs, the mean hid how bad the tail was).
//!
//! ## Default path untouched
//!
//! This is a NEW named scorer alongside the canonical four. Nothing here is
//! called by `ga_fitness`, `ga_fitness_growth`, `archive_score`,
//! `window_score` or `quality_score`, and `SCORING_VERSION_CURRENT` is not
//! bumped: existing runs and artifacts are unaffected. It becomes reachable
//! only through an opt-in objective variant behind the
//! `search-monthly-upside-objective` cargo feature on the autoresearch side
//! (default OFF), which is excluded from the drawable space and named in the
//! space report. The `DiscoveryConfig::fitness_table` selector and the
//! evaluator's completed-month return vector landed in the neo-671 follow-up:
//! `FitnessTable::MonthlyUpsideV1` dispatches
//! `genetic::evolution_math::apply_metrics` to this scorer with that vector
//! (docs/autoresearch-loop.md §14.7).

use neoethos_core::utils::series::median_ignore_nan;
use serde::{Deserialize, Serialize};

/// A bounded research profile. The target is supplied by the frozen goal set;
/// the preset chooses only growth, median-return and downside weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MonthlyGoalProfileV1 {
    preset: u8,
    monthly_target_ppm: u32,
    goal_progress_selection: bool,
}

impl<'de> Deserialize<'de> for MonthlyGoalProfileV1 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            preset: u8,
            monthly_target_ppm: u32,
            goal_progress_selection: bool,
        }
        let wire = Wire::deserialize(deserializer)?;
        let value = Self {
            preset: wire.preset,
            monthly_target_ppm: wire.monthly_target_ppm,
            goal_progress_selection: wire.goal_progress_selection,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl MonthlyGoalProfileV1 {
    pub const PRESET_COUNT: u8 = 25;

    pub fn new(preset: u8, monthly_target: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            monthly_target.is_finite() && monthly_target > 0.0 && monthly_target <= 1.0,
            "monthly goal must be finite and inside (0, 1]"
        );
        let ppm = (monthly_target * 1_000_000.0).round();
        anyhow::ensure!(
            (monthly_target - ppm / 1_000_000.0).abs() <= 1e-12,
            "monthly goal must be exactly representable in parts per million"
        );
        let value = Self {
            preset,
            monthly_target_ppm: ppm as u32,
            goal_progress_selection: true,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.preset < Self::PRESET_COUNT,
            "unknown monthly-goal preset"
        );
        anyhow::ensure!(
            (1..=1_000_000).contains(&self.monthly_target_ppm),
            "monthly-goal target is outside its admitted range"
        );
        Ok(())
    }

    pub fn preset(self) -> u8 {
        self.preset
    }
    pub fn target_return(self) -> f64 {
        f64::from(self.monthly_target_ppm) / 1_000_000.0
    }
    pub fn target_ppm(self) -> u32 {
        self.monthly_target_ppm
    }
    pub fn uses_goal_progress_selection(self) -> bool {
        self.goal_progress_selection
    }
    /// Paired control: hold the scorer/target fixed and change only selection.
    pub fn with_legacy_selection_control(mut self) -> Self {
        self.goal_progress_selection = false;
        self
    }

    pub fn weights(self) -> anyhow::Result<[f64; 3]> {
        self.validate()?;
        // Fix the growth scale: positive rescalings of all three weights
        // produce the same ordering and must not become separate search arms.
        Ok([
            1.0,
            [0.0, 0.25, 0.5, 1.0, 2.0][usize::from(self.preset % 5)],
            [0.125, 0.25, 0.5, 1.0, 2.0][usize::from(self.preset / 5)],
        ])
    }
}

/// Observed monthly log-growth pace and median return, normalized to the
/// frozen monthly goal, minus the mean loss of the worst 10% of months.
/// All positive returns retain a growth gradient. Win rate supplies no bonus.
/// This is research guidance; the independent acceptance gates remain binding.
pub fn ga_fitness_monthly_goal_progress_v1(
    months: &[f64],
    metrics: &[f64; 11],
    profile: MonthlyGoalProfileV1,
) -> f64 {
    let Ok([growth_weight, median_weight, tail_weight]) = profile.weights() else {
        return f64::NEG_INFINITY;
    };
    if metrics.iter().any(|value| !value.is_finite())
        || months
            .iter()
            .any(|value| !value.is_finite() || *value <= -1.0)
    {
        return f64::NEG_INFINITY;
    }
    if metrics[8] < 1.0 || months.is_empty() {
        return -100.0;
    }
    let goal = profile.target_return();
    let growth =
        months.iter().map(|value| value.ln_1p()).sum::<f64>() / months.len() as f64 / goal.ln_1p();
    let median = median_ignore_nan(months) / goal;
    let mut ordered = months.to_vec();
    ordered.sort_by(f64::total_cmp);
    let tail_count = ordered.len().div_ceil(10);
    let tail_mean = ordered[..tail_count].iter().sum::<f64>() / tail_count as f64;
    let downside = -tail_mean.min(0.0) / goal;
    let score = growth_weight * growth + median_weight * median - tail_weight * downside;
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

/// Median monthly return contribution: clipped to `[-10, +200]` percent.
pub const MEDIAN_CLAMP_MIN_PCT: f64 = -10.0;
pub const MEDIAN_CLAMP_MAX_PCT: f64 = 200.0;
/// One-sided upside bonus: `min(max(best - median, 0) * WEIGHT, CAP)`.
pub const UPSIDE_BONUS_WEIGHT: f64 = 0.25;
pub const UPSIDE_BONUS_CAP_PCT: f64 = 50.0;
/// Tail-loss penalty applies ONLY when the worst month is below -2.0%.
pub const TAIL_THRESHOLD_PCT: f64 = 2.0;
/// `tail_penalty = (TAIL_THRESHOLD_PCT + worst_pct) * TAIL_PENALTY_WEIGHT`.
pub const TAIL_PENALTY_WEIGHT: f64 = 1.0;
/// Small win-rate floor bonus: `+5.0` iff the win rate meets the floor.
pub const WIN_RATE_FLOOR: f64 = 0.50;
pub const WIN_RATE_FLOOR_BONUS: f64 = 5.0;

/// Asymmetric monthly-upside GA fitness (v1), in percent-scale score units.
///
/// `monthly_returns` are fractional monthly returns (0.02 = +2%); `metrics`
/// is the canonical `[BacktestMetrics::to_metric_array]` order and supplies
/// the guards and the win-rate floor bonus:
///
/// * `metrics[1]` = Sharpe — non-finite ⇒ `f64::NEG_INFINITY` (the same
///   "do not propagate" sentinel as `[super::ga_fitness]`).
/// * `metrics[8]` = trade count — `< 1`, an empty `monthly_returns`, or any
///   non-finite month ⇒ `-100.0` (graduated, not -∞, per GA Fix B).
///
/// Never mutates its inputs.
pub fn ga_fitness_monthly_upside_v1(monthly_returns: &[f64], metrics: &[f64; 11]) -> f64 {
    // Guards shared with the canonical scorers (named.rs): non-finite Sharpe
    // → NEG_INFINITY (metrics unusable); zero trades, an empty month series,
    // or a non-finite month → -100.0 (graduated, not -∞, per GA Fix B).
    if !metrics[1].is_finite() {
        return f64::NEG_INFINITY;
    }
    if metrics[8] < 1.0 || monthly_returns.is_empty() {
        return -100.0;
    }
    if monthly_returns.iter().any(|r| !r.is_finite()) {
        return -100.0;
    }

    // All values are finite from here on, so median_ignore_nan degenerates to
    // the ordinary median (it is the shared, tested implementation).
    let median_pct = median_ignore_nan(monthly_returns) * 100.0;
    let best_pct = monthly_returns
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        * 100.0;
    let worst_pct = monthly_returns
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        * 100.0;

    let base = median_pct.clamp(MEDIAN_CLAMP_MIN_PCT, MEDIAN_CLAMP_MAX_PCT);

    // One-sided upside bonus: credit for how far the best month exceeded the
    // typical one, never negative, capped.
    let upside_bonus =
        ((best_pct - median_pct).max(0.0) * UPSIDE_BONUS_WEIGHT).min(UPSIDE_BONUS_CAP_PCT);

    // Tail-loss penalty: non-zero ONLY when the worst month falls below
    // -TAIL_THRESHOLD_PCT, scaling with the depth below it.
    let tail_penalty = if worst_pct < -TAIL_THRESHOLD_PCT {
        (TAIL_THRESHOLD_PCT + worst_pct) * TAIL_PENALTY_WEIGHT
    } else {
        0.0
    };

    let win_rate_bonus = if metrics[4] >= WIN_RATE_FLOOR {
        WIN_RATE_FLOOR_BONUS
    } else {
        0.0
    };

    base + upside_bonus + tail_penalty + win_rate_bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical `[f64; 11]` with only the slots this scorer reads populated.
    fn metrics(win_rate: f64, trades: f64) -> [f64; 11] {
        let mut m = [0.0; 11];
        m[1] = 1.5; // finite Sharpe — the scorer only guards finiteness
        m[4] = win_rate;
        m[8] = trades;
        m
    }

    #[test]
    fn monthly_goal_progress_cannot_prefer_near_zero_profit_for_crossing_the_win_floor() {
        let strong = [0.02; 12];
        let near_zero = [0.0001; 12];
        assert!(
            ga_fitness_monthly_upside_v1(&near_zero, &metrics(0.51, 100.0))
                > ga_fitness_monthly_upside_v1(&strong, &metrics(0.49, 100.0))
        );
        for preset in 0..MonthlyGoalProfileV1::PRESET_COUNT {
            let profile = MonthlyGoalProfileV1::new(preset, 0.04).unwrap();
            assert!(
                ga_fitness_monthly_goal_progress_v1(&strong, &metrics(0.49, 100.0), profile)
                    > ga_fitness_monthly_goal_progress_v1(
                        &near_zero,
                        &metrics(0.51, 100.0),
                        profile
                    ),
                "preset {preset} must reward economic progress over the win-rate discontinuity"
            );
        }
    }

    #[test]
    fn monthly_goal_profiles_must_change_a_return_downside_tradeoff() {
        let steady = [0.01; 12];
        let variable = [
            0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, -0.04, -0.04, -0.04,
        ];
        let observations = metrics(0.6, 100.0);
        let growth_profile = MonthlyGoalProfileV1::new(0, 0.04).unwrap();
        let downside_profile = MonthlyGoalProfileV1::new(20, 0.04).unwrap();
        assert!(
            ga_fitness_monthly_goal_progress_v1(&variable, &observations, growth_profile)
                > ga_fitness_monthly_goal_progress_v1(&steady, &observations, growth_profile)
        );
        assert!(
            ga_fitness_monthly_goal_progress_v1(&steady, &observations, downside_profile)
                > ga_fitness_monthly_goal_progress_v1(&variable, &observations, downside_profile)
        );
    }

    #[test]
    fn monthly_goal_profiles_reject_invalid_or_silently_rounded_targets() {
        for target in [0.0, -0.04, f64::NAN, f64::INFINITY, 1.1, 0.0000001] {
            assert!(MonthlyGoalProfileV1::new(0, target).is_err());
        }
        assert!(MonthlyGoalProfileV1::new(25, 0.04).is_err());
        let profile = MonthlyGoalProfileV1::new(24, 0.04).unwrap();
        assert_eq!(profile.target_ppm(), 40_000);
        assert_eq!(profile.weights().unwrap(), [1.0, 2.0, 2.0]);
        let serialized = serde_json::to_vec(&profile).unwrap();
        assert_eq!(
            serde_json::from_slice::<MonthlyGoalProfileV1>(&serialized).unwrap(),
            profile
        );
        assert!(
            serde_json::from_str::<MonthlyGoalProfileV1>(
                r#"{"preset":255,"monthly_target_ppm":40000,"goal_progress_selection":true}"#
            )
            .is_err()
        );
        let untrusted = MonthlyGoalProfileV1 {
            preset: 255,
            monthly_target_ppm: 40_000,
            goal_progress_selection: true,
        };
        assert!(untrusted.validate().is_err());
        assert_eq!(
            ga_fitness_monthly_goal_progress_v1(&[0.02; 12], &metrics(0.6, 100.0), untrusted),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn monthly_goal_progress_refuses_bankruptcy_and_nonfinite_months() {
        let profile = MonthlyGoalProfileV1::new(0, 0.04).unwrap();
        for months in [
            vec![0.04, -1.0],
            vec![0.04, -1.1],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert_eq!(
                ga_fitness_monthly_goal_progress_v1(&months, &metrics(0.6, 100.0), profile),
                f64::NEG_INFINITY
            );
        }
    }

    #[test]
    fn zero_variance_scores_the_median_with_no_penalty() {
        // The whole point: a flat month series is not punished for its small
        // variance. base = 3.0, no bonus (best == median), no tail, no wr
        // bonus (wr 0.4 < floor).
        let months = [0.03; 10];
        let s = ga_fitness_monthly_upside_v1(&months, &metrics(0.40, 100.0));
        assert!(
            (s - 3.0).abs() < 1e-9,
            "flat +3%/month must score exactly 3.0, got {s}"
        );
    }

    #[test]
    fn median_clamps_to_minus_ten_and_plus_two_hundred() {
        let huge = [5.0; 4]; // +500%/month → base clamps to +200, no bonus/tail
        let s = ga_fitness_monthly_upside_v1(&huge, &metrics(0.40, 100.0));
        assert!(
            (s - 200.0).abs() < 1e-9,
            "median clamp high bound broken: {s}"
        );

        // -50%/month → base clamps to -10 AND the tail penalty applies:
        // (2.0 + -50.0) * 1.0 = -48.0 → -58.0 total.
        let awful = [-0.5; 4];
        let s = ga_fitness_monthly_upside_v1(&awful, &metrics(0.40, 100.0));
        assert!(
            (s - (-58.0)).abs() < 1e-9,
            "median clamp + tail broken: {s}"
        );
    }

    #[test]
    fn tail_penalty_applies_only_below_the_threshold() {
        // Worst month -1.9%: above the -2.0% threshold → zero tail penalty.
        let calm = [0.03, 0.03, 0.03, 0.03, -0.019];
        let s_calm = ga_fitness_monthly_upside_v1(&calm, &metrics(0.40, 100.0));
        assert!(
            (s_calm - 3.0).abs() < 1e-9,
            "worst -1.9% must cost nothing: {s_calm}"
        );

        // Worst month -2.1%: penalty = (2.0 + -2.1) * 1.0 = -0.1.
        let clipped = [0.03, 0.03, 0.03, 0.03, -0.021];
        let s_clipped = ga_fitness_monthly_upside_v1(&clipped, &metrics(0.40, 100.0));
        assert!(
            (s_clipped - 2.9).abs() < 1e-9,
            "worst -2.1% must cost 0.1: {s_clipped}"
        );
    }

    #[test]
    fn tail_penalty_crushes_a_minus_129_pct_worst_month() {
        // The experiment's extreme catastrophic rung: nine +4% months cannot
        // mean-compensate one -129% month. median 4.0, bonus 0.0, tail
        // (2.0 + -129.0) = -127.0, wr 0.78 → +5.0 ⇒ -118.0.
        let catastrophic = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, -1.29];
        let s = ga_fitness_monthly_upside_v1(&catastrophic, &metrics(0.78, 100.0));
        assert!((s - (-118.0)).abs() < 1e-9, "tail penalty math broken: {s}");
    }

    #[test]
    fn upside_bonus_rewards_a_larger_best_month_without_punishing_flatness() {
        // Same median (+2%/month): the candidate whose best month exceeded the
        // median earns the one-sided bonus, the flat candidate is not punished.
        let flat = [0.02; 10];
        let lumpy = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.08];
        let s_flat = ga_fitness_monthly_upside_v1(&flat, &metrics(0.40, 100.0));
        let s_lumpy = ga_fitness_monthly_upside_v1(&lumpy, &metrics(0.40, 100.0));
        assert!(
            (s_flat - 2.0).abs() < 1e-9,
            "flat must score exactly the median: {s_flat}"
        );
        // (8 - 2) * 0.25 = 1.5 → 3.5.
        assert!(
            (s_lumpy - 3.5).abs() < 1e-9,
            "upside bonus math broken: {s_lumpy}"
        );
    }

    #[test]
    fn win_rate_floor_bonus_is_small_and_floor_gated() {
        let months = [0.02; 10];
        let below = ga_fitness_monthly_upside_v1(&months, &metrics(0.49, 100.0));
        let at_floor = ga_fitness_monthly_upside_v1(&months, &metrics(0.50, 100.0));
        assert!(
            (at_floor - below - WIN_RATE_FLOOR_BONUS).abs() < 1e-9,
            "the floor bonus must be exactly +{WIN_RATE_FLOOR_BONUS} at wr >= {WIN_RATE_FLOOR}: \
             {below} vs {at_floor}"
        );
    }

    #[test]
    fn median_handles_odd_even_and_unsorted_input() {
        // Odd count, unsorted: median = 0.02 (2%), best 3% → bonus (3-2)*0.25.
        let odd = [0.01, 0.03, 0.02];
        let s = ga_fitness_monthly_upside_v1(&odd, &metrics(0.40, 100.0));
        assert!((s - 2.25).abs() < 1e-9, "odd-count median broken: {s}");

        // Even count, unsorted: median = (0.02 + 0.03)/2 = 2.5%, best 4% →
        // bonus (4 - 2.5) * 0.25 = 0.375.
        let even = [0.04, 0.01, 0.03, 0.02];
        let s = ga_fitness_monthly_upside_v1(&even, &metrics(0.40, 100.0));
        assert!((s - 2.875).abs() < 1e-9, "even-count median broken: {s}");
    }

    #[test]
    fn selection_experiment_mean_reward_picks_catastrophic_upside_reward_picks_sane() {
        // The neo-671 selection-experiment shape, reconstructed:
        //
        // * catastrophic — the early-lock rung the mean/Sharpe rewards chose:
        //   nine +4% months, one -13% month, 78% win rate.
        // * sane — the adaptive rung the asymmetric reward chose:
        //   steady +2% months, worst month positive, 60% win rate.
        //
        // The MEAN monthly return is 2.3% for the catastrophic rung versus
        // 2.0% for the sane one, so a mean reward selects the candidate whose
        // worst month is -13%. The upside reward must select the sane one.
        let catastrophic = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, -0.13];
        let sane = [0.02; 10];

        let mean_reward = |months: &[f64]| months.iter().sum::<f64>() / months.len() as f64;
        assert!(
            mean_reward(&catastrophic) > mean_reward(&sane),
            "the experiment's premise: the mean must prefer the catastrophic rung \
             ({} > {})",
            mean_reward(&catastrophic),
            mean_reward(&sane)
        );

        let s_catastrophic = ga_fitness_monthly_upside_v1(&catastrophic, &metrics(0.78, 100.0));
        let s_sane = ga_fitness_monthly_upside_v1(&sane, &metrics(0.60, 100.0));
        assert!(
            s_sane > s_catastrophic,
            "the upside reward must select the sane-worst candidate: \
             sane {s_sane} vs catastrophic {s_catastrophic}"
        );
        // Pins, so a silent constant drift cannot re-break the selection:
        // catastrophic = 4.0 + 0.0 - 11.0 + 5.0 = -2.0; sane = 2.0 + 0.0 + 5.0 = 7.0.
        assert!(
            (s_catastrophic - (-2.0)).abs() < 1e-9,
            "catastrophic pin: {s_catastrophic}"
        );
        assert!((s_sane - 7.0).abs() < 1e-9, "sane pin: {s_sane}");
    }

    #[test]
    fn guards_mirror_the_canonical_named_scorers() {
        // NaN Sharpe → NEG_INFINITY (same "do not propagate" sentinel).
        let mut nan_sharpe = metrics(0.60, 100.0);
        nan_sharpe[1] = f64::NAN;
        assert_eq!(
            ga_fitness_monthly_upside_v1(&[0.02; 10], &nan_sharpe),
            f64::NEG_INFINITY
        );

        // Zero trades → -100.0 (graduated, per GA Fix B).
        assert_eq!(
            ga_fitness_monthly_upside_v1(&[0.02; 10], &metrics(0.60, 0.0)),
            -100.0
        );

        // No monthly observations → -100.0.
        assert_eq!(
            ga_fitness_monthly_upside_v1(&[], &metrics(0.60, 100.0)),
            -100.0
        );

        // A corrupted month (non-finite) → -100.0, not a guessed median.
        let corrupted = [0.02, 0.02, f64::NAN, 0.02];
        assert_eq!(
            ga_fitness_monthly_upside_v1(&corrupted, &metrics(0.60, 100.0)),
            -100.0
        );
    }

    #[test]
    fn inputs_are_never_mutated() {
        let months = vec![0.04, 0.01, 0.03, 0.02, -0.13];
        let before_months = months.clone();
        let m = metrics(0.78, 100.0);
        let before_metrics = m;
        let _ = ga_fitness_monthly_upside_v1(&months, &m);
        assert_eq!(months, before_months, "monthly returns were mutated");
        assert_eq!(m, before_metrics, "metrics were mutated");
    }
}
