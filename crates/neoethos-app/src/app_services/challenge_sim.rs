//! Prop-firm challenge **first-passage** Monte Carlo — "with THIS portfolio's
//! trade distribution, what is the probability of hitting the profit target
//! BEFORE a loss barrier, and at what risk-per-trade is it highest?"
//!
//! Framing (Curupira, "Starting from the end: prop firm Monte Carlo",
//! curupira.dev 2026 + operator directive 2026-07-02): a challenge is not
//! "make money forever" — it is a bounded random walk between an upper
//! barrier (+profit target) and two lower barriers (max loss, daily loss),
//! optionally inside a finite day window. Passing probability is dominated by
//! the R-multiple distribution and the risk-per-trade, not by the strategy's
//! long-run average. So this report answers, per candidate sizing:
//!   pass% (phase 1), funded% (phase 1 × phase 2), bust%, median days —
//! and sweeps risk-per-trade to find the first-passage-optimal size, which is
//! NOT the long-run-growth-optimal (Kelly) size.
//!
//! Trade source: the portfolio's sibling `*.trades.json` R-multiples, sampled
//! WITH replacement (bootstrap — unlike tail_risk's permutation, a challenge
//! path is open-ended so we need arbitrary-length sequences). Day cadence is
//! taken from the artifact's own entry timestamps.
//!
//! HONESTY: bootstrap assumes iid trades; real losing streaks cluster worse
//! (news days, regime breaks). Treat pass% as an upper bound and the daily
//! barrier as more dangerous live than simulated.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rand::Rng;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChallengeSweepPoint {
    pub risk_pct: f64,
    pub pass_phase1_pct: f64,
    pub funded_pct: f64,
    pub bust_pct: f64,
    pub timeout_pct: f64,
    /// Median calendar days to clear phase 1 (passing paths only; 0 if none).
    pub median_days_phase1: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChallengeReport {
    pub portfolio: String,
    pub trades: usize,
    pub iterations: usize,
    /// FTMO-style rule set the walk was bounded by.
    pub profit_target_pct: f64,
    pub phase2_target_pct: f64,
    pub daily_loss_pct: f64,
    pub max_loss_pct: f64,
    pub day_limit_phase1: usize,
    pub day_limit_phase2: usize,
    /// Trade cadence derived from the artifact's entry timestamps.
    pub trades_per_day: f64,
    /// The sweep over candidate risk-per-trade sizes.
    pub sweep: Vec<ChallengeSweepPoint>,
    /// Risk% with the highest funded probability in the sweep.
    pub best_risk_pct: f64,
    pub best_funded_pct: f64,
    /// Attempts for ≥90% cumulative probability of funding at best risk
    /// (the "budget for multiple attempts" number; 0 = unreachable).
    pub attempts_for_90pct: usize,
    pub note: String,
}

/// Derive the sibling `<stem>.trades.json` from a portfolio path — same rule
/// as `tail_risk::trades_path_for`.
fn trades_path_for(portfolio_path: &str) -> PathBuf {
    let p = portfolio_path.trim();
    let stem = p
        .strip_suffix(".live_portfolio.json")
        .or_else(|| p.strip_suffix(".json"))
        .unwrap_or(p);
    PathBuf::from(format!("{stem}.trades.json"))
}

/// One bounded-random-walk attempt. Returns (passed, busted, days_used).
/// Barriers are FTMO-convention: measured against the INITIAL balance —
/// equity floor `1 − max_loss`, daily floor `day_start − daily_loss`,
/// target `1 + target`. `!passed && !busted` ⇒ timeout (day limit hit).
fn run_attempt(
    r_multiples: &[f64],
    risk_fraction: f64,
    target: f64,
    daily_loss: f64,
    max_loss: f64,
    trades_per_day: f64,
    day_limit: usize,
    rng: &mut impl Rng,
) -> (bool, bool, usize) {
    let mut equity = 1.0_f64;
    let mut day_start = 1.0_f64;
    let mut day: usize = 1;
    // Fractional day clock: each trade advances 1/trades_per_day days.
    let day_step = 1.0 / trades_per_day.max(1e-6);
    let mut clock = 0.0_f64;

    // Hard cap on total trades so a degenerate zero-edge walk terminates.
    let max_trades = (day_limit as f64 * trades_per_day).ceil() as usize + 1;
    for _ in 0..max_trades {
        let r = r_multiples[rng.random_range(0..r_multiples.len())];
        equity *= 1.0 + (risk_fraction * r).clamp(-0.9, 0.9);

        // Barrier checks after every trade (intra-day breach counts).
        if equity <= 1.0 - max_loss || day_start - equity >= daily_loss {
            return (false, true, day);
        }
        if equity >= 1.0 + target {
            return (true, false, day);
        }

        clock += day_step;
        while clock >= 1.0 {
            clock -= 1.0;
            day += 1;
            day_start = equity;
            if day > day_limit {
                return (false, false, day_limit);
            }
        }
    }
    (false, false, day.min(day_limit))
}

/// BLOCKING. Bootstrap the portfolio's R-multiples through FTMO-style
/// barriers across a sweep of risk-per-trade sizes.
pub fn run_challenge_sim(portfolio_path: &str, iterations: usize) -> Result<ChallengeReport> {
    let trades_file = trades_path_for(portfolio_path);
    let raw = std::fs::read_to_string(&trades_file).with_context(|| {
        format!(
            "no trade log next to the portfolio ({}) — re-run discovery on this \
             pair (older artifacts predate per-trade logging)",
            trades_file.display()
        )
    })?;
    let logged: serde_json::Value = serde_json::from_str(&raw).context("parse trades.json")?;
    let mut entries: Vec<(i64, f64)> = Vec::new(); // (entry_time_ms, r_multiple)
    if let Some(arr) = logged.as_array() {
        for strat in arr {
            for t in strat.get("trades").and_then(|v| v.as_array()).unwrap_or(&Vec::new()) {
                let entry_time = t.get("entry_time").and_then(|v| v.as_i64()).unwrap_or(0);
                let r = t.get("r_multiple").and_then(|v| v.as_f64()).unwrap_or(0.0);
                entries.push((entry_time, r));
            }
        }
    }
    entries.retain(|e| e.1.is_finite());
    if entries.len() < 30 {
        anyhow::bail!(
            "only {} logged trades — too few to bootstrap a challenge distribution (need ≥30)",
            entries.len()
        );
    }
    entries.sort_by_key(|e| e.0);
    let r_multiples: Vec<f64> = entries.iter().map(|e| e.1).collect();
    if !r_multiples.iter().any(|r| r.abs() > 1e-12) {
        anyhow::bail!(
            "artifact predates r_multiple logging (all zero) — re-run discovery on this pair"
        );
    }

    // Cadence from the artifact's own clock: trades per CALENDAR day scaled
    // to trading days (5/7). Clamped so a 3-trade weekend artifact cannot
    // claim 50 trades/day and a decade-long one cannot claim ~0.
    let span_ms = (entries.last().unwrap().0 - entries.first().unwrap().0).max(0);
    let span_days = (span_ms as f64 / 86_400_000.0).max(1.0);
    let trades_per_day = (entries.len() as f64 / (span_days * 5.0 / 7.0)).clamp(0.2, 50.0);

    // FTMO-classic rule set. The 30/60-day limits were formally dropped by
    // FTMO in 2024, but keeping them makes the estimate CONSERVATIVE and
    // matches the operator's "stable monthly income" horizon.
    const TARGET_P1: f64 = 0.10;
    const TARGET_P2: f64 = 0.05;
    const DAILY_LOSS: f64 = 0.05;
    const MAX_LOSS: f64 = 0.10;
    const DAY_LIMIT_P1: usize = 30;
    const DAY_LIMIT_P2: usize = 60;

    let iterations = iterations.clamp(500, 10_000);
    // First-passage-optimal sizing is found empirically — sweep the sizes a
    // small trader would actually consider.
    let sweep_risks = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03];

    let mut rng = rand::rng();
    let mut sweep: Vec<ChallengeSweepPoint> = Vec::with_capacity(sweep_risks.len());
    for &risk in &sweep_risks {
        let mut pass1 = 0usize;
        let mut bust = 0usize;
        let mut timeout = 0usize;
        let mut days_pass: Vec<usize> = Vec::new();
        let mut pass2 = 0usize;
        let mut p2_runs = 0usize;
        for _ in 0..iterations {
            let (ok, busted, days) = run_attempt(
                &r_multiples, risk, TARGET_P1, DAILY_LOSS, MAX_LOSS,
                trades_per_day, DAY_LIMIT_P1, &mut rng,
            );
            if ok {
                pass1 += 1;
                days_pass.push(days);
            } else if busted {
                bust += 1;
            } else {
                timeout += 1;
            }
        }
        // Phase 2 measured independently (fresh balance, lower target, longer
        // window) — funded probability = P(phase1) × P(phase2).
        for _ in 0..iterations {
            let (ok, _, _) = run_attempt(
                &r_multiples, risk, TARGET_P2, DAILY_LOSS, MAX_LOSS,
                trades_per_day, DAY_LIMIT_P2, &mut rng,
            );
            p2_runs += 1;
            if ok {
                pass2 += 1;
            }
        }
        days_pass.sort_unstable();
        let p1 = pass1 as f64 / iterations as f64;
        let p2 = if p2_runs > 0 { pass2 as f64 / p2_runs as f64 } else { 0.0 };
        sweep.push(ChallengeSweepPoint {
            risk_pct: risk * 100.0,
            pass_phase1_pct: p1 * 100.0,
            funded_pct: p1 * p2 * 100.0,
            bust_pct: bust as f64 / iterations as f64 * 100.0,
            timeout_pct: timeout as f64 / iterations as f64 * 100.0,
            median_days_phase1: days_pass
                .get(days_pass.len() / 2)
                .copied()
                .unwrap_or(0) as f64,
        });
    }

    let best = sweep
        .iter()
        .max_by(|a, b| {
            a.funded_pct
                .partial_cmp(&b.funded_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or_else(|| sweep[0].clone());
    let p_funded = best.funded_pct / 100.0;
    let attempts_for_90pct = if p_funded > 1e-9 {
        ((0.1_f64).ln() / (1.0 - p_funded).max(1e-12).ln()).ceil() as usize
    } else {
        0
    };

    let note = if best.funded_pct < 5.0 {
        "This portfolio's distribution barely clears prop-firm barriers at ANY size — \
         it needs a better edge, not better sizing. Bootstrap assumes iid trades; live \
         streaks cluster worse."
            .to_string()
    } else {
        format!(
            "First-passage-optimal size ≈ {:.2}% risk/trade → {:.0}% chance of funding per \
             attempt ({} attempts budgets ≥90%). NOTE: this is the CHALLENGE-optimal size, \
             not the long-run Kelly size — drop back to normal sizing once funded. \
             Bootstrap assumes iid trades; treat as an upper bound.",
            best.risk_pct, best.funded_pct, attempts_for_90pct.max(1)
        )
    };

    Ok(ChallengeReport {
        portfolio: portfolio_path.to_string(),
        trades: entries.len(),
        iterations,
        profit_target_pct: TARGET_P1 * 100.0,
        phase2_target_pct: TARGET_P2 * 100.0,
        daily_loss_pct: DAILY_LOSS * 100.0,
        max_loss_pct: MAX_LOSS * 100.0,
        day_limit_phase1: DAY_LIMIT_P1,
        day_limit_phase2: DAY_LIMIT_P2,
        trades_per_day,
        sweep,
        best_risk_pct: best.risk_pct,
        best_funded_pct: best.funded_pct,
        attempts_for_90pct,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempt_passes_with_strong_edge_and_busts_with_negative_edge() {
        let mut rng = rand::rng();
        // Strong edge: 70% +2R / 30% −1R. At 1% risk the walk should clear
        // +10% before −10%/−5%-daily nearly always inside 30 days at 10/day.
        let good: Vec<f64> = (0..100).map(|i| if i < 70 { 2.0 } else { -1.0 }).collect();
        let mut passes = 0;
        for _ in 0..200 {
            let (ok, _, _) =
                run_attempt(&good, 0.01, 0.10, 0.05, 0.10, 10.0, 30, &mut rng);
            if ok {
                passes += 1;
            }
        }
        assert!(passes > 150, "strong edge must usually pass, got {passes}/200");

        // Pure negative edge: 30% +1R / 70% −1R must essentially never pass.
        let bad: Vec<f64> = (0..100).map(|i| if i < 30 { 1.0 } else { -1.0 }).collect();
        let mut bad_passes = 0;
        for _ in 0..200 {
            let (ok, _, _) =
                run_attempt(&bad, 0.01, 0.10, 0.05, 0.10, 10.0, 30, &mut rng);
            if ok {
                bad_passes += 1;
            }
        }
        assert!(bad_passes < 10, "negative edge must almost never pass, got {bad_passes}/200");
    }

    #[test]
    fn daily_loss_barrier_busts_before_max_loss() {
        let mut rng = rand::rng();
        // Every trade loses 1R. At 3% risk, two same-day losses breach the 5%
        // daily barrier long before the 10% total barrier at 10 trades/day.
        let all_loss = vec![-1.0_f64; 10];
        let (ok, busted, days) =
            run_attempt(&all_loss, 0.03, 0.10, 0.05, 0.10, 10.0, 30, &mut rng);
        assert!(!ok && busted, "all-loss walk must bust");
        assert_eq!(days, 1, "3% risk × all-loss must breach the DAILY barrier on day 1");
    }

    #[test]
    fn trades_path_derivation_matches_tail_risk_convention() {
        assert_eq!(
            trades_path_for("X.live_portfolio.json"),
            PathBuf::from("X.trades.json")
        );
        assert_eq!(trades_path_for("Y.json"), PathBuf::from("Y.trades.json"));
    }
}
