//! Trade-sequence Monte Carlo tail risk — "what is the WORST this portfolio
//! can plausibly do to my account at my current sizing?"
//!
//! A single backtest shows ONE ordering of trades; the drawdown you actually
//! live through depends on the ordering you get. Reshuffling the realized
//! trade sequence thousands of times yields the DISTRIBUTION of max drawdowns
//! — the risky-mode number that matters before pressing Start on a small
//! account is the 95th percentile, not the backtest's single path.
//!
//! Sizing model: each trade's `r_multiple` (net P&L / initial risk) compounds
//! the equity at the CURRENT `risk.risk_per_trade`:
//!     equity ← equity × (1 + risk_fraction × R)
//! so the report answers for YOUR sizing today, not the backtest's. Artifacts
//! that predate `r_multiple` fall back to `pnl_pct` (flagged in the report).
//!
//! HONESTY NOTE: shuffling assumes trade returns are exchangeable (no serial
//! dependence). Real losing streaks can cluster worse than iid — treat p95 as
//! a FLOOR on prudence, not a ceiling.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TailRiskReport {
    pub portfolio: String,
    pub trades: usize,
    pub iterations: usize,
    /// The per-trade risk fraction the simulation compounded at.
    pub risk_fraction: f64,
    /// "rMultiple" | "pnlPct" — which per-trade return source was available.
    pub mode: String,
    pub max_dd_p50_pct: f64,
    pub max_dd_p90_pct: f64,
    pub max_dd_p95_pct: f64,
    pub max_dd_p99_pct: f64,
    pub worst_dd_pct: f64,
    /// Probability the equity path loses ≥ `ruin_threshold_pct` at some point.
    pub ruin_threshold_pct: f64,
    pub ruin_probability_pct: f64,
    pub median_final_multiple: f64,
    pub note: String,
}

/// Derive the sibling `<stem>.trades.json` from a portfolio path
/// (`X.live_portfolio.json` → `X.trades.json`; `X.json` → `X.trades.json`).
fn trades_path_for(portfolio_path: &str) -> PathBuf {
    let p = portfolio_path.trim();
    let stem = p
        .strip_suffix(".live_portfolio.json")
        .or_else(|| p.strip_suffix(".json"))
        .unwrap_or(p);
    PathBuf::from(format!("{stem}.trades.json"))
}

/// Max drawdown (as a fraction of the running peak) of a compounded equity
/// path over per-trade returns `rets` (each already scaled: equity ×(1+ret)).
fn max_drawdown(rets: &[f64]) -> (f64, f64) {
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for &r in rets {
        equity *= (1.0 + r).max(0.0); // equity can hit 0 (ruin), never negative
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
        if equity <= 0.0 {
            return (1.0, 0.0);
        }
    }
    (max_dd, equity)
}

/// BLOCKING. Load the portfolio's logged trades, Monte-Carlo the sequence and
/// report the drawdown distribution at the CURRENT risk sizing.
pub fn run_tail_risk(
    portfolio_path: &str,
    iterations: usize,
    risk_override: Option<f64>,
) -> Result<TailRiskReport> {
    let trades_file = trades_path_for(portfolio_path);
    let raw = std::fs::read_to_string(&trades_file).with_context(|| {
        format!(
            "no trade log next to the portfolio ({}) — re-run discovery on this \
             pair (older artifacts predate per-trade logging)",
            trades_file.display()
        )
    })?;
    // Shape: Vec<LoggedStrategyTrades { strategy_id, trades: Vec<Trade> }>.
    let logged: serde_json::Value = serde_json::from_str(&raw).context("parse trades.json")?;
    let mut entries: Vec<(i64, f64, Option<f64>)> = Vec::new(); // (entry_time, r_multiple, pnl_pct)
    if let Some(arr) = logged.as_array() {
        for strat in arr {
            for t in strat.get("trades").and_then(|v| v.as_array()).unwrap_or(&Vec::new()) {
                let entry_time = t.get("entry_time").and_then(|v| v.as_i64()).unwrap_or(0);
                let r = t.get("r_multiple").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let pct = t.get("pnl_pct").and_then(|v| v.as_f64());
                entries.push((entry_time, r, pct));
            }
        }
    }
    entries.sort_by_key(|e| e.0);
    if entries.len() < 20 {
        anyhow::bail!(
            "only {} logged trades — too few for a meaningful drawdown distribution (need ≥20)",
            entries.len()
        );
    }

    // Current sizing from config (override wins).
    let risk_fraction = risk_override
        .filter(|r| r.is_finite() && *r > 0.0 && *r <= 0.5)
        .or_else(|| {
            neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
                .ok()
                .map(|s| s.risk.risk_per_trade)
        })
        .unwrap_or(0.01)
        .clamp(0.0001, 0.5);

    // Per-trade return at THIS sizing: prefer risk-normalized R-multiples;
    // fall back to the artifact's own pnl_pct when R is absent (older runs).
    let have_r = entries.iter().any(|e| e.1.abs() > 1e-12);
    let (mode, mut rets): (String, Vec<f64>) = if have_r {
        (
            "rMultiple".into(),
            entries.iter().map(|e| risk_fraction * e.1).collect(),
        )
    } else {
        (
            "pnlPct".into(),
            entries
                .iter()
                .map(|e| e.2.unwrap_or(0.0) / 100.0)
                .collect(),
        )
    };
    // Clamp pathological single-trade returns (bad artifacts) to ±90%.
    for r in rets.iter_mut() {
        *r = r.clamp(-0.9, 0.9);
    }

    let iterations = iterations.clamp(200, 20_000);
    const RUIN_THRESHOLD: f64 = 0.5; // losing half the account = ruin for a small trader

    let mut dds: Vec<f64> = Vec::with_capacity(iterations);
    let mut finals: Vec<f64> = Vec::with_capacity(iterations);
    let mut ruined = 0usize;
    let mut rng = rand::rng();
    let mut seq = rets.clone();
    for _ in 0..iterations {
        seq.shuffle(&mut rng);
        let (dd, fin) = max_drawdown(&seq);
        if dd >= RUIN_THRESHOLD {
            ruined += 1;
        }
        dds.push(dd);
        finals.push(fin);
    }
    dds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    finals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f64 {
        let idx = ((dds.len() as f64 - 1.0) * p).round() as usize;
        dds[idx.min(dds.len() - 1)] * 100.0
    };

    let p95 = pct(0.95);
    let ruin_prob = ruined as f64 / iterations as f64 * 100.0;
    let note = if ruin_prob >= 1.0 {
        format!(
            "DANGER: at {:.2}% risk/trade this portfolio has a {ruin_prob:.1}% chance of \
             losing half the account on an unlucky ordering. Cut the risk before going live.",
            risk_fraction * 100.0
        )
    } else if p95 > 30.0 {
        format!(
            "CAUTION: 1-in-20 orderings draw down ≥{p95:.0}%. Consider lower risk/trade; \
             remember shuffling assumes independent trades — real streaks can cluster worse."
        )
    } else {
        format!(
            "At {:.2}% risk/trade the drawdown distribution looks survivable \
             (p95 {p95:.0}%). Shuffling assumes independent trades — treat as a floor.",
            risk_fraction * 100.0
        )
    };

    Ok(TailRiskReport {
        portfolio: portfolio_path.to_string(),
        trades: entries.len(),
        iterations,
        risk_fraction,
        mode,
        max_dd_p50_pct: pct(0.50),
        max_dd_p90_pct: pct(0.90),
        max_dd_p95_pct: p95,
        max_dd_p99_pct: pct(0.99),
        worst_dd_pct: dds.last().copied().unwrap_or(0.0) * 100.0,
        ruin_threshold_pct: RUIN_THRESHOLD * 100.0,
        ruin_probability_pct: ruin_prob,
        median_final_multiple: finals[finals.len() / 2],
        note,
    })
}
