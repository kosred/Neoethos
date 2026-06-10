//! Demo forward-test promotion gate.
//!
//! This is the SECOND gate a strategy must clear, distinct from the backtest
//! [`promotion_gate`](super::promotion_gate). The backtest gate asks "were the
//! historical metrics good enough?"; this gate asks the harder, out-of-sample
//! question: "did the strategy actually HOLD UP on a live demo account, within
//! a tolerance of what the backtest promised?".
//!
//! Locked design decision (docs/v0.5-autonomous-trader-design.md §9, decision
//! #5): the gate is TRADE-COUNT based, not calendar based — a strategy is
//! promotion-eligible only after at least `min_demo_trades` (default 100) REAL
//! demo fills AND its live forward metrics land within `forward_tolerance`
//! (default 20%) of the backtest metrics it was promoted on. Calendar time is
//! the wrong unit: a scalper hits 100 trades in a week, a swing strategy in
//! months — both need the same statistical confidence before risking real
//! money. The operator still approves the final real-money switch; this gate
//! only marks ELIGIBILITY, never an automatic promotion.
//!
//! Reuses [`PromotionMetrics`] + [`CriterionResult`] so the UI renders the same
//! per-criterion evidence ("live PF 1.40 vs backtest 1.60 floor 1.28 — pass").

use super::promotion_gate::{CriterionResult, PromotionMetrics};
use serde::{Deserialize, Serialize};

/// Config for the demo forward-test gate. Units match [`PromotionMetrics`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DemoForwardGateConfig {
    /// When false, the gate is a no-op pass (the operator can still gate
    /// manually). Default true.
    pub enabled: bool,
    /// Minimum number of REAL demo fills before the live metrics are
    /// statistically meaningful. Default 100 — a great live Sharpe over 6
    /// trades is noise, exactly as `min_trades` guards the backtest gate.
    pub min_demo_trades: u64,
    /// Allowed degradation of live-vs-backtest, as a fraction. 0.20 = the live
    /// metric may be up to 20% worse than backtest and still pass. Applied as a
    /// FLOOR for higher-is-better metrics (`live >= backtest * (1 - tol)`) and a
    /// CAP for lower-is-better metrics (`live <= backtest * (1 + tol)`).
    pub forward_tolerance: f64,
}

impl Default for DemoForwardGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_demo_trades: 100,
            forward_tolerance: 0.20,
        }
    }
}

/// The gate's verdict: `eligible` plus the full per-criterion breakdown and a
/// one-line summary. `eligible == true` means "the operator MAY now promote to
/// real money", never an automatic switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DemoForwardDecision {
    pub eligible: bool,
    pub criteria: Vec<CriterionResult>,
    pub summary: String,
}

/// Evaluate a strategy's LIVE demo forward-test against the BACKTEST metrics it
/// was promoted on.
///
/// `demo_trades` is the count of real demo fills observed so far (the gate's
/// statistical-significance floor). `live` is the metrics computed from those
/// demo fills; `backtest` is the metrics the backtest gate already approved.
///
/// AND semantics: a single failed criterion blocks eligibility, so the operator
/// never sees a green light unless trade count AND every tracked metric held up
/// out-of-sample within tolerance.
pub fn evaluate_demo_forward_gate(
    demo_trades: u64,
    live: &PromotionMetrics,
    backtest: &PromotionMetrics,
    config: &DemoForwardGateConfig,
) -> DemoForwardDecision {
    if !config.enabled {
        return DemoForwardDecision {
            eligible: true,
            criteria: Vec::new(),
            summary: "Demo forward gate disabled — eligibility not enforced.".to_string(),
        };
    }

    let tol = config.forward_tolerance.max(0.0);
    // Higher-is-better: live must stay at or above backtest*(1-tol).
    let floor = |bt: f64| bt * (1.0 - tol);
    // Lower-is-better: live must stay at or below backtest*(1+tol).
    let cap = |bt: f64| bt * (1.0 + tol);

    let criteria = vec![
        CriterionResult {
            name: "Demo trade count".to_string(),
            passed: demo_trades >= config.min_demo_trades,
            actual: demo_trades as f64,
            threshold: config.min_demo_trades as f64,
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Live profit factor vs backtest".to_string(),
            passed: live.profit_factor >= floor(backtest.profit_factor),
            actual: live.profit_factor,
            threshold: floor(backtest.profit_factor),
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Live Sharpe vs backtest".to_string(),
            passed: live.sharpe >= floor(backtest.sharpe),
            actual: live.sharpe,
            threshold: floor(backtest.sharpe),
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Live win rate vs backtest".to_string(),
            passed: live.win_rate >= floor(backtest.win_rate),
            actual: live.win_rate,
            threshold: floor(backtest.win_rate),
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Live max drawdown % vs backtest".to_string(),
            passed: live.max_drawdown_pct <= cap(backtest.max_drawdown_pct),
            actual: live.max_drawdown_pct,
            threshold: cap(backtest.max_drawdown_pct),
            comparison: "<=".to_string(),
        },
    ];

    let failed: Vec<&str> = criteria
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.name.as_str())
        .collect();
    let eligible = failed.is_empty();
    let summary = if eligible {
        format!(
            "Eligible — {} demo trades and all live metrics within {:.0}% of backtest.",
            demo_trades,
            tol * 100.0
        )
    } else if demo_trades < config.min_demo_trades {
        format!(
            "Not yet — {}/{} demo trades; need more fills before judging live performance.",
            demo_trades, config.min_demo_trades
        )
    } else {
        format!(
            "Blocked — {} of {} criteria failed: {}",
            failed.len(),
            criteria.len(),
            failed.join(", ")
        )
    };

    DemoForwardDecision {
        eligible,
        criteria,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backtest() -> PromotionMetrics {
        PromotionMetrics {
            sharpe: 1.5,
            win_rate: 0.55,
            profit_factor: 1.60,
            max_drawdown_pct: 10.0,
            trades: 300,
        }
    }

    #[test]
    fn enough_trades_and_within_tolerance_is_eligible() {
        // Live degraded ~12% on PF/Sharpe, DD ~15% worse — all inside 20%.
        let live = PromotionMetrics {
            sharpe: 1.32,
            win_rate: 0.50,
            profit_factor: 1.41,
            max_drawdown_pct: 11.5,
            trades: 140,
        };
        let d = evaluate_demo_forward_gate(140, &live, &backtest(), &DemoForwardGateConfig::default());
        assert!(d.eligible, "should be eligible: {}", d.summary);
        assert!(d.criteria.iter().all(|c| c.passed));
    }

    #[test]
    fn too_few_demo_trades_is_not_yet() {
        let live = backtest(); // metrics fine, but only 40 trades
        let d = evaluate_demo_forward_gate(40, &live, &backtest(), &DemoForwardGateConfig::default());
        assert!(!d.eligible);
        assert!(d.summary.starts_with("Not yet"), "summary: {}", d.summary);
        // The trade-count criterion is the one that failed.
        assert!(!d.criteria[0].passed);
    }

    #[test]
    fn degraded_profit_factor_beyond_tolerance_blocks() {
        // PF 1.10 vs backtest 1.60 → floor 1.28, fails (>20% worse).
        let live = PromotionMetrics {
            profit_factor: 1.10,
            ..backtest()
        };
        let d = evaluate_demo_forward_gate(200, &live, &backtest(), &DemoForwardGateConfig::default());
        assert!(!d.eligible);
        assert!(d.summary.starts_with("Blocked"), "summary: {}", d.summary);
    }

    #[test]
    fn live_drawdown_too_much_worse_blocks() {
        // DD 13% vs backtest 10% → cap 12%, fails.
        let live = PromotionMetrics {
            max_drawdown_pct: 13.0,
            ..backtest()
        };
        let d = evaluate_demo_forward_gate(200, &live, &backtest(), &DemoForwardGateConfig::default());
        assert!(!d.eligible);
    }

    #[test]
    fn disabled_gate_is_unconditional_pass() {
        let cfg = DemoForwardGateConfig {
            enabled: false,
            ..DemoForwardGateConfig::default()
        };
        let d = evaluate_demo_forward_gate(0, &backtest(), &backtest(), &cfg);
        assert!(d.eligible);
        assert!(d.criteria.is_empty());
    }
}
