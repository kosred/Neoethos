//! Promotion Gate — the quality bar a discovered + trained strategy
//! portfolio must clear before it is promoted to live trading (F-330).
//!
//! The Strategy Lab pipeline is Discovery → Training → Validation →
//! **Promotion Gate**. The first three stages produce a portfolio and
//! its backtest/walk-forward metrics; this gate is the final, explicit
//! decision point: does the portfolio meet the operator's minimum
//! Sharpe / win-rate / profit-factor / drawdown / trade-count bar?
//!
//! This module is deliberately **pure**: it takes already-computed
//! metrics + a threshold config and returns a structured decision with
//! a per-criterion breakdown. It does NOT read files, run backtests,
//! or touch the network — those live in the neoethos-app pipeline
//! orchestrator that calls `evaluate_promotion`. Keeping it pure makes
//! the gate trivially testable and lets both the HTTP endpoint and the
//! CLI share one source of truth for "is this good enough".
//!
//! ## Metric sources
//!
//! The inputs map onto fields the discovery/training pipeline already
//! produces (see `neoethos_search::genetic::Gene` +
//! `app_services::discovery::ModelTargetEntry`): `sharpe`, `win_rate`,
//! `profit_factor`, `max_drawdown_pct`, `trades`. Calmar is
//! intentionally omitted — it needs an annualised-return input the
//! portfolio artifacts don't currently carry; `max_drawdown_pct` is
//! the drawdown guard instead.

use serde::{Deserialize, Serialize};

/// Operator-tunable thresholds for the promotion gate.
///
/// Defaults are deliberately moderate — a retail/standard account bar.
/// A PropFirm preset (or the operator via Settings) can tighten them.
/// `enabled: false` makes the gate a no-op pass-through, for operators
/// who want the pipeline to promote whatever it finds (e.g. demo-only
/// experimentation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionGateConfig {
    /// When false, every portfolio promotes regardless of metrics.
    pub enabled: bool,
    /// Minimum acceptable Sharpe ratio (out-of-sample preferred).
    pub min_sharpe: f64,
    /// Minimum win rate as a fraction in `[0, 1]` (0.45 = 45%).
    pub min_win_rate: f64,
    /// Minimum profit factor (gross profit / gross loss). 1.0 = break
    /// even before costs; we want a margin above that.
    pub min_profit_factor: f64,
    /// Maximum tolerated peak-to-trough drawdown, as a percentage
    /// (25.0 = 25%). Strategies that bled more than this in backtest
    /// are rejected even if other metrics look good.
    pub max_drawdown_pct: f64,
    /// Minimum number of trades the metrics must be based on. A
    /// stellar Sharpe over 4 trades is noise, not signal.
    pub min_trades: u64,
}

impl Default for PromotionGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_sharpe: 1.0,
            min_win_rate: 0.45,
            min_profit_factor: 1.2,
            max_drawdown_pct: 25.0,
            min_trades: 30,
        }
    }
}

/// The metrics a portfolio (or a single strategy) presents to the gate.
/// Units match the config thresholds exactly: `win_rate` is a fraction,
/// `max_drawdown_pct` is a percentage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionMetrics {
    pub sharpe: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub trades: u64,
}

/// The result of checking one threshold. `passed` is the verdict;
/// `actual` vs `threshold` (with the `comparison` operator) is the
/// evidence the UI renders so the operator sees WHY a portfolio was
/// rejected, not just that it was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CriterionResult {
    pub name: String,
    pub passed: bool,
    pub actual: f64,
    pub threshold: f64,
    /// Human-readable comparator: `">="` for floors, `"<="` for caps.
    pub comparison: String,
}

/// The gate's verdict on a portfolio: the overall `promoted` boolean
/// plus the full per-criterion breakdown and a one-line summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionDecision {
    pub promoted: bool,
    pub criteria: Vec<CriterionResult>,
    pub summary: String,
}

/// Evaluate a portfolio's metrics against the gate config.
///
/// Returns a [`PromotionDecision`] with one [`CriterionResult`] per
/// threshold. The portfolio is promoted only when EVERY criterion
/// passes (AND semantics — a single failure blocks promotion). When
/// the gate is disabled the decision is an unconditional pass with an
/// empty criteria list.
pub fn evaluate_promotion(
    metrics: &PromotionMetrics,
    config: &PromotionGateConfig,
) -> PromotionDecision {
    if !config.enabled {
        return PromotionDecision {
            promoted: true,
            criteria: Vec::new(),
            summary: "Promotion gate disabled — portfolio auto-promoted.".to_string(),
        };
    }

    // Audit B07 (2026-07-13): reject non-finite evidence BEFORE threshold
    // comparisons. The floor checks (`NaN >= x` → false) happened to fail
    // safe, but `+inf` sailed through every floor (an infinite profit
    // factor would auto-pass), and a NaN that reached the UI serialized as
    // `null` with a passing sibling criterion — indistinguishable from a
    // legitimate rejection. Non-finite metrics mean the evidence pipeline
    // is broken; name the field instead of pretending to gate on it.
    let non_finite: Vec<(&str, f64)> = [
        ("Sharpe ratio", metrics.sharpe),
        ("Win rate", metrics.win_rate),
        ("Profit factor", metrics.profit_factor),
        ("Max drawdown %", metrics.max_drawdown_pct),
    ]
    .into_iter()
    .filter(|(_, v)| !v.is_finite())
    .collect();
    if !non_finite.is_empty() {
        let criteria = non_finite
            .iter()
            .map(|(name, value)| CriterionResult {
                name: format!("{name} (finite)"),
                passed: false,
                actual: *value,
                threshold: 0.0,
                comparison: "finite".to_string(),
            })
            .collect::<Vec<_>>();
        let fields: Vec<&str> = non_finite.iter().map(|(name, _)| *name).collect();
        return PromotionDecision {
            promoted: false,
            criteria,
            summary: format!(
                "Rejected: non-finite metric(s) {} — the evidence pipeline produced \
                 NaN/inf, which cannot be gated on. Fix the metric source before promoting.",
                fields.join(", ")
            ),
        };
    }

    let criteria = vec![
        CriterionResult {
            name: "Sharpe ratio".to_string(),
            passed: metrics.sharpe >= config.min_sharpe,
            actual: metrics.sharpe,
            threshold: config.min_sharpe,
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Win rate".to_string(),
            passed: metrics.win_rate >= config.min_win_rate,
            actual: metrics.win_rate,
            threshold: config.min_win_rate,
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Profit factor".to_string(),
            passed: metrics.profit_factor >= config.min_profit_factor,
            actual: metrics.profit_factor,
            threshold: config.min_profit_factor,
            comparison: ">=".to_string(),
        },
        CriterionResult {
            name: "Max drawdown %".to_string(),
            passed: metrics.max_drawdown_pct <= config.max_drawdown_pct,
            actual: metrics.max_drawdown_pct,
            threshold: config.max_drawdown_pct,
            comparison: "<=".to_string(),
        },
        CriterionResult {
            name: "Trade count".to_string(),
            passed: metrics.trades >= config.min_trades,
            actual: metrics.trades as f64,
            threshold: config.min_trades as f64,
            comparison: ">=".to_string(),
        },
    ];

    let failed: Vec<&str> = criteria
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.name.as_str())
        .collect();
    let promoted = failed.is_empty();
    let summary = if promoted {
        format!(
            "All {} criteria passed — portfolio is eligible for promotion.",
            criteria.len()
        )
    } else {
        format!(
            "{} of {} criteria failed: {}",
            failed.len(),
            criteria.len(),
            failed.join(", ")
        )
    };

    PromotionDecision {
        promoted,
        criteria,
        summary,
    }
}

/// Aggregate a portfolio's per-strategy metrics into one
/// [`PromotionMetrics`] for a portfolio-level gate decision.
///
/// Aggregation rules, chosen so the gate is conservative (a weak
/// portfolio can't hide behind one stellar strategy):
///   - `sharpe`, `win_rate`, `profit_factor` → mean across strategies
///   - `max_drawdown_pct` → the WORST (max) single-strategy drawdown
///   - `trades` → sum across strategies
///
/// Returns `None` for an empty portfolio — the caller should treat
/// "nothing to promote" as a non-promotion rather than a pass.
pub fn aggregate_portfolio(entries: &[PromotionMetrics]) -> Option<PromotionMetrics> {
    if entries.is_empty() {
        return None;
    }
    let n = entries.len() as f64;
    Some(PromotionMetrics {
        sharpe: entries.iter().map(|e| e.sharpe).sum::<f64>() / n,
        win_rate: entries.iter().map(|e| e.win_rate).sum::<f64>() / n,
        profit_factor: entries.iter().map(|e| e.profit_factor).sum::<f64>() / n,
        // Audit B07: `f64::max` IGNORES NaN (`f64::max(0.0, NaN) == 0.0`), so a
        // member with NaN drawdown silently vanished from the worst-drawdown
        // aggregate — the one field where a broken member could hide behind
        // healthy siblings (the mean fields propagate NaN on their own).
        // Propagate NaN explicitly; `evaluate_promotion` then rejects it
        // loudly as non-finite evidence.
        max_drawdown_pct: entries.iter().map(|e| e.max_drawdown_pct).fold(
            0.0_f64,
            |acc, dd| {
                if dd.is_nan() { f64::NAN } else { acc.max(dd) }
            },
        ),
        trades: entries.iter().map(|e| e.trades).sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strong() -> PromotionMetrics {
        PromotionMetrics {
            sharpe: 1.8,
            win_rate: 0.56,
            profit_factor: 1.6,
            max_drawdown_pct: 12.0,
            trades: 240,
        }
    }

    #[test]
    fn strong_portfolio_is_promoted() {
        let d = evaluate_promotion(&strong(), &PromotionGateConfig::default());
        assert!(d.promoted, "summary was: {}", d.summary);
        assert_eq!(d.criteria.len(), 5);
        assert!(d.criteria.iter().all(|c| c.passed));
    }

    #[test]
    fn low_sharpe_blocks_promotion_and_names_the_criterion() {
        let mut m = strong();
        m.sharpe = 0.4; // below default 1.0
        let d = evaluate_promotion(&m, &PromotionGateConfig::default());
        assert!(!d.promoted);
        let sharpe = d.criteria.iter().find(|c| c.name == "Sharpe ratio").unwrap();
        assert!(!sharpe.passed);
        assert_eq!(sharpe.comparison, ">=");
        assert!(d.summary.contains("Sharpe ratio"));
    }

    #[test]
    fn excessive_drawdown_blocks_promotion() {
        let mut m = strong();
        m.max_drawdown_pct = 40.0; // above default 25%
        let d = evaluate_promotion(&m, &PromotionGateConfig::default());
        assert!(!d.promoted);
        let dd = d.criteria.iter().find(|c| c.name == "Max drawdown %").unwrap();
        assert!(!dd.passed);
        assert_eq!(dd.comparison, "<=");
    }

    #[test]
    fn too_few_trades_blocks_promotion() {
        let mut m = strong();
        m.trades = 5; // below default 30
        let d = evaluate_promotion(&m, &PromotionGateConfig::default());
        assert!(!d.promoted);
        assert!(d.summary.contains("Trade count"));
    }

    #[test]
    fn disabled_gate_always_promotes() {
        let mut m = strong();
        m.sharpe = -3.0; // catastrophic, but gate is off
        let cfg = PromotionGateConfig {
            enabled: false,
            ..PromotionGateConfig::default()
        };
        let d = evaluate_promotion(&m, &cfg);
        assert!(d.promoted);
        assert!(d.criteria.is_empty());
    }

    #[test]
    fn aggregate_uses_mean_and_worst_drawdown() {
        let entries = vec![
            PromotionMetrics {
                sharpe: 2.0,
                win_rate: 0.6,
                profit_factor: 1.8,
                max_drawdown_pct: 10.0,
                trades: 100,
            },
            PromotionMetrics {
                sharpe: 1.0,
                win_rate: 0.5,
                profit_factor: 1.2,
                max_drawdown_pct: 22.0,
                trades: 80,
            },
        ];
        let agg = aggregate_portfolio(&entries).unwrap();
        assert!((agg.sharpe - 1.5).abs() < 1e-9);
        assert!((agg.win_rate - 0.55).abs() < 1e-9);
        assert!((agg.max_drawdown_pct - 22.0).abs() < 1e-9); // worst
        assert_eq!(agg.trades, 180); // sum
    }

    #[test]
    fn non_finite_metrics_fail_every_field_loudly() {
        // Audit B07: NaN, +inf and -inf must be rejected as broken evidence,
        // never compared against thresholds. +inf profit factor used to PASS.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut m = strong();
            m.profit_factor = bad;
            let d = evaluate_promotion(&m, &PromotionGateConfig::default());
            assert!(!d.promoted, "non-finite profit factor ({bad}) must block");
            assert!(
                d.summary.contains("non-finite"),
                "summary must name the failure mode: {}",
                d.summary
            );
            assert!(
                d.criteria.iter().any(|c| c.name.contains("Profit factor")),
                "criteria must name the offending field"
            );
        }
    }

    #[test]
    fn nan_drawdown_cannot_hide_in_portfolio_aggregation() {
        // Audit B07: f64::max ignores NaN, so a NaN-drawdown member used to
        // vanish from the worst-drawdown aggregate. It must propagate and
        // then fail the gate as non-finite evidence.
        let mut broken = strong();
        broken.max_drawdown_pct = f64::NAN;
        let agg = aggregate_portfolio(&[strong(), broken]).expect("non-empty");
        assert!(
            agg.max_drawdown_pct.is_nan(),
            "NaN member drawdown must poison the aggregate, not vanish"
        );
        let d = evaluate_promotion(&agg, &PromotionGateConfig::default());
        assert!(!d.promoted, "poisoned aggregate must not promote");
    }

    #[test]
    fn aggregate_empty_portfolio_is_none() {
        assert!(aggregate_portfolio(&[]).is_none());
    }
}
