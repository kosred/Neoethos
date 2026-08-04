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
///
/// **This struct is the single recipient of those five thresholds.** It
/// is simultaneously (a) the operator's config — it is embedded in
/// [`crate::config::ModelsConfig::promotion_gate`] and deserialised from
/// `config.yaml`, (b) the value [`evaluate_promotion`] enforces, and
/// (c) the value the `/strategy_lab/promotion` endpoint echoes to the
/// UI. There is deliberately no mirror struct with its own `Default`:
/// a mirror is free to drift from the enforced copy, and a drifted
/// display copy is indistinguishable from a correct one in every
/// artifact the operator can see.
///
/// Serialisation is camelCase because the HTTP DTO is camelCase, but
/// every field also accepts its snake_case spelling so the operator's
/// `config.yaml` reads like the rest of the file. Both spellings
/// deserialise to the same field — see
/// `snake_case_and_camel_case_yaml_deserialise_identically`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PromotionGateConfig {
    /// When false, every portfolio promotes regardless of metrics.
    pub enabled: bool,
    /// Minimum acceptable Sharpe ratio (out-of-sample preferred).
    #[serde(alias = "min_sharpe")]
    pub min_sharpe: f64,
    /// Minimum win rate as a fraction in `[0, 1]` (0.45 = 45%).
    #[serde(alias = "min_win_rate")]
    pub min_win_rate: f64,
    /// Minimum profit factor (gross profit / gross loss). 1.0 = break
    /// even before costs; we want a margin above that.
    #[serde(alias = "min_profit_factor")]
    pub min_profit_factor: f64,
    /// Maximum tolerated peak-to-trough drawdown, as a percentage
    /// (25.0 = 25%). Strategies that bled more than this in backtest
    /// are rejected even if other metrics look good.
    #[serde(alias = "max_drawdown_pct")]
    pub max_drawdown_pct: f64,
    /// Minimum number of trades the metrics must be based on. A
    /// stellar Sharpe over 4 trades is noise, not signal.
    #[serde(alias = "min_trades")]
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

    // ─── 2026-08-04: "the gate ran, but read nobody's settings" ──────────
    //
    // `neoethos_app::server::strategy_lab::load_gate_config` took a
    // `&Settings`, ignored it, and returned `PromotionGateConfig::default()`.
    // The thresholds are now a real config field. These tests pin the two
    // properties that make that safe: the values did not move today, and
    // the operator's file is actually read.

    #[test]
    fn promotion_gate_config_default_matches_the_gates_own_default() {
        // The literal bar, spelled out. Adding the config knob must not
        // shift a single decision on any existing install; if someone
        // retunes these, this test makes it a deliberate, visible act
        // rather than a silent change to what "promoted" means.
        let d = PromotionGateConfig::default();
        assert!(d.enabled, "gate must stay ON by default");
        assert_eq!(d.min_sharpe, 1.0);
        assert_eq!(d.min_win_rate, 0.45);
        assert_eq!(d.min_profit_factor, 1.2);
        assert_eq!(d.max_drawdown_pct, 25.0);
        assert_eq!(d.min_trades, 30);
    }

    #[test]
    fn models_config_promotion_gate_default_equals_the_gates_own_default() {
        // The config field must not become a mirror with its own drifting
        // Default — that is precisely the failure this field closes.
        assert_eq!(
            crate::config::ModelsConfig::default().promotion_gate,
            PromotionGateConfig::default(),
            "ModelsConfig.promotion_gate must delegate to the gate's own Default"
        );
    }

    #[test]
    fn a_config_without_the_promotion_gate_key_keeps_the_previous_thresholds() {
        // Every existing config.yaml on disk predates this field. Loading
        // one must reproduce the exact pre-fix behaviour — otherwise the
        // fix silently re-gates the operator's already-promoted portfolios.
        let models: crate::config::ModelsConfig =
            serde_yaml_ng::from_str("ml_models: [lightgbm]\n").expect("legacy config deserialises");
        assert_eq!(models.promotion_gate, PromotionGateConfig::default());
    }

    #[test]
    fn an_operator_set_threshold_survives_a_yaml_round_trip() {
        // The whole point: a value the operator writes must reach the gate.
        let yaml = "\
ml_models: [lightgbm]
promotion_gate:
  min_sharpe: 2.5
  min_trades: 500
";
        let models: crate::config::ModelsConfig =
            serde_yaml_ng::from_str(yaml).expect("operator config deserialises");
        assert_eq!(models.promotion_gate.min_sharpe, 2.5);
        assert_eq!(models.promotion_gate.min_trades, 500);
        // Unspecified fields keep the documented default rather than 0.
        assert_eq!(models.promotion_gate.min_profit_factor, 1.2);
        assert!(models.promotion_gate.enabled);

        // And the gate enforces what was read — a portfolio that clears the
        // default bar must be REJECTED against the operator's tighter one.
        let m = strong(); // sharpe 1.8, 240 trades
        assert!(evaluate_promotion(&m, &PromotionGateConfig::default()).promoted);
        let d = evaluate_promotion(&m, &models.promotion_gate);
        assert!(
            !d.promoted,
            "operator's min_sharpe 2.5 / min_trades 500 must block sharpe 1.8 / 240 trades: {}",
            d.summary
        );
        assert!(d.summary.contains("Sharpe ratio"), "{}", d.summary);
        assert!(d.summary.contains("Trade count"), "{}", d.summary);
    }

    #[test]
    fn snake_case_and_camel_case_yaml_deserialise_identically() {
        // Serialisation is camelCase (the HTTP DTO the UI receives), but
        // config.yaml is snake_case everywhere else. Both must land on the
        // same field — a spelling that silently no-ops would recreate the
        // exact bug this field fixes: a knob the operator sets and nothing
        // reads.
        let snake: PromotionGateConfig =
            serde_yaml_ng::from_str("min_sharpe: 1.7\nmax_drawdown_pct: 8.0\nmin_win_rate: 0.6\n")
                .expect("snake_case deserialises");
        let camel: PromotionGateConfig =
            serde_yaml_ng::from_str("minSharpe: 1.7\nmaxDrawdownPct: 8.0\nminWinRate: 0.6\n")
                .expect("camelCase deserialises");
        assert_eq!(snake, camel);
        assert_eq!(snake.min_sharpe, 1.7);
        assert_eq!(snake.max_drawdown_pct, 8.0);
        assert_eq!(snake.min_win_rate, 0.6);

        // The wire format the UI already consumes is unchanged.
        let json = serde_json::to_string(&PromotionGateConfig::default()).expect("serialises");
        assert!(json.contains("\"minSharpe\""), "{json}");
        assert!(json.contains("\"maxDrawdownPct\""), "{json}");
    }

    #[test]
    fn a_disabled_gate_set_from_config_actually_disables_the_gate() {
        // `enabled: false` was reachable in the struct but not from disk.
        let models: crate::config::ModelsConfig =
            serde_yaml_ng::from_str("promotion_gate:\n  enabled: false\n").expect("deserialises");
        let mut catastrophic = strong();
        catastrophic.sharpe = -3.0;
        let d = evaluate_promotion(&catastrophic, &models.promotion_gate);
        assert!(d.promoted, "operator's `enabled: false` must reach the gate");
    }
}
