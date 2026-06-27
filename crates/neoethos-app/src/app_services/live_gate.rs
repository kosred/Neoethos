//! Demo forward-test gate enforcement for the autonomous LIVE path.
//!
//! Before the engine is allowed to trade REAL money, the strategy must clear
//! the demo forward-test gate ([`neoethos_core::domain::demo_gate`]): at least
//! N real DEMO fills AND live metrics within tolerance of the backtest it was
//! promoted on. On a Demo account the gate is a no-op pass — a demo account is
//! exactly how you accumulate those fills in the first place.
//!
//! This never *initiates* trading; it only BLOCKS the live start when the
//! strategy hasn't yet proven itself on demo. The operator still presses Start.

use anyhow::{Context, Result};

use neoethos_core::Settings;
use neoethos_core::broker_config::CTraderBrokerEnvironment;
use neoethos_core::domain::demo_gate::{
    DemoForwardDecision, DemoForwardGateConfig, evaluate_demo_forward_gate,
};
use neoethos_core::domain::promotion_gate::PromotionMetrics;

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::journal_stats::compute_stats;
use crate::app_services::journal_store::{query_closed_trades, query_equity};

/// True when the active broker environment routes to REAL money.
pub fn active_env_is_live() -> bool {
    matches!(
        load_broker_settings().ctrader.environment,
        CTraderBrokerEnvironment::Live
    )
}

/// Read the BACKTEST [`PromotionMetrics`] for a `*.live_portfolio.json` from its
/// sibling `*.quality.json` (written by discovery). Units are normalised to
/// match the gate + the journal: `win_rate` as a fraction, `max_drawdown_pct`
/// as a PERCENT (quality.json stores it as a fraction, e.g. 0.118 → 11.8).
fn backtest_metrics_for(portfolio_path: &str) -> Result<PromotionMetrics> {
    let p = std::path::Path::new(portfolio_path);
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    // discovery writes `<base>.json.live_portfolio.json` + `<base>.json.quality.json`
    let stem = name
        .strip_suffix(".live_portfolio.json")
        .unwrap_or(name);
    let dir = p.parent().unwrap_or_else(|| std::path::Path::new("."));
    let qpath = dir.join(format!("{stem}.quality.json"));
    let text = std::fs::read_to_string(&qpath)
        .with_context(|| format!("read backtest quality {}", qpath.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse quality {}", qpath.display()))?;
    let f = |k: &str| v.get(k).and_then(|x| x.as_f64());

    let mut win_rate = f("win_rate").unwrap_or(0.0);
    if win_rate > 1.0 {
        win_rate /= 100.0; // defensive: accept a percent too
    }
    let dd_frac = f("max_drawdown_pct").unwrap_or(0.0);

    Ok(PromotionMetrics {
        sharpe: f("sharpe_ratio").unwrap_or(0.0),
        win_rate,
        profit_factor: f("profit_factor").unwrap_or(0.0),
        max_drawdown_pct: dd_frac * 100.0,
        trades: f("total_trades").unwrap_or(0.0) as u64,
    })
}

/// Evaluate the demo forward gate for a live portfolio: demo fills from the
/// journal (filtered to the portfolio's symbol) measured against the backtest
/// metrics the strategy was promoted on. Equity-derived metrics (Sharpe, max
/// drawdown) use the account equity curve, which is exact while the live engine
/// trades a single symbol — a slightly conservative blend otherwise.
pub fn evaluate_for_portfolio(portfolio_path: &str) -> Result<DemoForwardDecision> {
    let artifact = neoethos_search::load_live_portfolio_json(portfolio_path)
        .with_context(|| format!("load live portfolio {portfolio_path}"))?;
    let symbol = artifact.symbol;
    let backtest = backtest_metrics_for(portfolio_path)?;

    let data_dir = Settings::load()
        .map(|s| s.system.data_dir)
        .unwrap_or_else(|_| std::path::PathBuf::from("data"));
    let trades: Vec<_> = query_closed_trades(&data_dir, None, None)
        .into_iter()
        .filter(|t| t.symbol.eq_ignore_ascii_case(&symbol))
        .collect();
    let equity = query_equity(&data_dir, None, None);
    let stats = compute_stats(&trades, &equity);

    let live = PromotionMetrics {
        sharpe: stats.sharpe.unwrap_or(0.0),
        win_rate: stats.win_rate_pct / 100.0,
        // No losses yet → PF is None (∞); treat as meeting the backtest floor.
        profit_factor: stats.profit_factor.unwrap_or(backtest.profit_factor),
        max_drawdown_pct: stats.max_drawdown_pct,
        trades: stats.total_trades as u64,
    };

    Ok(evaluate_demo_forward_gate(
        stats.total_trades as u64,
        &live,
        &backtest,
        &DemoForwardGateConfig::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtest_metrics_convert_drawdown_fraction_to_percent() {
        let dir = std::env::temp_dir().join(format!("neo_gate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let q = dir.join("X_H1.json.quality.json");
        std::fs::write(
            &q,
            r#"{"win_rate":0.45,"profit_factor":1.78,"sharpe_ratio":4.28,"max_drawdown_pct":0.118,"total_trades":300}"#,
        )
        .unwrap();
        let pf = dir.join("X_H1.json.live_portfolio.json");
        let m = backtest_metrics_for(pf.to_str().unwrap()).unwrap();
        assert!((m.win_rate - 0.45).abs() < 1e-9);
        assert!((m.profit_factor - 1.78).abs() < 1e-9);
        assert!((m.max_drawdown_pct - 11.8).abs() < 1e-6, "dd% = {}", m.max_drawdown_pct);
        assert_eq!(m.trades, 300);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
