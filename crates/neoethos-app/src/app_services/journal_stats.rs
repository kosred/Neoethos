//! Pure performance-stats engine for the trade journal.
//!
//! Computes the professional metric set (per the researched myfxbook /
//! MT5 / FTMO / QuantStats standard) from the two raw artifacts the
//! journal persists: a closed-trade list (→ P/L + trade-distribution
//! metrics) and an equity series (→ drawdown / Sharpe / recovery).
//!
//! Pure + fully unit-testable: no I/O, no panics, no div-by-zero.
//! Undefined ratios (profit factor with no losses, Sharpe with <2
//! samples) are `None` → serialize as `null`, never `inf`/`NaN`.

use crate::app_services::journal_store::{ClosedTrade, EquitySample};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct JournalStats {
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub breakeven: usize,
    pub win_rate_pct: f64,
    pub net_profit: f64,
    pub gross_profit: f64,
    /// Sum of losing trades (<= 0).
    pub gross_loss: f64,
    /// `gross_profit / |gross_loss|`; `None` when there are no losses.
    pub profit_factor: Option<f64>,
    pub avg_win: f64,
    /// Mean loss (<= 0).
    pub avg_loss: f64,
    /// `avg_win / |avg_loss|`; `None` when there are no losses.
    pub payoff_ratio: Option<f64>,
    /// Net profit per trade.
    pub expectancy: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    // ── Equity-series derived ──
    pub max_drawdown_abs: f64,
    pub max_drawdown_pct: f64,
    /// `net_profit / |max_drawdown_abs|`; `None` when there's no drawdown.
    pub recovery_factor: Option<f64>,
    /// Per-sample Sharpe (mean/stddev of equity returns); caller
    /// annualizes. `None` with fewer than 2 usable returns.
    pub sharpe: Option<f64>,
}

/// Compute the full stats bundle. Defensive: empty inputs → all-zero
/// stats; every ratio that could divide by zero is guarded and returns
/// `None` rather than `inf`/`NaN`.
pub fn compute_stats(trades: &[ClosedTrade], equity: &[EquitySample]) -> JournalStats {
    let mut s = JournalStats {
        total_trades: trades.len(),
        ..Default::default()
    };

    let mut cur_win_streak = 0usize;
    let mut cur_loss_streak = 0usize;
    let mut win_sum = 0.0f64;
    let mut loss_sum = 0.0f64; // negative
    let mut largest_win = f64::NEG_INFINITY;
    let mut largest_loss = f64::INFINITY;

    for t in trades {
        let p = t.net_profit;
        s.net_profit += p;
        if p > 0.0 {
            s.wins += 1;
            s.gross_profit += p;
            win_sum += p;
            cur_win_streak += 1;
            cur_loss_streak = 0;
            s.max_consecutive_wins = s.max_consecutive_wins.max(cur_win_streak);
            largest_win = largest_win.max(p);
        } else if p < 0.0 {
            s.losses += 1;
            s.gross_loss += p;
            loss_sum += p;
            cur_loss_streak += 1;
            cur_win_streak = 0;
            s.max_consecutive_losses = s.max_consecutive_losses.max(cur_loss_streak);
            largest_loss = largest_loss.min(p);
        } else {
            s.breakeven += 1;
            cur_win_streak = 0;
            cur_loss_streak = 0;
        }
    }

    if !trades.is_empty() {
        let n = trades.len() as f64;
        s.win_rate_pct = (s.wins as f64 / n) * 100.0;
        s.expectancy = s.net_profit / n;
    }
    if s.wins > 0 {
        s.avg_win = win_sum / s.wins as f64;
    }
    if s.losses > 0 {
        s.avg_loss = loss_sum / s.losses as f64; // negative
    }
    if s.gross_loss != 0.0 {
        s.profit_factor = Some(s.gross_profit / s.gross_loss.abs());
    }
    if s.losses > 0 && s.avg_loss != 0.0 {
        s.payoff_ratio = Some(s.avg_win / s.avg_loss.abs());
    }
    s.largest_win = if largest_win.is_finite() { largest_win } else { 0.0 };
    s.largest_loss = if largest_loss.is_finite() {
        largest_loss
    } else {
        0.0
    };

    // ── Equity-derived ──
    if !equity.is_empty() {
        let mut peak = f64::NEG_INFINITY;
        let mut max_dd_abs = 0.0f64;
        let mut max_dd_pct = 0.0f64;
        for e in equity {
            peak = peak.max(e.equity);
            if peak.is_finite() && peak > 0.0 {
                let dd = peak - e.equity;
                if dd > max_dd_abs {
                    max_dd_abs = dd;
                }
                let dd_pct = (dd / peak) * 100.0;
                if dd_pct > max_dd_pct {
                    max_dd_pct = dd_pct;
                }
            }
        }
        s.max_drawdown_abs = max_dd_abs;
        s.max_drawdown_pct = max_dd_pct;
        if max_dd_abs > 0.0 {
            s.recovery_factor = Some(s.net_profit / max_dd_abs);
        }

        // Sharpe over per-sample equity returns.
        let mut rets: Vec<f64> = Vec::new();
        for w in equity.windows(2) {
            let prev = w[0].equity;
            let cur = w[1].equity;
            if prev.abs() > f64::EPSILON {
                rets.push((cur - prev) / prev);
            }
        }
        if rets.len() >= 2 {
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var =
                rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() as f64 - 1.0);
            let sd = var.sqrt();
            if sd > f64::EPSILON {
                s.sharpe = Some(mean / sd);
            }
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::journal_store::{ClosedTrade, EquitySample};

    fn t(net: f64) -> ClosedTrade {
        ClosedTrade {
            schema_version: 1,
            recorded_at_unix_ms: 0,
            position_id: 0,
            symbol: "EURUSD".into(),
            side: "BUY".into(),
            lots: 0.1,
            entry_ts_ms: None,
            entry_price: None,
            exit_ts_ms: None,
            exit_price: None,
            gross_profit: net,
            commission: 0.0,
            swap: 0.0,
            net_profit: net,
            balance_after: None,
            account_id: None,
        }
    }

    #[test]
    fn empty_inputs_are_all_zero_no_panic() {
        let s = compute_stats(&[], &[]);
        assert_eq!(s.total_trades, 0);
        assert_eq!(s.net_profit, 0.0);
        assert!(s.profit_factor.is_none());
        assert!(s.sharpe.is_none());
    }

    #[test]
    fn basic_trade_stats() {
        let trades = [t(10.0), t(-5.0), t(3.0)];
        let s = compute_stats(&trades, &[]);
        assert_eq!(s.total_trades, 3);
        assert_eq!(s.wins, 2);
        assert_eq!(s.losses, 1);
        assert!((s.net_profit - 8.0).abs() < 1e-9);
        assert!((s.gross_profit - 13.0).abs() < 1e-9);
        assert!((s.gross_loss + 5.0).abs() < 1e-9);
        assert!((s.profit_factor.unwrap() - 2.6).abs() < 1e-9);
        assert!((s.win_rate_pct - (2.0 / 3.0 * 100.0)).abs() < 1e-9);
        assert_eq!(s.largest_win, 10.0);
        assert_eq!(s.largest_loss, -5.0);
        assert_eq!(s.max_consecutive_wins, 1);
        assert_eq!(s.max_consecutive_losses, 1);
    }

    #[test]
    fn no_losses_leaves_profit_factor_none() {
        let s = compute_stats(&[t(5.0), t(7.0)], &[]);
        assert!(s.profit_factor.is_none());
        assert!(s.payoff_ratio.is_none());
        assert_eq!(s.max_consecutive_wins, 2);
    }

    #[test]
    fn drawdown_from_equity() {
        let eq = |ts: i64, e: f64| EquitySample {
            ts_ms: ts,
            balance: e,
            equity: e,
            account_id: None,
        };
        let s = compute_stats(&[], &[eq(1, 100.0), eq(2, 110.0), eq(3, 90.0), eq(4, 120.0)]);
        assert!((s.max_drawdown_abs - 20.0).abs() < 1e-9);
        assert!((s.max_drawdown_pct - (20.0 / 110.0 * 100.0)).abs() < 1e-9);
    }
}
