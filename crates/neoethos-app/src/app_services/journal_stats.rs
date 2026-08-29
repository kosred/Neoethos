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

/// Max drawdown measured from the cumulative P&L of a **scoped set of trades**,
/// instead of from an account equity curve.
///
/// # Why this exists (2026-08-09, item #197)
///
/// [`JournalStats::max_drawdown_pct`] is computed peak-to-trough over
/// [`EquitySample`]s, and an equity sample is an ACCOUNT-level fact: one number
/// for the whole cTrader account, moved by every engine and every manual order
/// on it. That is the right measurement for "how did the account do" and the
/// wrong one for "how did THIS strategy do".
///
/// The demo forward-test gate needs the second question answered: it compares
/// the live figure against ONE strategy's `quality.json`, whose
/// `max_drawdown_pct` is that strategy's own equity curve. Feeding it the
/// account union curve fails a qualified strategy whenever another engine was
/// drawing down at the same time, and passes an unqualified one on a quiet
/// account. Neither error is visible in the result.
///
/// An [`EquitySample`] cannot be scoped to a symbol — the account has one
/// balance, not one per instrument — so the strategy's curve is RECONSTRUCTED
/// from the scoped trades' own realised P&L, laid on top of a baseline account
/// equity that supplies the percentage denominator.
///
/// **What this is not.** It is a *closed-trade* curve: it moves only when a
/// trade closes, so intra-trade excursion is invisible and the figure is a
/// lower bound on the true peak-to-trough. The backtest number it is compared
/// against is built the same way, which is exactly why it is the comparable one.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeDrawdown {
    /// Peak-to-trough of `baseline_equity + cumulative net P&L`, in account currency.
    pub max_drawdown_abs: f64,
    /// The same, as a percent of the running peak.
    pub max_drawdown_pct: f64,
    /// The denominator: account equity the scoped curve was laid on top of.
    pub baseline_equity: f64,
    /// Trades that actually moved the curve.
    pub trades_used: usize,
    /// Trades discarded because their `net_profit` was NaN/infinite. Counted
    /// rather than dropped silently — a discard on a decision path is a fact
    /// the caller has to be able to see.
    pub trades_skipped_non_finite: usize,
}

/// Reconstruct a scoped strategy's drawdown from its own closed trades.
///
/// `trades` are re-sorted by effective timestamp internally, so the caller's
/// ordering is never load-bearing. `baseline_equity` is the account equity the
/// curve starts from and the percentage denominator.
///
/// `None` when there is nothing measurable — no trades, or a baseline that is
/// not a positive finite number. **`None` means "not measured", never "zero
/// drawdown"**; a caller on a gating path must treat it as a refusal to
/// measure, not as a pass.
pub fn max_drawdown_from_trade_pnl(
    trades: &[ClosedTrade],
    baseline_equity: f64,
) -> Option<TradeDrawdown> {
    if trades.is_empty() || !baseline_equity.is_finite() || baseline_equity <= 0.0 {
        return None;
    }
    let mut ordered: Vec<(i64, f64)> = trades
        .iter()
        .map(|t| (t.effective_ts_ms(), t.net_profit))
        .collect();
    ordered.sort_by_key(|(ts, _)| *ts);

    let mut equity = baseline_equity;
    let mut peak = baseline_equity;
    let mut max_dd_abs = 0.0f64;
    let mut max_dd_pct = 0.0f64;
    let mut used = 0usize;
    let mut skipped = 0usize;
    for (_, pnl) in &ordered {
        if !pnl.is_finite() {
            skipped += 1;
            continue;
        }
        used += 1;
        equity += *pnl;
        if equity > peak {
            peak = equity;
        }
        let dd = peak - equity;
        if dd > max_dd_abs {
            max_dd_abs = dd;
        }
        if peak > 0.0 {
            let dd_pct = (dd / peak) * 100.0;
            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
            }
        }
    }
    if used == 0 {
        return None;
    }
    Some(TradeDrawdown {
        max_drawdown_abs: max_dd_abs,
        max_drawdown_pct: max_dd_pct,
        baseline_equity,
        trades_used: used,
        trades_skipped_non_finite: skipped,
    })
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
    s.largest_win = if largest_win.is_finite() {
        largest_win
    } else {
        0.0
    };
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
            environment: None,
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
            environment: None,
        };
        let s = compute_stats(
            &[],
            &[eq(1, 100.0), eq(2, 110.0), eq(3, 90.0), eq(4, 120.0)],
        );
        assert!((s.max_drawdown_abs - 20.0).abs() < 1e-9);
        assert!((s.max_drawdown_pct - (20.0 / 110.0 * 100.0)).abs() < 1e-9);
    }

    fn t_at(net: f64, exit_ms: i64) -> ClosedTrade {
        let mut trade = t(net);
        trade.exit_ts_ms = Some(exit_ms);
        trade.recorded_at_unix_ms = exit_ms;
        trade
    }

    #[test]
    fn trade_scoped_drawdown_walks_the_scoped_curve_only() {
        // 1000 → 1100 → 1010 → 1210. Peak 1100, trough 1010 → 90 abs, 8.18%.
        let trades = [t_at(100.0, 1), t_at(-90.0, 2), t_at(200.0, 3)];
        let d = max_drawdown_from_trade_pnl(&trades, 1000.0).expect("measurable");
        assert!((d.max_drawdown_abs - 90.0).abs() < 1e-9);
        assert!((d.max_drawdown_pct - (90.0 / 1100.0 * 100.0)).abs() < 1e-9);
        assert_eq!(d.trades_used, 3);
        assert_eq!(d.trades_skipped_non_finite, 0);
        assert!((d.baseline_equity - 1000.0).abs() < 1e-9);
    }

    /// THE DEFECT #197 CLOSES. Another engine on the same account drew the
    /// ACCOUNT curve down 25%; this strategy's own trades never lost more than
    /// ~1.8%. The account-curve figure refuses a strategy that qualified.
    #[test]
    fn another_engines_drawdown_does_not_land_on_this_strategy() {
        let eq = |ts: i64, e: f64| EquitySample {
            ts_ms: ts,
            balance: e,
            equity: e,
            account_id: None,
            environment: Some("Demo".to_string()),
        };
        // This strategy: +20, -20, +20 on a 1000 baseline → 1.96% worst.
        let trades = [t_at(20.0, 1), t_at(-20.0, 2), t_at(20.0, 3)];
        // The account, meanwhile, went 1000 → 1000 → 750 (a different engine).
        let account = [eq(1, 1000.0), eq(2, 1000.0), eq(3, 750.0)];

        let account_view = compute_stats(&trades, &account);
        assert!(
            account_view.max_drawdown_pct > 24.0,
            "the account curve shows the union: {}",
            account_view.max_drawdown_pct
        );

        let scoped = max_drawdown_from_trade_pnl(&trades, 1000.0).expect("measurable");
        assert!(
            scoped.max_drawdown_pct < 2.0,
            "this strategy's own drawdown is small: {}",
            scoped.max_drawdown_pct
        );
    }

    #[test]
    fn ordering_is_not_the_callers_responsibility() {
        let forward = [t_at(100.0, 1), t_at(-90.0, 2), t_at(200.0, 3)];
        let shuffled = [t_at(200.0, 3), t_at(100.0, 1), t_at(-90.0, 2)];
        let a = max_drawdown_from_trade_pnl(&forward, 1000.0).expect("measurable");
        let b = max_drawdown_from_trade_pnl(&shuffled, 1000.0).expect("measurable");
        assert!((a.max_drawdown_pct - b.max_drawdown_pct).abs() < 1e-12);
    }

    /// Unmeasurable must be distinguishable from "no drawdown" — a gate that
    /// reads 0.0 for "I could not measure" is a gate that admits on ignorance.
    #[test]
    fn an_unusable_baseline_refuses_to_measure_rather_than_reporting_zero() {
        let trades = [t_at(100.0, 1), t_at(-90.0, 2)];
        assert!(max_drawdown_from_trade_pnl(&trades, 0.0).is_none());
        assert!(max_drawdown_from_trade_pnl(&trades, -5.0).is_none());
        assert!(max_drawdown_from_trade_pnl(&trades, f64::NAN).is_none());
        assert!(max_drawdown_from_trade_pnl(&[], 1000.0).is_none());
    }

    #[test]
    fn non_finite_pnl_is_counted_not_silently_dropped() {
        let trades = [t_at(100.0, 1), t_at(f64::NAN, 2), t_at(-50.0, 3)];
        let d = max_drawdown_from_trade_pnl(&trades, 1000.0).expect("measurable");
        assert_eq!(d.trades_used, 2);
        assert_eq!(d.trades_skipped_non_finite, 1);
        assert!((d.max_drawdown_abs - 50.0).abs() < 1e-9);
    }
}
