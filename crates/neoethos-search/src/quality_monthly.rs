use super::{Trade, evaluation_weekdays};
use anyhow::{Result, anyhow, ensure};
use chrono::{Datelike, Months, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Realized account-currency PnL by UTC EXIT month. Returns and drawdown use
/// percent units; win_rate is a fraction. Undefined ratios stay None. The
/// drawdown resets at each month opening and measures closed-trade balance,
/// excluding intratrade floating losses. Existing scorer inputs are unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyStrategyMetricsV1 {
    pub month_utc: String,
    pub period_start_ms: i64,
    pub period_end_ms: i64,
    pub is_partial: bool,
    pub closed_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: Option<f64>,
    pub starting_balance: f64,
    pub ending_balance: f64,
    pub net_profit: f64,
    pub return_pct: Option<f64>,
    pub gross_profit: f64,
    /// Positive magnitude of realized losses.
    pub gross_loss: f64,
    /// None when there are no losses; gross_profit/gross_loss remain visible.
    pub profit_factor: Option<f64>,
    pub max_closed_balance_drawdown_money: f64,
    pub max_closed_balance_drawdown_pct: Option<f64>,
}

/// One row for EVERY calendar month touched by the supplied evaluation span.
/// Charges already included in Trade.pnl are counted once. Historical reports
/// without the evaluated span must label their trade-only interval separately.
pub fn monthly_closed_trade_metrics_v1(
    trades: &[Trade],
    initial_balance: f64,
    start_ms: i64,
    end_ms: i64,
) -> Result<Vec<MonthlyStrategyMetricsV1>> {
    evaluation_weekdays(start_ms, end_ms)?;
    ensure!(
        initial_balance.is_finite() && initial_balance > 0.0,
        "monthly ledger requires a positive finite initial balance"
    );
    for trade in trades {
        ensure!(
            trade.pnl.is_finite()
                && trade.entry_time >= start_ms
                && trade
                    .exit_time
                    .is_some_and(|exit| exit >= trade.entry_time && exit <= end_ms),
            "monthly ledger contains non-finite PnL or a trade outside the evaluation interval"
        );
    }
    let mut ordered = trades.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|trade| (trade.exit_time, trade.entry_time));
    let mut pending = ordered.into_iter().peekable();
    let mut month = Utc
        .timestamp_millis_opt(start_ms)
        .single()
        .unwrap()
        .date_naive()
        .with_day(1)
        .unwrap();
    let mut balance = initial_balance;
    let mut rows = Vec::new();
    loop {
        let next = month
            .checked_add_months(Months::new(1))
            .ok_or_else(|| anyhow!("monthly ledger calendar overflow"))?;
        let boundary = |date: chrono::NaiveDate| {
            Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
                .timestamp_millis()
        };
        let month_start = boundary(month);
        let next_start = boundary(next);
        let opening = balance;
        let mut row = MonthlyStrategyMetricsV1 {
            month_utc: format!("{:04}-{:02}", month.year(), month.month()),
            period_start_ms: start_ms.max(month_start),
            period_end_ms: end_ms.min(next_start - 1),
            is_partial: start_ms > month_start || end_ms < next_start - 1,
            starting_balance: opening,
            ending_balance: opening,
            net_profit: 0.0,
            return_pct: None,
            closed_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            gross_profit: 0.0,
            gross_loss: 0.0,
            win_rate: None,
            profit_factor: None,
            max_closed_balance_drawdown_money: 0.0,
            max_closed_balance_drawdown_pct: (opening > 0.0).then_some(0.0),
        };
        let mut peak = opening;
        while pending
            .peek()
            .is_some_and(|trade| trade.exit_time.unwrap() < next_start)
        {
            let trade = pending.next().unwrap();
            row.closed_trades += 1;
            row.net_profit += trade.pnl;
            if trade.pnl > 0.0 {
                row.winning_trades += 1;
                row.gross_profit += trade.pnl;
            } else if trade.pnl < 0.0 {
                row.losing_trades += 1;
                row.gross_loss -= trade.pnl;
            }
            balance += trade.pnl;
            peak = peak.max(balance);
            row.max_closed_balance_drawdown_money =
                row.max_closed_balance_drawdown_money.max(peak - balance);
            if peak > 0.0 {
                row.max_closed_balance_drawdown_pct = Some(
                    row.max_closed_balance_drawdown_pct
                        .unwrap_or(0.0)
                        .max((peak - balance) / peak * 100.0),
                )
                .filter(|v| v.is_finite());
            }
        }
        ensure!(
            [
                balance,
                row.net_profit,
                row.gross_profit,
                row.gross_loss,
                row.max_closed_balance_drawdown_money
            ]
            .iter()
            .all(|v| v.is_finite()),
            "monthly ledger money overflow"
        );
        row.ending_balance = balance;
        row.return_pct = (opening > 0.0)
            .then(|| row.net_profit / opening * 100.0)
            .filter(|v| v.is_finite());
        row.win_rate =
            (row.closed_trades > 0).then(|| row.winning_trades as f64 / row.closed_trades as f64);
        row.profit_factor = (row.gross_loss > 0.0)
            .then(|| row.gross_profit / row.gross_loss)
            .filter(|v| v.is_finite());
        rows.push(row);
        if next_start > end_ms {
            break;
        }
        month = next;
    }
    Ok(rows)
}
