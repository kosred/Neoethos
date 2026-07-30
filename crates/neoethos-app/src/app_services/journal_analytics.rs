//! The part of a trading journal that answers questions.
//!
//! [`journal_store`](super::journal_store) already keeps a proper round-trip
//! record — entry and exit price and time, side, lots, commission, swap, net —
//! and [`journal_stats`](super::journal_stats) computes the headline figures.
//! What was missing is everything between: a total tells you the account lost
//! money, it does not tell you *where*.
//!
//! Three things are derived here.
//!
//! **Per trade, in units that compare.** Money depends on position size, so two
//! trades of the same quality look different if one was 0.1 lots and the other
//! 1.0. Pips and R-multiples do not. R also needs the risk taken, which the
//! broker record does not carry, so it is reconstructed from the loss on the
//! losing trades — the stop is what a loser paid.
//!
//! **Excursion.** How far a trade went in favour before it closed is the whole
//! "the profit was there and it went away" question, and the broker never
//! reports it. It is recovered by replaying the stored price series across the
//! trade's own window, so a live trade gets the same MFE/MAE the backtest
//! reports for its trades.
//!
//! **Breakdowns.** By symbol, by hour, by weekday, by direction. A single hour
//! histogram settles questions like "is it only trading the London session"
//! that no amount of reasoning about the config can.

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::journal_store::ClosedTrade;

/// One trade, with everything the raw record leaves implicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DerivedTrade {
    pub position_id: i64,
    pub symbol: String,
    pub side: String,
    pub lots: f64,
    pub entry_ts_ms: Option<i64>,
    pub exit_ts_ms: Option<i64>,
    pub net_profit: f64,
    /// Hours held. `None` when either timestamp is missing.
    pub duration_hours: Option<f64>,
    /// Price move in the trade's favour, signed by direction.
    pub pips: Option<f64>,
    /// Net profit in units of the risk taken. `None` until a risk estimate
    /// exists for this symbol — see [`estimate_risk_per_lot`].
    pub r_multiple: Option<f64>,
    /// Best and worst the trade ever was, in pips, recovered from the price
    /// series. `None` when no bars cover the window.
    pub mfe_pips: Option<f64>,
    pub mae_pips: Option<f64>,
    /// Of the favourable excursion, how much was actually kept. Negative when
    /// a trade that was ahead closed at a loss.
    pub capture_ratio: Option<f64>,
    /// UTC hour and weekday of entry, for the breakdowns.
    pub entry_hour_utc: Option<u32>,
    pub entry_weekday: Option<String>,
}

/// A group of trades and what they did, for one breakdown bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BucketSummary {
    pub bucket: String,
    pub trades: usize,
    pub wins: usize,
    pub win_rate_pct: f64,
    pub net_profit: f64,
    /// Net profit per trade — the figure that says whether this bucket is worth
    /// trading at all, which a total hides when the counts differ wildly.
    pub expectancy: f64,
    pub net_pips: f64,
}

/// The whole journal, sliced the ways that locate a problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JournalAnalytics {
    pub trades: Vec<DerivedTrade>,
    pub by_symbol: Vec<BucketSummary>,
    pub by_hour_utc: Vec<BucketSummary>,
    pub by_weekday: Vec<BucketSummary>,
    pub by_side: Vec<BucketSummary>,
    /// Mean favourable excursion, mean kept, over trades that have both.
    pub avg_mfe_pips: Option<f64>,
    pub avg_capture_ratio: Option<f64>,
    /// Hours in which the account never traded. Directly answers "is it only
    /// active in one session" without inferring it from config.
    pub inactive_hours_utc: Vec<u32>,
}

/// Bars covering a trade window, supplied by the caller so this module stays
/// free of storage concerns and testable without a store.
pub trait PriceWindow {
    /// Highs and lows between `from_ms` and `to_ms` inclusive, in price terms.
    fn window(&self, symbol: &str, from_ms: i64, to_ms: i64) -> Option<(Vec<f64>, Vec<f64>)>;
    /// One pip in price terms for this symbol.
    fn pip_size(&self, symbol: &str) -> Option<f64>;
}

/// The risk a stop represents, per lot, inferred from the trades themselves.
///
/// A broker fill carries no stop distance, so R cannot be read off a record.
/// But a strategy that stops out repeatedly pays the same amount each time,
/// scaled by size — so the median loss per lot across losing trades estimates
/// the risk without needing the strategy's parameters. The median rather than
/// the mean because a trade closed early for other reasons should not move it.
///
/// `None` when there are no losses to learn from; R is then reported as `None`
/// rather than guessed, since an R computed against a made-up denominator is
/// worse than no R at all.
pub fn estimate_risk_per_lot(trades: &[ClosedTrade]) -> Option<f64> {
    let mut losses: Vec<f64> = trades
        .iter()
        .filter(|t| t.net_profit < 0.0 && t.lots > 0.0)
        .map(|t| -t.net_profit / t.lots)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if losses.is_empty() {
        return None;
    }
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(losses[losses.len() / 2])
}

fn is_long(side: &str) -> bool {
    side.trim().eq_ignore_ascii_case("BUY")
}

fn weekday_name(ts_ms: i64) -> Option<String> {
    DateTime::<Utc>::from_timestamp_millis(ts_ms).map(|dt| dt.weekday().to_string())
}

/// Turn one stored trade into a derived one.
pub fn derive_trade(
    trade: &ClosedTrade,
    risk_per_lot: Option<f64>,
    prices: Option<&dyn PriceWindow>,
) -> DerivedTrade {
    let duration_hours = match (trade.entry_ts_ms, trade.exit_ts_ms) {
        (Some(entry), Some(exit)) if exit >= entry => {
            Some((exit - entry) as f64 / 3_600_000.0)
        }
        _ => None,
    };
    let pip_size = prices.and_then(|p| p.pip_size(&trade.symbol));
    let long = is_long(&trade.side);
    let pips = match (trade.entry_price, trade.exit_price, pip_size) {
        (Some(entry), Some(exit), Some(pip)) if pip > 0.0 => {
            Some(if long { (exit - entry) / pip } else { (entry - exit) / pip })
        }
        _ => None,
    };
    let r_multiple = risk_per_lot
        .filter(|risk| *risk > 0.0 && trade.lots > 0.0)
        .map(|risk| trade.net_profit / (risk * trade.lots));

    // Excursion, replayed from the price series the account actually traded.
    let (mfe_pips, mae_pips) = match (
        trade.entry_ts_ms,
        trade.exit_ts_ms,
        trade.entry_price,
        pip_size,
        prices,
    ) {
        (Some(entry_ts), Some(exit_ts), Some(entry_px), Some(pip), Some(prices)) if pip > 0.0 => {
            match prices.window(&trade.symbol, entry_ts, exit_ts) {
                Some((highs, lows)) if !highs.is_empty() => {
                    let best = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let worst = lows.iter().cloned().fold(f64::INFINITY, f64::min);
                    if long {
                        (
                            Some((best - entry_px) / pip),
                            Some((entry_px - worst) / pip),
                        )
                    } else {
                        (
                            Some((entry_px - worst) / pip),
                            Some((best - entry_px) / pip),
                        )
                    }
                }
                _ => (None, None),
            }
        }
        _ => (None, None),
    };
    // Only meaningful when the trade was actually ahead at some point.
    let capture_ratio = match (pips, mfe_pips) {
        (Some(realised), Some(mfe)) if mfe > 0.0 => Some(realised / mfe),
        _ => None,
    };

    let entry_hour_utc = trade
        .entry_ts_ms
        .and_then(DateTime::<Utc>::from_timestamp_millis)
        .map(|dt| dt.hour());

    DerivedTrade {
        position_id: trade.position_id,
        symbol: trade.symbol.clone(),
        side: trade.side.clone(),
        lots: trade.lots,
        entry_ts_ms: trade.entry_ts_ms,
        exit_ts_ms: trade.exit_ts_ms,
        net_profit: trade.net_profit,
        duration_hours,
        pips,
        r_multiple,
        mfe_pips,
        mae_pips,
        capture_ratio,
        entry_hour_utc,
        entry_weekday: trade.entry_ts_ms.and_then(weekday_name),
    }
}

fn summarise(bucket: String, group: &[&DerivedTrade]) -> BucketSummary {
    let trades = group.len();
    let wins = group.iter().filter(|t| t.net_profit > 0.0).count();
    let net_profit: f64 = group.iter().map(|t| t.net_profit).sum();
    let net_pips: f64 = group.iter().filter_map(|t| t.pips).sum();
    BucketSummary {
        bucket,
        trades,
        wins,
        win_rate_pct: if trades > 0 {
            100.0 * wins as f64 / trades as f64
        } else {
            0.0
        },
        net_profit,
        expectancy: if trades > 0 {
            net_profit / trades as f64
        } else {
            0.0
        },
        net_pips,
    }
}

fn group_by<K, F>(trades: &[DerivedTrade], key: F) -> Vec<BucketSummary>
where
    K: Ord + std::fmt::Display,
    F: Fn(&DerivedTrade) -> Option<K>,
{
    let mut buckets: BTreeMap<K, Vec<&DerivedTrade>> = BTreeMap::new();
    for trade in trades {
        if let Some(k) = key(trade) {
            buckets.entry(k).or_default().push(trade);
        }
    }
    buckets
        .into_iter()
        .map(|(k, group)| summarise(k.to_string(), &group))
        .collect()
}

/// Derive every trade and slice the result.
pub fn analyse(trades: &[ClosedTrade], prices: Option<&dyn PriceWindow>) -> JournalAnalytics {
    let risk_per_lot = estimate_risk_per_lot(trades);
    let derived: Vec<DerivedTrade> = trades
        .iter()
        .map(|t| derive_trade(t, risk_per_lot, prices))
        .collect();

    let mean = |values: Vec<f64>| -> Option<f64> {
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    let avg_mfe_pips = mean(derived.iter().filter_map(|t| t.mfe_pips).collect());
    let avg_capture_ratio = mean(derived.iter().filter_map(|t| t.capture_ratio).collect());

    let traded_hours: std::collections::HashSet<u32> =
        derived.iter().filter_map(|t| t.entry_hour_utc).collect();
    let inactive_hours_utc: Vec<u32> = if traded_hours.is_empty() {
        Vec::new()
    } else {
        (0..24u32).filter(|h| !traded_hours.contains(h)).collect()
    };

    JournalAnalytics {
        by_symbol: group_by(&derived, |t| Some(t.symbol.clone())),
        // Zero-padded so string ordering is clock ordering: "09" before "10".
        by_hour_utc: group_by(&derived, |t| t.entry_hour_utc.map(|h| format!("{h:02}"))),
        by_weekday: group_by(&derived, |t| t.entry_weekday.clone()),
        by_side: group_by(&derived, |t| Some(t.side.clone())),
        avg_mfe_pips,
        avg_capture_ratio,
        inactive_hours_utc,
        trades: derived,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedPrices {
        pip: f64,
        highs: Vec<f64>,
        lows: Vec<f64>,
    }

    impl PriceWindow for FixedPrices {
        fn window(&self, _symbol: &str, _from: i64, _to: i64) -> Option<(Vec<f64>, Vec<f64>)> {
            Some((self.highs.clone(), self.lows.clone()))
        }
        fn pip_size(&self, _symbol: &str) -> Option<f64> {
            Some(self.pip)
        }
    }

    fn trade(id: i64, side: &str, entry: f64, exit: f64, net: f64, entry_ts: i64) -> ClosedTrade {
        ClosedTrade {
            schema_version: 2,
            recorded_at_unix_ms: entry_ts + 3_600_000,
            position_id: id,
            symbol: "EURUSD".to_string(),
            side: side.to_string(),
            lots: 1.0,
            account_id: Some("acct".to_string()),
            entry_ts_ms: Some(entry_ts),
            entry_price: Some(entry),
            exit_ts_ms: Some(entry_ts + 3_600_000),
            exit_price: Some(exit),
            gross_profit: net,
            commission: 0.0,
            swap: 0.0,
            net_profit: net,
            balance_after: None,
        }
    }

    /// The question the journal exists to answer: the trade was 30 pips ahead
    /// and closed 10 up, so two thirds of the move was handed back. Nothing in
    /// the broker's record says that.
    #[test]
    fn a_long_reports_how_much_of_its_favourable_move_it_kept() {
        let prices = FixedPrices {
            pip: 0.0001,
            highs: vec![1.1030, 1.1015],
            lows: vec![1.0995, 1.0990],
        };
        let d = derive_trade(
            &trade(1, "BUY", 1.1000, 1.1010, 100.0, 1_700_000_000_000),
            Some(200.0),
            Some(&prices),
        );
        assert!((d.pips.unwrap() - 10.0).abs() < 1e-6);
        assert!((d.mfe_pips.unwrap() - 30.0).abs() < 1e-6);
        assert!((d.mae_pips.unwrap() - 10.0).abs() < 1e-6);
        assert!((d.capture_ratio.unwrap() - 1.0 / 3.0).abs() < 1e-6);
        assert!((d.r_multiple.unwrap() - 0.5).abs() < 1e-9);
        assert!((d.duration_hours.unwrap() - 1.0).abs() < 1e-9);
    }

    /// Direction has to invert the excursion, or every short reads as a
    /// disaster that was never in profit.
    #[test]
    fn a_short_measures_excursion_downward() {
        let prices = FixedPrices {
            pip: 0.0001,
            highs: vec![1.1010],
            lows: vec![1.0970],
        };
        let d = derive_trade(
            &trade(2, "SELL", 1.1000, 1.0990, 100.0, 1_700_000_000_000),
            None,
            Some(&prices),
        );
        assert!((d.pips.unwrap() - 10.0).abs() < 1e-6);
        // Price fell 30 pips below entry — favourable for a short.
        assert!((d.mfe_pips.unwrap() - 30.0).abs() < 1e-6);
        // And rose 10 above it — adverse.
        assert!((d.mae_pips.unwrap() - 10.0).abs() < 1e-6);
        // No losses to learn a risk from, so R is withheld rather than invented.
        assert!(d.r_multiple.is_none());
    }

    #[test]
    fn risk_per_lot_is_the_median_loss_and_ignores_the_winners() {
        let trades = vec![
            trade(1, "BUY", 1.1, 1.1, -200.0, 0),
            trade(2, "BUY", 1.1, 1.1, -190.0, 0),
            trade(3, "BUY", 1.1, 1.1, -210.0, 0),
            trade(4, "BUY", 1.1, 1.1, 5000.0, 0),
        ];
        assert_eq!(estimate_risk_per_lot(&trades), Some(200.0));
        assert_eq!(estimate_risk_per_lot(&[]), None);
    }

    /// The London-session question, answered by counting rather than reasoning.
    #[test]
    fn the_hours_the_account_never_traded_are_listed() {
        let hour = 3_600_000i64;
        let at = |h: i64| 1_700_000_000_000 - (1_700_000_000_000 % (24 * hour)) + h * hour;
        let trades = vec![
            trade(1, "BUY", 1.1000, 1.1010, 100.0, at(8)),
            trade(2, "BUY", 1.1000, 1.0990, -100.0, at(9)),
            trade(3, "SELL", 1.1000, 1.0990, 100.0, at(9)),
        ];
        let analytics = analyse(&trades, None);
        assert!(!analytics.inactive_hours_utc.contains(&8));
        assert!(!analytics.inactive_hours_utc.contains(&9));
        assert!(analytics.inactive_hours_utc.contains(&15));
        assert_eq!(analytics.inactive_hours_utc.len(), 22);

        // Hour 9 holds two trades that cancel out; the bucket must show both and
        // an expectancy of zero rather than being read as "no edge, no data".
        let nine = analytics
            .by_hour_utc
            .iter()
            .find(|b| b.bucket == "09")
            .expect("hour 09 bucket");
        assert_eq!(nine.trades, 2);
        assert_eq!(nine.wins, 1);
        assert!((nine.net_profit).abs() < 1e-9);

        // Zero-padding keeps the buckets in clock order.
        let order: Vec<&str> = analytics
            .by_hour_utc
            .iter()
            .map(|b| b.bucket.as_str())
            .collect();
        assert_eq!(order, vec!["08", "09"]);

        assert_eq!(analytics.by_side.len(), 2);
        assert_eq!(analytics.by_symbol.len(), 1);
    }

    /// Without a price source the journal still works; it simply reports less,
    /// rather than reporting zeros that look like measurements.
    #[test]
    fn excursion_is_absent_not_zero_when_no_prices_are_available() {
        let d = derive_trade(
            &trade(1, "BUY", 1.1000, 1.1010, 100.0, 1_700_000_000_000),
            Some(200.0),
            None,
        );
        assert!(d.mfe_pips.is_none());
        assert!(d.capture_ratio.is_none());
        assert!(d.pips.is_none(), "pips need a pip size to be meaningful");
        assert!(d.r_multiple.is_some(), "R needs no price series");
    }
}
