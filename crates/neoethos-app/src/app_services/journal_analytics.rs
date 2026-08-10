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
    /// The denominator [`r_multiple`](Self::r_multiple) was divided by, and
    /// where it came from — `"symbol"`, `"all_symbols"` or `"none"`. R is an
    /// ESTIMATE (see [`estimate_risk_per_lot`]); a reader shown a number with
    /// no denominator cannot tell a measured R from an inferred one.
    pub risk_per_lot: Option<f64>,
    pub risk_basis: String,
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

/// How much of the journal each derived figure could actually be computed for.
///
/// Every optional field on a [`DerivedTrade`] is `None` for a reason — no price
/// series for that symbol, no entry timestamp on the row, no losses to infer a
/// stop from. A reader shown a column of blanks cannot tell "the trades never
/// went anywhere" from "the excursion could not be recovered", and those two
/// readings lead to opposite decisions. This counts every one of them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyticsCoverage {
    pub trades_total: usize,
    /// Trades with a pip figure — needs entry price, exit price and a pip size.
    pub with_pips: usize,
    /// Trades whose MFE/MAE was recovered from the stored price series.
    pub with_excursion: usize,
    pub with_r_multiple: usize,
    pub with_duration: usize,
    /// Rows the broker record left without an entry timestamp: no holding time,
    /// no entry-hour bucket, and no window to replay prices over.
    pub missing_entry_time: usize,
    /// Rows that had everything the excursion replay needs EXCEPT bars covering
    /// the window — i.e. the price store, not the journal, is what is missing.
    pub missing_price_series: usize,
    /// The R denominator per symbol, and the fallback used where a symbol had
    /// too few losses of its own to infer one.
    pub risk_per_lot_by_symbol: BTreeMap<String, f64>,
    pub risk_per_lot_all_symbols: Option<f64>,
    /// Symbols that fell back to the all-symbol estimate, and why it matters:
    /// a stop on XAUUSD and a stop on EURUSD are not the same money per lot.
    pub symbols_using_fallback_risk: Vec<String>,
    /// Minimum losing trades a symbol needs before its own estimate is trusted.
    pub min_losses_for_symbol_risk: usize,
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
    /// What could and could not be computed, and why — see [`AnalyticsCoverage`].
    pub coverage: AnalyticsCoverage,
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
/// What can be recovered is what a stop-out costs: the strategy pays roughly the
/// same each time, scaled by size, so a high quantile of loss-per-lot lands in
/// that cluster.
///
/// It has to be a high quantile, not the middle. The operator's own 238 demand
/// trades have a median loss-per-lot of 0.00 — most losers close early, well
/// inside the stop — and dividing by that produced R values in the hundreds of
/// thousands. A number that large is obviously wrong; one merely ten times off
/// would not have been, which is why the statistic is chosen against real data
/// rather than by what reads well.
///
/// This is an estimate of a *typical* stop, and volatility-scaled stops move
/// per trade, so R here compares trades within a strategy rather than measuring
/// each one's own risk exactly.
///
/// `None` when there is nothing to learn from — no losses, or a cluster that
/// rounds to zero. An R computed against a denominator near zero is worse than
/// no R at all.
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
    // Nine in ten losses were no worse than this — close to the stop without
    // being the single worst fill, which slippage or a gap can distort.
    let index = ((losses.len() as f64 * 0.9).ceil() as usize)
        .saturating_sub(1)
        .min(losses.len() - 1);
    let estimate = losses[index];
    // A stop that rounds to nothing is not a stop; refuse rather than divide.
    (estimate > 1e-6).then_some(estimate)
}

/// Losing trades a symbol must have of its own before its stop estimate is
/// trusted over the all-symbol one. Below this the quantile is reading noise.
pub const MIN_LOSSES_FOR_SYMBOL_RISK: usize = 5;

/// The stop estimate PER SYMBOL.
///
/// [`estimate_risk_per_lot`] over a mixed journal divides every trade by one
/// number, and a stop is money-per-lot: on the operator's instruments a
/// one-lot XAUUSD stop and a one-lot EURUSD stop differ by more than an order
/// of magnitude. Pooling them makes every R on the smaller instrument look
/// tiny and every R on the larger one look enormous — in the one view built to
/// compare trades across instruments.
///
/// Only symbols with at least [`MIN_LOSSES_FOR_SYMBOL_RISK`] losses of their
/// own appear; the caller falls back to the pooled estimate for the rest and
/// records which symbols those were.
pub fn estimate_risk_per_lot_by_symbol(trades: &[ClosedTrade]) -> BTreeMap<String, f64> {
    let mut by_symbol: BTreeMap<String, Vec<ClosedTrade>> = BTreeMap::new();
    for trade in trades {
        by_symbol
            .entry(trade.symbol.clone())
            .or_default()
            .push(trade.clone());
    }
    by_symbol
        .into_iter()
        .filter_map(|(symbol, rows)| {
            let losses = rows
                .iter()
                .filter(|t| t.net_profit < 0.0 && t.lots > 0.0)
                .count();
            if losses < MIN_LOSSES_FOR_SYMBOL_RISK {
                return None;
            }
            estimate_risk_per_lot(&rows).map(|risk| (symbol, risk))
        })
        .collect()
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
    derive_trade_with_basis(trade, risk_per_lot, "all_symbols", prices)
}

/// As [`derive_trade`], but records WHERE the R denominator came from.
pub fn derive_trade_with_basis(
    trade: &ClosedTrade,
    risk_per_lot: Option<f64>,
    risk_basis: &'static str,
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
        risk_per_lot: r_multiple.and(risk_per_lot),
        risk_basis: if r_multiple.is_some() { risk_basis } else { "none" }.to_string(),
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
///
/// The R denominator is estimated PER SYMBOL where a symbol has enough losses
/// of its own ([`MIN_LOSSES_FOR_SYMBOL_RISK`]), falling back to the pooled
/// estimate elsewhere — a stop is money-per-lot, and pooling XAUUSD with EURUSD
/// makes every R on both wrong. Which basis each trade used is on the trade, and
/// the fallbacks are named in [`JournalAnalytics::coverage`].
pub fn analyse(trades: &[ClosedTrade], prices: Option<&dyn PriceWindow>) -> JournalAnalytics {
    let pooled_risk = estimate_risk_per_lot(trades);
    let per_symbol_risk = estimate_risk_per_lot_by_symbol(trades);

    let mut fallback_symbols: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let derived: Vec<DerivedTrade> = trades
        .iter()
        .map(|t| match per_symbol_risk.get(&t.symbol) {
            Some(risk) => derive_trade_with_basis(t, Some(*risk), "symbol", prices),
            None => {
                if pooled_risk.is_some() {
                    fallback_symbols.insert(t.symbol.clone());
                }
                derive_trade_with_basis(t, pooled_risk, "all_symbols", prices)
            }
        })
        .collect();

    // Every blank, counted and attributed. `missing_price_series` is the case
    // where the journal row had everything the replay needs and the price store
    // did not cover the window — the difference between "the trade never went
    // anywhere" and "we could not see where it went".
    let coverage = AnalyticsCoverage {
        trades_total: derived.len(),
        with_pips: derived.iter().filter(|t| t.pips.is_some()).count(),
        with_excursion: derived.iter().filter(|t| t.mfe_pips.is_some()).count(),
        with_r_multiple: derived.iter().filter(|t| t.r_multiple.is_some()).count(),
        with_duration: derived.iter().filter(|t| t.duration_hours.is_some()).count(),
        missing_entry_time: trades.iter().filter(|t| t.entry_ts_ms.is_none()).count(),
        missing_price_series: trades
            .iter()
            .zip(derived.iter())
            .filter(|(raw, d)| {
                raw.entry_ts_ms.is_some()
                    && raw.exit_ts_ms.is_some()
                    && raw.entry_price.is_some()
                    && d.mfe_pips.is_none()
            })
            .count(),
        risk_per_lot_by_symbol: per_symbol_risk,
        risk_per_lot_all_symbols: pooled_risk,
        symbols_using_fallback_risk: fallback_symbols.into_iter().collect(),
        min_losses_for_symbol_risk: MIN_LOSSES_FOR_SYMBOL_RISK,
    };

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
        coverage,
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
            environment: Some("Demo".to_string()),
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
    fn risk_per_lot_finds_the_stop_and_ignores_the_winners() {
        let trades = vec![
            trade(1, "BUY", 1.1, 1.1, -200.0, 0),
            trade(2, "BUY", 1.1, 1.1, -190.0, 0),
            trade(3, "BUY", 1.1, 1.1, -210.0, 0),
            trade(4, "BUY", 1.1, 1.1, 5000.0, 0),
        ];
        let estimate = estimate_risk_per_lot(&trades).expect("three losses to learn from");
        assert!(
            (190.0..=210.0).contains(&estimate),
            "expected the stop cluster, got {estimate}"
        );
        assert_eq!(estimate_risk_per_lot(&[]), None);
    }

    /// The shape of the operator's real journal: most losers close early, well
    /// inside the stop, and only a minority actually pay it. Taking the middle
    /// of that distribution gives ~0 and turns every R into a number in the
    /// hundreds of thousands — which is how this was caught.
    #[test]
    fn early_exits_do_not_drag_the_stop_estimate_to_zero() {
        let mut trades: Vec<ClosedTrade> = (0..70)
            .map(|i| trade(i, "BUY", 1.1, 1.1, -0.01, 0))
            .collect();
        trades.extend((70..100).map(|i| trade(i, "BUY", 1.1, 1.1, -150.0, 0)));
        let estimate = estimate_risk_per_lot(&trades).expect("a stop cluster exists");
        assert!(
            estimate > 100.0,
            "the stop cluster is 150 per lot; estimate came out {estimate}"
        );

        // With nothing but near-zero losses there is no stop to find, and R must
        // be withheld rather than computed against a denominator of ~0.
        let all_tiny: Vec<ClosedTrade> = (0..50)
            .map(|i| trade(i, "BUY", 1.1, 1.1, -1e-9, 0))
            .collect();
        assert_eq!(estimate_risk_per_lot(&all_tiny), None);
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

    fn trade_on(symbol: &str, net: f64, lots: f64) -> ClosedTrade {
        let mut t = trade(0, "BUY", 1.1, 1.1, net, 1_700_000_000_000);
        t.symbol = symbol.to_string();
        t.lots = lots;
        t
    }

    /// A stop is money-per-lot. Pooling a 1-lot XAUUSD stop with a 1-lot
    /// EURUSD stop divides every trade by a denominator that belongs to
    /// neither — in the one view built to compare trades ACROSS instruments.
    #[test]
    fn r_uses_each_symbols_own_stop_where_there_is_one() {
        let mut trades: Vec<ClosedTrade> = Vec::new();
        // EURUSD: stop costs ~200 per lot.
        trades.extend((0..8).map(|_| trade_on("EURUSD", -200.0, 1.0)));
        // XAUUSD: stop costs ~4000 per lot — twenty times the money.
        trades.extend((0..8).map(|_| trade_on("XAUUSD", -4000.0, 1.0)));
        // One winner on each, same size as its own stop → R should be ~+1.
        trades.push(trade_on("EURUSD", 200.0, 1.0));
        trades.push(trade_on("XAUUSD", 4000.0, 1.0));

        let per_symbol = estimate_risk_per_lot_by_symbol(&trades);
        assert!((per_symbol["EURUSD"] - 200.0).abs() < 1.0);
        assert!((per_symbol["XAUUSD"] - 4000.0).abs() < 1.0);

        let analytics = analyse(&trades, None);
        let winners: Vec<&DerivedTrade> = analytics
            .trades
            .iter()
            .filter(|t| t.net_profit > 0.0)
            .collect();
        assert_eq!(winners.len(), 2);
        for w in winners {
            assert_eq!(w.risk_basis, "symbol");
            assert!(
                (w.r_multiple.expect("R") - 1.0).abs() < 0.05,
                "{} won exactly one stop's worth, R = {:?}",
                w.symbol,
                w.r_multiple
            );
        }
    }

    /// A symbol with too few losses of its own must borrow the pooled estimate
    /// AND say so, rather than quietly presenting a borrowed denominator as its
    /// own measurement.
    #[test]
    fn a_thin_symbol_falls_back_and_the_fallback_is_named() {
        let mut trades: Vec<ClosedTrade> = (0..8).map(|_| trade_on("EURUSD", -200.0, 1.0)).collect();
        trades.push(trade_on("GBPJPY", -180.0, 1.0)); // one loss: not enough
        trades.push(trade_on("GBPJPY", 90.0, 1.0));

        let analytics = analyse(&trades, None);
        assert_eq!(
            analytics.coverage.symbols_using_fallback_risk,
            vec!["GBPJPY".to_string()]
        );
        assert!(analytics.coverage.risk_per_lot_by_symbol.contains_key("EURUSD"));
        assert!(!analytics.coverage.risk_per_lot_by_symbol.contains_key("GBPJPY"));
        let gbp = analytics
            .trades
            .iter()
            .find(|t| t.symbol == "GBPJPY" && t.net_profit > 0.0)
            .expect("the GBPJPY winner");
        assert_eq!(gbp.risk_basis, "all_symbols");
    }

    /// A column of blanks must be distinguishable from a column of zeros, and
    /// "the price store has no bars" from "the journal has no entry time".
    #[test]
    fn coverage_separates_a_missing_row_field_from_a_missing_price_series() {
        let mut with_everything = trade(1, "BUY", 1.1000, 1.1010, 100.0, 1_700_000_000_000);
        with_everything.lots = 1.0;
        let mut no_entry_time = trade(2, "BUY", 1.1000, 1.0990, -100.0, 1_700_000_000_000);
        no_entry_time.entry_ts_ms = None;
        no_entry_time.lots = 1.0;

        // No price source at all: everything excursion-shaped is unavailable.
        let analytics = analyse(&[with_everything, no_entry_time], None);
        assert_eq!(analytics.coverage.trades_total, 2);
        assert_eq!(analytics.coverage.with_excursion, 0);
        assert_eq!(analytics.coverage.missing_entry_time, 1);
        assert_eq!(
            analytics.coverage.missing_price_series, 1,
            "one row had everything the replay needs and still got no bars"
        );
        assert_eq!(analytics.coverage.with_duration, 1);
        assert_eq!(analytics.coverage.min_losses_for_symbol_risk, MIN_LOSSES_FOR_SYMBOL_RISK);
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
