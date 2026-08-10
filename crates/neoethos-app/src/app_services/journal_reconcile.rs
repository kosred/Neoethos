//! Bridge live broker deals → the trade journal.
//!
//! Called best-effort from the production account-refresh path
//! (`server::bridge` after a successful `load_account_runtime`). Converts
//! each REALIZED (closing) deal in `recent_deals` into a [`ClosedTrade`]
//! (idempotent on `position_id`) and appends an equity sample whenever a
//! new trade closes.
//!
//! Defensive by contract (operator directive): runs off the main thread,
//! never panics, never blocks the refresh, never propagates an error — a
//! journal hiccup must not affect trading. Resolution that can't happen
//! (config unreadable) is silently skipped.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::app_services::ctrader_account::{CTraderAccountRuntimeSnapshot, CTraderDealSnapshot};
use crate::app_services::journal_store::{self, ClosedTrade, EquitySample};

/// Resolve the data dir from the live `config.yaml`; `None` (skip) on
/// any failure — never an error on the refresh path.
fn data_dir() -> Option<PathBuf> {
    let path = crate::server::state::current_config_path();
    neoethos_core::Settings::from_yaml(&path)
        .ok()
        .map(|s| s.system.data_dir)
}

/// Build a [`ClosedTrade`] from a deal — but only for REALIZED (closing)
/// deals, which are the ones that carry `net_profit`. Opening fills
/// (`net_profit == None`) are not closed trades and are skipped.
/// When each position was opened, from the data already in hand.
///
/// A closing deal reports the exit and the entry *price*, never the entry
/// *time*, so `entry_ts_ms` was written as `None` — and stayed `None` for all
/// 238 trades in the operator's journal. Without it there is no holding time,
/// no entry-hour breakdown, and no window to replay prices over for excursion:
/// three of the questions a journal exists to answer, unanswerable.
///
/// Two sources, both already fetched. A position still open carries
/// `open_timestamp_ms` directly. For one already closed, its opening deal is in
/// the same deal history — the earliest execution stamped with that position id.
fn position_open_times(runtime: &CTraderAccountRuntimeSnapshot) -> HashMap<i64, i64> {
    let mut opened: HashMap<i64, i64> = HashMap::new();
    for deal in &runtime.recent_deals {
        opened
            .entry(deal.position_id)
            .and_modify(|first| *first = (*first).min(deal.execution_timestamp_ms))
            .or_insert(deal.execution_timestamp_ms);
    }
    for position in &runtime.reconcile.positions {
        if let Some(ts) = position.open_timestamp_ms {
            opened.insert(position.position_id, ts);
        }
    }
    opened
}

/// The cTrader broker environment currently selected on disk, as the string the
/// journal stores (`"Demo"` / `"Live"`).
///
/// 2026-08-09: this is the fact the demo forward-test gate needs and could not
/// previously read off a journal row. It is deliberately taken here, on the
/// reconcile path, because this is the moment the fill is turned into a record —
/// the environment that was live when the deal executed is the environment the
/// account-refresh that produced this snapshot ran against.
fn current_environment_label() -> String {
    crate::app_services::broker_persistence::load_broker_settings()
        .ctrader
        .environment
        .as_str()
        .to_string()
}

fn closed_trade_from_deal(
    d: &CTraderDealSnapshot,
    names: &HashMap<i64, String>,
    account_id: &str,
    environment: &str,
    opened_at: &HashMap<i64, i64>,
) -> Option<ClosedTrade> {
    let net = d.net_profit?;
    // Resolve the broker symbol NAME from the catalog the bridge threads in.
    // While the catalog is still EMPTY (cold start race), DEFER: recording is
    // idempotent on position_id and the deal stays in the broker's recent-deals
    // window, so the next refresh (catalog populated) records it with the real
    // name. A populated catalog that lacks this id (exotic) falls back to #id.
    let symbol = match names.get(&d.symbol_id) {
        Some(n) => n.clone(),
        None if names.is_empty() => return None,
        None => format!("#{}", d.symbol_id),
    };
    // The deal that CLOSES a position trades the OPPOSITE side of the position
    // (a long is closed by a SELL deal). The journal reports the POSITION.
    let side = match d.trade_side.trim().to_ascii_uppercase().as_str() {
        "BUY" => "SELL".to_string(),
        "SELL" => "BUY".to_string(),
        _ => d.trade_side.clone(), // unknown label — keep verbatim
    };
    // `filled_volume` is BASE UNITS (12000 = 0.12 lots EURUSD); report lots.
    let lots = neoethos_core::symbol_metadata::resolve(&symbol)
        .filter(|m| m.contract_size.is_finite() && m.contract_size > 0.0)
        .map(|m| d.filled_volume / m.contract_size)
        .unwrap_or(d.filled_volume);
    Some(ClosedTrade {
        schema_version: journal_store::new_schema_version(),
        recorded_at_unix_ms: journal_store::now_unix_ms(),
        position_id: d.position_id,
        symbol,
        side,
        lots,
        // Per-account scoping (2026-07-02): the journal serves ONE account's
        // history at a time — stamp every row with its owner.
        account_id: Some(account_id.to_string()),
        // Environment scoping (2026-08-09): the demo forward-test gate counts
        // DEMO fills. Without this stamp there was no way to tell one from the
        // other on a stored row.
        environment: Some(environment.to_string()),
        // Only when it is genuinely earlier than the close. An opening deal that
        // arrived in the same batch as its closing one would otherwise stamp a
        // zero-length trade, which reads as a measurement rather than a gap.
        entry_ts_ms: opened_at
            .get(&d.position_id)
            .copied()
            .filter(|ts| *ts < d.execution_timestamp_ms),
        entry_price: d.entry_price,
        exit_ts_ms: Some(d.execution_timestamp_ms),
        exit_price: d.execution_price,
        gross_profit: d.gross_profit.unwrap_or(0.0),
        commission: d.fee.unwrap_or(0.0),
        swap: d.swap.unwrap_or(0.0),
        net_profit: net,
        balance_after: None, // balance-after wiring is a follow-up polish
    })
}

/// Record any newly-closed deals + (if any closed) an equity sample.
/// Best-effort: all failures are logged and swallowed.
pub fn reconcile_best_effort(
    runtime: &CTraderAccountRuntimeSnapshot,
    names: &HashMap<i64, String>,
) {
    let Some(dir) = data_dir() else {
        return;
    };

    let account_id = runtime.reconcile.account_id.to_string();
    let environment = current_environment_label();
    let mut recorded_any = false;
    let opened_at = position_open_times(runtime);

    // Every deal that does NOT become a journal row is counted and reasoned.
    // The journal is a decision path — the demo forward-test gate counts these
    // rows before real money is allowed — so "it just didn't show up" must
    // never be a possible explanation.
    let mut recorded = 0usize;
    let mut already_present = 0usize;
    let mut write_failed = 0usize;
    let mut opening_fills = 0usize;
    let mut deferred_cold_catalog = 0usize;
    for deal in &runtime.recent_deals {
        if deal.net_profit.is_none() {
            // An opening fill is not a closed trade; nothing is lost.
            opening_fills += 1;
            continue;
        }
        let Some(trade) =
            closed_trade_from_deal(deal, names, &account_id, &environment, &opened_at)
        else {
            // The only remaining refusal: the broker symbol catalog has not
            // populated yet. Recording is idempotent on `position_id` and the
            // deal stays in the broker's recent-deals window, so the next
            // refresh records it under its real name. Deferred, not dropped.
            deferred_cold_catalog += 1;
            continue;
        };
        match journal_store::record_closed_trade(&dir, &trade) {
            Ok(true) => {
                recorded += 1;
                recorded_any = true;
            }
            Ok(false) => already_present += 1,
            Err(e) => {
                write_failed += 1;
                tracing::warn!(
                    target: "neoethos_app::journal_reconcile",
                    error = %e, position_id = trade.position_id,
                    "failed to record closed trade (continuing)"
                );
            }
        }
    }
    if deferred_cold_catalog > 0 || write_failed > 0 {
        tracing::warn!(
            target: "neoethos_app::journal_reconcile",
            deals_seen = runtime.recent_deals.len(),
            recorded, already_present, opening_fills,
            deferred_cold_catalog, write_failed,
            "journal reconcile did not record every closing deal — \
             {deferred_cold_catalog} deferred until the broker symbol catalog \
             populates (retried on the next refresh), {write_failed} failed to write"
        );
    } else if recorded > 0 {
        tracing::debug!(
            target: "neoethos_app::journal_reconcile",
            deals_seen = runtime.recent_deals.len(),
            recorded, already_present, opening_fills,
            %environment,
            "journal reconcile: closing deals recorded"
        );
    }

    // Sample equity only when a new trade closed — keeps the equity file
    // bounded (one point per realized trade) instead of one per heartbeat.
    if recorded_any {
        let balance = runtime.trader.balance;
        journal_store::append_equity_sample_best_effort(
            &dir,
            &EquitySample {
                ts_ms: journal_store::now_unix_ms(),
                balance,
                // Floating PnL isn't summed here (kept off the hot path);
                // balance is the realized-equity anchor for the curve.
                equity: balance,
                account_id: Some(account_id.clone()),
                environment: Some(environment.clone()),
            },
        );
    }
}

#[cfg(test)]
mod entry_time_tests {
    use super::*;

    /// A populated catalog. An empty one makes `closed_trade_from_deal` defer
    /// on purpose — cold-start race — so passing one would test that instead.
    fn catalog() -> HashMap<i64, String> {
        HashMap::from([(1i64, "EURUSD".to_string())])
    }

    fn deal(position_id: i64, ts: i64) -> CTraderDealSnapshot {
        CTraderDealSnapshot {
            deal_id: ts,
            order_id: ts,
            position_id,
            symbol_id: 1,
            trade_side: "BUY".to_string(),
            deal_status: "FILLED".to_string(),
            volume: 100_000.0,
            filled_volume: 100_000.0,
            execution_timestamp_ms: ts,
            execution_price: Some(1.1),
            entry_price: Some(1.1),
            gross_profit: Some(1.0),
            fee: Some(0.0),
            swap: Some(0.0),
            pnl_conversion_fee: Some(0.0),
            net_profit: Some(1.0),
        }
    }

    /// The opening deal is in the same history as the closing one, so the entry
    /// time never needed a new request — it was being discarded.
    #[test]
    fn the_opening_deal_supplies_the_entry_time() {
        let deals = vec![deal(500, 1_700_000_000_000), deal(500, 1_700_003_600_000)];
        let mut opened: HashMap<i64, i64> = HashMap::new();
        for d in &deals {
            opened
                .entry(d.position_id)
                .and_modify(|first| *first = (*first).min(d.execution_timestamp_ms))
                .or_insert(d.execution_timestamp_ms);
        }
        assert_eq!(opened.get(&500), Some(&1_700_000_000_000));

        let closing = &deals[1];
        let trade = closed_trade_from_deal(closing, &catalog(), "acct", "Demo", &opened)
            .expect("a closing deal with a net figure is a closed trade");
        assert_eq!(trade.entry_ts_ms, Some(1_700_000_000_000));
        assert_eq!(trade.exit_ts_ms, Some(1_700_003_600_000));
    }

    /// A position whose opening deal is not in the window must report no entry
    /// time rather than borrowing the close's — a zero-length trade would read
    /// as a measurement, and every derived figure would inherit the lie.
    #[test]
    fn an_absent_opening_deal_leaves_the_entry_time_unknown() {
        let closing = deal(501, 1_700_003_600_000);
        let mut opened: HashMap<i64, i64> = HashMap::new();
        opened.insert(501, 1_700_003_600_000); // only the close is known

        let trade = closed_trade_from_deal(&closing, &catalog(), "acct", "Demo", &opened)
            .expect("still a closed trade");
        assert_eq!(
            trade.entry_ts_ms, None,
            "an entry stamped at the exit is not an entry time"
        );
    }
}
