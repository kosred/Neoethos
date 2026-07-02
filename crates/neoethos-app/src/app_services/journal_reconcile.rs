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
fn closed_trade_from_deal(
    d: &CTraderDealSnapshot,
    names: &HashMap<i64, String>,
    account_id: &str,
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
        entry_ts_ms: None,
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
    let mut recorded_any = false;
    for deal in &runtime.recent_deals {
        let Some(trade) = closed_trade_from_deal(deal, names, &account_id) else {
            continue;
        };
        match journal_store::record_closed_trade(&dir, &trade) {
            Ok(true) => recorded_any = true,
            Ok(false) => {} // already recorded
            Err(e) => {
                tracing::warn!(
                    target: "neoethos_app::journal_reconcile",
                    error = %e, position_id = trade.position_id,
                    "failed to record closed trade (continuing)"
                );
            }
        }
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
            },
        );
    }
}
