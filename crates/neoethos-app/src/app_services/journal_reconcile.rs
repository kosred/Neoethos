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
) -> Option<ClosedTrade> {
    let net = d.net_profit?;
    Some(ClosedTrade {
        schema_version: journal_store::new_schema_version(),
        recorded_at_unix_ms: journal_store::now_unix_ms(),
        position_id: d.position_id,
        // Resolve the broker symbol NAME from the catalog the bridge threads in;
        // fall back to `#<id>` only when the catalog hasn't populated yet.
        symbol: names
            .get(&d.symbol_id)
            .cloned()
            .unwrap_or_else(|| format!("#{}", d.symbol_id)),
        side: d.trade_side.clone(),
        lots: d.filled_volume,
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

    let mut recorded_any = false;
    for deal in &runtime.recent_deals {
        let Some(trade) = closed_trade_from_deal(deal, names) else {
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
            },
        );
    }
}
