//! Prop-firm risk gate + volume-conversion helpers.
//!
//! This module owns the pre-trade safety checks (daily drawdown, total
//! drawdown, risk-per-trade) and the bounds-checked volume conversions that
//! feed cTrader on-wire requests. It was carved out of the trading.rs
//! god-file to make every audit-fix sit next to the function it protects.
//!
//! PRESERVED FIXES (do not change behavior without auditor sign-off):
//! - Batch 1 / audit-fix F5: `units_to_ctrader_protocol_volume` returns
//!   `Result<i64>` and hard-fails on non-finite / out-of-range inputs
//!   instead of saturating to `i64::MAX`.
//! - Batch 1 / audit-fix F6: `ctrader_protocol_volume_from_lots` applies the
//!   same overflow guard before the cast to `i64`.
//! - Batch 3b: `validate_and_convert_lot_size_to_ctrader_volume` enforces the
//!   broker min/max/step volume constraints over the converted protocol
//!   volume.
//! - Batch B (Pass 3): `prop_firm_pre_trade_check` hard-fails on an empty
//!   symbol name or unresolved cTrader metadata rather than falling back to
//!   synthetic pip-value defaults.
//! - Batch 9 / audit-fix F7: `prop_firm_pre_trade_check` clamps the
//!   `pip_position` argument to `[-10, 10]` at gate entry so a malformed
//!   symbol metadata cannot make `10.0_f64.powi(pip_position)` go to
//!   `inf`/`0.0` and silently flip the gate.

use crate::app_services::ctrader_data::CTraderSymbolInfo;
use crate::app_services::ctrader_messages::CTraderNewOrderRequest;
use crate::app_state::OrderTicketState;

/// Convert "units" (cTrader's strategy-side unit count) to the on-wire
/// `volume` integer (units × 100, since cTrader expresses volumes in
/// 1/100 of a unit).
///
/// SECURITY (audit-fix F5): the previous `as i64` cast silently saturated
/// to `i64::MAX` for non-finite or out-of-range inputs, so a malformed
/// upstream value could in principle slip a max-volume order past every
/// downstream check. We now hard-fail at the conversion boundary so the
/// operator sees the bad input instead of a giant order.
pub(super) fn units_to_ctrader_protocol_volume(volume: f64) -> anyhow::Result<i64> {
    let scaled = volume * 100.0;
    if !scaled.is_finite() || scaled.abs() >= i64::MAX as f64 {
        anyhow::bail!(
            "cTrader protocol volume overflow converting units: volume={volume}"
        );
    }
    Ok(scaled.round() as i64)
}

pub(super) fn ctrader_protocol_volume_from_units(volume: f64) -> anyhow::Result<i64> {
    units_to_ctrader_protocol_volume(volume)
}

pub(super) fn ctrader_protocol_volume_from_lots(
    lots: f64,
    symbol: &CTraderSymbolInfo,
) -> anyhow::Result<i64> {
    let lot_size = symbol
        .lot_size
        .ok_or_else(|| anyhow::anyhow!("cTrader symbol metadata is missing lotSize"))?;
    // SECURITY (audit-fix F6): same overflow class as F5 — guard the
    // product before the silent saturating cast.
    let scaled = lots * lot_size as f64;
    if !scaled.is_finite() || scaled.abs() >= i64::MAX as f64 {
        anyhow::bail!(
            "cTrader protocol volume overflow converting lots: lots={lots} lot_size={lot_size}"
        );
    }
    Ok(scaled.round() as i64)
}

pub(super) fn validate_and_convert_lot_size_to_ctrader_volume(
    ticket: &OrderTicketState,
    max_lot_size: f64,
    symbol: &CTraderSymbolInfo,
) -> anyhow::Result<i64> {
    if ticket.lot_size <= 0.0 {
        return Err(anyhow::anyhow!("lot size must be greater than zero"));
    }
    if ticket.lot_size > max_lot_size {
        return Err(anyhow::anyhow!(
            "lot size {:.2} exceeds app risk limit {:.2}",
            ticket.lot_size,
            max_lot_size
        ));
    }
    let protocol_volume = ctrader_protocol_volume_from_lots(ticket.lot_size, symbol)?;
    if let Some(min_volume) = symbol.min_volume
        && protocol_volume < min_volume
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} is below broker minimum {:.2}",
            ticket.lot_size,
            min_volume as f64 / symbol.lot_size.unwrap_or(1) as f64
        ));
    }
    if let Some(max_volume) = symbol.max_volume
        && protocol_volume > max_volume
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} exceeds broker maximum {:.2}",
            ticket.lot_size,
            max_volume as f64 / symbol.lot_size.unwrap_or(1) as f64
        ));
    }
    if let Some(step_volume) = symbol.step_volume
        && step_volume > 0
        && protocol_volume % step_volume != 0
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} does not align with broker step volume",
            ticket.lot_size
        ));
    }
    Ok(protocol_volume)
}

pub(super) fn prop_firm_pre_trade_check(
    risk: &forex_core::config::RiskConfig,
    order: &CTraderNewOrderRequest,
    account_equity: f64,
    initial_equity: f64,
    day_start_equity: f64,
    pip_position: i32,
    symbol_name: &str,
) -> anyhow::Result<()> {
    // SECURITY (audit-fix F7): `10.0_f64.powi(pip_position)` returns `inf`
    // when `pip_position >= 308` and `0.0` when `pip_position <= -308`,
    // either of which silently breaks the risk-per-trade gate below:
    // `pip_distance` either explodes (gate rejects every order) or
    // collapses to zero (gate passes every order). FX pip positions are
    // exactly 2 (JPY) or 4 (everything else), so anything outside ±10
    // is malformed symbol metadata and must be rejected at the gate
    // boundary rather than risk-sized.
    if !(-10..=10).contains(&pip_position) {
        return Err(anyhow::anyhow!(
            "invalid pip_position {pip_position} for symbol {symbol_name}: \
             outside supported FX range [-10, 10]; refusing to size order"
        ));
    }
    if risk.require_stop_loss && order.stop_loss.is_none() {
        return Err(anyhow::anyhow!(
            "Mandatory stop-loss rule violated: order missing stop_loss"
        ));
    }

    if day_start_equity > 0.0 {
        let daily_drawdown_ratio =
            ((day_start_equity - account_equity) / day_start_equity).max(0.0);
        if daily_drawdown_ratio >= risk.daily_drawdown_limit {
            return Err(anyhow::anyhow!(
                "Daily drawdown limit reached: current {:.2}% >= max {:.2}% (measured via Equity)",
                daily_drawdown_ratio * 100.0,
                risk.daily_drawdown_limit * 100.0
            ));
        }
    }

    if initial_equity > 0.0 {
        let total_drawdown_ratio = ((initial_equity - account_equity) / initial_equity).max(0.0);
        if total_drawdown_ratio >= risk.total_drawdown_limit {
            return Err(anyhow::anyhow!(
                "Total drawdown limit reached: current {:.2}% >= max {:.2}% (measured via Equity)",
                total_drawdown_ratio * 100.0,
                risk.total_drawdown_limit * 100.0
            ));
        }
    }

    // HARD risk-per-trade gate (D4+D5). Previously this was a `tracing::warn`
    // that did not block the order, and the loss estimate used `* 10.0` —
    // i.e. it assumed every pip is worth $10/std-lot. Wrong for non-USD
    // accounts and for any quote currency != account currency. We now
    // compute the real per-pip account-currency value via the forex-search
    // cost model and reject the order if it would exceed the configured
    // `risk_per_trade` percentage. Override the live FX rate the model
    // needs for cross pairs via `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`.
    if let (Some(sl), Some(entry_estimate)) =
        (order.stop_loss, order.limit_price.or(order.stop_price))
    {
        let pip_multiplier = 10.0_f64.powi(pip_position);
        let pip_distance = (entry_estimate - sl).abs() * pip_multiplier;

        // The risk gate refuses to size a position without authoritative
        // per-symbol pip-value metadata. Synthesised defaults (empty
        // symbol → EURUSD heuristic, `* 10.0` USD-per-pip assumption)
        // are not acceptable in a prop-firm production path: a single
        // mispriced JPY or cross pair would silently bypass the
        // configured `risk_per_trade` ceiling. Real metadata must come
        // from the cTrader symbol-metadata table (populated by the
        // ctrader connector) or operator-supplied disk overrides.
        let symbol = symbol_name.trim();
        if symbol.is_empty() {
            return Err(anyhow::anyhow!(
                "Risk gate cannot size order: symbol name was not supplied. \
                 Refusing to fall back to synthetic pip-value defaults."
            ));
        }
        let metadata = forex_core::symbol_metadata::resolve(symbol).ok_or_else(|| {
            anyhow::anyhow!(
                "Risk gate cannot size order: no cTrader symbol metadata for {symbol}. \
                 Populate data/symbol_metadata.json (or the FOREX_BOT_SYMBOL_METADATA \
                 override) from the cTrader ProtoOASymbol records before trading."
            )
        })?;
        // Pip value in account currency per standard lot. We hard-fail
        // for cross pairs without a quote→account conversion rate
        // rather than silently using a possibly-wrong fallback.
        // `FOREX_BOT_PROP_ACCOUNT_CURRENCY` / `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`
        // are reserved for operator overrides only — never synthesized.
        let account_currency = std::env::var("FOREX_BOT_PROP_ACCOUNT_CURRENCY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Risk gate cannot size order: FOREX_BOT_PROP_ACCOUNT_CURRENCY \
                     is unset. The account currency must be supplied by the broker \
                     (cTrader trader profile) — no synthetic default allowed."
                )
            })?;
        let quote_to_account_rate = std::env::var("FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0);
        let pip_value_per_lot = metadata.pip_value_in_account(
            &account_currency,
            quote_to_account_rate,
            Some(entry_estimate),
        );
        if !pip_value_per_lot.is_finite() || pip_value_per_lot <= 0.0 {
            return Err(anyhow::anyhow!(
                "Risk gate cannot size order: pip value for {symbol} in {account_currency} \
                 is not resolvable (cross-pair without quote→account FX rate?). \
                 Set FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE or supply a broker-sourced \
                 conversion rate; no synthetic fallback is permitted."
            ));
        }
        // cTrader volume is in cents of a standard lot, so divide.
        let estimated_loss = pip_distance * (order.volume as f64 / 100.0) * pip_value_per_lot;
        let max_loss = risk.risk_per_trade * account_equity;
        if estimated_loss > max_loss {
            return Err(anyhow::anyhow!(
                "Risk-per-trade exceeded: estimated loss {:.2} > max allowed {:.2} ({:.2}%) at {:.1} pips",
                estimated_loss,
                max_loss,
                risk.risk_per_trade * 100.0,
                pip_distance
            ));
        }
    }

    Ok(())
}
