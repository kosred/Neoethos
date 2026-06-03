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

use crate::app_services::ctrader_data::{
    CTraderSymbolInfo, SymbolDistanceType, SymbolFinancials,
};
use crate::app_services::ctrader_messages::{CTraderNewOrderRequest, CTraderTradeSide};
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
        anyhow::bail!("cTrader protocol volume overflow converting units: volume={volume}");
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
    risk: &neoethos_core::config::RiskConfig,
    order: &CTraderNewOrderRequest,
    account_equity: f64,
    initial_equity: f64,
    day_start_equity: f64,
    pip_position: i32,
    symbol_name: &str,
    // Latest mid-market price (e.g. `(bid + ask) / 2`) supplied by the caller
    // from the broker's live spot stream. Used as the entry-price estimate
    // for Market orders (which carry neither `limit_price` nor `stop_price`).
    // `None` is acceptable for Market orders without stop-loss (the
    // risk-per-trade gate is N/A); for Market orders WITH stop-loss the
    // gate hard-fails (see the new branch below) — Note.
    market_price_for_entry: Option<f64>,
    // **Phase B (2026-05-27)**: the broker's per-symbol financial &
    // schedule projection from `ProtoOASymbol`. Supplies the
    // trading-mode flag, short-selling permission, and the
    // SL/TP minimum-distance constraints that cTrader would otherwise
    // reject server-side with `TRADING_BAD_STOPS` /
    // `TRADING_DISABLED`. `None` is accepted for the test fixtures and
    // legacy paths where the broker catalog hasn't been threaded
    // through yet — those paths simply skip the new gates, preserving
    // pre-Phase-B behaviour.
    financials: Option<&SymbolFinancials>,
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

    // ── Phase B broker-side gating (2026-05-27 cycle-3) ──────────────
    //
    // When the broker's per-symbol financial projection is supplied
    // (production path — `orders.rs` resolves it from the
    // `CTraderSymbolInfo`), enforce the three constraints cTrader
    // would otherwise reject at the server with `TRADING_DISABLED`
    // or `TRADING_BAD_STOPS`. Each check fail-loud at the gate so
    // the operator sees the directive instead of the broker's
    // opaque error code returned via WSS.
    //
    // Legacy / test paths that pass `None` skip these — preserves
    // pre-Phase-B behaviour exactly.
    if let Some(fin) = financials {
        // (1) Trading-mode gating — the symbol must allow opening
        //     new positions. `CLOSE_ONLY_MODE`, `DISABLED_*` etc.
        //     are rejected here so the operator sees why instead
        //     of waiting for the broker to bounce the order.
        if !fin.can_open_new_position() {
            return Err(anyhow::anyhow!(
                "Symbol {symbol_name} is not currently accepting new \
                 positions (cTrader trading_mode = {:?}). Operator must \
                 wait for the symbol to re-enable or pick a different \
                 instrument.",
                fin.trading_mode
            ));
        }

        // (2) Short-selling gating — if the order is a SELL and the
        //     broker has short-selling disabled for this symbol,
        //     reject. The default when the broker omits the field
        //     is `true` (most pairs allow shorts), per
        //     `SymbolFinancials::short_selling_allowed`.
        if order.trade_side == CTraderTradeSide::Sell && !fin.short_selling_allowed() {
            return Err(anyhow::anyhow!(
                "Short-selling is disabled by the broker for {symbol_name}. \
                 Long-only trading is permitted; switch the order side to \
                 BUY or pick a different instrument."
            ));
        }

        // (3) SL / TP minimum-distance gating — cTrader rejects
        //     stops or take-profits closer to the entry than the
        //     `MinDistance` thresholds carried on `ProtoOASymbol`.
        //     We pre-emptively reject so the operator sees the
        //     concrete distance instead of `TRADING_BAD_STOPS`.
        //
        //     Units are determined by `distance_set_in`:
        //       - SymbolDistanceInPoints     → distance × 10^-(pip_position+1)
        //         (i.e. distance is in `points`, where 1 point =
        //         the smallest price increment for the symbol).
        //       - SymbolDistanceInPercentage → distance% of entry.
        //
        //     `distance_set_in == None` is treated as "broker did not
        //     specify units" → skip the check rather than guess.
        let entry_for_distance = order
            .limit_price
            .or(order.stop_price)
            .or(market_price_for_entry);
        if let (Some(entry), Some(dist_kind)) = (entry_for_distance, fin.distance_set_in) {
            // 1 point = 10^-(pip_position+1) of price. For 5-digit FX
            // (pip_position=4): point = 1e-5. For 3-digit JPY
            // (pip_position=2): point = 1e-3.
            let point_size = 10f64.powi(-(pip_position + 1));
            let check_distance = |label: &str, target: f64, min_points: Option<u32>| -> anyhow::Result<()> {
                let Some(min_points) = min_points else {
                    return Ok(()); // broker didn't supply a minimum
                };
                if min_points == 0 {
                    return Ok(());
                }
                let actual_price_distance = (entry - target).abs();
                let min_price_distance = match dist_kind {
                    SymbolDistanceType::SymbolDistanceInPoints => {
                        min_points as f64 * point_size
                    }
                    SymbolDistanceType::SymbolDistanceInPercentage => {
                        // Proto stores the value × 100 for the
                        // percentage variant (per cTrader docs);
                        // undo by /100 then ×entry. Defensive abs
                        // on `entry` to avoid the rare negative
                        // synthetic instrument.
                        (min_points as f64 / 100.0) * entry.abs() / 100.0
                    }
                };
                if actual_price_distance < min_price_distance {
                    return Err(anyhow::anyhow!(
                        "{label} for {symbol_name} is too close to entry: \
                         distance {:.5} < broker minimum {:.5} \
                         (entry={:.5}, target={:.5}, {} points). \
                         Widen the {label} and retry.",
                        actual_price_distance,
                        min_price_distance,
                        entry,
                        target,
                        min_points,
                    ));
                }
                Ok(())
            };

            if let Some(sl) = order.stop_loss {
                check_distance("stop-loss", sl, fin.sl_distance_points)?;
            }
            if let Some(tp) = order.take_profit {
                check_distance("take-profit", tp, fin.tp_distance_points)?;
            }
        }
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

    // Pre-condition checks that must run regardless of order type.
    // An empty symbol name must be rejected unconditionally — we must
    // not fall through to a synthetic EURUSD pip-value default in any
    // prop-firm code path.
    let symbol = symbol_name.trim();
    if symbol.is_empty() {
        return Err(anyhow::anyhow!(
            "Risk gate cannot size order: symbol name was not supplied. \
             Refusing to fall back to synthetic pip-value defaults."
        ));
    }

    // When a stop-loss is present the operator expects risk sizing to
    // happen. Require the account currency now (before the entry-price
    // check below) so a misconfigured environment is surfaced even for
    // market orders that carry a stop-loss but no explicit entry price.
    if order.stop_loss.is_some() {
        // **F-565 fix (2026-05-25 — F-CORE3 Phase B)**: was direct
        // `std::env::var(...)`. Now routes through
        // `neoethos_core::env_overrides::prop_firm_account_currency()`
        // — the canonical typed getter. Same behaviour, single
        // grep-able source for the env-var name.
        neoethos_core::env_overrides::prop_firm_account_currency().ok_or_else(|| {
            anyhow::anyhow!(
                "Risk gate cannot size order: NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY \
                 is unset. The account currency must be supplied by the broker \
                 (cTrader trader profile) — no synthetic default allowed."
            )
        })?;
        // The full pip-value computation runs inside the entry-price block
        // below; we only validate presence here.
    }

    // HARD risk-per-trade gate (D4+D5). Previously this was a `tracing::warn`
    // that did not block the order, and the loss estimate used `* 10.0` —
    // i.e. it assumed every pip is worth $10/std-lot. Wrong for non-USD
    // accounts and for any quote currency != account currency. We now
    // compute the real per-pip account-currency value via the neoethos-search
    // cost model and reject the order if it would exceed the configured
    // `risk_per_trade` percentage. Override the live FX rate the model
    // needs for cross pairs via `NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`.
    //
    // Note — entry-estimate fallback for Market orders.
    // Pre-fix: a Market order carries neither `limit_price` nor `stop_price`,
    // so the original `(Some(sl), Some(entry))` if-let pattern was always
    // `(Some(sl), None)` for Market orders → the entire risk-per-trade
    // computation was SKIPPED even when a stop-loss was set. That meant a
    // 100-lot Market BUY with a wide SL passed the gate silently. We now:
    //   (a) accept an optional `market_price_for_entry` from the caller
    //       (typically the mid of the latest cTrader spot quote);
    //   (b) use it as the entry estimate when the order itself carries no
    //       limit/stop price;
    //   (c) hard-fail if `stop_loss` is set but NO entry estimate of any
    //       kind is available — refusing to size an order without an
    //       authoritative entry price is the safe choice in a prop-firm
    //       production path.
    let entry_estimate = order
        .limit_price
        .or(order.stop_price)
        .or(market_price_for_entry);
    if order.stop_loss.is_some() && entry_estimate.is_none() {
        return Err(anyhow::anyhow!(
            "Risk gate cannot size order: stop_loss is set but no entry-price \
             estimate is available (Market order with no `limit_price`/`stop_price` \
             and no live mid-market quote). Wait for the broker to deliver a fresh \
             bid/ask spot update before retrying, or switch to a Limit order with \
             an explicit target_price. Refusing to bypass the risk-per-trade gate."
        ));
    }
    if let (Some(sl), Some(entry_estimate)) = (order.stop_loss, entry_estimate) {
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
        // (symbol emptiness already checked above; no need to re-check)
        let metadata = neoethos_core::symbol_metadata::resolve(symbol).ok_or_else(|| {
            anyhow::anyhow!(
                "Risk gate cannot size order: no cTrader symbol metadata for {symbol}. \
                 Populate data/symbol_metadata.json (or the NEOETHOS_BOT_SYMBOL_METADATA \
                 override) from the cTrader ProtoOASymbol records before trading."
            )
        })?;
        // Pip value in account currency per standard lot. We hard-fail
        // for cross pairs without a quote→account conversion rate
        // rather than silently using a possibly-wrong fallback.
        // `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` / `NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`
        // are reserved for operator overrides only — never synthesized.
        // **F-565 fix (2026-05-25 — F-CORE3 Phase B)**: both env reads
        // route through the canonical `neoethos_core::env_overrides`
        // registry.
        let account_currency = neoethos_core::env_overrides::prop_firm_account_currency()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Risk gate cannot size order: NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY \
                     is unset. The account currency must be supplied by the broker \
                     (cTrader trader profile) — no synthetic default allowed."
                )
            })?;
        let quote_to_account_rate =
            neoethos_core::env_overrides::prop_firm_quote_to_account_rate();
        let pip_value_per_lot = metadata.pip_value_in_account(
            &account_currency,
            quote_to_account_rate,
            Some(entry_estimate),
        );
        if !pip_value_per_lot.is_finite() || pip_value_per_lot <= 0.0 {
            return Err(anyhow::anyhow!(
                "Risk gate cannot size order: pip value for {symbol} in {account_currency} \
                 is not resolvable (cross-pair without quote→account FX rate?). \
                 Set NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE or supply a broker-sourced \
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
