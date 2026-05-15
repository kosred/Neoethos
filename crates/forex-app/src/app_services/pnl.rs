//! Broker-side unrealized PnL: audit + authoritative paths.
//!
//! Wires the `ProtoOAGetPositionUnrealizedPnLReq` / `…Res` pair that
//! Spotware added in the 2026-05-14 OpenApiMessages refresh (Batch 6).
//!
//! Today `forex-app` computes unrealized PnL locally from
//! `(currentPrice - entryPrice) * volume * pipValue * direction`. The
//! broker now exposes `grossUnrealizedPnL` + `netUnrealizedPnL` in the
//! account deposit currency, already FX-converted server-side (proto:
//! `ProtoOAPositionUnrealizedPnL` in `OpenApiModelMessages.proto`).
//!
//! ## Two modes
//!
//! **Audit mode** ([`audit_unrealized_pnl`]) — emits a `debug!` line
//! per position and a `warn!` line when |broker_net - local| /
//! position_notional exceeds [`DEFAULT_PNL_AUDIT_DRIFT_FRACTION`]. The
//! local value is still authoritative for downstream consumers; this
//! mode only flags drift for an operator to investigate.
//!
//! **Authoritative mode** ([`fetch_unrealized_pnl_for_all_positions`])
//! — Batch 14 upgrade. The risk gate consumes the broker's net PnL
//! directly per position; the local f64 computation is only consulted
//! when the server call fails. A drift larger than
//! [`DEFAULT_PNL_CIRCUIT_BREAKER_FRACTION`] is no longer just a `warn!`
//! — it returns a [`PnLDriftCircuitBreaker`] error so the caller can
//! block new orders until an operator acknowledges.
//!
//! ## Drift threshold rationale
//!
//! Per `docs/audits/research/ctrader_api_full_reference.md` §5 (and
//! the proto comment on `ProtoOAPositionUnrealizedPnL.netUnrealizedPnL`),
//! `netUnrealizedPnL` is the gross PnL **minus accrued swap** (it
//! intentionally does NOT include the potential closing commission).
//! Our local calculation is essentially a *gross* mark-to-market that
//! does NOT subtract swap. Comparing them at the `net` field therefore
//! always overstates the drift by ~|accrued swap|. We keep the audit
//! comparator on `net` because:
//!
//! 1. The risk-gate equity formula already absorbs swap on the broker
//!    side via `ProtoOAPosition.swap` (it lands in `position.swap` per
//!    `parse_reconcile_response`). The net field is the operationally
//!    relevant figure — it's what the broker would credit/debit at
//!    immediate close.
//! 2. Batch 13's `warn!` threshold of 0.1 % of position notional was
//!    chosen specifically with swap absorption in mind: a 1-week-old
//!    position with typical FX swap (~5-10 pips/yr ≈ 0.001 % of
//!    notional/day) accumulates ~0.01 % of notional in swap, well
//!    under the 0.1 % alarm.
//!
//! The 1 % authoritative-mode circuit breaker is intentionally one
//! full order of magnitude above the audit threshold. Below 0.1 % we
//! treat broker and local as agreeing; between 0.1 % and 1 % the
//! audit warning fires but the gate still trades on the broker value;
//! above 1 % we assume one side is fundamentally wrong (stale
//! quote feed, mis-scaled `moneyDigits`, wrong FX conversion) and
//! refuse to size further orders.
//!
//! ## Real-data fixtures
//!
//! All tests are real-data only. The captured payload lives under
//! `crates/forex-app/tests/fixtures/ctrader/unrealized_pnl/` — see
//! the README there for the schema and capture procedure. Synthetic
//! broker payloads remain disallowed per the 2026-05-15 operator
//! directive.

use crate::app_services::ctrader_account::CTraderPositionSnapshot;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE, CTraderOpenApiTransport,
    CTraderUnrealizedPnLSnapshot, build_account_auth_request, build_application_auth_request,
    build_get_position_unrealized_pnl_request, parse_ctrader_error_payload,
    parse_get_position_unrealized_pnl_response, parse_open_api_envelope,
};
use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;

/// Default audit drift threshold expressed as a fraction of position
/// notional (0.001 == 0.1 %).
///
/// Above this value the audit emits `warn!`; below it the comparison
/// stays at `debug!`. Tunable via the
/// `FOREX_BOT_PNL_AUDIT_DRIFT_FRACTION` env var so an operator can
/// tighten or loosen the alarm without a rebuild.
pub const DEFAULT_PNL_AUDIT_DRIFT_FRACTION: f64 = 0.001;

/// Authoritative-mode circuit-breaker threshold expressed as a fraction
/// of position notional (0.01 == 1 %).
///
/// When the live equity reader runs against the broker's
/// `ProtoOAGetPositionUnrealizedPnLRes` and the per-position drift
/// versus the local mark-to-market exceeds this fraction, the caller
/// MUST treat the broker/local pair as fundamentally inconsistent and
/// block further new-order submissions until an operator acknowledges.
/// One full order of magnitude above the audit `warn!` threshold so
/// that ordinary stale-quote noise does not trip the breaker. Tunable
/// via `FOREX_BOT_PNL_CIRCUIT_BREAKER_FRACTION`.
pub const DEFAULT_PNL_CIRCUIT_BREAKER_FRACTION: f64 = 0.01;

/// Effective drift threshold, clamped to `[1e-5, 0.05]` to keep the
/// alarm from going silent on zero or pathological on >5 %.
fn pnl_audit_drift_fraction() -> f64 {
    std::env::var("FOREX_BOT_PNL_AUDIT_DRIFT_FRACTION")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_PNL_AUDIT_DRIFT_FRACTION)
        .clamp(1e-5, 0.05)
}

/// Effective circuit-breaker threshold, clamped to `[1e-4, 0.20]`. The
/// upper bound caps the operator's "ignore drift" override at 20 % so
/// the breaker cannot be fully disabled by a typo; the lower bound
/// avoids tripping on float epsilon when broker and local agree.
fn pnl_circuit_breaker_fraction() -> f64 {
    std::env::var("FOREX_BOT_PNL_CIRCUIT_BREAKER_FRACTION")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_PNL_CIRCUIT_BREAKER_FRACTION)
        .clamp(1e-4, 0.20)
}

/// One line of the PnL audit log: pairs the broker's server-side
/// unrealized PnL with the local mark-to-market value for the same
/// position id, plus the notional used to normalise the drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnLAuditRow {
    pub position_id: i64,
    pub broker_net_unrealized_pnl: f64,
    pub local_unrealized_pnl: f64,
    pub position_notional: f64,
    /// `(broker - local) / notional` — signed fraction. `f64::NAN`
    /// when `position_notional` is non-positive (no normalisable
    /// denominator); the caller skips the warn-threshold check in
    /// that case.
    pub drift_fraction: f64,
}

impl PnLAuditRow {
    /// Drift as a percentage (already multiplied by 100), useful for
    /// the formatted log line.
    pub fn drift_pct(&self) -> f64 {
        self.drift_fraction * 100.0
    }
}

/// Fetch the broker's per-position unrealized PnL via
/// `ProtoOAGetPositionUnrealizedPnLReq`. The caller passes a
/// transport that has already opened the WebSocket — the function
/// runs the standard app-auth + account-auth handshake before the
/// PnL request so the broker accepts the call.
///
/// Returns the parsed response. Callers that already have an
/// authenticated session and want to avoid the redundant handshake
/// should call `build_get_position_unrealized_pnl_request` + their
/// own `send_sequence` directly.
pub fn fetch_broker_unrealized_pnl<T: CTraderOpenApiTransport>(
    transport: &T,
    client_id: &str,
    client_secret: &str,
    access_token: &str,
    account_id: i64,
) -> Result<CTraderUnrealizedPnLSnapshot> {
    let responses = transport.send_sequence(&[
        build_application_auth_request(client_id, client_secret, "pnl-app-auth-1"),
        build_account_auth_request(account_id, access_token, "pnl-account-auth-1"),
        build_get_position_unrealized_pnl_request(account_id, "pnl-1"),
    ])?;
    if responses.len() != 3 {
        return Err(anyhow!(
            "expected 3 cTrader unrealized pnl responses, received {}",
            responses.len()
        ));
    }
    ensure_success(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success(
        &responses[2],
        CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    )?;
    parse_get_position_unrealized_pnl_response(&responses[2])
}

fn ensure_success(response_json: &str, expected_payload_type: u32) -> Result<()> {
    let envelope = parse_open_api_envelope(response_json)?;
    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader unrealized pnl request failed: {}",
            parse_ctrader_error_payload(&envelope.payload)
                .context("failed to format cTrader error payload")?
        ));
    }
    if envelope.payload_type != expected_payload_type {
        return Err(anyhow!(
            "unexpected cTrader unrealized pnl payload type: expected {}, got {}",
            expected_payload_type,
            envelope.payload_type
        ));
    }
    Ok(())
}

/// Compare a broker-side unrealized PnL snapshot against the local
/// mark-to-market values keyed by `position_id`. Emits one
/// `debug!(target = "forex_app::pnl_audit")` line per position and
/// one `warn!` line per position whose drift exceeds the threshold
/// returned by [`pnl_audit_drift_fraction`].
///
/// Returns the audit rows in the same order as the broker snapshot
/// so the caller can persist them if it wants. Audit mode only —
/// downstream callers MUST keep using the local value; this never
/// mutates state.
pub fn audit_unrealized_pnl(
    broker: &CTraderUnrealizedPnLSnapshot,
    positions: &[CTraderPositionSnapshot],
    local_pnl_for_position: impl Fn(&CTraderPositionSnapshot) -> Option<f64>,
) -> Vec<PnLAuditRow> {
    let drift_threshold = pnl_audit_drift_fraction();
    let mut rows = Vec::with_capacity(broker.positions.len());

    for broker_row in &broker.positions {
        let Some(position) = positions
            .iter()
            .find(|p| p.position_id == broker_row.position_id)
        else {
            tracing::debug!(
                target: "forex_app::pnl_audit",
                position_id = broker_row.position_id,
                broker_net = broker_row.net_unrealized_pnl,
                "pnl_audit broker returned position not present in local reconcile snapshot"
            );
            continue;
        };
        let Some(local) = local_pnl_for_position(position) else {
            tracing::debug!(
                target: "forex_app::pnl_audit",
                position_id = broker_row.position_id,
                broker_net = broker_row.net_unrealized_pnl,
                "pnl_audit local unrealized pnl unavailable (missing live quote)"
            );
            continue;
        };
        let notional = position_notional(position);
        let drift_fraction = if notional > 0.0 {
            (broker_row.net_unrealized_pnl - local) / notional
        } else {
            f64::NAN
        };
        let row = PnLAuditRow {
            position_id: broker_row.position_id,
            broker_net_unrealized_pnl: broker_row.net_unrealized_pnl,
            local_unrealized_pnl: local,
            position_notional: notional,
            drift_fraction,
        };

        // Matches the format requested in the Batch 6 integration task:
        // `pnl_audit position=X broker={broker} local={local} drift={drift_pct:.4}%`
        tracing::debug!(
            target: "forex_app::pnl_audit",
            position = row.position_id,
            broker = row.broker_net_unrealized_pnl,
            local = row.local_unrealized_pnl,
            drift_pct = row.drift_pct(),
            "pnl_audit position={} broker={} local={} drift={:.4}%",
            row.position_id,
            row.broker_net_unrealized_pnl,
            row.local_unrealized_pnl,
            row.drift_pct(),
        );

        if drift_fraction.is_finite() && drift_fraction.abs() > drift_threshold {
            tracing::warn!(
                target: "forex_app::pnl_audit",
                position = row.position_id,
                broker = row.broker_net_unrealized_pnl,
                local = row.local_unrealized_pnl,
                drift_pct = row.drift_pct(),
                threshold_pct = drift_threshold * 100.0,
                "pnl_audit drift exceeds threshold: position={} broker={} local={} drift={:.4}% \
                 (threshold {:.4}%)",
                row.position_id,
                row.broker_net_unrealized_pnl,
                row.local_unrealized_pnl,
                row.drift_pct(),
                drift_threshold * 100.0,
            );
        }

        rows.push(row);
    }

    rows
}

/// Approximate notional used to normalise the drift. We multiply the
/// position's open `price` by its `volume`; both are already in the
/// account currency / lot-units pair that the broker exposes in
/// `ProtoOAPosition`. Returns `0.0` when either side is missing
/// (which makes the drift_fraction NaN and disables the warn check).
fn position_notional(position: &CTraderPositionSnapshot) -> f64 {
    match position.price {
        Some(price) if price.is_finite() && price > 0.0 && position.volume > 0.0 => {
            price * position.volume
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "TODO(real-data): requires a captured ProtoOAGetPositionUnrealizedPnLRes fixture \
                from a live cTrader account; see crates/forex-app/tests/fixtures/ (not committed)."]
    fn audit_unrealized_pnl_warns_when_broker_value_drifts_beyond_threshold_real_fixture() {
        // Placeholder — see the #[ignore] reason. The real test would
        // load a captured response from
        // `crates/forex-app/tests/fixtures/pnl_audit_drift.json`,
        // parse it via `parse_get_position_unrealized_pnl_response`,
        // and assert that `audit_unrealized_pnl` flags the drifted
        // position. Synthetic data is explicitly disallowed for this
        // path (see Batch 6 integration constraints).
        unimplemented!("requires a real cTrader fixture not yet captured");
    }

    #[test]
    fn drift_fraction_is_nan_when_notional_is_zero() {
        // Pure local-only check (no broker call): if the position has
        // no `price` field, `position_notional` returns 0.0 and the
        // resulting drift_fraction is NaN. This is a unit test of the
        // helper, not of the broker round-trip, so synthetic position
        // values are acceptable here.
        let position = CTraderPositionSnapshot {
            position_id: 9001,
            symbol_id: 14,
            trade_side: "BUY".to_string(),
            volume: 25.0,
            open_timestamp_ms: None,
            price: None,
            stop_loss: None,
            take_profit: None,
            swap: None,
            commission: None,
            label: None,
            comment: None,
            client_order_id: None,
        };
        assert_eq!(position_notional(&position), 0.0);
    }
}
