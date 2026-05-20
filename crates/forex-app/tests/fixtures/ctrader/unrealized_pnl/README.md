# `ProtoOAGetPositionUnrealizedPnLRes` fixtures

This directory holds **real, captured** `ProtoOAGetPositionUnrealizedPnLRes`
JSON envelopes used by the Batch 14 authoritative-PnL tests in
`crates/forex-app/src/app_services/pnl.rs::tests`. Synthetic broker
payloads are forbidden here per the 2026-05-15 operator directive
(no-silent-fallback / no-synthetic-data).

## Schema reference

Source of truth: `crates/forex-app/proto/OpenApiMessages.proto:789-794`
(message `ProtoOAGetPositionUnrealizedPnLRes`, payload type **2188**)
and `OpenApiModelMessages.proto:714-718` (the per-row
`ProtoOAPositionUnrealizedPnL` body). Documented in
`docs/audits/research/ctrader_api_full_reference.md` §4.1 (payload-type
table) and §5 (model schemas).

```json
{
  "clientMsgId": "<operator-supplied or echoed from the request>",
  "payloadType": 2188,
  "payload": {
    "ctidTraderAccountId": <int64 — required>,
    "moneyDigits": <uint32 — required; 10^moneyDigits applied to gross/net>,
    "positionUnrealizedPnL": [
      {
        "positionId": <int64 — required>,
        "grossUnrealizedPnL": <int64 — required; scaled by 10^moneyDigits>,
        "netUnrealizedPnL":   <int64 — required; scaled by 10^moneyDigits>
      },
      ...
    ]
  }
}
```

Per the proto comments:

- `grossUnrealizedPnL` — gross PnL in the account deposit currency.
- `netUnrealizedPnL` — `gross - accrued swap` in deposit currency. Does
  **not** include the closing commission that would apply if the
  position were liquidated immediately.

`moneyDigits` is the per-response exponent applied to BOTH gross and
net rows. The proto comment quotes the canonical example:

> `moneyDigits = 8` should be interpreted as the value multiplied by
> 10^8 with the 'real' value equal to `10053099944 / 10^8 = 100.53099944`.

`crates/forex-app/src/app_services/ctrader_money.rs::scale_ctrader_money_int`
is the only sanctioned scaler — every fixture-driven test routes
through it (see `parse_get_position_unrealized_pnl_response`).

## Expected fixture file names

The Batch 14 test suite is `#[ignore = "TODO(real-data): ..."]` until
the operator captures and lands the following files (commit them
under this directory):

| File | Test that consumes it | Captured scenario |
|------|------------------------|-------------------|
| `pnl_audit_steady_state.json` | `fetch_authoritative_returns_one_row_per_broker_position_real_fixture` | Healthy reconcile tick — broker and local agree within `DEFAULT_PNL_AUDIT_DRIFT_FRACTION` (0.1 % of position notional). At least 2 open positions so the `HashMap` indexing is exercised. |
| `pnl_audit_drift.json` | `audit_unrealized_pnl_warns_when_broker_value_drifts_beyond_threshold_real_fixture` | Audit `warn!` case — at least one position with broker/local drift in `[0.1 %, 1 %]` of position notional. |
| `pnl_circuit_breaker_trip.json` | `circuit_breaker_trips_when_drift_exceeds_one_percent_real_fixture` | Circuit-breaker case — at least one position with broker/local drift > 1 % of position notional. Easiest reproduction: pause the local spot-feed for ~30 s while a 1-lot position is open in a fast-moving session, then capture. |

Each file is the **raw JSON envelope** the cTrader Open API sends on
the WebSocket (port `5036`) — i.e. the same string our
`ProductionCTraderOpenApiTransport` reads off the socket and passes to
`parse_get_position_unrealized_pnl_response`. Keep `clientMsgId` as
returned by the broker (do not hand-edit). The host (`live.ctraderapi.com`
or `demo.ctraderapi.com`) and the originating account id should be
recorded in the commit message but **not** in the file body —
re-using the broker's verbatim bytes is the entire point.

## Capture procedure

1. Start the bot against a demo (or live) cTrader account with at
   least one open position. The demo host is
   `demo.ctraderapi.com:5036`; auth bundle as usual via the cTrader
   OAuth flow.
2. Wrap the `ProductionCTraderOpenApiTransport::send_sequence` call
   with a one-line `tracing::debug!(target: "forex_app::pnl_capture",
   raw = %text, "captured pnl response")` immediately after the
   `responses.push(text.to_string())` for payload type 2188. The
   captured `text` is exactly the bytes this README references — no
   re-serialization, no field reordering.
3. Run a real reconcile + PnL fetch, redirecting the trace output to a
   file. `RUST_LOG=forex_app::pnl_capture=debug cargo run ...` is
   enough.
4. Extract the matching log line, save the JSON body to the
   appropriate file in this directory, and remove the temporary
   capture tracing line.
5. For the drift / circuit-breaker scenarios, either capture during a
   genuine drift incident or temporarily pause the local
   `ctrader_streaming` task while a position is open so the broker's
   net value diverges from the locally-cached mark-to-market by the
   target amount.
6. Commit the file with a short note in the commit body identifying
   the broker host, the originating account id (range only — e.g.
   "demo account 100…1234"), and the capture date.

## Threshold cross-reference

The audit warn threshold (0.1 %) and the authoritative-mode circuit
breaker (1 %) are both `pub const` at the top of
`crates/forex-app/src/app_services/pnl.rs`. Both are operator-tunable
through `FOREX_BOT_PNL_AUDIT_DRIFT_FRACTION` and
`FOREX_BOT_PNL_CIRCUIT_BREAKER_FRACTION`. Rationale lives in that
module's header doc comment.

The threshold is evaluated against `netUnrealizedPnL` (not gross). The
choice is documented in the same module header — broker `net` is what
the operator owes/is-owed at immediate close, and prop-firm equity
drawdown is measured on that quantity.
