# cTrader Single-Account Execution Design

**Date:** 2026-03-24

## Goal

Add real single-account `cTrader` execution to the existing authenticated runtime so the app can submit, amend, cancel, and close trades using documented Open API requests and event responses, while surfacing lot size, order status, P/L, and journal-style trade logs in the UI.

## Scope

This tranche adds:

1. Single-account `cTrader` order submission:
   - `ProtoOANewOrderReq`
   - `ProtoOAAmendOrderReq`
   - `ProtoOACancelOrderReq`
   - `ProtoOAClosePositionReq`
2. Result parsing through documented event surfaces:
   - `ProtoOAExecutionEvent`
   - `ProtoOAOrderErrorEvent`
3. Operator-facing order ticket data in the execution panel:
   - side
   - lot size
   - order type
   - SL/TP
   - comment/label
   - slippage
4. Journal/history rows with trading outcomes:
   - timestamp
   - symbol
   - side
   - lot size
   - entry/exit price
   - order/deal/position ids
   - gross P/L
   - fees
   - net P/L
   - status / error code

This tranche does **not** add:

- multi-account fan-out
- copy trading
- background streaming execution daemon
- automatic bot order routing

## Architecture

`ctrader_messages.rs` remains the documented message layer and owns payload constants, request builders, and response matching helpers.

`ctrader_execution.rs` becomes the single-account execution orchestration layer. It sends the documented message sequence over the existing Open API transport and maps responses into a typed `ExecutionOutcome`.

`trading.rs` remains the app-facing service layer. It validates the order ticket, converts lot size into protocol volume, calls the execution backend, refreshes runtime state after success, and updates UI-facing status and journal rows.

`execution_panel.rs` stays thin. It renders order-ticket inputs and operator actions but does not talk directly to the transport.

## Data Flow

1. Operator fills order ticket in UI.
2. App validates lot size against symbol/account/runtime constraints.
3. App builds the documented `cTrader` request.
4. Transport sends request on an already authenticated account session.
5. App waits for:
   - `ProtoOAExecutionEvent` on success path
   - `ProtoOAOrderErrorEvent` on error path
6. App maps result to `ExecutionOutcome`.
7. App refreshes account runtime snapshot and rebuilds `Positions`, `Orders`, `History`, and `Journal`.

## Lot Size

The UI must speak in `lot size`, while the protocol still uses `volume` in cTrader units. The service layer owns that conversion.

Before any order leaves the app:

- validate positive lot size
- validate against symbol `minVolume`, `maxVolume`, `stepVolume` when available
- fail closed if runtime metadata is insufficient for a safe conversion

Invalid lot size must never produce a fake request or fake success state.

## P/L And Journal Semantics

`History` and `Journal` rows must include P/L explicitly when the documented event/deal payload provides it.

- Use realized/gross values from the API when present.
- Use fee/commission values from the API when present.
- Compute `net P/L = gross P/L + fees` only when both pieces are actually known.
- If a field is unavailable, surface that honestly instead of inventing values.

The canonical log remains the forensic source of truth, but the operator UI should no longer require opening the log to understand order outcomes.

## Error Handling

- `ProtoOAOrderErrorEvent` maps to explicit failed operator outcomes.
- No optimistic success on request send.
- No silent retries in this tranche.
- Invalid lot size, missing account runtime, missing symbol metadata, and disconnected session all fail closed before transport.

## Testing

TDD-first:

- failing request/response tests in `ctrader_messages.rs`
- failing execution orchestration tests in `ctrader_execution.rs`
- failing service/UI mapping tests in `trading.rs` and `execution_panel.rs`

Verification gates:

- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`

## Official Sources

- [cTrader Open API](https://help.ctrader.com/open-api/)
- [cTrader Messages](https://help.ctrader.com/open-api/messages/)
- [cTrader Account Authentication](https://help.ctrader.com/open-api/account-authentication/)
- [Spotware Open API proto messages](https://github.com/spotware/openapi-proto-messages)
