# cTrader Execution Foundation Design

**Date:** 2026-03-24

**Goal:** Extend the existing `cTrader` tier-1 adapter from account runtime parity to execution parity by adding recent deals/fills visibility and documented order-operation request builders without inventing undocumented behavior.

## Scope

This tranche adds four things:

1. `ProtoOADealListReq` / `ProtoOADealListRes` support for recent fills.
2. `cTrader` runtime parsing for recent deals alongside trader and reconcile snapshots.
3. `ExecutionSurfaceSnapshot` parity so the UI shows `cTrader` fills/timeline the same way it already does for `MT5`.
4. Documented request builders for:
   - `ProtoOANewOrderReq`
   - `ProtoOAAmendOrderReq`
   - `ProtoOACancelOrderReq`
   - `ProtoOAClosePositionReq`

This tranche does **not** add live order submission yet. It establishes the typed message layer and runtime/deal surface needed before wiring actual execution commands.

## Architecture

`ctrader_messages.rs` remains the documented Open API message layer. It owns payload IDs, JSON request builders, response matching, and low-level transport helpers.

`ctrader_account.rs` remains the account-runtime layer. It grows from `trader + reconcile` to `trader + reconcile + deals`, keeping parsing and transport orchestration isolated from UI concerns.

`trading.rs` remains the app-facing service layer. It consumes the richer `cTrader` runtime snapshot and maps it into the existing `ExecutionSurfaceSnapshot` contract used by the UI.

## Data Flow

For connected `cTrader` runtime:

1. App auth
2. Account auth
3. Trader request
4. Reconcile request
5. Deal list request
6. Parse responses into a typed runtime snapshot
7. Format positions, pending orders, and recent fills for the operator UI

The order-operation builders will be introduced at the message layer and covered by tests, but they will not yet be executed from the UI.

## Error Handling

- If the deal list response is missing or malformed, the runtime load fails explicitly.
- If a response returns `ProtoOAErrorRes`, we surface the parsed error text and do not degrade silently.
- If `cTrader` runtime is not connected, the UI continues to show honest warnings instead of fake fills or fake success.

## Testing

TDD-first:

- failing tests for `ctrader_messages.rs` request builders
- failing tests for `ctrader_account.rs` deal parsing and runtime orchestration
- failing tests for `trading.rs` execution-surface parity

Verification gates:

- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`

## Official Sources

- [cTrader Open API](https://help.ctrader.com/open-api/)
- [cTrader Messages](https://help.ctrader.com/open-api/messages/)
- [Spotware Open API proto messages](https://github.com/spotware/openapi-proto-messages)
