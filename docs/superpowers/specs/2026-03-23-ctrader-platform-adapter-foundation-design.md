# cTrader Platform Adapter Foundation Design

## Goal

Promote `cTrader` to the tier-1 backend for market data and execution by implementing the documented `cTrader Open API` surface in staged, testable tranches, while keeping `MT5` as a secondary adapter.

## Official-source basis

This design is based on official `cTrader Open API` documentation and primary protocol references:

- [Getting started](https://help.ctrader.com/open-api/)
- [App and account authentication](https://help.ctrader.com/open-api/account-authentication/)
- [Attain symbol data](https://help.ctrader.com/open-api/symbol-data/)
- [Messages](https://help.ctrader.com/open-api/messages/)
- [Proxies and endpoints](https://help.ctrader.com/open-api/proxies-endpoints/)

Confirmed documented capabilities relevant to our product:

- App auth and account auth
- Historical trend bars and tick data
- Live quotes, live trend bars, depth quotes
- Symbols and symbol metadata
- Open positions and pending orders reconciliation
- Trading operations, including documented order requests and execution events

Not yet treated as documented API capability for our implementation scope:

- Opening new demo or live accounts directly via the trading API surface. This may exist as product or broker onboarding functionality, but it is not currently treated as a standard documented `Open API` trading contract in this design.

## Product direction

`cTrader` becomes the primary backend for:

- Charts
- Historical market data
- Live market data
- Positions / orders / deals
- Trading execution
- Multi-account execution fan-out

`MT5` remains supported as:

- a local bridge path
- a secondary execution and diagnostics backend

The application remains broker-adapter based, but `cTrader` is now the priority path.

## Architecture

The implementation is split into small layers.

### 1. cTrader protocol layer

New low-level modules will own documented message building, parsing, request/response correlation, and transport behavior.

Planned files:

- `crates/forex-app/src/app_services/ctrader_messages.rs`
- `crates/forex-app/src/app_services/ctrader_open_api.rs`

Responsibilities:

- payload type constants
- typed request/response envelopes
- JSON transport matching by `clientMsgId`
- fail-closed error handling for `ProtoErrorRes` / `ProtoOAOrderErrorEvent`

### 2. cTrader domain adapters

Small focused adapters per concern:

- `crates/forex-app/src/app_services/ctrader_data.rs`
- later `ctrader_account.rs`
- later `ctrader_execution.rs`
- later `ctrader_streaming.rs`

Responsibilities:

- symbols and historical data
- account auth and account metadata
- live reconciliation and orders
- subscriptions and streaming events

### 3. App-level broker-agnostic integration

`TradingSession` remains the app boundary.

Responsibilities:

- expose typed snapshots to the UI
- choose active adapter
- cache and normalize market/execution state
- never expose raw cTrader protocol details directly to the UI

### 4. UI consumption

The UI consumes only typed snapshots:

- chart market data snapshot
- execution surface snapshot
- account/session snapshot
- warnings / degraded / failed states

## Tranche map

This program is too large for a single implementation pass. It is deliberately decomposed.

### Tranche 1: Symbols + historical data

Deliver:

- symbol discovery
- symbol metadata normalization
- historical trend bars
- historical tick data contract if documented mapping is straightforward
- chart integration using real `cTrader` historical data

Primary documented messages:

- `ProtoOASymbolsListReq`
- `ProtoOASymbolsListRes`
- `ProtoOAGetTrendbarsReq`
- `ProtoOAGetTrendbarsRes`
- `ProtoOAGetTickDataReq`
- `ProtoOAGetTickDataRes`

### Tranche 2: Account auth + current account state

Deliver:

- `ProtoOAAccountAuthReq`
- authenticated account session lifecycle
- trader/account snapshot
- reconcile current open positions and pending orders

Primary documented messages:

- `ProtoOAAccountAuthReq`
- `ProtoOAAccountAuthRes`
- `ProtoOATraderReq`
- `ProtoOATraderRes`
- `ProtoOAReconcileReq`
- `ProtoOAReconcileRes`

### Tranche 3: Live market subscriptions

Deliver:

- spot quotes
- live trend bars
- depth quotes
- streaming snapshot updates into chart/workspace

Primary documented messages:

- `ProtoOASubscribeSpotsReq`
- `ProtoOASubscribeSpotsRes`
- `ProtoOASpotEvent`
- `ProtoOASubscribeLiveTrendbarReq`
- `ProtoOASubscribeLiveTrendbarRes`
- `ProtoOASubscribeDepthQuotesReq`
- `ProtoOASubscribeDepthQuotesRes`
- `ProtoOADepthEvent`

### Tranche 4: Trading operations

Deliver:

- new orders
- cancel/amend flows
- close position
- execution event handling
- order error handling

Primary documented messages:

- `ProtoOANewOrderReq`
- `ProtoOACancelOrderReq`
- `ProtoOAAmendOrderReq`
- `ProtoOAAmendPositionSLTPReq`
- `ProtoOAClosePositionReq`
- `ProtoOAExecutionEvent`
- `ProtoOAOrderErrorEvent`

### Tranche 5: Multi-account execution

Deliver:

- per-account auth/session registry
- multi-account execution targets
- fan-out submit/cancel/close
- per-account result tracking

## Data contracts

The first tranche should introduce app-level typed contracts instead of leaking raw envelopes.

### Symbol contract

- `CTraderSymbolInfo`
  - `symbol_id`
  - `symbol_name`
  - `display_name`
  - `base_asset`
  - `quote_asset`
  - `digits`
  - `pip_position`
  - `is_archived`
  - `is_trading_enabled`

### Historical bars contract

- `HistoricalBarsRequest`
  - `account_id`
  - `symbol_id`
  - `timeframe`
  - `from_timestamp_ms`
  - `to_timestamp_ms`
  - `count`

- `HistoricalBar`
  - `timestamp_ms`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`

- `HistoricalBarsResult`
  - `symbol_id`
  - `timeframe`
  - `bars`
  - `has_more`
  - `warnings`

## Critical implementation rules

- Only use documented `cTrader` API functionality.
- No speculative “probably exists” messages or endpoints.
- No silent fallbacks on malformed or unsupported responses.
- No UI dependency on raw JSON envelopes.
- All response matching must be correlated, not “first frame wins”.
- Error payloads must be surfaced explicitly.

## Testing and verification

Each tranche must pass:

- focused unit tests for message builders/parsers
- adapter integration tests
- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`

For market data tranches, chart/UI snapshots must use real adapter results and must not regress to placeholders.

## Current validated baseline

Already implemented and verified before this design:

- browser-based `cTrader` OAuth bootstrap
- secure token persistence and restore
- `Live/Demo` environment selection
- account discovery and target synchronization
- hardened callback and account-discovery transport correlation

This design builds on that verified auth/discovery foundation instead of replacing it.
