# cTrader Live Subscriptions And Reconcile Design

## Goal

Add the next documented `cTrader Open API` tranche after historical data:

- authenticated account session over the Open API transport
- trader/account snapshot retrieval
- reconcile snapshots for current positions and pending orders
- live spot and live trendbar subscriptions for the selected symbol/timeframe

This tranche must keep `cTrader` as the tier-1 backend and expose only broker-agnostic app snapshots to the UI.

## Official-source basis

This design is based only on official `cTrader Open API` documentation:

- [App and account authentication](https://help.ctrader.com/open-api/account-authentication/)
- [Messages](https://help.ctrader.com/open-api/messages/)
- [Attain symbol data](https://help.ctrader.com/open-api/symbol-data/)
- [Proxies and endpoints](https://help.ctrader.com/open-api/proxies-endpoints/)

Confirmed documented requirements used by this design:

- `ProtoOAAccountAuthReq` requires an already authorized application connection and takes `ctidTraderAccountId` plus `accessToken`
- `ProtoOATraderReq` / `ProtoOATraderRes` expose trader account information
- `ProtoOAReconcileReq` / `ProtoOAReconcileRes` expose current open positions and pending orders
- `ProtoOASubscribeSpotsReq` subscribes a symbol and may request timestamps
- `ProtoOASubscribeLiveTrendbarReq` requires an existing spot subscription
- `ProtoOASpotEvent` carries bid, ask, timestamp, and repeated `trendbar`
- JSON transport uses port `5036`, while live and demo environments must remain fully separated

## Product direction

After this tranche:

- the `Chart` panel should be able to show historical candles plus live cTrader updates when an authenticated cTrader account is active
- the `Execution` panel and bottom strip should show real cTrader positions and pending orders from reconcile snapshots
- the UI should continue to fail closed with explicit degraded reasons whenever account auth, subscriptions, or reconcile are unavailable

This tranche does not yet add trading operations such as `new/amend/cancel/close` orders. It prepares the runtime needed for those later commands.

## Architecture

The new behavior is split into three focused layers.

### 1. Protocol helpers

`ctrader_messages.rs` will be extended with documented builders and constants for:

- `ProtoOAAccountAuthReq`
- `ProtoOATraderReq`
- `ProtoOAReconcileReq`
- `ProtoOASubscribeSpotsReq`
- `ProtoOAUnsubscribeSpotsReq`
- `ProtoOASubscribeLiveTrendbarReq`
- `ProtoOAUnsubscribeLiveTrendbarReq`

It will also own any small shared helpers needed to correlate responses and interpret known payload ids.

### 2. Focused cTrader adapters

Two small focused modules will own the next runtime seams:

- `ctrader_account.rs`
  - account-auth builders/parsers
  - trader snapshot parsing
  - reconcile response parsing
- `ctrader_streaming.rs`
  - spot/live-trendbar subscribe/unsubscribe builders
  - spot event parsing
  - normalized live quote/live candle update contracts

These modules must stay transport-agnostic and testable through stub transports or pure parser tests.

### 3. App-level integration

`TradingSession` remains the only UI-facing boundary.

Responsibilities:

- authenticate the selected cTrader account at the Open API level
- fetch trader and reconcile snapshots
- maintain a small cached live snapshot for the selected symbol/timeframe
- merge historical candles with live spot/trendbar updates into `MarketChartSnapshot`
- replace `cTrader execution feed is not wired yet` with real reconcile-backed data when available

The UI remains snapshot-driven and should not learn the cTrader protocol directly.

## Runtime flow

The documented runtime sequence is:

1. restore or create a valid cTrader token session
2. app-auth the Open API connection with `ProtoOAApplicationAuthReq`
3. account-auth the selected `ctidTraderAccountId` with `ProtoOAAccountAuthReq`
4. optionally request trader metadata with `ProtoOATraderReq`
5. request reconcile snapshot with `ProtoOAReconcileReq`
6. subscribe to spots with `ProtoOASubscribeSpotsReq`
7. subscribe to live trendbars with `ProtoOASubscribeLiveTrendbarReq`
8. consume `ProtoOASpotEvent` updates and project them into app-level market snapshots

If any step fails, the app must:

- keep the snapshot honest
- surface the exact degraded reason to the UI
- avoid fake positions/orders or fake live price updates

## Data contracts

This tranche introduces app-level contracts instead of leaking protocol envelopes.

### Account snapshot

- `CTraderTraderSnapshot`
  - `account_id`
  - `broker_name`
  - `account_type`
  - `balance`
  - `equity`
  - `margin`
  - `free_margin`
  - `leverage`
  - `is_live`

### Reconcile snapshot

- `CTraderReconcileSnapshot`
  - `account_id`
  - `positions`
  - `pending_orders`
  - `protection_orders` when returned

This contract is then projected into the existing `ExecutionSurfaceSnapshot`.

### Live market snapshot

- `CTraderSpotSnapshot`
  - `symbol_id`
  - `bid`
  - `ask`
  - `timestamp_ms`
- `CTraderLiveTrendbarUpdate`
  - `symbol_id`
  - `timeframe`
  - `open`
  - `high`
  - `low`
  - `close`
  - `timestamp_ms`

These are then projected into the existing `MarketChartSnapshot`.

## UI behavior

The `Chart` panel stays snapshot-driven.

- when cTrader account auth and subscriptions are active:
  - historical candles come from the already-implemented cTrader historical path
  - newest market state is updated from live spot/trendbar events
- when the runtime is not active:
  - the chart stays degraded with explicit warnings

The `Execution` panel and bottom strip:

- show reconcile-backed positions and pending orders for cTrader when available
- otherwise keep explicit warnings and diagnostics

No fake success path is allowed.

## Testing

This tranche must be developed with TDD and verified through:

- protocol-builder unit tests
- parser unit tests
- adapter tests with stub transports
- `TradingSession` tests for:
  - account-authenticated cTrader reconcile snapshots
  - cTrader live market snapshot projection
  - honest degraded behavior when auth/account/subscription requirements are missing
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace -- --nocapture`

## Out of scope

This tranche does not include:

- order submit / amend / cancel / close
- multi-account execution fan-out
- refresh-token renewal automation
- demo/live account creation
- depth-of-market UI rendering

Those remain later documented tranches.
