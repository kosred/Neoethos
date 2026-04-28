# cTrader OpenAPI Integration Audit Report

**Date**: 2026-04-27  
**Scope**: Static code audit of all `ctrader_*.rs` modules + execution of existing unit-test suite + new integration tests  
**Audited files**:
- `ctrader_auth.rs` — token model
- `ctrader_live_auth.rs` — OAuth 2.0 loopback flow, account discovery
- `ctrader_messages.rs` — payload-type constants, JSON message builders, production transport
- `ctrader_data.rs` — symbol resolution, historical bars & tick data
- `ctrader_execution.rs` — order execution, session singleton, idempotency cache
- `ctrader_streaming.rs` — live spot/trendbar subscriptions
- `ctrader_session.rs` — async session (not wired)
- `ctrader_proto_messages.rs` — protobuf helpers (not wired)
- `app_services/trading.rs` — wiring / orchestration

---

## 1. Feature Status Matrix

| Feature | Code Present | Tests Pass | Wired to UI | Requires User Action |
|---|---|---|---|---|
| OAuth login (browser-based) | ✅ | ✅ | ✅ | Live credentials |
| Token storage (keyring) | ✅ | ✅ | ✅ | — |
| Token auto-refresh | ✅ | ✅ | ✅ | — |
| Account discovery (list accounts) | ✅ | ✅ | ✅ | Live credentials |
| Symbol resolution (name → id) | ✅ | ✅ | ✅ | — |
| Historical bars (trendbars) | ✅ | ✅ | ✅ | — |
| Historical tick data | ✅ | ✅ | ✅ | — |
| Live spot quote streaming | ✅ | ✅ | ✅ | Live session |
| Live trendbar streaming | ✅ | ✅ | ✅ | Live session |
| New order (market/limit/stop) | ✅ | ✅ | ✅ | Live credentials |
| Amend order | ✅ | ✅ | ✅ | Live credentials |
| Cancel order | ✅ | ✅ | ✅ | Live credentials |
| Close position | ✅ | ✅ | ✅ | Live credentials |
| Reconcile (open positions/orders) | ✅ | ✅ | ✅ | — |
| Deal list (trade history) | ✅ | ✅ | ✅ | — |
| Async session / heartbeat | ✅ code | ❌ not wired | ❌ | — |

---

## 2. Architecture Overview

The cTrader integration uses **two transport layers**:

### 2a. Synchronous `tungstenite` (active, used everywhere)
All production data paths open a new TLS WebSocket connection via `tungstenite::connect()`, run a sequential request-response chain, and close the connection. No persistent connection is maintained between API calls.

- **Data** (`ctrader_data.rs`): opens connection → app-auth → account-auth → symbols-list → symbol-by-id → bars/ticks → close
- **Execution** (`ctrader_execution.rs`): maintains a session singleton with 30-second auth key reuse; 2-attempt retry with reconnect on failure
- **Streaming** (`ctrader_streaming.rs`): maintains a session singleton; subscribe-spots + subscribe-trendbars; polls for spot events on-demand

### 2b. Async `tokio-tungstenite` (NOT wired — dead code)
`ctrader_session.rs` implements a full-duplex async session with a 10-second heartbeat loop and read/write task separation. This module has `#![allow(dead_code)]` at the top and is **not connected to any live code path**. It was likely prototyped for a future persistent-connection mode.

---

## 3. Bugs Found & Fixed

### BUG-001 — Misleading constant arithmetic in `execute_with_transport` (FIXED)

**File**: `crates/forex-app/src/app_services/ctrader_execution.rs`, line 542  
**Function**: `execute_with_transport` (marked `#[allow(dead_code)]`)

**Before**:
```rust
ensure_payload_type(
    &responses[0],
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE - 2,  // 2103-2 = 2101
)?;
```

**After**:
```rust
ensure_payload_type(
    &responses[0],
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,  // 2101
)?;
```

**Impact**: Arithmetically identical at runtime (2103-2=2101=`APPLICATION_AUTH_RESPONSE`). However it is a latent correctness trap: if the constants were ever reordered or if a reader changes one without the other, validation would silently accept a wrong payload type. The production path `execute_via_session` already uses the correct named constant on line 453. Fixed to match.

### BUG-002 — `TradingPanelMode::Disconnected` returned live status instead of "Offline" (FIXED)

**File**: `crates/forex-app/src/app_services/trading.rs`, `snapshot()` function  

**Before**: `TradingPanelMode::Disconnected => state.status_msg.clone()` — returned `"cTrader Ready"` for a fresh disconnected session, causing `is_online` detection in the watchlist to misfire.

**After**: `TradingPanelMode::Disconnected => "Offline".to_string()` — always shows "Offline" when not connected, consistent with the `is_online` check `!= "Offline"` in `watchlist_panel.rs:28`.

---

## 4. Observations & Recommendations

### OBS-001 — Demo "opens in browser" is correct behavior (not a bug)

The OAuth 2.0 loopback flow is the only supported authorization method for cTrader OpenAPI. The flow:

1. App binds `127.0.0.1:PORT` TCP listener
2. Calls `open::that(authorize_url)` — opens system browser at `https://id.ctrader.com/my/settings/openapi/grantingaccess/?...`
3. User grants access in browser
4. Browser redirects to `http://127.0.0.1:PORT/callback?code=...`
5. App captures authorization code, exchanges for tokens via HTTPS POST
6. Tokens stored in OS keyring

**For Demo accounts**, cTrader's OAuth server at `id.ctrader.com` is the same; the environment (`demo.ctraderapi.com` vs `live.ctraderapi.com`) is only used for the WebSocket API, not for login. So Demo login also opens the browser. This is correct and cannot be changed without violating cTrader's OAuth policy.

**Improvement opportunity**: Show a progress indicator inside the app while waiting for the callback — currently there's no visible indicator between clicking "Connect" and the callback arriving.

### OBS-002 — New connection per data request (performance)

`ProductionCTraderOpenApiTransport::send_sequence()` opens a fresh TLS WebSocket for every call. For chart history this means 3 separate connections:

- Connection 1: app-auth + account-auth + symbols-list
- Connection 2: symbol-by-id  
- Connection 3: trendbars + bid-ticks + ask-ticks

Each requires a full TLS handshake (~150–300ms on typical network). Low priority improvement: merge auth + data steps into fewer `send_sequence` calls.

### OBS-003 — `ctrader_session.rs` and `ctrader_proto_messages.rs` are dead code

Both modules have `#![allow(dead_code)]` and implement a persistent heartbeat-enabled async session not wired to any live code path. These are safe to leave as-is until a future persistent-streaming milestone.

### OBS-004 — Streaming session uses global singleton without account keying

`ctrader_streaming.rs` caches the streaming session in a `OnceLock<Mutex<Option<...>>>`. If credentials change, the 30-second staleness TTL forces re-subscribe. Acceptable for single-account use; would need keying by account id for multi-account.

---

## 5. Test Coverage Added

New integration tests in `ctrader_integration_tests.rs` (all using stub transports):

| Test | What it validates |
|---|---|
| `app_auth_request_payload_type_is_2100` | Correct request constant |
| `app_auth_response_constant_is_2101` | Correct response constant |
| `error_response_constant_is_2142` | Error payload type |
| `symbol_resolution_sends_auth_then_symbols_list_then_detail` | Full 4-message sequence, correct ordering |
| `symbol_resolution_is_case_insensitive_and_strips_slash` | "EUR/USD" matches "eurusd" |
| `symbol_resolution_fails_when_symbol_not_in_list` | Error with symbol name in message |
| `symbol_resolution_surfaces_ctrader_error_on_app_auth_failure` | Error payload propagates |
| `bars_only_flow_sends_5_messages_and_returns_bar` | 5-message sequence, price scaling |
| `full_chart_history_flow_sends_7_messages_and_returns_bars_and_ticks` | 7-message sequence + live plan |
| `account_discovery_sends_app_auth_then_account_list` | Correct payload types (2100, 2149) |
| `account_discovery_surfaces_app_auth_error` | Error propagation |
| `demo_environment_uses_demo_endpoint` | `demo.ctraderapi.com` |
| `live_environment_uses_live_endpoint` | `live.ctraderapi.com` |
| `trendbar_price_scaling_5_digits_is_correct` | low/open/close/high delta-decoding |
| `trendbar_timestamp_conversion_minutes_to_ms` | `utcTimestampInMinutes × 60000` |
| `trendbar_period_mapping_covers_all_standard_timeframes` | M1/M5/M15/M30/H1/H4/D1/W1 |
| `trendbar_period_mapping_is_case_insensitive` | "m1", "h1", "d1" |

---

## 6. Items Requiring User Action (Live Testing)

These cannot be validated via static analysis or stub transports:

| Item | What to do |
|---|---|
| **OAuth login end-to-end** | Run app, go to Settings → Broker, configure `client_id` + `client_secret` + `redirect_uri`, click "Connect". Browser opens; grant access; app should show "AccountsAvailable" state. |
| **Demo account live quotes** | After login, select a symbol in the watchlist. Market panel should show live Bid/Ask prices (requires `demo.ctraderapi.com:5036` reachable). |
| **Place test market order** | In Execution panel, select Market order, 0.01 lot, set stop-loss, click Buy. Verify position appears in Exposure section. |
| **Close position** | With open position selected, click Close. Verify it disappears from Exposure. |
| **Token refresh** | Token lifetime is ~30 days. To force test: edit the stored token's `created_at_unix` to simulate near-expiry, restart the app. Should silently refresh and continue. |
| **Streaming continuity** | Keep app open with a chart visible for >30 seconds. Live trendbar updates should merge into the chart candles without gaps. |

---

## 7. Test Run Results

All tests pass after fixes.

```
test result: ok. 195 passed; 0 failed
```

*(run `cargo test -p forex-app` to verify)*
