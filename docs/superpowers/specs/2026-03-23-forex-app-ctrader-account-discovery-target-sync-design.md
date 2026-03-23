# Forex App cTrader Account Discovery And Target Sync Design

**Date:** 2026-03-23  
**Status:** Approved for implementation  
**Scope:** `forex-app` cTrader post-auth account discovery, account catalog surfacing, and execution-target synchronization

---

## 1. Goal

After successful `cTrader` OAuth login or secure-session restore, the app must discover the granted trader accounts, surface them in the `System` UI, and synchronize them into the existing execution-target model without breaking `MT5` or `DXtrade`.

This tranche does **not** implement full cTrader market/trading streams yet. It only delivers:

- app-auth/account-discovery transport seam
- discovered-account catalog
- safe merge into `BrokerAccountTarget`
- operator target selection from discovered accounts
- honest session/readiness/status reporting

---

## 2. Why This Tranche Exists

The current repo already has:

- real `cTrader` browser login
- loopback callback capture
- real token exchange
- secure token persistence
- session restore on startup

But it still does **not** have real account discovery. The current `AccountsAvailable` state is reached by replaying manually typed UI account targets back into the auth session, not by querying the broker.

That is the wrong boundary for multi-account execution and future copy-trading.

---

## 3. Official cTrader Contract

Based on the official `cTrader Open API` docs reviewed on 2026-03-23:

- after OAuth token acquisition, the correct sequence is:
  - `ProtoOAApplicationAuthReq`
  - `ProtoOAGetAccountListByAccessTokenReq`
  - `ProtoOAAccountAuthReq` per selected account
- one access token can enumerate the accounts already granted for that cTID
- additional accounts created later require the user to go through consent again
- live and demo accounts must be separated by `isLive` and authorized on the matching environment connection

Reference pages:

- [cTrader Open API account authentication](https://help.ctrader.com/open-api/account-authentication/)
- [cTrader Open API connection](https://help.ctrader.com/open-api/connection/)
- [cTrader Open API messages](https://help.ctrader.com/open-api/messages/)

---

## 4. Current Local Constraints

Relevant current files:

- `crates/forex-app/src/app_services/ctrader_auth.rs`
- `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- `crates/forex-app/src/app_services/secure_store.rs`
- `crates/forex-app/src/app_services/trading.rs`
- `crates/forex-app/src/app_services/broker_config.rs`
- `crates/forex-app/src/ui/system_status.rs`

Current gaps:

- `BrokerAccountTarget` is too thin to represent discovered remote accounts cleanly
- `CTraderAuthSnapshot` exposes counts but not the discovered catalog
- `TradingSession::connect()` still says remote live auth is not wired yet
- `reset_runtime_state()` clears in-memory cTrader auth/discovery state aggressively
- `System` UI still treats cTrader targets as freeform text boxes

---

## 5. Architecture

### 5.1 Separate Truths

The implementation must keep two separate truths:

- **Discovered accounts**
  - what `cTrader` says exists and is currently granted
- **Execution targets**
  - which accounts the operator has armed for fan-out execution

These are related but not identical.

### 5.2 Compatibility Rule

Do **not** replace the generic `BrokerAccountTarget` contract with a cTrader-specific type.

Instead:

- keep `BrokerAccountTarget` as the shared execution-target shape
- add a cTrader-only discovered-account catalog next to it
- synchronize the discovered catalog into targets by `account_id`
- preserve operator target toggles during resync

### 5.3 State Rule

The app must distinguish:

- token restored
- accounts discovered
- targets synchronized
- accounts account-authenticated

Those are different states and must not be collapsed into one generic “ready” status.

---

## 6. File Layout

### Modify

- `crates/forex-app/src/app_services/ctrader_auth.rs`
  - enrich auth/account state
- `crates/forex-app/src/app_services/ctrader_live_auth.rs`
  - add account-discovery transport contract
- `crates/forex-app/src/app_services/broker_config.rs`
  - add minimal discovery metadata and richer readiness states
- `crates/forex-app/src/app_services/trading.rs`
  - orchestrate discovery + sync + UI-visible status
- `crates/forex-app/src/ui/system_status.rs`
  - render discovered accounts as selectable targets

### Optional small helper file if needed

- `crates/forex-app/src/app_services/ctrader_account_transport.rs`
  - only if the transport seam becomes too large for `ctrader_live_auth.rs`

This tranche should avoid growing any one file significantly if a clean extraction is warranted.

---

## 7. Data Model

### 7.1 New cTrader discovered account shape

Add a richer discovered-account struct for cTrader, for example:

- `account_id`
- `broker_title`
- `trader_login`
- `is_live`
- `enabled_for_execution`
- `available`

This may extend `CTraderAccountSummary` or sit alongside it, but the model must be rich enough to:

- render the discovered catalog in UI
- merge into targets safely
- differentiate live vs demo

### 7.2 Shared target model

Keep `BrokerAccountTarget`, but extend only if needed with generic optional metadata such as:

- `source_adapter: Option<String>`
- `discovered: bool`
- `available: bool`

If these fields are added, they must remain optional and must not break current `MT5` and `DXtrade` behavior.

### 7.3 Readiness/session state

`BrokerSessionState` should evolve beyond:

- `Disconnected`
- `Configured`
- `ReadyForAuth`
- `Authenticated`
- `Failed`

It should be able to express:

- `Authorizing`
- `SessionRestored`
- `DiscoveringAccounts`
- `AccountsAvailable`

without lying about true runtime readiness.

---

## 8. Runtime Flow

### Happy path

1. Operator completes cTrader app credentials
2. Operator starts login or restores saved session
3. App obtains/restores token bundle
4. App opens the cTrader transport seam
5. App app-authenticates the connection
6. App requests granted account list
7. App stores discovered accounts in the cTrader auth/session state
8. App synchronizes discovered accounts into `broker_settings.ctrader.accounts`
9. Existing execution target toggles are preserved by `account_id`
10. UI renders discovered catalog + enabled target counts

### Resync path

1. Operator runs account discovery again
2. App refreshes discovered catalog
3. Existing target rows are merged by `account_id`
4. Missing accounts are marked unavailable/stale, not silently deleted

### Failure rules

- if app auth fails: mark explicit `Failed`
- if account-list request fails: mark explicit `Failed`
- if no granted accounts are returned: explicit degraded/empty state, not fake success
- if restored tokens exist but account discovery has not yet run: show `SessionRestored`, not `AccountsAvailable`

---

## 9. UI Contract

The `System` panel must change from freeform cTrader account entry to a hybrid discovery-driven view:

- keep credential fields
- keep auth actions
- add `Discover Accounts`
- show discovered account table
- allow `enabled_for_execution` toggles directly on discovered accounts
- keep manual editing only as fallback, not the primary flow

The UI must display:

- auth/session state
- account discovery state
- discovered count
- enabled target count
- live/demo distinction if available
- explicit reasons when discovery is unavailable

---

## 10. Testing Strategy

### Unit tests

- discovered accounts are retained in auth snapshot
- discovered accounts sync into targets by `account_id`
- existing `enabled_for_execution` survives resync
- missing discovered accounts are marked unavailable, not silently dropped
- `restore_ctrader_session()` does not incorrectly claim accounts are available
- readiness/session state reflects `SessionRestored` vs `AccountsAvailable`

### UI tests

- system dashboard shows discovered accounts section
- discovered accounts surface target count correctly
- cTrader manual fallback is suppressed or de-emphasized when discovery data exists

### Verification commands

- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `target/debug/forex-app.exe --headless --local --config config.yaml`

---

## 11. Risks

- losing enabled targets on account resync
- mixing secret storage with non-secret discovered-account metadata
- clearing discovery state accidentally in `reset_runtime_state()`
- overstating readiness after token restore but before account discovery
- coupling `MT5` or `DXtrade` to cTrader-specific discovery fields

---

## 12. Non-Goals

This tranche does **not** include:

- full cTrader price stream
- full cTrader trading operations
- copy-trading fan-out execution
- refresh-token renewal daemon
- DXtrade live auth
- chart overlays from cTrader fills

Those are follow-up tranches.

---

## 13. Acceptance Criteria

- cTrader account discovery is backed by real cTrader account-list flow, not manual replay
- discovered accounts are visible in the UI
- execution targets are synchronized from discovered accounts without losing operator toggles
- restored token-only state is distinguished from accounts-discovered state
- no new warnings on verified paths
- `MT5` and `DXtrade` behavior remains intact
