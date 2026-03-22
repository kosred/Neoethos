# Forex App Broker Credentials Readiness Design

## Goal

Add a broker credentials and execution-target readiness layer to `crates/forex-app` so the UI can model `MT5`, `cTrader`, and `DXtrade` as real operator-configurable runtimes before live remote authentication is wired.

This tranche is intentionally limited to configuration, readiness, and multi-account target selection. It does not attempt live OAuth flows, token exchange, secure persistence, or remote API calls yet.

## Scope

This design covers:

- app-level broker configuration state
- adapter readiness validation
- multi-account execution target selection
- `System` and `Execution` panel integration
- honest connect gating and operator feedback

This design does not cover:

- real `cTrader` OAuth browser flow
- real `DXtrade` login/network calls
- token storage or refresh handling
- order fan-out / copy trading execution
- persistence to disk or OS keychain

## Product Direction

The app already treats trading connectivity as adapter-based. This tranche turns that into a real operator-facing configuration surface:

- the user can choose an adapter
- the user can enter adapter-specific configuration
- the app can say whether that adapter is:
  - incomplete
  - ready for auth
  - authenticated
  - failed
- the user can define multiple execution targets now, so the later copy-trading engine lands on a stable contract instead of another refactor

## Architecture

Recommended approach:

- keep broker configuration state inside `TradingSession`, not `AppState`
- introduce a focused `broker_config` service module
- keep the UI thin:
  - `System` renders forms and target toggles
  - `Execution` asks the session if connect is allowed
- keep remote adapters explicit and honest:
  - missing credentials disable connect
  - configured-but-unwired remote adapters report `ready` but still say live auth is not implemented

This keeps business rules out of rendering code and avoids touching unrelated app state across the workspace.

## Configuration Model

The new configuration layer should include:

- `BrokerSettingsState`
  - `mt5`
  - `ctrader`
  - `dxtrade`

- `BrokerAccountTarget`
  - account id
  - label
  - execution enabled flag

- `BrokerSessionState`
  - `Disconnected`
  - `Configured`
  - `ReadyForAuth`
  - `Authenticated`
  - `Failed`

- `AdapterReadinessSnapshot`
  - adapter name
  - session state
  - status line
  - missing required fields
  - enabled execution target count
  - connect allowed flag

For `cTrader`, readiness should follow the documented OAuth application contract:

- `client_id`
- `client_secret`
- `redirect_uri`

For `DXtrade`, readiness should stay generic and instance-oriented:

- `platform_url`
- `username`
- `password`

For `MT5`, readiness should remain local-terminal oriented and should not regress the current local bridge flow.

## UI Changes

`System` should gain:

- adapter-specific config form
- readiness summary
- missing field reporting
- account target list with enable/disable toggles
- target count summary

`Execution` should gain:

- connect button disabled when the selected adapter is not ready
- explicit hover/help reason for the disabled state
- explicit degraded message when a remote adapter is configured but live auth is not wired yet

## Verification

This tranche is complete only if:

- focused readiness tests fail first and then pass
- `cargo test -p forex-app -- --nocapture` passes
- `cargo clippy -p forex-app --all-targets -- -D warnings` passes
- `cargo test --workspace` passes
- `cargo clippy --workspace --all-targets -- -D warnings` passes
- `target/debug/forex-app.exe --headless --local --config config.yaml` still succeeds

## References

- [cTrader Open API](https://help.ctrader.com/open-api/)
- [cTrader Open API account authentication](https://help.ctrader.com/open-api/account-authentication/)
- [DXtrade APIs](https://dx.trade/apis/)
