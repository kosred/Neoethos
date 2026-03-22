# Forex App cTrader Auth Session Design

## Goal

Add the first real `cTrader` authentication/session contract to `crates/forex-app`, built on top of the new broker readiness layer, without yet introducing live HTTP calls or token persistence.

The immediate target is a correct operator-visible auth state machine:

- build authorize URL
- enter or receive authorization code
- build token exchange request contract
- represent account-list/session readiness
- surface these states cleanly in the UI

## Scope

This design covers:

- `cTrader` auth state machine
- authorization URL construction
- authorization code capture contract
- token exchange request contract
- account list snapshot contract
- `System` panel auth state/reporting

This design does not cover:

- real browser launch flow
- real localhost callback listener
- real token exchange HTTP calls
- token refresh handling
- secure token persistence
- live account auth protobuf exchange

## Architecture

Recommended approach:

- add a focused `ctrader_auth` service module
- keep `TradingSession` as the owner of the current auth session state
- keep `System` as the primary operator surface for auth progression
- do not leak raw stringly auth state across the UI

The state machine should be explicit and typed so the next tranche can add real network transport without rewriting the UI contract.

## Auth Model

The first auth model should include:

- `CTraderAuthState`
  - `NotConfigured`
  - `ReadyToAuthorize`
  - `AwaitingAuthorizationCode`
  - `AuthorizationCodeReceived`
  - `AccessTokenReady`
  - `AccountsAvailable`
  - `Authenticated`
  - `Failed`

- `CTraderAuthSnapshot`
  - current state
  - status line
  - authorize URL
  - authorization code present flag
  - token request ready flag
  - account count
  - next operator action

- `CTraderTokenExchangeRequest`
  - `grant_type`
  - `code`
  - `client_id`
  - `client_secret`
  - `redirect_uri`

- `CTraderAccountSummary`
  - account id
  - broker title / label
  - execution enabled flag

## Behavior

The auth contract should behave like this:

- missing `client_id` or `redirect_uri` -> `NotConfigured`
- valid readiness config -> `ReadyToAuthorize`
- `start_ctrader_auth()` -> `AwaitingAuthorizationCode` and expose authorize URL
- `receive_ctrader_authorization_code(code)` -> `AuthorizationCodeReceived`
- `build_ctrader_token_exchange_request()` -> returns typed request and moves snapshot to `AccessTokenReady`
- `set_ctrader_accounts(accounts)` -> `AccountsAvailable`

No real network call should happen yet.

## UI Changes

`System` should gain a `cTrader Auth` section when the selected adapter is `cTrader`:

- auth state badge/text
- authorize URL availability
- `Start cTrader Auth`
- local authorization code field
- token request readiness summary
- account list summary when present

The UI should never pretend that auth completed if only a code was received.

## Verification

This tranche is complete only if:

- failing tests prove the state machine was added intentionally
- `cargo test -p forex-app -- --nocapture` passes
- `cargo clippy -p forex-app --all-targets -- -D warnings` passes
- `cargo test --workspace` passes
- `cargo clippy --workspace --all-targets -- -D warnings` passes
- `target/debug/forex-app.exe --headless --local --config config.yaml` still succeeds

## References

- [cTrader Open API](https://help.ctrader.com/open-api/)
- [cTrader Open API account authentication](https://help.ctrader.com/open-api/account-authentication/)
