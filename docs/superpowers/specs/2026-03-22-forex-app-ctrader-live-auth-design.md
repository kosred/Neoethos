# Forex App cTrader Live Auth Design

## Goal

Add the first real live `cTrader` authentication flow to `crates/forex-app` using the approved operator model:

- open the system browser for login
- receive the authorization callback through a local loopback listener
- exchange the authorization code for tokens
- persist the token bundle securely per operating system
- restore the `cTrader` session after restart

This tranche must extend the current typed auth state machine instead of bypassing it with ad-hoc network code.

## Scope

This design covers:

- system-browser `cTrader` login launch
- fixed primary callback port with small fallback port set
- local loopback callback listener
- real token exchange call to the documented `cTrader` token endpoint
- secure token storage in OS-native secret stores
- restore/delete session behavior in the app service layer
- operator-visible status and actions in `System`

This design does not cover:

- live `ProtoOAApplicationAuthReq` / `ProtoOAAccountAuthReq` account socket flows
- market-data or live-order `cTrader` execution
- automatic refresh-token renewal loops
- encrypted plaintext file fallback
- `DXtrade` live auth

## Architecture

Recommended approach:

- keep `ctrader_auth.rs` as the typed auth/session state machine
- add a focused `ctrader_live_auth.rs` module for live browser/callback/token-exchange orchestration
- add a focused `secure_store.rs` module for token persistence
- keep `TradingSession` as the owner of the active `cTrader` auth/session state
- surface all operator actions through `System`, not through scattered ad-hoc buttons

This keeps the network transport, secure storage, and UI wiring separated from each other, while preserving the existing `TradingSession` boundary that already owns adapter readiness and auth progression.

## Auth Model

The existing `CTraderAuthState` should be extended with live-runtime states:

- `NotConfigured`
- `ReadyToAuthorize`
- `AwaitingAuthorizationCode`
- `ListeningForCallback`
- `AuthorizationCodeReceived`
- `ExchangingToken`
- `AccessTokenReady`
- `RestoredFromStorage`
- `AccountsAvailable`
- `Failed`

The snapshot should expose:

- current state
- status line
- authorize URL
- callback port in use
- authorization code present flag
- token request ready flag
- token persistence status
- account count
- enabled target count
- next operator action

The live-auth module should introduce:

- `CTraderTokenBundle`
  - `access_token`
  - `refresh_token`
  - `token_type`
  - `expires_in`
  - `scope`
  - `created_at_unix`
- `CTraderLoopbackConfig`
  - `primary_port`
  - `fallback_ports`
  - `callback_path`
- `CTraderTokenExchangeResponse`
  - typed token payload mapped from the documented endpoint

## Runtime Flow

The production flow should be:

1. operator configures `client_id`, `client_secret`, `redirect_uri`
2. app binds the first available loopback port from the allowed port set
3. app starts a one-shot local callback listener
4. app opens the system browser with the authorize URL
5. `cTrader` redirects back with the authorization code
6. app validates the callback path and extracts the `code`
7. app calls the documented token endpoint
8. app persists the token bundle through the OS secure store
9. app updates the auth snapshot to `RestoredFromStorage` or `AccessTokenReady`
10. app can restore the session on future startup or adapter selection

Failure rules:

- if no allowed callback port can bind, fail explicitly
- if browser launch fails, fail explicitly
- if callback payload is malformed or missing `code`, fail explicitly
- if token exchange fails, do not mark the session authenticated
- if secure storage fails, do not mark the session complete
- no plaintext token fallback

## Secure Storage

Token persistence must use OS-native secure storage:

- Windows Credential Manager
- macOS Keychain
- Linux Secret Service or equivalent supported by the selected secure-store crate

The app may persist non-secret metadata in normal config later, but access/refresh tokens must not be written to plain files in this tranche.

The restore path must:

- load the saved token bundle
- validate that required fields are present
- hydrate the `cTrader` auth snapshot into a restored state
- allow explicit operator action to clear the saved session

## UI Changes

The `System` panel should gain a real `cTrader` live-auth control area when the selected adapter is `cTrader`:

- `Start cTrader Login`
- callback port / redirect status
- current live auth state
- token persistence status
- restored-session status on startup
- `Clear Saved Session`

The UI must remain honest:

- if the browser launch succeeded but the callback was never received, show waiting state
- if tokens were exchanged but not saved, show failure
- if a session was restored, say so explicitly

## Dependencies

This tranche is expected to add:

- `reqwest` for HTTP token exchange
- `serde` / `serde_json` for typed token payloads
- `keyring` for secure token persistence
- a small browser-launch helper such as `open`
- existing `tokio` runtime for loopback listener orchestration

## Verification

This tranche is complete only if:

- failing tests prove the live auth flow was added intentionally
- `cargo test -p forex-app -- --nocapture` passes
- `cargo clippy -p forex-app --all-targets -- -D warnings` passes
- `cargo test --workspace` passes
- `cargo clippy --workspace --all-targets -- -D warnings` passes
- `target/debug/forex-app.exe --headless --local --config config.yaml` still succeeds

## References

- [cTrader Open API](https://help.ctrader.com/open-api/)
- [cTrader App and account authentication](https://help.ctrader.com/open-api/account-authentication/)
- [cTrader application registration](https://help.ctrader.com/open-api/api-application/)
- [Rust keyring crate](https://docs.rs/keyring/latest/keyring/)
- [Rust oauth2 crate](https://docs.rs/oauth2/latest/oauth2/)
