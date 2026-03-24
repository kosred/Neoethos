# cTrader Token Refresh Session Hardening Design

## Goal

Harden the existing `cTrader` authenticated session so long-running app usage can survive access-token expiry without forcing the operator to log in again.

## Scope

This tranche adds only the documented `cTrader` token refresh lifecycle and secure-store update behavior. It does not add new trading operations, multi-account fan-out, or background daemon infrastructure.

## Official Contract

Based on the official `cTrader Open API` documentation:

- Access tokens expire after about 30 days.
- Refresh tokens do not expire on their own.
- The same `GET https://openapi.ctrader.com/apps/token` endpoint is used for refresh.
- Refresh requests must use `grant_type=refresh_token`.
- A successful refresh returns new `accessToken` and `refreshToken` values.
- The previously issued token values are invalidated after refresh.

## Architecture

The app already has:

- `CTraderTokenBundle` in `ctrader_auth.rs`
- secure persistence in `secure_store.rs`
- live auth/token exchange in `ctrader_live_auth.rs`
- session/account runtime wiring in `trading.rs`

This tranche extends that design with:

1. `Token freshness helpers`
   - add expiry/near-expiry evaluation to `CTraderTokenBundle`
   - expose whether refresh is required before using the session

2. `Refresh request/response contract`
   - add a dedicated refresh request builder in `ctrader_live_auth.rs`
   - reuse the same token response parsing as the initial code exchange

3. `TradingSession auto-refresh seam`
   - before cTrader account discovery, account runtime load, chart history, and live chart requests, the session ensures a fresh token bundle exists
   - refreshed bundles are written back to secure storage immediately
   - if refresh fails, the request fails closed and the existing session is not treated as valid

4. `Auth state continuity`
   - restored sessions remain restored after a successful refresh
   - account-discovered sessions remain account-discovered after a successful refresh
   - no fake “authenticated” state is introduced without a valid stored bundle

## Error Handling

- Missing stored bundle: fail with an explicit “stored token bundle required” style error.
- Missing refresh token: fail closed, do not silently keep using the stale access token.
- Refresh HTTP failure: bubble explicit error, do not degrade to stale-token usage.
- Secure-store save failure after refresh: treat refresh as failed.
- Invalid refreshed payload: fail closed.

## Testing

The tranche requires TDD coverage for:

- refresh URL construction
- expiry / near-expiry helpers
- refresh response parsing
- trading-session auto-refresh before cTrader account-runtime/chart paths
- secure-store overwrite with refreshed bundle

## Non-Goals

- background refresh scheduler
- token refresh via protobuf path
- order execution changes
- training/discovery data-source integration
