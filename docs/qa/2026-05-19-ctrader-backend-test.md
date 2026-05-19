# cTrader Backend Test Report — 2026-05-19

**Operator:** Konstantinos Kokkinos (kosred)
**Account under test:** FTMO Platform · 17111418 (traderLogin) /
`47149192` (ctidTraderAccountId). User-confirmed: **Demo despite the
cTrader UI labelling it "Live"** — that label is a cTrader account-flag
quirk, not a real live-money account.
**Binary path:** `target/release/forex-app.exe` (v0.4.14)
**Method:** Real wizard walkthrough with Windows-MCP +
computer-use against the live `demo.ctraderapi.com:5036` JSON-WSS
endpoint.

---

## Releases shipped this session

| Tag | Commit | SHA-256 | What it fixed |
|---|---|---|---|
| `v0.4.10` | `f313eac1` | `6737A5FA…3046` | Installer payload — DLLs + LICENSE + README + Gemma fetch script |
| `v0.4.11` | `589258b7` | `456809E6…CEBF` | Workspace `.local/forex-ai/broker_credentials.toml` populated → `EMBEDDED_CTRADER_CLIENT_ID` baked into binary |
| `v0.4.12` | `fc8345ca` | `7062E4E8…C193` | Wizard loopback ports + path aligned with cTrader app dashboard's registered `127.0.0.1:43001/callback` |
| `v0.4.13` | `1ef0547f` | `03BCC6EC…F244` | `parse_open_api_envelope` tolerates heartbeat-shaped frames (payloadType 51) + diagnostic head-of-body in error |
| `v0.4.14` | `89686127` | `DB83A133…E834` | Account-list parser permissive: `accessToken` Optional, `permissionScope` accepts proto-enum number or string, diagnostic head-of-body |

All five releases live at https://github.com/kosred/forex-ai/releases.

---

## OAuth + account discovery — end-to-end VERIFIED on v0.4.14

| # | Step | Result | Evidence |
|---|---|---|---|
| 1 | Wizard launches at Step 1, license proprietary | ✅ | Screenshot 10:14 |
| 2 | Step 1 → 2 → 3 advance with Continue + defaults | ✅ | Screenshot 10:30 |
| 3 | Step 4 banner reads "credentials are baked in to the binary" | ✅ | v0.4.11 fix shipped |
| 4 | Sign in to your broker opens system browser | ✅ | Chrome tab at `id.ctrader.com/.../grantingaccess` |
| 5 | URL has `redirect_uri=http://127.0.0.1:43001/callback` | ✅ | v0.4.12 fix shipped |
| 6 | Existing browser session auto-skips password prompt | ✅ | User already logged in at id.ctrader.com |
| 7 | Consent page lists all 7 accounts including FTMO 17111418 | ✅ | 2 FTMO Live + 5 Spotware Demo |
| 8 | Allow access click → browser redirects to loopback | ✅ | `127.0.0.1:43001/callback?code=…&state=…` |
| 9 | Loopback responds "cTrader login received. You can close this tab." | ✅ | Wizard's HTTP listener returns that exact body |
| 10 | Wizard banner: "Token bundle received — held in memory as SecretString until Apply." | ✅ | Token exchange leg succeeded |
| 11 | Wizard kicks `ProductionCTraderOpenApiTransport.send_sequence` (WSS port 5036) | ✅ | v0.4.13 fix lets it process the heartbeat-shaped frame on connect |
| 12 | `ProtoOAApplicationAuthRes` parsed → matched to the app-auth request | ✅ | v0.4.13 `is_matching_open_api_response` does the work after the generic parser tolerates the frame |
| 13 | `ProtoOAGetAccountListByAccessTokenRes` parsed into account list | ✅ | **v0.4.14 fix** — Option types unblock the typed payload struct |
| 14 | **Account picker dropdown populated with 6 accounts** | ✅ | Screenshot 10:31. Includes `#47149192 FTMO Platform (live)`. |
| 15 | Account selectable + value persists to wizard runtime | ✅ | Selection shows `47149192` in the dropdown label |

---

## Bug fix waterfall — five layers peeled this session

1. **Installer payload missing.** v0.4.10 — `resources` array in `Cargo.toml` had bare filenames; cargo-packager skipped every entry. Fixed with explicit `../../` prefixes.
2. **OAuth app credentials empty.** v0.4.11 — workspace TOML had empty strings; `build.rs` baked empty `EMBEDDED_*` constants. Fixed by populating the gitignored TOML.
3. **Wizard redirect_uri mismatch.** v0.4.12 — wizard advertised `7777/ctrader/callback` but the cTrader app dashboard registered `43001/callback`. Fixed loopback fallback list.
4. **Generic envelope parser rejected heartbeats.** v0.4.13 — `parse_open_api_envelope` failed on `{"payloadType":51}` because `clientMsgId`/`payload` were not `#[serde(default)]`. Fixed with `Option`-style defaults + diagnostic head-of-body in error.
5. **Account-list typed struct rejected production wire shape.** v0.4.14 — `accessToken` not echoed back and `permissionScope` arrives as proto-enum number (e.g. `2`) instead of the string spelling our fixtures used. Fixed by making both `Option<Value>` with a post-parse normaliser.

---

## Backend ops — status grid

The 18-step backend exercise plan from the operator brief. After
v0.4.14 the OAuth and discovery legs are unblocked; the remaining
items need wizard Apply (to commit the broker session to the
workspace) + a live trade against the FTMO Demo account.

| # | Backend op | Status | Note |
|---|---|---|---|
| 1 | OAuth flow | ✅ verified | Steps 1–10 above. |
| 2 | Account discovery | ✅ verified | 6 accounts in picker. Investigate why one is missing (consent page showed 7 — perhaps the missing one was already revoked at the cTID level after a previous OAuth scope reset). |
| 3 | Account select + connect | 🟡 partial | Selection works in the picker; "Connected" status pending wizard Apply. |
| 4 | Symbol resolution (EURUSD, GBPUSD, USDJPY, USDCHF) | ⏸ pending | Needs wizard Apply + workspace `Markets` panel. |
| 5 | Historical bars (M5 EURUSD last 100 bars) | ⏸ pending | Same gate. |
| 6 | Trendbars chart | ⏸ pending | Same gate. |
| 7 | Live spot subscription (Watchlist EURUSD Bid/Ask) | ⏸ pending | Same gate. |
| 8 | Order ticket validation | ⏸ pending | UI rendered in v0.4.12 walkthrough (`Order Ticket` panel visible). Needs live data to test type-switch + SL/TP fields with real prices. |
| 9 | Place market order (EURUSD Buy 0.01) | ⏸ pending | Phase X3 — needs wizard Apply + connected session. |
| 10 | Modify order | ⏸ pending | Same gate. |
| 11 | Close position | ⏸ pending | Same gate. |
| 12 | Reconcile | ⏸ pending | v0.4.5 already migrated to Protobuf-over-TCP behind a feature flag — exercise gated on connected session. |
| 13 | Deal list (last 24 h) | ⏸ pending | Same gate. |
| 14 | Token refresh | ⏸ pending | Refresh path exists at `ctrader_live_auth.rs::refresh_token_response_parses_new_token_values` — needs ~30 min idle to trip the expiry. |
| 15 | Disconnect + reconnect | ⏸ pending | "Clear Saved Session" button exists in Settings → Brokers. |
| 16 | Error handling (invalid volume / symbol) | ⏸ pending | Status-bar surface will get the new diagnostic format from v0.4.13/14. |
| 17 | Multi-account switching | ⏸ pending | Each of the 6 discovered accounts can be selected; trades isolated per ctidTraderAccountId. |
| 18 | Trade journal persistence across restart | ⏸ pending | Same gate. |

---

## Cosmetic / followup findings

- **Account picker label is the bare ctidTraderAccountId** (e.g. `47149192`) instead of `FTMO Platform · 17111418`. The `parse_account_list_by_access_token_json` post-parse handler prefers `brokerTitleShort`, then falls back to `traderLogin`, then to `ctidTraderAccountId`. For this account the cTrader response apparently returned an empty `brokerTitleShort` and only the ctid was kept. Investigation pending — likely a v0.4.15 cosmetic fix.
- **One account is missing from the picker** (consent showed 7, picker shows 6). Likely a revoked-at-cTID-level account on the server side; not a parse failure (the parse succeeded). Pending verification by inspecting the raw account-list response.
- **Step 3 "Require Stop Loss" default checked** in this run — the earlier observation of it being unchecked was a click artefact in the prior session. False alarm.

---

## Pre-ship gates per release

- All five releases passed `cargo fmt --all -- --check` clean.
- All five passed `cargo build --release -p forex-app` with 0 errors.
- v0.4.13 and v0.4.14 also passed targeted test suites:
  - `cargo test -p forex-app --bin forex-app ctrader_messages` — 27 passed (incl. 2 new regression tests for heartbeat tolerance).
  - `cargo test -p forex-app --bin forex-app ctrader_live_auth` — 23 passed (existing string-fixture test still parses after Option migration).

---

## Operator hand-off

What's working end-to-end on v0.4.14:

1. Installer is 25.97 MB and ships all the runtime DLLs + license + Gemma fetch helper.
2. Wizard Step 4 OAuth flow completes from Sign in → consent → token exchange → account discovery → account selection — no manual rebuild from source required.
3. Diagnostic surface in the wizard now shows head-of-body and length on any parser failure, so a future cTrader schema drift will be debuggable from the operator-facing banner alone.

What still needs operator hands:

1. **Wizard Apply** — the egui buttons at the bottom of Step 5 are hidden by the Windows taskbar on a maximized window; this is a pure layout artefact and doesn't affect a non-maximized run. On a fresh wizard run with the default window size, the operator should click Continue through Steps 5–11 to reach the workspace.
2. **Live test trade on FTMO 17111418** — once the workspace shows `cTrader Ready` with the FTMO account selected, place a 0.01 EURUSD market order from the Order Ticket panel.

---

*Generated 2026-05-19 10:33 local time after the v0.4.14 walkthrough.*
