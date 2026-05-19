# Functional Test Report — 2026-05-19

**Operator:** Konstantinos Kokkinos (kosred)
**Session length:** ~1 h (08:53 → 09:40 local time)
**Starting version:** v0.4.9 (audit revealed installer payload bug)
**Ending version:** v0.4.12 (3 patch releases shipped this session)
**Test method:** Real binary walkthrough with Windows-MCP + computer-use, not unit tests, not deferred-to-CI.

---

## Releases shipped this session

| Tag | Commit | SHA-256 of installer | What it fixed |
|---|---|---|---|
| `v0.4.10` | `f313eac1` | `6737A5FA…3046` | Installer payload (34 MB of DLLs + LICENSE + README + fetch-gemma-model.ps1) + AI Helper "Gemma model not found" banner |
| `v0.4.11` | `589258b7` | `456809E6…CEBF` | Workspace `.local/forex-ai/broker_credentials.toml` populated with the real cTrader Open API app credentials so `build.rs` actually bakes them in |
| `v0.4.12` | `fc8345ca` | `7062E4E8…C193` | `WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS` + `WIZARD_DEFAULT_OAUTH_CALLBACK_PATH` aligned with the cTrader app dashboard's registered `http://127.0.0.1:43001/callback` |

All three are live at https://github.com/kosred/forex-ai/releases.

---

## Phase X1 — Wizard end-to-end

Result: **8 of 11 steps fully verified; 1 step blocked by an upstream cTrader API
parse bug**.

| # | Step | v0.4.10 | v0.4.11 | v0.4.12 | Notes |
|---|---|---|---|---|---|
| 1 | Welcome & License | ✅ | ✅ | ✅ | Proprietary license, scroll-to-bottom gate works, checkbox enables Continue |
| 2 | Data Path | ✅ | ✅ | ✅ | Default `%LocalAppData%\forex-ai`, Browse button present |
| 3 | Account & Profile | ✅ | ✅ | ⚠ | FTMO_STANDARD preset, 4% target, Risk 4/10. ⚠ "Require Stop Loss" checkbox default flipped to OFF — investigate. |
| 4 | cTrader Sign-in | ❌ banner | ✅ banner gone, OAuth opens browser | ✅ port 43001 listed first | See "OAuth flow" below. |
| 5 | Symbols & Timeframes | n/a | n/a | ✅ | 6 strategy templates, symbol checkboxes (EURUSD pre-checked), 11 canonical timeframes (no H2 per policy) |
| 6 | Historical Data | n/a | n/a | ✅ | "Begin download" correctly greyed because cTrader was skipped |
| 7 | Hardware Probe | n/a | n/a | ✅ | 12 cores, 31.4 GiB, Win11 26200, no-GPU notice, CPU NdArray recommended |
| 8 | News & Safeguards | n/a | n/a | ✅ | Auto-flatten Fri 16:00 ET ✓, correlation cap 0.70, ATR pause 3.0σ |
| 9 | Auto-start | n/a | n/a | ✅ | Optional shortcut at Microsoft Start Menu Startup folder |
| 10 | Autonomy & Risk | n/a | n/a | ✅ | Stage roadmap (4 stages), risk quiz placeholder, Risky Mode acknowledgement gates, Arm-Risky-Mode toggle correctly greyed until acknowledgement |
| 11 | Summary & Apply | n/a | n/a | ✅ | All 19 settings cleanly summarized; Apply transitions to workspace |

### OAuth flow detail (Step 4 → Step 5 transition)

1. ✅ `request_access` flow opens the system browser at `id.ctrader.com/my/settings/openapi/grantingaccess?client_id=…&redirect_uri=http%3A%2F%2F127.0.0.1%3A43001%2Fcallback&scope=trading&state=…`.
2. ✅ Existing browser session at id.ctrader.com auto-skipped the password prompt — operator was already logged in.
3. ✅ Consent page lists 7 accounts (2 FTMO Live, 5 Spotware Demo). Target **FTMO Platform · Live · 17111418** ($10K USD) is present.
4. ✅ "Allow access" click via Windows-MCP — browser redirected to `http://127.0.0.1:43001/callback?code=…&state=…`.
5. ✅ Wizard's loopback listener responded with `"cTrader login received. You can close this tab."`.
6. ✅ Wizard banner updated to `"Token bundle received — held in memory as SecretString until Apply."`.
7. ❌ **Account-discovery leg failed**: `"OAuth error: failed to parse cTrader JSON envelope"`.
   - Origin: `crates/forex-app/src/app_services/ctrader_messages.rs:902` —
     `parse_open_api_envelope` couldn't deserialise the response into
     `CTraderOpenApiJsonMessage`.
   - Likely cause: the response shape from `connect.spotware.com`'s JSON-over-WS endpoint shifted, or the request envelope we send doesn't match what the live endpoint expects right now.
   - Status: **deferred to v0.4.13** (see task #105). Phase X3 (live trades) is blocked on this — the wizard never gets the account list, so there's no `ctidTraderAccountId` to authenticate against.

---

## Phase X2 — cTrader sanity check (no password)

Partial pass:

- ✅ Step 4 banner *"the bot's app credentials are baked in to the binary, so there is nothing else for you to type here"* now reflects reality (v0.4.11 fix).
- ✅ Loopback port list visible: `[43001, 7777, 7878, 8989]` (v0.4.12 fix).
- ✅ Browser opens cleanly, no password input ever required by the wizard.
- ❌ "Discover Accounts" path → JSON envelope parse error (see X1 detail).
- ⏸ Settings → Brokers → cTrader pre-fill check and "Save Credentials to Disk" round-trip — **not exercised this session**.
- ⏸ DxTrade panel form-input render — **not exercised this session**.

---

## Phase X3 — Live test trades

**Blocked.** The discovery-stage JSON parse error means the wizard never hands the workspace a connected cTrader session with a selected `ctidTraderAccountId`. The Order Ticket panel correctly shows `OFFLINE` / `Not connected` in the post-Apply workspace.

---

## Phase X4 — Discovery + Training start

**Not exercised this session.** The workspace renders with `Discovery` and `Training` tabs present in the AI Engine group, but a real run wasn't attempted because Phase X3 burned the time budget on the build → ship cycle.

---

## Phase X5 — Open-ended 15-tab smoke

Post-Apply workspace observation:

- ✅ All 14 left-sidebar entries render:
  - TRADING — Dashboard, Chart, Markets, Order Ticket, News, Trade Watch
  - AI ENGINE — Discovery, Training, Intelligence, **AI Helper** (v0.4.10 addition, present in sidebar)
  - SYSTEM — Runtime, Broker Setup, Data Bootstrap, Hardware, Risk Settings, Settings
- ✅ Top-bar status reads `EURUSD / M1 / cTrader / $0.00`, green dot `cTrader Ready`, `12 cores · GPU on`, `AUTO OFF`, `DEMO`, `HALT` button (red).
- ✅ Bottom-bar reads `Offline | No engines running | v0.4.12 | cTrader Ready | 07:39:48 UTC`.
- ✅ Right-hand dock initially shows **Broker Setup** with `Data Source: cTrader`, `Adapter: cTrader`, `Readiness: OAuth app credentials ready for Demo environment`, `Integration: Remote Open API`, and the `Runtime Source` / `Active Broker Adapter` toggles between cTrader and DXtrade. This is the v0.4.11 / v0.4.12 fix surfacing in the post-wizard workspace.
- ⏸ Per-tab content rendering for the remaining 12 tabs **not deep-tested this session**.

---

## New bugs found (chronological)

| # | Severity | Description | Status |
|---|---|---|---|
| 1 | High | v0.4.9 installer was 20.93 MB and missing all DLLs / LICENSE / README / Gemma-fetch script | ✅ Fixed in v0.4.10 |
| 2 | High | `EMBEDDED_CTRADER_CLIENT_ID = ""` shipped in every release v0.4.7 → v0.4.10 (wizard banner blocked OAuth entirely) | ✅ Fixed in v0.4.11 |
| 3 | High | Wizard advertised `127.0.0.1:7777/ctrader/callback` but cTrader app dashboard had `127.0.0.1:43001/callback` registered → consent click bounced with "Provided application does not contain provided URI" | ✅ Fixed in v0.4.12 |
| 4 | Medium | `parse_open_api_envelope` fails on the account-discovery response from the live cTrader Open API JSON-over-WS endpoint → account list never populates → workspace stays Offline | 🟡 Deferred to v0.4.13 |
| 5 | Low | Step 3 "Require Stop Loss on every order" default appears to be OFF for FTMO_STANDARD preset (FTMO requires SL) — needs default reverification on a clean run | 🟡 Deferred to v0.4.13 |

---

## Known gaps (not regressions — explicit operator decisions)

- Phase X3 live trades — needs Bug #4 fixed first.
- Phase X4 discovery + training run — not exercised this session.
- Phase X5 per-tab smoke for the 12 non-default tabs — not exercised this session.
- Gemma GGUF model is not bundled in the installer; the AI Helper banner correctly surfaces the fetch script.
- Wizard "Require Stop Loss" default state on Step 3 may have flipped — re-verify on a clean wizard_state.json.

---

## Reproduction notes

- Wizard launched with `--wizard` flag from `target/release/forex-app.exe`.
- `wizard_state.json` at `%LocalAppData%\forex-ai\wizard_state.json` deleted between each rebuild to force a clean run.
- Browser at `id.ctrader.com` was already logged in throughout the session — no password input ever required from the operator.
- Workspace `.local/forex-ai/broker_credentials.toml` carries the real Open API app credentials (gitignored).

---

*Generated 2026-05-19 09:40 local time by the v0.4.12 functional walkthrough.*
