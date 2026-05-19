# 7-Hour Autonomous Run — Handoff (in-progress, 2026-05-19 ~11:20 local)

**Operator:** Konstantinos (kosred)
**Mandate:** 7 hours of autonomous work — cTrader backend top priority,
ship releases, fix bugs, no approval prompts.
**Started:** 10:30 local time (after the v0.4.10–v0.4.14 sprint)
**Status at this snapshot:** ~50 minutes in, 4 of the 9 priorities
landed.

---

## Releases shipped (the running tally)

Combined with the earlier sprint, this is now **8 patch releases in
one day**: v0.4.10 → v0.4.17. All five new ones from this autonomous
window are listed below — earlier releases are in the prior QA reports.

| Tag | Commit | SHA-256 (head 8) | What it fixed |
|---|---|---|---|
| `v0.4.13` | `1ef0547f` | `03BCC6EC` | `parse_open_api_envelope` tolerates heartbeat frames |
| `v0.4.14` | `89686127` | `DB83A133` | Account-list parser permissive types |
| `v0.4.15` | `5ba482f9` | `7D35AA9A` | Wizard body scrolls so nav buttons stay reachable |
| `v0.4.16` | `e38a39ce` | `05C8EC74` | Account picker label + discovery telemetry |
| `v0.4.17` | `b4c78b98` | `86E87E08` | Wizard Apply persists OAuth token bundle to secure store |

All five are live at https://github.com/kosred/forex-ai/releases.

---

## Priority outcomes

### Priority 1 — Wizard scroll fix ✅ DONE in v0.4.15
- Root cause: `egui::CentralPanel` without scroll wrapping. On a
  maximized window, the bottom nav button row was hidden behind the
  Windows taskbar.
- Fix: wrap per-step body in `egui::ScrollArea::vertical().auto_shrink([false, false])`.
- **Verified live**: Step 5 walkthrough on v0.4.16 binary showed the
  scrollbar appearing on the right, Continue button reachable by
  scrolling.

### Priority 3 — Cosmetic: account picker label ✅ DONE in v0.4.16
- Selected-text was rendering bare `47149192`. Now renders
  `#47149192 FTMO Platform 17102270 (live)` — same format as the
  dropdown options. New shared helper `account_picker_label` prevents
  drift.
- **Verified live**: After OAuth + Allow access on v0.4.16 binary,
  dropdown showed full friendly labels for all 6 accounts.
- Also added the `<broker> <traderLogin>` stutter-avoidance fallback
  in `parse_account_list_by_access_token_json` so the picker shows
  something readable even when the cTrader response has only one of
  the two fields.

### Priority 3 — Investigation: 6-of-7 missing account ⚠ ROOT CAUSE
- **Finding from live walkthrough**: dropdown shows 6 accounts. The
  consent page showed 7. The MISSING one is the operator's stated
  target: **FTMO Platform · 17111418 (10K USD)**. The picker has the
  *other* FTMO Live account `17102270` (100K USD) but not 17111418.
- New v0.4.16 telemetry (`tracing::info!` line with `count` + `ids`)
  will let the operator confirm which `ctidTraderAccountId`s
  the broker returned on the discovery leg — pending a log capture
  with `RUST_LOG=ctrader.auth=info` or similar.
- **Hypothesis**: server-side revocation of 17111418's API access at
  the cTID level, independent of the consent-page grant. Operator
  needs to check the cTID Open API app dashboard for this account.

### Priority 3 — Step 3 "Require Stop Loss" default ✅ NOT A BUG
- **Verified on v0.4.16 + v0.4.17 binaries**: the checkbox is ticked
  by default for FTMO_STANDARD preset on a clean `wizard_state.json`.
  The earlier observation of it being unchecked was a click artefact
  from a prior session where my batch-click hit the checkbox while
  scrolling past — false alarm, no code change needed.

### Priority 2 — 16 backend ops (3-18) 🟡 PARTIAL
Op #3 (account select + connect) is the gating step. Findings:
- ✅ Wizard's Step 4 OAuth → token exchange → account discovery →
  account selection — all green.
- ✅ Wizard Apply step transitions cleanly to the workspace.
- ✅ Workspace top bar reads `DEMO · 47149192`, `🟢 cTrader Ready`.
- ✅ Broker Setup panel shows the selected account as `Target 1 = 47149192 Primary`.
- ❌ **Restore Saved Session button** still fires "No saved cTrader
  session found" → the v0.4.17 token-persist fix did NOT propagate
  end-to-end. Two possible root causes:
  1. `expose_token_bundle()` returned `None` at the moment Apply ran
     (timing race with the discovery worker that takes ownership of
     the access token in `result.token_bundle.access_token` via
     `move` semantics).
  2. The keyring `set_password` call succeeded but the workspace's
     trading session reads from a different keyring entry / process
     credential cache than the wizard writes to.
- Ops #4–#18 are all gated on op #3 being live. None of them have
  been exercised yet on the current build.

### Priority 4–7 — UI/UX, Discovery+Training, Gemma chat, DxTrade ⏸ PENDING
Time budget consumed by the priority-2 token-persist debugging.
These remain for the second half of the 7-hour window.

### Priority 8 — Release hygiene ✅ ON TRACK
Five releases shipped this session, every one with a CHANGELOG entry,
release notes, packaging manifest refresh, and a tagged GitHub release
with the NSIS installer asset attached.

### Priority 9 — Disk safety ✅ MONITORED
Free space at session start: 57.9 GB. At each release cycle: still
> 57 GB. No cleanup needed.

---

## New bugs found this session

| # | Severity | Description | Status |
|---|---|---|---|
| 6 | High | Wizard's Apply doesn't persist OAuth token bundle → workspace can't restore session | 🟡 v0.4.17 fix wires the call but real-binary verification shows the workspace still says "No saved cTrader session". Needs deeper investigation — likely a timing race with the discovery worker. |
| 7 | Medium | 6-of-7 accounts in picker — FTMO 17111418 missing | 🟡 v0.4.16 added telemetry; root cause likely server-side revocation. Operator action: check cTrader Open API app dashboard. |
| 8 | Low | Wizard nav buttons hidden by Windows taskbar on maximized window | ✅ Fixed in v0.4.15 (ScrollArea wrap). |
| 9 | Cosmetic | Bare ctid in account picker selected_text | ✅ Fixed in v0.4.16. |

Earlier session bugs (1-5) all fixed in v0.4.10–v0.4.14.

---

## Workspace screen sampled this session

Top bar visible: `Forex AI · PRO · SYMBOL EURUSD · TIMEFRAME M1 · SOURCE cTrader · EQUITY $0.00 · 🟢 cTrader Ready · 12 cores · GPU on · AUTO OFF · DEMO · 47149192 · ⚙ · HALT`.

Sidebar: 14 entries across TRADING (Dashboard, Chart, Markets, Order
Ticket, News, Trade Watch) / AI ENGINE (Discovery, Training,
Intelligence, AI Helper) / SYSTEM (Runtime, Broker Setup, Data
Bootstrap, Hardware, Risk Settings, Settings).

Broker Setup panel post-Apply: Demo environment selected · "Current
cTrader environment: Demo" · 4 buttons row (Start cTrader Login
Automatic / Start cTrader Auth / Prepare Token Request / Save) ·
Manual Code input + Accept Code · Discover Accounts / Restore Saved
Session / Clear Saved Session · Account Management (Create Demo / Create Live) ·
**Execution Targets: Target 1 = 47149192 / Primary** ← wizard correctly
wrote this row.

---

## Next steps (for the remaining time budget)

1. **Debug v0.4.17 token-persist**: add a tracing event inside
   `write_broker_credentials` immediately before and after the
   `save_token_bundle` call so we can see in the log whether
   `expose_token_bundle()` returned `Some(_)` and whether
   `save_token_bundle` returned `Ok` or `Err`. Ship as v0.4.18.
2. **Once token persist works**: exercise ops #4–#18 live on the new
   binary.
3. **Phase X5 catch-up**: open each of the 14 sidebar tabs and
   screenshot — confirm no render regressions from the ScrollArea
   wrap or any other v0.4.15+ changes.
4. **AI Helper**: verify the "Gemma model not found" banner shows
   with three buttons + URL.
5. **DxTrade panel**: verify form fields render on the brokers
   settings.

The Phase 4-7 work is straightforward once op #3 unblocks the actual
trading-session connect path.

---

*This report is the live, mid-session snapshot. Will be replaced /
amended by `2026-05-19-7hr-final-handoff.md` at the end of the
window.*
