# cTrader Live Test Checklist — 2026-05-12

**Build under test:** master @ `4a91caf0` (forex-app TradingView palette + status bar) on top of all Phase-8c, persistence, embedded-credentials, GPU-migration work.

**Demo account:** FTMO Demo `17111418` (replaces old `17102270`).
**cTrader ID:** `konstantinoskokkinos1982@gmail.com`.

**Pre-flight (already verified autonomously):**
- ✅ Embedded cTrader credentials baked into binary from `.local/forex-ai/broker_credentials.toml` (build.rs).
- ✅ `%APPDATA%\forex-ai\broker_credentials.toml` synchronized with the real Client ID/Secret on 2026-05-12 (stale `client-123` placeholder previously blocked OAuth — fixed; backup saved as `broker_credentials.toml.bak-2026-05-12`).
- ✅ Release binary at `target/release/forex-app.exe`.

---

## Walkthrough

| # | Action | Expected result | Pass/Fail |
|---|--------|-----------------|-----------|
| 1 | Launch `target/release/forex-app.exe` (no flags) | GUI opens at 1200×800 with "Forex AI - Pure Rust Terminal" title. Top bar shows brand + PRO badge + SYMBOL/TIMEFRAME/SOURCE/EQUITY ribbon. Bottom status bar shows `● Offline   No engines running   v0.2.0`. Status pill on right shows current message (likely "loading"/"ready"). | |
| 2 | Sidebar → System → Brokers | Center panel renders "Broker Setup" with summary cards: Data Source / Adapter / Readiness / Integration / Targets. Below: Runtime Source toggle (cTrader / Local), Active Broker Adapter row, Adapter Configuration form. | |
| 3 | In the cTrader form, verify Client ID is `26884_ZJBPTG1PzFd0Pw48UvjTmjK8SxspnCDxq4POYfZ5ZAYXzpoUqO` and Client Secret is non-empty (50 chars). | If empty or `client-123`, the persisted file is still wrong — re-run sync step in pre-flight. | |
| 4 | Environment = **Demo** is selected (radio). | Required: FTMO demo account `17111418` only lives on the Demo cTrader environment. | |
| 5 | Click **Start cTrader Login (Automatic)**. | Default browser opens to `https://connect.spotware.com/oauth/...` with the app's client_id. Status bar in the app turns to "OAuth: awaiting browser…". | |
| 6 | In browser: log in with `konstantinoskokkinos1982@gmail.com` + password → click **Authorize**. | Browser redirects to `http://127.0.0.1:43001/callback?code=...&state=...`. A success page is shown by the app's local listener. App status flips to "OAuth: received code, exchanging…". | |
| 7 | Wait for token exchange. | App status becomes "OAuth: token acquired" or similar. cTrader Auth card shows `Connected` / non-empty status_line. | |
| 8 | Click **Discover Accounts**. | "Discover Accounts" populates the Targets list with all accounts under your cTID. **Expected to include FTMO Demo `17111418`** (and possibly the old `17102270` if it is still attached). | |
| 9 | In the Targets table, tick `17111418` as `enabled_for_execution`. Optionally untick `17102270`. | The target row shows enabled state; readiness card flips to a ready state if at least one target is enabled. | |
| 10 | Click **Save Credentials to Disk**. | Status bar: "saved broker credentials to disk". Verify on next launch the form auto-loads. | |
| 11 | Sidebar → Trading → Watchlist. | Watchlist shows EURUSD + other configured symbols. Live spot prices update every few seconds (bid/ask flash). If prices stay frozen for > 30 s, cTrader stream is broken. | |
| 12 | Sidebar → Trading → Execution. | Execution panel renders. Symbol selector defaults to EURUSD. Volume input visible. Side buttons (Buy/Sell) visible. | |
| 13 | **TEST ORDER** — Set symbol = EURUSD, volume = 0.01, side = Buy, **market order**. Click Execute. | Live trade journal records the new order. cTrader confirms execution within ~1 s. Watchlist shows an active position. Equity in top ribbon ticks accordingly. | |
| 14 | Refresh positions view → confirm `EURUSD 0.01 long` is open. | Position row with entry price, current P&L. | |
| 15 | Close the position (Close button or opposite market order). | Position disappears from active list. Journal records close. P&L realized. | |
| 16 | Top ribbon EQUITY updated to reflect realized P&L. | EQUITY value shifts (likely a few cents either way). | |
| 17 | File → Quit (or window close). | App writes shutdown log entry `app_shutdown / FINISHED`. Process exits cleanly with exit code 0. | |

---

## Pass criteria

All 17 steps PASS. If step 5 OAuth browser does not open or step 7 token exchange fails, the FTMO demo cannot be reached — abort and inspect `logs/qa-2026-05-12/build.log` + app log under `logs/`.

## Known sharp edges

- **APPDATA file authority** — Whatever is in `%APPDATA%\forex-ai\broker_credentials.toml` overrides both `.local/` and embedded constants. If somebody (or a test) writes garbage there, step 3 will show garbage. Backup file: `broker_credentials.toml.bak-2026-05-12`.
- **Old FTMO account 17102270** — May still be authorized on the cTID. Step 9 lets you untick it. Trading panel default execution adapter respects the `enabled_for_execution` flag.
- **Headless processes running** — If you have `forex-app.exe --headless` running discovery/training in another shell (Phase 2), they share the data dir but NOT the broker connection. They will not interfere with the GUI test.
