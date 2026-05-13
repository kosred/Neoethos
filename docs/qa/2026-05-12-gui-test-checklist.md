# GUI Visual + cTrader Live Test Checklist — 2026-05-12

**Build under test:** master @ `4a91caf0` (TradingView palette + 4.0% risk slider + status bar).
**Binary:** `target/release/forex-app.exe` (103.7 MB, dated May 12 00:38).
**GUI PID:** `17296` (started 17:31:57 from this session).
**Other live processes:** discovery (`24180`) + training (`14744`) headless + an older GUI (`14884` from 02:38 AM) — they share the data dir but not the OAuth port.

**Log files for this run:**
- `logs/gui-173140.log` (stdout, 3376 bytes after init, idle thereafter — expected)
- `logs/gui-173140.err.log` (empty)

**Pre-flight verified autonomously:**
- ✅ Release binary present, version `0.2.0`.
- ✅ GUI launched, OpenGL 3.3 Core context on AMD Radeon initialised cleanly.
- ✅ Broker credentials loaded from `%APPDATA%\forex-ai\broker_credentials.toml` (the file I synchronised earlier — client_id 56 chars prefix `26884_`, client_secret 50 chars, environment Demo).
- ✅ 0 WARN / 0 ERROR / 0 panic across `discovery`, `training`, `gui` logs in the first ~4 minutes.
- ✅ Memory stable: GUI held 124.7 MB across two samples 40 s apart with no growth.

> **Note on the layout description.** The original brief asked you to verify a *top-bar Navigate dropdown* + *Engine dropdown*. The current master moves those to a **left sidebar** (grouped Trading / AI Engine / System) and a **bottom status bar** (no engine buttons there — actions live inside each tab). The checklist below reflects what's actually in `crates/forex-app/src/main.rs` + `crates/forex-app/src/workspace/tabs.rs`. If you wanted the old top-bar dropdowns back, that's a separate UI change, not a regression.

---

## 1 — Top bar (single horizontal strip)

- [ ] Brand `Forex AI` on the left with a `PRO` accent badge next to it.
- [ ] Ribbon items in order: **SYMBOL** / **TIMEFRAME** / **SOURCE** / **EQUITY**.
- [ ] EQUITY value coloured green when > 0, muted otherwise.
- [ ] Right side: ⚙ settings icon → **AUTO ON/OFF** toggle pill (green when on, grey when off) → hardware indicator `N cores • GPU on/off` → vertical separator → status text + status dot.

*(Top bar **does not** contain a Navigate dropdown or an Engine dropdown — nav lives in the left sidebar, engine controls live inside the Discovery / Training tabs.)*

## 2 — Left sidebar (primary navigation)

- [ ] Resizable panel on the left, default ~220 px wide.
- [ ] Three section headers in order: **Trading**, **AI Engine**, **System**.
- [ ] Under *Trading*: Dashboard / Chart / Markets / Order Ticket / News / Trade Watch.
- [ ] Under *AI Engine*: Discovery / Training / Intelligence.
- [ ] Under *System*: Runtime / Broker Setup / Data Bootstrap / Hardware / Risk Settings / Settings.
- [ ] Clicking a tab highlights it with a left accent stripe and brings it to focus in the central panel.
- [ ] Each tab description appears as hover tooltip (e.g. "Genetic strategy search → portfolio" on Discovery).

## 3 — Default dock layout (central area)

- [ ] **Chart** tab is the active center tab.
- [ ] Left column (~18% width) holds **Markets** + **Dashboard** tabs.
- [ ] Right column top (~26% width) holds **Order Ticket** + **News** tabs.
- [ ] Right column bottom holds **Broker Setup** / **Runtime** / **Intelligence** / **Data Bootstrap** / **Hardware** / **Risk Settings** / **Settings** tabs (one of them visible at a time).
- [ ] Center bottom (~22% of center height) holds **Trade Watch** / **Discovery** / **Training**.

## 4 — Bottom status bar (slim strip)

- [ ] On the left: broker connection dot (grey + "Offline" before OAuth, green + "Connected" after).
- [ ] Active-engine tally: "No engines running" in muted text, OR `Running: Discovery, Training` in accent colour. (Note: this GUI process does NOT run the headless discovery/training — those are separate processes. The status bar reflects only what this GUI started.)
- [ ] App version `v0.2.0` displayed.
- [ ] On the right: UTC clock `HH:MM:SS UTC` + compact status text (truncated at 30 chars with `...`).

## 5 — Broker Setup tab (Settings → Brokers → cTrader)

- [ ] Summary cards row: Data Source / Adapter / Readiness / Integration / Targets (+ cTrader Auth once loaded).
- [ ] **Runtime Source** toggle: cTrader / Local.
- [ ] **Active Broker Adapter** row showing the supported adapters as selectable labels.
- [ ] **Adapter Configuration** form for cTrader:
  - [ ] Client ID pre-filled (length 56, starts with `26884_`).
  - [ ] Client Secret pre-filled (length 50 — not the old `secret-abc` placeholder).
  - [ ] Redirect URI = `http://127.0.0.1:43001/callback`.
  - [ ] Environment radio = **Demo**.
- [ ] First button row: **Start cTrader Login (Automatic)** / Start cTrader Auth / Prepare Token Request / **Save Credentials to Disk** (with hover tooltip about persistence).
- [ ] Manual Code text field below + **Accept Code** button (fallback path).
- [ ] Second button row: **Discover Accounts** / Restore Saved Session / Clear Saved Session.

## 6 — Discovery tab (manual sanity check, optional)

- [ ] Form fields: base TF / higher TFs (comma list) / population / generations / max indicators / target candidates / portfolio size / correlation threshold / min trades per day.
- [ ] **Start** button at the bottom of the form.
- [ ] Status card shows Queued / Running / Succeeded / Failed / Cancelled with a coloured dot.
- [ ] If the user clicks Start, a snapshot row appears with state `Queued` then `Running` and progress text.

*(Note: do **not** start a manual discovery in this GUI right now — the headless discovery process already has the data lock. Triggering a parallel run may fail with a file-lock error. Just verify the form renders.)*

## 7 — Training tab (visual only)

- [ ] Form with symbol selector + base TF + Start button.
- [ ] If the headless training is currently running, this tab will show **no** Running state for this GUI process (that's an out-of-process job).
- [ ] Snapshot card layout matches Discovery's.

## 8 — Theme & rendering

- [ ] TradingView-style dark palette (charcoal background, white-ish primary text, muted secondary, accent blue/green).
- [ ] Fonts: title, body, caption sizes visibly distinct.
- [ ] No text clipping / overflow in the top bar at 1200 × 800.
- [ ] Sidebar accent stripe paints when a tab is selected.
- [ ] Risk Settings tab: drawdown slider top end is **4.0%** (per commit `4a91caf0`).

## 9 — cTrader OAuth live test (when you are at the computer)

Run when you are ready to type your password into the browser:

| # | Action | Expected |
|---|--------|----------|
| 1 | In the **Broker Setup** tab, click **Start cTrader Login (Automatic)**. | Default browser opens to `connect.spotware.com/oauth/...?client_id=26884_...&redirect_uri=http://127.0.0.1:43001/callback&...`. App status text shows "OAuth: awaiting browser…" or similar. |
| 2 | In the browser: log in as `konstantinoskokkinos1982@gmail.com` + your password → click **Authorize**. | Browser redirects to `http://127.0.0.1:43001/callback?code=...`. App's local listener serves a small success page. |
| 3 | Return to the app. | Status flips to "OAuth: code received" → "token acquired". cTrader Auth summary card shows non-empty status_line. |
| 4 | Click **Discover Accounts**. | Targets table populates with all cTID-linked accounts — **expected: FTMO Demo `17111418`** (the new account). The old `17102270` may also appear. |
| 5 | Tick `enabled_for_execution` on `17111418` only. | Readiness card flips to ready. |
| 6 | Click **Save Credentials to Disk**. | Status text: "saved broker credentials to disk". |
| 7 | Sidebar → Trading → **Markets**. | Watchlist shows EURUSD + configured pairs. Bid/Ask cells update every few seconds. |
| 8 | Sidebar → Trading → **Order Ticket**. | Symbol = EURUSD, Volume = 0.01, Side = Buy, type = Market. Click **Execute**. |
| 9 | Confirm in trade journal + Trade Watch. | New `EURUSD 0.01 long` position. Equity in top ribbon ticks. |
| 10 | Close the position. | Trade journal records close. Equity reflects realised P&L. |

**Abort conditions:**
- Step 1 browser does not open → check default browser is set in Windows.
- Step 2 redirect goes to a different port → the redirect_uri in the embedded credentials does not match what the cTrader app portal has registered. Will need a credential rotation.
- Step 4 only shows account `17102270` (no `17111418`) → the new FTMO demo is not yet linked to your cTID. Add it via the FTMO/cTrader account-linking flow first.

---

## Known sharp edges

- **Two GUI windows are running simultaneously** (PID 14884 from this morning, PID 17296 from now). They both bind a local OAuth listener on port `43001` — only one will succeed. **Use PID 17296** (the new one) for the test, or close the old one first via its window's X button.
- **Headless discovery + training are running in background** (PIDs 24180 + 14744, ~2.5 GB RAM each on AUDUSD M1). They do not touch the OAuth port, but they DO hold a read lock on parquet data; do not start a manual discovery for AUDUSD from the GUI while they're alive.
- **Status text in the bottom bar can be stale** if no widget triggers a repaint — egui requests a repaint each tick only while a job is running. If the bar looks frozen, click anywhere in the window.
