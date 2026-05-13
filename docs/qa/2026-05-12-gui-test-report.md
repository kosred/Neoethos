# GUI Test Report — 2026-05-12

**Build under test:** master @ `4a91caf0` ("forex-app: TradingView palette + 4.0% risk slider + status bar").
**Binary:** `target/release/forex-app.exe` (98.9 MB on the Windows side, dated 12/5 02:38), version `0.2.0`.
**Test session:** 17:37:59 → 17:39:55 (60 s active monitoring), PID `7420`.
**Logs:** [logs/gui.log](logs/gui.log) (stdout) + [logs/gui.err.log](logs/gui.err.log) (stderr).
**Other running forex-app processes (left untouched per instructions):**
- PID 14884 — older GUI from 02:38 AM (window: "Forex AI - Pure Rust Terminal").
- PID 17296 — earlier GUI started at 17:31:57 (this session, before the brief was narrowed).
- PID 24180 — `--headless --auto-discovery` for AUDUSD, started 17:31:40.
- PID 14744 — `--headless --auto-training` for AUDUSD, started 17:31:51.

---

## A) GUI startup health

### Process snapshot

| Checkpoint | Time | Alive | RAM (MB) | CPU total (s) | Threads | Log lines | Log bytes | WARN/ERROR/panic |
|-----------:|------|:-----:|---------:|--------------:|--------:|----------:|----------:|-----------------:|
| t=0   | 17:37:59 | ✓ | 122.2 | — | — | — | — | — |
| +10 s | 17:38:09 | ✓ | 122.2 | 0.5 | — | 21 | 3 376 | 0 |
| +15 s | 17:38:42 | ✓ | 116.6 | 0.5 | 25 | 21 | 3 379 | 0 |
| +30 s | 17:39:10 | ✓ | 116.5 | 0.5 | 20 | 21 | 3 379 | 0 |
| +45 s | 17:39:32 | ✓ | 116.5 | 0.5 | 20 | 21 | 3 379 | 0 |
| +60 s | 17:39:55 | ✓ | 116.5 | 0.5 | 20 | 21 | 3 379 | 0 |

**Interpretation:**
- **Memory** settles from 122 MB at startup to 116.5 MB once init buffers free and stays absolutely flat. Zero leak in 60 s of idle.
- **CPU** spent 0.5 s in 116 s of wall time — egui is event-driven; with no input it spends ~0% CPU.
- **Threads** drop from 25 to 20 (init threads exit) and remain stable.
- **Logs stop growing** after the init sequence, which is expected because the only `info!()` channel events are job state changes and broker connection updates — none occurred. The DEBUG-level egui/glow output is one-shot at startup.

### Full startup transcript (21 lines, INFO + DEBUG)

| # | Level | Target | Message |
|---|-------|--------|---------|
| 1 | INFO  | `forex_core::logging`            | Logging initialized (verbose=true) |
| 2 | INFO  | `forex_core::logging`            | Canonical log file: `logs\forex-ai.log` |
| 3 | INFO  | `forex_app`                      | Starting Forex AI in GUI Mode... |
| 4 | DEBUG | `eframe`                         | Using the glow renderer |
| 5 | DEBUG | `eframe::native::glow_integration` | Event::Resumed |
| 6 | DEBUG | `eframe::native::glow_integration` | trying to create glutin Display with config { RGB888, A8, no depth/stencil/MSAA } |
| 7 | DEBUG | `eframe::native::glow_integration` | using the first config from config picker closure — `Wgl(Config { hdc=…, pixel_format_index=32 })` |
| 8 | DEBUG | `eframe::native::glow_integration` | successfully created GL Display: WGL + features `CONTEXT_NO_ERROR | FLOAT_PIXEL_FORMAT | SWAP_CONTROL | CONTEXT_RELEASE_BEHAVIOR | CREATE_ES_CONTEXT | MULTISAMPLING_PIXEL_FORMATS | SRGB_FRAMEBUFFERS` |
| 9 | DEBUG | `eframe::native::glow_integration` | creating gl context using raw window handle `Win32(Win32WindowHandle { hwnd=3801960, hinstance=… })` |
| 10 | DEBUG | `eframe::native::glow_integration` | Initializing `egui_winit` for viewport "FFFF" |
| 11 | DEBUG | `eframe::native::glow_integration` | Creating a `gl_surface` for viewport "FFFF" |
| 12 | DEBUG | `egui_glow::painter`             | OpenGL version: `3.3.0 Core Profile Context 22.20.27.09.230330` / renderer `AMD Radeon (TM) Graphics` / vendor `ATI Technologies Inc.` |
| 13 | DEBUG | `egui_glow::shader_version`      | Shader version: `Gl140 ("4.60")` |
| 14 | DEBUG | `egui_glow::painter`             | Shader header: `#version 140\n` |
| 15 | DEBUG | `egui_glow::painter`             | SRGB texture support: true |
| 16 | DEBUG | `egui_glow::painter`             | SRGB framebuffer support: true |
| 17 | DEBUG | `egui_glow::vao`                 | GL version: `3.3.0 Core Profile Context 22.20.27.09.230330` |
| 18 | INFO  | `forex_app::app_services::broker_persistence` | **loaded broker credentials from disk** `path=C:\Users\konst\AppData\Roaming\forex-ai\broker_credentials.toml` |
| 19–21 | _(trailing blank/continuation lines from DEBUG record formatting)_ |

**Tracing levels observed:** `INFO` and `DEBUG`. No `WARN`, `ERROR`, `TRACE`, or `FATAL` messages. `stderr` is **completely empty** (0 bytes) for the entire 60 s session.

### Findings — A

| Finding | Severity | Detail |
|---------|----------|--------|
| `CONTEXT_NO_ERROR` substring in OpenGL features line | Info | Looks like an "error" substring to naive grep but is the OpenGL `WGL_CONTEXT_OPENGL_NO_ERROR_ARB` capability flag. Not a real error. |
| GPU = AMD Radeon (integrated), driver `22.20.27.09.230330` | Info | OpenGL 3.3 Core. Sufficient for egui_glow. The compile-time `cubecl` and `tch` GPU stacks are training-side, not GUI-side. |
| Two GUI windows are now open simultaneously | Sharp edge | PID 14884 (this morning) + PID 17296 (earlier this session) + PID 7420 (this test). All three bind `OAuth listener:43001` lazily on user click — only the first to click wins. Recommend closing 14884 and 17296 before running the OAuth step. |
| `loaded broker credentials from disk` line on first INFO record | Pass | Confirms `%APPDATA%\forex-ai\broker_credentials.toml` is being loaded — the file I synchronised earlier with the real `26884_...` Client ID. Embedded fallback is not in play (file exists and has non-empty credentials). |

---

## B) cTrader integration — visual checklist (for the user at the computer)

Perform these in the **PID 7420** window. Order matters: do steps 1–3 before clicking any OAuth button so the form pre-fill is verifiable.

- [ ] Sidebar → **System** group → **Broker Setup** tab.
- [ ] Settings → Brokers → cTrader form shows:
  - [ ] **Client ID** pre-filled (length 56 chars, starts with `26884_`).
  - [ ] **Client Secret** pre-filled (50 chars; should *not* be `secret-abc`).
  - [ ] **Redirect URI** = `http://127.0.0.1:43001/callback`.
  - [ ] **Environment** radio = **Demo**.
- [ ] Click **«Start cTrader Login (Automatic)»**.
  - [ ] Default browser opens to a URL on `connect.spotware.com/oauth/...` (or `id.ctrader.com` depending on routing) with the embedded `client_id` and the redirect_uri above as query params.
  - [ ] The app's status text turns to "OAuth: awaiting browser…" or similar.
- [ ] Browser: log in as `konstantinoskokkinos1982@gmail.com` + your password.
- [ ] Browser: click **Authorize**.
  - [ ] Browser redirects to `http://127.0.0.1:43001/callback?code=...&state=...` and shows the app's success page.
  - [ ] App status flips to "Authentication successful" / "OAuth: token acquired" (depending on the exact substring in `ctrader_auth::CTraderAuthSnapshot::status_line`).
- [ ] Click **«Discover Accounts»**.
  - [ ] Targets list populates with all cTID-linked accounts.
  - [ ] **Expected presence:** FTMO Demo `17111418`. Possibly also `17102270` (old demo).
- [ ] Tick `enabled_for_execution` on `17111418`. (Untick `17102270` if shown.)
- [ ] Click **«Save Credentials to Disk»**.
  - [ ] Status text: "saved broker credentials to disk".
- [ ] cTrader Auth summary card now shows a connected `status_line`.
  - [ ] (Visual: top-bar status pill turns green; bottom status-bar shows green dot + "Connected".)
- [ ] Sidebar → Trading → **Markets** tab.
  - [ ] Watchlist rows render with live Bid/Ask cells.
  - [ ] Cells visibly tick/refresh every few seconds.
- [ ] Sidebar → Trading → **Order Ticket**.
  - [ ] Symbol = EURUSD, Volume = 0.01, Side = Buy, type = Market.
  - [ ] Click **Execute** → confirmation toast / status line.
  - [ ] Trade journal (Trade Watch tab) shows the new EURUSD 0.01 long entry with execution time + price.
  - [ ] Top-ribbon **EQUITY** ticks.
- [ ] Close the position from the journal or via an opposite market order.
  - [ ] Position disappears.
  - [ ] Realised P&L reflected in EQUITY.
- [ ] Click **«Clear Saved Session»**.
  - [ ] cTrader Auth card flips back to disconnected.
  - [ ] Status line "Cleared saved cTrader session" or similar.

**Abort conditions:** if step 4 browser does not open → check Windows default browser; if step 6 only shows account `17102270` → the new FTMO demo is not yet attached to the cTID (add via FTMO portal first).

---

## C) Training pipeline through the GUI — visual checklist

> The current master places **Discovery** and **Training** as their own sidebar tabs (under the **AI Engine** group), with Start / status / metrics inside each tab. There is **no separate "Engine dropdown" in the top bar** — that was an earlier design that got replaced by the sidebar nav + bottom status bar (per main.rs:567-575 comment "actions live in their tabs"). The bottom status bar shows an aggregate **«Running: Discovery, Training»** tally but no buttons.

Perform these in the **PID 7420** window. **Important:** the headless processes (PID 24180 discovery + PID 14744 training, both on AUDUSD M1) already hold the data lock for AUDUSD. To avoid lock conflicts, do the GUI training tests on a **different symbol** (EURUSD, GBPUSD, etc.) — or stop the headless ones first with `Stop-Process -Id 24180,14744`.

- [ ] Bottom status bar: "Running: Discovery, Training" tally is **absent** in this GUI (because *this* GUI process did not start them). The headless PIDs run out-of-process and their state is invisible here.
- [ ] Sidebar → **AI Engine** group → **Discovery** tab.
  - [ ] Form fields render: base TF, higher TFs (comma list), population, generations, max indicators, target candidates, portfolio size, correlation threshold, min trades per day.
  - [ ] Select symbol = **EURUSD** (avoid AUDUSD).
  - [ ] Click **Start**.
  - [ ] Status card updates from `Idle` → `Queued` → `Running` with a coloured dot.
  - [ ] Progress text shows the current stage / generation as the GA iterates.
- [ ] Sidebar → **AI Engine** → **Training** tab.
  - [ ] Symbol selector + base TF + Start button render.
  - [ ] Click **Start** with symbol = EURUSD.
  - [ ] Status card flips to Running.
  - [ ] Live epoch / loss metrics appear in the snapshot region (training_orchestrator writes progress messages).
- [ ] State transitions: Running → Done with green dot on success, or → Failed with red dot + reason.
- [ ] Each tab has a **Stop / Cancel** button that flips status to `Cancelled` cleanly. Sidebar nav remains responsive throughout — no UI freeze.
- [ ] Top bar **AUTO ON/OFF** toggle pill: clicking flips colour green ↔ grey. The state is `state.auto_trade_enabled` — separate from job triggers; verifies the toggle is wired.

> **Logs panel:** there is **no dedicated "Logs panel" tab** in this build. Job stdout/tracing goes to the file at `logs/forex-ai.log` and to the snapshot card's `stage`/`message` field. The per-tab snapshot card *is* the live log surface.

### Findings — C

| Finding | Severity | Detail |
|---------|----------|--------|
| GUI cannot observe out-of-process headless jobs | Info | The headless discovery/training (PIDs 24180/14744) are invisible from this GUI's `discovery_job`/`training_job` state. This is by design: each process owns its own `mpsc::channel`. If you want to drive the pipeline from the GUI, stop the headless processes first. |
| Symbol conflict on AUDUSD | Sharp edge | Starting a GUI discovery/training on AUDUSD while the headless ones are alive may yield a file/data-lock error. Use EURUSD or stop the headless first. |
| No top-bar Engine dropdown | Design change | Brief asked for one; master has replaced it with per-tab controls + bottom status bar tally. Sidebar + tabs are the canonical surface. |

---

## Critical findings

**None.** Zero panics, zero stderr, zero `WARN`/`ERROR`/`FATAL` records in 60 s of monitoring. Process memory flat. Credentials loaded from `%APPDATA%`. Window opened successfully on AMD Radeon glow renderer. The launch is clean enough to proceed straight to the cTrader visual + OAuth steps.

## Pending — needs the user at the computer

1. Type the cTrader password into the browser at OAuth step (4 in section B).
2. Visual verification of all UI items in sections B and C — none of which can be observed from logs alone.
3. Decision: close the older GUI windows (PIDs 14884 + 17296) before clicking the OAuth button to avoid port-43001 contention with PID 7420.
4. Decision: stop or keep the headless discovery/training PIDs (24180 + 14744) before driving Discovery/Training from the GUI on the same symbol.
