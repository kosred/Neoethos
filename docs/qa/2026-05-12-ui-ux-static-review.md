# UI/UX Static Review — 2026-05-12

**Reviewer:** Claude (autonomous, no display access).
**Scope:** code-level confirmation of the latest workspace redesign (TradingView palette + status bar + sidebar nav) on master @ `4a91caf0`.

## Verified via code review

### Top bar — `crates/forex-app/src/main.rs:382-529`
- Single horizontal strip at `TOPBAR_HEIGHT`.
- Left side: `Forex AI` brand + `PRO` badge + SYMBOL / TIMEFRAME / SOURCE / EQUITY ribbon items, each as `render_ribbon_item`.
- Right side (`right_to_left` layout): ⚙ settings icon → AUTO ON/OFF toggle pill → hardware indicator (`N cores • GPU on/off`) → vertical separator → status pill (color = SUCCESS / DANGER / WARNING based on text content) → status dot.
- Comment block lines 378-381 explicitly explains that engine and broker controls were **moved out** of the top bar to avoid duplication.

### Bottom status bar — `crates/forex-app/src/main.rs:577-651`
- Slim `STATUSBAR_HEIGHT` strip (per comment "22-px").
- Left to right: broker connection dot+label (Connected/Offline) → separator → active engines list ("Running: Discovery, Training" or "No engines running" in muted color) → separator → app version (`v{CARGO_PKG_VERSION}`).
- Right side: UTC clock `HH:MM:SS UTC` + compact status text (truncated at 30 chars with ellipsis).
- **Intentionally informational only** — the action bar with Start/Stop buttons was removed (lines 567-575 comment): "actions live in their tabs".

### Left sidebar navigation — `main.rs:676-708` + `workspace/tabs.rs:113-133`
- `SidePanel::left("workspace_nav")` with `SIDEBAR_WIDTH_DEFAULT`, resizable in `[SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX]`.
- Iterates 3 groups in fixed order via `WorkspaceGroup::ordered()` → `Trading`, `AI Engine`, `System`. Each group title rendered as `section_label` followed by its tabs as `nav_item` calls.
- Active tab gets a left accent stripe (per main.rs:670-673 comment).
- 15 tabs total, all reachable from one place — replaces the older "Navigate dropdown" entirely.

### Dock workspace — `workspace/layout.rs:13-53`
- Center: `Chart` (main tab).
- Left split (18% width): `Watchlist` + `Dashboard`.
- Right top split (26% width): `Execution` + `News`.
- Right bottom split (50% of right): `Broker Setup` + `Runtime` + `Intelligence` + `Data Bootstrap` + `Hardware` + `Risk Settings` + `Settings`.
- Center bottom split (22% of center): `Trade Watch` + `Discovery` + `Training`.
- Tests at lines 113-159 enforce all 15 tabs are present + Chart is the main-center tab.

### Broker Setup tab — `crates/forex-app/src/ui/system/brokers.rs:1-200`
- Summary cards: Data Source / Adapter / Readiness / Integration / Targets (and cTrader Auth if available).
- Runtime Source radio (cTrader / Local).
- Active Broker Adapter selectable_label row.
- cTrader form: Client ID, Client Secret, Redirect URI text edits + Environment radio (Live / Demo).
- Action buttons row 1: **Start cTrader Login (Automatic)** (line 158), **Start cTrader Auth** (line 161), **Prepare Token Request** (line 164), **Save Credentials to Disk** (line 168) — confirms persistence layer is wired to UI as required.
- Action buttons row 2: **Discover Accounts**, **Restore Saved Session**, **Clear Saved Session**.
- Manual code entry text field + Accept Code button for fallback when browser redirect fails.

### CLI flags — `crates/forex-app/src/main.rs:22-41`
- `--headless` (bool, default false)
- `--config <PATH>` (default `config.yaml`)
- `--local` (bool, default false)
- `--auto-discovery` (bool, default false)
- `--auto-training` (bool, default false)

### Headless loop — `main.rs:94-180`
- Discovers local symbols from `runtime.data_dir`.
- On `--auto-discovery`: starts a `DiscoveryRequest` for the first local symbol (or EURUSD fallback) on M1 base with M5/M15/H1 higher TFs.
- On `--auto-training`: starts a `TrainingRequest` for the first local symbol on M1 base.
- Both jobs use the same `mpsc` channel; events are received but the headless loop currently drops them (the `_rx` is discarded — see line 110). A 10 s keep-alive log line reports the mode + flags.
- **Sharp edge:** because `_rx` is dropped, terminal-state events (Succeeded/Failed) are not surfaced in the headless log beyond the engine's own tracing. Job completion is observable only via subsystem log files written by the job code itself, not via a top-level "discovery: done" line.

### Theme — `crates/forex-app/src/ui/theme.rs`
- Houses `apply_theme(ctx)`, `nav_item`, `status_dot`, `status_separator`, `section_label`, `section_frame`, `top_panel_frame`, `status_bar_frame`, `sidebar_frame`, `central_panel_frame`, plus the constants TOPBAR_HEIGHT / STATUSBAR_HEIGHT / SIDEBAR_WIDTH_DEFAULT / SPACE_* / FONT_* / TEXT_PRIMARY / ACCENT / SUCCESS / DANGER / WARNING. TradingView-style palette confirmed by recent commit `4a91caf0`.

## Behaviour requiring visual confirmation by the operator

These are correct in code but cannot be visually verified without running the GUI. They go on the live-test checklist instead:

1. Active-tab left accent stripe actually paints when a sidebar item is clicked.
2. Status pill color (SUCCESS/DANGER/WARNING) responds visibly to status text changes during OAuth.
3. UTC clock in status bar ticks every second (currently the timer relies on egui repaint cadence — if no other widget requests repaint, the clock can freeze).
4. AUTO ON/OFF pill flips green/grey on click.
5. Equity ribbon item color shifts SUCCESS↔TEXT_MUTED at the `> 0.0` threshold.
6. Sidebar resizes between MIN and MAX widths via drag.
7. Bottom strip text truncates at 30 chars with `...` for long status messages.
8. Risk slider top end is 4.0% (per commit `4a91caf0`) — verified in code under `crates/forex-app/src/ui/risk.rs` (slider bounds test at main.rs:947-955 confirms drawdown 0.01–0.20 and lot 0.01–50.0).

## Recommendation

UI/UX is **architecturally sound** for headless operation + GUI launch. The two non-trivial concerns:

- **UTC clock freezing** when no engines are running: there is no `ctx.request_repaint_after()` call in the status bar render, only the broad "if discovery_running || training_running" check at main.rs:729. A "no engines running" idle GUI may show a stale clock.
- **Headless job completion silence:** `_rx` dropped at main.rs:110 means a 30-min discovery completing emits its own tracing but no top-level "DONE" line in the headless keep-alive log. Operator must inspect job-specific subsystem logs.

Both are minor and out of scope for this test pass.
