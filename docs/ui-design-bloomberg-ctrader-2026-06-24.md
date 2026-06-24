# UI design research — Bloomberg Terminal + cTrader → NeoEthos

Researched (web) how the two reference platforms structure their screens, to set
NeoEthos's visual + structural direction. Grouping is the backbone.

## What Bloomberg Terminal does
- **Black UI, maximum density** — "every pixel is accountable", hierarchy earned by
  importance not decoration; whitespace sacrificed for data.
- **Color-coded semantics** — green = action, yellow = market sector, blue = panel
  switch, red = exit. Consistent meaning everywhere.
- **Command-driven** — function codes + command line; power users navigate by typing,
  not hunting menus. Conceals complexity behind a fast keyboard surface.
- **Tabbed/panel model** — was 4 fixed panels, now arbitrary tabs/windows, multi-monitor,
  fully customizable workspaces.

## What cTrader does (the layout to mirror)
- **Top:** app-switcher **Trade · Copy · Algo · Analyze** + account/workspace bar + chart toolbar.
- **Left:** **Market Watch** (symbols grouped into watchlists, live quotes, quick-trade).
- **Center:** **Charts** (multi-chart, detachable, indicators, timeframes).
- **Right:** **Active Symbol Panel (ASP)** — order entry + Depth of Market + sentiment +
  market details + stats + leverage.
- **Bottom:** **Trade Watch** — positions / orders / history / balance, tabbed.
- **Status bar:** session, server **latency**, time.
- **Workspaces** — save/restore panel layouts across devices.

## Trading-UX best practices (from the design literature)
- Density without chaos; **card-based** metrics with mini-graphs; **modular grid**;
  group similar indicators; legibility for fast decisions; consistency; progressive
  disclosure (core first, detail on demand).

---

## Direction for NeoEthos

### 1. The "Trade" workspace becomes a multi-panel cockpit (not one screen at a time)
Mirror cTrader: a single dense screen with docked panels instead of full-page nav for trading.
```
┌──────────────────────────────────────────────────────────────────────┐
│ TOP BAR: NeoEthos | workspace tabs | ⌘K search | acct (LIVE/DEMO·broker)│
│                                              | ●latency | conn pill     │
├───────────┬──────────────────────────────────────────┬────────────────┤
│ MARKET    │  CHART (candles + live bar + indicators)  │ ORDER TICKET    │
│ WATCH     │                                            │ buy/sell·lots   │
│ (asset    │                                            │ SL/TP·margin    │
│  groups,  │                                            ├────────────────┤
│  live     │                                            │ ACCOUNT         │
│  bid/ask) │                                            │ bal·eq·P/L      │
├───────────┴──────────────────────────────────────────┴────────────────┤
│ TRADE WATCH (tabs): Positions | Orders | History | Journal  (live P/L)  │
├────────────────────────────────────────────────────────────────────────┤
│ STATUS: cTrader · Demo · latency · data path · v0.5.0                    │
└────────────────────────────────────────────────────────────────────────┘
```
We already have every piece (Market Watch, Chart+indicators+live, order ticket, account
stream, positions). This re-arranges them into one cockpit.

### 2. Top-level groups = workspace tabs (cTrader's app-switcher)
- **Trade** (cockpit above) · **Autopilot** (Algo) · **Research** (Discovery/Training/Lab/
  Intelligence) · **Data & Files** · **Desk** (Journal/News/AI) · **System**.
- Keep the grouped left-rail too for the non-cockpit workspaces.

### 3. Bloomberg-isms to adopt
- **Command palette (Ctrl/⌘-K)** — jump to any symbol, screen, or action by typing.
- **Semantic color discipline** — green buy / red sell / **amber for headers, alerts,
  "pending/attention"** (the Bloomberg accent); monospace tabular numbers everywhere.
- **Latency / freshness pill** in the status bar (we already get freshnessSeconds).
- **Density pass** — tighter rows, right-aligned numerics, less padding in tables.

### 4. Card + modular grid for the dashboards
Research/Intelligence/Hardware/Account use card layouts with a mini-graph per metric
(equity curve, discovery progress, model accuracy) on a consistent modular grid.

### 5. Workspaces (later)
Save/restore panel layout + selected symbol/TF per workspace (cTrader parity), groundwork
for multi-monitor / detachable panels.

## Build order (when approved)
1. Trade cockpit (dock Market Watch + Chart + Order/Account + Trade-Watch into one screen).
2. Command palette (⌘K) + status-bar latency pill.
3. Density + semantic-color pass across tables/cards.
4. Card/mini-graph dashboards (equity curve from journal).
5. Saved workspaces.
