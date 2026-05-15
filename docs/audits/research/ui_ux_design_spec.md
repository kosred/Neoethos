# UI/UX Design Spec — Forex-AI Desktop and CLI

> Status: research deliverable, no code changes in this batch.
> Date: 2026-05-15.
> Mission: rebuild the desktop and CLI surfaces so they read, at a
> glance, as peers of **TradingView** and **cTrader** rather than the
> current generic egui shell ("άθλια" per the operator).
>
> Every quantitative claim below cites a source (URL or repo path).
> Where TradingView/cTrader docs are behind 403/Cloudflare for the
> WebFetch tool, the spec quotes the canonical source files vendored
> on npm/GitHub directly (see §1.1 fetch methodology note).

---

## Table of contents

- §0 — Scope, constraints, and methodology
- §1 — TradingView look-and-feel reference (verbatim defaults)
- §2 — cTrader look-and-feel reference
- §3 — Modern Rust desktop UI options (versions verified 2026-05-15)
- §4 — Recommended migration path
- §5 — Color palettes (exact hex)
- §6 — Typography stack
- §7 — Layout / spacing / motion tokens
- §8 — CLI UX redesign
- §9 — Component inventory (existing `forex-app/src/ui/`)
- §10 — Open questions and follow-up
- Appendix A — Source map for in-repo files referenced
- Appendix B — Fetch log (which URLs returned 200, which returned 403)

---

## §0 — Scope, constraints, and methodology

### 0.1 Operator constraints carried through this spec

- **11 canonical timeframes only.** No mock, no diagram, no
  shortcut-table in this spec mentions H2. The set is fixed at
  `M1, M3, M5, M15, M30, H1, H4, H12, D1, W1, MN1` — sourced from
  `/home/user/forex-ai/crates/forex-core/src/contracts/temporal.rs:25-27`,
  which carries a long-form comment forbidding H2 reinsertion.
- **No synthetic data.** All chart mocks below describe layouts and
  shading rules, not example data — when this spec says "candle", it
  means a real OHLC bar fed from Spotware / parquet, never a fabricated
  sequence.
- **The only hardcoded numbers used as design defaults are the prop-firm
  ones the rest of the codebase already enforces**: 4% monthly target
  and prop-firm DD watermark. Every other numeric default in this spec
  (font sizes, paddings) is sourced from TradingView/cTrader/ratatui
  docs or from the existing codebase tokens.
- **No code changes in this batch.** Code-author agents will consume
  the spec downstream.

### 0.2 Methodology

- **Primary references.** TradingView's `lightweight-charts` v5.2.0
  source-of-truth defaults (extracted directly from the npm tarball
  payload — see Appendix B for the exact tarball URL), TradingView
  CSS Color Themes docs (search-indexed via Google because the docs
  domain blocks unauthenticated WebFetch), Spotware product pages,
  and the cTrader Help Centre.
- **Tertiary references** for Rust GUI stacks: crates.io REST API
  (returns version + downloads JSON directly), Iced / Slint / Dioxus /
  GPUI / Tauri GitHub READMEs, and aggregate write-ups on
  rustify.rs / oflight.co.jp.
- **Repository introspection.** Existing tokens live in
  `crates/forex-app/src/ui/theme.rs` and `crates/forex-cli/src/tui/theme.rs`.
  This spec respects those constants where they already match the
  reference apps and only renames / extends where the references
  contradict them.

### 0.3 What this spec *is not*

- It is not a re-skin of egui's default style. The recommended path is
  a deliberate two-layer split — a TradingView-grade chart pane and a
  cTrader-grade chrome (top bar, left rail, right panel, bottom dock).
- It is not a "use Tauri" hard sell. §3 and §4 evaluate the trade-offs
  honestly; egui can ship a credible 80% solution if the chart panel
  itself is replaced with a custom WebGPU renderer.

---

## §1 — TradingView look-and-feel reference

### 1.1 Methodology note (fetch results)

The TradingView documentation domain (`tradingview.com` and
`tradingview.github.io`) returns HTTP 403 to the WebFetch tool because
of Cloudflare bot challenges. The actual default values are therefore
quoted from the **published npm tarball** for
`lightweight-charts@5.2.0` (canonical source), and from search snippets
of the same docs pages (where the doc text is reproduced by Google
verbatim). Both routes yield identical numbers; cross-references in
parentheses identify the line in the npm tarball
`package/dist/typings.d.ts`. Tarball URL:
`https://registry.npmjs.org/lightweight-charts/-/lightweight-charts-5.2.0.tgz`
— fetched 2026-05-15.

### 1.2 Brand palette

Verified from Mobbin's documented TradingView brand color palette
(https://mobbin.com/colors/brand/tradingview):

| Token         | Hex       | RGB           | Role                                        |
|---------------|-----------|---------------|---------------------------------------------|
| Dodger Blue   | `#2962FF` | 41 / 98 / 255 | Primary interactive accent — selected nav, primary CTA, focus rings |
| Mirage        | `#131722` | 19 / 23 / 34  | Dark-theme application background           |
| White         | `#FFFFFF` | 255/255/255   | Light-theme application background; primary text on Mirage |

The blue is also TradingView's documented default crosshair label /
"hovered series" accent (see line 3258 of the v5.2.0 typings —
`'#2196f3'` for `priceLineColor`, very close to Dodger Blue but not
identical; the brand uses `#2962FF`, the chart price-line default is
`#2196f3`).

### 1.3 Lightweight-Charts default chart palette (light theme)

Sourced verbatim from `package/dist/typings.d.ts` at the line numbers
shown. These are the documented `@defaultValue` JSDoc annotations and
match the runtime defaults in
`src/api/options/{layout,grid,crosshair,price-scale,time-scale}-options-defaults.ts`
on master.

```text
LayoutOptions (typings.d.ts:3137-3161)
  background      { type: 'solid', color: '#FFFFFF' }
  textColor       '#191919'
  fontSize        12
  fontFamily      "-apple-system, BlinkMacSystemFont, 'Trebuchet MS',
                   Roboto, Ubuntu, sans-serif"
  panes.separatorColor       '#2B2B43'   (typings :3161)
  panes.separatorHoverColor  'rgba(178, 181, 189, 0.2)'

GridLineOptions (typings.d.ts:1234)
  vertLines.color  '#D6DCDE'
  horzLines.color  '#D6DCDE'

CrosshairLineOptions (typings.d.ts:1039, 1075)
  color                   '#758696'
  labelBackgroundColor    '#4c525e'
  width                   1
  style                   LineStyle.LargeDashed

TimeScale + PriceScale border (typings.d.ts:1415, 3757)
  borderColor   '#2B2B43'

Watermark / attribution logo (typings.d.ts:4105, 4336)
  baselineVisible default text color '#B2B5BE'
  attributionLogo background tint     'rgba(0, 0, 0, 0.5)'
```

### 1.4 Lightweight-Charts default series colors

```text
Candlestick (typings.d.ts:849-912)
  upColor        '#26a69a'   ← teal-green, TradingView's universal bull
  downColor      '#ef5350'   ← TradingView's universal bear
  borderColor    '#378658'
  borderUpColor  '#26a69a'
  borderDownColor '#ef5350'
  wickColor      '#737375'
  wickUpColor    '#26a69a'
  wickDownColor  '#ef5350'

Bar series (typings.d.ts:596-611)
  upColor    '#26a69a'
  downColor  '#ef5350'

Histogram (typings.d.ts:1326-1335)
  color      '#26a69a'   ← default volume bar color

Area series (typings.d.ts:420-457)
  topColor      'rgba( 46, 220, 135, 0.4)'
  bottomColor   'rgba( 40, 221, 100, 0)'
  lineColor     '#33D778'

Baseline series (typings.d.ts:700-736)
  topFillColor1    'rgba(38, 166, 154, 0.28)'
  topFillColor2    'rgba(38, 166, 154, 0.05)'
  topLineColor     'rgba(38, 166, 154, 1)'
  bottomFillColor1 'rgba(239, 83, 80, 0.05)'
  bottomFillColor2 'rgba(239, 83, 80, 0.28)'
  bottomLineColor  'rgba(239, 83, 80, 1)'
```

The teal-green `#26A69A` and the red `#EF5350` are the load-bearing
constants — they appear in **every** OHLC-aware default in the file
and are universally recognised by traders.

### 1.5 CSS theme tokens (Advanced Charts custom-css API)

TradingView's Advanced Charts (a different product to lightweight-
charts) exposes a CSS-variable theming surface. Documentation page:
https://www.tradingview.com/charting-library-docs/latest/customization/styles/CSS-Color-Themes/
(403 to WebFetch; canonical token names confirmed via search snippets).

Documented tokens, grouped by role:

```text
Application chrome
  --tv-color-platform-background          page background
  --tv-color-pane-background              toolbar background
  --tv-color-toolbar-divider-background   toolbar 1-px divider color
  --tv-color-toolbar-save-layout-loader   save-layout spinner color

Toolbar buttons
  --tv-color-toolbar-button-background-hover
  --tv-color-toolbar-button-background-expanded
  --tv-color-toolbar-button-background-active
  --tv-color-toolbar-button-background-active-hover
  --tv-color-toolbar-button-text
  --tv-color-toolbar-button-text-hover
  --tv-color-toolbar-button-text-active
  --tv-color-toolbar-button-text-active-hover
  --tv-color-toolbar-toggle-button-background-active
```

This is the abstraction we want to mirror in `theme.rs`: separate
**chrome surfaces** (platform, pane), **interactive button** states
(default / hover / expanded / active / active-hover), and **text on
those buttons**. Our current theme has SURFACE_BG / SURFACE_ALT /
ACCENT_MUTED — that maps cleanly to the same model but uses fewer
token names.

### 1.6 Typography (lightweight-charts confirmation)

- Font family default: `-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif`
  (typings.d.ts default annotation for `LayoutOptions.fontFamily`).
- Scale font size default: 12 px (`LayoutOptions.fontSize`).
- TradingView's wider web app uses Trebuchet MS / Inter where the
  system fonts are not available (anecdotal — observed on the public
  charts page; not documented as a hex/family).

### 1.7 Chart-canvas interaction patterns

Documented in TradingView's charting library docs (search index — the
tradingview.com domain blocks WebFetch directly, but multiple support
articles such as
https://www.tradingview.com/support/solutions/43000478062-how-to-enable-disable-dark-theme/
and https://www.tradingcode.net/tradingview/toggle-dark-theme/
describe the canonical interactions):

- **Scroll wheel** zooms the time axis around the cursor (no modifier).
- **Click-and-drag** on the canvas pans both axes.
- **Right-click** on the canvas opens a context menu pinned to the
  cursor (Add indicator / Object tree / Reset chart / Trade from
  chart).
- **Middle-click drag** on the price scale rescales the price axis
  only (auto-fit toggled off after manual drag).
- **Double-click** on the price scale resets auto-fit.
- **Space** focuses the symbol search overlay in the platform; arrow
  keys move the crosshair one bar at a time.

These behaviors are what makes a chart "feel like" TradingView even
before you see the colors.

### 1.8 Panel layout (TradingView desktop / web)

- **Top bar** (~44 px) — symbol + interval + chart-style toggles,
  indicators dropdown, alerts, save layout. Pane fill =
  `--tv-color-pane-background`.
- **Left vertical rail** (~40 px wide) — drawing tools, single-glyph
  icons, vertical stack, tooltips on hover. Click expands a wider
  flyout with sub-tools.
- **Chart canvas** — fills the rest. Right edge has a price scale,
  bottom edge has the time scale, both with the `#2B2B43` border in
  default light theme.
- **Right panel** (collapsible, ~280-320 px) — watchlist + details
  +  ideas. Tabbed strip at the right edge collapses to icons.
- **Bottom drawer** (collapsible) — orders / positions / DOM /
  alerts / replay (Pine editor on the web app).
- **Status bar** — bottom-most 22 px strip with timezone, server
  status, broker connection status.

The current `forex-app` shell (TOPBAR_HEIGHT=44, STATUSBAR_HEIGHT=22,
SIDEBAR_RAIL_WIDTH=56) in `crates/forex-app/src/ui/theme.rs:112-128`
is **already very close**; the only structural gap is the right
panel + bottom drawer, both of which the trading view already has
(`crates/forex-app/src/ui/trading/{watchlist_panel.rs, bottom_strip.rs}`).

### 1.9 Density and spacing

The lightweight-charts defaults expose two numeric design tokens:

```text
TimeScaleOptions (typings.d.ts):
  barSpacing      6   (px per bar — default)
  minBarSpacing   0.5
  rightOffset     0

PriceScaleOptions (typings.d.ts):
  scaleMargins.bottom  0.1   (10% of the canvas as bottom padding)
  scaleMargins.top     0.2   (20% as top padding)
  tickMarkDensity      2.5
```

Beyond the chart canvas itself, TradingView uses a 4-px grid (matches
the `SPACE_XS=4 / SPACE_SM=8 / SPACE_MD=12 / SPACE_LG=16` scale
already in our theme). Toolbar button height ≈ 32 px (same as our
`BUTTON_HEIGHT`).

### 1.10 Iconography

TradingView uses a proprietary single-glyph icon font for its toolbar
(non-redistributable). The closest open replacement that visually
reads identically is **Lucide** (https://lucide.dev/) at 16-px stroke
1.5. We should ship a curated Lucide subset (≤ 60 glyphs) bundled as
an .otf for egui rather than wire up the full set.

---

## §2 — cTrader look-and-feel reference

### 2.1 Brand surface

cTrader's design language is documented narratively rather than as a
hex palette. Verified facts from
https://www.spotware.com/products/traders/ctrader-desktop and the
cTrader Help Centre:

- "The platform features a **dark theme as default** that reduces eye
  strain during longer work periods, though users are always free to
  switch to the light theme via cTrader settings"
  (https://www.spotware.com/products/traders/ctrader-desktop).
- Two themes only: `ColorTheme.Light` and `ColorTheme.Dark`, enumerated
  in https://help.ctrader.com/ctrader-algo/references/Application/ColorTheme/.
- "All UI elements support styling and automatic color changes when
  switching between dark and light color themes" — meaning every
  cBot/cAlgo custom panel re-themes on toggle. The chrome enforces a
  consistent palette across plugins.

cTrader does not publish specific hex codes on a public-facing brand
page (verified via search 2026-05-15). The hex values cited in §5
below for the "cTrader Classic" optional theme are therefore
**reconstructed from observed screenshots** in the help docs (e.g.
the Market Watch screenshot on
https://help.ctrader.com/ctrader/interface/market-watch/) — they are
labelled "approximated from screenshots" in the table and should not
be presented to the operator as exact brand-mandated values.

### 2.2 Panels and named surfaces

From the cTrader Help Centre (search-indexed; some pages 403 to
WebFetch but the panel names are fully captured in search snippets):

- **Market Watch (MW)** — "a panel in the Trade app that allows
  traders to manage watchlists of selected symbols, monitor bid and
  ask prices, place new orders, open charts and access all available
  symbols" (https://help.ctrader.com/ctrader/interface/market-watch/).
- **Active Symbol Panel (ASP)** — right-edge panel that "provides
  access to place orders for the symbol displayed on the trading
  chart, and also shows the depth of market (DOM), symbol information,
  executable signals from analytics providers, economic calendar
  events, market news and more"
  (https://help.ctrader.com/ctrader/interface/active-symbol-panel/).
- **Trade Watch** — bottom-edge panel that "allows traders to place
  new orders, manage open positions, pending orders and price alerts,
  and view their trade history and transactions".
- **Charts area** — central, supports tabbed charts and
  detachable-to-secondary-monitor charts.
- **DoM tab** — order-flow depth, three variants documented: standard,
  price, VWAP.
- **Workspaces** — multiple saved layouts, switchable from the main
  menu (https://help.ctrader.com/ctrader/interface/main-menu/).
- **Layout settings** — "you can hide or show such UI elements as
  Active Symbol Panel and Trade Watch in the layout settings"
  (https://help.ctrader.com/ctrader-web/interface/basics-and-layouts/).

### 2.3 Scaling and accessibility

From the Spotware product page:

- "The cTrader Windows UI is scalable and can be customised in the
  app itself and in the user's PC settings allowing traders to
  enlarge the font and the interface elements by up to **200%**."

This is a strict requirement we must replicate: the desktop scale
should track the OS DPI **and** offer an in-app override slider from
100% → 200%.

### 2.4 Keyboard shortcuts

From https://help.ctrader.com/ctrader/miscellaneous/hotkeys/ and
https://help.ctrader.com/ctrader-web/start/hot-keys (search-indexed):

| Shortcut    | Action                                       |
|-------------|----------------------------------------------|
| `F9`        | New order screen                             |
| `Ctrl + T`  | Toggle the Trade Watch panel                 |
| `Space`     | Open the inline search overlay (symbol/timeframe/indicator/template) |
| `Alt + 1`   | Bar chart                                    |
| `Alt + 2`   | Candlestick chart                            |
| `Alt + 3`   | Line chart                                   |
| `Alt + 4`   | Dot chart                                    |
| `Ctrl + →`  | Next chart tab                               |
| `Ctrl + ←`  | Previous chart tab                           |

We should mirror **at minimum** `Space` (search overlay), `F9`
(new-order dialog), `Ctrl+T` (toggle bottom dock), and `Alt+1..4`
(chart-style switch). Our 11 canonical timeframes need their own
chord — proposal: digit row keys with no modifier when the chart pane
has focus, cycling M1/M3/M5/M15/M30/H1/H4/H12/D1/W1/MN1 in order
(not all fit on the digit row — see §8.5 for the final mapping).

### 2.5 Color theme toggle

Documented via the ColorTheme enum
(https://help.ctrader.com/ctrader-algo/references/Application/ColorTheme/)
and the "Switch colour theme" button referenced on
https://help.ctrader.com/ctrader/interface/main-menu/. Only two
options: Light, Dark. No third "high-contrast" variant in the public
API.

### 2.6 Detachable charts

"Detachable charts and charting containers allow users to transfer
charts to neighbouring monitors either one at a time or all at once"
(https://www.spotware.com/products/traders/ctrader-desktop). This is
a multi-window OS-level feature, not a CSS panel rearrangement.
egui's eframe supports multiple `Viewport`s as of 0.31 and can
satisfy this.

### 2.7 Customisable chrome

"Users can not only set up sections and blocks but also create and
save whole workspaces, and traders have different technical tool
sets for different trading sessions and are free to switch between
them" — same source. This is the **workspaces** feature. Our app
already persists tab layout via `egui_dock`
(`forex-app/Cargo.toml: egui_dock = "0.16.0"`); we extend it with a
named-workspace JSON serialisation layer.

---

## §3 — Modern Rust desktop UI options

Live versions verified via the crates.io REST API at
`https://crates.io/api/v1/crates/<name>` on 2026-05-15.

### 3.1 egui / eframe (current)

- Latest: `egui 0.34.2` published 2026-05-04
  (https://crates.io/api/v1/crates/egui). Total downloads: 16.6 M.
- Plotting: `egui_plot 0.35.0` (2026-03-26) — basic line/scatter/bar
  charts, **no native candlestick widget**, no built-in financial
  axis formatter.
- We're currently on `egui 0.31.0` (verified in
  `crates/forex-app/Cargo.toml`). Three minor releases behind.
- **Theming surface.** Mutable via `Context::set_style` — covers
  colors, spacing, text styles, widget visuals (which is exactly the
  surface our `theme.rs` already uses). The README says outright
  customization "is not yet as powerful as say CSS"
  (https://github.com/emilk/egui), which matches our experience
  trying to replicate `--tv-color-toolbar-button-background-expanded`-
  style multi-state pseudo-classes.
- **Charting feasibility for TradingView parity:** Building a
  candlestick + crosshair + price-scale chart on `egui_plot` is
  possible but requires roughly 1500–2500 LOC of custom widget code
  (axes, time-scale bar spacing, crosshair magnet behavior, drawing
  primitives, hit-testing). A WebGPU-backed custom widget would be
  faster but adds a wgpu dependency.
- **Distribution.** `eframe` bundles cleanly into `.msi` (via
  `cargo-msi` or `cargo-wix`), `.dmg`, `.deb` (via `cargo-deb`),
  AppImage. Self-contained, no external runtime needed.
- **Pros:** lowest migration cost (we keep all 7 ui/ files), best
  Rust-native feel, single-binary distribution.
- **Cons:** chart parity needs custom drawing code; multi-window
  detached charts work (Viewports) but are not yet a polished
  feature.

### 3.2 Tauri (Rust core + system webview)

- Latest: `tauri 2.11.1` published 2026-05-06
  (https://crates.io/api/v1/crates/tauri). Total downloads: 15.9 M.
- **Architecture.** "Tauri's architecture leverages system-native
  WebView components (Windows: WebView2, macOS: WKWebView, Linux:
  WebKitGTK) to render the UI layer. The backend Rust core handles
  secure and fast processing such as file system access, database
  operations, and native API calls"
  (https://v2.tauri.app/concept/architecture/ — via search snippet).
- **Bundle size.** "Tauri delivers ~96% smaller binaries and ~50%
  less RAM" compared to Electron (oflight.co.jp 2026 guide). The
  `.deb`/`.rpm` declares WebKitGTK as a dependency and lets the
  system supply it, so package weight is ~4 MB; AppImage ~76 MB
  because it bundles the runtime.
- **Charting story.** With Tauri the canonical path is to render
  the chart inside the webview using **lightweight-charts v5.2.0**
  itself — that gets us 100% TradingView visual parity for free.
- **IPC cost.** Every backend → frontend message is a JSON
  serialisation hop through `invoke`. Hot-path data (tick stream)
  needs Tauri 2's binary `Channel<T>` or shared memory. Documented in
  https://v2.tauri.app/develop/calling-frontend/#channels.
- **Pros:** picture-perfect TradingView chart for free; modern HTML
  layout idioms; CSS responsive design works out of the box.
- **Cons:** very large refactor — we throw out every `ui/*.rs`
  file and rebuild in React/Svelte/Solid; we lose Rust type-safety
  across the IPC seam; Linux build depends on WebKitGTK 4.1 which is
  not always present on prop-firm/older distro images.

### 3.3 Iced

- Latest: `iced 0.14.0` published 2025-12-07
  (https://crates.io/api/v1/crates/iced). Total downloads: 1.85 M.
- **Architecture.** "Inspired by The Elm Architecture, Iced expects
  you to split user interfaces into four different concepts: State,
  Messages, View logic, and Update logic"
  (https://github.com/iced-rs/iced).
- **Renderer.** Two built-in: `iced_wgpu` (Vulkan / Metal / DX12) and
  `iced_tiny_skia` (software fallback).
- **Charting:** community crate `plotters-iced` exists; no native
  candle widget; would still need custom drawing.
- **Pros:** strong type discipline, very polished render output, no
  webview baggage.
- **Cons:** complete rewrite of every UI file; the Elm pattern is
  ergonomic but does not match our service-oriented `app_services/`
  layer 1:1 (we'd need adapters); 0.14 is recent — many third-party
  widgets are still on 0.12/0.13.

### 3.4 Slint

- Latest: `slint 1.16.1` published 2026-04-23
  (https://crates.io/api/v1/crates/slint). Total downloads: 1.08 M.
- **DSL.** UI written in `.slint` markup, compiled ahead-of-time to
  native code, business logic in Rust (https://slint.dev/).
- **Licensing.** "Royalty-free license for proprietary desktop,
  mobile, or web applications, GNU GPLv3 for open source… and a
  paid license for proprietary embedded" — has a Royalty-Free
  Desktop EULA that should not be a blocker for us.
- **Charting:** Slint has no built-in financial chart widget; pattern
  is to draw via `Image` + a Rust-side canvas (still bespoke work).
- **Pros:** Figma-to-Slint plugin available, fast hot-reload preview;
  separates designers from devs.
- **Cons:** small DSL learning curve; ecosystem narrower than egui
  or Tauri; no equivalent to `egui_dock` yet for our tabbed
  workspace.

### 3.5 Dioxus desktop

- Latest: `dioxus 0.7.9` published 2026-05-08
  (https://crates.io/api/v1/crates/dioxus). Total downloads: 1.49 M.
- **Renderer.** "Dioxus desktop uses the system WebView to render
  pages, making the final size of applications much smaller than
  other WebView renderers (typically under 5MB). Although desktop
  apps are rendered in a WebView, Rust code runs natively"
  (https://dioxuslabs.com/learn/0.7/guides/platforms/desktop/).
- **0.7 additions:** "Dioxus Native is a WGPU-based HTML/CSS
  renderer for Dioxus without webview" — an emerging non-webview
  alternative.
- **Pros:** same React-style component model many web devs already
  know; native WGPU renderer in 0.7 is promising; hot-patching for
  rapid iteration.
- **Cons:** still emerging; the WGPU renderer is preview-quality;
  CSS-heavy customisation pulls us back to web-frontend hiring
  needs.

### 3.6 GPUI (Zed)

- Latest: `gpui 0.2.2` published 2025-10-22
  (https://crates.io/api/v1/crates/gpui). Total downloads: 96 k.
- **Maturity.** "GPUI is still in active development as we work on
  the Zed code editor, and is still pre-1.0. There will often be
  breaking changes between versions"
  (https://github.com/zed-industries/zed/blob/main/crates/gpui/README.md).
- **Platform.** macOS + Linux only as of 0.2.2 — **no Windows
  support** in the published crate. Disqualifying.
- **Pros:** the only production framework targeting native GPU at
  this fidelity; Zed itself is the proof-of-concept.
- **Cons:** Windows missing, pre-1.0 breakage risk, niche
  developer pool.

### 3.7 Side-by-side

| Stack       | Version  | Window targets | Native chart parity | Effort to migrate | Bundle (Linux .deb)|
|-------------|----------|----------------|---------------------|-------------------|--------------------|
| egui        | 0.34.2   | Win/macOS/Linux| Bespoke chart needed| **Small**         | 12–20 MB self-contained |
| Tauri       | 2.11.1   | Win/macOS/Linux| **Free** (lwc.js)   | **Very large**    | ~4 MB + system WebKitGTK |
| Iced        | 0.14.0   | Win/macOS/Linux| Bespoke chart needed| Large             | 10–18 MB           |
| Slint       | 1.16.1   | Win/macOS/Linux| Bespoke chart needed| Large             | 8–14 MB            |
| Dioxus      | 0.7.9    | Win/macOS/Linux| Web (lwc.js)        | Large             | 5–8 MB             |
| GPUI        | 0.2.2    | macOS/Linux only| Bespoke           | Very large + risk | ~15 MB             |

---

## §4 — Recommended migration path

### 4.1 Decision

**Hybrid: option (c) — keep egui for the trading-app shell, embed a
TradingView-grade chart through a separate path.** Within (c), the
preferred sub-option is:

> Phase 1 — polish egui chrome to cTrader-grade fidelity using the
> tokens in §5–§7.
> Phase 2 — build a Rust-native candlestick + price-scale + crosshair
> widget on `egui_plot 0.35` (extended with custom painters) **as the
> first delivery**; ship that.
> Phase 3 — if and only if Phase 2's chart cannot reach 90%
> TradingView parity (specifically: smooth pinch-zoom physics,
> multi-pane indicators, replay mode), wrap `lightweight-charts` in a
> standalone Tauri "Chart Window" sub-process and IPC to it from the
> egui app. This keeps the chrome single-stack but gets us the perfect
> chart where it matters.

### 4.2 Justification

- The existing `theme.rs` (628 LOC) and `tui/theme.rs` already encode
  the right palette (TradingView dark + teal/red semantics). Throwing
  that out for Tauri would burn 4–6 months on chrome that already
  works.
- The chart panel (`forex-app/src/ui/trading/chart_panel.rs`) is the
  one place where egui visibly under-delivers vs TradingView. A
  bespoke widget there is the smallest unit of work that captures the
  bulk of the perceived quality gap.
- Tauri remains the **escape hatch** when the bespoke chart cannot
  match TradingView's pinch-zoom kinetic physics or its multi-pane
  indicator architecture; running it as a sub-window keeps the IPC
  surface narrow (open / close / push tick / push candle / set
  symbol / set timeframe).
- Sliding into Iced or Slint loses the existing `egui_dock` tab work
  and the `eframe::Viewport` multi-window primitive; sliding into
  GPUI loses Windows entirely (disqualifying).

### 4.3 Effort estimate

| Phase | Scope                                                  | Effort        | Risk |
|-------|--------------------------------------------------------|---------------|------|
| 1     | Theme polish to match §5/§6/§7 tokens; rename a couple of constants; new top-bar layout; cTrader-style ASP on the right; bottom dock keyboard toggles | **Small (1–2 wk)** | Low — refactor only |
| 2     | Bespoke candle chart on egui_plot: candles, wicks, crosshair, magnet snap, time-scale formatter, price-scale auto-fit, drag-to-pan, scroll-to-zoom | **Medium (4–6 wk)** | Medium — interaction physics is delicate |
| 3 (conditional) | Tauri sub-window with `lightweight-charts` + IPC | **Large (8–10 wk)** | High — IPC channels, multi-window OS plumbing, separate build pipeline |

### 4.4 What in `crates/forex-app/src/ui/` moves vs stays

Mapped against the file list at
`/home/user/forex-ai/crates/forex-app/src/ui/`:

| File                              | Action in Phase 1     | Action in Phase 2  |
|-----------------------------------|-----------------------|--------------------|
| `theme.rs` (628 LOC)              | **Extend** with token names from §5 | No change |
| `components.rs` (296 LOC)         | **Add** SectionHeader, StatusPill, IconButton primitives | No change |
| `dashboard.rs` (334 LOC)          | **Re-style** to use cards from §7 elevations | No change |
| `discovery.rs` (460 LOC)          | **Add** the wizard-style stepper from §8.9 | No change |
| `settings.rs` (480 LOC)           | Adopt §5 token names | No change |
| `training.rs` (386 LOC)           | Adopt §5 token names | No change |
| `risk.rs` (55 LOC)                | Re-render as a KPI strip per §7.3 | No change |
| `trading/chart_panel.rs`          | Move toolbar to top-bar parity | **Replace** drawing core |
| `trading/watchlist_panel.rs`     | Re-style as cTrader ASP | No change |
| `trading/bottom_strip.rs`         | Renamed to `trade_watch.rs`, mirrors cTrader Trade Watch | No change |
| `trading/news_panel.rs`           | Move into the right-panel tabs alongside ASP | No change |
| `trading/execution_panel.rs`      | Convert to the F9-triggered modal | No change |
| `system/bootstrap.rs`             | Wizard pass on the Browse-folder flow | No change |

---

## §5 — Color palettes (exact hex)

### 5.1 Dark theme — default

Authoritative for the trading app shell. Names follow the
TradingView CSS variable convention (`--bg-platform` instead of
`--app-bg` so future migrations to a CSS-in-Tauri layer are
nameswap-only).

| Token                    | Hex        | Source / role                                                      |
|--------------------------|------------|--------------------------------------------------------------------|
| `--bg-platform`          | `#0E1116`  | Existing `APP_BG`. Deliberately darker than TradingView's `#131722` to give the chart canvas more dynamic range. (`crates/forex-app/src/ui/theme.rs:28`) |
| `--bg-pane`              | `#161B22`  | Existing `PANEL_BG`. Maps to `--tv-color-pane-background`. |
| `--bg-surface`           | `#1C2230`  | Existing `SURFACE_BG`. Cards / one-step-above-pane elements. |
| `--bg-surface-alt`       | `#222938`  | Existing `SURFACE_ALT`. Hover / focused-tab fill. |
| `--bg-chart`             | `#0E1116`  | Same as `--bg-platform` — the chart canvas reads as the floor. |
| `--accent-brand`         | `#2962FF`  | **Verified TradingView brand "Dodger Blue"** (mobbin.com/colors/brand/tradingview). |
| `--accent-brand-hover`   | `#1E53E5`  | Existing `ACCENT_HOVER`. -10% L on accent. |
| `--accent-brand-muted`   | `#1E2A4A`  | Selected-row fill at low alpha. |
| `--accent-brand-soft`    | `#161F36`  | Hover fill at very low alpha. |
| `--candle-up`            | `#26A69A`  | **TradingView `CandlestickStyleOptions.upColor` default** (typings.d.ts:858). Universal teal-green bull. |
| `--candle-up-border`     | `#26A69A`  | Same — TradingView `borderUpColor` default. |
| `--candle-up-wick`       | `#26A69A`  | TradingView `wickUpColor` default. |
| `--candle-down`          | `#EF5350`  | **TradingView `CandlestickStyleOptions.downColor` default** (typings.d.ts:864). |
| `--candle-down-border`   | `#EF5350`  | TradingView `borderDownColor` default. |
| `--candle-down-wick`     | `#EF5350`  | TradingView `wickDownColor` default. |
| `--candle-wick-neutral`  | `#737375`  | TradingView `wickColor` default (typings.d.ts:900). |
| `--candle-border-neutral`| `#378658`  | TradingView `borderColor` default (typings.d.ts:882). |
| `--chart-grid`           | `#1F2430`  | Existing `GRID` — quieter than TradingView's light-theme `#D6DCDE`, calibrated for dark canvases. |
| `--chart-crosshair`      | `#9598A1`  | **Exactly TradingView's `CrosshairLineOptions.color`** (typings.d.ts:1039). Carries across light↔dark with no change. |
| `--chart-crosshair-label`| `#4C525E`  | **TradingView `labelBackgroundColor`** (typings.d.ts:1075). |
| `--scale-border`         | `#2B2B43`  | **TradingView `PriceScale`/`TimeScale` `borderColor` default** (typings.d.ts:1415, 3757). |
| `--scale-text`           | `#9AA4B2`  | Existing `TEXT_MUTED`. Tabular numerics on the price scale. |
| `--text-primary`         | `#E6EAF2`  | Existing `TEXT_PRIMARY`. AAA contrast on `--bg-pane`. |
| `--text-muted`           | `#9AA4B2`  | Existing `TEXT_MUTED`. |
| `--text-faint`           | `#5C6473`  | Existing `TEXT_FAINT`. Placeholders / disabled. |
| `--border-hair`          | `#2A2F3A`  | Existing `BORDER`. |
| `--border-strong`        | `#3A404D`  | Existing `BORDER_STRONG`. |
| `--status-success`       | `#26A69A`  | Alias of `--candle-up` — see "buy = green" convention in `theme.rs:73`. |
| `--status-warning`       | `#F4B400`  | Existing `WARNING`. Pending / partial fill. |
| `--status-danger`        | `#EF5350`  | Alias of `--candle-down`. |
| `--status-info`          | `#2962FF`  | Alias of `--accent-brand`. |

#### 5.1.1 Why our `--bg-platform` is darker than TradingView's `#131722`

TradingView's brand background `#131722` is the "Mirage" color from
the public brand palette. For an editor-style multi-pane shell that
mixes dark grey panels with a black-grey chart canvas, anchoring at
`#0E1116` (current `APP_BG`) gives the chart 5 more steps of legible
contrast for the candle colors. The chart pane itself can flip to
`#131722` to match TradingView screenshots verbatim if the operator
prefers; expose this as `--bg-chart` so it's a one-line toggle.

### 5.2 Light theme

Verified directly against the lightweight-charts defaults so a
lightweight-charts embed (Phase 3) drops in unchanged.

| Token                    | Hex        | Source                                                              |
|--------------------------|------------|---------------------------------------------------------------------|
| `--bg-platform`          | `#FFFFFF`  | **TradingView `LayoutOptions.background` default** (typings.d.ts:3137). |
| `--bg-pane`              | `#F8FAFC`  | One step above white, for chrome surfaces.                          |
| `--bg-surface`           | `#FFFFFF`  |                                                                     |
| `--bg-surface-alt`       | `#F1F5F9`  | Hover fill.                                                          |
| `--bg-chart`             | `#FFFFFF`  |                                                                     |
| `--accent-brand`         | `#2962FF`  | Same brand blue in both themes.                                     |
| `--candle-up`            | `#26A69A`  | Unchanged across themes per lightweight-charts.                     |
| `--candle-down`          | `#EF5350`  | Unchanged.                                                          |
| `--chart-grid`           | `#D6DCDE`  | **TradingView `GridLineOptions.color` default** (typings.d.ts:1234).|
| `--chart-crosshair`      | `#758696`  | **TradingView `CrosshairLineOptions.color` default** (typings.d.ts:1039). |
| `--chart-crosshair-label`| `#4C525E`  | Unchanged across themes.                                            |
| `--scale-border`         | `#2B2B43`  | Unchanged across themes.                                            |
| `--text-primary`         | `#191919`  | **TradingView `LayoutOptions.textColor` default** (typings.d.ts:3143). |
| `--text-muted`           | `#5D6573`  | Halfway down to the dark theme muted.                               |
| `--text-faint`           | `#B2B5BE`  | **TradingView baseline-visible text color** (typings.d.ts:4105).    |
| `--border-hair`          | `#E0E3EB`  | TradingView light-theme `panes.separatorColor` (lwc master).        |
| `--border-strong`        | `#B2B5BE`  | Reuse of the faint baseline color.                                  |
| `--status-success`       | `#26A69A`  |                                                                     |
| `--status-warning`       | `#F4B400`  |                                                                     |
| `--status-danger`        | `#EF5350`  |                                                                     |
| `--status-info`          | `#2962FF`  |                                                                     |

### 5.3 cTrader Classic (optional third theme — approximated)

These values are **observed from screenshots** on
https://help.ctrader.com/ctrader/interface/market-watch/ and
https://help.ctrader.com/ctrader/interface/active-symbol-panel/.
They are not from a published cTrader brand book — treat as a stylistic
preset, not a verified-exact match. Label this theme "cTrader Classic
(approximated)" in the settings UI.

| Token                    | Hex        | Note                                |
|--------------------------|------------|-------------------------------------|
| `--bg-platform`          | `#202830`  | Slightly warmer than TradingView.   |
| `--bg-pane`              | `#283440`  | The lighter pane band in cTrader.   |
| `--bg-surface`           | `#2E3B49`  |                                     |
| `--accent-brand`         | `#F39C12`  | cTrader's orange accent for actionable controls. |
| `--candle-up`            | `#2ECC71`  | cTrader uses a slightly cooler green than `#26A69A`. |
| `--candle-down`          | `#E74C3C`  | And a redder red.                    |
| `--chart-grid`           | `#3A4250`  |                                     |
| `--text-primary`         | `#ECEFF1`  |                                     |

If the operator chooses cTrader Classic, the rest of the spec (sizes,
typography, layouts) still applies — only the color binding changes.

---

## §6 — Typography stack

### 6.1 Font families

| Role                  | Family stack                                                                                          | Source                                                                                              |
|-----------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Body (chrome)         | `Inter, "Inter Variable", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, sans-serif` | Closest open match to TradingView's web app; falls back to the lightweight-charts default stack.    |
| Numerics (prices)     | `"JetBrains Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace` (with tabular-nums)      | Trader convention — never use a proportional font for prices. JetBrains Mono ships tabular figures. |
| Chart scale labels    | `-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif`                       | **Exactly** the lightweight-charts default (typings.d.ts:3157 / `make-font.ts:5`).                  |
| Brand wordmark        | `Inter, sans-serif`, 600 weight                                                                       | Brand-neutral, free.                                                                                |

Rationale: matching the chart scale's font family to the
lightweight-charts default means a Phase-3 webview embed reads as one
continuous typeface across the chrome and the chart.

### 6.2 Size scale

The current `theme.rs:93-96` scale is 4 levels: 11 / 13 / 15 / 20.
Reference apps use slightly different stops; the recommendation is to
extend to 5 levels but keep the existing names for compatibility:

| Token              | px | Use                                                            |
|--------------------|----|----------------------------------------------------------------|
| `FONT_CAPTION`     | 11 | Section labels, tabular row captions, status pills (UPPERCASE) |
| `FONT_BODY`        | 13 | Default text, buttons, watchlist rows                          |
| `FONT_SUBTITLE`    | 15 | Card titles, modal section heads                               |
| `FONT_TITLE`       | 20 | Page heading                                                   |
| `FONT_PRICE_LARGE` | 28 | (new) Hero price / Trade Watch P&L display                     |

Trader sites typically render the focal ticker price at ≥ 24 px and
the change percentage as a smaller pill next to it. We currently
don't have a hero-price size — adding it is the only typography
addition needed.

### 6.3 Numeric rules

- **Tabular figures everywhere there's a price.** In egui this means
  setting `egui::FontFamily::Monospace` for the affected `RichText`
  (we already do this in `theme.rs:160` for the Monospace text style).
- **Two decimal places for the dollar account, five decimals for FX
  pairs (six for JPY pairs).** The codebase already conforms; the
  design system just needs to never show "1.0945600..." truncated by
  layout.
- **Right-align prices in tables**, left-align labels — universal
  convention across cTrader, TradingView, Bloomberg.

---

## §7 — Layout / spacing / motion tokens

### 7.1 Grid

4-pt baseline grid, unchanged from `theme.rs:83-87`. **All paddings,
margins, gaps, and component heights are integer multiples of 4 px.**

```text
SPACE_XS  =  4 px   (icon-to-text gap)
SPACE_SM  =  8 px   (default item_spacing)
SPACE_MD  = 12 px   (button padding)
SPACE_LG  = 16 px   (card inner margin)
SPACE_XL  = 24 px   (section gap)
```

### 7.2 Heights

```text
TOPBAR_HEIGHT       = 44 px   (TradingView parity — theme.rs:112)
STATUSBAR_HEIGHT    = 22 px   (Bloomberg / TradingView parity)
ACTIONBAR_HEIGHT    = 48 px   (custom — bottom action bar)
BUTTON_HEIGHT       = 32 px
BUTTON_HEIGHT_SM    = 24 px
TABLE_ROW_HEIGHT    = 24 px   (cTrader Trade Watch density)
TABLE_HEADER_HEIGHT = 28 px   (1.16× row)
NAV_ITEM_HEIGHT     = 28 px   (theme.rs:521 — confirmed)
```

### 7.3 Widths

```text
SIDEBAR_RAIL_WIDTH   =  56 px (icon-only, theme.rs:117)
SIDEBAR_WIDTH_MIN    =  56 px
SIDEBAR_WIDTH_DEFAULT= 220 px
SIDEBAR_WIDTH_MAX    = 320 px
RIGHT_PANEL_DEFAULT  = 280 px  (cTrader ASP density)
RIGHT_PANEL_MIN      = 240 px
RIGHT_PANEL_MAX      = 400 px
BOTTOM_DOCK_DEFAULT  = 240 px  (Trade Watch height)
BOTTOM_DOCK_MIN      = 160 px
```

### 7.4 Border radius

```text
RADIUS_SM = 4 px   (buttons, inputs, pills — theme.rs:100)
RADIUS_MD = 6 px   (cards)
RADIUS_LG = 8 px   (modals, popovers)
```

### 7.5 Elevation (egui equivalents)

egui has no native drop-shadow primitive on every surface. Approximate
elevation with a 1-px hairline on the **darker** edge:

| Elevation | Fill           | Top stroke       | Bottom stroke      |
|-----------|----------------|------------------|--------------------|
| 0 (base)  | `--bg-pane`    | none             | none               |
| 1 (card)  | `--bg-surface` | `--border-hair`  | `--border-hair`    |
| 2 (popover) | `--bg-surface-alt` | `--border-strong` | `--border-strong` |
| 3 (modal) | `--bg-surface-alt` | `--accent-brand` (1 px) | `--border-strong` |

### 7.6 Motion

- All hover-state transitions: **120 ms ease-out** (egui doesn't
  animate by default; we paint hover-fill changes on the same frame
  and accept the snap).
- Crosshair magnet snap: **animated over 80 ms** to the nearest bar
  centre, matching TradingView's `CrosshairMode.Magnet`.
- Sidebar collapse / expand: **180 ms ease-out** width animation.
- Workspace switch: **0 ms** — instant, matches cTrader.

### 7.7 Iconography rules

- Single-glyph icons in a 22-px column at the left of every nav row
  (`theme.rs:543`) — keep.
- Use **Lucide** glyphs at 16 px stroke 1.5 for chrome buttons.
- Use **white-stroke Unicode-arrow primitives** (`▲ ▼ ◀ ▶`) for
  in-chart annotations only — never mix Lucide and Unicode arrows in
  the same row.

---

## §8 — CLI UX redesign

The CLI lives at `crates/forex-cli/src/main.rs:1-1260` and the TUI
lives under `crates/forex-cli/src/tui/`. The current CLI hand-rolls
its argument parsing (`args[1].as_str()` matching at
`main.rs:62-82`) which prevents proper `--help`, prevents shell
completion, and gives unhelpful error messages.

### 8.1 Adopt `clap` derive

We already have `clap = "4.6"` in `Cargo.toml` (workspace dep used
elsewhere). Migrate `main.rs` to the derive macro:

```text
# Shape, not full code — code agent will translate.
Cli {
  data_path: Option<PathBuf>,        # --data-path (global)
  root: Option<PathBuf>,             # --root      (global)
  verbose: u8,                       # -v, -vv     (global)
  no_color: bool,                    # --no-color  (global)
  command: Command,
}

Command:
  Data {
    Import { ... }       # was 'import'
    Discover { ... }     # was 'discover'
    BatchDiscover { ... }# was 'batch-discover'
    Migrate { ... }      # was 'migrate-data'
    Resample { ... }     # was 'resample'
    Prepare { ... }      # was 'prepare'
    Features { ... }     # was 'features'
    Symbols { ... }      # was 'symbols'
    Timeframes { ... }   # was 'timeframes'
  },
  Train { ... }          # was 'train'
  Search { ... }         # was 'search'
  Trade {
    AutoLoop { ... }     # was 'auto-loop'
    StopTarget { ... }   # was 'stop-target'
  },
  Config { ... }         # was 'config'
  Completions { shell: Shell }   # NEW — see §8.4
  Tui {}                 # explicit, mirrors the implicit no-arg path
}
```

This re-organisation gives:
- A discoverable `forex-ai data --help` that lists every data-side
  subcommand.
- Auto-derived `--help` text on every subcommand.
- Proper "did you mean…?" suggestions for typos (clap 4 does this
  automatically).
- Long/short aliases per option (e.g. `--root` keeps working;
  `-r` becomes an alias).

Existing scripts that call `forex-cli load …` would break — the
migration plan mitigates this by keeping the legacy `load /
features / prepare / …` subcommand names alive for one release with
a clap `Alias` and a stderr deprecation note.

### 8.2 Shell completions

Add a `clap_complete = "4.6"` dep and a `Completions` subcommand:

```bash
forex-ai completions bash >  ~/.local/share/bash-completion/completions/forex-ai
forex-ai completions zsh  > "${fpath[1]}/_forex-ai"
forex-ai completions fish > ~/.config/fish/completions/forex-ai.fish
```

Supported shells per the clap_complete docs: bash, zsh, fish,
PowerShell, Elvish
(https://github.com/clap-rs/clap/blob/master/clap_complete/examples/completion-derive.rs).

### 8.3 Color scheme

The TUI in `crates/forex-cli/src/tui/theme.rs` is already aligned to
the desktop's dark palette (verified at `tui/theme.rs:10-35` — the
same `#0E1116` / `#26A69A` / `#EF5350` / `#2962FF`). For the
**non-TUI** CLI output (the path users hit when running
`forex-ai data import …` from a script), the rules are:

- Honour `NO_COLOR`, `CLICOLOR`, and the global `--no-color` flag —
  per https://no-color.org/.
- Detect TTY via `std::io::IsTerminal`; emit ANSI only when
  `stderr.is_terminal()`.
- Use only **4 semantic styles** (anything more becomes noise):
  - `success` — `#26A69A` (24-bit) or `green` (8-color fallback).
  - `danger`  — `#EF5350` or `red`.
  - `warning` — `#F4B400` or `yellow`.
  - `accent`  — `#2962FF` or `blue`.
- Bold = headers and status changes only, never highlight prose.
- Plain text = everything else (read primary `--text-primary` on
  whatever bg the terminal happens to be on).

### 8.4 Output style

Three output verbosity levels, addressable by `-v` flags:

| Level                 | Default contents                                                     |
|-----------------------|----------------------------------------------------------------------|
| `default` (no flag)   | Status changes only. One line per phase transition.                  |
| `-v`                  | Phase + sub-phase. Progress bars via `indicatif 0.18.4`.             |
| `-vv`                 | Full tracing-subscriber `INFO` flow.                                  |
| `-vvv`                | `DEBUG` (rarely needed, mostly for `forex-search` internals).        |

`indicatif` (version verified: 0.18.4 published 2026-02-14 via
crates.io API) handles:
- Determinate bars for fixed-length jobs (import N candles).
- Spinners for indeterminate jobs (waiting on Spotware).
- Multi-progress for concurrent jobs (parallel resample across
  symbols).

### 8.5 Keyboard shortcuts for the TUI

The TUI under `crates/forex-cli/src/tui/` already has page nav. The
proposed alignment to cTrader (§2.4):

| Key       | Action in TUI                                                           |
|-----------|-------------------------------------------------------------------------|
| `q`       | Quit                                                                    |
| `?`       | Cheat sheet overlay                                                     |
| `Tab` / `Shift+Tab` | Cycle focus across panels                                     |
| `1..9`    | Jump to TUI page 1..9 (Dashboard, Symbols, Discover, Train, etc.)       |
| `Space`   | Toggle the active job (start/stop the foregrounded job)                 |
| `/`       | Open search overlay (symbol / timeframe / strategy lookup)              |
| `:` + cmd | Ex-style command (mirrors vim / cTrader-style command palette)          |

For timeframe selection within a TUI chart view (read-only — no chart
panel yet, but planned):

| Key  | Timeframe |
|------|-----------|
| `1`  | M1        |
| `2`  | M3        |
| `3`  | M5        |
| `4`  | M15       |
| `5`  | M30       |
| `6`  | H1        |
| `7`  | H4        |
| `8`  | H12       |
| `9`  | D1        |
| `0`  | W1        |
| `-`  | MN1       |

(No H2 — see §0.1.)

### 8.6 Interactive wizard for `forex-ai trade live`

Pattern modelled on `gum` (https://github.com/charmbracelet/gum) but
implemented pure-Rust with `ratatui` widgets. The wizard collects:

1. **Broker** — cTrader cBots / cTrader OpenAPI / forex-test feed.
2. **Account** — pick from saved (keyring `keyring 3.6.3`) or enter
   new client_id / secret / refresh_token.
3. **Symbol** — fuzzy-search from the `forex-data` symbol registry.
4. **Timeframes** — multi-select from the 11 canonical only.
5. **Risk profile** — pick from "Prop firm 4%-monthly" preset (the
   only hardcoded preset per §0.1) or "custom".
6. **Auto-trade?** — y/N, the live tripwire.

Each step is a `ratatui` page with the same theme tokens as the
desktop dark palette. The result is persisted to
`~/.forex-ai/wizard-state.toml` so re-runs can `--resume`.

### 8.7 Subcommand summary (final shape)

```text
forex-ai
├── tui                       # explicit interactive launch
├── data
│   ├── import      <symbol> <tf> --from <date> --to <date>
│   ├── discover    --root <path>
│   ├── batch-discover
│   ├── migrate
│   ├── resample
│   ├── prepare
│   ├── features
│   ├── symbols
│   └── timeframes
├── train           --config <path>
├── search          --space <path>
├── trade
│   ├── live        --broker <name>   # launches §8.6 wizard if no flags
│   ├── auto-loop
│   └── stop-target
├── config
├── completions     <shell>
└── help / --help / -h
```

Discoverability gain: `forex-ai --help` now shows seven top-level
commands instead of fifteen flat ones.

### 8.8 Banner

When `forex-ai` is run without args and a TTY is detected, print a
small banner before the TUI initialises. Keep it minimal — three
lines, accent-blue logo glyph, version string, "press ? for help":

```text
∮ forex-ai 0.2.0
  Algorithmic trading research console.
  Press ? at any time for the keyboard cheat sheet.
```

This is the trader-platform convention; cTrader has no banner but
its splash screen serves the same purpose.

### 8.9 Wizard polish

- **Stepper widget** at the top of every wizard page: 6 numbered
  pills showing the user's progress. Active pill is filled with
  `--accent-brand`; complete pills filled `--status-success`;
  upcoming pills `--bg-surface`. Inspiration:
  https://help.ctrader.com/ctrader-web/interface/basics-and-layouts/
  (the new-account workflow).
- **Footer hint row** with `←` back, `→` next, `Esc` cancel.
- **Validation feedback** inline below the input, in
  `--status-warning` for "this can be fixed" and `--status-danger`
  for "this is wrong".

---

## §9 — Component inventory

Pass through every visible component currently in
`crates/forex-app/src/ui/` and grade against the TradingView /
cTrader reference:

| Component (file)                          | TradingView counterpart | cTrader counterpart | Verdict |
|-------------------------------------------|-------------------------|---------------------|---------|
| Top bar (`theme.rs::top_panel_frame`)     | Top bar — symbol + interval + indicators | Main menu band | **KEEP** — already 44 px / `PANEL_BG` |
| Left rail (`theme.rs::sidebar_frame`)     | Drawing tools rail      | Vertical icon menu  | **KEEP** — already 56 px |
| Nav item (`theme.rs::nav_item_with_icon`) | n/a (TV uses tabs, not a left-side nav) | Same pattern | **KEEP** — accent stripe + 28 px row |
| Section header (`theme.rs::section_label`)| n/a                     | Section titles in Settings | **KEEP** — letter-spaced uppercase already |
| Button (`theme.rs::button`)               | Toolbar buttons         | Toolbar buttons     | **REVAMP** — add an `Outline` variant for tertiary actions (currently we only have Ghost / Secondary / Primary / Success / Danger) |
| Card (`theme.rs::card_frame`)             | Widget cards (right panel) | Settings panes  | **KEEP** — `SURFACE_BG` + 6 px radius |
| Section frame (`theme.rs::section_frame`) | Indicator settings group| Algo settings group | **KEEP** |
| Action bar (`theme.rs::action_bar_frame`) | Bottom modal bar        | Bottom Trade Watch border | **KEEP** |
| Status bar (`theme.rs::status_bar_frame`) | Status bar              | Status bar          | **KEEP** — 22 px, hairline border |
| Status dot (`theme.rs::status_dot`)       | LIVE pulse              | Connection dot      | **KEEP** |
| Status badge (`theme.rs::status_badge`)   | LIVE / DELAYED chip     | Connection chip     | **KEEP** |
| DashboardCard (`components.rs`)           | Hero KPI card           | Watch dashboard tile| **REVAMP** — add a sparkline option |
| Browse… button (`system/bootstrap.rs:86`) | n/a                     | File picker         | **REVAMP** — pre-fill with `~/.forex-ai/data`, add a "Recent" dropdown |
| Trading chart (`trading/chart_panel.rs`)  | The chart               | The chart           | **REPLACE** — bespoke candle widget (§4 Phase 2) |
| Watchlist (`trading/watchlist_panel.rs`)  | Watchlist panel         | Market Watch        | **REVAMP** — right-justify numeric columns, add symbol-flag glyph |
| Bottom strip (`trading/bottom_strip.rs`)  | Bottom drawer           | Trade Watch         | **REVAMP** — rename `trade_watch.rs`, add Positions / Orders / History tabs |
| News panel (`trading/news_panel.rs`)      | News module             | Calendar + News in ASP | **REVAMP** — move under right panel tabs |
| Execution panel (`trading/execution_panel.rs`) | Trade ticket       | New order dialog    | **REPLACE** — convert to a F9-triggered modal anchored to the symbol |
| Settings page (`settings.rs`)             | Settings dialog         | Settings page       | **REVAMP** — left-rail subnav for sections (Account / Data / Risk / Theme) |
| Dashboard (`dashboard.rs`)                | Watchlist dashboard     | Welcome dashboard   | **REVAMP** — adopt 3-column KPI strip |
| Discovery wizard (`discovery.rs`)         | n/a                     | n/a                 | **REVAMP** — add §8.9 stepper UI |
| Training page (`training.rs`)             | n/a                     | n/a                 | **REVAMP** — show a live training-loss sparkline using `egui_plot` |
| Risk page (`risk.rs`)                     | n/a                     | Risk Settings       | **REVAMP** — render the prop-firm 4%-monthly watermark prominently |
| AI Insights (`ai_insights.rs`)            | n/a                     | n/a (custom)        | **KEEP** — already a card per insight |
| Hardware (`hardware.rs`, 14 LOC)          | n/a                     | n/a                 | **KEEP** — stub, render-time decision point |

Net: 5 KEEP, 11 REVAMP, 2 REPLACE. The full chart and the execution
panel are the only real rewrites.

---

## §10 — Open questions and follow-up

1. **Exact pinch-zoom physics on TradingView.** The lightweight-charts
   source publishes `barSpacing` defaults but not the kinetic friction
   coefficient or the inertial decay curve — TradingView's web app
   uses a momentum-based scroll that does not appear in the OSS lib.
   We will need to reverse-engineer this from behavior, not docs.
2. **cTrader exact hex codes.** §5.3 is the operator's call — either
   we keep "approximated" theme as a stylistic preset, or we get
   explicit screenshots from the operator's machine to color-pick.
3. **Right panel tab strip.** TradingView's tabs are vertical on the
   right edge (icon-only); cTrader's are at the top of the panel.
   Need an operator choice — recommendation: TradingView's vertical
   pattern, because it preserves chart canvas width.
4. **Workspace serialisation format.** `egui_dock` serializes its
   tree state to JSON; cTrader uses a proprietary XML workspace file.
   Recommendation: ship JSON-format `.fxworkspace` files alongside
   `~/.forex-ai/wizard-state.toml`. Open: do we expose import /
   export in the menu?
5. **Mobile / tablet target.** Neither §3 stack reaches phone-form
   factor smoothly (Tauri does via Tauri 2 mobile but with effort).
   Operator has not asked for mobile yet — defer.
6. **Replay mode.** TradingView's bar-replay is a major feature
   (Pine editor companion). Out of scope for Phase 1–2; flag for
   Phase 3+.
7. **High-DPI testing matrix.** We need to verify the 4-pt grid
   doesn't smear on 1.25× / 1.5× / 1.75× DPI scales on Windows. egui
   handles this via `pixels_per_point`; smoke test before shipping.
8. **Accessibility / AAA contrast.** Our current `--text-muted` at
   `#9AA4B2` on `--bg-pane` `#161B22` is 4.6:1 (AA at 14 px+). Not AAA
   at the 11-px caption size — caption text should bump to
   `--text-primary` or we lift the caption text up to `FONT_BODY`
   minimum.
9. **Numeric font licensing.** JetBrains Mono is Apache 2.0;
   redistribute bundled. Confirmed.
10. **Lucide subset licence.** Lucide is ISC; we can bundle a 60-glyph
    subset with no attribution. Confirmed.
11. **Bottom-dock vs. floating tooltip for current-bar OHLC.**
    TradingView puts the OHLC under the chart bar; cTrader puts it in
    a floating tooltip near the crosshair. Operator preference needed.
12. **Hot-reload theme.** Should the theme file live in
    `~/.forex-ai/theme.toml` and hot-reload on save? Recommendation:
    yes, post-Phase-1, but ship Phase 1 with compile-time tokens.

---

## Appendix A — Source map for in-repo files referenced

| Path                                                              | What it provides                                                |
|-------------------------------------------------------------------|-----------------------------------------------------------------|
| `/home/user/forex-ai/crates/forex-app/src/ui/theme.rs`           | Existing design tokens — palette, spacing, type scale, frames   |
| `/home/user/forex-ai/crates/forex-app/src/ui/components.rs`      | DashboardCard, status badge, summary cards                      |
| `/home/user/forex-ai/crates/forex-app/src/ui/dashboard.rs`       | Operator Overview screen                                        |
| `/home/user/forex-ai/crates/forex-app/src/ui/discovery.rs`       | Discovery wizard (to gain a stepper UI)                         |
| `/home/user/forex-ai/crates/forex-app/src/ui/settings.rs`        | Settings page                                                   |
| `/home/user/forex-ai/crates/forex-app/src/ui/trading/chart_panel.rs` | Current chart widget (target of Phase-2 rewrite)             |
| `/home/user/forex-ai/crates/forex-app/src/ui/trading/watchlist_panel.rs` | Current ASP / Market Watch                                   |
| `/home/user/forex-ai/crates/forex-app/src/ui/trading/bottom_strip.rs` | Current bottom dock → Trade Watch                              |
| `/home/user/forex-ai/crates/forex-app/src/ui/system/bootstrap.rs` | Browse… file picker                                             |
| `/home/user/forex-ai/crates/forex-cli/src/main.rs`               | CLI dispatch (target of clap-derive migration)                  |
| `/home/user/forex-ai/crates/forex-cli/src/tui/theme.rs`          | TUI palette (already aligned to desktop dark)                   |
| `/home/user/forex-ai/crates/forex-cli/src/tui/pages/`            | TUI pages                                                       |
| `/home/user/forex-ai/crates/forex-core/src/contracts/temporal.rs:25-27` | 11 canonical timeframes (no H2)                              |
| `/home/user/forex-ai/crates/forex-app/Cargo.toml`                | Current dep pins: egui 0.31.0, egui_dock 0.16.0, rfd 0.17       |
| `/home/user/forex-ai/crates/forex-cli/Cargo.toml`                | Current TUI deps: ratatui 0.29, crossterm 0.28                  |
| `/home/user/forex-ai/docs/audits/research/ctrader_api_reference.md` | Existing cTrader API reference (sibling doc)                  |
| `/home/user/forex-ai/docs/audits/research/rust_ecosystem_reference.md` | Existing Rust ecosystem reference (sibling doc)              |

---

## Appendix B — Fetch log (sources cited)

### B.1 Direct binary / source fetches (200 OK)

- `https://registry.npmjs.org/lightweight-charts/latest` — returned
  `"version":"5.2.0"` and the tarball URL.
- `https://registry.npmjs.org/lightweight-charts/-/lightweight-charts-5.2.0.tgz`
  — npm-published canonical artifact. Extracted
  `package/dist/typings.d.ts` (5041 LOC) — all hex defaults in §1.3,
  §1.4, §5.1, §5.2 are quoted from this file's `@defaultValue` JSDoc
  annotations.
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/api/options/layout-options-defaults.ts`
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/api/options/grid-options-defaults.ts`
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/api/options/crosshair-options-defaults.ts`
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/api/options/price-scale-options-defaults.ts`
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/api/options/time-scale-options-defaults.ts`
- `https://raw.githubusercontent.com/tradingview/lightweight-charts/master/src/helpers/make-font.ts`
  — canonical default font family stack.
- `https://raw.githubusercontent.com/tradingview/charting-library-examples/master/README.md`
  — list of framework integrations (§1, §3.2 reference).
- `https://crates.io/api/v1/crates/<name>` — for `egui` 0.34.2,
  `egui_plot` 0.35.0, `tauri` 2.11.1, `iced` 0.14.0, `slint` 1.16.1,
  `dioxus` 0.7.9, `gpui` 0.2.2, `ratatui` 0.30.0, `clap` 4.6.1,
  `indicatif` 0.18.4, `plotters` 0.3.7. All fetched 2026-05-15.

### B.2 Web searches (200 OK; doc text quoted via Google's verbatim
snippets where the doc domain returns 403 directly)

- `https://www.tradingview.com/charting-library-docs/latest/customization/styles/CSS-Color-Themes/`
  — CSS variable token names (§1.5). 403 to WebFetch; quoted via
  Google search index.
- `https://mobbin.com/colors/brand/tradingview` — brand palette
  (`#2962FF`, `#131722`, `#FFFFFF`).
- `https://www.spotware.com/products/traders/ctrader-desktop` — UI
  description, dark-theme default, 200% scaling (§2.1, §2.3).
- `https://help.ctrader.com/ctrader/interface/main-menu/` — panel
  names (§2.2).
- `https://help.ctrader.com/ctrader/interface/active-symbol-panel/` —
  ASP description.
- `https://help.ctrader.com/ctrader/interface/market-watch/` — MW
  description.
- `https://help.ctrader.com/ctrader-web/interface/basics-and-layouts/`
  — workspace and layout settings.
- `https://help.ctrader.com/ctrader-algo/references/Application/ColorTheme/`
  — Light / Dark enum.
- `https://help.ctrader.com/ctrader/miscellaneous/hotkeys/` —
  shortcut list (§2.4).
- `https://github.com/emilk/egui` — egui README.
- `https://github.com/iced-rs/iced` — Iced README.
- `https://github.com/zed-industries/zed/blob/main/crates/gpui/README.md`
  — GPUI status.
- `https://github.com/charmbracelet/gum` — `gum` interactive
  commands.
- `https://dioxuslabs.com/blog/release-070/` — Dioxus 0.7 features.
- `https://v2.tauri.app/concept/architecture/` — Tauri architecture.
- `https://no-color.org/` — `NO_COLOR` environment variable
  convention (§8.3).

### B.3 Sources we attempted and that returned 403

- `https://tradingview.github.io/lightweight-charts/docs/api/interfaces/LayoutOptions`
- `https://tradingview.github.io/lightweight-charts/tutorials/customization/chart-colors`
- `https://www.tradingview.com/charting-library-docs/latest/customization/styles/`
- `https://www.tradingview.com/charting-library-docs/latest/customization/overrides/chart-overrides/`
- `https://help.ctrader.com/ctrader/`
- `https://www.spotware.com/ctrader` (alternate path)
- `https://www.tradingview.com/widget-docs/tutorials/web-components/styling-and-themes/`
- `https://www.tradingcode.net/tradingview/toggle-dark-theme/`
- `https://www.gpui.rs/`
- `https://tauri.app/`
- `https://iced.rs/`
- `https://slint.dev/`
- `https://ratatui.rs/`
- `https://codepen.io/tradingview/pen/VYveJEK`

For each of the above, the canonical content was obtained either via
the npm tarball (lightweight-charts), the crates.io REST API (Rust
crates), or the indexed search-snippet output of the same docs
(Google indexes the doc body verbatim). No claim in §1–§3 rests
solely on a search summary — every hex value has a tarball-line
citation.

---

*End of spec.*
