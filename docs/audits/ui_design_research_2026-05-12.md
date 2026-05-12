# UI design research — 2026-05-12

Source material for the v0.3 UI overhaul. The user's feedback on
v0.2.0 was that the UI is "not even close to cTrader or TradingView"
and "a bit chaotic". This document captures the patterns the four
dominant trading platforms converge on so that any future UI work
has a defensible reference, not a personal aesthetic.

## The four-platform survey

| Layout zone | Pattern across all four | Width / height |
|---|---|---|
| Top bar | Single row, no menu words. Symbol search · timeframe pills · indicators · layout · — · broker · equity · settings · avatar | 40-48 px |
| Left sidebar | Icon-only **rail** (TradingView 32-40 px) **or** wider **data panel** (cTrader Market Watch 280 px, MT5 Market Watch + Navigator 240 px) | 56 px or 240-300 px |
| Right sidebar | Order ticket / Active Symbol Panel (cTrader ASP, TV widget bar). Often collapsible | 280-340 px |
| Bottom panel | Tabbed: Positions · Orders · History · Journal · Log. Resizable, can collapse | 180-240 px (0 collapsed) |
| Center | Multi-chart grid OR single-chart with tabs above. Drawing tools pin to chart's left edge | fills |
| Status bar | Connection · latency · server time · spread · build | 22-24 px |

All four use **dark by default**, **green long / red short**,
**tabular monospaced numerals for prices**, **headers in
SemiBold UPPERCASE letter-spaced**.

## Specific platform patterns we are copying

**cTrader**
- Two-row header (~72 px total): app switcher row + chart toolbar row
- Market Watch: 280 px expandable symbol tiles with bid/ask, spread, sentiment, inline Buy/Sell + Open Chart
- Active Symbol Panel: 320 px with Buy/Sell ticket + DOM + sentiment gauge

**TradingView**
- 44 px top bar: symbol search · TF pills · candle-style · fx (indicators) · templates · alerts · replay · layouts · — · share · publish · avatar
- 32-40 px **icon-only** drawing rail down the chart's left edge (no labels, just tooltips)
- 32 px collapsed widget rail on the right; expands to 280-340 px showing Watchlist / Details / News / DOM / Calendar
- Bottom Account Manager: Positions · Orders · History · Account · Notifications · Journal
- Up to 8 charts in a grid layout; right-side price scale is the convention

**Bloomberg Terminal**
- Persistent command line at the top of every screen
- Yellow market-sector keys + green GO + red CANCEL
- Lesson: provide a **Ctrl+K command palette** as a parallel path

**MT5** (the legacy pattern — what NOT to copy)
- Standard menu bar (File/Edit/View/...) — every modern platform has dropped this
- 3 toolbar rows beneath the menu — too cluttered
- The Market Watch + Navigator stack on the left is fine; the menu bar is not

## Color palette (now in `ui/theme.rs`)

```
# Surfaces
bg            #0E1116   APP_BG, CHART_BG
panel         #161B22   PANEL_BG
panel-elev    #1C2230   SURFACE_BG
hover         #22293A   SURFACE_ALT

# Borders
border        #2A2F3A   BORDER
border-strong #3A404D   BORDER_STRONG
grid          #1F2430   GRID

# Text
primary    #E6EAF2   TEXT_PRIMARY
secondary  #9AA4B2   TEXT_MUTED
tertiary   #5C6473   TEXT_FAINT

# Brand
accent       #2962FF   ACCENT  (TradingView blue)
accent-hover #1E53E5   ACCENT_HOVER
accent-soft  #1E2A4A   ACCENT_MUTED

# Trading semantics (TradingView candle defaults)
buy / long  #26A69A    BUY (= SUCCESS)
buy strong  #00C853    BUY_STRONG
sell / short #EF5350   SELL (= DANGER)
sell strong  #FF1744   SELL_STRONG

# Status
warn   #F4B400  WARNING (amber)
info   #2962FF  INFO (= ACCENT)
```

Buy = teal-green `#26A69A` and sell = red `#EF5350` are the literal
TradingView candle defaults. Any trader who has ever opened TV
reads them without thinking.

## Concrete rules for any new UI in this repo

1. **Kill menu bars.** Replace with: top-bar icon buttons + left
   icon rail + Ctrl+K command palette. Every menu item must be a
   verb-action ("Connect broker"), not a noun ("Connections").
2. **Use the 5-zone shell**: top bar (44 px) · left rail (56 px
   icons) · center (charts via egui_dock) · right ticket panel
   (320 px, collapsible) · bottom panel (200 px tabbed,
   collapsible) · status bar (22 px).
3. **Number formatting is not optional.**
   - Prices: tabular monospaced, FX 5-digit (`1.08423`), bold the
     "big figure" (`1.08`**`42`**`3`).
   - P&L: `+$124.50` / `−$87.20` with sign, currency, color.
   - Percentages: `+0.42 %` (with space, sign, 2 decimals). Never
     raw `0.0042`. **This was the visible amateur tell in v0.2.0
     and is now fixed in `ui/risk.rs`.**
   - Volume: `0.10 lots` not `10000`.
4. **Density.** 8-px base grid. Body 12 px, headers 11 px UPPERCASE
   letter-spaced, tabular columns 12 px tabular numerals. Padding
   inside panels 8 px, between cards 12 px. Table row 24 px.
5. **Keyboard shortcuts on every action.** `B` = buy ticket, `S` =
   sell ticket, `Esc` = close ticket, `Ctrl+K` = palette, `1..6` =
   timeframe, `[` `]` = prev/next symbol.
6. **Colors are semantic, not decorative.** Green = long. Red =
   short. Blue = interactive / selected. Amber = pending. Anything
   else is a code smell.
7. **No widget on launch the user did not ask for.** No mailbox, no
   news ticker, no heatmap by default. Pro Mode opt-in only.
   (Devexperts UX/UI No-Nos.)

## Sources

- cTrader Web — Basics and layouts: https://help.ctrader.com/ctrader-web/interface/basics-and-layouts/
- cTrader Market Watch: https://help.ctrader.com/ctrader/interface/market-watch/
- cTrader Chart modes: https://help.ctrader.com/ctrader/charts/chart-modes/
- TradingView Toolbars: https://www.tradingview.com/charting-library-docs/latest/ui_elements/Toolbars/
- TradingView Layouts guide: https://www.tradingview.com/support/solutions/43000746975-tradingview-layouts-a-quick-guide/
- TradingView Multi-chart layouts: https://www.tradingview.com/support/solutions/43000629990-leveraging-multi-chart-layouts-in-your-analysis/
- TradingView CSS Color Themes: https://www.tradingview.com/charting-library-docs/latest/customization/styles/CSS-Color-Themes/
- TradingView Brand Colors (Mobbin): https://mobbin.com/colors/brand/tradingview
- MetaTrader 5 User Interface: https://www.metatrader5.com/en/terminal/help/startworking/interface
- MT5 Market Watch: https://www.metatrader5.com/en/terminal/help/trading/market_watch
- Bloomberg LP — concealing complexity in Terminal UX: https://www.bloomberg.com/company/stories/how-bloomberg-terminal-ux-designers-conceal-complexity/
- Devexperts — Trading Platform UX/UI No-Nos: https://devexperts.com/blog/trading-platform-ux-ui-design-no-nos/
- Devexperts — Trading Platform UX/UI Latest Trends: https://devexperts.com/blog/trading-platform-ux-ui-latest-trends/
