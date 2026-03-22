# Forex App Dockable Workspace Shell Design

## Goal

Replace the current top-tab shell in `crates/forex-app` with a dockable operator workspace built around the approved `A · Chart-first rail` direction.

The immediate target is not a full charting engine. The target is a correct workspace shell:

- dockable tabs and panels
- a composite `Charts + Trading` workspace
- preserved backend service boundaries
- a layout that already feels like a trading terminal instead of a debug utility

## Scope

This design covers:

- `crates/forex-app`
- dockable workspace orchestration
- the first `Charts + Trading` workspace shell
- integration of existing `Discovery`, `Training`, and `System` views into the dockable shell

This design does not yet cover:

- full chart engine implementation
- drawing tools
- live news/calendar ingestion
- workspace layout persistence
- floating detached windows beyond what the dock library gives us naturally
- multi-broker login UI expansion

## Product Direction

The approved visual direction is the chart-first rail layout:

- left rail for watchlist and broker status
- center chart surface as the dominant panel
- right rail for execution and news
- bottom strip for positions, orders, bot timeline, and notes

This means the app no longer thinks in terms of one `Trading` page. It thinks in terms of a composite trading workspace made from several smaller panels.

## Architectural Approach

Recommended approach:

- keep `egui/eframe`
- add `egui_dock` for workspace composition
- keep business logic in app services
- let the docking layer own only:
  - workspace layout
  - active tabs
  - panel routing
  - panel visibility

The workspace layer must not gain discovery, training, broker, or chart business logic.

## Workspace Model

The first dockable workspace should include these tabs:

- `Chart`
- `Watchlist`
- `Execution`
- `News`
- `Bottom Strip`
- `Discovery`
- `Training`
- `System`

This is a shell-level composition, not an instruction to fully implement all product behavior inside every panel on day one.

## First Tranche Scope

The first implementation tranche should deliver:

- real `egui_dock` shell integration
- default layout matching the approved chart-first rail concept
- routing of existing panels into the dockable shell
- new placeholder-first panels for:
  - `Chart`
  - `Watchlist`
  - `Execution`
  - `News`
  - `Bottom Strip`

Existing verified operator views should remain intact and get embedded into the docking shell:

- `Discovery`
- `Training`
- `System`

## File Structure Recommendation

Recommended ownership:

- `crates/forex-app/src/workspace/mod.rs`
  - public workspace exports

- `crates/forex-app/src/workspace/tabs.rs`
  - workspace tab enum
  - tab labels
  - tab categories

- `crates/forex-app/src/workspace/layout.rs`
  - default dock layout
  - workspace state container

- `crates/forex-app/src/workspace/viewer.rs`
  - `egui_dock::TabViewer` implementation
  - routes each tab to the correct renderer

- `crates/forex-app/src/ui/trading/chart_panel.rs`
  - chart surface shell
  - timeframe bar
  - bot marker placeholders

- `crates/forex-app/src/ui/trading/watchlist_panel.rs`
  - symbol list
  - runtime summary

- `crates/forex-app/src/ui/trading/execution_panel.rs`
  - order actions
  - risk summary

- `crates/forex-app/src/ui/trading/news_panel.rs`
  - news/event rail shell

- `crates/forex-app/src/ui/trading/bottom_strip.rs`
  - positions/orders/timeline/notes shell

- `crates/forex-app/src/main.rs`
  - remove top-tab page ownership
  - host the dockable workspace shell

- `crates/forex-app/src/app_state.rs`
  - add workspace state handle
  - stop treating `current_tab` as the main app navigation primitive

## Verification Requirements

The dockable workspace tranche is complete only if:

- `cargo test -p forex-app -- --nocapture` passes
- `cargo clippy -p forex-app --all-targets -- -D warnings` passes
- `cargo test --workspace` passes
- `cargo clippy --workspace --all-targets -- -D warnings` passes
- `target/debug/forex-app.exe --headless --local --config config.yaml` still succeeds

The docking shell must not regress:

- canonical logging
- discovery/training service contracts
- current broker adapter seams
