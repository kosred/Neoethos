# Forex App Modern UI Foundation Design

## Goal

Upgrade `crates/forex-app` from a functionally correct operator shell into a visually intentional, modern, cross-platform desktop terminal foundation without destabilizing the backend service layer.

The immediate target is not full TradingView-like charting. The immediate target is a strong visual and structural base:

- a consistent design system for cards, panels, badges, spacing, and colors
- a more professional operator shell across `Trading`, `Discovery`, `Training`, and `System`
- a cross-platform-safe UI style that remains compatible with Windows and macOS
- a packaging-aware foundation that avoids development-path assumptions

## Scope

This design covers:

- `crates/forex-app`
- app-wide theme and visual tokens
- shared UI primitives for dashboards and operator panels
- top-level shell polish in `main.rs`
- the first pass of modernizing the current operator tabs

This design does not yet cover:

- charting workspace
- dockable multi-window layouts
- manual order ticket
- news/calendar aggregation
- real `cTrader` or `DXtrade` connectivity
- installer generation or code signing

## Problem Statement

The app now has a materially better backend contract, but the UI still looks like an internal tool rather than a professional desktop terminal:

- repeated dashboard rendering logic is duplicated across tabs
- there is no shared visual language for cards, sections, badges, or shell surfaces
- the top navigation and panels are functional but visually flat
- styling is mostly default `egui`, which is serviceable but not product-grade

This creates three risks:

1. the UI looks less mature than the backend
2. future screens will drift visually because there is no design system
3. later charting and broker panels will compound duplication instead of building on reusable primitives

## Design Summary

The first modernization tranche introduces a small app-owned design system inside `crates/forex-app/src/ui`.

This design system owns:

- theme application
- palette and spacing tokens
- reusable operator card and section renderers
- consistent status and severity visuals
- shell-level layout polish for the top nav and main panels

The app keeps `egui/eframe`. We do not switch UI frameworks.

## Architectural Approach

Recommended approach:

- keep `eframe` and `egui`
- add a focused `ui/theme.rs` module
- consolidate shared dashboard primitives in `ui/components.rs`
- reuse the existing app-service layer and only improve the visual shell around it

This preserves the verified runtime contracts and moves the visual system into small reusable files rather than growing `main.rs` again.

## Visual Principles

The app should feel like a modern operator terminal:

- dark, high-contrast, low-noise background
- restrained but intentional accent color
- clear distinction between passive information, warnings, and failures
- cards and sections with consistent spacing and corner treatment
- no emoji-dependent visual hierarchy
- status and severity should be readable even before opening logs

The UI should feel intentionally designed, not merely customized defaults.

## Cross-Platform Constraints

The chosen visual system must remain safe for:

- Windows
- macOS
- Linux

This means:

- no platform-specific font dependencies in the supported path
- no hardcoded Windows-only rendering assumptions
- no external runtime assets required for the first modernization tranche

The app should remain bundle-friendly for future offline packaging:

- Windows MSI or packaged EXE
- macOS `.app`/DMG
- Linux AppImage and optional `.deb`/`.rpm`

The design system must not assume a development checkout path or internet access.

## File Structure Recommendation

Recommended new or modified ownership:

- `crates/forex-app/src/ui/theme.rs`
  - app-wide theme application
  - color/spacing tokens
  - shared frames and status helpers

- `crates/forex-app/src/ui/components.rs`
  - shared dashboard card and section types
  - shared renderers for cards, detail sections, report blocks, and open-log action

- `crates/forex-app/src/ui/discovery.rs`
  - discovery-specific data grouping only
  - reuse shared card/section rendering

- `crates/forex-app/src/ui/training.rs`
  - training-specific data grouping only
  - reuse shared card/section rendering

- `crates/forex-app/src/ui/trading.rs`
  - adapter-specific operator content only
  - reuse shared card/section rendering

- `crates/forex-app/src/ui/system_status.rs`
  - system-specific summaries only
  - reuse shared card/section rendering

- `crates/forex-app/src/main.rs`
  - apply theme once during app startup
  - keep shell wiring, not bespoke visual logic

## Verification Requirements

This tranche should be considered complete only if:

- `cargo test -p forex-app -- --nocapture` passes
- `cargo clippy -p forex-app --all-targets -- -D warnings` passes
- `cargo test --workspace` passes
- `cargo clippy --workspace --all-targets -- -D warnings` passes
- `target/debug/forex-app.exe --headless --local --config config.yaml` still succeeds

The modernization tranche must not regress the current app service, logging, or broker adapter seams.
