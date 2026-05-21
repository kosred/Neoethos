# mockups/

Static HTML design references for the neoethos trading UI.

## Status (V0.4.19)

**Authoritative visual reference** for the egui implementation under
`crates/neoethos-app/src/ui/`. The two HTML files in this directory describe
the target look-and-feel of the desktop GUI; the egui code mirrors them
panel-by-panel.

These files are **not** built or shipped — they are design artefacts
only. The actual UI is rendered by `neoethos-app` (egui/eframe) at runtime.

## Files

### `ui_mockup.html` (~285 KB)
Full-window design mock of the trading dashboard: chart panel, watchlist,
order ticket, execution surface, news feed, system tabs (Broker Setup,
Runtime, Intelligence, AI Helper, Data Bootstrap, Hardware, Risk
Settings, Settings), and the bottom action strip. Use as the source of
truth for colours (see `crate::ui::theme`), typography, spacing, and
panel borders.

### `tui_mockup.html` (~36 KB)
Stripped-down terminal-style variant intended as a future reference for
a headless / VPS-friendly TUI shell. The current `neoethos-cli` is a batch
tool — the TUI port (Task #41 in the V0.4 audit) is not yet implemented.

## Workflow

When changing the egui UI:

1. Pick the corresponding section of `ui_mockup.html` (or `tui_mockup.html`).
2. Match colours / spacing / typography from the mock.
3. The mock is **descriptive**, not prescriptive — if egui constraints
   (immediate-mode layout, no flexbox, etc.) force a divergence,
   document it in the panel's source comment and update the mock as a
   follow-up.

If the mock and the running app drift far enough that the mock is
misleading, **update the mock** (or delete it) — do not let stale
design references rot here. The `experiments/forex-flutter-ui/`
scaffold is a parked exploration of a Flutter port; it is **not** the
target UI and the mocks here are not meant to mirror it.
