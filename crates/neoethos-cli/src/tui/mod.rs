//! Interactive TUI for `neoethos-cli`.
//!
//! Launched when `neoethos-cli` is invoked without subcommand arguments,
//! the TUI provides a Bloomberg-style fixed-layout terminal:
//!
//! ```text
//! ┌── NeoEthos · TUI v0.2 ──────── 19:51:31 UTC · v0.2.0 ──┐
//! │ [ Dashboard ] Discover  Strategies  Symbols  Train  ⌘K │
//! │ ────────────────────────────────────────────────────── │
//! │  ┌─ Active jobs ─┐  ┌─ Last discovery ──┐  ┌─ Models ┐ │
//! │  │ ● discovery   │  │ EURUSD D1         │  │ 33 reg. │ │
//! │  │   12/42 work  │  │ 47 portfolio      │  │  3 prom │ │
//! │  └───────────────┘  └───────────────────┘  └─────────┘ │
//! └────────────────────────────────────────────────────────┘
//!   Tab: cycle pages · Enter: action · ⌘K: palette · q: quit
//! ```
//!
//! Each [`Page`] owns its own state + draw + key-handler. The
//! top-level [`App`] dispatches events to the active page and
//! re-renders on every tick (~30 fps, throttled).
//!
//! All long-running work (discovery, training, backtests) is
//! launched as a child process; the TUI just reads its log file
//! and renders progress. This keeps the TUI process responsive and
//! avoids re-implementing rayon/cubecl integration inside the
//! event loop.

mod app;
mod form;
mod jobs;
mod pages;
mod theme;
mod widgets;
pub mod wizard;

pub use app::run_tui;
pub use wizard::run_wizard_tui;
