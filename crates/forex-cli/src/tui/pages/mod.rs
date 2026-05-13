//! Page enum + dispatch. Each variant of [`Page`] maps to a module
//! with `draw(...)` + `handle_key(...)` functions. Adding a new page
//! is: add an enum variant, add a module, route in both functions.

use crossterm::event::KeyCode;
use ratatui::{buffer::Buffer, layout::Rect};

use crate::tui::app::AppShared;

pub mod auto_loop;
pub mod config_view;
pub mod dashboard;
pub mod discover;
pub mod funnel;
pub mod logs;
pub mod strategies;
pub mod symbols;
pub mod train;

/// Top-level pages of the TUI. The order here is the order they
/// appear in the top nav bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Dashboard,
    Discover,
    Strategies,
    Symbols,
    Train,
    Funnel,
    AutoLoop,
    Config,
    Logs,
}

impl Page {
    pub const ALL: &'static [Page] = &[
        Page::Dashboard,
        Page::Discover,
        Page::Strategies,
        Page::Symbols,
        Page::Train,
        Page::Funnel,
        Page::AutoLoop,
        Page::Config,
        Page::Logs,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Page::Dashboard => "Dashboard",
            Page::Discover => "Discover",
            Page::Strategies => "Strategies",
            Page::Symbols => "Symbols",
            Page::Train => "Train",
            Page::Funnel => "Funnel",
            Page::AutoLoop => "AutoLoop",
            Page::Config => "Config",
            Page::Logs => "Logs",
        }
    }

    /// Bottom-of-screen keyboard hints, contextual to the page.
    pub fn key_hints(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Page::Dashboard => &[
                ("Tab", "next page"),
                ("R", "refresh"),
                ("Q", "quit"),
            ],
            Page::Discover => &[
                ("↑↓", "focus field"),
                ("Enter", "edit"),
                ("Esc", "cancel"),
                ("L", "launch"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Strategies => &[
                ("Tab", "next page"),
                ("↑↓", "select"),
                ("/", "filter"),
                ("Enter", "details"),
                ("Q", "quit"),
            ],
            Page::Symbols => &[
                ("Tab", "next page"),
                ("↑↓", "select"),
                ("Q", "quit"),
            ],
            Page::Train => &[
                ("↑↓", "focus field"),
                ("Enter", "edit"),
                ("Esc", "cancel"),
                ("L", "launch"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Funnel => &[
                ("Tab", "page"),
                ("R", "refresh"),
                ("Q", "quit"),
            ],
            Page::AutoLoop => &[
                ("Tab", "page"),
                ("L", "launch"),
                ("Q", "quit"),
            ],
            Page::Config => &[
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Logs => &[
                ("Tab", "page"),
                ("Q", "quit"),
            ],
        }
    }

    pub fn draw(self, area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
        match self {
            Page::Dashboard => dashboard::draw(area, buf, shared),
            Page::Discover => discover::draw(area, buf, shared),
            Page::Strategies => strategies::draw(area, buf, shared),
            Page::Symbols => symbols::draw(area, buf, shared),
            Page::Train => train::draw(area, buf, shared),
            Page::Funnel => funnel::draw(area, buf, shared),
            Page::AutoLoop => auto_loop::draw(area, buf, shared),
            Page::Config => config_view::draw(area, buf, shared),
            Page::Logs => logs::draw(area, buf, shared),
        }
    }

    /// Page-local key handler — returns `true` if the page consumed
    /// the key.
    pub fn handle_key(self, code: KeyCode, shared: &mut AppShared) -> bool {
        match self {
            Page::Discover => discover::handle_key(code, shared),
            Page::Train => train::handle_key(code, shared),
            Page::AutoLoop => auto_loop::handle_key(code, shared),
            _ => false,
        }
    }

    /// "Activate" — fired by the mouse-click hit-tester when the user
    /// clicks the Launch pill. Distinct from Enter (which starts
    /// editing a focused form field).
    pub fn activate(self, shared: &mut AppShared) {
        match self {
            Page::Discover => discover::launch_now(shared),
            Page::Train => train::launch_now(shared),
            Page::AutoLoop => auto_loop::launch_now(shared),
            _ => {}
        }
    }
}
