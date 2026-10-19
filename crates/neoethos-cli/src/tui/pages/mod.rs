//! Page enum + dispatch. Each variant of [`Page`] maps to a module
//! with `draw(...)` + `handle_key(...)` functions. Adding a new page
//! is: add an enum variant, add a module, route in both functions.

use crossterm::event::KeyCode;
use ratatui::{buffer::Buffer, layout::Rect};

use crate::tui::app::AppShared;
use crate::tui::form::FormState;

/// A normal form key either mutates the fields or leaves one page-specific
/// action for its owner. Editing mode never leaks keystrokes to Launch/Stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FormKeyAction {
    Handled,
    Launch,
    Stop,
    Ignored,
}

pub(super) fn handle_form_key(
    code: KeyCode,
    form: &mut FormState,
    status: &mut String,
) -> FormKeyAction {
    if form.editing {
        return if handle_editing_form_key(code, form, status) {
            FormKeyAction::Handled
        } else {
            FormKeyAction::Ignored
        };
    }
    match code {
        KeyCode::Up | KeyCode::Char('k') => {
            form.focus_prev();
            FormKeyAction::Handled
        }
        KeyCode::Down | KeyCode::Char('j') => {
            form.focus_next();
            FormKeyAction::Handled
        }
        KeyCode::Enter => {
            let label = form.fields[form.focused].label;
            form.start_editing();
            *status = format!("Editing {} (Esc to cancel)", label);
            FormKeyAction::Handled
        }
        KeyCode::Char('x') => {
            form.clear_focused();
            *status = "Field cleared".to_string();
            FormKeyAction::Handled
        }
        KeyCode::Char('l') | KeyCode::Char('L') => FormKeyAction::Launch,
        KeyCode::Char('K') => FormKeyAction::Stop,
        _ => FormKeyAction::Ignored,
    }
}

/// Shared editing-mode handler for the existing Discover and Train form pages.
/// Page-specific launch/stop commands remain in each page's normal key handler.
pub(super) fn handle_editing_form_key(
    code: KeyCode,
    form: &mut FormState,
    status: &mut String,
) -> bool {
    match code {
        KeyCode::Esc => {
            form.stop_editing(false);
            *status = "Edit cancelled".to_string();
            true
        }
        KeyCode::Enter => {
            form.stop_editing(true);
            *status = format!(
                "Saved {} = {}",
                form.fields[form.focused].label,
                form.fields[form.focused].effective()
            );
            true
        }
        KeyCode::Backspace => {
            form.backspace();
            true
        }
        KeyCode::Char(c) => {
            form.type_char(c);
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod shared_editing_tests {
    use super::*;
    use crate::tui::form::Field;

    #[test]
    fn navigation_and_launch_are_separate_from_editing() {
        let mut form = FormState::new(vec![
            Field::new("First", "1", "first"),
            Field::new("Second", "2", "second"),
        ]);
        let mut status = String::new();
        assert_eq!(
            handle_form_key(KeyCode::Down, &mut form, &mut status),
            FormKeyAction::Handled
        );
        assert_eq!(form.focused, 1);
        assert_eq!(
            handle_form_key(KeyCode::Char('l'), &mut form, &mut status),
            FormKeyAction::Launch
        );
        assert_eq!(
            handle_form_key(KeyCode::Char('K'), &mut form, &mut status),
            FormKeyAction::Stop
        );
        assert_eq!(
            handle_form_key(KeyCode::Enter, &mut form, &mut status),
            FormKeyAction::Handled
        );
        assert!(form.editing);
        assert_eq!(
            handle_form_key(KeyCode::Char('l'), &mut form, &mut status),
            FormKeyAction::Handled
        );
        assert_eq!(form.fields[1].effective(), "2l");
    }

    #[test]
    fn editing_keystrokes_preserve_save_and_cancel_contract() {
        let mut form = FormState::new(vec![Field::new("Name", "base", "hint")]);
        let mut status = String::new();

        form.start_editing();
        assert!(handle_editing_form_key(
            KeyCode::Char('!'),
            &mut form,
            &mut status
        ));
        assert!(handle_editing_form_key(
            KeyCode::Esc,
            &mut form,
            &mut status
        ));
        assert_eq!(form.fields[0].effective(), "base");
        assert_eq!(status, "Edit cancelled");
        assert!(!form.editing);

        form.start_editing();
        assert!(handle_editing_form_key(
            KeyCode::Char('!'),
            &mut form,
            &mut status
        ));
        assert!(handle_editing_form_key(
            KeyCode::Backspace,
            &mut form,
            &mut status
        ));
        assert!(handle_editing_form_key(
            KeyCode::Char('?'),
            &mut form,
            &mut status
        ));
        assert!(!handle_editing_form_key(
            KeyCode::Up,
            &mut form,
            &mut status
        ));
        assert!(handle_editing_form_key(
            KeyCode::Enter,
            &mut form,
            &mut status
        ));
        assert_eq!(form.fields[0].effective(), "base?");
        assert_eq!(status, "Saved Name = base?");
        assert!(!form.editing);
    }
}

pub mod auto_loop;
pub mod chart;
pub mod config_view;
pub mod dashboard;
pub mod discover;
pub mod funnel;
pub mod logs;
pub mod native_research;
pub mod strategies;
pub mod symbols;
pub mod train;

/// Top-level pages of the TUI. The order here is the order they
/// appear in the top nav bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Dashboard,
    Discover,
    NativeResearch,
    Strategies,
    Symbols,
    Train,
    Funnel,
    AutoLoop,
    Config,
    Logs,
    Chart,
}

impl Page {
    pub const ALL: &'static [Page] = &[
        Page::Dashboard,
        Page::Discover,
        Page::NativeResearch,
        Page::Strategies,
        Page::Symbols,
        Page::Train,
        Page::Funnel,
        Page::AutoLoop,
        Page::Config,
        Page::Logs,
        Page::Chart,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Page::Dashboard => "Dashboard",
            Page::Discover => "Discover",
            Page::NativeResearch => "Native Research",
            Page::Strategies => "Strategies",
            Page::Symbols => "Symbols",
            Page::Train => "Train",
            Page::Funnel => "Funnel",
            Page::AutoLoop => "AutoLoop",
            Page::Config => "Config",
            Page::Logs => "Logs",
            Page::Chart => "Chart",
        }
    }

    /// Bottom-of-screen keyboard hints, contextual to the page.
    pub fn key_hints(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Page::Dashboard => &[
                ("?", "help"),
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
            Page::NativeResearch => &[
                ("↑↓", "focus field"),
                ("Enter", "edit"),
                ("L", "start"),
                ("S", "status"),
                ("K", "cancel"),
                ("N", "open page"),
            ],
            Page::Strategies => &[
                ("↑↓", "select"),
                ("V", "validate"),
                ("P", "promote"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Symbols => &[
                ("PgUp/PgDn", "inventory"),
                ("Home", "top"),
                ("E", "edit source"),
                ("I", "import data"),
                ("Tab", "page"),
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
                ("↑↓", "run"),
                ("PgUp/PgDn", "scroll"),
                ("Home", "top"),
                ("Q", "quit"),
            ],
            Page::AutoLoop => &[
                ("Tab", "page"),
                ("L", "launch"),
                ("K", "stop"),
                ("Q", "quit"),
            ],
            Page::Config => &[
                ("↑↓", "field"),
                ("Enter", "edit"),
                ("S", "save"),
                ("R", "reload"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Logs => &[
                ("↑↓", "scroll"),
                ("F", "follow tail"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
            Page::Chart => &[
                ("←→", "symbol"),
                ("↑↓", "timeframe"),
                ("Tab", "page"),
                ("Q", "quit"),
            ],
        }
    }

    pub fn draw(self, area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
        match self {
            Page::Dashboard => dashboard::draw(area, buf, shared),
            Page::Discover => discover::draw(area, buf, shared),
            Page::NativeResearch => native_research::draw(area, buf, shared),
            Page::Strategies => strategies::draw(area, buf, shared),
            Page::Symbols => symbols::draw(area, buf, shared),
            Page::Train => train::draw(area, buf, shared),
            Page::Funnel => funnel::draw(area, buf, shared),
            Page::AutoLoop => auto_loop::draw(area, buf, shared),
            Page::Config => config_view::draw(area, buf, shared),
            Page::Logs => logs::draw(area, buf, shared),
            Page::Chart => chart::draw(area, buf, shared),
        }
    }

    /// Page-local key handler — returns `true` if the page consumed
    /// the key.
    pub fn handle_key(self, code: KeyCode, shared: &mut AppShared) -> bool {
        match self {
            Page::Discover => discover::handle_key(code, shared),
            Page::NativeResearch => native_research::handle_key(code, shared),
            Page::Train => train::handle_key(code, shared),
            Page::AutoLoop => auto_loop::handle_key(code, shared),
            Page::Chart => chart::handle_key(code, shared),
            Page::Config => config_view::handle_key(code, shared),
            Page::Symbols => symbols::handle_key(code, shared),
            Page::Strategies => strategies::handle_key(code, shared),
            Page::Funnel => funnel::handle_key(code, shared),
            Page::Logs => logs::handle_key(code, shared),
            _ => false,
        }
    }

    /// "Activate" — fired by the mouse-click hit-tester when the user
    /// clicks the Launch pill. Distinct from Enter (which starts
    /// editing a focused form field).
    pub fn activate(self, shared: &mut AppShared) {
        match self {
            Page::Discover => discover::launch_now(shared),
            Page::NativeResearch => native_research::launch_now(shared),
            Page::Train => train::launch_now(shared),
            Page::AutoLoop => auto_loop::launch_now(shared),
            _ => {}
        }
    }
}
