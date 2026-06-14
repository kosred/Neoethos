//! Top-level TUI app state + event loop.
//!
//! `run_tui()` is the entry point — called from `main.rs` when
//! `neoethos-cli` is invoked with no subcommand. It sets up the
//! terminal, runs the render/event loop, and tears down on exit.

use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Padding, Paragraph, Widget};

use crate::tui::form::{FormState, make_discover_form, make_train_form};
use crate::tui::jobs::JobManager;
use crate::tui::pages::Page;
use crate::tui::theme;

/// State shared across all pages — dataset root, recent run
/// summaries, anything that needs to be read by multiple panels.
/// Pages own their own page-specific state separately.
/// A clickable hit-target the TUI publishes during render so the
/// mouse-event handler can dispatch the same intent the keyboard would.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub rect: Rect,
    pub action: HitAction,
}

#[derive(Debug, Clone, Copy)]
pub enum HitAction {
    /// Switch to the given page (top-bar tab click).
    GoToPage(Page),
    /// Trigger the page-local Enter handler (e.g. Discover/Train launch).
    Activate,
    /// Focus a specific form field on the given page (mouse click on row).
    /// Single click focuses; second click on the already-focused field
    /// starts editing. The Page argument lets us target the right
    /// form even if the click happens to be on a non-current page (it
    /// shouldn't, but defensive).
    FocusField { page: Page, index: usize },
}

/// A high-impact action staged behind a Y/N confirmation prompt. When
/// `AppShared::pending_confirmation` is `Some`, the next keypress is routed by
/// `app.rs`: `Y` runs the stored action, `N`/`Esc` cancels. Each page asks for
/// confirmation by setting this rather than acting immediately, so destructive
/// keys (save / import / stop) can't fire on a fat-finger.
#[derive(Debug, Clone)]
pub enum PendingAction {
    /// Save the Config form back to config.yaml (`config_view::S`).
    ConfigSave,
    /// Import data into the data/ layout (`symbols::I`).
    SymbolsImport,
    /// Stop the running discovery job (`discover::K`).
    DiscoverStop,
    /// Create the auto-loop stop flag (`auto_loop::K`).
    AutoLoopStop,
}

impl PendingAction {
    /// Human label shown in the "Confirm <label>?" prompt.
    pub fn label(&self) -> &'static str {
        match self {
            PendingAction::ConfigSave => "save config to config.yaml",
            PendingAction::SymbolsImport => "import data into data/",
            PendingAction::DiscoverStop => "stop the running discovery",
            PendingAction::AutoLoopStop => "create the auto-loop stop flag",
        }
    }

    /// Run the confirmed action against shared state.
    pub fn execute(self, shared: &mut AppShared) {
        match self {
            PendingAction::ConfigSave => crate::tui::pages::config_view::do_save(shared),
            PendingAction::SymbolsImport => crate::tui::pages::symbols::launch_import(shared),
            PendingAction::DiscoverStop => crate::tui::pages::discover::do_stop(shared),
            PendingAction::AutoLoopStop => crate::tui::pages::auto_loop::do_create_stop_flag(shared),
        }
    }
}

pub struct AppShared {
    pub data_root: PathBuf,
    pub build_version: &'static str,
    pub started_at: Instant,
    /// Last refresh time for the dataset inventory. Pages can compare
    /// against this to decide whether to re-read disk.
    pub last_refresh: Instant,
    /// Status line text — replaced by the most recent action / event.
    /// Always present so the bottom bar never goes blank.
    pub status: String,
    /// Background subprocess manager. Pages spawn `neoethos-cli`
    /// subprocesses through this — Discover/Train both use it.
    pub jobs: JobManager,
    /// Click hit-targets published by the most recent render. The mouse
    /// handler walks this list on every click to find what was hit.
    /// Cleared at the top of every frame and rebuilt by render code.
    pub hits: Vec<Hit>,
    /// Editable form for Discover page parameters.
    pub discover_form: FormState,
    /// Editable form for Train page parameters.
    pub train_form: FormState,
    /// Chart page state — selected symbol/timeframe + cached candles.
    pub chart_state: crate::tui::pages::chart::ChartState,
    /// Editable Config page form — loaded from the on-disk Settings, saved
    /// back to config.yaml so the user can change core settings from the TUI.
    pub config_form: FormState,
    /// Single-field form on the Symbols page: a source path to import data from
    /// (CSV/Parquet/Vortex/… → canonical data/ layout) without leaving the TUI.
    pub import_form: FormState,
    /// Selected row on the Strategies page (index into the scanned portfolio
    /// list), so the user can browse a portfolio's per-strategy metrics.
    pub strategies_selected: usize,
    /// Logs page scroll offset measured in lines UP from the tail (0 = follow
    /// the newest lines, the default).
    pub logs_scroll: usize,
    /// A high-impact action staged behind a Y/N confirmation prompt. When
    /// `Some`, the next keypress is routed to confirm/cancel (see
    /// `App::handle_key`) and a centered prompt is rendered over the page.
    pub pending_confirmation: Option<PendingAction>,
}

impl AppShared {
    fn new(data_root: PathBuf) -> Self {
        let root_str = data_root.display().to_string();
        let chart_state = crate::tui::pages::chart::ChartState::new(&data_root);
        Self {
            data_root,
            build_version: env!("CARGO_PKG_VERSION"),
            started_at: Instant::now(),
            last_refresh: Instant::now(),
            status: "Ready".to_string(),
            jobs: JobManager::new(),
            hits: Vec::new(),
            discover_form: make_discover_form(&root_str),
            train_form: make_train_form(&root_str),
            chart_state,
            config_form: crate::tui::pages::config_view::make_config_form(),
            import_form: FormState::new(vec![crate::tui::form::Field::new(
                "Import source",
                "",
                "Folder/file to import (CSV/TSV/JSON/Parquet/Vortex) → data/ layout",
            )]),
            strategies_selected: 0,
            logs_scroll: 0,
            pending_confirmation: None,
        }
    }

    /// Test (col, row) against the published hits, return the first
    /// matching action. The hits list is built by render code each
    /// frame, so this only sees what's currently on-screen.
    pub fn hit_test(&self, col: u16, row: u16) -> Option<HitAction> {
        self.hits
            .iter()
            .find(|h| {
                col >= h.rect.x
                    && col < h.rect.x + h.rect.width
                    && row >= h.rect.y
                    && row < h.rect.y + h.rect.height
            })
            .map(|h| h.action)
    }
}

/// The mutable TUI app: which page is active, the shared dataset
/// state, and a quit flag.
pub struct App {
    pub current: Page,
    pub shared: AppShared,
    pub quit: bool,
    /// When true, a help overlay listing every page + its keys is shown over
    /// the current page. Toggled with `?`; any key dismisses it.
    pub show_help: bool,
}

impl App {
    fn new(data_root: PathBuf) -> Self {
        Self {
            current: Page::Dashboard,
            shared: AppShared::new(data_root),
            quit: false,
            show_help: false,
        }
    }

    fn next_page(&mut self) {
        let pages = Page::ALL;
        let idx = pages.iter().position(|p| *p == self.current).unwrap_or(0);
        self.current = pages[(idx + 1) % pages.len()];
        self.shared.status = format!("Switched to {}", self.current.label());
    }

    fn prev_page(&mut self) {
        let pages = Page::ALL;
        let idx = pages.iter().position(|p| *p == self.current).unwrap_or(0);
        self.current = pages[(idx + pages.len() - 1) % pages.len()];
        self.shared.status = format!("Switched to {}", self.current.label());
    }

    fn handle_mouse(&mut self, ev: MouseEvent) {
        // We only care about left-button DOWN events — drags and scroll
        // wheel are no-ops for now. Hit-test against published click
        // targets and dispatch the matching action.
        if !matches!(ev.kind, MouseEventKind::Down(MouseButton::Left)) {
            return;
        }
        let action = match self.shared.hit_test(ev.column, ev.row) {
            Some(a) => a,
            None => return,
        };
        match action {
            HitAction::GoToPage(p) => {
                self.current = p;
                self.shared.status = format!("Switched to {}", p.label());
            }
            HitAction::Activate => {
                self.current.activate(&mut self.shared);
            }
            HitAction::FocusField { page, index } => {
                if page != self.current {
                    self.current = page;
                }
                let form = match page {
                    Page::Discover => &mut self.shared.discover_form,
                    Page::Train => &mut self.shared.train_form,
                    _ => return,
                };
                if form.focused == index {
                    // Second click on the already-focused field → edit.
                    form.start_editing();
                    self.shared.status = format!("Editing {}", form.fields[index].label);
                } else {
                    form.focus(index);
                    self.shared.status = format!("Focused {}", form.fields[index].label);
                }
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) {
        // Ctrl-C is always the kill-switch, regardless of edit state.
        if mods.contains(KeyModifiers::CONTROL) && matches!(code, KeyCode::Char('c')) {
            self.quit = true;
            return;
        }

        // Help overlay: while it's open, ANY key dismisses it.
        if self.show_help {
            self.show_help = false;
            return;
        }

        // Confirmation prompt: while one is staged, only Y/N/Esc are honored —
        // Y runs the action, N/Esc cancels. Every other key is swallowed so a
        // stray keystroke can't both dismiss the prompt AND do something else.
        if let Some(action) = self.shared.pending_confirmation.take() {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    let label = action.label();
                    action.execute(&mut self.shared);
                    // `execute` sets its own status; only override if it didn't.
                    if self.shared.status.is_empty() {
                        self.shared.status = format!("Confirmed: {label}");
                    }
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.shared.status = "Cancelled".to_string();
                }
                _ => {
                    // Unknown key: re-arm the prompt so it stays up until the
                    // user explicitly answers.
                    self.shared.pending_confirmation = Some(action);
                }
            }
            return;
        }

        // When ANY form is currently in edit mode, the page swallows
        // every key — otherwise typing 'q' would quit the app
        // mid-symbol-name. Esc breaks out of edit mode (handled by the
        // page); Tab still cycles fields (handled by the page);
        // outside edit mode the global shortcuts apply.
        let editing = self.shared.discover_form.editing
            || self.shared.train_form.editing
            || self.shared.config_form.editing
            || self.shared.import_form.editing;
        if editing {
            let _ = self.current.handle_key(code, &mut self.shared);
            return;
        }

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Tab => self.next_page(),
            KeyCode::BackTab => self.prev_page(),
            KeyCode::Char('1') => self.current = Page::Dashboard,
            KeyCode::Char('2') => self.current = Page::Discover,
            KeyCode::Char('3') => self.current = Page::Strategies,
            KeyCode::Char('4') => self.current = Page::Symbols,
            KeyCode::Char('5') => self.current = Page::Train,
            KeyCode::Char('6') => self.current = Page::Funnel,
            KeyCode::Char('7') => self.current = Page::AutoLoop,
            KeyCode::Char('8') => self.current = Page::Config,
            KeyCode::Char('9') => self.current = Page::Logs,
            KeyCode::Char('0') => self.current = Page::Chart,
            // Refresh: re-stamp last_refresh so the next render's
            // dataset summary is recomputed from disk and the status
            // bar shows "Refreshed Xs ago". The help text on every
            // page already promises this key works.
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.shared.last_refresh = Instant::now();
                // Re-read the editable config from disk too, so external edits
                // to config.yaml show up on the Config page after a refresh.
                self.shared.config_form =
                    crate::tui::pages::config_view::make_config_form();
                self.shared.status =
                    "Refreshed dataset inventory + config.".to_string();
            }
            other => {
                // Page-local: Up/Down focus, Enter to edit/launch, etc.
                let _ = self.current.handle_key(other, &mut self.shared);
            }
        }
    }
}

/// Run the TUI until the user quits. Returns when the user presses
/// `q` / `Esc` / `Ctrl-C`.
pub fn run_tui(data_root: Option<PathBuf>) -> Result<()> {
    let data_root = data_root.unwrap_or_else(|| PathBuf::from("data"));

    // Terminal setup.
    enable_raw_mode().context("enable raw terminal mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("init terminal backend")?;
    // DOCUMENTED-DEFAULT: best-effort clear; failure here would also break
    // the subsequent draw loop and surface there.
    terminal.clear().ok();

    let mut app = App::new(data_root);
    let res = event_loop(&mut terminal, &mut app);

    // Terminal teardown — always runs, even if event_loop bailed. These
    // are documented best-effort cleanups: at this point the program is
    // exiting, so the only thing we could do with an error is print it,
    // which would corrupt the now-restored terminal. Leave silent.
    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
    )
    .ok();
    terminal.show_cursor().ok();

    res
}

fn event_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<()> {
    let tick = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    loop {
        terminal
            .draw(|f| {
                // Reset the click-hit list at the start of every frame
                // so render code can rebuild it. Done here (and not in
                // `render`) so the borrow against `app` stays clean.
                app.shared.hits.clear();
                render(f.area(), f.buffer_mut(), app);
            })
            .context("render frame")?;
        if app.quit {
            return Ok(());
        }

        let timeout = tick
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_millis(0));

        if event::poll(timeout).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    if key.kind == KeyEventKind::Press {
                        app.handle_key(key.code, key.modifiers);
                    }
                }
                Ok(Event::Mouse(m)) => app.handle_mouse(m),
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick {
            last_tick = Instant::now();
            // Drain any new lines from running subprocesses into their
            // ring buffers so the next render sees fresh log output.
            app.shared.jobs.tick();
        }
    }
}

/// Layout: 3 rows — top bar (3 lines) · main area (rest) · status (1 line).
fn render(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &mut App) {
    // Fill the entire viewport with the app background so it does
    // not show terminal default. This is harmless in raw mode.
    let bg = Block::default().style(Style::default().bg(theme::APP_BG));
    bg.render(area, buf);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // top bar
            Constraint::Min(0),    // main
            Constraint::Length(1), // status
        ])
        .split(area);

    render_top_bar(rows[0], buf, app);
    app.current.draw(rows[1], buf, &mut app.shared);
    render_status_bar(rows[2], buf, app);
    if app.show_help {
        render_help_overlay(rows[1], buf);
    }
    if let Some(action) = &app.shared.pending_confirmation {
        render_confirm_overlay(rows[1], buf, action.label());
    }
}

/// Centered Y/N confirmation prompt rendered over the current page — reuses the
/// help-overlay's bordered, accent-titled style so it reads as a modal. The
/// next keypress is routed by `App::handle_key` (Y runs, N/Esc cancels).
fn render_confirm_overlay(area: Rect, buf: &mut ratatui::buffer::Buffer, label: &str) {
    let title = format!("Confirm {label}?");
    let w = ((title.chars().count() as u16) + 8)
        .max(34)
        .min(area.width);
    let h = 5u16.min(area.height);
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let popup = Rect {
        x,
        y,
        width: w,
        height: h,
    };
    Clear.render(popup, buf);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ACCENT))
        .title(Span::styled(
            " CONFIRM ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(theme::SURFACE_ALT))
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(popup);
    block.render(popup, buf);
    let lines = vec![
        Line::from(vec![Span::styled(
            title,
            theme::accent_style().add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            "[Y]es   ·   [N]o / Esc",
            theme::muted_style(),
        )]),
    ];
    Paragraph::new(lines).render(inner, buf);
}

/// Centered keyboard-help overlay: every page's keys at a glance, so the user
/// never has to guess. Toggled with `?`, dismissed by any key.
fn render_help_overlay(area: Rect, buf: &mut ratatui::buffer::Buffer) {
    let w = ((area.width as f32 * 0.72) as u16).clamp(40, area.width);
    let h = (Page::ALL.len() as u16 + 6).min(area.height);
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let popup = Rect {
        x,
        y,
        width: w,
        height: h,
    };
    Clear.render(popup, buf);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ACCENT))
        .title(Span::styled(
            " KEYBOARD HELP — press any key to close ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(theme::SURFACE_ALT))
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(popup);
    block.render(popup, buf);

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                "Global  ",
                theme::accent_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Tab/Shift-Tab pages · 1-0 jump · R refresh · ? help · Q / Ctrl-C quit",
                theme::muted_style(),
            ),
        ]),
        Line::raw(""),
    ];
    for p in Page::ALL {
        let hints: Vec<String> = p
            .key_hints()
            .iter()
            .map(|(k, a)| format!("{k} {a}"))
            .collect();
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<10}", p.label()),
                theme::accent_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(hints.join("  ·  "), theme::muted_style()),
        ]));
    }
    Paragraph::new(lines).render(inner, buf);
}

fn render_top_bar(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &mut App) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(theme::BORDER_STRONG))
        .style(theme::panel_block_style());
    let inner = block.inner(area);
    block.render(area, buf);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // brand
            Constraint::Min(0),     // page tabs
            Constraint::Length(28), // clock + version
        ])
        .split(inner);

    // Brand
    let brand = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            " NeoEthos ",
            Style::default()
                .bg(theme::ACCENT)
                .fg(theme::APP_BG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled("TUI", theme::caption_style().add_modifier(Modifier::BOLD)),
    ])])
    .style(theme::panel_block_style());
    brand.render(cols[0], buf);

    // Page tabs — published as click hits so mouse-clicking a tab
    // switches to it. We track each tab's pixel column range as we
    // build the spans so the hit rectangles match exactly.
    let mut spans: Vec<Span> = vec![];
    let tab_row = cols[1].y;
    let mut cursor_x = cols[1].x;
    for (i, p) in Page::ALL.iter().enumerate() {
        let style = if *p == app.current {
            theme::nav_active_style()
        } else {
            theme::nav_inactive_style()
        };
        let label = format!(" {} ", p.label());
        let label_w = label.chars().count() as u16;
        if cursor_x + label_w <= cols[1].x + cols[1].width {
            app.shared.hits.push(Hit {
                rect: Rect {
                    x: cursor_x,
                    y: tab_row,
                    width: label_w,
                    height: cols[1].height.max(1),
                },
                action: HitAction::GoToPage(*p),
            });
        }
        spans.push(Span::styled(label, style));
        cursor_x += label_w;
        if i + 1 < Page::ALL.len() {
            let sep = " · ";
            spans.push(Span::styled(sep, theme::caption_style()));
            cursor_x += sep.chars().count() as u16;
        }
    }
    let tabs = Paragraph::new(Line::from(spans)).style(theme::panel_block_style());
    tabs.render(cols[1], buf);

    // Clock + version
    let clock = utc_clock();
    let right = Paragraph::new(Line::from(vec![
        Span::styled(clock, theme::muted_style()),
        Span::raw("  "),
        Span::styled(
            format!("v{} ", app.shared.build_version),
            theme::caption_style(),
        ),
    ]))
    .style(theme::panel_block_style())
    .alignment(ratatui::layout::Alignment::Right);
    right.render(cols[2], buf);
}

fn render_status_bar(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    // (status bar is read-only: takes &App not &mut)
    let block = Block::default().style(Style::default().bg(theme::PANEL_BG).fg(theme::TEXT_MUTED));
    let inner = block.inner(area);
    block.render(area, buf);

    let mut hints: Vec<Span> = Vec::new();
    for (i, (key, action)) in app.current.key_hints().iter().enumerate() {
        if i > 0 {
            hints.push(Span::styled("  ·  ", theme::caption_style()));
        }
        hints.push(Span::styled(
            format!(" {} ", key),
            Style::default()
                .bg(theme::SURFACE_ALT)
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ));
        hints.push(Span::raw(" "));
        hints.push(Span::styled(*action, theme::muted_style()));
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(40)])
        .split(inner);

    Paragraph::new(Line::from(hints)).render(cols[0], buf);

    // Right column: status text on top, "refreshed Xs ago" below so the
    // operator can see how stale the dashboard's dataset summary is
    // without scrolling. Updates when the user presses R.
    let refreshed = app.shared.last_refresh.elapsed().as_secs();
    let right = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            app.shared.status.clone(),
            theme::muted_style(),
        )]),
        Line::from(vec![Span::styled(
            format!("refreshed {}s ago", refreshed),
            theme::caption_style(),
        )]),
    ])
    .alignment(ratatui::layout::Alignment::Right);
    right.render(cols[1], buf);
}

fn utc_clock() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let day = secs % 86_400;
    format!(
        "{:02}:{:02}:{:02} UTC",
        day / 3600,
        (day % 3600) / 60,
        day % 60
    )
}
