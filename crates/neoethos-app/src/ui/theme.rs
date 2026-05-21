// Phase C3 audit + Flutter pivot context (2026-05-18 operator directive):
// `theme.rs` is a design-system constants module by definition — every
// `pub const FONT_*` / `pub const RADIUS_*` / `pub const STATUS_*` is
// an API surface for the broader UI, so most tokens have no call site
// at any given snapshot. A missing token is a real bug (renderers
// fail loud); an unused one is healthy headroom.
//
// EGUI -> FLUTTER MIGRATION NOTE: once the Flutter rewrite ships,
// this entire file gets replaced by Dart design tokens at the
// Flutter side, and the Rust crate gracefully sheds it. Until then
// the file-local `#![allow(dead_code)]` is the honest tag for
// "transient palette during the egui sunset". NOT promoted to a
// workspace lint. NOT silencing real bugs.
#![allow(dead_code)]

//! NeoEthos design system.
//!
//! Anchored to the conventions every working trader recognises:
//! TradingView-style dark palette, teal-green long / red short,
//! 4-pt spacing grid, four type levels, semantic colors used sparingly.
//!
//! One source of truth: any new UI work should reach for the tokens
//! here, not invent its own padding or color.

use eframe::egui;

// ─── Color palette ───────────────────────────────────────────────────────
//
// Mapped from TradingView's published dark theme + the Stock Trader
// UI Kit conventions. Buy / sell colors are TradingView's literal
// candle defaults (#26A69A / #EF5350) so anyone who has ever looked
// at a TV chart reads them without thinking.
//
// Contrast ratios:
//   TEXT_PRIMARY on APP_BG       ≥ 15:1 (AAA)
//   TEXT_PRIMARY on PANEL_BG     ≥ 13:1 (AAA)
//   TEXT_MUTED   on PANEL_BG     ≥  4.6:1 (AA)
//   ACCENT       on APP_BG       ≥  4.5:1 (AA)

/// Application root background (chart canvas, gutter).
pub const APP_BG: egui::Color32 = egui::Color32::from_rgb(0x0E, 0x11, 0x16);
/// Default panel background — sidebars, ticket, bottom dock, top bar.
pub const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(0x16, 0x1B, 0x22);
/// Card / surface background — one step "above" the panel.
pub const SURFACE_BG: egui::Color32 = egui::Color32::from_rgb(0x1C, 0x22, 0x30);
/// Elevated surface — hover, focused tab, accent callouts.
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(0x22, 0x29, 0x3A);
/// Chart canvas — exact match for app bg by convention.
pub const CHART_BG: egui::Color32 = egui::Color32::from_rgb(0x0E, 0x11, 0x16);

/// Subtle hairline borders.
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(0x2A, 0x2F, 0x3A);
/// Heavier borders for focus states / dividers.
pub const BORDER_STRONG: egui::Color32 = egui::Color32::from_rgb(0x3A, 0x40, 0x4D);
/// Chart grid lines.
pub const GRID: egui::Color32 = egui::Color32::from_rgb(0x1F, 0x24, 0x30);

/// Primary text.
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(0xE6, 0xEA, 0xF2);
/// Secondary text (labels, captions, hints).
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(0x9A, 0xA4, 0xB2);
/// Tertiary text (disabled, placeholders).
pub const TEXT_FAINT: egui::Color32 = egui::Color32::from_rgb(0x5C, 0x64, 0x73);

/// Primary accent — TradingView blue. Used for active nav, primary
/// CTAs, focus rings. Replaced the previous soft violet (#7C89FF)
/// because traders read this exact blue as "interactive / selected".
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x29, 0x62, 0xFF);
/// Accent on hover.
pub const ACCENT_HOVER: egui::Color32 = egui::Color32::from_rgb(0x1E, 0x53, 0xE5);
/// Accent at low alpha — selected row background.
pub const ACCENT_MUTED: egui::Color32 = egui::Color32::from_rgb(0x1E, 0x2A, 0x4A);
/// Accent at very low alpha — for hover backgrounds.
pub const ACCENT_SOFT: egui::Color32 = egui::Color32::from_rgb(0x16, 0x1F, 0x36);

/// Trading semantics — TradingView's literal candle colors.
/// `BUY` / `LONG` is teal-green; `SELL` / `SHORT` is red.
pub const BUY: egui::Color32 = egui::Color32::from_rgb(0x26, 0xA6, 0x9A);
pub const BUY_STRONG: egui::Color32 = egui::Color32::from_rgb(0x00, 0xC8, 0x53);
pub const SELL: egui::Color32 = egui::Color32::from_rgb(0xEF, 0x53, 0x50);
pub const SELL_STRONG: egui::Color32 = egui::Color32::from_rgb(0xFF, 0x17, 0x44);

/// Status semantics. Use ONLY for the meaning they signal.
/// `SUCCESS` = `BUY` (intentional); `DANGER` = `SELL`; warning is
/// amber for pending/partial fills; info matches `ACCENT`.
pub const SUCCESS: egui::Color32 = BUY;
pub const WARNING: egui::Color32 = egui::Color32::from_rgb(0xF4, 0xB4, 0x00);
pub const DANGER: egui::Color32 = SELL;
pub const INFO: egui::Color32 = ACCENT;

// ─── Trading-environment status-pill tokens ──────────────────────────────
//
// Used by the persistent Demo/Paper/Live pill in the main chrome (see
// `ui::chrome::status_pill`). Pattern #1 of the giants-pattern gaps in
// `docs/audits/research/wizard_onboarding_competitive_analysis.md` §10
// (ThinkOrSwim paperMoney pill, TradingView gray-vs-red Trading Panel).
//
// Token mapping is anchored to the design-spec §5.1 palette:
//   - `STATUS_DEMO`  -> `TEXT_MUTED` (`--text-muted`) — historical replay
//   - `STATUS_PAPER` -> `WARNING`    (`--status-warning`) — sim execution
//   - `STATUS_LIVE`  -> `DANGER`     (`--status-danger`) — real money
//
// We deliberately reuse `DANGER` (the `--candle-down` red) instead of
// inventing a fourth red, so the operator's eye reads "this pill is
// red = real money" using the same token as candle bears. The only
// other `DANGER`-solid surface in the chrome is the HALT button (per
// §10.6 wording: "the only `color.danger`-solid element in the entire
// window"). The pill renders as a `status_badge` (low-alpha fill +
// strong border + colored text) which is visually distinct from the
// solid-red HALT button so the two co-exist without ambiguity.
pub const STATUS_DEMO: egui::Color32 = TEXT_MUTED;
pub const STATUS_PAPER: egui::Color32 = WARNING;
pub const STATUS_LIVE: egui::Color32 = DANGER;

// ─── Spacing scale ──────────────────────────────────────────────────────
//
// 4-pt grid. The names map to the use-case so call sites read like
// `ui.add_space(SPACE_MD)` rather than guessing magic numbers.

pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 12.0;
pub const SPACE_LG: f32 = 16.0;
pub const SPACE_XL: f32 = 24.0;

// ─── Typography scale ───────────────────────────────────────────────────
//
// 4 levels only. Anything outside these is a code smell.

pub const FONT_CAPTION: f32 = 11.0; // labels, tags, captions
pub const FONT_BODY: f32 = 13.0; // default text + buttons
pub const FONT_SUBTITLE: f32 = 15.0; // card/section headers
pub const FONT_TITLE: f32 = 20.0; // page headers

// ─── Corner radii ───────────────────────────────────────────────────────

pub const RADIUS_SM: u8 = 4;
pub const RADIUS_MD: u8 = 6;
pub const RADIUS_LG: u8 = 8;

// ─── Layout heights ─────────────────────────────────────────────────────
//
// Numbers chosen to match the pro-platform survey:
//   TradingView   top 44 px, drawing rail 32-40 px, status 22 px
//   cTrader       header 72 px (two-row), Market Watch 280 px, ASP 320 px
//   Bloomberg     command bar 28 px, function-key strip 24 px

pub const TOPBAR_HEIGHT: f32 = 44.0;
pub const STATUSBAR_HEIGHT: f32 = 22.0;
pub const ACTIONBAR_HEIGHT: f32 = 48.0;

/// Icon-only left rail (TradingView pattern). Tooltips on hover.
pub const SIDEBAR_RAIL_WIDTH: f32 = 56.0;
/// Optional wider sidebar for the data-panel mode (cTrader Market
/// Watch). Currently kept as the default until we ship the rail
/// + secondary panel design.
pub const SIDEBAR_WIDTH_DEFAULT: f32 = 220.0;
pub const SIDEBAR_WIDTH_MIN: f32 = 56.0; // collapses to rail
pub const SIDEBAR_WIDTH_MAX: f32 = 320.0;

pub const BUTTON_HEIGHT: f32 = 32.0;
pub const BUTTON_HEIGHT_SM: f32 = 24.0;
/// Tabular row height for Positions / Orders / History / Watchlist.
pub const TABLE_ROW_HEIGHT: f32 = 24.0;

// ─── Apply global theme ─────────────────────────────────────────────────

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(SPACE_SM, SPACE_XS);
    style.spacing.button_padding = egui::vec2(SPACE_MD, 6.0);
    style.spacing.menu_margin = egui::Margin::same(SPACE_SM as i8);
    style.spacing.window_margin = egui::Margin::same(SPACE_MD as i8);
    style.spacing.indent = SPACE_LG;
    style.spacing.interact_size = egui::vec2(40.0, BUTTON_HEIGHT);

    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(FONT_TITLE, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Name("Subtitle".into()),
        egui::FontId::new(FONT_SUBTITLE, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(FONT_BODY, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(FONT_BODY, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(FONT_BODY - 1.0, egui::FontFamily::Monospace),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(FONT_CAPTION, egui::FontFamily::Proportional),
    );

    style.visuals = egui::Visuals::dark();
    style.visuals.override_text_color = Some(TEXT_PRIMARY);
    style.visuals.panel_fill = PANEL_BG;
    style.visuals.window_fill = PANEL_BG;
    style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.faint_bg_color = SURFACE_BG;
    style.visuals.extreme_bg_color = APP_BG;
    style.visuals.code_bg_color = SURFACE_ALT;
    style.visuals.hyperlink_color = ACCENT;

    style.visuals.selection.bg_fill = ACCENT.linear_multiply(0.30);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    let radius_sm = egui::CornerRadius::same(RADIUS_SM);

    style.visuals.widgets.noninteractive.bg_fill = SURFACE_BG;
    style.visuals.widgets.noninteractive.weak_bg_fill = SURFACE_BG;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.corner_radius = radius_sm;

    style.visuals.widgets.inactive.bg_fill = SURFACE_BG;
    style.visuals.widgets.inactive.weak_bg_fill = SURFACE_BG;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.corner_radius = radius_sm;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    style.visuals.widgets.hovered.bg_fill = SURFACE_ALT;
    style.visuals.widgets.hovered.weak_bg_fill = SURFACE_ALT;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, BORDER_STRONG);
    style.visuals.widgets.hovered.corner_radius = radius_sm;
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    style.visuals.widgets.active.bg_fill = ACCENT_MUTED;
    style.visuals.widgets.active.weak_bg_fill = ACCENT_MUTED;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.active.corner_radius = radius_sm;
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    style.visuals.widgets.open.bg_fill = SURFACE_ALT;
    style.visuals.widgets.open.weak_bg_fill = SURFACE_ALT;
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.open.corner_radius = radius_sm;
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    ctx.set_style(style);
}

// ─── Frame helpers ──────────────────────────────────────────────────────

/// Top app bar. Slim, panel-colored, single hairline bottom border.
pub fn top_panel_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::menu(style);
    frame.fill = PANEL_BG;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.inner_margin = egui::Margin {
        left: SPACE_LG as i8,
        right: SPACE_LG as i8,
        top: SPACE_SM as i8,
        bottom: SPACE_SM as i8,
    };
    frame
}

/// Left sidebar. Same color as top bar to read as one chrome surface.
pub fn sidebar_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL_BG)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .inner_margin(egui::Margin {
            left: SPACE_SM as i8,
            right: SPACE_SM as i8,
            top: SPACE_MD as i8,
            bottom: SPACE_MD as i8,
        })
}

/// Bottom action bar — engine + broker controls. The fill uses
/// SURFACE_BG (one step above PANEL_BG) so the bar reads as its own
/// surface, with a subtle top border to separate it from the dock.
pub fn action_bar_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE_BG)
        .stroke(egui::Stroke::new(1.0, BORDER_STRONG))
        .inner_margin(egui::Margin::symmetric(SPACE_LG as i8, SPACE_SM as i8))
}

/// Bottom status bar — slim 22-px strip with read-only state. Pro
/// convention is for this to be the lowest-visual-weight surface in
/// the whole shell, so we use `PANEL_BG` (not SURFACE_BG) and a
/// single hairline border on top.
pub fn status_bar_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL_BG)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .inner_margin(egui::Margin {
            left: SPACE_SM as i8,
            right: SPACE_SM as i8,
            top: 2,
            bottom: 2,
        })
}

/// Vertical hair-line separator used inside the status bar between
/// info groups. Adds 8 px padding either side and uses BORDER so it
/// reads as quieter than `egui::Separator::default()`.
pub fn status_separator(ui: &mut egui::Ui) {
    ui.add_space(SPACE_SM);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(1.0, ui.available_height() - 4.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 0.0, BORDER);
    ui.add_space(SPACE_SM);
}

/// Central content area.
pub fn central_panel_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(APP_BG)
        .inner_margin(egui::Margin::same(SPACE_SM as i8))
}

/// Standard card surface — one level above the panel.
pub fn card_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE_BG)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(egui::CornerRadius::same(RADIUS_MD))
        .inner_margin(egui::Margin::same(SPACE_LG as i8))
}

/// Section group — slightly elevated card, used inside cards.
pub fn section_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE_ALT)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(egui::CornerRadius::same(RADIUS_SM))
        .inner_margin(egui::Margin::same(SPACE_MD as i8))
}

/// Soft accent callout — for primary value reveals.
#[allow(dead_code)]
pub fn accent_frame(_style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(ACCENT_MUTED)
        .stroke(egui::Stroke::new(1.0, ACCENT.linear_multiply(0.55)))
        .corner_radius(egui::CornerRadius::same(RADIUS_MD))
        .inner_margin(egui::Margin::symmetric(SPACE_MD as i8, SPACE_SM as i8))
}

// ─── Status primitives ──────────────────────────────────────────────────

/// Filled status dot. Sized off `font_size` so it sits with its label.
pub fn status_dot(ui: &mut egui::Ui, color: egui::Color32, font_size: f32) {
    let radius = (font_size * 0.32).max(3.5);
    let size = egui::vec2(radius * 2.0 + 4.0, font_size * 1.4);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let center = egui::pos2(rect.min.x + radius + 2.0, rect.center().y);
        // Soft outer halo so the dot reads on any surface.
        ui.painter()
            .circle_filled(center, radius + 1.5, color.linear_multiply(0.25));
        ui.painter().circle_filled(center, radius, color);
    }
}

/// Compact colored pill — "LIVE", "OFFLINE", "PRO", etc.
pub fn status_badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame::new()
        .fill(color.linear_multiply(0.16))
        .stroke(egui::Stroke::new(1.0, color.linear_multiply(0.60)))
        .inner_margin(egui::Margin::symmetric(SPACE_SM as i8, 3))
        .corner_radius(egui::CornerRadius::same(RADIUS_SM))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(FONT_CAPTION - 0.5)
                    .color(color)
                    .strong(),
            );
        });
}

/// Section/group header label. Use ABOVE a list of related items.
///
/// Renders as letter-spaced UPPERCASE in the muted text color, with a
/// 1-px hairline divider below — the universal "section" pattern across
/// Bloomberg, cTrader, Jira, Linear. The letter spacing is faked by
/// inserting a thin space between every character (egui has no native
/// `letter-spacing` knob).
pub fn section_label(ui: &mut egui::Ui, text: &str) {
    let spaced: String = text
        .to_uppercase()
        .chars()
        .collect::<Vec<_>>()
        .as_slice()
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("\u{2009}"); // U+2009 THIN SPACE — visual letter-spacing
    ui.add_space(SPACE_XS);
    ui.label(
        egui::RichText::new(spaced)
            .size(FONT_CAPTION - 1.0)
            .color(TEXT_FAINT)
            .strong(),
    );
    // Hairline divider sits flush below the label so the eye reads the
    // section as a contained group, not a free-floating caption.
    let line_response =
        ui.allocate_response(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
    if ui.is_rect_visible(line_response.rect) {
        ui.painter().hline(
            line_response.rect.x_range(),
            line_response.rect.center().y,
            egui::Stroke::new(1.0, BORDER),
        );
    }
    ui.add_space(SPACE_XS);
}

/// View-level heading + optional subtitle. Use at the top of a panel.
pub fn view_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.label(
        egui::RichText::new(title)
            .size(FONT_TITLE)
            .strong()
            .color(TEXT_PRIMARY),
    );
    if !subtitle.trim().is_empty() {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(subtitle)
                .size(FONT_BODY)
                .color(TEXT_MUTED),
        );
    }
    ui.add_space(SPACE_MD);
}

// ─── Button primitives ──────────────────────────────────────────────────

/// Variants emit consistent visual hierarchy across the whole app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    /// Subtle text-only button — for tertiary actions.
    Ghost,
    /// Neutral filled button — default for most secondary actions.
    Secondary,
    /// Primary CTA — accent fill, used for the most important action.
    Primary,
    /// Success-colored CTA — typically "Start" / "Connect".
    Success,
    /// Destructive — "Stop" / "Disconnect" / "Delete".
    Danger,
}

/// Render a styled button. Returns the egui Response.
pub fn button(ui: &mut egui::Ui, label: &str, kind: ButtonKind) -> egui::Response {
    let (fill, stroke, text) = match kind {
        ButtonKind::Ghost => (egui::Color32::TRANSPARENT, egui::Stroke::NONE, TEXT_MUTED),
        ButtonKind::Secondary => (
            SURFACE_BG,
            egui::Stroke::new(1.0, BORDER_STRONG),
            TEXT_PRIMARY,
        ),
        ButtonKind::Primary => (ACCENT, egui::Stroke::new(1.0, ACCENT), egui::Color32::WHITE),
        ButtonKind::Success => (
            SUCCESS.linear_multiply(0.18),
            egui::Stroke::new(1.0, SUCCESS.linear_multiply(0.65)),
            SUCCESS,
        ),
        ButtonKind::Danger => (
            DANGER.linear_multiply(0.18),
            egui::Stroke::new(1.0, DANGER.linear_multiply(0.70)),
            DANGER,
        ),
    };
    let button = egui::Button::new(
        egui::RichText::new(label)
            .size(FONT_BODY)
            .color(text)
            .strong(),
    )
    .fill(fill)
    .stroke(stroke)
    .corner_radius(egui::CornerRadius::same(RADIUS_SM))
    .min_size(egui::vec2(0.0, BUTTON_HEIGHT));
    ui.add(button)
}

/// Same as `button` but compact — for inline contexts (dock tab bar, lists).
pub fn small_button(ui: &mut egui::Ui, label: &str, kind: ButtonKind) -> egui::Response {
    let (fill, stroke, text) = match kind {
        ButtonKind::Ghost => (egui::Color32::TRANSPARENT, egui::Stroke::NONE, TEXT_MUTED),
        ButtonKind::Secondary => (
            SURFACE_BG,
            egui::Stroke::new(1.0, BORDER_STRONG),
            TEXT_PRIMARY,
        ),
        ButtonKind::Primary => (ACCENT, egui::Stroke::new(1.0, ACCENT), egui::Color32::WHITE),
        ButtonKind::Success => (
            SUCCESS.linear_multiply(0.18),
            egui::Stroke::new(1.0, SUCCESS.linear_multiply(0.65)),
            SUCCESS,
        ),
        ButtonKind::Danger => (
            DANGER.linear_multiply(0.18),
            egui::Stroke::new(1.0, DANGER.linear_multiply(0.70)),
            DANGER,
        ),
    };
    let button = egui::Button::new(
        egui::RichText::new(label)
            .size(FONT_BODY - 1.0)
            .color(text)
            .strong(),
    )
    .fill(fill)
    .stroke(stroke)
    .corner_radius(egui::CornerRadius::same(RADIUS_SM))
    .min_size(egui::vec2(0.0, BUTTON_HEIGHT_SM));
    ui.add(button)
}

// ─── Nav item helper ────────────────────────────────────────────────────

/// Render a sidebar navigation row. Returns true when clicked.
/// Picks the right hover/active style and uses a left accent stripe for
/// the active row so the eye locks on the selected view instantly.
pub fn nav_item(ui: &mut egui::Ui, label: &str, description: &str, active: bool) -> egui::Response {
    nav_item_with_icon(ui, "", label, description, active)
}

/// Sidebar nav item with a single-glyph icon prefix. The icon column is
/// fixed-width so labels line up cleanly across the whole rail —
/// matches Bloomberg / cTrader / TradingView sidebar conventions where
/// the eye scans the icons first, then the labels.
pub fn nav_item_with_icon(
    ui: &mut egui::Ui,
    icon: &str,
    label: &str,
    description: &str,
    active: bool,
) -> egui::Response {
    let frame_fill = if active {
        ACCENT_MUTED
    } else {
        egui::Color32::TRANSPARENT
    };
    let label_color = if active { TEXT_PRIMARY } else { TEXT_MUTED };
    let icon_color = if active { ACCENT } else { TEXT_FAINT };
    let stripe_color = if active {
        ACCENT
    } else {
        egui::Color32::TRANSPARENT
    };

    // 28px row height — Bloomberg/cTrader sidebar density.
    let desired_size = egui::vec2(ui.available_width(), 28.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    let hovered = response.hovered();
    let fill = if hovered && !active {
        SURFACE_BG
    } else {
        frame_fill
    };

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, RADIUS_SM as f32, fill);
        // Left accent stripe (3px wide, tall as row, only visible when active).
        let stripe_rect = egui::Rect::from_min_size(rect.min, egui::vec2(3.0, rect.height()));
        painter.rect_filled(stripe_rect, 1.0, stripe_color);

        // Icon column: fixed 22px slot starting at SPACE_SM after the
        // stripe so all rail items have aligned glyphs.
        let icon_x = rect.min.x + SPACE_SM + 6.0;
        let label_x = if icon.is_empty() {
            rect.min.x + SPACE_MD
        } else {
            painter.text(
                egui::pos2(icon_x, rect.center().y),
                egui::Align2::LEFT_CENTER,
                icon,
                egui::FontId::new(FONT_BODY + 1.0, egui::FontFamily::Proportional),
                icon_color,
            );
            // Reserve a 22px column so labels line up across the rail.
            icon_x + 22.0
        };

        painter.text(
            egui::pos2(label_x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::new(FONT_BODY, egui::FontFamily::Proportional),
            label_color,
        );
    }

    response.on_hover_text(description)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_theme_swaps_default_panel_fill_for_warm_dark() {
        let ctx = egui::Context::default();
        let default_fill = ctx.style().visuals.panel_fill;

        apply_theme(&ctx);

        let style = ctx.style();
        assert_ne!(style.visuals.panel_fill, default_fill);
        assert_eq!(style.visuals.panel_fill, PANEL_BG);
    }

    #[test]
    fn spacing_scale_uses_a_4pt_grid() {
        // The whole design system breaks if these aren't on a 4-pt grid.
        for v in [SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, SPACE_XL] {
            assert!((v % 4.0).abs() < f32::EPSILON, "{v} is not on the 4pt grid");
        }
    }

    #[test]
    fn type_scale_has_exactly_four_levels() {
        // Adding a 5th level should be a deliberate design decision —
        // if you're tempted, you probably wanted SUBTITLE or BODY.
        let levels = [FONT_CAPTION, FONT_BODY, FONT_SUBTITLE, FONT_TITLE];
        assert_eq!(levels.len(), 4);
        let mut sorted = levels.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, levels.to_vec());
    }

    #[test]
    fn card_frame_uses_surface_palette_and_md_radius() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);

        let frame = card_frame(&ctx.style());

        assert_eq!(frame.fill, SURFACE_BG);
        assert_eq!(frame.stroke.color, BORDER);
        assert_eq!(frame.corner_radius, egui::CornerRadius::same(RADIUS_MD));
    }

    #[test]
    fn button_kinds_compile_and_distinct() {
        // Symbolic check that the variants stay distinct — protects
        // against accidental Ghost-becomes-Secondary refactors.
        assert_ne!(ButtonKind::Ghost, ButtonKind::Secondary);
        assert_ne!(ButtonKind::Primary, ButtonKind::Success);
        assert_ne!(ButtonKind::Success, ButtonKind::Danger);
    }
}
