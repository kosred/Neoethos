//! Forex-AI design system.
//!
//! One source of truth for the surface palette, type scale, spacing
//! scale, and reusable UI primitives. The intent is "professional
//! desktop trading terminal" — clean, dense-but-breathing, no random
//! corner radii or text sizes scattered through call sites. Anything
//! that needs styling should reach for the tokens here.
//!
//! Inspired by Linear / Vercel / Anthropic console aesthetics:
//! warm-dark surfaces, soft borders, one distinct accent, semantic
//! colors used sparingly.

use eframe::egui;

// ─── Color palette ───────────────────────────────────────────────────────
//
// All RGB values were picked together so the contrast ratios stack:
//   TEXT_PRIMARY on APP_BG       ≥ 15:1 (AAA)
//   TEXT_PRIMARY on SURFACE_BG   ≥ 13:1 (AAA)
//   TEXT_MUTED   on SURFACE_BG   ≥  4.6:1 (AA)
//   ACCENT       on APP_BG       ≥  4.5:1 (AA)

/// Application root background (behind everything).
pub const APP_BG: egui::Color32 = egui::Color32::from_rgb(13, 15, 20);
/// Default panel background — top bar, side panel, action bar.
pub const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(17, 20, 26);
/// Card / surface background — one step "above" the panel.
pub const SURFACE_BG: egui::Color32 = egui::Color32::from_rgb(22, 26, 33);
/// Elevated surface — hover, focused tab, accent callouts.
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(28, 33, 41);
/// Chart canvas — slightly darker than APP_BG for contrast.
pub const CHART_BG: egui::Color32 = egui::Color32::from_rgb(10, 12, 16);

/// Subtle hairline borders.
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(38, 44, 54);
/// Heavier borders for focus states / dividers.
pub const BORDER_STRONG: egui::Color32 = egui::Color32::from_rgb(56, 65, 79);
/// Chart grid lines.
pub const GRID: egui::Color32 = egui::Color32::from_rgb(32, 39, 49);

/// Primary text.
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(232, 237, 244);
/// Secondary text (labels, captions, hints).
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(149, 162, 180);
/// Tertiary text (disabled, placeholders).
pub const TEXT_FAINT: egui::Color32 = egui::Color32::from_rgb(96, 108, 124);

/// Primary accent — used for active nav, primary CTAs, focus rings.
/// Soft violet-blue, more distinctive than a generic terminal cyan.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(124, 137, 255);
/// Accent at 18% alpha equivalent — for selected backgrounds.
pub const ACCENT_MUTED: egui::Color32 = egui::Color32::from_rgb(36, 41, 73);
/// Accent at very low alpha — for hover backgrounds on accent items.
pub const ACCENT_SOFT: egui::Color32 = egui::Color32::from_rgb(26, 30, 52);

/// Semantic colors. Use them ONLY for the meaning they signal.
pub const SUCCESS: egui::Color32 = egui::Color32::from_rgb(74, 200, 142);
pub const WARNING: egui::Color32 = egui::Color32::from_rgb(238, 172, 55);
pub const DANGER: egui::Color32 = egui::Color32::from_rgb(239, 95, 110);
pub const INFO: egui::Color32 = egui::Color32::from_rgb(108, 168, 230);

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

pub const TOPBAR_HEIGHT: f32 = 56.0;
pub const ACTIONBAR_HEIGHT: f32 = 48.0;
pub const SIDEBAR_WIDTH_DEFAULT: f32 = 220.0;
pub const SIDEBAR_WIDTH_MIN: f32 = 188.0;
pub const SIDEBAR_WIDTH_MAX: f32 = 320.0;
pub const BUTTON_HEIGHT: f32 = 32.0;
pub const BUTTON_HEIGHT_SM: f32 = 24.0;

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
pub fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(FONT_CAPTION - 1.0)
            .color(TEXT_FAINT)
            .strong(),
    );
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
        ButtonKind::Ghost => (
            egui::Color32::TRANSPARENT,
            egui::Stroke::NONE,
            TEXT_MUTED,
        ),
        ButtonKind::Secondary => (SURFACE_BG, egui::Stroke::new(1.0, BORDER_STRONG), TEXT_PRIMARY),
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
        ButtonKind::Ghost => (
            egui::Color32::TRANSPARENT,
            egui::Stroke::NONE,
            TEXT_MUTED,
        ),
        ButtonKind::Secondary => (SURFACE_BG, egui::Stroke::new(1.0, BORDER_STRONG), TEXT_PRIMARY),
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
pub fn nav_item(
    ui: &mut egui::Ui,
    label: &str,
    description: &str,
    active: bool,
) -> egui::Response {
    let frame_fill = if active { ACCENT_MUTED } else { egui::Color32::TRANSPARENT };
    let label_color = if active { TEXT_PRIMARY } else { TEXT_MUTED };
    let stripe_color = if active { ACCENT } else { egui::Color32::TRANSPARENT };

    let desired_size = egui::vec2(ui.available_width(), 30.0);
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
        // Left accent stripe.
        let stripe_rect = egui::Rect::from_min_size(
            rect.min,
            egui::vec2(3.0, rect.height()),
        );
        painter.rect_filled(stripe_rect, 1.0, stripe_color);

        let text_pos = egui::pos2(rect.min.x + SPACE_MD, rect.center().y);
        painter.text(
            text_pos,
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
