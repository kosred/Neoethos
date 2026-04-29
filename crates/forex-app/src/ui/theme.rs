use eframe::egui;

pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 149, 255);
pub const ACCENT_MUTED: egui::Color32 = egui::Color32::from_rgb(10, 34, 54);
pub const APP_BG: egui::Color32 = egui::Color32::from_rgb(7, 10, 14);
pub const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(12, 17, 23);
pub const SURFACE_BG: egui::Color32 = egui::Color32::from_rgb(15, 21, 29);
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(20, 28, 38);
pub const CHART_BG: egui::Color32 = egui::Color32::from_rgb(8, 12, 17);
pub const GRID: egui::Color32 = egui::Color32::from_rgb(28, 38, 50);
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(35, 47, 61);
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(236, 240, 244);
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(129, 145, 163);
pub const SUCCESS: egui::Color32 = egui::Color32::from_rgb(31, 194, 120);
pub const WARNING: egui::Color32 = egui::Color32::from_rgb(238, 172, 55);
pub const DANGER: egui::Color32 = egui::Color32::from_rgb(239, 83, 102);

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.menu_margin = egui::Margin::same(6);
    style.spacing.window_margin = egui::Margin::same(8);
    style.spacing.indent = 16.0;

    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(20.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Name("Heading2".into()),
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(12.0, egui::FontFamily::Monospace),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(11.0, egui::FontFamily::Proportional),
    );

    style.visuals = egui::Visuals::dark();
    style.visuals.override_text_color = Some(TEXT_PRIMARY);
    style.visuals.panel_fill = PANEL_BG;
    style.visuals.window_fill = PANEL_BG;
    style.visuals.faint_bg_color = SURFACE_BG;
    style.visuals.extreme_bg_color = APP_BG;
    style.visuals.code_bg_color = SURFACE_ALT;
    style.visuals.hyperlink_color = ACCENT;
    style.visuals.selection.bg_fill = ACCENT.linear_multiply(0.35);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.noninteractive.bg_fill = SURFACE_BG;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(4);
    style.visuals.widgets.inactive.bg_fill = SURFACE_BG;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(4);
    style.visuals.widgets.hovered.bg_fill = SURFACE_ALT;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(4);
    style.visuals.widgets.active.bg_fill = ACCENT_MUTED;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.active.corner_radius = egui::CornerRadius::same(4);
    style.visuals.widgets.open.bg_fill = SURFACE_ALT;
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.open.corner_radius = egui::CornerRadius::same(4);

    ctx.set_style(style);
}

pub fn top_panel_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::menu(style);
    frame.fill = PANEL_BG;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.inner_margin = egui::Margin::symmetric(12, 7);
    frame
}

pub fn central_panel_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::central_panel(style);
    frame.fill = APP_BG;
    frame.inner_margin = egui::Margin::same(6);
    frame
}

pub fn card_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::group(style);
    frame.fill = SURFACE_BG;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.corner_radius = egui::CornerRadius::same(6);
    frame.inner_margin = egui::Margin::symmetric(8, 6);
    frame
}

pub fn section_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::group(style);
    frame.fill = SURFACE_ALT;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.corner_radius = egui::CornerRadius::same(4);
    frame.inner_margin = egui::Margin::symmetric(8, 6);
    frame
}

/// Paint a filled status indicator circle at the current ui cursor.
/// Must be called inside a horizontal layout. Dot radius scales with `font_size`.
pub fn status_dot(ui: &mut egui::Ui, color: egui::Color32, font_size: f32) {
    let radius = (font_size * 0.28).max(3.0);
    let size = egui::vec2(radius * 2.0 + 4.0, font_size * 1.4);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().circle_filled(
            egui::pos2(rect.min.x + radius, rect.center().y),
            radius,
            color,
        );
    }
}

/// Render a compact colored status badge, e.g. "LIVE" or "OFFLINE".
/// Works in any layout direction (LTR or RTL).
pub fn status_badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame::new()
        .fill(color.linear_multiply(0.18))
        .stroke(egui::Stroke::new(1.0, color.linear_multiply(0.55)))
        .inner_margin(egui::Margin::symmetric(5, 2))
        .corner_radius(egui::CornerRadius::same(3))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(9.5).color(color).strong());
        });
}

/// Highlighted accent frame — used for key callout values.
#[allow(dead_code)]
pub fn accent_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::group(style);
    frame.fill = ACCENT_MUTED;
    frame.stroke = egui::Stroke::new(1.0, ACCENT.linear_multiply(0.5));
    frame.corner_radius = egui::CornerRadius::same(6);
    frame.inner_margin = egui::Margin::symmetric(10, 6);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_theme_changes_default_panel_fill_and_accent_palette() {
        let ctx = egui::Context::default();
        let default_fill = ctx.style().visuals.panel_fill;

        apply_theme(&ctx);

        let style = ctx.style();
        assert_ne!(style.visuals.panel_fill, default_fill);
        assert_eq!(style.visuals.panel_fill, PANEL_BG);
        assert_eq!(ACCENT, egui::Color32::from_rgb(0, 149, 255));
    }

    #[test]
    fn card_frame_uses_operator_surface_palette() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);

        let frame = card_frame(&ctx.style());

        assert_eq!(frame.fill, SURFACE_BG);
        assert_eq!(frame.stroke.color, BORDER);
        assert_eq!(frame.corner_radius, egui::CornerRadius::same(6));
    }
}
