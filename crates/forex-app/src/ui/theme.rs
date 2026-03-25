use eframe::egui;

pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 140, 255); // Vibrant Neon Blue
pub const ACCENT_MUTED: egui::Color32 = egui::Color32::from_rgb(18, 35, 52);
pub const APP_BG: egui::Color32 = egui::Color32::from_rgb(10, 14, 20); // Deep Obsidian
pub const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(16, 22, 30); // Dark Slate
pub const SURFACE_BG: egui::Color32 = egui::Color32::from_rgb(22, 30, 42); // Elevated Layer
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(28, 40, 56);
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(44, 58, 76);
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(240, 244, 248);
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(130, 148, 168);
pub const SUCCESS: egui::Color32 = egui::Color32::from_rgb(34, 215, 120); // Vibrant Green
pub const WARNING: egui::Color32 = egui::Color32::from_rgb(255, 170, 0);
pub const DANGER: egui::Color32 = egui::Color32::from_rgb(255, 69, 102); // Modern Red

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 12.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.menu_margin = egui::Margin::same(12);
    style.spacing.window_margin = egui::Margin::same(16);
    style.spacing.indent = 24.0;

    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(32.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Name("Heading2".into()),
        egui::FontId::new(24.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(14.0, egui::FontFamily::Monospace),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
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
    style.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(10);
    style.visuals.widgets.inactive.bg_fill = SURFACE_BG;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(10);
    style.visuals.widgets.hovered.bg_fill = SURFACE_ALT;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(10);
    style.visuals.widgets.active.bg_fill = ACCENT_MUTED;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.active.corner_radius = egui::CornerRadius::same(10);
    style.visuals.widgets.open.bg_fill = SURFACE_ALT;
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.open.corner_radius = egui::CornerRadius::same(10);

    ctx.set_style(style);
}

pub fn top_panel_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::menu(style);
    frame.fill = PANEL_BG;
    frame.stroke = egui::Stroke::new(0.0, egui::Color32::TRANSPARENT);
    frame.inner_margin = egui::Margin::symmetric(14, 10);
    frame
}

pub fn central_panel_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::central_panel(style);
    frame.fill = APP_BG;
    frame.inner_margin = egui::Margin::same(16);
    frame
}

pub fn card_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::group(style);
    frame.fill = SURFACE_BG;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.corner_radius = egui::CornerRadius::same(16);
    frame.inner_margin = egui::Margin::symmetric(14, 12);
    frame
}

pub fn section_frame(style: &egui::Style) -> egui::Frame {
    let mut frame = egui::Frame::group(style);
    frame.fill = SURFACE_ALT;
    frame.stroke = egui::Stroke::new(1.0, BORDER);
    frame.corner_radius = egui::CornerRadius::same(12);
    frame.inner_margin = egui::Margin::symmetric(18, 16);
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
        assert_eq!(ACCENT, egui::Color32::from_rgb(72, 170, 255));
    }

    #[test]
    fn card_frame_uses_operator_surface_palette() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);

        let frame = card_frame(&ctx.style());

        assert_eq!(frame.fill, SURFACE_BG);
        assert_eq!(frame.stroke.color, BORDER);
        assert_eq!(frame.corner_radius, egui::CornerRadius::same(16));
    }
}
