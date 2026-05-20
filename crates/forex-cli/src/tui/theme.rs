//! TUI palette — anchored to the TradingView dark palette already
//! used by `forex-app` (`docs/audits/ui_design_research_2026-05-12.md`).
//!
//! ratatui's `Color` is RGB-based when the terminal supports
//! truecolor, falling back to the nearest 256-color cube otherwise.
//! All constants here are picked from the same hex values as
//! `forex-app/src/ui/theme.rs`.

use ratatui::style::{Color, Modifier, Style};

// ─── Surfaces ──────────────────────────────────────────────────────
pub const APP_BG: Color = Color::Rgb(0x0E, 0x11, 0x16);
pub const PANEL_BG: Color = Color::Rgb(0x16, 0x1B, 0x22);
pub const SURFACE_BG: Color = Color::Rgb(0x1C, 0x22, 0x30);
pub const SURFACE_ALT: Color = Color::Rgb(0x22, 0x29, 0x3A);

// ─── Borders / dividers ───────────────────────────────────────────
pub const BORDER: Color = Color::Rgb(0x2A, 0x2F, 0x3A);
pub const BORDER_STRONG: Color = Color::Rgb(0x3A, 0x40, 0x4D);

// ─── Text ──────────────────────────────────────────────────────────
pub const TEXT_PRIMARY: Color = Color::Rgb(0xE6, 0xEA, 0xF2);
pub const TEXT_MUTED: Color = Color::Rgb(0x9A, 0xA4, 0xB2);
pub const TEXT_FAINT: Color = Color::Rgb(0x5C, 0x64, 0x73);

// ─── Brand ─────────────────────────────────────────────────────────
pub const ACCENT: Color = Color::Rgb(0x29, 0x62, 0xFF);
pub const ACCENT_SOFT: Color = Color::Rgb(0x1E, 0x2A, 0x4A);

// ─── Trading semantics ─────────────────────────────────────────────
pub const BUY: Color = Color::Rgb(0x26, 0xA6, 0x9A);
pub const SELL: Color = Color::Rgb(0xEF, 0x53, 0x50);
pub const WARN: Color = Color::Rgb(0xF4, 0xB4, 0x00);

// ─── Convenience styles ───────────────────────────────────────────

pub fn title_style() -> Style {
    Style::default()
        .fg(TEXT_PRIMARY)
        .add_modifier(Modifier::BOLD)
}

/// UPPERCASE letter-spaced caption — pro-trading convention.
pub fn caption_style() -> Style {
    Style::default().fg(TEXT_MUTED).add_modifier(Modifier::DIM)
}

pub fn muted_style() -> Style {
    Style::default().fg(TEXT_MUTED)
}

pub fn primary_style() -> Style {
    Style::default().fg(TEXT_PRIMARY)
}

pub fn accent_style() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

pub fn buy_style() -> Style {
    Style::default().fg(BUY).add_modifier(Modifier::BOLD)
}

pub fn sell_style() -> Style {
    Style::default().fg(SELL).add_modifier(Modifier::BOLD)
}

pub fn warn_style() -> Style {
    Style::default().fg(WARN)
}

pub fn nav_inactive_style() -> Style {
    Style::default().fg(TEXT_FAINT)
}

pub fn nav_active_style() -> Style {
    Style::default()
        .fg(ACCENT)
        .bg(ACCENT_SOFT)
        .add_modifier(Modifier::BOLD)
}

pub fn panel_block_style() -> Style {
    Style::default().bg(PANEL_BG).fg(TEXT_PRIMARY)
}
