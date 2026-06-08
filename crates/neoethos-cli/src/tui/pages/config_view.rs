//! Config page — an EDITABLE form over the most important settings, saved
//! back to the user's config.yaml via [`neoethos_core::Settings::save`] (the
//! same path the GUI writes, so the CLI/TUI/GUI stay in parity). Power users
//! can still edit the full config.yaml by hand for the long tail of knobs.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::form::{Field, FormState};
use crate::tui::theme;

/// Build the editable Config form from the on-disk Settings. Each field maps to
/// one Settings knob; [`save_config_form`] writes them back. Curated to the
/// settings a user changes most (symbol/data, compute mode, the discovery
/// search budget, prop-firm strictness) — the full long tail stays in
/// config.yaml.
pub fn make_config_form() -> FormState {
    let s = neoethos_core::Settings::load().unwrap_or_default();
    FormState::new(vec![
        Field::new("Symbol", s.system.symbol.clone(), "Primary symbol, e.g. EURUSD"),
        Field::new(
            "Account currency",
            s.system.account_currency.clone(),
            "Deposit currency, e.g. USD / EUR",
        ),
        Field::new("Base timeframe", s.system.base_timeframe.clone(), "e.g. M1, M5, M30, H1"),
        Field::new(
            "Compute mode",
            s.system.enable_gpu_preference.clone(),
            "auto | cpu | gpu  (auto picks the best device, never OOMs)",
        ),
        Field::new(
            "Data dir",
            s.system.data_dir.display().to_string(),
            "Absolute path to the data/ root (symbol=*/timeframe=*/)",
        ),
        Field::new(
            "Discovery mode",
            s.models.discovery_mode.clone(),
            "prop_firm | strict | risky",
        ),
        Field::new(
            "Population",
            s.models.prop_search_population.to_string(),
            "GA population per generation",
        ),
        Field::new(
            "Generations",
            s.models.prop_search_generations.to_string(),
            "GA generations (also time-bounded by Max hours)",
        ),
        Field::new(
            "Max hours/combo",
            format!("{}", s.models.prop_search_max_hours),
            "Wall-clock cap per symbol×TF combo",
        ),
        Field::new(
            "Portfolio size",
            s.models.prop_search_portfolio_size.to_string(),
            "Max strategies kept per combo",
        ),
        Field::new(
            "Prop-firm pass rate",
            format!("{}", s.models.prop_firm_min_pass_rate),
            "0 = ranking-only; 0.40–0.65 = stricter all-window consistency",
        ),
    ])
}

/// Apply the form's values to the on-disk Settings and save. Returns a status
/// string (count of changed knobs + path), or an error string on a bad value /
/// write failure. Loads a FRESH Settings so untouched long-tail knobs are
/// preserved exactly.
pub fn save_config_form(form: &FormState) -> String {
    let mut s = match neoethos_core::Settings::load() {
        Ok(s) => s,
        Err(e) => return format!("Config load failed: {e}"),
    };
    let mut changed = 0usize;
    let mut rejected: Vec<&str> = Vec::new();

    let set_string = |dst: &mut String, v: Option<&str>, changed: &mut usize| {
        if let Some(v) = v {
            let v = v.trim();
            if !v.is_empty() && v != dst {
                *dst = v.to_string();
                *changed += 1;
            }
        }
    };

    set_string(&mut s.system.symbol, form.value_for("Symbol"), &mut changed);
    set_string(
        &mut s.system.account_currency,
        form.value_for("Account currency"),
        &mut changed,
    );
    set_string(
        &mut s.system.base_timeframe,
        form.value_for("Base timeframe"),
        &mut changed,
    );

    // Compute mode + discovery mode are enumerated — validate before applying.
    if let Some(v) = form.value_for("Compute mode") {
        let v = v.trim().to_lowercase();
        if matches!(v.as_str(), "auto" | "cpu" | "gpu") {
            if v != s.system.enable_gpu_preference {
                s.system.enable_gpu_preference = v;
                changed += 1;
            }
        } else {
            rejected.push("compute-mode (use auto|cpu|gpu)");
        }
    }
    if let Some(v) = form.value_for("Discovery mode") {
        let v = v.trim().to_lowercase();
        if matches!(v.as_str(), "prop_firm" | "strict" | "risky") {
            if v != s.models.discovery_mode {
                s.models.discovery_mode = v;
                changed += 1;
            }
        } else {
            rejected.push("discovery-mode (use prop_firm|strict|risky)");
        }
    }

    if let Some(v) = form.value_for("Data dir") {
        let v = v.trim();
        if !v.is_empty() && std::path::Path::new(v) != s.system.data_dir {
            s.system.data_dir = std::path::PathBuf::from(v);
            changed += 1;
        }
    }

    // Numeric knobs — parse + reject bad input rather than silently zeroing.
    apply_usize(form, "Population", &mut s.models.prop_search_population, &mut changed, &mut rejected);
    apply_usize(form, "Generations", &mut s.models.prop_search_generations, &mut changed, &mut rejected);
    apply_usize(form, "Portfolio size", &mut s.models.prop_search_portfolio_size, &mut changed, &mut rejected);
    apply_f64(form, "Max hours/combo", &mut s.models.prop_search_max_hours, 0.0, f64::MAX, &mut changed, &mut rejected);
    apply_f64(form, "Prop-firm pass rate", &mut s.models.prop_firm_min_pass_rate, 0.0, 1.0, &mut changed, &mut rejected);

    if !rejected.is_empty() {
        return format!("Not saved — invalid: {}", rejected.join(", "));
    }
    if changed == 0 {
        return "No changes to save".to_string();
    }

    let path = neoethos_core::config::user_config_path();
    match s.save(&path) {
        Ok(()) => format!("Saved {changed} change(s) → {}", path.display()),
        Err(e) => format!("Save failed: {e}"),
    }
}

fn apply_usize(
    form: &FormState,
    label: &'static str,
    dst: &mut usize,
    changed: &mut usize,
    rejected: &mut Vec<&'static str>,
) {
    if let Some(raw) = form.value_for(label) {
        let raw = raw.trim();
        if raw.is_empty() {
            return;
        }
        match raw.parse::<usize>() {
            Ok(v) if v != *dst => {
                *dst = v;
                *changed += 1;
            }
            Ok(_) => {}
            Err(_) => rejected.push(label),
        }
    }
}

fn apply_f64(
    form: &FormState,
    label: &'static str,
    dst: &mut f64,
    lo: f64,
    hi: f64,
    changed: &mut usize,
    rejected: &mut Vec<&'static str>,
) {
    if let Some(raw) = form.value_for(label) {
        let raw = raw.trim();
        if raw.is_empty() {
            return;
        }
        match raw.parse::<f64>() {
            Ok(v) if v.is_finite() && v >= lo && v <= hi => {
                if (v - *dst).abs() > f64::EPSILON {
                    *dst = v;
                    *changed += 1;
                }
            }
            _ => rejected.push(label),
        }
    }
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    // While editing a field, keystrokes mutate the focused value. Scoped so the
    // `&mut` borrow ends before any branch below reassigns `config_form`.
    if shared.config_form.editing {
        let form = &mut shared.config_form;
        match code {
            KeyCode::Enter => form.stop_editing(true),
            KeyCode::Esc => form.stop_editing(false),
            KeyCode::Backspace => form.backspace(),
            KeyCode::Char(c) => form.type_char(c),
            _ => return false,
        }
        return true;
    }
    match code {
        KeyCode::Up => {
            shared.config_form.focus_prev();
            true
        }
        KeyCode::Down => {
            shared.config_form.focus_next();
            true
        }
        KeyCode::Enter => {
            shared.config_form.start_editing();
            true
        }
        KeyCode::Char('S') => {
            let msg = save_config_form(&shared.config_form);
            // Reload so the form reflects exactly what's now on disk.
            shared.config_form = make_config_form();
            shared.status = msg;
            true
        }
        // 'R' (reload) is handled by the global refresh key in app.rs, which
        // also rebuilds this form from disk.
        _ => false,
    }
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " CONFIG — edit & save to config.yaml ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let form = &shared.config_form;
    let mut lines: Vec<Line> = Vec::with_capacity(form.fields.len() * 2 + 4);
    for (i, field) in form.fields.iter().enumerate() {
        let focused = i == form.focused;
        let editing = focused && form.editing;
        let marker = if focused { "▸ " } else { "  " };
        let value = if editing {
            format!("{}▌", field.value)
        } else {
            field.effective().to_string()
        };
        let label_style = if focused {
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT_PRIMARY)
        };
        let value_style = if editing {
            Style::default().fg(theme::APP_BG).bg(theme::ACCENT)
        } else if focused {
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT_PRIMARY)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{marker}{:<22}", field.label), label_style),
            Span::styled(format!(" {value}"), value_style),
        ]));
        if focused {
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(format!("↳ {}", field.hint), theme::caption_style()),
            ]));
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![Span::styled(
        "  [↑↓] field   [Enter] edit   [Esc] cancel   [S] save→config.yaml   [R] reload",
        theme::caption_style().add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(vec![Span::styled(
        "  Long-tail knobs live in config.yaml; this edits the common ones with parity to CLI/GUI.",
        theme::caption_style(),
    )]));

    Paragraph::new(lines).render(inner, buf);
}
