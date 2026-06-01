//! Chart — braille candlestick view of a symbol/timeframe's Vortex data.
//!
//! Reads OHLCV via `neoethos_data::load_symbol_timeframe_tail` (the same
//! integrity-gated path the server chart uses) and renders the trailing
//! N candles on a ratatui `Canvas` with the Braille marker.
//!
//! Defensive by contract (see the operator's "clear-errors, no-unwrap"
//! directive): every failure mode — missing data dir, no symbols, a
//! missing/partial/truncated timeframe, an empty dataset, NaN/length-
//! mismatched columns — surfaces as a clear on-screen message instead of
//! a panic. There is no `.unwrap()`/`.expect()`/unchecked index in here.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Rectangle};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

/// Trailing candles to load + render.
const TAIL: usize = 240;

#[derive(Debug, Clone, Copy)]
struct Candle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

/// Per-session Chart page state. Lives in [`AppShared`].
pub struct ChartState {
    data_root: std::path::PathBuf,
    symbols: Vec<String>,
    symbol_idx: usize,
    timeframes: Vec<String>,
    tf_idx: usize,
    candles: Vec<Candle>,
    /// `(symbol, timeframe)` currently loaded into `candles`; `None` means
    /// "needs (re)load on next draw".
    loaded_key: Option<(String, String)>,
    /// Human-readable status shown when there's nothing to plot. Always a
    /// message — never a panic.
    status: String,
}

impl ChartState {
    pub fn new(data_root: &std::path::Path) -> Self {
        let mut s = Self {
            data_root: data_root.to_path_buf(),
            symbols: Vec::new(),
            symbol_idx: 0,
            timeframes: Vec::new(),
            tf_idx: 0,
            candles: Vec::new(),
            loaded_key: None,
            status: String::new(),
        };
        s.refresh_symbols();
        s
    }

    fn refresh_symbols(&mut self) {
        match neoethos_data::discover_symbols(&self.data_root) {
            Ok(syms) if !syms.is_empty() => {
                self.symbols = syms;
                if self.symbol_idx >= self.symbols.len() {
                    self.symbol_idx = 0;
                }
                self.refresh_timeframes();
            }
            Ok(_) => {
                self.symbols.clear();
                self.timeframes.clear();
                self.status = format!(
                    "No symbols under {}. Import data or run Data Bootstrap first.",
                    self.data_root.display()
                );
            }
            Err(e) => {
                self.symbols.clear();
                self.timeframes.clear();
                self.status = format!("Could not scan {}: {e}", self.data_root.display());
            }
        }
    }

    fn refresh_timeframes(&mut self) {
        self.loaded_key = None; // force a reload on next draw
        let Some(symbol) = self.symbols.get(self.symbol_idx).cloned() else {
            self.timeframes.clear();
            return;
        };
        match neoethos_data::discover_timeframes(&self.data_root, &symbol) {
            Ok(tfs) if !tfs.is_empty() => {
                self.timeframes = tfs;
                if self.tf_idx >= self.timeframes.len() {
                    self.tf_idx = 0;
                }
            }
            Ok(_) => {
                self.timeframes.clear();
                self.status = format!("{symbol}: no usable timeframes (all missing/partial).");
            }
            Err(e) => {
                self.timeframes.clear();
                self.status = format!("{symbol}: could not list timeframes: {e}");
            }
        }
    }

    fn current_symbol(&self) -> Option<&str> {
        self.symbols.get(self.symbol_idx).map(String::as_str)
    }
    fn current_tf(&self) -> Option<&str> {
        self.timeframes.get(self.tf_idx).map(String::as_str)
    }

    /// Load candles for the current `(symbol, tf)` if not already loaded.
    /// Every failure path sets `status` + clears `candles`; never panics.
    fn ensure_loaded(&mut self) {
        let (Some(symbol), Some(tf)) = (
            self.current_symbol().map(|s| s.to_string()),
            self.current_tf().map(|s| s.to_string()),
        ) else {
            self.candles.clear();
            return; // status already set by refresh_*
        };
        if self.loaded_key.as_ref() == Some(&(symbol.clone(), tf.clone())) {
            return; // already loaded — no per-frame disk thrash
        }
        self.loaded_key = Some((symbol.clone(), tf.clone()));
        self.candles.clear();
        self.status.clear();

        let ohlcv =
            match neoethos_data::load_symbol_timeframe_tail(&self.data_root, &symbol, &tf, TAIL) {
                Ok(o) => o,
                Err(e) => {
                    self.status = format!("{symbol} {tf}: {e}");
                    return;
                }
            };

        // Defensive: never index blindly. The data layer already enforces
        // equal column lengths, but re-check here so a future regression
        // surfaces as a message, not an out-of-bounds panic.
        let n = ohlcv.close.len();
        if n == 0 {
            self.status = format!("{symbol} {tf}: dataset is empty.");
            return;
        }
        if ohlcv.open.len() != n || ohlcv.high.len() != n || ohlcv.low.len() != n {
            self.status = format!("{symbol} {tf}: malformed OHLCV (column length mismatch).");
            return;
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (o, h, l, c) = (ohlcv.open[i], ohlcv.high[i], ohlcv.low[i], ohlcv.close[i]);
            if o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite() {
                out.push(Candle {
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                });
            }
        }
        if out.is_empty() {
            self.status = format!("{symbol} {tf}: no finite candles to plot.");
            return;
        }
        self.candles = out;
    }

    fn step_symbol(&mut self, delta: isize) {
        let len = self.symbols.len();
        if len == 0 {
            return;
        }
        let len_i = len as isize;
        self.symbol_idx = (((self.symbol_idx as isize + delta) % len_i + len_i) % len_i) as usize;
        self.tf_idx = 0;
        self.refresh_timeframes();
    }
    fn step_tf(&mut self, delta: isize) {
        let len = self.timeframes.len();
        if len == 0 {
            return;
        }
        let len_i = len as isize;
        self.tf_idx = (((self.tf_idx as isize + delta) % len_i + len_i) % len_i) as usize;
        self.loaded_key = None;
    }
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    shared.chart_state.ensure_loaded();
    let st = &shared.chart_state;

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .margin(1)
        .split(area);

    let sym = st.current_symbol().unwrap_or("—");
    let tf = st.current_tf().unwrap_or("—");
    let header = if st.candles.is_empty() {
        Line::from(vec![
            Span::styled(format!(" {sym} {tf} "), theme::accent_style()),
            Span::styled("   ←/→ symbol   ↑/↓ timeframe", theme::caption_style()),
        ])
    } else {
        let last = st.candles.last().map(|c| c.close).unwrap_or(0.0);
        let lo = st.candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let hi = st
            .candles
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        Line::from(vec![
            Span::styled(format!(" {sym} {tf} "), theme::accent_style()),
            Span::styled(
                format!(
                    "  {} bars · last {last:.5} · [{lo:.5} – {hi:.5}]",
                    st.candles.len()
                ),
                theme::muted_style(),
            ),
            Span::styled("   ←/→ symbol  ↑/↓ tf", theme::caption_style()),
        ])
    };
    Paragraph::new(header).render(rows[0], buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .style(theme::panel_block_style());

    if st.candles.is_empty() {
        let msg = if st.status.is_empty() {
            "Loading…".to_string()
        } else {
            st.status.clone()
        };
        let inner = block.inner(rows[1]);
        block.render(rows[1], buf);
        Paragraph::new(Line::styled(msg, theme::warn_style())).render(inner, buf);
        return;
    }

    let lo = st.candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let hi = st
        .candles
        .iter()
        .map(|c| c.high)
        .fold(f64::NEG_INFINITY, f64::max);
    // Pad the y-axis so wicks aren't clipped; handle a flat series.
    let (y_lo, y_hi) = if (hi - lo).abs() < f64::EPSILON {
        (lo - 1.0, hi + 1.0)
    } else {
        let pad = (hi - lo) * 0.05;
        (lo - pad, hi + pad)
    };
    let n = st.candles.len();
    // Snapshot into the owned closure so the paint fn doesn't borrow `st`.
    let candles = st.candles.clone();

    Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .x_bounds([0.0, n as f64])
        .y_bounds([y_lo, y_hi])
        .paint(move |ctx| {
            for (i, c) in candles.iter().enumerate() {
                let x = i as f64 + 0.5;
                let up = c.close >= c.open;
                let color = if up { theme::BUY } else { theme::SELL };
                // Wick.
                ctx.draw(&CanvasLine {
                    x1: x,
                    y1: c.low,
                    x2: x,
                    y2: c.high,
                    color,
                });
                // Body.
                let (body_lo, body_hi) = if up {
                    (c.open, c.close)
                } else {
                    (c.close, c.open)
                };
                ctx.draw(&Rectangle {
                    x: x - 0.32,
                    y: body_lo,
                    width: 0.64,
                    height: (body_hi - body_lo).max(f64::EPSILON),
                    color,
                });
            }
        })
        .render(rows[1], buf);
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    match code {
        KeyCode::Left | KeyCode::Char('h') => {
            shared.chart_state.step_symbol(-1);
            true
        }
        KeyCode::Right | KeyCode::Char('l') => {
            shared.chart_state.step_symbol(1);
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            shared.chart_state.step_tf(-1);
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            shared.chart_state.step_tf(1);
            true
        }
        _ => false,
    }
}
