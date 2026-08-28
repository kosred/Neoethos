//! Chart — braille candlestick view of a symbol/timeframe's Vortex data.
//!
//! Inventories exact canonical identities without hashing data on every UI
//! refresh, then fully verifies and generation-pins the selected identity
//! before rendering its trailing N candles on a ratatui `Canvas`.
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
    inventory_entries: Vec<neoethos_data::DataFileEntry>,
    inventory_rejections: Vec<String>,
    symbols: Vec<String>,
    symbol_idx: usize,
    timeframes: Vec<String>,
    tf_idx: usize,
    candles: Vec<Candle>,
    /// `(symbol, timeframe)` currently loaded into `candles`; `None` means
    /// "needs (re)load on next draw".
    loaded_key: Option<(String, String, String)>,
    /// Human-readable status shown when there's nothing to plot. Always a
    /// message — never a panic.
    status: String,
}

impl ChartState {
    pub fn new(data_root: &std::path::Path) -> Self {
        let mut s = Self {
            data_root: data_root.to_path_buf(),
            inventory_entries: Vec::new(),
            inventory_rejections: Vec::new(),
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
        match neoethos_data::DatasetDiscovery::scan_metadata(&self.data_root) {
            Ok(report) => {
                self.inventory_rejections = report
                    .skipped
                    .into_iter()
                    .map(|skipped| {
                        format!(
                            "path={} category={} detail={:?}",
                            skipped.path.display(),
                            skipped.reason.category(),
                            skipped.reason
                        )
                    })
                    .collect();
                self.inventory_entries.clear();
                for entry in report.entries {
                    if entry.symbol.is_none() || entry.timeframe.is_none() {
                        self.inventory_rejections.push(format!(
                            "path={} category=invalid_inventory_entry detail=missing symbol/timeframe for identity={} generation={} manifest_binding_sha256={} verification={:?}",
                            entry.path.display(),
                            entry.dataset_identity,
                            entry.generation,
                            entry.manifest_binding_sha256,
                            entry.verification
                        ));
                    } else {
                        self.inventory_entries.push(entry);
                    }
                }
                self.symbols = self
                    .inventory_entries
                    .iter()
                    .filter_map(|entry| entry.symbol.clone())
                    .collect();
                self.symbols.sort();
                self.symbols.dedup();
                if self.symbols.is_empty() {
                    self.timeframes.clear();
                    let rejected = if self.inventory_rejections.is_empty() {
                        String::new()
                    } else {
                        format!(" Rejected: {}", self.inventory_rejections.join(" | "))
                    };
                    self.status = format!(
                        "No canonical dataset identities under {}.",
                        self.data_root.display()
                    ) + &rejected;
                    return;
                }
                if self.symbol_idx >= self.symbols.len() {
                    self.symbol_idx = 0;
                }
                self.refresh_timeframes();
            }
            Err(e) => {
                self.inventory_entries.clear();
                self.inventory_rejections.clear();
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
        self.timeframes = self
            .inventory_entries
            .iter()
            .filter(|entry| entry.symbol.as_deref() == Some(symbol.as_str()))
            .filter_map(|entry| entry.timeframe.clone())
            .collect();
        self.timeframes.sort_by(|left, right| {
            let code = |timeframe: &str| {
                timeframe
                    .parse::<neoethos_data::CanonicalTimeframe>()
                    .map(|value| value.ctrader_protocol_code())
                    .unwrap_or(i32::MAX)
            };
            code(left).cmp(&code(right))
        });
        self.timeframes.dedup();
        if self.timeframes.is_empty() {
            self.status = format!("{symbol}: no canonical manifest timeframes.");
        } else if self.tf_idx >= self.timeframes.len() {
            self.tf_idx = 0;
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
        let matching = self
            .inventory_entries
            .iter()
            .filter(|entry| {
                entry.symbol.as_deref() == Some(symbol.as_str())
                    && entry.timeframe.as_deref() == Some(tf.as_str())
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            self.candles.clear();
            self.loaded_key = None;
            let exact = matching
                .iter()
                .map(|entry| {
                    format!(
                        "identity={} generation={}",
                        entry.dataset_identity, entry.generation
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            self.status = format!(
                "{symbol} {tf}: expected exactly one canonical identity, found {}. {exact}",
                matching.len()
            );
            return;
        }
        let entry = (*matching[0]).clone();
        let key = (symbol.clone(), tf.clone(), entry.generation.clone());
        if self.loaded_key.as_ref() == Some(&key) {
            return; // already loaded — no per-frame disk thrash
        }
        self.candles.clear();
        self.loaded_key = None;
        self.status.clear();
        if entry.verification != neoethos_data::DataVerificationStatus::ManifestOnly {
            self.status = format!(
                "{symbol} {tf}: inventory entry has unexpected verification {:?}",
                entry.verification
            );
            return;
        }
        let identity = match neoethos_data::CanonicalDatasetIdentity::from_path_component(
            &entry.dataset_identity,
        ) {
            Ok(identity) => identity,
            Err(error) => {
                self.status = format!("{symbol} {tf}: invalid exact identity: {error}");
                return;
            }
        };
        let loaded = match neoethos_data::core::canonical_ohlcv::load_canonical_timeframe(
            &self.data_root,
            &identity,
        ) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.status = format!(
                    "{symbol} {tf} identity={} generation={}: {error:#}",
                    entry.dataset_identity, entry.generation
                );
                return;
            }
        };
        let binding = match loaded.artifact().source_binding("tui-chart-source") {
            Ok(binding) => binding,
            Err(error) => {
                self.status = format!("{symbol} {tf}: source binding failed: {error}");
                return;
            }
        };
        if binding.generation_id() != entry.generation.as_str()
            || hex_sha256(binding.manifest_hash()) != entry.manifest_binding_sha256
        {
            self.loaded_key = None;
            self.status = format!(
                "{symbol} {tf}: canonical generation changed after inventory; refresh before loading"
            );
            return;
        }
        let mut ohlcv = loaded.ohlcv().clone();
        let total = ohlcv.len();
        if total > TAIL {
            let drop = total - TAIL;
            ohlcv.open.drain(..drop);
            ohlcv.high.drain(..drop);
            ohlcv.low.drain(..drop);
            ohlcv.close.drain(..drop);
            if let Some(timestamps) = ohlcv.timestamp.as_mut() {
                timestamps.drain(..drop);
            }
            if let Some(volume) = ohlcv.volume.as_mut() {
                volume.drain(..drop);
            }
        }

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
        self.loaded_key = Some(key);
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

    fn inventory_notice(&self) -> String {
        let current = match (self.current_symbol(), self.current_tf()) {
            (Some(symbol), Some(timeframe)) => self
                .inventory_entries
                .iter()
                .filter(|entry| {
                    entry.symbol.as_deref() == Some(symbol)
                        && entry.timeframe.as_deref() == Some(timeframe)
                })
                .map(|entry| {
                    format!(
                        "identity={} generation={} manifest_binding_sha256={} verification={:?}",
                        entry.dataset_identity,
                        entry.generation,
                        entry.manifest_binding_sha256,
                        entry.verification
                    )
                })
                .collect::<Vec<_>>()
                .join(" | "),
            _ => String::new(),
        };
        if self.inventory_rejections.is_empty() {
            current
        } else if current.is_empty() {
            format!(
                "REJECTED {}",
                self.inventory_rejections.join(" | REJECTED ")
            )
        } else {
            format!(
                "{current} | REJECTED {}",
                self.inventory_rejections.join(" | REJECTED ")
            )
        }
    }
}

fn hex_sha256(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    shared.chart_state.ensure_loaded();
    let st = &shared.chart_state;
    let inventory_notice = st.inventory_notice();
    let notice_height = u16::from(!inventory_notice.is_empty());

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(notice_height),
            Constraint::Min(4),
        ])
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
        let lo = st
            .candles
            .iter()
            .map(|c| c.low)
            .fold(f64::INFINITY, f64::min);
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
    if !inventory_notice.is_empty() {
        Paragraph::new(Line::styled(inventory_notice, theme::caption_style())).render(rows[1], buf);
    }

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
        let inner = block.inner(rows[2]);
        block.render(rows[2], buf);
        Paragraph::new(Line::styled(msg, theme::warn_style())).render(inner, buf);
        return;
    }

    let lo = st
        .candles
        .iter()
        .map(|c| c.low)
        .fold(f64::INFINITY, f64::min);
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
        .render(rows[2], buf);
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
