//! Maps a signal + current positions to a `TradeIntent`.
//!
//! Phase-1 policy (intentionally simple — the loop, not the alpha, is what we're
//! proving here):
//!   - Flat signal  → close any open position on the symbol (exit).
//!   - Directional signal, no open position → Open with SL/TP bracketed around
//!     the current mark by a fixed fraction.
//!   - Directional signal opposite to the open position → close it (reversal;
//!     the re-open happens on the next bar once flat — no pyramiding in Phase 1).
//!   - Directional signal same side as the open position → hold (no-op).
//!
//! Sizing is `base_volume × confidence` (floored). The real correlation-aware
//! fractional-Kelly sizing (design §9 decision 4) replaces this in a later phase.

use crate::contracts::{CloseReason, Direction, Signal, TradeIntent};
use crate::position::Position;

/// Phase-1 decision policy parameters.
#[derive(Debug, Clone)]
pub struct DecisionConfig {
    /// Lots at full (confidence == 1.0) conviction.
    pub base_volume: f64,
    /// Never size below this (so a low-confidence signal still trades a token).
    pub min_volume: f64,
    /// Stop distance as a fraction of the mark price (e.g. 0.005 = 0.5%).
    pub stop_frac: f64,
    /// Take-profit distance as a multiple of the stop distance (R-multiple).
    pub tp_r_multiple: f64,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            base_volume: 1.0,
            min_volume: 0.01,
            stop_frac: 0.005,
            tp_r_multiple: 2.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecisionEngine {
    cfg: DecisionConfig,
}

impl DecisionEngine {
    pub fn new(cfg: DecisionConfig) -> Self {
        Self { cfg }
    }

    fn size(&self, confidence: f64) -> f64 {
        let c = confidence.clamp(0.0, 1.0);
        (self.cfg.base_volume * c).max(self.cfg.min_volume)
    }

    /// Bracket (sl, tp) around `mark` for `dir` using the configured stop
    /// fraction + R-multiple. Returns `None` for a Flat direction.
    fn bracket(&self, dir: Direction, mark: f64) -> Option<(f64, f64)> {
        if mark <= 0.0 {
            return None;
        }
        let stop = mark * self.cfg.stop_frac;
        match dir {
            Direction::Long => Some((mark - stop, mark + stop * self.cfg.tp_r_multiple)),
            Direction::Short => Some((mark + stop, mark - stop * self.cfg.tp_r_multiple)),
            Direction::Flat => None,
        }
    }

    /// Decide the single intent (if any) for this signal given the open
    /// positions on its symbol and the current `mark` price.
    pub fn intent(&self, signal: &Signal, open: &[Position], mark: f64) -> Option<TradeIntent> {
        let existing = open.iter().find(|p| p.symbol == signal.symbol);

        match (signal.dir, existing) {
            // No actionable signal: close an open position, else nothing.
            (Direction::Flat, Some(p)) => Some(TradeIntent::Close {
                position_id: p.id.clone(),
                volume: None,
                reason: CloseReason::Signal,
            }),
            (Direction::Flat, None) => None,

            // Directional, flat book → open a bracketed position.
            (dir, None) => {
                let (sl, tp) = self.bracket(dir, mark)?;
                Some(TradeIntent::Open {
                    symbol: signal.symbol.clone(),
                    dir,
                    volume: self.size(signal.confidence),
                    sl: Some(sl),
                    tp: Some(tp),
                    source: signal.source,
                })
            }

            // Directional against the open position → close (reversal).
            (dir, Some(p)) if p.dir != dir => Some(TradeIntent::Close {
                position_id: p.id.clone(),
                volume: None,
                reason: CloseReason::Signal,
            }),

            // Same side → hold.
            (_, Some(_)) => None,
        }
    }
}
