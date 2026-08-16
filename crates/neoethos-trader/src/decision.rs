//! Maps a signal + current positions to a `TradeIntent`.
//!
//! ## Bracket policy (audit #226, corrected 2026-08-09)
//!
//! The bracket comes from the SIGNAL when the signal carries one
//! ([`crate::contracts::Signal::sl_pips`] / `tp_pips` — the gene's own stop,
//! adaptive or fixed, exactly as [`crate::gene_signal::combine_gene_signals_with_brackets`]
//! reproduces it from the discovery run). Only a signal with NO bracket falls
//! back to the synthetic `stop_frac`-of-price bracket below, and that fallback
//! is now logged.
//!
//! **Why that matters.** `stop_frac` defaults to 0.005 — 0.5 % of price. On
//! EURUSD at 1.08 that is a **54-pip stop**, against a GA population whose
//! stops live in the 6–20 pip band. Every replay run before 2026-08-09 sized
//! its risk against a stop up to 9× wider than the strategy's, so its win rate,
//! payoff and drawdown described a strategy that does not exist.
//!
//! ## Exit policy (audit #228)
//!
//! The GA's evaluator (`neoethos-search` `eval.rs`) closes a trade on stop,
//! target, or `max_hold_bars`. It has NO "the signal went flat" exit and NO
//! "the signal reversed" exit. This engine had both, unconditionally, so a
//! replayed trade could be closed on a bar the backtest would have held.
//!
//! Both are now explicit knobs. [`DecisionConfig::default`] keeps the
//! historical Phase-1 behaviour (both ON) because the momentum-stub path has
//! no bracket and no time stop, so with both OFF it would never close anything.
//! [`DecisionConfig::gene_parity`] turns both OFF and is what the REAL-gene
//! replay paths use, together with the engine's `max_hold_bars` time stop.
//!
//! Sizing is `base_volume × confidence` (floored). The real correlation-aware
//! fractional-Kelly sizing (design §9 decision 4) replaces this in a later phase.

use crate::contracts::{CloseReason, Direction, Signal, TradeIntent};
use crate::position::Position;

/// Default synthetic stop as a fraction of price, used ONLY when the signal
/// carries no bracket of its own. See the module docs for why 0.005 is not a
/// neutral choice.
pub const DEFAULT_SYNTHETIC_STOP_FRAC: f64 = 0.005;

/// Phase-1 decision policy parameters.
#[derive(Debug, Clone)]
pub struct DecisionConfig {
    /// Lots at full (confidence == 1.0) conviction.
    pub base_volume: f64,
    /// Never size below this (so a low-confidence signal still trades a token).
    pub min_volume: f64,
    /// Stop distance as a fraction of the mark price (e.g. 0.005 = 0.5%).
    /// **Fallback only** — used when the signal carries no `sl_pips`.
    pub stop_frac: f64,
    /// Take-profit distance as a multiple of the stop distance (R-multiple).
    /// Applies to the synthetic bracket only; a signal-supplied bracket brings
    /// its own take-profit.
    pub tp_r_multiple: f64,
    /// Price movement of one pip for this symbol, used to convert a
    /// signal-supplied bracket (in pips) into price. `0.0` disables the
    /// signal-supplied bracket entirely and forces the synthetic fallback —
    /// which is announced, not silent.
    pub pip_size: f64,
    /// Close an open position when the signal goes Flat. The GA evaluator does
    /// NOT do this (audit #228).
    pub close_on_flat_signal: bool,
    /// Close an open position when the signal flips to the other side. The GA
    /// evaluator does NOT do this either (audit #228).
    pub close_on_opposite_signal: bool,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            base_volume: 1.0,
            min_volume: 0.01,
            stop_frac: DEFAULT_SYNTHETIC_STOP_FRAC,
            tp_r_multiple: 2.0,
            pip_size: 0.0,
            // Phase-1 stub behaviour, unchanged: the momentum stub has no
            // bracket and no time stop, so removing the signal exits here
            // would leave positions open for the whole replay.
            close_on_flat_signal: true,
            close_on_opposite_signal: true,
        }
    }
}

impl DecisionConfig {
    /// Backtest-parity policy for the REAL-gene replay paths (audit #228):
    /// exits are stop, target, or the engine's `max_hold_bars` time stop —
    /// the same three the GA evaluator uses. `pip_size` must be the symbol's
    /// exact broker `ProtoOASymbol` pip size or the gene bracket
    /// cannot be converted to price.
    pub fn gene_parity(pip_size: f64) -> Self {
        Self {
            pip_size,
            close_on_flat_signal: false,
            close_on_opposite_signal: false,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecisionEngine {
    cfg: DecisionConfig,
    /// One-shot latch so the synthetic-bracket fallback is announced once per
    /// engine rather than once per bar.
    synthetic_bracket_warned: bool,
}

impl DecisionEngine {
    pub fn new(cfg: DecisionConfig) -> Self {
        Self {
            cfg,
            synthetic_bracket_warned: false,
        }
    }

    /// Read-only view of the policy in force (so the replay harness can report
    /// exactly which exits and which bracket the numbers were produced under).
    pub fn config(&self) -> &DecisionConfig {
        &self.cfg
    }

    fn size(&self, confidence: f64) -> f64 {
        let c = confidence.clamp(0.0, 1.0);
        (self.cfg.base_volume * c).max(self.cfg.min_volume)
    }

    /// Bracket (sl, tp) around `mark` for `dir`.
    ///
    /// Prefers the STRATEGY's own bracket carried on the signal; falls back to
    /// the synthetic `stop_frac` bracket and says so, once. Returns `None` for
    /// a Flat direction or a non-positive mark.
    fn bracket(&mut self, signal: &Signal, dir: Direction, mark: f64) -> Option<(f64, f64)> {
        if mark <= 0.0 {
            return None;
        }
        let strategy_stop = if self.cfg.pip_size > 0.0
            && signal.sl_pips.is_finite()
            && signal.sl_pips > 0.0
        {
            Some(signal.sl_pips * self.cfg.pip_size)
        } else {
            None
        };

        let (stop, target) = match strategy_stop {
            Some(stop) => {
                // The gene's own take-profit when it has one; otherwise the
                // configured R-multiple of the gene's own stop. Never a
                // fraction of price.
                let target = if signal.tp_pips.is_finite() && signal.tp_pips > 0.0 {
                    signal.tp_pips * self.cfg.pip_size
                } else {
                    stop * self.cfg.tp_r_multiple
                };
                (stop, target)
            }
            None => {
                if !self.synthetic_bracket_warned {
                    self.synthetic_bracket_warned = true;
                    tracing::warn!(
                        target: "neoethos_trader::decision",
                        symbol = %signal.symbol,
                        stop_frac = self.cfg.stop_frac,
                        pip_size = self.cfg.pip_size,
                        implied_stop_price = mark * self.cfg.stop_frac,
                        "NO STRATEGY BRACKET on this signal — falling back to a SYNTHETIC \
                         stop of stop_frac x price. These are NOT the strategy's stops, so \
                         the win rate, payoff and drawdown this run reports describe a \
                         different strategy (audit #226)."
                    );
                }
                let stop = mark * self.cfg.stop_frac;
                (stop, stop * self.cfg.tp_r_multiple)
            }
        };

        match dir {
            Direction::Long => Some((mark - stop, mark + target)),
            Direction::Short => Some((mark + stop, mark - target)),
            Direction::Flat => None,
        }
    }

    /// Decide the single intent (if any) for this signal given the open
    /// positions on its symbol and the current `mark` price.
    pub fn intent(
        &mut self,
        signal: &Signal,
        open: &[Position],
        mark: f64,
    ) -> Option<TradeIntent> {
        let existing = open.iter().find(|p| p.symbol == signal.symbol).cloned();

        match (signal.dir, existing) {
            // No actionable signal: close an open position ONLY if this policy
            // says signal-flat is an exit (audit #228 — the GA's is not).
            (Direction::Flat, Some(p)) => {
                if self.cfg.close_on_flat_signal {
                    Some(TradeIntent::Close {
                        position_id: p.id.clone(),
                        volume: None,
                        reason: CloseReason::Signal,
                    })
                } else {
                    None
                }
            }
            (Direction::Flat, None) => None,

            // Directional, flat book → open a bracketed position.
            (dir, None) => {
                let (sl, tp) = self.bracket(signal, dir, mark)?;
                Some(TradeIntent::Open {
                    symbol: signal.symbol.clone(),
                    dir,
                    volume: self.size(signal.confidence),
                    sl: Some(sl),
                    tp: Some(tp),
                    source: signal.source,
                })
            }

            // Directional against the open position → close (reversal), ONLY
            // if this policy says a reversal is an exit (audit #228).
            (dir, Some(p)) if p.dir != dir => {
                if self.cfg.close_on_opposite_signal {
                    Some(TradeIntent::Close {
                        position_id: p.id.clone(),
                        volume: None,
                        reason: CloseReason::Signal,
                    })
                } else {
                    None
                }
            }

            // Same side → hold.
            (_, Some(_)) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::SignalSource;

    fn sig(dir: Direction, sl_pips: f64, tp_pips: f64) -> Signal {
        Signal {
            symbol: "EURUSD".to_string(),
            dir,
            confidence: 1.0,
            source: SignalSource::Strategy,
            sl_pips,
            tp_pips,
        }
    }

    fn open_long() -> Position {
        Position {
            id: "p1".to_string(),
            symbol: "EURUSD".to_string(),
            dir: Direction::Long,
            volume: 1.0,
            entry_price: 1.08,
            sl: None,
            tp: None,
            source: SignalSource::Strategy,
            opened_at_bar: 0,
            trail_px: None,
        }
    }

    /// The gene's own 10-pip stop must be used, NOT 0.5 % of 1.08 (=54 pips).
    #[test]
    fn gene_bracket_wins_over_the_synthetic_stop() {
        let mut eng = DecisionEngine::new(DecisionConfig::gene_parity(0.0001));
        let intent = eng
            .intent(&sig(Direction::Long, 10.0, 25.0), &[], 1.08)
            .expect("directional signal on a flat book must open");
        match intent {
            TradeIntent::Open { sl, tp, .. } => {
                assert!((sl.unwrap() - (1.08 - 0.0010)).abs() < 1e-9, "sl {sl:?}");
                assert!((tp.unwrap() - (1.08 + 0.0025)).abs() < 1e-9, "tp {tp:?}");
            }
            other => panic!("expected an Open, got {other:?}"),
        }
    }

    /// With no bracket on the signal the synthetic stop is used — that is the
    /// documented fallback, and it is 54 pips on this price.
    #[test]
    fn no_gene_bracket_falls_back_to_stop_frac() {
        let mut eng = DecisionEngine::new(DecisionConfig::gene_parity(0.0001));
        let intent = eng.intent(&sig(Direction::Long, 0.0, 0.0), &[], 1.08).unwrap();
        match intent {
            TradeIntent::Open { sl, .. } => {
                let stop = 1.08 - sl.unwrap();
                assert!((stop - 1.08 * DEFAULT_SYNTHETIC_STOP_FRAC).abs() < 1e-9);
            }
            other => panic!("expected an Open, got {other:?}"),
        }
    }

    /// Backtest parity: a flat or reversed signal must NOT close a position,
    /// because the GA evaluator has neither exit (audit #228).
    #[test]
    fn gene_parity_does_not_close_on_flat_or_reversal() {
        let mut eng = DecisionEngine::new(DecisionConfig::gene_parity(0.0001));
        let open = vec![open_long()];
        assert!(
            eng.intent(&sig(Direction::Flat, 0.0, 0.0), &open, 1.08).is_none(),
            "flat signal must not close under backtest parity"
        );
        assert!(
            eng.intent(&sig(Direction::Short, 10.0, 20.0), &open, 1.08).is_none(),
            "reversal must not close under backtest parity"
        );
    }

    /// The Phase-1 stub policy is unchanged: it still closes on both.
    #[test]
    fn default_policy_still_closes_on_flat_and_reversal() {
        let mut eng = DecisionEngine::default();
        let open = vec![open_long()];
        assert!(eng.intent(&sig(Direction::Flat, 0.0, 0.0), &open, 1.08).is_some());
        assert!(eng.intent(&sig(Direction::Short, 0.0, 0.0), &open, 1.08).is_some());
    }
}
