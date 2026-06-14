//! Phase-1 stub `SignalEngine`.
//!
//! A deterministic momentum rule over the rolling bar window — just enough to
//! drive the loop with varied Long/Short/Flat calls so the replay exercises
//! opens, reversals, and exits. Phase 4 replaces this with the real `Gene`
//! evaluation + `SoftVotingEnsemble` regime-conditional blend (design §7); the
//! `SignalEngine` trait is the seam, so nothing downstream changes.

use crate::contracts::{Direction, LiveBar, PortfolioEntry, Signal, SignalEngine, SignalSource};

/// Momentum stub: compares the latest close against the close `lookback` bars
/// ago. Up → Long, down → Short, equal/insufficient-history → Flat.
#[derive(Debug, Clone)]
pub struct MomentumStubSignal {
    lookback: usize,
    confidence: f64,
}

impl MomentumStubSignal {
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback: lookback.max(1),
            confidence: 0.6,
        }
    }
}

impl Default for MomentumStubSignal {
    fn default() -> Self {
        Self::new(3)
    }
}

impl SignalEngine for MomentumStubSignal {
    fn evaluate(&mut self, entry: &PortfolioEntry, window: &[LiveBar]) -> Signal {
        let dir = if window.len() <= self.lookback {
            Direction::Flat
        } else {
            let last = window[window.len() - 1].c;
            let prev = window[window.len() - 1 - self.lookback].c;
            if last > prev {
                Direction::Long
            } else if last < prev {
                Direction::Short
            } else {
                Direction::Flat
            }
        };
        Signal {
            symbol: entry.symbol.clone(),
            dir,
            confidence: self.confidence,
            source: SignalSource::Strategy,
        }
    }
}
