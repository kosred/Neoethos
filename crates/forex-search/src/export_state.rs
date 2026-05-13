//! Validation/export state machine — P10.
//!
//! Replaces the binary "no strategies / portfolio.json" outcome with
//! the typed states the spec requires:
//!
//! ```text
//! NoCandidates       — GA found nothing in the archive
//! FiltersFailed      — candidates exist but none pass passes_filter
//! PortfolioSelected  — portfolio rows chosen but downstream gates not yet run
//! ValidationFailed   — walkforward/CPCV/prop-firm rejected the portfolio
//! ExportBlocked      — manifest/contract not yet ready
//! ExportReady        — artifacts written, manifest valid
//! ```
//!
//! Failed runs always save diagnostics — the funnel profile, best
//! rejected candidates, the config snapshot, and a one-line outcome
//! string — so the operator can debug from disk without re-running.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportState {
    NoCandidates,
    FiltersFailed,
    PortfolioSelected,
    ValidationFailed,
    ExportBlocked,
    ExportReady,
}

impl ExportState {
    pub fn label(self) -> &'static str {
        match self {
            Self::NoCandidates => "no_candidates",
            Self::FiltersFailed => "filters_failed",
            Self::PortfolioSelected => "portfolio_selected",
            Self::ValidationFailed => "validation_failed",
            Self::ExportBlocked => "export_blocked",
            Self::ExportReady => "export_ready",
        }
    }

    pub fn is_terminal_failure(self) -> bool {
        matches!(
            self,
            Self::NoCandidates | Self::FiltersFailed | Self::ValidationFailed | Self::ExportBlocked
        )
    }

    pub fn is_terminal_success(self) -> bool {
        matches!(self, Self::ExportReady)
    }

    /// Pick a state from the funnel counts. Used by the orchestrator
    /// when a work-unit finishes so it can report the precise reason
    /// rather than "empty portfolio".
    pub fn from_funnel(
        ranked: usize,
        passed_filter: usize,
        portfolio_size: usize,
        export_ready: bool,
    ) -> Self {
        if export_ready {
            return Self::ExportReady;
        }
        if portfolio_size > 0 {
            return Self::ValidationFailed;
        }
        if passed_filter > 0 {
            return Self::PortfolioSelected; // candidates exist past filter but
                                            // didn't survive prop-firm/corr gate
        }
        if ranked > 0 {
            return Self::FiltersFailed;
        }
        Self::NoCandidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn funnel_to_state_no_candidates_when_ranked_zero() {
        assert_eq!(
            ExportState::from_funnel(0, 0, 0, false),
            ExportState::NoCandidates
        );
    }

    #[test]
    fn funnel_to_state_filters_failed_when_ranked_but_no_filter() {
        assert_eq!(
            ExportState::from_funnel(50, 0, 0, false),
            ExportState::FiltersFailed
        );
    }

    #[test]
    fn funnel_to_state_export_ready_overrides_everything() {
        assert_eq!(
            ExportState::from_funnel(0, 0, 0, true),
            ExportState::ExportReady
        );
    }

    #[test]
    fn validation_failed_when_portfolio_picked_but_not_ready() {
        assert_eq!(
            ExportState::from_funnel(50, 10, 5, false),
            ExportState::ValidationFailed
        );
    }
}
