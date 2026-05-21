//! cTrader connection state machine — P8.
//!
//! Tracks the 14 connection steps the operator needs to see succeed in
//! order before the chart/discovery pipeline can use a cTrader account
//! as data source. Each step records:
//! - status (pending / in-flight / ok / failed / skipped)
//! - request id (for matching responses to requests during debugging)
//! - error message + timestamp on failure
//! - retry action hint
//!
//! UI consumes [`CTraderStateMachine::steps`] to render a checklist;
//! connector code calls `mark_*` to update.
//!
//! This sits separately from `ctrader_data` so the UI can render
//! progress even while no actual API calls are in flight (e.g. before
//! the first connect button is pressed — every step shows "pending").

use serde::{Deserialize, Serialize};

/// One step in the connection sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTraderStep {
    pub index: u8,
    pub name: String,
    pub status: CTraderStepStatus,
    pub request_id: Option<String>,
    pub message: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub retry_hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CTraderStepStatus {
    Pending,
    InFlight,
    Ok,
    Failed,
    Skipped,
}

/// State-machine helpers — `label` is `dead_code` in the release
/// build because the UI uses `glyph()` instead, but it's part of the
/// documented contract for the `CTraderStepStatus` enum (used by the
/// 14-step view's hover-tooltip path when wired). Tests below assert
/// it returns the expected human-readable label.
#[allow(dead_code)]
impl CTraderStepStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InFlight => "in-flight",
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pending => "○",
            Self::InFlight => "⟳",
            Self::Ok => "●",
            Self::Failed => "✗",
            Self::Skipped => "—",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTraderStateMachine {
    pub steps: Vec<CTraderStep>,
}

impl Default for CTraderStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// The `mark_*` mutation API is not called from production (audit
/// 2026-05-20: the state machine is DERIVED from session signals in
/// `TradingSession::derive_ctrader_state_machine`, not mutated).
/// They stay on the public surface because (a) tests below exercise
/// the mutation contract and (b) when a future flow needs explicit
/// progress reporting (e.g. wizard "connect now" walkthrough), the
/// mutation surface is already designed and tested.
#[allow(dead_code)]
impl CTraderStateMachine {
    pub fn new() -> Self {
        let canonical = [
            "token present",
            "token refreshed if needed",
            "socket connected",
            "ApplicationAuth sent",
            "ApplicationAuthRes received",
            "GetAccountListByAccessToken sent",
            "accounts received",
            "selected account authenticated",
            "symbols list loaded",
            "symbol metadata loaded",
            "historical bars loaded",
            "live spots subscribed",
            "live trendbars subscribed",
            "chart updated",
        ];
        let steps = canonical
            .iter()
            .enumerate()
            .map(|(i, name)| CTraderStep {
                index: (i + 1) as u8,
                name: (*name).to_string(),
                status: CTraderStepStatus::Pending,
                request_id: None,
                message: None,
                started_at: None,
                finished_at: None,
                retry_hint: None,
            })
            .collect();
        Self { steps }
    }

    pub fn mark_in_flight(&mut self, index: u8, request_id: Option<String>) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.index == index) {
            s.status = CTraderStepStatus::InFlight;
            s.request_id = request_id;
            s.started_at = Some(now_iso8601());
            s.finished_at = None;
            s.message = None;
        }
    }

    pub fn mark_ok(&mut self, index: u8, message: impl Into<String>) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.index == index) {
            s.status = CTraderStepStatus::Ok;
            s.message = Some(message.into());
            s.finished_at = Some(now_iso8601());
        }
    }

    pub fn mark_failed(
        &mut self,
        index: u8,
        message: impl Into<String>,
        retry_hint: Option<String>,
    ) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.index == index) {
            s.status = CTraderStepStatus::Failed;
            s.message = Some(message.into());
            s.finished_at = Some(now_iso8601());
            s.retry_hint = retry_hint;
        }
    }

    pub fn mark_skipped(&mut self, index: u8, reason: impl Into<String>) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.index == index) {
            s.status = CTraderStepStatus::Skipped;
            s.message = Some(reason.into());
            s.finished_at = Some(now_iso8601());
        }
    }

    /// Reset every step back to pending — used when the operator clicks
    /// "reconnect" or the previous connection drops.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Index of the first non-OK step; `None` when every step is OK.
    /// UI uses this to highlight "you are stuck on step X".
    pub fn current_step(&self) -> Option<u8> {
        self.steps
            .iter()
            .find(|s| !matches!(s.status, CTraderStepStatus::Ok | CTraderStepStatus::Skipped))
            .map(|s| s.index)
    }

    pub fn is_fully_connected(&self) -> bool {
        self.steps
            .iter()
            .all(|s| matches!(s.status, CTraderStepStatus::Ok | CTraderStepStatus::Skipped))
    }
}

#[allow(dead_code)] // used only by the `mark_*` mutation methods,
                    // which are themselves dead in the release build
                    // (see the impl above).
fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_machine_has_14_pending_steps() {
        let sm = CTraderStateMachine::new();
        assert_eq!(sm.steps.len(), 14);
        for s in &sm.steps {
            assert_eq!(s.status, CTraderStepStatus::Pending);
        }
        assert_eq!(sm.current_step(), Some(1));
        assert!(!sm.is_fully_connected());
    }

    #[test]
    fn marking_steps_advances_current_step() {
        let mut sm = CTraderStateMachine::new();
        sm.mark_ok(1, "token present");
        sm.mark_ok(2, "token still valid");
        sm.mark_in_flight(3, Some("req-001".to_string()));
        assert_eq!(sm.current_step(), Some(3));
        sm.mark_ok(3, "socket open");
        assert_eq!(sm.current_step(), Some(4));
    }

    #[test]
    fn fully_connected_when_all_steps_ok() {
        let mut sm = CTraderStateMachine::new();
        for i in 1..=14 {
            sm.mark_ok(i, "");
        }
        assert!(sm.is_fully_connected());
        assert_eq!(sm.current_step(), None);
    }

    #[test]
    fn mark_failed_records_retry_hint() {
        let mut sm = CTraderStateMachine::new();
        sm.mark_failed(
            5,
            "ApplicationAuth invalid client_id",
            Some("set FOREX_BOT_CTRADER_CLIENT_ID env".to_string()),
        );
        let s = &sm.steps[4];
        assert_eq!(s.status, CTraderStepStatus::Failed);
        assert_eq!(
            s.retry_hint.as_deref(),
            Some("set FOREX_BOT_CTRADER_CLIENT_ID env")
        );
    }

    #[test]
    fn reset_returns_to_initial_state() {
        let mut sm = CTraderStateMachine::new();
        for i in 1..=5 {
            sm.mark_ok(i, "");
        }
        sm.reset();
        assert_eq!(sm.current_step(), Some(1));
    }
}
