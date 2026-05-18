//! Pending-suggestion framework — Role 2 of Gemma's trading
//! integration.
//!
//! Phase G0 — types + in-memory queue stub. G6b wires this to
//! the Flutter SSE event stream and the real `submit_order`
//! path with `OrderSource::AiSuggested` provenance.
//!
//! ## What this is
//!
//! When the user asks Gemma "what trade do you suggest on
//! EUR/USD?", Gemma calls a `GatedTrading` tool from the
//! catalog (see `tools.rs`). The tool **does not execute the
//! order**. Instead it creates a [`PendingSuggestion`], hands
//! it to the [`SuggestionQueue`], and returns a placeholder
//! result to Gemma. The Flutter chat UI subscribes to the
//! queue, renders the suggestion with Approve / Reject buttons,
//! and POSTs the user's decision back. Only on **Approve**
//! does the actual `submit_order` fire — through the existing
//! autonomous-trading path, tagged with
//! `OrderSource::AiSuggested` so Risky Mode treats it like any
//! other AI-origin order (every gate still applies).
//!
//! ## Two-role separation
//!
//! - **Role 1 — ensemble expert** (`expert.rs`): `GemmaExpert`
//!   votes inside `SoftVotingEnsemble`. Soft-vote decides the
//!   trade; no UI loop. This module is not involved.
//! - **Role 2 — conversational suggester** (this module): the
//!   chat UI is in the loop; nothing executes without an
//!   explicit user click.
//!
//! ## `OrderSource::AiSuggested` — future variant
//!
//! `forex_app::app_services::trading::OrderSource` today has
//! two variants: `Manual` and `Ai`. The suggested-trade path
//! needs a third — `AiSuggested` — so the audit / accounting
//! layer can distinguish "ensemble fired autonomously" from
//! "user clicked Approve on a Gemma suggestion". The variant
//! gets added in the G6b commit when the cross-crate wiring
//! lands. Until then, `PendingSuggestion::order_source` carries
//! a string discriminator (`"ai_suggested"`) so the audit
//! payload is forward-compatible.

use crate::error::GemmaError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Duration;

/// Direction shorthand for a trade suggestion. Mirrors the
/// `GemmaVoteDirection` from `expert.rs` but kept separate so
/// the wire shapes for Role 1 (ensemble vote) and Role 2 (user
/// suggestion) can evolve independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SuggestionSide {
    Buy,
    Sell,
}

/// A single Gemma-emitted trade proposal awaiting user decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingSuggestion {
    /// Stable ID the UI uses to POST the Approve / Reject
    /// decision back. Mint with `OsRng`-backed entropy in the
    /// runtime (same pattern as `generate_order_code` in the
    /// DXtrade adapter).
    pub suggestion_id: String,
    /// When Gemma created the suggestion. Used to detect
    /// timeouts.
    pub created_at_unix_ms: i64,
    /// Hard expiry. After this UTC instant the suggestion is no
    /// longer actionable — the UI must clear it and the queue
    /// must mark it `TimedOut`.
    pub expires_at_unix_ms: i64,
    /// Symbol the proposal applies to, in slash format
    /// (`"EUR/USD"`, `"XAU/USD"`).
    pub symbol: String,
    pub side: SuggestionSide,
    /// Volume in broker-canonical units. The same shape
    /// `DxTradeNewOrder.volume` carries.
    pub volume: i64,
    /// Optional limit price (for LIMIT orders). `None` ⇒
    /// market order.
    #[serde(default)]
    pub limit_price: Option<f64>,
    /// Optional stop-loss in price units.
    #[serde(default)]
    pub stop_loss_price: Option<f64>,
    /// Optional take-profit in price units.
    #[serde(default)]
    pub take_profit_price: Option<f64>,
    /// Short reasoning string Gemma produced — shown in the UI
    /// alongside the buttons so the user has context. ≤ 280
    /// chars enforced at queue-push time.
    pub reasoning: String,
    /// Provenance tag carried through to the audit trail. G0
    /// uses the string `"ai_suggested"` because the real
    /// `OrderSource::AiSuggested` variant lands in G6b along
    /// with the cross-crate wiring.
    #[serde(default = "default_order_source")]
    pub order_source: String,
}

fn default_order_source() -> String {
    "ai_suggested".to_string()
}

/// Maximum allowed `reasoning` length, in characters.
pub const SUGGESTION_REASONING_MAX_CHARS: usize = 280;

/// Outcome of a `PendingSuggestion` after the user has had a
/// chance to decide. Maps directly onto the audit-log payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SuggestionResolution {
    /// User clicked Approve. The real `submit_order` follows.
    Approved { approved_at_unix_ms: i64 },
    /// User clicked Reject. No trade fires.
    Rejected {
        rejected_at_unix_ms: i64,
        /// Optional free-text the user typed in the rejection
        /// dialog. Audit-logged for post-mortem analysis.
        #[serde(default)]
        reason: Option<String>,
    },
    /// Expired without a decision. No trade fires; queue
    /// removes the suggestion.
    TimedOut { expired_at_unix_ms: i64 },
}

/// Queue trait. The runtime owns one; the tool layer pushes
/// suggestions in; the API layer (G8) drains them out to the
/// SSE stream. G0 ships an in-memory implementation for tests;
/// G6b wires the real production queue (still in-memory but
/// hooked to `crossbeam_channel::Sender<PendingSuggestion>` for
/// the SSE bridge).
pub trait SuggestionQueue: Send + Sync {
    /// Push a new suggestion. Returns the assigned
    /// `suggestion_id` (the queue may overwrite a caller-
    /// provided ID to ensure uniqueness).
    fn push(&self, suggestion: PendingSuggestion) -> Result<String, GemmaError>;

    /// Resolve a suggestion by ID with the user's decision.
    /// Returns `Ok(true)` when the suggestion existed and was
    /// updated, `Ok(false)` when it had already expired /
    /// resolved.
    fn resolve(
        &self,
        suggestion_id: &str,
        resolution: SuggestionResolution,
    ) -> Result<bool, GemmaError>;

    /// Snapshot all currently-pending suggestions, oldest-first.
    /// Useful for the API surface to bootstrap a freshly-opened
    /// chat UI ("here's what's still waiting on you").
    fn pending(&self) -> Result<Vec<PendingSuggestion>, GemmaError>;

    /// Number of pending suggestions. Useful for the rate-cap
    /// check in the tool layer.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory queue — G0 stub plus what G6b actually ships as
/// the production queue (the SSE bridge in G8 will just attach
/// a `crossbeam_channel` sender alongside).
pub struct InMemorySuggestionQueue {
    inner: Mutex<Vec<PendingSuggestion>>,
}

impl InMemorySuggestionQueue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }
}

impl Default for InMemorySuggestionQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl SuggestionQueue for InMemorySuggestionQueue {
    fn push(&self, suggestion: PendingSuggestion) -> Result<String, GemmaError> {
        validate(&suggestion)?;
        let id = suggestion.suggestion_id.clone();
        let mut q = self
            .inner
            .lock()
            .map_err(|_| poisoned("suggestion queue mutex poisoned"))?;
        // Reject collisions defensively — the caller's RNG
        // SHOULD prevent these but a hash-style fallback in the
        // runtime might emit a duplicate.
        if q.iter().any(|s| s.suggestion_id == id) {
            return Err(GemmaError::ToolDenied {
                name: "suggestion_queue.push".to_string(),
                reason: format!("duplicate suggestion_id {id}"),
            });
        }
        q.push(suggestion);
        Ok(id)
    }

    fn resolve(
        &self,
        suggestion_id: &str,
        _resolution: SuggestionResolution,
    ) -> Result<bool, GemmaError> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| poisoned("suggestion queue mutex poisoned"))?;
        if let Some(idx) = q.iter().position(|s| s.suggestion_id == suggestion_id) {
            q.remove(idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn pending(&self) -> Result<Vec<PendingSuggestion>, GemmaError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| poisoned("suggestion queue mutex poisoned"))?
            .clone())
    }

    fn len(&self) -> usize {
        self.inner.lock().map(|q| q.len()).unwrap_or(0)
    }
}

fn poisoned(msg: &str) -> GemmaError {
    GemmaError::AuditWriteFailed {
        reason: msg.to_string(),
    }
}

fn validate(s: &PendingSuggestion) -> Result<(), GemmaError> {
    if s.suggestion_id.trim().is_empty() {
        return Err(GemmaError::ToolDenied {
            name: "suggestion_queue.push".to_string(),
            reason: "suggestion_id is empty".to_string(),
        });
    }
    if s.symbol.trim().is_empty() {
        return Err(GemmaError::ToolDenied {
            name: "suggestion_queue.push".to_string(),
            reason: "symbol is empty".to_string(),
        });
    }
    if s.volume <= 0 {
        return Err(GemmaError::ToolDenied {
            name: "suggestion_queue.push".to_string(),
            reason: format!("volume must be positive, got {}", s.volume),
        });
    }
    if s.reasoning.chars().count() > SUGGESTION_REASONING_MAX_CHARS {
        return Err(GemmaError::ToolDenied {
            name: "suggestion_queue.push".to_string(),
            reason: format!("reasoning exceeds {SUGGESTION_REASONING_MAX_CHARS} chars",),
        });
    }
    if s.expires_at_unix_ms <= s.created_at_unix_ms {
        return Err(GemmaError::ToolDenied {
            name: "suggestion_queue.push".to_string(),
            reason: "expires_at must be strictly after created_at".to_string(),
        });
    }
    Ok(())
}

/// Helper: compute `expires_at_unix_ms` from a base instant +
/// the operator-configured timeout. Pure function; testable.
pub fn compute_expiry(created_at: DateTime<Utc>, timeout: Duration) -> i64 {
    let after = created_at + chrono::Duration::from_std(timeout).unwrap_or_default();
    after.timestamp_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str) -> PendingSuggestion {
        PendingSuggestion {
            suggestion_id: id.to_string(),
            created_at_unix_ms: 1_700_000_000_000,
            expires_at_unix_ms: 1_700_000_060_000,
            symbol: "EUR/USD".to_string(),
            side: SuggestionSide::Buy,
            volume: 100_000,
            limit_price: None,
            stop_loss_price: Some(1.05),
            take_profit_price: Some(1.12),
            reasoning: "ensemble shows trend up, RSI oversold".to_string(),
            order_source: default_order_source(),
        }
    }

    #[test]
    fn push_and_pending_round_trip_one_suggestion() {
        let q = InMemorySuggestionQueue::new();
        let id = q.push(sample("s-1")).expect("push ok");
        assert_eq!(id, "s-1");
        assert_eq!(q.len(), 1);
        let pending = q.pending().expect("pending");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].suggestion_id, "s-1");
        assert_eq!(pending[0].order_source, "ai_suggested");
    }

    #[test]
    fn push_rejects_duplicate_suggestion_id() {
        let q = InMemorySuggestionQueue::new();
        q.push(sample("s-1")).unwrap();
        let err = q.push(sample("s-1")).expect_err("must bail");
        assert!(
            matches!(err, GemmaError::ToolDenied { ref reason, .. } if reason.contains("duplicate"))
        );
    }

    #[test]
    fn push_rejects_empty_symbol() {
        let q = InMemorySuggestionQueue::new();
        let mut s = sample("s-1");
        s.symbol.clear();
        let err = q.push(s).expect_err("must bail");
        assert!(
            matches!(err, GemmaError::ToolDenied { ref reason, .. } if reason.contains("symbol"))
        );
    }

    #[test]
    fn push_rejects_non_positive_volume() {
        let q = InMemorySuggestionQueue::new();
        let mut s = sample("s-1");
        s.volume = 0;
        assert!(q.push(s).is_err());
        let mut s2 = sample("s-2");
        s2.volume = -1;
        assert!(q.push(s2).is_err());
    }

    #[test]
    fn push_rejects_reasoning_over_char_cap() {
        let q = InMemorySuggestionQueue::new();
        let mut s = sample("s-1");
        s.reasoning = "x".repeat(SUGGESTION_REASONING_MAX_CHARS + 1);
        let err = q.push(s).expect_err("must bail");
        assert!(
            matches!(err, GemmaError::ToolDenied { ref reason, .. } if reason.contains("reasoning"))
        );
    }

    #[test]
    fn push_rejects_expiry_at_or_before_created() {
        let q = InMemorySuggestionQueue::new();
        let mut s = sample("s-1");
        s.expires_at_unix_ms = s.created_at_unix_ms; // equal
        assert!(q.push(s).is_err());
        let mut s2 = sample("s-2");
        s2.expires_at_unix_ms = s2.created_at_unix_ms - 1; // before
        assert!(q.push(s2).is_err());
    }

    #[test]
    fn resolve_drops_the_suggestion_from_pending() {
        let q = InMemorySuggestionQueue::new();
        q.push(sample("s-1")).unwrap();
        let ok = q
            .resolve(
                "s-1",
                SuggestionResolution::Approved {
                    approved_at_unix_ms: 1_700_000_010_000,
                },
            )
            .unwrap();
        assert!(ok);
        assert!(q.is_empty());
    }

    #[test]
    fn resolve_unknown_id_returns_false() {
        let q = InMemorySuggestionQueue::new();
        let ok = q
            .resolve(
                "missing",
                SuggestionResolution::TimedOut {
                    expired_at_unix_ms: 0,
                },
            )
            .unwrap();
        assert!(!ok);
    }

    #[test]
    fn pre_versioning_resolution_round_trips_through_json() {
        // Pin the resolution wire shape — the UI consumes this
        // via SSE.
        let r = SuggestionResolution::Rejected {
            rejected_at_unix_ms: 42,
            reason: Some("price drifted".to_string()),
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("rejected"));
        let back: SuggestionResolution = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn compute_expiry_is_purely_additive() {
        let base = DateTime::<Utc>::from_timestamp_millis(1_700_000_000_000).unwrap();
        let after = compute_expiry(base, Duration::from_secs(60));
        assert_eq!(after, 1_700_000_060_000);
    }

    #[test]
    fn order_source_defaults_to_ai_suggested_when_missing() {
        // Forward-compat: a wire payload without `order_source`
        // (older runtime) must default to "ai_suggested".
        let raw = r#"{
            "suggestion_id": "s-1",
            "created_at_unix_ms": 1,
            "expires_at_unix_ms": 2,
            "symbol": "EUR/USD",
            "side": "BUY",
            "volume": 100,
            "reasoning": "test"
        }"#;
        let parsed: PendingSuggestion = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.order_source, "ai_suggested");
    }
}
