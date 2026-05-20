//! Error type for the forex-gemma helper crate.
//!
//! Phase G0. Every public API in this crate returns
//! `Result<T, GemmaError>` or `anyhow::Result<T>` (with
//! `GemmaError` convertible via the `?` operator). The variants
//! cover the failure modes the rest of the bot needs to
//! distinguish — primarily so the chrome / Flutter UI can render
//! the right message (refusal vs. internal failure vs. tool
//! rejection vs. "feature off").

use thiserror::Error;

/// Top-level error type for the Gemma helper.
///
/// Variants are deliberately coarse — fine-grained categorisation
/// (e.g. "embedding vector dimension mismatch") goes into the
/// `source` chain via `anyhow::Error`.
#[derive(Debug, Error)]
pub enum GemmaError {
    /// The inference runtime failed mid-generation. Examples:
    /// model file missing, GPU out of memory, tokenizer error.
    /// Distinct from a topic-gate refusal — this one means
    /// "something inside the bot broke" rather than "the user
    /// asked something we don't answer".
    #[error("Gemma inference failed: {reason}")]
    InferenceFailed { reason: String },

    /// The topic gate refused the input or the output. NOT an
    /// error condition from the user's perspective — the canned
    /// refusal text is the response. Surfaced as an error to the
    /// runtime so the audit log can record it distinctly from a
    /// successful answer.
    #[error("topic gate refused: {reason}")]
    TopicGateRefused {
        reason: String,
        canned_response: String,
    },

    /// Gemma tried to invoke a tool name that isn't registered.
    /// Most often a hallucinated function call.
    #[error("tool not found: {name}")]
    ToolNotFound { name: String },

    /// Gemma tried to invoke a gated tool while the gate is
    /// closed (e.g. trading tools disabled, rate-limit reached,
    /// per-call approval pending).
    #[error("tool denied: {name} — {reason}")]
    ToolDenied { name: String, reason: String },

    /// The audit log writer couldn't persist the row. Treated as
    /// a hard failure for trading-tool calls (no audit ⇒ no
    /// trade) but soft-warned for chat-only calls.
    #[error("audit log write failed: {reason}")]
    AuditWriteFailed { reason: String },

    /// The on-disk config or audit-log schema version is outside
    /// the readable range this build supports. Re-uses
    /// `forex_core::SchemaVersionError` semantics.
    #[error("schema version mismatch: {reason}")]
    SchemaVersionMismatch { reason: String },

    /// Generic configuration problem (invalid field combo,
    /// missing required field, etc.). Caught by validation at
    /// load time.
    #[error("Gemma config invalid: {reason}")]
    ConfigInvalid { reason: String },

    /// Caller used a Gemma API while the feature is off. Most
    /// often hit when `forex-app` is built without the
    /// `gemma-helper` feature but a call path through the
    /// stub-trait surface fires. The error message tells the
    /// operator exactly which feature to enable.
    #[error("Gemma helper is not enabled (rebuild with `--features gemma-helper`)")]
    NotEnabled,
}

impl GemmaError {
    /// Convenience constructor for fail-loud stubs in G0. Returns
    /// a `GemmaError::InferenceFailed` whose `reason` names the
    /// pending phase. Mirrors the cTrader / DXtrade stub style.
    pub fn pending(phase_tag: &str) -> Self {
        Self::InferenceFailed {
            reason: format!("{phase_tag} not yet implemented in this build"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_carries_phase_tag_in_message() {
        let err = GemmaError::pending("G1 inference");
        let msg = err.to_string();
        assert!(msg.contains("G1 inference"));
        assert!(msg.contains("not yet implemented"));
    }

    #[test]
    fn topic_gate_refused_keeps_canned_response_for_caller() {
        let err = GemmaError::TopicGateRefused {
            reason: "out-of-scope".to_string(),
            canned_response: "Μπορώ να βοηθήσω μόνο με ερωτήσεις σχετικά με το forex-ai."
                .to_string(),
        };
        if let GemmaError::TopicGateRefused {
            canned_response, ..
        } = &err
        {
            assert!(canned_response.contains("forex-ai"));
        } else {
            panic!("variant mismatch");
        }
    }

    #[test]
    fn variants_are_distinguishable_by_to_string() {
        // The chrome / Flutter UI dispatches on the error type
        // string, so the message prefixes must stay distinct.
        assert!(GemmaError::NotEnabled.to_string().contains("not enabled"));
        assert!(
            GemmaError::ToolNotFound {
                name: "x".to_string()
            }
            .to_string()
            .contains("tool not found")
        );
        assert!(
            GemmaError::ToolDenied {
                name: "submit_order".to_string(),
                reason: "rate-limit".to_string(),
            }
            .to_string()
            .contains("tool denied")
        );
    }
}
