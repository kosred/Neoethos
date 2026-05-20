//! HTTP / SSE API surface that the Flutter client talks to.
//!
//! Phase G0 — request/response types only, no live HTTP handlers.

use crate::gate::LanguageHint;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub prompt: String,
    #[serde(default)]
    pub language: LanguageHint,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    RefusedByGate {
        reason: String,
        canned_response: String,
    },
    TokenDelta {
        text: String,
    },
    ToolCallStarted {
        tool_name: String,
        args: serde_json::Value,
    },
    ToolResult {
        tool_name: String,
        outcome: String,
        result: Option<serde_json::Value>,
    },
    /// Gemma emitted a trade suggestion. The UI renders Approve / Reject;
    /// the user's decision goes back via `POST /gemma/suggestion/decide`.
    /// The trade does NOT execute until the user clicks Approve.
    TradePendingApproval {
        suggestion_id: String,
        symbol: String,
        side: String,
        volume: i64,
        #[serde(default)]
        limit_price: Option<f64>,
        #[serde(default)]
        stop_loss_price: Option<f64>,
        #[serde(default)]
        take_profit_price: Option<f64>,
        reasoning: String,
        expires_at_unix_ms: i64,
    },
    TurnFinished {
        latency_ms: i64,
        audit_id: Option<String>,
    },
    Errored {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureFlagToggle {
    pub flag: FeatureFlag,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFlag {
    AuditFullText,
    TradingTools,
    PerTradeApproval,
    GatedTools,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuggestionDecision {
    pub suggestion_id: String,
    pub approved: bool,
    #[serde(default)]
    pub reject_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_round_trips_through_json() {
        let req = ChatRequest {
            session_id: "s-1".to_string(),
            prompt: "What does the ensemble say?".to_string(),
            language: LanguageHint::English,
            max_tokens: Some(256),
        };
        let s = serde_json::to_string(&req).expect("ser");
        let back: ChatRequest = serde_json::from_str(&s).expect("de");
        assert_eq!(back, req);
    }

    #[test]
    fn chat_request_accepts_missing_optional_fields() {
        let json = r#"{"session_id":"s","prompt":"hi"}"#;
        let req: ChatRequest = serde_json::from_str(json).expect("de");
        assert_eq!(req.language, LanguageHint::Unknown);
        assert_eq!(req.max_tokens, None);
    }

    #[test]
    fn chat_event_token_delta_serializes_as_tagged_enum() {
        let evt = ChatEvent::TokenDelta {
            text: "hello".to_string(),
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains(r#""kind":"token_delta""#));
        assert!(s.contains(r#""text":"hello""#));
    }

    #[test]
    fn chat_event_variants_are_distinguishable_in_json() {
        let evts = vec![
            ChatEvent::RefusedByGate {
                reason: "x".to_string(),
                canned_response: "y".to_string(),
            },
            ChatEvent::ToolCallStarted {
                tool_name: "list_positions".to_string(),
                args: serde_json::json!({}),
            },
            ChatEvent::ToolResult {
                tool_name: "list_positions".to_string(),
                outcome: "ok".to_string(),
                result: Some(serde_json::json!({"count": 0})),
            },
            ChatEvent::TradePendingApproval {
                suggestion_id: "sug-1".to_string(),
                symbol: "EUR/USD".to_string(),
                side: "BUY".to_string(),
                volume: 100_000,
                limit_price: None,
                stop_loss_price: Some(1.05),
                take_profit_price: Some(1.12),
                reasoning: "trend up".to_string(),
                expires_at_unix_ms: 1_700_000_060_000,
            },
            ChatEvent::TurnFinished {
                latency_ms: 1234,
                audit_id: Some("audit-1".to_string()),
            },
            ChatEvent::Errored {
                reason: "bug".to_string(),
            },
        ];
        for e in evts {
            let json = serde_json::to_string(&e).unwrap();
            let back: ChatEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, e);
        }
    }

    #[test]
    fn feature_flag_toggle_pins_known_flags() {
        for f in [
            FeatureFlag::AuditFullText,
            FeatureFlag::TradingTools,
            FeatureFlag::PerTradeApproval,
            FeatureFlag::GatedTools,
        ] {
            let toggle = FeatureFlagToggle {
                flag: f,
                enabled: true,
            };
            let s = serde_json::to_string(&toggle).unwrap();
            let back: FeatureFlagToggle = serde_json::from_str(&s).unwrap();
            assert_eq!(back, toggle);
        }
    }

    #[test]
    fn suggestion_decision_round_trips_through_json() {
        let req = SuggestionDecision {
            suggestion_id: "sug-1".to_string(),
            approved: false,
            reject_reason: Some("price drifted".to_string()),
        };
        let s = serde_json::to_string(&req).unwrap();
        let back: SuggestionDecision = serde_json::from_str(&s).unwrap();
        assert_eq!(back, req);
    }
}
