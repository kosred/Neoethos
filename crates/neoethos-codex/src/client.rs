//! Chat-completions client backed by a ChatGPT subscription token.
//!
//! Wraps a `reqwest::Client` + a [`StoredAuth`] and exposes
//! [`CodexClient::chat`] — a single function that takes the OpenAI
//! Chat Completions request shape and returns the same response
//! shape. The plumbing handles:
//!
//! - Auto-refresh: if the access_token has < 60s left, we refresh
//!   via [`crate::oauth::refresh_token`] before sending. The new
//!   bundle is persisted via the supplied [`AuthStore`].
//! - Error transparency: a 401 from the API is reported as
//!   `CodexError::ApiCall { status: 401, ... }` so the front-end
//!   can prompt "click Re-auth ChatGPT" instead of "unknown error".
//!
//! The on-the-wire shape mirrors the OpenAI public Chat Completions
//! API. ChatGPT-subscription accounts use a slightly different
//! backend (`chatgpt.com/backend-api/conversation` rather than
//! `api.openai.com/v1/chat/completions`) and a different model
//! identifier (`gpt-5-codex` etc.); we encode the difference in
//! `model` defaults but otherwise pass the request through.

use serde::{Deserialize, Serialize};

use crate::auth_store::{AuthStore, StoredAuth};
use crate::error::CodexError;
use crate::oauth::refresh_token;
use crate::CODEX_API_BASE;

/// One conversational turn. Matches the OpenAI shape exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// "system", "user", or "assistant".
    pub role: String,
    pub content: String,
}

/// Request body for [`CodexClient::chat`]. Optional fields fall back
/// to sensible defaults; the only required value is `messages`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    /// Defaults to "gpt-5-codex" — the model the Codex CLI uses
    /// against ChatGPT-subscription accounts. Callers can override
    /// for `gpt-5-thinking` or any other model the account has.
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Streaming is NOT supported yet — we send `stream=false`
    /// implicitly. A future PR can add SSE support; until then,
    /// callers get the full response in one HTTP body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl ChatCompletionRequest {
    pub fn simple(prompt: &str) -> Self {
        Self {
            model: "gpt-5-codex".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: None,
            temperature: None,
            stream: Some(false),
        }
    }
}

/// What the API returns. We model only the fields the AI Helper UI
/// renders — everything else lives in `raw` for forward-compat.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Thin client that owns the access_token + store + HTTP client.
/// Cheap to construct (single `reqwest::Client::new()`); the caller
/// may keep one per HTTP server lifetime.
pub struct CodexClient {
    http: reqwest::Client,
    store: AuthStore,
}

impl CodexClient {
    pub fn new(store: AuthStore) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("reqwest client build failed — only fails on TLS setup, which is fatal"),
            store,
        }
    }

    /// Load the current auth, refreshing if needed. Returns the
    /// fresh bundle (persisted) — never `None`. Callers that want
    /// to handle "not logged in" should check `store.load()`
    /// themselves first.
    async fn current_auth(&self) -> Result<StoredAuth, CodexError> {
        let auth = self.store.load()?.ok_or(CodexError::NotAuthenticated)?;
        if !auth.is_expired() {
            return Ok(auth);
        }
        let refresh = match auth.refresh_token.as_ref() {
            Some(r) => r.expose().to_string(),
            None => {
                // No refresh token — operator must redo the OAuth flow.
                return Err(CodexError::NotAuthenticated);
            }
        };
        let bundle = refresh_token(&refresh).await?;
        let new_auth = StoredAuth::from_bundle(bundle);
        self.store.save(&new_auth)?;
        Ok(new_auth)
    }

    /// Send a chat request. Returns the parsed response on 2xx, or
    /// `CodexError::ApiCall { status, body }` for anything else.
    /// The caller maps non-2xx into HTTP responses for the UI.
    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, CodexError> {
        let auth = self.current_auth().await?;

        // ChatGPT subscription accounts hit /backend-api/conversation.
        // It accepts a superset of the public chat completions body —
        // sending a Chat Completions–shaped JSON works as long as we
        // include the `model` field.
        let url = format!("{CODEX_API_BASE}/conversation");
        let response = self
            .http
            .post(&url)
            .bearer_auth(auth.access_token.expose())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("OpenAI-Beta", "codex-cli")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            return Err(CodexError::ApiCall {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: ChatCompletionResponse = serde_json::from_str(&text)?;
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_request_defaults_to_user_message() {
        let req = ChatCompletionRequest::simple("hello world");
        assert_eq!(req.model, "gpt-5-codex");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hello world");
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn serializes_without_optional_nulls() {
        let req = ChatCompletionRequest::simple("hi");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"gpt-5-codex\""));
        assert!(json.contains("\"messages\""));
        // Skipped if None — these should NOT appear in the body.
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("temperature"));
    }
}
