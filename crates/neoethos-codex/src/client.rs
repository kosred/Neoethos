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
                // **2026-05-26 fix (Κωνσταντίνος, Option B)**: enable
                // automatic cookie persistence across requests so any
                // `cf_clearance` cookie that Cloudflare issues after a
                // successful (browser-like) request is reused on the
                // next call. Without this, every request was a "cold"
                // visitor to Cloudflare's edge — challenge-eligible
                // and 403-prone. With this + the spoofed UA below in
                // `chat()`, the second-and-onward request normally
                // sails through.
                .cookie_store(true)
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
            // **2026-05-26 fix (Κωνσταντίνος)**: Cloudflare at chatgpt.com
            // returns 403 with anti-bot challenge HTML when the User-Agent
            // looks like a non-browser HTTP client (e.g. reqwest's default).
            // Spoofing a current Chrome desktop UA gets the request past
            // the JS-challenge gate in most cases. If Cloudflare still
            // 403s (e.g. rate-limited from this IP), the 403 handler
            // below now surfaces a friendly message instead of dumping
            // 16 KB of challenge HTML into the UI.
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                 AppleWebKit/537.36 (KHTML, like Gecko) \
                 Chrome/131.0.0.0 Safari/537.36",
            )
            .header("Accept-Language", "en-US,en;q=0.9")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            // **2026-05-26 fix**: detect Cloudflare anti-bot challenge
            // (always returned as HTML containing `cf_chl` markers) and
            // replace with a short operator-friendly message. Raw HTML
            // in the error string was making the AI-Helper screen show
            // a 16 KB unreadable blob; now it shows one actionable line.
            let body_lower = text.to_lowercase();
            let friendly = if status.as_u16() == 403
                && (body_lower.contains("cf_chl")
                    || body_lower.contains("cloudflare")
                    || body_lower.contains("<html"))
            {
                "ChatGPT subscription endpoint is rate-limited by \
                 Cloudflare. Wait ~6 minutes and try again, or use the \
                 ChatGPT web app from this machine first to refresh the \
                 anti-bot cookie."
                    .to_string()
            } else {
                text
            };
            return Err(CodexError::ApiCall {
                status: status.as_u16(),
                body: friendly,
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
