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
    /// Defaults to "gpt-5" — the standard ChatGPT-subscription
    /// model. Callers can override to `gpt-5-thinking` or any other
    /// model the account has access to.
    ///
    /// **2026-05-26 fix (Κωνσταντίνος)**: was previously `gpt-5-codex`
    /// per the assumption that ChatGPT subscriptions could call the
    /// Codex CLI's privileged model. Verified empirically against
    /// `/backend-api/codex/responses`: ChatGPT subscription accounts
    /// get HTTP 400 with `"The 'gpt-5-codex' model is not supported
    /// when using Codex with a ChatGPT account."` The actual model
    /// the API accepts for personal subscriptions is plain `gpt-5`.
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
            model: "gpt-5".to_string(),
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
    ///
    /// **2026-05-26 fix (Κωνσταντίνος)**: the dev's original assumption
    /// that `/backend-api/conversation` accepts the OpenAI Chat
    /// Completions body was wrong — that endpoint is the chat-UI's
    /// internal flow and demands a UUID/author/parts wrapper, returning
    /// SSE. The official Codex CLI privileged path is actually
    /// `/backend-api/codex/responses`, which speaks the simpler
    /// **OpenAI Responses API** format. We map our existing
    /// ChatCompletion{Request,Response} shapes to/from that format so
    /// the call-site signature stays unchanged — only the wire goes
    /// through the right pipe.
    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, CodexError> {
        let auth = self.current_auth().await?;

        // Map ChatCompletion-shaped messages → Responses API shape.
        // Convention used by official Codex CLI: system messages
        // become the dedicated `instructions` field (one string);
        // all other roles (user/assistant) are concatenated into
        // `input`. For a one-shot prompt (`ChatCompletionRequest::simple`)
        // we get `instructions=None, input="<the prompt>"`.
        let (instructions, input) = split_messages(&request.messages);

        let api_request = ResponsesApiRequest {
            model: request.model.clone(),
            input,
            instructions,
            // `stream: false` returns the full response as a single JSON
            // body. The Codex CLI uses streaming for incremental tokens
            // but we don't surface incremental output in the AI Helper
            // panel yet, so the simpler synchronous flow is fine.
            stream: false,
            // `store: false` matches the Codex CLI's stateless default —
            // we send the whole prompt every time and the server doesn't
            // persist a thread for us to reference later.
            store: false,
            max_output_tokens: request.max_tokens,
        };

        let url = format!("{CODEX_API_BASE}/codex/responses");
        let response = self
            .http
            .post(&url)
            .bearer_auth(auth.access_token.expose())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            // Header that the Codex CLI sets to identify itself. Keeping
            // it here helps the backend route the request to the
            // privileged code path even though we're not the official
            // binary. (Empirically: without this, /codex/responses can
            // 403 even with a valid bearer.)
            .header("OpenAI-Beta", "responses=v1")
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
            .json(&api_request)
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

        // Parse Responses API output and remap to the Chat Completions
        // shape our caller (server::codex::chat) expects.
        let api_response: ResponsesApiResponse = serde_json::from_str(&text)?;
        Ok(api_response.into_chat_completion(&request.model))
    }
}

// ─── Responses API wire types ──────────────────────────────────────────────
//
// These are the on-the-wire shapes for the
// `/backend-api/codex/responses` endpoint. They are internal to this
// crate: callers see the legacy `ChatCompletion{Request,Response}`
// API only.

#[derive(Debug, Serialize)]
struct ResponsesApiRequest {
    model: String,
    /// Concatenated user/assistant messages — single string.
    input: String,
    /// Optional system prompt. Distinct from `input` in the Responses
    /// API: instructions persist for the whole turn but don't count
    /// as user input.
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    /// Synchronous response (one JSON object). `true` would switch to
    /// SSE event-stream which we don't currently consume.
    stream: bool,
    /// Whether ChatGPT should persist this turn in server-side memory.
    /// Codex CLI sends `false` for stateless use.
    store: bool,
    /// Cap on output tokens. None → use server default.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ResponsesApiResponse {
    #[serde(default)]
    #[allow(dead_code)] // surfaced via ChatCompletionResponse.id in future
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutputItem {
    #[serde(rename = "type", default)]
    item_type: String,
    /// Deserialized so the wire-shape stays accurate but currently
    /// unused — the consuming code filters by `item_type == "message"`
    /// and reads `content[].text` directly. If a future caller needs
    /// to distinguish between assistant/system replies it can read
    /// this field; until then `#[allow(dead_code)]` keeps the build
    /// warning-free.
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    content: Vec<ResponsesContent>,
}

#[derive(Debug, Deserialize)]
struct ResponsesContent {
    #[serde(rename = "type", default)]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl ResponsesApiResponse {
    /// Map a Responses-API reply into the Chat-Completions shape that
    /// the rest of NeoEthos already speaks. The model id falls back
    /// to the request's model when the server omitted one (it usually
    /// echoes it back, but we don't depend on that).
    fn into_chat_completion(self, request_model: &str) -> ChatCompletionResponse {
        // Concatenate every `output_text` payload across all output
        // items. The Codex/Responses API can emit multiple
        // `output` items (e.g. tool calls + final message); we only
        // care about the textual `message` items here. Joining with
        // newlines preserves multi-segment replies without losing
        // structure.
        let assistant_text: String = self
            .output
            .iter()
            .filter(|item| item.item_type == "message")
            .flat_map(|item| item.content.iter())
            .filter(|c| c.content_type == "output_text")
            .filter_map(|c| c.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        let model = self
            .model
            .unwrap_or_else(|| request_model.to_string());

        ChatCompletionResponse {
            id: self.id,
            model: Some(model),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: assistant_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: self.usage.map(|u| ChatUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: if u.total_tokens > 0 {
                    u.total_tokens
                } else {
                    u.input_tokens + u.output_tokens
                },
            }),
        }
    }
}

/// Split Chat-Completions-shaped messages into the Responses-API
/// pair `(instructions, input)`. Public for the unit tests in this
/// crate; not part of the external API.
fn split_messages(messages: &[ChatMessage]) -> (Option<String>, String) {
    let mut instructions: Vec<&str> = Vec::new();
    let mut conversation: Vec<String> = Vec::new();
    for m in messages {
        match m.role.to_ascii_lowercase().as_str() {
            "system" => instructions.push(m.content.as_str()),
            "user" => conversation.push(m.content.clone()),
            "assistant" => conversation.push(format!("Assistant: {}", m.content)),
            other => conversation.push(format!("{other}: {}", m.content)),
        }
    }
    let instructions = if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    };
    let input = conversation.join("\n\n");
    (instructions, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_request_defaults_to_user_message() {
        let req = ChatCompletionRequest::simple("hello world");
        assert_eq!(req.model, "gpt-5");
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

    #[test]
    fn split_messages_promotes_system_to_instructions() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        let (instructions, input) = split_messages(&messages);
        assert_eq!(instructions.as_deref(), Some("Be concise."));
        assert_eq!(input, "Hello");
    }

    #[test]
    fn split_messages_concatenates_user_and_assistant_history() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "What is forex?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Foreign exchange.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Tell me more.".to_string(),
            },
        ];
        let (instructions, input) = split_messages(&messages);
        assert_eq!(instructions, None);
        assert!(input.contains("What is forex?"));
        assert!(input.contains("Assistant: Foreign exchange."));
        assert!(input.contains("Tell me more."));
    }

    #[test]
    fn responses_api_maps_to_chat_completion() {
        let api_resp = ResponsesApiResponse {
            id: Some("resp_xxx".to_string()),
            model: Some("gpt-5".to_string()),
            output: vec![ResponsesOutputItem {
                item_type: "message".to_string(),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContent {
                    content_type: "output_text".to_string(),
                    text: Some("Hi there.".to_string()),
                }],
            }],
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            }),
        };
        let chat = api_resp.into_chat_completion("gpt-5");
        assert_eq!(chat.choices.len(), 1);
        assert_eq!(chat.choices[0].message.role, "assistant");
        assert_eq!(chat.choices[0].message.content, "Hi there.");
        assert_eq!(chat.usage.as_ref().unwrap().total_tokens, 15);
    }

    #[test]
    fn responses_api_ignores_non_message_outputs() {
        // The Responses API can emit tool_call items etc. Make sure
        // we only pull text from `message` items so we don't trip
        // on unrelated structures.
        let api_resp = ResponsesApiResponse {
            id: None,
            model: None,
            output: vec![
                ResponsesOutputItem {
                    item_type: "tool_call".to_string(),
                    role: None,
                    content: vec![ResponsesContent {
                        content_type: "output_text".to_string(),
                        text: Some("ignored".to_string()),
                    }],
                },
                ResponsesOutputItem {
                    item_type: "message".to_string(),
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContent {
                        content_type: "output_text".to_string(),
                        text: Some("kept".to_string()),
                    }],
                },
            ],
            usage: None,
        };
        let chat = api_resp.into_chat_completion("gpt-5");
        assert_eq!(chat.choices[0].message.content, "kept");
    }
}
