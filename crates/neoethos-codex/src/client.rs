//! Chat-completions client backed by a ChatGPT subscription token.
//!
//! Wraps a `reqwest::Client` + a [`StoredAuth`] and exposes
//! [`CodexClient::chat`] — a single function that takes a Chat
//! Completions request shape and returns the same response
//! shape. The plumbing handles:
//!
//! - Auto-refresh: if the access_token has < 60s left, we refresh
//!   via [`crate::oauth::refresh_token`] before sending. The new
//!   bundle is persisted via the supplied [`AuthStore`].
//! - Error transparency: a 401 from the API is reported as
//!   `CodexError::ApiCall { status: 401, ... }` so the front-end
//!   can prompt "click Re-auth ChatGPT" instead of "unknown error".
//!
//! The on-the-wire request shape mirrors the familiar Chat Completions
//! API. ChatGPT-subscription accounts talk to the privileged Codex
//! backend (`chatgpt.com/backend-api/codex/responses`) with the
//! `gpt-5.5` model identifier; we encode that in the `model` default
//! but otherwise pass the request through.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::auth_store::{AuthStore, StoredAuth};
use crate::error::CodexError;
use crate::oauth::refresh_token;
use crate::CODEX_API_BASE;

/// Process-stable installation ID sent on every `/codex/responses` request
/// via the `x-codex-installation-id` header. The official Codex CLI
/// generates one on first launch and persists it under `~/.codex/`; we
/// just create one per backend process so the header is well-formed and
/// stable across the session. We do NOT try to mimic the persistence
/// scheme because (a) collisions with a real `~/.codex/install.json`
/// would corrupt that file for users who also run the official CLI and
/// (b) the server doesn't appear to enforce continuity across processes —
/// the header is used for telemetry / per-install rate-limiting, not
/// auth (auth is the Bearer token).
///
/// Format mirrors the CLI: 32 lowercase hex chars (= 16 random bytes).
fn process_installation_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rng().fill_bytes(&mut bytes);
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    })
}

/// One conversational turn. Matches the Chat Completions shape exactly.
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
    /// Defaults to "gpt-5.5" — the ONLY model the ChatGPT-subscription
    /// Codex endpoint accepts (see [`ChatCompletionRequest::simple`]).
    /// Callers can override, but every other name we tested returns
    /// HTTP 400 "not supported when using Codex with a ChatGPT account".
    ///
    /// **F-291 (2026-05-29)**: verified empirically against
    /// `/backend-api/codex/responses` with a live subscription token —
    /// every other model identifier we tried returned HTTP 400; only
    /// `gpt-5.5` is accepted on the subscription path.
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
            // F-291: gpt-5.5 is the only model the ChatGPT-subscription
            // Codex endpoint accepts.
            model: "gpt-5.5".to_string(),
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

    /// Force a token refresh regardless of expiry (audit B15) — used when the
    /// server rejects the current token with 401 even though our local
    /// freshness check thought it was valid (revoked server-side, clock skew,
    /// or expired in the send window).
    async fn force_refresh(&self, auth: &StoredAuth) -> Result<StoredAuth, CodexError> {
        let refresh = auth
            .refresh_token
            .as_ref()
            .ok_or(CodexError::NotAuthenticated)?
            .expose()
            .to_string();
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
    /// that `/backend-api/conversation` accepts the Chat
    /// Completions body was wrong — that endpoint is the chat-UI's
    /// internal flow and demands a UUID/author/parts wrapper, returning
    /// SSE. The official Codex CLI privileged path is actually
    /// `/backend-api/codex/responses`, which speaks the simpler
    /// **Responses API** format. We map our existing
    /// ChatCompletion{Request,Response} shapes to/from that format so
    /// the call-site signature stays unchanged — only the wire goes
    /// through the right pipe.
    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, CodexError> {
        let mut auth = self.current_auth().await?;

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
            // **F-291**: MUST be true — the endpoint rejects a
            // non-streaming request ("Stream must be set to true"). We
            // still drain the whole SSE stream synchronously below; the
            // AI Helper panel doesn't render incremental tokens yet.
            stream: true,
            // `store: false` matches the Codex CLI's stateless default —
            // we send the whole prompt every time and the server doesn't
            // persist a thread for us to reference later.
            store: false,
            // **F-291 Cycle 5 (2026-05-31)**: `max_output_tokens` is
            // NOT accepted by the ChatGPT-subscription Codex
            // `/responses` endpoint — it returns HTTP 400
            // "Unsupported parameter: max_output_tokens" the moment the
            // UI passes a non-null token cap. (The earlier smoke test
            // missed it because it sent a prompt with no token cap.) The
            // endpoint caps output server-side, so we simply don't send
            // the field.
        };

        let url = format!("{CODEX_API_BASE}/codex/responses");
        // B15: send, and on a 401 refresh the token ONCE and retry — a token
        // that expired in the send window or was revoked server-side now
        // recovers automatically instead of failing the operator's request.
        // `status()` only borrows the response, so we inspect it for the retry
        // decision and still read the body below.
        let mut refreshed = false;
        let response = loop {
            let resp = self.send_responses(&auth, &api_request, &url).await?;
            if resp.status().as_u16() == 401 && !refreshed && auth.refresh_token.is_some() {
                refreshed = true;
                auth = self.force_refresh(&auth).await?;
                continue;
            }
            break resp;
        };

        let status = response.status();
        let text = response.text().await?;
        return self.finish_chat(&request, status, text);
    }

    /// Build + send ONE Responses-API request. Split out of `chat` so the 401
    /// refresh-retry (B15) can re-send with a fresh token.
    async fn send_responses(
        &self,
        auth: &StoredAuth,
        api_request: &ResponsesApiRequest,
        url: &str,
    ) -> Result<reqwest::Response, CodexError> {
        let response = self
            .http
            .post(url)
            .bearer_auth(auth.access_token.expose())
            // **F-291**: SSE response — ask for the event stream.
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            // **2026-05-29 (F-291 Cycle 3 research)**: the official Codex
            // CLI source (github.com/openai/codex/codex-rs/core/src/
            // client.rs) sets three identity headers the backend
            // appears to require for the `/codex/responses` privileged
            // path. Without them the server returns
            // `"The 'X' model is not supported when using Codex with a
            // ChatGPT account."` for every model name, including the
            // ones the user's plan *does* entitle. Adding them at the
            // very least makes us look like a real CLI install:
            //
            //   - `OAI-Product-Sku: codex`  →  routes the request to
            //     the privileged Codex path (vs. plain ChatGPT chat).
            //   - `x-codex-installation-id: <32-hex>` →  per-install
            //     telemetry / rate-limiting key. We generate a random
            //     stable-per-process ID since we have no `~/.codex/
            //     install.json` to read.
            //   - `OpenAI-Beta: responses=v1` →  enables the synchronous
            //     Responses-API path (the WS path uses a different
            //     beta value `responses_websockets=2026-02-06`).
            //
            // KNOWN LIMITATION: even with all three set, the server may
            // still 400 on every model name if the operator's ChatGPT
            // plan doesn't include Codex entitlement (Plus has limited
            // Codex; Pro/Team/Enterprise have more). In that case the
            // friendly-error fallback below is the best UX we can give —
            // the only remediation is for the operator to upgrade to (or
            // sign in with) a ChatGPT plan that includes Codex
            // entitlement. We never fall back to a paid API key.
            .header("OpenAI-Beta", "responses=v1")
            .header("OAI-Product-Sku", "codex")
            .header("x-codex-installation-id", process_installation_id())
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
            .json(api_request)
            .send()
            .await?;
        Ok(response)
    }

    /// Turn a drained Responses-API reply (status + body) into the final chat
    /// result: map non-2xx to friendly operator errors, else parse the SSE
    /// stream. Split out of `chat` so the send can be retried (B15) before this
    /// runs. Not async — the body is already read and the SSE parse is pure.
    fn finish_chat(
        &self,
        request: &ChatCompletionRequest,
        status: reqwest::StatusCode,
        text: String,
    ) -> Result<ChatCompletionResponse, CodexError> {
        if !status.is_success() {
            // **2026-05-26 fix**: detect Cloudflare anti-bot challenge
            // (always returned as HTML containing `cf_chl` markers) and
            // replace with a short operator-friendly message. Raw HTML
            // in the error string was making the AI-Helper screen show
            // a 16 KB unreadable blob; now it shows one actionable line.
            //
            // **2026-05-29 (F-291)**: also intercept the per-account
            // model-not-supported 400 ("The 'X' model is not supported
            // when using Codex with a ChatGPT account.") and replace
            // with an actionable line that names the user-facing
            // remediation (plan upgrade / model picker / API key
            // fallback). Without this, the AI Helper just shows the
            // raw 400 body which doesn't tell the operator what to do.
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
            } else if status.as_u16() == 400
                && body_lower.contains("not supported when using codex")
            {
                format!(
                    "Your ChatGPT plan does not entitle '{model}' via the \
                     Codex API. Codex access is currently limited to \
                     ChatGPT Pro / Team / Enterprise plans (Plus has \
                     restricted access). Either upgrade the plan or sign \
                     in with a different account that has Codex \
                     entitlement.",
                    model = request.model,
                )
            } else {
                text
            };
            return Err(CodexError::ApiCall {
                status: status.as_u16(),
                body: friendly,
            });
        }

        // **F-291**: the success body is an SSE event stream (the
        // `text()` above already drained it to completion since we don't
        // surface incremental tokens yet). Aggregate the `output_text`
        // deltas + usage and remap to the Chat Completions shape our
        // caller (server::codex::chat) expects.
        // B15: a terminal error / truncated stream now fails loudly (HTTP 502)
        // instead of returning an empty "success".
        let (assistant_text, usage) =
            parse_sse_response(&text).map_err(|body| CodexError::ApiCall { status: 502, body })?;
        Ok(ChatCompletionResponse {
            id: None,
            model: Some(request.model.clone()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: assistant_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage,
        })
    }
}

// ─── Responses API wire types ──────────────────────────────────────────────
//
// These are the on-the-wire shapes for the
// `/backend-api/codex/responses` endpoint. They are internal to this
// crate: callers see the legacy `ChatCompletion{Request,Response}`
// API only.

/// Fallback `instructions` when the caller didn't supply a system
/// message. The Codex `/responses` endpoint REQUIRES a non-empty
/// `instructions` string ("Instructions are required") so we never
/// send an empty one.
const DEFAULT_INSTRUCTIONS: &str =
    "You are NeoEthos AI Helper, a concise assistant embedded in a Rust \
     forex trading terminal. Answer clearly and briefly. You can discuss \
     trading, markets, economics, and how to use the platform.";

#[derive(Debug, Serialize)]
struct ResponsesApiRequest {
    model: String,
    /// **F-291 (2026-05-29)**: MUST be a list of message objects. A
    /// bare string is rejected with "Input must be a list".
    input: Vec<ResponsesInputItem>,
    /// **F-291**: REQUIRED and non-empty. Sending none (or an empty
    /// string) returns "Instructions are required".
    instructions: String,
    /// **F-291**: MUST be `true`. The endpoint only speaks SSE; a
    /// non-streaming request returns "Stream must be set to true".
    /// We drain the whole event stream synchronously in `chat()` and
    /// aggregate the text deltas (no incremental UI surface yet).
    stream: bool,
    /// Whether ChatGPT should persist this turn in server-side memory.
    /// Codex CLI sends `false` for stateless use.
    store: bool,
    // **F-291 Cycle 5**: no `max_output_tokens` — the ChatGPT-account
    // `/responses` endpoint rejects it with HTTP 400 "Unsupported
    // parameter: max_output_tokens". Output is capped server-side.
}

/// One message in the Responses-API `input` list.
#[derive(Debug, Serialize, PartialEq)]
struct ResponsesInputItem {
    role: String,
    content: Vec<ResponsesInputContent>,
}

/// One content part inside an input message. The Codex endpoint
/// expects parts typed `input_text`.
#[derive(Debug, Serialize, PartialEq)]
struct ResponsesInputContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

/// Parse a Responses-API SSE body into `(aggregated_text, usage)`.
///
/// The endpoint emits Server-Sent Event frames, one JSON object per
/// `data:` line:
/// ```text
/// event: response.output_text.delta
/// data: {"type":"response.output_text.delta","delta":"Hello",...}
///
/// event: response.completed
/// data: {"type":"response.completed","response":{"usage":{...}}}
/// ```
/// We concatenate every `response.output_text.delta` `delta`, and read
/// token usage from the terminal `response.completed` frame when
/// present. If no deltas arrived we fall back to the cumulative `text`
/// carried by a `response.output_text.done` frame so a reply is never
/// silently dropped. Unknown frame types are ignored — the wire format
/// gains event types over time and we only depend on the text ones.
///
/// Returns the assistant text + usage on a
/// clean stream, or `Err(message)` when the stream carried a TERMINAL ERROR or
/// was TRUNCATED (audit B15). Previously this silently returned whatever
/// partial text it had scraped — so a mid-stream server error or a cut
/// connection looked like a successful (empty/partial) answer to the operator.
fn parse_sse_response(body: &str) -> Result<(String, Option<ChatUsage>), String> {
    let mut text = String::new();
    let mut done_text: Option<String> = None;
    let mut usage: Option<ChatUsage> = None;
    let mut saw_terminal = false; // response.completed / response.done / [DONE]

    for line in body.lines() {
        let payload = match line.strip_prefix("data:") {
            Some(p) => p.trim(),
            None => continue,
        };
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            saw_terminal = true;
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue, // tolerate keep-alive / partial frames
        };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("response.output_text.delta") => {
                if let Some(d) = v.get("delta").and_then(|d| d.as_str()) {
                    text.push_str(d);
                }
            }
            Some("response.output_text.done") => {
                if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                    done_text = Some(t.to_string());
                }
            }
            // B15: a terminal error event must fail the call, not be ignored.
            Some("error") | Some("response.failed") | Some("response.error") => {
                let msg = v
                    .pointer("/response/error/message")
                    .or_else(|| v.pointer("/error/message"))
                    .or_else(|| v.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("the AI provider reported a stream error");
                return Err(format!("Codex stream error: {msg}"));
            }
            Some("response.completed") | Some("response.done") => {
                saw_terminal = true;
                if let Some(u) = v.pointer("/response/usage") {
                    let input_tokens =
                        u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                            as u32;
                    let output_tokens =
                        u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                            as u32;
                    let total_tokens = u
                        .get("total_tokens")
                        .and_then(|x| x.as_u64())
                        .unwrap_or((input_tokens + output_tokens) as u64)
                        as u32;
                    usage = Some(ChatUsage {
                        prompt_tokens: input_tokens,
                        completion_tokens: output_tokens,
                        total_tokens,
                    });
                }
            }
            _ => {}
        }
    }

    // Prefer the streamed deltas; fall back to the done-frame's
    // cumulative text only when no deltas were seen.
    let final_text = if text.is_empty() {
        done_text.unwrap_or_default()
    } else {
        text
    };
    // B15: a stream that produced NO text and never reached a terminal frame
    // was truncated (dropped connection mid-answer). Surface it as an error
    // instead of returning an empty "success".
    if final_text.is_empty() && !saw_terminal {
        return Err(
            "Codex stream ended without a reply (connection truncated). Try again.".to_string(),
        );
    }
    Ok((final_text, usage))
}

/// Split Chat-Completions-shaped messages into the Responses-API pair
/// `(instructions, input)`. System messages collapse into the single
/// `instructions` string (defaulting to [`DEFAULT_INSTRUCTIONS`] when
/// absent — the endpoint requires non-empty); every other role becomes
/// an input message object. Public for the unit tests in this crate;
/// not part of the external API.
fn split_messages(messages: &[ChatMessage]) -> (String, Vec<ResponsesInputItem>) {
    let mut instructions: Vec<&str> = Vec::new();
    let mut input: Vec<ResponsesInputItem> = Vec::new();
    for m in messages {
        match m.role.to_ascii_lowercase().as_str() {
            "system" => instructions.push(m.content.as_str()),
            role => input.push(ResponsesInputItem {
                role: if role == "assistant" {
                    "assistant".to_string()
                } else {
                    "user".to_string()
                },
                content: vec![ResponsesInputContent {
                    content_type: "input_text".to_string(),
                    text: m.content.clone(),
                }],
            }),
        }
    }
    let instructions = if instructions.is_empty() {
        DEFAULT_INSTRUCTIONS.to_string()
    } else {
        instructions.join("\n\n")
    };
    (instructions, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_request_defaults_to_user_message() {
        let req = ChatCompletionRequest::simple("hello world");
        assert_eq!(req.model, "gpt-5.5");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hello world");
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn serializes_without_optional_nulls() {
        let req = ChatCompletionRequest::simple("hi");
        let json = serde_json::to_string(&req).unwrap();
        // **F-291 (2026-05-29 Cycle 4)**: pinned to `gpt-5.5` — the only
        // model the live ChatGPT-subscription Codex endpoint accepts.
        // (Was `gpt-5`, which 400s with the per-account error; the
        // 2026-05-26 `gpt-5-codex`→`gpt-5` switch was a half-fix.)
        assert!(json.contains("\"model\":\"gpt-5.5\""));
        assert!(json.contains("\"messages\""));
        // Skipped if None — these should NOT appear in the body.
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn process_installation_id_is_stable_and_hex() {
        // The header must round-trip the CLI's 32-hex shape; if it isn't,
        // the server may reject the request as malformed before even
        // reaching the model-entitlement check.
        let a = process_installation_id();
        let b = process_installation_id();
        assert_eq!(a, b, "installation id must be process-stable");
        assert_eq!(a.len(), 32, "installation id must be 32 hex chars");
        assert!(
            a.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "installation id must be lowercase hex only: {a}"
        );
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
        assert_eq!(instructions, "Be concise.");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0].role, "user");
        assert_eq!(input[0].content[0].content_type, "input_text");
        assert_eq!(input[0].content[0].text, "Hello");
    }

    #[test]
    fn split_messages_defaults_instructions_when_no_system() {
        // **F-291**: the endpoint requires a non-empty `instructions`
        // string, so a conversation with no system message must still
        // produce one (the default), and every non-system turn becomes
        // its own input message object with its role preserved.
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
        assert_eq!(instructions, DEFAULT_INSTRUCTIONS);
        assert_eq!(input.len(), 3);
        assert_eq!(input[0].role, "user");
        assert_eq!(input[1].role, "assistant");
        assert_eq!(input[2].role, "user");
        assert_eq!(input[1].content[0].text, "Foreign exchange.");
    }

    #[test]
    fn responses_request_serializes_as_input_list_with_instructions() {
        // **F-291** wire contract: `input` is a list, `instructions` is
        // a non-empty string, `stream` is true. Any of these wrong and
        // the endpoint 400s before it even checks model entitlement.
        let (instructions, input) = split_messages(&[ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }]);
        let req = ResponsesApiRequest {
            model: "gpt-5.5".to_string(),
            input,
            instructions,
            stream: true,
            store: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"input\":["), "input must serialize as a list");
        assert!(json.contains("\"type\":\"input_text\""));
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"instructions\":\""));
        assert!(
            !json.contains("\"instructions\":\"\""),
            "instructions must be non-empty"
        );
        // **F-291 Cycle 5 regression guard**: the endpoint 400s on
        // `max_output_tokens` ("Unsupported parameter"). It must never
        // appear on the wire.
        assert!(
            !json.contains("max_output_tokens"),
            "max_output_tokens is rejected by the /responses endpoint"
        );
    }

    #[test]
    fn parse_sse_aggregates_text_deltas_and_usage() {
        let body = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n",
            "\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\", world\"}\n",
            "\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":3,\"total_tokens\":13}}}\n",
            "\n",
        );
        let (text, usage) = parse_sse_response(body).expect("clean stream");
        assert_eq!(text, "Hello, world");
        let u = usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 3);
        assert_eq!(u.total_tokens, 13);
    }

    #[test]
    fn parse_sse_falls_back_to_done_frame_when_no_deltas() {
        let body = concat!(
            "event: response.output_text.done\n",
            "data: {\"type\":\"response.output_text.done\",\"text\":\"Full reply.\"}\n",
            "\n",
        );
        let (text, usage) = parse_sse_response(body).expect("done frame");
        assert_eq!(text, "Full reply.");
        assert!(usage.is_none());
    }

    #[test]
    fn parse_sse_tolerates_keepalive_and_garbage_lines() {
        let body = concat!(
            ": keep-alive comment\n",
            "data: \n",
            "data: not-json\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n",
            "data: [DONE]\n",
        );
        let (text, _usage) = parse_sse_response(body).expect("has terminal [DONE]");
        assert_eq!(text, "ok");
    }

    #[test]
    fn parse_sse_surfaces_a_terminal_error_event() {
        // B15: a mid-stream error event must fail, not return partial text.
        let body = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n",
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"message\":\"rate limit exceeded\"}}\n",
        );
        let err = parse_sse_response(body).expect_err("error event must fail");
        assert!(err.contains("rate limit exceeded"), "got: {err}");
    }

    #[test]
    fn parse_sse_flags_a_truncated_stream() {
        // B15: a stream with no text and no terminal frame = truncated.
        let body = ": keep-alive only, connection dropped\n";
        let err = parse_sse_response(body).expect_err("truncation must fail");
        assert!(err.contains("truncated"), "got: {err}");
    }
}
