//! `/auth/codex/*` and `/codex/chat` HTTP routes — ChatGPT
//! subscription fallback for the AI Helper.
//!
//! Flow:
//!   1. UI calls `POST /auth/codex/start`. We generate a fresh
//!      `AuthorizationRequest`, stash it in `AppApiState`, spawn the
//!      loopback callback listener on 127.0.0.1:1455, and return the
//!      authorize URL to the front-end. The front-end opens that URL
//!      in the operator's default browser.
//!   2. Operator signs in at auth.openai.com. The issuer redirects
//!      to `http://localhost:1455/auth/callback?code=...&state=...`.
//!      Our listener captures the query, verifies state matches,
//!      exchanges the code for tokens, persists to `~/.codex/auth.json`,
//!      and returns "complete".
//!   3. UI polls `GET /auth/codex/status` every second until status
//!      becomes `complete` (or `error`). On complete it flips the AI
//!      Helper to "ChatGPT" mode.
//!   4. `POST /codex/chat` proxies a chat request through the stored
//!      bearer token.
//!
//! All state for an in-flight OAuth attempt lives in
//! `AppApiState::codex_auth_state` (a `Mutex<Option<...>>`). At most
//! one login can be in progress at a time — clicking Connect twice
//! quickly returns an error from the second call instead of leaking
//! a second callback listener.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use neoethos_codex::{
    AuthStore, AuthorizationRequest, CallbackResult, CallbackServer, CodexClient,
    ChatCompletionRequest, ChatMessage, StoredAuth, exchange_code,
};

use super::state::AppApiState;

// ─── GET /auth/codex/status ───────────────────────────────────────────────

/// Status DTO consumed by the Flutter Settings panel.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexStatusDto {
    /// True when `~/.codex/auth.json` exists and has a non-empty
    /// access_token. We do NOT verify the token is still valid here
    /// (that would force a network call on every render); we just
    /// report what's on disk. The first chat request triggers a
    /// refresh if the token has expired.
    pub authenticated: bool,
    /// Cached email from the id_token claims. Lets the UI show
    /// "Signed in as foo@example.com" without decoding the JWT
    /// itself.
    pub email: Option<String>,
    /// True iff a login flow is currently in progress (the operator
    /// has clicked Connect but hasn't completed the browser
    /// redirect). UI uses this to show a "waiting for browser" hint
    /// instead of "click Connect again".
    pub login_in_progress: bool,
    /// Last error from the most recent attempt, if any. Cleared when
    /// the operator starts a fresh attempt.
    pub last_error: Option<String>,
    /// On-disk path to `auth.json`. Useful for the UI's "Open Codex
    /// config folder" button.
    pub auth_path: String,
}

pub async fn status(State(state): State<AppApiState>) -> Json<CodexStatusDto> {
    let store = AuthStore::at_default();
    let stored = store.load().ok().flatten();
    let codex_state = state.codex.lock().await;
    let (login_in_progress, last_error) = match codex_state.as_ref() {
        Some(s) => (s.in_flight.is_some(), s.last_error.clone()),
        None => (false, None),
    };

    Json(CodexStatusDto {
        authenticated: stored.is_some(),
        email: stored.as_ref().and_then(|s| s.email.clone()),
        login_in_progress,
        last_error,
        auth_path: store.path().display().to_string(),
    })
}

// ─── POST /auth/codex/start ────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexStartDto {
    /// URL the front-end must open in the operator's default
    /// browser. We never call `webbrowser::open` from here because
    /// the operator's launcher preferences live on the desktop side
    /// (Flutter does it with `url_launcher`).
    pub authorize_url: String,
    /// Loopback port the callback listener bound to. Echoed back so
    /// the UI can show "waiting on http://localhost:1455 ..." in the
    /// status string. Always 1455 today.
    pub callback_port: u16,
}

pub async fn start(State(state): State<AppApiState>) -> Response {
    // Reject overlapping logins so we don't end up with two listeners
    // racing on the same port and a confused user.
    {
        let codex_state = state.codex.lock().await;
        if let Some(s) = codex_state.as_ref() {
            if s.in_flight.is_some() {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "A ChatGPT login is already in progress. \
                                 Complete or close the browser tab first."
                    })),
                )
                    .into_response();
            }
        }
    }

    let request = AuthorizationRequest::new();
    let authorize_url = request.build_authorize_url();
    let callback_port = neoethos_codex::CODEX_CALLBACK_PORT;

    // Bind the listener BEFORE returning so a port-busy error is
    // surfaced synchronously instead of leaking to a background task.
    let listener = match CallbackServer::bind().await {
        Ok(l) => l,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": err.to_string(),
                })),
            )
                .into_response();
        }
    };

    // Stash the auth request so the callback task can verify the
    // state token. We also keep a flag the UI polls.
    let request_for_task = request.clone();
    {
        let mut codex_state = state.codex.lock().await;
        let slot = codex_state.get_or_insert_with(CodexFlowState::default);
        slot.in_flight = Some(request.clone());
        slot.last_error = None;
    }

    // Drive the rest of the flow on a background task so the HTTP
    // request returns immediately. The task is fire-and-forget — its
    // result lands in `state.codex` for the status poller.
    let state_clone = state.clone();
    tokio::spawn(async move {
        // 5 minutes is generous; if the operator hasn't completed
        // by then they almost certainly abandoned the tab.
        let outcome = listener.wait_for_callback(300).await;
        let final_error = drive_callback_to_completion(state_clone.clone(), &request_for_task, outcome).await;
        let mut codex_state = state_clone.codex.lock().await;
        let slot = codex_state.get_or_insert_with(CodexFlowState::default);
        slot.in_flight = None;
        slot.last_error = final_error;
    });

    Json(CodexStartDto {
        authorize_url,
        callback_port,
    })
    .into_response()
}

/// Run the callback → token-exchange → persist pipeline. Returns
/// `None` on success (caller clears `last_error`) and `Some(reason)`
/// on any failure so the UI can show it.
async fn drive_callback_to_completion(
    _state: AppApiState,
    request: &AuthorizationRequest,
    outcome: Result<CallbackResult, neoethos_codex::CodexError>,
) -> Option<String> {
    let callback = match outcome {
        Ok(cb) => cb,
        Err(err) => return Some(err.to_string()),
    };
    let (code, state_param) = match callback {
        CallbackResult::Success { code, state } => (code, state),
        CallbackResult::Error { error, description } => {
            let mut msg = error;
            if let Some(d) = description {
                msg.push_str(": ");
                msg.push_str(&d);
            }
            return Some(msg);
        }
    };
    if state_param != request.state {
        return Some(
            "OAuth state mismatch — refusing to continue (possible CSRF).".to_string(),
        );
    }
    let bundle = match exchange_code(&code, request).await {
        Ok(b) => b,
        Err(err) => return Some(err.to_string()),
    };
    let auth = StoredAuth::from_bundle(bundle);
    let store = AuthStore::at_default();
    if let Err(err) = store.save(&auth) {
        return Some(err.to_string());
    }
    tracing::info!(
        target: "neoethos_app::server::codex",
        email = ?auth.email,
        "ChatGPT account linked via Codex OAuth"
    );
    None
}

// ─── POST /auth/codex/logout ───────────────────────────────────────────────

pub async fn logout(State(state): State<AppApiState>) -> Response {
    let store = AuthStore::at_default();
    if let Err(err) = store.delete() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response();
    }
    let mut codex_state = state.codex.lock().await;
    *codex_state = None;
    Json(serde_json::json!({"ok": true})).into_response()
}

// ─── POST /codex/chat ──────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct CodexChatBody {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    #[serde(rename = "maxTokens")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexChatResponseDto {
    pub model: String,
    pub response: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub async fn chat(
    State(_state): State<AppApiState>,
    Json(body): Json<CodexChatBody>,
) -> Response {
    if body.prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt must not be empty"})),
        )
            .into_response();
    }
    let store = AuthStore::at_default();
    let client = CodexClient::new(store);

    let mut request = ChatCompletionRequest::simple(&body.prompt);
    if let Some(model) = body.model {
        request.model = model;
    }
    request.max_tokens = body.max_tokens;
    // Override the simple() default — the AI Helper wants the system
    // role to position the assistant as a forex co-pilot. Prepend
    // instead of replacing so power-users can pass a multi-message
    // body via a future endpoint.
    request.messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: "You are the NeoEthos AI Helper. Answer trading and \
                      market-history questions concisely. Refuse \
                      financial advice; explain mechanics and risk \
                      instead."
                .to_string(),
        },
    );

    match client.chat(request).await {
        Ok(resp) => {
            // F-291: display fallback aligned with the only accepted
            // Codex model. resp.model is normally already set to the
            // request model by the client, so this is belt-and-braces.
            let model = resp.model.unwrap_or_else(|| "gpt-5.5".to_string());
            let response = resp
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default();
            let usage = resp.usage.unwrap_or(neoethos_codex::client::ChatUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });
            Json(CodexChatResponseDto {
                model,
                response,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            })
            .into_response()
        }
        Err(err) => {
            let message = err.to_string();
            let status = match &err {
                neoethos_codex::CodexError::NotAuthenticated => StatusCode::UNAUTHORIZED,
                neoethos_codex::CodexError::ApiCall { status, .. } => {
                    StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
                }
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(serde_json::json!({"error": message}))).into_response()
        }
    }
}

// ─── Shared mutable state for a flow in progress ───────────────────────────

/// Carried in [`AppApiState::codex`]. The presence of `Some(...)` means
/// at least one login has been attempted in this process lifetime;
/// the inner `in_flight` distinguishes "currently waiting on the
/// browser" from "completed (success or failure)".
#[derive(Debug, Default)]
pub struct CodexFlowState {
    /// Cached request from the most recent `start()` call. Kept
    /// alive so the callback task can verify the state token and
    /// then submit the PKCE verifier to the token endpoint.
    pub in_flight: Option<AuthorizationRequest>,
    /// Plain-language description of the most recent failure, if any.
    /// `None` after a successful login.
    pub last_error: Option<String>,
}
