//! # neoethos-codex
//!
//! Optional ChatGPT subscription–backed fallback for the AI Helper.
//!
//! When the local Gemma runtime is unavailable (no GGUF on disk,
//! incompatible model, low VRAM, or the inference crashed as in #197 /
//! #208), an operator with an active ChatGPT Plus or Pro subscription
//! can authenticate via the same PKCE OAuth flow that the official
//! `codex` CLI uses, and route AI Helper chats through their
//! subscription instead.
//!
//! ## Layout
//!
//! - [`pkce`] — code_verifier / code_challenge generation. Pure crypto,
//!   no I/O, no async.
//! - [`oauth`] — authorization URL builder + token-endpoint exchange.
//!   Returns parsed [`TokenBundle`] structs.
//! - [`callback`] — single-shot loopback HTTP listener bound to
//!   `127.0.0.1:1455`. Captures the `?code=...&state=...` redirect,
//!   serves a small branded "you may now close this tab" HTML page,
//!   then shuts down.
//! - [`auth_store`] — read/write of `~/.codex/auth.json` (the exact
//!   path and schema the Codex CLI uses). Operators who already
//!   authenticated with `codex login` get picked up automatically;
//!   we never force a second flow.
//! - [`client`] — minimal `chat/completions` proxy. Mirrors the
//!   Chat Completions request/response shape so callers can
//!   reuse their existing prompt scaffolding.
//! - [`error`] — `CodexError` umbrella for HTTP / token / JSON
//!   failures.
//!
//! ## Why we don't reuse the OAuth code from `broker_control`
//!
//! That code path is tightly coupled to the cTrader endpoints
//! (Spotware-specific scopes, `client_id` + `client_secret`,
//! Authorization Code without PKCE). Codex uses a *public* client
//! (no secret, PKCE-only) against the Codex (ChatGPT) issuer, which is a
//! materially different flow. Sharing code between the two would
//! require an OAuth abstraction layer that costs more than just
//! writing ~250 lines of straightforward PKCE.
//!
//! ## Why we DON'T need a feature flag
//!
//! No native deps; pure HTTP + JSON + crypto. The crate adds ≈ 50 KB
//! to the binary regardless of whether the operator uses it. A flag
//! would only hide the option from the UI, which is the opposite of
//! what we want — the entire point is to give operators a working
//! AI Helper when the Gemma path is broken.

#![doc(html_root_url = "https://docs.rs/neoethos-codex/0.4.99")]
#![deny(unused_must_use)]

pub mod auth_store;
pub mod callback;
pub mod client;
pub mod error;
pub mod oauth;
pub mod pkce;

pub use auth_store::{AuthStore, StoredAuth, default_auth_path};
pub use callback::{CallbackResult, CallbackServer};
pub use client::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CodexClient};
pub use error::CodexError;
pub use oauth::{AuthorizationRequest, TokenBundle, exchange_code};
pub use pkce::PkceChallenge;

/// Published OAuth client ID for the official Codex CLI. This is a
/// public identifier — not a secret — taken from the official
/// `codex` source. Hard-coding it here means our flow is
/// interoperable with the Codex CLI: an operator who runs
/// `codex login` and then opens NeoEthos sees a "Connected as
/// ${email}" badge without re-authenticating.
pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// OAuth issuer that owns the authorize + token endpoints.
pub const CODEX_ISSUER: &str = "https://auth.openai.com";

/// PKCE redirect URI. The Codex CLI binds 127.0.0.1:1455; we mirror
/// it so the issuer's allow-list (which is fixed for this client_id)
/// accepts our callbacks.
pub const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";

/// Loopback port for the callback listener. Anything else here will
/// be rejected by the issuer.
pub const CODEX_CALLBACK_PORT: u16 = 1455;

/// API base for the ChatGPT-subscription privileged backend. The
/// **actual** programmatic endpoint that the official Codex CLI uses
/// is `/backend-api/codex/responses` — NOT `/conversation` (which is
/// the chat-UI's own endpoint and rejects external-shaped bodies
/// with 422 "Invalid conversation body").
///
/// The `/codex/responses` endpoint speaks the Responses API
/// format: `{model, input, stream, …}` rather than the Chat
/// Completions `{messages: [...]}` shape. Response body is a plain
/// JSON object containing `output: [{content: [{type: "output_text",
/// text: "..."}]}]`. See [`crate::client`] for the request/response
/// shaping we do internally to keep the call-site API (which still
/// looks like Chat Completions) stable.
///
/// **2026-05-26 fix (Κωνσταντίνος)**: was previously
/// `chatgpt.com/backend-api/conversation` per the dev's
/// misunderstanding — that endpoint is the chat-UI's internal
/// conversation flow and demands a very different schema
/// (`{action: "next", messages: [{id, author, content: {parts}}],
/// parent_message_id, …}` + SSE event-stream response). We switched
/// to the documented Codex CLI path which uses a simpler synchronous
/// shape that we can map cleanly to/from the existing
/// `ChatCompletion{Request,Response}` types.
pub const CODEX_API_BASE: &str = "https://chatgpt.com/backend-api";

/// OAuth scopes we request. `openid` + `profile` give us the
/// email/name claims; `offline_access` is what gets us a
/// `refresh_token` so we don't force re-auth every hour.
pub const CODEX_SCOPES: &str = "openid profile email offline_access";
