//! Umbrella error type for the crate.
//!
//! Callers will mostly see this surfaced as the body of an HTTP 500 / 503
//! through the `/auth/codex/*` and `/codex/chat` endpoints — the
//! `Display` impl is therefore phrased for end-users rather than
//! developers.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodexError {
    #[error("OAuth callback listener failed to bind 127.0.0.1:1455 — is another login already in progress? ({0})")]
    CallbackBind(std::io::Error),

    #[error("OAuth callback timed out after {0}s. Try clicking Connect again.")]
    CallbackTimeout(u64),

    #[error("OAuth callback received an error from OpenAI: {0}")]
    CallbackError(String),

    #[error("OAuth state mismatch — the callback did not match the request we sent. Aborting to defend against CSRF.")]
    StateMismatch,

    #[error("Token endpoint returned {status}: {body}")]
    TokenExchange { status: u16, body: String },

    #[error("auth.json on disk is corrupted or from a newer schema: {0}")]
    AuthStoreParse(String),

    #[error("auth.json could not be written ({path}): {source}")]
    AuthStoreWrite {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Codex API call failed ({status}): {body}")]
    ApiCall { status: u16, body: String },

    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON (de)serialisation error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Not authenticated — operator must complete the OAuth flow before this call.")]
    NotAuthenticated,
}
