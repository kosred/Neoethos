//! OAuth 2.0 authorization-code-with-PKCE wire format.
//!
//! Two responsibilities:
//!   1. Build the URL we redirect the operator's browser to
//!      (`/oauth/authorize?...`).
//!   2. Trade the authorization `code` we get back for tokens
//!      (`POST /oauth/token`).
//!
//! The actual transport of the browser redirect happens in
//! [`crate::callback`]; this module is HTTP-only and synchronous from
//! the caller's point of view.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::CodexError;
use crate::pkce::PkceChallenge;
use crate::{CODEX_CLIENT_ID, CODEX_ISSUER, CODEX_REDIRECT_URI, CODEX_SCOPES};

/// Inputs for building an authorization URL. The struct lives only
/// for one OAuth attempt; we keep it as a value type so callers can
/// stash it in their app state across the browser round-trip without
/// fighting the borrow checker.
#[derive(Debug, Clone)]
pub struct AuthorizationRequest {
    pub pkce: PkceChallenge,
    /// CSRF guard — a short random token we round-trip via the
    /// `state` query parameter. The callback handler verifies it
    /// matches before continuing.
    pub state: String,
}

impl AuthorizationRequest {
    /// Generate a fresh request: new PKCE pair + new state token.
    pub fn new() -> Self {
        let mut state_bytes = [0u8; 24];
        rand::rng().fill_bytes(&mut state_bytes);
        let state = URL_SAFE_NO_PAD.encode(state_bytes);
        Self {
            pkce: PkceChallenge::generate(),
            state,
        }
    }

    /// Build the URL we send the operator's browser to. This is the
    /// only thing the front-end gets from `/auth/codex/start`; the
    /// Flutter side hands it to `url_launcher` and that's the end of
    /// our involvement until the redirect fires.
    pub fn build_authorize_url(&self) -> String {
        // Manually URL-encoding because `url::form_urlencoded` is
        // verbose and pulling in another dep just for this is silly.
        // All the values we serialise here are ASCII-only and safe
        // to drop into a query string with minimal escaping.
        format!(
            "{issuer}/oauth/authorize\
             ?response_type=code\
             &client_id={client_id}\
             &redirect_uri={redirect}\
             &scope={scope}\
             &code_challenge={cc}\
             &code_challenge_method={method}\
             &state={state}",
            issuer = CODEX_ISSUER,
            client_id = url_encode(CODEX_CLIENT_ID),
            redirect = url_encode(CODEX_REDIRECT_URI),
            scope = url_encode(CODEX_SCOPES),
            cc = self.pkce.code_challenge,
            method = self.pkce.method(),
            state = url_encode(&self.state),
        )
    }
}

impl Default for AuthorizationRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Token-endpoint response. We only model the fields we actually
/// consume; OpenAI's response includes more (id_token claims, etc.)
/// and we keep them in [`Self::raw`] so the persistence layer can
/// round-trip them verbatim into `~/.codex/auth.json`.
#[derive(Debug, Clone)]
pub struct TokenBundle {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub token_type: String,
    /// Issuer-reported lifetime in seconds. We convert to an absolute
    /// timestamp at persistence time so the on-disk schema is
    /// independent of "when was this loaded".
    pub expires_in_seconds: Option<u64>,
    /// Raw JSON we got from the token endpoint. Stored as-is so we
    /// don't lose forward-compat fields (e.g. id_token claims that
    /// some Codex CLI versions persist for offline display of
    /// `signed_in_as`).
    pub raw: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TokenResponseRaw {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default = "default_token_type")]
    token_type: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

#[derive(Debug, Serialize)]
struct TokenRequestBody<'a> {
    grant_type: &'a str,
    code: &'a str,
    redirect_uri: &'a str,
    client_id: &'a str,
    code_verifier: &'a str,
}

/// Exchange the authorization `code` returned by the callback for a
/// fresh [`TokenBundle`]. This call:
///   - never touches the disk;
///   - never re-uses an existing `reqwest::Client` (we build a
///     short-lived one — the token endpoint sees ≤ 1 request per
///     login, so connection pooling buys nothing);
///   - leaves the caller responsible for persisting the result.
///
/// Errors are explicit: a non-2xx response surfaces the issuer's
/// JSON body verbatim in [`CodexError::TokenExchange`] so the
/// `/auth/codex/status` endpoint can show "invalid_grant" or
/// whatever the issuer actually said.
pub async fn exchange_code(
    code: &str,
    request: &AuthorizationRequest,
) -> Result<TokenBundle, CodexError> {
    let client = reqwest::Client::builder()
        // 30s is generous; the token endpoint typically responds in
        // < 1s. Anything timing out past 30s is almost certainly a
        // network issue worth bubbling to the operator.
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let body = TokenRequestBody {
        grant_type: "authorization_code",
        code,
        redirect_uri: CODEX_REDIRECT_URI,
        client_id: CODEX_CLIENT_ID,
        code_verifier: &request.pkce.code_verifier,
    };

    let url = format!("{CODEX_ISSUER}/oauth/token");
    let response = client
        .post(&url)
        .form(&body)
        .header("Accept", "application/json")
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    if !status.is_success() {
        return Err(CodexError::TokenExchange {
            status: status.as_u16(),
            body: response_text,
        });
    }

    let raw: serde_json::Value = serde_json::from_str(&response_text)?;
    let parsed: TokenResponseRaw = serde_json::from_str(&response_text)?;

    Ok(TokenBundle {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        id_token: parsed.id_token,
        token_type: parsed.token_type,
        expires_in_seconds: parsed.expires_in,
        raw,
    })
}

/// Refresh an expired bundle using the refresh_token. Same wire
/// shape as `exchange_code` but `grant_type=refresh_token`.
pub async fn refresh_token(refresh_token: &str) -> Result<TokenBundle, CodexError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    #[derive(Serialize)]
    struct RefreshBody<'a> {
        grant_type: &'a str,
        refresh_token: &'a str,
        client_id: &'a str,
    }

    let body = RefreshBody {
        grant_type: "refresh_token",
        refresh_token,
        client_id: CODEX_CLIENT_ID,
    };

    let url = format!("{CODEX_ISSUER}/oauth/token");
    let response = client
        .post(&url)
        .form(&body)
        .header("Accept", "application/json")
        .send()
        .await?;

    let status = response.status();
    let text = response.text().await?;

    if !status.is_success() {
        return Err(CodexError::TokenExchange {
            status: status.as_u16(),
            body: text,
        });
    }

    let raw: serde_json::Value = serde_json::from_str(&text)?;
    let parsed: TokenResponseRaw = serde_json::from_str(&text)?;

    Ok(TokenBundle {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        id_token: parsed.id_token,
        token_type: parsed.token_type,
        expires_in_seconds: parsed.expires_in,
        raw,
    })
}

/// Minimal application/x-www-form-urlencoded percent-encoder. Covers
/// the small alphabet we actually emit (scopes are space-separated
/// ASCII; client_id is base64-ish). Not RFC-3986-complete, just
/// enough to round-trip our values.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorize_url_contains_all_required_params() {
        let req = AuthorizationRequest::new();
        let url = req.build_authorize_url();
        assert!(url.starts_with("https://auth.openai.com/oauth/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains(&format!("code_challenge={}", req.pkce.code_challenge)));
        assert!(url.contains(&format!("state={}", req.state)));
        // Space in scope must be URL-encoded.
        assert!(url.contains("scope=openid%20profile%20email%20offline_access"));
        // localhost colon must be URL-encoded.
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback"));
    }

    #[test]
    fn state_is_a_fresh_random_value_per_request() {
        let a = AuthorizationRequest::new();
        let b = AuthorizationRequest::new();
        assert_ne!(a.state, b.state);
        assert!(a.state.len() >= 30);
    }

    #[test]
    fn url_encoder_handles_known_special_chars() {
        assert_eq!(url_encode("openid profile"), "openid%20profile");
        assert_eq!(url_encode("http://localhost:1455"), "http%3A%2F%2Flocalhost%3A1455");
        assert_eq!(url_encode("abc-123_xyz.~"), "abc-123_xyz.~");
    }
}
