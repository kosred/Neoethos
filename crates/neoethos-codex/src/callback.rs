//! Single-shot loopback HTTP listener for the OAuth redirect.
//!
//! When the operator authorizes us on auth.openai.com, the browser
//! is redirected to `http://localhost:1455/auth/callback?code=...&state=...`.
//! This module spins up a minimal HTTP/1.1 server on port 1455, waits
//! for exactly one request, replies with a branded "you may close this
//! tab" page, and returns the parsed query parameters.
//!
//! We deliberately avoid axum / hyper here: pulling in a full HTTP
//! stack for a 1-request listener doubles the dependency surface for
//! no benefit. The hand-rolled parser handles only the GET shape the
//! issuer actually sends.
//!
//! ## Failure modes we tolerate
//!
//! - **User closes the tab before authorizing**: we just block until
//!   the timeout fires; caller sees `CodexError::CallbackTimeout`.
//! - **Issuer returns ?error=access_denied**: we still parse the
//!   query, surface `CallbackResult::Error { ... }`, and serve the
//!   error page.
//! - **Port 1455 already in use**: bind fails fast with
//!   `CallbackError::CallbackBind` — the operator may have a stale
//!   Codex CLI flow open. We don't try to be cute about killing
//!   them; just report.

use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;

use crate::error::CodexError;
use crate::CODEX_CALLBACK_PORT;

/// What the loopback listener captured. We don't decide validity
/// here — that's the caller's job (verify `state`, then exchange
/// `code` for tokens).
#[derive(Debug, Clone)]
pub enum CallbackResult {
    Success { code: String, state: String },
    Error { error: String, description: Option<String> },
}

/// Holds the listener so the caller can drop it explicitly. We
/// intentionally don't expose this as a long-lived server — the
/// design is "one bind, one request, then drop".
pub struct CallbackServer {
    listener: TcpListener,
}

impl CallbackServer {
    /// Bind 127.0.0.1:1455. Fails immediately with
    /// `CodexError::CallbackBind` if the port is busy, which is
    /// usually a clearer message than letting an HTTP request
    /// time out down the line.
    pub async fn bind() -> Result<Self, CodexError> {
        let addr = format!("127.0.0.1:{CODEX_CALLBACK_PORT}");
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(CodexError::CallbackBind)?;
        Ok(Self { listener })
    }

    /// Block until either:
    ///   - a `GET /auth/callback?...` request lands, in which case we
    ///     parse the query and return `Ok(CallbackResult::...)`;
    ///   - the timeout fires, in which case we return
    ///     `Err(CallbackError::CallbackTimeout)`.
    ///
    /// The HTTP reply is a small branded page that explains the next
    /// step. The operator's browser shows this; the listener then
    /// shuts down.
    pub async fn wait_for_callback(
        &self,
        timeout_secs: u64,
    ) -> Result<CallbackResult, CodexError> {
        let accept_result = timeout(
            Duration::from_secs(timeout_secs),
            self.listener.accept(),
        )
        .await
        .map_err(|_| CodexError::CallbackTimeout(timeout_secs))?;

        let (mut socket, _peer) = accept_result.map_err(CodexError::CallbackBind)?;

        // Read just enough of the request to parse the request line.
        // Real-world Chrome/Edge/Firefox redirects send the request
        // in one syscall < 1 KB; we read up to 4 KB to be safe and
        // throw away the headers.
        let mut buf = [0u8; 4096];
        let n = socket
            .read(&mut buf)
            .await
            .map_err(CodexError::CallbackBind)?;
        let request = String::from_utf8_lossy(&buf[..n]);

        let parsed = parse_request_line_query(&request);

        // Reply with a small branded page regardless of success/error;
        // the operator sees something useful instead of "ERR_EMPTY_RESPONSE".
        let response_body = match &parsed {
            Ok(CallbackResult::Success { .. }) => SUCCESS_HTML,
            Ok(CallbackResult::Error { error, .. }) => {
                tracing::warn!(
                    target: "neoethos_codex::callback",
                    error = %error,
                    "OAuth callback returned an error"
                );
                ERROR_HTML
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_codex::callback",
                    error = %err,
                    "OAuth callback request was malformed"
                );
                ERROR_HTML
            }
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            response_body.len(),
            response_body
        );
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.shutdown().await;

        parsed
    }
}

/// Parse the query parameters from the first line of an HTTP request.
/// The browser sends `GET /auth/callback?code=XYZ&state=ABC HTTP/1.1`.
/// We only need `code`, `state`, and the OAuth error fields.
fn parse_request_line_query(request: &str) -> Result<CallbackResult, CodexError> {
    let first_line = request.lines().next().unwrap_or("");
    // Splitting on space → ["GET", "/auth/callback?...", "HTTP/1.1"].
    let target = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| CodexError::CallbackError("malformed HTTP request line".to_string()))?;

    let query = target.split_once('?').map(|(_, q)| q).unwrap_or("");

    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    let mut error: Option<String> = None;
    let mut description: Option<String> = None;

    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let decoded = percent_decode(value);
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error = Some(decoded),
            "error_description" => description = Some(decoded),
            _ => {}
        }
    }

    if let Some(error) = error {
        return Ok(CallbackResult::Error { error, description });
    }
    match (code, state) {
        (Some(code), Some(state)) => Ok(CallbackResult::Success { code, state }),
        _ => Err(CodexError::CallbackError(
            "callback missing required parameters (code, state)".to_string(),
        )),
    }
}

/// Minimal percent-decoder for the values that show up in OAuth
/// callbacks. Codex never includes raw spaces, but `state` can
/// contain `-`, `_`, `.`, `~` (already legal) and base64 chars; only
/// `+` shows up as ` ` in form bodies, never query strings. So we
/// only need to handle `%XX` sequences.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_digit(bytes[i + 1]);
            let lo = hex_digit(bytes[i + 2]);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

const SUCCESS_HTML: &str = r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>NeoEthos · ChatGPT connected</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #0e1418; color: #e2e8f0; display: flex; align-items: center;
       justify-content: center; height: 100vh; margin: 0; }
.card { max-width: 420px; padding: 32px 36px; background: #182028; border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.45); text-align: center; }
h1 { font-size: 22px; margin: 0 0 12px; color: #4ade80; }
p  { font-size: 15px; line-height: 1.5; margin: 8px 0; color: #cbd5e1; }
small { display:block; margin-top:20px; color:#64748b; font-size:12px; }
</style></head>
<body><div class="card">
<h1>✓ ChatGPT connected</h1>
<p>Your ChatGPT subscription is now linked to NeoEthos. You can close this tab and return to the app — the AI Helper will use your account for chats.</p>
<small>NeoEthos · neoethos-codex</small>
</div></body></html>"#;

const ERROR_HTML: &str = r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>NeoEthos · authentication failed</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #1a0e0e; color: #fee2e2; display: flex; align-items: center;
       justify-content: center; height: 100vh; margin: 0; }
.card { max-width: 420px; padding: 32px 36px; background: #2a1818; border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.45); text-align: center; }
h1 { font-size: 22px; margin: 0 0 12px; color: #f87171; }
p  { font-size: 15px; line-height: 1.5; margin: 8px 0; color: #fecaca; }
small { display:block; margin-top:20px; color:#9ca3af; font-size:12px; }
</style></head>
<body><div class="card">
<h1>✗ Authentication failed</h1>
<p>The sign-in provider returned an error or the request was malformed. Close this tab, return to NeoEthos, and click Connect ChatGPT again.</p>
<small>NeoEthos · neoethos-codex</small>
</div></body></html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_success_callback() {
        let req = "GET /auth/callback?code=abc123&state=xyz789 HTTP/1.1\r\nHost: localhost:1455\r\n\r\n";
        let parsed = parse_request_line_query(req).unwrap();
        match parsed {
            CallbackResult::Success { code, state } => {
                assert_eq!(code, "abc123");
                assert_eq!(state, "xyz789");
            }
            _ => panic!("expected Success"),
        }
    }

    #[test]
    fn parses_error_callback() {
        let req = "GET /auth/callback?error=access_denied&error_description=user%20said%20no HTTP/1.1\r\n\r\n";
        let parsed = parse_request_line_query(req).unwrap();
        match parsed {
            CallbackResult::Error { error, description } => {
                assert_eq!(error, "access_denied");
                assert_eq!(description.as_deref(), Some("user said no"));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn missing_code_yields_error() {
        let req = "GET /auth/callback?state=xyz HTTP/1.1\r\n\r\n";
        let err = parse_request_line_query(req).unwrap_err();
        assert!(matches!(err, CodexError::CallbackError(_)));
    }

    #[test]
    fn percent_decode_basic() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
        assert_eq!(percent_decode("a%2Bb"), "a+b");
        assert_eq!(percent_decode("plain"), "plain");
        // Truncated percent sequence is treated literally.
        assert_eq!(percent_decode("ab%2"), "ab%2");
    }
}
