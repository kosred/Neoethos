use crate::app_services::ctrader_auth::CTraderTokenBundle;
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use std::sync::{Arc, Mutex};

pub const CTRADER_DEFAULT_SCOPE: &str = "trading";
pub const CTRADER_TOKEN_ENDPOINT_BASE: &str = "https://openapi.ctrader.com";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLoopbackConfig {
    allowed_ports: Vec<u16>,
    callback_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderCallbackPayload {
    pub authorization_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLiveAuthRequest {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uri: String,
    pub scope: String,
    pub loopback: CTraderLoopbackConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLiveAuthResult {
    pub callback_port: u16,
    pub authorization_code: String,
    pub token_bundle: CTraderTokenBundle,
}

pub trait CTraderLiveAuthBackend: Send + Sync {
    fn run(&self, request: CTraderLiveAuthRequest) -> Result<CTraderLiveAuthResult>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderLiveAuthBackend;

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderLiveAuthBackend {
    outcome: Arc<Mutex<Option<Result<CTraderLiveAuthResult, String>>>>,
}

#[derive(Debug, Deserialize)]
struct CTraderTokenExchangeResponse {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "refreshToken")]
    refresh_token: String,
    #[serde(rename = "tokenType")]
    token_type: String,
    #[serde(rename = "expiresIn")]
    expires_in: i64,
    scope: Option<String>,
}

impl CTraderLoopbackConfig {
    pub fn new(primary_port: u16, fallback_ports: Vec<u16>, callback_path: impl Into<String>) -> Self {
        let mut allowed_ports = vec![primary_port];
        allowed_ports.extend(fallback_ports);
        Self {
            allowed_ports,
            callback_path: callback_path.into(),
        }
    }

    pub fn callback_path(&self) -> &str {
        &self.callback_path
    }

    pub fn allowed_ports(&self) -> &[u16] {
        &self.allowed_ports
    }
}

impl ProductionCTraderLiveAuthBackend {
    fn bind_loopback_listener(&self, config: &CTraderLoopbackConfig) -> Result<(u16, TcpListener)> {
        for port in config.allowed_ports() {
            match TcpListener::bind(("127.0.0.1", *port)) {
                Ok(listener) => return Ok((*port, listener)),
                Err(_) => continue,
            }
        }
        Err(anyhow!("failed to bind any cTrader callback port"))
    }

    fn capture_authorization_code(
        &self,
        listener: TcpListener,
        expected_path: &str,
    ) -> Result<String> {
        let (mut stream, _) = listener.accept().context("failed to accept cTrader callback")?;
        let mut buffer = [0_u8; 4096];
        let bytes_read = stream
            .read(&mut buffer)
            .context("failed to read cTrader callback request")?;
        let request = String::from_utf8_lossy(&buffer[..bytes_read]);
        let request_line = request
            .lines()
            .next()
            .context("cTrader callback request was empty")?;
        let request_target = request_line
            .split_whitespace()
            .nth(1)
            .context("cTrader callback request line was malformed")?;
        let payload = parse_callback_request(request_target, expected_path)?;

        let response = concat!(
            "HTTP/1.1 200 OK\r\n",
            "Content-Type: text/html; charset=utf-8\r\n",
            "Content-Length: 74\r\n",
            "Connection: close\r\n",
            "\r\n",
            "<html><body><h1>cTrader login received.</h1>You can close this tab.</body></html>",
        );
        stream
            .write_all(response.as_bytes())
            .context("failed to write cTrader callback response")?;
        stream.flush().context("failed to flush cTrader callback response")?;
        Ok(payload.authorization_code)
    }

    fn exchange_token(
        &self,
        request: &CTraderLiveAuthRequest,
        callback_port: u16,
        authorization_code: &str,
    ) -> Result<CTraderTokenBundle> {
        let redirect_uri = rewrite_redirect_uri_port(&request.redirect_uri, callback_port)?;
        let url = build_token_exchange_url(
            CTRADER_TOKEN_ENDPOINT_BASE,
            "authorization_code",
            authorization_code,
            &redirect_uri,
            &request.client_id,
            &request.client_secret,
        );
        let response = reqwest::blocking::get(url)
            .context("failed to call cTrader token endpoint")?
            .error_for_status()
            .context("cTrader token endpoint returned an error status")?;
        let payload: CTraderTokenExchangeResponse = response
            .json()
            .context("failed to parse cTrader token response")?;
        Ok(CTraderTokenBundle {
            access_token: payload.access_token,
            refresh_token: payload.refresh_token,
            token_type: payload.token_type,
            expires_in: payload.expires_in,
            scope: payload.scope.unwrap_or_else(|| request.scope.clone()),
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("system clock is before UNIX_EPOCH")?
                .as_secs() as i64,
        })
    }
}

impl CTraderLiveAuthBackend for ProductionCTraderLiveAuthBackend {
    fn run(&self, request: CTraderLiveAuthRequest) -> Result<CTraderLiveAuthResult> {
        let (callback_port, listener) = self.bind_loopback_listener(&request.loopback)?;
        let authorize_url = build_authorize_url(
            &request.client_id,
            &request.redirect_uri,
            callback_port,
            &request.scope,
        );
        open::that(authorize_url).context("failed to open system browser for cTrader login")?;
        let authorization_code =
            self.capture_authorization_code(listener, request.loopback.callback_path())?;
        let token_bundle = self.exchange_token(&request, callback_port, &authorization_code)?;
        Ok(CTraderLiveAuthResult {
            callback_port,
            authorization_code,
            token_bundle,
        })
    }
}

#[cfg(test)]
impl StubCTraderLiveAuthBackend {
    pub fn success(result: CTraderLiveAuthResult) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(result)))),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
        }
    }
}

#[cfg(test)]
impl Default for StubCTraderLiveAuthBackend {
    fn default() -> Self {
        Self::failure("stub cTrader live auth backend was not configured")
    }
}

#[cfg(test)]
impl CTraderLiveAuthBackend for StubCTraderLiveAuthBackend {
    fn run(&self, _request: CTraderLiveAuthRequest) -> Result<CTraderLiveAuthResult> {
        match self
            .outcome
            .lock()
            .expect("stub cTrader live auth backend lock poisoned")
            .take()
            .unwrap_or_else(|| Err("stub cTrader live auth backend was already consumed".to_string()))
        {
            Ok(result) => Ok(result),
            Err(message) => Err(anyhow!(message)),
        }
    }
}

pub fn build_default_loopback_config(redirect_uri: &str) -> Result<CTraderLoopbackConfig> {
    let (_, remainder) = redirect_uri
        .split_once("://")
        .context("redirect URI is missing scheme")?;
    let (host_port, suffix) = remainder
        .split_once('/')
        .map_or((remainder, ""), |(host_port, suffix)| (host_port, suffix));
    let port = host_port
        .split(':')
        .nth(1)
        .context("redirect URI is missing port")?
        .parse::<u16>()
        .context("redirect URI port is invalid")?;
    let callback_path = format!("/{}", suffix.trim_start_matches('/'));
    Ok(CTraderLoopbackConfig::new(
        port,
        vec![port.saturating_add(1), port.saturating_add(2)],
        callback_path,
    ))
}

pub fn build_authorize_url(
    client_id: &str,
    redirect_uri: &str,
    callback_port: u16,
    scope: &str,
) -> String {
    let redirect_uri = rewrite_redirect_uri_port(redirect_uri, callback_port)
        .unwrap_or_else(|_| redirect_uri.to_string());
    format!(
        "https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={}&redirect_uri={}&scope={}&product=web",
        percent_encode(client_id),
        percent_encode(&redirect_uri),
        percent_encode(scope),
    )
}

pub fn parse_callback_request(request_target: &str, expected_path: &str) -> Result<CTraderCallbackPayload> {
    let (path, query) = request_target
        .split_once('?')
        .map_or((request_target, ""), |(path, query)| (path, query));
    if path != expected_path {
        return Err(anyhow!("unexpected callback path: {path}"));
    }

    let authorization_code = query
        .split('&')
        .find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == "code").then(|| value.to_string())
        })
        .filter(|code| !code.trim().is_empty())
        .context("missing authorization code")?;

    Ok(CTraderCallbackPayload { authorization_code })
}

pub fn build_token_exchange_url(
    base_url: &str,
    grant_type: &str,
    code: &str,
    redirect_uri: &str,
    client_id: &str,
    client_secret: &str,
) -> String {
    format!(
        "{}/apps/token?grant_type={}&code={}&redirect_uri={}&client_id={}&client_secret={}",
        base_url.trim_end_matches('/'),
        percent_encode(grant_type),
        percent_encode(code),
        percent_encode(redirect_uri),
        percent_encode(client_id),
        percent_encode(client_secret),
    )
}

fn rewrite_redirect_uri_port(redirect_uri: &str, callback_port: u16) -> Result<String> {
    let (scheme, remainder) = redirect_uri
        .split_once("://")
        .context("redirect URI is missing scheme")?;
    let (host_port, suffix) = remainder
        .split_once('/')
        .map_or((remainder, ""), |(host_port, suffix)| (host_port, suffix));
    let host = host_port
        .split(':')
        .next()
        .context("redirect URI host is missing")?;
    let suffix = if suffix.is_empty() {
        String::new()
    } else {
        format!("/{}", suffix.trim_start_matches('/'))
    };
    Ok(format!("{scheme}://{host}:{callback_port}{suffix}"))
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![char::from(byte).to_string()]
            }
            _ => vec![format!("%{byte:02X}")],
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorize_url_uses_selected_callback_port() {
        let config = CTraderLoopbackConfig::new(43001, vec![43002, 43003], "/callback");

        let authorize_url = build_authorize_url(
            "client-id",
            "http://127.0.0.1:43001/callback",
            43002,
            "trading",
        );

        assert!(authorize_url.contains("client_id=client-id"));
        assert!(authorize_url.contains(
            "redirect_uri=http%3A%2F%2F127.0.0.1%3A43002%2Fcallback"
        ));
        assert_eq!(config.allowed_ports(), &[43001, 43002, 43003]);
    }

    #[test]
    fn callback_parser_accepts_expected_path_and_extracts_code() {
        let parsed = parse_callback_request("/callback?code=auth-code-123", "/callback")
            .expect("callback should parse");

        assert_eq!(parsed.authorization_code, "auth-code-123");
    }

    #[test]
    fn token_exchange_request_uses_documented_query_parameters() {
        let url = build_token_exchange_url(
            "https://openapi.ctrader.com",
            "authorization_code",
            "auth-code-123",
            "http://127.0.0.1:43001/callback",
            "client-id",
            "secret-456",
        );

        assert_eq!(
            url,
            "https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=auth-code-123&redirect_uri=http%3A%2F%2F127.0.0.1%3A43001%2Fcallback&client_id=client-id&client_secret=secret-456"
        );
    }
}
