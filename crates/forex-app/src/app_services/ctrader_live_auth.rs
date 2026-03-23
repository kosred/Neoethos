use crate::app_services::ctrader_auth::CTraderTokenBundle;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};
use tungstenite::{connect, Message};

#[cfg(test)]
use std::sync::{Arc, Mutex};

pub const CTRADER_DEFAULT_SCOPE: &str = "trading";
pub const CTRADER_TOKEN_ENDPOINT_BASE: &str = "https://openapi.ctrader.com";
pub const CTRADER_OA_APPLICATION_AUTH_PAYLOAD_TYPE: u32 = 2100;
pub const CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE: u32 = 2101;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_PAYLOAD_TYPE: u32 = 2149;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE: u32 = 2150;
pub const CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE: u32 = 2142;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLoopbackConfig {
    allowed_ports: Vec<u16>,
    callback_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderCallbackPayload {
    pub authorization_code: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderEnvironment {
    Live,
    Demo,
}

impl CTraderEnvironment {
    pub fn endpoint_host(self) -> &'static str {
        match self {
            Self::Live => "live.ctraderapi.com",
            Self::Demo => "demo.ctraderapi.com",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAccountDiscoveryRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
}

impl CTraderAccountDiscoveryRequest {
    pub fn endpoint_host(&self) -> &'static str {
        self.environment.endpoint_host()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAccountDiscoveryResult {
    pub access_token: String,
    pub permission_scope: String,
    pub accounts: Vec<crate::app_services::ctrader_auth::CTraderDiscoveredAccount>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTraderOpenApiJsonMessage {
    #[serde(rename = "clientMsgId")]
    pub client_msg_id: String,
    #[serde(rename = "payloadType")]
    pub payload_type: u32,
    pub payload: Value,
}

pub trait CTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>>;
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

pub trait CTraderAccountDiscoveryBackend: Send + Sync {
    fn discover_accounts(
        &self,
        request: &CTraderAccountDiscoveryRequest,
    ) -> Result<CTraderAccountDiscoveryResult>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderLiveAuthBackend;

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderLiveAuthBackend {
    outcome: Arc<Mutex<Option<Result<CTraderLiveAuthResult, String>>>>,
}

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderAccountDiscoveryBackend {
    outcome: Arc<Mutex<Option<Result<CTraderAccountDiscoveryResult, String>>>>,
    last_request: Arc<Mutex<Option<CTraderAccountDiscoveryRequest>>>,
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

impl CTraderAccountDiscoveryBackend for ProductionCTraderLiveAuthBackend {
    fn discover_accounts(
        &self,
        request: &CTraderAccountDiscoveryRequest,
    ) -> Result<CTraderAccountDiscoveryResult> {
        discover_ctrader_accounts(request)
    }
}

#[cfg(test)]
impl StubCTraderAccountDiscoveryBackend {
    pub fn success(result: CTraderAccountDiscoveryResult) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(result)))),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn last_request(&self) -> Option<CTraderAccountDiscoveryRequest> {
        self.last_request
            .lock()
            .expect("stub cTrader account discovery request lock poisoned")
            .clone()
    }
}

#[cfg(test)]
impl CTraderAccountDiscoveryBackend for StubCTraderAccountDiscoveryBackend {
    fn discover_accounts(
        &self,
        request: &CTraderAccountDiscoveryRequest,
    ) -> Result<CTraderAccountDiscoveryResult> {
        *self
            .last_request
            .lock()
            .expect("stub cTrader account discovery request lock poisoned") = Some(request.clone());
        match self
            .outcome
            .lock()
            .expect("stub cTrader account discovery lock poisoned")
            .take()
            .unwrap_or_else(|| Err("stub cTrader account discovery backend was already consumed".to_string()))
        {
            Ok(result) => Ok(result),
            Err(message) => Err(anyhow!(message)),
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

pub fn build_application_auth_json(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_APPLICATION_AUTH_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "clientId": client_id,
            "clientSecret": client_secret,
        }),
    }
}

pub fn build_account_list_by_access_token_json(
    request: &CTraderAccountDiscoveryRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "accessToken": request.access_token,
        }),
    }
}

#[derive(Debug, Deserialize)]
struct CTraderAccountListResponseEnvelope {
    #[serde(rename = "clientMsgId")]
    _client_msg_id: Option<String>,
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: CTraderAccountListResponsePayload,
}

#[derive(Debug, Deserialize)]
struct CTraderAccountListResponsePayload {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "permissionScope")]
    permission_scope: String,
    #[serde(rename = "ctidTraderAccount")]
    accounts: Vec<CTraderCtidTraderAccount>,
}

#[derive(Debug, Deserialize)]
struct CTraderCtidTraderAccount {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: u64,
    #[serde(rename = "isLive")]
    is_live: Option<bool>,
    #[serde(rename = "traderLogin")]
    trader_login: Option<i64>,
    #[serde(rename = "brokerTitleShort")]
    broker_title_short: Option<String>,
}

pub fn parse_account_list_by_access_token_json(
    response_json: &str,
) -> Result<CTraderAccountDiscoveryResult> {
    let envelope: CTraderAccountListResponseEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader account list response")?;
    if envelope.payload_type != CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader account list payload type: {}",
            envelope.payload_type
        ));
    }

    let accounts = envelope
        .payload
        .accounts
        .into_iter()
        .map(|account| {
            let account_id = account.ctid_trader_account_id.to_string();
            let broker_title = account.broker_title_short.unwrap_or_default();
            let account_name = if broker_title.trim().is_empty() {
                account
                    .trader_login
                    .map(|login| login.to_string())
                    .unwrap_or_else(|| account_id.clone())
            } else {
                broker_title.clone()
            };
            crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                account_id,
                broker_title,
                account_name,
                trader_login: account.trader_login,
                is_live: account.is_live,
                enabled_for_execution: false,
            }
        })
        .collect();

    Ok(CTraderAccountDiscoveryResult {
        access_token: envelope.payload.access_token,
        permission_scope: envelope.payload.permission_scope,
        accounts,
    })
}

pub fn perform_account_discovery_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderAccountDiscoveryRequest,
) -> Result<CTraderAccountDiscoveryResult> {
    let responses = transport.send_sequence(&[
        build_application_auth_json(&request.client_id, &request.client_secret, "app-auth-1"),
        build_account_list_by_access_token_json(request, "account-list-1"),
    ])?;
    if responses.is_empty() {
        return Err(anyhow!("expected cTrader app auth response, received none"));
    }
    if responses.len() == 1 {
        let app_auth_envelope: CTraderOpenApiJsonMessage = serde_json::from_str(&responses[0])
            .context("failed to parse cTrader app auth response")?;
        if app_auth_envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
            let error = parse_ctrader_error_payload(&app_auth_envelope.payload)?;
            return Err(anyhow!("cTrader app auth failed: {}", error));
        }
        return Err(anyhow!(
            "expected cTrader account list response after app auth, received only app auth response"
        ));
    }
    if responses.len() != 2 {
        return Err(anyhow!(
            "expected 2 cTrader discovery responses, received {}",
            responses.len()
        ));
    }

    let app_auth_envelope: CTraderOpenApiJsonMessage = serde_json::from_str(&responses[0])
        .context("failed to parse cTrader app auth response")?;
    if app_auth_envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        let error = parse_ctrader_error_payload(&app_auth_envelope.payload)?;
        return Err(anyhow!("cTrader app auth failed: {}", error));
    }
    if app_auth_envelope.payload_type != CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader app auth payload type: {}",
            app_auth_envelope.payload_type
        ));
    }

    parse_account_list_by_access_token_json(&responses[1])
}

pub fn discover_ctrader_accounts(
    request: &CTraderAccountDiscoveryRequest,
) -> Result<CTraderAccountDiscoveryResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.endpoint_host());
    perform_account_discovery_with_transport(&transport, request)
}

fn parse_ctrader_error_payload(payload: &Value) -> Result<String> {
    #[derive(Debug, Deserialize)]
    struct CTraderErrorPayload {
        #[serde(rename = "errorCode")]
        error_code: String,
        description: Option<String>,
    }

    let error: CTraderErrorPayload = serde_json::from_value(payload.clone())
        .context("failed to parse cTrader error payload")?;
    Ok(match error.description {
        Some(description) if !description.trim().is_empty() => {
            format!("{}: {}", error.error_code, description)
        }
        _ => error.error_code,
    })
}

#[cfg(test)]
#[derive(Default)]
struct StubCTraderOpenApiTransport {
    sent_messages: std::sync::Mutex<Vec<CTraderOpenApiJsonMessage>>,
    responses: std::sync::Mutex<Vec<Result<String>>>,
}

#[cfg(test)]
impl StubCTraderOpenApiTransport {
    fn with_responses(responses: Vec<Result<String>>) -> Self {
        Self {
            sent_messages: std::sync::Mutex::new(Vec::new()),
            responses: std::sync::Mutex::new(responses),
        }
    }

    fn sent_messages(&self) -> Vec<CTraderOpenApiJsonMessage> {
        self.sent_messages.lock().expect("sent_messages lock").clone()
    }
}

#[cfg(test)]
impl CTraderOpenApiTransport for StubCTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        self.sent_messages
            .lock()
            .expect("sent_messages lock")
            .extend(messages.iter().cloned());

        let mut responses = self.responses.lock().expect("responses lock");
        let mut output = Vec::with_capacity(messages.len());
        for _ in messages {
            if responses.is_empty() {
                break;
            }
            let response = responses.remove(0).map_err(|err| anyhow!(err))?;
            let payload_type = parse_open_api_payload_type(&response)?;
            output.push(response);
            if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                break;
            }
        }
        Ok(output)
    }
}

#[allow(dead_code)]
struct ProductionCTraderOpenApiTransport {
    endpoint_host: String,
}

#[allow(dead_code)]
impl ProductionCTraderOpenApiTransport {
    fn new(endpoint_host: impl Into<String>) -> Self {
        Self {
            endpoint_host: endpoint_host.into(),
        }
    }
}

#[allow(dead_code)]
impl CTraderOpenApiTransport for ProductionCTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        let url = format!("wss://{}:5036", self.endpoint_host);
        let (mut socket, _) =
            connect(url.as_str()).with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        let mut responses = Vec::with_capacity(messages.len());

        for message in messages {
            let serialized = serde_json::to_string(message)
                .context("failed to serialize cTrader open api message")?;
            socket
                .send(Message::Text(serialized.into()))
                .context("failed to send cTrader open api message")?;

            loop {
                match socket.read().context("failed to read cTrader open api response")? {
                    Message::Text(text) => {
                        if text.trim().is_empty() {
                            return Err(anyhow!("empty cTrader open api response"));
                        }
                        let payload_type = parse_open_api_payload_type(text.as_ref())?;
                        responses.push(text.to_string());
                        if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        break;
                    }
                    Message::Binary(bytes) => {
                        let text = String::from_utf8(bytes.to_vec())
                            .context("failed to decode cTrader binary response")?;
                        if text.trim().is_empty() {
                            return Err(anyhow!("empty cTrader open api response"));
                        }
                        let payload_type = parse_open_api_payload_type(&text)?;
                        responses.push(text);
                        if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        break;
                    }
                    Message::Ping(payload) => {
                        socket
                            .send(Message::Pong(payload))
                            .context("failed to reply to cTrader ping")?;
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) => {
                        return Err(anyhow!("cTrader open api socket closed unexpectedly"));
                    }
                    Message::Frame(_) => {}
                }
            }
        }
        let _ = socket.close(None);
    Ok(responses)
    }
}

fn parse_open_api_payload_type(response_json: &str) -> Result<u32> {
    let envelope: CTraderOpenApiJsonMessage = serde_json::from_str(response_json)
        .context("failed to parse cTrader JSON envelope")?;
    Ok(envelope.payload_type)
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
        )
        .expect("authorize url should build");

        assert!(authorize_url.contains("client_id=client-id"));
        assert!(authorize_url.contains(
            "redirect_uri=http%3A%2F%2F127.0.0.1%3A43002%2Fcallback"
        ));
        assert_eq!(config.allowed_ports(), &[43001, 43002, 43003]);
    }

    #[test]
    fn authorize_url_rejects_malformed_redirect_uri() {
        let err = build_authorize_url("client-id", "not-a-valid-redirect", 43002, "trading")
            .expect_err("malformed redirect must fail");

        assert!(err.to_string().contains("redirect URI"));
    }

    #[test]
    fn callback_parser_accepts_expected_path_and_extracts_code() {
        let parsed = parse_callback_request("/callback?code=auth-code-123", "/callback")
            .expect("callback should parse");

        assert_eq!(parsed.authorization_code, "auth-code-123");
    }

    #[test]
    fn callback_parser_decodes_percent_encoded_authorization_code() {
        let parsed = parse_callback_request("/callback?code=auth%2Bcode%252F123", "/callback")
            .expect("callback should decode");

        assert_eq!(parsed.authorization_code, "auth+code%2F123");
    }

    #[test]
    fn callback_parser_surfaces_ctrader_denial_errors() {
        let err = parse_callback_request(
            "/callback?error=access_denied&error_description=operator%20cancelled",
            "/callback",
        )
        .expect_err("denied callback should fail");

        assert!(err.to_string().contains("cTrader authorization denied: access_denied"));
        assert!(err.to_string().contains("operator cancelled"));
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

    #[test]
    fn application_auth_request_uses_documented_payload_type() {
        let message = build_application_auth_json("client-id", "secret-456", "cm-id-2");

        assert_eq!(message.client_msg_id, "cm-id-2");
        assert_eq!(message.payload_type, 2100);
        assert_eq!(
            message.payload.get("clientId").and_then(|value| value.as_str()),
            Some("client-id")
        );
        assert_eq!(
            message.payload.get("clientSecret").and_then(|value| value.as_str()),
            Some("secret-456")
        );
    }

    #[test]
    fn account_discovery_request_uses_documented_json_payload_type() {
        let request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "access-token-123".to_string(),
            environment: CTraderEnvironment::Demo,
        };

        let message = build_account_list_by_access_token_json(&request, "cm-id-1");

        assert_eq!(message.client_msg_id, "cm-id-1");
        assert_eq!(message.payload_type, 2149);
        assert_eq!(
            message.payload.get("accessToken").and_then(|value| value.as_str()),
            Some("access-token-123")
        );
    }

    #[test]
    fn account_list_response_parses_discovered_accounts() {
        let response = serde_json::json!({
            "clientMsgId": "server-msg-1",
            "payloadType": 2150,
            "payload": {
                "accessToken": "access-token-123",
                "permissionScope": "SCOPE_TRADE",
                "ctidTraderAccount": [
                    {
                        "ctidTraderAccountId": 101,
                        "isLive": true,
                        "traderLogin": 500101,
                        "brokerTitleShort": "Broker A"
                    },
                    {
                        "ctidTraderAccountId": 202,
                        "isLive": false,
                        "traderLogin": 500202,
                        "brokerTitleShort": "Broker B"
                    }
                ]
            }
        });

        let result = parse_account_list_by_access_token_json(&response.to_string())
            .expect("account list response should parse");

        assert_eq!(result.access_token, "access-token-123");
        assert_eq!(result.permission_scope, "SCOPE_TRADE");
        assert_eq!(result.accounts.len(), 2);
        assert_eq!(result.accounts[0].account_id, "101");
        assert_eq!(result.accounts[0].broker_title, "Broker A");
        assert_eq!(result.accounts[0].trader_login, Some(500101));
        assert_eq!(result.accounts[0].is_live, Some(true));
        assert_eq!(result.accounts[1].is_live, Some(false));
    }

    #[test]
    fn account_discovery_request_can_be_built_for_live_and_demo_environments() {
        let live_request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "live-token".to_string(),
            environment: CTraderEnvironment::Live,
        };
        let demo_request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "demo-token".to_string(),
            environment: CTraderEnvironment::Demo,
        };

        assert_eq!(live_request.endpoint_host(), "live.ctraderapi.com");
        assert_eq!(demo_request.endpoint_host(), "demo.ctraderapi.com");
    }

    #[test]
    fn account_discovery_exchange_sends_app_auth_then_account_list() {
        let transport = StubCTraderOpenApiTransport::with_responses(vec![
            Ok(serde_json::json!({
                "clientMsgId": "app-auth-1",
                "payloadType": 2101,
                "payload": {}
            })
            .to_string()),
            Ok(serde_json::json!({
                "clientMsgId": "account-list-1",
                "payloadType": 2150,
                "payload": {
                    "accessToken": "access-token-123",
                    "permissionScope": "SCOPE_TRADE",
                    "ctidTraderAccount": [
                        {
                            "ctidTraderAccountId": 101,
                            "isLive": true,
                            "traderLogin": 500101,
                            "brokerTitleShort": "Broker A"
                        }
                    ]
                }
            })
            .to_string()),
        ]);
        let request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "access-token-123".to_string(),
            environment: CTraderEnvironment::Live,
        };

        let result = perform_account_discovery_with_transport(&transport, &request)
            .expect("account discovery should succeed");
        let sent = transport.sent_messages();

        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].payload_type, 2100);
        assert_eq!(sent[1].payload_type, 2149);
        assert_eq!(result.accounts.len(), 1);
        assert_eq!(result.accounts[0].account_id, "101");
        assert_eq!(result.accounts[0].is_live, Some(true));
    }

    #[test]
    fn account_discovery_exchange_surfaces_ctrader_error_payload() {
        let transport = StubCTraderOpenApiTransport::with_responses(vec![Ok(
            serde_json::json!({
                "clientMsgId": "app-auth-1",
                "payloadType": 2142,
                "payload": {
                    "errorCode": "INVALID_ACCESS_TOKEN",
                    "description": "Access token is expired"
                }
            })
            .to_string(),
        )]);
        let request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "access-token-123".to_string(),
            environment: CTraderEnvironment::Live,
        };

        let err = perform_account_discovery_with_transport(&transport, &request)
            .expect_err("error payload should fail the exchange");

        assert!(err.to_string().contains("INVALID_ACCESS_TOKEN"));
    }

    #[test]
    fn account_discovery_exchange_surfaces_account_list_error_payload() {
        let transport = StubCTraderOpenApiTransport::with_responses(vec![
            Ok(serde_json::json!({
                "clientMsgId": "app-auth-1",
                "payloadType": 2101,
                "payload": {}
            })
            .to_string()),
            Ok(serde_json::json!({
                "clientMsgId": "account-list-1",
                "payloadType": 2142,
                "payload": {
                    "errorCode": "ACCOUNTS_LIST_FAILED",
                    "description": "Access token has no linked accounts"
                }
            })
            .to_string()),
        ]);
        let request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "access-token-123".to_string(),
            environment: CTraderEnvironment::Live,
        };

        let err = perform_account_discovery_with_transport(&transport, &request)
            .expect_err("account list error payload should fail the exchange");

        assert!(err.to_string().contains("cTrader account list failed"));
        assert!(err.to_string().contains("ACCOUNTS_LIST_FAILED"));
    }

    #[test]
    fn account_discovery_exchange_ignores_unrelated_frames_until_expected_response() {
        let transport = StubCTraderOpenApiTransport::with_responses(vec![
            Ok(serde_json::json!({
                "clientMsgId": "noise-1",
                "payloadType": 9999,
                "payload": {}
            })
            .to_string()),
            Ok(serde_json::json!({
                "clientMsgId": "app-auth-1",
                "payloadType": 2101,
                "payload": {}
            })
            .to_string()),
            Ok(serde_json::json!({
                "clientMsgId": "noise-2",
                "payloadType": 9998,
                "payload": {}
            })
            .to_string()),
            Ok(serde_json::json!({
                "clientMsgId": "account-list-1",
                "payloadType": 2150,
                "payload": {
                    "accessToken": "access-token-123",
                    "permissionScope": "SCOPE_TRADE",
                    "ctidTraderAccount": [
                        {
                            "ctidTraderAccountId": 101,
                            "isLive": true,
                            "traderLogin": 500101,
                            "brokerTitleShort": "Broker A"
                        }
                    ]
                }
            })
            .to_string()),
        ]);
        let request = CTraderAccountDiscoveryRequest {
            client_id: "client-id".to_string(),
            client_secret: "secret-456".to_string(),
            access_token: "access-token-123".to_string(),
            environment: CTraderEnvironment::Live,
        };

        let result = perform_account_discovery_with_transport(&transport, &request)
            .expect("account discovery should ignore unrelated frames");

        assert_eq!(result.accounts.len(), 1);
        assert_eq!(result.accounts[0].account_id, "101");
    }
}
