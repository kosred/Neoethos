use crate::app_services::ctrader_auth::CTraderTokenBundle;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE, CTraderOpenApiJsonMessage,
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
    build_account_list_by_access_token_request, build_application_auth_request,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(test)]
use crate::app_services::ctrader_messages::{
    expected_response_payload_type, is_matching_open_api_response, parse_open_api_envelope,
};
#[cfg(test)]
use std::sync::{Arc, Mutex};

pub const CTRADER_DEFAULT_SCOPE: &str = "trading";
pub const CTRADER_TOKEN_ENDPOINT_BASE: &str = "https://openapi.ctrader.com";
const CTRADER_CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);
const CTRADER_CALLBACK_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLoopbackConfig {
    allowed_ports: Vec<u16>,
    callback_path: String,
    bind_host: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderCallbackPayload {
    pub authorization_code: String,
    /// Opaque `state` value echoed back by the cTrader authorization server.
    /// SECURITY (audit-fix F2): the OAuth client MUST compare this against the
    /// `state` it generated before opening the browser; mismatch indicates a
    /// CSRF / authorization-response-injection attempt and the callback must
    /// be rejected without exchanging the code.
    pub state: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderTokenRefreshRequest {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    pub scope: String,
}

pub trait CTraderLiveAuthBackend: Send + Sync {
    fn run(&self, request: CTraderLiveAuthRequest) -> Result<CTraderLiveAuthResult>;
    fn refresh_token_bundle(
        &self,
        request: &CTraderTokenRefreshRequest,
    ) -> Result<CTraderTokenBundle>;
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
    refresh_outcome: Arc<Mutex<Option<Result<CTraderTokenBundle, String>>>>,
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
    access_token: Option<String>,
    #[serde(rename = "refreshToken")]
    refresh_token: Option<String>,
    #[serde(rename = "tokenType")]
    token_type: Option<String>,
    #[serde(rename = "expiresIn")]
    expires_in: Option<i64>,
    scope: Option<String>,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
    description: Option<String>,
}

impl CTraderLoopbackConfig {
    pub fn new(
        primary_port: u16,
        fallback_ports: Vec<u16>,
        callback_path: impl Into<String>,
    ) -> Self {
        let mut allowed_ports = vec![primary_port];
        allowed_ports.extend(fallback_ports);
        Self {
            allowed_ports,
            callback_path: callback_path.into(),
            bind_host: "127.0.0.1".to_string(),
        }
    }

    pub fn with_bind_host(
        primary_port: u16,
        fallback_ports: Vec<u16>,
        callback_path: impl Into<String>,
        bind_host: impl Into<String>,
    ) -> Self {
        let mut config = Self::new(primary_port, fallback_ports, callback_path);
        config.bind_host = bind_host.into();
        config
    }

    pub fn callback_path(&self) -> &str {
        &self.callback_path
    }

    pub fn allowed_ports(&self) -> &[u16] {
        &self.allowed_ports
    }

    pub fn bind_host(&self) -> &str {
        &self.bind_host
    }
}

impl ProductionCTraderLiveAuthBackend {
    fn bind_loopback_listener(&self, config: &CTraderLoopbackConfig) -> Result<(u16, TcpListener)> {
        for port in config.allowed_ports() {
            match TcpListener::bind((config.bind_host(), *port)) {
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
        self.capture_authorization_code_with_timeout(
            listener,
            expected_path,
            CTRADER_CALLBACK_TIMEOUT,
        )
    }

    fn capture_authorization_code_with_timeout(
        &self,
        listener: TcpListener,
        expected_path: &str,
        timeout: Duration,
    ) -> Result<String> {
        // Back-compat: keep the pre-F2 entrypoint working for unit tests
        // that don't exercise the state-validation path. Production
        // `run()` always uses `..._with_state` below.
        self.capture_authorization_code_with_state(listener, expected_path, None, timeout)
    }

    fn capture_authorization_code_with_state(
        &self,
        listener: TcpListener,
        expected_path: &str,
        expected_state: Option<&str>,
        timeout: Duration,
    ) -> Result<String> {
        listener
            .set_nonblocking(true)
            .context("failed to configure cTrader callback listener")?;
        let started = Instant::now();
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    return self.read_authorization_code_from_stream(
                        stream,
                        expected_path,
                        expected_state,
                    );
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    if started.elapsed() >= timeout {
                        return Err(anyhow!(
                            "timed out waiting for cTrader callback; verify the registered redirect_uri and browser login flow"
                        ));
                    }
                    let remaining = timeout.saturating_sub(started.elapsed());
                    std::thread::sleep(CTRADER_CALLBACK_POLL_INTERVAL.min(remaining));
                }
                Err(err) => return Err(err).context("failed to accept cTrader callback"),
            }
        }
    }

    fn read_authorization_code_from_stream(
        &self,
        mut stream: TcpStream,
        expected_path: &str,
        expected_state: Option<&str>,
    ) -> Result<String> {
        stream
            .set_nonblocking(false)
            .context("failed to configure cTrader callback stream")?;
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
        // SECURITY (audit-fix F2): when a state token was issued, the
        // callback must carry the same value or we refuse to surface the
        // authorization code to the rest of the flow.
        let payload = match expected_state {
            Some(state) => parse_callback_request_with_state(request_target, expected_path, state)?,
            None => parse_callback_request(request_target, expected_path)?,
        };

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
        stream
            .flush()
            .context("failed to flush cTrader callback response")?;
        Ok(payload.authorization_code)
    }

    fn exchange_token(
        &self,
        request: &CTraderLiveAuthRequest,
        callback_port: u16,
        authorization_code: &str,
    ) -> Result<CTraderTokenBundle> {
        let redirect_uri = rewrite_redirect_uri_port(&request.redirect_uri, callback_port)?;
        // reqwest >= 0.13.3 logs only the host, not the full URL — protects client_secret from leaking to debug/trace logs.
        // The cTrader token endpoint is documented as a GET with credentials as URL query
        // parameters (see help.ctrader.com/open-api/account-authentication/). Moving
        // `client_secret` into a POST body breaks against the real broker.
        let url = build_token_exchange_endpoint_url(CTRADER_TOKEN_ENDPOINT_BASE);
        let query = build_token_exchange_form(
            "authorization_code",
            authorization_code,
            &redirect_uri,
            &request.client_id,
            &request.client_secret,
        );
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .query(&query)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .context("failed to call cTrader token endpoint")?
            .error_for_status()
            .context("cTrader token endpoint returned an error status")?;
        let body = response
            .text()
            .context("failed to read cTrader token response body")?;
        parse_token_bundle_response(&body, &request.scope, current_unix_seconds()?)
    }
}

impl CTraderLiveAuthBackend for ProductionCTraderLiveAuthBackend {
    fn run(&self, request: CTraderLiveAuthRequest) -> Result<CTraderLiveAuthResult> {
        // Step 1/5 — bind loopback listener
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "bind_loopback",
            host = %request.loopback.bind_host(),
            scope = %request.scope,
            "starting cTrader OAuth flow"
        );
        let (callback_port, listener) = self.bind_loopback_listener(&request.loopback)
            .with_context(|| format!(
                "OAuth step 1/5 (bind_loopback) failed — could not bind any of the allowed callback ports {:?} on host {}. \
                 Another process is likely holding the port; close any other ForexAI instance or pick a different port range in settings.",
                request.loopback.allowed_ports(),
                request.loopback.bind_host(),
            ))?;
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "bind_loopback",
            callback_port,
            "loopback listener bound"
        );

        // Step 2/5 — build authorize URL
        // SECURITY (audit-fix F2): mint a fresh CSRF-state token for this
        // flow, embed it in the authorize URL, and pass it to the callback
        // capture loop so the redirect-uri handler can reject any callback
        // that doesn't echo it back.
        let oauth_state = generate_oauth_state();
        let authorize_url = build_authorize_url_with_state(
            &request.client_id,
            &request.redirect_uri,
            callback_port,
            &request.scope,
            &oauth_state,
        )
        .with_context(|| {
            format!(
                "OAuth step 2/5 (build_authorize_url) failed — the redirect_uri '{}' \
             configured in your cTrader app must be a valid http(s) URL on the loopback host. \
             Check the redirect URI registered in https://openapi.ctrader.com.",
                request.redirect_uri,
            )
        })?;
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "build_authorize_url",
            client_id_prefix = %request.client_id.chars().take(6).collect::<String>(),
            state_len = oauth_state.len(),
            "authorize URL constructed with CSRF state"
        );

        // Step 3/5 — open system browser
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "open_browser",
            "launching system browser for cTrader login (this opens openapi.ctrader.com in your default browser)"
        );
        open::that(&authorize_url).with_context(|| format!(
            "OAuth step 3/5 (open_browser) failed — could not launch a system browser to {}. \
             On a headless server you cannot complete OAuth this way; run on a desktop session, \
             or copy the URL above to a browser on another machine and paste the redirect back manually.",
            authorize_url,
        ))?;

        // Step 4/5 — wait for callback
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "wait_for_callback",
            callback_port,
            timeout_secs = CTRADER_CALLBACK_TIMEOUT.as_secs(),
            "waiting for browser to redirect back to the loopback listener"
        );
        let authorization_code = self
            .capture_authorization_code_with_state(
                listener,
                request.loopback.callback_path(),
                Some(oauth_state.as_str()),
                CTRADER_CALLBACK_TIMEOUT,
            )
            .with_context(|| format!(
                "OAuth step 4/5 (wait_for_callback) failed — no valid callback received on port {} within the timeout window. \
                 Common causes: (a) the redirect_uri registered in your cTrader app does not match http://{}:{}{} ; \
                 (b) the browser was closed before approving; (c) firewall/AV blocked the loopback connection; \
                 (d) the callback `state` did not match — possible CSRF, rejected.",
                callback_port,
                request.loopback.bind_host(),
                callback_port,
                request.loopback.callback_path(),
            ))?;
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "wait_for_callback",
            "authorization_code received"
        );

        // Step 5/5 — exchange code for token bundle
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "exchange_token",
            "exchanging authorization_code for access + refresh token"
        );
        let token_bundle = self.exchange_token(&request, callback_port, &authorization_code)
            .with_context(|| {
                "OAuth step 5/5 (exchange_token) failed — the cTrader token endpoint returned an error. \
                 Check that your client_id and client_secret in settings match the cTrader app you registered, \
                 and that the app has not been revoked at https://openapi.ctrader.com."
                    .to_string()
            })?;
        tracing::info!(
            target: "forex_app::ctrader::oauth",
            step = "exchange_token",
            scope = %request.scope,
            access_token_len = token_bundle.access_token.len(),
            refresh_token_present = !token_bundle.refresh_token.is_empty(),
            "token bundle issued — OAuth flow complete"
        );
        Ok(CTraderLiveAuthResult {
            callback_port,
            authorization_code,
            token_bundle,
        })
    }

    fn refresh_token_bundle(
        &self,
        request: &CTraderTokenRefreshRequest,
    ) -> Result<CTraderTokenBundle> {
        // reqwest >= 0.13.3 logs only the host, not the full URL — protects client_secret from leaking to debug/trace logs.
        // Per cTrader's documented OAuth flow the refresh path is also `GET /apps/token`
        // with credentials as URL query parameters.
        let url = build_token_exchange_endpoint_url(CTRADER_TOKEN_ENDPOINT_BASE);
        let query = build_refresh_token_exchange_form(
            &request.refresh_token,
            &request.client_id,
            &request.client_secret,
        );
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .query(&query)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .context("failed to call cTrader refresh token endpoint")?
            .error_for_status()
            .context("cTrader refresh token endpoint returned an error status")?;
        let body = response
            .text()
            .context("failed to read cTrader refresh token response body")?;
        parse_token_bundle_response(&body, &request.scope, current_unix_seconds()?)
    }
}

#[cfg(test)]
impl StubCTraderLiveAuthBackend {
    pub fn success(result: CTraderLiveAuthResult) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(result)))),
            refresh_outcome: Arc::new(Mutex::new(Some(Err(
                "stub cTrader refresh backend was not configured".to_string(),
            )))),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
            refresh_outcome: Arc::new(Mutex::new(Some(Err(
                "stub cTrader refresh backend was not configured".to_string(),
            )))),
        }
    }

    pub fn with_refresh_success(result: CTraderTokenBundle) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(
                "stub cTrader live auth backend was not configured".to_string(),
            )))),
            refresh_outcome: Arc::new(Mutex::new(Some(Ok(result)))),
        }
    }

    pub fn with_refresh_failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(
                "stub cTrader live auth backend was not configured".to_string(),
            )))),
            refresh_outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
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

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
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
            .unwrap_or_else(|| {
                Err("stub cTrader account discovery backend was already consumed".to_string())
            }) {
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
            .unwrap_or_else(|| {
                Err("stub cTrader live auth backend was already consumed".to_string())
            }) {
            Ok(result) => Ok(result),
            Err(message) => Err(anyhow!(message)),
        }
    }

    fn refresh_token_bundle(
        &self,
        _request: &CTraderTokenRefreshRequest,
    ) -> Result<CTraderTokenBundle> {
        match self
            .refresh_outcome
            .lock()
            .expect("stub cTrader refresh backend lock poisoned")
            .take()
            .unwrap_or_else(|| Err("stub cTrader refresh backend was already consumed".to_string()))
        {
            Ok(result) => Ok(result),
            Err(message) => Err(anyhow!(message)),
        }
    }
}

pub fn build_default_loopback_config(redirect_uri: &str) -> Result<CTraderLoopbackConfig> {
    let parts = parse_redirect_uri_parts(redirect_uri)?;
    if !is_loopback_redirect_host(&parts.bind_host) {
        return Err(anyhow!(
            "cTrader loopback redirect URI host must be localhost, 127.0.0.1, or [::1]"
        ));
    }
    Ok(CTraderLoopbackConfig::with_bind_host(
        parts.port,
        vec![parts.port.saturating_add(1), parts.port.saturating_add(2)],
        callback_path_from_suffix(&parts.suffix),
        parts.bind_host,
    ))
}

fn is_loopback_redirect_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

pub fn build_authorize_url(
    client_id: &str,
    redirect_uri: &str,
    callback_port: u16,
    scope: &str,
) -> Result<String> {
    // Back-compat shim for callers/tests that don't supply state. Real
    // production code must use `build_authorize_url_with_state` and verify
    // the echoed `state` on the callback. See audit-fix F2.
    let redirect_uri = rewrite_redirect_uri_port(redirect_uri, callback_port)?;
    Ok(format!(
        "https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={}&redirect_uri={}&scope={}&product=web",
        percent_encode(client_id),
        percent_encode(&redirect_uri),
        percent_encode(scope),
    ))
}

/// Build the cTrader authorize URL with a CSRF `state` parameter.
///
/// SECURITY (audit-fix F2): RFC 6749 §10.12 mandates the OAuth client both
/// (a) include an unguessable, per-flow `state` value in the authorize
/// request and (b) verify that the same value is echoed back on the
/// redirect-uri callback. Without this check, an attacker who can induce
/// the user's browser to follow a crafted callback URL can splice in their
/// own authorization code and trick the bot into binding to the attacker's
/// cTrader account. The token-exchange round-trip would still succeed
/// because the broker accepts whichever code is supplied.
///
/// Generate `state` with [`generate_oauth_state`], pass it here, then pass
/// the SAME string to [`parse_callback_request_with_state`].
pub fn build_authorize_url_with_state(
    client_id: &str,
    redirect_uri: &str,
    callback_port: u16,
    scope: &str,
    state: &str,
) -> Result<String> {
    if state.trim().is_empty() {
        return Err(anyhow!("OAuth state token must not be empty"));
    }
    let redirect_uri = rewrite_redirect_uri_port(redirect_uri, callback_port)?;
    Ok(format!(
        "https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={}&redirect_uri={}&scope={}&state={}&product=web",
        percent_encode(client_id),
        percent_encode(&redirect_uri),
        percent_encode(scope),
        percent_encode(state),
    ))
}

/// Generate a cryptographically random, URL-safe OAuth `state` token using
/// the OS entropy source. 32 bytes of entropy → 256-bit unguessability,
/// which is well above the OAuth 2.0 Security Best Current Practice
/// minimum (128 bits).
pub fn generate_oauth_state() -> String {
    use base64::Engine as _;
    use rand::TryRngCore;
    let mut bytes = [0_u8; 32];
    // OsRng pulls from /dev/urandom on Linux and BCryptGenRandom on Windows.
    // We tolerate a single retry if the OS RNG transiently fails — if both
    // attempts fail the system is in a state where issuing an OAuth flow
    // would be unsafe anyway, so we panic. The OS RNG failing twice in
    // succession indicates a kernel-level entropy fault.
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .or_else(|_| rand::rngs::OsRng.try_fill_bytes(&mut bytes))
        .expect("OS RNG failed to produce OAuth state entropy");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

pub fn parse_callback_request(
    request_target: &str,
    expected_path: &str,
) -> Result<CTraderCallbackPayload> {
    let (path, query) = request_target
        .split_once('?')
        .map_or((request_target, ""), |(path, query)| (path, query));
    if path != expected_path {
        return Err(anyhow!("unexpected callback path: {path}"));
    }

    let mut authorization_code = None;
    let mut state = None;
    let mut denial_error = None;
    let mut denial_description = None;

    for pair in query.split('&').filter(|pair| !pair.trim().is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let decoded_value = percent_decode(value)?;
        match key {
            "code" => authorization_code = Some(decoded_value),
            "state" => state = Some(decoded_value),
            "error" => denial_error = Some(decoded_value),
            "error_description" => denial_description = Some(decoded_value),
            _ => {}
        }
    }

    if let Some(error) = denial_error.filter(|error| !error.trim().is_empty()) {
        let description = denial_description
            .filter(|description| !description.trim().is_empty())
            .map(|description| format!(": {description}"))
            .unwrap_or_default();
        return Err(anyhow!(
            "cTrader authorization denied: {error}{description}"
        ));
    }

    let authorization_code = authorization_code
        .filter(|code| !code.trim().is_empty())
        .context("missing authorization code")?;

    Ok(CTraderCallbackPayload {
        authorization_code,
        state,
    })
}

/// Same as [`parse_callback_request`] but additionally enforces the OAuth
/// `state` CSRF check (audit-fix F2). The caller passes the `state` value
/// that was sent to the authorization server; this function rejects the
/// callback if the echoed value is missing, empty, or doesn't match.
///
/// We use a length-checked equality first (cheap reject) and then a
/// constant-time byte compare so a network-adjacent attacker can't measure
/// timing to learn a prefix of our state.
pub fn parse_callback_request_with_state(
    request_target: &str,
    expected_path: &str,
    expected_state: &str,
) -> Result<CTraderCallbackPayload> {
    if expected_state.trim().is_empty() {
        return Err(anyhow!(
            "OAuth state validation requested but no state token was issued; refusing to accept callback"
        ));
    }
    let payload = parse_callback_request(request_target, expected_path)?;
    let received = payload
        .state
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow!(
                "cTrader callback missing `state` parameter — possible CSRF; refusing to exchange authorization code"
            )
        })?;
    if !constant_time_eq(expected_state.as_bytes(), received.as_bytes()) {
        return Err(anyhow!(
            "cTrader callback `state` mismatch — possible CSRF or authorization-response-injection; refusing to exchange authorization code"
        ));
    }
    Ok(payload)
}

/// Constant-time byte comparison. Avoids leaking the length of the matching
/// prefix via timing side-channels.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// URL for the cTrader token endpoint. Per the cTrader help-centre
/// (help.ctrader.com/open-api/account-authentication/), this endpoint is
/// invoked as a `GET` with `client_id`/`client_secret`/`code`/`redirect_uri`
/// passed as URL query parameters. The matching parameter set is produced by
/// [`build_token_exchange_form`] / [`build_refresh_token_exchange_form`] and
/// is passed via `RequestBuilder::query(...)` at the call site.
pub fn build_token_exchange_endpoint_url(base_url: &str) -> String {
    format!("{}/apps/token", base_url.trim_end_matches('/'))
}

/// Build the URL-query parameter list for the authorization-code grant
/// against `/apps/token`. The parameters are intentionally returned as a
/// `Vec<(name, value)>` so they can be handed to `RequestBuilder::query(...)`
/// (Spotware's documented flow uses GET + query string for this endpoint).
pub fn build_token_exchange_form(
    grant_type: &str,
    code: &str,
    redirect_uri: &str,
    client_id: &str,
    client_secret: &str,
) -> Vec<(&'static str, String)> {
    vec![
        ("grant_type", grant_type.to_string()),
        ("code", code.to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("client_id", client_id.to_string()),
        ("client_secret", client_secret.to_string()),
    ]
}

/// Same as [`build_token_exchange_form`] but for the `refresh_token` grant.
pub fn build_refresh_token_exchange_form(
    refresh_token: &str,
    client_id: &str,
    client_secret: &str,
) -> Vec<(&'static str, String)> {
    vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.to_string()),
        ("client_id", client_id.to_string()),
        ("client_secret", client_secret.to_string()),
    ]
}

pub fn build_application_auth_json(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    build_application_auth_request(client_id, client_secret, client_msg_id)
}

pub fn build_account_list_by_access_token_json(
    request: &CTraderAccountDiscoveryRequest,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    build_account_list_by_access_token_request(&request.access_token, client_msg_id)
}

// v0.4.13 — the wire format from the live `demo.ctraderapi.com:5036`
// endpoint differs from the integration-test fixture in two ways that
// blew up the strict struct shape that previously lived here:
//
//   * `accessToken` is not always echoed back on the account-list
//     response — the server already returned it on the token-exchange
//     leg and treats the re-echo as optional. The strict `String`
//     field rejected envelopes without it.
//   * `permissionScope` arrives as the proto enum's numeric value
//     (`SCOPE_TRADE` → `2`, `SCOPE_VIEW` → `1`) in JSON-over-WSS
//     responses from production, even though the older fixtures used
//     the string spelling. A typed `String` field rejected the int.
//
// The struct is now permissive: both fields are `Option<Value>` and
// the post-parse handler treats absence as "no extra metadata,
// proceed with the account list".
#[derive(Debug, Deserialize)]
struct CTraderAccountListResponseEnvelope {
    #[serde(rename = "clientMsgId", default)]
    _client_msg_id: Option<String>,
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: CTraderAccountListResponsePayload,
}

#[derive(Debug, Deserialize)]
struct CTraderAccountListResponsePayload {
    #[serde(rename = "accessToken", default)]
    access_token: Option<String>,
    #[serde(rename = "permissionScope", default)]
    permission_scope: Option<Value>,
    #[serde(rename = "ctidTraderAccount", default)]
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
        .with_context(|| {
            // v0.4.13 — include the head of the offending body and the
            // total length so the wizard's "OAuth error: …" surface has
            // enough signal to triage a future schema drift without
            // extra logs. Same diagnostic shape as `parse_open_api_envelope`.
            let total = response_json.len();
            let head: String = response_json.chars().take(200).collect();
            format!(
                "failed to parse cTrader account list response \
                 (len={total}, head={head:?})"
            )
        })?;
    if envelope.payload_type != CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader account list payload type: {}",
            envelope.payload_type
        ));
    }

    let accounts: Vec<crate::app_services::ctrader_auth::CTraderDiscoveredAccount> = envelope
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

    // v0.4.13 — both fields are now `Option`. The `access_token` flows
    // back to the wizard so it can be re-applied if missing on subsequent
    // legs; empty-string fallback matches the pre-change contract for
    // downstream consumers that read it via direct field access.
    // `permission_scope` is reduced to its display form: a number arrives
    // as the proto enum value (e.g. "2" for SCOPE_TRADE), a string echoes
    // through verbatim; either way the result is human-readable.
    let access_token = envelope.payload.access_token.unwrap_or_default();
    let permission_scope = match envelope.payload.permission_scope {
        Some(Value::String(s)) => s,
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
        None => String::new(),
    };

    // v0.4.16 — log how many accounts the broker returned, so the
    // 6-of-7 missing-account observation from the 2026-05-19
    // walkthrough is debuggable from the operator's log without
    // re-running the wizard with a debugger attached. We log the IDs
    // only (no tokens, no permissionScope) so the line is safe to
    // ship at INFO level.
    let account_ids: Vec<String> = accounts
        .iter()
        .map(|a: &crate::app_services::ctrader_auth::CTraderDiscoveredAccount| a.account_id.clone())
        .collect();
    tracing::info!(
        target: "ctrader.auth",
        count = accounts.len(),
        ids = ?account_ids,
        "cTrader account-list response parsed"
    );

    Ok(CTraderAccountDiscoveryResult {
        access_token,
        permission_scope,
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

    let app_auth_envelope: CTraderOpenApiJsonMessage =
        serde_json::from_str(&responses[0]).context("failed to parse cTrader app auth response")?;
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

    let account_list_envelope: CTraderOpenApiJsonMessage = serde_json::from_str(&responses[1])
        .context("failed to parse cTrader account list response envelope")?;
    if account_list_envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        let error = parse_ctrader_error_payload(&account_list_envelope.payload)?;
        return Err(anyhow!("cTrader account list failed: {}", error));
    }

    parse_account_list_by_access_token_json(&responses[1])
}

pub fn discover_ctrader_accounts(
    request: &CTraderAccountDiscoveryRequest,
) -> Result<CTraderAccountDiscoveryResult> {
    let transport = ProductionCTraderOpenApiTransport::new(request.endpoint_host());
    perform_account_discovery_with_transport(&transport, request)
}

pub fn parse_token_bundle_response(
    response_json: &str,
    fallback_scope: &str,
    created_at_unix: i64,
) -> Result<CTraderTokenBundle> {
    let payload: CTraderTokenExchangeResponse =
        serde_json::from_str(response_json).context("failed to parse cTrader token response")?;
    if let Some(error_code) = payload.error_code.filter(|value| !value.trim().is_empty()) {
        let description = payload
            .description
            .filter(|value| !value.trim().is_empty())
            .map(|value| format!(": {value}"))
            .unwrap_or_default();
        return Err(anyhow!(
            "cTrader token exchange failed: {error_code}{description}"
        ));
    }

    Ok(CTraderTokenBundle {
        access_token: payload
            .access_token
            .filter(|value| !value.trim().is_empty())
            .context("cTrader token response did not include accessToken")?,
        refresh_token: payload
            .refresh_token
            .filter(|value| !value.trim().is_empty())
            .context("cTrader token response did not include refreshToken")?,
        token_type: payload
            .token_type
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "bearer".to_string()),
        expires_in: payload
            .expires_in
            .context("cTrader token response did not include expiresIn")?,
        scope: payload
            .scope
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| fallback_scope.to_string()),
        created_at_unix,
    })
}

fn parse_ctrader_error_payload(payload: &Value) -> Result<String> {
    #[derive(Debug, Deserialize)]
    struct CTraderErrorPayload {
        #[serde(rename = "errorCode")]
        error_code: String,
        description: Option<String>,
    }

    let error: CTraderErrorPayload =
        serde_json::from_value(payload.clone()).context("failed to parse cTrader error payload")?;
    Ok(match error.description {
        Some(description) if !description.trim().is_empty() => {
            format!("{}: {}", error.error_code, description)
        }
        _ => error.error_code,
    })
}

fn current_unix_seconds() -> Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_secs() as i64)
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
        self.sent_messages
            .lock()
            .expect("sent_messages lock")
            .clone()
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
        for message in messages {
            let expected_payload_type = expected_response_payload_type(message.payload_type)?;
            loop {
                if responses.is_empty() {
                    return Err(anyhow!(
                        "stub cTrader transport ran out of responses before matching payload {}",
                        expected_payload_type
                    ));
                }
                let response = responses.remove(0).map_err(|err| anyhow!(err))?;
                let envelope = parse_open_api_envelope(&response)?;
                if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                    output.push(response);
                    return Ok(output);
                }
                if is_matching_open_api_response(&envelope, message, expected_payload_type) {
                    output.push(response);
                    break;
                }
            }
        }
        Ok(output)
    }
}

fn rewrite_redirect_uri_port(redirect_uri: &str, callback_port: u16) -> Result<String> {
    let parts = parse_redirect_uri_parts(redirect_uri)?;
    Ok(format!(
        "{}://{}:{}{}",
        parts.scheme, parts.host_for_uri, callback_port, parts.suffix
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RedirectUriParts {
    scheme: String,
    host_for_uri: String,
    bind_host: String,
    port: u16,
    suffix: String,
}

fn parse_redirect_uri_parts(redirect_uri: &str) -> Result<RedirectUriParts> {
    let (scheme, remainder) = redirect_uri
        .split_once("://")
        .context("redirect URI is missing scheme")?;
    let (authority, suffix) = remainder
        .split_once('/')
        .map_or((remainder, ""), |(authority, suffix)| (authority, suffix));
    let (host_for_uri, bind_host, port) = parse_redirect_authority(authority)?;
    let suffix = if suffix.is_empty() {
        String::new()
    } else {
        format!("/{}", suffix.trim_start_matches('/'))
    };
    Ok(RedirectUriParts {
        scheme: scheme.to_string(),
        host_for_uri,
        bind_host,
        port,
        suffix,
    })
}

fn parse_redirect_authority(authority: &str) -> Result<(String, String, u16)> {
    if authority.trim().is_empty() {
        return Err(anyhow!("redirect URI host is missing"));
    }
    if let Some(remainder) = authority.strip_prefix('[') {
        let (host, after_host) = remainder
            .split_once(']')
            .context("redirect URI IPv6 host is missing closing bracket")?;
        let port = after_host
            .strip_prefix(':')
            .context("redirect URI is missing port")?
            .parse::<u16>()
            .context("redirect URI port is invalid")?;
        if host.trim().is_empty() {
            return Err(anyhow!("redirect URI host is missing"));
        }
        return Ok((format!("[{host}]"), host.to_string(), port));
    }

    let (host, port) = authority
        .rsplit_once(':')
        .context("redirect URI is missing port")?;
    if host.trim().is_empty() {
        return Err(anyhow!("redirect URI host is missing"));
    }
    if host.contains(':') {
        return Err(anyhow!("redirect URI IPv6 host must be bracketed"));
    }
    let port = port
        .parse::<u16>()
        .context("redirect URI port is invalid")?;
    Ok((host.to_string(), host.to_string(), port))
}

fn callback_path_from_suffix(suffix: &str) -> String {
    format!("/{}", suffix.trim_start_matches('/'))
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

fn percent_decode(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'%' if idx + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[idx + 1..idx + 3])
                    .context("invalid percent-encoded callback value")?;
                let byte = u8::from_str_radix(hex, 16)
                    .context("invalid percent-encoded callback value")?;
                decoded.push(byte);
                idx += 3;
            }
            b'+' => {
                decoded.push(b' ');
                idx += 1;
            }
            byte => {
                decoded.push(byte);
                idx += 1;
            }
        }
    }
    String::from_utf8(decoded).context("callback value is not valid UTF-8")
}

#[cfg(test)]
#[path = "ctrader_live_auth_tests.rs"]
mod tests;
