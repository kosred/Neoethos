//! HTTP access to the running NeoEthos backend + the demo guard.
//!
//! Everything the MCP tools do funnels through [`Backend`]. Three invariants
//! live here, enforced by the compiler where possible:
//!
//! 1. **Backend-not-running is a first-class, loud failure.** Every request
//!    that cannot reach the backend returns the fixed, actionable
//!    [`Backend::unreachable_message`] — no retries, no silent degradation,
//!    no in-process fallback (design §1.2).
//! 2. **The demo guard is a compile-time choke point.** The only function
//!    that can POST to a trade-affecting route is [`Backend::trade_post`],
//!    whose signature requires a [`DemoProof`]. The only constructor of
//!    `DemoProof` is [`Backend::ensure_demo`], which re-verifies the active
//!    account against the live backend on EVERY call (no caching). A handler
//!    that forgets the guard does not compile (design §2).
//! 3. **No unwrap/panic on fallible paths.** Every failure becomes a
//!    [`ToolError`] whose message the MCP client renders verbatim.

use serde_json::{Value, json};

/// The exact set of trade-affecting backend routes. [`Backend::trade_post`]
/// refuses to post anywhere else, and nothing else in this crate may post
/// here. Kept as data (not just convention) so tests can assert coverage.
pub const TRADE_ROUTES: &[&str] = &[
    "/orders",
    "/orders/pending",
    "/orders/cancel",
    "/positions/close",
    "/positions/protection",
    "/actions/{id}/confirm",
    "/autonomous/start",
    "/strategy_lab/promote",
    "/risk/preset",
];

/// The exact set of demo-guarded MCP tools (design §3). A unit test asserts
/// this list equals the set of tools annotated `destructive_hint=true ∧
/// read_only_hint≠true` minus `update_settings` (hint-only) — belt and
/// braces on top of the [`DemoProof`] compile error.
pub const GUARDED_TOOLS: &[&str] = &[
    "open_position",
    "close_position",
    "modify_position_protection",
    "place_pending_order",
    "cancel_pending_order",
    "confirm_pending_action",
    "autopilot_start",
    "promote_strategies",
    "apply_risk_preset",
];

/// A tool-level failure. The MCP layer renders `self.0` verbatim as an
/// `isError: true` tool result, so every message here must stand alone and
/// tell the agent what happened and what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError(pub String);

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ToolError {}

/// Proof that [`Backend::ensure_demo`] verified the active broker account is
/// a demo account *for this specific call*. Zero-field, private constructor,
/// deliberately NOT `Clone`/`Copy` — a handler cannot mint one, stash one,
/// or reuse one across calls. The only path to a `DemoProof` is a fresh
/// round-trip through the guard.
pub struct DemoProof(());

/// Thin HTTP client of the running NeoEthos backend.
pub struct Backend {
    client: reqwest::Client,
    /// Base URL without a trailing slash, e.g. `http://127.0.0.1:7423`.
    base_url: String,
    /// Optional bearer forwarded when the operator enabled
    /// `NEOETHOS_API_TOKEN` on the backend (passed to us as `--token`).
    token: Option<String>,
}

impl Backend {
    /// Build the client. `base_url` is normalized (trailing `/` stripped).
    ///
    /// Timeouts: connect fails fast (the backend is loopback — if it takes
    /// more than 5 s to accept, it is not running), while the overall
    /// request budget is 600 s to cover the one deliberately long
    /// synchronous tool (`replay_backtest`) and slow `fetch_history`
    /// downloads, matching the shipped Codex `tool_timeout_sec = 600`.
    pub fn new(base_url: &str, token: Option<String>) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build the HTTP client: {e}"))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The fixed message for a backend that cannot be reached (design §1.2).
    /// Verbatim per the design doc — tests assert on it.
    pub fn unreachable_message(&self) -> String {
        format!(
            "NeoEthos backend is not reachable at {} (connection refused). Start the desktop \
             app or run 'neoethos-app --server', then retry. This control plane never falls \
             back to acting on its own.",
            self.base_url
        )
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .request(method, format!("{}{path}", self.base_url));
        if let Some(token) = &self.token {
            builder = builder.bearer_auth(token);
        }
        builder
    }

    /// Map a transport-level failure (connection refused, timeout, …) to the
    /// fixed unreachable message. HTTP errors (status codes) never take this
    /// path — they carry the backend's own body via [`Self::read_response`].
    fn transport_error(&self, err: reqwest::Error) -> ToolError {
        ToolError(format!(
            "{} (transport error: {err})",
            self.unreachable_message()
        ))
    }

    /// Read a response: 2xx → parsed JSON body (or `null` when empty);
    /// anything else → a `ToolError` carrying the HTTP status and the
    /// backend's error body verbatim.
    async fn read_response(&self, resp: reqwest::Response) -> Result<Value, ToolError> {
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ToolError(format!("failed to read the backend response body: {e}")))?;
        let value: Value = if text.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or_else(|_| json!({ "raw": text }))
        };
        if status.is_success() {
            Ok(value)
        } else {
            Err(ToolError(format!(
                "backend returned HTTP {status}: {value}"
            )))
        }
    }

    /// GET a backend route with optional query pairs.
    pub async fn get_json(&self, path: &str, query: &[(&str, String)]) -> Result<Value, ToolError> {
        let resp = self
            .request(reqwest::Method::GET, path)
            .query(query)
            .send()
            .await
            .map_err(|e| self.transport_error(e))?;
        self.read_response(resp).await
    }

    /// POST a JSON body to a NON-trade backend route. Posting to a trade
    /// route through here is a bug; it fails loudly instead of executing.
    pub async fn post_json(&self, path: &str, body: &Value) -> Result<Value, ToolError> {
        if Self::is_trade_route(path) {
            return Err(ToolError(format!(
                "INTERNAL GUARD VIOLATION: '{path}' is a trade-affecting route and may only \
                 be reached through trade_post() with a DemoProof. This call was refused."
            )));
        }
        self.post_json_unchecked(path, body).await
    }

    async fn post_json_unchecked(&self, path: &str, body: &Value) -> Result<Value, ToolError> {
        let resp = self
            .request(reqwest::Method::POST, path)
            .json(body)
            .send()
            .await
            .map_err(|e| self.transport_error(e))?;
        self.read_response(resp).await
    }

    fn is_trade_route(path: &str) -> bool {
        if TRADE_ROUTES.contains(&path) {
            return true;
        }
        // `/actions/{id}/confirm` is templated; match the concrete form.
        path.starts_with("/actions/") && path.ends_with("/confirm")
    }

    /// The ONLY function that may POST to a trade-affecting route. The
    /// `DemoProof` parameter is consumed (moved), so one guard round-trip
    /// authorizes exactly one trade-affecting request.
    pub async fn trade_post(
        &self,
        path: &str,
        body: &Value,
        _proof: DemoProof,
    ) -> Result<Value, ToolError> {
        if !Self::is_trade_route(path) {
            return Err(ToolError(format!(
                "INTERNAL GUARD VIOLATION: trade_post() was called for '{path}', which is not \
                 in the registered trade-route set. This call was refused."
            )));
        }
        self.post_json_unchecked(path, body).await
    }

    // ── The demo guard ────────────────────────────────────────────────────

    /// Verify — fresh, against the live backend, with no caching — that the
    /// active broker account is a demo account. This is the ONLY constructor
    /// of [`DemoProof`].
    ///
    /// Checks (design §2):
    /// 1. `GET /broker/status` must report `connected == true` and
    ///    `environment == "Demo"`. `environment == "Live"` refuses with the
    ///    dedicated live message.
    /// 2. `GET /broker/accounts` must report `isLive == false` for the
    ///    active account. `null`/missing (unknown) FAILS — fail-closed.
    /// 3. Any HTTP failure during 1–2 fails the guard.
    pub async fn ensure_demo(&self) -> Result<DemoProof, ToolError> {
        let status = self.get_json("/broker/status", &[]).await.map_err(|e| {
            ToolError(format!(
                "DEMO GUARD REFUSED (fail-closed): could not read /broker/status to verify \
                 the account environment. Underlying error: {e}"
            ))
        })?;
        let connected = status
            .get("connected")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let environment = status
            .get("environment")
            .and_then(Value::as_str)
            .unwrap_or("(missing)")
            .to_string();
        let account_id = status
            .get("accountId")
            .and_then(Value::as_str)
            .unwrap_or("(unknown)")
            .to_string();

        if environment == "Live" {
            return Err(ToolError(format!(
                "DEMO GUARD REFUSED: the active broker environment is 'Live' (account \
                 {account_id}). This control plane operates demo accounts only — no \
                 trade-affecting tool will execute against a live environment, ever. Switch \
                 the active account to Demo in the desktop app, then retry."
            )));
        }

        let accounts = self.get_json("/broker/accounts", &[]).await.map_err(|e| {
            ToolError(format!(
                "DEMO GUARD REFUSED (fail-closed): could not read /broker/accounts to verify \
                 the active account is a demo account. Underlying error: {e}"
            ))
        })?;
        let is_live: Option<bool> = accounts
            .get("accounts")
            .and_then(Value::as_array)
            .and_then(|list| {
                list.iter().find(|a| {
                    a.get("accountId").and_then(Value::as_str) == Some(account_id.as_str())
                })
            })
            .and_then(|a| a.get("isLive"))
            .and_then(Value::as_bool);

        if connected && environment == "Demo" && is_live == Some(false) {
            Ok(DemoProof(()))
        } else {
            Err(ToolError(format!(
                "DEMO GUARD REFUSED (fail-closed): cannot positively verify the active \
                 account is a demo account (broker connected={connected}, \
                 environment={environment}, is_live={is_live:?}). Trade-affecting tools stay \
                 locked until /broker/status reports connected=true + environment=Demo and \
                 /broker/accounts reports is_live=false for the active account."
            )))
        }
    }
}
