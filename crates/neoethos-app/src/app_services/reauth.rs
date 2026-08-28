//! Headless cTrader OAuth refresh. Extracted from `main.rs::--reauth`
//! so it can be invoked both from the CLI flag and from the running
//! HTTP server (`POST /broker/reauth`) without spawning a child
//! process.
//!
//! The flow:
//!   1. Load broker credentials from disk.
//!   2. Bind a loopback HTTP listener (43001 with 43002/43003 fallbacks).
//!   3. Open the Spotware consent screen in the user's default browser.
//!   4. Wait for the redirect carrying the auth code.
//!   5. Exchange the code for a token bundle.
//!   6. Persist the bundle to the Windows Credential Manager (keyring).
//!
//! The whole flow is synchronous blocking I/O internally (reqwest blocking
//! for the token exchange, std::net listener for the callback), so callers
//! must invoke this from a `spawn_blocking` worker — not directly on the
//! tokio runtime.

use anyhow::{Result, anyhow};

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_live_auth::{
    CTraderLiveAuthBackend, CTraderLiveAuthRequest, CTraderLoopbackConfig,
    ProductionCTraderLiveAuthBackend,
};
use crate::app_services::secure_store::production_ctrader_token_store;

/// Result returned to the caller. We deliberately don't leak the access
/// token itself — the caller already has the bundle persisted via the
/// keyring, and any further consumer should read it from there.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReauthOutcome {
    pub callback_port: u16,
    pub refresh_token_present: bool,
    pub access_token_len: usize,
    pub message: String,
}

/// Run the OAuth flow end-to-end. Must be called from a blocking
/// context (e.g. `tokio::task::spawn_blocking`).
pub fn run_reauth_flow_blocking() -> Result<ReauthOutcome> {
    let settings = load_broker_settings();
    let ct = &settings.ctrader;
    if ct.client_id.is_empty() || ct.client_secret.is_empty() {
        return Err(anyhow!(
            "cTrader client_id / client_secret are empty — set them in \
             .local/neoethos/broker_credentials.toml (or the embed env \
             vars) before re-authing"
        ));
    }

    let loopback = CTraderLoopbackConfig::new(43001, vec![43002, 43003], "/callback");
    let request = CTraderLiveAuthRequest {
        client_id: ct.client_id.clone(),
        client_secret: ct.client_secret.clone(),
        redirect_uri: ct.redirect_uri.clone(),
        // `trading` scope: ProtoOAAccountAuthReq requires it. Hard-coded
        // here so a misconfigured broker_credentials.toml can't downgrade
        // us back to `accounts`-only, which makes account-auth fail with
        // RET_ACCOUNT_DISABLED downstream.
        scope: "trading".to_string(),
        loopback,
    };

    let backend = ProductionCTraderLiveAuthBackend;
    let result = backend
        .run(request)
        .map_err(|e| anyhow!("OAuth flow failed: {e}"))?;

    production_ctrader_token_store()
        .save_token_bundle(&result.token_bundle)
        .map_err(|e| anyhow!("save_token_bundle failed: {e}"))?;

    Ok(ReauthOutcome {
        callback_port: result.callback_port,
        refresh_token_present: !result.token_bundle.refresh_token.is_empty(),
        access_token_len: result.token_bundle.access_token.len(),
        message: format!(
            "OAuth refresh complete via callback port {}; token saved to keyring",
            result.callback_port
        ),
    })
}
