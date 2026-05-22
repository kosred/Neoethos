//! Broker control endpoints.
//!
//! POST /broker/reauth — kick off the full cTrader OAuth flow. Opens
//! a browser window, captures the loopback callback, exchanges the
//! auth code for a token bundle, persists it to the keyring. Blocks
//! the HTTP response until the flow either completes or fails
//! (typical wall-clock time: 10–30 s depending on how fast the
//! operator clicks "Continue" in the consent screen).
//!
//! The bridge picks up the new token automatically on its next 5 s
//! refresh — no server restart needed.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::app_services::broker_config::{
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, BrokerSettingsState,
    CTraderBrokerEnvironment, CTraderBrokerSettings, DxTradeBrokerSettings,
};
use crate::app_services::broker_persistence::{load_broker_settings, save_broker_settings};
use crate::app_services::reauth::run_reauth_flow_blocking;

use super::state::AppApiState;

// ─── GET / POST /broker/credentials ───────────────────────────────────────

/// Wire DTO for the cTrader credentials form. Mirrors
/// `CTraderBrokerSettings` minus the transient `authorization_code_input`
/// field. The accountId is exposed as a single string for the UI even
/// though `BrokerSettingsState` carries a Vec<BrokerAccountTarget>;
/// most operators have one account, and the Vec stays as-is on disk.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialsDto {
    pub client_id: String,
    pub client_secret: String,
    #[serde(default)]
    pub redirect_uri: String,
    /// "Demo" or "Live"; defaults to "Demo" for safety.
    #[serde(default)]
    pub environment: String,
    /// Single account id the operator wants to trade against.
    #[serde(default)]
    pub account_id: String,
}

pub async fn credentials_get(State(_state): State<AppApiState>) -> Response {
    let settings = tokio::task::spawn_blocking(load_broker_settings).await;
    match settings {
        Ok(s) => {
            let ct = &s.ctrader;
            // Never echo client_secret in full — return a length-only
            // confirmation so the operator can see "yes, a secret is
            // saved" without us leaking it back over the wire.
            let secret_mask = if ct.client_secret.is_empty() {
                String::new()
            } else {
                format!("****{} (length {})",
                    &ct.client_secret[ct.client_secret.len().saturating_sub(4)..],
                    ct.client_secret.len(),
                )
            };
            Json(serde_json::json!({
                "clientId": ct.client_id,
                "clientSecretMask": secret_mask,
                "clientSecretConfigured": !ct.client_secret.is_empty(),
                "redirectUri": ct.redirect_uri,
                "environment": ct.environment.as_str(),
                "accountId": ct.accounts.first().map(|a| a.account_id.clone()).unwrap_or_default(),
            }))
            .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("settings load panicked: {err}")})),
        )
            .into_response(),
    }
}

pub async fn credentials_post(
    State(_state): State<AppApiState>,
    Json(body): Json<CredentialsDto>,
) -> Response {
    // Trim everything before validation so a stray trailing newline
    // from a paste doesn't fail an empty-check.
    let client_id = body.client_id.trim().to_string();
    let client_secret = body.client_secret.trim().to_string();
    let redirect_uri = body.redirect_uri.trim().to_string();
    let environment_raw = body.environment.trim();
    let account_id = body.account_id.trim().to_string();

    if client_id.is_empty() || client_secret.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "clientId and clientSecret are required",
            })),
        )
            .into_response();
    }

    let environment = match environment_raw.to_ascii_lowercase().as_str() {
        "live" => CTraderBrokerEnvironment::Live,
        _ => CTraderBrokerEnvironment::Demo, // default = safer
    };

    // Default redirect URI matches the loopback OAuth flow the rest of
    // the codebase expects (port 43001). Operators rarely need to
    // change this, but the field exists for white-label setups.
    let redirect_uri = if redirect_uri.is_empty() {
        "http://127.0.0.1:43001/callback".to_string()
    } else {
        redirect_uri
    };

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        // Preserve any existing DxTrade config on disk — we're only
        // updating the cTrader section here.
        let mut current = load_broker_settings();
        current.schema_version = BROKER_CREDENTIALS_SCHEMA_VERSION;
        current.ctrader = CTraderBrokerSettings {
            client_id,
            client_secret,
            redirect_uri,
            authorization_code_input: String::new(),
            environment,
            accounts: if account_id.is_empty() {
                Vec::new()
            } else {
                vec![BrokerAccountTarget {
                    account_id: account_id.clone(),
                    label: account_id.clone(),
                    enabled_for_execution: true,
                }]
            },
        };
        save_broker_settings(&current)?;
        Ok(())
    })
    .await;

    match result {
        Ok(Ok(())) => Json(serde_json::json!({
            "ok": true,
            "message": "Credentials saved. Open Broker Setup → Re-authenticate to fetch a fresh token.",
        }))
        .into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("save task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}

#[allow(dead_code)] // used by tests that pin the wire shape
fn _unused() -> DxTradeBrokerSettings { DxTradeBrokerSettings::default() }
#[allow(dead_code)]
fn _unused2() -> BrokerSettingsState { BrokerSettingsState::default() }

pub async fn reauth(State(_state): State<AppApiState>) -> Response {
    // run_reauth_flow_blocking() does sync filesystem + reqwest::blocking
    // + std::net listener I/O. We MUST hop to spawn_blocking — calling
    // it directly on the tokio runtime would either panic on drop
    // ("Cannot drop a runtime in a context where blocking is not
    // allowed") or block the reactor for the full duration of the OAuth
    // flow, stalling every other route.
    match tokio::task::spawn_blocking(run_reauth_flow_blocking).await {
        Ok(Ok(outcome)) => Json(outcome).into_response(),
        Ok(Err(err)) => {
            tracing::warn!(
                target: "neoethos_app::server::broker_control",
                error = %err,
                "POST /broker/reauth: OAuth flow failed"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": err.to_string(),
                })),
            )
                .into_response()
        }
        Err(join_err) => {
            tracing::error!(
                target: "neoethos_app::server::broker_control",
                error = %join_err,
                "POST /broker/reauth: blocking task panicked"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("reauth task panicked: {join_err}"),
                })),
            )
                .into_response()
        }
    }
}

