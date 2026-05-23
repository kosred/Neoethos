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
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, CTRADER_OAUTH_REDIRECT_URI,
    CTraderBrokerEnvironment, CTraderBrokerSettings,
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
                format!(
                    "****{} (length {})",
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
    let client_secret_in = body.client_secret.trim().to_string();
    let redirect_uri = body.redirect_uri.trim().to_string();
    let environment_raw = body.environment.trim();
    let account_id = body.account_id.trim().to_string();

    // Empty-secret semantics: the Settings UI prompts "leave blank to
    // keep existing" — so an empty secret in the payload is NOT a
    // validation failure when the on-disk secret is already populated.
    // We merge: anything empty in the payload inherits the current
    // saved value. Only when BOTH input and saved are empty do we
    // 400. Same merge logic for client_id (UI may pre-fill it but the
    // operator could clear the field to swap apps; we still need the
    // *new* value, but if they leave it blank we keep the old one).
    let environment = match environment_raw.to_ascii_lowercase().as_str() {
        "live" => CTraderBrokerEnvironment::Live,
        _ => CTraderBrokerEnvironment::Demo, // default = safer
    };

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        // Preserve any existing DxTrade config + load the current
        // cTrader secret so we can keep it when the form left the
        // field blank.
        let mut current = load_broker_settings();
        let saved_client_id = current.ctrader.client_id.clone();
        let saved_client_secret = current.ctrader.client_secret.clone();

        let merged_client_id = if client_id.is_empty() {
            saved_client_id
        } else {
            client_id
        };
        let merged_client_secret = if client_secret_in.is_empty() {
            saved_client_secret
        } else {
            client_secret_in
        };

        if merged_client_id.is_empty() || merged_client_secret.is_empty() {
            anyhow::bail!(
                "clientId and clientSecret are required (no saved value to fall back on)"
            );
        }

        // Default redirect URI matches the loopback OAuth flow the rest
        // of the codebase expects (port 43001). Operators rarely need
        // to change this, but the field exists for white-label setups.
        let merged_redirect_uri = if redirect_uri.is_empty() {
            CTRADER_OAUTH_REDIRECT_URI.to_string()
        } else {
            redirect_uri
        };

        current.schema_version = BROKER_CREDENTIALS_SCHEMA_VERSION;
        current.ctrader = CTraderBrokerSettings {
            client_id: merged_client_id,
            client_secret: merged_client_secret,
            redirect_uri: merged_redirect_uri,
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
