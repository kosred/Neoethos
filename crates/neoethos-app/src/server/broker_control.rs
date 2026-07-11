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
use crate::app_services::broker_api::fetch_broker_accounts_blocking;
use crate::app_services::broker_persistence::{load_broker_settings, save_broker_settings};
use crate::app_services::reauth::run_reauth_flow_blocking;

use super::errors::{actionable_error, internal_panic};
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
        Err(join_err) => internal_panic("Loading saved credentials", join_err),
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
        // Load the current cTrader secret so we can keep it when the
        // form left the field blank.
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
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not save your cTrader credentials. Make sure the app data folder \
             (%APPDATA%\\neoethos) is writable, then try again.",
            &err,
        ),
        Err(join_err) => internal_panic("Saving credentials", join_err),
    }
}

// ─── POST /broker/account/select ──────────────────────────────────────────

/// Wire DTO for `POST /broker/account/select`. The operator picks an
/// account from the Settings dropdown (sourced from `/broker/accounts`)
/// and we make it the *active* one by promoting it to the front of the
/// on-disk `[[ctrader.accounts]]` list — `resolve_creds()` always takes
/// `accounts.first()`, so first == active.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSelectDto {
    /// Numeric cTID as a string — cTrader ids can exceed i32 range so
    /// the wire shape is text, mirroring `BrokerAccountDto::account_id`.
    pub account_id: String,
}

/// Outcome of the blocking select task — distinguishes the three
/// terminal states so the async wrapper can map each to the right
/// HTTP status without stringly-typed sniffing.
enum SelectOutcome {
    /// Account was already on disk and is now first (or already was).
    Promoted,
    /// Account wasn't on disk but is a valid OAuth-granted cTID, so we
    /// added it as the active (first) entry.
    AddedFromGrant,
    /// `account_id` matched neither the on-disk list nor the live OAuth
    /// grant — a genuinely-unknown id. Maps to 404.
    NotFound,
}

/// `POST /broker/account/select` — set the *active* cTrader account.
///
/// MVP behaviour (no runtime hot-swap yet): load broker_credentials.toml
/// and make the requested account **first** in the `[[ctrader.accounts]]`
/// list. `resolve_creds()` always reads `accounts.first()`, so first ==
/// active; the next NeoEthos start picks up the freshly-promoted account.
/// We return `requiresRestart: true` so the UI prompts accordingly.
///
/// Two cases, both non-destructive (existing accounts are preserved, never
/// cleared the way `credentials_post` does — clearing would force a
/// re-OAuth to recover the rest of the granted set):
///
///   1. The id is already in the on-disk list → **reorder** it to the
///      front.
///   2. The id is NOT on disk yet → it's almost always a different
///      account from the same OAuth grant (the Settings dropdown is fed
///      by `/broker/accounts`, which lists the *full* grant, while
///      `credentials_post` only ever persists one account). We validate
///      it against the live grant and, if present there, **prepend** it
///      as a fresh enabled target. This is the case that actually makes
///      multi-account selection work — without it, picking any account
///      other than the single persisted one would be a no-op.
///
/// Errors:
///   - 400 if `accountId` is blank.
///   - 404 if the id is in neither the on-disk list nor the live OAuth
///     grant (stale UI / typo / revoked access).
pub async fn account_select(
    State(_state): State<AppApiState>,
    Json(body): Json<AccountSelectDto>,
) -> Response {
    let account_id = body.account_id.trim().to_string();
    if account_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "accountId must be non-empty"})),
        )
            .into_response();
    }

    let select_id = account_id.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<SelectOutcome> {
        let mut current = load_broker_settings();
        let accounts = &mut current.ctrader.accounts;

        if let Some(pos) = accounts.iter().position(|a| a.account_id == select_id) {
            // Case 1 — already persisted. Promote to front. If it's
            // already first this is a no-op; skip the write so we don't
            // bump the file mtime (and trip the credentials-drift
            // healer) for nothing.
            if pos != 0 {
                let picked = accounts.remove(pos);
                accounts.insert(0, picked);
                current.schema_version = BROKER_CREDENTIALS_SCHEMA_VERSION;
                save_broker_settings(&current)?;
            }
            return Ok(SelectOutcome::Promoted);
        }

        // Case 2 — not on disk. Confirm it's a real account from the
        // live OAuth grant before we add it, so a typo'd / revoked id
        // becomes a clean 404 instead of silently pinning a bad cTID
        // (which is exactly the CH_ACCESS_TOKEN_INVALID footgun this
        // whole picker exists to eliminate).
        let grant = fetch_broker_accounts_blocking()?;
        let Some(granted) = grant.accounts.iter().find(|a| a.account_id == select_id) else {
            return Ok(SelectOutcome::NotFound);
        };

        // Prepend as the active target, preserving the granted label +
        // execution flag so the on-disk row matches what the user saw
        // in the dropdown.
        accounts.insert(
            0,
            BrokerAccountTarget {
                account_id: granted.account_id.clone(),
                label: if granted.account_name.is_empty() {
                    granted.account_id.clone()
                } else {
                    granted.account_name.clone()
                },
                enabled_for_execution: granted.enabled_for_execution,
            },
        );
        current.schema_version = BROKER_CREDENTIALS_SCHEMA_VERSION;
        save_broker_settings(&current)?;
        Ok(SelectOutcome::AddedFromGrant)
    })
    .await;

    match result {
        Ok(Ok(SelectOutcome::Promoted)) | Ok(Ok(SelectOutcome::AddedFromGrant)) => {
            Json(serde_json::json!({
                "ok": true,
                "selectedAccountId": account_id,
                "requiresRestart": true,
            }))
            .into_response()
        }
        Ok(Ok(SelectOutcome::NotFound)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "account '{account_id}' is not in broker_credentials.toml \
                     nor in the current cTrader OAuth grant; re-authenticate \
                     (Broker Setup → Re-authenticate) and pick it from the \
                     refreshed list"
                ),
            })),
        )
            .into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to save the selected account. Make sure broker_credentials.toml isn't \
             locked by another process, then try again.",
            &err,
        ),
        Err(join_err) => internal_panic("Selecting the account", join_err),
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
            actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Authentication failed. Make sure no other app is using the OAuth callback \
                 port, then try Broker Setup → Re-authenticate. If the consent page didn't \
                 open, check your default browser.",
                &err,
            )
        }
        Err(join_err) => {
            tracing::error!(
                target: "neoethos_app::server::broker_control",
                error = %join_err,
                "POST /broker/reauth: blocking task panicked"
            );
            internal_panic("Re-authentication", join_err)
        }
    }
}
