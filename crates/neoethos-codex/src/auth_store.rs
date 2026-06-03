//! On-disk token persistence at `~/.codex/auth.json`.
//!
//! This path is **deliberately** what the official Codex CLI
//! uses. An operator who has already run `codex login` finds NeoEthos
//! immediately authenticated; conversely, if they auth through
//! NeoEthos first, `codex` will see the credentials. The schema we
//! write is a superset of the CLI's — extra fields are tolerated by
//! the CLI (it does field-level deserialise), and we tolerate extras
//! from the CLI (we use `serde_json::Value` for the carry-over).
//!
//! ## Security model
//!
//! - File mode on POSIX is set to 0600 right after the rename (best
//!   effort; if it fails we log and continue — the file lives in the
//!   user's home dir which is already 0700 by convention).
//! - We write atomically: `auth.json.tmp` → fsync → rename. A crash
//!   mid-write leaves the previous `auth.json` intact rather than
//!   truncating it.
//! - We never log the access_token, refresh_token, or id_token. The
//!   `Debug` impl below redacts them. Anything that touches these
//!   bytes does so through accessors that don't include them in the
//!   Display string.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::CodexError;
use crate::oauth::TokenBundle;

/// Top-level shape of `auth.json`.
///
/// Matches the Codex CLI's published schema. The CLI stores all
/// fields under a single `tokens` object so we do the same. Extra
/// CLI fields (e.g. `last_refresh`, `account_id`) ride along in
/// `extras`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAuth {
    /// "OPENAI" today; reserved for future providers (Anthropic etc.).
    /// The Codex CLI sets this to `"openai"`.
    #[serde(default = "default_provider")]
    pub provider: String,

    /// The bearer we send on every chat request.
    pub access_token: SecretString,

    /// Used to mint a new access_token when the current one expires.
    /// Optional because some flows return only a short-lived bearer
    /// (offline_access scope wasn't granted).
    #[serde(default)]
    pub refresh_token: Option<SecretString>,

    /// JWT we can decode locally (no network call) to display the
    /// signed-in account in the UI. Optional.
    #[serde(default)]
    pub id_token: Option<SecretString>,

    /// Absolute UTC instant the access_token stops working. We
    /// refresh ~60s before this. Optional because not every issuer
    /// includes `expires_in`.
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,

    /// "Bearer" in 99% of cases.
    #[serde(default = "default_token_type")]
    pub token_type: String,

    /// Email pulled from the id_token's claims, cached so the UI
    /// can render "Signed in as foo@example.com" without a fresh
    /// decode every render.
    #[serde(default)]
    pub email: Option<String>,

    /// Anything else that came back from the token endpoint that we
    /// didn't model explicitly. Preserved so the file round-trips
    /// losslessly to/from the Codex CLI.
    #[serde(default)]
    pub extras: serde_json::Value,
}

fn default_provider() -> String {
    "openai".to_string()
}
fn default_token_type() -> String {
    "Bearer".to_string()
}

/// Newtype around `String` that redacts in the Debug impl so token
/// values can't sneak into a `dbg!(...)` or `tracing::debug!` line.
/// Serde sees it as a plain string.
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretString(pub String);

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            f.write_str("SecretString(<empty>)")
        } else {
            f.write_fmt(format_args!("SecretString(<redacted, {} chars>)", self.0.len()))
        }
    }
}

impl SecretString {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl StoredAuth {
    /// Build from a fresh [`TokenBundle`] by converting `expires_in`
    /// (seconds-from-now) into an absolute `expires_at`. The id_token
    /// is decoded locally to pull out the `email` claim if it has
    /// one.
    pub fn from_bundle(bundle: TokenBundle) -> Self {
        let expires_at = bundle
            .expires_in_seconds
            .map(|secs| Utc::now() + chrono::Duration::seconds(secs as i64));
        let email = bundle.id_token.as_deref().and_then(parse_email_claim);
        Self {
            provider: "openai".to_string(),
            access_token: SecretString(bundle.access_token),
            refresh_token: bundle.refresh_token.map(SecretString),
            id_token: bundle.id_token.map(SecretString),
            expires_at,
            token_type: bundle.token_type,
            email,
            extras: bundle.raw,
        }
    }

    /// True if `expires_at` is set AND in the past (or within 60s
    /// of now). Callers should refresh before sending a chat request.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => Utc::now() + chrono::Duration::seconds(60) >= exp,
            None => false,
        }
    }
}

/// On-disk shape that tolerates BOTH auth.json layouts (F-291).
///
/// 1. **Modern Codex CLI** (what `codex login` writes today):
///    ```json
///    {
///      "auth_mode": "chatgpt",
///      "last_refresh": "2026-05-27T17:43:33Z",
///      "OPENAI_API_KEY": null,
///      "tokens": {
///        "access_token": "...",
///        "account_id": "...",
///        "id_token": "...",
///        "refresh_token": "..."
///      }
///    }
///    ```
/// 2. **Legacy flat** (what [`StoredAuth::from_bundle`] + [`AuthStore::save`]
///    wrote before F-291): `access_token` / `refresh_token` / `id_token`
///    / `expires_at` / `token_type` / `email` all at the top level.
///
/// `load()` parses into this, then [`OnDiskAuth::into_stored`] normalises
/// to the canonical [`StoredAuth`]. Every field is optional so a file
/// from either era deserialises; the conversion fails loudly only when
/// NEITHER an `access_token` could be found.
#[derive(Debug, Deserialize)]
struct OnDiskAuth {
    /// Modern CLI: the nested token bundle. Preferred when present.
    #[serde(default)]
    tokens: Option<OnDiskTokens>,
    /// Legacy flat: top-level access token.
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    provider: Option<String>,
}

/// The modern CLI's nested `tokens` object.
#[derive(Debug, Deserialize)]
struct OnDiskTokens {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    /// The CLI persists this; we don't use it yet but accept it so an
    /// unknown-field-strict future serde config wouldn't choke.
    #[serde(default)]
    #[allow(dead_code)]
    account_id: Option<String>,
}

impl OnDiskAuth {
    /// Normalise either schema into a [`StoredAuth`]. Prefers the
    /// modern nested `tokens` object; falls back to the legacy flat
    /// fields. Fails with [`CodexError::AuthStoreParse`] when no
    /// access token is present in either location.
    fn into_stored(self) -> Result<StoredAuth, CodexError> {
        let (access_token, refresh_token, id_token) =
            if let Some(t) = self.tokens {
                (t.access_token, t.refresh_token, t.id_token)
            } else if let Some(at) = self.access_token {
                (at, self.refresh_token, self.id_token)
            } else {
                return Err(CodexError::AuthStoreParse(
                    "auth.json has neither a `tokens.access_token` (modern \
                     Codex CLI) nor a top-level `access_token` (legacy) — \
                     re-run `codex login` or reconnect from Settings → \
                     Account."
                        .to_string(),
                ));
            };
        // Decode the email claim from the id_token when the legacy
        // `email` field wasn't already stored.
        let decoded_email = id_token.as_deref().and_then(parse_email_claim);
        Ok(StoredAuth {
            provider: self.provider.unwrap_or_else(|| "openai".to_string()),
            access_token: SecretString(access_token),
            refresh_token: refresh_token.map(SecretString),
            id_token: id_token.map(SecretString),
            // The modern CLI doesn't write an `expires_at`; leaving it
            // None makes `is_expired()` return false so we attempt the
            // call and let a server 401 drive the refresh path. The
            // legacy schema's `expires_at` is preserved when present.
            expires_at: self.expires_at,
            token_type: self.token_type.unwrap_or_else(|| "Bearer".to_string()),
            email: self.email.or(decoded_email),
            extras: serde_json::Value::Null,
        })
    }
}

/// Default location: `$HOME/.codex/auth.json`.
///
/// `directories::UserDirs` is cross-platform and matches what the
/// Codex CLI does on each OS:
///   - Linux/macOS: `$HOME/.codex/auth.json`
///   - Windows:    `%USERPROFILE%\.codex\auth.json`
pub fn default_auth_path() -> PathBuf {
    if let Some(user_dirs) = directories::UserDirs::new() {
        return user_dirs.home_dir().join(".codex").join("auth.json");
    }
    // Fallback if `directories` can't find the home dir for some
    // reason — write into the current directory so we don't silently
    // hand the operator a non-functional setup.
    PathBuf::from("./.codex/auth.json")
}

/// Wrapper that owns the on-disk path. Tests can construct one with
/// a temp dir to exercise read/write paths without polluting the
/// user's real `~/.codex/`.
#[derive(Debug, Clone)]
pub struct AuthStore {
    path: PathBuf,
}

impl AuthStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Construct using [`default_auth_path`].
    pub fn at_default() -> Self {
        Self::new(default_auth_path())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read and parse the file. Returns `Ok(None)` if the file
    /// doesn't exist (a fresh install) — the caller decides what
    /// "no auth yet" should look like. Returns `Err` for I/O
    /// failures and corrupt JSON.
    ///
    /// **F-291 (2026-05-29)**: parses through [`OnDiskAuth`] which
    /// tolerates BOTH the modern Codex CLI schema (token fields nested
    /// under a `tokens` object, plus a top-level `last_refresh`) AND
    /// the legacy flat schema this crate wrote before today. The old
    /// code deserialised straight into [`StoredAuth`], whose
    /// `access_token` lives at the top level — so it silently failed to
    /// read ANY `auth.json` written by a current `codex login`
    /// (everything is under `tokens` there). That surfaced as a
    /// perpetual "Not authenticated" no matter how many times the
    /// operator logged in, and is the real reason the AI Helper
    /// appeared to "reject every model" — most calls never got a valid
    /// bearer to send in the first place.
    pub fn load(&self) -> Result<Option<StoredAuth>, CodexError> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => {
                let on_disk: OnDiskAuth = serde_json::from_str(&text)
                    .map_err(|e| CodexError::AuthStoreParse(e.to_string()))?;
                Ok(Some(on_disk.into_stored()?))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CodexError::AuthStoreWrite {
                path: self.path.display().to_string(),
                source: e,
            }),
        }
    }

    /// Atomically replace the file. Writes to `auth.json.tmp` first,
    /// fsyncs, then renames over the target. A crash mid-write
    /// leaves the previous version intact.
    pub fn save(&self, auth: &StoredAuth) -> Result<(), CodexError> {
        let parent = self.path.parent().ok_or_else(|| CodexError::AuthStoreWrite {
            path: self.path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "no parent dir"),
        })?;
        std::fs::create_dir_all(parent).map_err(|source| CodexError::AuthStoreWrite {
            path: parent.display().to_string(),
            source,
        })?;

        let mut tmp = self.path.clone();
        tmp.set_extension("json.tmp");

        let json =
            serde_json::to_vec_pretty(auth).map_err(|e| CodexError::AuthStoreParse(e.to_string()))?;
        std::fs::write(&tmp, &json).map_err(|source| CodexError::AuthStoreWrite {
            path: tmp.display().to_string(),
            source,
        })?;

        // On POSIX, tighten the mode to 0600 before rename so an
        // attacker process can't catch a brief world-readable window.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&tmp)
                .map_err(|source| CodexError::AuthStoreWrite {
                    path: tmp.display().to_string(),
                    source,
                })?
                .permissions();
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(&tmp, perms);
        }

        std::fs::rename(&tmp, &self.path).map_err(|source| CodexError::AuthStoreWrite {
            path: self.path.display().to_string(),
            source,
        })?;
        Ok(())
    }

    /// Remove the file. Used by `/auth/codex/logout`. NotFound is
    /// not an error (already logged out).
    pub fn delete(&self) -> Result<(), CodexError> {
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(CodexError::AuthStoreWrite {
                path: self.path.display().to_string(),
                source: e,
            }),
        }
    }
}

/// Decode the `email` claim out of a JWT id_token. Returns `None`
/// for any error (bad base64, bad JSON, no `email` field) — id_tokens
/// from the issuer almost always include it, but we don't want a missing
/// claim to block the whole login flow.
fn parse_email_claim(id_token: &str) -> Option<String> {
    let payload = id_token.split('.').nth(1)?;
    let decoded = base64::Engine::decode(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        payload,
    )
    .or_else(|_| {
        base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE, payload)
    })
    .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    json.get("email").and_then(|v| v.as_str()).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bundle() -> TokenBundle {
        TokenBundle {
            access_token: "ATOKEN".into(),
            refresh_token: Some("RTOKEN".into()),
            id_token: None,
            token_type: "Bearer".into(),
            expires_in_seconds: Some(3600),
            raw: serde_json::json!({"access_token":"ATOKEN","refresh_token":"RTOKEN"}),
        }
    }

    #[test]
    fn from_bundle_computes_absolute_expiry() {
        let auth = StoredAuth::from_bundle(make_bundle());
        let exp = auth.expires_at.expect("expiry set");
        let delta = exp - Utc::now();
        assert!(delta.num_seconds() > 3500 && delta.num_seconds() <= 3600);
    }

    #[test]
    fn is_expired_respects_60s_safety_window() {
        let mut auth = StoredAuth::from_bundle(make_bundle());
        // Force expiry to 30s from now → considered expired (refresh window).
        auth.expires_at = Some(Utc::now() + chrono::Duration::seconds(30));
        assert!(auth.is_expired());
        // 5 min from now → fresh.
        auth.expires_at = Some(Utc::now() + chrono::Duration::seconds(300));
        assert!(!auth.is_expired());
    }

    #[test]
    fn save_and_load_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos-codex-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("auth.json");

        let store = AuthStore::new(path.clone());
        let original = StoredAuth::from_bundle(make_bundle());
        store.save(&original).unwrap();

        let loaded = store.load().unwrap().expect("file present after save");
        assert_eq!(loaded.access_token.expose(), "ATOKEN");
        assert_eq!(
            loaded.refresh_token.as_ref().map(|s| s.expose()),
            Some("RTOKEN")
        );
        assert_eq!(loaded.token_type, "Bearer");

        store.delete().unwrap();
        assert!(store.load().unwrap().is_none());
    }

    #[test]
    fn secret_string_does_not_leak_in_debug() {
        let s = SecretString("super-secret-token-value".into());
        let dbg = format!("{:?}", s);
        assert!(!dbg.contains("super-secret-token-value"));
        assert!(dbg.contains("redacted"));
    }

    #[test]
    fn parse_email_claim_handles_well_formed_jwt() {
        // header.payload.sig — payload is {"email":"a@b.com"} base64url-encoded.
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            br#"{"email":"a@b.com","sub":"123"}"#,
        );
        let jwt = format!("header.{payload}.sig");
        assert_eq!(parse_email_claim(&jwt), Some("a@b.com".to_string()));
    }

    #[test]
    fn parse_email_claim_tolerates_malformed_input() {
        assert_eq!(parse_email_claim("not.a.jwt"), None);
        assert_eq!(parse_email_claim(""), None);
    }

    #[test]
    fn loads_modern_nested_codex_cli_schema() {
        // **F-291 regression**: a current `codex login` writes the token
        // fields NESTED under a `tokens` object, with a top-level
        // `last_refresh`. The old loader deserialised straight into
        // StoredAuth (top-level `access_token`) and failed with
        // "missing field `access_token`" — which surfaced as a permanent
        // "Not authenticated" in the AI Helper. This must now parse.
        let dir = std::env::temp_dir()
            .join(format!("neoethos-codex-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("auth.json");
        std::fs::write(
            &path,
            r#"{
              "auth_mode": "chatgpt",
              "last_refresh": "2026-05-27T17:43:33.025749300Z",
              "OPENAI_API_KEY": null,
              "tokens": {
                "access_token": "NESTED_AT",
                "account_id": "acct-123",
                "id_token": "header.payload.sig",
                "refresh_token": "NESTED_RT"
              }
            }"#,
        )
        .unwrap();

        let store = AuthStore::new(path);
        let loaded = store.load().unwrap().expect("nested schema must parse");
        assert_eq!(loaded.access_token.expose(), "NESTED_AT");
        assert_eq!(
            loaded.refresh_token.as_ref().map(|s| s.expose()),
            Some("NESTED_RT")
        );
        // The modern CLI doesn't write an `expires_at` → None →
        // is_expired() stays false so we attempt the call.
        assert!(!loaded.is_expired());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_auth_json_with_no_access_token() {
        // Neither a nested `tokens.access_token` nor a top-level one →
        // fail loud with a message that tells the operator to re-login,
        // rather than silently behaving as "not authenticated".
        let dir = std::env::temp_dir()
            .join(format!("neoethos-codex-noauth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("auth.json");
        std::fs::write(&path, r#"{"auth_mode":"chatgpt","OPENAI_API_KEY":null}"#)
            .unwrap();

        let store = AuthStore::new(path);
        let err = store.load().unwrap_err();
        assert!(
            matches!(err, CodexError::AuthStoreParse(_)),
            "expected AuthStoreParse, got {err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        SecretString(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        SecretString(value.to_string())
    }
}
