//! Persistence layer for [`BrokerSettingsState`].
//!
//! Loads and saves broker connection credentials (cTrader Open API client ID,
//! client secret, redirect URI, etc.) to a TOML file outside the repository,
//! so the application can pre-populate the Settings → Brokers UI on startup
//! instead of requiring the user to retype credentials every launch.
//!
//! # Lookup order
//!
//! Resolved by [`credentials_file_path`]:
//!
//! 1. The path in environment variable `FOREX_AI_BROKER_CREDENTIALS_PATH`,
//!    if set. Useful for tests and CI.
//! 2. `<config_dir>/forex-ai/broker_credentials.toml` where `<config_dir>` is
//!    `dirs::config_dir()` — `%APPDATA%` on Windows, `$XDG_CONFIG_HOME` (or
//!    `~/.config`) on Linux, `~/Library/Application Support` on macOS.
//! 3. `<cwd>/.local/forex-ai/broker_credentials.toml` as a fallback for
//!    development environments where the user hasn't set up `%APPDATA%`.
//!
//! # Security
//!
//! The TOML file is intended to live OUTSIDE the git repository. The
//! `.gitignore` should include both `**/broker_credentials.toml` and
//! `.local/` to prevent accidental commits.
//!
//! Two transient fields are explicitly NEVER serialized:
//!
//! - `CTraderBrokerSettings::authorization_code_input` — short-lived OAuth
//!   intermediate value
//! - `DxTradeBrokerSettings::password` — must be re-entered each session
//!
//! These are enforced at the type level via `#[serde(skip_serializing,
//! skip_deserializing, default)]` on the field declarations.

use crate::app_services::broker_config::BrokerSettingsState;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::{env, fs};

const APP_CONFIG_SUBDIR: &str = "forex-ai";
const CREDENTIALS_FILENAME: &str = "broker_credentials.toml";
const ENV_OVERRIDE_VAR: &str = "FOREX_AI_BROKER_CREDENTIALS_PATH";

/// Resolves the path to the broker credentials TOML file.
///
/// Order of resolution:
/// 1. `$FOREX_AI_BROKER_CREDENTIALS_PATH` if non-empty
/// 2. `<dirs::config_dir>/forex-ai/broker_credentials.toml`
/// 3. `<cwd>/.local/forex-ai/broker_credentials.toml`
///
/// Returns the first candidate that EXISTS. If none exists, returns the
/// preferred candidate (env override → config_dir → local) so the caller can
/// create it.
pub fn credentials_file_path() -> Result<PathBuf> {
    let candidates = candidate_paths()?;

    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    candidates
        .into_iter()
        .next()
        .context("no candidate path could be resolved for broker credentials")
}

fn candidate_paths() -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(3);

    if let Ok(custom) = env::var(ENV_OVERRIDE_VAR) {
        if !custom.trim().is_empty() {
            paths.push(PathBuf::from(custom));
        }
    }

    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join(APP_CONFIG_SUBDIR).join(CREDENTIALS_FILENAME));
    }

    if let Ok(cwd) = env::current_dir() {
        paths.push(
            cwd.join(".local")
                .join(APP_CONFIG_SUBDIR)
                .join(CREDENTIALS_FILENAME),
        );
    }

    if paths.is_empty() {
        anyhow::bail!("unable to determine broker credentials file path on this platform");
    }
    Ok(paths)
}

/// Loads broker settings from disk.
///
/// Returns [`BrokerSettingsState::default`] if no file exists at any of the
/// candidate paths, or if the file is malformed (after logging the parse
/// error). This function never panics — startup must succeed even with a
/// missing or corrupt credentials file.
pub fn load_broker_settings() -> BrokerSettingsState {
    let path = match credentials_file_path() {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!(error = %err, "broker credentials path resolution failed");
            return BrokerSettingsState::default();
        }
    };

    if !path.is_file() {
        tracing::debug!(
            path = %path.display(),
            "no broker credentials file found; using empty defaults"
        );
        return BrokerSettingsState::default();
    }

    match fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str::<BrokerSettingsState>(&contents) {
            Ok(settings) => {
                tracing::info!(
                    path = %path.display(),
                    "loaded broker credentials from disk"
                );
                settings
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to parse broker credentials TOML; using defaults"
                );
                BrokerSettingsState::default()
            }
        },
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "failed to read broker credentials file; using defaults"
            );
            BrokerSettingsState::default()
        }
    }
}

/// Persists broker settings to disk at the resolved credentials path.
///
/// Creates the parent directory if missing. Writes TOML in the standard
/// formatting. Transient fields (`authorization_code_input`, DxTrade
/// `password`) are excluded by their serde annotations.
pub fn save_broker_settings(settings: &BrokerSettingsState) -> Result<()> {
    let path = credentials_file_path()?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create directory for broker credentials at {}",
                parent.display()
            )
        })?;
    }

    let serialized = toml::to_string_pretty(settings)
        .context("failed to serialize broker credentials to TOML")?;

    fs::write(&path, serialized)
        .with_context(|| format!("failed to write broker credentials to {}", path.display()))?;

    tracing::info!(path = %path.display(), "saved broker credentials to disk");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::broker_config::{
        BrokerAccountTarget, CTraderBrokerEnvironment, CTraderBrokerSettings,
        DxTradeBrokerSettings,
    };
    use std::sync::Mutex;

    /// `env::set_var`/`env::var` are process-global. Serialize the env-mutating
    /// tests so they don't race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_path<F: FnOnce(&std::path::Path)>(path: &std::path::Path, body: F) {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        // SAFETY: the lock above ensures no concurrent env access from
        // these tests; cargo test still parallelizes outer tests but the
        // env-touching ones share the lock.
        unsafe {
            env::set_var(ENV_OVERRIDE_VAR, path);
        }
        body(path);
        unsafe {
            env::remove_var(ENV_OVERRIDE_VAR);
        }
    }

    fn populated_settings() -> BrokerSettingsState {
        BrokerSettingsState {
            ctrader: CTraderBrokerSettings {
                client_id: "client-123".to_string(),
                client_secret: "secret-abc".to_string(),
                redirect_uri: "http://127.0.0.1:43001/callback".to_string(),
                authorization_code_input: "should-not-persist".to_string(),
                environment: CTraderBrokerEnvironment::Demo,
                accounts: vec![BrokerAccountTarget {
                    account_id: "ctr-001".to_string(),
                    label: "Primary".to_string(),
                    enabled_for_execution: true,
                }],
            },
            dxtrade: DxTradeBrokerSettings {
                platform_url: "https://demo.dx.example".to_string(),
                username: "user42".to_string(),
                password: "should-not-persist-either".to_string(),
                accounts: vec![],
            },
        }
    }

    #[test]
    fn load_returns_default_when_file_missing() {
        let dir = tempdir_or_skip();
        let path = dir.join("does-not-exist.toml");
        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            assert_eq!(loaded, BrokerSettingsState::default());
        });
    }

    #[test]
    fn save_then_load_roundtrip_preserves_ctrader_credentials() {
        let dir = tempdir_or_skip();
        let path = dir.join("creds.toml");

        with_env_path(&path, |_| {
            let original = populated_settings();
            save_broker_settings(&original).expect("save should succeed");

            let loaded = load_broker_settings();
            assert_eq!(loaded.ctrader.client_id, "client-123");
            assert_eq!(loaded.ctrader.client_secret, "secret-abc");
            assert_eq!(
                loaded.ctrader.redirect_uri,
                "http://127.0.0.1:43001/callback"
            );
            assert_eq!(loaded.ctrader.environment, CTraderBrokerEnvironment::Demo);
            assert_eq!(loaded.ctrader.accounts.len(), 1);
            assert_eq!(loaded.ctrader.accounts[0].account_id, "ctr-001");
            assert!(loaded.ctrader.accounts[0].enabled_for_execution);
        });
    }

    #[test]
    fn dxtrade_password_is_not_persisted() {
        let dir = tempdir_or_skip();
        let path = dir.join("creds.toml");

        with_env_path(&path, |_| {
            let original = populated_settings();
            save_broker_settings(&original).expect("save should succeed");

            // Read the raw file to confirm the password literal is absent.
            let raw = fs::read_to_string(&path).expect("read");
            assert!(
                !raw.contains("should-not-persist-either"),
                "DxTrade password leaked into TOML:\n{raw}"
            );

            // After load, the password field is reset to the field default.
            let loaded = load_broker_settings();
            assert_eq!(loaded.dxtrade.password, "");
            assert_eq!(loaded.dxtrade.username, "user42");
        });
    }

    #[test]
    fn ctrader_authorization_code_input_is_not_persisted() {
        let dir = tempdir_or_skip();
        let path = dir.join("creds.toml");

        with_env_path(&path, |_| {
            let original = populated_settings();
            save_broker_settings(&original).expect("save should succeed");

            let raw = fs::read_to_string(&path).expect("read");
            assert!(
                !raw.contains("should-not-persist"),
                "authorization_code_input leaked into TOML:\n{raw}"
            );

            let loaded = load_broker_settings();
            assert_eq!(loaded.ctrader.authorization_code_input, "");
        });
    }

    #[test]
    fn malformed_toml_falls_back_to_default() {
        let dir = tempdir_or_skip();
        let path = dir.join("malformed.toml");
        fs::write(&path, "not = valid \n[unclosed").expect("write");

        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            assert_eq!(loaded, BrokerSettingsState::default());
        });
    }

    /// Emulate `tempfile::tempdir` without adding the dependency: use the
    /// system temp + a per-test PID/nanos suffix. Skip the test gracefully if
    /// the directory cannot be created (e.g. tightly sandboxed CI).
    fn tempdir_or_skip() -> PathBuf {
        use std::time::SystemTime;

        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let path = env::temp_dir().join(format!("forex-ai-bp-{pid}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir should be creatable");
        path
    }
}
