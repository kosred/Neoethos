//! Canonical on-disk shape for `broker_credentials.toml`.
//!
//! This module owns the schema and the filesystem I/O that BOTH
//! `neoethos-app` (HTTP `/broker/credentials` endpoint) and
//! `neoethos-cli` (`credentials set` subcommand) share. Living in
//! `neoethos-core` rather than `neoethos-app` breaks the workspace
//! cycle that previously blocked the CLI from writing credentials
//! without depending on the GUI binary.
//!
//! ## What lives here
//!
//! - The pure data structs (`BrokerSettingsState`, `CTraderBrokerSettings`,
//!   `BrokerAccountTarget`, `CTraderBrokerEnvironment`).
//! - Constants (schema version, Spotware sign-up URLs).
//! - Path resolution (`credentials_file_path`) — env override →
//!   platform config dir → cwd `.local/` fallback.
//! - Read/write helpers (`load_from_disk`, `save_to_disk`) that DO
//!   NOT include the compile-time embedded-credentials overlay.
//!
//! ## What does NOT live here
//!
//! - `apply_embedded_fallback` — that pulls from
//!   `neoethos-app::app_services::embedded_credentials` which is
//!   binary-specific (the `build.rs` stamps constants per build).
//!   Kept in `neoethos-app`.
//! - Runtime connection/readiness state. Consumers derive that from current
//!   credentials and broker probes; it is not part of this on-disk schema.
//!
//! ## Security
//!
//! One transient field is NEVER serialized:
//! - `CTraderBrokerSettings::authorization_code_input` — short-lived OAuth value

use crate::schema_version::{HasSchemaVersion, SchemaVersion, default_v1};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Current schema version of the `broker_credentials.toml` on-disk
/// contract. Bump when fields are renamed/removed or their types
/// change in a way that `#[serde(default)]` can't bridge.
pub const BROKER_CREDENTIALS_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1);

/// Spotware's canonical cTrader demo-account sign-up page. White-label
/// brokers running cTrader under their own domain can patch this
/// single constant instead of editing the UI.
pub const CTRADER_CREATE_DEMO_ACCOUNT_URL: &str = "https://app.ctrader.com/accounts/create-demo";

/// Spotware's canonical cTrader live-account sign-up page.
pub const CTRADER_CREATE_LIVE_ACCOUNT_URL: &str = "https://app.ctrader.com/accounts/create-live";

/// OAuth redirect URI the loopback listener binds to during the
/// cTrader consent flow. Hardcoded in 5+ sites before #150 —
/// promoted here so a future port change doesn't drift between
/// the listener, the CLI default, and the embedded fallback
/// state. Format: scheme + host + port + path. Spotware echoes
/// this string back in the consent URL, so any change here MUST
/// be mirrored in the cTrader application portal's registered
/// redirect-uri list — otherwise the post-consent redirect
/// fails with `unauthorized_client`.
pub const CTRADER_OAUTH_REDIRECT_URI: &str = "http://127.0.0.1:43001/callback";

const APP_CONFIG_SUBDIR: &str = "neoethos";
const CREDENTIALS_FILENAME: &str = "broker_credentials.toml";
// The registry still owns the environment-variable name. Credentials use a
// strict locator: a present but invalid override is NOT the default profile.

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BrokerAccountTarget {
    pub account_id: String,
    pub label: String,
    #[serde(default)]
    pub enabled_for_execution: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CTraderBrokerEnvironment {
    #[default]
    Live,
    Demo,
}

impl CTraderBrokerEnvironment {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Demo => "Demo",
        }
    }

    /// Parse "Live"/"Demo" (case-insensitive). Returns `None` for
    /// anything else so callers can decide whether to default or
    /// reject.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "live" => Some(Self::Live),
            "demo" => Some(Self::Demo),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CTraderBrokerSettings {
    #[serde(default)]
    pub client_id: String,
    #[serde(default)]
    pub client_secret: String,
    #[serde(default)]
    pub redirect_uri: String,
    /// Transient input. NEVER persisted to disk for security.
    #[serde(skip_serializing, skip_deserializing, default)]
    pub authorization_code_input: String,
    #[serde(default)]
    pub environment: CTraderBrokerEnvironment,
    #[serde(default)]
    pub accounts: Vec<BrokerAccountTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrokerSettingsState {
    /// Schema version of the on-disk `broker_credentials.toml`
    /// contract. Defaults to v1 (the pre-versioning shape) when
    /// missing, so files written by older builds load without
    /// breaking. See [`BROKER_CREDENTIALS_SCHEMA_VERSION`].
    #[serde(default = "default_v1")]
    pub schema_version: SchemaVersion,
    #[serde(default)]
    pub ctrader: CTraderBrokerSettings,
}

impl Default for BrokerSettingsState {
    fn default() -> Self {
        Self {
            schema_version: BROKER_CREDENTIALS_SCHEMA_VERSION,
            ctrader: CTraderBrokerSettings::default(),
        }
    }
}

impl HasSchemaVersion for BrokerSettingsState {
    const CURRENT: SchemaVersion = BROKER_CREDENTIALS_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

/// Resolves the path to the broker credentials TOML file.
///
/// Order of resolution:
/// 1. `$NEOETHOS_BROKER_CREDENTIALS_PATH` when present (env override
///    is AUTHORITATIVE — bypasses the existing-file lookup so tests
///    target an isolated temp path without accidentally falling
///    through to the operator's real `~/AppData/Roaming/neoethos/
///    broker_credentials.toml`). Empty or invalid explicit values return an
///    error before default-path enumeration.
/// 2. `<dirs::config_dir>/neoethos/broker_credentials.toml`.
/// 3. `<cwd>/.local/neoethos/broker_credentials.toml`.
///
/// Returns the first candidate that EXISTS. If none exists, returns
/// the highest-priority candidate so callers can create it there.
pub fn credentials_file_path() -> Result<PathBuf> {
    if let Some(custom) = credentials_profile_path()? {
        return Ok(custom);
    }

    let candidates = candidate_credentials_paths()?;

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

/// Strict explicit profile shared by filesystem credentials and OAuth tokens.
/// Absent selects the normal profile; invalid never falls through to it.
pub fn credentials_profile_path() -> Result<Option<PathBuf>> {
    let raw = env::var_os(crate::env_overrides::ENV_BROKER_CREDENTIALS_PATH);
    resolve_credentials_profile(raw.as_deref())
}

fn resolve_credentials_profile(raw: Option<&std::ffi::OsStr>) -> Result<Option<PathBuf>> {
    let Some(raw) = raw else { return Ok(None) };
    let path = raw
        .to_str()
        .context("broker credentials profile path is not Unicode")?
        .trim();
    anyhow::ensure!(
        !path.is_empty() && !path.contains('\0'),
        "broker credentials profile path is empty or contains NUL"
    );
    // Never canonicalize: missing files are valid bootstrap destinations and
    // creating one must not change its OAuth namespace. Alternate spellings may
    // select separate profiles, but can never select the default token entry.
    let path =
        std::path::absolute(path).context("cannot resolve broker credentials profile path")?;
    anyhow::ensure!(
        path.file_name().is_some(),
        "broker credentials profile must select a file"
    );
    anyhow::ensure!(
        path.to_str().is_some(),
        "resolved broker credentials profile path is not Unicode"
    );
    Ok(Some(path))
}

/// Every candidate path that `credentials_file_path` would try, in
/// priority order (canonical first). Exposed (vs the previous
/// private helper) so the neoethos-app layer can detect when more
/// than one of these is populated and migrate the stale copies —
/// otherwise the user re-auth'd from a different CWD and now has
/// two files with different `account_id`s drifting against each
/// other (#141).
pub fn candidate_credentials_paths() -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(2);

    if let Some(config_dir) = dirs::config_dir() {
        paths.push(
            config_dir
                .join(APP_CONFIG_SUBDIR)
                .join(CREDENTIALS_FILENAME),
        );
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

/// Read the TOML at `path` and parse it. Returns the parsed state on
/// success. The caller decides how to handle errors (defaults vs.
/// loud-fail) — this layer just does the bytes-to-struct part.
///
/// Returns `Ok(None)` if the file does not exist. That distinction
/// matters: a missing file is a "use defaults" signal, whereas an
/// unreadable / malformed file is a real error.
pub fn load_from_disk(path: &Path) -> Result<Option<BrokerSettingsState>> {
    if !path.is_file() {
        return Ok(None);
    }
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let parsed: BrokerSettingsState = toml::from_str(&contents)
        .with_context(|| format!("failed to parse TOML at {}", path.display()))?;
    crate::schema_version::check_schema_version_readable(&parsed, "broker_credentials.toml")
        .with_context(|| format!("schema-version mismatch in {}", path.display()))?;
    Ok(Some(parsed))
}

/// Persist `settings` to `path`. Creates the parent directory if it
/// doesn't exist. Always stamps the current schema version on the
/// outgoing file, regardless of what the in-memory struct carries —
/// this protects against code paths that constructed the struct
/// manually and forgot to set `schema_version`.
pub fn save_to_disk(path: &Path, settings: &BrokerSettingsState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create directory for broker credentials at {}",
                parent.display()
            )
        })?;
    }

    let mut to_write = settings.clone();
    to_write.schema_version = BROKER_CREDENTIALS_SCHEMA_VERSION;
    let serialized = toml::to_string_pretty(&to_write)
        .context("failed to serialize broker credentials to TOML")?;
    fs::write(path, serialized)
        .with_context(|| format!("failed to write broker credentials to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Aliased, never re-spelled: see the note beside `CREDENTIALS_FILENAME`.
    const ENV_OVERRIDE_VAR: &str = crate::env_overrides::ENV_BROKER_CREDENTIALS_PATH;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvOverrideGuard;
    impl Drop for EnvOverrideGuard {
        fn drop(&mut self) {
            unsafe {
                env::remove_var(ENV_OVERRIDE_VAR);
            }
        }
    }

    fn with_env_path<F: FnOnce(&Path)>(path: &Path, body: F) {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            env::set_var(ENV_OVERRIDE_VAR, path);
        }
        let _env_guard = EnvOverrideGuard;
        body(path);
    }

    fn tempdir_or_skip() -> PathBuf {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let path = env::temp_dir().join(format!("neoethos-core-bp-{pid}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir creatable");
        path
    }

    #[test]
    fn load_returns_none_when_file_missing() {
        let dir = tempdir_or_skip();
        let path = dir.join("does-not-exist.toml");
        let r = load_from_disk(&path).expect("missing should not error");
        assert!(r.is_none());
    }

    #[test]
    fn save_then_load_roundtrip_preserves_fields() {
        let dir = tempdir_or_skip();
        let path = dir.join("creds.toml");
        let original = BrokerSettingsState {
            schema_version: BROKER_CREDENTIALS_SCHEMA_VERSION,
            ctrader: CTraderBrokerSettings {
                client_id: "abc".to_string(),
                client_secret: "xyz".to_string(),
                redirect_uri: "http://127.0.0.1:43001/callback".to_string(),
                environment: CTraderBrokerEnvironment::Demo,
                accounts: vec![BrokerAccountTarget {
                    account_id: "ct-1".to_string(),
                    label: "primary".to_string(),
                    enabled_for_execution: true,
                }],
                ..Default::default()
            },
        };
        save_to_disk(&path, &original).expect("save");
        let loaded = load_from_disk(&path).expect("load").expect("some");
        assert_eq!(loaded.ctrader.client_id, "abc");
        assert_eq!(loaded.ctrader.environment, CTraderBrokerEnvironment::Demo);
        assert_eq!(loaded.ctrader.accounts.len(), 1);
        assert_eq!(loaded.schema_version, BROKER_CREDENTIALS_SCHEMA_VERSION);
    }

    #[test]
    fn transient_fields_are_not_persisted() {
        let dir = tempdir_or_skip();
        let path = dir.join("transient.toml");
        let original = BrokerSettingsState {
            ctrader: CTraderBrokerSettings {
                client_id: "abc".to_string(),
                client_secret: "secret".to_string(),
                authorization_code_input: "DO-NOT-PERSIST".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        save_to_disk(&path, &original).expect("save");
        let raw = fs::read_to_string(&path).expect("read");
        assert!(!raw.contains("DO-NOT-PERSIST"));
    }

    #[test]
    fn env_override_wins_over_default_lookup() {
        let dir = tempdir_or_skip();
        let path = dir.join("override.toml");
        with_env_path(&path, |p| {
            let resolved = credentials_file_path().expect("resolve");
            assert_eq!(resolved, p);
        });
    }

    #[test]
    fn invalid_explicit_credentials_profile_never_falls_through_to_defaults() {
        for raw in ["", " \n ", "bad\0profile"] {
            assert!(resolve_credentials_profile(Some(std::ffi::OsStr::new(raw))).is_err());
        }
        with_env_path(Path::new("  \n "), |_| {
            assert!(credentials_file_path().is_err());
        });
        assert!(resolve_credentials_profile(None).unwrap().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn non_unicode_credentials_profile_is_not_the_default_profile() {
        use std::os::windows::ffi::OsStringExt;
        let raw = std::ffi::OsString::from_wide(&[0xd800]);
        assert!(resolve_credentials_profile(Some(&raw)).is_err());
    }

    #[test]
    fn environment_parse_is_case_insensitive() {
        assert_eq!(
            CTraderBrokerEnvironment::parse("Demo"),
            Some(CTraderBrokerEnvironment::Demo)
        );
        assert_eq!(
            CTraderBrokerEnvironment::parse("LIVE"),
            Some(CTraderBrokerEnvironment::Live)
        );
        assert_eq!(CTraderBrokerEnvironment::parse("xyz"), None);
    }
}
