//! App-side persistence wrapper for [`BrokerSettingsState`].
//!
//! The raw I/O lives in `neoethos-core::broker_config` so the CLI can
//! reuse it. This module layers the `neoethos-app`-specific
//! embedded-credentials fallback (compile-time constants baked in by
//! `build.rs`) on top of the load path.
//!
//! # Lookup order (highest priority first)
//!
//! 1. `$NEOETHOS_BROKER_CREDENTIALS_PATH` selects an isolated credentials and
//!    OAuth profile. It never scans/heals default credential files. Invalid
//!    explicit paths fail closed rather than selecting the operator's profile.
//! 2. `<dirs::config_dir>/neoethos/broker_credentials.toml` — `%APPDATA%` on
//!    Windows, `$XDG_CONFIG_HOME` on Linux, `~/Library/Application Support` on
//!    macOS.
//! 3. `<cwd>/.local/neoethos/broker_credentials.toml` — dev machine fallback.
//! 4. Compile-time embedded constants from [`crate::app_services::embedded_credentials`]
//!    — baked into the binary by `build.rs` for zero-config distribution.
//!    THIS LAYER LIVES ONLY IN `neoethos-app` — the CLI's `credentials set`
//!    path skips it on purpose (the operator is writing fresh values).
//!
//! # Security
//!
//! The TOML file is intended to live OUTSIDE the git repository.
//! One transient field is explicitly NEVER serialized:
//!
//! - `CTraderBrokerSettings::authorization_code_input` — short-lived OAuth value

use crate::app_services::broker_config::BrokerSettingsState;
use anyhow::Result;

/// Loads broker settings, applying the four-level resolution chain.
///
/// Never panics. Returns settings with cTrader credentials populated from
/// the highest-priority source that has a non-empty `client_id`.
pub fn load_broker_settings() -> BrokerSettingsState {
    load_for_profile(
        neoethos_core::broker_config::credentials_profile_path(),
        || {
            // #141: Detect + heal credentials drift before we read. If the
            // user re-authenticated from a different CWD, they end up with
            // two files (e.g. `%APPDATA%\neoethos\broker_credentials.toml`
            // AND `<cwd>/.local/neoethos/broker_credentials.toml`) that
            // contain DIFFERENT `account_id` rows. The load path picks the
            // first one that exists, which is non-deterministic across
            // launches if the CWD changes. Healing here renames the stale
            // copies to `*.bak.<timestamp>` so subsequent loads are
            // canonical.
            let _ = heal_credentials_drift();
            load_from_filesystem()
        },
    )
}

fn load_for_profile(
    profile: Result<Option<std::path::PathBuf>>,
    load_default: impl FnOnce() -> BrokerSettingsState,
) -> BrokerSettingsState {
    let mut settings = match profile {
        Ok(Some(path)) => load_from_path(&path),
        Ok(None) => load_default(),
        Err(error) => {
            tracing::error!(%error, "invalid broker credentials profile; default credentials are not consulted");
            return BrokerSettingsState::default();
        }
    };
    apply_embedded_fallback(&mut settings);
    settings
}

/// Detect more than one populated candidate path and migrate the
/// stale ones to `*.bak.<unix-ms>` so future loads are
/// deterministic. The "freshest" path (highest mtime) wins. Backs
/// up the loser instead of deleting it so the operator can recover
/// the previous `account_id` if needed.
///
/// Best-effort: any IO error is logged and swallowed. The function
/// is called from `load_broker_settings` before the actual load,
/// so subsequent reads see the cleaned-up disk state.
fn heal_credentials_drift() -> Result<()> {
    let candidates = neoethos_core::broker_config::candidate_credentials_paths()?;
    heal_credentials_drift_candidates(candidates)
}

fn heal_credentials_drift_candidates(candidates: Vec<std::path::PathBuf>) -> Result<()> {
    let existing: Vec<_> = candidates.iter().filter(|p| p.is_file()).cloned().collect();
    if existing.len() < 2 {
        return Ok(()); // nothing to heal
    }

    // Find the one with the latest modified time — that's the
    // "fresh" credentials the user actually wrote during their
    // last re-auth.
    let mut with_mtime: Vec<(std::path::PathBuf, std::time::SystemTime)> = existing
        .into_iter()
        .filter_map(|p| {
            std::fs::metadata(&p)
                .and_then(|m| m.modified())
                .ok()
                .map(|t| (p, t))
        })
        .collect();
    with_mtime.sort_by(|a, b| b.1.cmp(&a.1)); // newest first
    let Some((canonical, _)) = with_mtime.first() else {
        return Ok(());
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    for (path, _) in with_mtime.iter().skip(1) {
        let backup = path.with_extension(format!("toml.bak.{now_ms}"));
        match std::fs::rename(path, &backup) {
            Ok(()) => tracing::warn!(
                target: "neoethos_app::broker_persistence",
                stale = %path.display(),
                backup = %backup.display(),
                canonical = %canonical.display(),
                "credentials drift: renamed stale copy to backup — \
                 the canonical file (latest mtime) wins"
            ),
            Err(err) => tracing::warn!(
                target: "neoethos_app::broker_persistence",
                stale = %path.display(),
                error = %err,
                "credentials drift: could not rename stale copy — \
                 manual cleanup may be needed"
            ),
        }
    }
    Ok(())
}

/// Default filesystem portion only (levels 2–3); explicit profiles are already
/// resolved before this function is reachable.
fn default_credentials_file_path() -> Result<std::path::PathBuf> {
    // The explicit profile was already captured by the caller. Do not read the
    // process environment again while selecting the default profile's file.
    let candidates = neoethos_core::broker_config::candidate_credentials_paths()?;
    candidates
        .iter()
        .find(|path| path.is_file())
        .or_else(|| candidates.first())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no default broker credentials path could be resolved"))
}

fn load_from_filesystem() -> BrokerSettingsState {
    let path = match default_credentials_file_path() {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!(error = %err, "broker credentials path resolution failed");
            return BrokerSettingsState::default();
        }
    };

    load_from_path(&path)
}

fn load_from_path(path: &std::path::Path) -> BrokerSettingsState {
    match neoethos_core::broker_config::load_from_disk(path) {
        Ok(Some(s)) => {
            tracing::info!(
                path = %path.display(),
                schema_version = %s.schema_version,
                "loaded broker credentials from disk"
            );
            s
        }
        Ok(None) => {
            tracing::debug!(
                path = %path.display(),
                "no broker credentials file found; will use embedded defaults"
            );
            BrokerSettingsState::default()
        }
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "failed to load broker credentials TOML; will try embedded defaults"
            );
            BrokerSettingsState::default()
        }
    }
}

/// Level-4 fallback: fill any empty cTrader fields from compile-time constants.
/// User-supplied values (non-empty) are never overwritten.
fn apply_embedded_fallback(settings: &mut BrokerSettingsState) {
    use crate::app_services::embedded_credentials::{
        EMBEDDED_CTRADER_CLIENT_ID, EMBEDDED_CTRADER_CLIENT_SECRET, EMBEDDED_CTRADER_REDIRECT_URI,
    };

    if EMBEDDED_CTRADER_CLIENT_ID.is_empty() {
        return; // binary was built without embedded credentials — nothing to do
    }

    let ct = &mut settings.ctrader;
    let used_embedded =
        ct.client_id.is_empty() || ct.client_secret.is_empty() || ct.redirect_uri.is_empty();

    if ct.client_id.is_empty() {
        ct.client_id = EMBEDDED_CTRADER_CLIENT_ID.to_string();
    }
    if ct.client_secret.is_empty() {
        ct.client_secret = EMBEDDED_CTRADER_CLIENT_SECRET.to_string();
    }
    if ct.redirect_uri.is_empty() {
        ct.redirect_uri = EMBEDDED_CTRADER_REDIRECT_URI.to_string();
    }

    if used_embedded {
        tracing::info!(
            "using embedded compile-time cTrader credentials \
             (no user-level config file with credentials found)"
        );
    }
}

/// Persists broker settings to disk at the resolved credentials path.
///
/// Creates the parent directory if missing. Writes TOML in the standard
/// formatting. The transient field (`authorization_code_input`) is
/// excluded by its serde annotation. The shared
/// writer in `neoethos-core` always stamps the current schema version
/// regardless of what the in-memory value carries.
pub fn save_broker_settings(settings: &BrokerSettingsState) -> Result<()> {
    let path = match neoethos_core::broker_config::credentials_profile_path()? {
        Some(path) => path,
        None => default_credentials_file_path()?,
    };
    neoethos_core::broker_config::save_to_disk(&path, settings)?;
    tracing::info!(path = %path.display(), "saved broker credentials to disk");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::broker_config::{
        BrokerAccountTarget, CTraderBrokerEnvironment, CTraderBrokerSettings,
    };
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::{env, fs};

    /// The path-override env var lives in
    /// `neoethos_core::env_overrides::ENV_BROKER_CREDENTIALS_PATH`
    /// now. The local alias keeps the in-file `env::set_var` /
    /// `env::remove_var` test calls readable while still tracking the
    /// canonical const — so a rename breaks the test file too.
    const ENV_OVERRIDE_VAR: &str = neoethos_core::env_overrides::ENV_BROKER_CREDENTIALS_PATH;

    /// `env::set_var`/`env::var` are process-global. Serialize the env-mutating
    /// tests so they don't race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that always restores the env even when `body` panics.
    /// Without this guard, a panicking test left `ENV_OVERRIDE_VAR` set
    /// AND poisoned `ENV_LOCK`, which then blew up every subsequent
    /// env-touching test with `env lock poisoned`. Now we recover from
    /// poison via `into_inner` and Drop runs in any path.
    struct EnvOverrideGuard;
    impl Drop for EnvOverrideGuard {
        fn drop(&mut self) {
            // SAFETY: `with_env_path` holds `ENV_LOCK` for the lifetime
            // of this guard — no other test can be touching the env.
            unsafe {
                env::remove_var(ENV_OVERRIDE_VAR);
            }
        }
    }

    fn with_env_path<F: FnOnce(&std::path::Path)>(path: &std::path::Path, body: F) {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: the lock above ensures no concurrent env access from
        // these tests; cargo test still parallelizes outer tests but
        // the env-touching ones share the lock.
        unsafe {
            env::set_var(ENV_OVERRIDE_VAR, path);
        }
        let _env_guard = EnvOverrideGuard;
        body(path);
        // _env_guard's Drop fires here (or on panic) — env always cleared.
    }

    fn populated_settings() -> BrokerSettingsState {
        BrokerSettingsState {
            schema_version: crate::app_services::broker_config::BROKER_CREDENTIALS_SCHEMA_VERSION,
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
        }
    }

    #[test]
    fn explicit_profile_loader_never_enumerates_or_heals_default_paths() {
        let dir = tempdir_or_skip();
        let path = dir.join("isolated.toml");
        let no_default = || panic!("explicit profile must not enter the default loader/healer");
        let missing = load_for_profile(Ok(Some(path.clone())), no_default);
        assert!(missing.ctrader.accounts.is_empty());
        neoethos_core::broker_config::save_to_disk(&path, &populated_settings()).unwrap();
        let loaded = load_for_profile(Ok(Some(path.clone())), no_default);
        assert_eq!(loaded.ctrader.client_id, "client-123");
        assert_eq!(loaded.ctrader.accounts[0].account_id, "ctr-001");
        fs::write(&path, "[malformed").unwrap();
        let malformed = load_for_profile(Ok(Some(path)), no_default);
        assert!(malformed.ctrader.accounts.is_empty());
    }

    #[test]
    fn invalid_profile_loader_never_falls_back_but_absent_profile_keeps_defaults() {
        let invalid = load_for_profile(Err(anyhow::anyhow!("invalid explicit profile")), || {
            panic!("invalid profile must not enter the default loader/healer")
        });
        assert_eq!(invalid, BrokerSettingsState::default());
        let loaded = load_for_profile(Ok(None), populated_settings);
        assert_eq!(loaded.ctrader.client_id, "client-123");
        assert_eq!(loaded.ctrader.accounts[0].account_id, "ctr-001");
    }

    #[test]
    fn load_returns_embedded_or_default_when_file_missing() {
        use crate::app_services::embedded_credentials::EMBEDDED_CTRADER_CLIENT_ID;

        let dir = tempdir_or_skip();
        let path = dir.join("does-not-exist.toml");
        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            if EMBEDDED_CTRADER_CLIENT_ID.is_empty() {
                // No embedded credentials baked in — expect empty defaults.
                assert_eq!(loaded.ctrader.client_id, "");
            } else {
                // Embedded credentials should fill the gap.
                assert_eq!(loaded.ctrader.client_id, EMBEDDED_CTRADER_CLIENT_ID);
            }
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
    fn embedded_fallback_fills_empty_client_id() {
        use crate::app_services::embedded_credentials::EMBEDDED_CTRADER_CLIENT_ID;

        if EMBEDDED_CTRADER_CLIENT_ID.is_empty() {
            // Binary was built without embedded credentials — test is vacuously passing.
            return;
        }

        let dir = tempdir_or_skip();
        // Write a TOML that is valid but has no ctrader section (all defaults = empty).
        let path = dir.join("empty_creds.toml");
        fs::write(&path, "[ctrader]\n").expect("write");

        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            assert_eq!(
                loaded.ctrader.client_id, EMBEDDED_CTRADER_CLIENT_ID,
                "empty client_id should be filled from embedded constant"
            );
        });
    }

    #[test]
    fn user_credentials_win_over_embedded() {
        use crate::app_services::embedded_credentials::EMBEDDED_CTRADER_CLIENT_ID;

        if EMBEDDED_CTRADER_CLIENT_ID.is_empty() {
            return; // no embedded credentials to compete with
        }

        let dir = tempdir_or_skip();
        let path = dir.join("user_creds.toml");

        with_env_path(&path, |_| {
            let original = populated_settings(); // has client_id = "client-123"
            save_broker_settings(&original).expect("save");

            let loaded = load_broker_settings();
            assert_eq!(
                loaded.ctrader.client_id, "client-123",
                "user-supplied client_id must not be overwritten by embedded constant"
            );
            assert_ne!(
                loaded.ctrader.client_id, EMBEDDED_CTRADER_CLIENT_ID,
                "embedded constant must not win when user value is present"
            );
        });
    }

    #[test]
    fn save_then_load_round_trips_schema_version() {
        use crate::app_services::broker_config::BROKER_CREDENTIALS_SCHEMA_VERSION;
        let dir = tempdir_or_skip();
        let path = dir.join("creds.toml");
        with_env_path(&path, |_| {
            let original = populated_settings();
            save_broker_settings(&original).expect("save");
            let loaded = load_broker_settings();
            // The save path stamps the CURRENT schema version on
            // every write, so the loaded value must match the
            // constant regardless of what was constructed in memory.
            assert_eq!(loaded.schema_version, BROKER_CREDENTIALS_SCHEMA_VERSION);
        });
    }

    #[test]
    fn loading_pre_versioning_toml_defaults_to_v1() {
        use neoethos_core::SchemaVersion;
        let dir = tempdir_or_skip();
        let path = dir.join("pre_v1_creds.toml");
        // Write a TOML that LACKS the schema_version field — this
        // is what files written by builds before Phase D4 look
        // like. The `#[serde(default = "default_v1")]` attribute
        // must kick in and treat it as v1.
        let raw = "[ctrader]\nclient_id = \"old\"\n";
        fs::write(&path, raw).expect("write");
        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            assert_eq!(loaded.schema_version, SchemaVersion::new(1));
            assert_eq!(loaded.ctrader.client_id, "old");
        });
    }

    #[test]
    fn loading_too_new_schema_version_falls_back_to_default() {
        // Simulate a TOML written by a FUTURE build whose schema
        // version this binary doesn't understand. The loader must
        // fail loud (log an error) and return defaults rather than
        // silently mis-parsing potentially-incompatible data.
        let dir = tempdir_or_skip();
        let path = dir.join("future_creds.toml");
        let raw = "schema_version = 999\n[ctrader]\nclient_id = \"future\"\n";
        fs::write(&path, raw).expect("write");
        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            // Falls back to default-but-then-embedded-fallback-applied.
            // The key invariant: it does NOT carry the "future" client_id.
            assert_ne!(loaded.ctrader.client_id, "future");
        });
    }

    #[test]
    fn heal_credentials_drift_renames_stale_copy() {
        // Exercise the actual production healer with owned candidate files.
        // Never enumerate the operator's directories or mutate process CWD.
        use std::time::Duration;

        let dir = tempdir_or_skip();
        let local_dir = dir.join(".local").join("neoethos");
        fs::create_dir_all(&local_dir).expect("local dir");
        let local_file = local_dir.join("broker_credentials.toml");
        fs::write(&local_file, "[ctrader]\nclient_id = \"OLDER\"\n").expect("local file");

        let canonical_file = dir.join("canonical_creds.toml");
        fs::write(&canonical_file, "[ctrader]\nclient_id = \"NEWER\"\n").expect("canonical");
        // Force canonical's mtime to be NEWER than local's.
        // SystemTime::now() vs local_file's stamp is enough on
        // most filesystems, but be explicit by sleeping a beat
        // and rewriting the canonical.
        std::thread::sleep(Duration::from_millis(20));
        fs::write(&canonical_file, "[ctrader]\nclient_id = \"NEWER\"\n")
            .expect("canonical retouched");

        heal_credentials_drift_candidates(vec![local_file.clone(), canonical_file.clone()])
            .expect("actual healer");
        assert!(!local_file.exists());
        assert_eq!(
            fs::read_to_string(&canonical_file).unwrap(),
            "[ctrader]\nclient_id = \"NEWER\"\n"
        );
        let backups: Vec<_> = fs::read_dir(&local_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(backups.len(), 1);
        assert!(
            backups[0]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("broker_credentials.toml.bak.")
        );
        assert_eq!(
            fs::read_to_string(&backups[0]).unwrap(),
            "[ctrader]\nclient_id = \"OLDER\"\n"
        );
    }

    #[test]
    fn malformed_toml_falls_back_to_default() {
        use crate::app_services::embedded_credentials::EMBEDDED_CTRADER_CLIENT_ID;

        let dir = tempdir_or_skip();
        let path = dir.join("malformed.toml");
        fs::write(&path, "not = valid \n[unclosed").expect("write");

        with_env_path(&path, |_| {
            let loaded = load_broker_settings();
            // Filesystem load fell back to default (good — the malformed
            // TOML did not panic). Then the embedded fallback overlay
            // ran, so the loaded result equals "default + embedded".
            let mut expected = BrokerSettingsState::default();
            apply_embedded_fallback(&mut expected);
            assert_eq!(loaded, expected);
            // Sanity-check that the failure path went through the
            // default rather than parsing junk into real fields.
            if EMBEDDED_CTRADER_CLIENT_ID.is_empty() {
                assert_eq!(loaded.ctrader.client_id, "");
            } else {
                assert_eq!(loaded.ctrader.client_id, EMBEDDED_CTRADER_CLIENT_ID);
            }
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
        let path = env::temp_dir().join(format!("neoethos-bp-{pid}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir should be creatable");
        path
    }
}
