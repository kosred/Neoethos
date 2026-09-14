use crate::app_services::ctrader_auth::CTraderTokenBundle;
use anyhow::{Context, Result, anyhow};
use keyring::Entry;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::{Arc, Mutex};

pub trait SecretStoreBackend: Clone {
    fn set_secret(&self, service: &str, user: &str, secret: &str) -> Result<()>;
    fn get_secret(&self, service: &str, user: &str) -> Result<Option<String>>;
    fn delete_secret(&self, service: &str, user: &str) -> Result<()>;
}

/// Trait dispatch over the secure token store. Production code uses
/// `load_token_bundle_with_legacy_fallback` because that's the path
/// that migrates from the pre-v0.4.13 keyring entry name; the direct
/// `load_token_bundle` is reachable via the inherent impl on
/// `CTraderSecureStore` (used by tests that pin the no-migration
/// contract). Both shapes stay on the trait surface so a future
/// alternative backend (file-vault, OS keychain wrapper, etc.) can
/// be plugged in without touching call sites.
#[allow(dead_code)] // load_token_bundle trait method; see doc above
pub trait CTraderTokenStore: Send + Sync {
    fn save_token_bundle(&self, bundle: &CTraderTokenBundle) -> Result<()>;
    fn load_token_bundle(&self) -> Result<Option<CTraderTokenBundle>>;
    fn load_token_bundle_with_legacy_fallback(&self) -> Result<Option<CTraderTokenBundle>>;
    fn clear_token_bundle(&self) -> Result<()>;
}

pub const CTRADER_TOKEN_STORE_SERVICE: &str = "neoethos";
pub const CTRADER_TOKEN_STORE_USER: &str = "ctrader.default";
pub const LEGACY_CTRADER_TOKEN_STORE_SERVICE: &str = "neoethos.test";
pub const LEGACY_CTRADER_TOKEN_STORE_USER: &str = "ctrader.account";

#[derive(Clone, Default)]
pub struct KeyringSecretStoreBackend;

impl SecretStoreBackend for KeyringSecretStoreBackend {
    fn set_secret(&self, service: &str, user: &str, secret: &str) -> Result<()> {
        Entry::new(service, user)
            .context("failed to create keyring entry")?
            .set_password(secret)
            .context("failed to write secret to keyring")?;
        Ok(())
    }

    fn get_secret(&self, service: &str, user: &str) -> Result<Option<String>> {
        let entry = Entry::new(service, user).context("failed to create keyring entry")?;
        match entry.get_password() {
            Ok(secret) => Ok(Some(secret)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(anyhow!(error)).context("failed to read secret from keyring"),
        }
    }

    fn delete_secret(&self, service: &str, user: &str) -> Result<()> {
        let entry = Entry::new(service, user).context("failed to create keyring entry")?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(anyhow!(error)).context("failed to delete secret from keyring"),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Default)]
pub struct MemorySecretStoreBackend {
    entries: Arc<Mutex<HashMap<(String, String), String>>>,
}

#[cfg(test)]
impl MemorySecretStoreBackend {
    pub fn seed(&self, service: &str, user: &str, secret: String) {
        self.entries
            .lock()
            .expect("memory secret store lock poisoned")
            .insert((service.to_string(), user.to_string()), secret);
    }
}

#[cfg(test)]
impl SecretStoreBackend for MemorySecretStoreBackend {
    fn set_secret(&self, service: &str, user: &str, secret: &str) -> Result<()> {
        self.entries
            .lock()
            .expect("memory secret store lock poisoned")
            .insert((service.to_string(), user.to_string()), secret.to_string());
        Ok(())
    }

    fn get_secret(&self, service: &str, user: &str) -> Result<Option<String>> {
        Ok(self
            .entries
            .lock()
            .expect("memory secret store lock poisoned")
            .get(&(service.to_string(), user.to_string()))
            .cloned())
    }

    fn delete_secret(&self, service: &str, user: &str) -> Result<()> {
        self.entries
            .lock()
            .expect("memory secret store lock poisoned")
            .remove(&(service.to_string(), user.to_string()));
        Ok(())
    }
}

#[derive(Clone)]
pub struct CTraderSecureStore<B: SecretStoreBackend = KeyringSecretStoreBackend> {
    service: String,
    // Invalid explicit profiles carry their error to every fallible operation,
    // without ever constructing or consulting the default keyring entry.
    user: std::result::Result<String, String>,
    backend: B,
}

impl<B: SecretStoreBackend> CTraderSecureStore<B> {
    pub fn new(service: impl Into<String>, user: impl Into<String>, backend: B) -> Self {
        Self {
            service: service.into(),
            user: Ok(user.into()),
            backend,
        }
    }

    pub fn save_token_bundle(&self, bundle: &CTraderTokenBundle) -> Result<()> {
        let user = self.user()?;
        let secret =
            serde_json::to_string(bundle).context("failed to serialize cTrader token bundle")?;
        self.backend
            .set_secret(&self.service, user, &secret)
            .context("failed to persist cTrader token bundle")
    }

    pub fn load_token_bundle(&self) -> Result<Option<CTraderTokenBundle>> {
        let user = self.user()?;
        let Some(secret) = self
            .backend
            .get_secret(&self.service, user)
            .context("failed to load cTrader token bundle")?
        else {
            return Ok(None);
        };

        decode_token_bundle(&secret).map(Some)
    }

    pub fn load_token_bundle_with_legacy_fallback(&self) -> Result<Option<CTraderTokenBundle>> {
        if let Some(bundle) = self.load_token_bundle()? {
            return Ok(Some(bundle));
        }
        if self.service != CTRADER_TOKEN_STORE_SERVICE || self.user()? != CTRADER_TOKEN_STORE_USER {
            return Ok(None);
        }

        let Some(secret) = self
            .backend
            .get_secret(
                LEGACY_CTRADER_TOKEN_STORE_SERVICE,
                LEGACY_CTRADER_TOKEN_STORE_USER,
            )
            .context("failed to load legacy cTrader token bundle")?
        else {
            return Ok(None);
        };
        let bundle = decode_token_bundle(&secret)?;
        self.save_token_bundle(&bundle)
            .context("failed to migrate legacy cTrader token bundle")?;
        Ok(Some(bundle))
    }

    pub fn clear_token_bundle(&self) -> Result<()> {
        let user = self.user()?;
        self.backend
            .delete_secret(&self.service, user)
            .context("failed to clear cTrader token bundle")
    }

    fn user(&self) -> Result<&str> {
        self.user.as_deref().map_err(|error| anyhow!("{error}"))
    }
}

/// An explicit credentials file is a separate OAuth profile, not an alias for
/// the operator's default login. Existing overrides therefore require their own
/// authorization; neither default nor legacy tokens are silently imported.
pub fn production_ctrader_token_store() -> CTraderSecureStore<KeyringSecretStoreBackend> {
    token_store_for_profile(
        neoethos_core::broker_config::credentials_profile_path(),
        KeyringSecretStoreBackend,
    )
}

fn token_store_for_profile<B: SecretStoreBackend>(
    profile: Result<Option<PathBuf>>,
    backend: B,
) -> CTraderSecureStore<B> {
    let user = profile
        .and_then(|profile| match profile {
            None => Ok(CTRADER_TOKEN_STORE_USER.to_string()),
            Some(path) => {
                let path = path
                    .to_str()
                    .context("resolved broker credentials profile path is not Unicode")?;
                let mut hash = Sha256::new();
                hash.update(b"neoethos.ctrader.credentials-profile.v1\0");
                hash.update(path.as_bytes());
                Ok(format!("ctrader.profile.v1.{:x}", hash.finalize()))
            }
        })
        .map_err(|error| format!("invalid broker credentials profile: {error:#}"));
    CTraderSecureStore {
        service: CTRADER_TOKEN_STORE_SERVICE.to_string(),
        user,
        backend,
    }
}

fn decode_token_bundle(secret: &str) -> Result<CTraderTokenBundle> {
    let value: serde_json::Value =
        serde_json::from_str(secret).context("failed to parse stored cTrader token bundle")?;
    let required_fields = ["access_token", "refresh_token", "token_type", "scope"];
    if required_fields.iter().any(|field| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(|value| value.trim().is_empty())
            .unwrap_or(true)
    }) {
        return Err(anyhow!("incomplete cTrader token bundle in secure storage"));
    }
    // 2026-06-10: validate the shape beyond non-emptiness. An OAuth bearer
    // bundle must carry token_type=="bearer" and a positive expires_in; a
    // malformed/zero lifetime would make the refresh-ahead window misfire
    // (treat a live token as perpetually expired, or never refresh). Reject
    // early with a clear message instead of trusting a junk bundle.
    if let Some(token_type) = value.get("token_type").and_then(serde_json::Value::as_str)
        && !token_type.trim().eq_ignore_ascii_case("bearer")
    {
        return Err(anyhow!(
            "unexpected cTrader token_type {token_type:?} in secure storage (expected \"bearer\")"
        ));
    }
    if value
        .get("expires_in")
        .and_then(serde_json::Value::as_i64)
        .map(|secs| secs <= 0)
        .unwrap_or(true)
    {
        return Err(anyhow!(
            "cTrader token bundle has a missing or non-positive expires_in in secure storage"
        ));
    }
    serde_json::from_value(value).context("failed to decode stored cTrader token bundle")
}

impl<B> CTraderTokenStore for CTraderSecureStore<B>
where
    B: SecretStoreBackend + Send + Sync + 'static,
{
    fn save_token_bundle(&self, bundle: &CTraderTokenBundle) -> Result<()> {
        CTraderSecureStore::save_token_bundle(self, bundle)
    }

    fn load_token_bundle(&self) -> Result<Option<CTraderTokenBundle>> {
        CTraderSecureStore::load_token_bundle(self)
    }

    fn load_token_bundle_with_legacy_fallback(&self) -> Result<Option<CTraderTokenBundle>> {
        CTraderSecureStore::load_token_bundle_with_legacy_fallback(self)
    }

    fn clear_token_bundle(&self) -> Result<()> {
        CTraderSecureStore::clear_token_bundle(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct ProfileSpy {
        memory: MemorySecretStoreBackend,
        calls: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl SecretStoreBackend for ProfileSpy {
        fn set_secret(&self, service: &str, user: &str, secret: &str) -> Result<()> {
            self.calls.lock().unwrap().push(("set".into(), user.into()));
            self.memory.set_secret(service, user, secret)
        }

        fn get_secret(&self, service: &str, user: &str) -> Result<Option<String>> {
            self.calls.lock().unwrap().push(("get".into(), user.into()));
            self.memory.get_secret(service, user)
        }

        fn delete_secret(&self, service: &str, user: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(("delete".into(), user.into()));
            self.memory.delete_secret(service, user)
        }
    }

    fn profile_bundle() -> CTraderTokenBundle {
        CTraderTokenBundle {
            access_token: "profile-access".into(),
            refresh_token: "profile-refresh".into(),
            token_type: "bearer".into(),
            expires_in: 3600,
            scope: "trading".into(),
            created_at_unix: 1_774_147_200,
        }
    }

    #[test]
    fn explicit_profiles_never_read_migrate_or_clear_default_tokens() {
        let backend = ProfileSpy::default();
        // Invalid payloads also make any unintended legacy/default decoding fail.
        backend.memory.seed(
            CTRADER_TOKEN_STORE_SERVICE,
            CTRADER_TOKEN_STORE_USER,
            "operator-token".into(),
        );
        backend.memory.seed(
            LEGACY_CTRADER_TOKEN_STORE_SERVICE,
            LEGACY_CTRADER_TOKEN_STORE_USER,
            "legacy-token".into(),
        );
        let original = backend.memory.entries.lock().unwrap().clone();
        let a = token_store_for_profile(
            Ok(Some(
                std::path::absolute("profile-a/credentials.toml").unwrap(),
            )),
            backend.clone(),
        );
        let b = token_store_for_profile(
            Ok(Some(
                std::path::absolute("profile-b/credentials.toml").unwrap(),
            )),
            backend.clone(),
        );
        assert_ne!(a.user().unwrap(), b.user().unwrap());
        assert_eq!(a.load_token_bundle_with_legacy_fallback().unwrap(), None);
        a.save_token_bundle(&profile_bundle()).unwrap();
        assert_eq!(
            a.load_token_bundle_with_legacy_fallback().unwrap(),
            Some(profile_bundle())
        );
        assert_eq!(b.load_token_bundle_with_legacy_fallback().unwrap(), None);
        b.clear_token_bundle().unwrap();
        assert_eq!(a.load_token_bundle().unwrap(), Some(profile_bundle()));
        a.clear_token_bundle().unwrap();
        assert_eq!(*backend.memory.entries.lock().unwrap(), original);
        let calls = backend.calls.lock().unwrap();
        assert_eq!(calls.len(), 7);
        assert!(
            calls
                .iter()
                .all(|(_, user)| user.starts_with("ctrader.profile.v1."))
        );
    }

    #[test]
    fn invalid_explicit_profile_has_zero_secret_backend_access() {
        for reason in ["empty", "not Unicode", "cannot resolve absolute path"] {
            let backend = ProfileSpy::default();
            let store = token_store_for_profile(Err(anyhow!("{reason}")), backend.clone());
            assert!(store.load_token_bundle().is_err());
            assert!(store.load_token_bundle_with_legacy_fallback().is_err());
            assert!(store.save_token_bundle(&profile_bundle()).is_err());
            assert!(store.clear_token_bundle().is_err());
            assert!(backend.calls.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn explicit_profile_namespace_does_not_change_when_file_is_created() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "neoethos-token-profile-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("credentials.toml");
        let before =
            token_store_for_profile(Ok(Some(path.clone())), MemorySecretStoreBackend::default());
        std::fs::write(&path, "[ctrader]\n").unwrap();
        let after =
            token_store_for_profile(Ok(Some(path.clone())), MemorySecretStoreBackend::default());
        assert_eq!(before.user().unwrap(), after.user().unwrap());
        let default = token_store_for_profile(Ok(None), MemorySecretStoreBackend::default());
        assert_eq!(default.user().unwrap(), CTRADER_TOKEN_STORE_USER);
        assert_ne!(before.user().unwrap(), default.user().unwrap());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn production_ctrader_token_store_identity_is_not_test_scoped() {
        assert_eq!(CTRADER_TOKEN_STORE_SERVICE, "neoethos");
        assert_eq!(CTRADER_TOKEN_STORE_USER, "ctrader.default");
        assert!(!CTRADER_TOKEN_STORE_SERVICE.contains(".test"));
    }

    #[test]
    fn secure_store_round_trip_saves_loads_and_clears_bundle() {
        let backend = MemorySecretStoreBackend::default();
        let store = CTraderSecureStore::new(
            CTRADER_TOKEN_STORE_SERVICE,
            CTRADER_TOKEN_STORE_USER,
            backend.clone(),
        );
        let bundle = CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: 1_774_147_200,
        };

        store
            .save_token_bundle(&bundle)
            .expect("save should succeed");
        let restored = store.load_token_bundle().expect("load should succeed");
        assert_eq!(restored, Some(bundle));

        store.clear_token_bundle().expect("clear should succeed");
        assert_eq!(
            store.load_token_bundle().expect("load should succeed"),
            None
        );
    }

    #[test]
    fn production_store_migrates_legacy_test_scoped_bundle() {
        let backend = MemorySecretStoreBackend::default();
        let production_store = CTraderSecureStore::new(
            CTRADER_TOKEN_STORE_SERVICE,
            CTRADER_TOKEN_STORE_USER,
            backend.clone(),
        );
        let legacy_store = CTraderSecureStore::new(
            LEGACY_CTRADER_TOKEN_STORE_SERVICE,
            LEGACY_CTRADER_TOKEN_STORE_USER,
            backend,
        );
        let bundle = CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: 1_774_147_200,
        };

        legacy_store
            .save_token_bundle(&bundle)
            .expect("save legacy bundle should succeed");
        let restored = production_store
            .load_token_bundle_with_legacy_fallback()
            .expect("legacy fallback should load");

        assert_eq!(restored, Some(bundle.clone()));
        assert_eq!(
            production_store
                .load_token_bundle()
                .expect("production load should succeed"),
            Some(bundle)
        );
    }

    #[test]
    fn secure_store_rejects_incomplete_bundle_payloads() {
        let backend = MemorySecretStoreBackend::default();
        backend.seed(
            "neoethos.test",
            "ctrader.account",
            "{\"access_token\":\"access\"}".to_string(),
        );
        let store = CTraderSecureStore::new("neoethos.test", "ctrader.account", backend);

        let error = store
            .load_token_bundle()
            .expect_err("incomplete payload must fail");
        assert!(error.to_string().contains("incomplete"));
    }
}
