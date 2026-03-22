use crate::app_services::ctrader_auth::CTraderTokenBundle;
use anyhow::{anyhow, Context, Result};
use keyring::Entry;

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::{Arc, Mutex};

pub trait SecretStoreBackend: Clone {
    fn set_secret(&self, service: &str, user: &str, secret: &str) -> Result<()>;
    fn get_secret(&self, service: &str, user: &str) -> Result<Option<String>>;
    fn delete_secret(&self, service: &str, user: &str) -> Result<()>;
}

pub trait CTraderTokenStore: Send + Sync {
    fn save_token_bundle(&self, bundle: &CTraderTokenBundle) -> Result<()>;
    fn load_token_bundle(&self) -> Result<Option<CTraderTokenBundle>>;
    fn clear_token_bundle(&self) -> Result<()>;
}

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
    user: String,
    backend: B,
}

impl<B: SecretStoreBackend> CTraderSecureStore<B> {
    pub fn new(service: impl Into<String>, user: impl Into<String>, backend: B) -> Self {
        Self {
            service: service.into(),
            user: user.into(),
            backend,
        }
    }

    pub fn save_token_bundle(&self, bundle: &CTraderTokenBundle) -> Result<()> {
        let secret = serde_json::to_string(bundle).context("failed to serialize cTrader token bundle")?;
        self.backend
            .set_secret(&self.service, &self.user, &secret)
            .context("failed to persist cTrader token bundle")
    }

    pub fn load_token_bundle(&self) -> Result<Option<CTraderTokenBundle>> {
        let Some(secret) = self
            .backend
            .get_secret(&self.service, &self.user)
            .context("failed to load cTrader token bundle")?
        else {
            return Ok(None);
        };

        let value: serde_json::Value =
            serde_json::from_str(&secret).context("failed to parse stored cTrader token bundle")?;
        let required_fields = ["access_token", "refresh_token", "token_type", "scope"];
        if required_fields.iter().any(|field| {
            value
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(|value| value.trim().is_empty())
                .unwrap_or(true)
        })
        {
            return Err(anyhow!("incomplete cTrader token bundle in secure storage"));
        }
        let bundle: CTraderTokenBundle =
            serde_json::from_value(value).context("failed to decode stored cTrader token bundle")?;

        Ok(Some(bundle))
    }

    pub fn clear_token_bundle(&self) -> Result<()> {
        self.backend
            .delete_secret(&self.service, &self.user)
            .context("failed to clear cTrader token bundle")
    }
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

    fn clear_token_bundle(&self) -> Result<()> {
        CTraderSecureStore::clear_token_bundle(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_store_round_trip_saves_loads_and_clears_bundle() {
        let backend = MemorySecretStoreBackend::default();
        let store = CTraderSecureStore::new(
            "forex-ai.test",
            "ctrader.account",
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

        store.save_token_bundle(&bundle).expect("save should succeed");
        let restored = store.load_token_bundle().expect("load should succeed");
        assert_eq!(restored, Some(bundle));

        store.clear_token_bundle().expect("clear should succeed");
        assert_eq!(store.load_token_bundle().expect("load should succeed"), None);
    }

    #[test]
    fn secure_store_rejects_incomplete_bundle_payloads() {
        let backend = MemorySecretStoreBackend::default();
        backend.seed(
            "forex-ai.test",
            "ctrader.account",
            "{\"access_token\":\"access\"}".to_string(),
        );
        let store = CTraderSecureStore::new(
            "forex-ai.test",
            "ctrader.account",
            backend,
        );

        let error = store.load_token_bundle().expect_err("incomplete payload must fail");
        assert!(error.to_string().contains("incomplete"));
    }
}
