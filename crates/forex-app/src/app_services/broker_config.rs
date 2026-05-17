use crate::app_services::trading::TradingAdapterKind;
use forex_core::{default_v1, HasSchemaVersion, SchemaVersion};
use serde::{Deserialize, Serialize};

/// Current schema version of the `broker_credentials.toml` on-disk
/// contract. Bump when fields are renamed/removed or their types
/// change in a way that `#[serde(default)]` can't bridge.
///
/// v1 (current): the original layout from before Phase D4.
/// New optional fields and field renamings within v1 are
/// backward-compatible via `#[serde(default)]`. A future v2 is
/// reserved for a hypothetical breaking change (e.g. moving
/// credentials into a per-broker dictionary).
pub const BROKER_CREDENTIALS_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerSessionState {
    Disconnected,
    ReadyForAuth,
    Authenticated,
    #[cfg_attr(not(test), allow(dead_code))]
    Failed,
}

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterReadinessSnapshot {
    pub adapter_name: String,
    pub session_state: BrokerSessionState,
    pub status_line: String,
    pub missing_fields: Vec<String>,
    pub target_count: usize,
    pub can_attempt_connect: bool,
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

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DxTradeBrokerSettings {
    #[serde(default)]
    pub platform_url: String,
    #[serde(default)]
    pub username: String,
    /// NEVER persisted to disk. The user enters this each session.
    #[serde(skip_serializing, skip_deserializing, default)]
    pub password: String,
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
    #[serde(default)]
    pub dxtrade: DxTradeBrokerSettings,
}

impl Default for BrokerSettingsState {
    fn default() -> Self {
        Self {
            schema_version: BROKER_CREDENTIALS_SCHEMA_VERSION,
            ctrader: CTraderBrokerSettings::default(),
            dxtrade: DxTradeBrokerSettings::default(),
        }
    }
}

impl HasSchemaVersion for BrokerSettingsState {
    const CURRENT: SchemaVersion = BROKER_CREDENTIALS_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

impl BrokerSettingsState {
    pub fn readiness(&self, adapter: TradingAdapterKind) -> AdapterReadinessSnapshot {
        match adapter {
            TradingAdapterKind::CTrader => {
                let missing_fields = required_missing_fields(&[
                    ("client_id", &self.ctrader.client_id),
                    ("client_secret", &self.ctrader.client_secret),
                    ("redirect_uri", &self.ctrader.redirect_uri),
                ]);
                let ready_line = format!(
                    "OAuth app credentials ready for {} environment.",
                    self.ctrader.environment.as_str()
                );
                build_remote_readiness(
                    adapter,
                    missing_fields,
                    count_enabled_targets(&self.ctrader.accounts),
                    &ready_line,
                )
            }
            TradingAdapterKind::DxTrade => {
                let missing_fields = required_missing_fields(&[
                    ("platform_url", &self.dxtrade.platform_url),
                    ("username", &self.dxtrade.username),
                    ("password", &self.dxtrade.password),
                ]);
                build_remote_readiness(
                    adapter,
                    missing_fields,
                    count_enabled_targets(&self.dxtrade.accounts),
                    "Remote broker credentials ready.",
                )
            }
        }
    }
}

fn build_remote_readiness(
    adapter: TradingAdapterKind,
    missing_fields: Vec<String>,
    target_count: usize,
    ready_line: &str,
) -> AdapterReadinessSnapshot {
    let session_state = if missing_fields.is_empty() {
        BrokerSessionState::ReadyForAuth
    } else {
        BrokerSessionState::Disconnected
    };
    let status_line = if missing_fields.is_empty() {
        ready_line.to_string()
    } else {
        format!(
            "{} configuration incomplete: missing {}",
            adapter.as_str(),
            missing_fields.join(", ")
        )
    };

    AdapterReadinessSnapshot {
        adapter_name: adapter.as_str().to_string(),
        session_state,
        status_line,
        missing_fields: missing_fields.clone(),
        target_count,
        can_attempt_connect: missing_fields.is_empty(),
    }
}

fn required_missing_fields(fields: &[(&str, &str)]) -> Vec<String> {
    fields
        .iter()
        .filter(|(_, value)| value.trim().is_empty())
        .map(|(label, _)| (*label).to_string())
        .collect()
}

fn count_enabled_targets(accounts: &[BrokerAccountTarget]) -> usize {
    accounts
        .iter()
        .filter(|account| account.enabled_for_execution)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::TradingAdapterKind;

    #[test]
    fn ctrader_readiness_requires_oauth_fields_and_counts_targets() {
        let mut settings = BrokerSettingsState::default();

        let missing = settings.readiness(TradingAdapterKind::CTrader);
        assert_eq!(missing.session_state, BrokerSessionState::Disconnected);
        assert_eq!(
            missing.missing_fields,
            vec![
                "client_id".to_string(),
                "client_secret".to_string(),
                "redirect_uri".to_string()
            ]
        );
        assert_eq!(missing.target_count, 0);
        assert!(!missing.can_attempt_connect);

        settings.ctrader.client_id = "app-client".to_string();
        settings.ctrader.client_secret = "top-secret".to_string();
        settings.ctrader.redirect_uri = "http://localhost:3000/callback".to_string();
        settings.ctrader.environment = CTraderBrokerEnvironment::Demo;
        settings.ctrader.accounts.push(BrokerAccountTarget {
            account_id: "ctr-001".to_string(),
            label: "Primary".to_string(),
            enabled_for_execution: true,
        });
        settings.ctrader.accounts.push(BrokerAccountTarget {
            account_id: "ctr-002".to_string(),
            label: "Standby".to_string(),
            enabled_for_execution: false,
        });

        let ready = settings.readiness(TradingAdapterKind::CTrader);
        assert_eq!(ready.session_state, BrokerSessionState::ReadyForAuth);
        assert!(ready.missing_fields.is_empty());
        assert_eq!(ready.target_count, 1);
        assert!(ready.can_attempt_connect);
        assert_eq!(
            ready.status_line,
            "OAuth app credentials ready for Demo environment."
        );
    }

    #[test]
    fn dxtrade_readiness_requires_platform_and_credentials() {
        let mut settings = BrokerSettingsState::default();

        let missing = settings.readiness(TradingAdapterKind::DxTrade);
        assert_eq!(missing.session_state, BrokerSessionState::Disconnected);
        assert_eq!(
            missing.missing_fields,
            vec![
                "platform_url".to_string(),
                "username".to_string(),
                "password".to_string()
            ]
        );
        assert!(!missing.can_attempt_connect);

        settings.dxtrade.platform_url = "https://broker.example/specs".to_string();
        settings.dxtrade.username = "ops-user".to_string();
        settings.dxtrade.password = "ops-pass".to_string();

        let ready = settings.readiness(TradingAdapterKind::DxTrade);
        assert_eq!(ready.session_state, BrokerSessionState::ReadyForAuth);
        assert!(ready.missing_fields.is_empty());
        assert!(ready.can_attempt_connect);
    }
}
