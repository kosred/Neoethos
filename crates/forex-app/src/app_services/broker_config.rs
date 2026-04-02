use crate::app_services::trading::TradingAdapterKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerSessionState {
    Disconnected,
    Configured,
    ReadyForAuth,
    Authenticated,
    #[cfg_attr(not(test), allow(dead_code))]
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BrokerAccountTarget {
    pub account_id: String,
    pub label: String,
    pub enabled_for_execution: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Mt5BrokerSettings {
    pub terminal_path: String,
    pub server: String,
    pub login: String,
    pub accounts: Vec<BrokerAccountTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CTraderBrokerSettings {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uri: String,
    pub authorization_code_input: String,
    pub environment: CTraderBrokerEnvironment,
    pub accounts: Vec<BrokerAccountTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DxTradeBrokerSettings {
    pub platform_url: String,
    pub username: String,
    pub password: String,
    pub accounts: Vec<BrokerAccountTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BrokerSettingsState {
    pub mt5: Mt5BrokerSettings,
    pub ctrader: CTraderBrokerSettings,
    pub dxtrade: DxTradeBrokerSettings,
}

impl BrokerSettingsState {
    pub fn readiness(&self, adapter: TradingAdapterKind) -> AdapterReadinessSnapshot {
        match adapter {
            TradingAdapterKind::Mt5 => {
                let target_count = count_enabled_targets(&self.mt5.accounts);
                AdapterReadinessSnapshot {
                    adapter_name: adapter.as_str().to_string(),
                    session_state: BrokerSessionState::Configured,
                    status_line: "Local terminal bridge selected.".to_string(),
                    missing_fields: Vec::new(),
                    target_count,
                    can_attempt_connect: true,
                }
            }
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
