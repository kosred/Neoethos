//! Broker-credentials *behavior* layer. The pure data types live in
//! `neoethos-core` so both this crate and `neoethos-cli` can write the
//! same `broker_credentials.toml`. This module owns the things that
//! depend on `TradingAdapterKind` (a `neoethos-app` concept) and the
//! transient runtime concepts that don't belong on-disk.
//!
//! Re-exports from `neoethos-core` keep import sites in the rest of
//! `neoethos-app` unchanged after the Phase B SoT migration. Existing
//! `use crate::app_services::broker_config::{BrokerSettingsState, ...}`
//! lines still work; new code in `neoethos-cli` imports the same
//! names directly from `neoethos_core::broker_config`.

use crate::app_services::trading::TradingAdapterKind;

pub use neoethos_core::broker_config::{
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, BrokerSettingsState,
    CTRADER_OAUTH_REDIRECT_URI, CTraderBrokerEnvironment, CTraderBrokerSettings,
    DxTradeBrokerSettings,
};
// `CTRADER_CREATE_DEMO_ACCOUNT_URL` / `CTRADER_CREATE_LIVE_ACCOUNT_URL`
// constants live in `neoethos-core::broker_config` and were
// previously re-exported here for the legacy egui "create demo
// account" buttons (removed in #89). When the Flutter shell wants
// to surface those links it can import them directly from
// neoethos-core — no need for a pass-through re-export.

/// Runtime-only flavour of "what state is the broker session in".
/// Not persisted to disk — recomputed from credentials + live
/// connection probes. Kept in `neoethos-app` because only the GUI
/// binary maintains a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerSessionState {
    Disconnected,
    ReadyForAuth,
    Authenticated,
    #[cfg_attr(not(test), allow(dead_code))]
    Failed,
}

/// Read-only readiness snapshot consumed by the UI's broker-setup
/// banner. Tells the operator what's missing before they can try to
/// connect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterReadinessSnapshot {
    pub adapter_name: String,
    pub session_state: BrokerSessionState,
    pub status_line: String,
    pub missing_fields: Vec<String>,
    pub target_count: usize,
    pub can_attempt_connect: bool,
}

/// Extension trait so `BrokerSettingsState` (defined in `neoethos-core`)
/// can still answer the app-side `readiness()` question without
/// reintroducing the `neoethos-app -> neoethos-app` dependency cycle.
pub trait BrokerSettingsReadiness {
    fn readiness(&self, adapter: TradingAdapterKind) -> AdapterReadinessSnapshot;
}

impl BrokerSettingsReadiness for BrokerSettingsState {
    fn readiness(&self, adapter: TradingAdapterKind) -> AdapterReadinessSnapshot {
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
                    ("domain", &self.dxtrade.domain),
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
                "domain".to_string(),
                "password".to_string()
            ]
        );
        assert!(!missing.can_attempt_connect);

        settings.dxtrade.platform_url = "https://broker.example/specs".to_string();
        settings.dxtrade.username = "ops-user".to_string();
        settings.dxtrade.domain = "default".to_string();
        settings.dxtrade.password = "ops-pass".to_string();

        let ready = settings.readiness(TradingAdapterKind::DxTrade);
        assert_eq!(ready.session_state, BrokerSessionState::ReadyForAuth);
        assert!(ready.missing_fields.is_empty());
        assert!(ready.can_attempt_connect);
    }
}
