//! App-side re-exports of the canonical broker-credentials data types.
//! The schema and persistence implementation live in `neoethos-core` so the
//! app and CLI read and write the same `broker_credentials.toml` contract.
//!
//! Re-exports from `neoethos-core` keep import sites in the rest of
//! `neoethos-app` unchanged after the Phase B SoT migration. Existing
//! `use crate::app_services::broker_config::{BrokerSettingsState, ...}`
//! lines still work; new code in `neoethos-cli` imports the same
//! names directly from `neoethos_core::broker_config`.

pub use neoethos_core::broker_config::{
    BROKER_CREDENTIALS_SCHEMA_VERSION, BrokerAccountTarget, BrokerSettingsState,
    CTRADER_OAUTH_REDIRECT_URI, CTraderBrokerEnvironment, CTraderBrokerSettings,
};
// `CTRADER_CREATE_DEMO_ACCOUNT_URL` / `CTRADER_CREATE_LIVE_ACCOUNT_URL`
// constants live in `neoethos-core::broker_config` and were
// previously re-exported here for the legacy egui "create demo
// account" buttons (removed in #89). When the Flutter shell wants
// to surface those links it can import them directly from
// neoethos-core — no need for a pass-through re-export.
