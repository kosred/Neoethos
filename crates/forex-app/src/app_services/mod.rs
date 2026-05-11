pub mod bootstrap_writer;
pub mod broker_config;
pub mod broker_persistence;
pub mod ctrader_account;
pub mod ctrader_auth;
pub mod ctrader_bootstrap;
pub mod ctrader_data;
pub mod ctrader_execution;
#[cfg(test)]
mod ctrader_integration_tests;
pub mod ctrader_live_auth;
pub mod ctrader_messages;
pub mod ctrader_openapi;
pub mod ctrader_proto_messages;
pub mod ctrader_session;
pub mod ctrader_streaming;
pub mod discovery;
pub mod embedded_credentials;
pub mod jobs;
pub mod live_journal;
pub mod secure_store;
pub mod trading;
pub mod training;

use crate::app_services::jobs::JobSnapshot;

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
    LlmNewsUpdated(String),
    Heartbeat,
    CTraderConnectUpdated(crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot),
    BootstrapUpdated(JobSnapshot),
    ConnectOutcome(Result<String, String>),
}
