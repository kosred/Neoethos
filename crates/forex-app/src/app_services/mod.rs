pub mod broker_config;
pub mod ctrader_account;
pub mod ctrader_messages;
pub mod ctrader_auth;
pub mod ctrader_data;
pub mod ctrader_live_auth;
pub mod ctrader_streaming;
pub mod discovery;
pub mod jobs;
pub mod secure_store;
pub mod trading;
pub mod training;

use crate::app_services::jobs::JobSnapshot;

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
}
