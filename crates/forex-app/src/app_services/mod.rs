pub mod broker_config;
pub mod ctrader_auth;
pub mod discovery;
pub mod jobs;
pub mod trading;
pub mod training;

use crate::app_services::jobs::JobSnapshot;

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
}
