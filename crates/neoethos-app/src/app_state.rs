pub use crate::app_services::execution_admission::platform_import_auxiliary_slot_limit;
use crate::app_services::execution_admission::{
    AdmissionError, ExecutionAdmissionClient, ExecutionAdmissionCoordinator,
    ExecutionAdmissionSnapshot,
};
use neoethos_core::Settings;
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{AuxiliarySlotLimit, CpuPermitBroker, WorkerLimit};
use std::path::PathBuf;

/// Process-lifetime owner of async CPU admission and the lease-bound Rayon
/// executor. Dropping this state stops and joins the coordinator thread.
pub struct AppExecutionState {
    coordinator: Option<ExecutionAdmissionCoordinator>,
    executor: BudgetedCpuExecutor,
}

impl AppExecutionState {
    pub fn new(
        broker: CpuPermitBroker,
        max_cached_worker_threads: WorkerLimit,
    ) -> std::io::Result<Self> {
        Self::new_with_auxiliary_slots(
            broker,
            max_cached_worker_threads,
            platform_import_auxiliary_slot_limit(),
        )
    }

    pub fn new_with_auxiliary_slots(
        broker: CpuPermitBroker,
        max_cached_worker_threads: WorkerLimit,
        auxiliary_slots: AuxiliarySlotLimit,
    ) -> std::io::Result<Self> {
        let executor =
            BudgetedCpuExecutor::new_for_broker(broker.clone(), max_cached_worker_threads);
        Ok(Self {
            coordinator: Some(ExecutionAdmissionCoordinator::start_with_auxiliary_slots(
                broker,
                auxiliary_slots,
            )?),
            executor,
        })
    }

    pub fn admission_client(&self) -> ExecutionAdmissionClient {
        self.coordinator
            .as_ref()
            .expect("app execution state owns its coordinator until shutdown")
            .client()
    }

    pub fn executor(&self) -> &BudgetedCpuExecutor {
        &self.executor
    }

    pub fn admission_snapshot(&self) -> ExecutionAdmissionSnapshot {
        self.coordinator
            .as_ref()
            .expect("app execution state owns its coordinator until shutdown")
            .admission_snapshot()
    }

    pub fn shutdown(mut self) -> Result<(), AdmissionError> {
        self.coordinator
            .take()
            .expect("app execution state coordinator shuts down only once")
            .shutdown()
    }
}

impl std::fmt::Debug for AppExecutionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AppExecutionState")
            .field("coordinator", &self.coordinator)
            .field("executor", &self.executor)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct AppRuntimeConfig {
    pub config_path: String,
    pub data_dir: PathBuf,
    pub start_local: bool,
    /// Auto-start discovery on headless launch (VPS/WSL2 use-case).
    /// The UI start/stop controls are one of several interfaces to this subsystem.
    pub auto_discovery: bool,
    /// Auto-start training on headless launch (VPS/WSL2 use-case).
    pub auto_training: bool,
}

impl AppRuntimeConfig {
    pub fn from_settings(
        config_path: String,
        start_local: bool,
        auto_discovery: bool,
        auto_training: bool,
        settings: &Settings,
    ) -> Self {
        Self {
            config_path,
            data_dir: settings.system.data_dir.clone(),
            start_local,
            auto_discovery,
            auto_training,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn app_runtime_config_uses_settings_data_dir() {
        let mut settings = Settings::default();
        settings.system.data_dir = PathBuf::from("custom-data-root");

        let runtime = AppRuntimeConfig::from_settings(
            "config.yaml".to_string(),
            true,
            false,
            false,
            &settings,
        );

        assert_eq!(runtime.data_dir, PathBuf::from("custom-data-root"));
        assert!(runtime.start_local);
        assert!(!runtime.auto_discovery);
        assert!(!runtime.auto_training);
    }
}
