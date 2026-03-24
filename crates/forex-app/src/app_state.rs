use crate::app_services::jobs::JobSnapshot;
use forex_core::{logging::canonical_log_path, Settings};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppRuntimeConfig {
    pub config_path: String,
    pub data_dir: PathBuf,
    pub start_local: bool,
}

impl AppRuntimeConfig {
    pub fn from_settings(config_path: String, start_local: bool, settings: &Settings) -> Self {
        Self {
            config_path,
            data_dir: settings.system.data_dir.clone(),
            start_local,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    MT5,
    Local,
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub runtime: AppRuntimeConfig,
    pub data_source: DataSource,
    pub status_msg: String,
    pub selected_pair: String,
    pub chart_timeframe: String,
    pub available_symbols: Vec<String>,
    pub discovery_job: Option<JobSnapshot>,
    pub training_job: Option<JobSnapshot>,
    pub bootstrap_form: BootstrapFormState,
    pub bootstrap_job: Option<JobSnapshot>,
    pub canonical_log_path: PathBuf,
    pub hardware: HardwareState,
    pub risk: RiskState,
}

impl AppState {
    pub fn new(runtime: AppRuntimeConfig, available_symbols: Vec<String>) -> Self {
        let selected_pair = available_symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());

        Self {
            data_source: if runtime.start_local {
                DataSource::Local
            } else {
                DataSource::MT5
            },
            status_msg: if runtime.start_local {
                "Local Mode".to_string()
            } else {
                "Offline".to_string()
            },
            canonical_log_path: canonical_log_path(),
            runtime,
            selected_pair: selected_pair.clone(),
            chart_timeframe: "M1".to_string(),
            available_symbols,
            discovery_job: None,
            training_job: None,
            bootstrap_form: BootstrapFormState::default_for_symbol(&selected_pair),
            bootstrap_job: None,
            hardware: HardwareState::default(),
            risk: RiskState::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapFormState {
    pub pairs_input: String,
    pub timeframes_input: String,
    pub years: u32,
}

impl BootstrapFormState {
    pub fn default_for_symbol(symbol: &str) -> Self {
        Self {
            pairs_input: symbol.to_string(),
            timeframes_input: "M1,M5,M15,H1".to_string(),
            years: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareState {
    pub cpu_cores: i32,
    pub gpu_enabled: bool,
}

impl Default for HardwareState {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get() as i32,
            gpu_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiskState {
    pub daily_drawdown_limit: f32,
    pub max_lot_size: f32,
}

impl Default for RiskState {
    fn default() -> Self {
        Self {
            daily_drawdown_limit: 4.5,
            max_lot_size: 10.0,
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

        let runtime = AppRuntimeConfig::from_settings("config.yaml".to_string(), true, &settings);

        assert_eq!(runtime.data_dir, PathBuf::from("custom-data-root"));
        assert!(runtime.start_local);
    }

    #[test]
    fn app_state_uses_first_symbol_and_keeps_job_slots_empty() {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: true,
        };

        let state = AppState::new(runtime, vec!["GBPUSD".to_string(), "EURUSD".to_string()]);

        assert_eq!(state.selected_pair, "GBPUSD");
        assert_eq!(state.chart_timeframe, "M1");
        assert!(state.discovery_job.is_none());
        assert!(state.training_job.is_none());
        assert!(state.bootstrap_job.is_none());
        assert_eq!(state.bootstrap_form.pairs_input, "GBPUSD");
        assert_eq!(state.bootstrap_form.timeframes_input, "M1,M5,M15,H1");
        assert_eq!(state.bootstrap_form.years, 1);
        assert_eq!(state.canonical_log_path, PathBuf::from("logs").join("forex-ai.log"));
        assert_eq!(state.hardware.cpu_cores, num_cpus::get() as i32);
        assert!(state.hardware.gpu_enabled);
        assert_eq!(state.risk.daily_drawdown_limit, 4.5);
        assert_eq!(state.risk.max_lot_size, 10.0);
    }

    #[test]
    fn app_state_falls_back_to_eurusd_when_symbol_list_is_empty() {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: false,
        };

        let state = AppState::new(runtime, Vec::new());

        assert_eq!(state.selected_pair, "EURUSD");
        assert_eq!(state.chart_timeframe, "M1");
        assert_eq!(state.status_msg, "Offline");
        assert_eq!(state.bootstrap_form.pairs_input, "EURUSD");
    }

    #[test]
    fn hardware_state_defaults_match_host_core_count_and_gpu_enabled() {
        let state = HardwareState::default();

        assert_eq!(state.cpu_cores, num_cpus::get() as i32);
        assert!(state.gpu_enabled);
    }

    #[test]
    fn risk_state_defaults_match_existing_guard_values() {
        let state = RiskState::default();

        assert_eq!(state.daily_drawdown_limit, 4.5);
        assert_eq!(state.max_lot_size, 10.0);
    }

    #[test]
    fn bootstrap_form_defaults_can_be_built_for_a_symbol() {
        let state = BootstrapFormState::default_for_symbol("AUDUSD");

        assert_eq!(state.pairs_input, "AUDUSD");
        assert_eq!(state.timeframes_input, "M1,M5,M15,H1");
        assert_eq!(state.years, 1);
    }
}
