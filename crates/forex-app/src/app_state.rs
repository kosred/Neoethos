use crate::app_services::jobs::JobSnapshot;
use forex_core::{Settings, logging::canonical_log_path};
use std::path::PathBuf;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    CTrader,
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
    pub order_ticket: OrderTicketState,
    pub canonical_log_path: PathBuf,
    pub hardware: HardwareState,
    pub risk: forex_core::config::RiskConfig,
    pub dashboard_panel: crate::ui::dashboard::DashboardPanel,
    pub ai_insights_panel: crate::ui::ai_insights::AiInsightsPanel,
    /// v0.4.8 — chat scrollback + input for the AI Helper panel. Lives
    /// in `AppState` so the conversation survives tab switches and is
    /// re-rendered on every frame without losing the operator's
    /// in-progress prompt. The chat is in-memory only by design — no
    /// persistence to disk until the audit log (G7 / disk-backed
    /// JsonlAuditLog) is wired in.
    pub ai_helper_panel: crate::ui::ai_helper::AiHelperState,
    pub llm_news_filter: forex_core::domain::news_filter::NewsFilter,
    pub discovery_form: DiscoveryFormState,
    pub auto_trade_enabled: bool,

    // Account Real-time Data
    pub account_balance: f64,
    pub account_equity: f64,
}

impl AppState {
    pub fn new(
        runtime: AppRuntimeConfig,
        settings: &Settings,
        available_symbols: Vec<String>,
    ) -> Self {
        let selected_pair = available_symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());
        let mut llm_news_filter = forex_core::domain::news_filter::NewsFilter::new(
            settings.news.openai_news_enabled,
            settings.news.news_lookahead_minutes as i64,
            settings.news.news_kill_window_min as i64,
        );
        llm_news_filter.llm_provider = if settings.news.openai_news_enabled {
            "openai".to_string()
        } else {
            "perplexity".to_string()
        };

        Self {
            data_source: if runtime.start_local {
                DataSource::Local
            } else {
                DataSource::CTrader
            },
            status_msg: if runtime.start_local {
                "Local Mode".to_string()
            } else {
                "cTrader Ready".to_string()
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
            order_ticket: OrderTicketState::default(),
            hardware: HardwareState::default(),
            risk: settings.risk.clone(),
            dashboard_panel: crate::ui::dashboard::DashboardPanel::new(),
            ai_insights_panel: crate::ui::ai_insights::AiInsightsPanel::new(),
            ai_helper_panel: crate::ui::ai_helper::AiHelperState::new(),
            llm_news_filter,
            discovery_form: DiscoveryFormState::from_settings(settings),
            auto_trade_enabled: false,
            account_balance: 0.0,
            account_equity: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveryFormState {
    pub base_tf: String,
    pub higher_tfs: String,
    pub max_indicators: u32,
    pub population: u32,
    pub generations: u32,
    pub target_candidates: u32,
    pub portfolio_size: u32,
    pub correlation_threshold: f32,
    pub min_trades_per_day: f32,
}

impl Default for DiscoveryFormState {
    fn default() -> Self {
        Self {
            base_tf: "M1".to_string(),
            higher_tfs: "M5, M15, H1".to_string(),
            max_indicators: 12,
            population: 100,
            generations: 5,
            target_candidates: 200,
            portfolio_size: 100,
            correlation_threshold: 0.7,
            min_trades_per_day: 0.5, // Relaxed default
        }
    }
}

impl DiscoveryFormState {
    pub fn from_settings(settings: &Settings) -> Self {
        let discovery = forex_search::DiscoveryConfig::from_settings(settings);
        let higher_tfs = settings.system.higher_timeframes.join(", ");

        Self {
            base_tf: settings.system.base_timeframe.clone(),
            higher_tfs: if higher_tfs.trim().is_empty() {
                "M5, M15, H1".to_string()
            } else {
                higher_tfs
            },
            max_indicators: discovery.max_indicators as u32,
            population: discovery.population as u32,
            generations: discovery.generations as u32,
            target_candidates: discovery.candidate_count as u32,
            portfolio_size: discovery.portfolio_size as u32,
            correlation_threshold: discovery.corr_threshold as f32,
            min_trades_per_day: discovery.min_trades_per_day as f32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapFormState {
    pub pairs_input: String,
    pub timeframes_input: String,
    pub years: u32,
    /// Task #9 — additional source folders the operator wants to
    /// scan and import alongside the primary `state.runtime.data_dir`.
    /// Each path is run through `forex_data::DatasetDiscovery::scan`
    /// independently so the format auto-detection works per-source
    /// (an MT4 export folder + a Spotware Parquet dump + a CSV
    /// archive can all live in the import set simultaneously).
    pub external_sources: Vec<std::path::PathBuf>,
}

impl BootstrapFormState {
    pub fn default_for_symbol(symbol: &str) -> Self {
        Self {
            pairs_input: symbol.to_string(),
            timeframes_input: "M1,M5,M15,H1".to_string(),
            years: 1,
            external_sources: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderTicketState {
    pub order_type: OrderType,
    pub target_price: f64,
    pub auto_lot_sizing: bool,
    pub auto_risk_pct: f64,
    pub stop_loss_pips: f64,
    pub lot_size: f64,
    pub slippage_in_points: i32,
    pub smart_sl_enabled: bool,
    pub smart_rr_ratio: f64,
    pub trailing_stop: bool,
    pub comment: String,
    pub label: String,
    pub selected_position_id: Option<i64>,
    pub selected_order_id: Option<i64>,
}

impl Default for OrderTicketState {
    fn default() -> Self {
        Self {
            order_type: OrderType::Market,
            target_price: 0.0,
            auto_lot_sizing: false,
            auto_risk_pct: 1.0,
            stop_loss_pips: 20.0,
            lot_size: 0.10,
            slippage_in_points: 10,
            smart_sl_enabled: true,
            smart_rr_ratio: 2.0,
            trailing_stop: false,
            comment: String::new(),
            label: "manual".to_string(),
            selected_position_id: None,
            selected_order_id: None,
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

    #[test]
    fn app_state_uses_first_symbol_and_keeps_job_slots_empty() {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: true,
            auto_discovery: false,
            auto_training: false,
        };

        let state = AppState::new(
            runtime,
            &forex_core::Settings::default(),
            vec!["GBPUSD".to_string(), "EURUSD".to_string()],
        );

        assert_eq!(state.selected_pair, "GBPUSD");
        assert_eq!(state.chart_timeframe, "M1");
        assert!(state.discovery_job.is_none());
        assert!(state.training_job.is_none());
        assert!(state.bootstrap_job.is_none());
        assert_eq!(state.bootstrap_form.pairs_input, "GBPUSD");
        assert_eq!(state.bootstrap_form.timeframes_input, "M1,M5,M15,H1");
        assert_eq!(state.bootstrap_form.years, 1);
        assert_eq!(state.order_ticket.order_type, OrderType::Market);
        assert_eq!(state.order_ticket.lot_size, 0.10);
        assert!(!state.auto_trade_enabled);
        assert_eq!(state.order_ticket.slippage_in_points, 10);
        assert_eq!(state.order_ticket.label, "manual");
        assert!(state.order_ticket.selected_position_id.is_none());
        assert!(state.order_ticket.selected_order_id.is_none());
        assert_eq!(
            state.canonical_log_path,
            PathBuf::from("logs").join("forex-ai.log")
        );
        assert_eq!(state.hardware.cpu_cores, num_cpus::get() as i32);
        assert!(state.hardware.gpu_enabled);
        assert_eq!(state.risk.daily_drawdown_limit, 0.04);
        assert_eq!(state.risk.total_drawdown_limit, 0.07);
        assert_eq!(state.risk.max_lot_size, 10.0);
        assert_eq!(state.risk.risk_per_trade, 0.03);
        assert!(state.risk.require_stop_loss);
    }

    #[test]
    fn app_state_falls_back_to_eurusd_when_symbol_list_is_empty() {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: false,
            auto_discovery: false,
            auto_training: false,
        };

        let state = AppState::new(runtime, &forex_core::Settings::default(), Vec::new());

        assert_eq!(state.selected_pair, "EURUSD");
        assert_eq!(state.chart_timeframe, "M1");
        assert_eq!(state.status_msg, "cTrader Ready");
        assert_eq!(state.bootstrap_form.pairs_input, "EURUSD");
    }

    #[test]
    fn hardware_state_defaults_match_host_core_count_and_gpu_enabled() {
        let state = HardwareState::default();

        assert_eq!(state.cpu_cores, num_cpus::get() as i32);
        assert!(state.gpu_enabled);
    }

    #[test]
    fn bootstrap_form_defaults_can_be_built_for_a_symbol() {
        let state = BootstrapFormState::default_for_symbol("AUDUSD");

        assert_eq!(state.pairs_input, "AUDUSD");
        assert_eq!(state.timeframes_input, "M1,M5,M15,H1");
        assert_eq!(state.years, 1);
    }

    #[test]
    fn order_ticket_defaults_are_operator_friendly() {
        let state = OrderTicketState::default();

        assert_eq!(state.order_type, OrderType::Market);
        assert_eq!(state.lot_size, 0.10);
        assert_eq!(state.slippage_in_points, 10);
        assert_eq!(state.label, "manual");
        assert!(state.comment.is_empty());
        assert!(state.selected_position_id.is_none());
        assert!(state.selected_order_id.is_none());
    }
}
