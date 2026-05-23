// In Windows release builds, link as a GUI subsystem app so the
// backend process spawned by the Flutter shell does not pop up a
// black console window behind the UI. In debug builds we keep the
// console so `cargo run -- --server` still shows tracing output
// on stdout. Linux/macOS ignore the attribute.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app_services;
mod app_state;
mod server;

use app_state::AppRuntimeConfig;
use clap::Parser;
use neoethos_core::Settings;
use neoethos_core::logging::{
    setup_logging, show_double_click_help_dialog_if_orphaned, write_subsystem_record,
};
use neoethos_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    headless: bool,

    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long, default_value_t = false)]
    local: bool,

    /// Auto-start discovery on launch (headless VPS/WSL2 use-case).
    #[arg(long, default_value_t = false)]
    auto_discovery: bool,

    /// Auto-start training on launch (headless VPS/WSL2 use-case).
    #[arg(long, default_value_t = false)]
    auto_training: bool,

    /// Run the live cTrader API test harness.
    #[arg(long, default_value_t = false)]
    api_test: bool,

    /// Output path for the api-test JSON report.
    #[arg(long, default_value = "api-test-report.json")]
    api_test_output: String,

    /// Add a 1-second pause between api-test flows.
    #[arg(long, default_value_t = false)]
    api_test_slow: bool,

    /// Restrict the api-test run to flows whose `name` matches this glob.
    #[arg(long)]
    api_test_only: Option<String>,

    /// Run as a headless HTTP API server on port 7423.
    ///
    /// This is the default behavior as of v0.4.20. Passing `--server`
    /// explicitly is equivalent to passing nothing at all; the flag is
    /// preserved for older scripts and the Flutter `BackendSupervisor`.
    #[arg(long, default_value_t = false)]
    server: bool,

    /// Headless cTrader OAuth flow.
    ///
    /// Opens the system browser to the Spotware consent page, captures the
    /// redirect, exchanges the authorization code for a fresh token bundle,
    /// and writes it to the OS keyring. Run this when `--server` logs
    /// `RET_ACCOUNT_DISABLED` or "Authentication failed".
    #[arg(long, default_value_t = false)]
    reauth: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    setup_logging(true)?;
    neoethos_search::install_search_runtime_overrides_from_env();
    let settings = Settings::from_yaml(&args.config)?;
    let runtime = AppRuntimeConfig::from_settings(
        args.config.clone(),
        args.local,
        args.auto_discovery,
        args.auto_training,
        &settings,
    );
    write_subsystem_record(
        SubsystemSection::App,
        app_record(
            "app_startup",
            "STARTED",
            format!(
                "starting neoethos headless={} local={} config={}",
                args.headless, args.local, args.config
            ),
        ),
    )?;

    if args.api_test {
        info!("Starting neoethos-app in API-TEST mode (cTrader demo)...");
        let cfg = app_services::api_test::ApiTestConfig {
            environment: app_services::api_test::ApiTestEnvironment::Demo,
            output_path: std::path::PathBuf::from(&args.api_test_output),
            slow: args.api_test_slow,
            only_filter: args.api_test_only.clone(),
        };
        app_services::api_test::run_api_test_suite(cfg).await?;
        return Ok(());
    }

    if args.reauth {
        info!("Running cTrader OAuth flow (browser will open)...");
        let outcome = tokio::task::spawn_blocking(app_services::reauth::run_reauth_flow_blocking)
            .await
            .map_err(|e| anyhow::anyhow!("OAuth blocking task panicked: {e}"))??;

        info!(
            callback_port = outcome.callback_port,
            access_token_len = outcome.access_token_len,
            refresh_token_present = outcome.refresh_token_present,
            "OAuth flow complete; token bundle saved to keyring"
        );
        info!("Token bundle saved. You can now run: neoethos-app --server");
        return Ok(());
    }

    if args.headless {
        info!("Starting neoethos in headless server mode...");
        run_headless_loop(runtime).await;
        return Ok(());
    }

    let _ = args.server; // keep the flag valid for back-compat.
    info!("Starting neoethos in HTTP server mode (Flutter front-end backend)...");

    // Pop the help dialog BEFORE we await on serve(). The function is
    // a no-op on non-Windows, in debug builds, and when the Flutter
    // shell set NEOETHOS_LAUNCHED_BY_FLUTTER=1. In the orphaned-
    // double-click path it blocks the main thread until the user
    // clicks OK — which is fine; the dialog is the entire UI surface
    // for that user right now.
    show_double_click_help_dialog_if_orphaned("http://127.0.0.1:7423");

    let state = server::state::AppApiState::new();
    server::bridge::spawn(state.clone());

    // Hydrate the in-memory signal journal (#127, #131) from disk
    // so explain_recent_trades can narrate signals that fired in
    // earlier sessions. Logs the row count at info; first-launch
    // is a clean zero.
    let restored = app_services::signal_journal::restore_from_disk();
    info!(
        target: "neoethos_app::main",
        restored,
        "signal journal hydrated"
    );

    // Spawn the Gemma news watcher (#128). The task is a no-op
    // when `news.gemma_news_watcher_enabled = false` — the loop
    // exits immediately. Behind the gemma-backend feature so the
    // default build doesn't pull in the watcher code path.
    #[cfg(feature = "gemma-backend")]
    {
        let watcher_config =
            app_services::gemma_news_watcher::WatcherConfig::from_news_config(&settings.news);
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _watcher_handle = app_services::gemma_news_watcher::spawn(
            state.clone(),
            watcher_config,
            cancel,
        );
        // We deliberately drop the handle — the watcher should live
        // for the entire process lifetime. The serve() call below
        // never returns in the happy path; on a signal-shutdown the
        // task is dropped along with everything else.
    }

    server::serve(state).await?;
    Ok(())
}

async fn run_headless_loop(runtime: AppRuntimeConfig) {
    use app_services::{
        discovery::{DiscoveryRequest, start_discovery_job},
        training::{TrainingRequest, start_training_job},
    };
    use std::path::PathBuf;

    info!("Loading configuration from: {}", runtime.config_path);

    let symbols = match neoethos_data::discover_symbols(&runtime.data_dir) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::main",
                data_dir = %runtime.data_dir.display(),
                error = %err,
                "headless: discover_symbols failed; continuing with empty symbol list"
            );
            Vec::new()
        }
    };
    info!(
        "Headless: mapped {} local symbols in '{}'",
        symbols.len(),
        runtime.data_dir.display()
    );

    let (tx, _rx) = mpsc::channel(1000);

    if runtime.auto_discovery {
        let symbol = symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());
        info!("Headless: auto-starting discovery for {}", symbol);
        let request = DiscoveryRequest {
            data_root: runtime.data_dir.clone(),
            symbol,
            base_tf: "M1".to_string(),
            higher_tfs: vec!["M5".to_string(), "M15".to_string(), "H1".to_string()],
            config: neoethos_search::DiscoveryConfig::default(),
            prop_firm_rules: neoethos_search::PropFirmRiskRules::default(),
        };
        match start_discovery_job(request, tx.clone()) {
            Ok(_handle) => info!("Headless: discovery job started"),
            Err(err) => error!("Headless: failed to start discovery: {}", err),
        }
    }

    if runtime.auto_training {
        let symbol = symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());
        info!("Headless: auto-starting training for {}", symbol);
        let request = TrainingRequest {
            config_path: runtime.config_path.clone(),
            models_dir: PathBuf::from("models"),
            symbol,
            base_tf: "M1".to_string(),
        };
        match start_training_job(request, tx.clone()) {
            Ok(_handle) => info!("Headless: training job started"),
            Err(err) => error!("Headless: failed to start training: {}", err),
        }
    }

    let mode = if runtime.start_local {
        "LOCAL"
    } else {
        "CTRADER"
    };
    if let Err(err) = write_subsystem_record(
        SubsystemSection::App,
        app_record(
            "headless_start",
            "READY",
            format!(
                "mode={} auto_discovery={} auto_training={}",
                mode, runtime.auto_discovery, runtime.auto_training
            ),
        ),
    ) {
        error!("Failed to write APP section log: {}", err);
    }

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
    loop {
        interval.tick().await;
        info!(
            "Headless keep-alive: Cores={} Mode={} Discovery={} Training={}",
            num_cpus::get(),
            mode,
            runtime.auto_discovery,
            runtime.auto_training,
        );
    }
}

pub(crate) fn app_record(
    operation: &str,
    status: &str,
    message: impl Into<String>,
) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!("app-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::App,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message: message.into(),
        body: String::new(),
    }
}

fn system_time_string() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => format!("unix:{}", d.as_secs()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::main",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "unix:pre-1970".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AppRuntimeConfig, app_record};
    use crate::app_state::DataSource;
    use neoethos_core::Settings;
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
    fn app_record_targets_app_section() {
        let record = app_record("headless_start", "STARTED", "headless startup");

        assert_eq!(
            record.subsystem,
            neoethos_core::sectioned_log::SubsystemSection::App
        );
        assert_eq!(record.operation, "headless_start");
        assert_eq!(record.status, "STARTED");
        assert_eq!(record.message, "headless startup");
    }

    #[test]
    fn trading_panel_mode_disables_live_controls_in_local_mode() {
        let mode = crate::app_services::trading::panel_mode(DataSource::Local, false);

        assert_eq!(
            mode,
            crate::app_services::trading::TradingPanelMode::LocalOnly
        );
    }

    #[test]
    fn trading_panel_mode_switches_to_connected_when_ctrader_is_connected() {
        let mode = crate::app_services::trading::panel_mode(DataSource::CTrader, true);

        assert_eq!(
            mode,
            crate::app_services::trading::TradingPanelMode::Connected
        );
    }
}
