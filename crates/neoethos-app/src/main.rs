// In Windows release builds, link as a GUI subsystem app so the
// backend process spawned by the Flutter shell does not pop up a
// black console window behind the UI. In debug builds we keep the
// console so `cargo run -- --server` still shows tracing output
// on stdout. Linux/macOS ignore the attribute.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

// Modules now live in the library crate (src/lib.rs) so the Tauri desktop
// shell can reuse them in-process. The bin consumes them via `neoethos_app::`.
use neoethos_app::{app_services, app_state, server};

use app_state::AppRuntimeConfig;
use clap::Parser;
use neoethos_core::Settings;
use neoethos_core::execution_budget::{
    StartupEvent, StartupRuntimeKind, StartupTrace, format_startup_diagnostics,
    parse_parent_cpu_assignment, startup_diagnostics_requested,
};
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

    /// Multi-timeframe Discovery sweep on a single symbol; writes
    /// per-TF outcomes to `validation-runs/<ts>/sweep.csv` and exits.
    ///
    /// Mutually exclusive with `--auto-discovery` / `--auto-training`.
    /// Honors `--validation-tfs` (default `M5,M15,M30,H1,H4,D1`) and
    /// `--validation-tf-timeout-secs` (default 1800). Driven by
    /// `config.yaml` via `DiscoveryConfig::from_settings` so the
    /// operator's population/generation counts apply.
    #[arg(long, default_value_t = false)]
    validation_mode: bool,

    /// Comma-separated timeframes to sweep when `--validation-mode` is set.
    /// M1 is intentionally omitted from the default — it's noise-dominated
    /// and pure cost for the GA search.
    #[arg(long, default_value = "M5,M15,M30,H1,H4,D1")]
    validation_tfs: String,

    /// Per-TF hard timeout in seconds for `--validation-mode`. A TF that
    /// blows the cap is recorded as `Timeout` in the CSV and the sweep
    /// continues with the next TF.
    #[arg(long, default_value_t = 1800)]
    validation_tf_timeout_secs: u64,

    /// Floor on the GA generation count for each `--validation-mode` TF.
    /// Overrides `DiscoveryConfig.generations` from `config.yaml` when the
    /// configured value is lower. This exists because operator configs
    /// often set 20 generations as a "fast smoke test" value — but a
    /// validation sweep needs every TF to actually exercise the search,
    /// otherwise short-data TFs (D1/H4) finish in <1s with a tiny archive
    /// (#215). Set to 0 to honor whatever `config.yaml` says.
    #[arg(long, default_value_t = 20)]
    validation_min_generations: usize,

    // REMOVED 2026-08-09 (dead-code purge, batch D2): `--launched-by-flutter`.
    // Nothing in the workspace ever passed it — grep for `launched-by-flutter`
    // across crates/, desktop/, mesh/, mcp/, scripts/ and .github/ found only
    // doc comments. Flutter died 2026-06-22. Because the flag defaulted to
    // false and no spawner set it, removing it does NOT change behaviour: the
    // orphan help dialog was already unconditional on this path.
    /// PID of the parent GUI process. When set, the backend self-terminates
    /// within ~2s of that process exiting, so it never lingers as a windowless
    /// orphan holding port 7423 (the "app won't open" bug: the next launch
    /// sees a live backend and silently exits). Still honoured — any
    /// supervisor that spawns this binary detached should pass it.
    #[arg(long)]
    parent_pid: Option<u32>,

    /// **Phase C (2026-05-28)** — capture the real `ProtoOASymbolByIdRes`
    /// payload for each comma-separated symbol from the configured
    /// cTrader account, then exit. Each capture dumps a `.raw.json`
    /// (verbatim broker bytes) and a `.decoded.json` (the parser's
    /// projection) under `--capture-output` (default
    /// `crates/neoethos-app/tests/fixtures`).
    ///
    /// Used to verify Phase A.1 `SymbolFinancials` schema assumptions
    /// against real bytes instead of proto comments. Mutually
    /// exclusive with all other modes; exits 0 on success.
    ///
    /// Example:
    ///   neoethos-app --capture-symbols EURUSD,USDJPY,XAUUSD,BTCUSD
    #[arg(long)]
    capture_symbols: Option<String>,

    /// Output directory for `--capture-symbols`. Created if missing.
    /// Default: `crates/neoethos-app/tests/fixtures` (relative to
    /// the current working directory, so run from the repo root).
    #[arg(long)]
    capture_output: Option<String>,

    /// **Phase D.1 (2026-05-28)** — fetch the entire broker symbol
    /// catalog (typically 800-900 entries) plus full
    /// `ProtoOASymbolByIdRes` payloads, batched 50 symbols per
    /// request. Writes verbatim broker envelopes + a symbol index
    /// + bootstrap metadata under
    /// `data/broker_symbols/<env>/` (overridable via
    /// `--bootstrap-output`). One-shot, ~30-90 s for 830 symbols.
    ///
    /// Why: the GA cost model has been operating on synthetic
    /// $7/lot commission + 1.5-pip spread fallbacks because no code
    /// path ever populated `SymbolMetadata.commission_per_lot` from
    /// the broker. This bootstrap is the foundation for Phase D.2
    /// (delete the synthetic fallbacks; metadata reads from this
    /// catalog).
    #[arg(long, default_value_t = false)]
    bootstrap_broker_catalog: bool,

    /// Output directory for `--bootstrap-broker-catalog`. Created
    /// if missing. Default: `data/broker_symbols` (relative to the
    /// current working directory).
    #[arg(long)]
    bootstrap_output: Option<String>,

    /// **Phase D.2d (2026-05-28)** — re-run ONLY the catalog →
    /// `SymbolMetadataTable` conversion step against the on-disk
    /// `raw_batches/` cache previously written by
    /// `--bootstrap-broker-catalog`, without touching the broker.
    ///
    /// Use this after a schema change to `SymbolMetadata` (e.g. new
    /// fields added) to regenerate `data/symbol_metadata.json` with
    /// the new shape, while preserving the operator's broker capture
    /// (so we don't burn quota re-fetching 830 symbols).
    ///
    /// Reads `<bootstrap-output>/<env>/raw_batches/*.json`
    /// + `light_symbols.json` + `asset_list.json`. Writes
    /// `data/symbol_metadata.json`. No network I/O. ~1 s.
    #[arg(long, default_value_t = false)]
    rebuild_symbol_metadata: bool,

    /// Internal parent-to-child CPU assignment. The value only narrows the
    /// automatic effective-logical-threads-minus-two process budget.
    #[arg(long, value_name = "POSITIVE_INTEGER")]
    cpu_threads: Option<String>,

    /// Resolve and install startup policy, build the managed Tokio runtime,
    /// print one structured diagnostic line to stderr, then exit.
    #[arg(long, default_value_t = false)]
    startup_diagnostics: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut startup_trace = StartupTrace::default();
    // Linux signal masks are per-thread. SourceSeal must reserve/block its
    // SIGIO + real-time signal set on the initial thread before Tokio, Tauri,
    // logging, or any dependency can create another thread.
    neoethos_data::initialize_source_seal_before_runtime()?;
    startup_trace.record(StartupEvent::ImportSignalPreflightCompleted)?;
    let raw_args: Vec<String> = std::env::args().collect();
    let args = Args::parse_from(&raw_args);
    // #101 follow-up + #179: the help dialog must fire BEFORE the
    // config-load step below, otherwise an orphaned double-click whose
    // CWD lacks `config.yaml` exits silently with `windows_subsystem =
    // "windows"`. Skipped in debug builds / non-Windows (the helper
    // handles both internally).
    //
    // 2026-08-09 (dead-code purge D2): this used to be gated on
    // `!args.launched_by_flutter`, a CLI flag no spawner ever passed. Both
    // that flag and the older `NEOETHOS_LAUNCHED_BY_FLUTTER` env fallback are
    // gone. The call is now unconditional, which is exactly what it already
    // was at runtime — nothing set either signal after Flutter died 2026-06-22.
    show_double_click_help_dialog_if_orphaned("http://127.0.0.1:7423");

    let settings = Settings::from_yaml(&args.config)?;
    startup_trace.record(StartupEvent::ConfigurationLoaded)?;

    let parent_cpu_assignment = parse_parent_cpu_assignment(&raw_args)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    startup_trace.record(StartupEvent::ParentCpuCapParsed)?;
    let coordination_scope = if parent_cpu_assignment.is_some() {
        neoethos_core::execution_budget::CoordinationScope::ManagedProcessTree
    } else {
        neoethos_core::execution_budget::CoordinationScope::ProcessLocal
    };
    let budget_inputs = neoethos_core::ExecutionBudgetInputs::from_settings_and_parent(
        &settings,
        parent_cpu_assignment.map(|limit| limit.get()),
        coordination_scope,
    )?;
    budget_inputs.clone().resolve()?;
    startup_trace.record(StartupEvent::CpuBudgetResolved)?;
    let installed =
        neoethos_core::execution_budget::install_process_budget(budget_inputs.request().clone())?;
    startup_trace.record(StartupEvent::CpuBudgetInstalled)?;

    // Logging and environment diagnostics may initialize process-global
    // state, so they run only after the immutable CPU budget is installed.
    setup_logging(true)?;
    neoethos_core::env_overrides::log_active_overrides_at_startup();

    // Config-consolidation (audit S05): every runtime override comes from the
    // single config via ONE shared installer, before any executor can observe
    // a default or initialize a dependency-owned worker pool.
    neoethos_app::install_runtime_overrides_from_settings(&settings);
    startup_trace.record(StartupEvent::RuntimeSettingsInstalled)?;

    let runtime_workers = installed.resolved().effective_worker_limit.get();
    let managed_runtime = build_managed_runtime(runtime_workers)?;
    startup_trace.record(StartupEvent::TokioRuntimeBuilt)?;
    let startup_diagnostics = startup_diagnostics_requested(&raw_args);
    debug_assert_eq!(startup_diagnostics, args.startup_diagnostics);
    debug_assert_eq!(args.cpu_threads.is_some(), parent_cpu_assignment.is_some());
    if startup_diagnostics {
        eprintln!(
            "{}",
            format_startup_diagnostics(
                "neoethos-app",
                installed,
                StartupRuntimeKind::Tokio,
                Some(runtime_workers),
                &startup_trace,
            )
        );
        return Ok(());
    }

    startup_trace.record(StartupEvent::ApplicationBuilderStarted)?;
    info!(
        target: "neoethos_app::startup",
        "{}",
        format_startup_diagnostics(
            "neoethos-app",
            installed,
            StartupRuntimeKind::Tokio,
            Some(runtime_workers),
            &startup_trace,
        )
    );
    managed_runtime.block_on(async_main(args, settings))
}

fn build_managed_runtime(worker_threads: usize) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
}

async fn async_main(args: Args, settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    // Tie this backend's lifetime to the GUI that spawned it (#179 follow-up).
    // A detached spawn would otherwise linger as a windowless orphan holding
    // port 7423 — which makes the NEXT app launch see a live backend and
    // silently exit, i.e. "the app won't open". The watchdog self-terminates
    // this process within ~2s of the parent exiting.
    if let Some(parent_pid) = args.parent_pid {
        spawn_parent_watchdog(parent_pid);
    }

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

    if args.validation_mode {
        // Mutually exclusive with the existing headless auto-* paths.
        // We could let clap enforce this via `conflicts_with`, but
        // returning a typed error here gives the operator a clearer
        // message and keeps the CLI parser config readable.
        if args.auto_discovery || args.auto_training {
            error!(
                "--validation-mode is mutually exclusive with --auto-discovery / --auto-training; \
                 drop the auto-* flag(s) and re-run"
            );
            return Err(anyhow::anyhow!(
                "--validation-mode conflicts with --auto-discovery / --auto-training"
            )
            .into());
        }
        info!(
            target: "neoethos_app::validation",
            tfs = %args.validation_tfs,
            tf_timeout_secs = args.validation_tf_timeout_secs,
            "Starting neoethos-app in VALIDATION-MODE (multi-TF Discovery sweep)..."
        );
        let exit_code = app_services::validation::run_validation_sweep(
            &runtime,
            &settings,
            &args.validation_tfs,
            args.validation_tf_timeout_secs,
            args.validation_min_generations,
        )
        .await?;
        std::process::exit(exit_code);
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

    if args.rebuild_symbol_metadata {
        // Phase D.2d (2026-05-28) — regenerate data/symbol_metadata.json
        // from the cached raw_batches WITHOUT a broker round-trip.
        // Picks the env_dir based on broker settings (demo vs live).
        let output_root = args
            .bootstrap_output
            .as_deref()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(app_services::capture_symbols::default_bootstrap_root);
        let env_label = app_services::capture_symbols::env_label_from_settings();
        let env_dir = output_root.join(env_label);
        info!(
            target: "neoethos_app::rebuild_symbol_metadata",
            env_dir = %env_dir.display(),
            "rebuilding SymbolMetadataTable from cached raw_batches"
        );
        let env_dir_clone = env_dir.clone();
        let table = tokio::task::spawn_blocking(move || {
            app_services::capture_symbols::build_symbol_metadata_table_from_catalog(&env_dir_clone)
        })
        .await
        .map_err(|e| anyhow::anyhow!("rebuild blocking task panicked: {e}"))??;
        let metadata_path = std::path::PathBuf::from("data").join("symbol_metadata.json");
        table
            .save_to_disk(&metadata_path)
            .map_err(|e| anyhow::anyhow!("write {}: {e}", metadata_path.display()))?;
        info!(
            target: "neoethos_app::rebuild_symbol_metadata",
            entries = table.entries.len(),
            path = %metadata_path.display(),
            "rebuild complete"
        );
        eprintln!(
            "[rebuild] wrote {} entries → {}",
            table.entries.len(),
            metadata_path.display()
        );
        return Ok(());
    }

    if args.bootstrap_broker_catalog {
        // Phase D.1 (2026-05-28) — pull the full broker catalog into
        // disk so Phase D.2 can replace `baked_in_default()` +
        // synthetic cost fallbacks with broker-supplied real data.
        let output_root = args
            .bootstrap_output
            .as_deref()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(app_services::capture_symbols::default_bootstrap_root);
        let env_label = app_services::capture_symbols::env_label_from_settings();
        info!(
            target: "neoethos_app::bootstrap_broker_catalog",
            env_label,
            output_root = %output_root.display(),
            "starting full broker catalog bootstrap"
        );
        let output_root_clone = output_root.clone();
        tokio::task::spawn_blocking(move || {
            app_services::capture_symbols::run_bootstrap(env_label, &output_root_clone)
        })
        .await
        .map_err(|e| anyhow::anyhow!("bootstrap blocking task panicked: {e}"))??;
        info!(
            "Bootstrap complete. Catalog under {}/{}",
            output_root.display(),
            env_label
        );
        return Ok(());
    }

    if let Some(raw) = args.capture_symbols.as_deref() {
        // Phase C (2026-05-28) — one-shot fixture capture. Runs the
        // 3-step `ProtoOAApplicationAuthReq → ProtoOAAccountAuthReq →
        // ProtoOASymbolByIdReq` sequence per requested symbol, writes
        // the raw broker envelope to disk, and exits. Driven by the
        // audit finding that we have ZERO recorded broker payloads
        // — Phase A.1 schema assumptions need real bytes to verify.
        let symbols = app_services::capture_symbols::parse_symbol_list(raw);
        if symbols.is_empty() {
            return Err(anyhow::anyhow!(
                "--capture-symbols requires a comma-separated list of symbol names, got: {raw:?}"
            )
            .into());
        }
        let output_dir = args
            .capture_output
            .as_deref()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(app_services::capture_symbols::default_output_dir);
        info!(
            target: "neoethos_app::capture_symbols",
            symbols = ?symbols,
            output_dir = %output_dir.display(),
            "starting cTrader fixture capture"
        );
        let symbols_clone = symbols.clone();
        let output_clone = output_dir.clone();
        tokio::task::spawn_blocking(move || {
            app_services::capture_symbols::run_capture(&symbols_clone, &output_clone)
        })
        .await
        .map_err(|e| anyhow::anyhow!("capture blocking task panicked: {e}"))??;
        info!("Capture complete. Fixtures under {}", output_dir.display());
        return Ok(());
    }

    if args.headless {
        info!("Starting neoethos in headless server mode...");
        run_headless_loop(runtime).await;
        return Ok(());
    }

    let _ = args.server; // keep the flag valid for back-compat.
    info!("Starting neoethos in HTTP server mode (Flutter front-end backend)...");

    // F-553/F-576 closure (2026-05-25): install the CLI `--config`
    // value process-wide so route handlers AND free functions
    // (engines_control::resolve_data_root, etc.) all see the same
    // path. The OnceLock-backed install happens BEFORE any
    // AppApiState construction so state.config_path() reads the
    // resolved value.
    server::state::install_config_path(args.config.clone());
    // REMOVED 2026-08-09 (dead-code purge, batch D2): the F-270
    // `install_launched_by_flutter` call. Flutter's BackendSupervisor — the
    // only reader of the /healthz field it fed — died 2026-06-22.
    let state = server::state::AppApiState::new();
    // F-231-related closure (2026-05-25): install the process-wide
    // account-refresh trigger so the deep cTrader execution-event
    // parser (and any future spontaneous-event listener) can flip
    // the dashboard within ~750 ms of a fill / close / margin call
    // without threading AppApiState through every call site. Same
    // pattern as `install_config_path` directly above.
    server::state::install_account_refresh_trigger(state.account_refresh_tx_clone());
    server::bridge::spawn(state.clone());
    // Autonomous LLM supervisor heartbeat (no-op until enabled from the UI).
    app_services::supervisor::spawn(state.clone());
    // Session-aware spread sampler (records the broker's real per-hour spreads).
    app_services::spread_stats::spawn();
    // Auto-cull retirement → rediscovery queue drainer (Settings-gated).
    app_services::rediscovery::spawn(state.clone());

    // Spawn the live spot streamer (#137). Best-effort — if creds
    // are missing or the broker rejects auth, the helper logs and
    // returns false; the HTTP server still comes up. When the
    // user re-auths and restarts the binary, the next attempt
    // picks up the fresh token automatically.
    let spawned = tokio::task::spawn_blocking(
        app_services::live_spots_streamer::try_spawn_with_defaults_blocking,
    )
    .await
    .unwrap_or(false);
    if spawned {
        info!("Live spot streamer spawned — /live/spots will populate as ticks arrive");
    } else {
        info!(
            "Live spot streamer not spawned (creds/token missing or unreachable) — \
             /live/spots will return an empty list"
        );
    }

    server::serve(state).await?;
    Ok(())
}

/// Spawn a background thread that self-terminates this backend within ~2s of
/// the parent GUI (`parent_pid`) exiting. Without this, the detached spawn
/// would leave a windowless backend holding port 7423 after the GUI
/// closes/crashes, which makes the next app launch see a live backend and
/// silently exit — the "app won't open" bug.
/// Captures the parent's start time so OS PID reuse can't masquerade as the
/// parent still being alive; fail-safe (keeps the backend running) if the
/// parent can't be located at startup.
fn spawn_parent_watchdog(parent_pid: u32) {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let _ = std::thread::Builder::new()
        .name("parent-watchdog".into())
        .spawn(move || {
            let target = Pid::from_u32(parent_pid);
            let mut sys = System::new();
            // Establish the parent's identity (start time) before enforcing.
            let mut parent_start: Option<u64> = None;
            for _ in 0..5 {
                sys.refresh_processes(ProcessesToUpdate::Some(&[target]), true);
                if let Some(parent) = sys.process(target) {
                    parent_start = Some(parent.start_time());
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            if parent_start.is_none() {
                tracing::warn!(
                    target: "neoethos_app::watchdog",
                    parent_pid,
                    "parent-pid watchdog could not locate the parent GUI at startup; \
                     leaving the backend running (no self-terminate on a possibly-stale pid)"
                );
                return;
            }
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                sys.refresh_processes(ProcessesToUpdate::Some(&[target]), true);
                let parent_alive = sys
                    .process(target)
                    .map(|parent| Some(parent.start_time()) == parent_start)
                    .unwrap_or(false);
                if !parent_alive {
                    tracing::warn!(
                        target: "neoethos_app::watchdog",
                        parent_pid,
                        "parent GUI exited — backend self-terminating to free port 7423 (no orphan)"
                    );
                    std::process::exit(0);
                }
            }
        });
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
        // ── 2026-08-10, config consolidation ─────────────────────────────
        // This built `DiscoveryConfig::default()`. Every discovery knob the
        // operator had set — population, generations, gates, cost model,
        // `min_history_years` — was IGNORED on the headless auto-discovery
        // path, while the UI path (`server/engines_control.rs:205`) and the
        // validation sweep (`app_services/validation.rs:283`) both used
        // `from_settings`. Same binary, same config file, three different
        // answers depending on how the run was started.
        //
        // FAIL LOUD, do not substitute. If the config cannot be read we do
        // NOT quietly fall back to the compiled defaults and run anyway —
        // that is the failure wearing the costume of a choice. The run is
        // refused and the reason is named.
        let discovery_setup = match Settings::from_yaml(&runtime.config_path) {
            Ok(settings) => match neoethos_search::DiscoveryConfig::try_from_settings(&settings) {
                Ok(cfg) => {
                    let symbol = settings.system.resolve_symbol();
                    let base_tf = settings.system.resolve_base_timeframe();
                    match app_services::discovery::resolve_unique_background_dataset_identity(
                        &runtime.data_dir,
                        &symbol,
                        &base_tf,
                    ) {
                        Ok(dataset_identity) => {
                            let higher_tfs = settings.system.resolve_higher_timeframes(&base_tf);
                            info!(
                                config_path = %runtime.config_path,
                                dataset_identity = %dataset_identity.to_path_component(),
                                population = cfg.population,
                                generations = cfg.generations,
                                min_history_years = cfg.runtime_overrides.min_history_years,
                                "Headless: discovery config and exact dataset identity resolved"
                            );
                            Some((dataset_identity, higher_tfs, cfg))
                        }
                        Err(err) => {
                            error!(
                                symbol = %symbol,
                                base_tf = %base_tf,
                                error = %err,
                                "Headless: refusing auto-discovery because background selection did not resolve exactly one canonical dataset identity"
                            );
                            None
                        }
                    }
                }
                Err(err) => {
                    error!(
                        config_path = %runtime.config_path,
                        error = %err,
                        "Headless: refusing to auto-start discovery because exact broker financial evidence is unavailable"
                    );
                    None
                }
            },
            Err(err) => {
                error!(
                    config_path = %runtime.config_path,
                    error = %err,
                    "Headless: REFUSING to auto-start discovery — the config file \
                     could not be read, and running on the compiled defaults would \
                     search a different space than the operator configured without \
                     saying so. Fix the config and restart. (Auto-training, if \
                     requested, is unaffected and still runs.)"
                );
                None
            }
        };
        if let Some((dataset_identity, higher_tfs, discovery_config)) = discovery_setup {
            match app_services::discovery::pin_current_discovery_input(
                &runtime.data_dir,
                &dataset_identity,
                &higher_tfs,
            ) {
                Ok(pinned_input) => {
                    let request = DiscoveryRequest {
                        data_root: runtime.data_dir.clone(),
                        pinned_input: std::sync::Arc::new(pinned_input),
                        higher_tfs,
                        config: discovery_config,
                        prop_firm_rules: neoethos_search::PropFirmRiskRules::default(),
                    };
                    match start_discovery_job(request, tx.clone()) {
                        Ok(_handle) => info!("Headless: discovery job started"),
                        Err(err) => error!("Headless: failed to start discovery: {}", err),
                    }
                }
                Err(err) => error!(
                    dataset_identity = %dataset_identity.to_path_component(),
                    error = %err,
                    "Headless: exact discovery generations could not be pinned; acquire/refresh data before retrying"
                ),
            }
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
}
