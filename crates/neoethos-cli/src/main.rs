use anyhow::{Context, Result};
use neoethos_core::logging::{setup_logging, write_subsystem_record};
use neoethos_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::time::{SystemTime, UNIX_EPOCH};

mod gpu_bench;
mod gpu_bench_population;
mod gpu_bench_prepare;
mod gpu_bench_snapshot;
mod tui;

fn main() -> Result<()> {
    setup_logging(false)?;
    // Say out loud what the environment is changing underneath this run.
    //
    // This function was written for exactly this call and had ZERO callers — its
    // own doc says "Designed to be called once in the binary's main() after
    // setup_logging". Meanwhile the workspace carries 215 distinct NEOETHOS_*
    // names across 183 `env::var` sites, and an env var that silently alters a
    // run is how `apply_mode_overrides`, the search runtime overrides and the
    // OOS holdout each spent months meaning something weaker than they claimed.
    // Nothing here changes behaviour; it makes the behaviour visible, which is
    // the precondition for retiring these in favour of the single config.
    neoethos_core::env_overrides::log_active_overrides_at_startup();
    // Config-consolidation: search runtime overrides come from the single
    // config (canonical user config.yaml), not the environment. (S2a:
    // genetic search; rest staged.)
    //
    // ⚠ AUDIT #125/#289, CLI HALF — this line read
    // `Settings::load().unwrap_or_default()` and its own comment admitted
    // "Falls back to defaults if it can't be loaded". `Settings::default()` is
    // NOT the shipped configuration: `ModelsConfig::default()` still encodes
    // the pre-2026-06-06 posture and re-arms `require_walkforward_for_export`
    // and `prop_firm_min_pass_rate`, so a config that failed to load silently
    // changed WHICH STRATEGIES MAY REACH LIVE — on the binary that actually
    // runs discovery. The same defect was recorded for the desktop shell
    // (#125, `desktop/src-tauri/src/lib.rs:65`) and closed there; nothing in
    // the 323 items noticed this one.
    //
    // Now: a failed load is fatal for every subcommand that decides money or
    // spends hours, and survivable ONLY for the diagnostic commands you would
    // reach for to fix it (`config`, `credentials`, `setup`, `wizard`,
    // `--help`, `--version`). Those run on built-in defaults with an
    // unmissable banner, never quietly.
    let raw_args: Vec<String> = std::env::args().collect();
    // Owned, so the `raw_args` vector can be moved into `args` further down.
    let subcommand: String = raw_args.get(1).cloned().unwrap_or_default();
    let mut startup_settings = match neoethos_core::Settings::load() {
        Ok(s) => s,
        Err(err) => {
            let path = neoethos_core::config::user_config_path();
            let diagnostic = matches!(
                subcommand.as_str(),
                // NOT the empty subcommand: a bare `neoethos-cli` opens the
                // TUI, which can START ENGINES. It takes over the terminal, so
                // the banner above scrolls away unread. `config` is the way in.
                "config"
                    | "credentials"
                    | "setup"
                    | "wizard"
                    | "--help"
                    | "-h"
                    | "help"
                    | "--version"
                    | "-V"
                    | "version"
            );
            eprintln!("──────────────────────────────────────────────────────────────");
            eprintln!("CONFIG NOT LOADED");
            eprintln!("  tried: $CONFIG_FILE, then {}, then ./config.yaml", path.display());
            eprintln!("  {err:#}");
            eprintln!("  Built-in defaults are NOT your settings: they re-arm the export");
            eprintln!("  gates and risk limits you set deliberately, and they decide which");
            eprintln!("  strategies may reach live money.");
            eprintln!("──────────────────────────────────────────────────────────────");
            tracing::error!(
                target: "neoethos_cli",
                config = %path.display(),
                error = %format!("{err:#}"),
                subcommand = %subcommand,
                diagnostic_command = diagnostic,
                "config.yaml could not be loaded"
            );
            if !diagnostic {
                anyhow::bail!(
                    "refusing to run `{subcommand}` without a readable config.\n\
                     Put a valid config.yaml at {} (or in the working directory, or point \
                     $CONFIG_FILE at one) and re-run. `neoethos-cli config` still works \
                     without it.",
                    path.display()
                );
            }
            eprintln!(
                "Continuing on built-in defaults because `{subcommand}` is a diagnostic command."
            );
            neoethos_core::Settings::default()
        }
    };
    // CPU budget for THIS process — an internal parent→child handoff, not an
    // operator knob.
    //
    // It used to travel as the environment variable `NEOETHOS_BOT_CPU_BUDGET`,
    // set by `spawn_discover_combo` on each `discover` child the schedule
    // orchestrator starts. That was the config-chaos pattern in miniature: it
    // appeared in no config file and in no knob catalog, yet a stale export
    // left in a shell silently re-partitioned the cores of every later run,
    // and the old `.ok().and_then(parse.ok())` chain swallowed a typo without
    // a word — `NEOETHOS_BOT_CPU_BUDGET=eight` meant "no assignment".
    //
    // It is now `--cpu-threads`: visible in the process list, scoped to the
    // one invocation it was written for, and FATAL when malformed.
    let process_cpu_assignment = match parse_flag(&raw_args, "--cpu-threads") {
        None => None,
        Some(raw) => match raw.trim().parse::<usize>() {
            Ok(v) if v > 0 => Some(v),
            _ => anyhow::bail!(
                "--cpu-threads expects a positive integer, got `{raw}`. This flag is set by \
                 the schedule orchestrator on the children it spawns; there is no reason to \
                 pass it by hand. Refusing to run rather than guess a core budget."
            ),
        },
    };
    warn_retired_env_vars();
    startup_settings.apply_process_cpu_assignment(process_cpu_assignment);
    neoethos_search::install_search_runtime_overrides_from_settings(&startup_settings);
    neoethos_models::tree_models::config::install_tree_runtime_from_settings(&startup_settings);
    neoethos_models::statistical::common::install_statistical_runtime_from_settings(
        &startup_settings,
    );
    neoethos_models::genetic::install_genetic_runtime_from_settings(&startup_settings);
    neoethos_core::system::install_hardware_runtime_overrides_from_settings(&startup_settings);
    neoethos_data::install_data_runtime_overrides(
        startup_settings.models.data_runtime.normalize_features,
        startup_settings
            .models
            .data_runtime
            .rebuild_stale_higher_tfs,
    );
    // Same vector the config-load guard above already collected.
    let args: Vec<String> = raw_args;
    if args.len() < 2 {
        // No subcommand → launch interactive TUI. Use `--help` for
        // legacy bare help; explicit subcommands keep working
        // unchanged for scripting.
        if let Err(err) = write_subsystem_record(
            SubsystemSection::Cli,
            cli_record("tui", "STARTED", "launching interactive TUI"),
        ) {
            tracing::warn!(
                target: "neoethos_cli",
                error = %err,
                "failed to write CLI 'tui STARTED' subsystem record"
            );
        }
        let res = tui::run_tui(None);
        if let Err(err) = write_subsystem_record(
            SubsystemSection::Cli,
            cli_record(
                "tui",
                if res.is_ok() { "SUCCESS" } else { "FAILED" },
                match &res {
                    Ok(_) => "TUI session ended cleanly".to_string(),
                    Err(err) => format!("TUI session ended with error: {err}"),
                },
            ),
        ) {
            tracing::warn!(
                target: "neoethos_cli",
                error = %err,
                "failed to write CLI 'tui {}' subsystem record",
                if res.is_ok() { "SUCCESS" } else { "FAILED" }
            );
        }
        return res;
    }
    if matches!(args[1].as_str(), "--help" | "-h" | "help") {
        print_help();
        return Ok(());
    }
    let command = args[1].clone();
    write_subsystem_record(
        SubsystemSection::Cli,
        cli_record(
            &command,
            "STARTED",
            format!("starting CLI command {}", command),
        ),
    )?;

    let result = match args[1].as_str() {
        "symbols" => cmd_symbols(&args[2..]),
        "timeframes" => cmd_timeframes(&args[2..]),
        "load" => cmd_load(&args[2..]),
        "features" => cmd_features(&args[2..]),
        "prepare" => cmd_prepare(&args[2..]),
        "resample" => cmd_resample(&args[2..]),
        "train" => cmd_train(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "discover" => cmd_discover(&args[2..]),
        "discovery-promote-weekly" => cmd_discovery_promote_weekly(&args[2..]),
        "trader-replay" => cmd_trader_replay(&args[2..]),
        "forward-test" => cmd_forward_test(&args[2..]),
        "blend-test" => cmd_blend_test(&args[2..]),
        "batch-discover" => cmd_batch_discover(&args[2..]),
        "bench" => gpu_bench::run(&args[2..]),
        "bench-prepare" => gpu_bench_prepare::run_prepare(&args[2..]),
        "bench-matrix" => gpu_bench_prepare::run_matrix(&args[2..]),
        "bench-collate" => gpu_bench_prepare::run_collate(&args[2..]),
        "bench-preflight-report" => gpu_bench_prepare::run_preflight_report(&args[2..]),
        "migrate-data" => cmd_migrate_data(&args[2..]),
        "slice-dataset" => cmd_slice_dataset(&args[2..]),
        "import" => cmd_import(&args[2..]),
        "config" => cmd_config(&args[2..]),
        "auto-loop" => cmd_auto_loop(&args[2..]),
        "schedule" => cmd_schedule(&args[2..]),
        "stop-target" => cmd_stop_target(&args[2..]),
        "wizard" => cmd_wizard(&args[2..]),
        "setup" => cmd_setup(&args[2..]),
        "credentials" => cmd_credentials(&args[2..]),
        _ => {
            print_help();
            Ok(())
        }
    };

    match &result {
        Ok(_) => {
            write_subsystem_record(
                SubsystemSection::Cli,
                cli_record(
                    &command,
                    "SUCCESS",
                    format!("CLI command {} completed", command),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Cli,
                cli_record(
                    &command,
                    "FAILED",
                    format!("CLI command {} failed: {}", command, err),
                ),
            )?;
        }
    }

    result
}

fn cmd_load(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let mut root = parse_root(args, settings.as_ref());
    let mut symbol = None;
    let mut timeframe = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--root" => {
                if let Some(val) = iter.next() {
                    root = val.to_string();
                }
            }
            "--symbol" => {
                if let Some(val) = iter.next() {
                    symbol = Some(val.to_string());
                }
            }
            "--timeframe" => {
                if let Some(val) = iter.next() {
                    timeframe = Some(val.to_string());
                }
            }
            _ => {}
        }
    }

    let symbol = symbol.unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe = timeframe.unwrap_or_else(|| default_base_tf(settings.as_ref()));

    let ohlcv = neoethos_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    println!("Loaded {} {} rows: {}", symbol, timeframe, ohlcv.len());
    Ok(())
}

/// `slice-dataset --symbol EURUSD --base M1 --root <SRC> --out-root <DST>
///                --from-date 2018-01-01 --to-date 2021-01-01`
///
/// Additive, NON-destructive: reads the source `(symbol, base)` Vortex
/// dataset from `<SRC>`, keeps only the bars whose timestamp falls in the
/// half-open range `[from-date, to-date)` (UTC), and writes the filtered
/// subset to `<DST>/symbol=<SYM>/timeframe=<TF>/data.vortex` in the SAME
/// canonical Vortex layout the loader reads — so a subsequent
/// `discover --root <DST> --symbol <SYM> --base <TF>` runs on the slice.
///
/// Purpose: OOM-safe walk-forward. A multi-year M1 dataset that overflows
/// RAM on a weak machine can be chopped into e.g. 3-year windows that each
/// fit, discovered independently, and stitched by the operator.
///
/// Reuses the exact discovery IO path:
///   - reader: `neoethos_data::load_symbol_timeframe` (same as `discover`)
///   - date→row mapping + filter: `neoethos_data::slice_ohlcv_by_date_range_ms`
///   - writer: `neoethos_data::write_symbol_timeframe_vortex`
///     (canonical `write_ohlcv_vortex` under the hood)
///
/// Fails loud when the source is missing/empty or the range yields 0 rows.
fn cmd_slice_dataset(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());

    let out_root = parse_flag(args, "--out-root")
        .or_else(|| parse_flag(args, "--out"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "slice-dataset requires --out-root <DST> (the new root dir to write the slice into)"
            )
        })?;

    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    if symbol.is_empty() {
        anyhow::bail!(
            "slice-dataset: no --symbol supplied and config.yaml could not provide one — \
             pass --symbol explicitly (e.g. --symbol EURUSD)"
        );
    }
    // `--base` is the primary name (matches discover/prepare); `--timeframe`
    // is accepted as an alias for parity with `load`.
    let base = parse_flag(args, "--base")
        .or_else(|| parse_flag(args, "--timeframe"))
        .unwrap_or_else(|| default_base_tf(settings.as_ref()));
    if base.is_empty() {
        anyhow::bail!(
            "slice-dataset: no --base supplied and config.yaml could not provide one — \
             pass --base explicitly (e.g. --base M1)"
        );
    }

    let from_date = parse_flag(args, "--from-date").ok_or_else(|| {
        anyhow::anyhow!(
            "slice-dataset requires --from-date YYYY-MM-DD (inclusive lower bound, UTC)"
        )
    })?;
    let to_date = parse_flag(args, "--to-date").ok_or_else(|| {
        anyhow::anyhow!("slice-dataset requires --to-date YYYY-MM-DD (exclusive upper bound, UTC)")
    })?;

    let from_ms = parse_ymd_to_epoch_ms(&from_date, "--from-date")?;
    let to_ms = parse_ymd_to_epoch_ms(&to_date, "--to-date")?;
    if to_ms <= from_ms {
        anyhow::bail!(
            "slice-dataset: --to-date ({to_date}) must be strictly after --from-date ({from_date}) \
             (the range is half-open [from, to))"
        );
    }

    // Reader — identical to the `discover` command's load path.
    let source = neoethos_data::load_symbol_timeframe(&root, &symbol, &base).map_err(|err| {
        anyhow::anyhow!("slice-dataset: failed to load source {symbol} {base} from {root}: {err}")
    })?;
    let source_rows = source.len();
    if source_rows == 0 {
        anyhow::bail!(
            "slice-dataset: source {symbol} {base} at {root} is empty — nothing to slice"
        );
    }

    // Date→row mapping + half-open filter (shared data-crate helper).
    let (slice, span) = neoethos_data::slice_ohlcv_by_date_range_ms(&source, from_ms, to_ms)
        .map_err(|err| anyhow::anyhow!("slice-dataset: {err}"))?;
    let kept_rows = slice.len();
    if kept_rows == 0 {
        anyhow::bail!(
            "slice-dataset: 0 rows of {symbol} {base} fall in [{from_date}, {to_date}) — \
             the requested window does not overlap the source data \
             (source has {source_rows} rows). Widen the date range or check the dataset."
        );
    }

    // Writer — canonical Vortex layout, byte-compatible with the loader.
    let written = neoethos_data::write_symbol_timeframe_vortex(&out_root, &symbol, &base, &slice)
        .map_err(|err| {
        anyhow::anyhow!("slice-dataset: failed to write slice to {out_root}: {err}")
    })?;

    let (first_ms, last_ms) = span.expect("span is Some when kept_rows > 0");
    println!(
        "slice-dataset {symbol} {base}: [{from_date}, {to_date})  source rows={source_rows}  kept rows={kept_rows}"
    );
    println!(
        "  kept span: {} .. {}",
        format_epoch_ms_date(first_ms),
        format_epoch_ms_date(last_ms)
    );
    println!("  written: {}", written.display());
    Ok(())
}

/// Parse a `YYYY-MM-DD` date as midnight UTC and return epoch milliseconds.
/// Fails loud with the offending flag name when the string isn't a valid date.
fn parse_ymd_to_epoch_ms(date: &str, flag: &str) -> Result<i64> {
    let naive = chrono::NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d").map_err(|err| {
        anyhow::anyhow!("slice-dataset: {flag} '{date}' is not a valid YYYY-MM-DD date: {err}")
    })?;
    let dt = naive.and_hms_opt(0, 0, 0).ok_or_else(|| {
        anyhow::anyhow!("slice-dataset: {flag} '{date}' could not be set to midnight")
    })?;
    Ok(dt.and_utc().timestamp_millis())
}

/// Render an epoch-ms timestamp as a `YYYY-MM-DD` UTC date for the kept-span
/// summary line.
fn format_epoch_ms_date(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("ms:{ms}"))
}

fn cmd_symbols(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbols = neoethos_data::discover_symbols(root)?;
    println!("Symbols ({}):", symbols.len());
    for sym in symbols {
        println!("  {}", sym);
    }
    Ok(())
}

fn cmd_timeframes(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let tfs = neoethos_data::discover_timeframes(root, &symbol)?;
    println!("Timeframes for {} ({}):", symbol, tfs.len());
    for tf in tfs {
        println!("  {}", tf);
    }
    Ok(())
}

fn cmd_features(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe =
        parse_flag(args, "--timeframe").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let ohlcv = neoethos_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    let features = neoethos_data::compute_hpc_features(&ohlcv)?;
    println!(
        "Features {} {} -> rows={}, cols={}",
        symbol,
        timeframe,
        features.n_samples(),
        features.n_features()
    );
    Ok(())
}

fn cmd_prepare(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let higher = parse_flag(args, "--higher")
        .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref(), &base));
    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();
    let dataset = neoethos_data::load_symbol_dataset(&root, &symbol)?;
    let features =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs)?;
    println!(
        "Prepared {} base={} rows={} cols={}",
        symbol,
        base,
        features.n_samples(),
        features.n_features()
    );
    Ok(())
}

/// `discovery-promote-weekly [--symbol X --tf Y] [--cache-dir ...] [--portfolio ...]`
/// — the weekly-refresh promotion step of the search-memory feature.
///
/// Loads THIS run's discovery ledger (`<cache-dir>/{SYMBOL}_{TF}.discovery_ledger.json`,
/// written by every discovery run) and merges its recorded genes into the live
/// portfolio under the **additive** policy: a ledger gene is "new" when its
/// canonical signature hash is not already present among the live portfolio's
/// genes; existing genes are always carried. Prints a growth summary
/// ("added N new, carried M, total K").
///
/// SCOPE NOTE (deferred — see report): the ledger records gene *signatures*
/// (hash + indicator names + SMC flags + fitness), not the full `Gene`
/// (indices/weights). The live portfolio stores full genes. So we can detect
/// which ledger genes are NEW relative to the live portfolio and carry the
/// existing full genes forward, but we cannot synthesize full genes for ledger-
/// only records here — adding them as live genes would require the discovery
/// run to also persist full genes per ledger entry (a future enhancement). We
/// therefore write a merged **growth-summary JSON** next to the portfolio and
/// report the new-vs-carried breakdown.
fn cmd_discovery_promote_weekly(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let ledger_cfg = settings
        .as_ref()
        .map(|s| s.models.discovery_ledger.clone())
        .unwrap_or_default();

    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let tf = parse_flag(args, "--tf")
        .or_else(|| parse_flag(args, "--base"))
        .unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let cache_dir = parse_flag(args, "--cache-dir").unwrap_or(ledger_cfg.cache_dir);

    let ledger = neoethos_search::load_prior_ledger(&cache_dir, &symbol, &tf).ok_or_else(|| {
        anyhow::anyhow!(
            "no discovery ledger found at {} — run a discovery for {} {} first \
             (the ledger is written automatically when models.discovery_ledger.enabled = true)",
            neoethos_search::ledger_path(&cache_dir, &symbol, &tf).display(),
            symbol,
            tf
        )
    })?;

    // Ledger genes recorded this run (portfolio + archive), de-duplicated by hash.
    let mut ledger_hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in ledger.portfolio.iter().chain(ledger.archive.iter()) {
        ledger_hashes.insert(rec.hash.clone());
    }

    // The live portfolio whose full genes we carry. Default path mirrors the
    // discover command's `{out}.live_portfolio.json`, keyed off the ledger's
    // cache layout; override with --portfolio.
    let portfolio_path = parse_flag(args, "--portfolio")
        .unwrap_or_else(|| format!("{}/{}_{}.live_portfolio.json", cache_dir, symbol, tf));

    let mut existing_hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
    let existing_count = match neoethos_search::load_live_portfolio_json(&portfolio_path) {
        Ok(artifact) => {
            for gene in &artifact.genes {
                existing_hashes
                    .insert(neoethos_search::genetic::gene_signature_hash(gene).to_string());
            }
            artifact.genes.len()
        }
        Err(_) => {
            println!(
                "(no existing live portfolio at {} — treating all ledger genes as new)",
                portfolio_path
            );
            0
        }
    };

    let new_genes: Vec<&neoethos_search::GeneRecord> = ledger
        .portfolio
        .iter()
        .chain(ledger.archive.iter())
        .filter(|rec| !existing_hashes.contains(&rec.hash))
        .collect();
    // Distinct new hashes (a hash could appear in both portfolio + archive).
    let new_hashes: std::collections::HashSet<&String> =
        new_genes.iter().map(|r| &r.hash).collect();
    let added = new_hashes.len();
    let carried = existing_count;
    let total = carried + added;

    // Write a merged growth-summary JSON next to the portfolio so the weekly run
    // leaves an auditable record of what grew.
    #[derive(serde::Serialize)]
    struct PromotionSummary<'a> {
        symbol: &'a str,
        tf: &'a str,
        policy: &'a str,
        carried: usize,
        added: usize,
        total: usize,
        new_genes: Vec<&'a neoethos_search::GeneRecord>,
    }
    let summary_path = format!("{}/{}_{}.weekly_promotion.json", cache_dir, symbol, tf);
    let summary = PromotionSummary {
        symbol: &symbol,
        tf: &tf,
        policy: &ledger_cfg.promotion_policy,
        carried,
        added,
        total,
        new_genes: new_genes.clone(),
    };
    if let Err(err) = neoethos_core::storage::json::write_json_atomic(&summary_path, &summary) {
        tracing::warn!(
            target: "neoethos_cli::discovery_promote_weekly",
            error = %err,
            path = %summary_path,
            "failed to write weekly-promotion summary (non-fatal)"
        );
    }

    println!(
        "discovery-promote-weekly {} {} (policy={}): added {} new, carried {}, total {}",
        symbol, tf, ledger_cfg.promotion_policy, added, carried, total
    );
    println!(
        "  ledger: {}",
        neoethos_search::ledger_path(&cache_dir, &symbol, &tf).display()
    );
    println!("  summary written: {}", summary_path);
    if added > 0 {
        println!("  new strategies this run:");
        for rec in new_genes.iter().take(20) {
            let flags = if rec.smc_flags.is_empty() {
                "-".to_string()
            } else {
                rec.smc_flags.clone()
            };
            println!(
                "    fitness={:.4} sharpe={:.3} trades={:.0} smc=[{}] indicators={:?}",
                rec.fitness, rec.sharpe, rec.trades, flags, rec.indicator_names
            );
        }
        if new_genes.len() > 20 {
            println!("    ... and {} more", new_genes.len() - 20);
        }
    }
    Ok(())
}

/// `trader-replay --symbol EURUSD --base M1 [--root data]` — offline dry-run of
/// the autonomous-trader engine over real on-disk history. Drives the SAME
/// `neoethos_trader` engine the app's `/autonomous/replay` endpoint does (UI↔CLI
/// parity) with ZERO broker calls, and prints the resulting EngineStats. Symbol
/// and base resolve through the shared `SystemConfig` resolvers, same as
/// `discover`.
fn cmd_trader_replay(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    // With --portfolio <live_portfolio.json>, run the REAL discovered genes
    // (symbol/base come from the artifact). Without it, run the momentum stub on
    // --symbol/--base. Both drive the SAME engine (parity with /autonomous/replay).
    let stats = if let Some(portfolio) = parse_flag(args, "--portfolio") {
        // `--blend off|confirm|scale` (default off). With confirm/scale the
        // discovered genes' size is gated by the per-(symbol,base_tf)
        // SoftVotingEnsemble loaded from `--models-root` (default `models`) —
        // gene-dominant meta-labeling; ML never flips direction. `off` is
        // byte-identical to the gene-only path.
        let blend_arg = parse_flag(args, "--blend").unwrap_or_else(|| "off".to_string());
        let mode = match blend_arg.trim().to_ascii_lowercase().as_str() {
            "off" | "genes" | "genes_only" | "genesonly" => neoethos_trader::BlendMode::GenesOnly,
            "confirm" | "mlconfirm" => neoethos_trader::BlendMode::MlConfirm,
            "scale" | "mlscale" => neoethos_trader::BlendMode::MlScale,
            other => anyhow::bail!("--blend must be off|confirm|scale (got '{other}')"),
        };
        if matches!(mode, neoethos_trader::BlendMode::GenesOnly) {
            neoethos_trader::replay_portfolio_from_dir(
                &root,
                &portfolio,
                replay_engine_config(settings.as_ref(), &default_symbol(settings.as_ref())),
            )?
        } else {
            let models_root =
                parse_flag(args, "--models-root").unwrap_or_else(|| "models".to_string());
            // Both multipliers go through `BlendConfig::from_config_values` —
            // the ONE constructor that validates them. Writing the fields
            // directly (what this did until 2026-08-10) bypassed the inversion
            // refusal, so the CLI could hand the replay a blend the constructor
            // would have rejected: `--veto-below 0.80 --gate-floor 0.10` used to
            // be accepted silently and vetoes every bar the floor exists to keep
            // tradeable. The old `.clamp(0.0, 1.0)` was the other half of the
            // problem — it turned `--gate-floor 1.5` into 1.0 with no message.
            let requested_gate_floor = parse_blend_knob(args, "--gate-floor")?;
            let requested_veto_below = parse_blend_knob(args, "--veto-below")?;
            let blend = neoethos_trader::BlendConfig::from_config_values(
                mode,
                requested_gate_floor,
                requested_veto_below,
            );
            // Print the EFFECTIVE numbers next to what was asked for, so a
            // refusal (also logged at warn by the constructor, with both values)
            // is visible on stdout too — this replay's every position size
            // depends on which of the two won.
            println!(
                "  blend mode={blend_arg} models_root={models_root} gate_floor={:.2} veto_below={:.2}",
                blend.gate_floor, blend.veto_below
            );
            for (name, requested, used) in [
                ("gate-floor", requested_gate_floor, blend.gate_floor),
                ("veto-below", requested_veto_below, blend.veto_below),
            ] {
                if let Some(req) = requested {
                    if req != used {
                        println!(
                            "  WARNING: --{name} {req} REFUSED (out of [0,1] or inverted pair) — using {used}"
                        );
                    }
                }
            }
            neoethos_trader::replay_blend_from_dir(
                &root,
                &portfolio,
                &models_root,
                replay_engine_config(settings.as_ref(), &default_symbol(settings.as_ref())),
                blend,
            )?
        }
    } else {
        let symbol =
            parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
        let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
        if symbol.trim().is_empty() || base.trim().is_empty() {
            anyhow::bail!(
                "trader-replay needs --symbol and --base (or a reachable config.yaml with \
                 system.symbol / system.base_timeframe), or pass \
                 --portfolio <live_portfolio.json> to run the discovered genes"
            );
        }
        neoethos_trader::replay_symbol_from_dir(
            &root,
            &symbol,
            &base,
            replay_engine_config(settings.as_ref(), &symbol),
        )?
    };
    println!("trader-replay (offline dry-run, zero broker calls):");
    println!(
        "  bars={} signals={} intents={} executed={} blocked={}",
        stats.bars_processed,
        stats.signals_evaluated,
        stats.intents_emitted,
        stats.intents_executed,
        stats.intents_blocked
    );
    println!(
        "  positions: opened={} closed={} open_now={}",
        stats.positions_opened, stats.positions_closed, stats.open_positions
    );
    println!(
        "  realized_pnl={:.5} equity={:.2}",
        stats.realized_pnl, stats.equity
    );
    Ok(())
}

/// `forward-test --portfolio <live_portfolio.json> [--root data] [--oos-from 2023-01-01]`
/// — FAITHFUL out-of-sample test: runs each gene's REAL strategy (its own SL/TP +
/// risk-based confidence-scaled sizing + full costs) via the discovery backtest
/// engine on the holdout window, features computed warm over the FULL series then
/// sliced to [oos-from, end). Reports per-gene IS-vs-OOS + Walk-Forward Efficiency.
fn cmd_forward_test(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let config = settings
        .as_ref()
        .map(neoethos_search::DiscoveryConfig::from_settings)
        .unwrap_or_default();
    let portfolio = parse_flag(args, "--portfolio").ok_or_else(|| {
        anyhow::anyhow!("forward-test requires --portfolio <live_portfolio.json>")
    })?;
    let oos_from = parse_flag(args, "--oos-from").unwrap_or_else(|| "2023-01-01".to_string());
    let oos_ms = parse_ymd_to_epoch_ms(&oos_from, "--oos-from")?;

    let results = neoethos_search::faithful_oos_eval(
        &config,
        std::path::Path::new(&root),
        std::path::Path::new(&portfolio),
        oos_ms,
    )?;

    println!("FAITHFUL OOS forward-test (gene real SL/TP + risk sizing; holdout from {oos_from}):");
    println!(
        "{:<16}{:>5}{:>5}{:>8}{:>8}{:>8}{:>8}{:>10}{:>7}{:>8}  verdict",
        "gene",
        "#ind",
        "#smc",
        "IS_PF",
        "IS_DD%",
        "OOS_PF",
        "OOS_DD%",
        "OOS_net",
        "OOS_tr",
        "WFE_shp"
    );
    let mut survives = 0usize;
    for r in &results {
        let net = r.oos.net_profit;
        let oos_dd = r.oos.max_drawdown * 100.0;
        let is_survivor = r.oos.trade_count >= 30
            && r.wfe_sharpe >= 0.5
            && r.oos.profit_factor >= 1.3
            && oos_dd <= 10.0
            && net > 0.0;
        let verdict = if r.oos.trade_count < 30 {
            "DEAD (<30 tr)"
        } else if is_survivor {
            "SURVIVES"
        } else if net > 0.0 && r.oos.profit_factor >= 1.0 {
            "weak+"
        } else {
            "FAILS-OOS"
        };
        if is_survivor {
            survives += 1;
        }
        println!(
            "{:<16}{:>5}{:>5}{:>8.2}{:>8.1}{:>8.2}{:>8.1}{:>10.0}{:>7}{:>8.2}  {}",
            r.strategy_id,
            r.n_indicators,
            r.n_smc,
            r.is_profit_factor,
            r.is_max_drawdown * 100.0,
            r.oos.profit_factor,
            oos_dd,
            net,
            r.oos.trade_count,
            r.wfe_sharpe,
            verdict
        );
    }
    println!(
        "SURVIVES={}/{} (WFE_sharpe>=0.5, OOS PF>=1.3, OOS DD<=10%, >=30 trades, net>0)",
        survives,
        results.len()
    );
    Ok(())
}

/// `blend-test --portfolio <live_portfolio.json> --models-root <models_oos_locked>
/// [--root data] [--gate-floor 0.34] [--veto-below 0.15]`
///
/// Stage 4 — re-validate the gene↔ML blend on the NETTED portfolio the live
/// engine actually trades (verdict #1: the trader nets the genes via
/// combine_gene_signals, so we compare on the netted signal, not per-gene). Runs
/// the SAME trader engine three ways — GenesOnly (baseline) vs MlConfirm vs
/// MlScale — over identical bars, and prints a paired EngineStats table + a
/// non-degradation verdict. The blend ships ON only if it does NOT degrade
/// genes-only.
///
/// IMPORTANT: point `--models-root` at a LEAK-FREE root (`cli train --oos-from
/// <date> --models-dir models_oos_locked`), else the ensemble has seen the
/// evaluation window and the comparison is contaminated. Note the trader engine
/// uses a uniform SL/TP model (decision.rs), so the absolute P&L is a simplified
/// figure — but a GenesOnly-vs-blend comparison on the SAME engine is a valid
/// apples-to-apples accept/reject (only the gated signal differs). The rigorous
/// per-gene faithful number remains `forward-test`.
fn cmd_blend_test(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let portfolio = parse_flag(args, "--portfolio")
        .ok_or_else(|| anyhow::anyhow!("blend-test requires --portfolio <live_portfolio.json>"))?;
    let models_root =
        parse_flag(args, "--models-root").unwrap_or_else(|| "models_oos_locked".to_string());
    // Same rule as `trader-replay`: the flags are handed to
    // `BlendConfig::from_config_values`, never written into the fields. This
    // site used to carry its OWN copies of the shipped defaults (0.34 / 0.15)
    // and its own `.clamp(0.0, 1.0)` — three ways to drift from
    // `DEFAULT_BLEND_GATE_FLOOR` / `DEFAULT_BLEND_VETO_BELOW` and one way
    // (an inverted pair) to run all three arms of the comparison with a blend
    // that vetoes every bar, which reads as "the blend is safe, it opened
    // nothing".
    let requested_gate_floor = parse_blend_knob(args, "--gate-floor")?;
    let requested_veto_below = parse_blend_knob(args, "--veto-below")?;
    let validated = neoethos_trader::BlendConfig::from_config_values(
        neoethos_trader::BlendMode::GenesOnly,
        requested_gate_floor,
        requested_veto_below,
    );
    let (gate_floor, veto_below) = (validated.gate_floor, validated.veto_below);
    for (name, requested, used) in [
        ("gate-floor", requested_gate_floor, gate_floor),
        ("veto-below", requested_veto_below, veto_below),
    ] {
        if let Some(req) = requested {
            if req != used {
                println!(
                    "  WARNING: --{name} {req} REFUSED (out of [0,1] or inverted pair) — using {used}"
                );
            }
        }
    }

    // All three arms get the SAME validated multipliers — only `mode` differs,
    // which is the entire point of the comparison.
    let run = |mode| -> Result<neoethos_trader::EngineStats> {
        neoethos_trader::replay_blend_from_dir(
            &root,
            &portfolio,
            &models_root,
            replay_engine_config(settings.as_ref(), &default_symbol(settings.as_ref())),
            neoethos_trader::BlendConfig::from_config_values(
                mode,
                Some(gate_floor),
                Some(veto_below),
            ),
        )
    };
    let genes = run(neoethos_trader::BlendMode::GenesOnly)?;
    let confirm = run(neoethos_trader::BlendMode::MlConfirm)?;
    let scale = run(neoethos_trader::BlendMode::MlScale)?;

    println!(
        "blend-test (NETTED trader engine, models_root={models_root}, gate_floor={gate_floor:.2}, veto_below={veto_below:.2}):"
    );
    let row = |name: &str, s: &neoethos_trader::EngineStats| {
        println!(
            "  {name:<8} pnl={:>12.5} equity={:>12.2} opened={:>5} closed={:>5} blocked={:>5} signals={:>6}",
            s.realized_pnl,
            s.equity,
            s.positions_opened,
            s.positions_closed,
            s.intents_blocked,
            s.signals_evaluated
        );
    };
    row("genes", &genes);
    row("confirm", &confirm);
    row("scale", &scale);

    // Non-degradation accept gate vs the genes-only baseline (same engine/bars).
    let verdict = |name: &str, s: &neoethos_trader::EngineStats| {
        let pnl_ok = s.realized_pnl >= genes.realized_pnl - 1e-9;
        let eq_ok = s.equity >= genes.equity - 1e-9;
        let traded = s.positions_opened >= 1;
        if pnl_ok && eq_ok && traded {
            println!(
                "  -> {name}: ACCEPT (>= genes-only on realized_pnl AND equity, still trades)"
            );
        } else if !traded {
            println!("  -> {name}: REJECT (blend vetoed every trade)");
        } else {
            println!("  -> {name}: REJECT (degrades vs genes-only)");
        }
    };
    verdict("confirm", &confirm);
    verdict("scale", &scale);
    println!(
        "NOTE: relative comparison on the trader's uniform-SL/TP engine; ensure --models-root is \
         leak-free (train --oos-from). Per-gene faithful numbers: `forward-test`."
    );
    Ok(())
}

fn cmd_resample(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let target = parse_flag(args, "--target").unwrap_or_else(|| "H1".to_string());
    let dataset = neoethos_data::load_symbol_dataset(&root, &symbol)?;
    let base_ohlcv = dataset
        .frames
        .get(&base)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;
    let resampled = neoethos_data::resample_ohlcv(base_ohlcv, &target)?;
    println!(
        "Resampled {} {} -> {} rows={}",
        symbol,
        base,
        target,
        resampled.len()
    );
    Ok(())
}

fn cmd_train(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, String)> {
        let settings_opt = resolve_cli_settings(args)?;
        // Folder-browse support (2026-05-14): `--data-path <folder>`
        // scans the folder, prints a discovery summary, and (if
        // `--dry-run` is also set) exits before training kicks off.
        if has_flag(args, "--data-path") || has_flag(args, "--dry-run") {
            let root = parse_root(args, settings_opt.as_ref());
            let _ = print_dataset_discovery_summary(&root)?;
            if has_flag(args, "--dry-run") {
                let dry_symbol = parse_flag(args, "--symbol")
                    .unwrap_or_else(|| default_symbol(settings_opt.as_ref()));
                let dry_base = parse_flag(args, "--base")
                    .unwrap_or_else(|| default_base_tf(settings_opt.as_ref()));
                return Ok((dry_symbol, dry_base));
            }
        }
        let settings = settings_opt.unwrap_or_else(neoethos_core::Settings::default);
        let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| settings.system.symbol.clone());
        let base =
            parse_flag(args, "--base").unwrap_or_else(|| settings.system.base_timeframe.clone());
        // Stage 4 leak-free OOS-locked retrain: `--oos-from YYYY-MM-DD` truncates
        // each symbol's training to rows strictly before the cutoff (minus the
        // triple-barrier purge), so the experts can be used in an OOS blend
        // validation on [cutoff, end) without look-ahead. The locked experts MUST
        // go to a SEPARATE root so production `models/` is never overwritten.
        let oos_ms = match parse_flag(args, "--oos-from") {
            Some(d) => Some(parse_ymd_to_epoch_ms(&d, "--oos-from")?),
            None => None,
        };
        let default_models_dir = if oos_ms.is_some() {
            "models_oos_locked"
        } else {
            "models"
        };
        let models_dir =
            parse_flag(args, "--models-dir").unwrap_or_else(|| default_models_dir.to_string());
        if oos_ms.is_some() {
            let norm = models_dir.replace('\\', "/");
            if models_dir == "models" || norm.ends_with("/models") {
                anyhow::bail!(
                    "--oos-from trains LEAK-LOCKED experts; refusing to write them to the \
                     production '{models_dir}' root. Use a distinct --models-dir \
                     (e.g. models_oos_locked)."
                );
            }
        }
        let mut orchestrator = neoethos_models::TrainingOrchestrator::new(
            settings,
            std::path::PathBuf::from(models_dir),
        );
        if let Some(ms) = oos_ms {
            orchestrator = orchestrator.with_oos_lock_from_ms(ms);
        }

        orchestrator.train_symbol(&symbol, &base)?;

        println!("Pure Rust training complete for {}", symbol);
        Ok((symbol, base))
    })();

    match &result {
        Ok((symbol, base)) => {
            write_subsystem_record(
                SubsystemSection::Training,
                section_record(
                    SubsystemSection::Training,
                    "train",
                    "SUCCESS",
                    format!("training completed for {} {}", symbol, base),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Training,
                section_record(
                    SubsystemSection::Training,
                    "train",
                    "FAILED",
                    format!("training failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

fn cmd_search(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let defaults = settings
        .as_ref()
        .map(neoethos_search::DiscoveryConfig::from_settings)
        .unwrap_or_default();
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let higher = parse_flag(args, "--higher")
        .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref(), &base));
    let genes: usize = parse_flag(args, "--genes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(defaults.population);
    let max_indicators: usize = parse_flag(args, "--max-indicators")
        .and_then(|v| v.parse().ok())
        .unwrap_or(defaults.max_indicators);
    let generations: usize = parse_flag(args, "--generations")
        .and_then(|v| v.parse().ok())
        .unwrap_or(defaults.generations);

    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

    let dataset = neoethos_data::load_symbol_dataset(&root, &symbol)?;
    let dataset = neoethos_data::ensure_timeframes_with_resample(
        &dataset,
        &base,
        neoethos_data::MANDATORY_TFS,
    )?;
    let features =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs)?;
    let base_ohlcv = dataset
        .frames
        .get(&base)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;

    let result =
        neoethos_search::evolve_search(&features, base_ohlcv, genes, generations, max_indicators)?;
    let mut best_idx = 0usize;
    let mut best_profit = f64::MIN;
    for (idx, metrics) in result.metrics.iter().enumerate() {
        let net_profit = metrics[0];
        if net_profit > best_profit {
            best_profit = net_profit;
            best_idx = idx;
        }
    }
    println!(
        "Search {} genes={} best_idx={} net_profit={:.2}",
        symbol, genes, best_idx, best_profit
    );
    Ok(())
}

/// What a streaming sweep leaves behind, beyond the per-batch portfolios: the
/// run-level canonical feature list, the survivors remapped onto it, and the
/// census of every batch the sweep abandoned.
///
/// Carried as a struct so the artifact-writing block below reads the same
/// whether the sweep ran one batch or forty.
struct StreamingArtifactBundle {
    canonical: neoethos_search::orchestration::CanonicalFeatureIndex,
    survivors: Vec<neoethos_search::orchestration::CanonicalSurvivor>,
    ledger: neoethos_search::orchestration::StreamingRunLedger,
    next_cursor: usize,
    space_len: usize,
    batch_columns: usize,
    streamed: bool,
}

fn cmd_discover(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, String, usize, usize)> {
        let settings = resolve_cli_settings(args)?;
        let defaults = settings
            .as_ref()
            .map(neoethos_search::DiscoveryConfig::from_settings)
            .unwrap_or_default();
        let root = parse_root(args, settings.as_ref());
        // Folder-browse support (2026-05-14): when `--data-path` or
        // `--dry-run` are supplied, scan the folder and emit a
        // dataset-layout summary before the GA pipeline starts.
        if has_flag(args, "--data-path") || has_flag(args, "--dry-run") {
            let _ = print_dataset_discovery_summary(&root)?;
            if has_flag(args, "--dry-run") {
                let dry_symbol = parse_flag(args, "--symbol")
                    .unwrap_or_else(|| default_symbol(settings.as_ref()));
                let dry_base = parse_flag(args, "--base")
                    .unwrap_or_else(|| default_base_tf(settings.as_ref()));
                return Ok((dry_symbol, dry_base, 0, 0));
            }
        }
        let symbol =
            parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
        let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
        let higher = parse_flag(args, "--higher")
            .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref(), &base));
        // F-304 fix (2026-05-28): bind the account currency for the
        // cost model. Resolution order:
        //   1. `--account-currency` CLI flag (operator-explicit)
        //   2. `Settings.system.account_currency` (from config.yaml or
        //      cTrader trader profile written back by the bridge)
        // (The legacy `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env override was
        // removed in v0.4.36 — config / CLI is the source.)
        // Empty propagates downstream — the cost-model NaN guard will
        // reject the run with a clear error message rather than
        // silently producing NaN spread/pip values that the sanitizer
        // scrubs to 0.0 (= GA sees zero-trade candidates).
        let account_currency = parse_flag(args, "--account-currency")
            .or_else(|| {
                settings
                    .as_ref()
                    .map(|s| s.system.account_currency.clone())
                    .filter(|c| !c.trim().is_empty())
            })
            .unwrap_or_default();
        let population: usize = parse_flag(args, "--population")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.population);
        let generations: usize = parse_flag(args, "--generations")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.generations);
        let max_indicators: usize = parse_flag(args, "--max-indicators")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.max_indicators);
        let candidate_count: usize = parse_flag(args, "--candidates")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.candidate_count);
        let portfolio_size: usize = parse_flag(args, "--portfolio-size")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.portfolio_size);
        let corr_threshold: f64 = parse_flag(args, "--corr")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.corr_threshold);
        let min_trades_per_day: f64 = parse_flag(args, "--min-trades")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.min_trades_per_day);
        let out = parse_flag(args, "--out")
            .unwrap_or_else(|| "cache/vector_ta_knowledge.json".to_string());

        let higher_list: Vec<String> = higher
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

        // agent 2026-06-05 perf fix: load ONLY base + higher TFs, not every
        // timeframe. `load_symbol_dataset` loaded EVERY canonical TF (incl M1's
        // ~5.27M rows) for every combo, then `ensure_timeframes_with_resample`
        // cloned the whole frame map — the dominant per-combo pre-GA cost
        // (minutes, GPU idle). `ensure_timeframes_with_resample` skips TFs <= base
        // and only resamples MISSING higher TFs from the base, so base + higher
        // (filtered to what exists on disk) is sufficient. M1's 5.27M rows are now
        // loaded only for the M1-base combo, not for every combo.
        let mut want_tfs: Vec<String> = vec![base.clone()];
        for h in &higher_list {
            if !want_tfs.contains(h) {
                want_tfs.push(h.clone());
            }
        }
        want_tfs
            .retain(|tf| neoethos_data::symbol_timeframe_vortex_path(&root, &symbol, tf).exists());
        if !want_tfs.iter().any(|t| t == &base) {
            want_tfs.push(base.clone());
        }
        let want_refs: Vec<&str> = want_tfs.iter().map(|s| s.as_str()).collect();
        let dataset =
            neoethos_data::load_symbol_dataset_with_timeframes(&root, &symbol, &want_refs)?;
        let dataset = neoethos_data::ensure_timeframes_with_resample(
            &dataset,
            &base,
            neoethos_data::MANDATORY_TFS,
        )?;
        let base_ohlcv = dataset
            .frames
            .get(&base)
            .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;

        let config = neoethos_search::DiscoveryConfig {
            timeframe_label: base.clone(),
            // F-304 fix (2026-05-28): bind the CLI-resolved symbol +
            // account currency BEFORE `..defaults.clone()` so the
            // cost-model receives the operator's chosen values, not
            // the (potentially stale or empty) settings copy. Empty
            // values still propagate and trip the run-loud guard.
            evaluation_symbol: symbol.clone(),
            evaluation_account_currency: account_currency.clone(),
            population,
            generations,
            max_indicators,
            candidate_count,
            portfolio_size,
            corr_threshold,
            min_trades_per_day,
            filtering: defaults.filtering,
            ..defaults.clone()
        }
        .apply_mode_overrides();
        // ── THE STREAMING WORKING-SET SWEEP (opt-in) ────────────────────────
        //
        // `--stream-sweep` advances the working set through the
        // (indicator, period) space in batches instead of building ONE cube and
        // searching it. `--stream-max-batches N` spends at most N batches
        // (default: until the space is exhausted).
        //
        // The batch WIDTH is deliberately not a flag: it comes from free RAM
        // and the widest frame via `hpc_ta::streaming_batch_columns`, so peak
        // memory is a function of the hardware and never of what the operator
        // typed (the never-OOM invariant).
        //
        // WITHOUT the flag this is byte-for-byte the previous code: one
        // `prepare_multitimeframe_features`, one holdout cycle, the same
        // artifacts. That is the parity case, and it is the default.
        let stream_sweep = has_flag(args, "--stream-sweep");
        let stream_max_batches: usize = parse_flag(args, "--stream-max-batches")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        // Audit B02/B03 (2026-07-13): the CLI used to run discovery on the
        // FULL series — no held-out tail, so every "validation" window had
        // already been seen during selection. The holdout wrapper withholds
        // the last 20% and attaches honest forward-test/prop-firm evidence.
        let (result, streaming) = if !stream_sweep {
            let features =
                neoethos_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs)?;
            let result = neoethos_search::run_discovery_cycle_with_holdout(
                &features,
                base_ohlcv,
                &config,
                neoethos_search::PropFirmRiskRules::default(),
            )?;
            (result, None)
        } else {
            let mut outcome = neoethos_search::orchestration::run_streaming_working_set(
                &neoethos_search::orchestration::StreamingPlan::streaming(stream_max_batches),
                base_ohlcv.close.len(),
                // The ONLY sanctioned build entry point: it installs the batch
                // as the working set and restores the previous one afterwards,
                // even on panic. `None` is documented as byte-identical to
                // `prepare_multitimeframe_features`.
                |batch| {
                    neoethos_data::prepare_multitimeframe_features_batch(
                        &dataset,
                        &base,
                        &higher_refs,
                        batch,
                    )
                },
                |features| {
                    neoethos_search::run_discovery_cycle_with_holdout(
                        features,
                        base_ohlcv,
                        &config,
                        neoethos_search::PropFirmRiskRules::default(),
                    )
                },
            )?;
            if outcome.batches.is_empty() {
                // Loud, and it NAMES every abandoned cursor. An empty sweep
                // that says only "no portfolio" is the silent drop with extra
                // steps.
                let rejected: Vec<(usize, &'static str)> = outcome
                    .ledger
                    .rejected_rows()
                    .iter()
                    .map(|row| (row.cursor, row.outcome.as_str()))
                    .collect();
                anyhow::bail!(
                    "streaming sweep produced no portfolio: {} batches attempted, {} abandoned \
                     (counts: {:?}; abandoned cursors: {:?}). The sweep covered pairs \
                     [0, {}) of {} at {} columns per batch. Nothing was lost silently — every \
                     cursor above has a reason in the run log \
                     (target=neoethos_search::batch_ledger).",
                    outcome.ledger.batches_seen(),
                    outcome.ledger.batches_rejected(),
                    outcome.ledger.counts_by_outcome(),
                    rejected,
                    outcome.next_cursor,
                    outcome.space_len,
                    outcome.batch_columns
                );
            }
            // Provenance for the run-level artifact is collected BEFORE the
            // primary batch is moved out for the per-run artifacts below.
            let survivors = outcome.survivors();
            let bundle = StreamingArtifactBundle {
                canonical: outcome.canonical.clone(),
                survivors,
                ledger: outcome.ledger.clone(),
                next_cursor: outcome.next_cursor,
                space_len: outcome.space_len,
                batch_columns: outcome.batch_columns,
                streamed: outcome.streamed,
            };
            // The FIRST surviving batch takes today's artifact paths, so a
            // single-batch sweep writes exactly the file set a non-streaming
            // run writes. Later batches are written beside it, keyed by cursor.
            let primary = outcome.batches.remove(0);
            let extra: Vec<(usize, neoethos_search::DiscoveryResult)> = outcome
                .batches
                .into_iter()
                .map(|batch| (batch.cursor, batch.result))
                .collect();
            (primary.result, Some((bundle, extra)))
        };
        if let Some(parent) = std::path::Path::new(&out).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        // F-306 fix (2026-05-28): save funnel + quality + trade log
        // BEFORE the empty-portfolio guard. The previous order made
        // every empty run leave ZERO artifacts on disk, blocking
        // post-mortem diagnosis. Mirrors the app server pattern at
        // `crates/neoethos-app/src/app_services/discovery.rs:988-1007`.
        let funnel_path = format!("{out}.funnel.json");
        if let Err(err) = neoethos_search::save_funnel_json(&funnel_path, &result) {
            tracing::warn!(
                target: "neoethos_cli::discover",
                error = %err,
                path = %funnel_path,
                "save_funnel_json failed (non-fatal — continuing to other artifacts)"
            );
        }
        if !result.quality_metrics.is_empty() {
            let quality_path = format!("{out}.quality.json");
            if let Err(err) = neoethos_search::save_quality_report_json(&quality_path, &result) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %quality_path,
                    "save_quality_report_json failed (non-fatal)"
                );
            }
        }
        if !result.logged_trades.is_empty() {
            let trade_log_path = format!("{out}.trades.json");
            if let Err(err) = neoethos_search::save_trade_log_json(&trade_log_path, &result) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %trade_log_path,
                    "save_trade_log_json failed (non-fatal)"
                );
            }
        }
        // Now the empty-portfolio guard. Diagnostics are already on
        // disk, so the operator can post-mortem even when this fires.
        neoethos_search::ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, base))?;
        neoethos_search::save_portfolio_json(&out, &result)?;
        // Phase 4 (2026-06-04): also emit the self-describing live portfolio
        // artifact (full genes + effective_feature_names + base/higher TFs +
        // normalize flag) the autonomous trader loads to evaluate the discovered
        // strategies with backtest parity. Additive + non-fatal.
        {
            let live_path = format!("{out}.live_portfolio.json");
            if let Err(err) = neoethos_search::save_live_portfolio_json(
                &live_path,
                &symbol,
                &base,
                &config.higher_timeframes,
                &result,
            ) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %live_path,
                    "save_live_portfolio_json failed (non-fatal)"
                );
            }
        }
        let profile_path = format!("{out}.profile.json");
        neoethos_search::save_discovery_profile_json(&profile_path, &config, &result)?;
        if !result.canonical_backtest_artifacts.is_empty() {
            let backtest_dir = format!("{out}.canonical_backtests");
            neoethos_search::save_canonical_backtest_artifacts(&backtest_dir, &result)?;
        }
        if !result.walkforward_validation_artifacts.is_empty() {
            let validation_dir = format!("{out}.walkforward_validations");
            neoethos_search::save_walkforward_validation_artifacts(&validation_dir, &result)?;
        }
        // ── The streaming run artifact ──────────────────────────────────────
        //
        // Written ONLY on a streaming run, and never instead of the per-batch
        // artifacts: each surviving batch keeps its own portfolio JSON, whose
        // genes address that batch's own `effective_feature_names` and are
        // therefore still internally consistent. THIS file is the run-level
        // view — Option C's canonical name list, the survivors remapped onto
        // it with the cursor that produced each one, and the batch census.
        if let Some((bundle, extra)) = streaming {
            for (cursor, batch_result) in &extra {
                let batch_out = format!("{out}.batch{cursor}.json");
                // Non-fatal, deliberately: `save_portfolio_json` runs the
                // export-readiness gate, and a later batch failing it must not
                // discard the batches already written. Nothing is lost — the
                // genes themselves are in the run-level artifact below, with
                // this cursor on them — but the failure is named, not swallowed.
                if let Err(err) = neoethos_search::save_portfolio_json(&batch_out, batch_result) {
                    tracing::warn!(
                        target: "neoethos_cli::discover",
                        error = %err,
                        cursor = *cursor,
                        path = %batch_out,
                        "per-batch portfolio export failed (non-fatal — this batch's survivors \
                         are still in the streaming run artifact)"
                    );
                }
            }
            let genes: Vec<neoethos_search::Gene> = bundle
                .survivors
                .iter()
                .map(|survivor| survivor.gene.clone())
                .collect();
            // INVARIANT 4 again, at the artifact boundary this time: nothing
            // leaves the process addressing a name that does not exist.
            bundle
                .canonical
                .assert_indices_in_range(&genes, "cli streaming run portfolio")?;
            let artifact = neoethos_search::orchestration::StreamingRunPortfolio {
                schema_version:
                    neoethos_search::orchestration::STREAMING_RUN_PORTFOLIO_SCHEMA_VERSION,
                symbol: symbol.clone(),
                base_timeframe: base.clone(),
                higher_timeframes: config.higher_timeframes.clone(),
                canonical_feature_names: bundle.canonical.names().to_vec(),
                survivors: bundle.survivors,
                next_cursor: bundle.next_cursor,
                space_len: bundle.space_len,
                batch_columns: bundle.batch_columns,
                ledger: bundle.ledger,
            };
            let streaming_path = format!("{out}.streaming.json");
            std::fs::write(&streaming_path, serde_json::to_string_pretty(&artifact)?)?;
            println!(
                "Streaming sweep streamed={} batches={} kept={} abandoned={} survivors={} \
                 canonical_features={} cursor={}/{} batch_columns={} out={}",
                bundle.streamed,
                artifact.ledger.batches_seen(),
                artifact.ledger.batches_kept(),
                artifact.ledger.batches_rejected(),
                artifact.survivors.len(),
                artifact.canonical_feature_names.len(),
                artifact.next_cursor,
                artifact.space_len,
                artifact.batch_columns,
                streaming_path
            );
        }
        println!(
            "Discovery {} portfolio={} candidates={} out={}",
            symbol,
            result.portfolio.len(),
            result.candidates.len(),
            out
        );
        Ok((
            symbol,
            base,
            result.portfolio.len(),
            result.candidates.len(),
        ))
    })();

    match &result {
        Ok((symbol, base, portfolio, candidates)) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "discover",
                    "SUCCESS",
                    format!(
                        "discovery completed for {} {} portfolio={} candidates={}",
                        symbol, base, portfolio, candidates
                    ),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "discover",
                    "FAILED",
                    format!("discovery failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

fn cmd_batch_discover(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, usize, usize)> {
        let settings = resolve_cli_settings(args)?;
        let root = parse_root(args, settings.as_ref());
        let symbols_raw = parse_flag(args, "--symbols").unwrap_or_default();
        let tfs_raw = parse_flag(args, "--timeframes")
            .unwrap_or_else(|| default_batch_timeframes_csv(settings.as_ref()));
        let out_dir =
            parse_flag(args, "--out-dir").unwrap_or_else(|| "cache/discovery".to_string());

        let symbols: Vec<String> = if symbols_raw.is_empty() {
            neoethos_data::discover_symbols(&root)?
        } else {
            symbols_raw
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .collect()
        };

        let tfs: Vec<String> = tfs_raw
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect();

        let mut config = settings
            .as_ref()
            .map(neoethos_search::DiscoveryConfig::from_settings)
            .unwrap_or_default();
        // Explicit overrides win over the config-derived values (same
        // precedence as env > config elsewhere). These let the TUI Discover
        // form's Population/Generations/Portfolio-size fields actually take
        // effect instead of being silently dropped (parity fix 2026-06-08).
        if let Some(p) = parse_flag(args, "--population")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            config.population = p;
        }
        if let Some(g) = parse_flag(args, "--generations")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            config.generations = g;
        }
        if let Some(ps) = parse_flag(args, "--portfolio-size")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            config.portfolio_size = ps;
        }
        let orchestrator = neoethos_search::DiscoveryOrchestrator::new(&root, &out_dir, config);

        let summary = orchestrator.run_batch(&symbols, &tfs)?;

        println!(
            "Batch discovery complete. Results in {} (saved={} work_units={} skipped_symbols={} skipped_timeframes={} feature_failures={} empty_portfolios={})",
            out_dir,
            summary.portfolios_saved,
            summary.work_units_seen,
            summary.skipped_symbols,
            summary.skipped_timeframes,
            summary.feature_failures,
            summary.empty_portfolios
        );
        Ok((out_dir, summary.portfolios_saved, summary.work_units_seen))
    })();

    match &result {
        Ok((out_dir, saved, work_units)) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "batch-discover",
                    "SUCCESS",
                    format!(
                        "batch discovery completed out_dir={} saved={} work_units={}",
                        out_dir, saved, work_units
                    ),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "batch-discover",
                    "FAILED",
                    format!("batch discovery failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

/// Recursive universal data importer — converts CSV/TSV/JSON/JSONL/
/// Parquet/Vortex files anywhere under `--source` into the canonical
/// `data/symbol={SYM}/timeframe={TF}/data.vortex` layout under
/// `--root`. Symbol/timeframe are inferred from path components or the
/// filename. Failed conversions are quarantined; the report is written
/// to `<root>/import_report.json`.
/// Print the resolved config: every setting with raw value, resolved
/// value, source (config / sentinel-expanded / env / default), and
/// notes. The TUI's Config page renders the same data.
fn cmd_config(args: &[String]) -> Result<()> {
    if args.first().map(String::as_str) == Some("normalize") {
        return cmd_config_normalize(&args[1..]);
    }
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);

    if has_flag(args, "--json") {
        let text = serde_json::to_string_pretty(&resolved)
            .map_err(|e| anyhow::anyhow!("serialize resolved config: {e}"))?;
        println!("{}", text);
        return Ok(());
    }

    println!("Resolved configuration");
    println!("======================");
    println!(
        "{:<10} {:<28} {:<28} {:<28} {:<8}",
        "section", "field", "raw", "resolved", "source"
    );
    println!("{}", "-".repeat(110));
    for row in resolved.display_table() {
        println!(
            "{:<10} {:<28} {:<28} {:<28} {:<8}",
            row[0], row[1], row[2], row[3], row[4]
        );
    }
    println!();
    println!("Notes:");
    for f in &resolved.display_fields {
        if let Some(note) = &f.note {
            println!("  {} / {}: {}", f.section, f.field, note);
        }
    }
    Ok(())
}

/// `config normalize` — show, and optionally rewrite, the operator's store as
/// an OVERRIDE DOCUMENT instead of a full snapshot.
///
/// Why this command exists. Older builds saved settings by dumping every field,
/// so a store written then is a photograph of that build's defaults. From that
/// moment it shadows every default the codebase improves — not by disagreeing
/// with them, but by repeating older ones that were never chosen. The operator
/// cannot see this by reading his file: a deliberate limit and a fossilised
/// default are the same line of YAML. The measured case on this machine was a
/// 509-line store in which seven gates sat at values no one had selected,
/// including a payoff floor of 0.0 against a default of 2.0 and an export gate
/// disabled against a default that requires walk-forward.
///
/// The fix is not to patch values — that is the same mistake with fresher
/// numbers. It is to make the file carry ONLY what diverges, so that every
/// future default arrives on its own. `Settings::save` already writes that
/// shape; nothing called it on an existing store, so no store was ever
/// converted.
///
///   config normalize            Print each divergence beside the default it
///                               shadows. Reads nothing else, writes nothing.
///   config normalize --write    Back the store up, rewrite it as overrides,
///                               reload it, and REFUSE (restoring the backup)
///                               unless the reloaded settings are identical.
fn cmd_config_normalize(args: &[String]) -> Result<()> {
    let write = has_flag(args, "--write");
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let provenance = settings.provenance().describe();
    let path = settings.provenance().path().map(std::path::Path::to_path_buf);
    let before = settings.overrides_against_defaults()?;

    println!("Config store: {provenance}");
    if let Some(p) = &path {
        let lines = std::fs::read_to_string(p)
            .map(|t| t.lines().count())
            .unwrap_or(0);
        println!("On disk     : {lines} lines");
    }
    println!(
        "Overrides   : {} ({} diverge from default, {} money keys carried by rule)",
        before.len(),
        before.iter().filter(|o| o.diverges).count(),
        before.iter().filter(|o| o.money_key).count()
    );
    println!();
    println!("{:<46} {:<32} {:<32} {}", "key", "your file", "default", "");
    println!("{}", "-".repeat(120));
    for o in &before {
        let default = o.default.as_deref().unwrap_or("<not in schema>");
        let mark = match (o.money_key, o.diverges, o.default.is_none()) {
            (_, _, true) => "LEFTOVER",
            (true, false, _) => "money (= default)",
            (true, true, _) => "MONEY",
            (false, true, _) => "",
            (false, false, _) => "",
        };
        println!(
            "{:<46} {:<32} {:<32} {}",
            truncate(&o.path, 46),
            truncate(&o.live, 32),
            truncate(default, 32),
            mark
        );
    }
    println!();

    if !write {
        println!(
            "Nothing written. Re-run with --write to convert the store to this \
             shape; every key not listed above then follows the compiled default \
             instead of a frozen copy of it."
        );
        return Ok(());
    }

    let Some(path) = path else {
        anyhow::bail!(
            "no config file to normalize — this process resolved to the compiled \
             defaults ({provenance}). There is nothing on disk to convert."
        );
    };

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("read the clock: {e}"))?
        .as_secs();
    let backup = path.with_extension(format!("yaml.pre-normalize-{stamp}"));
    std::fs::copy(&path, &backup)
        .map_err(|e| anyhow::anyhow!("back up {} to {}: {e}", path.display(), backup.display()))?;
    println!("Backup      : {}", backup.display());

    settings.save(&path)?;

    // Refuse to leave a rewritten store in place unless it reloads to the same
    // settings. A conversion that changes behaviour is the exact failure this
    // command is meant to end, so it is checked rather than assumed — and on
    // any mismatch the operator's original file is put back before we return.
    let restore = |why: String| -> anyhow::Error {
        let _ = std::fs::copy(&backup, &path);
        anyhow::anyhow!("{why}\nThe original store has been restored from {}.", backup.display())
    };
    let reloaded = neoethos_core::Settings::from_yaml(&path)
        .map_err(|e| restore(format!("the normalized store failed to load: {e}")))?;
    let after = reloaded
        .overrides_against_defaults()
        .map_err(|e| restore(format!("the normalized store could not be re-read: {e}")))?;
    if after != before {
        let mut diff = Vec::new();
        for o in &before {
            if !after.contains(o) {
                diff.push(format!("  lost:    {} = {}", o.path, o.live));
            }
        }
        for o in &after {
            if !before.contains(o) {
                diff.push(format!("  gained:  {} = {}", o.path, o.live));
            }
        }
        return Err(restore(format!(
            "the normalized store does not round-trip — {} change(s):\n{}",
            diff.len(),
            diff.join("\n")
        )));
    }

    let lines = std::fs::read_to_string(&path)
        .map(|t| t.lines().count())
        .unwrap_or(0);
    println!("Written     : {} ({lines} lines, was a full snapshot)", path.display());
    println!("Verified    : reloads to identical settings; every other key now follows the default.");
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Auto search-train loop (P9). Forward-only:
///   import → discover → train → export → next (symbol, timeframe)
///
/// Controls:
///   --symbols X,Y,Z         (default: auto-detect from data root)
///   --timeframes M3,M5,...  (default: ResolvedConfig.timeframes.canonical_default)
///   --skip-training         (run discover + export only)
///   --max-jobs N            (stop after N work-units, 0 = no limit)
///   --resume                (continue from cache/auto_loop_checkpoint.json)
///   --stop-flag PATH        (file whose existence stops the loop after current job)
///
/// Persists checkpoint to cache/auto_loop_checkpoint.json — on crash,
/// re-run with --resume to continue.
/// Multi-GPU / hybrid scheduler driver.
///
/// Enumerates symbol×TF combos, asks `scheduler::plan_combo` how each should be
/// admitted, then runs single-card / CPU combos across all cards concurrently —
/// each as a subprocess `discover` pinned via
/// `NEOETHOS_BOT_SEARCH_EVAL_{WGPU,CUDA}_DEVICE`. Oversized populations are
/// reported as requiring single-device chunking; they are never described as
/// cross-card shards that the worker does not execute.
///
/// `--dry-run` prints the full admission plan WITHOUT spawning — the way to
/// validate the decisions against a real hardware probe + real on-disk data
/// sizes (works on any machine, no GPU needed).
fn cmd_schedule(args: &[String]) -> Result<()> {
    use neoethos_core::scheduler::{
        AdmissionPolicy, ComboItem, ComboShape, WorkScheduler, plan_combo,
    };

    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);
    let root = parse_root(args, Some(&settings));
    let dry_run = has_flag(args, "--dry-run");
    // Stage 1 schedules DISCOVERY only; training co-scheduling on separate
    // cards is Stage 3, so there is no `--skip-training` knob here yet.
    let resume = has_flag(args, "--resume");
    let stop_flag =
        parse_flag(args, "--stop-flag").unwrap_or_else(|| "cache/schedule_stop.flag".to_string());
    // feature-cube column count is hard to know without building the cube; a
    // conservative estimate is fine because the planner's safety margins
    // (0.75 RAM / 0.80 VRAM / 2× overhead) absorb the error. Overridable.
    let feature_count: usize = parse_flag(args, "--features")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    // bytes-per-bar for the load-free row estimate (vortex-compressed OHLCV).
    let bytes_per_bar: f64 = parse_flag(args, "--bytes-per-bar")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12.0);
    let population = resolved.search.population;

    // Set the data root for any child orchestrator BEFORE any thread spawns.
    unsafe {
        std::env::set_var("NEOETHOS_BOT_DATA_ROOT", &root);
    }

    let symbols: Vec<String> = match parse_flag(args, "--symbols") {
        Some(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_uppercase()).collect(),
        _ => neoethos_data::discover_symbols(&root)?,
    };
    let tfs: Vec<String> = parse_flag(args, "--timeframes")
        .unwrap_or_else(|| resolved.timeframes.canonical_default.join(","))
        .split(',')
        .map(|x| x.trim().to_uppercase())
        .collect();

    let mut probe = neoethos_core::system::HardwareProbe::new();
    let hw = probe.detect();
    let policy = AdmissionPolicy::default();
    println!(
        "Hardware: {} cores, {:.0}GB RAM avail, {} detected GPU(s) names={:?} VRAM={:?}GB",
        hw.cpu_cores, hw.available_ram_gb, hw.num_gpus, hw.gpu_names, hw.gpu_mem_gb
    );

    // Build admission plans for the runtime's actual one-device-per-worker
    // contract. Oversized populations remain schedulable only because the
    // evaluator chunks them on that same device.
    let mut id_combo: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    let mut schedulable: Vec<ComboItem> = Vec::new();
    for sym in &symbols {
        for tf in &tfs {
            let id = format!("{sym}/{tf}");
            let rows = estimate_series_rows(&root, sym, tf, bytes_per_bar);
            if rows == 0 {
                println!("  skip {id}: no vortex data found on disk");
                continue;
            }
            let shape = ComboShape::new(rows, population, feature_count);
            let plan = plan_combo(shape, &hw, &policy);
            id_combo.insert(id.clone(), (sym.clone(), tf.clone()));
            schedulable.push(ComboItem::new(id, shape, plan));
        }
    }

    println!(
        "\n=== Admission plan: {} schedulable ===",
        schedulable.len()
    );
    for it in &schedulable {
        let tag = if it.plan.cards_per_combo == 0 {
            "  [CPU lane]"
        } else if !it.plan.fits_on_gpu {
            "  [does NOT fit on one card — chunk]"
        } else {
            ""
        };
        println!(
            "  [{:?}] {:<14} rows≈{:>9}  assigned_cards={} population/device={} cpu={} RAM≈{:.1}GB VRAM/device≈{:.1}GB{}",
            it.plan.class,
            it.id,
            it.shape.series_rows,
            it.plan.cards_per_combo,
            it.plan.genes_per_card,
            it.plan.cpu_threads_per_worker,
            it.plan.est_ram_per_combo_gb,
            it.plan.est_vram_per_card_gb,
            tag
        );
    }
    if dry_run {
        println!("\n--dry-run: plan only, no processes spawned.");
        return Ok(());
    }

    let checkpoint_path = std::path::PathBuf::from("cache").join("schedule_checkpoint.json");
    let mut completed: Vec<String> = Vec::new();
    if resume && checkpoint_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&checkpoint_path) {
            if let Ok(prev) = serde_json::from_str::<ScheduleCheckpoint>(&text) {
                completed = prev.completed.clone();
                schedulable.retain(|it| !completed.contains(&it.id));
                println!(
                    "Resuming: {} already done; {} remaining",
                    completed.len(),
                    schedulable.len()
                );
            }
        }
    }

    let mut sched = WorkScheduler::new(schedulable, &hw, &policy);
    let mut children: std::collections::HashMap<String, std::process::Child> =
        std::collections::HashMap::new();
    let mut failed: Vec<String> = Vec::new();
    let total = sched.pending_len();
    println!(
        "\nScheduling {} combos across {} card(s)...",
        total,
        sched.total_cards()
    );

    loop {
        let stopping = std::path::Path::new(&stop_flag).exists();
        if stopping && children.is_empty() {
            println!("Stop-flag found and no in-flight work — exiting.");
            break;
        }
        if !stopping {
            for a in sched.poll() {
                let (sym, tf) = id_combo.get(&a.id).cloned().unwrap_or_default();
                match spawn_discover_combo(&a, &sym, &tf, &root, &resolved, &hw) {
                    Ok(child) => {
                        println!(
                            "  ▶ {} on cards {:?} (population/device {}, {} cpu threads)",
                            a.id, a.card_ids, a.genes_per_card, a.cpu_threads
                        );
                        children.insert(a.id.clone(), child);
                    }
                    Err(e) => {
                        eprintln!("  spawn {} failed: {e:#}", a.id);
                        sched.complete(&a.id);
                        failed.push(a.id.clone());
                    }
                }
            }
        }
        if children.is_empty() {
            if sched.is_done() || stopping {
                break;
            }
            // Nothing running and nothing dispatchable — avoid a spin loop.
            eprintln!(
                "scheduler stalled with {} pending and no free capacity — stopping",
                sched.pending_len()
            );
            break;
        }

        let ids: Vec<String> = children.keys().cloned().collect();
        let mut any_done = false;
        for id in ids {
            let finished = match children.get_mut(&id).map(|c| c.try_wait()) {
                Some(Ok(Some(status))) => Some(status),
                Some(Err(e)) => {
                    eprintln!("  wait {id}: {e}");
                    None
                }
                _ => None,
            };
            if let Some(status) = finished {
                any_done = true;
                children.remove(&id);
                sched.complete(&id);
                if status.success() {
                    completed.push(id.clone());
                    println!("  ✔ {id} done ({}/{})", completed.len(), total);
                    write_schedule_checkpoint(&checkpoint_path, &completed);
                } else {
                    // Stage 1 has no CPU lane yet (that is Stage 3), so a GPU
                    // failure is logged and resources freed — NOT retried in a
                    // loop. The OOM->CPU requeue path is exercised once Stage 3
                    // lands a real fast-CPU fallback.
                    failed.push(id.clone());
                    eprintln!(
                        "  ✗ {id} FAILED (exit {:?}) — freed; not retried (CPU lane is Stage 3)",
                        status.code()
                    );
                }
            }
        }
        if !any_done {
            std::thread::sleep(std::time::Duration::from_millis(750));
        }
    }

    println!(
        "\nSchedule done: {} completed, {} failed.",
        completed.len(),
        failed.len()
    );
    if !failed.is_empty() {
        println!("  failed: {failed:?}");
    }
    Ok(())
}

/// Cheap, load-free estimate of the bar count for a symbol/timeframe from the
/// on-disk vortex file size. The orchestrator must NOT materialise data (M1 is
/// ~80GB expanded), so we estimate from bytes. Feeds the admission planner,
/// whose conservative margins + the subprocess's exact load make the estimate
/// non-critical to correctness.
fn estimate_series_rows(root: &str, symbol: &str, tf: &str, bytes_per_bar: f64) -> usize {
    let path = neoethos_data::symbol_timeframe_vortex_path(root, symbol, tf);
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) as f64;
    if bytes <= 0.0 {
        return 0;
    }
    (bytes / bytes_per_bar.max(1.0)).ceil() as usize
}

/// Spawn one `discover` subprocess pinned to its assigned card. The planner and
/// worker both use the single-device contract (`card_ids[0]`).
fn spawn_discover_combo(
    a: &neoethos_core::scheduler::Assignment,
    symbol: &str,
    timeframe: &str,
    root: &str,
    resolved: &neoethos_core::resolved_config::ResolvedConfig,
    hardware: &neoethos_core::system::HardwareProfile,
) -> Result<std::process::Child> {
    let exe = std::env::current_exe().context("locating current executable")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("discover")
        .arg("--symbol")
        .arg(symbol)
        .arg("--base")
        .arg(timeframe)
        .arg("--root")
        .arg(root)
        .arg("--population")
        .arg(resolved.search.population.to_string())
        .arg("--generations")
        .arg(resolved.search.generations.to_string())
        .arg("--portfolio-size")
        .arg(resolved.search.portfolio_size.to_string())
        .arg("--out")
        .arg(format!("cache/schedule/{symbol}_{timeframe}.json"))
        // The child's CPU budget travels as an ARGUMENT, not as
        // `NEOETHOS_BOT_CPU_BUDGET`. An env var here was indistinguishable
        // from an operator knob and outlived the invocation it was meant for;
        // an argument is scoped to this child and visible in the process list.
        // Read back in `main()` via `parse_flag(&raw_args, "--cpu-threads")`.
        .arg("--cpu-threads")
        .arg(a.cpu_threads.to_string());
    // NOTE: intra-combo GPU sharding (the plural *_DEVICES env) is DISABLED at
    // the dispatch layer — the cubecl wgpu multi-device path panics
    // at runtime ("Memory page 0 doesn't exist" in cubecl-runtime client.rs) and
    // falls back to slow CPU recompute. Validated 2026-06-07 on the 2×A6000 VPS:
    // M1 ran 58 min on CPU (gen 305/20000, 0 results) before we caught it. Until
    // that cubecl multi-device issue is fixed, every combo runs on the PROVEN
    // single-device path, pinned to its first assigned card (H4 validated: card
    // at 120 MiB, clean run). Combo-level throughput (one combo per card,
    // concurrent) is preserved; the multi-device code in eval.rs stays in place,
    // gated behind the plural env which we simply no longer set.
    for (key, value) in gpu_assignment_env(a, hardware) {
        cmd.env(key, value);
    }
    // `NEOETHOS_BOT_DATA_ROOT` still travels as an env var because its reader
    // is `neoethos-models/src/training_orchestrator.rs:329`, in a crate this
    // change does not own. The `--root` argument above already carries the same
    // value to the search; the env copy exists only for the in-process trainer.
    // Collapsing it is a neoethos-models edit, routed, not done here.
    cmd.env("NEOETHOS_BOT_DATA_ROOT", root);
    cmd.spawn()
        .with_context(|| format!("spawning discover for {symbol}/{timeframe}"))
}

fn gpu_assignment_env(
    assignment: &neoethos_core::scheduler::Assignment,
    hardware: &neoethos_core::system::HardwareProfile,
) -> std::collections::BTreeMap<&'static str, String> {
    use neoethos_core::system::AcceleratorBackend;

    let mut envs = std::collections::BTreeMap::new();
    let Some(&slot) = assignment.card_ids.first() else {
        return envs;
    };
    if let Some(device) = hardware.accelerator_devices.get(slot) {
        match device.backend {
            AcceleratorBackend::Cuda => {
                envs.insert(
                    "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE",
                    device.backend_index.to_string(),
                );
            }
            backend if backend.is_wgpu_family() => {
                if let Some(selector) = device.cubecl_wgpu_selector() {
                    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE", selector);
                }
            }
            AcceleratorBackend::Cpu | AcceleratorBackend::Rocm => {}
            _ => {}
        }
        return envs;
    }

    // Backwards compatibility for synthetic/legacy profiles that contain only
    // `gpu_mem_gb` and therefore cannot describe a backend or adapter class.
    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE", slot.to_string());
    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE", slot.to_string());
    envs
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ScheduleCheckpoint {
    updated_at: String,
    completed: Vec<String>,
}

fn write_schedule_checkpoint(path: &std::path::Path, completed: &[String]) {
    let ck = ScheduleCheckpoint {
        updated_at: chrono::Utc::now().to_rfc3339(),
        completed: completed.to_vec(),
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&ck) {
        let _ = std::fs::write(path, text);
    }
}

fn cmd_auto_loop(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);
    let root = parse_root(args, Some(&settings));

    // Set NEOETHOS_BOT_DATA_ROOT for the in-process training orchestrator
    // (cmd_train doesn't honor --root yet, see training_orchestrator.rs).
    // SAFETY: called before any thread spawn — we are still in
    // single-threaded init here (setup_logging and the search-runtime
    // overrides installer above only mutate tracing/global config; rayon
    // and tokio threads are not started until cmd_discover/cmd_train run,
    // which happen below). Per std::env::set_var docs, on Linux/macOS the
    // ONLY safe option is to mutate env before any other thread exists;
    // doing this inside the per-symbol loop would race with rayon worker
    // threads spawned by the prior cmd_discover call.
    unsafe {
        std::env::set_var("NEOETHOS_BOT_DATA_ROOT", &root);
    }
    let symbols_raw = parse_flag(args, "--symbols").unwrap_or_default();
    let tfs_raw = parse_flag(args, "--timeframes")
        .unwrap_or_else(|| resolved.timeframes.canonical_default.join(","));
    let skip_training = has_flag(args, "--skip-training");
    let max_jobs: usize = parse_flag(args, "--max-jobs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let resume = has_flag(args, "--resume");
    let stop_flag =
        parse_flag(args, "--stop-flag").unwrap_or_else(|| "cache/auto_loop_stop.flag".to_string());
    let checkpoint_path = std::path::PathBuf::from("cache").join("auto_loop_checkpoint.json");

    let symbols: Vec<String> = if symbols_raw.is_empty() {
        neoethos_data::discover_symbols(&root)?
    } else {
        symbols_raw
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    };
    let tfs: Vec<String> = tfs_raw
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .collect();

    // Build the (symbol, timeframe) work queue.
    let mut work_queue: Vec<(String, String)> = symbols
        .iter()
        .flat_map(|s| tfs.iter().map(move |t| (s.clone(), t.clone())))
        .collect();
    let total_units = work_queue.len();

    // Resume support: read checkpoint and skip already-completed pairs.
    let mut completed: Vec<(String, String)> = Vec::new();
    if resume && checkpoint_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&checkpoint_path) {
            if let Ok(prev) = serde_json::from_str::<AutoLoopCheckpoint>(&text) {
                completed = prev.completed.clone();
                work_queue.retain(|w| !completed.contains(w));
                println!(
                    "Resuming from checkpoint: {} already completed; {} remaining",
                    completed.len(),
                    work_queue.len()
                );
            }
        }
    }

    let mut jobs_run = 0usize;
    println!(
        "Auto-loop start: {} work units ({} symbols × {} timeframes); skip_training={}; stop-flag={}",
        work_queue.len(),
        symbols.len(),
        tfs.len(),
        skip_training,
        stop_flag
    );

    for (sym, tf) in work_queue.into_iter() {
        if std::path::Path::new(&stop_flag).exists() {
            println!("Stop-flag found at {} — exiting loop", stop_flag);
            break;
        }
        if max_jobs > 0 && jobs_run >= max_jobs {
            println!("Reached --max-jobs={}; exiting", max_jobs);
            break;
        }

        println!(
            "[{}/{}] discovering {} {}",
            jobs_run + 1,
            total_units,
            sym,
            tf
        );
        let discover_args: Vec<String> = vec![
            "discover".to_string(),
            "--symbol".to_string(),
            sym.clone(),
            "--base".to_string(),
            tf.clone(),
            // 2026-06-04 parity: no hardcoded `--higher H4`. Omitting `--higher`
            // lets cmd_discover resolve the higher-TF ladder from config relative
            // to THIS base (`tf`) via the shared `resolve_higher_timeframes`, so
            // every base in the sweep gets its correct top-down context — same as
            // a standalone `discover` run (H4-as-base no longer self-references).
            "--root".to_string(),
            root.clone(),
            "--population".to_string(),
            resolved.search.population.to_string(),
            "--generations".to_string(),
            resolved.search.generations.to_string(),
            "--portfolio-size".to_string(),
            resolved.search.portfolio_size.to_string(),
            "--out".to_string(),
            format!("cache/auto_loop/{}_{}.json", sym, tf),
        ];
        let discover_ok = match cmd_discover(&discover_args) {
            Ok(()) => {
                println!("  discover OK");
                true
            }
            Err(err) => {
                eprintln!("  discover FAILED: {err:#}");
                // Continue to next; don't bail the whole loop.
                false
            }
        };

        let mut train_ok = true;
        if !skip_training {
            // NEOETHOS_BOT_DATA_ROOT was set at the top of `cmd_auto_loop`
            // before any thread spawned; cmd_train reads it via
            // training_orchestrator::train_symbol.
            let train_args: Vec<String> = vec![
                "train".to_string(),
                "--symbol".to_string(),
                sym.clone(),
                "--base".to_string(),
                tf.clone(),
                "--models-dir".to_string(),
                "cache/auto_loop_models".to_string(),
            ];
            match cmd_train(&train_args) {
                Ok(()) => println!("  train OK"),
                Err(err) => {
                    eprintln!("  train FAILED: {err:#}");
                    train_ok = false;
                }
            }
        }

        // Audit B14 (2026-07-13): mark this combo COMPLETE only when its
        // stages actually succeeded, so `--resume` RETRIES failed work rather
        // than silently skipping it. Previously `completed.push` ran
        // unconditionally, so a transient discovery/training failure (OOM, bad
        // data, a crash mid-run) was checkpointed as "done" and the user
        // permanently lost those strategies on resume.
        if discover_ok && train_ok {
            completed.push((sym.clone(), tf.clone()));
        } else {
            eprintln!(
                "  [{sym} {tf}] NOT marked complete (discover_ok={discover_ok}, \
                 train_ok={train_ok}) — will retry on --resume"
            );
        }
        let checkpoint = AutoLoopCheckpoint {
            started_at: completed
                .first()
                .map(|_| chrono::Utc::now().to_rfc3339())
                .unwrap_or_default(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            completed: completed.clone(),
            remaining: total_units.saturating_sub(completed.len()),
        };
        if let Some(dir) = checkpoint_path.parent() {
            if let Err(err) = std::fs::create_dir_all(dir) {
                tracing::warn!(
                    target: "neoethos_cli",
                    dir = %dir.display(),
                    error = %err,
                    "auto_loop: failed to create checkpoint directory"
                );
            }
        }
        match serde_json::to_string_pretty(&checkpoint) {
            Ok(text) => {
                if let Err(err) = std::fs::write(&checkpoint_path, text) {
                    tracing::warn!(
                        target: "neoethos_cli",
                        path = %checkpoint_path.display(),
                        error = %err,
                        "auto_loop: failed to write checkpoint"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_cli",
                    error = %err,
                    "auto_loop: failed to serialize checkpoint"
                );
            }
        }

        jobs_run += 1;
    }

    println!(
        "Auto-loop done: {}/{} work units processed; checkpoint at {}",
        completed.len(),
        total_units,
        checkpoint_path.display()
    );
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AutoLoopCheckpoint {
    started_at: String,
    updated_at: String,
    completed: Vec<(String, String)>,
    remaining: usize,
}

fn cmd_import(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let source = parse_flag(args, "--source").unwrap_or_else(|| root.clone());
    let force = has_flag(args, "--force");

    // Folder-browse support (2026-05-14): when the operator points
    // `--data-path` at a folder, scan it and print a summary so they
    // can confirm the layout before any conversion runs. `--dry-run`
    // exits after the summary.
    if has_flag(args, "--data-path") || has_flag(args, "--dry-run") {
        let _ = print_dataset_discovery_summary(&source)?;
        if has_flag(args, "--dry-run") {
            return Ok(());
        }
    }

    let report =
        neoethos_data::core::universal_importer::import_directory_recursive(&source, &root, force)?;

    let report_path = std::path::PathBuf::from(&root).join("import_report.json");
    if let Err(err) = report.save_to_disk(&report_path) {
        tracing::warn!(
            target: "neoethos_cli",
            path = %report_path.display(),
            error = %err,
            "universal import: failed to save report"
        );
    }

    println!(
        "Universal import: source={} root={} files_seen={} imported={} skipped={} quarantined={} failed={}",
        source,
        root,
        report.files_seen,
        report.imported,
        report.skipped,
        report.quarantined,
        report.failed
    );
    println!("  full report: {}", report_path.display());
    for r in report.results.iter().take(20) {
        println!(
            "  [{:?}] {} -> {} rows ({})",
            r.status, r.source, r.rows, r.message
        );
    }
    if report.results.len() > 20 {
        println!("  ... ({} more in report)", report.results.len() - 20);
    }
    Ok(())
}

fn cmd_migrate_data(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let force = has_flag(args, "--force");
    let delete_source = has_flag(args, "--delete-source");
    let summary = neoethos_data::migrate_legacy_parquet_tree(&root, force, delete_source)?;

    println!(
        "Vortex migration root={} converted={} skipped={} failed={}",
        root,
        summary.total_converted(),
        summary.total_skipped(),
        summary.total_failed()
    );

    for record in &summary.converted {
        println!(
            "  converted {} {} rows={} -> {}",
            record.job.symbol,
            record.job.timeframe,
            record.rows,
            record.job.vortex_path.display()
        );
    }
    for record in &summary.skipped {
        println!(
            "  skipped {} {} rows={} -> {}",
            record.job.symbol,
            record.job.timeframe,
            record.rows,
            record.job.vortex_path.display()
        );
    }
    for failure in &summary.failed {
        println!(
            "  failed {} {} -> {} ({})",
            failure.job.symbol,
            failure.job.timeframe,
            failure.job.parquet_path.display(),
            failure.error
        );
    }

    if summary.total_failed() > 0 {
        anyhow::bail!(
            "vortex migration completed with {} failed datasets",
            summary.total_failed()
        );
    }

    Ok(())
}

fn cmd_stop_target(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe =
        parse_flag(args, "--timeframe").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let pip_size: f64 = parse_flag(args, "--pip")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0001);
    let signal: i8 = parse_flag(args, "--signal")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let ohlcv = neoethos_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    let settings = neoethos_search::StopTargetSettings::default();
    let result = neoethos_search::infer_stop_target_pips(
        &ohlcv.open,
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        &settings,
        pip_size,
        signal,
    );
    if let Some((sl, tp, rr)) = result {
        println!(
            "Stop/Target {} {}: SL={:.2} pips TP={:.2} pips RR={:.2}",
            symbol, timeframe, sl, tp, rr
        );
    } else {
        println!("Stop/Target {} {}: insufficient data", symbol, timeframe);
    }
    Ok(())
}

/// `neoethos-cli wizard` — TUI counterpart of the desktop first-run
/// wizard. Spec §8 (`installer_wizard_ux_spec.md`).
fn cmd_wizard(_args: &[String]) -> Result<()> {
    tui::run_wizard_tui()
}

/// `neoethos-cli setup` — Task #61 headless setup helper. Closes the
/// CLI parity gap: prints canonical credentials paths, shows which
/// config files exist on disk, and emits ready-to-paste TOML / JSON
/// templates for the operator to scp into place on a headless host.
///
/// Sub-modes:
///   `neoethos-cli setup`             — same as `setup show`
///   `neoethos-cli setup show`        — list expected paths + existence
///   `neoethos-cli setup ctrader`     — print broker_credentials.toml template
///   `neoethos-cli setup paths`       — print just the canonical directories
///
/// We intentionally do NOT write binary state here — the on-disk
/// schemas live in `neoethos-app::app_services` which the CLI crate
/// can't depend on (creates a cycle). Operators paste the template
/// into the canonical path manually OR drive the egui wizard once
/// on a desktop and `scp` the resulting `broker_credentials.toml`
/// to the headless host.
fn cmd_setup(args: &[String]) -> Result<()> {
    let mode = args.first().map(String::as_str).unwrap_or("show");
    match mode {
        "show" => setup_show(),
        "ctrader" => setup_ctrader_template(),
        "paths" => setup_paths(),
        "--help" | "-h" | "help" => {
            setup_help();
            Ok(())
        }
        other => {
            eprintln!(
                "neoethos-cli setup: unknown sub-mode '{other}'. \
                 Try 'neoethos-cli setup --help'."
            );
            setup_help();
            Ok(())
        }
    }
}

fn setup_help() {
    println!("neoethos-cli setup — headless credentials helper");
    println!();
    println!("USAGE:");
    println!("    neoethos-cli setup [SUBCOMMAND]");
    println!();
    println!("SUBCOMMANDS:");
    println!("    show       List expected config paths + which already exist (default)");
    println!("    paths      Print just the canonical directories, one per line (scripting)");
    println!("    ctrader    Emit a broker_credentials.toml template for the cTrader broker");
    println!();
    println!("    The CLI does NOT write binary state — paste the template into the canonical");
    println!("    path printed by `setup paths`. Drive the egui wizard once on a desktop if you");
    println!("    prefer a graphical flow, then `scp` the resulting `broker_credentials.toml`.");
}

/// Canonical user-config directory — matches the resolution in
/// `neoethos-app::broker_persistence::credentials_file_path` exactly so
/// `neoethos-cli setup` prints the same paths the GUI writes to.
/// Order: env override → `dirs::config_dir()/neoethos` → `.local/neoethos`.
fn canonical_user_config_dir() -> std::path::PathBuf {
    // Test-seam env var: matches `BROKER_CREDENTIALS_PATH_ENV_VAR` in
    // neoethos-app so an operator running a sandboxed CLI session sees
    // the same override path the GUI does.
    // **F-CORE3 closure (2026-05-25)**: routed through the canonical
    // `neoethos_core::env_overrides::broker_credentials_path_override`
    // typed getter — single grep-able source for the env-var name.
    if let Some(custom) = neoethos_core::env_overrides::broker_credentials_path_override()
        && let Some(parent) = std::path::Path::new(&custom).parent()
    {
        return parent.to_path_buf();
    }
    if let Some(config_dir) = dirs::config_dir() {
        return config_dir.join("neoethos");
    }
    // Last-resort dev-machine fallback — mirrors the third candidate
    // in `neoethos-app::broker_persistence::candidate_paths`.
    std::path::PathBuf::from(".local/neoethos")
}

fn setup_show() -> Result<()> {
    let config_dir = canonical_user_config_dir();
    println!("NeoEthos headless setup status");
    println!("==============================");
    println!();
    println!("Canonical config directory:");
    println!("  {}", config_dir.display());
    if !config_dir.exists() {
        println!("    ! directory does not exist yet — `mkdir -p` it before pasting templates");
    }
    println!();
    let entries: &[(&str, &str)] = &[
        (
            "broker_credentials.toml",
            "cTrader OAuth credentials (client_id, redirect_uri, accounts, environment)",
        ),
        (
            "risky_mode_state.json",
            "Risky Mode arm + ack ledger (written by the desktop wizard's Apply step)",
        ),
        (
            "wizard_state.json",
            "Wizard completion sentinel + per-step status (resume-from-disk hint)",
        ),
        (
            "risk_acknowledgement.json",
            "Append-only ledger of the 5-question risk-quiz acknowledgements (Task #68)",
        ),
    ];
    println!("Expected files:");
    for (name, description) in entries {
        let path = config_dir.join(name);
        let mark = if path.exists() { "✓" } else { "·" };
        println!("  [{}] {}", mark, path.display());
        println!("      {}", description);
    }
    println!();
    println!("Run `neoethos-cli setup ctrader` for a paste-ready cTrader template.");
    Ok(())
}

fn setup_paths() -> Result<()> {
    let dir = canonical_user_config_dir();
    println!("{}", dir.display());
    Ok(())
}

fn setup_ctrader_template() -> Result<()> {
    let dir = canonical_user_config_dir();
    let path = dir.join("broker_credentials.toml");
    println!("# Paste this into:");
    println!("#   {}", path.display());
    println!("# Replace the placeholder values with the credentials from the cTrader Open API");
    println!("# Developer Portal (https://openapi.ctrader.com). For accounts with");
    println!("# `enabled_for_execution = true`, the bot will route orders. Leaving the");
    println!("# array empty is fine — the GUI's account-discovery step populates it.");
    println!();
    println!("schema_version = 1");
    println!();
    println!("[ctrader]");
    println!("environment = \"Demo\"  # or \"Live\" — match the cTrader account's tier");
    println!("client_id = \"<your cTrader app client_id>\"");
    println!("client_secret = \"<your cTrader app client_secret>\"");
    println!("redirect_uri = \"http://127.0.0.1:43001/callback\"");
    println!("accounts = []");
    Ok(())
}

// REMOVED 2026-08-09: `setup news` printed a `news_api.toml` template and told
// the operator that "the news API key drives ... [the] blackout filter".
// Nothing in this repo has ever read `news_api.toml`, and batch D3 deleted
// `neoethos_core::domain::news_filter` — the only type that ever held a news
// API key — together with the `news.perplexity_enabled`,
// `news.news_lookahead_minutes` and `news.news_kill_window_min` knobs.
//
// The LIVE news gate is `neoethos_app::app_services::news_calendar`, whose
// blackout window is HARDCODED (15 min before / 10 min after) and whose only
// operator knob is `news.news_trading_mode`. If that window should become
// settable, add ONE knob and give it a reader in `news_calendar` — do not
// restore a command that advertises a capability with no implementation.

fn parse_root(args: &[String], settings: Option<&neoethos_core::Settings>) -> String {
    let root = resolve_root(args, settings);
    // Point the FX resolver at whatever store this command decided on, so a
    // `--root` run converts cross-pair pip values against that store rather
    // than the one named in config.yaml.
    neoethos_search::fx_rates::set_store_root(&root);
    root
}

fn resolve_root(args: &[String], settings: Option<&neoethos_core::Settings>) -> String {
    // `--data-path` is the operator-facing flag added 2026-05-14 for
    // folder-browsing workflows; `--root` remains for backwards
    // compatibility with existing scripts. `--data-path` wins when
    // both are supplied because it's the more explicit name.
    if let Some(p) = parse_flag(args, "--data-path") {
        return p;
    }
    parse_flag(args, "--root").unwrap_or_else(|| {
        settings
            .map(|settings| settings.system.data_dir.to_string_lossy().to_string())
            .unwrap_or_else(|| "data".to_string())
    })
}

/// Run `DatasetDiscovery::scan` on the supplied root and print a
/// human-readable summary table to stdout. Returns the report so the
/// caller can react (e.g. honour `--dry-run`).
///
/// Shell-completion hint: when this codebase migrates to clap-derive,
/// the `--data-path` argument should be annotated with
/// `value_hint = clap::ValueHint::DirPath` so shells that respect the
/// hint can complete directory paths. Today the CLI uses manual arg
/// parsing, so the hint is documented here as a future-work marker.
fn print_dataset_discovery_summary(root: &str) -> Result<neoethos_data::DatasetDiscovery> {
    let report = neoethos_data::DatasetDiscovery::scan(root)?;
    println!("Scanned: {}", report.root.display());
    if report.is_empty() && report.skipped.is_empty() {
        // Real-data only: never silently fall back to a packaged demo
        // dataset. Surface the empty result so the operator can pick
        // a different folder.
        println!(
            "  (no data files found at depth ≤ {})",
            neoethos_data::MAX_WALK_DEPTH
        );
        return Ok(report);
    }

    let total = report.entries.len();
    let format_breakdown: Vec<String> = report
        .format_counts()
        .into_iter()
        .map(|(fmt, n)| format!("{}: {}", fmt.as_str(), n))
        .collect();
    println!(
        "Files found:        {} ({})",
        total,
        format_breakdown.join(", ")
    );

    let symbols = report.symbols();
    let symbols_preview: String = if symbols.len() > 6 {
        format!(
            "{}, ...",
            symbols
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        symbols.join(", ")
    };
    println!(
        "Symbols detected:   {}  ({})",
        symbols.len(),
        symbols_preview
    );

    let tfs = report.timeframes();
    println!("Timeframes:         {}", tfs.join(", "));

    if !report.skipped.is_empty() {
        let buckets = report.skip_counts_by_category();
        // Per-bucket detail: e.g. "unsupported_timeframe: H2 (x4)".
        let mut detail_parts: Vec<String> = Vec::new();
        for (cat, count) in &buckets {
            let example_labels: Vec<String> = report
                .skipped
                .iter()
                .filter(|s| s.reason.category() == cat)
                .filter_map(|s| match &s.reason {
                    neoethos_data::SkipReason::UnsupportedTimeframe(label) => Some(label.clone()),
                    neoethos_data::SkipReason::UnknownExtension(ext) => Some(format!(".{ext}")),
                    neoethos_data::SkipReason::TooLarge(bytes) => {
                        Some(format!("{} MiB", bytes / (1024 * 1024)))
                    }
                    neoethos_data::SkipReason::Unreadable(_) => None,
                })
                .collect();
            let mut uniq: Vec<String> = example_labels;
            uniq.sort();
            uniq.dedup();
            let labels = if uniq.is_empty() {
                "".to_string()
            } else {
                format!(": {}", uniq.join(", "))
            };
            detail_parts.push(format!("{count} {cat}{labels}"));
        }
        println!(
            "Skipped:            {}   ({})",
            report.skipped.len(),
            detail_parts.join("; ")
        );
    }

    Ok(report)
}

fn resolve_cli_settings(args: &[String]) -> Result<Option<neoethos_core::Settings>> {
    if let Some(config_path) = parse_flag(args, "--config") {
        return neoethos_core::Settings::from_yaml(&config_path).map(Some);
    }

    // Without --config, resolve the way the rest of the process does.
    //
    // This checked the working directory for `config.yaml` and stopped there,
    // while `Settings::load()` — which installs the runtime overrides at
    // startup, main.rs:18 — prefers the user config under %LOCALAPPDATA% and
    // only falls back to the relative path. Run from the repo and the two
    // disagreed inside one process.
    //
    // They are not variants of each other. Measured on this machine
    // (2026-08-01): the user config says trading_mode prop_firm, preset ftmo,
    // prop_search_device auto; the repo template, last touched two weeks
    // earlier, says risky, none, and `cpu` — the line recorded as the cause of
    // eight months of discovery never reaching the card. Which one a run got
    // depended on the directory it was started from.
    //
    // `load()` still ends at the relative path, so a workspace checkout with no
    // user config behaves exactly as before.
    neoethos_core::Settings::load().map(Some)
}


/// `EngineConfig` with the operator's REAL broker costs, instead of
/// `EngineConfig::default()` which charges nothing.
///
/// `ReplayCostModel` and `from_pips` were written and then never called: all
/// four replay entry points here and the one in `neoethos-app` passed
/// `EngineConfig::default()`, whose `costs` is `ReplayCostModel::zero()`. So
/// every replay an operator could actually run filled at the mark — no spread,
/// no slippage, no commission — and the only thing standing between him and a
/// flattering number was a disclosure warning. That warning is the honest
/// minimum; charging the costs is the fix.
///
/// The pip size comes from the symbol table, never a guess: EURUSD is 0.0001
/// and USDJPY is 0.01, and using the wrong one misprices the spread by a factor
/// of a hundred. An unknown symbol therefore keeps the ZERO model and says so
/// by name — a wrong cost is worse than a declared absent one, because it looks
/// like it was charged.
fn replay_engine_config(
    settings: Option<&neoethos_core::Settings>,
    symbol: &str,
) -> neoethos_trader::EngineConfig {
    let mut cfg = neoethos_trader::EngineConfig::default();
    let Some(settings) = settings else {
        tracing::warn!(
            target: "neoethos_cli::replay",
            "no config resolved — this replay fills at the mark, charging nothing"
        );
        return cfg;
    };
    let Some(meta) = neoethos_core::symbol_metadata::global_table().lookup(symbol) else {
        tracing::warn!(
            target: "neoethos_cli::replay",
            symbol,
            "symbol is not in the metadata table, so its pip size is unknown and the              spread cannot be converted to price units. This replay charges NOTHING.              Fix the symbol or add it to the table rather than trusting the result."
        );
        return cfg;
    };
    let risk = &settings.risk;
    cfg.costs = neoethos_trader::ReplayCostModel::from_pips(
        risk.backtest_spread_pips,
        risk.slippage_pips,
        risk.commission_per_lot,
        meta.pip_size,
    );
    tracing::info!(
        target: "neoethos_cli::replay",
        symbol,
        spread_pips = risk.backtest_spread_pips,
        slippage_pips = risk.slippage_pips,
        commission_per_lot = risk.commission_per_lot,
        pip_size = meta.pip_size,
        "replay costs charged from the operator's config"
    );
    cfg
}

fn default_symbol(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"EURUSD"` when settings was None — a synthetic default that the
    // no-synthetic-data directive forbids. Now returns the configured
    // symbol when settings loaded, empty string otherwise. Downstream
    // code rejects empty symbols (see `default_pip_size` returning NaN
    // for empty input → fitness guard rejects) so the operator gets a
    // clear "symbol required" error instead of silent EURUSD execution.
    // SHARED resolution (2026-06-04 parity unification): the Some branch now
    // delegates to `SystemConfig::resolve_symbol` in neoethos-core — the SAME
    // function the app server calls — so UI and CLI can never diverge. Only the
    // None-path F-CORE2 error logging stays CLI-specific.
    match settings {
        Some(settings) => settings.system.resolve_symbol(),
        None => {
            tracing::error!(
                target: "neoethos_cli::defaults",
                "No --symbol supplied and config.yaml could not be loaded; \
                 cannot synthesise a default per F-CORE2 doctrine — supply \
                 --symbol explicitly or ensure config.yaml is reachable."
            );
            String::new()
        }
    }
}

fn default_base_tf(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"M1"` when settings was None. Same fix as `default_symbol`.
    // 2026-06-04 parity: Some branch delegates to the shared core resolver.
    match settings {
        Some(settings) => settings.system.resolve_base_timeframe(),
        None => {
            tracing::error!(
                target: "neoethos_cli::defaults",
                "No --timeframe supplied and config.yaml could not be loaded; \
                 cannot synthesise a default per F-CORE2 doctrine — supply \
                 --timeframe explicitly or ensure config.yaml is reachable."
            );
            String::new()
        }
    }
}

/// Resolve the higher-TF CSV for the **effective** `base` (which may be a
/// `--base` override, not the config base). Delegates the actual selection to
/// the shared `SystemConfig::resolve_higher_timeframes` so the CLI and the app
/// server always pick the same ladder.
fn default_higher_tfs_csv(settings: Option<&neoethos_core::Settings>, base: &str) -> String {
    settings
        .map(|settings| settings.system.resolve_higher_timeframes(base).join(","))
        .unwrap_or_default()
}

fn default_batch_timeframes_csv(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"M1,M5,M15,H1,H4"` when settings was None — a synthetic default
    // that the no-synthetic-data directive forbids. Now returns empty
    // when settings can't load; downstream sweep code surfaces a clear
    // "no timeframes specified" error.
    if let Some(settings) = settings {
        let mut timeframes = vec![settings.system.base_timeframe.clone()];
        let higher_timeframes = if settings.system.multi_resolution_enabled
            && !settings.system.multi_resolution_timeframes.is_empty()
        {
            &settings.system.multi_resolution_timeframes
        } else {
            &settings.system.higher_timeframes
        };
        for timeframe in higher_timeframes {
            if !timeframes.contains(timeframe) {
                timeframes.push(timeframe.clone());
            }
        }
        return timeframes.join(",");
    }

    tracing::error!(
        target: "neoethos_cli::defaults",
        "No --timeframes supplied and config.yaml could not be loaded; \
         cannot synthesise a default per F-CORE2 doctrine — supply \
         --timeframes explicitly or ensure config.yaml is reachable."
    );
    String::new()
}

/// Environment variables this binary used to obey and no longer does, each
/// paired with what replaced it.
///
/// NOTHING branches on this list — it is a report, not a decision. It exists
/// because the failure mode this whole consolidation is closing is the SILENT
/// one: an operator who exported a name that used to work, and a binary that
/// ignored it without saying so. A deleted lever that is still being pulled
/// must announce that it is deleted.
const RETIRED_ENV_VARS: &[(&str, &str)] = &[(
    "NEOETHOS_BOT_CPU_BUDGET",
    "--cpu-threads (set by the schedule orchestrator on the children it spawns)",
)];

/// Name every retired variable that is still set in this process's
/// environment, state its replacement, and say plainly that the value was
/// ignored. Called once from `main()`.
fn warn_retired_env_vars() {
    for (name, replacement) in RETIRED_ENV_VARS {
        let Some(value) = std::env::var_os(name) else {
            continue;
        };
        let value = value.to_string_lossy().to_string();
        eprintln!(
            "IGNORED ENV VAR: {name}={value} — this variable was deleted. Nothing read it, \
             and NOTHING in this run was changed by it. Its replacement is: {replacement}."
        );
        tracing::warn!(
            target: "neoethos_cli",
            env_var = %name,
            value = %value,
            replacement = %replacement,
            "a deleted environment variable is still set; its value was ignored"
        );
    }
}

/// Parse an optional blend-multiplier flag (`--gate-floor` / `--veto-below`).
///
/// An unparseable value is a HARD ERROR, never a silent `None`. The previous
/// `.and_then(|v| v.parse().ok())` turned `--gate-floor 0,34` (comma) or a typo
/// into "operator said nothing" and ran the default — on a knob that scales
/// every position in the replay. Absent still means absent; RANGE and inversion
/// validation belong to `BlendConfig::from_config_values`, which logs both the
/// configured and the used number when it refuses.
fn parse_blend_knob(args: &[String], name: &str) -> Result<Option<f64>> {
    match parse_flag(args, name) {
        None => Ok(None),
        Some(raw) => match raw.trim().parse::<f64>() {
            Ok(v) => Ok(Some(v)),
            Err(err) => anyhow::bail!(
                "{name} expects a number in [0,1] (got '{raw}'): {err}. This value \
                 scales every position's size — refusing to guess."
            ),
        },
    }
}

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
    }
    None
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

/// `credentials` subcommand — write `broker_credentials.toml` headlessly.
///
/// This is the CLI parity for the Flutter Settings → cTrader credentials
/// form. Same file, same schema, same path-resolution rules — the
/// shared writer lives in `neoethos_core::broker_config` so the two
/// frontends can never drift.
///
/// Subcommands:
///   credentials show
///     Read the current `broker_credentials.toml` and print a redacted
///     summary (client_secret is shown as `••••<last4>` only). Useful
///     for verifying which set the binary is picking up via the
///     `NEOETHOS_BROKER_CREDENTIALS_PATH` env override.
///
///   credentials set --client-id <id> [--client-secret <secret>]
///                   [--redirect-uri <uri>] [--environment Demo|Live]
///                   [--account-id <cTID>]
///     Merge-update the on-disk file. Unspecified fields keep their
///     current value (merge semantics match `POST /broker/credentials`).
///     If --client-secret is provided but blank, the existing secret
///     is preserved (same rule as the UI's "Leave blank to keep" form).
fn cmd_credentials(args: &[String]) -> Result<()> {
    if args.is_empty() {
        anyhow::bail!(
            "credentials requires a subcommand: `show` or `set`. \
             Run `neoethos-cli credentials show` to read the current \
             on-disk values."
        );
    }
    match args[0].as_str() {
        "show" => cmd_credentials_show(),
        "set" => cmd_credentials_set(&args[1..]),
        other => {
            anyhow::bail!("unknown credentials subcommand `{other}`. Expected `show` or `set`.")
        }
    }
}

fn cmd_credentials_show() -> Result<()> {
    let path = neoethos_core::broker_config::credentials_file_path()?;
    let loaded = neoethos_core::broker_config::load_from_disk(&path)?;
    println!("Path: {}", path.display());
    match loaded {
        None => {
            println!("(no file at that path — defaults will be used)");
        }
        Some(state) => {
            println!("Schema version: {}", state.schema_version);
            println!("\n[ctrader]");
            println!("  client_id    : {}", maybe_blank(&state.ctrader.client_id));
            println!(
                "  client_secret: {}",
                redact_secret(&state.ctrader.client_secret)
            );
            println!(
                "  redirect_uri : {}",
                maybe_blank(&state.ctrader.redirect_uri)
            );
            println!("  environment  : {}", state.ctrader.environment.as_str());
            println!("  accounts     : {} entries", state.ctrader.accounts.len());
            for (i, a) in state.ctrader.accounts.iter().enumerate() {
                println!(
                    "    [{i}] id={} label={} enabled={}",
                    a.account_id, a.label, a.enabled_for_execution
                );
            }
        }
    }
    Ok(())
}

fn cmd_credentials_set(args: &[String]) -> Result<()> {
    let mut client_id: Option<String> = None;
    let mut client_secret: Option<String> = None;
    let mut redirect_uri: Option<String> = None;
    let mut environment: Option<String> = None;
    let mut account_id: Option<String> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--client-id" => client_id = iter.next().cloned(),
            "--client-secret" => client_secret = iter.next().cloned(),
            "--redirect-uri" => redirect_uri = iter.next().cloned(),
            "--environment" => environment = iter.next().cloned(),
            "--account-id" => account_id = iter.next().cloned(),
            other => anyhow::bail!(
                "unknown flag `{other}` for `credentials set`. \
                 Supported flags: --client-id, --client-secret, \
                 --redirect-uri, --environment, --account-id."
            ),
        }
    }

    let path = neoethos_core::broker_config::credentials_file_path()?;
    let mut state = neoethos_core::broker_config::load_from_disk(&path)?.unwrap_or_default();

    if let Some(v) = client_id {
        state.ctrader.client_id = v.trim().to_string();
    }
    // Empty-secret semantics match the UI: blank means "keep current".
    if let Some(v) = client_secret {
        if !v.is_empty() {
            state.ctrader.client_secret = v;
        }
    }
    if let Some(v) = redirect_uri {
        state.ctrader.redirect_uri = v.trim().to_string();
    }
    if let Some(v) = environment {
        let parsed =
            neoethos_core::broker_config::CTraderBrokerEnvironment::parse(&v).ok_or_else(|| {
                anyhow::anyhow!("invalid --environment value `{v}`. Expected `Demo` or `Live`.")
            })?;
        state.ctrader.environment = parsed;
    }
    if let Some(v) = account_id {
        let trimmed = v.trim().to_string();
        if !trimmed.is_empty() {
            // Replace the entire account list with the single target —
            // matches the UI's behaviour (the dropdown sends one
            // accountId, the server overwrites the targets vec).
            state.ctrader.accounts = vec![neoethos_core::broker_config::BrokerAccountTarget {
                account_id: trimmed,
                label: String::new(),
                enabled_for_execution: true,
            }];
        }
    }

    if state.ctrader.client_id.trim().is_empty() {
        anyhow::bail!(
            "client_id is required (it is currently blank on disk and no \
             --client-id was supplied). Provide --client-id at least once \
             before saving."
        );
    }
    if state.ctrader.client_secret.is_empty() {
        anyhow::bail!(
            "client_secret is required (it is currently blank on disk and \
             no --client-secret was supplied). Provide --client-secret at \
             least once before saving."
        );
    }
    if state.ctrader.redirect_uri.trim().is_empty() {
        // Sourced from neoethos-core so the listener, the CLI
        // default, and the embedded fallback can't drift apart
        // (#150).
        state.ctrader.redirect_uri =
            neoethos_core::broker_config::CTRADER_OAUTH_REDIRECT_URI.to_string();
    }

    neoethos_core::broker_config::save_to_disk(&path, &state)?;
    println!(
        "Wrote {} ({} ctrader.accounts)",
        path.display(),
        state.ctrader.accounts.len()
    );
    println!(
        "Next step: open the GUI and run Broker Setup → Re-authenticate to fetch an OAuth token."
    );
    Ok(())
}

fn redact_secret(s: &str) -> String {
    if s.is_empty() {
        return "(blank)".to_string();
    }
    let last4: String = s
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("••••{last4} (len={})", s.len())
}

fn maybe_blank(s: &str) -> &str {
    if s.is_empty() { "(blank)" } else { s }
}

fn print_help() {
    println!("neoethos-cli");
    println!("  symbols --root data");
    println!("  timeframes --symbol EURUSD --root data");
    println!("  load --symbol EURUSD --timeframe M1 --root data");
    println!("  features --symbol EURUSD --timeframe M1 --root data");
    println!("  prepare --symbol EURUSD --base M1 --higher H1,H4 --root data");
    println!("  resample --symbol EURUSD --base M1 --target H1 --root data");
    println!(
        "  bench --dry-run --fixture tiny --prototype a --backend cuda --out cache/gpu-bench/plan.json"
    );
    println!("  train --symbol EURUSD --base M1 --higher H1,H4 --horizon 1 --root data");
    println!(
        "  search --symbol EURUSD --base M1 --higher H1,H4 --genes 64 --generations 5 --max-indicators 12 --root data"
    );
    println!(
        "  discover --symbol EURUSD --base M1 --higher H1,H4 --population 100 --generations 5 --max-indicators 12 --portfolio-size 100 --candidates 200 --corr 0.7 --min-trades 1 --out cache/vector_ta_knowledge.json --root data"
    );
    println!(
        "  discover ... --stream-sweep [--stream-max-batches N]  Sweep the (indicator, period) space in BATCHES instead of building one cube: a batch is sized from FREE RAM (never a flag), a discovery cycle runs per batch, and a batch whose candidates cannot clear the CONFIGURED expectancy floor is abandoned before the quality screen. Survivors from different batches are remapped onto one run-level feature list; every abandoned batch is named by cursor in <out>.streaming.json."
    );
    println!(
        "  discovery-promote-weekly [--symbol EURUSD --tf M1] [--cache-dir cache/search] [--portfolio <live_portfolio.json>]  Weekly-refresh: merge this run's discovery ledger into the live portfolio (additive by gene-signature hash) and print 'added N new, carried M, total K'."
    );
    println!(
        "  trader-replay [--symbol EURUSD --base M1 | --portfolio <live_portfolio.json>] [--root data] [--blend off|confirm|scale] [--models-root models]  Offline dry-run of the autonomous trader (zero broker calls; same engine as /autonomous/replay). With --portfolio runs the REAL genes; --blend gates their size by the ML ensemble (gene-dominant)."
    );
    println!(
        "  blend-test --portfolio <live_portfolio.json> --models-root models_oos_locked [--root data] [--gate-floor 0.34] [--veto-below 0.15]  Re-validate the gene<->ML blend on the NETTED portfolio: GenesOnly vs MlConfirm vs MlScale on the same engine + non-degradation verdict. Point --models-root at a LEAK-FREE root (train --oos-from)."
    );
    println!(
        "  train --symbol EURUSD --base H1 [--models-dir models] [--oos-from 2023-01-01]  Train the ML ensemble. --oos-from trains LEAK-LOCKED experts (rows < cutoff, purged) to a SEPARATE root for OOS blend validation."
    );
    println!("  migrate-data --root data [--force] [--delete-source]");
    println!(
        "  slice-dataset --symbol EURUSD --base M1 --root <SRC> --out-root <DST> --from-date 2018-01-01 --to-date 2021-01-01"
    );
    println!(
        "                               Write the [from,to) UTC date-range subset of a Vortex dataset to a NEW root"
    );
    println!(
        "                               (discover --root <DST> runs on the slice). Enables OOM-safe walk-forward chunking."
    );
    println!("  stop-target --symbol EURUSD --timeframe M1 --pip 0.0001 --signal 1 --root data");
    println!("  wizard                       Launch the interactive first-run wizard (TUI).");
    println!("  setup [show|paths|ctrader]  Headless credentials helper (Task #61).");
    println!("                               Prints canonical paths + ready-to-paste templates.");
    println!("  credentials show             Show on-disk broker_credentials.toml (redacted).");
    println!("  credentials set --client-id X --client-secret Y [--redirect-uri Z]");
    println!("                  [--environment Demo|Live] [--account-id N]");
    println!("                               Merge-update broker_credentials.toml. Same writer");
    println!("                               as the GUI Settings screen — never drifts.");
    println!();
    println!("  --data-path <folder>   Browse a folder and auto-discover dataset layout");
    println!("                         (subfolders for symbol/timeframe, Hive-style or flat).");
    println!("                         Supported on: train, discover, import.");
    println!("  --dry-run              With --data-path, print the discovery summary and exit.");
}

fn cli_record(operation: &str, status: &str, message: impl Into<String>) -> SectionedRunRecord {
    section_record(SubsystemSection::Cli, operation, status, message)
}

fn section_record(
    section: SubsystemSection,
    operation: &str,
    status: &str,
    message: impl Into<String>,
) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!(
            "{}-{}-{}",
            section.as_str().to_lowercase(),
            operation,
            now.replace(':', "-")
        ),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: section,
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
    // F-282 + F-656 fix (2026-05-25): match the neoethos-app pattern —
    // never panic on pre-1970 clock; emit a sentinel + structured warn
    // so the operator sees the clock skew without losing the whole CLI.
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => format!("unix:{}", d.as_secs()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_cli::main",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "unix:pre-1970".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cli_record, gpu_assignment_env, section_record};
    use neoethos_core::sectioned_log::SubsystemSection;

    #[test]
    fn integrated_wgpu_assignment_uses_typed_selector_without_cuda_pin() {
        use neoethos_core::scheduler::Assignment;
        use neoethos_core::system::{
            AcceleratorBackend, AcceleratorDevice, AcceleratorDeviceClass, HardwareProfile,
            TrainingPrecision,
        };

        let assignment = Assignment {
            id: "EURUSD/M1".to_string(),
            card_ids: vec![0],
            genes_per_card: 128,
            cpu_threads: 3,
            class: neoethos_core::scheduler::ComboClass::Light,
        };
        let profile = HardwareProfile {
            schema_version: neoethos_core::system::HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 12,
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            gpu_names: vec!["AMD Radeon Graphics".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![0.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "AMD Radeon Graphics".to_string(),
                backend: AcceleratorBackend::Vulkan,
                device_class: AcceleratorDeviceClass::IntegratedGpu,
                backend_index: 0,
                memory_gb: 0.0,
                supported_precisions: vec![TrainingPrecision::Fp32],
                compute_capability: None,
                source: "test".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };

        let envs = gpu_assignment_env(&assignment, &profile);

        assert_eq!(
            envs.get("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE"),
            Some(&"integrated:0".to_string())
        );
        assert!(!envs.contains_key("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE"));
    }

    #[test]
    fn cli_record_targets_cli_section() {
        let record = cli_record("load", "SUCCESS", "load completed");
        assert_eq!(record.subsystem, SubsystemSection::Cli);
        assert_eq!(record.operation, "load");
        assert_eq!(record.status, "SUCCESS");
    }

    #[test]
    fn section_record_targets_requested_subsystem() {
        let record = section_record(
            SubsystemSection::Discovery,
            "discover",
            "FAILED",
            "discovery failed",
        );
        assert_eq!(record.subsystem, SubsystemSection::Discovery);
        assert_eq!(record.operation, "discover");
        assert_eq!(record.status, "FAILED");
        assert_eq!(record.message, "discovery failed");
    }
}
