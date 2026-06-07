use anyhow::Result;
use neoethos_core::logging::{setup_logging, write_subsystem_record};
use neoethos_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod tui;

fn main() -> Result<()> {
    setup_logging(false)?;
    // Config-consolidation: search runtime overrides come from the single
    // config (canonical user config.yaml), not the environment. Falls back
    // to defaults if it can't be loaded. (S2a: genetic search; rest staged.)
    let startup_settings = neoethos_core::Settings::load().unwrap_or_default();
    neoethos_search::install_search_runtime_overrides_from_settings(&startup_settings);
    neoethos_models::tree_models::config::install_tree_runtime_from_settings(&startup_settings);
    neoethos_core::system::install_hardware_runtime_overrides_from_settings(&startup_settings);
    neoethos_data::install_data_runtime_overrides(
        startup_settings.models.data_runtime.normalize_features,
        startup_settings.models.data_runtime.rebuild_stale_higher_tfs,
    );
    let args: Vec<String> = std::env::args().collect();
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
        "migrate-data" => cmd_migrate_data(&args[2..]),
        "slice-dataset" => cmd_slice_dataset(&args[2..]),
        "import" => cmd_import(&args[2..]),
        "config" => cmd_config(&args[2..]),
        "auto-loop" => cmd_auto_loop(&args[2..]),
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
        anyhow::anyhow!("slice-dataset requires --from-date YYYY-MM-DD (inclusive lower bound, UTC)")
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
    let cache = neoethos_data::FeatureCache::new("cache/features", 60, true);
    let features = neoethos_data::prepare_multitimeframe_features(
        &dataset,
        &base,
        &higher_refs,
        Some(&cache),
    )?;
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
    let portfolio_path = parse_flag(args, "--portfolio").unwrap_or_else(|| {
        format!("{}/{}_{}.live_portfolio.json", cache_dir, symbol, tf)
    });

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
    if let Err(err) =
        neoethos_core::storage::json::write_json_atomic(&summary_path, &summary)
    {
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
    println!("  ledger: {}", neoethos_search::ledger_path(&cache_dir, &symbol, &tf).display());
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
            other => anyhow::bail!(
                "--blend must be off|confirm|scale (got '{other}')"
            ),
        };
        if matches!(mode, neoethos_trader::BlendMode::GenesOnly) {
            neoethos_trader::replay_portfolio_from_dir(
                &root,
                &portfolio,
                neoethos_trader::EngineConfig::default(),
            )?
        } else {
            let models_root = parse_flag(args, "--models-root").unwrap_or_else(|| "models".to_string());
            let mut blend = neoethos_trader::BlendConfig {
                mode,
                ..Default::default()
            };
            if let Some(f) = parse_flag(args, "--gate-floor").and_then(|v| v.parse::<f64>().ok()) {
                blend.gate_floor = f.clamp(0.0, 1.0);
            }
            if let Some(v) = parse_flag(args, "--veto-below").and_then(|v| v.parse::<f64>().ok()) {
                blend.veto_below = v.clamp(0.0, 1.0);
            }
            println!(
                "  blend mode={blend_arg} models_root={models_root} gate_floor={:.2} veto_below={:.2}",
                blend.gate_floor, blend.veto_below
            );
            neoethos_trader::replay_blend_from_dir(
                &root,
                &portfolio,
                &models_root,
                neoethos_trader::EngineConfig::default(),
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
            neoethos_trader::EngineConfig::default(),
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

    println!(
        "FAITHFUL OOS forward-test (gene real SL/TP + risk sizing; holdout from {oos_from}):"
    );
    println!(
        "{:<16}{:>5}{:>5}{:>8}{:>8}{:>8}{:>8}{:>10}{:>7}{:>8}  verdict",
        "gene", "#ind", "#smc", "IS_PF", "IS_DD%", "OOS_PF", "OOS_DD%", "OOS_net", "OOS_tr", "WFE_shp"
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
    let portfolio = parse_flag(args, "--portfolio").ok_or_else(|| {
        anyhow::anyhow!("blend-test requires --portfolio <live_portfolio.json>")
    })?;
    let models_root =
        parse_flag(args, "--models-root").unwrap_or_else(|| "models_oos_locked".to_string());
    let gate_floor = parse_flag(args, "--gate-floor")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.34)
        .clamp(0.0, 1.0);
    let veto_below = parse_flag(args, "--veto-below")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.15)
        .clamp(0.0, 1.0);

    let run = |mode| -> Result<neoethos_trader::EngineStats> {
        neoethos_trader::replay_blend_from_dir(
            &root,
            &portfolio,
            &models_root,
            neoethos_trader::EngineConfig::default(),
            neoethos_trader::BlendConfig {
                mode,
                gate_floor,
                veto_below,
            },
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
            println!("  -> {name}: ACCEPT (>= genes-only on realized_pnl AND equity, still trades)");
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
    let features = neoethos_data::prepare_multitimeframe_features(
        &dataset,
        &base,
        &higher_refs,
        Some(&neoethos_data::FeatureCache::new(
            "cache/features",
            60,
            true,
        )),
    )?;
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
        want_tfs.retain(|tf| {
            neoethos_data::symbol_timeframe_vortex_path(&root, &symbol, tf).exists()
        });
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
        let features = neoethos_data::prepare_multitimeframe_features(
            &dataset,
            &base,
            &higher_refs,
            Some(&neoethos_data::FeatureCache::new(
                "cache/features",
                60,
                true,
            )),
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
        let result = neoethos_search::run_discovery_cycle(&features, base_ohlcv, &config)?;
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

        let config = settings
            .as_ref()
            .map(neoethos_search::DiscoveryConfig::from_settings)
            .unwrap_or_default();
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
        match cmd_discover(&discover_args) {
            Ok(()) => println!("  discover OK"),
            Err(err) => {
                eprintln!("  discover FAILED: {err:#}");
                // Continue to next; don't bail the whole loop.
            }
        }

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
                Err(err) => eprintln!("  train FAILED: {err:#}"),
            }
        }

        completed.push((sym.clone(), tf.clone()));
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
///   `neoethos-cli setup news`        — print news API key template
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
        "news" => setup_news_template(),
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
    println!("    news       Emit a TOML snippet for the news API key");
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
    println!("Run `neoethos-cli setup news` for the news API key snippet.");
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
    println!();
    println!("[dxtrade]");
    println!("platform_url = \"\"");
    println!("username = \"\"");
    println!("password = \"\"");
    println!("domain = \"default\"");
    println!("accounts = []");
    Ok(())
}

fn setup_news_template() -> Result<()> {
    let dir = canonical_user_config_dir();
    let path = dir.join("news_api.toml");
    println!("# Paste this into:");
    println!("#   {}", path.display());
    println!("# The news API key drives Step 8 of the wizard (LLM-curated news +");
    println!("# blackout filter). Perplexity is the default provider; keep the key");
    println!("# OUT of shell history — `chmod 600` the file after pasting.");
    println!();
    println!("provider = \"perplexity\"");
    println!("api_key = \"<your Perplexity API key>\"");
    Ok(())
}

fn parse_root(args: &[String], settings: Option<&neoethos_core::Settings>) -> String {
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

fn parse_config_path(args: &[String]) -> String {
    parse_flag(args, "--config").unwrap_or_else(|| "config.yaml".to_string())
}

fn resolve_cli_settings(args: &[String]) -> Result<Option<neoethos_core::Settings>> {
    if let Some(config_path) = parse_flag(args, "--config") {
        return neoethos_core::Settings::from_yaml(&config_path).map(Some);
    }

    let default_config_path = parse_config_path(args);
    let default_path = Path::new(&default_config_path);
    if default_path.exists() {
        return neoethos_core::Settings::from_yaml(default_path).map(Some);
    }

    Ok(None)
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
            println!("\n[dxtrade]");
            println!(
                "  platform_url : {}",
                maybe_blank(&state.dxtrade.platform_url)
            );
            println!("  username     : {}", maybe_blank(&state.dxtrade.username));
            println!("  domain       : {}", maybe_blank(&state.dxtrade.domain));
            println!("  password     : (never persisted)");
            println!("  accounts     : {} entries", state.dxtrade.accounts.len());
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
    println!("  train --symbol EURUSD --base M1 --higher H1,H4 --horizon 1 --root data");
    println!(
        "  search --symbol EURUSD --base M1 --higher H1,H4 --genes 64 --generations 5 --max-indicators 12 --root data"
    );
    println!(
        "  discover --symbol EURUSD --base M1 --higher H1,H4 --population 100 --generations 5 --max-indicators 12 --portfolio-size 100 --candidates 200 --corr 0.7 --min-trades 1 --out cache/vector_ta_knowledge.json --root data"
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
    println!("  train --symbol EURUSD --base H1 [--models-dir models] [--oos-from 2023-01-01]  Train the ML ensemble. --oos-from trains LEAK-LOCKED experts (rows < cutoff, purged) to a SEPARATE root for OOS blend validation.");
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
    println!("  setup [show|paths|ctrader|news]  Headless credentials helper (Task #61).");
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
    use super::{cli_record, section_record};
    use neoethos_core::sectioned_log::SubsystemSection;

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
