use anyhow::Result;
use forex_core::logging::{setup_logging, write_subsystem_record};
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<()> {
    setup_logging(false)?;
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        let _ = write_subsystem_record(
            SubsystemSection::Cli,
            cli_record("help", "SUCCESS", "printed CLI help"),
        );
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
        "batch-discover" => cmd_batch_discover(&args[2..]),
        "migrate-data" => cmd_migrate_data(&args[2..]),
        "stop-target" => cmd_stop_target(&args[2..]),
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

    let ohlcv = forex_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    println!("Loaded {} {} rows: {}", symbol, timeframe, ohlcv.len());
    Ok(())
}

fn cmd_symbols(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbols = forex_data::discover_symbols(root)?;
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
    let tfs = forex_data::discover_timeframes(root, &symbol)?;
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
    let ohlcv = forex_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    let features = forex_data::compute_hpc_features(&ohlcv)?;
    println!(
        "Features {} {} -> rows={}, cols={}",
        symbol,
        timeframe,
        features.data.nrows(),
        features.data.ncols()
    );
    Ok(())
}

fn cmd_prepare(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let higher =
        parse_flag(args, "--higher").unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref()));
    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();
    let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
    let cache = forex_data::FeatureCache::new("cache/features", 60, true);
    let features =
        forex_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs, Some(&cache))?;
    println!(
        "Prepared {} base={} rows={} cols={}",
        symbol,
        base,
        features.data.nrows(),
        features.data.ncols()
    );
    Ok(())
}

fn cmd_resample(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let target = parse_flag(args, "--target").unwrap_or_else(|| "H1".to_string());
    let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
    let base_ohlcv = dataset
        .frames
        .get(&base)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;
    let resampled = forex_data::resample_ohlcv(base_ohlcv, &target)?;
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
        let settings = resolve_cli_settings(args)?.unwrap_or_else(forex_core::Settings::default);
        let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| settings.system.symbol.clone());
        let base =
            parse_flag(args, "--base").unwrap_or_else(|| settings.system.base_timeframe.clone());
        let models_dir = parse_flag(args, "--models-dir").unwrap_or_else(|| "models".to_string());
        let orchestrator =
            forex_models::TrainingOrchestrator::new(settings, std::path::PathBuf::from(models_dir));

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
        .map(forex_search::DiscoveryConfig::from_settings)
        .unwrap_or_default();
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let higher =
        parse_flag(args, "--higher").unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref()));
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

    let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
    let dataset =
        forex_data::ensure_timeframes_with_resample(&dataset, &base, forex_data::MANDATORY_TFS)?;
    let features = forex_data::prepare_multitimeframe_features(
        &dataset,
        &base,
        &higher_refs,
        Some(&forex_data::FeatureCache::new("cache/features", 60, true)),
    )?;
    let base_ohlcv = dataset
        .frames
        .get(&base)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;

    let result =
        forex_search::evolve_search(&features, base_ohlcv, genes, generations, max_indicators)?;
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
            .map(forex_search::DiscoveryConfig::from_settings)
            .unwrap_or_default();
        let root = parse_root(args, settings.as_ref());
        let symbol =
            parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
        let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
        let higher = parse_flag(args, "--higher")
            .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref()));
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

        let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
        let dataset = forex_data::ensure_timeframes_with_resample(
            &dataset,
            &base,
            forex_data::MANDATORY_TFS,
        )?;
        let features = forex_data::prepare_multitimeframe_features(
            &dataset,
            &base,
            &higher_refs,
            Some(&forex_data::FeatureCache::new("cache/features", 60, true)),
        )?;
        let base_ohlcv = dataset
            .frames
            .get(&base)
            .ok_or_else(|| anyhow::anyhow!("base timeframe missing: {}", base))?;

        let config = forex_search::DiscoveryConfig {
            timeframe_label: base.clone(),
            population,
            generations,
            max_indicators,
            candidate_count,
            portfolio_size,
            corr_threshold,
            min_trades_per_day,
            filtering: defaults.filtering,
            ..defaults.clone()
        };
        let result = forex_search::run_discovery_cycle(&features, base_ohlcv, &config)?;
        forex_search::ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, base))?;
        if let Some(parent) = std::path::Path::new(&out).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        forex_search::save_portfolio_json(
            &out,
            &result.portfolio,
            &result.effective_feature_names,
        )?;
        let profile_path = format!("{out}.profile.json");
        forex_search::save_discovery_profile_json(&profile_path, &config, &result)?;
        if !result.quality_metrics.is_empty() {
            let quality_path = format!("{out}.quality.json");
            forex_search::save_quality_report_json(&quality_path, &result)?;
        }
        if !result.logged_trades.is_empty() {
            let trade_log_path = format!("{out}.trades.json");
            forex_search::save_trade_log_json(&trade_log_path, &result)?;
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
            forex_data::discover_symbols(&root)?
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
            .map(forex_search::DiscoveryConfig::from_settings)
            .unwrap_or_default();
        let orchestrator = forex_search::DiscoveryOrchestrator::new(&root, &out_dir, config);

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

fn cmd_migrate_data(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let force = has_flag(args, "--force");
    let delete_source = has_flag(args, "--delete-source");
    let summary = forex_data::migrate_legacy_parquet_tree(&root, force, delete_source)?;

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

    let ohlcv = forex_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    let settings = forex_search::StopTargetSettings::default();
    let result = forex_search::infer_stop_target_pips(
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

fn parse_root(args: &[String], settings: Option<&forex_core::Settings>) -> String {
    parse_flag(args, "--root").unwrap_or_else(|| {
        settings
            .map(|settings| settings.system.data_dir.to_string_lossy().to_string())
            .unwrap_or_else(|| "data".to_string())
    })
}

fn parse_config_path(args: &[String]) -> String {
    parse_flag(args, "--config").unwrap_or_else(|| "config.yaml".to_string())
}

fn resolve_cli_settings(args: &[String]) -> Result<Option<forex_core::Settings>> {
    if let Some(config_path) = parse_flag(args, "--config") {
        return forex_core::Settings::from_yaml(&config_path).map(Some);
    }

    let default_config_path = parse_config_path(args);
    let default_path = Path::new(&default_config_path);
    if default_path.exists() {
        return forex_core::Settings::from_yaml(default_path).map(Some);
    }

    Ok(None)
}

fn default_symbol(settings: Option<&forex_core::Settings>) -> String {
    settings
        .map(|settings| settings.system.symbol.clone())
        .unwrap_or_else(|| "EURUSD".to_string())
}

fn default_base_tf(settings: Option<&forex_core::Settings>) -> String {
    settings
        .map(|settings| settings.system.base_timeframe.clone())
        .unwrap_or_else(|| "M1".to_string())
}

fn default_higher_tfs_csv(settings: Option<&forex_core::Settings>) -> String {
    settings
        .map(|settings| {
            if settings.system.multi_resolution_enabled
                && !settings.system.multi_resolution_timeframes.is_empty()
            {
                settings
                    .system
                    .multi_resolution_timeframes
                    .iter()
                    .filter(|timeframe| {
                        !timeframe.eq_ignore_ascii_case(&settings.system.base_timeframe)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(",")
            } else {
                settings.system.higher_timeframes.join(",")
            }
        })
        .unwrap_or_default()
}

fn default_batch_timeframes_csv(settings: Option<&forex_core::Settings>) -> String {
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

    "M1,M5,M15,H1,H4".to_string()
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

fn print_help() {
    println!("forex-cli");
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
    println!("  migrate-data --root data [--force] [--delete-source]");
    println!("  stop-target --symbol EURUSD --timeframe M1 --pip 0.0001 --signal 1 --root data");
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
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_secs();
    format!("unix:{seconds}")
}

#[cfg(test)]
mod tests {
    use super::{cli_record, section_record};
    use forex_core::sectioned_log::SubsystemSection;

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
