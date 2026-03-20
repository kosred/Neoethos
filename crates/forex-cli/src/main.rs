use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        return Ok(());
    }
    match args[1].as_str() {
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
        "stop-target" => cmd_stop_target(&args[2..]),
        _ => {
            print_help();
            Ok(())
        }
    }
}

fn cmd_load(args: &[String]) -> Result<()> {
    let mut root = "data".to_string();
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

    let symbol = symbol.unwrap_or_else(|| "EURUSD".to_string());
    let timeframe = timeframe.unwrap_or_else(|| "M1".to_string());

    let ohlcv = forex_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    println!("Loaded {} {} rows: {}", symbol, timeframe, ohlcv.len());
    Ok(())
}

fn cmd_symbols(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbols = forex_data::discover_symbols(root)?;
    println!("Symbols ({}):", symbols.len());
    for sym in symbols {
        println!("  {}", sym);
    }
    Ok(())
}

fn cmd_timeframes(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let tfs = forex_data::discover_timeframes(root, &symbol)?;
    println!("Timeframes for {} ({}):", symbol, tfs.len());
    for tf in tfs {
        println!("  {}", tf);
    }
    Ok(())
}

fn cmd_features(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let timeframe = parse_flag(args, "--timeframe").unwrap_or_else(|| "M1".to_string());
    let ohlcv = forex_data::load_symbol_timeframe(&root, &symbol, &timeframe)?;
    let features = forex_data::compute_talib_features(&ohlcv)?;
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
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = parse_flag(args, "--base").unwrap_or_else(|| "M1".to_string());
    let higher = parse_flag(args, "--higher").unwrap_or_else(|| "".to_string());
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
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = parse_flag(args, "--base").unwrap_or_else(|| "M1".to_string());
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
    let _root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = parse_flag(args, "--base").unwrap_or_else(|| "M1".to_string());
    let models_dir = parse_flag(args, "--models-dir").unwrap_or_else(|| "models".to_string());
    let config_path = parse_flag(args, "--config").unwrap_or_else(|| "config.yaml".to_string());

    let settings = forex_core::Settings::from_yaml(&config_path)?;
    let orchestrator = forex_models::TrainingOrchestrator::new(settings, std::path::PathBuf::from(models_dir));
    
    orchestrator.train_symbol(&symbol, &base)?;
    
    println!("Pure Rust training complete for {}", symbol);
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = parse_flag(args, "--base").unwrap_or_else(|| "M1".to_string());
    let higher = parse_flag(args, "--higher").unwrap_or_else(|| "".to_string());
    let genes: usize = parse_flag(args, "--genes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let max_indicators: usize = parse_flag(args, "--max-indicators")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let generations: usize = parse_flag(args, "--generations")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

    let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
    let dataset =
        forex_data::ensure_timeframes_with_resample(&dataset, &base, &forex_data::MANDATORY_TFS)?;
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
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = parse_flag(args, "--base").unwrap_or_else(|| "M1".to_string());
    let higher = parse_flag(args, "--higher").unwrap_or_else(|| "".to_string());
    let population: usize = parse_flag(args, "--population")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let generations: usize = parse_flag(args, "--generations")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_indicators: usize = parse_flag(args, "--max-indicators")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let candidate_count: usize = parse_flag(args, "--candidates")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let portfolio_size: usize = parse_flag(args, "--portfolio-size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let corr_threshold: f64 = parse_flag(args, "--corr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let min_trades_per_day: f64 = parse_flag(args, "--min-trades")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let out = parse_flag(args, "--out").unwrap_or_else(|| "cache/talib_knowledge.json".to_string());

    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

    let dataset = forex_data::load_symbol_dataset(&root, &symbol)?;
    let dataset =
        forex_data::ensure_timeframes_with_resample(&dataset, &base, &forex_data::MANDATORY_TFS)?;
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
        population,
        generations,
        max_indicators,
        candidate_count,
        portfolio_size,
        corr_threshold,
        min_trades_per_day,
        filtering: Default::default(),
    };
    let result = forex_search::run_discovery_cycle(&features, base_ohlcv, &config)?;
    forex_search::ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, base))?;
    if let Some(parent) = std::path::Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    forex_search::save_portfolio_json(&out, &result.portfolio, &features.names)?;
    println!(
        "Discovery {} portfolio={} candidates={} out={}",
        symbol,
        result.portfolio.len(),
        result.candidates.len(),
        out
    );
    Ok(())
}

fn cmd_batch_discover(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbols_raw = parse_flag(args, "--symbols").unwrap_or_else(|| "".to_string());
    let tfs_raw = parse_flag(args, "--timeframes").unwrap_or_else(|| "M1,M5,M15,H1,H4".to_string());
    let out_dir = parse_flag(args, "--out-dir").unwrap_or_else(|| "cache/discovery".to_string());
    
    let symbols: Vec<String> = if symbols_raw.is_empty() {
        forex_data::discover_symbols(&root)?
    } else {
        symbols_raw.split(',').map(|s| s.trim().to_uppercase()).collect()
    };
    
    let tfs: Vec<String> = tfs_raw.split(',').map(|s| s.trim().to_uppercase()).collect();
    
    let config = forex_search::DiscoveryConfig::default();
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
    Ok(())
}

fn cmd_stop_target(args: &[String]) -> Result<()> {
    let root = parse_root(args);
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let timeframe = parse_flag(args, "--timeframe").unwrap_or_else(|| "M1".to_string());
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

fn parse_root(args: &[String]) -> String {
    parse_flag(args, "--root").unwrap_or_else(|| "data".to_string())
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

fn print_help() {
    println!("forex-cli");
    println!("  symbols --root data");
    println!("  timeframes --symbol EURUSD --root data");
    println!("  load --symbol EURUSD --timeframe M1 --root data");
    println!("  features --symbol EURUSD --timeframe M1 --root data");
    println!("  prepare --symbol EURUSD --base M1 --higher H1,H4 --root data");
    println!("  resample --symbol EURUSD --base M1 --target H1 --root data");
    println!("  train --symbol EURUSD --base M1 --higher H1,H4 --horizon 1 --root data");
    println!("  search --symbol EURUSD --base M1 --higher H1,H4 --genes 64 --generations 5 --max-indicators 12 --root data");
    println!("  discover --symbol EURUSD --base M1 --higher H1,H4 --population 100 --generations 5 --max-indicators 12 --portfolio-size 100 --candidates 200 --corr 0.7 --min-trades 1 --out cache/talib_knowledge.json --root data");
    println!("  stop-target --symbol EURUSD --timeframe M1 --pip 0.0001 --signal 1 --root data");
}
