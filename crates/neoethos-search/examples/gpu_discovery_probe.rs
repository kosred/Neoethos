//! One real discovery on real data, instrumented, without the ML stack.
//!
//! The GPU question needs a real workload to answer: the bench reaches 47-50 M
//! candidate-bars per second on the same kernel where discovery manages 0.46.
//! The suspicion is that events are counted per in-signal bar rather than per
//! trade — `population_count_events_kernel` increments once for every bar whose
//! signal is non-zero — so a gene that is in the market most of the time emits
//! hundreds of thousands of events where it makes a few thousand trades. Past
//! the session's capacity the population is split into ever smaller batches and
//! per-launch cost takes over.
//!
//! Running the CLI would answer it too, but the CLI links the whole model stack
//! (torch, catboost, xgboost) which costs hours to build on a rented card and
//! once crashed with SIGILL on a host whose CPU the vendored code did not
//! expect. `neoethos-search` depends only on data, core, and the CUDA shim, so
//! this example is minutes to build and exercises exactly the path in question.
//!
//! Usage:
//!   gpu_discovery_probe --root <store> --symbol EURUSD --base M3 \
//!       [--higher M15,H1,H4] [--population 512] [--generations 3]

use anyhow::{Context, Result};
use std::time::Instant;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let root = flag(&args, "--root").unwrap_or_else(|| "data".to_string());
    let symbol = flag(&args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base = flag(&args, "--base").unwrap_or_else(|| "M3".to_string());
    let higher = flag(&args, "--higher").unwrap_or_else(|| "M15,H1,H4".to_string());
    let higher_list: Vec<String> = higher
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

    let settings = neoethos_core::Settings::load()
        .context("loading config.yaml — the probe uses the same settings as a real run")?;
    let mut config = neoethos_search::DiscoveryConfig::from_settings(&settings);
    config.evaluation_symbol = symbol.clone();
    if let Some(population) = flag(&args, "--population").and_then(|v| v.parse::<usize>().ok()) {
        config.population = population;
    }
    if let Some(generations) = flag(&args, "--generations").and_then(|v| v.parse::<usize>().ok()) {
        config.generations = generations;
    }
    // The FX resolver needs the store this probe was pointed at, not the one
    // config.yaml happens to name.
    neoethos_search::fx_rates::set_store_root(&root);

    tracing::info!(root, symbol, base, higher, "loading dataset");
    let loaded = Instant::now();
    let dataset = neoethos_data::load_symbol_dataset(&root, &symbol)
        .with_context(|| format!("loading {symbol} from {root}"))?;
    let features = neoethos_data::prepare_multitimeframe_features(
        &dataset,
        &base,
        &higher_refs,
        None,
    )
    .context("building multi-timeframe features")?;
    let base_ohlcv = dataset
        .frames
        .get(&base)
        .with_context(|| format!("{symbol} has no {base} series in this store"))?;
    let bars = features.n_samples();
    tracing::info!(
        bars,
        features = features.names.len(),
        seconds = format!("{:.1}", loaded.elapsed().as_secs_f64()),
        "dataset ready"
    );

    let population = config.population;
    let generations = config.generations;
    let started = Instant::now();
    let result = neoethos_search::run_discovery_cycle(&features, base_ohlcv, &config)?;
    let elapsed = started.elapsed().as_secs_f64();

    // Candidate-bars is the unit the bench and the scaling sweep report, so the
    // two numbers can be put side by side. One generation evaluates the whole
    // population against every bar.
    let candidate_bars = population as f64 * bars as f64 * generations.max(1) as f64;
    tracing::info!(
        population,
        generations,
        bars,
        seconds = format!("{elapsed:.1}"),
        throughput_m_cand_bars_per_s = format!("{:.2}", candidate_bars / elapsed / 1.0e6),
        portfolio = result.portfolio.len(),
        candidates = result.candidates.len(),
        "PROBE RESULT — compare against 47-50 M/s in gpu_eval_bench and 966 M/s \
         at population 131 072 in the scaling sweep"
    );
    Ok(())
}
