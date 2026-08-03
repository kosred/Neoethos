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

    let mut settings = neoethos_core::Settings::load()
        .context("loading config.yaml — the probe uses the same settings as a real run")?;

    // Knobs that decide whether the pipeline has anything to do.
    //
    // Measured on M3: 2 046 of 2 048 genes per generation are rejected by the
    // archive's profit threshold, so the quality screen and validation — half
    // the run on H1 — never execute, and a "75 s M3 run" measures an empty
    // pipeline. Reaching those knobs meant editing config.yaml and rebuilding,
    // which is not something you do a hundred times.
    //
    // `--archive-mode active` keeps every gene that trades regardless of
    // profit. That is the right setting for timing a full pipeline and the
    // wrong one for judging strategies; it is a flag precisely so the two
    // cannot be confused.
    if let Some(mode) = flag(&args, "--archive-mode") {
        settings.models.search_runtime.archive_mode = mode;
    }
    if let Some(min_net) = flag(&args, "--archive-min-net").and_then(|v| v.parse::<f64>().ok()) {
        settings.models.search_runtime.archive_min_net = min_net;
    }
    // What the CLI does at startup, and what this probe never did.
    //
    // `neoethos-cli` calls this once after loading Settings; without it the
    // genetic knobs fall back to `unwrap_or_default()`, so archive mode,
    // selection policy, SMC gate and immigrant ratio all silently ignored
    // config.yaml. The probe reported `archive_mode=active` from its own flag
    // while the GA ran `mode=net` — the flag was read, applied to Settings, and
    // then never installed.
    //
    // Every timing taken with this probe before now used GA defaults rather
    // than the configured search. The comparisons between them still hold; a
    // comparison against a CLI run would not have.
    neoethos_search::install_search_runtime_overrides_from_settings(&settings);

    let mut config = neoethos_search::DiscoveryConfig::from_settings(&settings);
    if let Some(min_trades) = flag(&args, "--min-trades-per-day").and_then(|v| v.parse::<f64>().ok())
    {
        config.min_trades_per_day = min_trades;
    }
    // The base filter, which is where the candidates actually go.
    //
    // Measured on M3: ranked=22 486 -> post_passes_filter=2. The archive was
    // already open (`--archive-mode active`); it is these floors that reject
    // 99.99 %, and with two survivors the pipeline never reaches validation —
    // so a timing taken there measures the GA alone and nothing said so.
    //
    // `apply_mode_overrides` runs inside `from_settings`, so these are applied
    // after it: a flag the mode could silently overwrite would be worse than no
    // flag at all.
    if let Some(v) = flag(&args, "--min-win-rate").and_then(|v| v.parse::<f64>().ok()) {
        config.filtering.min_win_rate = v;
    }
    if let Some(v) = flag(&args, "--max-dd").and_then(|v| v.parse::<f64>().ok()) {
        config.filtering.max_dd = v;
    }
    if let Some(v) = flag(&args, "--min-profit-factor").and_then(|v| v.parse::<f64>().ok()) {
        config.filtering.min_profit_factor = v;
    }
    if let Some(v) = flag(&args, "--min-sharpe").and_then(|v| v.parse::<f64>().ok()) {
        config.filtering.min_sharpe = v;
    }
    tracing::info!(
        archive_mode = %settings.models.search_runtime.archive_mode,
        archive_min_net = settings.models.search_runtime.archive_min_net,
        min_trades_per_day = config.min_trades_per_day,
        min_win_rate = config.filtering.min_win_rate,
        max_dd = config.filtering.max_dd,
        min_profit_factor = config.filtering.min_profit_factor,
        min_sharpe = config.filtering.min_sharpe,
        "probe knobs in effect"
    );
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
    // Only the timeframes this run uses. The unrestricted loader pulls every
    // series in the store, and M1 alone is 5.27 M bars — minutes of load time
    // and gigabytes of RAM for data the probe never reads.
    let wanted: Vec<&str> = std::iter::once(base.as_str())
        .chain(higher_refs.iter().copied())
        .collect();
    let dataset = neoethos_data::load_symbol_dataset_with_timeframes(&root, &symbol, &wanted)
        .with_context(|| format!("loading {symbol} {wanted:?} from {root}"))?;
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

    // Deliberately NOT a throughput figure comparable to the bench.
    //
    // The first version divided candidate-bars by this elapsed time and printed
    // 1.49 M/s. Wrong twice over: it used the feature frame's 1 757 261 bars
    // while discovery had capped the run to 439 315, and it divided by dataset
    // load, feature build, validation and Monte-Carlo as well as the evaluation
    // the bench actually times. Both errors inflate, and the result still read
    // like a measurement.
    //
    // The comparable number is already in this log — `eval`'s own per-population
    // line carries the genes and bars it really evaluated. What belongs here is
    // the shape of the run.
    tracing::info!(
        population,
        generations,
        feature_frame_bars = bars,
        total_seconds = format!("{elapsed:.1}"),
        portfolio = result.portfolio.len(),
        candidates = result.candidates.len(),
        "PROBE RESULT — whole-cycle timing, not throughput. For throughput read \
         the `event budget usage` lines: candidate-bars is n_genes x bars there, \
         over the interval between consecutive population evaluations."
    );
    report_profile_distribution(&result);
    Ok(())
}

/// What shape of strategy the search actually produced.
///
/// The operator's target is 57-65 % of trades winning at about 2.2:1. Gating on
/// that is one line; knowing whether the search can *reach* it is the question,
/// and a gate that rejects everything answers it only by silence. So the
/// distribution is printed with no gate applied — quantiles rather than a mean,
/// because a handful of extreme candidates would otherwise decide the answer.
fn report_profile_distribution(result: &neoethos_search::DiscoveryResult) {
    let metrics = &result.quality_metrics;
    if metrics.is_empty() {
        tracing::info!(
            target: "gpu_discovery_probe",
            "no candidates were scored, so there is no distribution to report"
        );
        return;
    }
    let quantiles = |mut values: Vec<f64>| -> (f64, f64, f64, f64) {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| values[((values.len() as f64 * q) as usize).min(values.len() - 1)];
        (at(0.10), at(0.50), at(0.90), at(0.99))
    };
    let (wr10, wr50, wr90, wr99) = quantiles(metrics.iter().map(|m| m.win_rate).collect());
    let (pr10, pr50, pr90, pr99) = quantiles(metrics.iter().map(|m| m.payoff_ratio).collect());
    let (im10, im50, im90, im99) = quantiles(metrics.iter().map(|m| m.in_market_pct).collect());

    tracing::info!(
        target: "gpu_discovery_probe",
        candidates = metrics.len(),
        "PROFILE DISTRIBUTION — what the search actually produces"
    );
    tracing::info!(target: "gpu_discovery_probe",
        "  win rate      p10 {:.0}%  p50 {:.0}%  p90 {:.0}%  p99 {:.0}%",
        wr10 * 100.0, wr50 * 100.0, wr90 * 100.0, wr99 * 100.0);
    tracing::info!(target: "gpu_discovery_probe",
        "  payoff        p10 {pr10:.2}  p50 {pr50:.2}  p90 {pr90:.2}  p99 {pr99:.2}");
    tracing::info!(target: "gpu_discovery_probe",
        "  in market     p10 {:.0}%  p50 {:.0}%  p90 {:.0}%  p99 {:.0}%",
        im10 * 100.0, im50 * 100.0, im90 * 100.0, im99 * 100.0);

    // The joint condition is the real question: plenty of candidates clear each
    // bar separately while none clears both, and only the pair is the target.
    let wins_enough = metrics.iter().filter(|m| m.win_rate >= 0.57).count();
    let pays_enough = metrics.iter().filter(|m| m.payoff_ratio >= 2.2).count();
    let both = metrics
        .iter()
        .filter(|m| m.win_rate >= 0.57 && m.payoff_ratio >= 2.2)
        .count();
    let selective = metrics
        .iter()
        .filter(|m| m.win_rate >= 0.57 && m.payoff_ratio >= 2.2 && m.in_market_pct <= 0.35)
        .count();
    tracing::info!(target: "gpu_discovery_probe",
        "  meeting the target: win>=57% {wins_enough} | payoff>=2.2 {pays_enough} |          BOTH {both} | both + under 35% exposure {selective}  (of {} scored)",
        metrics.len());
}
