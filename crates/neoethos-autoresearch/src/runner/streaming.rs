//! The real executor: one search = one streaming working-set sweep on the card.
//!
//! This is the only place in the crate that reads a bar. It exists behind
//! [`super::SweepExecutor`] so the state machine — the part that has to be
//! provably resumable and provably unable to reach a broker — can be exercised
//! end to end without a GPU, a dataset or a feature cube.
//!
//! ## Two decisions made here, once, loudly
//!
//! **1. Every search is charged at the PESSIMISTIC edge of the frozen cost
//! band.** `evaluation_spread_pips` is set to `cost_band_pips.1` and
//! `evaluation_commission_per_trade` to `0.0`, so the whole round trip is the
//! pessimistic edge and `StrategyMetrics::profit_per_trade` **is**
//! `E_screen_pess` with no unit conversion anywhere. This is not the loop
//! varying a frozen field: the value is fully determined by `cost_band_pips`,
//! which no `SearchConfigDelta` can touch. It is `docs/autoresearch-loop.md` §8
//! — *"charge the COST BAND, not a point estimate"* — applied at the only place
//! where it costs nothing: a result that survives only at the optimistic edge is
//! not a result, so the loop simply never computes one.
//!
//! **2. The loaded span STOPS before the OOS window.** The dataset is sliced by
//! time at `oos_start_ms` before a single feature is built, and
//! [`super::SweepExecutor::windows`] reports the bounds it actually used so the
//! startup self-check can compare them. A leaked out-of-sample window cannot be
//! un-leaked.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;

use neoethos_search::discovery::DiscoveryConfig;
use neoethos_search::genetic::Gene;

use crate::journal::{CostBandCounts, OosWindow};
use crate::session::SweepId;

use super::{OosEvidence, SearchOutcome, SearchRequest, SweepExecutor};

/// The fraction of the series, by time, held out and touched once.
const OOS_FRACTION: f64 = 0.20;

/// One search = one discovery run over a bounded working-set sweep.
pub struct StreamingSweepExecutor {
    /// The dataset as loaded, SLICED to end before the OOS window.
    in_sample: neoethos_data::SymbolDataset,
    /// The full dataset, kept for the single out-of-sample evaluation.
    full: neoethos_data::SymbolDataset,
    base_timeframe: String,
    higher_timeframes: Vec<String>,
    search_span_ms: (i64, i64),
    oos_window: OosWindow,
    oos_bars: usize,
    bars_per_expected_trade: f64,
    streaming_requested: bool,
    /// Where per-search discovery ledgers are written before their matrices are
    /// copied into the session store.
    scratch: PathBuf,
}

impl StreamingSweepExecutor {
    /// Load the data, derive the OOS window, and bound the search span.
    pub fn resolve(
        settings: &neoethos_core::config::Settings,
        base_config: &DiscoveryConfig,
    ) -> Result<Self> {
        let root = settings.system.data_dir.clone();
        let symbol = base_config.evaluation_symbol.clone();
        if symbol.trim().is_empty() {
            bail!(
                "the autoresearch loop was asked to run with an empty symbol. It will not pick \
                 one: which instrument is being searched is the operator's decision."
            );
        }
        let full = neoethos_data::load_symbol_dataset(&root, &symbol).with_context(|| {
            format!(
                "loading {symbol} from {}. The autoresearch loop reads data and never downloads \
                 it.",
                root.display()
            )
        })?;

        let base_timeframe = base_config.timeframe_label.clone();
        let base = full.timeframe(&base_timeframe).ok_or_else(|| {
            anyhow::anyhow!(
                "{symbol} has no {base_timeframe} frame; available: {:?}",
                full.timeframes()
            )
        })?;
        let timestamps = base.timestamp.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{symbol} {base_timeframe} has no timestamp column, so the out-of-sample window \
                 cannot be defined BY TIME. It will not be defined by row index instead: a row \
                 split across a data gap is not a time split, and the whole point of the window \
                 is that nothing in the search has seen it."
            )
        })?;
        if timestamps.len() < 100 {
            bail!(
                "{symbol} {base_timeframe} has only {} bars — too few to split into a search span \
                 and an out-of-sample window",
                timestamps.len()
            );
        }

        // The OOS window is the final OOS_FRACTION of the span BY TIME.
        let first = *timestamps.first().expect("non-empty");
        let last = *timestamps.last().expect("non-empty");
        let span = (last - first).max(1);
        let oos_start_ms = last - ((span as f64) * OOS_FRACTION) as i64;
        let oos_window = OosWindow {
            start_ms: oos_start_ms,
            end_ms: last,
        };
        let oos_bars = timestamps.iter().filter(|t| **t >= oos_start_ms).count();

        // The search span STOPS one millisecond before the window opens.
        let search_span_ms = (first, oos_start_ms - 1);

        let mut sliced: HashMap<String, neoethos_data::Ohlcv> = HashMap::new();
        for (tf, frame) in &full.frames {
            let (cut, _) = neoethos_data::slice_ohlcv_by_date_range_ms(
                frame,
                search_span_ms.0,
                search_span_ms.1,
            )
            .map_err(|err| {
                anyhow::anyhow!("slicing {symbol} {tf} to the in-sample span failed: {err}")
            })?;
            sliced.insert(tf.clone(), cut);
        }
        let in_sample = neoethos_data::SymbolDataset {
            symbol: symbol.clone(),
            frames: sliced,
        };

        // Bars per expected trade, from the operator's own trade-rate floor and
        // the base frame's own bar spacing. Derived, so the OOS length check
        // means the same thing on M5 and on H1.
        let bars_per_day = bars_per_day(timestamps);
        let bars_per_expected_trade = if base_config.min_trades_per_day > 0.0 {
            bars_per_day / base_config.min_trades_per_day
        } else {
            bars_per_day
        };

        let scratch = std::env::temp_dir()
            .join("neoethos-autoresearch")
            .join(&symbol);
        std::fs::create_dir_all(&scratch)
            .with_context(|| format!("creating the scratch dir {}", scratch.display()))?;

        Ok(Self {
            in_sample,
            full,
            base_timeframe,
            higher_timeframes: base_config.higher_timeframes.clone(),
            search_span_ms,
            oos_window,
            oos_bars,
            bars_per_expected_trade,
            streaming_requested: true,
            scratch,
        })
    }

    /// The pessimistic-edge cost model, applied to a resolved config.
    ///
    /// Refuses rather than guessing when the band is absent: the startup
    /// self-check already aborts on a band that cannot discriminate, so reaching
    /// here without one means the check was bypassed.
    fn charge_pessimistic_edge(config: &DiscoveryConfig) -> Result<DiscoveryConfig> {
        let Some((_, pessimistic)) = config.cost_band_pips else {
            bail!(
                "no cost_band_pips on the resolved configuration, so the pessimistic edge cannot \
                 be charged. The startup self-check aborts on this; reaching a search without a \
                 band means it was bypassed."
            );
        };
        Ok(DiscoveryConfig {
            evaluation_spread_pips: pessimistic,
            evaluation_commission_per_trade: 0.0,
            ..config.clone()
        })
    }
}

fn bars_per_day(timestamps: &[i64]) -> f64 {
    const MS_PER_DAY: f64 = 86_400_000.0;
    let first = *timestamps.first().unwrap_or(&0);
    let last = *timestamps.last().unwrap_or(&0);
    let days = ((last - first).max(1) as f64 / MS_PER_DAY).max(1.0);
    (timestamps.len() as f64 / days).max(1.0)
}

impl SweepExecutor for StreamingSweepExecutor {
    fn describe(&self) -> String {
        format!(
            "streaming working-set sweep on {} {} (+{:?}); search span {}..{}, OOS {}..{} \
             ({} bars, touched once); every search charged at the PESSIMISTIC edge of the frozen \
             cost band",
            self.in_sample.symbol,
            self.base_timeframe,
            self.higher_timeframes,
            self.search_span_ms.0,
            self.search_span_ms.1,
            self.oos_window.start_ms,
            self.oos_window.end_ms,
            self.oos_bars
        )
    }

    fn streaming_requested(&self) -> bool {
        self.streaming_requested
    }

    fn windows(&self) -> Result<((i64, i64), OosWindow, usize, f64)> {
        Ok((
            self.search_span_ms,
            self.oos_window,
            self.oos_bars,
            self.bars_per_expected_trade,
        ))
    }

    fn execute(&mut self, request: &SearchRequest<'_>) -> Result<SearchOutcome> {
        let began = std::time::Instant::now();
        let config = Self::charge_pessimistic_edge(request.config)?;

        // Each search writes its ledger into its own directory, so 100 searches
        // in one sweep cannot overwrite one another's trial-returns matrix.
        let ledger_dir = self
            .scratch
            .join(request.sweep.to_string())
            .join(format!("slot_{:03}", request.slot));
        std::fs::create_dir_all(&ledger_dir)?;
        let config = DiscoveryConfig {
            discovery_ledger_enabled: true,
            discovery_ledger_cache_dir: ledger_dir.to_string_lossy().to_string(),
            ..config
        };

        let base = self.in_sample.timeframe(&self.base_timeframe).ok_or_else(|| {
            anyhow::anyhow!("the in-sample dataset lost its {} frame", self.base_timeframe)
        })?;
        let higher: Vec<&str> = self.higher_timeframes.iter().map(String::as_str).collect();

        let plan = neoethos_search::orchestration::StreamingPlan::streaming(0);
        let dataset = &self.in_sample;
        let base_tf = self.base_timeframe.clone();
        let permutation = request.permutation;

        let outcome = neoethos_search::orchestration::run_streaming_working_set(
            &plan,
            base.close.len(),
            // The ONLY sanctioned build entry point: it installs the batch as
            // the working set and restores the previous one afterwards, even on
            // panic.
            |batch| {
                let mut frame = neoethos_data::prepare_multitimeframe_features_batch(
                    dataset, &base_tf, &higher, batch,
                )?;
                // THE SHUFFLE CONTROL, applied at the only correct point: after
                // the features exist and before the search sees them. Prices,
                // labels, costs, exit geometry, gene encoding and the GA seed
                // are untouched — only the relationship between the features
                // and the future is destroyed.
                if let Some(permutation) = permutation {
                    permutation.apply(&mut frame).context(
                        "applying the shuffle control's permutation to the feature block",
                    )?;
                }
                Ok(frame)
            },
            |features| {
                neoethos_search::run_discovery_cycle_with_holdout(
                    features,
                    base,
                    &config,
                    neoethos_search::PropFirmRiskRules::default(),
                )
            },
        );

        let wall_ms = began.elapsed().as_millis() as u64;

        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                // A per-SEARCH failure. Named, counted, and the sweep continues:
                // one bad region must not throw away the regions already
                // searched.
                return Ok(SearchOutcome {
                    slot: request.slot,
                    config_hash: request.config_hash.to_string(),
                    trials_offered: 0,
                    statistics: neoethos_search::deflated::TrialStatisticsReport::unreadable(
                        format!("the search itself failed: {err:#}"),
                    ),
                    // `discriminates: false` on a failed search, so S3 refuses
                    // it. A failure must never inherit a passing band verdict.
                    cost_band: CostBandCounts::default(),
                    rejections: Vec::new(),
                    survivors: 0,
                    e_screen_pess: None,
                    n_trades: 0,
                    champion_returns: Vec::new(),
                    champion_period_keys: Vec::new(),
                    champion_strategy_id: String::new(),
                    streamed: false,
                    batch_columns: 0,
                    next_cursor: 0,
                    wall_ms,
                    error: Some(format!("{err:#}")),
                });
            }
        };

        // ── S5 COLLECT ──────────────────────────────────────────────────────
        let manifest = neoethos_search::trial_returns::load_manifest(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
        );
        // `trials_offered` is the honest denominator the writer recorded — every
        // candidate the screen was offered, not the survivors. It is read from
        // the manifest rather than counted here, so the loop's N and the DSR's N
        // are the same number from the same source.
        let trials_offered = manifest.as_ref().map(|m| m.trials_offered).unwrap_or(0);

        let statistics = match neoethos_search::deflated::read_matrix(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
        ) {
            Ok(matrix) => {
                // Copy the binary into the session store BEFORE the scratch dir
                // is reused, so the promotion candidate's evidence survives.
                let source = neoethos_search::trial_returns::binary_path(
                    &config.discovery_ledger_cache_dir,
                    &config.evaluation_symbol,
                    &config.timeframe_label,
                );
                if let Some(parent) = request.trial_returns_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let _ = std::fs::copy(&source, &request.trial_returns_path);
                neoethos_search::deflated::analyse_matrix(&matrix, trials_offered)
            }
            Err(err) => neoethos_search::deflated::TrialStatisticsReport::unreadable(format!(
                "{err:#}"
            )),
        };

        // The best survivor, and the number that decides the screen. Because
        // the run was charged at the pessimistic band edge,
        // `profit_per_trade` IS `E_screen_pess` — no conversion, no assumed lot
        // size, no second cost model.
        let mut best: Option<&neoethos_search::quality::StrategyMetrics> = None;
        let mut survivors = 0usize;
        for batch in &outcome.batches {
            for metrics in &batch.result.quality_metrics {
                survivors += 1;
                if best.is_none_or(|b| metrics.profit_per_trade > b.profit_per_trade) {
                    best = Some(metrics);
                }
            }
        }

        let rejections = outcome
            .batches
            .iter()
            .filter_map(|b| b.result.funnel_profile.as_ref())
            .flat_map(|profile| {
                profile
                    .stages
                    .iter()
                    .flat_map(|stage| stage.top_reasons.iter().cloned())
            })
            .fold(HashMap::<String, usize>::new(), |mut acc, (name, count)| {
                *acc.entry(name).or_default() += count;
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();

        let (champion_returns, champion_period_keys, champion_strategy_id) = match &best {
            Some(metrics) => champion_series(&config, metrics),
            None => (Vec::new(), Vec::new(), String::new()),
        };

        Ok(SearchOutcome {
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            trials_offered,
            statistics,
            // The census is produced INSIDE the search and `DiscoveryResult`
            // has no field carrying it out (§17.5), so every survivor is
            // reported as `unmeasured` and `discriminates` is carried from the
            // band the run actually charged. That combination FAILS screen
            // conjunct S3 rather than passing it silently: an unmeasured band
            // recorded as `survives` would make every census read clean, which
            // is worse than no census because a reader takes it as evidence.
            cost_band: CostBandCounts {
                unmeasured: survivors,
                discriminates: neoethos_search::discovery::cost_band_discriminates(
                    request.config.cost_band_pips,
                    neoethos_search::run_identity::cost_pips_round_trip(
                        request.config.evaluation_spread_pips,
                        request.config.evaluation_commission_per_trade,
                        request.config.evaluation_config(None).pip_value_per_lot,
                    ),
                ),
                ..CostBandCounts::default()
            },
            rejections,
            survivors,
            e_screen_pess: best.map(|m| m.profit_per_trade),
            n_trades: best.map(|m| m.total_trades).unwrap_or(0),
            champion_returns,
            champion_period_keys,
            champion_strategy_id,
            streamed: outcome.streamed,
            batch_columns: outcome.batch_columns,
            next_cursor: outcome.next_cursor,
            wall_ms,
            error: None,
        })
    }

    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        config: &DiscoveryConfig,
    ) -> Result<OosEvidence> {
        // The adaptive-stop hazard, refused rather than approximated.
        //
        // When adaptive stops are installed, a gene's effective SL is
        // `stop_vol_mult x` the dataset's per-bar volatility rather than its own
        // `sl_pips`, and `neoethos-search` deliberately keeps the resolver
        // private (`GeneEvalSettingsResolver`) so no caller can screen a
        // DIFFERENT strategy from the one that was scored — measured, the
        // divergence was 30,331 trades against 1,727 on one signal. There is no
        // public entry point that resolves them from outside the crate, so
        // rather than reconstruct the rule here and quietly evaluate a
        // different strategy, this refuses and names what is missing.
        if neoethos_search::stop_target::adaptive_stops_enabled() {
            bail!(
                "adaptive stops are installed, and there is no PUBLIC neoethos-search entry point \
                 that resolves a gene's effective stop from outside the crate \
                 (`GeneEvalSettingsResolver` is pub(crate)). Reconstructing the rule here would \
                 evaluate a DIFFERENT strategy from the one that was scored — measured at 30,331 \
                 trades against 1,727 on one signal — and it would do so on the one window this \
                 whole design exists to protect. REQUIRED ELSEWHERE: a public \
                 `neoethos_search::evaluate_portfolio_on_window(genes, features, ohlcv, config, \
                 from_ms, to_ms) -> Vec<Trade>` that routes through the same resolver the GA \
                 scored with. Until it exists, promotion is refused rather than approximated."
            );
        }

        let config = Self::charge_pessimistic_edge(config)?;
        let base = self.full.timeframe(&self.base_timeframe).ok_or_else(|| {
            anyhow::anyhow!("the dataset lost its {} frame", self.base_timeframe)
        })?;
        let timestamps = base
            .timestamp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the base frame has no timestamp column"))?;

        // Features are built over the FULL series on purpose: every indicator
        // here is causal, so a value at bar `i` is a function of bars `<= i`
        // only. Building them over the whole series and then reading ONLY the
        // out-of-sample tail leaks nothing forward, and it gives the tail the
        // same warm-up the search had — which slicing first would not.
        let higher: Vec<&str> = self.higher_timeframes.iter().map(String::as_str).collect();
        let features = neoethos_data::prepare_multitimeframe_features(
            &self.full,
            &self.base_timeframe,
            &higher,
        )?;

        let first_oos = timestamps
            .iter()
            .position(|t| *t >= self.oos_window.start_ms)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no bar falls inside the out-of-sample window {}..{}",
                    self.oos_window.start_ms,
                    self.oos_window.end_ms
                )
            })?;

        let genes = load_promotion_genes(&self.scratch, sweep, slot)?;
        if genes.is_empty() {
            bail!(
                "{sweep} slot {slot} has no recorded portfolio genes, so there is nothing to \
                 evaluate out of sample"
            );
        }

        let mut per_trade_net: Vec<f64> = Vec::new();
        let mut r_multiples: Vec<f64> = Vec::new();
        let mut by_month: HashMap<i64, f64> = HashMap::new();

        for gene in &genes {
            let signals = neoethos_search::genetic::signals_for_gene(&features, gene);
            let settings = oos_backtest_settings(&config, gene);
            let trades = neoethos_search::eval::simulate_trades_core(
                &base.close,
                &base.high,
                &base.low,
                timestamps,
                &signals,
                &settings,
            );
            for trade in trades
                .iter()
                .filter(|t| t.entry_time >= self.oos_window.start_ms)
            {
                per_trade_net.push(trade.pnl);
                r_multiples.push(trade.r_multiple);
                let month = month_key(trade.entry_time);
                *by_month.entry(month).or_default() += trade.pnl;
            }
        }

        let _ = first_oos;
        let mut period_keys: Vec<i64> = by_month.keys().copied().collect();
        period_keys.sort_unstable();
        let monthly_returns: Vec<f64> = period_keys
            .iter()
            .map(|k| by_month.get(k).copied().unwrap_or(0.0) / config.initial_balance.max(1.0))
            .collect();

        const MS_PER_DAY: f64 = 86_400_000.0;
        let days = ((self.oos_window.end_ms - self.oos_window.start_ms).max(1) as f64
            / MS_PER_DAY)
            .max(1.0);

        Ok(OosEvidence {
            window: self.oos_window,
            trades_per_day: per_trade_net.len() as f64 / days,
            // Every number above was booked at the PESSIMISTIC edge, so the
            // band verdict is by construction and not by a second pass.
            band_survives: per_trade_net.iter().sum::<f64>() > 0.0,
            per_trade_net_pips: per_trade_net,
            r_multiples,
            monthly_returns,
            period_keys,
        })
    }
}

/// UTC month key, as `trial_returns::month_keys_spanning` produces them.
fn month_key(ms: i64) -> i64 {
    neoethos_search::trial_returns::month_keys_spanning(ms, ms)
        .first()
        .copied()
        .unwrap_or(ms)
}

/// The champion's per-month return series — this search's candidate row for the
/// session champion matrix (`pbo_session`).
///
/// Read from the trial-returns matrix the run just wrote, which is where the
/// per-trial series already lives; nothing is recomputed, so the row and the
/// statistics beside it describe the same trials.
fn champion_series(
    config: &DiscoveryConfig,
    metrics: &neoethos_search::quality::StrategyMetrics,
) -> (Vec<f64>, Vec<i64>, String) {
    let Ok(matrix) = neoethos_search::deflated::read_matrix(
        &config.discovery_ledger_cache_dir,
        &config.evaluation_symbol,
        &config.timeframe_label,
    ) else {
        return (Vec::new(), Vec::new(), metrics.strategy_id.clone());
    };
    let row = matrix
        .rows
        .iter()
        .find(|r| r.strategy_id == metrics.strategy_id);
    match row {
        Some(row) => (
            row.returns.clone(),
            matrix.period_keys.clone(),
            metrics.strategy_id.clone(),
        ),
        None => (Vec::new(), Vec::new(), metrics.strategy_id.clone()),
    }
}

/// The promotion candidate's genes, from the survivors artifact the sweep
/// wrote.
fn load_promotion_genes(scratch: &std::path::Path, sweep: SweepId, slot: usize) -> Result<Vec<Gene>> {
    let path = scratch
        .join(sweep.to_string())
        .join(format!("slot_{slot:03}"))
        .join("survivors.json");
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "reading the promotion candidate's survivors from {}. Without them the OOS \
             evaluation has nothing to evaluate, and it will NOT re-search to find something.",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))
}

/// Backtest settings for an already-selected gene on the OOS window.
///
/// Only reached when adaptive stops are OFF — see the refusal in
/// [`StreamingSweepExecutor::evaluate_oos`] — so the gene's own `sl_pips` /
/// `tp_pips` are the effective stops, exactly as they were when it was scored.
fn oos_backtest_settings(
    config: &DiscoveryConfig,
    gene: &Gene,
) -> neoethos_search::eval::BacktestSettings {
    let evaluation = config.evaluation_config(None);
    let mut settings = neoethos_search::eval::BacktestSettings::default();
    settings.sl_pips = if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
        gene.sl_pips
    } else {
        20.0
    };
    settings.tp_pips = if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
        gene.tp_pips
    } else {
        40.0
    };
    settings.max_hold_bars = gene.max_hold_bars;
    settings.pip_value_per_lot = evaluation.pip_value_per_lot;
    settings.spread_pips = config.evaluation_spread_pips;
    settings.commission_per_trade = config.evaluation_commission_per_trade;
    settings
}
