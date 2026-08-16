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
        // EMPTIED FIRST. The scratch root is keyed by SYMBOL while sweep ids
        // restart at 1 in every session, so this exact directory may already
        // hold a PREVIOUS session's matrix and manifest — written under a
        // different cost model and a different goal set. Matrix writes are
        // non-fatal by design, so if this search's write fails, whatever is left
        // here is what S5 would read back. Removing it first means a failed
        // write reads as "no matrix" instead of as "somebody else's matrix".
        //
        // Nothing durable lives here: the matrix and the promotion evidence are
        // copied into the SESSION STORE at S5, and this is the only writer.
        if ledger_dir.exists() {
            std::fs::remove_dir_all(&ledger_dir).with_context(|| {
                format!(
                    "emptying the scratch ledger directory {} before {} slot {} runs. It is \
                     keyed by symbol and sweep, not by session, so a stale matrix left here \
                     would be attributed to this search.",
                    ledger_dir.display(),
                    request.sweep,
                    request.slot
                )
            })?;
        }
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
                    apply_shuffle_control(&permutation, &mut frame).context(
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
        //
        // ATTRIBUTION FIRST. Everything below is read back out of
        // `discovery_ledger_cache_dir`, and until this check existed it was read
        // back BY PATH ALONE. `TrialReturnsManifest::config_hash` carries the
        // identity of the run that wrote the matrix and its doc comment names
        // this exact attack; the loop simply never read it.
        //
        // The attack is not hypothetical here. The scratch root is
        // `temp_dir()/neoethos-autoresearch/<symbol>` — keyed by SYMBOL, not by
        // session — while sweep ids restart at 1 in every session, so session
        // B's `sweep 1/slot_007` IS session A's directory. Matrix writes are
        // non-fatal by design, so a failed write in B leaves A's matrix sitting
        // there, and B would have adopted it: a different cost model, a
        // different goal set, and DSR/PBO asserted computable on that basis.
        //
        // `None` is NOT a match. It means the writer had no stamp, which is
        // "cannot be attributed" — and an unattributable matrix is exactly as
        // unusable as a foreign one.
        let manifest = neoethos_search::trial_returns::load_manifest(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
        );
        let attribution = attribute_manifest(
            manifest.as_ref().map(|m| m.config_hash.as_deref()),
            request.config_hash,
            &config.discovery_ledger_cache_dir,
        );

        // `trials_offered` is the honest denominator the writer recorded — every
        // candidate the screen was offered, not the survivors. It is read from
        // the manifest rather than counted here, so the loop's N and the DSR's N
        // are the same number from the same source. An unattributable manifest
        // contributes ZERO: its number belongs to some other run, and carrying
        // it would put another session's trials into this session's N.
        let trials_offered = match (&attribution, &manifest) {
            (Ok(()), Some(m)) => m.trials_offered,
            _ => 0,
        };

        let read = match &attribution {
            // NOT read at all. Reading it would also copy it into the session
            // store, where `pbo_session` and the promotion path both read it
            // back — which is how a foreign matrix would outlive the scratch
            // directory that explains it.
            Err(why) => Err(anyhow::anyhow!("{why}")),
            Ok(()) => neoethos_search::deflated::read_matrix(
                &config.discovery_ledger_cache_dir,
                &config.evaluation_symbol,
                &config.timeframe_label,
            ),
        };

        let statistics = match read {
            Ok(matrix) => {
                // Copy the binary into the session store BEFORE the scratch dir
                // is reused, so the promotion candidate's evidence survives.
                //
                // THE RESULT IS CHECKED. It used to be `let _ = …`, which is
                // the crate's `unused_must_use = "deny"` being defeated by hand
                // on the one line whose comment claims the evidence survives:
                // the scratch root is keyed by SYMBOL and is reused by the next
                // session, so a copy that silently did not happen is a matrix
                // that is gone by the time anything reads it — and `pbo_session`
                // and the promotion path both read it. A failure here is a disk
                // fault, not a research result, so it fails the SWEEP loudly
                // rather than producing a row whose evidence quietly is not
                // there.
                let source = neoethos_search::trial_returns::binary_path(
                    &config.discovery_ledger_cache_dir,
                    &config.evaluation_symbol,
                    &config.timeframe_label,
                );
                if let Some(parent) = request.trial_returns_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(&source, &request.trial_returns_path).with_context(|| {
                    format!(
                        "copying {} slot {}'s trial-returns matrix from {} into the session store \
                         at {}. The scratch directory is keyed by SYMBOL and is reused by every \
                         session, so a matrix that is not copied is a matrix that is gone — and \
                         the session champion matrix, pbo_session and the promotion path all read \
                         it back from the session store.",
                        request.sweep,
                        request.slot,
                        source.display(),
                        request.trial_returns_path.display()
                    )
                })?;
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

        // The champion's return series is read out of the SAME directory, so it
        // is governed by the SAME attribution. A series that cannot be
        // attributed must not reach the session champion matrix: it is the
        // series `pbo_session` is computed from, and a foreign one there would
        // move the number that gates every promotion.
        let (champion_returns, champion_period_keys, champion_strategy_id) =
            match (&attribution, &best) {
                (Ok(()), Some(metrics)) => champion_series(&config, metrics),
                (Err(_), Some(metrics)) => {
                    (Vec::new(), Vec::new(), metrics.strategy_id.clone())
                }
                (_, None) => (Vec::new(), Vec::new(), String::new()),
            };

        // ── S5 COLLECT: THE PROMOTION EVIDENCE ──────────────────────────────
        //
        // The genes this search selected, with the names their indices address,
        // written into the SESSION STORE under this slot's `config_hash`.
        //
        // It is written HERE — by the search, at the moment the genes exist —
        // and not reconstructed later, because later there is nothing to
        // reconstruct from: the scratch directory is keyed by SYMBOL and the
        // next session reuses it. A promotion candidate can be a sweep that ran
        // days earlier and is resumed into, and the one out-of-sample touch is
        // not a thing to be attempted on evidence that may or may not still be
        // on disk.
        //
        // A search that selected nothing writes NOTHING, so "no artifact" and
        // "an artifact that says nothing survived" are the same observable and
        // the promotion path refuses on both rather than evaluating an empty
        // portfolio and reporting its zero trades as a result.
        let genes_persisted = persist_promotion_evidence(request, &outcome)?;
        tracing::debug!(
            target: "neoethos_autoresearch::streaming",
            sweep = %request.sweep,
            slot = request.slot,
            config_hash = %request.config_hash,
            genes = genes_persisted,
            path = %request.promotion_evidence_path.display(),
            "promotion evidence written"
        );

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
                        request
                            .config
                            .try_evaluation_config(None)?
                            .pip_value_per_lot,
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

    /// Everything this executor can refuse WITHOUT reading an out-of-sample bar.
    ///
    /// The runner calls this BEFORE journalling `OosTouchSpent`, so each of
    /// these refusals leaves the single window unspent and a later resume — with
    /// adaptive stops off, or with the frame restored — can still use it. Every
    /// check here is repeated inside [`Self::evaluate_oos`], which is what makes
    /// skipping the preflight expensive rather than dangerous.
    fn oos_preflight(&self, portfolio: &super::PromotionPortfolio) -> Result<()> {
        neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::Promotion)
            .map_err(anyhow::Error::new)?;

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
        //
        // It refuses in the PREFLIGHT because it is a property of the build and
        // not of the candidate: it would have been true before the touch was
        // journalled as spent, and spending the window on a call that cannot
        // return is how an irreplaceable resource gets consumed by a bug.
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
        if portfolio.genes.is_empty() {
            bail!(
                "{} slot {} carries no genes, so there is nothing to evaluate out of sample",
                portfolio.sweep,
                portfolio.slot
            );
        }
        let base = self.full.timeframe(&self.base_timeframe).ok_or_else(|| {
            anyhow::anyhow!("the dataset lost its {} frame", self.base_timeframe)
        })?;
        let timestamps = base
            .timestamp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the base frame has no timestamp column"))?;
        if !timestamps.iter().any(|t| *t >= self.oos_window.start_ms) {
            bail!(
                "no bar falls inside the out-of-sample window {}..{}",
                self.oos_window.start_ms,
                self.oos_window.end_ms
            );
        }
        Ok(())
    }

    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        config: &DiscoveryConfig,
        portfolio: &super::PromotionPortfolio,
    ) -> Result<OosEvidence> {
        // Re-asserted rather than assumed: the runner calls this first, and a
        // check that only runs when somebody remembers to call it is not a
        // check.
        self.oos_preflight(portfolio)?;
        if portfolio.sweep != sweep || portfolio.slot != slot {
            bail!(
                "the evidence handed to the out-of-sample evaluation says it is {} slot {}, but \
                 the touch is being spent on {sweep} slot {slot}. Refusing to evaluate one \
                 configuration's genes under another's name.",
                portfolio.sweep,
                portfolio.slot
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
        let raw = neoethos_data::prepare_multitimeframe_features(
            &self.full,
            &self.base_timeframe,
            &higher,
        )?;

        // THE GENES ARE PROJECTED BY NAME, NEVER BY POSITION.
        //
        // A gene's `indices` are positions into the feature list its search
        // actually used — after discovery's prefilter and after the streaming
        // loop's canonical remap. They are NOT positions into whatever
        // `prepare_multitimeframe_features` builds here, and the two lists agree
        // only by accident. Reading column 47 of the wrong list evaluates a
        // different strategy and calls the answer out of sample, which is the
        // one error this window cannot survive — so the projection is by name,
        // through the same `project_features_to_effective` the live path uses,
        // and a name that is absent is a REFUSAL.
        let features = neoethos_search::live_portfolio::project_features_to_effective(
            &raw,
            &portfolio.feature_names,
        )
        .with_context(|| {
            format!(
                "projecting the out-of-sample feature block onto the {} feature names {sweep} \
                 slot {slot}'s genes address (streamed={}). A missing name means the search \
                 selected on a column this build does not produce — for a STREAMED run that is \
                 the working-set extension the sweep installed, which the plain whole-vocabulary \
                 build does not contain. REQUIRED ELSEWHERE: a public neoethos-search entry point \
                 that rebuilds a recorded working set by name so the out-of-sample touch can \
                 reproduce the exact column list the search used. Until it exists this REFUSES \
                 rather than reading a different column and reporting the number as out-of-sample \
                 evidence.",
                portfolio.feature_names.len(),
                portfolio.streamed
            )
        })?;

        let mut per_trade_net: Vec<f64> = Vec::new();
        let mut r_multiples: Vec<f64> = Vec::new();
        let mut by_month: HashMap<i64, f64> = HashMap::new();

        for gene in &portfolio.genes {
            let signals = neoethos_search::genetic::signals_for_gene(&features, gene);
            let settings = oos_backtest_settings(&config, gene)?;
            let trades = neoethos_search::simulate_trades_broker_real(
                &base.close,
                &base.high,
                &base.low,
                timestamps,
                &signals,
                &settings,
            )?;
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

/// Can the ledger under `cache_dir` be attributed to the search that is asking?
///
/// The manifest is keyed only on `{SYMBOL}_{TF}.trial_returns.json` — a path,
/// not an identity — and `TrialReturnsManifest::config_hash` is the identity
/// that goes with it. Pure, so the decision is testable without a card, a
/// dataset or a feature cube.
///
/// `stamp` is `None` when no manifest is present at all, and `Some(None)` when
/// one is present but carries no `config_hash`.
///
/// **`Some(None)` is not a match.** It means the writer had no stamp, which is
/// "cannot be attributed"; an unattributable matrix is exactly as unusable as
/// a foreign one, and treating it as a match is how the previous session's
/// numbers would enter this one.
fn attribute_manifest(
    stamp: Option<Option<&str>>,
    expected_config_hash: &str,
    cache_dir: &str,
) -> Result<(), String> {
    let Some(stamp) = stamp else {
        return Err(format!(
            "no trial-returns manifest under {cache_dir}, so nothing read back from that \
             directory can be attributed to this search. DSR and PBO are NOT computable and are \
             refused rather than computed from whatever the directory happens to hold."
        ));
    };
    match stamp {
        Some(hash) if hash == expected_config_hash => Ok(()),
        Some(other) => Err(format!(
            "the trial-returns matrix under {cache_dir} was written by configuration {other}, not \
             by this search's {expected_config_hash}. The scratch root is keyed by SYMBOL and \
             sweep ids restart at 1 every session, so this is a PREVIOUS session's matrix — under \
             a different cost model and a different goal set. Refused: a statistic deflated \
             against another run's trials is not this run's statistic."
        )),
        None => Err(format!(
            "the trial-returns manifest under {cache_dir} carries NO config_hash, so the matrix \
             beside it cannot be attributed to this search. `None` means the writer had no stamp; \
             it does not mean 'matches'."
        )),
    }
}

/// The champion's per-month return series — this search's candidate row for the
/// session champion matrix (`pbo_session`).
///
/// Read from the trial-returns matrix the run just wrote, which is where the
/// per-trial series already lives; nothing is recomputed, so the row and the
/// statistics beside it describe the same trials.
///
/// Only ever called on a ledger [`attribute_manifest`] has attributed to this
/// search: it reads the same directory, so it inherits the same identity check.
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

/// Persist what this search selected, so the one out-of-sample touch has
/// something to evaluate. Returns how many genes were written.
///
/// **The genes are the CANONICAL ones and the names are the run-level list.**
/// `StreamingBatchSurvivor::result.portfolio`'s indices address that batch's own
/// `effective_feature_names`, so index 47 means a different column in every
/// batch; `canonical_portfolio`'s indices address the run-level
/// `StreamingRunOutcome::canonical`, one list for the whole run. Writing the
/// canonical pair is what lets the evaluation project by name later — writing
/// the per-batch pair would need one name list per gene and would silently
/// collapse to the last batch's.
///
/// A run that selected nothing writes NOTHING and says so with `0`. That is the
/// honest observable: the promotion path refuses on a missing artifact, and an
/// artifact recording an empty portfolio would be a promise of evidence that
/// contains none.
fn persist_promotion_evidence<R>(
    request: &SearchRequest<'_>,
    outcome: &neoethos_search::orchestration::StreamingRunOutcome<R>,
) -> Result<usize> {
    let genes: Vec<Gene> = outcome
        .batches
        .iter()
        .flat_map(|batch| batch.canonical_portfolio.iter().cloned())
        .collect();
    if genes.is_empty() {
        return Ok(0);
    }
    let batches = outcome
        .batches
        .iter()
        .filter(|batch| !batch.canonical_portfolio.is_empty())
        .count();
    let written = genes.len();
    let portfolio = super::PromotionPortfolio {
        schema: super::PROMOTION_EVIDENCE_SCHEMA.to_string(),
        sweep: request.sweep,
        slot: request.slot,
        // THE STAMP. It is the slot's `config_hash` exactly as the runner handed
        // it in — never recomputed here — so the loader's check compares the
        // same number the journal recorded rather than two independent
        // derivations that could drift apart.
        config_hash: request.config_hash.to_string(),
        feature_names: outcome.canonical.names().to_vec(),
        genes,
        streamed: outcome.streamed,
        batches,
    };
    write_json_atomically(&request.promotion_evidence_path, &portfolio).with_context(|| {
        format!(
            "writing {} slot {}'s promotion evidence ({written} genes over {} feature names). \
             Without it the single out-of-sample touch has nothing to evaluate, and this path \
             will NOT re-search to find something later.",
            request.sweep,
            request.slot,
            portfolio.feature_names.len()
        )
    })?;
    Ok(written)
}

/// Write JSON through a temporary file and a rename, fsyncing the bytes before
/// the rename.
///
/// A half-written promotion artifact is worse than a missing one: the missing
/// one is refused by name, the half-written one is a parse error at the moment
/// the window is about to be spent.
fn write_json_atomically<T: serde::Serialize>(
    path: &std::path::Path,
    value: &T,
) -> Result<()> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec(value)
        .with_context(|| format!("serialising {}", path.display()))?;
    {
        let mut file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.flush()?;
        file.sync_all()
            .with_context(|| format!("fsyncing {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Apply a control sweep's permutation to a feature block, in place.
///
/// **The features move; nothing else does.** `timestamps` are copied across
/// untouched and prices, labels, costs, exit geometry and the GA seed are not
/// reachable from here at all — that asymmetry *is* the experiment. Every
/// feature keeps its marginal distribution and (under `CircularRotation`) its
/// autocorrelation; the only thing destroyed is its alignment with the future.
///
/// The frame is `[samples × features]` and dense in ROW-major order, hence
/// [`crate::shuffle::BlockLayout::RowMajor`]. A layout mistake here would rotate
/// across features instead of across time and quietly produce a control that is
/// not a control, so the shape is asserted rather than assumed: if the rebuilt
/// array does not take the original `(rows, cols)`, this returns an error and
/// the sweep FAILS. It never falls back to the unpermuted frame — a control
/// that silently ran live data is the one result that would poison the null.
fn apply_shuffle_control(
    permutation: &crate::shuffle::FeaturePermutation,
    frame: &mut neoethos_data::FeatureFrame,
) -> Result<()> {
    let rows = frame.n_samples();
    let cols = frame.n_features();
    if rows == 0 || cols == 0 {
        bail!(
            "the shuffle control was handed a {rows}x{cols} feature block. There is nothing to \
             permute, so the control sweep would be a second live run wearing the control's label."
        );
    }

    let dense = frame.to_dense_samples_major();
    let src = dense.as_slice().ok_or_else(|| {
        anyhow::anyhow!(
            "the dense [samples x features] block is not in standard row-major order, so the \
             rotation would move data across features instead of across time"
        )
    })?;

    let rotated = permutation
        .apply(src, rows, cols, crate::shuffle::BlockLayout::RowMajor)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let array = ndarray::Array2::from_shape_vec((rows, cols), rotated)
        .context("rebuilding the permuted feature block at its original shape")?;

    *frame = neoethos_data::FeatureFrame::from_array(
        frame.timestamps.clone(),
        frame.names.clone(),
        array,
    );
    Ok(())
}

/// Backtest settings for an already-selected gene on the OOS window.
///
/// Only reached when adaptive stops are OFF — see the refusal in
/// [`StreamingSweepExecutor::oos_preflight`], which fires BEFORE the window is
/// journalled as spent and is re-asserted inside
/// [`StreamingSweepExecutor::evaluate_oos`] — so the gene's own `sl_pips` /
/// `tp_pips` are the effective stops, exactly as they were when it was scored.
///
/// **This mirrors `discovery::discovery_backtest_settings` field for field.**
/// That function is private to `neoethos-search`, so the mirror is by hand, and
/// a hand-written mirror is exactly where a policy drifts apart one field at a
/// time. It matters more here than anywhere else in the crate: the one
/// out-of-sample touch is spent through this function, so any field that is
/// cheaper here than it was in-sample turns "it survived out of sample" into
/// "it survived a cheaper simulation out of sample" — and the touch cannot be
/// taken back.
///
/// `max_hold_bars` comes from the EVALUATION CONFIG, not from the gene: `Gene`
/// carries no `max_hold_bars` field at all (`genetic/strategy_gene.rs`) — the
/// hold limit is a policy of the run, identical for every candidate in it.
///
/// REQUIRED ELSEWHERE (see `docs/autoresearch-loop.md` §14): make
/// `discovery_backtest_settings` `pub` in `crates/neoethos-search/src/discovery.rs`
/// so the OOS touch calls THE settings builder instead of a copy of it.
fn oos_backtest_settings(
    config: &DiscoveryConfig,
    gene: &Gene,
) -> Result<neoethos_search::eval::BacktestSettings> {
    let evaluation = config.try_evaluation_config(None)?;
    Ok(neoethos_search::eval::BacktestSettings {
        sl_pips: if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            20.0
        },
        tp_pips: if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            40.0
        },
        max_hold_bars: evaluation.max_hold_bars,
        // All four exit-geometry fields together, as in-sample.
        trailing_enabled: evaluation.trailing_enabled,
        trailing_atr_multiplier: evaluation.trailing_atr_multiplier,
        trailing_be_trigger_r: evaluation.trailing_be_trigger_r,
        trailing_min_lock_pips: evaluation.trailing_min_lock_pips,
        pip_value: evaluation.pip_value,
        pip_value_per_lot: evaluation.pip_value_per_lot,
        // Every cost the screen charged, charged again out of sample.
        spread_pips: evaluation.spread_pips,
        commission_per_trade: evaluation.commission_per_trade,
        session_spread_profile: config.session_spread_pips.map(|curve| {
            neoethos_search::eval::SessionSpreadProfile {
                asian_pips: curve[0],
                overlap_pips: curve[1],
                late_ny_pips: curve[2],
            }
        }),
        swap_long_pips_per_day: config.swap_long_pips_per_day,
        swap_short_pips_per_day: config.swap_short_pips_per_day,
        // #75/#217 (2026-08-10): the in-sample builder stopped hardcoding this
        // and reads `config.kill_zones_enabled` (which is
        // `risk.kill_zones_enabled`, the same field live reads). The mirror
        // follows it — a `true` here against a `false` in-sample would make the
        // one out-of-sample touch a DIFFERENT simulation from the one that
        // scored the gene.
        kill_zones_enabled: config.kill_zones_enabled,
        risk_per_trade_min: config.risk_per_trade_min,
        risk_per_trade_max: config.risk_per_trade_max,
        ..neoethos_search::eval::BacktestSettings::default()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// THE PROMOTION EVIDENCE, TESTED FROM BOTH ENDS
//
// These tests exist because the defect they cover was not a wrong number: the
// read site existed, the comment above it described what it read, and NOTHING
// IN THE WORKSPACE EVER WROTE THE FILE. A unit test of the reader alone would
// have passed — it would have asserted that a missing file errors. So the
// writer and the reader are exercised against each other, and every refusal is
// asserted by the words it uses: a refusal nobody can recognise is a crash with
// better manners.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{PROMOTION_EVIDENCE_SCHEMA, PromotionPortfolio, load_promotion_portfolio};

    fn gene(indices: &[usize]) -> Gene {
        Gene {
            indices: indices.to_vec(),
            weights: vec![1.0; indices.len()],
            strategy_id: "s-1".to_string(),
            ..Gene::default()
        }
    }

    // ── C1: the ledger is attributed, not merely located ────────────────────

    #[test]
    fn a_matrix_written_by_this_search_is_attributed_to_it() {
        attribute_manifest(Some(Some("fnv64:mine")), "fnv64:mine", "/scratch")
            .expect("the hashes agree");
    }

    #[test]
    fn a_previous_sessions_matrix_is_refused_and_names_the_configuration() {
        // The scratch root is keyed by SYMBOL and sweep ids restart at 1 every
        // session, so `sweep 1/slot_007` is a directory two sessions share.
        // Matrix writes are non-fatal by design, so B can find A's matrix there.
        let err = attribute_manifest(Some(Some("fnv64:theirs")), "fnv64:mine", "/scratch")
            .expect_err("a foreign matrix must not be adopted");
        assert!(err.contains("fnv64:theirs"), "{err}");
        assert!(err.contains("fnv64:mine"), "{err}");
        assert!(err.contains("PREVIOUS session"), "{err}");
    }

    #[test]
    fn an_unstamped_manifest_cannot_be_attributed_and_is_never_treated_as_a_match() {
        let err = attribute_manifest(Some(None), "fnv64:mine", "/scratch")
            .expect_err("None means 'cannot be attributed', not 'matches'");
        assert!(err.contains("NO config_hash"), "{err}");
    }

    #[test]
    fn an_absent_manifest_refuses_rather_than_reading_whatever_is_there() {
        let err = attribute_manifest(None, "fnv64:mine", "/scratch")
            .expect_err("no manifest is no attribution");
        assert!(err.contains("NOT computable"), "{err}");
    }

    fn portfolio(config_hash: &str) -> PromotionPortfolio {
        PromotionPortfolio {
            schema: PROMOTION_EVIDENCE_SCHEMA.to_string(),
            sweep: SweepId(3),
            slot: 7,
            config_hash: config_hash.to_string(),
            feature_names: vec!["rsi_14".into(), "atr_14".into(), "ema_50".into()],
            genes: vec![gene(&[0, 2])],
            streamed: true,
            batches: 2,
        }
    }

    fn write(dir: &std::path::Path, value: &PromotionPortfolio) -> PathBuf {
        let path = dir.join("promotion").join("slot_007.json");
        write_json_atomically(&path, value).expect("the artifact must be writable");
        path
    }

    #[test]
    fn the_evidence_the_executor_writes_is_the_evidence_the_runner_loads() {
        // The whole defect in one test: a writer and a reader that agree.
        let dir = tempfile::tempdir().unwrap();
        let written = portfolio("fnv64:aaaa");
        let path = write(dir.path(), &written);

        let loaded = load_promotion_portfolio(&path, SweepId(3), 7, "fnv64:aaaa")
            .expect("the artifact it just wrote must load");
        assert_eq!(loaded, written);
        assert_eq!(loaded.genes.len(), 1);
        assert_eq!(loaded.feature_names.len(), 3);
    }

    #[test]
    fn a_missing_artifact_is_a_named_refusal_that_leaves_the_window_unspent() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_promotion_portfolio(
            &dir.path().join("promotion").join("slot_007.json"),
            SweepId(3),
            7,
            "fnv64:aaaa",
        )
        .expect_err("a missing artifact must refuse");
        let text = format!("{err:#}");
        assert!(text.contains("UNSPENT"), "{text}");
        assert!(text.contains("SESSION STORE"), "{text}");
    }

    #[test]
    fn the_stamp_is_what_makes_a_file_at_the_right_path_into_evidence() {
        // A file at the right path is not proof that it describes the right
        // configuration: a resumed session, a re-run sweep or a copied directory
        // all put bytes there. The stamp is checked, and it is checked BEFORE
        // the window is spent.
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), &portfolio("fnv64:aaaa"));
        let err = load_promotion_portfolio(&path, SweepId(3), 7, "fnv64:bbbb")
            .expect_err("a foreign stamp must refuse");
        let text = format!("{err:#}");
        assert!(text.contains("fnv64:bbbb"), "{text}");
        assert!(text.contains("NOT spent"), "{text}");
    }

    #[test]
    fn one_slots_evidence_is_never_read_as_anothers() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), &portfolio("fnv64:aaaa"));
        let err = load_promotion_portfolio(&path, SweepId(3), 8, "fnv64:aaaa")
            .expect_err("a slot mismatch must refuse");
        assert!(format!("{err:#}").contains("wearing one path"));
    }

    #[test]
    fn an_empty_portfolio_is_not_a_promotion_candidate() {
        let empty = PromotionPortfolio {
            genes: Vec::new(),
            ..portfolio("fnv64:aaaa")
        };
        let err = empty
            .assert_names(SweepId(3), 7, "fnv64:aaaa", std::path::Path::new("x.json"))
            .expect_err("nothing to evaluate must refuse");
        assert!(format!("{err:#}").contains("nothing to evaluate"));
    }

    #[test]
    fn genes_without_the_names_their_indices_address_are_refused() {
        // An index without a name is a number that reads whatever happens to be
        // in that position — which, on the one out-of-sample window, is a
        // different strategy reported under this one's name.
        let nameless = PromotionPortfolio {
            feature_names: Vec::new(),
            ..portfolio("fnv64:aaaa")
        };
        let err = nameless
            .assert_names(SweepId(3), 7, "fnv64:aaaa", std::path::Path::new("x.json"))
            .expect_err("no names must refuse");
        assert!(format!("{err:#}").contains("no feature names"));

        let out_of_range = PromotionPortfolio {
            genes: vec![gene(&[0, 9])],
            ..portfolio("fnv64:aaaa")
        };
        let err = out_of_range
            .assert_names(SweepId(3), 7, "fnv64:aaaa", std::path::Path::new("x.json"))
            .expect_err("an index past the names must refuse");
        assert!(format!("{err:#}").contains("internally inconsistent"));
    }

    #[test]
    fn a_schema_this_build_does_not_know_is_refused_rather_than_guessed() {
        let future = PromotionPortfolio {
            schema: "neoethos.autoresearch.promotion_evidence.v99".to_string(),
            ..portfolio("fnv64:aaaa")
        };
        let err = future
            .assert_names(SweepId(3), 7, "fnv64:aaaa", std::path::Path::new("x.json"))
            .expect_err("an unknown schema must refuse");
        assert!(format!("{err:#}").contains("v99"));
    }

    #[test]
    fn nothing_is_written_for_a_search_that_selected_nothing() {
        // "No artifact" and "an artifact recording an empty portfolio" must be
        // the same observable, so the promotion path has exactly one thing to
        // refuse instead of two.
        let dir = tempfile::tempdir().unwrap();
        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let path = dir.path().join("promotion").join("slot_007.json");
        let request = SearchRequest {
            sweep: SweepId(3),
            slot: 7,
            config: &config,
            config_hash: "fnv64:aaaa",
            trial_returns_path: dir.path().join("unused.bin"),
            promotion_evidence_path: path.clone(),
            permutation: None,
        };
        let outcome: neoethos_search::orchestration::StreamingRunOutcome<()> =
            neoethos_search::orchestration::StreamingRunOutcome {
                batches: Vec::new(),
                canonical:
                    neoethos_search::orchestration::batch_ledger::CanonicalFeatureIndex::new(),
                ledger: neoethos_search::orchestration::batch_ledger::StreamingRunLedger::new(),
                streamed: false,
                next_cursor: 0,
                space_len: 0,
                batch_columns: 0,
            };
        assert_eq!(persist_promotion_evidence(&request, &outcome).unwrap(), 0);
        assert!(
            !path.exists(),
            "a search that selected nothing must write nothing"
        );
    }
}
