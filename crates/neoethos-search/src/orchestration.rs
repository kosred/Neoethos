// The batch ledger + the canonical feature index (Option C). Declared here
// rather than in `lib.rs` so this change stays inside the files it owns while a
// parallel wave is building; `lib.rs` may hoist it to a top-level `pub mod
// batch_ledger;` later without touching a line of its contents.
#[path = "batch_ledger.rs"]
pub mod batch_ledger;

use crate::discovery::{
    DiscoveryConfig, DiscoveryResult, discovery_per_kind_evidence_hashes,
    ensure_non_empty_portfolio, run_discovery_cycle_with_holdout,
    save_canonical_backtest_artifacts, save_discovery_profile_json, save_portfolio_json,
    save_quality_report_json, save_trade_log_json, save_walkforward_validation_artifacts,
};
use crate::genetic::Gene;
use crate::validation::PropFirmRiskRules;
use anyhow::{Result, bail};
use neoethos_data::{
    FeatureFrame, MANDATORY_TFS, ensure_timeframes_with_resample, load_symbol_dataset,
    prepare_multitimeframe_features,
};
use std::path::Path;
use std::sync::Arc;
use tracing::info;

pub use batch_ledger::{
    BatchLedgerEntry, BatchOutcome, CanonicalFeatureIndex, CanonicalSurvivor,
    STREAMING_RUN_PORTFOLIO_SCHEMA_VERSION, StreamingRunLedger, StreamingRunPortfolio,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchDiscoverySummary {
    pub symbols_seen: usize,
    pub work_units_seen: usize,
    pub portfolios_saved: usize,
    pub skipped_symbols: usize,
    pub skipped_timeframes: usize,
    pub feature_failures: usize,
    pub empty_portfolios: usize,
    pub discovery_failures: usize,
    /// Portfolios whose persisted profile reports
    /// `validation_evidence_complete = false` because at least one of
    /// the four producer-side validation kinds did not ship for that
    /// run. The live-execution simulation hash is structurally absent
    /// today, so a portfolio is counted here only when a producer-side
    /// kind is missing — not because the simulator has not been wired.
    pub portfolios_with_missing_producer_evidence: usize,
}

impl BatchDiscoverySummary {
    fn finalize(self) -> Result<Self> {
        if self.portfolios_saved == 0 {
            anyhow::bail!(
                "Batch discovery produced no usable portfolios \
                 (symbols={}, work_units={}, skipped_symbols={}, skipped_timeframes={}, \
                 feature_failures={}, empty_portfolios={}, discovery_failures={}). \
                 Check cache/discovery/*.json funnel files for per-pair rejection reasons, \
                 or run a single pair to see the detailed funnel.",
                self.symbols_seen,
                self.work_units_seen,
                self.skipped_symbols,
                self.skipped_timeframes,
                self.feature_failures,
                self.empty_portfolios,
                self.discovery_failures
            );
        }
        Ok(self)
    }
}

pub struct DiscoveryOrchestrator {
    pub data_root: String,
    pub output_dir: String,
    pub config: DiscoveryConfig,
}

impl DiscoveryOrchestrator {
    pub fn new(data_root: &str, output_dir: &str, config: DiscoveryConfig) -> Self {
        Self {
            data_root: data_root.to_string(),
            output_dir: output_dir.to_string(),
            config,
        }
    }

    pub fn run_batch(
        &self,
        symbols: &[String],
        timeframes: &[String],
    ) -> Result<BatchDiscoverySummary> {
        std::fs::create_dir_all(&self.output_dir)?;
        let mut summary = BatchDiscoverySummary::default();

        for symbol in symbols {
            summary.symbols_seen += 1;
            info!("Processing symbol: {}", symbol);
            let ds = match load_symbol_dataset(&self.data_root, symbol) {
                Ok(d) => d,
                Err(e) => {
                    summary.skipped_symbols += 1;
                    info!("Skipping symbol {}: {}", symbol, e);
                    continue;
                }
            };

            for tf in timeframes {
                summary.work_units_seen += 1;
                info!("  Timeframe: {}", tf);
                let ds_ready = match ensure_timeframes_with_resample(&ds, tf, MANDATORY_TFS) {
                    Ok(d) => d,
                    Err(e) => {
                        summary.skipped_timeframes += 1;
                        info!("    Skipping tf {}: {}", tf, e);
                        continue;
                    }
                };

                let htfs: Vec<&str> = self
                    .config
                    .higher_timeframes
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let features = match prepare_multitimeframe_features(&ds_ready, tf, &htfs) {
                    Ok(f) => f,
                    Err(e) => {
                        summary.feature_failures += 1;
                        info!("    Feature prep failed: {}", e);
                        continue;
                    }
                };

                let base_ohlcv = match ds_ready.frames.get(tf) {
                    Some(o) => o,
                    None => {
                        summary.feature_failures += 1;
                        info!("    Skipping tf {}: base ohlcv missing", tf);
                        continue;
                    }
                };
                let mut runtime_config = self.config.clone().apply_mode_overrides();
                runtime_config.timeframe_label = tf.clone();
                // Bind the current symbol so the cost-model guard doesn't fire.
                // The base config carries settings.system.symbol as a default,
                // which is wrong for every symbol except the one in config.yaml.
                runtime_config.evaluation_symbol = symbol.clone();
                // Previously this used `?` and aborted the whole batch on a
                // single discovery failure, while every other error in the
                // loop counted toward `summary.skipped_*` and continued.
                // Audit B02/B03 (2026-07-13): batch discovery used to see the
                // FULL series; the holdout wrapper withholds the OOS tail.
                let result = match run_discovery_cycle_with_holdout(
                    &features,
                    base_ohlcv,
                    &runtime_config,
                    PropFirmRiskRules::default(),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        summary.discovery_failures += 1;
                        info!("    Discovery failed for {} {}: {}", symbol, tf, e);
                        continue;
                    }
                };
                if let Err(err) = ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, tf))
                {
                    summary.empty_portfolios += 1;
                    info!("    {}", err);
                    continue;
                }

                info!("    Found {} strategies", result.portfolio.len());

                let out_path = Path::new(&self.output_dir).join(format!("{}_{}.json", symbol, tf));
                save_portfolio_json(&out_path, &result)?;
                let profile_path =
                    Path::new(&self.output_dir).join(format!("{}_{}_profile.json", symbol, tf));
                save_discovery_profile_json(profile_path, &runtime_config, &result)?;
                if !result.quality_metrics.is_empty() {
                    let quality_path =
                        Path::new(&self.output_dir).join(format!("{}_{}_quality.json", symbol, tf));
                    save_quality_report_json(quality_path, &result)?;
                }
                if !result.logged_trades.is_empty() {
                    let trade_log_path = Path::new(&self.output_dir)
                        .join(format!("{}_{}_trade_logs.json", symbol, tf));
                    save_trade_log_json(trade_log_path, &result)?;
                }
                if !result.canonical_backtest_artifacts.is_empty() {
                    let backtest_dir = Path::new(&self.output_dir)
                        .join(format!("{}_{}_canonical_backtests", symbol, tf));
                    save_canonical_backtest_artifacts(&backtest_dir, &result)?;
                }
                if !result.walkforward_validation_artifacts.is_empty() {
                    let validation_dir = Path::new(&self.output_dir)
                        .join(format!("{}_{}_walkforward_validations", symbol, tf));
                    save_walkforward_validation_artifacts(&validation_dir, &result)?;
                }
                if let Ok(hashes) = discovery_per_kind_evidence_hashes(&result)
                    && !hashes.all_producer_kinds_present()
                {
                    summary.portfolios_with_missing_producer_evidence += 1;
                }
                summary.portfolios_saved += 1;
            }
        }
        summary.finalize()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// THE STREAMING WORKING-SET LOOP
//
// WHERE IT LIVES, AND WHY HERE.
//
// `run_discovery_cycle_with_progress` takes `features: &FeatureFrame` as a
// PARAMETER and never rebuilds it. The working set is therefore chosen entirely
// OUTSIDE the cycle — in whoever calls `prepare_multitimeframe_features` and
// then the cycle. That pair is this file (`DiscoveryOrchestrator::run_batch`)
// and `neoethos-cli/src/main.rs::cmd_discover`. So the loop goes AROUND the
// pair; neither `discovery.rs` nor `genetic/search_engine.rs` changes for it to
// run, and the parity case is trivially expressible: one batch covering the
// whole space, predicate silent, one build, one cycle — today's path.
//
// WHAT IT DOES NOT DO. It does not apply the early-reject predicate. That fires
// INSIDE the cycle, at `discovery.rs` where `ranked_candidates` is formed —
// before `Gene::requires_quality_screen`, before walk-forward, before CPCV,
// which is the only place early enough to skip the 50.4 % of wall time the
// measured run spent rejecting 174 of 174 candidates. This loop OBSERVES that
// verdict through `discovery::batch_rejection_ledger()` and records it in its
// own census.
//
// CROSS-BATCH SURVIVORS: OPTION C (docs/streaming-parameter-search.md, settled
// 2026-08-10). Survivors from different batches DO share a run. Correlation
// pruning already works across them (it consumes per-bar `Vec<i8>` direction
// vectors, not the cube, and batches differ in columns, not in the time axis),
// and the live path projects BY NAME. The one real defect is that a gene's
// `indices` are positions into ITS batch's `effective_feature_names`, so index
// 47 means different columns in different batches. This loop remaps every
// survivor at BATCH EXIT into `CanonicalFeatureIndex` — append-only, hard-error
// on an unresolvable name, range-asserted at run end.
// ═════════════════════════════════════════════════════════════════════════════

/// What a batch's search produced, in the only two terms the loop needs. A
/// trait so the loop can be exercised without constructing a full
/// [`DiscoveryResult`], and so the remap logic is testable in isolation from
/// the pipeline it protects.
pub trait BatchSearchResult {
    /// The batch's OWN feature-name list — what its genes' indices address.
    fn local_feature_names(&self) -> &[String];
    /// The genes the batch selected.
    fn portfolio_genes(&self) -> &[Gene];
}

impl BatchSearchResult for DiscoveryResult {
    fn local_feature_names(&self) -> &[String] {
        &self.effective_feature_names
    }

    fn portfolio_genes(&self) -> &[Gene] {
        &self.portfolio
    }
}

/// Whether to stream, and for how long.
///
/// The batch WIDTH is deliberately absent: it comes from
/// `hpc_ta::streaming_batch_columns`, i.e. from free RAM and the widest frame,
/// never from a config constant or a CLI flag (design doc §6, "Non-negotiable
/// when it lands"). What an operator gets to choose is whether to sweep and how
/// many batches to spend, not how much memory to risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingPlan {
    pub enabled: bool,
    /// `0` = until the parameter space is exhausted.
    pub max_batches: usize,
}

impl Default for StreamingPlan {
    fn default() -> Self {
        Self::disabled()
    }
}

impl StreamingPlan {
    /// Today's behaviour: one whole-vocabulary pass.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            max_batches: 0,
        }
    }

    pub const fn streaming(max_batches: usize) -> Self {
        Self {
            enabled: true,
            max_batches,
        }
    }
}

/// One batch that produced survivors.
pub struct StreamingBatchSurvivor<R> {
    /// The batch's cursor into the (indicator, period) space — its provenance.
    pub cursor: usize,
    pub pairs: usize,
    /// The batch's own result, UNTOUCHED. Its `effective_feature_names` and its
    /// genes' local indices remain mutually consistent, so it can still be
    /// saved as today's per-run artifact.
    pub result: R,
    /// The same genes with CANONICAL indices — positions into
    /// [`StreamingRunOutcome::canonical`]. Never mixed with `result.portfolio`.
    pub canonical_portfolio: Vec<Gene>,
}

/// Everything a streaming run produced, including everything it threw away.
pub struct StreamingRunOutcome<R> {
    /// When `streamed` is true: the batches that produced survivors.
    ///
    /// When `streamed` is FALSE (streaming was not requested, or the machine
    /// could not size a batch): exactly one entry — the single whole-vocabulary
    /// pass — WHETHER OR NOT it selected anything, so the caller's own
    /// empty-portfolio guard is the one that fires. That is what keeps the
    /// non-streaming path byte-for-byte today's path.
    pub batches: Vec<StreamingBatchSurvivor<R>>,
    pub canonical: CanonicalFeatureIndex,
    pub ledger: StreamingRunLedger,
    /// True only if at least one real working set was installed.
    pub streamed: bool,
    /// Cursor the sweep would resume from — the whole resumable state.
    pub next_cursor: usize,
    pub space_len: usize,
    pub batch_columns: usize,
}

// Debug is implemented BY HAND, not derived, so that it does NOT require
// `R: Debug`. A derived impl would make the diagnosability of a streaming run
// depend on whether the caller's search-result type happens to be printable —
// exactly the wrong dependency for a value that only ever gets printed when
// something has already gone wrong (`expect`, `expect_err`, a bubbled `Result`).
// The per-batch result is shown as an opaque placeholder; the provenance that
// actually explains a failure — cursor, pair count, survivor count, ledger — is
// printed in full.
impl<R> std::fmt::Debug for StreamingBatchSurvivor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingBatchSurvivor")
            .field("cursor", &self.cursor)
            .field("pairs", &self.pairs)
            .field("result", &format_args!("<{}>", std::any::type_name::<R>()))
            .field("canonical_portfolio", &self.canonical_portfolio)
            .finish()
    }
}

impl<R> std::fmt::Debug for StreamingRunOutcome<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingRunOutcome")
            .field("batches", &self.batches)
            .field("canonical", &self.canonical)
            .field("ledger", &self.ledger)
            .field("streamed", &self.streamed)
            .field("next_cursor", &self.next_cursor)
            .field("space_len", &self.space_len)
            .field("batch_columns", &self.batch_columns)
            .finish()
    }
}

impl<R> StreamingRunOutcome<R> {
    /// Every surviving gene with CANONICAL indices, tagged with the batch that
    /// produced it.
    pub fn survivors(&self) -> Vec<CanonicalSurvivor> {
        self.batches
            .iter()
            .flat_map(|batch| {
                batch
                    .canonical_portfolio
                    .iter()
                    .map(move |gene| CanonicalSurvivor {
                        source_cursor: batch.cursor,
                        gene: gene.clone(),
                    })
            })
            .collect()
    }

    pub fn survivor_count(&self) -> usize {
        self.batches
            .iter()
            .map(|b| b.canonical_portfolio.len())
            .sum()
    }
}

/// Does `name` address the `(indicator, period)` stem `stem`?
///
/// Segment-aware on purpose: `rsi_4` must not be seen inside `rsi_40`, and the
/// stem must be matchable anywhere in the name because higher-timeframe columns
/// carry a timeframe prefix and multi-output indicators carry an output suffix.
fn column_addresses_stem(name_segments: &[&str], stem_segments: &[&str]) -> bool {
    if stem_segments.is_empty() || name_segments.len() < stem_segments.len() {
        return false;
    }
    name_segments
        .windows(stem_segments.len())
        .any(|window| window == stem_segments)
}

/// THE ANTI-NO-OP GUARD.
///
/// The loop is worthless — worse, actively misleading — if the feature build
/// ignores the working set: every batch would compute the same columns, the
/// census would report a sweep that never happened, and the cursor would march
/// through a parameter space the run never touched. That failure is SILENT by
/// construction, so it is checked rather than assumed.
///
/// Conservative on purpose: it fires only when NOT ONE of the batch's pairs
/// appears anywhere in the built frame. A batch of hundreds of pairs producing
/// zero matching columns cannot happen if the working set was applied — while
/// individual pairs legitimately vanish (an id whose kernel failed, a period
/// the `#212` pre-flight emitted as all-NaN under a different name).
fn assert_working_set_observed(
    batch: &neoethos_data::core::hpc_ta::SweepBatch,
    features: &FeatureFrame,
) -> Result<usize> {
    if batch.pairs.is_empty() {
        return Ok(0);
    }
    let name_segments: Vec<Vec<&str>> = features
        .names
        .iter()
        .map(|n| n.split('_').collect::<Vec<&str>>())
        .collect();
    let mut matched = 0usize;
    for pair in &batch.pairs {
        let stem = format!("{}_{}", pair.id, pair.period);
        let stem_segments: Vec<&str> = stem.split('_').collect();
        if name_segments
            .iter()
            .any(|segments| column_addresses_stem(segments, &stem_segments))
        {
            matched += 1;
        }
    }
    if matched == 0 {
        let sample: Vec<String> = batch
            .pairs
            .iter()
            .take(6)
            .map(|p| format!("{}_{}", p.id, p.period))
            .collect();
        bail!(
            "streaming batch at cursor {} planned {} (indicator, period) pairs ({} columns) but \
             the feature build emitted NOT ONE column addressing ANY of them (frame has {} \
             columns; wanted e.g. {:?}). The working set was not observed by the build — either \
             `neoethos_data::with_extended_sweep_working_set` / \
             `prepare_multitimeframe_features_batch` was bypassed, or the sweep still takes the \
             budget-capped prefix and ignores the installed batch. Refusing to continue: every \
             batch would compute the same columns and the run would report a sweep it never \
             performed.",
            batch.cursor,
            batch.pairs.len(),
            batch.planned_columns,
            features.names.len(),
            sample
        );
    }
    Ok(matched)
}

/// THE LOOP. Advance the working set through the (indicator, period) space, run
/// one discovery cycle per batch, carry survivors forward under one canonical
/// index.
///
/// * `budget_rows` — the BASE timeframe's bar count. The batch width is sized
///   against the run's widest frame so it is not a function of which timeframe
///   is being built (per-TF widths must stay equal or the cube cannot be
///   assembled).
/// * `build_features` — must route through
///   `neoethos_data::prepare_multitimeframe_features_batch` (or
///   `with_extended_sweep_working_set`) so the batch it is handed is actually
///   installed. Handed `None` for the non-streaming pass, which that function
///   documents as byte-identical to `prepare_multitimeframe_features`.
/// * `run_cycle` — one discovery cycle on the built frame.
///
/// Failure policy, stated rather than implied:
///
/// * a per-batch build or cycle ERROR is recorded, named, and the sweep
///   continues — one bad region must not throw away the regions already
///   searched. If NO batch reaches the search, `StreamingRunLedger::finish`
///   turns that into a loud infrastructure failure rather than "found nothing".
/// * a remap failure is FATAL and propagates: it means a survivor's term cannot
///   be resolved to a name, and the alternatives are dropping the term (a
///   strategy nobody designed) or dropping the gene silently.
/// * the non-streaming pass propagates build/cycle errors exactly as today.
pub fn run_streaming_working_set<R, B, C>(
    plan: &StreamingPlan,
    budget_rows: usize,
    mut build_features: B,
    mut run_cycle: C,
) -> Result<StreamingRunOutcome<R>>
where
    R: BatchSearchResult,
    B: FnMut(Option<Arc<neoethos_data::core::hpc_ta::SweepBatch>>) -> Result<FeatureFrame>,
    C: FnMut(&FeatureFrame) -> Result<R>,
{
    let mut canonical = CanonicalFeatureIndex::new();
    let mut ledger = StreamingRunLedger::new();
    let mut batches: Vec<StreamingBatchSurvivor<R>> = Vec::new();

    // The hardware probe + space enumeration happen ONLY when streaming was
    // asked for, so today's path does not acquire a new dependency on free-RAM
    // measurement just by existing.
    let mut search = plan
        .enabled
        .then(|| crate::discovery::StreamingSearch::new(budget_rows));
    let batch_columns = search.as_ref().map(|s| s.batch_columns()).unwrap_or(0);
    let space_len = search.as_ref().map(|s| s.space_len()).unwrap_or(0);

    // ── The single-pass path ────────────────────────────────────────────────
    // Taken when streaming was not requested, and when it WAS requested but
    // this machine cannot afford a batch or the space is empty. The second case
    // is named in the census (`streaming_unavailable`) rather than silently
    // behaving like the first: the run is complete and loses nothing — it just
    // did not stream, and a reader must be able to see that.
    if !plan.enabled || batch_columns == 0 || space_len == 0 {
        if plan.enabled {
            tracing::warn!(
                target: "neoethos_search::streaming",
                batch_columns,
                space_len,
                budget_rows,
                "streaming was requested but no working set could be sized (batch_columns == 0 \
                 means free RAM affords no extension beyond the resident vocabulary; \
                 space_len == 0 means the (indicator, period) space is empty). Falling back to \
                 the single whole-vocabulary pass — nothing is lost, but this run did NOT sweep."
            );
        }
        let rejected_before = crate::discovery::batch_rejection_ledger().batches_rejected;
        let features = build_features(None)?;
        let result = run_cycle(&features)?;
        let rejected_after = crate::discovery::batch_rejection_ledger().batches_rejected;
        let predicate_fired = rejected_after > rejected_before;
        let canonical_portfolio =
            canonical.remap_portfolio(result.portfolio_genes(), result.local_feature_names(), 0)?;
        let outcome = if plan.enabled {
            BatchOutcome::StreamingUnavailable
        } else if predicate_fired {
            BatchOutcome::PredicateEarlyReject
        } else if result.portfolio_genes().is_empty() {
            BatchOutcome::EmptyPortfolio
        } else {
            BatchOutcome::SurvivorsPromoted
        };
        let detail = match outcome {
            BatchOutcome::StreamingUnavailable => format!(
                "no working set could be sized (batch_columns={batch_columns}, \
                 space_len={space_len}); ran the whole-vocabulary pass instead \
                 (early_reject_predicate_fired={predicate_fired}, \
                 portfolio={})",
                result.portfolio_genes().len()
            ),
            BatchOutcome::PredicateEarlyReject => {
                "early-reject predicate fired inside the cycle — see the \
                 neoethos_search::batch_ledger census for the numbers"
                    .to_string()
            }
            BatchOutcome::EmptyPortfolio => {
                "the cycle selected no strategies".to_string()
            }
            _ => String::new(),
        };
        ledger.record(BatchLedgerEntry::new(
            0,
            0,
            0,
            0,
            outcome,
            result.portfolio_genes().len(),
            detail,
        ));
        // Returned WHETHER OR NOT it selected anything: the caller's own
        // empty-portfolio guard is the one that must fire. Parity.
        batches.push(StreamingBatchSurvivor {
            cursor: 0,
            pairs: 0,
            result,
            canonical_portfolio,
        });
        ledger.finish("streaming_working_set")?;
        ledger.log_summary("streaming_working_set");
        crate::discovery::log_batch_rejection_summary("streaming_working_set");
        let all: Vec<Gene> = batches
            .iter()
            .flat_map(|b| b.canonical_portfolio.iter().cloned())
            .collect();
        canonical.assert_indices_in_range(&all, "streaming_working_set run end")?;
        return Ok(StreamingRunOutcome {
            batches,
            canonical,
            ledger,
            streamed: false,
            next_cursor: 0,
            space_len,
            batch_columns,
        });
    }

    // ── The streaming path ──────────────────────────────────────────────────
    let Some(mut search) = search.take() else {
        bail!(
            "internal: the streaming path was entered without a sized sweep. Unreachable unless \
             the single-pass guard above stopped covering `plan.enabled == false`."
        );
    };
    tracing::info!(
        target: "neoethos_search::streaming",
        budget_rows,
        batch_columns,
        space_len,
        max_batches = plan.max_batches,
        "streaming working-set sweep starting — batch width is derived from FREE RAM and the \
         widest frame, never from a config constant; the cursor is the run's resumable state."
    );
    let mut ran = 0usize;
    loop {
        if plan.max_batches != 0 && ran >= plan.max_batches {
            ledger.record(BatchLedgerEntry::new(
                search.cursor(),
                search.cursor(),
                0,
                0,
                BatchOutcome::BudgetExhausted,
                0,
                format!(
                    "batch budget of {} spent; {} of {} pairs remain unswept. Resume from this \
                     cursor.",
                    plan.max_batches,
                    space_len.saturating_sub(search.cursor()),
                    space_len
                ),
            ));
            break;
        }
        let Some(batch) = search.next_batch() else {
            break;
        };
        let cursor = batch.cursor;
        let next_cursor = batch.next_cursor;
        let pairs = batch.pairs.len();
        let planned_columns = batch.planned_columns;
        ran += 1;

        let rejected_before = crate::discovery::batch_rejection_ledger().batches_rejected;
        let features = match build_features(Some(Arc::clone(&batch))) {
            Ok(f) => f,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_search::streaming",
                    cursor,
                    error = %err,
                    "feature build failed for this batch — recorded and skipped; sweep continues"
                );
                ledger.record(BatchLedgerEntry::new(
                    cursor,
                    next_cursor,
                    pairs,
                    planned_columns,
                    BatchOutcome::FeatureBuildFailed,
                    0,
                    format!("{err:#}"),
                ));
                continue;
            }
        };

        // Fatal by design — see `assert_working_set_observed`.
        let matched_pairs = assert_working_set_observed(&batch, &features)?;

        let result = match run_cycle(&features) {
            Ok(r) => r,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_search::streaming",
                    cursor,
                    error = %err,
                    "discovery cycle failed for this batch — recorded and skipped"
                );
                ledger.record(BatchLedgerEntry::new(
                    cursor,
                    next_cursor,
                    pairs,
                    planned_columns,
                    BatchOutcome::DiscoveryCycleFailed,
                    0,
                    format!("{err:#}"),
                ));
                continue;
            }
        };
        let rejected_after = crate::discovery::batch_rejection_ledger().batches_rejected;

        if rejected_after > rejected_before {
            ledger.record(BatchLedgerEntry::new(
                cursor,
                next_cursor,
                pairs,
                planned_columns,
                BatchOutcome::PredicateEarlyReject,
                0,
                "the early-reject predicate rejected this batch inside the cycle, before the \
                 quality screen / walk-forward / CPCV — see the neoethos_search::batch_ledger \
                 census for the expectancy, payoff and trade counts behind the verdict"
                    .to_string(),
            ));
            continue;
        }
        if result.portfolio_genes().is_empty() {
            ledger.record(BatchLedgerEntry::new(
                cursor,
                next_cursor,
                pairs,
                planned_columns,
                BatchOutcome::EmptyPortfolio,
                0,
                "the cycle ran to completion on this batch and selected no strategies (it paid \
                 for the full screen — this is NOT a predicate rejection)"
                    .to_string(),
            ));
            continue;
        }

        // ── BATCH EXIT. Remap HERE, invariant 2. ────────────────────────────
        // Nothing downstream ever sees a local index, because the survivor is
        // translated before it is stored.
        let canonical_portfolio = canonical.remap_portfolio(
            result.portfolio_genes(),
            result.local_feature_names(),
            cursor,
        )?;
        tracing::info!(
            target: "neoethos_search::streaming",
            cursor,
            next_cursor,
            pairs,
            planned_columns,
            matched_pairs,
            portfolio = canonical_portfolio.len(),
            canonical_names = canonical.len(),
            "batch kept — survivors remapped into the run-level canonical feature index"
        );
        ledger.record(BatchLedgerEntry::new(
            cursor,
            next_cursor,
            pairs,
            planned_columns,
            BatchOutcome::SurvivorsPromoted,
            canonical_portfolio.len(),
            String::new(),
        ));
        batches.push(StreamingBatchSurvivor {
            cursor,
            pairs,
            result,
            canonical_portfolio,
        });
    }

    ledger.finish("streaming_working_set")?;
    ledger.log_summary("streaming_working_set");
    crate::discovery::log_batch_rejection_summary("streaming_working_set");

    // INVARIANT 4, at run end, over every gene the run is about to hand on.
    let all: Vec<Gene> = batches
        .iter()
        .flat_map(|b| b.canonical_portfolio.iter().cloned())
        .collect();
    canonical.assert_indices_in_range(&all, "streaming_working_set run end")?;

    Ok(StreamingRunOutcome {
        batches,
        canonical,
        ledger,
        streamed: true,
        next_cursor: search.cursor(),
        space_len,
        batch_columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeResult {
        names: Vec<String>,
        portfolio: Vec<Gene>,
    }

    impl BatchSearchResult for FakeResult {
        fn local_feature_names(&self) -> &[String] {
            &self.names
        }
        fn portfolio_genes(&self) -> &[Gene] {
            &self.portfolio
        }
    }

    fn frame(names: &[&str]) -> FeatureFrame {
        FeatureFrame::from_array(
            vec![0i64, 1],
            names.iter().map(|n| n.to_string()).collect(),
            ndarray::Array2::<f32>::zeros((2, names.len())),
        )
    }

    #[test]
    fn stem_matching_is_segment_aware() {
        let name: Vec<&str> = "h1_rsi_4_signal".split('_').collect();
        let rsi4: Vec<&str> = "rsi_4".split('_').collect();
        let rsi40: Vec<&str> = "rsi_40".split('_').collect();
        assert!(column_addresses_stem(&name, &rsi4));
        assert!(
            !column_addresses_stem(&name, &rsi40),
            "rsi_4 must not be read as rsi_40"
        );
    }

    /// PARITY. Streaming disabled ⇒ exactly one build with NO working set, one
    /// cycle, and the cycle's own result handed back untouched.
    #[test]
    fn disabled_plan_builds_once_with_no_working_set() {
        let mut builds = 0usize;
        let mut installed_batches = 0usize;
        let mut cycles = 0usize;
        let outcome = run_streaming_working_set(
            &StreamingPlan::disabled(),
            1_000,
            |batch| {
                builds += 1;
                if batch.is_some() {
                    installed_batches += 1;
                }
                Ok(frame(&["a", "b"]))
            },
            |_features| {
                cycles += 1;
                Ok(FakeResult {
                    names: vec!["a".to_string(), "b".to_string()],
                    portfolio: vec![Gene {
                        strategy_id: "g".to_string(),
                        indices: vec![1],
                        weights: vec![1.0],
                        ..Default::default()
                    }],
                })
            },
        )
        .expect("single pass");
        assert_eq!(builds, 1, "one build, exactly as today");
        assert_eq!(installed_batches, 0, "no working set may be installed");
        assert_eq!(cycles, 1, "one cycle, exactly as today");
        assert!(!outcome.streamed);
        assert_eq!(outcome.batches.len(), 1);
        assert_eq!(outcome.ledger.batches_seen(), 1);
        assert_eq!(outcome.ledger.batches_rejected(), 0);
        // The remap preserved WHICH NAME the term addresses — the property
        // that matters; the canonical list is the union of referenced names,
        // so it is deliberately smaller than the batch's own list.
        let gene = &outcome.batches[0].canonical_portfolio[0];
        assert_eq!(outcome.canonical.names()[gene.indices[0]], "b");
        assert_eq!(outcome.batches[0].result.portfolio[0].indices, vec![1]);
    }

    /// PARITY, second half. An empty portfolio from the single pass still comes
    /// back to the caller, so `ensure_non_empty_portfolio` stays the guard that
    /// fires.
    #[test]
    fn disabled_plan_returns_an_empty_portfolio_to_the_caller() {
        let outcome = run_streaming_working_set(
            &StreamingPlan::disabled(),
            1_000,
            |_batch| Ok(frame(&["a"])),
            |_features| {
                Ok(FakeResult {
                    names: vec!["a".to_string()],
                    portfolio: Vec::new(),
                })
            },
        )
        .expect("single pass");
        assert_eq!(outcome.batches.len(), 1);
        assert!(outcome.batches[0].result.portfolio.is_empty());
        assert_eq!(
            outcome.ledger.count_of(BatchOutcome::EmptyPortfolio),
            1,
            "and it is NAMED, not silently dropped"
        );
    }

    #[test]
    fn a_build_error_is_named_and_never_swallows_the_run() {
        let mut ledger = StreamingRunLedger::new();
        ledger.record(BatchLedgerEntry::new(
            0,
            10,
            10,
            80,
            BatchOutcome::FeatureBuildFailed,
            0,
            "cube assembly refused",
        ));
        ledger.record(BatchLedgerEntry::new(
            10,
            20,
            10,
            80,
            BatchOutcome::SurvivorsPromoted,
            2,
            "",
        ));
        ledger.finish("test").expect("one batch reached the search");
        assert_eq!(ledger.rejected_rows().len(), 1);
        assert_eq!(ledger.rejected_rows()[0].detail, "cube assembly refused");
    }

    #[test]
    fn batch_summary_rejects_zero_saved_portfolios() {
        let summary = BatchDiscoverySummary {
            symbols_seen: 1,
            work_units_seen: 2,
            skipped_symbols: 0,
            skipped_timeframes: 1,
            feature_failures: 1,
            empty_portfolios: 0,
            portfolios_saved: 0,
            ..Default::default()
        };

        let err = summary
            .finalize()
            .expect_err("expected zero-save batch to fail");
        let msg = err.to_string();
        assert!(
            msg.contains("no usable portfolios"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("work_units=2"), "unexpected error: {msg}");
    }

    #[test]
    fn batch_summary_accepts_at_least_one_saved_portfolio() {
        let summary = BatchDiscoverySummary {
            portfolios_saved: 1,
            work_units_seen: 3,
            ..Default::default()
        };

        let finalized = summary
            .finalize()
            .expect("expected non-empty batch success");
        assert_eq!(finalized.portfolios_saved, 1);
    }
}
