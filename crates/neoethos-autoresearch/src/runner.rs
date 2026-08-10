//! The state machine — one iteration is one **sweep**.
//!
//! `docs/autoresearch-loop.md` §3. This is the only module that calls the
//! search.
//!
//! ```text
//!   S0  Open / Resume ── fold the journal, verify goal_hash + judge_hash + cost_hash
//!    │
//!    ├─> S1  PROPOSE   draw 100 proposals as ONE batch (75 posterior + 25 uniform floor)
//!    │   S2  GUARD     trust region, dedupe by config_hash, coverage debt
//!    │   S3  STAMP     ResolvedConfigStamp -> config_hash; assert_payoff_floor_reachable
//!    │   S4  RUN       SweepStarted (INTENT, fsynced) BEFORE any work; then 100 searches
//!    │   S5  COLLECT   statistics, the ten rejection counters, censuses, survivors
//!    │   S6  SCREEN    in-sample only; never touches OOS
//!    │   S7  RECORD    SweepCompleted (OUTCOME, fsynced) + PosteriorUpdated
//!    │   S8  BLOCK?    every 4 live sweeps -> the shuffle control, BlockJudged
//!    │   S9  DECIDE    R1 reached -> S10 | R2 refuted -> stop | budget -> stop
//!    └───────────────  else -> S1
//!        S10 PROMOTE   evaluate ONLY on OOS, once; then STOP either way
//! ```
//!
//! **Two-phase journalling is what makes "a crash loses one sweep, not the
//! session" true rather than hoped.** The intent record is fsynced before any
//! work starts; on resume a `SweepStarted` with no outcome becomes
//! `SweepAbandoned`, counted and named, and its `trials_offered` — recovered
//! from the sweep's partial ledger — **still counts into `N_session`**. A trial
//! that ran is a trial that ran.
//!
//! ## What this module will not do
//!
//! It never places an order, never contacts a broker, never writes
//! `live_portfolio.json` and never writes `config.yaml`. On R1 it writes
//! `proposal.json` beside the verdict and stops. The operator promotes.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::time::Instant;

use neoethos_core::config::Settings;
use neoethos_search::deflated::TrialStatisticsReport;
use neoethos_search::discovery::DiscoveryConfig;

use crate::contracts::{self, ResolvedInputs};
use crate::goals::GoalSet;
use crate::journal::{
    CostBandCounts, NamedValues, OosWindow, Record, SearchRecord, SweepKind,
};
use crate::session::{
    BestEver, ChampionRow, SessionHeader, SessionId, SessionStore, SessionWriter, SweepEvidence,
    SweepId,
};

// ─────────────────────────────────────────────────────────────────────────────
// Arguments
// ─────────────────────────────────────────────────────────────────────────────

/// What the CLI passes in. `docs/autoresearch-loop.md` §14.2.
///
/// **There is deliberately no field naming a goal constant, a cost field or a
/// judge threshold.** If there were, the loop's goals would be writable from the
/// command line, which is the same defect wearing a different hat.
#[derive(Debug, Clone)]
pub struct RunArgs {
    pub symbol: String,
    pub max_sweeps: usize,
    pub max_hours: f64,
    /// Resume an existing session. Without it, a new session id.
    pub session: Option<String>,
    /// Scenario label. Without it, the goal set's primary scenario.
    pub scenario: Option<String>,
    /// Draw and stamp proposals; run nothing.
    pub dry_run: bool,
}

impl RunArgs {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            max_sweeps: crate::DEFAULT_MAX_SWEEPS,
            max_hours: f64::INFINITY,
            session: None,
            scenario: None,
            dry_run: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The executor port — the seam between the state machine and the card
// ─────────────────────────────────────────────────────────────────────────────

/// What one search is asked to do.
#[derive(Debug, Clone)]
pub struct SearchRequest<'a> {
    pub sweep: SweepId,
    pub slot: usize,
    /// The proposal's delta already applied to the session's baseline.
    pub config: &'a DiscoveryConfig,
    pub config_hash: &'a str,
    /// Where this search's trial-returns matrix must be written.
    pub trial_returns_path: std::path::PathBuf,
    /// `Some` for a shuffle control: the permutation to apply to the feature
    /// block before the search sees it. Prices, labels, costs, exit geometry,
    /// gene encoding and the GA seed are untouched — only the relationship
    /// between features and the future is destroyed.
    pub permutation: Option<&'a crate::shuffle::FeaturePermutation>,
}

/// One search's outcome, as the loop collects it at S5.
///
/// **The loop owns this type** because the loop owns S5 COLLECT; the judge
/// consumes it and returns verdicts. Neither side reaches across.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub slot: usize,
    pub config_hash: String,
    /// Every screened candidate this search offered. The honest N contribution,
    /// counted whether or not anything survived.
    pub trials_offered: usize,
    pub statistics: TrialStatisticsReport,
    pub cost_band: CostBandCounts,
    /// The ten named rejection counters, as `(name, count)`.
    pub rejections: Vec<(String, usize)>,
    pub survivors: usize,
    /// Per-trade net expectancy of the best survivor at the PESSIMISTIC edge of
    /// the cost band. `None` when nothing survived.
    pub e_screen_pess: Option<f64>,
    pub n_trades: usize,
    /// The champion's monthly return series and its month keys — this search's
    /// candidate row for the session champion matrix.
    pub champion_returns: Vec<f64>,
    pub champion_period_keys: Vec<i64>,
    pub champion_strategy_id: String,
    pub streamed: bool,
    pub batch_columns: usize,
    pub next_cursor: usize,
    pub wall_ms: u64,
    /// A per-search failure. Named and counted; the sweep CONTINUES.
    pub error: Option<String>,
}

impl SearchOutcome {
    /// The compact journal row.
    pub fn to_record(&self) -> SearchRecord {
        SearchRecord {
            slot: self.slot,
            config_hash: self.config_hash.clone(),
            trials_offered: self.trials_offered,
            survivors: self.survivors,
            e_screen_pess: self.e_screen_pess,
            n_trades: self.n_trades,
            dsr: self.statistics.dsr_value(),
            pbo: self.statistics.pbo_value(),
            dsr_refusal: self.statistics.deflated_sharpe_refusal.clone(),
            pbo_refusal: self.statistics.pbo_refusal.clone(),
            cost_band: self.cost_band,
            rejections: self.rejections.clone(),
            streamed: self.streamed,
            batch_columns: self.batch_columns,
            next_cursor: self.next_cursor,
            wall_ms: self.wall_ms,
            error: self.error.clone(),
        }
    }
}

/// The evidence a promotion is judged on. Produced ONCE, on the OOS window.
#[derive(Debug, Clone)]
pub struct OosEvidence {
    pub window: OosWindow,
    /// Per-trade net result in pips, charged at the PESSIMISTIC edge of the
    /// cost band. A result that survives only at the optimistic edge is not a
    /// result.
    pub per_trade_net_pips: Vec<f64>,
    /// The same trades as R-multiples — the input to the goal functional.
    pub r_multiples: Vec<f64>,
    pub monthly_returns: Vec<f64>,
    pub period_keys: Vec<i64>,
    pub trades_per_day: f64,
    /// `CostBandVerdict::SurvivesBand` across the whole band, not just the
    /// point estimate.
    pub band_survives: bool,
}

/// The seam between the state machine and whatever actually runs a search.
///
/// It exists so that the loop — the part that must be provably resumable and
/// provably unable to reach a broker — can be exercised end to end without a
/// GPU, a dataset or a feature cube. A state machine that can only be tested by
/// running a real sweep is a state machine whose crash-recovery path is tested
/// by hoping.
pub trait SweepExecutor {
    /// One line naming what will actually run, written into the log at S0.
    fn describe(&self) -> String;

    /// Did the operator's configuration ask for the streaming working-set
    /// sweep? Read by the block-1 self-check (§13.2.6).
    fn streaming_requested(&self) -> bool;

    /// The windows this executor will honour:
    /// `((search_span_start_ms, search_span_end_ms), oos_window, oos_bars,
    /// bars_per_expected_trade)`.
    ///
    /// It is on the trait rather than computed by the runner because only the
    /// executor knows what it actually loaded, and §8.2 requires the loaded
    /// span to be checked against the OOS window by the party that bounded it.
    /// There is deliberately **no default body**: a default would be a place
    /// for a silent zero to hide, and a zero here reads as "no overlap" — the
    /// exact false negative that would let a sweep fit on the window its
    /// verdict is supposed to test it against.
    fn windows(&self) -> Result<((i64, i64), OosWindow, usize, f64)>;

    /// Run one search.
    ///
    /// Returning `Err` fails the SWEEP; returning an `Ok` outcome whose `error`
    /// is `Some` fails only that SEARCH and the sweep continues, which is the
    /// documented failure policy: one bad region must not throw away the
    /// regions already searched.
    fn execute(&mut self, request: &SearchRequest<'_>) -> Result<SearchOutcome>;

    /// Evaluate — **never search** — a promotion candidate on the OOS window.
    ///
    /// Nothing is fitted and nothing is selected here: the candidate's genes
    /// were chosen in sample and are simply backtested forward.
    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        config: &DiscoveryConfig,
    ) -> Result<OosEvidence>;
}

/// An executor that refuses to run anything.
///
/// Not used by [`run`] — a `--dry-run` still resolves the REAL executor, so the
/// dry run exercises the dataset load, the window derivation and the whole
/// startup self-check, which is most of what a dry run is for. This one exists
/// for the state-machine tests and as the reference implementation of the port:
/// it does not pretend to produce statistics, because an empty outcome is
/// indistinguishable from a sweep that found nothing.
pub struct RefusingExecutor {
    windows: ((i64, i64), OosWindow, usize, f64),
}

impl RefusingExecutor {
    pub fn new(search_span: (i64, i64), oos: OosWindow, oos_bars: usize, bars_per_trade: f64) -> Self {
        Self {
            windows: (search_span, oos, oos_bars, bars_per_trade),
        }
    }
}

impl SweepExecutor for RefusingExecutor {
    fn describe(&self) -> String {
        "REFUSING EXECUTOR — no search is executed and no bar is read".to_string()
    }

    fn streaming_requested(&self) -> bool {
        false
    }

    fn windows(&self) -> Result<((i64, i64), OosWindow, usize, f64)> {
        Ok(self.windows)
    }

    fn execute(&mut self, request: &SearchRequest<'_>) -> Result<SearchOutcome> {
        bail!(
            "the refusing executor was asked to run {} slot {}. Reaching S4 with it means the \
             runner's dry-run branch is wrong. Refusing to return an empty outcome, which would \
             be indistinguishable from a sweep that found nothing.",
            request.sweep,
            request.slot
        )
    }

    fn evaluate_oos(
        &mut self,
        _sweep: SweepId,
        _slot: usize,
        _config: &DiscoveryConfig,
    ) -> Result<OosEvidence> {
        bail!("the refusing executor never touches the out-of-sample window")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolved startup state
// ─────────────────────────────────────────────────────────────────────────────

/// Everything S0 resolved, held for the life of the session.
struct LoopContext {
    goals: GoalSet,
    scenario: crate::goals::Scenario,
    thresholds: crate::judge::JudgeThresholds,
    base_config: DiscoveryConfig,
    cost_hash: String,
    judge_hash: String,
    store: SessionStore,
    oos_window: OosWindow,
    max_sweeps: usize,
    max_hours: f64,
    symbol: String,
}

/// The frozen cost model and validation geometry, as `(field path, value)`.
///
/// Both go into `cost_hash` rather than into `judge_hash`, and the reason is
/// worth writing down: a judge threshold is a bar the loop must clear; these
/// are properties of the MEASUREMENT itself. `cost_band_pips` decides what a
/// trade is charged and the CPCV geometry decides which rows a candidate was
/// ever scored on — change either and the earlier sweeps' numbers no longer
/// mean what the later ones will, whatever the judge's bars are.
///
/// They are denormalised into the session header by field path, so a reader
/// holding only `journal.jsonl` can see exactly what was hashed.
fn frozen_cost_fields(config: &DiscoveryConfig, pip_value_per_lot: f64) -> NamedValues {
    NamedValues(vec![
        (
            "evaluation_spread_pips".into(),
            format!("{}", config.evaluation_spread_pips),
        ),
        (
            "evaluation_commission_per_trade".into(),
            format!("{}", config.evaluation_commission_per_trade),
        ),
        (
            "swap_long_pips_per_day".into(),
            format!("{}", config.swap_long_pips_per_day),
        ),
        (
            "swap_short_pips_per_day".into(),
            format!("{}", config.swap_short_pips_per_day),
        ),
        (
            "session_spread_pips".into(),
            format!("{:?}", config.session_spread_pips),
        ),
        ("cost_band_pips".into(), format!("{:?}", config.cost_band_pips)),
        ("pip_value_per_lot".into(), format!("{pip_value_per_lot}")),
        ("enable_cpcv".into(), format!("{}", config.enable_cpcv)),
        ("cpcv_n_splits".into(), format!("{}", config.cpcv_n_splits)),
        (
            "cpcv_n_test_groups".into(),
            format!("{}", config.cpcv_n_test_groups),
        ),
        (
            "cpcv_embargo_pct".into(),
            format!("{}", config.cpcv_embargo_pct),
        ),
        ("cpcv_purge_pct".into(), format!("{}", config.cpcv_purge_pct)),
        ("cpcv_max_rows".into(), format!("{}", config.cpcv_max_rows)),
    ])
}

fn hash_named(values: &NamedValues) -> String {
    let canonical = values
        .0
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";");
    crate::fnv64(&canonical)
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run (or resume) a session to a verdict.
///
/// This is the ONE function the CLI calls (§14.2). It resolves the real
/// executor and hands off to [`run_with_executor`].
pub fn run(args: RunArgs, settings: &Settings) -> Result<crate::verdict::SessionVerdict> {
    let base_config = DiscoveryConfig {
        evaluation_symbol: args.symbol.clone(),
        ..DiscoveryConfig::from_settings(settings)
    }
    .apply_mode_overrides();

    // A `--dry-run` resolves the REAL executor too: the dataset load, the
    // window derivation and the whole startup self-check are most of what a dry
    // run is for. It simply stops after S3 STAMP without executing anything.
    let mut executor = streaming::StreamingSweepExecutor::resolve(settings, &base_config)
        .context("resolving the search executor for the autoresearch loop")?;
    run_with_executor(args, settings, base_config, &mut executor)
}

/// The state machine proper, against any [`SweepExecutor`].
pub fn run_with_executor(
    args: RunArgs,
    settings: &Settings,
    base_config: DiscoveryConfig,
    executor: &mut dyn SweepExecutor,
) -> Result<crate::verdict::SessionVerdict> {
    let started = Instant::now();

    // ── S0 OPEN / RESUME ────────────────────────────────────────────────────
    let (mut ctx, mut writer, mut proposer) = open_or_resume(&args, settings, base_config, executor)?;

    // A session that already reached a verdict is not resumed into more sweeps.
    // The OOS window is spent, and a second promotion would need a second one.
    if let Some(verdict) = writer.session().stopped.clone() {
        tracing::info!(
            target: "neoethos_autoresearch::runner",
            session = %ctx.store.session_id(),
            "this session has already stopped with a verdict; nothing further to run"
        );
        return Ok(verdict);
    }

    if args.dry_run {
        return dry_run(&mut ctx, &mut writer, &mut proposer);
    }

    // ── the loop ────────────────────────────────────────────────────────────
    loop {
        // Budget first, so a session that has run out stops with a verdict
        // rather than beginning a sweep it cannot finish.
        if let Some(reason) = budget_exhausted(&ctx, writer.session(), started) {
            return stop(&mut ctx, &mut writer, reason);
        }

        let sweep = writer.session().next_sweep();

        // ── S1 PROPOSE / S2 GUARD / S3 STAMP ────────────────────────────────
        //
        // All 100 proposals are drawn BEFORE any of them runs, so a sweep is one
        // experiment with one sweep_hash and is reproducible from
        // (session_seed, sweep_index) alone.
        let drawn = proposer
            .draw_sweep(sweep, writer.session())
            .with_context(|| format!("drawing the 100 proposals of {sweep}"))?;

        journal_draw(&mut writer, sweep, &drawn)?;

        if drawn.exhausted {
            // U5 — the reachable space is enumerated. That is a result about a
            // searched space, not a failure.
            return stop(
                &mut ctx,
                &mut writer,
                crate::verdict::StopReason::ProposerExhausted,
            );
        }
        if drawn.proposals.is_empty() {
            bail!(
                "the proposer returned no proposals for {sweep} and did not report exhaustion. \
                 Refusing to journal a SweepStarted for a sweep with nothing to run — an empty \
                 sweep that completes silently is the silent drop with extra steps."
            );
        }

        // ── S4 RUN / S5 COLLECT ─────────────────────────────────────────────
        //
        // A sweep that fails outright still STOPS WITH A VERDICT, in the same
        // report shape as a success. An abort that produces no `verdict.json`
        // is the multi-hour run that ended with no artifact — the failure mode
        // this repository has actually lived through.
        let evidence = match run_sweep(
            &mut ctx,
            &mut writer,
            executor,
            sweep,
            SweepKind::Live,
            &drawn.proposals,
            None,
        ) {
            Ok(evidence) => evidence,
            Err(err) => {
                return stop(
                    &mut ctx,
                    &mut writer,
                    crate::verdict::StopReason::InfrastructureAbort {
                        detail: format!("{sweep} failed: {err:#}"),
                    },
                );
            }
        };

        // §13.2.5 — on the FIRST live sweep, an unreadable matrix everywhere
        // means the DSR/PBO wiring is ABSENT, not that the configuration was
        // bad.
        if writer.session().live_sweeps_run() == 1 {
            let readable = evidence
                .statistics
                .iter()
                .filter(|s| s.deflated_sharpe.is_some() || s.pbo.is_some())
                .count();
            if let Err(missing) = contracts::assert_first_sweep_readable(
                readable,
                "every search in sweep 1 returned an unreadable trial-returns matrix",
            ) {
                return stop(
                    &mut ctx,
                    &mut writer,
                    crate::verdict::StopReason::InfrastructureAbort {
                        detail: missing.to_string(),
                    },
                );
            }
        }

        // ── S6 SCREEN (in-sample only; never touches OOS) ────────────────────
        let null = crate::shuffle::ShuffleNull::from_session(writer.session());
        let screen = crate::judge::screen_sweep(
            &evidence,
            &ctx.thresholds,
            writer.session().n_session(),
            &null,
        );

        // ── S7 RECORD ───────────────────────────────────────────────────────
        for (slot, result) in screen.per_slot.iter().enumerate() {
            writer.append(Record::Screened {
                sweep,
                slot,
                screen_result: result.clone(),
                failing_conjunct: result.failing_conjunct_label(),
            })?;
        }
        let credits = proposer.credits_for(&drawn.proposals, &screen.per_slot);
        writer.append(Record::PosteriorUpdated { sweep, credits })?;

        if let Some(champion) = screen.champion.clone() {
            writer.session_mut().push_champion(champion.clone());
            ctx.store.write_champions(&writer.session().champions)?;
            writer.session_mut().offer_best_ever(BestEver {
                sweep,
                slot: screen.champion_slot.unwrap_or(0),
                config_hash: champion.config_hash.clone(),
                e_screen_pess: champion.e_screen_pess,
                n_trades: screen.champion_n_trades,
                dsr: screen.champion_dsr,
                pbo: screen.champion_pbo,
            });
        }

        // ── S8 BLOCK BOUNDARY ───────────────────────────────────────────────
        if writer.session().block_boundary_due() {
            if let Err(err) = run_block_control(&mut ctx, &mut writer, executor, &mut proposer) {
                // A block with no control is a block whose results are
                // unfalsifiable, so this stops the session rather than
                // continuing without a null.
                return stop(
                    &mut ctx,
                    &mut writer,
                    crate::verdict::StopReason::InfrastructureAbort {
                        detail: format!("the shuffle control failed: {err:#}"),
                    },
                );
            }
            // §13.2.6 — after block 1, a streaming session that never streamed
            // is a session running a single-pass search under a streaming label.
            if writer.session().blocks.len() == 1 {
                let (sweeps_in_block, streamed) = {
                    let block = writer
                        .session()
                        .live_sweeps_in_block(crate::session::BlockId(0));
                    let streamed = block
                        .iter()
                        .filter(|s| s.per_search.iter().any(|p| p.streamed))
                        .count();
                    (block.len(), streamed)
                };
                if let Err(missing) = contracts::assert_block_one_streamed(
                    executor.streaming_requested(),
                    sweeps_in_block,
                    streamed,
                ) {
                    return stop(
                        &mut ctx,
                        &mut writer,
                        crate::verdict::StopReason::InfrastructureAbort {
                            detail: missing.to_string(),
                        },
                    );
                }
            }
        }

        // ── S9 DECIDE ───────────────────────────────────────────────────────
        match crate::verdict::decide(writer.session(), &ctx.goals, &ctx.thresholds) {
            crate::verdict::Decision::Continue => {
                gc(&mut ctx, &writer)?;
            }
            crate::verdict::Decision::Promote(candidate) => {
                return promote(&mut ctx, &mut writer, executor, candidate);
            }
            crate::verdict::Decision::Stop(reason) => {
                return stop(&mut ctx, &mut writer, reason);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S0
// ─────────────────────────────────────────────────────────────────────────────

fn open_or_resume(
    args: &RunArgs,
    settings: &Settings,
    base_config: DiscoveryConfig,
    executor: &dyn SweepExecutor,
) -> Result<(LoopContext, SessionWriter, crate::proposer::Proposer)> {
    // The goals abort here (G1–G6) if any of them is not a goal.
    let goals = GoalSet::load(settings).map_err(contracts::MissingCapability::Goals)?;
    let scenario = match &args.scenario {
        Some(label) => goals
            .scenario_by_label(label)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no scenario is labelled {label:?}. Available: {:?}. The loop will NOT fall \
                     back to the primary scenario — silently optimising toward a different goal \
                     than the one you typed is the failure this crate exists to prevent. \
                     (scenarios source: {})",
                    goals.scenarios().iter().map(|s| s.label()).collect::<Vec<_>>(),
                    goals.scenarios_source()
                )
            })?,
        None => goals.primary().clone(),
    };

    let thresholds = crate::judge::JudgeThresholds::frozen();
    let judge_hash = thresholds.judge_hash();

    let pip_value_per_lot = base_config.evaluation_config(None).pip_value_per_lot;
    let costs = frozen_cost_fields(&base_config, pip_value_per_lot);
    let cost_hash = hash_named(&costs);

    // The OOS window is defined ONCE, here, and recorded in the header.
    let (search_span, oos_window, oos_bars, bars_per_trade) = executor.windows()?;

    let root = SessionStore::root()?;
    let created_utc = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let session_id = match &args.session {
        Some(raw) => SessionId::parse(raw)?,
        None => {
            let base = SessionId::new(&created_utc, goals.goal_hash(), &judge_hash, &cost_hash);
            if args.dry_run {
                // A dry run must never contaminate a real session's
                // config_hash set: proposals drawn but never run would be
                // refused as duplicates for the rest of the session's life.
                SessionId::parse(&format!("{base}-dry"))?
            } else {
                base
            }
        }
    };
    let store = SessionStore::open_under(&root, session_id.clone())?;

    let reproduction = format!(
        "neoethos autoresearch --symbol {} --max-sweeps {} --max-hours {} --session {}{}",
        args.symbol,
        args.max_sweeps,
        args.max_hours,
        session_id,
        if args.dry_run { " --dry-run" } else { "" }
    );

    let header = SessionHeader {
        schema: crate::SESSION_SCHEMA.to_string(),
        session_id: session_id.clone(),
        session_seed: session_seed_for(&session_id),
        symbol: args.symbol.clone(),
        created_utc: created_utc.clone(),
        goal_hash: goals.goal_hash().to_string(),
        judge_hash: judge_hash.clone(),
        cost_hash: cost_hash.clone(),
        scenarios_source: goals.scenarios_source().to_string(),
        identity_source: identity_source(),
        oos_window,
        reproduction: reproduction.clone(),
    };

    let resuming = store.header_path().exists();
    if resuming {
        let stored = store.read_header()?;
        store.assert_resumable(
            &stored,
            goals.goal_hash(),
            &judge_hash,
            &cost_hash,
            &args.symbol,
        )?;
    } else {
        store.write_header_once(&header)?;
    }

    let mut writer = SessionWriter::open(store.journal_path())?;

    // The very first record of a new journal.
    if writer.journal().is_empty() {
        writer.append(Record::SessionOpened {
            session_id: session_id.clone(),
            session_seed: header.session_seed,
            goal_hash: header.goal_hash.clone(),
            judge_hash: header.judge_hash.clone(),
            cost_hash: header.cost_hash.clone(),
            goals: NamedValues(
                goals
                    .field_values()
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), v.clone()))
                    .collect(),
            ),
            scenarios_source: goals.scenarios_source().to_string(),
            judge: thresholds.named_values(),
            costs: costs.clone(),
            oos_window,
            priors: "Beta(1, 1) on every axis-A level and every axis-B variant — the loop starts \
                     with no opinion"
                .to_string(),
            budget: NamedValues(vec![
                ("max_sweeps".into(), args.max_sweeps.to_string()),
                ("max_hours".into(), args.max_hours.to_string()),
                ("sweep_searches".into(), crate::SWEEP_SEARCHES.to_string()),
            ]),
            identity_source: identity_source(),
            symbol: args.symbol.clone(),
        })?;
    }

    // A sweep the process died inside. Counted, named, and its trials still
    // count into N_session.
    if let Some(open) = writer.session().open_sweep.clone() {
        let recovered = store.recover_partial_trials(open.sweep);
        tracing::warn!(
            target: "neoethos_autoresearch::runner",
            sweep = %open.sweep,
            trials_offered = recovered,
            "resuming: {} was started and never finished — the process died inside it. Recording \
             it as ABANDONED. Its {recovered} trials STILL count into N_session, because a trial \
             that ran is a trial that ran.",
            open.sweep
        );
        writer.append(Record::SweepAbandoned {
            sweep: open.sweep,
            trials_offered: recovered,
            reason: "process died between the SweepStarted intent and its outcome".to_string(),
        })?;
    }

    // §13.2 — the startup self-check. Every failure aborts.
    let resolved = ResolvedInputs {
        goals: &goals,
        cost_band_pips: base_config.cost_band_pips,
        baseline_cost_pips: neoethos_search::run_identity::cost_pips_round_trip(
            base_config.evaluation_spread_pips,
            base_config.evaluation_commission_per_trade,
            pip_value_per_lot,
        ),
        baseline_payoff_floor: base_config.target_profile.min_payoff_ratio,
        baseline_payoff_inputs: neoethos_search::run_identity::payoff_inputs_for_config(
            &base_config,
            pip_value_per_lot,
        ),
        search_span_ms: search_span,
        oos_window_ms: (oos_window.start_ms, oos_window.end_ms),
        oos_bars,
        n_min_oos: ctx_n_min_oos(&base_config, oos_window),
        oos_bars_per_expected_trade: bars_per_trade,
        mirror_risky: (
            base_config.risky_start_balance,
            base_config.risky_target_balance,
            base_config.risky_horizon_days,
        ),
    };
    contracts::startup_selfcheck(&resolved)?;

    tracing::info!(
        target: "neoethos_autoresearch::runner",
        session = %session_id,
        symbol = %args.symbol,
        resuming,
        sweeps_run = writer.session().sweeps_run(),
        n_session = writer.session().n_session(),
        goal_hash = %header.goal_hash,
        judge_hash = %header.judge_hash,
        cost_hash = %header.cost_hash,
        scenario = %scenario.label(),
        scenarios_source = %goals.scenarios_source(),
        executor = %executor.describe(),
        "autoresearch session open"
    );

    let proposer = crate::proposer::Proposer::restore(
        writer.session(),
        header.session_seed,
        &base_config,
    )?;

    let ctx = LoopContext {
        goals,
        scenario,
        thresholds,
        base_config,
        cost_hash,
        judge_hash,
        store,
        oos_window,
        max_sweeps: args.max_sweeps,
        max_hours: args.max_hours,
        symbol: args.symbol.clone(),
    };
    Ok((ctx, writer, proposer))
}

/// A deterministic session seed from the session id, so `(session_seed,
/// sweep_index)` replays a sweep's proposals exactly and the id in the
/// directory name is enough to reproduce them.
fn session_seed_for(id: &SessionId) -> u64 {
    let hash = crate::fnv64(id.as_str());
    u64::from_str_radix(hash.trim_start_matches("fnv64:"), 16).unwrap_or(0x9E37_79B9_7F4A_7C15)
}

/// §14.3 — what makes two proposals the same experiment.
///
/// `ResolvedConfigStamp` does not carry the GA seed today, so two proposals
/// differing ONLY in `replicate_seed` hash identically and would be refused as
/// duplicates — silently deleting the replicate dimension the judge needs to
/// estimate within-cell variance. Until §14.3 lands, identity is the PAIR, and
/// the session header says so out loud rather than letting the dedupe quietly
/// eat replicates.
fn identity_source() -> String {
    "config_hash + external replicate_seed (run_identity.rs lacks ga_seed — \
     docs/autoresearch-loop.md §14.3)"
        .to_string()
}

/// `N_MIN_OOS` is **derived**, never a constant (§15).
///
/// `min_trades_per_month × months_in_window`, where the months come from the
/// OOS window's own wall-clock duration and `min_trades_per_month` from the
/// configured `min_trades_per_day`. Deriving it is the point: a fixed trade
/// count degrades differently on every dataset — it is a strict bar on a
/// two-month window and no bar at all on a two-year one.
fn ctx_n_min_oos(config: &DiscoveryConfig, oos_window: OosWindow) -> usize {
    const MS_PER_DAY: f64 = 86_400_000.0;
    let days = ((oos_window.end_ms - oos_window.start_ms).max(0) as f64 / MS_PER_DAY).max(1.0);
    let months = (days / 30.0).max(1.0);
    let per_month = (config.min_trades_per_day.max(0.05) * 30.0).max(1.0);
    (per_month * months).round() as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// S4 / S5
// ─────────────────────────────────────────────────────────────────────────────

/// Run one sweep's 100 searches. `SweepStarted` is fsynced BEFORE any work.
fn run_sweep(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    executor: &mut dyn SweepExecutor,
    sweep: SweepId,
    kind: SweepKind,
    proposals: &[crate::proposal::Proposal],
    permutation: Option<&crate::shuffle::FeaturePermutation>,
) -> Result<SweepEvidence> {
    let sweep_hash = sweep_hash_of(proposals);
    ctx.store.prepare_sweep(sweep)?;
    ctx.store
        .write_sweep_artifact(sweep, "proposals.json", &proposals)?;

    // INTENT — fsynced before a single bar is read. Everything after this point
    // is recoverable; everything before it never happened.
    writer.append(Record::SweepStarted {
        sweep,
        sweep_hash: sweep_hash.clone(),
        kind: kind.clone(),
        started_ms: chrono::Utc::now().timestamp_millis(),
    })?;

    let began = Instant::now();
    let mut outcomes: Vec<SearchOutcome> = Vec::with_capacity(proposals.len());
    let mut trials_offered = 0usize;

    for (slot, proposal) in proposals.iter().enumerate() {
        let config = match proposal.resolve(&ctx.base_config) {
            Ok(config) => config,
            Err(err) => {
                // A proposal that cannot be resolved is a refusal, not a
                // silently skipped slot.
                writer.append(Record::ProposalRefused {
                    sweep,
                    slot,
                    config_hash: proposal.config_hash.clone(),
                    reason: format!("delta could not be applied: {err:#}"),
                })?;
                continue;
            }
        };
        let request = SearchRequest {
            sweep,
            slot,
            config: &config,
            config_hash: &proposal.config_hash,
            trial_returns_path: ctx.store.trial_returns_path(sweep, slot),
            permutation,
        };

        match executor.execute(&request) {
            Ok(outcome) => {
                trials_offered += outcome.trials_offered;
                // The partial ledger is what makes a crashed sweep's
                // `trials_offered` honest on resume.
                ctx.store
                    .record_partial(sweep, slot, outcome.trials_offered)?;
                outcomes.push(outcome);
            }
            Err(err) => {
                // A whole-sweep failure. The OUTCOME record is still written,
                // so the sweep is never left open, and its partial trials still
                // count.
                let recovered = ctx.store.recover_partial_trials(sweep);
                writer.append(Record::SweepFailed {
                    sweep,
                    trials_offered: recovered.max(trials_offered),
                    error: format!("{err:#}"),
                })?;
                return Err(err).with_context(|| format!("{sweep} slot {slot}"));
            }
        }
    }

    let wall_ms = began.elapsed().as_millis() as u64;
    let per_search: Vec<SearchRecord> = outcomes.iter().map(SearchOutcome::to_record).collect();
    let statistics: Vec<TrialStatisticsReport> =
        outcomes.iter().map(|o| o.statistics.clone()).collect();

    ctx.store
        .write_sweep_artifact(sweep, "statistics.json", &statistics)?;
    ctx.store
        .write_sweep_artifact(sweep, "censuses.json", &per_search)?;

    // OUTCOME.
    writer.append(Record::SweepCompleted {
        sweep,
        trials_offered,
        per_search: per_search.clone(),
        wall_ms,
    })?;

    let champion = best_champion(sweep, &outcomes);

    Ok(SweepEvidence {
        sweep,
        kind,
        sweep_hash,
        per_search,
        statistics,
        champion,
        trials_offered,
        wall_ms,
    })
}

/// The ordered hash of a sweep's 100 `config_hash`es — one experiment, one
/// identity.
fn sweep_hash_of(proposals: &[crate::proposal::Proposal]) -> String {
    let canonical = proposals
        .iter()
        .map(|p| p.config_hash.as_str())
        .collect::<Vec<_>>()
        .join("|");
    crate::fnv64(&canonical)
}

/// The sweep's champion row for the session champion matrix: the survivor with
/// the highest PESSIMISTIC-edge expectancy.
fn best_champion(sweep: SweepId, outcomes: &[SearchOutcome]) -> Option<ChampionRow> {
    outcomes
        .iter()
        .filter(|o| o.error.is_none() && !o.champion_returns.is_empty())
        .filter_map(|o| o.e_screen_pess.map(|e| (o, e)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(o, e)| ChampionRow {
            sweep,
            config_hash: o.config_hash.clone(),
            strategy_id: o.champion_strategy_id.clone(),
            period_keys: o.champion_period_keys.clone(),
            monthly_returns: o.champion_returns.clone(),
            e_screen_pess: e,
        })
}

fn journal_draw(
    writer: &mut SessionWriter,
    sweep: SweepId,
    drawn: &crate::proposer::DrawnSweep,
) -> Result<()> {
    for duplicate in &drawn.duplicates {
        writer.append(Record::ProposalDuplicate {
            sweep,
            slot: duplicate.slot,
            config_hash: duplicate.config_hash.clone(),
            first_seen_sweep: duplicate.first_seen_sweep,
        })?;
    }
    for refused in &drawn.refused {
        writer.append(Record::ProposalRefused {
            sweep,
            slot: refused.slot,
            config_hash: refused.config_hash.clone(),
            reason: refused.reason.to_string(),
        })?;
    }
    for (slot, proposal) in drawn.proposals.iter().enumerate() {
        writer.append(Record::ProposalDrawn {
            sweep,
            slot,
            proposal: proposal.clone(),
            origin: format!("{:?}", proposal.origin),
        })?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// S8 — the block boundary
// ─────────────────────────────────────────────────────────────────────────────

/// Re-run the block's **best-screening** sweep's exact 100 proposals, byte for
/// byte, against rotated features.
///
/// Comparing against the block's best is the right comparison because the loop
/// reports maxima: a null built from average runs would be too easy to beat,
/// and a falsification that is easy to pass falsifies nothing.
fn run_block_control(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    executor: &mut dyn SweepExecutor,
    proposer: &mut crate::proposer::Proposer,
) -> Result<()> {
    let block = crate::session::BlockId((writer.session().blocks.len()) as u32);
    let Some(source) = crate::shuffle::best_sweep_of_block(writer.session(), block) else {
        bail!(
            "{block} closed but no live sweep in it could be identified as the best-screening \
             one. The shuffle control has nothing to re-run, and a block with no control is a \
             block whose results are unfalsifiable."
        );
    };
    let Some(proposals) = writer.session().proposals_of(source).map(<[_]>::to_vec) else {
        bail!(
            "{block}'s best sweep {source} has no recorded proposals, so its control cannot re-run \
             them byte for byte. The journal is the memory; this means a ProposalDrawn record is \
             missing."
        );
    };

    let permutation = crate::shuffle::FeaturePermutation::draw(
        crate::shuffle::ControlKind::CircularRotation,
        writer.session(),
        block,
    );

    let control_sweep = writer.session().next_sweep();
    let evidence = run_sweep(
        ctx,
        writer,
        executor,
        control_sweep,
        SweepKind::Control {
            control: permutation.kind(),
            source_sweep: source,
            block,
        },
        &proposals,
        Some(&permutation),
    )?;

    let control_best = evidence
        .per_search
        .iter()
        .filter_map(|s| s.e_screen_pess)
        .fold(None::<f64>, |acc, e| Some(acc.map_or(e, |a| a.max(e))));
    let control_dsr = evidence
        .per_search
        .iter()
        .filter_map(|s| s.dsr)
        .fold(None::<f64>, |acc, d| Some(acc.map_or(d, |a| a.max(d))));

    writer.append(Record::ShuffleControlCompleted {
        block,
        kind: permutation.kind(),
        tau: permutation.tau(),
        source_sweep: source,
        trials_offered: evidence.trials_offered,
        e_screen_pess: control_best,
        dsr: control_dsr,
    })?;

    let judgement = crate::shuffle::judge_block(writer.session(), block, control_best, control_dsr);
    writer.append(Record::BlockJudged {
        block,
        delta_expectancy: judgement.delta_expectancy,
        p_block: judgement.p_block,
        dsr_gap: judgement.dsr_gap,
        indistinguishable: judgement.indistinguishable,
    })?;

    if judgement.indistinguishable {
        // Every success credited from this block is retracted and re-credited
        // as a failure, by replaying the journal. The posterior is a fold, so
        // this is exact, not an approximation.
        let sweep_ids: Vec<SweepId> = writer
            .session()
            .live_sweeps_in_block(block)
            .iter()
            .map(|s| s.sweep)
            .collect();
        writer.append(Record::PosteriorRetracted {
            block,
            sweep_ids,
            reason: "block indistinguishable".to_string(),
        })?;
        proposer.reload(writer.session());
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// S10 — the one OOS touch
// ─────────────────────────────────────────────────────────────────────────────

fn promote(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    executor: &mut dyn SweepExecutor,
    candidate: crate::verdict::PromotionCandidate,
) -> Result<crate::verdict::SessionVerdict> {
    if writer.session().oos_spent() {
        bail!(
            "the promotion path was reached with the OOS touch ALREADY SPENT. A configuration \
             that touches the out-of-sample window more than once has spent it, and the second \
             reading is not out of sample. Refusing."
        );
    }

    let proposals = writer
        .session()
        .proposals_of(candidate.sweep)
        .map(<[_]>::to_vec)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the promotion candidate's sweep {} has no recorded proposals, so its \
                 configuration cannot be reconstructed for the OOS evaluation",
                candidate.sweep
            )
        })?;
    let proposal = proposals.get(candidate.slot).ok_or_else(|| {
        anyhow::anyhow!(
            "{} has no proposal at slot {}",
            candidate.sweep,
            candidate.slot
        )
    })?;
    let config = proposal.resolve(&ctx.base_config)?;

    // The budget is spent BEFORE the evaluation, so a crash inside it cannot
    // leave the session believing the window is still clean.
    writer.append(Record::OosTouchSpent {
        window: ctx.oos_window,
        sweep: candidate.sweep,
        candidate: proposal.config_hash.clone(),
    })?;

    let oos = executor
        .evaluate_oos(candidate.sweep, candidate.slot, &config)
        .with_context(|| {
            format!(
                "evaluating {} slot {} on the out-of-sample window",
                candidate.sweep, candidate.slot
            )
        })?;

    let outcome = crate::judge::promote(
        &candidate,
        &oos,
        &ctx.thresholds,
        &ctx.goals,
        &ctx.scenario,
        writer.session(),
    )?;

    if outcome.promoted {
        writer.append(Record::Promoted {
            sweep: candidate.sweep,
            goal_metric: outcome.goal_metric,
            goal_bar: outcome.goal_bar,
            frontier: outcome.frontier.clone(),
        })?;
        // The loop's ONLY tradeable output. It is not `live_portfolio.json`, it
        // is not written where the trading side reads, and nothing here sizes a
        // position or contacts a broker. The operator promotes.
        ctx.store.write_proposal(&serde_json::json!({
            "schema": "neoethos.autoresearch.proposal.v1",
            "session_id": ctx.store.session_id().as_str(),
            "sweep": candidate.sweep,
            "slot": candidate.slot,
            "config_hash": proposal.config_hash,
            "goal_hash": ctx.goals.goal_hash(),
            "judge_hash": ctx.judge_hash,
            "cost_hash": ctx.cost_hash,
            "scenario": ctx.scenario.label(),
            "goal_metric": outcome.goal_metric,
            "goal_bar": outcome.goal_bar,
            "frontier": outcome.frontier,
            "oos_window": ctx.oos_window,
            "oos_trades": oos.per_trade_net_pips.len(),
            "proposal": proposal,
            "NOT_A_PROMOTION": "This file is a PROPOSAL. Nothing in the trading path reads it. \
                                Promoting it is the operator's act.",
        }))?;
        stop(ctx, writer, crate::verdict::StopReason::GoalReached(candidate))
    } else {
        let conjunct = outcome
            .failing_conjunct
            .clone()
            .unwrap_or_else(|| "unnamed".to_string());
        writer.append(Record::PromotionRefused {
            sweep: candidate.sweep,
            failing_conjunct: conjunct.clone(),
        })?;
        // STOP either way — the OOS window is spent.
        stop(
            ctx,
            writer,
            crate::verdict::StopReason::PromotionRefused {
                sweep: candidate.sweep,
                failing_conjunct: conjunct,
            },
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stopping, budget, GC, dry run
// ─────────────────────────────────────────────────────────────────────────────

fn budget_exhausted(
    ctx: &LoopContext,
    session: &crate::session::Session,
    started: Instant,
) -> Option<crate::verdict::StopReason> {
    if session.sweeps_run() >= ctx.max_sweeps {
        return Some(crate::verdict::StopReason::BudgetExhausted {
            sweeps_run: session.sweeps_run(),
            max_sweeps: ctx.max_sweeps,
        });
    }
    if ctx.max_hours.is_finite() && started.elapsed().as_secs_f64() / 3600.0 >= ctx.max_hours {
        return Some(crate::verdict::StopReason::WallClock {
            hours: ctx.max_hours,
        });
    }
    None
}

fn stop(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    reason: crate::verdict::StopReason,
) -> Result<crate::verdict::SessionVerdict> {
    let verdict = crate::verdict::build(
        writer.session(),
        &ctx.goals,
        &ctx.scenario,
        &ctx.thresholds,
        reason,
        crate::verdict::VerdictContext {
            session_id: ctx.store.session_id().as_str().to_string(),
            symbol: ctx.symbol.clone(),
            cost_hash: ctx.cost_hash.clone(),
            judge_hash: ctx.judge_hash.clone(),
            oos_window: ctx.oos_window,
        },
    );
    writer.append(Record::SessionStopped {
        verdict: verdict.clone(),
    })?;
    ctx.store.write_verdict(&verdict)?;
    tracing::info!(
        target: "neoethos_autoresearch::runner",
        session = %ctx.store.session_id(),
        "{}",
        writer.session().census_line()
    );
    Ok(verdict)
}

/// §3.2 — delete the trial-returns matrices this session no longer needs, under
/// a named census. Never silently.
fn gc(ctx: &mut LoopContext, writer: &SessionWriter) -> Result<()> {
    let mut keep: HashSet<SweepId> = HashSet::new();
    if let Some(best) = &writer.session().best_ever {
        keep.insert(best.sweep);
    }
    for summary in &writer.session().sweeps {
        if !summary.kind.is_live() {
            keep.insert(summary.sweep);
        }
    }
    let census = ctx.store.gc_trial_matrices(&keep)?;
    if census.deleted > 0 {
        tracing::info!(
            target: "neoethos_autoresearch::runner",
            kept = census.kept,
            deleted = census.deleted,
            bytes_reclaimed = census.bytes_reclaimed,
            free_bytes_after = ?census.free_bytes_after,
            rule = %census.rule,
            "sweeps_gc"
        );
    }
    Ok(())
}

/// `--dry-run`: draw and stamp one sweep's proposals, write them, run nothing.
///
/// It runs under its own session id (`…-dry`) so the proposals it draws can
/// never be refused as duplicates by a later real session.
fn dry_run(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    proposer: &mut crate::proposer::Proposer,
) -> Result<crate::verdict::SessionVerdict> {
    let sweep = writer.session().next_sweep();
    let drawn = proposer.draw_sweep(sweep, writer.session())?;
    journal_draw(writer, sweep, &drawn)?;
    ctx.store
        .prepare_sweep(sweep)?;
    ctx.store
        .write_sweep_artifact(sweep, "proposals.json", &drawn.proposals)?;
    tracing::info!(
        target: "neoethos_autoresearch::runner",
        sweep = %sweep,
        proposals = drawn.proposals.len(),
        duplicates = drawn.duplicates.len(),
        refused = drawn.refused.len(),
        sweep_hash = %sweep_hash_of(&drawn.proposals),
        "DRY RUN — proposals drawn and stamped; nothing was executed"
    );
    stop(ctx, writer, crate::verdict::StopReason::OperatorStop)
}

// ─────────────────────────────────────────────────────────────────────────────
// The real executor
// ─────────────────────────────────────────────────────────────────────────────

pub mod streaming;

#[cfg(test)]
mod tests;
