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
//! ## A slot NAMES its proposal
//!
//! Every `slot` in this module — in the journal, in the per-slot ledger
//! directory, in the trial-returns filename, in the best-ever record and in the
//! promotion candidate — is [`crate::proposal::Proposal::slot`], **not** a
//! position in any vector. Refusals at S4 are the norm rather than the
//! exception: `Proposal::resolve` bails on every non-zero `streaming_batches`
//! draw and every non-`Start` cursor draw, both of which the proposer samples
//! uniformly. So a refused proposal keeps its row (see
//! `SearchOutcome::refused_before_running`) instead of being dropped, the rows
//! carry their slot explicitly, and the two places the judge hands back a
//! POSITION — `SweepScreen::per_slot` and `SweepScreen::champion_slot` — are
//! translated back into slots in exactly one function, `align_screen`.
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
    /// The slot that NAMES this proposal — [`crate::proposal::Proposal::slot`],
    /// not a position in any vector. The executor keys its per-slot ledger
    /// directory by it and must echo it back unchanged in
    /// [`SearchOutcome::slot`].
    pub slot: usize,
    /// The proposal's delta already applied to the session's baseline.
    pub config: &'a DiscoveryConfig,
    pub config_hash: &'a str,
    /// Where this search's trial-returns matrix must be written.
    pub trial_returns_path: std::path::PathBuf,
    /// Where this search's PROMOTION EVIDENCE must be written — the genes it
    /// selected, the feature names those genes address, and the `config_hash`
    /// stamp that says which configuration selected them.
    ///
    /// It is a path in the SESSION STORE and not in the executor's scratch
    /// directory, and that is the whole point: the scratch root is keyed by
    /// SYMBOL and is reused by every session, while the promotion candidate may
    /// be a sweep that ran days earlier and is resumed into. Evidence that lives
    /// anywhere else is evidence the single out-of-sample touch cannot rely on.
    pub promotion_evidence_path: std::path::PathBuf,
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
    /// The slot that names the PROPOSAL this outcome belongs to — echoed back
    /// from [`SearchRequest::slot`] unchanged. The runner checks it and FAILS
    /// the sweep on a mismatch: an outcome that renames itself would attribute
    /// this search's statistics, its trial matrix and possibly the one
    /// out-of-sample touch to a different configuration.
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

/// How the `error` of a row that never ran begins. Written into the journal so
/// a census reader can tell "this search failed" from "this search never
/// happened" without holding the runner's source.
pub const NOT_RUN_PREFIX: &str = "NOT RUN — ";

impl SearchOutcome {
    /// The row for a slot whose proposal could not be resolved into a runnable
    /// configuration.
    ///
    /// **It exists so `per_search` keeps one row per PROPOSAL.** The moment a
    /// refusal removes a row, every index after it names a different experiment
    /// than it did before — and refusals are the NORM here, not an edge case:
    /// [`crate::proposal::Proposal::resolve`] bails on every non-zero
    /// `streaming_batches` draw and every non-`Start` cursor draw, both of which
    /// the proposer samples uniformly. Keeping the row is what makes "position"
    /// and "identity" the same number again, and the identity is also carried
    /// explicitly in `slot` so that even a compacted vector can be re-keyed.
    ///
    /// The row is honest about itself: zero trials, an unreadable statistics
    /// report, a cost band that does not discriminate, and a named `error` that
    /// begins with [`NOT_RUN_PREFIX`]. The judge therefore screens it as a
    /// FAILURE rather than scoring a search that never happened, and the runner
    /// excludes it from the posterior credit — an arm that was never tested is
    /// not evidence about that arm.
    fn refused_before_running(slot: usize, config_hash: &str, reason: &str) -> Self {
        let detail = format!(
            "{NOT_RUN_PREFIX}the proposal was refused before a bar was read and NO SEARCH TOOK \
             PLACE, so this row carries no evidence about the configuration it names. It is kept \
             so the sweep's rows stay one-per-proposal and slot {slot} still names slot {slot}. \
             Reason: {reason}"
        );
        Self {
            slot,
            config_hash: config_hash.to_string(),
            trials_offered: 0,
            statistics: TrialStatisticsReport::unreadable(detail.clone()),
            // `discriminates: false`, so screen conjunct S3 refuses it. A row
            // that never ran must never inherit a passing band verdict.
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
            wall_ms: 0,
            error: Some(detail),
        }
    }

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

/// The schema of the per-slot promotion-evidence artifact.
pub const PROMOTION_EVIDENCE_SCHEMA: &str = "neoethos.autoresearch.promotion_evidence.v1";

/// What a search selected IN SAMPLE, persisted at S5 so the one out-of-sample
/// touch has something to evaluate.
///
/// **Written by the executor into the session store, read by the runner before
/// the window is spent.** Three things make it trustworthy rather than merely
/// present:
///
/// 1. **It is stamped.** `config_hash` is the slot's stamp as the runner handed
///    it to the executor, and the loader refuses an artifact whose stamp is not
///    the promotion candidate's. A file found at the right path is not evidence
///    that it describes the right configuration — the session store is keyed by
///    session, but a resumed session, a re-run sweep or a hand-copied directory
///    all put bytes at that path.
/// 2. **It carries the names its gene indices address.** A `Gene`'s `indices`
///    are positions into the feature list that search actually used — after
///    discovery's prefilter, and after the streaming loop's canonical remap.
///    They are NOT positions into whatever `prepare_multitimeframe_features`
///    happens to build later. Carrying the names is what lets the out-of-sample
///    evaluation project by NAME and refuse loudly when a name is absent,
///    instead of silently reading a different column and calling the answer out
///    of sample.
/// 3. **It is honest about being empty.** A search that selected nothing writes
///    no artifact, so "no evidence" and "evidence saying nothing survived" are
///    the same observable, and the promotion path refuses instead of evaluating
///    an empty portfolio and reporting its zero trades as a result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PromotionPortfolio {
    pub schema: String,
    pub sweep: SweepId,
    /// The SLOT that names the proposal — never a position in a vector.
    pub slot: usize,
    /// The stamp of the configuration that selected these genes.
    pub config_hash: String,
    /// The feature names the genes' `indices` address, in that exact order.
    pub feature_names: Vec<String>,
    /// The portfolio, exactly as the search selected it.
    pub genes: Vec<neoethos_search::genetic::Gene>,
    /// Did this search stream? Recorded because a streamed run's feature names
    /// can include working-set columns that a plain whole-vocabulary build does
    /// not contain, which is the one way the projection below can fail.
    pub streamed: bool,
    /// How many batches contributed genes.
    pub batches: usize,
}

impl PromotionPortfolio {
    /// Refuse an artifact that does not describe the configuration the touch is
    /// about to be spent on.
    ///
    /// Every check here is a refusal and none is a repair: the whole purpose of
    /// the stamp is that the answer to "is this the right evidence?" is either
    /// YES or a stop, never a guess.
    pub fn assert_names(
        &self,
        sweep: SweepId,
        slot: usize,
        config_hash: &str,
        path: &std::path::Path,
    ) -> Result<()> {
        if self.schema != PROMOTION_EVIDENCE_SCHEMA {
            bail!(
                "{} carries schema {:?} but this build reads {PROMOTION_EVIDENCE_SCHEMA}. Refusing \
                 to spend the single out-of-sample touch on an artifact whose shape this build \
                 does not know it agrees with.",
                path.display(),
                self.schema
            );
        }
        if self.sweep != sweep || self.slot != slot {
            bail!(
                "{} says it is {} slot {} but it was loaded as the evidence for {sweep} slot \
                 {slot}. Two searches are wearing one path.",
                path.display(),
                self.sweep,
                self.slot
            );
        }
        if self.config_hash != config_hash {
            bail!(
                "{} is stamped {} but the promotion candidate is stamped {config_hash}. THIS IS \
                 THE CHECK THAT CANNOT BE SKIPPED: the artifact would otherwise hand a DIFFERENT \
                 configuration's genes to the one out-of-sample evaluation, and the result would \
                 be reported under the candidate's name. Refusing; the window is NOT spent.",
                path.display(),
                self.config_hash
            );
        }
        if self.genes.is_empty() {
            bail!(
                "{} records no genes, so there is nothing to evaluate out of sample. A search that \
                 selected nothing is not a promotion candidate, and this path will NOT re-search \
                 to find one.",
                path.display()
            );
        }
        if self.feature_names.is_empty() {
            bail!(
                "{} records {} genes but no feature names, so nothing can say WHICH columns their \
                 indices address. An index without a name is a number that reads whatever happens \
                 to be in that position.",
                path.display(),
                self.genes.len()
            );
        }
        for (position, gene) in self.genes.iter().enumerate() {
            if let Some(out_of_range) = gene
                .indices
                .iter()
                .find(|index| **index >= self.feature_names.len())
            {
                bail!(
                    "{}: gene {position} addresses feature index {out_of_range} but only {} names \
                     were recorded. The artifact is internally inconsistent, so no projection of \
                     it can be trusted.",
                    path.display(),
                    self.feature_names.len()
                );
            }
        }
        Ok(())
    }
}

/// Where a slot's promotion evidence lives inside the SESSION STORE.
///
/// Deliberately NOT under `trial_returns/`: that directory is what
/// [`SessionStore::gc_trial_matrices`] empties, and the promotion candidate may
/// be a sweep whose matrices were collected long before the touch is spent.
fn promotion_evidence_path(
    store: &SessionStore,
    sweep: SweepId,
    slot: usize,
) -> std::path::PathBuf {
    store
        .sweep_dir(sweep)
        .join("promotion")
        .join(format!("slot_{slot:03}.json"))
}

/// Load and verify a promotion candidate's evidence. Called BEFORE the window is
/// journalled as spent.
fn load_promotion_portfolio(
    path: &std::path::Path,
    sweep: SweepId,
    slot: usize,
    config_hash: &str,
) -> Result<PromotionPortfolio> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "reading the promotion candidate's evidence from {}. It is written at S5 by the \
             search that selected the genes, into the SESSION STORE. Its absence means either \
             that search selected nothing, or that it ran under a build that did not write it — \
             and in both cases the out-of-sample window stays UNSPENT rather than being spent on \
             a path that cannot succeed.",
            path.display()
        )
    })?;
    let portfolio: PromotionPortfolio = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    portfolio.assert_names(sweep, slot, config_hash, path)?;
    Ok(portfolio)
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

    /// Everything this executor can refuse WITHOUT reading the out-of-sample
    /// window, asked before the window is journalled as spent.
    ///
    /// It exists because the expensive half of "the promotion path cannot
    /// succeed" was never that it failed — it was that it failed *after* the
    /// budget record was written, so the one irreplaceable window was consumed
    /// by a call that could not have returned. Anything an executor knows in
    /// advance (a capability it lacks, a frame it no longer holds, an empty
    /// portfolio) belongs here, where a refusal costs nothing.
    ///
    /// The default is `Ok(())` and it means exactly one thing: *this executor
    /// has no condition it can check in advance*. It is not a place to hide a
    /// missing check — `evaluate_oos` is still required to re-assert anything
    /// this returns, so a preflight that is skipped cannot let a bad touch
    /// through, it can only make the refusal expensive.
    fn oos_preflight(&self, portfolio: &PromotionPortfolio) -> Result<()> {
        let _ = portfolio;
        Ok(())
    }

    /// Evaluate — **never search** — a promotion candidate on the OOS window.
    ///
    /// Nothing is fitted and nothing is selected here: the candidate's genes
    /// were chosen in sample, are handed in as `portfolio`, and are simply
    /// backtested forward. The genes are a PARAMETER rather than something this
    /// call goes and finds, because the party that must prove the evidence
    /// exists is the party that decides whether the window gets spent, and that
    /// party is the runner.
    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        config: &DiscoveryConfig,
        portfolio: &PromotionPortfolio,
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

    /// It refuses HERE, before the budget record is written, so a session that
    /// reaches promotion with this executor stops with the window still clean.
    fn oos_preflight(&self, _portfolio: &PromotionPortfolio) -> Result<()> {
        bail!(
            "the refusing executor never touches the out-of-sample window, and it says so BEFORE \
             the touch is journalled as spent"
        )
    }

    fn evaluate_oos(
        &mut self,
        _sweep: SweepId,
        _slot: usize,
        _config: &DiscoveryConfig,
        _portfolio: &PromotionPortfolio,
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
        ..DiscoveryConfig::try_from_settings(settings)?
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
    // `run()` already reaches this refusal through
    // `DiscoveryConfig::try_from_settings`, but this function is public so
    // test harnesses and alternate callers can supply a config directly.
    // Gate again at the actual execution boundary: no synthetic executor may
    // turn guessed costs or returns into an autoresearch verdict.
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;

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
        let run = match run_sweep(
            &mut ctx,
            &mut writer,
            executor,
            sweep,
            SweepKind::Live,
            &drawn.proposals,
            None,
        ) {
            Ok(run) => run,
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
        let evidence = &run.evidence;

        // Refusals at S4 are routine — `Proposal::resolve` bails on every
        // non-zero `streaming_batches` and every non-`Start` cursor, both of
        // which the proposer samples — so a sweep with SOME refusals is normal
        // and each one is journalled by name. A sweep with NOTHING left is not:
        // it means the runner cannot express the space the proposer draws from,
        // which is a wiring finding and not a research result.
        if run.not_run.len() == drawn.proposals.len() {
            let (first_slot, first_reason) = run
                .not_run
                .first()
                .map(|(s, r)| (*s, r.clone()))
                .unwrap_or((0, "<unnamed>".to_string()));
            return stop(
                &mut ctx,
                &mut writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "every one of {sweep}'s {} proposals was refused before it ran, so the \
                         sweep produced no evidence at all. The first was slot {first_slot}: \
                         {first_reason}",
                        drawn.proposals.len()
                    ),
                },
            );
        }

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
            evidence,
            &ctx.thresholds,
            writer.session().n_session(),
            &null,
        );

        // The judge returns POSITIONS; every record below is keyed by the SLOT
        // that names a proposal. This is the one place the two are related, and
        // it refuses rather than guessing.
        let aligned = match align_screen(
            sweep,
            &drawn.proposals,
            evidence,
            &screen,
            &run.not_run,
        ) {
            Ok(aligned) => aligned,
            Err(err) => {
                return stop(
                    &mut ctx,
                    &mut writer,
                    crate::verdict::StopReason::InfrastructureAbort {
                        detail: format!("{sweep}: {err:#}"),
                    },
                );
            }
        };

        // ── S7 RECORD ───────────────────────────────────────────────────────
        for (slot, result) in &aligned.rows {
            writer.append(Record::Screened {
                sweep,
                slot: *slot,
                screen_result: result.clone(),
                failing_conjunct: result.failing_conjunct_label(),
            })?;
        }
        let credits =
            proposer.credits_for(&aligned.credit_proposals, &aligned.credit_screens);
        writer.append(Record::PosteriorUpdated { sweep, credits })?;

        // The champion ROW and the BEST-EVER record are two different things and
        // are no longer gated on one another. The row needs a per-period return
        // series and only feeds `pbo_session`; best-ever needs the slot, the hash
        // and the number, and feeds R2's U2 — the condition that lets the loop
        // say GOAL UNREACHABLE. Coupling them made the loop's ability to state a
        // refutation conditional on a matrix write that is non-fatal by design.
        if let (Some(config_hash), Some(e_screen_pess)) = (
            screen.champion_config_hash.clone(),
            screen.champion_e_screen_pess,
        ) {
            // A champion with no slot is a champion nobody can find again: the
            // best-ever record, the GC's keep-set and the promotion candidate
            // are all keyed by it. `unwrap_or(0)` here would silently name slot
            // 0's configuration as the session's best.
            let Some(champion_slot) = aligned.champion_slot else {
                return stop(
                    &mut ctx,
                    &mut writer,
                    crate::verdict::StopReason::InfrastructureAbort {
                        detail: format!(
                            "{sweep} produced a passing screen but named no slot for it, so \
                             nothing could say WHICH configuration the session's best-ever result \
                             belongs to."
                        ),
                    },
                );
            };
            if let Some(champion) = screen.champion.clone() {
                writer.push_champion(champion)?;
                ctx.store.write_champions(&writer.session().champions)?;
            } else if let Some(why) = &screen.champion_refusal {
                // Counted and NAMED. A sweep silently absent from the matrix
                // moves `pbo_session`, which gates every promotion.
                tracing::warn!(
                    target: "neoethos_autoresearch",
                    %sweep,
                    reason = %why,
                    "this sweep contributes NO row to the session champion matrix"
                );
            }
            writer.offer_best_ever(BestEver {
                sweep,
                slot: champion_slot,
                config_hash,
                e_screen_pess,
                n_trades: screen.champion_n_trades,
                dsr: screen.champion_dsr,
                pbo: screen.champion_pbo,
            })?;
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
                gc(&mut ctx, &mut writer, sweep)?;
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

    let pip_value_per_lot = base_config
        .try_evaluation_config(None)?
        .pip_value_per_lot;
    let costs = frozen_cost_fields(&base_config, pip_value_per_lot);
    let cost_hash = hash_named(&costs);

    // The OOS window is defined ONCE, here, and recorded in the header.
    let (search_span, oos_window, oos_bars, bars_per_trade) = executor.windows()?;

    // The judge, with its two floors DERIVED from the windows just resolved.
    // Built here and not earlier because both floors need those windows and
    // both enter `judge_hash`.
    let thresholds = session_thresholds(&base_config, search_span, oos_window);
    let judge_hash = thresholds.judge_hash();

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
        // Read off the JUDGE, not re-derived: the self-check's job is to verify
        // the floor that will actually be enforced. Two call sites of the same
        // derivation is how the startup message came to describe a bar the
        // judge did not apply.
        n_min_oos: thresholds.n_min_oos,
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

/// **The judge a session runs under.** §15 — the two trade floors are DERIVED,
/// and they are installed where they are ENFORCED.
///
/// `JudgeThresholds::frozen()` carries 30-trade placeholders. The judge applies
/// `n_min_screen` at screen conjunct S5 and `n_min_oos` at the single
/// out-of-sample touch, so deriving the floors for the startup self-check and
/// then judging with the placeholders means the line the operator reads at
/// startup — "verified: this window can hold N trades" — describes a bar
/// nothing ever applies. On a two-year OOS window at half a trade a day the
/// derived floor is around 360 against an enforced 30: a twelve-times weaker
/// bar, on the one window that cannot be re-read.
///
/// Both floors enter `judge_hash`, which the session id is built from and every
/// resume is checked against, so a session whose windows moved refuses to
/// resume rather than continuing under a floor its earlier sweeps were never
/// judged by.
fn session_thresholds(
    config: &DiscoveryConfig,
    search_span: (i64, i64),
    oos_window: OosWindow,
) -> crate::judge::JudgeThresholds {
    crate::judge::JudgeThresholds::frozen().with_derived_floors(
        ctx_n_min_screen(config, search_span, oos_window),
        ctx_n_min_oos(config, oos_window),
    )
}

/// The ONE derivation behind both trade floors (§15).
///
/// `min_trades_per_month × months_in_window`, where the months come from the
/// window's own wall-clock duration and `min_trades_per_month` from the
/// configured `min_trades_per_day`. Deriving it is the point: a fixed trade
/// count degrades differently on every dataset — it is a strict bar on a
/// two-month window and no bar at all on a two-year one.
///
/// Both floors go through here and through [`crate::judge::n_min_trades`], so
/// the in-sample bar and the out-of-sample bar MEAN the same thing measured on
/// different windows, rather than being two similar-looking formulas that drift
/// apart. The result is never zero: a floor of zero lets a candidate that
/// closed no trade clear the trade conjunct.
fn derived_trade_floor(config: &DiscoveryConfig, start_ms: i64, end_ms: i64) -> usize {
    const MS_PER_DAY: f64 = 86_400_000.0;
    let days = ((end_ms - start_ms).max(0) as f64 / MS_PER_DAY).max(1.0);
    let months = (days / 30.0).max(1.0);
    let per_month = (config.min_trades_per_day.max(0.05) * 30.0).max(1.0);
    crate::judge::n_min_trades(per_month, months)
}

/// `N_MIN_OOS` is **derived**, never a constant (§15) — from the OOS window.
fn ctx_n_min_oos(config: &DiscoveryConfig, oos_window: OosWindow) -> usize {
    derived_trade_floor(config, oos_window.start_ms, oos_window.end_ms)
}

/// `N_MIN_SCREEN` is **derived**, never a constant (§15) — from the IN-SAMPLE
/// span, which is the loaded span with the reserved OOS tail removed.
///
/// The screen never sees an OOS bar, so deriving its floor from the whole
/// loaded span would demand trades from months the screen is not allowed to
/// look at. When the window arithmetic is degenerate — an OOS start at or
/// before the span start, which the startup self-check rejects separately — the
/// span falls back to the whole loaded range rather than collapsing to a
/// one-day window and a floor of one, because a floor of one is no floor.
fn ctx_n_min_screen(
    config: &DiscoveryConfig,
    search_span: (i64, i64),
    oos_window: OosWindow,
) -> usize {
    let (span_start, span_end) = search_span;
    let in_sample_end = if oos_window.start_ms > span_start {
        oos_window.start_ms.min(span_end)
    } else {
        span_end
    };
    derived_trade_floor(config, span_start, in_sample_end)
}

// ─────────────────────────────────────────────────────────────────────────────
// A SLOT NAMES ITS PROPOSAL — it is never a position in a vector
// ─────────────────────────────────────────────────────────────────────────────
//
// The proposer emits exactly one proposal per slot `0..SWEEP_SEARCHES`, and the
// slot is carried on the proposal itself, in every journal record, in the
// per-slot ledger directory and in the trial-returns filename. Everything below
// looks a slot up BY THAT VALUE. Nothing indexes a vector with it without first
// proving the vector is dense, because the one vector that is NOT dense —
// `per_search`, which drops a row for every proposal the runner could not
// resolve — used to be indexed with a number that came from a different vector,
// and the place that mismatch landed was the single out-of-sample touch.

/// Prove a sweep's proposals are one-per-slot in draw order.
///
/// A gap here means a `ProposalDrawn` record is missing from the journal, and
/// the journal is the memory: with a gap, `session.proposals_of()` back-fills
/// the hole with a CLONE of a neighbour, so a slot that was never drawn would
/// answer with somebody else's configuration. Refused, loudly, before a bar is
/// read.
fn assert_draw_order_is_dense(
    sweep: SweepId,
    proposals: &[crate::proposal::Proposal],
) -> Result<()> {
    for (position, proposal) in proposals.iter().enumerate() {
        if proposal.slot != position {
            bail!(
                "{sweep}'s proposal at position {position} names slot {}. The draw order IS the \
                 slot: the proposer emits one proposal per slot and the journal, the ledger \
                 directory, the trial-returns matrix and the out-of-sample touch are all keyed by \
                 it. Refusing to run a sweep whose positions and slots disagree — that \
                 disagreement is invisible until it spends the one window this design exists to \
                 protect.",
                proposal.slot
            );
        }
    }
    Ok(())
}

/// The proposal a slot NAMES, cross-checked against the `config_hash` the
/// record carried.
///
/// The hash check is the belt: if the two ever disagree, two different
/// configurations are wearing one slot and the caller refuses rather than
/// running whichever one an index happened to reach.
fn proposal_named_by<'a>(
    sweep: SweepId,
    proposals: &'a [crate::proposal::Proposal],
    slot: usize,
    expected_hash: Option<&str>,
) -> Result<&'a crate::proposal::Proposal> {
    let proposal = proposals
        .iter()
        .find(|p| p.slot == slot)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{sweep} has no proposal naming slot {slot}. A slot names its proposal; it is \
                 never a position in a vector. Slots present: {:?}",
                proposals.iter().map(|p| p.slot).collect::<Vec<_>>()
            )
        })?;
    if let Some(expected) = expected_hash
        && !expected.is_empty()
        && !proposal.config_hash.is_empty()
        && proposal.config_hash != expected
    {
        bail!(
            "{sweep} slot {slot} names the proposal stamped {} but the record carried {expected}. \
             Two different configurations are wearing one slot. Refusing rather than running \
             whichever one the index reached — this check exists because a compacted vector \
             indexed by a dense number is exactly how the wrong configuration would be handed the \
             single out-of-sample touch.",
            proposal.config_hash
        );
    }
    Ok(proposal)
}

/// Refuse an outcome that renames the slot it was asked to run.
fn assert_outcome_names_its_request(
    sweep: SweepId,
    requested: usize,
    returned: usize,
) -> Result<()> {
    if requested != returned {
        bail!(
            "{sweep}: the executor was asked to run slot {requested} and returned an outcome \
             naming slot {returned}. A slot names its proposal, so an outcome that renames itself \
             would file this search's statistics, its trial-returns matrix and possibly the one \
             out-of-sample touch under a different configuration."
        );
    }
    Ok(())
}

/// The screen, re-keyed from POSITIONS in `per_search` to the SLOTS that name
/// the proposals.
///
/// The judge returns positions because it iterates the evidence it was handed;
/// the runner owns the mapping back to identity, and does it in exactly one
/// place so a future change to either side cannot reintroduce the drift.
struct AlignedScreen {
    /// `(slot, result)` for every screened row.
    rows: Vec<(usize, crate::judge::ScreenResult)>,
    /// The proposals whose searches ACTUALLY RAN, index-aligned with
    /// `credit_screens` — the posterior's input. Slots that never ran are
    /// excluded: an arm that was never tested is not evidence about that arm,
    /// and crediting it a failure would teach the loop to avoid a region it
    /// never looked at.
    credit_proposals: Vec<crate::proposal::Proposal>,
    credit_screens: Vec<crate::judge::ScreenResult>,
    /// The slot naming this sweep's champion, translated out of its position.
    champion_slot: Option<usize>,
}

fn align_screen(
    sweep: SweepId,
    proposals: &[crate::proposal::Proposal],
    evidence: &SweepEvidence,
    screen: &crate::judge::SweepScreen,
    not_run: &[(usize, String)],
) -> Result<AlignedScreen> {
    if screen.per_slot.len() != evidence.per_search.len() {
        bail!(
            "{sweep}: the screen returned {} results for {} search records. They are index-aligned \
             by contract, so a length disagreement means a result would be filed against a search \
             it did not judge.",
            screen.per_slot.len(),
            evidence.per_search.len()
        );
    }

    let mut rows = Vec::with_capacity(screen.per_slot.len());
    let mut credit_proposals = Vec::new();
    let mut credit_screens = Vec::new();
    for (position, result) in screen.per_slot.iter().enumerate() {
        let record = &evidence.per_search[position];
        let slot = record.slot;
        let proposal =
            proposal_named_by(sweep, proposals, slot, Some(record.config_hash.as_str()))?;
        rows.push((slot, result.clone()));
        if not_run.iter().any(|(s, _)| *s == slot) {
            continue;
        }
        credit_proposals.push(proposal.clone());
        credit_screens.push(result.clone());
    }

    let champion_slot = match screen.champion_slot {
        None => None,
        Some(position) => Some(
            evidence
                .per_search
                .get(position)
                .map(|r| r.slot)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{sweep}: the screen named position {position} as its champion but there \
                         are only {} search records. The champion's slot is what the best-ever \
                         record, the GC's keep-set and the promotion candidate are all keyed by.",
                        evidence.per_search.len()
                    )
                })?,
        ),
    };

    Ok(AlignedScreen {
        rows,
        credit_proposals,
        credit_screens,
        champion_slot,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// S4 / S5
// ─────────────────────────────────────────────────────────────────────────────

/// One sweep, as the runner observed it.
///
/// `SweepEvidence` is what the judge is handed; `not_run` is what only the
/// runner knows — the slots whose proposal could not be resolved into a runnable
/// configuration. Counted and NAMED, never dropped.
struct SweepRun {
    evidence: SweepEvidence,
    not_run: Vec<(usize, String)>,
}

/// Run one sweep's 100 searches. `SweepStarted` is fsynced BEFORE any work.
fn run_sweep(
    ctx: &mut LoopContext,
    writer: &mut SessionWriter,
    executor: &mut dyn SweepExecutor,
    sweep: SweepId,
    kind: SweepKind,
    proposals: &[crate::proposal::Proposal],
    permutation: Option<&crate::shuffle::FeaturePermutation>,
) -> Result<SweepRun> {
    // Before a single record is written: the positions and the slots must be the
    // same numbers, or nothing downstream can be trusted to name what it means.
    assert_draw_order_is_dense(sweep, proposals)?;

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
    let mut not_run: Vec<(usize, String)> = Vec::new();
    let mut trials_offered = 0usize;

    for proposal in proposals {
        // The slot comes from the PROPOSAL, never from the loop counter. The two
        // are equal — `assert_draw_order_is_dense` proved it above — and taking
        // it from the proposal is what keeps them equal when somebody later
        // filters this loop.
        let slot = proposal.slot;
        let config = match proposal.resolve(&ctx.base_config) {
            Ok(config) => config,
            Err(err) => {
                // A proposal that cannot be resolved is a refusal, not a
                // silently skipped slot — and NOT a dropped row either. It is
                // journalled by name AND keeps its place in `per_search`, so
                // every slot after it still names the proposal it named before.
                let reason = format!("{err:#}");
                writer.append(Record::ProposalRefused {
                    sweep,
                    slot,
                    config_hash: proposal.config_hash.clone(),
                    reason: format!("delta could not be applied: {reason}"),
                })?;
                outcomes.push(SearchOutcome::refused_before_running(
                    slot,
                    &proposal.config_hash,
                    &reason,
                ));
                not_run.push((slot, reason));
                continue;
            }
        };
        let request = SearchRequest {
            sweep,
            slot,
            config: &config,
            config_hash: &proposal.config_hash,
            trial_returns_path: ctx.store.trial_returns_path(sweep, slot),
            promotion_evidence_path: promotion_evidence_path(&ctx.store, sweep, slot),
            permutation,
        };

        let executed = match executor.execute(&request) {
            Ok(outcome) => assert_outcome_names_its_request(sweep, slot, outcome.slot)
                .map(|()| outcome),
            Err(err) => Err(err),
        };

        match executed {
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

    // The rows are one-per-proposal by construction. Checked rather than
    // asserted in a comment, and checked BEFORE the outcome record is written,
    // because "by construction" is what this defect was wearing the last time.
    if outcomes.len() != proposals.len() {
        let recovered = ctx.store.recover_partial_trials(sweep);
        let detail = format!(
            "{sweep} produced {} search records for {} proposals. Every proposal contributes \
             exactly one row — a refused one contributes a NOT-RUN row — so a short vector means a \
             slot was dropped and every slot after it now names a different experiment.",
            outcomes.len(),
            proposals.len()
        );
        // The sweep is closed rather than left open: a crash loses one sweep,
        // and so does a wiring fault.
        writer.append(Record::SweepFailed {
            sweep,
            trials_offered: recovered.max(trials_offered),
            error: detail.clone(),
        })?;
        bail!(detail);
    }

    // ── DEFLATE AGAINST N_session ───────────────────────────────────────────
    //
    // The executor deflated each report against ITS OWN trial count, because
    // that is the only number a single search can know. The honest denominator
    // is every candidate this SESSION has offered, including the sweep that has
    // just finished — a fold over the journal, and therefore not final until
    // this sweep is recorded. That is the whole reason this step lives in the
    // runner and cannot be pushed into `SweepExecutor`.
    //
    // `n_session_after` is the fold the `SweepCompleted` below will produce, and
    // it is PROVED against the fold immediately after the append rather than
    // trusted: a projection that silently disagreed with the journal would
    // reintroduce exactly the mismatch this fixes, one level down.
    let n_session_after = writer.session().n_session() + trials_offered;
    for outcome in &mut outcomes {
        let redeflated = outcome.statistics.redeflated_against(n_session_after);
        outcome.statistics = redeflated;
    }

    let per_search: Vec<SearchRecord> = outcomes.iter().map(SearchOutcome::to_record).collect();
    let statistics: Vec<TrialStatisticsReport> =
        outcomes.iter().map(|o| o.statistics.clone()).collect();

    ctx.store
        .write_sweep_artifact(sweep, "statistics.json", &statistics)?;
    ctx.store
        .write_sweep_artifact(sweep, "censuses.json", &per_search)?;

    // OUTCOME. Written AFTER the re-deflation so that the journal row, the
    // `statistics.json` artifact and the evidence the judge screens are the
    // same three numbers. `SearchRecord::dsr` is read by the shuffle control's
    // block judgement; a journal holding the un-deflated value while the judge
    // used the deflated one would make those two comparisons disagree about
    // what a DSR is.
    writer.append(Record::SweepCompleted {
        sweep,
        trials_offered,
        per_search: per_search.clone(),
        wall_ms,
    })?;

    // The projection above must BE the fold. If it is not, every report in this
    // sweep was deflated against a number that is not `n_session`, the judge's
    // N contract will refuse all of them, and the cause is here rather than in
    // the sweep. The sweep is already closed by the record above, so the session
    // keeps its history and the caller turns this into a stop WITH a verdict.
    let folded = writer.session().n_session();
    if folded != n_session_after {
        bail!(
            "{sweep}: the statistics were deflated against N = {n_session_after} but recording the \
             sweep left the session-wide fold at {folded}. The deflation denominator and the \
             journal's own count of offered trials must be the same number — a screen judged \
             against anything else is not deflated, it is flattered."
        );
    }

    let champion_rows = champion_rows(sweep, &outcomes);

    Ok(SweepRun {
        evidence: SweepEvidence {
            sweep,
            kind,
            sweep_hash,
            per_search,
            statistics,
            champion_rows,
            trials_offered,
            wall_ms,
        },
        not_run,
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

/// The champion row **every slot** offers, index-aligned with `outcomes`.
///
/// The runner MATERIALISES; it does not select. It used to do both — argmax
/// `e_screen_pess` over every non-errored search, screened or not — and the
/// judge then re-stamped that one row with the identity of the best *passing*
/// slot. Two different selection rules, one row, and the stamping hid the
/// disagreement: the session champion matrix received a REJECT's return series
/// under the winner's `config_hash`. Selection now happens once, in the judge,
/// over slots that actually passed the screen.
///
/// A row needs both a return series and a number; a search missing either
/// offers `None` rather than a half-built row.
fn champion_rows(sweep: SweepId, outcomes: &[SearchOutcome]) -> Vec<Option<ChampionRow>> {
    outcomes
        .iter()
        .map(|o| {
            if o.error.is_some() || o.champion_returns.is_empty() {
                return None;
            }
            o.e_screen_pess.map(|e| ChampionRow {
                sweep,
                config_hash: o.config_hash.clone(),
                strategy_id: o.champion_strategy_id.clone(),
                period_keys: o.champion_period_keys.clone(),
                monthly_returns: o.champion_returns.clone(),
                e_screen_pess: e,
            })
        })
        .collect()
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
    // The slot journalled is the PROPOSAL's slot, not its position in this
    // vector. They are equal — proved right here — and taking the value from the
    // proposal is what keeps the journal's key an identity rather than an
    // accident of iteration order.
    assert_draw_order_is_dense(sweep, &drawn.proposals)?;
    for proposal in &drawn.proposals {
        writer.append(Record::ProposalDrawn {
            sweep,
            slot: proposal.slot,
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
    // The control re-runs the source sweep's proposals BYTE FOR BYTE, so the
    // same proposals are refused at S4 in the same slots and the two sweeps'
    // rows stay comparable slot by slot. That comparability is why the refused
    // rows are kept rather than dropped.
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
    )?
    .evidence;

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

    // ── EVERYTHING BELOW THIS LINE HAPPENS BEFORE THE WINDOW IS SPENT ───────
    //
    // The window is spent when the touch HAPPENS, not when the runner decides
    // to attempt one. Every step that can fail without reading an
    // out-of-sample bar — reconstructing the configuration, loading the genes,
    // asking the executor whether it can evaluate at all — is taken first, and
    // each failure is a NAMED REFUSAL that stops the session with a verdict and
    // leaves `oos_spent()` false. A resume can then try again once the missing
    // thing exists, which is the entire difference between "this session has
    // nothing to say" and "this session cannot ever say anything".
    //
    // None of these refusals is an `Err` out of `promote`, because an `Err` here
    // propagates out of `run` and the session ends with NO `SessionStopped` and
    // NO `verdict.json` — the multi-hour run that produced no artifact.
    // Read out into an OWNED vector in its own statement: the session borrow
    // ends here, so every refusal below is free to take `writer` mutably and
    // stop with a verdict.
    let recorded = writer
        .session()
        .proposals_of(candidate.sweep)
        .map(<[_]>::to_vec);
    let Some(proposals) = recorded else {
        return stop(
            ctx,
            writer,
            crate::verdict::StopReason::InfrastructureAbort {
                detail: format!(
                    "the promotion candidate's sweep {} has no recorded proposals, so its \
                     configuration cannot be reconstructed for the out-of-sample evaluation. The \
                     journal is the memory; this means a ProposalDrawn record is missing. The \
                     window is NOT spent.",
                    candidate.sweep
                ),
            },
        );
    };
    // THE ONE PLACE A MISNAMED SLOT COULD NOT BE TAKEN BACK.
    //
    // `candidate.slot` NAMES a proposal; it is not an offset into this vector.
    // The lookup is by identity and it is cross-checked against the
    // `config_hash` the screen recorded, so if the two ever disagree the touch
    // is REFUSED rather than spent on whichever configuration an index reached.
    let proposal = match proposal_named_by(
        candidate.sweep,
        &proposals,
        candidate.slot,
        Some(candidate.config_hash.as_str()),
    ) {
        Ok(proposal) => proposal.clone(),
        Err(err) => {
            return stop(
                ctx,
                writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "resolving the promotion candidate for {} slot {} before spending the \
                         single out-of-sample touch: {err:#}. The window is NOT spent.",
                        candidate.sweep, candidate.slot
                    ),
                },
            );
        }
    };
    let config = match proposal.resolve(&ctx.base_config) {
        Ok(config) => config,
        Err(err) => {
            return stop(
                ctx,
                writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "the promotion candidate {} slot {} could no longer be resolved into a \
                         runnable configuration: {err:#}. The window is NOT spent.",
                        candidate.sweep, candidate.slot
                    ),
                },
            );
        }
    };

    // THE EVIDENCE IS LOADED AND STAMP-CHECKED BEFORE THE BUDGET IS SPENT.
    //
    // This is the half of the promotion path that used to be guaranteed to
    // fail: the genes were read from a file in the executor's scratch directory
    // that nothing in the workspace ever wrote, and the `OosTouchSpent` record
    // was already on disk by the time anyone found out. The evidence now lives
    // in the session store, is written at S5 by the search that selected it,
    // and carries the slot's `config_hash` so that a file at the right path
    // still has to prove it describes the right configuration.
    let evidence_path = promotion_evidence_path(&ctx.store, candidate.sweep, candidate.slot);
    let portfolio = match load_promotion_portfolio(
        &evidence_path,
        candidate.sweep,
        candidate.slot,
        &proposal.config_hash,
    ) {
        Ok(portfolio) => portfolio,
        Err(err) => {
            return stop(
                ctx,
                writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "the promotion candidate {} slot {} has no usable in-sample evidence, so \
                         the single out-of-sample touch was NOT spent and the window stays clean \
                         for a resume: {err:#}",
                        candidate.sweep, candidate.slot
                    ),
                },
            );
        }
    };

    // Anything the executor can refuse without reading a bar, refused now.
    if let Err(err) = executor.oos_preflight(&portfolio) {
        return stop(
            ctx,
            writer,
            crate::verdict::StopReason::InfrastructureAbort {
                detail: format!(
                    "the executor cannot evaluate {} slot {} out of sample and said so BEFORE the \
                     window was spent: {err:#}",
                    candidate.sweep, candidate.slot
                ),
            },
        );
    }

    // ── THE TOUCH ───────────────────────────────────────────────────────────
    //
    // The budget is spent immediately BEFORE the evaluation and after every
    // check that could have avoided it, so a crash inside the evaluation cannot
    // leave the session believing the window is still clean, and a refusal that
    // never read a bar cannot leave it believing the window is gone.
    writer.append(Record::OosTouchSpent {
        window: ctx.oos_window,
        sweep: candidate.sweep,
        candidate: proposal.config_hash.clone(),
    })?;

    let oos = match executor.evaluate_oos(
        candidate.sweep,
        candidate.slot,
        &config,
        &portfolio,
    ) {
        Ok(oos) => oos,
        Err(err) => {
            // The window IS spent — the record above is already fsynced — so
            // this stops with a verdict rather than losing the session. A
            // session that spent its window and reported nothing is the worst
            // of both outcomes.
            return stop(
                ctx,
                writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "evaluating {} slot {} on the out-of-sample window failed AFTER the touch \
                         was journalled as spent, so this session cannot try again: {err:#}",
                        candidate.sweep, candidate.slot
                    ),
                },
            );
        }
    };

    let outcome = match crate::judge::promote(
        &candidate,
        &oos,
        &ctx.thresholds,
        &ctx.goals,
        &ctx.scenario,
        writer.session(),
    ) {
        Ok(outcome) => outcome,
        Err(err) => {
            return stop(
                ctx,
                writer,
                crate::verdict::StopReason::InfrastructureAbort {
                    detail: format!(
                        "the out-of-sample evidence for {} slot {} could not be judged: {err:#}",
                        candidate.sweep, candidate.slot
                    ),
                },
            );
        }
    };

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
            // WHICH genes were evaluated, and where they are. The counts are
            // here so a reader can tell a portfolio from a single strategy
            // without opening the artifact; the artifact itself is not inlined,
            // because this file is the loop's only tradeable output and it stays
            // a description of a configuration rather than a deployable one.
            "promotion_evidence": {
                "path": evidence_path.display().to_string(),
                "config_hash": portfolio.config_hash,
                "genes": portfolio.genes.len(),
                "feature_names": portfolio.feature_names.len(),
                "streamed": portfolio.streamed,
                "batches": portfolio.batches,
            },
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
///
/// **The pass is JOURNALLED, and that is the point of the record.**
/// `AbandonCensus::gc_deleted_matrices` and `gc_bytes_reclaimed` are rendered by
/// every verdict and by `census_line` at every stop; until
/// [`Record::GarbageCollected`] existed there was no record for the fold to fold,
/// so both numbers were structurally zero while this function deleted every
/// non-kept sweep's matrices. The operator was told nothing had been deleted by a
/// session that had deleted all of it, and a `tracing::info!` that only fired
/// when `deleted > 0` was the whole of the evidence.
///
/// Every pass is recorded, including one that reclaimed nothing: *"the GC ran and
/// found nothing"* and *"the GC never ran"* are different facts, and only the
/// second is a wiring finding.
fn gc(ctx: &mut LoopContext, writer: &mut SessionWriter, after: SweepId) -> Result<()> {
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
    tracing::info!(
        target: "neoethos_autoresearch::runner",
        sweep = %after,
        kept = census.kept,
        deleted = census.deleted,
        bytes_reclaimed = census.bytes_reclaimed,
        free_bytes_after = ?census.free_bytes_after,
        rule = %census.rule,
        "sweeps_gc"
    );
    writer.append(Record::GarbageCollected {
        sweep: after,
        census,
    })
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
