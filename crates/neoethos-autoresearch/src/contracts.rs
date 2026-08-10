//! Contracts consumed, and how absence fails.
//!
//! `docs/autoresearch-loop.md` §13.
//!
//! This module exists for one purpose:
//!
//! > **An absent symbol must fail at COMPILE time, and an absent behaviour must
//! > fail at STARTUP, naming itself.**
//!
//! Never a silent no-op that makes the loop appear to run for days while doing
//! nothing. That failure mode — a run that looks like work and produces a
//! confident number — is the specific thing this whole crate is built to make
//! impossible, and it starts here.

// ─────────────────────────────────────────────────────────────────────────────
// §13.1 — COMPILE-TIME. One `use` of every symbol the loop depends on.
//
// These imports are deliberately unused. They are the contract: if
// `neoethos-search` loses one of them, this crate does not compile and the
// error names the missing symbol on this line. That is the required behaviour,
// and it is why `#[allow(unused_imports)]` is scoped to this module and to
// nothing else.
//
// The two APIs landing in parallel with this crate are the streaming inner loop
// (`run_streaming_working_set` + `StreamingPlan` + `StreamingRunOutcome`) and
// the DSR/PBO reader (`analyse_matrix` + `pbo_cscv` + `DecodedTrialMatrix`).
// If either is absent the build stops here rather than at a call site three
// modules away.
// ─────────────────────────────────────────────────────────────────────────────
#[allow(unused_imports)]
mod required_symbols {
    // The streaming inner loop, with the Gene.expectancy early-reject predicate
    // (booked AFTER spread, commission and swap).
    use neoethos_search::orchestration::{
        BatchSearchResult, StreamingPlan, StreamingRunOutcome, run_streaming_working_set,
    };
    // Canonical indices — what makes a cross-batch portfolio self-describing.
    use neoethos_search::batch_ledger::{
        BatchLedgerEntry, BatchOutcome, CanonicalFeatureIndex, CanonicalSurvivor,
        StreamingRunLedger,
    };
    // The search configuration, the search-side refusals, and the cost band.
    use neoethos_search::discovery::{
        CostBandCensus, CostBandVerdict, DiscoveryConfig, DiscoveryMode, TargetProfile,
        TargetProfileRejection, cost_band_discriminates,
    };
    // Run identity: two sweeps with the same hash are the same experiment.
    use neoethos_search::run_identity::{
        MEASURED_TRAILING_PAYOFF_CEILING, PayoffCeilingInputs, ResolvedConfigStamp,
        assert_payoff_floor_reachable, max_achievable_payoff,
    };
    // The DSR / PBO reader.
    use neoethos_search::deflated::{
        DecodedTrialMatrix, DecodedTrialRow, DeflatedSharpeReport, PboReport,
        TrialStatisticsReport, analyse_matrix, analyse_run, deflated_sharpe, pbo_cscv,
    };
    // The per-trial return matrix the judge reads.
    use neoethos_search::trial_returns::{
        TRIAL_RETURNS_MAGIC, TrialReturnsManifest, load_manifest, month_keys_spanning,
        period_returns,
    };
    // The goal functional (axis-B variant B5 and the promotion inequality).
    use neoethos_search::goal_report::{DEFAULT_RISK_LEVELS, GoalReport, RiskOutcome, build_report};
    use neoethos_search::genetic::Gene;
    // The operator's config and the prop-firm baselines.
    use neoethos_core::config::{PropFirmGateConfig, Settings};
    use neoethos_core::domain::prop_firm::{PropFirmConstraints, PropFirmPreset};
}

// ─────────────────────────────────────────────────────────────────────────────
// THE CONSUMED CONTRACT, ACROSS THE MODULE BOUNDARY
//
// Three builders wrote this crate concurrently. The loop (session + state
// machine) consumes the following symbols from the proposer's and the judge's
// modules. They are listed here, in one place, so that the seam is reviewable
// and so a mismatch surfaces as a named compile error rather than as a
// half-integrated loop.
//
// Three types cross the boundary INSIDE serialised journal records, so they
// carry a hard derive requirement: `Clone + PartialEq + Debug + Serialize +
// Deserialize`. They are `proposal::Proposal`, `judge::ScreenResult`,
// `verdict::SessionVerdict`, and `shuffle::ControlKind` — the last also needs
// `Eq`, because `journal::SweepKind` derives it.
//
// FROM THE PROPOSER (`space` / `proposal` / `proposer` / `objective`):
//
//   space::FROZEN_FIELDS: &[&str]
//   space::SpaceDescription                        // Serialize + Clone + Debug
//   space::SearchConfigDelta                       // Serialize/Deserialize + Clone + PartialEq + Debug
//   objective::ObjectiveVariant                    // + ::all() -> &'static [Self], ::label(&self) -> &'static str
//   objective::RefusalLevels                       // Serialize/Deserialize + Clone + PartialEq + Debug
//   proposal::Proposal { parent, axis_a, axis_b, refusals, replicate_seed, origin, config_hash }
//   proposal::Proposal::coverage_levels(&self) -> Vec<(&'static str, String)>
//        — ("population", "4096"), …, ("axis_b", "B3_CostElastic"). Drives U4.
//   proposal::Proposal::resolve(&self, base: &DiscoveryConfig) -> anyhow::Result<DiscoveryConfig>
//        — the ONE call the runner makes to turn a proposal into a runnable config.
//   proposal::ProposalOrigin                       // Posterior | ExplorationFloor | External { note }; Debug
//   proposer::Proposer::restore(session: &Session, session_seed: u64, base: &DiscoveryConfig)
//                                                   -> anyhow::Result<Proposer>
//   proposer::Proposer::draw_sweep(&mut self, sweep: SweepId, session: &Session)
//                                                   -> anyhow::Result<DrawnSweep>
//   proposer::Proposer::credits_for(&self, &[Proposal], &[judge::ScreenResult])
//                                                   -> Vec<journal::PosteriorCredit>
//        — binary reward, ONE success per level per sweep (§5.3 mechanism 3).
//   proposer::Proposer::reload(&mut self, session: &Session)
//        — re-read the folded posterior after a PosteriorRetracted.
//   proposer::DrawnSweep { proposals: Vec<Proposal>, duplicates: Vec<DuplicateDraw>,
//                          refused: Vec<RefusedDraw>, trust_region_reverted: usize,
//                          exhausted: bool }
//   proposer::DuplicateDraw { slot: usize, config_hash: String, first_seen_sweep: SweepId }
//   proposer::RefusedDraw   { slot: usize, config_hash: String, reason: impl Display }
//
// FROM THE JUDGE (`judge` / `shuffle` / `verdict`):
//
//   judge::JudgeThresholds::frozen() -> Self
//   judge::JudgeThresholds::judge_hash(&self) -> String
//   judge::JudgeThresholds::named_values(&self) -> journal::NamedValues
//   judge::ScreenResult                            // Passed | Failed { conjunct } | Unavailable { reason }
//   judge::ScreenResult::failing_conjunct_index(&self) -> Option<usize>   // 0..7, for the census array
//   judge::ScreenResult::failing_conjunct_label(&self) -> Option<String>
//   judge::screen_sweep(&session::SweepEvidence, &JudgeThresholds, n_session: usize,
//                       &shuffle::ShuffleNull) -> judge::SweepScreen
//   judge::SweepScreen { per_slot: Vec<ScreenResult>, champion: Option<session::ChampionRow>,
//                        champion_slot: Option<usize>, champion_n_trades: usize,
//                        champion_dsr: Option<f64>, champion_pbo: Option<f64> }
//   judge::promote(&verdict::PromotionCandidate, &runner::OosEvidence, &JudgeThresholds,
//                  &GoalSet, &goals::Scenario, &Session) -> anyhow::Result<PromotionOutcome>
//   judge::PromotionOutcome { promoted: bool, failing_conjunct: Option<String>,
//                             goal_metric: f64, goal_bar: f64,
//                             frontier: Vec<journal::RiskOutcomeRecord> }
//   shuffle::ControlKind                           // CircularRotation | ColumnPermutation
//   shuffle::ShuffleNull::from_session(&Session) -> ShuffleNull
//   shuffle::FeaturePermutation::draw(ControlKind, &Session, BlockId) -> FeaturePermutation
//   shuffle::FeaturePermutation::{kind() -> ControlKind, tau() -> f64,
//                                 apply(&self, &mut FeatureFrame) -> anyhow::Result<()>}
//   shuffle::best_sweep_of_block(&Session, BlockId) -> Option<SweepId>
//   shuffle::judge_block(&Session, BlockId, control_best: Option<f64>, control_dsr: Option<f64>)
//                                                   -> shuffle::BlockJudgement
//   shuffle::BlockJudgement { delta_expectancy, p_block, dsr_gap, indistinguishable }
//   shuffle::ShuffleSummary                        // rendered under every headline number
//   verdict::SessionVerdict, verdict::Verdict, verdict::InconclusiveReason
//   verdict::PromotionCandidate { sweep: SweepId, slot: usize }
//   verdict::decide(&Session, &GoalSet, &JudgeThresholds) -> verdict::Decision
//   verdict::Decision                              // Continue | Promote(PromotionCandidate) | Stop(StopReason)
//   verdict::StopReason                            // GoalReached(PromotionCandidate)
//                                                  // | PromotionRefused { sweep, failing_conjunct }
//                                                  // | ProposerExhausted
//                                                  // | BudgetExhausted { sweeps_run, max_sweeps }
//                                                  // | WallClock { hours } | OperatorStop
//                                                  // | InfrastructureAbort { detail: String }
//        — EVERY abort path in the runner routes through StopReason so that a failed
//          session still writes verdict.json in the same shape as a success. An
//          abort that produces no artifact is the multi-hour run that ended with
//          nothing, which this repository has actually lived through.
//   verdict::VerdictContext { session_id: String, symbol: String, cost_hash: String,
//                             judge_hash: String, oos_window: journal::OosWindow }
//   verdict::build(&Session, &GoalSet, &goals::Scenario, &JudgeThresholds, StopReason,
//                  VerdictContext) -> SessionVerdict
//
// The loop supplies the EVIDENCE (`runner::SearchOutcome`, `runner::OosEvidence`,
// `session::SweepEvidence`, `session::ChampionRow`, `session::CoverageCounters`,
// `session::PosteriorLedger`, `session::AbandonCensus`) because it owns S5
// COLLECT and the fold; the judge supplies the VERDICTS because it owns the
// statistics. Neither side reaches across.
// ─────────────────────────────────────────────────────────────────────────────

use crate::goals::{GoalRefusal, GoalSet, MirrorDisagreement};
use neoethos_search::discovery::cost_band_discriminates;
use neoethos_search::run_identity::{PayoffCeilingInputs, assert_payoff_floor_reachable};
use std::fmt;

/// Everything the startup self-check needs, resolved. Passed by reference so
/// the check is pure and testable without a process-wide install.
#[derive(Debug, Clone)]
pub struct ResolvedInputs<'a> {
    pub goals: &'a GoalSet,
    /// `DiscoveryConfig::cost_band_pips` — `(optimistic, pessimistic)`.
    pub cost_band_pips: Option<(f64, f64)>,
    /// The round-trip cost the run already charges, in pips:
    /// `spread + commission / pip_value_per_lot`.
    pub baseline_cost_pips: f64,
    /// The BASELINE configuration's payoff floor and its resolved geometry.
    pub baseline_payoff_floor: f64,
    pub baseline_payoff_inputs: PayoffCeilingInputs,
    /// The search window, in epoch ms, that sweeps are allowed to load.
    pub search_span_ms: (i64, i64),
    /// The out-of-sample window, in epoch ms. Defined once, touched once.
    pub oos_window_ms: (i64, i64),
    /// Bars inside the OOS window.
    pub oos_bars: usize,
    /// The derived minimum OOS trade count (§15: derived from
    /// `min_trades_per_month × months_in_window`, never a constant).
    pub n_min_oos: usize,
    /// Bars per trade the derivation assumed, so the message can show its work.
    pub oos_bars_per_expected_trade: f64,
    /// The search-side mirror of the risky goal
    /// (`DiscoveryConfig::risky_start_balance` / `risky_target_balance` /
    /// `risky_horizon_days`).
    pub mirror_risky: (f64, f64, f64),
}

/// A named, aborting startup failure. Every variant carries the numbers a
/// reader needs to act, because *"the loop refused to start"* with no number is
/// the silent drop again, one level up.
#[derive(Debug, Clone, PartialEq)]
pub enum MissingCapability {
    /// §13.2.1 — a goal refusal (G1–G6).
    Goals(GoalRefusal),
    /// §1.1 — the search-side mirror disagreed with the `system.*` authority.
    GoalMirror(MirrorDisagreement),
    /// §13.2.2 / §7.3 — the cost band cannot tell anything apart from the
    /// baseline it is measured against.
    CostBandCannotDiscriminate {
        baseline_cost_pips: f64,
        optimistic_edge: Option<f64>,
        pessimistic_edge: Option<f64>,
    },
    /// §13.2.3 — the OOS window is empty, too short, or overlaps the span the
    /// sweeps will load.
    OosWindowUnusable { detail: String },
    /// §13.2.4 — the BASELINE configuration's payoff floor is unreachable under
    /// its own geometry. The space cannot express the question.
    BaselineFloorUnreachable { floor: f64, detail: String },
    /// §13.2.5 — the FIRST live sweep produced no readable trial-returns
    /// matrix. On sweep 1 that means the wiring is absent, not that the
    /// configuration was bad.
    FirstSweepUnreadable { reason: String },
    /// §13.2.6 — streaming was requested but no sweep in block 1 ever sized a
    /// batch. The loop is not sweeping and would be running a single-pass
    /// search under a streaming label.
    StreamingRequestedButNeverStreamed { sweeps_in_block: usize },
}

impl fmt::Display for MissingCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Goals(refusal) => write!(f, "{refusal}"),
            Self::GoalMirror(disagreement) => write!(f, "{disagreement}"),
            Self::CostBandCannotDiscriminate {
                baseline_cost_pips,
                optimistic_edge,
                pessimistic_edge,
            } => write!(
                f,
                "STARTUP ABORT — the cost band cannot discriminate. The run already charges \
                 {baseline_cost_pips} pips round-trip, against band edges {optimistic_edge:?} / \
                 {pessimistic_edge:?}. Cost is monotone, so a candidate that cleared the baseline \
                 clears any cheaper cost BY CONSTRUCTION: every survivor would be recorded as \
                 SurvivesBand and the census would read clean on every run — which is worse than \
                 no census, because a reader takes it as evidence. Objective variant B3 \
                 (cost-elastic) and screen conjunct S3 both depend on this band, so the session \
                 refuses to start. The band's PESSIMISTIC edge must exceed the baseline \
                 round-trip cost, set from the broker's measured worst-case spread plus \
                 commission and not from a guess. The loop does not make this change: see \
                 docs/autoresearch-loop.md §14.4."
            ),
            Self::OosWindowUnusable { detail } => write!(
                f,
                "STARTUP ABORT — the out-of-sample window is unusable: {detail}. The OOS window \
                 is defined ONCE and touched ONCE (docs/autoresearch-loop.md §8.2); a window that \
                 cannot carry a verdict makes every promotion unfalsifiable, and a leaked OOS \
                 window cannot be un-leaked."
            ),
            Self::BaselineFloorUnreachable { floor, detail } => write!(
                f,
                "STARTUP ABORT — the BASELINE configuration's payoff floor ({floor}) is \
                 unreachable under its own resolved geometry, so the space cannot express the \
                 question the loop is being asked. Every sweep's answer would be fixed before a \
                 bar was read. {detail}"
            ),
            Self::FirstSweepUnreadable { reason } => write!(
                f,
                "ABORT — the FIRST live sweep produced no readable trial-returns matrix \
                 ({reason}). On sweep 1 that means the DSR/PBO wiring is ABSENT, not that the \
                 configuration was bad: without it the judge has no statistic and every later \
                 sweep would be screened by a conjunction that can never be satisfied. On any \
                 LATER sweep this is that sweep's own failure, counted and named \
                 (docs/autoresearch-loop.md §13.2.5)."
            ),
            Self::StreamingRequestedButNeverStreamed { sweeps_in_block } => write!(
                f,
                "ABORT — streaming was requested but batch_columns == 0 on all {sweeps_in_block} \
                 sweeps of block 1: no working set was ever sized. The loop would be running a \
                 single-pass whole-vocabulary search under a streaming label, and every sweep \
                 would compute the same columns — a sweep it never performed \
                 (docs/autoresearch-loop.md §13.2.6)."
            ),
        }
    }
}

impl std::error::Error for MissingCapability {}

/// §13.2 — the startup self-check. Each failure **aborts the session**.
///
/// Checks 5 and 6 cannot run here because they are statements about the first
/// sweep and the first block; they are [`assert_first_sweep_readable`] and
/// [`assert_block_one_streamed`], called by the runner at the point where they
/// become answerable. They are listed in the same enum so the abort reads the
/// same way whenever it fires.
pub fn startup_selfcheck(cfg: &ResolvedInputs<'_>) -> Result<(), MissingCapability> {
    // 1. The goals already loaded (G1–G6 fired inside `GoalSet::load`), plus
    //    the mirror agreement, which is a goal check that needs the search
    //    config and therefore could not live inside `GoalSet::load`.
    cfg.goals
        .assert_mirror_agrees(cfg.mirror_risky.0, cfg.mirror_risky.1, cfg.mirror_risky.2)
        .map_err(MissingCapability::GoalMirror)?;

    // 2. A band that cannot discriminate makes every census read clean.
    if !cost_band_discriminates(cfg.cost_band_pips, cfg.baseline_cost_pips) {
        return Err(MissingCapability::CostBandCannotDiscriminate {
            baseline_cost_pips: cfg.baseline_cost_pips,
            optimistic_edge: cfg.cost_band_pips.map(|(o, _)| o),
            pessimistic_edge: cfg.cost_band_pips.map(|(_, p)| p),
        });
    }

    // 3. The OOS window: non-empty, long enough, and disjoint from the span the
    //    sweeps will load. All three, in one place, because a window that fails
    //    any of them is not a window.
    let (oos_start, oos_end) = cfg.oos_window_ms;
    let (span_start, span_end) = cfg.search_span_ms;
    if oos_end <= oos_start {
        return Err(MissingCapability::OosWindowUnusable {
            detail: format!("the window is empty ({oos_start} .. {oos_end})"),
        });
    }
    if span_end > oos_start {
        return Err(MissingCapability::OosWindowUnusable {
            detail: format!(
                "the loaded search span ends at {span_end}, which is INSIDE the OOS window \
                 starting at {oos_start} — the search would fit on data the verdict is supposed \
                 to test it against (search span {span_start} .. {span_end})"
            ),
        });
    }
    if cfg.oos_bars == 0 {
        return Err(MissingCapability::OosWindowUnusable {
            detail: "the window contains no bars".to_string(),
        });
    }
    // "Long enough" is DERIVED, never a constant: a fixed bar count means one
    // thing on M5 and something else entirely on H1.
    let expected_oos_trades = if cfg.oos_bars_per_expected_trade > 0.0 {
        cfg.oos_bars as f64 / cfg.oos_bars_per_expected_trade
    } else {
        0.0
    };
    if expected_oos_trades < cfg.n_min_oos as f64 {
        return Err(MissingCapability::OosWindowUnusable {
            detail: format!(
                "the window holds {} bars, which at {} bars per expected trade is about {:.0} \
                 trades — below the derived floor of {} (min_trades_per_month x months in \
                 window). A promotion decided on fewer trades than that is a coin flip with a \
                 confidence interval attached",
                cfg.oos_bars, cfg.oos_bars_per_expected_trade, expected_oos_trades, cfg.n_min_oos
            ),
        });
    }

    // 4. The baseline configuration must be able to EXPRESS the question.
    if let Err(err) =
        assert_payoff_floor_reachable(cfg.baseline_payoff_floor, &cfg.baseline_payoff_inputs)
    {
        return Err(MissingCapability::BaselineFloorUnreachable {
            floor: cfg.baseline_payoff_floor,
            detail: format!("{err:#}"),
        });
    }

    Ok(())
}

/// §13.2.5 — deferred self-check, answerable only after the first live sweep.
///
/// `readable` is false when the sweep's `TrialStatisticsReport` came back
/// unreadable for every one of its searches. On sweep 1 that is the ABSENT
/// WIRING, not a bad configuration.
pub fn assert_first_sweep_readable(
    readable_searches: usize,
    reason: impl Into<String>,
) -> Result<(), MissingCapability> {
    if readable_searches == 0 {
        return Err(MissingCapability::FirstSweepUnreadable {
            reason: reason.into(),
        });
    }
    Ok(())
}

/// §13.2.6 — deferred self-check, answerable only at the end of block 1.
///
/// Only meaningful when streaming was actually requested: a non-streaming
/// session is not failing at anything by not streaming.
pub fn assert_block_one_streamed(
    streaming_requested: bool,
    sweeps_in_block: usize,
    sweeps_that_streamed: usize,
) -> Result<(), MissingCapability> {
    if streaming_requested && sweeps_in_block > 0 && sweeps_that_streamed == 0 {
        return Err(MissingCapability::StreamingRequestedButNeverStreamed { sweeps_in_block });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goals::GoalSet;
    use neoethos_core::config::Settings;

    fn goals() -> GoalSet {
        let mut s = Settings::default();
        s.system.trading_mode = "risky".to_string();
        s.system.risky_start_balance_usd = 100.0;
        s.system.risky_target_balance_usd = 50_000.0;
        s.system.risky_horizon_days = 180;
        GoalSet::load(&s).unwrap()
    }

    fn payoff_inputs() -> PayoffCeilingInputs {
        // A geometry that can reach a floor of 1.0 without a trail: widest TP
        // 40 pips against tightest SL 10, cost 1 pip.
        PayoffCeilingInputs {
            sl_min_pips: 10.0,
            sl_max_pips: 40.0,
            tp_min_pips: 10.0,
            tp_max_pips: 40.0,
            initializer_rr_max: 4.0,
            initializer_rr_min: 1.0,
            atr_pips: None,
            cost_pips_round_trip: 1.0,
            trailing_enabled: false,
            trailing_be_trigger_r: 0.0,
            trailing_give_back_r: 0.0,
            trailing_min_lock_pips: 0.0,
        }
    }

    fn inputs(goals: &GoalSet) -> ResolvedInputs<'_> {
        ResolvedInputs {
            goals,
            // Pessimistic edge ABOVE the baseline: the band can discriminate.
            cost_band_pips: Some((2.0, 5.0)),
            baseline_cost_pips: 3.4,
            baseline_payoff_floor: 1.0,
            baseline_payoff_inputs: payoff_inputs(),
            search_span_ms: (0, 800),
            oos_window_ms: (800, 1_000),
            oos_bars: 20_000,
            n_min_oos: 30,
            oos_bars_per_expected_trade: 100.0,
            mirror_risky: (100.0, 50_000.0, 180.0),
        }
    }

    #[test]
    fn a_healthy_configuration_passes() {
        let g = goals();
        startup_selfcheck(&inputs(&g)).expect("this configuration is startable");
    }

    #[test]
    fn the_shipped_cost_band_is_refused_naming_all_three_numbers() {
        // MEASURED at the shipped configuration: baseline 3.4 pips against band
        // edges 1.6 / 2.4 — BOTH cheaper than the cost already charged.
        let g = goals();
        let mut cfg = inputs(&g);
        cfg.cost_band_pips = Some((1.6, 2.4));
        let err = startup_selfcheck(&cfg).expect_err("a band below the baseline must abort");
        let rendered = err.to_string();
        assert!(rendered.contains("3.4"));
        assert!(rendered.contains("1.6"));
        assert!(rendered.contains("2.4"));
        assert!(rendered.contains("§14.4"));
    }

    #[test]
    fn an_absent_cost_band_is_refused() {
        let g = goals();
        let mut cfg = inputs(&g);
        cfg.cost_band_pips = None;
        assert!(matches!(
            startup_selfcheck(&cfg),
            Err(MissingCapability::CostBandCannotDiscriminate { .. })
        ));
    }

    #[test]
    fn a_search_span_reaching_into_the_oos_window_aborts() {
        // A leaked OOS window cannot be un-leaked, so this is an abort and
        // never a warning.
        let g = goals();
        let mut cfg = inputs(&g);
        cfg.search_span_ms = (0, 900);
        let err = startup_selfcheck(&cfg).expect_err("an overlap must abort");
        assert!(err.to_string().contains("INSIDE the OOS window"));
    }

    #[test]
    fn an_oos_window_too_short_to_carry_a_verdict_aborts() {
        let g = goals();
        let mut cfg = inputs(&g);
        cfg.oos_bars = 500; // 5 expected trades against a floor of 30
        let err = startup_selfcheck(&cfg).expect_err("too few OOS trades must abort");
        assert!(err.to_string().contains("coin flip"));
    }

    #[test]
    fn a_mirror_disagreement_aborts() {
        let g = goals();
        let mut cfg = inputs(&g);
        cfg.mirror_risky = (100.0, 500.0, 180.0);
        assert!(matches!(
            startup_selfcheck(&cfg),
            Err(MissingCapability::GoalMirror(_))
        ));
    }

    #[test]
    fn an_unreachable_baseline_floor_aborts() {
        let g = goals();
        let mut cfg = inputs(&g);
        // A floor of 20.0 against a 40:10 geometry is arithmetically impossible.
        cfg.baseline_payoff_floor = 20.0;
        assert!(matches!(
            startup_selfcheck(&cfg),
            Err(MissingCapability::BaselineFloorUnreachable { .. })
        ));
    }

    #[test]
    fn the_first_sweep_being_unreadable_is_an_abort_and_a_later_one_is_not() {
        assert!(assert_first_sweep_readable(0, "no manifest").is_err());
        assert!(assert_first_sweep_readable(1, "no manifest").is_ok());
    }

    #[test]
    fn not_streaming_is_only_a_failure_when_streaming_was_asked_for() {
        assert!(assert_block_one_streamed(false, 4, 0).is_ok());
        assert!(assert_block_one_streamed(true, 4, 0).is_err());
        assert!(assert_block_one_streamed(true, 4, 1).is_ok());
    }
}
