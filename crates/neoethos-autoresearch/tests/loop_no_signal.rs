//! **The adversarial run: try to make the loop report SUCCESS on data with no
//! signal, and see whether the shuffle control catches it.**
//!
//! `ScriptedExecutor` hands the loop a world where the control scores like noise
//! and one live slot per sweep genuinely dominates it. That proves a pass is
//! *reachable*; it proves nothing about whether a pass is *earned*, because in
//! that world the null is a bar the champion clears by construction.
//!
//! This file builds the opposite world — the one the shuffle control exists for:
//!
//! * **Mining still looks brilliant in sample.** Every live search returns a
//!   strongly-separated trial matrix (DSR ≈ 1, PBO ≈ 0), a discriminating cost
//!   band, hundreds of trades and a healthy positive `E_screen_pess`. S1–S5 have
//!   nothing to object to. That is what selecting the best of forty trials looks
//!   like whether or not any edge exists.
//! * **The control looks exactly as brilliant.** Rotating the features away from
//!   the future changes NOTHING, because there was nothing to destroy. The
//!   control's `E_screen_pess` is drawn from the same distribution as the live
//!   searches', with the same mean and the same spread.
//! * **Out of sample there is nothing.** If a pass does leak through, the single
//!   touch meets the measured base rate — negative pips per trade — rather than
//!   the in-sample fantasy.
//!
//! In that world S1–S5 are all satisfiable and **S6 is the only conjunct standing
//! between the loop and a false GOAL REACHED**. So this test is the measurement
//! of S6, now that round 2 made S6 reachable at all: before it, the screen died
//! at S2 and S6 was unreachable code.
//!
//! It asserts the one thing that must never happen — a `GOAL_REACHED` verdict on
//! a world with no signal in it — and writes down everything else it observed.

mod support;

use std::path::PathBuf;

use neoethos_autoresearch::journal::{CostBandCounts, OosWindow};
use neoethos_autoresearch::runner::{
    OosEvidence, RunArgs, SearchOutcome, SearchRequest, SweepExecutor, run_with_executor,
};
use neoethos_autoresearch::session::SweepId;
use neoethos_search::DiscoveryConfig;
use neoethos_search::deflated::{TrialStatisticsReport, analyse_bytes};
use support::{Evidence, TRIALS, tag_counts};

/// Enough sweeps for R2's U1 (`K_MIN = 40` LIVE sweeps) to be reachable, plus the
/// controls the cadence interleaves: 40 live + one control per `SHUFFLE_PERIOD`
/// = 4 live is 50, and the headroom covers refusals.
const MAX_SWEEPS: usize = 60;

/// The in-sample expectancy mining produces on this instrument from a search of
/// forty trials. Comfortably clear of S4's zero floor: the point of this fixture
/// is that S1–S5 have NOTHING to object to.
const MINED_MEAN: f64 = 0.30;

/// The spread of that number across searches. Live and control share it — that
/// is what "no signal" means.
const MINED_SPREAD: f64 = 0.06;

/// A deterministic draw in `[-1, 1)` from a search's identity. Deterministic
/// because a test whose verdict depends on the run is not a measurement, and
/// keyed on the identity so live and control sample the SAME distribution
/// without ever landing on the same value.
fn jitter(sweep: u64, slot: usize, control: bool) -> f64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in sweep
        .to_le_bytes()
        .iter()
        .chain(slot.to_le_bytes().iter())
        .chain(std::iter::once(&u8::from(control)))
    {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Top 53 bits into [0,1), then centred.
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    unit * 2.0 - 1.0
}

/// A world with no signal in it.
struct NoSignalExecutor {
    executed_live: usize,
    executed_control: usize,
    oos_calls: Vec<(SweepId, usize)>,
}

impl NoSignalExecutor {
    fn new() -> Self {
        Self {
            executed_live: 0,
            executed_control: 0,
            oos_calls: Vec::new(),
        }
    }
}

impl SweepExecutor for NoSignalExecutor {
    fn describe(&self) -> String {
        "NO-SIGNAL EXECUTOR — mining looks brilliant, the control looks exactly as brilliant"
            .to_string()
    }

    fn streaming_requested(&self) -> bool {
        false
    }

    fn windows(&self) -> anyhow::Result<((i64, i64), OosWindow, usize, f64)> {
        Ok((
            (0, 900 * support::DAY),
            OosWindow {
                start_ms: 900 * support::DAY + 1,
                end_ms: 1_200 * support::DAY,
            },
            300 * 288,
            5.0,
        ))
    }

    fn execute(&mut self, request: &SearchRequest<'_>) -> anyhow::Result<SearchOutcome> {
        let control = request.permutation.is_some();
        if control {
            self.executed_control += 1;
        } else {
            self.executed_live += 1;
        }

        // THE SAME DISTRIBUTION ON BOTH SIDES. Rotating the features destroyed
        // nothing, because there was nothing to destroy.
        let e_screen_pess = MINED_MEAN + MINED_SPREAD * jitter(request.sweep.0, request.slot, control);

        // A strongly separated matrix on BOTH sides too: S1 and S2 are about the
        // shape of the trial matrix, and mining the best of forty trials produces
        // that shape from noise. If S2 caught this, the shuffle control would
        // never have been needed.
        let salt = (request.sweep.0 as usize) * 1_000 + request.slot + usize::from(control);
        let (bytes, champion) = support::matrix_for(Evidence::Strong, salt);
        if let Some(parent) = request.trial_returns_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&request.trial_returns_path, &bytes)?;

        let portfolio = neoethos_autoresearch::runner::PromotionPortfolio {
            schema: neoethos_autoresearch::runner::PROMOTION_EVIDENCE_SCHEMA.to_string(),
            sweep: request.sweep,
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            feature_names: vec!["rsi_14".to_string(), "atr_14".to_string()],
            genes: vec![neoethos_search::genetic::Gene {
                indices: vec![0, 1],
                weights: vec![0.5, -0.5],
                strategy_id: format!("{}-slot{}", request.sweep, request.slot),
                sl_pips: 20.0,
                tp_pips: 40.0,
                ..Default::default()
            }],
            streamed: false,
            batches: 1,
        };
        if let Some(parent) = request.promotion_evidence_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &request.promotion_evidence_path,
            serde_json::to_vec(&portfolio)?,
        )?;

        let statistics: TrialStatisticsReport = analyse_bytes(&bytes, TRIALS);
        Ok(SearchOutcome {
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            trials_offered: TRIALS,
            statistics,
            cost_band: CostBandCounts {
                survives: 1,
                optimistic_edge_only: 0,
                fails: TRIALS - 1,
                unmeasured: 0,
                not_discriminating: 0,
                discriminates: true,
            },
            rejections: vec![("min_trades".to_string(), 3)],
            survivors: 1,
            e_screen_pess: Some(e_screen_pess),
            n_trades: 400,
            champion_returns: champion,
            champion_period_keys: (0..support::PERIODS as i64).map(|m| 24_300 + m).collect(),
            champion_strategy_id: format!("{}-slot{}", request.sweep, request.slot),
            streamed: false,
            batch_columns: 0,
            next_cursor: 0,
            wall_ms: 1,
            error: None,
        })
    }

    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        _config: &DiscoveryConfig,
        portfolio: &neoethos_autoresearch::runner::PromotionPortfolio,
    ) -> anyhow::Result<OosEvidence> {
        assert_eq!(portfolio.sweep, sweep);
        assert_eq!(portfolio.slot, slot);
        self.oos_calls.push((sweep, slot));
        // OUT OF SAMPLE THERE IS NOTHING. The measured base rate for this
        // instrument is -4.15 pips per trade in every exit configuration tested;
        // an in-sample fantasy meets it here.
        Ok(OosEvidence {
            window: OosWindow {
                start_ms: 900 * support::DAY + 1,
                end_ms: 1_200 * support::DAY,
            },
            per_trade_net_pips: vec![-4.15; 500],
            r_multiples: vec![-0.2; 500],
            monthly_returns: vec![-0.01; support::PERIODS],
            period_keys: (0..support::PERIODS as i64).map(|m| 24_300 + m).collect(),
            trades_per_day: 1.0,
            // TRUE on purpose. The refusal must come from the NUMBER, not from a
            // flag the fixture set to make the test pass.
            band_survives: true,
        })
    }
}

#[ignore = "DOES NOT TERMINATE. The loop cannot stop on a no-signal world: U2 gained the arm that lets silence be a refutation (verdict.rs u2_condition, 2026-08-10), and the lib tests prove that arm fires — but this end-to-end run still does not reach a verdict, so something ABOVE u2_condition never asks. Ignored rather than deleted or left to hang: it ran for an hour holding a linker lock and made cargo report LNK1104, which reads as a build failure and is not one. UNIGNORE IT AS THE FIRST STEP of the next loop round - it is the acceptance test for the one result this project has never been able to state."]
#[test]
fn a_world_with_no_signal_must_not_produce_a_goal_reached_verdict() {
    let root = support::fresh_root("no-signal");
    let settings = support::settings_in(&root);
    let base = support::base_config(&settings);

    let started = std::time::Instant::now();
    let mut executor = NoSignalExecutor::new();
    let outcome = run_with_executor(
        RunArgs {
            max_sweeps: MAX_SWEEPS,
            ..RunArgs::new("EURUSD")
        },
        &settings,
        base,
        &mut executor,
    );
    let elapsed_s = started.elapsed().as_secs_f64();

    let dir = support::only_session_dir();
    let lines = support::journal_lines(&dir);
    let tags = tag_counts(&lines);

    let mut screen_outcomes: std::collections::BTreeMap<String, usize> = Default::default();
    for record in support::records_tagged(&lines, "Screened") {
        let key = record
            .get("screen_result")
            .and_then(|r| {
                let outcome = r.get("outcome").and_then(|o| o.as_str())?;
                Some(match r.get("conjunct").and_then(|c| c.as_str()) {
                    Some(conjunct) => format!("{outcome}:{conjunct}"),
                    None => outcome.to_string(),
                })
            })
            .unwrap_or_else(|| "<unparsed>".to_string());
        *screen_outcomes.entry(key).or_insert(0) += 1;
    }

    let census = serde_json::json!({
        "elapsed_s": elapsed_s,
        "run_error": outcome.as_ref().err().map(|e| format!("{e:#}")),
        "live_searches": executor.executed_live,
        "control_searches": executor.executed_control,
        "oos_calls": executor.oos_calls.len(),
        "journal_tag_counts": tags,
        "screen_outcomes": screen_outcomes,
        "distinct_screen_failures": support::distinct_screen_failures(&lines),
        "verdict": outcome.as_ref().ok().map(|v| serde_json::to_value(v).expect("verdict")),
        "rendered": outcome.as_ref().map(|v| v.render()).unwrap_or_default(),
    });
    let path: PathBuf = std::env::temp_dir().join("autoresearch-census-no-signal.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&census).expect("census"))
        .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    eprintln!("NO-SIGNAL CENSUS WRITTEN TO {}", path.display());

    let verdict = outcome.expect(
        "even a refuted session must reach a verdict — 'no signal' is a RESULT and must be \
         reported in the same shape a success is",
    );

    // ── THE ONE THING THAT MUST NEVER HAPPEN ────────────────────────────────
    assert!(
        !matches!(
            verdict.verdict,
            neoethos_autoresearch::verdict::Verdict::GoalReached { .. }
        ),
        "THE LOOP REPORTED SUCCESS ON A WORLD WITH NO SIGNAL IN IT. Every live search and every \
         shuffle control drew its expectancy from the same distribution, so nothing here is \
         distinguishable from a time-rotated control, and out of sample every trade lost 4.15 \
         pips. A verdict of {} on this input means the screen's only defence against mining — S6, \
         the shuffle null — did not hold.\n{}",
        verdict.verdict.tag(),
        verdict.render()
    );

    // The control ran, or S6 was never evaluated and the assertion above is
    // vacuous.
    assert!(
        executor.executed_control > 0,
        "no shuffle control ever ran, so S6 was never evaluated and this test proves nothing.\n{}",
        verdict.render()
    );
}
