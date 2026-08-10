//! **The card-free executor the port was declared for.**
//!
//! `runner::SweepExecutor`'s own doc says it exists so that the loop — *"the
//! part that must be provably resumable and provably unable to reach a broker —
//! can be exercised end to end without a GPU, a dataset or a feature cube. A
//! state machine that can only be tested by running a real sweep is a state
//! machine whose crash-recovery path is tested by hoping."*
//!
//! That sentence was true and unused. The only two implementations in the
//! workspace were `RefusingExecutor`, which refuses by design, and
//! `StreamingSweepExecutor`, which needs a card, a dataset and a feature cube —
//! so **nothing had ever driven the state machine to a verdict**, and the
//! defects that survived two adversarial reviews were all in the stretch between
//! "a sweep completes" and "a verdict is written".
//!
//! This module is that missing third implementation, landed as a fixture rather
//! than left in a scratch directory. It lives under `tests/` and not under
//! `src/` on purpose: it FABRICATES statistics, and a fabricator that ships in
//! the library is one `use` away from being reachable from a real run.
//!
//! ## What it is honest about
//!
//! * It writes a **real, decodable trial-returns matrix** to the path the runner
//!   hands it — `neoethos_search::trial_returns::encode`, the same encoder the
//!   real search uses. A fixture that wrote junk bytes would pass a runner that
//!   never re-reads the matrix and fail one that does, which makes it a test of
//!   the fixture rather than of the loop.
//! * A **control** sweep (the request carries a permutation) scores like noise.
//!   That is the whole point of the shuffle control: rotating the features
//!   destroys the relationship between them and the future, so a control that
//!   returned the live sweep's numbers would make S6 unfalsifiable and the null
//!   meaningless.
//! * A refused search, a failing search and a passing search are all
//!   representable, and the failures are spread across S2–S5 so the abandon
//!   census has something to count.
//!
//! ## What it does NOT do
//!
//! It never reads a bar, never opens a dataset, never touches a broker and never
//! writes outside the session store and the path the runner names.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use neoethos_autoresearch::journal::{CostBandCounts, OosWindow};
use neoethos_autoresearch::runner::{
    OosEvidence, SearchOutcome, SearchRequest, SweepExecutor,
};
use neoethos_autoresearch::session::{SessionStore, SweepId};
use neoethos_core::config::Settings;
use neoethos_search::DiscoveryConfig;
use neoethos_search::deflated::{TrialStatisticsReport, analyse_bytes};
use neoethos_search::trial_returns::{TrialReturnMatrix, TrialReturnRow, encode};

pub const DAY: i64 = 86_400_000;

// ─────────────────────────────────────────────────────────────────────────────
// The matrix
//
// The shapes below are not arbitrary — every one of them is the smallest value
// that clears a REFUSAL THRESHOLD named in `neoethos_search::deflated`, because
// a fixture one row under a floor tests the floor and not the loop.
// ─────────────────────────────────────────────────────────────────────────────

/// `MIN_PERIODS_FOR_DSR = 24`, and `MIN_PERIODS_FOR_PBO = 16`. 24 is therefore
/// the shortest grid on which BOTH statistics are computable — and it is also
/// what pins CSCV to 12 groups of 2, the cheapest admissible split, which
/// matters because this runs a few hundred times inside one debug-build test.
pub const PERIODS: usize = 24;

/// `MIN_TRIALS_FOR_DSR = MIN_TRIALS_FOR_PBO = 10`, but ten is not enough for the
/// DSR to be PASSABLE, which is a different question and the one that matters
/// here.
///
/// The selection adjustment is `SR* = sd(SR across trials) · k(N)` with
/// `k(N) ≈ 3.5 … 4.9` over any N this test will reach, and a champion standing
/// `e` above the pack contributes about `e/√T` to that standard deviation. So
/// `SR* ≈ k·e/√T`, and the champion can only clear its own expected maximum
/// when `√T > k`, i.e. **T > k² ≈ 24**. At the ten-row floor no champion can
/// ever pass S2b however good it is — the fixture would be testing arithmetic.
/// Forty gives `SR* ≈ 0.67·e` and a comfortable positive excess.
pub const TRIALS: usize = 40;

/// First month key of the grid. Any ascending run works; the judge aligns
/// champion rows by key.
const FIRST_MONTH_KEY: i64 = 24_300;

/// What kind of evidence a search produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// One row dominates the pack by enough to clear the deflated Sharpe's own
    /// expected maximum. DSR ≈ 1, PBO ≈ 0.
    Strong,
    /// Every row scores like the others. The best row does NOT beat what a
    /// search of this size produces from noise, so S2b refuses it — which is the
    /// honest verdict on a matrix of noise, not a contrived failure.
    Noise,
}

fn month_keys() -> Vec<i64> {
    (0..PERIODS as i64).map(|m| FIRST_MONTH_KEY + m).collect()
}

/// The champion's series: alternating, so its skew is 0 and its kurtosis is 1
/// and the DSR's variance term is exactly 1 — no accidental refusal from a
/// degenerate fourth moment.
fn champion_returns(lift: f64) -> Vec<f64> {
    (0..PERIODS)
        .map(|p| if p % 2 == 0 { 0.03 * lift } else { 0.02 * lift })
        .collect()
}

/// A pack row: the same alternating shape with a small drift, so every row has a
/// computable, DISTINCT Sharpe. Distinct matters for CSCV — identical rows make
/// the rank denominator meaningless and the statistic refuses.
fn pack_returns(index: usize, drift: f64, offset: f64) -> Vec<f64> {
    (0..PERIODS)
        .map(|p| {
            let swing = if p % 2 == 0 { 0.01 } else { -0.01 };
            swing + drift * index as f64 + offset
        })
        .collect()
}

/// The pack a NOISE matrix is made of: Sharpes spread symmetrically about zero,
/// so the best of them sits well inside `SR*` and S2b refuses it by name. That
/// is the honest verdict on a matrix of noise, not a contrived failure.
fn noise_row(index: usize) -> Vec<f64> {
    pack_returns(index, 0.000_5, -0.000_5 * (TRIALS as f64 / 2.0))
}

/// Build one search's matrix and return `(encoded bytes, champion series)`.
///
/// `salt` varies the strategy ids so two searches never write byte-identical
/// files, which would hide a path mix-up in the store.
pub fn matrix_for(evidence: Evidence, salt: usize) -> (Vec<u8>, Vec<f64>) {
    let lift = 1.0 + 0.01 * (salt % 7) as f64;
    let mut rows: Vec<TrialReturnRow> = Vec::with_capacity(TRIALS);

    let champion = match evidence {
        Evidence::Strong => champion_returns(lift),
        // A "champion" that is simply the top of the pack: its Sharpe is inside
        // the spread the search itself generates.
        Evidence::Noise => noise_row(TRIALS - 1),
    };

    match evidence {
        Evidence::Strong => {
            rows.push(TrialReturnRow {
                candidate_index: 0,
                strategy_id: format!("champ-{salt}"),
                returns: champion.clone(),
                trades_outside_grid: 0,
            });
            for i in 1..TRIALS {
                rows.push(TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("pack-{salt}-{i}"),
                    returns: pack_returns(i, 0.000_1, 0.0),
                    trades_outside_grid: 0,
                });
            }
        }
        Evidence::Noise => {
            for i in 0..TRIALS {
                rows.push(TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("noise-{salt}-{i}"),
                    returns: noise_row(i),
                    trades_outside_grid: 0,
                });
            }
        }
    }

    let matrix = TrialReturnMatrix {
        period_keys: month_keys(),
        initial_balance: 10_000.0,
        rows,
    };
    let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
    (encode(&matrix, &refs), champion)
}

// ─────────────────────────────────────────────────────────────────────────────
// The executor
// ─────────────────────────────────────────────────────────────────────────────

/// One executed search, as the test observed it.
#[derive(Debug, Clone, PartialEq)]
pub struct Executed {
    pub sweep: SweepId,
    pub slot: usize,
    pub config_hash: String,
    pub control: bool,
}

/// How the fixture should damage the promotion evidence it writes at S5.
///
/// Each variant reproduces a REAL failure the round-2 work claims is now
/// refused before the irreplaceable out-of-sample window is spent. They are
/// modelled as executor behaviour rather than as post-hoc file edits because
/// that is where they actually come from: the evidence is written by the search
/// that selected the genes, so a build that does not write it, a scratch
/// directory shared by two sessions, and a search that selected nothing are all
/// differences in what the executor put on disk — not in what someone did to
/// the disk afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sabotage {
    /// Write it honestly.
    None,
    /// Write NOTHING — the original defect exactly. One place read
    /// `survivors.json`; nothing in the workspace wrote it. The promotion path
    /// was therefore guaranteed to fail, and it journalled `OosTouchSpent`
    /// before finding out.
    OmitEvidence,
    /// Write it at the right path, stamped with ANOTHER configuration's hash.
    /// This is what a scratch root keyed by symbol produces when sweep ids
    /// restart at 1 every session: session B's sweep-1/slot-007 IS session A's.
    ForeignStamp,
    /// A search that selected nothing, writing a well-formed artifact with an
    /// empty portfolio. "No evidence" and "evidence saying nothing survived"
    /// must be the same observable.
    EmptyGenes,
}

/// A deterministic, card-free [`SweepExecutor`].
pub struct ScriptedExecutor {
    completed: usize,
    current_sweep: u64,
    completed_in_sweep: usize,
    die_in_sweep_after: Option<(u64, usize)>,
    sabotage: Sabotage,
    pub executed: Vec<Executed>,
    pub oos_calls: Vec<(SweepId, usize)>,
}

impl ScriptedExecutor {
    pub fn new() -> Self {
        Self {
            completed: 0,
            current_sweep: 0,
            completed_in_sweep: 0,
            die_in_sweep_after: None,
            sabotage: Sabotage::None,
            executed: Vec::new(),
            oos_calls: Vec::new(),
        }
    }

    /// An executor that damages its promotion evidence in one named way, and is
    /// otherwise identical — so the ONLY difference between this run and the
    /// end-to-end run is the evidence, and any change in the verdict is
    /// attributable to it.
    pub fn sabotaging(sabotage: Sabotage) -> Self {
        Self {
            sabotage,
            ..Self::new()
        }
    }

    /// Panic once `after` searches of a sweep numbered `from` or higher have
    /// completed — a process death **inside** a sweep, through the real runner.
    ///
    /// Keyed on the sweep and not on a running total on purpose. Most proposals
    /// are refused before they run (`Proposal::resolve` bails on every non-zero
    /// `streaming_batches` and every non-`Start` cursor, both of which the
    /// proposer samples uniformly), so the number of searches a sweep actually
    /// executes is not a constant. A raw count would land the death in a
    /// different place on any change to the proposer, and a crash test whose
    /// crash moves is a crash test that stops testing the thing it was written
    /// for: an intent record on disk with no outcome, and a partial ledger with
    /// real trials in it.
    pub fn dying_in_sweep(from: u64, after: usize) -> Self {
        Self {
            die_in_sweep_after: Some((from, after)),
            ..Self::new()
        }
    }

    pub fn searches_run(&self) -> usize {
        self.executed.len()
    }
}

impl Default for ScriptedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// What one slot is scripted to produce, and which screen conjunct it is meant
/// to exercise.
///
/// Spread across S2–S5 on purpose: a fixture where every failure is the same
/// conjunct cannot tell a census that counts from a census that always reports
/// the same number.
fn script_for(slot: usize) -> (Evidence, CostBandCounts, Option<f64>, usize) {
    let survives = CostBandCounts {
        survives: 1,
        optimistic_edge_only: 0,
        fails: TRIALS - 1,
        unmeasured: 0,
        not_discriminating: 0,
        discriminates: true,
    };
    match slot % 5 {
        // The one that can pass, once the shuffle null exists.
        3 => (Evidence::Strong, survives, Some(0.30), 400),
        // S2 — a matrix of noise.
        0 => (Evidence::Noise, survives, Some(0.05), 400),
        // S3 — the band cannot discriminate, so passing it is arithmetic.
        1 => (
            Evidence::Strong,
            CostBandCounts {
                discriminates: false,
                ..survives
            },
            Some(0.10),
            400,
        ),
        // S4 — the measured base rate, not a near miss.
        2 => (Evidence::Strong, survives, Some(-0.02), 400),
        // S5 — too few trades to mean anything.
        _ => (Evidence::Strong, survives, Some(0.20), 4),
    }
}

impl SweepExecutor for ScriptedExecutor {
    fn describe(&self) -> String {
        "SCRIPTED EXECUTOR — no card, no dataset, no feature cube, no broker".to_string()
    }

    fn streaming_requested(&self) -> bool {
        false
    }

    fn windows(&self) -> anyhow::Result<((i64, i64), OosWindow, usize, f64)> {
        Ok((
            (0, 900 * DAY),
            OosWindow {
                start_ms: 900 * DAY + 1,
                end_ms: 1_200 * DAY,
            },
            300 * 288,
            5.0,
        ))
    }

    fn execute(&mut self, request: &SearchRequest<'_>) -> anyhow::Result<SearchOutcome> {
        if request.sweep.0 != self.current_sweep {
            self.current_sweep = request.sweep.0;
            self.completed_in_sweep = 0;
        }
        if let Some((from, after)) = self.die_in_sweep_after
            && request.sweep.0 >= from
            && self.completed_in_sweep >= after
        {
            panic!(
                "SCRIPTED EXECUTOR: simulated process death inside {} slot {}, after {} of its \
                 searches had completed",
                request.sweep, request.slot, self.completed_in_sweep
            );
        }

        let control = request.permutation.is_some();
        let (evidence, cost_band, e_screen_pess, n_trades) = if control {
            // The control destroyed the relationship between the features and
            // the future. It must score like noise, or the null it feeds is a
            // bar nothing can fail.
            (
                Evidence::Noise,
                CostBandCounts {
                    survives: 0,
                    optimistic_edge_only: 0,
                    fails: TRIALS,
                    unmeasured: 0,
                    not_discriminating: 0,
                    discriminates: true,
                },
                Some(-0.05),
                400,
            )
        } else {
            script_for(request.slot)
        };

        let salt = (request.sweep.0 as usize) * 1_000 + request.slot;
        let (bytes, champion) = matrix_for(evidence, salt);

        // A REAL matrix at the path the runner named. The runner is entitled to
        // re-read it — the honest N is session-wide and only the runner knows it
        // — so a fixture that wrote junk here would fail a correct runner.
        if let Some(parent) = request.trial_returns_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&request.trial_returns_path, &bytes)?;

        // THE PROMOTION EVIDENCE, written where a real executor writes it.
        //
        // The runner refuses to spend the out-of-sample window without it, so a
        // fixture that skipped this could never drive the loop to a promotion —
        // and that is the point: NOTHING in the workspace used to write this
        // file, while one place read it, so the promotion path could not
        // succeed and consumed the window finding out. A search that selected
        // nothing writes nothing, exactly as the real executor does.
        if e_screen_pess.is_some() && self.sabotage != Sabotage::OmitEvidence {
            let genes = if self.sabotage == Sabotage::EmptyGenes {
                Vec::new()
            } else {
                vec![neoethos_search::genetic::Gene {
                    indices: vec![0, 1],
                    weights: vec![0.5, -0.5],
                    strategy_id: format!("{}-slot{}", request.sweep, request.slot),
                    sl_pips: 20.0,
                    tp_pips: 40.0,
                    ..Default::default()
                }]
            };
            let portfolio = neoethos_autoresearch::runner::PromotionPortfolio {
                schema: neoethos_autoresearch::runner::PROMOTION_EVIDENCE_SCHEMA.to_string(),
                sweep: request.sweep,
                slot: request.slot,
                // The STAMP: the slot's own config_hash, so the runner's check
                // that the evidence describes the promoted configuration is
                // exercised rather than bypassed.
                config_hash: if self.sabotage == Sabotage::ForeignStamp {
                    // A DIFFERENT session's stamp at the right path — the exact
                    // collision a symbol-keyed scratch root produces.
                    "fnv64:another-session".to_string()
                } else {
                    request.config_hash.to_string()
                },
                feature_names: vec!["rsi_14".to_string(), "atr_14".to_string()],
                genes,
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
        }

        self.completed += 1;
        self.completed_in_sweep += 1;
        self.executed.push(Executed {
            sweep: request.sweep,
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            control,
        });

        let statistics: TrialStatisticsReport = analyse_bytes(&bytes, TRIALS);
        // The live champion improves with the sweep index, so `best_ever` has to
        // ADVANCE and can be checked by value rather than by presence.
        let lift = if control {
            0.0
        } else {
            0.01 * request.sweep.0 as f64
        };

        Ok(SearchOutcome {
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            trials_offered: TRIALS,
            statistics,
            cost_band,
            rejections: vec![
                ("base_quality.profile_payoff_ratio".to_string(), 7),
                ("min_trades".to_string(), 3),
            ],
            survivors: usize::from(cost_band.survives > 0),
            e_screen_pess: e_screen_pess.map(|e| e + lift),
            n_trades,
            champion_returns: champion,
            champion_period_keys: month_keys(),
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
        // The genes are a PARAMETER: the runner proved they exist, and proved
        // they are stamped with THIS slot's config_hash, before it journalled
        // the window as spent. A fixture that ignored them would let the
        // promotion path pass with no evidence behind it — which is exactly the
        // shape of the defect this signature exists to prevent.
        assert_eq!(portfolio.sweep, sweep, "the evidence must name its sweep");
        assert_eq!(portfolio.slot, slot, "the evidence must name its slot");
        assert!(
            !portfolio.genes.is_empty(),
            "the runner must never spend the window on an empty portfolio"
        );
        self.oos_calls.push((sweep, slot));
        Ok(OosEvidence {
            window: OosWindow {
                start_ms: 900 * DAY + 1,
                end_ms: 1_200 * DAY,
            },
            per_trade_net_pips: vec![1.0; 500],
            r_multiples: vec![0.2; 500],
            monthly_returns: vec![0.03; PERIODS],
            period_keys: month_keys(),
            trades_per_day: 1.0,
            band_survives: true,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The store, redirected
// ─────────────────────────────────────────────────────────────────────────────

/// Point the WHOLE process's store at `root` and build the operator-shaped
/// settings the loop reads its goals from.
///
/// `NEOETHOS_USER_DATA_DIR` is upstream of `config::user_config_path`, which is
/// what `SessionStore::root` derives the autoresearch store from. It is a
/// process-global, so each of these tests lives in its OWN integration-test
/// file: cargo gives each file its own binary, and therefore its own
/// environment.
pub fn settings_in(root: &Path) -> Settings {
    // SAFETY: one writer, before any thread is spawned, and each integration
    // test file is its own process.
    unsafe {
        std::env::set_var("NEOETHOS_USER_DATA_DIR", root);
    }
    let mut settings = Settings::default();
    settings.system.trading_mode = "risky".to_string();
    settings.system.risky_start_balance_usd = 1_000.0;
    settings.system.risky_target_balance_usd = 200_000.0;
    settings.system.risky_horizon_days = 180;
    settings
}

/// A clean store root under the OS temp directory, removed first so a rerun
/// never resumes yesterday's session.
pub fn fresh_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("neoethos-autoresearch-test-{name}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("creating the test store root");
    root
}

pub fn base_config(settings: &Settings) -> DiscoveryConfig {
    DiscoveryConfig {
        evaluation_symbol: "EURUSD".to_string(),
        ..DiscoveryConfig::from_settings(settings)
    }
    .apply_mode_overrides()
}

/// The one session directory under the store.
pub fn only_session_dir() -> PathBuf {
    let root = SessionStore::root().expect("the autoresearch store root");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("reading the store root {}: {e}", root.display()))
        .map(|e| e.expect("a store entry").path())
        .filter(|p| p.is_dir() && p.join("journal.jsonl").exists())
        .collect();
    assert_eq!(
        dirs.len(),
        1,
        "expected exactly one session under {}, found {dirs:?}",
        root.display()
    );
    dirs.pop().expect("the session directory")
}

// ─────────────────────────────────────────────────────────────────────────────
// Reading the journal back — the artifact the assertions are made against
//
// Deliberately parsed as raw JSON rather than through `Record`: the journal is
// the artifact an operator reads with `tail` in five years, and a test that can
// only read it through this build's own enum cannot notice a record that
// serialises to something a human cannot follow.
// ─────────────────────────────────────────────────────────────────────────────

pub fn journal_lines(dir: &Path) -> Vec<serde_json::Value> {
    let path = dir.join("journal.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("journal line is not JSON: {e}\n{l}"))
        })
        .collect()
}

pub fn tag_of(line: &serde_json::Value) -> String {
    line.get("record")
        .and_then(|r| r.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("<untyped>")
        .to_string()
}

pub fn records_tagged<'a>(
    lines: &'a [serde_json::Value],
    tag: &str,
) -> Vec<&'a serde_json::Value> {
    lines
        .iter()
        .filter(|l| tag_of(l) == tag)
        .map(|l| l.get("record").expect("a record"))
        .collect()
}

pub fn tag_counts(lines: &[serde_json::Value]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for line in lines {
        *counts.entry(tag_of(line)).or_insert(0) += 1;
    }
    counts
}

/// `ScreenResult` is `#[serde(tag = "outcome", rename_all = "snake_case")]`, so a
/// pass is `{"outcome":"passed", …}`. Read by tag rather than by shape: a shape
/// check would quietly stop matching the day a field is added.
fn screen_outcome(record: &serde_json::Value) -> Option<&str> {
    record
        .get("screen_result")
        .and_then(|r| r.get("outcome"))
        .and_then(|o| o.as_str())
}

/// Every `Screened` record whose verdict was a PASS.
pub fn passed_screens<'a>(lines: &'a [serde_json::Value]) -> Vec<&'a serde_json::Value> {
    records_tagged(lines, "Screened")
        .into_iter()
        .filter(|r| screen_outcome(r) == Some("passed"))
        .collect()
}

/// Every distinct `Screened` failure detail, for a failure message that names
/// WHY nothing passed instead of only that nothing did. This is the diagnostic
/// that located the unsatisfiable N contract.
pub fn distinct_screen_failures(lines: &[serde_json::Value]) -> Vec<String> {
    let mut out: std::collections::BTreeSet<String> = Default::default();
    for record in records_tagged(lines, "Screened") {
        if screen_outcome(record) == Some("passed") {
            continue;
        }
        let Some(result) = record.get("screen_result") else {
            continue;
        };
        let rendered = serde_json::to_string(result).unwrap_or_default();
        out.insert(rendered.chars().take(320).collect());
    }
    out.into_iter().collect()
}

/// Every file under `root` with this exact name. Used to prove a NEGATIVE — that
/// the loop wrote no trading artifact anywhere it could reach.
pub fn files_named(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(files_named(&path, name));
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            out.push(path);
        }
    }
    out
}

/// Count of trial-returns matrices still on disk, by sweep directory name.
pub fn matrices_on_disk(dir: &Path) -> std::collections::BTreeMap<String, usize> {
    let mut out = std::collections::BTreeMap::new();
    let sweeps = dir.join("sweeps");
    let Ok(entries) = std::fs::read_dir(&sweeps) else {
        return out;
    };
    for entry in entries.flatten() {
        let n = std::fs::read_dir(entry.path().join("trial_returns"))
            .map(|d| d.flatten().count())
            .unwrap_or(0);
        out.insert(entry.file_name().to_string_lossy().to_string(), n);
    }
    out
}
