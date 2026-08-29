//! The journal — append-only JSONL, crash-safe, replayable.
//!
//! `docs/autoresearch-loop.md` §11.
//!
//! > **The journal is the memory.** Every trial with its full configuration and
//! > its result, appended, and the next proposal is a function of that history.
//!
//! This is the one idea taken whole from Karpathy's autoresearch loop, including
//! the property that matters most here: *the history is the artifact*, so every
//! claim the loop makes can be re-derived from it. A loop whose history cannot
//! be audited is a random number generator with good manners.
//!
//! ## Why JSONL and not one big JSON
//!
//! Rejected: a single JSON document rewritten each sweep. This repository has
//! `exit 137` OOM kills and multi-hour runs that ended with no artifact at all;
//! a whole-file rewrite loses the ENTIRE session in exactly the failure mode
//! that actually happens here. Rejected: a database — a dependency, a lock, and
//! a file a human cannot read with `tail`.
//!
//! ## Two-phase, and why it is the whole of "a crash loses one sweep"
//!
//! [`Record::SweepStarted`] is serialised, written and **fsynced before any work
//! begins** (intent); the outcome record is written after (outcome). On replay a
//! `SweepStarted` with no matching outcome is a sweep the process died inside,
//! and the resume path turns it into [`Record::SweepAbandoned`] — counted,
//! named, and with its `trials_offered` still counting into `N_session`, because
//! **a trial that ran is a trial that ran** and forgetting it would be a loop
//! lying to itself with extra steps.
//!
//! ## The truncated tail
//!
//! A process killed mid-`write` leaves a partial final line. On replay that tail
//! is truncated from the file and recorded as [`Record::TruncatedTail`] — counted,
//! never silently absorbed, exactly the way `neoethos_search::trial_returns`
//! handles its own truncated tail. A **complete** line that fails to parse is a
//! different thing entirely and is a hard error naming the line number: that is
//! corruption in the middle of a history, and continuing past it would mean
//! folding a session from records the loop cannot read.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::session::{
    BestEver, BlockId, ChampionRow, DatasetReceiptV1, GcCensus, SessionId, SweepId,
};

// ─────────────────────────────────────────────────────────────────────────────
// Serialisable mirrors of upstream types
//
// `neoethos_search::discovery::CostBandCensus` and
// `neoethos_search::goal_report::RiskOutcome` are NOT `Serialize` today. Rather
// than reach into another crate's derive list (forbidden territory) the journal
// carries its own field-for-field mirrors, built by `From`. If those derives
// land, these can be deleted — see the REQUIRED ELSEWHERE note in
// `docs/autoresearch-loop.md`.
// ─────────────────────────────────────────────────────────────────────────────

/// Mirror of `neoethos_search::discovery::CostBandCensus`, plus the one fact
/// the census itself cannot carry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostBandCounts {
    pub survives: usize,
    pub optimistic_edge_only: usize,
    pub fails: usize,
    pub unmeasured: usize,
    pub not_discriminating: usize,
    /// `cost_band_discriminates(band, baseline)` for the run that produced these
    /// counts.
    ///
    /// Carried BESIDE the counts and not derived from them, because a census of
    /// all-zeros looks identical whether the band was discriminating and nothing
    /// survived, or the band could not tell anything apart in the first place —
    /// and the second reads clean, *"which is worse than no census, because a
    /// reader takes it as evidence."* Screen conjunct S3 reads this field.
    pub discriminates: bool,
}

impl CostBandCounts {
    /// Build from the search's own census plus the band verdict.
    pub fn from_census(
        census: &neoethos_search::discovery::CostBandCensus,
        discriminates: bool,
    ) -> Self {
        Self {
            survives: census.survives,
            optimistic_edge_only: census.optimistic_edge_only,
            fails: census.fails,
            unmeasured: census.unmeasured,
            not_discriminating: census.not_discriminating,
            discriminates,
        }
    }
}

/// Mirror of `neoethos_search::goal_report::RiskOutcome`.
///
/// The FULL frontier is journalled on a promotion, never just the level that
/// passed: the operator has to see the whole curve, including the risk levels
/// where `P(ruin)` is worse than `P(reach)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskOutcomeRecord {
    pub risk_fraction: f64,
    pub p_reach_target: f64,
    pub p_ruin: f64,
    pub median_terminal: f64,
    pub mean_terminal: f64,
    pub median_days_to_target: Option<f64>,
    pub reached_paths: usize,
    pub paths: usize,
}

impl From<&neoethos_search::goal_report::RiskOutcome> for RiskOutcomeRecord {
    fn from(o: &neoethos_search::goal_report::RiskOutcome) -> Self {
        Self {
            risk_fraction: o.risk_fraction,
            p_reach_target: o.p_reach_target,
            p_ruin: o.p_ruin,
            median_terminal: o.median_terminal,
            mean_terminal: o.mean_terminal,
            median_days_to_target: o.median_days_to_target,
            reached_paths: o.reached_paths,
            paths: o.paths,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Record payloads
// ─────────────────────────────────────────────────────────────────────────────

/// Live sweep, or shuffle control. Control sweeps spend budget and their trials
/// count into `N_session` exactly like live ones — the null is not free and
/// pretending it is would flatter every deflated statistic in the session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SweepKind {
    Live,
    /// A control re-run of `source_sweep`'s exact 100 proposals against
    /// permuted features. `control` names WHICH permutation, because the
    /// circular rotation is the one that enters screen conjunct S6 and the
    /// per-column permutation is a diagnostic that enters no inequality.
    Control {
        control: crate::shuffle::ControlKind,
        source_sweep: SweepId,
        block: BlockId,
    },
}

impl SweepKind {
    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live)
    }
}

/// One search's compact outcome, as it lands in the journal.
///
/// The FULL `TrialStatisticsReport`, the censuses and the survivors go to the
/// sweep's artifact directory; this is the row a human reads with `tail`, and
/// the row the fold needs. Keeping it small matters: the journal is read in
/// full on every resume of a session that may run for days.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRecord {
    pub slot: usize,
    /// From `ResolvedConfigStamp`. Two searches with the same hash searched
    /// under the same decisions.
    pub config_hash: String,
    /// Every screened candidate this search offered — the honest N contribution.
    pub trials_offered: usize,
    pub survivors: usize,
    /// Per-trade net expectancy of the best survivor, charged at the
    /// PESSIMISTIC edge of the cost band. `None` when nothing survived.
    pub e_screen_pess: Option<f64>,
    pub n_trades: usize,
    pub dsr: Option<f64>,
    pub pbo: Option<f64>,
    /// Present exactly when the corresponding statistic is absent. A refusal is
    /// carried, never dropped: "no number" is never good news.
    pub dsr_refusal: Option<String>,
    pub pbo_refusal: Option<String>,
    pub cost_band: CostBandCounts,
    /// The ten named rejection counters, as `(name, count)`. Named, not
    /// positional, so a new counter upstream widens the census instead of
    /// silently shifting every column.
    pub rejections: Vec<(String, usize)>,
    pub streamed: bool,
    pub batch_columns: usize,
    pub next_cursor: usize,
    pub wall_ms: u64,
    /// A per-search failure. The sweep CONTINUES; one bad search must not throw
    /// away the ninety-nine that worked.
    pub error: Option<String>,
}

/// One posterior credit. `success` is the binary, shuffle-corrected reward —
/// never the continuous in-sample score, which on this data is dominated by
/// selection noise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosteriorCredit {
    pub factor: String,
    pub level: String,
    pub success: bool,
}

/// The goal constants and the judge thresholds, denormalised into
/// [`Record::SessionOpened`] as `(path, value)` pairs.
///
/// Denormalised on purpose: the header must be readable by a person holding
/// only `journal.jsonl`, without this crate, in five years. A hash alone proves
/// nothing to a reader who cannot see what was hashed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedValues(pub Vec<(String, String)>);

/// The out-of-sample window, in epoch ms. Defined once in the header, touched
/// once in the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OosWindow {
    pub start_ms: i64,
    pub end_ms: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// The record enum — §11.3
// ─────────────────────────────────────────────────────────────────────────────

/// Everything that ever happened, in order.
///
/// One tagged enum, because the fold has to be total: a `match` that gains a
/// variant without gaining an arm does not compile, which is what stops a new
/// event type from being silently ignored by [`crate::session::Session::apply`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Record {
    /// Written ONCE, first line of the journal.
    SessionOpened {
        session_id: SessionId,
        session_seed: u64,
        goal_hash: String,
        judge_hash: String,
        cost_hash: String,
        /// Every goal constant, by field path and resolved value.
        goals: NamedValues,
        /// What the scenario list was built from. `"… (single)"` says out loud
        /// that the operator's second scenario is not enumerated.
        scenarios_source: String,
        /// Every judge threshold, by name and value.
        judge: NamedValues,
        /// Every frozen cost-model field, by field path and value.
        costs: NamedValues,
        oos_window: OosWindow,
        /// The proposer's priors, written down so a reader can see the loop
        /// started with no opinion.
        priors: String,
        /// `max_sweeps` and the wall-clock budget.
        budget: NamedValues,
        /// How a proposal's identity is computed — it says so out loud when
        /// `ResolvedConfigStamp` lacks `ga_seed` and replicates therefore need
        /// an external seed to avoid being deduplicated away (§14.3).
        identity_source: String,
        symbol: String,
        dataset_receipt: DatasetReceiptV1,
    },

    ProposalDrawn {
        sweep: SweepId,
        slot: usize,
        proposal: crate::proposal::Proposal,
        origin: String,
    },
    ProposalDuplicate {
        sweep: SweepId,
        slot: usize,
        config_hash: String,
        first_seen_sweep: SweepId,
    },
    /// Refused BEFORE it ran — e.g. a payoff floor unreachable under the
    /// proposal's own geometry. Never run, always counted, always named.
    ProposalRefused {
        sweep: SweepId,
        slot: usize,
        config_hash: String,
        reason: String,
    },

    /// INTENT. Fsynced before any work begins.
    SweepStarted {
        sweep: SweepId,
        sweep_hash: String,
        kind: SweepKind,
        started_ms: i64,
    },
    /// OUTCOME.
    SweepCompleted {
        sweep: SweepId,
        trials_offered: usize,
        per_search: Vec<SearchRecord>,
        wall_ms: u64,
    },
    SweepFailed {
        sweep: SweepId,
        trials_offered: usize,
        error: String,
    },
    /// Written on RESUME, for a `SweepStarted` with no outcome.
    SweepAbandoned {
        sweep: SweepId,
        trials_offered: usize,
        reason: String,
    },

    Screened {
        sweep: SweepId,
        slot: usize,
        screen_result: crate::judge::ScreenResult,
        failing_conjunct: Option<String>,
    },

    /// The session's best screened result advanced.
    ///
    /// **This record exists because `Session::best_ever` is a FOLD PRODUCT and
    /// not session state.** It was, briefly, a field the runner assigned through
    /// a `&mut Session`, and the consequence was exact and severe: a debug build
    /// tripped the §11.4 fold-equality assertion on the very next append, and a
    /// release build lost the pointer on every restart. Losing it disables the
    /// one thing this project has never been able to say — R2's U2 condition
    /// reads `best_ever` and, with `None` on the left of the comparison, cannot
    /// be satisfied, so THE LOOP CAN NO LONGER REPORT THE GOAL UNREACHABLE. It
    /// also un-keeps the previous best sweep's trial matrices at the next GC,
    /// which takes the evidence with it.
    ///
    /// Written ONLY when the candidate strictly improves on the current best, so
    /// the record's name is true; the fold applies the same comparison
    /// ([`crate::session::advances_best`]) rather than assigning blind, which
    /// makes the folded pointer monotone whatever order a reader replays in.
    BestEverAdvanced { best: BestEver },

    /// One sweep's champion row — the `pbo_session` input.
    ///
    /// Journalled for the same reason as [`Record::BestEverAdvanced`]: the
    /// session champion matrix is the input to PBO of the loop's OWN selection
    /// procedure, and a matrix that restarts empty on every resume means
    /// `pbo_session` refuses forever and NO PROMOTION IS POSSIBLE. The row is
    /// carried whole — it is small, and re-deriving it would need the sweep's
    /// full statistics, which the GC is allowed to delete.
    ///
    /// The row carries its own `sweep`, so there is no second copy of that id to
    /// disagree with it.
    ChampionRecorded { row: ChampionRow },

    PosteriorUpdated {
        sweep: SweepId,
        credits: Vec<PosteriorCredit>,
    },
    /// A block judged INDISTINGUISHABLE from its shuffle control retracts every
    /// success it credited. The posterior is a fold, so this is exact rather
    /// than an approximation.
    PosteriorRetracted {
        block: BlockId,
        sweep_ids: Vec<SweepId>,
        reason: String,
    },

    ShuffleControlCompleted {
        block: BlockId,
        kind: crate::shuffle::ControlKind,
        /// The rotation offset, as a fraction of the series length.
        tau: f64,
        source_sweep: SweepId,
        trials_offered: usize,
        /// The control's best `E_screen_pess` — the observation that joins the
        /// session null.
        e_screen_pess: Option<f64>,
        dsr: Option<f64>,
    },
    BlockJudged {
        block: BlockId,
        delta_expectancy: f64,
        p_block: f64,
        dsr_gap: f64,
        indistinguishable: bool,
    },

    /// The OOS window is spent. There is one of these per session, ever.
    OosTouchSpent {
        window: OosWindow,
        sweep: SweepId,
        candidate: String,
    },
    Promoted {
        sweep: SweepId,
        goal_metric: f64,
        goal_bar: f64,
        frontier: Vec<RiskOutcomeRecord>,
    },
    PromotionRefused {
        sweep: SweepId,
        failing_conjunct: String,
    },

    /// One garbage-collection pass over the trial-returns matrices (§3.2).
    ///
    /// **This record exists because the two GC counters were rendered by every
    /// verdict and ASSIGNED BY NOBODY.** `AbandonCensus::gc_deleted_matrices`
    /// and `gc_bytes_reclaimed` are printed on every report — *"trial matrices
    /// GC'd 0 (0 bytes reclaimed)"* — while `SessionStore::gc_trial_matrices`
    /// was quietly deleting every non-kept sweep's matrices on the way past.
    /// There was no `Record` variant for a GC pass, so the fold
    /// had nothing to fold and the operator was told zero matrices had been
    /// deleted by a session that had deleted all of them. A census that reads
    /// clean while the deletions happen is worse than no census, because a
    /// reader takes it as evidence — the same sentence
    /// [`CostBandCounts::discriminates`] exists for.
    ///
    /// Journalled on **every** pass, including a pass that deleted nothing: "the
    /// GC ran and found nothing to reclaim" and "the GC never ran" are different
    /// facts, and only one of them is a wiring finding.
    ///
    /// The census is carried WHOLE rather than re-declared field by field, so
    /// the number in the log line and the number in the verdict are the same
    /// value and not two values that agree today.
    GarbageCollected {
        /// The sweep the pass followed. A GC pass is not a sweep, but it always
        /// happens after one, and keying it lets a reader line the reclaimed
        /// bytes up against the sweep whose matrices went.
        sweep: SweepId,
        census: GcCensus,
    },

    SessionStopped {
        verdict: crate::verdict::SessionVerdict,
    },

    /// A partial final line found on replay, truncated from the file and
    /// counted here. Never silently absorbed.
    TruncatedTail { bytes: usize },
}

impl Record {
    /// The sweep this record is about, when it is about one. Used by the fold
    /// and by the posterior retraction replay.
    pub fn sweep(&self) -> Option<SweepId> {
        match self {
            Self::ProposalDrawn { sweep, .. }
            | Self::ProposalDuplicate { sweep, .. }
            | Self::ProposalRefused { sweep, .. }
            | Self::SweepStarted { sweep, .. }
            | Self::SweepCompleted { sweep, .. }
            | Self::SweepFailed { sweep, .. }
            | Self::SweepAbandoned { sweep, .. }
            | Self::Screened { sweep, .. }
            | Self::PosteriorUpdated { sweep, .. }
            | Self::OosTouchSpent { sweep, .. }
            | Self::Promoted { sweep, .. }
            | Self::PromotionRefused { sweep, .. }
            | Self::GarbageCollected { sweep, .. } => Some(*sweep),
            Self::ShuffleControlCompleted { source_sweep, .. } => Some(*source_sweep),
            Self::BestEverAdvanced { best } => Some(best.sweep),
            Self::ChampionRecorded { row } => Some(row.sweep),
            Self::SessionOpened { .. }
            | Self::PosteriorRetracted { .. }
            | Self::BlockJudged { .. }
            | Self::SessionStopped { .. }
            | Self::TruncatedTail { .. } => None,
        }
    }

    /// A short tag for logs and for the census's named examples.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::SessionOpened { .. } => "SessionOpened",
            Self::ProposalDrawn { .. } => "ProposalDrawn",
            Self::ProposalDuplicate { .. } => "ProposalDuplicate",
            Self::ProposalRefused { .. } => "ProposalRefused",
            Self::SweepStarted { .. } => "SweepStarted",
            Self::SweepCompleted { .. } => "SweepCompleted",
            Self::SweepFailed { .. } => "SweepFailed",
            Self::SweepAbandoned { .. } => "SweepAbandoned",
            Self::Screened { .. } => "Screened",
            Self::BestEverAdvanced { .. } => "BestEverAdvanced",
            Self::ChampionRecorded { .. } => "ChampionRecorded",
            Self::PosteriorUpdated { .. } => "PosteriorUpdated",
            Self::PosteriorRetracted { .. } => "PosteriorRetracted",
            Self::ShuffleControlCompleted { .. } => "ShuffleControlCompleted",
            Self::BlockJudged { .. } => "BlockJudged",
            Self::OosTouchSpent { .. } => "OosTouchSpent",
            Self::Promoted { .. } => "Promoted",
            Self::PromotionRefused { .. } => "PromotionRefused",
            Self::GarbageCollected { .. } => "GarbageCollected",
            Self::SessionStopped { .. } => "SessionStopped",
            Self::TruncatedTail { .. } => "TruncatedTail",
        }
    }
}

/// One line of `journal.jsonl`.
///
/// `seq` and `at_ms` are envelope, not content: they let a reader order the
/// file by hand and let the replay report *"the tail was cut after record
/// 4,182"* rather than *"the tail was cut"*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalLine {
    pub schema: String,
    pub seq: u64,
    pub at_ms: i64,
    pub record: Record,
}

/// Schema-only view used before the version-specific [`Record`] payload is
/// deserialized. Serde ignores `record` here, which lets a legacy envelope be
/// reported as the typed version mismatch it is instead of mislabelling its
/// now-incomplete payload as corruption.
#[derive(Deserialize)]
struct JournalEnvelope {
    schema: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// The journal itself
// ─────────────────────────────────────────────────────────────────────────────

/// What replay found before the first record was handed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayReport {
    pub lines_read: usize,
    /// Bytes of a partial final line, truncated from the file.
    pub truncated_tail_bytes: usize,
}

/// An append-only JSONL journal, fsynced per record.
///
/// Holds every record in memory as well as on disk, because
/// [`crate::session::Session`] is defined as a pure fold over exactly this
/// sequence and the equality between the two is asserted after every append.
// `Debug` because the refusal tests assert on `Journal::open(..).expect_err(..)`,
// and `expect_err` has to be able to print the Ok value it did NOT get. A
// refusal test that cannot name what came back instead is a test that reports
// "expected an error" and nothing else.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: File,
    records: Vec<Record>,
    next_seq: u64,
    replay: ReplayReport,
}

impl Journal {
    /// Open an existing journal (replaying it) or create a new empty one.
    ///
    /// A partial final line is TRUNCATED FROM THE FILE here, before the handle
    /// is returned for appending: leaving it in place would corrupt the next
    /// record by concatenating it onto a half-written one.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating the journal directory {}", parent.display()))?;
        }

        let mut replay = ReplayReport::default();
        let mut records = Vec::new();

        if path.exists() {
            let mut bytes = Vec::new();
            File::open(&path)
                .with_context(|| format!("opening the journal {}", path.display()))?
                .read_to_end(&mut bytes)
                .with_context(|| format!("reading the journal {}", path.display()))?;

            // Everything up to and including the last newline is complete.
            // Anything after it is a line the process did not finish writing.
            let complete_len = match bytes.iter().rposition(|b| *b == b'\n') {
                Some(idx) => idx + 1,
                None => 0,
            };
            replay.truncated_tail_bytes = bytes.len() - complete_len;
            if replay.truncated_tail_bytes > 0 {
                tracing::warn!(
                    target: "neoethos_autoresearch::journal",
                    path = %path.display(),
                    bytes = replay.truncated_tail_bytes,
                    "the journal ended in a PARTIAL LINE — the process was killed mid-write. \
                     Truncating it and recording a TruncatedTail. This is counted, never \
                     silently absorbed."
                );
                let file = OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .with_context(|| format!("truncating the journal {}", path.display()))?;
                file.set_len(complete_len as u64)
                    .with_context(|| format!("truncating the journal {}", path.display()))?;
                file.sync_all().with_context(|| {
                    format!("fsyncing the truncated journal {}", path.display())
                })?;
            }

            let text = std::str::from_utf8(&bytes[..complete_len]).with_context(|| {
                format!(
                    "the journal {} is not valid UTF-8. This is corruption in the middle of a \
                     history, not a truncated tail; refusing to fold a session from records the \
                     loop cannot read.",
                    path.display()
                )
            })?;

            for (index, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let envelope: JournalEnvelope = serde_json::from_str(line).with_context(|| {
                    format!(
                        "journal {} line {} is a COMPLETE line that does not parse. That is \
                         corruption in the middle of the history, not a truncated tail — every \
                         later record's meaning depends on the ones before it, so the session \
                         cannot be folded past this point. Nothing has been modified.",
                        path.display(),
                        index + 1
                    )
                })?;
                if envelope.schema != crate::JOURNAL_SCHEMA {
                    bail!(
                        "journal {} line {} carries schema \"{}\", but this build writes \"{}\". \
                         Two journals with different schemas are not one history; start a new \
                         session rather than migrating.",
                        path.display(),
                        index + 1,
                        envelope.schema,
                        crate::JOURNAL_SCHEMA
                    );
                }
                let parsed: JournalLine = serde_json::from_str(line).with_context(|| {
                    format!(
                        "journal {} line {} is a COMPLETE line in schema {} whose payload does \
                         not parse. That is corruption in the middle of the history, not a \
                         truncated tail — every later record's meaning depends on the ones \
                         before it, so the session cannot be folded past this point. Nothing \
                         has been modified.",
                        path.display(),
                        index + 1,
                        crate::JOURNAL_SCHEMA
                    )
                })?;
                records.push(parsed.record);
                replay.lines_read += 1;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .with_context(|| format!("opening the journal {} for append", path.display()))?;

        let next_seq = replay.lines_read as u64;
        Ok(Self {
            path,
            file,
            records,
            next_seq,
            replay,
        })
    }

    /// Append one record, then **fsync**.
    ///
    /// The fsync is the whole point and it is not negotiable for throughput:
    /// without it a `SweepStarted` can be in the page cache when the machine
    /// dies, and the resume path would then have no idea a sweep had begun.
    /// At most a few hundred records per sweep — against sweeps that take
    /// minutes — the cost is not measurable.
    pub fn append(&mut self, record: Record) -> Result<()> {
        let line = JournalLine {
            schema: crate::JOURNAL_SCHEMA.to_string(),
            seq: self.next_seq,
            at_ms: chrono::Utc::now().timestamp_millis(),
            record,
        };
        let mut encoded = serde_json::to_string(&line)
            .with_context(|| format!("serialising a {} record", line.record.tag()))?;
        // A record containing a raw newline would split into two lines and make
        // the file unparseable. serde_json never emits one, so this is an
        // assertion about serde_json rather than an escape: if it ever becomes
        // false, fail here rather than corrupt the history.
        if encoded.contains('\n') {
            bail!(
                "a {} record serialised to JSON containing a newline, which would split it \
                 across two journal lines. Refusing to write it.",
                line.record.tag()
            );
        }
        encoded.push('\n');

        self.file
            .write_all(encoded.as_bytes())
            .with_context(|| format!("appending to the journal {}", self.path.display()))?;
        self.file
            .flush()
            .with_context(|| format!("flushing the journal {}", self.path.display()))?;
        self.file
            .sync_data()
            .with_context(|| format!("fsyncing the journal {}", self.path.display()))?;

        self.next_seq += 1;
        self.records.push(line.record);
        Ok(())
    }

    /// Every record, in order. The input to [`crate::session::Session::fold`].
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn replay_report(&self) -> ReplayReport {
        self.replay
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// n_session — the honest N, defined ONCE, as a fold
// ─────────────────────────────────────────────────────────────────────────────

/// The trial count the deflated Sharpe deflates against: **every configuration
/// the loop has ever run, across the whole session**, not the survivors of one
/// sweep.
///
/// Live, control, failed and abandoned sweeps ALL count. There is no code path
/// that decrements this and no field that stores it — it is recomputed from the
/// journal on every use, which is what makes *"a loop that resets its own N
/// between sweeps"* structurally impossible rather than forbidden by policy.
///
/// **Counted from the TERMINAL SWEEP records only.** A control counts because a
/// control IS a sweep: `run_block_control` drives it through `run_sweep`, which
/// appends its own `SweepCompleted`. [`Record::ShuffleControlCompleted`] is the
/// control's SUMMARY — its `trials_offered` is the same number, restated for a
/// reader holding only `journal.jsonl` — and adding it here counted every
/// control twice, inflating N by roughly one sweep per block. Over-counting N
/// does not flatter a deflated statistic, but it does make the headline N a
/// number no other artifact in the session can reproduce.
///
/// This is `docs/autoresearch-loop.md` §8.1, verbatim.
pub fn n_session(journal: &[Record]) -> usize {
    journal
        .iter()
        .filter_map(|r| match r {
            Record::SweepCompleted { trials_offered, .. }
            | Record::SweepFailed { trials_offered, .. }
            | Record::SweepAbandoned { trials_offered, .. } => Some(*trials_offered),
            _ => None,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp dir")
    }

    fn completed(sweep: u64, trials: usize) -> Record {
        Record::SweepCompleted {
            sweep: SweepId(sweep),
            trials_offered: trials,
            per_search: Vec::new(),
            wall_ms: 1,
        }
    }

    fn best_ever(sweep: u64, e: f64) -> BestEver {
        BestEver {
            sweep: SweepId(sweep),
            slot: 3,
            config_hash: format!("fnv64:{sweep:016x}"),
            e_screen_pess: e,
            n_trades: 250,
            dsr: Some(0.97),
            pbo: Some(0.31),
        }
    }

    fn gc_record(sweep: u64, deleted: usize, bytes: u64) -> Record {
        Record::GarbageCollected {
            sweep: SweepId(sweep),
            census: crate::session::GcCensus {
                kept: 2,
                deleted,
                bytes_reclaimed: bytes,
                rule: "keep the best-ever sweep and every control".to_string(),
                free_bytes_after: Some(1_234_567),
            },
        }
    }

    /// The GC pass survives the process, so the counters the verdict renders can
    /// be re-derived from the journal alone.
    ///
    /// Pinned at the journal level because the failure this closes was that the
    /// record did not exist at all: `AbandonCensus::gc_deleted_matrices` and
    /// `gc_bytes_reclaimed` were printed by every report and written by nothing,
    /// so the operator was told zero matrices had been deleted by a session that
    /// had deleted all of them.
    #[test]
    fn a_garbage_collection_pass_survives_a_close_and_reopen() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        let pass = gc_record(9, 300, 14_680_064);
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(pass.clone()).unwrap();
        }
        let reopened = Journal::open(&path).unwrap();
        assert_eq!(reopened.records(), [pass].as_slice());
        assert_eq!(reopened.replay_report().truncated_tail_bytes, 0);
    }

    fn champion(sweep: u64) -> ChampionRow {
        ChampionRow {
            sweep: SweepId(sweep),
            config_hash: format!("fnv64:{sweep:016x}"),
            strategy_id: format!("g{sweep}"),
            period_keys: vec![24_300, 24_301, 24_302],
            monthly_returns: vec![0.01, -0.02, 0.03],
            e_screen_pess: 0.4,
        }
    }

    #[test]
    fn records_survive_a_close_and_reopen() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(completed(1, 100)).unwrap();
            j.append(completed(2, 100)).unwrap();
        }
        let j = Journal::open(&path).unwrap();
        assert_eq!(j.records().len(), 2);
        assert_eq!(j.replay_report().truncated_tail_bytes, 0);
    }

    #[test]
    fn a_partial_final_line_is_truncated_and_counted_not_absorbed() {
        // This is the crash. The process died halfway through writing record 3.
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(completed(1, 100)).unwrap();
            j.append(completed(2, 100)).unwrap();
        }
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(b"{\"schema\":\"neoethos.autoresearch.journal.v2\",\"seq\":2,\"at")
                .unwrap();
        }

        let j = Journal::open(&path).unwrap();
        // The two COMPLETE records survive.
        assert_eq!(j.records().len(), 2);
        // The partial line is counted, not silently swallowed.
        assert!(j.replay_report().truncated_tail_bytes > 0);

        // And the file is now clean, so the next append cannot be concatenated
        // onto the half-written line.
        let mut j = j;
        j.append(completed(3, 100)).unwrap();
        let reopened = Journal::open(&path).unwrap();
        assert_eq!(reopened.records().len(), 3);
        assert_eq!(reopened.replay_report().truncated_tail_bytes, 0);
    }

    #[test]
    fn a_corrupt_complete_line_is_a_hard_error_naming_the_line() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(completed(1, 100)).unwrap();
        }
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(b"{\"this\":\"is not a journal line\"}\n")
                .unwrap();
        }
        let err = Journal::open(&path).expect_err("a corrupt complete line must not be skipped");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("line 2"), "got: {rendered}");
        assert!(rendered.contains("corruption in the middle"));
    }

    #[test]
    fn n_session_counts_live_control_failed_and_abandoned_alike() {
        // A trial that ran is a trial that ran. Forgetting the failed and
        // abandoned ones would understate N and therefore FLATTER every
        // deflated Sharpe in the session.
        //
        // The CONTROL counts once, through its own `SweepCompleted` — a control
        // is a sweep. `ShuffleControlCompleted` restates the same number for a
        // reader and must not be added again: this journal is what a real block
        // boundary writes, and 100 + 40 + 17 + 100 is the double count.
        let records = vec![
            completed(1, 100),
            Record::SweepFailed {
                sweep: SweepId(2),
                trials_offered: 40,
                error: "gpu fell over".into(),
            },
            Record::SweepAbandoned {
                sweep: SweepId(3),
                trials_offered: 17,
                reason: "process died".into(),
            },
            // The control sweep itself…
            completed(4, 100),
            // …and its summary, carrying the SAME 100.
            Record::ShuffleControlCompleted {
                block: BlockId(0),
                kind: crate::shuffle::ControlKind::CircularRotation,
                tau: 0.37,
                source_sweep: SweepId(1),
                trials_offered: 100,
                e_screen_pess: Some(-0.2),
                dsr: None,
            },
        ];
        assert_eq!(n_session(&records), 100 + 40 + 17 + 100);
    }

    #[test]
    fn n_session_has_no_decrement_path() {
        // Every record type that is NOT a terminal sweep outcome contributes
        // zero, and none of them can subtract. Asserted rather than asserted-in-
        // prose because "the loop cannot reset its own N" is the claim the whole
        // judge rests on.
        let noise = vec![
            Record::TruncatedTail { bytes: 12 },
            // The two records that carry `best_ever` and the champion matrix
            // into the fold. They are STATE, not trials: neither may move N.
            Record::BestEverAdvanced {
                best: best_ever(1, 0.4),
            },
            Record::ChampionRecorded { row: champion(1) },
            Record::BlockJudged {
                block: BlockId(0),
                delta_expectancy: -1.0,
                p_block: 1.0,
                dsr_gap: -0.5,
                indistinguishable: true,
            },
            // A GC pass deletes EVIDENCE, never trials. If it could move N the
            // session's deflation would get easier every time the disk was
            // tidied.
            gc_record(4, 97, 4_096),
            Record::PosteriorRetracted {
                block: BlockId(0),
                sweep_ids: vec![SweepId(1)],
                reason: "block indistinguishable".into(),
            },
        ];
        assert_eq!(n_session(&noise), 0);

        let mut with_a_sweep = noise.clone();
        with_a_sweep.push(completed(9, 100));
        assert_eq!(n_session(&with_a_sweep), 100);
    }

    /// `best_ever` and the champion matrix are journalled, so they survive the
    /// process that produced them.
    ///
    /// Pinned at the JOURNAL level as well as the fold level because the failure
    /// this closes was that they were never written at all: a session that lost
    /// them on restart could no longer satisfy R2's U2 condition (so it could
    /// never report the goal unreachable) and restarted `pbo_session` with an
    /// empty matrix (so no promotion was possible). Both are floating-point
    /// payloads, so this also pins that they survive JSON round-tripping
    /// bit-for-bit rather than approximately.
    #[test]
    fn best_ever_and_champions_survive_a_close_and_reopen() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        let best = best_ever(7, -0.019_531_25);
        let row = champion(7);
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(Record::ChampionRecorded { row: row.clone() })
                .unwrap();
            j.append(Record::BestEverAdvanced { best: best.clone() })
                .unwrap();
        }
        let reopened = Journal::open(&path).unwrap();
        assert_eq!(
            reopened.records(),
            [
                Record::ChampionRecorded { row },
                Record::BestEverAdvanced { best },
            ]
            .as_slice(),
            "the two pieces of state that were once held outside the journal must come back \
             from it unchanged"
        );
        assert_eq!(reopened.replay_report().truncated_tail_bytes, 0);
    }

    #[test]
    fn a_journal_from_another_schema_is_refused_rather_than_migrated() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        std::fs::write(
            &path,
            b"{\"schema\":\"neoethos.autoresearch.journal.v0\",\"seq\":0,\"at_ms\":0,\
              \"record\":{\"type\":\"TruncatedTail\",\"bytes\":0}}\n",
        )
        .unwrap();
        let err = Journal::open(&path).expect_err("a foreign schema must be refused");
        assert!(format!("{err:#}").contains("not one history"));
    }

    #[test]
    fn a_legacy_v1_header_is_a_typed_schema_mismatch_not_payload_corruption() {
        let dir = tmp();
        let path = dir.path().join("journal.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"schema\":\"neoethos.autoresearch.journal.v1\",\"seq\":0,\"at_ms\":0,",
                "\"record\":{\"type\":\"SessionOpened\",\"session_id\":\"ar-legacy\",",
                "\"session_seed\":42,\"goal_hash\":\"fnv64:aaaa\",",
                "\"judge_hash\":\"fnv64:bbbb\",\"cost_hash\":\"fnv64:cccc\",",
                "\"goals\":[],\"scenarios_source\":\"legacy\",\"judge\":[],\"costs\":[],",
                "\"oos_window\":{\"start_ms\":900,\"end_ms\":1000},",
                "\"priors\":\"legacy\",\"budget\":[],\"identity_source\":\"legacy\",",
                "\"symbol\":\"EURUSD\"}}\n"
            ),
        )
        .unwrap();

        let err = Journal::open(&path).expect_err("v1 must not deserialize as a v2 record");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("journal.v1"), "got: {rendered}");
        assert!(rendered.contains("journal.v2"), "got: {rendered}");
        assert!(rendered.contains("not one history"), "got: {rendered}");
        assert!(
            !rendered.contains("COMPLETE line that does not parse"),
            "a known legacy envelope is a version mismatch, not corruption: {rendered}"
        );
    }
}
