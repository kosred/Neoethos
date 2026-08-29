//! The session — **a pure fold over the journal**, and the store it lives in.
//!
//! `docs/autoresearch-loop.md` §11.
//!
//! > **The in-memory [`Session`] is a pure fold over the journal. No state
//! > exists anywhere else.**
//!
//! That single invariant is what makes *"a crash loses one sweep, not the
//! session"* a property rather than an intention, and it is what makes
//! `N_session` un-resettable: there is no field holding it and no code path
//! that decrements it, so a loop cannot quietly restart its own trial count
//! between sweeps — which would be lying to itself with extra steps while every
//! deflated Sharpe in the report got easier to clear.
//!
//! Everything in this module is therefore one of exactly two things:
//!
//! * [`Session`] and its parts — **derived**, rebuilt by [`Session::fold`] from
//!   `journal.jsonl` on every open, never loaded from a snapshot;
//! * [`SessionStore`] — the directory, the immutable header, and the per-sweep
//!   artifact files, which are *evidence* rather than state. Deleting all of the
//!   artifacts would cost the loop its ability to re-derive a promotion's
//!   detail; deleting the journal would cost it the session.
//!
//! The equality between the two is asserted after **every** appended record by
//! [`SessionWriter::append`] in debug builds and by
//! `fold_matches_incremental_apply` in the tests.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::journal::{Journal, OosWindow, PosteriorCredit, Record, SearchRecord, SweepKind};

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// The three identifiers are DEFINED HERE and imported everywhere else.
//
// One definition, not two. `journal.rs`, `runner.rs`, `shuffle.rs`, `judge.rs`,
// `verdict.rs` and `proposal.rs` all name `crate::session::{SweepId, BlockId,
// SessionId}`; if each module defined its own, the proposer's `Session` and the
// shuffle's block bookkeeping would be talking about different types with the
// same name — the kind of split that compiles in each file and fails only where
// they meet.
//
// `session` owns them because the session IS the thing they index: a sweep id is
// a position in the fold, a block id is derived from `live_sweeps_run()`, and a
// session id names the journal the fold reads. `verdict` renders them; rendering
// is not ownership.
// ─────────────────────────────────────────────────────────────────────────────

/// One sweep — `SWEEP_SEARCHES` proposals drawn, run and judged as a single
/// experiment. Monotone within a session and never reused across a resume:
/// [`Session::next_sweep`] derives it from the highest id the fold ever saw.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(transparent)]
pub struct SweepId(pub u64);

/// `sweep-<n>`, and the format is LOAD-BEARING: [`SessionStore::sweep_dir`] is
/// `sweeps/{sweep}`, so this string is a DIRECTORY NAME, and
/// [`SessionStore::gc_trial_matrices`] reads those directories back by
/// `strip_prefix("sweep-")`. A prettier rendering with a space or a `#` in it
/// would produce directories the GC silently skips — every matrix kept forever,
/// no error, and a store that grows without bound over a multi-day session.
impl std::fmt::Display for SweepId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sweep-{}", self.0)
    }
}

/// One block — [`crate::SHUFFLE_PERIOD`] live sweeps, after which the block's
/// best sweep is re-run against rotated features (§9). Derived from the live
/// sweep count, never stored, so it cannot drift from the sweeps it names.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(transparent)]
pub struct BlockId(pub u32);

/// `block-<n>`, matching [`SweepId`]'s rendering so both are safe path
/// components even where only one of them is used as one today.
impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block-{}", self.0)
    }
}

/// A session — one goal set, one judge, one out-of-sample window, one journal.
///
/// The inner `String` is deliberately NOT public: it is also a directory name
/// under the autoresearch store, so every value must have come through
/// [`SessionId::new`] or [`SessionId::parse`], and [`SessionId::parse`] rejects
/// anything that is not a safe path component.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Construction and validation for [`SessionId`].
///
/// A session id is also a DIRECTORY NAME under the autoresearch store, so
/// [`SessionId::parse`] rejects anything that is not a safe path component: a
/// `--session ../../etc` would otherwise let a command-line argument address a
/// path outside the store.
impl SessionId {
    /// `ar-<utc>-<receipt-sha256>-<hash8>`. The full receipt identity makes the
    /// directory specific to the exact immutable generations and IS/OOS split;
    /// the final hash still separates different goal/judge/cost frames.
    pub fn new(
        now_utc: &str,
        goal_hash: &str,
        judge_hash: &str,
        cost_hash: &str,
        dataset_receipt_id: &DatasetReceiptId,
    ) -> Self {
        let fingerprint = crate::fnv64(&format!("{goal_hash}|{judge_hash}|{cost_hash}"));
        let short = fingerprint
            .trim_start_matches("fnv64:")
            .get(..8)
            .unwrap_or("00000000");
        Self(format!(
            "ar-{now_utc}-{}-{short}",
            dataset_receipt_id.as_str()
        ))
    }

    /// Parse an operator-supplied `--session` value.
    pub fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            bail!("a session id cannot be empty");
        }
        if !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            bail!(
                "session id {trimmed:?} contains characters other than [A-Za-z0-9_-]. The session \
                 id is also the directory name under the autoresearch store, so anything else \
                 would let it address a path outside it."
            );
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact dataset receipt
// ─────────────────────────────────────────────────────────────────────────────

pub const DATASET_RECEIPT_SCHEMA: &str = "neoethos.autoresearch.dataset-receipt.v1";
const DATASET_RECEIPT_ID_DOMAIN: &[u8] = b"neoethos.autoresearch.dataset-receipt.identity\0";
const DATASET_RECEIPT_ID_VERSION: u16 = 1;

/// Collision-resistant path key for one exact [`DatasetReceiptV1`]. The full
/// receipt is always persisted and exact-compared; this digest is only a safe,
/// bounded directory component.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DatasetReceiptId(String);

impl DatasetReceiptId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DatasetReceiptId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Exact immutable artifact selected for one direct timeframe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectTimeframeReceiptV1 {
    #[serde(with = "canonical_dataset_identity_serde")]
    pub dataset_identity: neoethos_data::CanonicalDatasetIdentity,
    pub manifest_schema_id: String,
    pub manifest_sha256: [u8; 32],
    pub generation_id: String,
    pub vortex_sha256: [u8; 32],
    pub row_count: u64,
    pub timestamp_start_ms: i64,
    pub timestamp_end_ms: i64,
}

impl DirectTimeframeReceiptV1 {
    pub fn from_artifact(artifact: &neoethos_data::CanonicalDatasetArtifactV1) -> Result<Self> {
        let binding = artifact.source_binding("dataset-receipt")?;
        Ok(Self {
            dataset_identity: artifact.identity().clone(),
            manifest_schema_id: binding.manifest_schema_id().to_owned(),
            manifest_sha256: *binding.manifest_hash(),
            generation_id: artifact.generation_id().to_owned(),
            vortex_sha256: *binding.vortex_hash(),
            row_count: artifact.row_count(),
            timestamp_start_ms: artifact.timestamp_start_ms(),
            timestamp_end_ms: artifact.timestamp_end_ms(),
        })
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.manifest_schema_id.trim().is_empty(),
            "direct timeframe receipt has an empty manifest schema id"
        );
        ensure!(
            !self.generation_id.trim().is_empty(),
            "direct timeframe receipt has an empty generation id"
        );
        ensure!(
            self.row_count > 0,
            "direct timeframe receipt {} has no rows",
            self.dataset_identity.to_path_component()
        );
        ensure!(
            self.timestamp_start_ms <= self.timestamp_end_ms,
            "direct timeframe receipt {} has inverted timestamp bounds",
            self.dataset_identity.to_path_component()
        );
        Ok(())
    }
}

/// Exact half-open in-sample interval. Every direct timeframe consumes only
/// rows whose canonical timestamp is in this interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InSampleWindowV1 {
    pub start_ms: i64,
    pub end_exclusive_ms: i64,
}

/// Frozen identity of all data one autoresearch session is allowed to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetReceiptV1 {
    pub schema: String,
    #[serde(with = "canonical_dataset_identity_serde")]
    pub anchor_identity: neoethos_data::CanonicalDatasetIdentity,
    pub direct_timeframes: Vec<DirectTimeframeReceiptV1>,
    pub in_sample_window: InSampleWindowV1,
    pub oos_window: OosWindow,
}

impl DatasetReceiptV1 {
    pub fn new(
        anchor_identity: neoethos_data::CanonicalDatasetIdentity,
        mut direct_timeframes: Vec<DirectTimeframeReceiptV1>,
        in_sample_window: InSampleWindowV1,
        oos_window: OosWindow,
    ) -> Result<Self> {
        direct_timeframes.sort_by(|left, right| {
            left.dataset_identity
                .to_path_component()
                .cmp(&right.dataset_identity.to_path_component())
        });
        let receipt = Self {
            schema: DATASET_RECEIPT_SCHEMA.to_owned(),
            anchor_identity,
            direct_timeframes,
            in_sample_window,
            oos_window,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn identity(&self) -> DatasetReceiptId {
        let digest = Sha256::digest(self.canonical_identity_bytes());
        DatasetReceiptId(format!("dr1-{}", encode_hex(&digest)))
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DATASET_RECEIPT_SCHEMA,
            "dataset receipt schema {:?} is not {DATASET_RECEIPT_SCHEMA}",
            self.schema
        );
        ensure!(
            self.in_sample_window.start_ms < self.in_sample_window.end_exclusive_ms,
            "dataset receipt in-sample window must be non-empty and half-open"
        );
        ensure!(
            self.in_sample_window.end_exclusive_ms == self.oos_window.start_ms,
            "dataset receipt IS cutoff {} does not equal OOS start {}",
            self.in_sample_window.end_exclusive_ms,
            self.oos_window.start_ms
        );
        ensure!(
            self.oos_window.start_ms <= self.oos_window.end_ms,
            "dataset receipt OOS window is inverted"
        );
        ensure!(
            !self.direct_timeframes.is_empty(),
            "dataset receipt has no direct timeframe artifacts"
        );

        let mut seen_timeframes = std::collections::BTreeSet::new();
        let mut anchor_artifact = None;
        let mut previous_identity = None::<String>;
        for direct in &self.direct_timeframes {
            direct.validate()?;
            ensure!(
                direct.dataset_identity.scope() == self.anchor_identity.scope()
                    && direct.dataset_identity.symbol_name() == self.anchor_identity.symbol_name()
                    && direct.dataset_identity.bar_timestamp_convention()
                        == self.anchor_identity.bar_timestamp_convention(),
                "direct timeframe {} is outside anchor series {}",
                direct.dataset_identity.to_path_component(),
                self.anchor_identity.to_path_component()
            );
            ensure!(
                seen_timeframes.insert(direct.dataset_identity.timeframe()),
                "dataset receipt repeats direct timeframe {}",
                direct.dataset_identity.timeframe()
            );
            let encoded = direct.dataset_identity.to_path_component();
            if let Some(previous) = &previous_identity {
                ensure!(
                    previous < &encoded,
                    "dataset receipt direct timeframes are not in canonical order"
                );
            }
            previous_identity = Some(encoded);
            if direct.dataset_identity == self.anchor_identity {
                anchor_artifact = Some(direct);
            }
        }
        let anchor_artifact = anchor_artifact.context(
            "dataset receipt does not contain the selected anchor identity as a direct artifact",
        )?;
        ensure!(
            self.in_sample_window.start_ms >= anchor_artifact.timestamp_start_ms
                && self.oos_window.end_ms <= anchor_artifact.timestamp_end_ms,
            "dataset receipt windows are outside the selected anchor generation"
        );
        Ok(())
    }

    fn canonical_identity_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(512 + self.direct_timeframes.len() * 256);
        bytes.extend_from_slice(DATASET_RECEIPT_ID_DOMAIN);
        bytes.extend_from_slice(&DATASET_RECEIPT_ID_VERSION.to_be_bytes());
        push_receipt_bytes(&mut bytes, self.schema.as_bytes());
        push_receipt_bytes(
            &mut bytes,
            self.anchor_identity.to_path_component().as_bytes(),
        );
        bytes.extend_from_slice(&(self.direct_timeframes.len() as u64).to_be_bytes());
        for direct in &self.direct_timeframes {
            push_receipt_bytes(
                &mut bytes,
                direct.dataset_identity.to_path_component().as_bytes(),
            );
            push_receipt_bytes(&mut bytes, direct.manifest_schema_id.as_bytes());
            bytes.extend_from_slice(&direct.manifest_sha256);
            push_receipt_bytes(&mut bytes, direct.generation_id.as_bytes());
            bytes.extend_from_slice(&direct.vortex_sha256);
            bytes.extend_from_slice(&direct.row_count.to_be_bytes());
            bytes.extend_from_slice(&direct.timestamp_start_ms.to_be_bytes());
            bytes.extend_from_slice(&direct.timestamp_end_ms.to_be_bytes());
        }
        bytes.extend_from_slice(&self.in_sample_window.start_ms.to_be_bytes());
        bytes.extend_from_slice(&self.in_sample_window.end_exclusive_ms.to_be_bytes());
        bytes.extend_from_slice(&self.oos_window.start_ms.to_be_bytes());
        bytes.extend_from_slice(&self.oos_window.end_ms.to_be_bytes());
        bytes
    }
}

fn push_receipt_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_be_bytes());
    target.extend_from_slice(value);
}

fn encode_hex(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

mod canonical_dataset_identity_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(
        identity: &neoethos_data::CanonicalDatasetIdentity,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&identity.to_path_component())
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<neoethos_data::CanonicalDatasetIdentity, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        neoethos_data::CanonicalDatasetIdentity::from_path_component(&encoded)
            .map_err(serde::de::Error::custom)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The immutable header
// ─────────────────────────────────────────────────────────────────────────────

/// `session.json` — written ONCE, fsynced, never rewritten.
///
/// It is the frame the whole session is judged inside. On resume the three
/// hashes are recomputed from the current config and compared; a difference
/// **refuses the resume, naming the field**. It does not migrate and does not
/// adapt: results judged by two different judges are not comparable and must
/// never share a history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionHeader {
    pub schema: String,
    pub session_id: SessionId,
    pub session_seed: u64,
    pub symbol: String,
    pub dataset_receipt: DatasetReceiptV1,
    pub created_utc: String,
    pub goal_hash: String,
    pub judge_hash: String,
    pub cost_hash: String,
    pub scenarios_source: String,
    pub identity_source: String,
    pub oos_window: OosWindow,
    /// The reproduction command, written at open so it exists even if the
    /// session never reaches a verdict.
    pub reproduction: String,
}

impl SessionHeader {
    fn validate(&self) -> Result<()> {
        self.dataset_receipt.validate()?;
        ensure!(
            self.dataset_receipt.anchor_identity.symbol_name() == self.symbol,
            "session header symbol {} disagrees with dataset receipt anchor symbol {}",
            self.symbol,
            self.dataset_receipt.anchor_identity.symbol_name()
        );
        ensure!(
            self.dataset_receipt.oos_window == self.oos_window,
            "session header OOS window disagrees with its exact dataset receipt"
        );
        Ok(())
    }
}

/// Why a resume was refused. Each variant names the hash that moved, because
/// *"the config changed"* is not actionable and *"the judge's PBO ceiling moved
/// from 0.50 to 0.60"* is.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeRefusal {
    GoalsChanged { stored: String, current: String },
    JudgeChanged { stored: String, current: String },
    CostModelChanged { stored: String, current: String },
    SymbolChanged { stored: String, current: String },
    DatasetReceiptChanged { stored: String, current: String },
    SchemaChanged { stored: String, current: String },
}

impl fmt::Display for ResumeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (what, why, stored, current) = match self {
            Self::GoalsChanged { stored, current } => (
                "goal_hash",
                "the GOAL CONSTANTS moved. Every sweep in this journal was run toward a different \
                 target, so continuing would mix two experiments under one N and one verdict",
                stored,
                current,
            ),
            Self::JudgeChanged { stored, current } => (
                "judge_hash",
                "a JUDGE THRESHOLD moved. Results judged by two different judges are not \
                 comparable; a session that resumed across a threshold change could reach any \
                 conclusion you liked by moving the bar half-way through",
                stored,
                current,
            ),
            Self::CostModelChanged { stored, current } => (
                "cost_hash",
                "the COST MODEL moved. Expectancy is a function of what a trade is charged, so \
                 the earlier sweeps' numbers no longer mean what the later ones will",
                stored,
                current,
            ),
            Self::SymbolChanged { stored, current } => (
                "symbol",
                "this session was opened on a different instrument",
                stored,
                current,
            ),
            Self::DatasetReceiptChanged { stored, current } => (
                "dataset_receipt",
                "the selected immutable generation set or the exact IS/OOS window moved. Results from two data receipts cannot share one history or one out-of-sample touch",
                stored,
                current,
            ),
            Self::SchemaChanged { stored, current } => (
                "schema",
                "this session was written by a different build's record format",
                stored,
                current,
            ),
        };
        write!(
            f,
            "RESUME REFUSED — {what} changed: stored {stored:?}, current {current:?}. {why}. Start \
             a NEW session (omit --session); this one is left untouched \
             (docs/autoresearch-loop.md §11.4)."
        )
    }
}

impl std::error::Error for ResumeRefusal {}

// ─────────────────────────────────────────────────────────────────────────────
// Fold products
// ─────────────────────────────────────────────────────────────────────────────

/// A `SweepStarted` with no outcome yet. On resume this is a sweep the process
/// died inside.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenSweep {
    pub sweep: SweepId,
    pub sweep_hash: String,
    pub kind: SweepKind,
    pub started_ms: i64,
}

/// A sweep that reached a terminal record.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepSummary {
    pub sweep: SweepId,
    pub kind: SweepKind,
    pub sweep_hash: String,
    pub trials_offered: usize,
    pub outcome: SweepOutcome,
    pub per_search: Vec<SearchRecord>,
    pub wall_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SweepOutcome {
    Completed,
    Failed(String),
    /// Found on resume: the process died between intent and outcome.
    Abandoned(String),
}

/// The best screened result the session has ever seen. R1/R2 evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestEver {
    pub sweep: SweepId,
    pub slot: usize,
    pub config_hash: String,
    /// Per-trade net expectancy of the best survivor at the PESSIMISTIC edge of
    /// the cost band. The optimistic edge is never the headline: a result that
    /// survives only at the cheap end of a cost nobody can pin down is not a
    /// result.
    pub e_screen_pess: f64,
    pub n_trades: usize,
    pub dsr: Option<f64>,
    pub pbo: Option<f64>,
}

/// One row of the **session champion matrix** — the input to `pbo_session`.
///
/// One row per sweep, carrying that sweep's champion's monthly return series.
/// This is PBO of the **loop's own selection procedure**, which is the object
/// the operator's instruction names; `pbo_sweep` is PBO within a single sweep.
/// Both must clear 0.50.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChampionRow {
    pub sweep: SweepId,
    pub config_hash: String,
    pub strategy_id: String,
    /// Month keys, so the judge can align rows drawn from sweeps whose windows
    /// differ. Rows that cannot be aligned are dropped BY THE JUDGE, with a
    /// count — never here, silently.
    pub period_keys: Vec<i64>,
    pub monthly_returns: Vec<f64>,
    pub e_screen_pess: f64,
}

/// How often a factor level has been drawn. R2/U4 needs both numbers: *a space
/// is not refuted if it was not searched.*
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LevelCoverage {
    /// Individual proposals carrying this level.
    pub draws: usize,
    /// Distinct sweeps in which it appeared at least once — *"a sweep's worth"*.
    pub sweeps: usize,
}

/// Coverage per axis-A factor level and per axis-B variant.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoverageCounters {
    levels: BTreeMap<(String, String), LevelCoverage>,
    seen_in_sweep: BTreeSet<(String, String, u64)>,
}

impl CoverageCounters {
    fn credit(&mut self, sweep: SweepId, factor: &str, level: &str) {
        let key = (factor.to_string(), level.to_string());
        let entry = self.levels.entry(key.clone()).or_default();
        entry.draws += 1;
        if self
            .seen_in_sweep
            .insert((key.0.clone(), key.1.clone(), sweep.0))
        {
            entry.sweeps += 1;
        }
    }

    pub fn get(&self, factor: &str, level: &str) -> LevelCoverage {
        self.levels
            .get(&(factor.to_string(), level.to_string()))
            .copied()
            .unwrap_or_default()
    }

    /// Every `(factor, level, coverage)` seen so far, in stable order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, LevelCoverage)> {
        self.levels
            .iter()
            .map(|((f, l), c)| (f.as_str(), l.as_str(), *c))
    }

    /// Levels of `factor` with fewer than `min_sweeps` sweeps' worth of draws.
    /// The proposer's coverage debt reads this; R2/U4 reads it too.
    pub fn under_covered(&self, factor: &str, min_sweeps: usize) -> Vec<(String, LevelCoverage)> {
        self.levels
            .iter()
            .filter(|((f, _), c)| f == factor && c.sweeps < min_sweeps)
            .map(|((_, l), c)| (l.clone(), *c))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// The report shape (`verdict::CoverageSummary`), which is the evidence
    /// behind U4 and the body of `SpaceDescription`.
    ///
    /// The split is by factor name: `"objective"` is axis B, everything else —
    /// the eleven axis-A factors and the four refusal dimensions — is axis A.
    /// The refusal dimensions belong on the A side because U4 requires each of
    /// their levels to have been drawn at least once, the same bar the axis-A
    /// levels carry; only the objective variants carry the `B_MIN_DRAWS = 3`
    /// bar.
    pub fn to_summary(&self) -> crate::verdict::CoverageSummary {
        let mut axis_a = Vec::new();
        let mut axis_b = Vec::new();
        for ((factor, level), c) in &self.levels {
            let entry = crate::verdict::CoverageEntry {
                factor: factor.clone(),
                level: level.clone(),
                sweeps_drawn: c.sweeps,
                proposals_drawn: c.draws,
            };
            if factor == "objective" {
                axis_b.push(entry);
            } else {
                axis_a.push(entry);
            }
        }
        crate::verdict::CoverageSummary { axis_a, axis_b }
    }
}

/// The folded Beta–Bernoulli evidence, per `(factor, level)`.
///
/// The **proposer** samples from this; the **session** owns it, because it is a
/// fold over `PosteriorUpdated` and `PosteriorRetracted` and nothing else. That
/// split is what makes retraction exact: when a block is judged
/// INDISTINGUISHABLE from its shuffle control, every success it credited is
/// flipped to a failure by replaying its own credits, not by an approximate
/// decrement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PosteriorLedger {
    counts: BTreeMap<(String, String), (u64, u64)>,
    credited_by_sweep: BTreeMap<u64, Vec<PosteriorCredit>>,
    retracted_sweeps: BTreeSet<u64>,
}

impl PosteriorLedger {
    fn credit(&mut self, sweep: SweepId, credits: &[PosteriorCredit]) {
        for c in credits {
            let entry = self
                .counts
                .entry((c.factor.clone(), c.level.clone()))
                .or_insert((0, 0));
            if c.success {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
        self.credited_by_sweep
            .entry(sweep.0)
            .or_default()
            .extend_from_slice(credits);
    }

    fn retract(&mut self, sweeps: &[SweepId]) {
        for sweep in sweeps {
            if !self.retracted_sweeps.insert(sweep.0) {
                // Already retracted. Retracting twice would turn one lucky block
                // into two penalties, which is a different distortion, not a
                // safer one.
                continue;
            }
            let Some(credits) = self.credited_by_sweep.get(&sweep.0).cloned() else {
                continue;
            };
            for c in credits.iter().filter(|c| c.success) {
                let entry = self
                    .counts
                    .entry((c.factor.clone(), c.level.clone()))
                    .or_insert((0, 0));
                entry.0 = entry.0.saturating_sub(1);
                entry.1 += 1;
            }
        }
    }

    /// `(successes, failures)` for a level. The proposer adds the `Beta(1, 1)`
    /// prior itself — the prior is the proposer's, the evidence is the
    /// session's.
    pub fn evidence(&self, factor: &str, level: &str) -> (u64, u64) {
        self.counts
            .get(&(factor.to_string(), level.to_string()))
            .copied()
            .unwrap_or((0, 0))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, u64, u64)> {
        self.counts
            .iter()
            .map(|((f, l), (s, fl))| (f.as_str(), l.as_str(), *s, *fl))
    }

    pub fn retracted_sweeps(&self) -> impl Iterator<Item = SweepId> + '_ {
        self.retracted_sweeps.iter().copied().map(SweepId)
    }
}

/// A block's judgement against its shuffle control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockOutcome {
    pub block: BlockId,
    pub delta_expectancy: f64,
    pub p_block: f64,
    pub dsr_gap: f64,
    pub indistinguishable: bool,
}

// §12 — every abandoned configuration, counted AND named.
//
// DEFINED HERE and carried whole into `SessionVerdict`. One definition, one
// shape in the fold and in the report — a second struct with the same name and
// different fields is how a count in the log stops matching the count in the
// verdict.
//
// NO SILENT DROPS (non-negotiable 4) is this struct: every way a configuration
// can leave the loop without a result has a counter, and each counter is
// incremented at exactly one place, in `Session::apply`.

/// Every abandoned configuration, counted and named.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AbandonCensus {
    /// Proposals that reached the journal. The denominator for everything else.
    pub proposals_drawn: usize,
    /// Refused because an identical `config_hash` had already been run — the
    /// same experiment twice is one experiment, not two trials.
    pub duplicates_refused: usize,
    /// Refused BEFORE running: the payoff floor is unreachable under the
    /// proposal's own exit geometry, so its answer was fixed before a bar was
    /// read.
    pub payoff_floor_unreachable: usize,
    /// Refused because the proposal left the trust region and was reverted.
    pub trust_region_reverted: usize,
    /// Individual searches inside a sweep that returned an error.
    pub searches_errored: usize,
    /// Whole sweeps that failed with an error.
    pub sweeps_failed: usize,
    /// Whole sweeps found open on resume: the process died between the intent
    /// record and the outcome record.
    pub sweeps_abandoned: usize,
    /// Screen failures by conjunct, indexed by
    /// [`crate::judge::ScreenConjunct::index`]: `S1..S6` plus `Unavailable`.
    pub screen_failed_by_conjunct: [usize; 7],
    /// Trial-return matrices deleted by the store's garbage collector.
    pub gc_deleted_matrices: usize,
    pub gc_bytes_reclaimed: u64,
    /// `(reason, config_hash)` for the first [`CENSUS_EXAMPLE_LIMIT`] drops. A
    /// count tells a reader how many died; an example tells them which question
    /// to ask next, which is why the census carries both.
    pub examples: Vec<(String, String)>,
}

impl AbandonCensus {
    /// Everything that left the loop without a result. Counted, named, and
    /// **never elided when zero.**
    ///
    /// A renderer that prints only the non-zero rows lets a reader mistake "this
    /// counter does not exist" for "this counter is zero", which is the exact
    /// difference between a drop that was measured and a drop that was never
    /// instrumented. Every row is printed on every report.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "ABANDON CENSUS — every configuration that left without a result"
        );
        let _ = writeln!(
            s,
            "  proposals drawn                 {}",
            self.proposals_drawn
        );
        let _ = writeln!(
            s,
            "  refused: duplicate config_hash  {}",
            self.duplicates_refused
        );
        let _ = writeln!(
            s,
            "  refused: payoff floor unreachable {}",
            self.payoff_floor_unreachable
        );
        let _ = writeln!(
            s,
            "  refused: trust region reverted  {}",
            self.trust_region_reverted
        );
        let _ = writeln!(
            s,
            "  searches errored                {}",
            self.searches_errored
        );
        let _ = writeln!(
            s,
            "  sweeps failed                   {}",
            self.sweeps_failed
        );
        let _ = writeln!(
            s,
            "  sweeps abandoned (crash)        {}",
            self.sweeps_abandoned
        );
        let _ = writeln!(s, "  screen failures by conjunct:");
        for (i, label) in crate::judge::ScreenConjunct::ALL_LABELS.iter().enumerate() {
            let _ = writeln!(s, "    {label:<16} {}", self.screen_failed_by_conjunct[i]);
        }
        let _ = writeln!(
            s,
            "  trial matrices GC'd             {} ({} bytes reclaimed)",
            self.gc_deleted_matrices, self.gc_bytes_reclaimed
        );
        if self.examples.is_empty() {
            let _ = writeln!(s, "  named examples: none");
        } else {
            let _ = writeln!(
                s,
                "  named examples (first {} of the drops):",
                self.examples.len()
            );
            for (reason, config_hash) in &self.examples {
                let _ = writeln!(s, "    {config_hash}  {reason}");
            }
        }
        s
    }
}

/// How many named examples the census keeps. Bounded because it is rendered in
/// every report and a session runs for days.
pub const CENSUS_EXAMPLE_LIMIT: usize = 32;

/// Record a named example against the census.
///
/// A free function rather than an inherent method so the census stays exactly
/// the shape `verdict` declares. A count tells a reader how many died; an
/// example tells them which question to ask next, which is why the census
/// carries both.
fn note(census: &mut AbandonCensus, reason: &str, config_hash: &str) {
    if census.examples.len() < CENSUS_EXAMPLE_LIMIT {
        census
            .examples
            .push((reason.to_string(), config_hash.to_string()));
    }
}

/// Does `candidate` strictly improve on the current best?
///
/// **One definition, used twice**: [`SessionWriter::offer_best_ever`] asks it
/// before deciding whether a record is worth journalling, and
/// [`Session::apply`] asks it again when folding that record. If the two sides
/// used different rules, a resumed session would compute a different
/// `best_ever` from the same journal than the process that wrote it — which is
/// precisely the divergence §11.4's assertion exists to catch, moved one level
/// down where the assertion could not see it.
///
/// Strict `>`, so an equal result does not displace the incumbent: the earlier
/// one is the one whose trial matrices the GC has been keeping, and swapping to
/// a tie would discard evidence for no gain. A NaN candidate loses every
/// comparison and can therefore never become the best — the right answer, since
/// "no number" is never good news.
pub fn advances_best(current: Option<&BestEver>, candidate: &BestEver) -> bool {
    match current {
        None => candidate.e_screen_pess.is_finite(),
        Some(best) => candidate.e_screen_pess > best.e_screen_pess,
    }
}

/// What the loop hands the judge at S6: one sweep's evidence, complete.
///
/// The loop owns this type because the loop owns S5 COLLECT. The judge consumes
/// it and returns verdicts; neither side reaches across.
#[derive(Debug, Clone)]
pub struct SweepEvidence {
    pub sweep: SweepId,
    pub kind: SweepKind,
    pub sweep_hash: String,
    /// Index-aligned with `statistics`.
    pub per_search: Vec<SearchRecord>,
    pub statistics: Vec<neoethos_search::deflated::TrialStatisticsReport>,
    /// The champion row **each slot** offers, index-aligned with `per_search`.
    /// `None` where that search produced no per-period champion series at all.
    ///
    /// One row PER SLOT, never one pre-selected row. The runner used to hand the
    /// judge a single row — the argmax of `e_screen_pess` over every non-errored
    /// search, SCREENED OR NOT — and the judge then stamped it with the
    /// `config_hash` and expectancy of the best *passing* slot. When those two
    /// argmaxes disagreed (the routine case: the highest-expectancy slot is
    /// typically the overfit outlier that fails S1/S2/S5) the session champion
    /// matrix received a REJECTED configuration's return series wearing the
    /// winner's identity, and the stamping is what hid it. `pbo_session` — the
    /// PBO of the loop's own selection procedure, which gates every promotion —
    /// is computed from exactly those rows.
    ///
    /// So the selection happens ONCE, in the judge, over slots that passed.
    pub champion_rows: Vec<Option<ChampionRow>>,
    pub trials_offered: usize,
    pub wall_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// The session
// ─────────────────────────────────────────────────────────────────────────────

/// Everything that carries between iterations — and nothing else.
///
/// `docs/autoresearch-loop.md` §3.1. Every field here is produced by
/// [`Session::apply`] from a journal record; there is no constructor that takes
/// values, and no setter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Session {
    /// From `SessionOpened`. `None` on a journal whose first record has not
    /// been written yet.
    pub opened: Option<OpenedHeader>,
    /// Terminal sweep records, in order.
    pub sweeps: Vec<SweepSummary>,
    /// A `SweepStarted` whose outcome has not arrived. On resume: a crash.
    pub open_sweep: Option<OpenSweep>,
    /// `config_hash -> the sweep that first proposed it`. Duplicate refusal.
    pub config_hashes: BTreeMap<String, SweepId>,
    pub coverage: CoverageCounters,
    pub posterior: PosteriorLedger,
    pub blocks: Vec<BlockOutcome>,
    /// The shuffle null: every control sweep's best `E_screen_pess`, in block
    /// order.
    pub null_observations: Vec<f64>,
    /// Controls that produced NO number. Carried separately because
    /// `null_observations` is a `Vec<f64>` on a `PartialEq` struct and a NaN
    /// sentinel would make the session unequal to itself.
    ///
    /// A refusal is not an observation and it is not nothing: `shuffle.rs`
    /// promises every control is *"counted, never dropped"*, and this field is
    /// where the count lives. Without it `ShuffleSummary::null_refusals` — which
    /// is rendered under EVERY headline the loop prints — was structurally zero.
    pub null_refusals: usize,
    /// The best screened result the session has ever seen.
    ///
    /// Folded from [`Record::BestEverAdvanced`] and from nothing else. It was
    /// once assigned through a `&mut Session` and the cost was exact: a debug
    /// build tripped the §11.4 assertion on the next append, and a release build
    /// silently dropped the pointer on every restart — which takes R2's U2
    /// condition with it, so the loop could no longer report the goal
    /// UNREACHABLE, and un-keeps the previous best sweep's trial matrices at the
    /// next GC, so the evidence went too.
    pub best_ever: Option<BestEver>,
    /// One row per sweep — the `pbo_session` input.
    ///
    /// Folded from [`Record::ChampionRecorded`], for the same reason: a champion
    /// matrix that restarts empty makes `pbo_session` refuse forever, and a
    /// refusal is a fail, so NO PROMOTION IS POSSIBLE after any restart.
    pub champions: Vec<ChampionRow>,
    pub oos_touches_spent: u32,
    pub census: AbandonCensus,
    /// Set by `SessionStopped`. A stopped session is never resumed into more
    /// sweeps.
    pub stopped: Option<crate::verdict::SessionVerdict>,
    /// Bytes of partial journal lines truncated on resume. Carried here because
    /// the report census has no field for it; it is ALSO recorded as a named
    /// census example, so it cannot fall out of a report.
    pub journal_truncated_bytes: usize,
    /// Every terminal sweep's `trials_offered`, in order. `n_session` is its
    /// sum, and nothing else ever touches it.
    terminal_trials: Vec<(SweepId, usize)>,
    /// The highest sweep id ever seen, so a resumed session never reuses one.
    highest_sweep: u64,
    /// Proposals drawn, by sweep — so the shuffle control can re-run a sweep's
    /// EXACT proposals, byte for byte.
    proposals: BTreeMap<u64, Vec<crate::proposal::Proposal>>,
    /// Screen results, by sweep, index-aligned with that sweep's proposals.
    screens: BTreeMap<u64, Vec<crate::judge::ScreenResult>>,
}

/// The header fields, as folded from `SessionOpened`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenedHeader {
    pub session_id: SessionId,
    pub session_seed: u64,
    pub symbol: String,
    pub dataset_receipt: DatasetReceiptV1,
    pub goal_hash: String,
    pub judge_hash: String,
    pub cost_hash: String,
    pub oos_window: OosWindow,
    pub scenarios_source: String,
    pub identity_source: String,
}

impl Session {
    /// Rebuild the session from its journal. **The only constructor.**
    pub fn fold(records: &[Record]) -> Result<Self> {
        let mut session = Self::default();
        for record in records {
            session.apply(record)?;
        }
        Ok(session)
    }

    /// One fold step.
    ///
    /// The `match` is exhaustive by construction: a new [`Record`] variant that
    /// does not gain an arm here does not compile, which is what stops an event
    /// type from being added and then silently ignored by the state that every
    /// verdict is computed from.
    pub fn apply(&mut self, record: &Record) -> Result<()> {
        match record {
            Record::SessionOpened {
                session_id,
                session_seed,
                symbol,
                goal_hash,
                judge_hash,
                cost_hash,
                oos_window,
                scenarios_source,
                identity_source,
                dataset_receipt,
                ..
            } => {
                if self.opened.is_some() {
                    bail!(
                        "the journal carries TWO SessionOpened records. A session has exactly one \
                         frame — one goal set, one judge, one OOS window — and a second header \
                         would mean two experiments folded into one N and one verdict."
                    );
                }
                dataset_receipt.validate()?;
                ensure!(
                    dataset_receipt.anchor_identity.symbol_name() == symbol,
                    "SessionOpened symbol disagrees with its dataset receipt anchor"
                );
                ensure!(
                    dataset_receipt.oos_window == *oos_window,
                    "SessionOpened OOS window disagrees with its dataset receipt"
                );
                self.opened = Some(OpenedHeader {
                    session_id: session_id.clone(),
                    session_seed: *session_seed,
                    symbol: symbol.clone(),
                    dataset_receipt: dataset_receipt.clone(),
                    goal_hash: goal_hash.clone(),
                    judge_hash: judge_hash.clone(),
                    cost_hash: cost_hash.clone(),
                    oos_window: *oos_window,
                    scenarios_source: scenarios_source.clone(),
                    identity_source: identity_source.clone(),
                });
            }

            Record::ProposalDrawn {
                sweep,
                slot,
                proposal,
                ..
            } => {
                self.note_sweep(*sweep);
                self.census.proposals_drawn += 1;
                self.config_hashes
                    .entry(proposal.config_hash.clone())
                    .or_insert(*sweep);
                for (factor, level) in proposal.coverage_levels() {
                    self.coverage.credit(*sweep, factor, &level);
                }
                let slots = self.proposals.entry(sweep.0).or_default();
                if slots.len() <= *slot {
                    slots.resize_with(*slot + 1, || proposal.clone());
                }
                slots[*slot] = proposal.clone();
            }

            Record::ProposalDuplicate {
                sweep, config_hash, ..
            } => {
                self.note_sweep(*sweep);
                self.census.duplicates_refused += 1;
                note(&mut self.census, "duplicate", config_hash);
            }

            Record::ProposalRefused {
                sweep,
                config_hash,
                reason,
                ..
            } => {
                self.note_sweep(*sweep);
                // A proposal whose payoff floor is unreachable under its own
                // geometry is refused BEFORE it runs: its answer was fixed
                // before a bar was read.
                if reason.contains("payoff") || reason.contains("floor") {
                    self.census.payoff_floor_unreachable += 1;
                } else if reason.contains("trust region") {
                    self.census.trust_region_reverted += 1;
                }
                note(&mut self.census, reason, config_hash);
            }

            Record::SweepStarted {
                sweep,
                sweep_hash,
                kind,
                started_ms,
            } => {
                if let Some(open) = &self.open_sweep {
                    bail!(
                        "the journal carries a SweepStarted for {sweep} while {} is still open. \
                         Two-phase journalling requires an outcome (or a SweepAbandoned written \
                         on resume) before the next intent, or the fold cannot tell which sweep a \
                         later record belongs to.",
                        open.sweep
                    );
                }
                self.note_sweep(*sweep);
                self.open_sweep = Some(OpenSweep {
                    sweep: *sweep,
                    sweep_hash: sweep_hash.clone(),
                    kind: kind.clone(),
                    started_ms: *started_ms,
                });
            }

            Record::SweepCompleted {
                sweep,
                trials_offered,
                per_search,
                wall_ms,
            } => {
                let open = self.take_open(*sweep)?;
                self.census.searches_errored +=
                    per_search.iter().filter(|s| s.error.is_some()).count();
                for failed in per_search.iter().filter(|s| s.error.is_some()) {
                    if let Some(err) = &failed.error {
                        note(&mut self.census, err, &failed.config_hash);
                    }
                }
                self.sweeps.push(SweepSummary {
                    sweep: *sweep,
                    kind: open.kind,
                    sweep_hash: open.sweep_hash,
                    trials_offered: *trials_offered,
                    outcome: SweepOutcome::Completed,
                    per_search: per_search.clone(),
                    wall_ms: *wall_ms,
                });
                self.terminal_trials.push((*sweep, *trials_offered));
            }

            Record::SweepFailed {
                sweep,
                trials_offered,
                error,
            } => {
                let open = self.take_open(*sweep)?;
                self.census.sweeps_failed += 1;
                note(&mut self.census, error, &open.sweep_hash);
                self.sweeps.push(SweepSummary {
                    sweep: *sweep,
                    kind: open.kind,
                    sweep_hash: open.sweep_hash,
                    trials_offered: *trials_offered,
                    outcome: SweepOutcome::Failed(error.clone()),
                    per_search: Vec::new(),
                    wall_ms: 0,
                });
                // A FAILED sweep's trials still ran, so they still count.
                self.terminal_trials.push((*sweep, *trials_offered));
            }

            Record::SweepAbandoned {
                sweep,
                trials_offered,
                reason,
            } => {
                let open = self.take_open(*sweep)?;
                self.census.sweeps_abandoned += 1;
                note(&mut self.census, reason, &open.sweep_hash);
                self.sweeps.push(SweepSummary {
                    sweep: *sweep,
                    kind: open.kind,
                    sweep_hash: open.sweep_hash,
                    trials_offered: *trials_offered,
                    outcome: SweepOutcome::Abandoned(reason.clone()),
                    per_search: Vec::new(),
                    wall_ms: 0,
                });
                // A trial that ran is a trial that ran.
                self.terminal_trials.push((*sweep, *trials_offered));
            }

            Record::Screened {
                sweep,
                slot,
                screen_result,
                ..
            } => {
                self.note_sweep(*sweep);
                if let Some(index) = screen_result.census_index()
                    && index < self.census.screen_failed_by_conjunct.len()
                {
                    self.census.screen_failed_by_conjunct[index] += 1;
                }
                let slots = self.screens.entry(sweep.0).or_default();
                // A GAP IS NOT A REPEAT. The vector is keyed by SLOT and the
                // records arrive by slot, so a record for slot k landing in a
                // vector of length j < k means slots j..k have no screen at all.
                // Filling them with a CLONE of the arriving result made every
                // one of them a Pass carrying slot k's statistics — and
                // `promotion_candidate` selects out of exactly this vector, so
                // the fabricated passes competed for the one irreplaceable
                // out-of-sample touch. They are filled with a REFUSAL instead:
                // "nothing could be judged here" is the truth, and it is a
                // truth that can never be promoted.
                while slots.len() <= *slot {
                    let missing = slots.len();
                    slots.push(crate::judge::ScreenResult::Unavailable {
                        detail: format!(
                            "no Screened record has arrived for slot {missing} of {sweep}. The \
                             screen vector is keyed by slot, so this slot was never judged — it \
                             is NOT the verdict of the neighbouring slot, and it can never be a \
                             promotion candidate."
                        ),
                    });
                }
                slots[*slot] = screen_result.clone();
            }

            Record::BestEverAdvanced { best } => {
                // MONOTONE, not blind assignment. The writer only appends a
                // record that strictly improves, so on a forward replay the two
                // are the same thing; taking the max as well means the folded
                // pointer cannot be walked BACKWARDS by a record a reader
                // replays out of order, and makes the fold idempotent under a
                // duplicated line.
                self.note_sweep(best.sweep);
                if advances_best(self.best_ever.as_ref(), best) {
                    self.best_ever = Some(best.clone());
                }
            }

            Record::ChampionRecorded { row } => {
                self.note_sweep(row.sweep);
                // One row per sweep is what `pbo_session` is defined over. Two
                // rows from one sweep would let a single sweep vote twice in the
                // PBO of the loop's own selection procedure, which is a
                // measurement that flatters itself — so this is refused rather
                // than de-duplicated, because a de-duplication would hide the
                // caller that wrote it twice.
                if let Some(existing) = self.champions.iter().find(|c| c.sweep == row.sweep) {
                    bail!(
                        "the journal carries TWO ChampionRecorded records for {} (config {} then \
                         {}). The session champion matrix is defined as ONE ROW PER SWEEP — the \
                         input to pbo_session, the PBO of the loop's own selection procedure — so \
                         a second row would let one sweep vote twice in the statistic that decides \
                         whether the loop is fooling itself.",
                        row.sweep,
                        existing.config_hash,
                        row.config_hash
                    );
                }
                self.champions.push(row.clone());
            }

            Record::PosteriorUpdated { sweep, credits } => {
                self.posterior.credit(*sweep, credits);
            }

            Record::PosteriorRetracted { sweep_ids, .. } => {
                self.posterior.retract(sweep_ids);
            }

            Record::ShuffleControlCompleted {
                block,
                e_screen_pess,
                source_sweep,
                ..
            } => {
                // ── N_session ────────────────────────────────────────────────
                //
                // NOT credited here. The control's trials count into N_session
                // exactly like a live sweep's — the null is not free — and that
                // is precisely how they are already counted: a control IS a
                // sweep. `run_block_control` drives it through `run_sweep`,
                // which appends its own `SweepCompleted` with this same
                // `trials_offered`, and the arm above credits it. Crediting it
                // a second time here, under the SOURCE sweep's id, inflated
                // N_session by one control per block — about 20% — which does
                // not flatter the deflated statistics but does make every
                // headline N a number no other artifact can reproduce.
                //
                // `trials_offered` is left in the record: it is what makes the
                // control's own line readable by a person holding only
                // `journal.jsonl`.

                // ── the null ─────────────────────────────────────────────────
                match e_screen_pess {
                    Some(e) => self.null_observations.push(*e),
                    // A control that produced NO number is COUNTED, never
                    // dropped. `shuffle.rs` promises exactly this and the fold
                    // used to break the promise silently: `null_refusals` was
                    // structurally zero under every headline the loop printed,
                    // so a session whose controls were all broken read as a
                    // session with a clean null. It is also NAMED in the
                    // census, because a counter with no example beside it is a
                    // number nobody can act on.
                    None => {
                        self.null_refusals += 1;
                        note(
                            &mut self.census,
                            "a shuffle control produced no E_screen_pess, so it joins the null as \
                             a REFUSAL and not as an observation",
                            &format!("{block} (source {source_sweep})"),
                        );
                    }
                }
            }

            Record::BlockJudged {
                block,
                delta_expectancy,
                p_block,
                dsr_gap,
                indistinguishable,
            } => {
                self.blocks.push(BlockOutcome {
                    block: *block,
                    delta_expectancy: *delta_expectancy,
                    p_block: *p_block,
                    dsr_gap: *dsr_gap,
                    indistinguishable: *indistinguishable,
                });
            }

            Record::OosTouchSpent { .. } => {
                self.oos_touches_spent += 1;
            }

            Record::Promoted { .. } | Record::PromotionRefused { .. } => {}

            Record::GarbageCollected { sweep, census } => {
                self.note_sweep(*sweep);
                // THE TWO COUNTERS THAT WERE RENDERED BY EVERY VERDICT AND
                // ASSIGNED BY NOBODY.
                //
                // `AbandonCensus::render` prints "trial matrices GC'd N (B bytes
                // reclaimed)" on every report and `census_line` prints it at
                // every stop, while `SessionStore::gc_trial_matrices` was
                // deleting every non-kept sweep's matrices with only a
                // `tracing::info!` to show for it. There was no record, so the
                // fold had nothing to fold, so the numbers were structurally
                // zero — the operator was told nothing had been deleted by a
                // session that had deleted all of it.
                //
                // Accumulated across passes rather than overwritten: the census
                // is a session total, and a GC that ran fifteen times must not
                // report only the fifteenth.
                let first_pass_that_reclaimed =
                    self.census.gc_deleted_matrices == 0 && census.deleted > 0;
                self.census.gc_deleted_matrices += census.deleted;
                self.census.gc_bytes_reclaimed += census.bytes_reclaimed;
                // The RULE, named once, on the first pass that actually deleted
                // something — not on every pass: the example list is capped at
                // CENSUS_EXAMPLE_LIMIT and is rendered in full, so a line per
                // pass would crowd out the named drops it exists to carry.
                if first_pass_that_reclaimed {
                    note(&mut self.census, &census.rule, &format!("gc after {sweep}"));
                }
            }

            Record::SessionStopped { verdict } => {
                self.stopped = Some(verdict.clone());
            }

            Record::TruncatedTail { bytes } => {
                // Not a field of the report census, so it is carried on the
                // session AND pushed into the census's named examples. A
                // truncated tail that appears in neither would be exactly the
                // silent absorption §11.2 forbids.
                self.journal_truncated_bytes += bytes;
                note(
                    &mut self.census,
                    "journal truncated tail (process killed mid-write)",
                    &format!("{bytes} bytes"),
                );
            }
        }
        Ok(())
    }

    fn note_sweep(&mut self, sweep: SweepId) {
        self.highest_sweep = self.highest_sweep.max(sweep.0);
    }

    fn take_open(&mut self, sweep: SweepId) -> Result<OpenSweep> {
        match self.open_sweep.take() {
            Some(open) if open.sweep == sweep => Ok(open),
            Some(open) => {
                self.open_sweep = Some(open.clone());
                bail!(
                    "an outcome record for {sweep} arrived while {} was the open sweep. The \
                     journal's two-phase ordering is broken and the fold cannot attribute this \
                     outcome.",
                    open.sweep
                )
            }
            None => bail!(
                "an outcome record for {sweep} arrived with NO open sweep. Every outcome must be \
                 preceded by its SweepStarted intent — that ordering is what makes \"a crash \
                 loses one sweep, not the session\" true."
            ),
        }
    }

    // ── derived reads ───────────────────────────────────────────────────────

    /// The honest N. Equal to [`crate::journal::n_session`] over the same
    /// records, by construction and by test.
    pub fn n_session(&self) -> usize {
        self.terminal_trials.iter().map(|(_, n)| *n).sum()
    }

    /// The next sweep id. Monotone across resumes: a reused id would let two
    /// experiments share one artifact directory.
    pub fn next_sweep(&self) -> SweepId {
        SweepId(self.highest_sweep + 1)
    }

    pub fn sweeps_run(&self) -> usize {
        self.sweeps.len()
    }

    pub fn live_sweeps_run(&self) -> usize {
        self.sweeps.iter().filter(|s| s.kind.is_live()).count()
    }

    pub fn searches_run(&self) -> usize {
        self.sweeps.iter().map(|s| s.per_search.len()).sum()
    }

    /// The block index the NEXT live sweep belongs to.
    pub fn current_block(&self) -> BlockId {
        BlockId((self.live_sweeps_run() / crate::SHUFFLE_PERIOD) as u32)
    }

    /// True when the live sweeps just completed close a block, so the shuffle
    /// control is due.
    pub fn block_boundary_due(&self) -> bool {
        let live = self.live_sweeps_run();
        live > 0
            && live % crate::SHUFFLE_PERIOD == 0
            && self.blocks.len() < live / crate::SHUFFLE_PERIOD
    }

    pub fn indistinguishable_blocks(&self) -> usize {
        self.blocks.iter().filter(|b| b.indistinguishable).count()
    }

    /// The proposals of a sweep, in draw order — what the shuffle control
    /// re-runs byte for byte.
    pub fn proposals_of(&self, sweep: SweepId) -> Option<&[crate::proposal::Proposal]> {
        self.proposals.get(&sweep.0).map(|v| v.as_slice())
    }

    pub fn screens_of(&self, sweep: SweepId) -> Option<&[crate::judge::ScreenResult]> {
        self.screens.get(&sweep.0).map(|v| v.as_slice())
    }

    /// The live sweeps of a block, best-screening first is NOT applied here —
    /// the judge decides "best". This returns membership only.
    pub fn live_sweeps_in_block(&self, block: BlockId) -> Vec<&SweepSummary> {
        self.sweeps
            .iter()
            .filter(|s| s.kind.is_live())
            .enumerate()
            .filter(|(index, _)| (*index / crate::SHUFFLE_PERIOD) as u32 == block.0)
            .map(|(_, s)| s)
            .collect()
    }

    pub fn has_stopped(&self) -> bool {
        self.stopped.is_some()
    }

    /// Has this session's one OOS touch been spent?
    pub fn oos_spent(&self) -> bool {
        self.oos_touches_spent > 0
    }

    /// Distinct `config_hash`es the session ever stamped — the SIZE of the space
    /// an "unreachable" verdict is a claim about.
    ///
    /// R2 says *"no configuration in THIS SPACE reaches the goal"*, and a
    /// refutation with no number attached to "this space" is not checkable.
    pub fn distinct_configurations(&self) -> usize {
        self.config_hashes.len()
    }

    /// One line, the shape the batch ledger and the indicator ledger already
    /// use. The verdict renders the full census; this is what goes in the log at
    /// every stop, so a tail of the run always ends with the drop counts.
    pub fn census_line(&self) -> String {
        format!(
            "ABANDON CENSUS: drawn={} duplicates={} unreachable_floor={} trust_region={} \
             searches_errored={} sweeps_failed={} sweeps_abandoned={} \
             screen_failed_by_conjunct={:?} gc_deleted={} ({} bytes) journal_truncated={} bytes \
             examples={}",
            self.census.proposals_drawn,
            self.census.duplicates_refused,
            self.census.payoff_floor_unreachable,
            self.census.trust_region_reverted,
            self.census.searches_errored,
            self.census.sweeps_failed,
            self.census.sweeps_abandoned,
            self.census.screen_failed_by_conjunct,
            self.census.gc_deleted_matrices,
            self.census.gc_bytes_reclaimed,
            self.journal_truncated_bytes,
            self.census.examples.len()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The writer — journal and session, in lockstep, always
// ─────────────────────────────────────────────────────────────────────────────

/// Operating-system lease for the sole process allowed to append one session.
///
/// The lock file is deliberately kept after release. Unlinking a locked file
/// would let another process create and lock a new inode while the old owner
/// still held the original one. Closing this handle (including process death)
/// releases the OS lock; acquisition is non-blocking, so this never waits while
/// holding a CPU-admission permit or an in-process mutex.
struct SessionWriterLease {
    _file: std::fs::File,
}

impl SessionWriterLease {
    fn acquire(journal_path: &Path) -> Result<Self> {
        if let Some(parent) = journal_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating the session writer lease directory {}",
                    parent.display()
                )
            })?;
        }
        let lock_path = journal_path.with_extension("writer.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| {
                format!(
                    "opening the active session writer lease {}",
                    lock_path.display()
                )
            })?;
        file.try_lock().with_context(|| {
            format!(
                "acquiring the exclusive active session writer lease {}. Another process may \
                 still own this session; acquisition is deliberately non-blocking",
                lock_path.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

/// Owns the journal and the folded session together, so nothing can append to
/// one without updating the other.
///
/// This is the only way the runner writes state. It exists because the
/// alternative — two objects updated by hand at seventeen call sites — is
/// exactly how the fold and the in-memory state drift apart, and the drift is
/// invisible until a resume produces a different session from the one that
/// crashed.
pub struct SessionWriter {
    journal: Journal,
    session: Session,
    // Declared last so the journal handle is dropped before its ownership
    // lease. The lease itself is held for this writer's complete lifetime.
    _lease: SessionWriterLease,
}

impl SessionWriter {
    /// Open (or resume) the journal at `path` and fold it.
    ///
    /// If the replay found a truncated tail, a [`Record::TruncatedTail`] is
    /// appended immediately, so the fact survives into the verdict's census
    /// rather than living only in a log line nobody kept.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let lease = SessionWriterLease::acquire(path)?;
        let journal = Journal::open(path)?;
        let session = Session::fold(journal.records())
            .context("folding the journal into a session on resume")?;
        let truncated = journal.replay_report().truncated_tail_bytes;
        let mut writer = Self {
            journal,
            session,
            _lease: lease,
        };
        if truncated > 0 {
            writer.append(Record::TruncatedTail { bytes: truncated })?;
        }
        Ok(writer)
    }

    /// Append a record to the journal (fsynced) and apply it to the session.
    ///
    /// The debug assertion is the §11.4 invariant, checked after **every**
    /// record rather than at the end: a fold that diverges from the incremental
    /// state is a resume that silently produces a different session, and the
    /// only moment it is cheap to catch is the record that caused it.
    pub fn append(&mut self, record: Record) -> Result<()> {
        self.journal.append(record.clone())?;
        self.session.apply(&record).with_context(|| {
            format!(
                "applying a {} record to the session. The record IS on disk — the journal is the \
                 authority — so a resume will replay it and hit the same error.",
                record.tag()
            )
        })?;
        debug_assert_eq!(
            Session::fold(self.journal.records()).ok().as_ref(),
            Some(&self.session),
            "the folded session diverged from the incremental one after a {} record. \
             docs/autoresearch-loop.md §11.4 requires them to be equal after EVERY record.",
            record.tag()
        );
        Ok(())
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Offer a candidate as the session's new best-ever result.
    ///
    /// Returns whether it advanced. **There is no `session_mut`**: this used to
    /// be `writer.session_mut().offer_best_ever(..)`, an assignment into a field
    /// of the fold's own product, and it is exactly the state-outside-the-fold
    /// that a debug build catches on the next append and a release build loses
    /// on the next restart. The comparison happens BEFORE the append so a
    /// non-advancing offer costs no journal line, and it is the same
    /// [`advances_best`] the fold applies, so the two can never disagree.
    pub fn offer_best_ever(&mut self, candidate: BestEver) -> Result<bool> {
        if !advances_best(self.session.best_ever.as_ref(), &candidate) {
            return Ok(false);
        }
        self.append(Record::BestEverAdvanced { best: candidate })?;
        Ok(true)
    }

    /// Record one sweep's champion row into the session champion matrix.
    ///
    /// Journalled, for the same reason: an in-memory matrix restarts empty, and
    /// an empty matrix makes `pbo_session` refuse — which is a FAIL, so no
    /// promotion would ever be possible again after a restart.
    pub fn push_champion(&mut self, row: ChampionRow) -> Result<()> {
        self.append(Record::ChampionRecorded { row })
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The store
// ─────────────────────────────────────────────────────────────────────────────

/// What a garbage-collection pass did. Never silent: §3.2 requires the rule and
/// the byte count in the census.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GcCensus {
    pub kept: usize,
    pub deleted: usize,
    pub bytes_reclaimed: u64,
    pub rule: String,
    /// Free bytes on the store's filesystem, PROBED — never a constant, so the
    /// budget means the same thing on a laptop and on a rented box.
    pub free_bytes_after: Option<u64>,
}

/// The on-disk session directory.
///
/// Root is resolved through the existing store resolver
/// (`neoethos_core::config::user_config_path`'s parent), **never a hardcoded
/// path** — the operator's `NEOETHOS_USER_DATA_DIR` redirect has to move the
/// autoresearch store with everything else, or a resumed session would silently
/// address a different directory from the one the last run wrote.
#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
    dir: PathBuf,
    session_id: SessionId,
}

impl SessionStore {
    /// `<store>/autoresearch`.
    pub fn root() -> Result<PathBuf> {
        let config_path = neoethos_core::config::user_config_path();
        let store = config_path.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "the resolved config path {} has no parent directory, so the autoresearch store \
                 root cannot be derived from it",
                config_path.display()
            )
        })?;
        Ok(store.join("autoresearch"))
    }

    pub fn open(session_id: SessionId) -> Result<Self> {
        let root = Self::root()?;
        Self::open_under(root, session_id)
    }

    /// Open under an explicit root. Used by tests, and by any caller that has
    /// already resolved the store itself.
    pub fn open_under(root: impl AsRef<Path>, session_id: SessionId) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let dir = root.join(session_id.as_str());
        std::fs::create_dir_all(dir.join("sweeps"))
            .with_context(|| format!("creating the session directory {}", dir.display()))?;
        Ok(Self {
            root,
            dir,
            session_id,
        })
    }

    /// Every session id under the store root, newest directory name last.
    pub fn list(root: impl AsRef<Path>) -> Result<Vec<SessionId>> {
        let root = root.as_ref();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(root)
            .with_context(|| format!("listing the autoresearch store {}", root.display()))?
        {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Ok(id) = SessionId::parse(name) {
                ids.push(id);
            }
        }
        ids.sort();
        Ok(ids)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn root_dir(&self) -> &Path {
        &self.root
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn journal_path(&self) -> PathBuf {
        self.dir.join("journal.jsonl")
    }

    pub fn header_path(&self) -> PathBuf {
        self.dir.join("session.json")
    }

    pub fn sweep_dir(&self, sweep: SweepId) -> PathBuf {
        self.dir.join("sweeps").join(sweep.to_string())
    }

    /// Per-search trial-return matrices. §11.1 names one `trial_returns.bin`
    /// per sweep; §3.2 says there are 100 per sweep (one per search), so they
    /// live in a subdirectory keyed by slot. The layout is stated here rather
    /// than implied so a reader of the store knows what a file is without
    /// opening it.
    pub fn trial_returns_path(&self, sweep: SweepId, slot: usize) -> PathBuf {
        self.sweep_dir(sweep)
            .join("trial_returns")
            .join(format!("slot_{slot:03}.bin"))
    }

    /// The per-sweep progress ledger, appended and fsynced after **each**
    /// search.
    ///
    /// It is not session state — the journal is — but it is what makes
    /// `SweepAbandoned { trials_offered }` honest after a crash: without it the
    /// resume would have to record `0` for a sweep that had run eighty searches,
    /// understating `N_session` and therefore FLATTERING every deflated Sharpe
    /// that followed.
    pub fn partial_ledger_path(&self, sweep: SweepId) -> PathBuf {
        self.sweep_dir(sweep).join("partial.jsonl")
    }

    /// Write the header exactly once. A second call with a DIFFERENT header is
    /// an error; with the same header it is a no-op, so a crash between
    /// `create_dir_all` and the first journal append does not wedge the store.
    pub fn write_header_once(&self, header: &SessionHeader) -> Result<()> {
        header.validate()?;
        let path = self.header_path();
        if path.exists() {
            let existing = self.read_header()?;
            if &existing != header {
                bail!(
                    "{} already exists and differs from the header this run would write. The \
                     session header is IMMUTABLE — it is the frame every sweep in this journal \
                     was judged inside. Start a new session rather than rewriting it.",
                    path.display()
                );
            }
            return Ok(());
        }
        write_json_atomically(&path, header)
    }

    pub fn read_header(&self) -> Result<SessionHeader> {
        let path = self.header_path();
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading the session header {}", path.display()))?;
        let header: SessionHeader = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing the session header {}", path.display()))?;
        header.validate()?;
        Ok(header)
    }

    /// Refuse a resume whose frame has moved. §11.4.
    pub fn assert_resumable(
        &self,
        stored: &SessionHeader,
        goal_hash: &str,
        judge_hash: &str,
        cost_hash: &str,
        symbol: &str,
        dataset_receipt: &DatasetReceiptV1,
    ) -> Result<(), ResumeRefusal> {
        if stored.schema != crate::SESSION_SCHEMA {
            return Err(ResumeRefusal::SchemaChanged {
                stored: stored.schema.clone(),
                current: crate::SESSION_SCHEMA.to_string(),
            });
        }
        if stored.goal_hash != goal_hash {
            return Err(ResumeRefusal::GoalsChanged {
                stored: stored.goal_hash.clone(),
                current: goal_hash.to_string(),
            });
        }
        if stored.judge_hash != judge_hash {
            return Err(ResumeRefusal::JudgeChanged {
                stored: stored.judge_hash.clone(),
                current: judge_hash.to_string(),
            });
        }
        if stored.cost_hash != cost_hash {
            return Err(ResumeRefusal::CostModelChanged {
                stored: stored.cost_hash.clone(),
                current: cost_hash.to_string(),
            });
        }
        if stored.symbol != symbol {
            return Err(ResumeRefusal::SymbolChanged {
                stored: stored.symbol.clone(),
                current: symbol.to_string(),
            });
        }
        if &stored.dataset_receipt != dataset_receipt {
            return Err(ResumeRefusal::DatasetReceiptChanged {
                stored: stored.dataset_receipt.identity().to_string(),
                current: dataset_receipt.identity().to_string(),
            });
        }
        Ok(())
    }

    /// Create a sweep's artifact directory before the sweep runs.
    pub fn prepare_sweep(&self, sweep: SweepId) -> Result<PathBuf> {
        let dir = self.sweep_dir(sweep);
        std::fs::create_dir_all(dir.join("trial_returns"))
            .with_context(|| format!("creating the sweep directory {}", dir.display()))?;
        Ok(dir)
    }

    /// Append one line to a sweep's partial ledger and fsync it.
    pub fn record_partial(&self, sweep: SweepId, slot: usize, trials_offered: usize) -> Result<()> {
        use std::io::Write as _;
        let path = self.partial_ledger_path(sweep);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening the partial ledger {}", path.display()))?;
        writeln!(
            file,
            "{{\"slot\":{slot},\"trials_offered\":{trials_offered}}}"
        )
        .with_context(|| format!("appending to the partial ledger {}", path.display()))?;
        file.flush()?;
        file.sync_data()
            .with_context(|| format!("fsyncing the partial ledger {}", path.display()))?;
        Ok(())
    }

    /// Recover `trials_offered` for a sweep the process died inside.
    ///
    /// Returns `0` when the sweep died before its first search finished, which
    /// is the truth: nothing had run.
    pub fn recover_partial_trials(&self, sweep: SweepId) -> usize {
        let path = self.partial_ledger_path(sweep);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return 0;
        };
        text.lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|v| v.get("trials_offered").and_then(|t| t.as_u64()))
            .map(|t| t as usize)
            .sum()
    }

    /// Write a JSON artifact into a sweep's directory.
    pub fn write_sweep_artifact<T: Serialize>(
        &self,
        sweep: SweepId,
        name: &str,
        value: &T,
    ) -> Result<()> {
        write_json_atomically(&self.sweep_dir(sweep).join(name), value)
    }

    pub fn write_champions(&self, rows: &[ChampionRow]) -> Result<()> {
        write_json_atomically(&self.dir.join("session_champions.json"), &rows)
    }

    /// Written ONCE, at stop.
    pub fn write_verdict<T: Serialize>(&self, verdict: &T) -> Result<()> {
        write_json_atomically(&self.dir.join("verdict.json"), verdict)
    }

    /// The proposal artifact — the loop's ONLY output that describes a
    /// tradeable thing.
    ///
    /// It is deliberately not `live_portfolio.json` and it is deliberately not
    /// written anywhere the trading side reads. The loop proposes; the operator
    /// promotes.
    pub fn write_proposal<T: Serialize>(&self, proposal: &T) -> Result<()> {
        write_json_atomically(&self.dir.join("proposal.json"), proposal)
    }

    /// Delete the trial-returns matrices this session no longer needs.
    ///
    /// The rule is named in the census and the bytes are counted. Kept: the
    /// session's best-ever sweep, every promotion candidate, every shuffle
    /// control (§3.2). ~14 MiB per run × 100 per sweep is why this exists at
    /// all, and *never silently* is why it returns a census.
    pub fn gc_trial_matrices(&self, keep: &HashSet<SweepId>) -> Result<GcCensus> {
        let sweeps_dir = self.dir.join("sweeps");
        let mut census = GcCensus {
            kept: 0,
            deleted: 0,
            bytes_reclaimed: 0,
            rule: "keep the trial-returns matrices of: the session's best-ever sweep, every \
                   promotion candidate, and every shuffle control. Delete the rest — the \
                   TrialStatisticsReport and the manifest are kept for ALL sweeps, so every \
                   statistic in every report stays re-derivable"
                .to_string(),
            free_bytes_after: None,
        };
        if !sweeps_dir.exists() {
            return Ok(census);
        }
        for entry in std::fs::read_dir(&sweeps_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(id) = name
                .strip_prefix("sweep-")
                .and_then(|n| n.parse::<u64>().ok())
                .map(SweepId)
            else {
                continue;
            };
            let matrices = entry.path().join("trial_returns");
            if !matrices.exists() {
                continue;
            }
            if keep.contains(&id) {
                census.kept += 1;
                continue;
            }
            for matrix in std::fs::read_dir(&matrices)? {
                let matrix = matrix?;
                let bytes = matrix.metadata().map(|m| m.len()).unwrap_or(0);
                match std::fs::remove_file(matrix.path()) {
                    Ok(()) => {
                        census.deleted += 1;
                        census.bytes_reclaimed += bytes;
                    }
                    Err(err) => {
                        // Counted and named. A GC that cannot delete is not a
                        // failure of the session, but it must not be silent
                        // either — the next run needs to know the disk did not
                        // come back.
                        tracing::warn!(
                            target: "neoethos_autoresearch::session",
                            path = %matrix.path().display(),
                            error = %err,
                            "could not delete a trial-returns matrix during GC; the bytes were \
                             NOT reclaimed and this sweep still occupies them"
                        );
                    }
                }
            }
        }
        census.free_bytes_after = neoethos_search::trial_returns::available_bytes_for(&self.dir);
        Ok(census)
    }
}

/// Write JSON to `path` via a temporary file and a rename, fsyncing the data
/// before the rename.
///
/// A half-written `verdict.json` is worse than a missing one: a reader takes it
/// as the answer.
fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialising {}", path.display()))?;
    {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
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

/// `n_session` over a folded session, for callers that hold a [`Session`]
/// rather than the raw records. Equal to [`crate::journal::n_session`] by test.
pub fn n_session(session: &Session) -> usize {
    session.n_session()
}

/// Bridge for callers that hold the raw record slice rather than a folded
/// [`Session`]. Same number, same definition — asserted by
/// `n_session_is_the_same_number_by_both_definitions`.
pub use crate::journal::n_session as n_session_of_records;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::CostBandCounts;

    fn direct_receipt(
        timeframe: neoethos_data::CanonicalTimeframe,
        generation_id: &str,
        hash_byte: u8,
    ) -> DirectTimeframeReceiptV1 {
        DirectTimeframeReceiptV1 {
            dataset_identity: neoethos_data::CanonicalDatasetIdentity::external(
                "autoresearch-session-test",
                "EURUSD",
                timeframe,
                neoethos_data::BarTimestampConvention::BarOpen,
            )
            .expect("test dataset identity"),
            manifest_schema_id: "neoethos.dataset-manifest.v1".to_owned(),
            manifest_sha256: [hash_byte; 32],
            generation_id: generation_id.to_owned(),
            vortex_sha256: [hash_byte.wrapping_add(1); 32],
            row_count: 101,
            timestamp_start_ms: 0,
            timestamp_end_ms: 1_000,
        }
    }

    fn dataset_receipt(generation_id: &str) -> DatasetReceiptV1 {
        let base = direct_receipt(neoethos_data::CanonicalTimeframe::M5, generation_id, 3);
        DatasetReceiptV1::new(
            base.dataset_identity.clone(),
            vec![base],
            InSampleWindowV1 {
                start_ms: 0,
                end_exclusive_ms: 900,
            },
            OosWindow {
                start_ms: 900,
                end_ms: 1_000,
            },
        )
        .expect("test dataset receipt")
    }

    fn header_record(id: &str) -> Record {
        Record::SessionOpened {
            session_id: SessionId(id.to_string()),
            session_seed: 42,
            goal_hash: "fnv64:aaaa".into(),
            judge_hash: "fnv64:bbbb".into(),
            cost_hash: "fnv64:cccc".into(),
            goals: crate::journal::NamedValues(vec![(
                "system.risky_target_balance_usd".into(),
                "50000".into(),
            )]),
            scenarios_source: "system.risky_* (single)".into(),
            judge: crate::journal::NamedValues(vec![("pbo_max".into(), "0.5".into())]),
            costs: crate::journal::NamedValues(vec![]),
            oos_window: OosWindow {
                start_ms: 900,
                end_ms: 1000,
            },
            priors: "Beta(1, 1) on every level".into(),
            budget: crate::journal::NamedValues(vec![("max_sweeps".into(), "200".into())]),
            identity_source: "config_hash".into(),
            symbol: "EURUSD".into(),
            dataset_receipt: dataset_receipt("generation-a"),
        }
    }

    fn search(slot: usize, hash: &str, e: Option<f64>) -> SearchRecord {
        SearchRecord {
            slot,
            config_hash: hash.into(),
            trials_offered: 10,
            survivors: 1,
            e_screen_pess: e,
            n_trades: 100,
            dsr: Some(0.96),
            pbo: Some(0.30),
            dsr_refusal: None,
            pbo_refusal: None,
            cost_band: CostBandCounts::default(),
            rejections: vec![("base_quality.profile_payoff_ratio".into(), 3)],
            streamed: true,
            batch_columns: 512,
            next_cursor: 900,
            wall_ms: 1_000,
            error: None,
        }
    }

    fn started(sweep: u64) -> Record {
        Record::SweepStarted {
            sweep: SweepId(sweep),
            sweep_hash: format!("fnv64:{sweep:016x}"),
            kind: SweepKind::Live,
            started_ms: 1,
        }
    }

    fn completed(sweep: u64, trials: usize) -> Record {
        Record::SweepCompleted {
            sweep: SweepId(sweep),
            trials_offered: trials,
            per_search: vec![search(0, "fnv64:one", Some(0.4))],
            wall_ms: 2_000,
        }
    }

    fn best(sweep: u64, slot: usize, e: f64) -> BestEver {
        BestEver {
            sweep: SweepId(sweep),
            slot,
            config_hash: format!("fnv64:{sweep:016x}-{slot}"),
            e_screen_pess: e,
            n_trades: 250,
            dsr: Some(0.97),
            pbo: Some(0.31),
        }
    }

    /// The month grid and the return generator are the SAME ones
    /// `judge::the_session_champion_matrix_is_built_in_memory_from_public_fields`
    /// pins as accepted by `pbo_cscv`, so a resume test that asserts
    /// `pbo_session` returns a number is testing the resume and not the CSCV
    /// admission rules.
    fn champion_months() -> Vec<i64> {
        (24_300..24_336).collect()
    }

    fn champion_row(sweep: u64) -> ChampionRow {
        let i = sweep as i64 - 1;
        ChampionRow {
            sweep: SweepId(sweep),
            config_hash: format!("fnv64:{sweep:016x}"),
            strategy_id: format!("g{sweep}"),
            period_keys: champion_months(),
            monthly_returns: (0..36)
                .map(|m| 0.01 * ((i + m) % 7) as f64 - 0.02)
                .collect(),
            e_screen_pess: 0.4,
        }
    }

    #[test]
    fn the_fold_equals_the_incremental_apply_after_every_record() {
        // §11.4, the invariant the whole resume path rests on.
        //
        // `best_ever` and the champion matrix are IN this list on purpose. They
        // were once the two fields the runner assigned through a `&mut Session`,
        // which made the fold and the incremental state diverge on the first
        // sweep that produced a champion — a debug-build panic on the next
        // append, and in a release build a silent loss of both on every restart.
        // Anything that moves them back out of the journal fails here.
        let records = vec![
            header_record("ar-test"),
            started(1),
            completed(1, 100),
            Record::ChampionRecorded {
                row: champion_row(1),
            },
            Record::BestEverAdvanced {
                best: best(1, 0, 0.40),
            },
            // A record that does NOT advance. The fold is monotone, so this
            // leaves the pointer where it was — and the fold and the incremental
            // apply have to agree about that too.
            Record::BestEverAdvanced {
                best: best(1, 7, 0.11),
            },
            started(2),
            Record::SweepFailed {
                sweep: SweepId(2),
                trials_offered: 33,
                error: "the card fell over".into(),
            },
        ];

        let mut incremental = Session::default();
        for (index, record) in records.iter().enumerate() {
            incremental.apply(record).unwrap();
            let folded = Session::fold(&records[..=index]).unwrap();
            assert_eq!(
                folded,
                incremental,
                "the fold diverged from the incremental state after record {index} ({})",
                record.tag()
            );
        }

        // And the fold produced the right values, not merely equal ones: a fold
        // that dropped both fields would also be "equal" to an incremental apply
        // that dropped both.
        assert_eq!(incremental.champions.len(), 1);
        assert_eq!(incremental.champions[0].sweep, SweepId(1));
        let best_ever = incremental.best_ever.as_ref().expect("a best-ever pointer");
        assert_eq!(
            best_ever.slot, 0,
            "the lower offer must not displace the best"
        );
        assert!((best_ever.e_screen_pess - 0.40).abs() < 1e-12);
    }

    #[test]
    fn a_second_champion_row_for_one_sweep_is_refused_not_appended() {
        // The session champion matrix is ONE ROW PER SWEEP. A second row would
        // let one sweep vote twice in `pbo_session` — the statistic that decides
        // whether the loop is fooling itself.
        let records = vec![
            header_record("ar-test"),
            Record::ChampionRecorded {
                row: champion_row(1),
            },
            Record::ChampionRecorded {
                row: champion_row(1),
            },
        ];
        let err = Session::fold(&records).expect_err("two rows from one sweep must not fold");
        assert!(
            format!("{err:#}").contains("ONE ROW PER SWEEP"),
            "got: {err:#}"
        );
    }

    #[test]
    fn the_best_ever_pointer_only_ever_moves_up() {
        // Replay order is not a promise the fold gets to rely on, so the fold
        // takes the max rather than assigning. A NaN loses every comparison and
        // can never become the best: "no number" is never good news.
        assert!(advances_best(None, &best(1, 0, -3.0)));
        assert!(!advances_best(None, &best(1, 0, f64::NAN)));
        assert!(advances_best(Some(&best(1, 0, 0.10)), &best(2, 0, 0.11)));
        assert!(!advances_best(Some(&best(1, 0, 0.10)), &best(2, 0, 0.10)));
        assert!(!advances_best(
            Some(&best(1, 0, 0.10)),
            &best(2, 0, f64::NAN)
        ));

        let records = vec![
            header_record("ar-test"),
            Record::BestEverAdvanced {
                best: best(1, 0, 0.40),
            },
            Record::BestEverAdvanced {
                best: best(2, 0, 0.05),
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.best_ever.as_ref().unwrap().sweep, SweepId(1));
    }

    #[test]
    fn n_session_is_the_same_number_by_both_definitions() {
        let records = vec![
            header_record("ar-test"),
            started(1),
            completed(1, 100),
            started(2),
            Record::SweepAbandoned {
                sweep: SweepId(2),
                trials_offered: 17,
                reason: "process died".into(),
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.n_session(), crate::journal::n_session(&records));
        assert_eq!(session.n_session(), 117);
    }

    #[test]
    fn a_crash_loses_one_sweep_not_the_session() {
        // Two-phase journalling: the intent for sweep 2 is on disk, the outcome
        // never arrived.
        let records = vec![
            header_record("ar-test"),
            started(1),
            completed(1, 100),
            started(2),
        ];
        let session = Session::fold(&records).unwrap();

        // Sweep 1 survived intact.
        assert_eq!(session.sweeps_run(), 1);
        assert_eq!(session.n_session(), 100);
        // Sweep 2 is visibly open — the resume path turns this into
        // SweepAbandoned, counted and named.
        let open = session.open_sweep.as_ref().expect("sweep 2 is open");
        assert_eq!(open.sweep, SweepId(2));
        // And the next sweep id never reuses 2's directory.
        assert_eq!(session.next_sweep(), SweepId(3));
    }

    #[test]
    fn an_abandoned_sweeps_trials_still_count() {
        let mut records = vec![header_record("ar-test"), started(1)];
        records.push(Record::SweepAbandoned {
            sweep: SweepId(1),
            trials_offered: 62,
            reason: "process died".into(),
        });
        let session = Session::fold(&records).unwrap();
        assert_eq!(
            session.n_session(),
            62,
            "a trial that ran is a trial that ran"
        );
        assert_eq!(session.census.sweeps_abandoned, 1);
    }

    #[test]
    fn a_second_writer_cannot_open_the_same_active_session_until_the_first_drops() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");

        let first = SessionWriter::open(&path).expect("the first writer owns the session");
        let error = match SessionWriter::open(&path) {
            Ok(_) => panic!("two live writers must never own one session journal"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("active session writer"),
            "the refusal must name the exclusive writer lease: {error:#}"
        );

        drop(first);
        SessionWriter::open(&path)
            .expect("dropping the owner must release the operating-system writer lease");
    }

    #[test]
    fn session_writer_lease_child_process() {
        let Ok(journal_path) = std::env::var("NEOETHOS_TEST_SESSION_WRITER_JOURNAL") else {
            return;
        };
        let ready_path = std::env::var("NEOETHOS_TEST_SESSION_WRITER_READY")
            .expect("the lease-child ready path accompanies its journal path");
        let _writer = SessionWriter::open(&journal_path).expect("child owns the writer lease");
        std::fs::write(&ready_path, b"ready").expect("child announces lease ownership");
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    #[test]
    fn another_process_is_refused_and_process_death_releases_the_writer_lease() {
        let dir = tempfile::tempdir().unwrap();
        let journal_path = dir.path().join("journal.jsonl");
        let ready_path = dir.path().join("child-ready");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("session::tests::session_writer_lease_child_process")
            .arg("--nocapture")
            .env("NEOETHOS_TEST_SESSION_WRITER_JOURNAL", &journal_path)
            .env("NEOETHOS_TEST_SESSION_WRITER_READY", &ready_path)
            .spawn()
            .expect("spawn writer-lease child process");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready_path.exists() {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("writer-lease child exited before acquiring the lock: {status}");
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("writer-lease child did not acquire the lock within 10 seconds");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let error = match SessionWriter::open(&journal_path) {
            Ok(_) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("a second process opened a journal whose child writer was active")
            }
            Err(error) => error,
        };
        child.kill().expect("terminate the lease-owning child");
        child.wait().expect("reap the lease-owning child");
        SessionWriter::open(&journal_path)
            .expect("operating-system process death must release the writer lease");
        assert!(format!("{error:#}").contains("active session writer"));
    }

    /// **The adversary's test.** Kill the loop after sweeps that produced
    /// champions — mid-sweep, with an intent on disk and no outcome — resume,
    /// and require that both `best_ever` and the champion matrix came back, and
    /// that the two capabilities they carry are still reachable.
    ///
    /// This is the whole of DEFECT 1. When these two lived outside the fold:
    ///
    /// * a DEBUG build panicked on the first append after the first champion,
    ///   because `SessionWriter::append` re-folds and compares — so this test
    ///   also fails as a panic, not as an assertion, if anything moves them back
    ///   out;
    /// * a RELEASE build lost both on every restart, and the loss is not
    ///   cosmetic. R2's U2 condition reads `best_ever`; with `None` it falls into
    ///   the arm that says *there is nothing to compare*, which can never be
    ///   satisfied, so **the loop could no longer report the goal UNREACHABLE** —
    ///   the one thing this project has never been able to say. And
    ///   `pbo_session` over an empty champion matrix REFUSES, and a refusal is a
    ///   fail, so **no promotion was possible** either.
    #[test]
    fn a_crash_loses_one_sweep_but_never_the_best_ever_or_the_champion_matrix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");

        // ── the process that did the work, and then died ────────────────────
        {
            let mut writer = SessionWriter::open(&path).unwrap();
            writer.append(header_record("ar-test")).unwrap();
            for sweep in 1..=12u64 {
                writer.append(started(sweep)).unwrap();
                writer.append(completed(sweep, 100)).unwrap();
                writer.push_champion(champion_row(sweep)).unwrap();
                let advanced = writer
                    .offer_best_ever(best(sweep, 0, 0.10 + 0.01 * sweep as f64))
                    .unwrap();
                assert!(
                    advanced,
                    "sweep {sweep} strictly improved on its predecessor"
                );
            }
            // A non-advancing offer costs no journal line and changes nothing.
            assert!(!writer.offer_best_ever(best(12, 9, -5.0)).unwrap());

            // Three shuffle-control observations — MIN_NULL_OBS — so the null
            // exists and U2 has something on both sides of its comparison.
            for (block, e) in [(0u32, 0.90), (1, 1.00), (2, 1.10)] {
                writer
                    .append(Record::ShuffleControlCompleted {
                        block: BlockId(block),
                        kind: crate::shuffle::ControlKind::CircularRotation,
                        tau: 0.37,
                        source_sweep: SweepId(4 * (block as u64 + 1)),
                        trials_offered: 100,
                        e_screen_pess: Some(e),
                        dsr: Some(0.5),
                    })
                    .unwrap();
            }

            // And now the crash: sweep 13's INTENT is on disk, its outcome never
            // arrives, and the process is gone before it can write anything else.
            writer.append(started(13)).unwrap();
        }

        // ── the resume ──────────────────────────────────────────────────────
        let resumed = SessionWriter::open(&path).unwrap();
        let session = resumed.session();

        // A crash loses ONE SWEEP, not the session.
        assert_eq!(session.sweeps_run(), 12);
        assert_eq!(
            session.open_sweep.as_ref().map(|o| o.sweep),
            Some(SweepId(13)),
            "the sweep the process died inside is visibly open, for the resume path to abandon"
        );
        assert_eq!(session.next_sweep(), SweepId(14));

        // Both pieces of state came back FROM THE JOURNAL.
        assert_eq!(session.champions.len(), 12, "the champion matrix survived");
        let best_ever = session
            .best_ever
            .as_ref()
            .expect("the best-ever pointer survived the process that produced it");
        assert_eq!(best_ever.sweep, SweepId(12));
        assert!((best_ever.e_screen_pess - 0.22).abs() < 1e-9);

        // CAPABILITY 1 — the loop can still say GOAL UNREACHABLE. U2 must be in
        // the arm that HAS both numbers, and must be satisfiable: the best live
        // result (+0.22) never beat the null's 95th percentile (+1.10).
        let check =
            crate::verdict::check_unreachable(session, &crate::judge::JudgeThresholds::frozen());
        let u2 = check
            .conditions
            .iter()
            .find(|c| c.id == "U2")
            .expect("U2 is always evaluated");
        assert!(
            u2.detail.contains("best live E_screen_pess"),
            "U2 fell into the no-best arm after a resume, which can never be satisfied: {}",
            u2.detail
        );
        assert!(
            u2.satisfied,
            "U2 must remain SATISFIABLE after a resume: {}",
            u2.detail
        );

        // CAPABILITY 2 — promotion is still possible. `pbo_session` is computed
        // over the champion matrix; with an empty matrix it refuses, and a
        // refusal is a fail.
        let (pbo, refusal) = crate::judge::session_pbo_value(session);
        assert!(
            pbo.is_some(),
            "pbo_session refused after a resume, so no promotion would ever be possible again: \
             {refusal:?}"
        );
    }

    /// A release build must not lose them either — and `debug_assert_eq!` is
    /// compiled out there, so the incremental state IS whatever the writer put
    /// in it. Comparing the writer's own view against a fold of its own journal
    /// is the check that survives `--release`.
    #[test]
    fn the_writers_view_equals_a_fold_of_its_own_journal_in_any_build() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut writer = SessionWriter::open(&path).unwrap();
        writer.append(header_record("ar-test")).unwrap();
        writer.append(started(1)).unwrap();
        writer.append(completed(1, 100)).unwrap();
        writer.push_champion(champion_row(1)).unwrap();
        writer.offer_best_ever(best(1, 0, 0.4)).unwrap();

        let folded = Session::fold(writer.journal().records()).unwrap();
        assert_eq!(
            &folded,
            writer.session(),
            "the writer's session must be a pure fold of its own journal in EVERY build — this \
             assertion is the release-build half of §11.4, where debug_assert_eq! is gone"
        );
        assert!(folded.best_ever.is_some());
        assert_eq!(folded.champions.len(), 1);
    }

    #[test]
    fn an_outcome_without_an_intent_is_a_hard_error() {
        let records = vec![header_record("ar-test"), completed(1, 100)];
        let err = Session::fold(&records).expect_err("an orphan outcome must not fold");
        assert!(format!("{err:#}").contains("NO open sweep"));
    }

    #[test]
    fn two_session_headers_are_refused() {
        let records = vec![header_record("ar-a"), header_record("ar-b")];
        assert!(Session::fold(&records).is_err());
    }

    #[test]
    fn a_retracted_block_flips_its_successes_to_failures_exactly() {
        // §5.3 mechanism 3: an INDISTINGUISHABLE block retracts every success it
        // credited, by replaying its own credits.
        let credits = vec![
            PosteriorCredit {
                factor: "population".into(),
                level: "4096".into(),
                success: true,
            },
            PosteriorCredit {
                factor: "axis_b".into(),
                level: "B3_CostElastic".into(),
                success: true,
            },
        ];
        let records = vec![
            header_record("ar-test"),
            Record::PosteriorUpdated {
                sweep: SweepId(1),
                credits: credits.clone(),
            },
            Record::PosteriorRetracted {
                block: BlockId(0),
                sweep_ids: vec![SweepId(1)],
                reason: "block indistinguishable".into(),
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.posterior.evidence("population", "4096"), (0, 1));
        assert_eq!(
            session.posterior.evidence("axis_b", "B3_CostElastic"),
            (0, 1)
        );
    }

    #[test]
    fn retracting_the_same_block_twice_does_not_double_penalise() {
        let credits = vec![PosteriorCredit {
            factor: "population".into(),
            level: "4096".into(),
            success: true,
        }];
        let records = vec![
            header_record("ar-test"),
            Record::PosteriorUpdated {
                sweep: SweepId(1),
                credits,
            },
            Record::PosteriorRetracted {
                block: BlockId(0),
                sweep_ids: vec![SweepId(1)],
                reason: "block indistinguishable".into(),
            },
            Record::PosteriorRetracted {
                block: BlockId(0),
                sweep_ids: vec![SweepId(1)],
                reason: "block indistinguishable".into(),
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.posterior.evidence("population", "4096"), (0, 1));
    }

    #[test]
    fn the_null_and_the_blocks_accumulate_in_order() {
        let records = vec![
            header_record("ar-test"),
            Record::ShuffleControlCompleted {
                block: BlockId(0),
                kind: crate::shuffle::ControlKind::CircularRotation,
                tau: 0.31,
                source_sweep: SweepId(3),
                trials_offered: 100,
                e_screen_pess: Some(0.31),
                dsr: Some(0.5),
            },
            Record::BlockJudged {
                block: BlockId(0),
                delta_expectancy: 0.11,
                p_block: 0.0,
                dsr_gap: 0.2,
                indistinguishable: false,
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.null_observations, vec![0.31]);
        assert_eq!(session.blocks.len(), 1);
        assert_eq!(session.indistinguishable_blocks(), 0);
        // The control's trials count into N exactly ONCE, through the control
        // sweep's own `SweepCompleted`. This journal has none, so the summary
        // record on its own contributes nothing — crediting it here is what
        // double-counted every control in a real session.
        assert_eq!(session.n_session(), 0);
        assert_eq!(
            session.n_session(),
            crate::journal::n_session(&records),
            "the two definitions of N must not drift apart"
        );
    }

    /// C5. A control that produced no number is COUNTED and NAMED, never
    /// dropped. `ShuffleSummary::null_refusals` rides under every headline the
    /// loop prints, and it used to be structurally zero.
    #[test]
    fn a_shuffle_control_with_no_number_is_counted_as_a_refusal_and_named() {
        let records = vec![
            header_record("ar-test"),
            Record::ShuffleControlCompleted {
                block: BlockId(0),
                kind: crate::shuffle::ControlKind::CircularRotation,
                tau: 0.31,
                source_sweep: SweepId(3),
                trials_offered: 100,
                e_screen_pess: None,
                dsr: None,
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert!(session.null_observations.is_empty());
        assert_eq!(session.null_refusals, 1);
        assert_eq!(
            crate::shuffle::ShuffleNull::from_session(&session).refused(),
            1
        );
        assert_eq!(
            crate::shuffle::ShuffleSummary::from_session(&session).null_refusals,
            1,
            "the headline must not print a clean null over a broken control"
        );
        assert!(
            session
                .census
                .examples
                .iter()
                .any(|(reason, _)| reason.contains("no E_screen_pess")),
            "counted is not enough — it must be NAMED: {:?}",
            session.census.examples
        );
    }

    /// C4. The control sweep's own `SweepCompleted` is the ONE place its trials
    /// are credited. This is the journal a real block boundary writes.
    #[test]
    fn a_control_sweeps_trials_are_counted_once_not_twice() {
        let records = vec![
            header_record("ar-test"),
            Record::SweepStarted {
                sweep: SweepId(1),
                sweep_hash: "fnv64:live".to_string(),
                kind: SweepKind::Live,
                started_ms: 1,
            },
            Record::SweepCompleted {
                sweep: SweepId(1),
                trials_offered: 100,
                per_search: Vec::new(),
                wall_ms: 1,
            },
            Record::SweepStarted {
                sweep: SweepId(2),
                sweep_hash: "fnv64:control".to_string(),
                kind: SweepKind::Control {
                    control: crate::shuffle::ControlKind::CircularRotation,
                    source_sweep: SweepId(1),
                    block: BlockId(0),
                },
                started_ms: 2,
            },
            Record::SweepCompleted {
                sweep: SweepId(2),
                trials_offered: 100,
                per_search: Vec::new(),
                wall_ms: 1,
            },
            Record::ShuffleControlCompleted {
                block: BlockId(0),
                kind: crate::shuffle::ControlKind::CircularRotation,
                tau: 0.31,
                source_sweep: SweepId(1),
                trials_offered: 100,
                e_screen_pess: Some(-0.2),
                dsr: None,
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.n_session(), 200, "not 300");
        assert_eq!(session.n_session(), crate::journal::n_session(&records));
    }

    #[test]
    fn the_truncated_tail_reaches_the_census() {
        let records = vec![
            header_record("ar-test"),
            Record::TruncatedTail { bytes: 57 },
        ];
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.journal_truncated_bytes, 57);
        assert!(
            session
                .census
                .examples
                .iter()
                .any(|(reason, _)| reason.contains("truncated tail")),
            "the truncated tail must also be a NAMED example, not only a counter"
        );
    }

    #[test]
    fn the_oos_touch_budget_is_spent_once() {
        let records = vec![
            header_record("ar-test"),
            Record::OosTouchSpent {
                window: OosWindow {
                    start_ms: 900,
                    end_ms: 1000,
                },
                sweep: SweepId(1),
                candidate: "fnv64:one".into(),
            },
        ];
        let session = Session::fold(&records).unwrap();
        assert!(session.oos_spent());
        assert_eq!(session.oos_touches_spent, 1);
    }

    #[test]
    fn the_store_refuses_a_resume_whose_judge_moved() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open_under(dir.path(), SessionId::parse("ar-test-0000").unwrap())
            .unwrap();
        let frozen_dataset_receipt = dataset_receipt("generation-a");
        let reproduction = format!(
            "neoethos autoresearch --dataset-identity {} --session {}",
            frozen_dataset_receipt.anchor_identity.to_path_component(),
            store.session_id()
        );
        let header = SessionHeader {
            schema: crate::SESSION_SCHEMA.to_string(),
            session_id: store.session_id().clone(),
            session_seed: 7,
            symbol: "EURUSD".into(),
            dataset_receipt: frozen_dataset_receipt,
            created_utc: "2026-08-10T00:00:00Z".into(),
            goal_hash: "fnv64:g".into(),
            judge_hash: "fnv64:j".into(),
            cost_hash: "fnv64:c".into(),
            scenarios_source: "system.risky_* (single)".into(),
            identity_source: "config_hash".into(),
            oos_window: OosWindow {
                start_ms: 900,
                end_ms: 1000,
            },
            reproduction,
        };
        store.write_header_once(&header).unwrap();
        // Idempotent for the same header.
        store.write_header_once(&header).unwrap();

        let stored = store.read_header().unwrap();
        assert!(
            store
                .assert_resumable(
                    &stored,
                    "fnv64:g",
                    "fnv64:j",
                    "fnv64:c",
                    "EURUSD",
                    &dataset_receipt("generation-a"),
                )
                .is_ok()
        );
        let err = store
            .assert_resumable(
                &stored,
                "fnv64:g",
                "fnv64:MOVED",
                "fnv64:c",
                "EURUSD",
                &dataset_receipt("generation-a"),
            )
            .expect_err("a moved judge must refuse the resume");
        assert!(matches!(err, ResumeRefusal::JudgeChanged { .. }));
        assert!(err.to_string().contains("judge_hash"));
    }

    #[test]
    fn dataset_receipt_identity_is_stable_ordered_and_sensitive_to_generation_and_window() {
        let base = direct_receipt(neoethos_data::CanonicalTimeframe::M5, "generation-base", 3);
        let higher = direct_receipt(
            neoethos_data::CanonicalTimeframe::H1,
            "generation-higher",
            5,
        );
        let make = |direct_timeframes: Vec<DirectTimeframeReceiptV1>, cutoff| {
            DatasetReceiptV1::new(
                base.dataset_identity.clone(),
                direct_timeframes,
                InSampleWindowV1 {
                    start_ms: 0,
                    end_exclusive_ms: cutoff,
                },
                OosWindow {
                    start_ms: cutoff,
                    end_ms: 1_000,
                },
            )
            .expect("dataset receipt")
        };
        let forward = make(vec![base.clone(), higher.clone()], 900);
        let reversed = make(vec![higher.clone(), base.clone()], 900);
        assert_eq!(forward, reversed);
        assert_eq!(forward.identity(), reversed.identity());

        let mut changed_higher = higher;
        changed_higher.generation_id = "generation-higher-replaced".to_owned();
        let changed_generation = make(vec![base.clone(), changed_higher], 900);
        assert_ne!(forward.identity(), changed_generation.identity());

        let changed_window = make(
            vec![
                base.clone(),
                direct_receipt(
                    neoethos_data::CanonicalTimeframe::H1,
                    "generation-higher",
                    5,
                ),
            ],
            800,
        );
        assert_ne!(forward.identity(), changed_window.identity());
    }

    #[test]
    fn resume_refuses_a_different_exact_dataset_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let store =
            SessionStore::open_under(dir.path(), SessionId::parse("ar-test-receipt").unwrap())
                .unwrap();
        let mut header = SessionHeader {
            schema: crate::SESSION_SCHEMA.to_string(),
            session_id: store.session_id().clone(),
            session_seed: 7,
            symbol: "EURUSD".into(),
            dataset_receipt: dataset_receipt("generation-a"),
            created_utc: "2026-08-17T00:00:00Z".into(),
            goal_hash: "fnv64:g".into(),
            judge_hash: "fnv64:j".into(),
            cost_hash: "fnv64:c".into(),
            scenarios_source: "test".into(),
            identity_source: "config_hash".into(),
            oos_window: OosWindow {
                start_ms: 900,
                end_ms: 1_000,
            },
            reproduction: "test".into(),
        };
        store.write_header_once(&header).unwrap();
        let stored = store.read_header().unwrap();
        header.dataset_receipt = dataset_receipt("generation-b");
        let err = store
            .assert_resumable(
                &stored,
                &header.goal_hash,
                &header.judge_hash,
                &header.cost_hash,
                &header.symbol,
                &header.dataset_receipt,
            )
            .expect_err("changed generation must refuse resume");
        assert!(matches!(err, ResumeRefusal::DatasetReceiptChanged { .. }));
    }

    #[test]
    fn a_partial_ledger_recovers_the_trials_a_crashed_sweep_had_run() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open_under(dir.path(), SessionId::parse("ar-test-0001").unwrap())
            .unwrap();
        store.prepare_sweep(SweepId(4)).unwrap();
        store.record_partial(SweepId(4), 0, 40).unwrap();
        store.record_partial(SweepId(4), 1, 37).unwrap();
        // The process dies here.
        assert_eq!(store.recover_partial_trials(SweepId(4)), 77);
        // A sweep that never got that far honestly reports zero.
        assert_eq!(store.recover_partial_trials(SweepId(9)), 0);
    }

    #[test]
    fn a_session_id_cannot_address_a_path_outside_the_store() {
        assert!(SessionId::parse("../../etc").is_err());
        assert!(SessionId::parse("ar-2026/../x").is_err());
        assert!(SessionId::parse("").is_err());
        assert!(SessionId::parse("ar-20260810T101500Z-1a2b3c4d").is_ok());
    }

    /// `SweepId`'s `Display` IS the on-disk directory name, and
    /// `gc_trial_matrices` parses it back with `strip_prefix("sweep-")`.
    ///
    /// Pinned here because the failure mode is silent: a prettier rendering
    /// (`"sweep #1"`) still writes and still reads within one process, and the
    /// only symptom is a garbage collector that matches nothing and a store that
    /// grows for days without one error line.
    #[test]
    fn the_sweep_directory_name_is_the_name_the_gc_parses_back() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open_under(dir.path(), SessionId::parse("ar-test-0003").unwrap())
            .unwrap();
        for id in [SweepId(1), SweepId(42)] {
            let name = store
                .sweep_dir(id)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(name, format!("sweep-{}", id.0));
            assert_eq!(
                name.strip_prefix("sweep-")
                    .and_then(|n| n.parse::<u64>().ok()),
                Some(id.0),
                "the GC must be able to recover the SweepId from the directory name"
            );
        }
    }

    #[test]
    fn gc_names_its_rule_and_counts_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open_under(dir.path(), SessionId::parse("ar-test-0002").unwrap())
            .unwrap();
        for sweep in [SweepId(1), SweepId(2)] {
            store.prepare_sweep(sweep).unwrap();
            std::fs::write(store.trial_returns_path(sweep, 0), vec![0u8; 1024]).unwrap();
        }
        let mut keep = std::collections::HashSet::new();
        keep.insert(SweepId(1));
        let census = store.gc_trial_matrices(&keep).unwrap();
        assert_eq!(census.kept, 1);
        assert_eq!(census.deleted, 1);
        assert_eq!(census.bytes_reclaimed, 1024);
        assert!(census.rule.contains("best-ever"));
        assert!(store.trial_returns_path(SweepId(1), 0).exists());
        assert!(!store.trial_returns_path(SweepId(2), 0).exists());
    }

    /// The GC counters the verdict renders are FOLDED, not zero.
    ///
    /// Before `Record::GarbageCollected` existed, `gc_deleted_matrices` and
    /// `gc_bytes_reclaimed` were rendered by `AbandonCensus::render` and by
    /// `census_line` on every single report and were written by NOTHING. The
    /// operator was told zero matrices had been deleted while the session had
    /// deleted all of them — the exact shape of "a census that reads clean is
    /// worse than no census, because a reader takes it as evidence".
    #[test]
    fn a_garbage_collection_pass_is_counted_into_the_census() {
        let pass = |sweep: u64, deleted: usize, bytes: u64| Record::GarbageCollected {
            sweep: SweepId(sweep),
            census: GcCensus {
                kept: 1,
                deleted,
                bytes_reclaimed: bytes,
                rule: "keep the best-ever sweep, the promotion candidate and every control"
                    .to_string(),
                free_bytes_after: Some(9_000),
            },
        };
        let records = vec![
            header_record("ar-test"),
            // A pass that reclaimed nothing is still a pass and is still
            // recorded — "the GC found nothing" and "the GC never ran" are
            // different facts.
            pass(1, 0, 0),
            pass(2, 97, 4_096),
            pass(3, 100, 8_192),
        ];
        let session = Session::fold(&records).unwrap();

        // ACCUMULATED across passes. A session that GC'd fifteen times must not
        // report only the fifteenth.
        assert_eq!(session.census.gc_deleted_matrices, 197);
        assert_eq!(session.census.gc_bytes_reclaimed, 12_288);
        // And the counters are visible in the two places every report reads.
        assert!(
            session
                .census
                .render()
                .contains("197 (12288 bytes reclaimed)")
        );
        assert!(session.census_line().contains("gc_deleted=197"));

        // The RULE is named ONCE — on the first pass that actually reclaimed
        // something — so it cannot crowd out the named drops the example list
        // exists to carry.
        let rules = session
            .census
            .examples
            .iter()
            .filter(|(_, hash)| hash.starts_with("gc after"))
            .count();
        assert_eq!(rules, 1, "examples: {:?}", session.census.examples);

        // A GC pass deletes EVIDENCE, never trials.
        assert_eq!(session.n_session(), 0);
    }

    #[test]
    fn block_boundaries_fall_every_four_live_sweeps() {
        let mut records = vec![header_record("ar-test")];
        for sweep in 1..=4u64 {
            records.push(started(sweep));
            records.push(completed(sweep, 100));
        }
        let session = Session::fold(&records).unwrap();
        assert_eq!(session.live_sweeps_run(), 4);
        assert!(session.block_boundary_due());
        assert_eq!(session.live_sweeps_in_block(BlockId(0)).len(), 4);
    }
}
