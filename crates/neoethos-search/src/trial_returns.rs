//! **Every trial's per-period return series**, persisted.
//!
//! This is the single most important thing in the 2026-08-09 measurement slice.
//!
//! Until now a discovery run persisted the WINNER'S SUMMARY: a portfolio, a
//! handful of scalar metrics, an archive of top genes. From a summary you cannot
//! compute the Deflated Sharpe Ratio, and you cannot compute the Probability of
//! Backtest Overfitting — both need the RETURN SERIES OF EVERY TRIAL, because
//! both are statements about the SEARCH, not about the winner. Without them no
//! result this project produces is falsifiable: "we found a strategy with Sharpe
//! 1.8" and "we ran 15 360 trials and the best of them had Sharpe 1.8" are
//! different claims, and only the second one is checkable.
//!
//! So: one row per screened candidate, one column per calendar month of the
//! evaluated span, values in f64. Written next to the discovery ledger.
//!
//! ## Size, stated rather than hoped
//!
//! The matrix is dense f64, row-major, trial-major:
//!
//! ```text
//!   bytes = HEADER (40) + 8 × periods + Σ_rows (2 + len(strategy_id) + 8 + 8 × periods)
//! ```
//!
//! At the real shape — 15 360 trials × 120 months (ten years) and ~18-byte
//! strategy ids — that is
//!
//! ```text
//!   40 + 960 + 15 360 × (2 + 18 + 8 + 960) = 1 000 + 15 360 × 988
//!                                          = 15 176 680 bytes ≈ 14.47 MiB
//! ```
//!
//! per (symbol, timeframe) run. JSON would be roughly 4× that and would round
//! the f64s, so the payload is binary and the metadata is JSON beside it.
//!
//! ## The cap is derived from the disk, not from a constant
//!
//! A hardcoded "keep the first 5 000 trials" is the same class of defect as
//! every other silent cap in this tree: it degrades differently on every
//! machine and says nothing when it bites. Instead the writer probes the free
//! space on the ledger's own filesystem and spends at most
//! `1 / DISK_SHARE_DIVISOR` of it. When the matrix does not fit, rows are
//! dropped BY A RECORDED RULE (a uniform stride, so the retained sample is not
//! the top of a fitness-ordered list) and the manifest states how many were
//! offered, how many written, how many dropped, why, and what the budget was.
//! NO SILENT DROPS.
//!
//! ## It is written AS THE SCREEN RUNS, not at the end
//!
//! Review finding (2026-08-09): the first cut held every row in RAM until the
//! whole parallel screen — Monte-Carlo perturbation and sensitivity launches
//! included, i.e. the long part — had finished, and wrote once afterwards. This
//! project's own record has `exit 137` OOM kills and multi-hour runs that ended
//! with no artifact, so the ONE file that makes a result falsifiable was being
//! lost in precisely the failure mode that actually happens.
//!
//! [`TrialReturnsWriter`] therefore appends each chunk of the screen as that
//! chunk completes and patches the header's trial count after every flush, so a
//! process killed mid-run leaves a SHORTER BUT VALID matrix on disk rather than
//! nothing. A partial matrix with an honest `trials_written` is worth more than
//! a complete one that never reached the disk.
//!
//! ## What this buys
//!
//! Nothing in money. It buys falsifiability: with this file, DSR and PBO become
//! computable, and a claim about a discovered strategy can be checked against
//! the search that produced it.
//!
//! ## The consumer exists — 2026-08-10
//!
//! This section previously read *"Nothing in this workspace READS this matrix
//! yet … the precondition landed, the falsifiability it buys is not purchased
//! until a consumer exists."* [`crate::deflated`] is that consumer.
//!
//! [`TrialReturnsWriter::finish`] decodes the bytes it has just re-read for the
//! hash and computes both statistics from them:
//!
//! * the **Deflated Sharpe Ratio** of the best trial, against `trials_offered`
//!   — the honest denominator, which is the count the SCREEN offered, not the
//!   survivors and not the rows a disk-budget stride retained; and
//! * **PBO by CSCV** over the calendar-month grid.
//!
//! Both land in [`TrialReturnsManifest::statistics`], which the discovery ledger
//! already embeds wholesale (`discovery_ledger.rs`), and both are logged at run
//! end beside the `config_hash`. Either may be a REFUSAL: a matrix too short or
//! too small to support a statistic produces a named reason rather than a
//! confident number, and the refusal travels exactly like a value would.
//!
//! Note that the `pbo` field in `discovery.rs` is a per-candidate CPCV number, a
//! different object entirely — this one is a property of the SEARCH.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::artifact_io::{fnv1a64, write_json_atomic};
use crate::quality::Trade;

/// Schema tag written into the manifest and checked by any reader.
pub const TRIAL_RETURNS_SCHEMA: &str = "neoethos.trial_returns.v1";

/// Magic bytes at the head of the binary payload.
pub const TRIAL_RETURNS_MAGIC: &[u8; 8] = b"NEOTRIAL";

/// Fixed header size in bytes: magic(8) + version(4) + trials(4) + periods(4)
/// + flags(4) + initial_balance(8) + reserved(8).
const HEADER_BYTES: u64 = 40;

/// The writer spends at most `available_bytes / DISK_SHARE_DIVISOR` on one
/// run's trial matrix. 200 = 0.5 % of free space, i.e. ~1 GiB of headroom
/// yields a 5 MiB budget and ~3 GiB yields the full 14.5 MiB matrix described
/// above. The number is a share, not a size, precisely so it means the same
/// thing on the operator's laptop and on a rented box.
pub const DISK_SHARE_DIVISOR: u64 = 200;

/// Budget used when the filesystem probe cannot identify the target's disk.
/// Recorded in the manifest as such — never presented as a measured value.
pub const PROBE_FAILED_BUDGET_BYTES: u64 = 32 * 1024 * 1024;

/// One trial: which candidate it was, and what it returned per period.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialReturnRow {
    pub candidate_index: usize,
    pub strategy_id: String,
    /// One value per period key, in the SAME order as
    /// [`TrialReturnMatrix::period_keys`]. Units: fraction of the run's
    /// `initial_balance`. Periods with no closed trade are exactly `0.0`.
    pub returns: Vec<f64>,
    /// Trades whose entry timestamp fell outside the period grid. Counted, not
    /// dropped silently — a non-zero value here means the grid and the data
    /// disagree and every downstream statistic is short by that much.
    pub trades_outside_grid: usize,
}

/// Every trial of one (symbol, timeframe) run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrialReturnMatrix {
    /// Calendar-month keys, ascending: `year * 12 + month`. The SAME key
    /// `quality::analyze_monthly_consistency` buckets by, so `positive_months`
    /// and this matrix are two views of one thing.
    pub period_keys: Vec<i64>,
    pub rows: Vec<TrialReturnRow>,
    pub initial_balance: f64,
}

/// What was actually written, in numbers a reader can verify.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialReturnsManifest {
    pub schema: String,
    /// File name (not a path) of the binary payload, next to this manifest.
    pub binary_file: String,
    /// `fnv64:…` over the exact bytes written.
    pub binary_hash: String,
    pub timestamp_ms: i64,
    /// `ResolvedConfigStamp::config_hash` of the run that wrote this matrix.
    ///
    /// IDENTITY, not decoration. The manifest is keyed only on
    /// `{SYMBOL}_{TF}.trial_returns.json`, so without this a ledger written by a
    /// run that skipped the quality screen — or ran under a different config, or
    /// whose matrix write failed (non-fatal by design) — would silently attach
    /// the PREVIOUS run's matrix to itself and then assert on that basis that
    /// DSR and PBO are computable. `None` means the writer had no stamp; a
    /// consumer must treat that as "cannot be attributed", not as "matches".
    #[serde(default)]
    pub config_hash: Option<String>,
    pub symbol: String,
    pub timeframe: String,

    pub period_count: usize,
    pub first_period_key: i64,
    pub last_period_key: i64,

    /// Trials the screen offered — the honest denominator for DSR/PBO.
    pub trials_offered: usize,
    pub trials_written: usize,
    pub trials_dropped: usize,
    /// `"all"` or `"stride:{k}"` — how the written subset was chosen.
    pub retention_rule: String,
    /// `None` when nothing was dropped.
    pub drop_reason: Option<String>,
    /// Total trades that fell outside the period grid across the WRITTEN rows.
    pub trades_outside_grid: usize,
    /// The same count over every OFFERED row. Under a retention stride the two
    /// differ, and reporting only the first would understate the disagreement
    /// between the grid and the data in a manifest whose stated purpose is that
    /// nothing is dropped without being counted.
    #[serde(default)]
    pub trades_outside_grid_offered: usize,

    pub bytes_written: u64,
    pub bytes_budget: u64,
    pub disk_available_bytes: u64,
    pub disk_share_divisor: u64,
    /// `"filesystem_probe"` or `"probe_failed_conservative_default"`.
    pub budget_source: String,

    pub dtype: String,
    pub returns_unit: String,
    pub initial_balance: f64,

    /// **What the matrix actually says about the search that produced it.**
    ///
    /// The Deflated Sharpe Ratio and the CSCV/PBO, computed by
    /// [`crate::deflated`] from the very bytes this manifest hashes, at the
    /// moment they are written. Both may be REFUSALS — a refusal is a result and
    /// is recorded as one, so "the matrix was too short to support the
    /// statistic" can never be mistaken for "the statistic was never run".
    ///
    /// `None` only for a manifest written by a build older than 2026-08-10.
    #[serde(default)]
    pub statistics: Option<crate::deflated::TrialStatisticsReport>,
}

/// `<cache_dir>/{SYMBOL}_{TF}.trial_returns.json` — the manifest.
pub fn manifest_path(cache_dir: &str, symbol: &str, tf: &str) -> PathBuf {
    let mut p = PathBuf::from(cache_dir);
    p.push(trial_returns_stem(symbol, tf, "json"));
    p
}

/// `<cache_dir>/{SYMBOL}_{TF}.trial_returns.bin` — the payload.
pub fn binary_path(cache_dir: &str, symbol: &str, tf: &str) -> PathBuf {
    let mut p = PathBuf::from(cache_dir);
    p.push(trial_returns_stem(symbol, tf, "bin"));
    p
}

fn trial_returns_stem(symbol: &str, tf: &str, ext: &str) -> String {
    format!(
        "{}_{}.trial_returns.{ext}",
        symbol.trim().to_ascii_uppercase(),
        tf.trim().to_ascii_uppercase()
    )
}

/// Ascending calendar-month keys covering `[first_ms, last_ms]` inclusive.
///
/// Returns an empty grid for non-positive or inverted timestamps rather than
/// guessing — the caller logs that and writes nothing.
pub fn month_keys_spanning(first_ms: i64, last_ms: i64) -> Vec<i64> {
    if first_ms <= 0 || last_ms <= 0 || last_ms < first_ms {
        return Vec::new();
    }
    let key_of = |ms: i64| -> Option<i64> {
        Utc.timestamp_millis_opt(ms)
            .single()
            .map(|dt| (dt.year() as i64) * 12 + dt.month() as i64)
    };
    let (Some(a), Some(b)) = (key_of(first_ms), key_of(last_ms)) else {
        return Vec::new();
    };
    (a..=b).collect()
}

/// Bucket one trial's trades into the period grid.
///
/// Buckets by ENTRY time, matching `quality::analyze_monthly_consistency`, so a
/// month's number here is the same month's number there. Values are
/// `pnl / initial_balance`.
pub fn period_returns(
    trades: &[Trade],
    period_keys: &[i64],
    initial_balance: f64,
) -> (Vec<f64>, usize) {
    let mut out = vec![0.0_f64; period_keys.len()];
    let mut outside = 0usize;
    if period_keys.is_empty() {
        return (out, trades.len());
    }
    let base = period_keys[0];
    let balance = if initial_balance.is_finite() && initial_balance.abs() > f64::EPSILON {
        initial_balance
    } else {
        // Never divide by zero and never silently emit NaN into a file whose
        // whole purpose is to be statistically usable.
        1.0
    };
    for trade in trades {
        if trade.entry_time <= 0 {
            outside += 1;
            continue;
        }
        let Some(dt) = Utc.timestamp_millis_opt(trade.entry_time).single() else {
            outside += 1;
            continue;
        };
        let key = (dt.year() as i64) * 12 + dt.month() as i64;
        let idx = key - base;
        if idx < 0 || idx as usize >= out.len() {
            outside += 1;
            continue;
        }
        out[idx as usize] += trade.pnl / balance;
    }
    (out, outside)
}

/// Strip Windows' extended-length (`\\?\`) prefix from an absolute path.
///
/// MEASURED on the operator's box (2026-08-09 review): `Path::canonicalize` on
/// Windows returns a VERBATIM path — `\\?\C:\Users\konst\…` — while `sysinfo`
/// reports the mount point as `C:\`. `Path::starts_with` compares components,
/// and a `VerbatimDisk` prefix never matches a `Disk` prefix, so the probe below
/// matched NO disk on every single run: the budget silently became the 32 MiB
/// constant and the manifest recorded `disk_available_bytes: 0`. The module doc
/// claimed the cap came from the disk; on the machine this actually runs on, it
/// did not. Nothing caught it because the unit test exercised the pure
/// arithmetic and never called the probe.
fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    path.to_path_buf()
}

/// Free bytes on the filesystem backing `dir`, or `None` when no mounted disk
/// can be matched to it.
///
/// Longest-matching-mount-point wins, so `C:\Users\…` resolves to `C:\` and a
/// nested mount resolves to the nested one.
pub fn available_bytes_for(dir: &Path) -> Option<u64> {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let target = strip_verbatim_prefix(&canonical);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if target.starts_with(mount) {
            let depth = mount.components().count();
            let available = disk.available_space();
            if best.map(|(d, _)| depth > d).unwrap_or(true) {
                best = Some((depth, available));
            }
        }
    }
    best.map(|(_, available)| available)
}

/// Bytes one row occupies on disk.
fn row_bytes(strategy_id_len: usize, period_count: usize) -> u64 {
    // u16 id length + id bytes + u64 candidate index + f64 per period.
    2 + strategy_id_len as u64 + 8 + 8 * period_count as u64
}

/// How many rows fit in `budget`, given the fixed overhead and a representative
/// row size. Pure, so the cap is testable without a filesystem.
pub fn rows_within_budget(budget_bytes: u64, period_count: usize, mean_row_bytes: u64) -> usize {
    let fixed = HEADER_BYTES + 8 * period_count as u64;
    if budget_bytes <= fixed || mean_row_bytes == 0 {
        return 0;
    }
    ((budget_bytes - fixed) / mean_row_bytes) as usize
}

/// Uniform stride that keeps at most `keep` of `total` rows. `1` = keep all.
pub fn retention_stride(total: usize, keep: usize) -> usize {
    if keep == 0 || total == 0 {
        return 1;
    }
    if total <= keep {
        return 1;
    }
    total.div_ceil(keep)
}

/// Byte offset of the `trials` field inside the fixed header. The streaming
/// writer seeks here after every flush so a killed process leaves a header that
/// agrees with the rows actually on disk.
const HEADER_TRIALS_OFFSET: u64 = 12;

/// Header + period grid. `trials` is patched in place by the streaming writer.
fn encode_header(period_keys: &[i64], initial_balance: f64, trials: usize) -> Vec<u8> {
    let periods = period_keys.len();
    let mut out = Vec::with_capacity((HEADER_BYTES + 8 * periods as u64) as usize);
    out.extend_from_slice(TRIAL_RETURNS_MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes()); // version
    out.extend_from_slice(&(trials as u32).to_le_bytes());
    out.extend_from_slice(&(periods as u32).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // flags bit0: fraction of initial_balance
    out.extend_from_slice(&initial_balance.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // reserved
    for key in period_keys {
        out.extend_from_slice(&key.to_le_bytes());
    }
    debug_assert_eq!(out.len() as u64, HEADER_BYTES + 8 * periods as u64);
    out
}

/// One row, appended to `out`.
fn encode_row_into(out: &mut Vec<u8>, row: &TrialReturnRow, periods: usize) {
    let id = row.strategy_id.as_bytes();
    let len = id.len().min(u16::MAX as usize);
    out.extend_from_slice(&(len as u16).to_le_bytes());
    out.extend_from_slice(&id[..len]);
    out.extend_from_slice(&(row.candidate_index as u64).to_le_bytes());
    // A row whose length disagrees with the grid would corrupt every later
    // row's offset, so it is padded/truncated to the grid and the mismatch
    // is visible as a zero tail rather than as a shifted file.
    for i in 0..periods {
        let v = row.returns.get(i).copied().unwrap_or(0.0);
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Serialise the matrix. Little-endian throughout; f64 for every value,
/// per the project's f64-for-anything-we-consume rule.
pub fn encode(matrix: &TrialReturnMatrix, rows: &[&TrialReturnRow]) -> Vec<u8> {
    let periods = matrix.period_keys.len();
    let mut out = encode_header(&matrix.period_keys, matrix.initial_balance, rows.len());
    out.reserve(rows.len() * row_bytes(24, periods) as usize);
    for row in rows {
        encode_row_into(&mut out, row, periods);
    }
    out
}

/// Conservative per-row size used to derive the retention stride BEFORE any row
/// exists. Overestimating the row only makes the cap tighter, never the budget
/// overrun; strategy ids in this tree are ~18 bytes.
pub const ASSUMED_STRATEGY_ID_BYTES: usize = 32;

/// **Crash-safe, append-as-you-go writer for the trial matrix.**
///
/// Open before the screen starts, `append` each completed chunk, `finish` at the
/// end. The header's trial count is patched after every flush, so killing the
/// process at any point leaves a file whose header agrees with its rows.
///
/// The retention stride is fixed at `open` from `expected_trials`, so the kept
/// subset is the same uniform sample the batch writer produced — the rows arrive
/// fitness-ordered and keeping a prefix would sample only the winners.
pub struct TrialReturnsWriter {
    file: std::fs::File,
    path: PathBuf,
    cache_dir: String,
    symbol: String,
    timeframe: String,
    period_keys: Vec<i64>,
    initial_balance: f64,
    stride: usize,
    budget: u64,
    disk_available: u64,
    budget_source: &'static str,
    mean_row_bytes: u64,
    expected_trials: usize,
    offered: usize,
    written: usize,
    bytes_written: u64,
    outside_written: usize,
    outside_offered: usize,
    /// Identity of the run that produced these rows. Written into the manifest
    /// so a ledger cannot silently attach a PREVIOUS run's matrix to itself.
    config_hash: Option<String>,
}

impl TrialReturnsWriter {
    pub fn open(
        cache_dir: &str,
        symbol: &str,
        tf: &str,
        period_keys: Vec<i64>,
        initial_balance: f64,
        expected_trials: usize,
        config_hash: Option<String>,
    ) -> Result<Self> {
        let dir = PathBuf::from(cache_dir);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create trial-returns directory {}", dir.display()))?;

        let periods = period_keys.len();
        let (disk_available, budget, budget_source) = match available_bytes_for(&dir) {
            Some(available) => (available, available / DISK_SHARE_DIVISOR, "filesystem_probe"),
            None => (
                0,
                PROBE_FAILED_BUDGET_BYTES,
                "probe_failed_conservative_default",
            ),
        };
        let mean_row_bytes = row_bytes(ASSUMED_STRATEGY_ID_BYTES, periods);
        let capacity = rows_within_budget(budget, periods, mean_row_bytes);
        let stride = retention_stride(expected_trials, capacity);

        let path = binary_path(cache_dir, symbol, tf);
        let mut file = std::fs::File::create(&path)
            .with_context(|| format!("create trial-returns payload {}", path.display()))?;
        let header = encode_header(&period_keys, initial_balance, 0);
        file.write_all(&header)
            .with_context(|| format!("write trial-returns header {}", path.display()))?;
        file.flush().ok();
        let bytes_written = header.len() as u64;

        Ok(Self {
            file,
            path,
            cache_dir: cache_dir.to_string(),
            symbol: symbol.trim().to_ascii_uppercase(),
            timeframe: tf.trim().to_ascii_uppercase(),
            period_keys,
            initial_balance,
            stride,
            budget,
            disk_available,
            budget_source,
            mean_row_bytes,
            expected_trials,
            offered: 0,
            written: 0,
            bytes_written,
            outside_written: 0,
            outside_offered: 0,
            config_hash,
        })
    }

    /// Append one completed chunk and patch the header. After this returns Ok,
    /// the file on disk is a valid matrix of `written` rows.
    pub fn append(&mut self, rows: &[TrialReturnRow]) -> Result<()> {
        let periods = self.period_keys.len();
        let mut buf: Vec<u8> = Vec::with_capacity(rows.len() * self.mean_row_bytes as usize);
        for row in rows {
            let keep = self.stride <= 1 || self.offered % self.stride == 0;
            self.offered += 1;
            self.outside_offered += row.trades_outside_grid;
            if !keep {
                continue;
            }
            encode_row_into(&mut buf, row, periods);
            self.written += 1;
            self.outside_written += row.trades_outside_grid;
        }
        if buf.is_empty() {
            return Ok(());
        }
        self.file
            .write_all(&buf)
            .with_context(|| format!("append trial-returns rows to {}", self.path.display()))?;
        self.bytes_written += buf.len() as u64;
        self.patch_trial_count()?;
        Ok(())
    }

    fn patch_trial_count(&mut self) -> Result<()> {
        let end = self.file.stream_position()?;
        self.file.seek(SeekFrom::Start(HEADER_TRIALS_OFFSET))?;
        self.file.write_all(&(self.written as u32).to_le_bytes())?;
        self.file.seek(SeekFrom::Start(end))?;
        self.file.flush().ok();
        Ok(())
    }

    /// Close the payload and write the manifest beside it.
    pub fn finish(mut self, timestamp_ms: i64) -> Result<TrialReturnsManifest> {
        self.patch_trial_count()?;
        self.file
            .sync_all()
            .with_context(|| format!("sync trial-returns payload {}", self.path.display()))?;
        drop(self.file);

        // Hash the FINAL bytes. The header is patched after the rows are
        // written, and FNV-1a is a sequential fold, so a running hash of the
        // rows could not be composed with the final header — the file is read
        // back instead, which also proves it is on disk and readable.
        let bytes = std::fs::read(&self.path)
            .with_context(|| format!("re-read trial-returns payload {}", self.path.display()))?;

        // READ WHAT WE JUST WROTE. Until 2026-08-10 this module's own doc said
        // "nothing in this workspace READS this matrix yet", and a matrix nobody
        // reads makes no result falsifiable. The bytes are already in hand for
        // the hash, so the two statistics that judge the SEARCH — the Deflated
        // Sharpe Ratio and the CSCV/PBO — cost one decode and are computed here,
        // where the trial count is still honest (`self.offered`, not the count
        // of survivors, and not the count of rows a disk-budget stride retained).
        //
        // Never fatal: `analyse_bytes` turns a decode failure into a refusal on
        // both statistics rather than an error, because a run that produced its
        // artifacts must not be failed by its own self-assessment.
        let statistics = crate::deflated::analyse_bytes(&bytes, self.offered);

        let dropped = self.offered.saturating_sub(self.written);
        let retention_rule = if self.stride <= 1 {
            "all".to_string()
        } else {
            format!("stride:{}", self.stride)
        };
        let drop_reason = if dropped > 0 {
            Some(format!(
                "disk headroom: {} bytes available on the ledger filesystem, budget {} bytes \
                 (1/{DISK_SHARE_DIVISOR} share, source {}), {} bytes per row × {} expected rows \
                 does not fit; kept every {}th row",
                self.disk_available,
                self.budget,
                self.budget_source,
                self.mean_row_bytes,
                self.expected_trials,
                self.stride
            ))
        } else {
            None
        };

        let manifest = TrialReturnsManifest {
            schema: TRIAL_RETURNS_SCHEMA.to_string(),
            binary_file: self
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            binary_hash: format!("fnv64:{:016x}", fnv1a64(&bytes)),
            timestamp_ms,
            config_hash: self.config_hash.take(),
            symbol: self.symbol.clone(),
            timeframe: self.timeframe.clone(),
            period_count: self.period_keys.len(),
            first_period_key: self.period_keys.first().copied().unwrap_or(0),
            last_period_key: self.period_keys.last().copied().unwrap_or(0),
            trials_offered: self.offered,
            trials_written: self.written,
            trials_dropped: dropped,
            retention_rule,
            drop_reason,
            trades_outside_grid: self.outside_written,
            trades_outside_grid_offered: self.outside_offered,
            bytes_written: bytes.len() as u64,
            bytes_budget: self.budget,
            disk_available_bytes: self.disk_available,
            disk_share_divisor: DISK_SHARE_DIVISOR,
            budget_source: self.budget_source.to_string(),
            dtype: "f64_le".to_string(),
            returns_unit: "fraction_of_initial_balance".to_string(),
            initial_balance: self.initial_balance,
            statistics: Some(statistics.clone()),
        };
        let man = manifest_path(&self.cache_dir, &self.symbol, &self.timeframe);
        write_json_atomic(&man, &manifest)
            .with_context(|| format!("write trial-returns manifest {}", man.display()))?;

        // PRINTED AT RUN END, NEXT TO THE CONFIG HASH. The caller
        // (`discovery.rs`, the `writer.finish(...)` arm) logs the manifest with
        // `config_hash`; this is the same moment, and both values plus every
        // assumption behind them go to the operator's console rather than only
        // into a JSON file nobody opens. `deflated_sharpe`/`pbo` are emitted as
        // structured fields too, so a log scrape can find them without parsing
        // the rendered block.
        tracing::info!(
            target: "neoethos_search::trial_returns",
            symbol = %manifest.symbol,
            timeframe = %manifest.timeframe,
            config_hash = ?manifest.config_hash,
            trials_offered = manifest.trials_offered,
            trials_written = manifest.trials_written,
            deflated_sharpe = ?statistics.dsr_value(),
            pbo = ?statistics.pbo_value(),
            "trial-return matrix read back — the search judged by its own record\n{}",
            statistics.render()
        );

        Ok(manifest)
    }
}

/// Write the matrix and its manifest next to the discovery ledger.
///
/// Returns the manifest that was written. Errors are the caller's to log; the
/// caller treats a failure here as non-fatal to the run (a discovery result is
/// still a discovery result), but it must SAY SO rather than swallow it.
pub fn write_trial_returns(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
    matrix: &TrialReturnMatrix,
    timestamp_ms: i64,
) -> Result<TrialReturnsManifest> {
    // ONE code path: the batch form is the streaming writer with a single
    // append, so the two can never drift into disagreeing about the stride,
    // the budget or the accounting.
    let mut writer = TrialReturnsWriter::open(
        cache_dir,
        symbol,
        tf,
        matrix.period_keys.clone(),
        matrix.initial_balance,
        matrix.rows.len(),
        None,
    )?;
    writer.append(&matrix.rows)?;
    writer.finish(timestamp_ms)
}

/// Load a previously written manifest, or `None` when absent/unreadable.
/// Fail-soft by design: the ledger embeds this and a missing sidecar must not
/// fail a ledger write.
pub fn load_manifest(cache_dir: &str, symbol: &str, tf: &str) -> Option<TrialReturnsManifest> {
    let path = manifest_path(cache_dir, symbol, tf);
    if !path.exists() {
        return None;
    }
    match crate::artifact_io::read_json::<TrialReturnsManifest>(&path, "trial-returns manifest") {
        Ok(m) => Some(m),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_search::trial_returns",
                path = %path.display(),
                error = %err,
                "trial-returns manifest present but unreadable — the ledger will record its \
                 absence rather than pretend the series were never produced"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(entry_ms: i64, pnl: f64) -> Trade {
        Trade {
            entry_time: entry_ms,
            exit_time: Some(entry_ms + 3_600_000),
            pnl,
            pnl_pct: None,
            duration_hours: None,
            mfe: 0.0,
            mae: 0.0,
            r_multiple: 0.0,
        }
    }

    /// 2024-01-15T00:00:00Z and 2024-03-15T00:00:00Z.
    const JAN_2024: i64 = 1_705_276_800_000;
    const MAR_2024: i64 = 1_710_460_800_000;

    #[test]
    fn month_grid_is_inclusive_and_contiguous() {
        let keys = month_keys_spanning(JAN_2024, MAR_2024);
        assert_eq!(keys.len(), 3, "Jan, Feb, Mar");
        assert_eq!(keys[0], 2024 * 12 + 1);
        assert_eq!(keys[2], 2024 * 12 + 3);
        // Degenerate inputs produce an empty grid, never a guessed one.
        assert!(month_keys_spanning(0, MAR_2024).is_empty());
        assert!(month_keys_spanning(MAR_2024, JAN_2024).is_empty());
    }

    #[test]
    fn returns_bucket_by_entry_month_as_fractions_of_balance() {
        let keys = month_keys_spanning(JAN_2024, MAR_2024);
        let trades = vec![trade(JAN_2024, 100.0), trade(JAN_2024, -40.0), trade(MAR_2024, 20.0)];
        let (r, outside) = period_returns(&trades, &keys, 1000.0);
        assert_eq!(outside, 0);
        assert!((r[0] - 0.06).abs() < 1e-12, "Jan = (100 - 40)/1000");
        assert_eq!(r[1], 0.0, "Feb had no trades");
        assert!((r[2] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn trades_outside_the_grid_are_counted_never_dropped_silently() {
        let keys = month_keys_spanning(JAN_2024, JAN_2024);
        // One inside January, one in March, one with no timestamp.
        let trades = vec![trade(JAN_2024, 10.0), trade(MAR_2024, 10.0), trade(0, 10.0)];
        let (r, outside) = period_returns(&trades, &keys, 100.0);
        assert_eq!(outside, 2);
        assert!((r[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn zero_balance_never_produces_nan_in_a_statistics_file() {
        let keys = month_keys_spanning(JAN_2024, JAN_2024);
        let (r, _) = period_returns(&[trade(JAN_2024, 5.0)], &keys, 0.0);
        assert!(r[0].is_finite());
    }

    #[test]
    fn the_documented_size_arithmetic_is_what_the_encoder_produces() {
        // 3 trials × 4 periods, 6-byte ids.
        let matrix = TrialReturnMatrix {
            period_keys: vec![1, 2, 3, 4],
            initial_balance: 10_000.0,
            rows: (0..3)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: "gene01".to_string(),
                    returns: vec![0.1, -0.2, 0.0, 0.3],
                    trades_outside_grid: 0,
                })
                .collect(),
        };
        let rows: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        let bytes = encode(&matrix, &rows);
        let expected = HEADER_BYTES + 8 * 4 + 3 * row_bytes(6, 4);
        assert_eq!(bytes.len() as u64, expected);
        assert_eq!(&bytes[..8], TRIAL_RETURNS_MAGIC);
        // trial count and period count are readable at fixed offsets.
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 4);
    }

    #[test]
    fn the_real_shape_costs_what_the_module_doc_claims() {
        // 15 360 trials × 120 months, 18-byte ids.
        assert_eq!(row_bytes(18, 120), 988);
        let bytes = HEADER_BYTES + 8 * 120 + 15_360 * row_bytes(18, 120);
        assert_eq!(bytes, 15_176_680);
        assert!(bytes < 16 * 1024 * 1024, "under 16 MiB per run");
    }

    #[test]
    fn the_cap_comes_from_the_budget_not_from_a_constant() {
        let periods = 120;
        let row = row_bytes(18, periods);
        // A 3 GiB-free disk yields a 15 MiB budget: the whole matrix fits.
        let budget = (3 * 1024 * 1024 * 1024_u64) / DISK_SHARE_DIVISOR;
        assert!(rows_within_budget(budget, periods, row) >= 15_360);
        // A 1 GiB-free disk yields ~5.4 MiB: it does not, and the stride says so.
        let tight = (1024 * 1024 * 1024_u64) / DISK_SHARE_DIVISOR;
        let capacity = rows_within_budget(tight, periods, row);
        assert!(capacity < 15_360 && capacity > 0);
        let stride = retention_stride(15_360, capacity);
        assert!(stride >= 2);
        assert!(15_360_usize.div_ceil(stride) <= capacity);
        // A budget smaller than the fixed header keeps nothing, and says so
        // through a stride that the caller reports rather than by writing junk.
        assert_eq!(rows_within_budget(8, periods, row), 0);
        assert_eq!(retention_stride(10, 0), 1);
    }

    #[test]
    fn stride_selection_is_uniform_not_top_of_list() {
        // The rows arrive fitness-ordered; keeping a prefix would sample only
        // the winners, which is exactly the bias PBO exists to detect.
        let rows: Vec<usize> = (0..10).collect();
        let stride = retention_stride(10, 3);
        let kept: Vec<usize> = rows.iter().copied().step_by(stride).collect();
        assert_eq!(stride, 4);
        assert_eq!(kept, vec![0, 4, 8]);
    }

    #[test]
    fn write_round_trips_and_the_manifest_accounts_for_every_row() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_trial_returns_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.to_string_lossy().to_string();

        let matrix = TrialReturnMatrix {
            period_keys: month_keys_spanning(JAN_2024, MAR_2024),
            initial_balance: 10_000.0,
            rows: (0..5)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("gene_{i}"),
                    returns: vec![0.01 * i as f64, 0.0, -0.02],
                    trades_outside_grid: i % 2,
                })
                .collect(),
        };
        let m = write_trial_returns(&cache, "eurusd", "m5", &matrix, 1_717_000_000_000).unwrap();
        assert_eq!(m.schema, TRIAL_RETURNS_SCHEMA);
        assert_eq!(m.trials_offered, 5);
        assert_eq!(
            m.trials_written + m.trials_dropped,
            m.trials_offered,
            "every offered trial is accounted for"
        );
        assert_eq!(m.period_count, 3);
        assert_eq!(m.dtype, "f64_le");
        assert!(binary_path(&cache, "EURUSD", "M5").exists());
        assert_eq!(m.binary_file, "EURUSD_M5.trial_returns.bin");

        let reloaded = load_manifest(&cache, "EURUSD", "M5").expect("manifest");
        assert_eq!(reloaded, m);
        assert!(load_manifest(&cache, "NOPE", "M1").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The probe must work on the HOST, not only in the arithmetic.
    ///
    /// This is the assertion whose absence let `available_bytes_for` return
    /// `None` on every Windows run while the module doc claimed the cap came
    /// from the disk. It calls the probe against a directory that certainly
    /// exists on any machine that can run this test.
    #[test]
    fn the_disk_probe_resolves_a_real_directory_on_this_host() {
        let dir = std::env::temp_dir();
        let available = available_bytes_for(&dir);
        assert!(
            available.is_some(),
            "the filesystem probe found no mounted disk for {} — the budget would silently \
             fall back to the {PROBE_FAILED_BUDGET_BYTES}-byte constant and the manifest would \
             report disk_available_bytes: 0 while claiming to be disk-derived",
            dir.display()
        );
    }

    #[test]
    fn the_verbatim_prefix_is_stripped_so_a_mount_point_can_match() {
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\C:\Users\konst")),
            PathBuf::from(r"C:\Users\konst")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\UNC\server\share")),
            PathBuf::from(r"\\server\share")
        );
        // Anything without the prefix is returned untouched, including POSIX.
        assert_eq!(
            strip_verbatim_prefix(Path::new("/var/lib/neoethos")),
            PathBuf::from("/var/lib/neoethos")
        );
    }

    /// A process killed mid-screen must leave a SHORTER BUT VALID matrix, not
    /// nothing. The writer is never `finish`ed here — that is the whole point.
    #[test]
    fn a_run_killed_mid_screen_leaves_a_readable_partial_matrix() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_trial_partial_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.to_string_lossy().to_string();
        let keys = month_keys_spanning(JAN_2024, MAR_2024);

        let mut w = TrialReturnsWriter::open(
            &cache,
            "eurusd",
            "m5",
            keys.clone(),
            10_000.0,
            10,
            Some("fnv64:deadbeef".to_string()),
        )
        .unwrap();
        let chunk: Vec<TrialReturnRow> = (0..4)
            .map(|i| TrialReturnRow {
                candidate_index: i,
                strategy_id: format!("gene_{i}"),
                returns: vec![0.01, 0.0, -0.02],
                trades_outside_grid: 0,
            })
            .collect();
        w.append(&chunk).unwrap();
        // Simulate the kill: drop the writer without calling `finish`.
        drop(w);

        let bytes = std::fs::read(binary_path(&cache, "EURUSD", "M5")).unwrap();
        assert_eq!(&bytes[..8], TRIAL_RETURNS_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            4,
            "the header must already agree with the rows on disk"
        );
        assert_eq!(
            bytes.len() as u64,
            HEADER_BYTES + 8 * 3 + 4 * row_bytes(6, 3),
            "exactly the flushed rows, no truncated tail"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streamed_chunks_equal_one_batch_write_and_carry_the_run_identity() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_trial_stream_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.to_string_lossy().to_string();
        let keys = month_keys_spanning(JAN_2024, MAR_2024);
        let rows: Vec<TrialReturnRow> = (0..7)
            .map(|i| TrialReturnRow {
                candidate_index: i,
                strategy_id: format!("gene_{i}"),
                returns: vec![0.01 * i as f64, 0.0, -0.02],
                trades_outside_grid: i % 2,
            })
            .collect();

        let mut w = TrialReturnsWriter::open(
            &cache,
            "eurusd",
            "m5",
            keys.clone(),
            10_000.0,
            rows.len(),
            Some("fnv64:0123456789abcdef".to_string()),
        )
        .unwrap();
        for chunk in rows.chunks(3) {
            w.append(chunk).unwrap();
        }
        let m = w.finish(1_717_000_000_000).unwrap();

        assert_eq!(m.trials_offered, 7);
        assert_eq!(m.trials_written + m.trials_dropped, m.trials_offered);
        assert_eq!(m.config_hash.as_deref(), Some("fnv64:0123456789abcdef"));
        // Nothing was dropped here, so the two outside-grid counts agree.
        assert_eq!(m.trades_outside_grid, m.trades_outside_grid_offered);
        assert_eq!(m.trades_outside_grid, 3, "rows 1, 3, 5");

        // Byte-identical to the batch encoder over the same rows.
        let matrix = TrialReturnMatrix {
            period_keys: keys,
            rows: rows.clone(),
            initial_balance: 10_000.0,
        };
        let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        let expected = encode(&matrix, &refs);
        let actual = std::fs::read(binary_path(&cache, "EURUSD", "M5")).unwrap();
        assert_eq!(actual, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **The reader is CALLED, not merely written.**
    ///
    /// A statistic nobody invokes is the same defect one layer out from a matrix
    /// nobody reads. `finish` must attach the statistics block to every manifest
    /// it writes — and on a 3-month, 7-trial matrix both statistics must REFUSE
    /// with a named reason rather than emit a confident number.
    #[test]
    fn finish_reads_the_matrix_back_and_attaches_both_statistics() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_trial_stats_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.to_string_lossy().to_string();
        let keys = month_keys_spanning(JAN_2024, MAR_2024);
        let rows: Vec<TrialReturnRow> = (0..7)
            .map(|i| TrialReturnRow {
                candidate_index: i,
                strategy_id: format!("gene_{i}"),
                returns: vec![0.01 * i as f64, 0.0, -0.02],
                trades_outside_grid: 0,
            })
            .collect();

        let mut w = TrialReturnsWriter::open(
            &cache,
            "eurusd",
            "m5",
            keys,
            10_000.0,
            rows.len(),
            Some("fnv64:cafebabe".to_string()),
        )
        .unwrap();
        w.append(&rows).unwrap();
        let m = w.finish(1_717_000_000_000).unwrap();

        let stats = m.statistics.as_ref().expect("finish must attach statistics");
        assert_eq!(stats.matrix_rows, 7);
        assert_eq!(stats.matrix_periods, 3);
        assert_eq!(stats.trials_offered, 7);
        // Three months and seven trials cannot support either statistic.
        assert!(stats.deflated_sharpe.is_none());
        assert!(stats.pbo.is_none());
        let dsr_refusal = stats.deflated_sharpe_refusal.as_ref().unwrap();
        assert!(
            dsr_refusal.starts_with("REFUSED:"),
            "a refusal is a result and says so: {dsr_refusal}"
        );
        assert!(stats.pbo_refusal.as_ref().unwrap().starts_with("REFUSED:"));

        // And it survives the manifest's own JSON round trip, which is how it
        // reaches the discovery ledger.
        let reloaded = load_manifest(&cache, "EURUSD", "M5").expect("manifest");
        assert_eq!(reloaded.statistics, m.statistics);
        assert_eq!(reloaded, m);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A matrix long and wide enough produces VALUES, through the same call
    /// site — so the refusal test above cannot be passing because the reader is
    /// simply never able to compute anything.
    #[test]
    fn a_long_enough_matrix_produces_values_through_the_same_call_site() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_trial_stats_full_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.to_string_lossy().to_string();
        // 48 months × 40 trials.
        let keys: Vec<i64> = (0..48).map(|i| 24_000 + i).collect();
        let rows: Vec<TrialReturnRow> = (0..40)
            .map(|i| TrialReturnRow {
                candidate_index: i,
                strategy_id: format!("gene_{i}"),
                returns: (0..48)
                    .map(|j| {
                        let wobble = (((i * 31 + j * 17) % 23) as f64 - 11.0) * 0.002;
                        0.001 * i as f64 + wobble
                    })
                    .collect(),
                trades_outside_grid: 0,
            })
            .collect();

        let mut w =
            TrialReturnsWriter::open(&cache, "eurusd", "m5", keys, 10_000.0, rows.len(), None)
                .unwrap();
        w.append(&rows).unwrap();
        let m = w.finish(1_717_000_000_000).unwrap();

        let stats = m.statistics.as_ref().unwrap();
        let dsr = stats.deflated_sharpe.as_ref().expect("DSR computable at 48×40");
        assert!((0.0..=1.0).contains(&dsr.deflated_sharpe_ratio));
        assert_eq!(dsr.trials_n, 40, "the honest N is the row count, not the winners");
        let pbo = stats.pbo.as_ref().expect("PBO computable at 48×40");
        assert!((0.0..=1.0).contains(&pbo.pbo));
        assert!(stats.dsr_value().is_some() && stats.pbo_value().is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
