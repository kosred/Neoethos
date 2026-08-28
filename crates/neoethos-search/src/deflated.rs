//! **The reader for the trial-return matrix, and the two statistics that make a
//! discovery result falsifiable.**
//!
//! [`crate::trial_returns`] writes one row per SCREENED candidate — captured
//! before any gate, because a matrix built from survivors is the selected sample
//! and is precisely what PBO exists to detect. Until 2026-08-10 nothing read it.
//! Its own module doc said so: *"the precondition landed, the falsifiability it
//! buys is not purchased until a consumer exists."* This module is that consumer.
//!
//! Two statistics, both statements about the SEARCH rather than about the winner:
//!
//! * **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014). The probability
//!   that the champion's true Sharpe exceeds the bar implied by how many trials
//!   it was selected from, given the champion's own skew and kurtosis. The raw
//!   Sharpe is reported beside it so the gap is visible; a raw Sharpe quoted
//!   without the trial count it was mined from is not a claim, it is a headline.
//!
//! * **PBO via CSCV** (Bailey, Borwein, López de Prado & Zhu, 2015). Split the
//!   period grid into `S` contiguous groups, take every balanced half as
//!   in-sample, pick the in-sample champion, and measure how often that champion
//!   lands in the bottom half out-of-sample. `PBO > 0.5` means the selection
//!   procedure is worse than choosing at random.
//!
//! ## Refusal is a result
//!
//! Both statistics REFUSE rather than guess. A DSR computed from four monthly
//! observations is not a weak number, it is a wrong one — the skew and kurtosis
//! terms it corrects with are pure noise at that length, and the correction can
//! move the answer either way. Every refusal names WHICH precondition failed and
//! WHY, and refusals travel into the manifest and the ledger exactly like values
//! do, so "no number" is recorded rather than looking like "not run".
//!
//! ## Every output states its own assumptions
//!
//! A bare `0.87` is unusable: deflated against how many trials? Sharpe per month
//! or per year? Which trials were excluded and in which direction does that bias
//! the answer? Each report therefore carries an `assumptions` list that is
//! rendered next to the value, and the renderer prints them — they are not
//! documentation, they are part of the output.
//!
//! ## No new dependencies
//!
//! `Φ` and `Φ⁻¹` are implemented here in closed form (Abramowitz & Stegun 7.1.26
//! for the CDF, Acklam's rational approximation for the quantile) with their
//! error bounds stated in the assumptions, rather than pulling a statistics crate
//! into a workspace that is trying to shrink its dependency surface.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data_selection::CanonicalSearchInputReceiptV2;
use crate::trial_returns::{TRIAL_RETURNS_MAGIC, binary_path};

/// Schema tag for the statistics block embedded in the trial-returns manifest.
pub const TRIAL_STATISTICS_SCHEMA: &str = "neoethos.trial_statistics.v1";

/// Calendar months per year — the period grid is monthly (see
/// [`crate::trial_returns::month_keys_spanning`]).
pub const PERIODS_PER_YEAR: f64 = 12.0;

/// Euler–Mascheroni constant, used by the expected-maximum-Sharpe estimator.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

// ---------------------------------------------------------------------------
// Refusal thresholds. Named constants, not magic numbers buried in a branch:
// every one of them appears verbatim in the refusal text it produces.
// ---------------------------------------------------------------------------

/// Below this many periods the third and fourth moments the DSR corrects with
/// are noise, and the correction is as likely to flatter as to deflate.
pub const MIN_PERIODS_FOR_DSR: usize = 24;

/// The DSR's selection adjustment needs the VARIANCE of the Sharpe estimates
/// across trials. Estimating a variance from a handful of trials produces an
/// adjustment with no information in it.
pub const MIN_TRIALS_FOR_DSR: usize = 10;

/// CSCV needs at least [`MIN_CSCV_GROUPS`] groups of [`MIN_PERIODS_PER_GROUP`]
/// periods each.
pub const MIN_CSCV_GROUPS: usize = 8;
/// Upper bound on the group count: `C(18,9)` combinations already costs more
/// than the whole statistic is worth at the end of a run.
pub const MAX_CSCV_GROUPS: usize = 16;
/// A group holding a single period gives a Sharpe with zero degrees of freedom.
pub const MIN_PERIODS_PER_GROUP: usize = 2;
/// `MIN_CSCV_GROUPS * MIN_PERIODS_PER_GROUP`.
pub const MIN_PERIODS_FOR_PBO: usize = MIN_CSCV_GROUPS * MIN_PERIODS_PER_GROUP;
/// Ranking the champion among fewer trials than this makes the relative rank —
/// and therefore the logit PBO is a fraction of — too coarse to mean anything.
pub const MIN_TRIALS_FOR_PBO: usize = 10;

/// Work budget for the CSCV sweep, in `combinations × trials × groups` units.
/// The group count is chosen as the LARGEST that fits this budget, so the
/// statistic gets finer on small searches and stays bounded on huge ones. At the
/// real shape (15 360 trials, 120 months) this selects S = 10.
pub const CSCV_WORK_BUDGET: u128 = 64_000_000;

/// Performance stand-in for a sub-sample whose returns are constant. Finite, so
/// it can never poison a mean or a comparison with NaN.
const CONSTANT_SERIES_PERFORMANCE: f64 = 1.0e12;

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// One decoded trial.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedTrialRow {
    pub candidate_index: usize,
    pub strategy_id: String,
    pub returns: Vec<f64>,
}

/// A decoded trial-return matrix, plus what the decode itself observed.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedTrialMatrix {
    pub version: u32,
    pub flags: u32,
    pub initial_balance: f64,
    pub period_keys: Vec<i64>,
    pub rows: Vec<DecodedTrialRow>,
    /// The trial count the HEADER claims. The streaming writer patches it after
    /// every flush, so it should equal `rows.len()`; when it does not, the file
    /// was cut mid-row and [`Self::truncated_tail_bytes`] says by how much.
    pub header_trials: usize,
    /// Bytes after the last complete row. Non-zero means a partial row was
    /// discarded — counted, never silently absorbed.
    pub truncated_tail_bytes: usize,
}

impl DecodedTrialMatrix {
    pub fn periods(&self) -> usize {
        self.period_keys.len()
    }
    pub fn trials(&self) -> usize {
        self.rows.len()
    }
    /// True when the rows on disk disagree with the header's own count.
    pub fn is_truncated(&self) -> bool {
        self.truncated_tail_bytes > 0 || self.header_trials != self.rows.len()
    }
}

/// Why a matrix could not be decoded. A typed error rather than a string so the
/// caller cannot accidentally treat "no file" as "empty search".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes than the fixed 40-byte header.
    TooShortForHeader { bytes: usize },
    /// The magic did not match `NEOTRIAL`.
    BadMagic { found: [u8; 8] },
    /// A version this build does not know how to read.
    UnsupportedVersion { version: u32 },
    /// The header's period count does not fit in the file.
    TruncatedPeriodGrid { periods: usize, bytes: usize },
    /// A declared field ran past the end of the buffer.
    TruncatedRow { row: usize },
    /// A strategy id that is not UTF-8.
    BadStrategyId { row: usize },
    /// The file exists but holds no periods at all.
    EmptyPeriodGrid,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShortForHeader { bytes } => write!(
                f,
                "trial matrix is {bytes} bytes, shorter than the 40-byte header — nothing was \
                 ever flushed"
            ),
            Self::BadMagic { found } => write!(
                f,
                "trial matrix magic is {found:?}, expected {TRIAL_RETURNS_MAGIC:?} — this is not \
                 a trial-return matrix"
            ),
            Self::UnsupportedVersion { version } => write!(
                f,
                "trial matrix declares version {version}; this build reads version 1 only and \
                 will not guess at a layout it does not know"
            ),
            Self::TruncatedPeriodGrid { periods, bytes } => write!(
                f,
                "trial matrix header declares {periods} periods but the file is {bytes} bytes — \
                 the period grid itself is incomplete"
            ),
            Self::TruncatedRow { row } => {
                write!(f, "trial matrix row {row} runs past the end of the file")
            }
            Self::BadStrategyId { row } => {
                write!(f, "trial matrix row {row} has a non-UTF-8 strategy id")
            }
            Self::EmptyPeriodGrid => write!(
                f,
                "trial matrix has zero periods — no return series can be read from it"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn le_u64(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

fn le_i64(bytes: &[u8], at: usize) -> i64 {
    le_u64(bytes, at) as i64
}

fn le_f64(bytes: &[u8], at: usize) -> f64 {
    f64::from_bits(le_u64(bytes, at))
}

/// Decode the binary payload written by [`crate::trial_returns`].
///
/// Layout, read from the writer rather than assumed:
/// `magic[8] | version u32 | trials u32 | periods u32 | flags u32 |
///  initial_balance f64 | reserved u64` (40 bytes), then `periods × i64` keys,
/// then per row `id_len u16 | id bytes | candidate_index u64 | periods × f64`.
/// Little-endian throughout.
///
/// A file cut mid-row decodes to the complete rows that precede the cut, with
/// the leftover byte count reported — that is the documented crash-safety
/// contract of the writer, and honouring it here is what makes a killed run's
/// matrix usable instead of discarded.
pub fn decode(bytes: &[u8]) -> Result<DecodedTrialMatrix, DecodeError> {
    const HEADER: usize = 40;
    if bytes.len() < HEADER {
        return Err(DecodeError::TooShortForHeader { bytes: bytes.len() });
    }
    if &bytes[..8] != TRIAL_RETURNS_MAGIC {
        let mut found = [0u8; 8];
        found.copy_from_slice(&bytes[..8]);
        return Err(DecodeError::BadMagic { found });
    }
    let version = le_u32(bytes, 8);
    if version != 1 {
        return Err(DecodeError::UnsupportedVersion { version });
    }
    let header_trials = le_u32(bytes, 12) as usize;
    let periods = le_u32(bytes, 16) as usize;
    let flags = le_u32(bytes, 20);
    let initial_balance = le_f64(bytes, 24);
    // bytes[32..40] reserved.

    if periods == 0 {
        return Err(DecodeError::EmptyPeriodGrid);
    }
    let grid_end = HEADER
        .checked_add(
            periods
                .checked_mul(8)
                .ok_or(DecodeError::TruncatedPeriodGrid {
                    periods,
                    bytes: bytes.len(),
                })?,
        )
        .ok_or(DecodeError::TruncatedPeriodGrid {
            periods,
            bytes: bytes.len(),
        })?;
    if bytes.len() < grid_end {
        return Err(DecodeError::TruncatedPeriodGrid {
            periods,
            bytes: bytes.len(),
        });
    }
    let mut period_keys = Vec::with_capacity(periods);
    for i in 0..periods {
        period_keys.push(le_i64(bytes, HEADER + i * 8));
    }

    let mut rows: Vec<DecodedTrialRow> = Vec::new();
    let mut cursor = grid_end;
    loop {
        if cursor == bytes.len() {
            break;
        }
        let row_index = rows.len();
        // A partial trailing row is the killed-mid-write case: stop, and report
        // the leftover instead of pretending the file is clean.
        if cursor + 2 > bytes.len() {
            break;
        }
        let id_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        let id_start = cursor + 2;
        let idx_start = id_start + id_len;
        let returns_start = idx_start + 8;
        let row_end = returns_start + periods * 8;
        if row_end > bytes.len() {
            break;
        }
        let strategy_id = std::str::from_utf8(&bytes[id_start..idx_start])
            .map_err(|_| DecodeError::BadStrategyId { row: row_index })?
            .to_string();
        let candidate_index = le_u64(bytes, idx_start) as usize;
        let mut returns = Vec::with_capacity(periods);
        for i in 0..periods {
            returns.push(le_f64(bytes, returns_start + i * 8));
        }
        rows.push(DecodedTrialRow {
            candidate_index,
            strategy_id,
            returns,
        });
        cursor = row_end;
    }
    // Guard the pathological case where the loop made no progress at all on a
    // non-empty remainder, so a corrupt length field cannot look like clean EOF.
    let truncated_tail_bytes = bytes.len() - cursor;
    if truncated_tail_bytes > 0 && rows.is_empty() && header_trials > 0 {
        return Err(DecodeError::TruncatedRow { row: 0 });
    }

    Ok(DecodedTrialMatrix {
        version,
        flags,
        initial_balance,
        period_keys,
        rows,
        header_trials,
        truncated_tail_bytes,
    })
}

/// Read and decode only the payload whose manifest validates against the exact
/// canonical receipt and resolved configuration.
pub fn read_matrix(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
    expected_receipt: &CanonicalSearchInputReceiptV2,
    expected_config_hash: &str,
) -> anyhow::Result<DecodedTrialMatrix> {
    crate::trial_returns::load_manifest(
        cache_dir,
        symbol,
        tf,
        expected_receipt,
        expected_config_hash,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "no receipt-bound trial-returns manifest for {symbol}/{tf} and config {expected_config_hash}"
        )
    })?;
    read_matrix_at(&binary_path(
        cache_dir,
        expected_receipt,
        expected_config_hash,
    )?)
}

/// Internal decode helper. External consumers must enter through
/// [`read_matrix`], which validates the receipt/config-bound manifest first.
fn read_matrix_at(path: &Path) -> anyhow::Result<DecodedTrialMatrix> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("read trial matrix {}: {e}", path.display()))?;
    decode(&bytes).map_err(|e| anyhow::anyhow!("decode trial matrix {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Normal distribution, in closed form
// ---------------------------------------------------------------------------

/// `erf` via Abramowitz & Stegun 7.1.26. Absolute error ≤ 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1_f64;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Standard normal CDF. Absolute error ≤ 1.5e-7 (inherited from [`erf`]).
pub fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal quantile (Acklam's rational approximation). Relative error
/// ≤ 1.15e-9 on `(0, 1)`. Returns `None` outside the open unit interval rather
/// than clamping to an infinity that would silently propagate.
pub fn normal_ppf(p: f64) -> Option<f64> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return None;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    x.is_finite().then_some(x)
}

// ---------------------------------------------------------------------------
// Moments
// ---------------------------------------------------------------------------

/// Mean, sample standard deviation (n−1), skewness and NON-EXCESS kurtosis
/// (3.0 for a normal). `None` when the series is too short or not finite.
fn moments(xs: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let n = xs.len();
    if n < 4 || xs.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for x in xs {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    if m2 <= 0.0 || !m2.is_finite() {
        return None;
    }
    let sd_sample = (m2 * nf / (nf - 1.0)).sqrt();
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2);
    if ![mean, sd_sample, skew, kurt].iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((mean, sd_sample, skew, kurt))
}

/// Per-period Sharpe of a return series, or `None` when it is constant/degenerate.
fn per_period_sharpe(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 || xs.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let nf = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    if var <= 0.0 || !var.is_finite() {
        return None;
    }
    let sr = mean / var.sqrt();
    sr.is_finite().then_some(sr)
}

// ---------------------------------------------------------------------------
// Deflated Sharpe Ratio
// ---------------------------------------------------------------------------

/// The champion, deflated by the size of the search that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeflatedSharpeReport {
    /// Honest trial count used in the selection adjustment.
    pub trials_n: usize,
    /// Where `trials_n` came from — the matrix rows, or the larger offered count
    /// when a retention stride sub-sampled the file.
    pub trials_n_source: String,
    /// Trials that had a computable Sharpe; the cross-trial variance uses these.
    pub trials_with_sharpe: usize,
    pub periods: usize,
    pub period_unit: String,

    pub champion_strategy_id: String,
    pub champion_candidate_index: usize,

    /// Raw, per period. Reported beside the deflated value so the gap is visible.
    pub raw_sharpe_per_period: f64,
    /// Raw, annualised by √12 — the number a human recognises.
    pub raw_sharpe_annualised: f64,
    pub skewness: f64,
    /// Non-excess kurtosis (3.0 = normal).
    pub kurtosis: f64,

    /// Variance of the per-period Sharpe estimates ACROSS trials.
    pub sharpe_variance_across_trials: f64,
    /// `SR*` — the Sharpe a candidate is expected to reach by luck alone after
    /// `trials_n` trials. Per period.
    pub expected_max_sharpe_per_period: f64,
    pub expected_max_sharpe_annualised: f64,
    /// `raw − SR*`, per period. Negative means the champion did not even beat
    /// what the search would have produced from noise.
    pub excess_over_expected_max_per_period: f64,

    /// The statistic: `P(true Sharpe > SR*)` in `[0, 1]`.
    pub deflated_sharpe_ratio: f64,

    pub assumptions: Vec<String>,
}

/// `SR*` — the per-period Sharpe a candidate is expected to reach by luck alone
/// after `trials_n` independent trials.
///
/// Bailey & López de Prado (2014):
/// `SR* = sqrt(V) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]`
///
/// **One implementation.** [`deflated_sharpe`] computes it from a matrix and
/// [`redeflate_sharpe`] recomputes it at a larger `N`; a second copy of this
/// formula anywhere would let the two answers drift apart silently, and the
/// whole point of the deflation is that N is not negotiable.
fn expected_max_sharpe(sharpe_variance_across_trials: f64, trials_n: usize) -> Result<f64, String> {
    let n = trials_n as f64;
    let (Some(q1), Some(q2)) = (
        normal_ppf(1.0 - 1.0 / n),
        normal_ppf(1.0 - 1.0 / (n * std::f64::consts::E)),
    ) else {
        return Err(format!(
            "REFUSED: the expected-maximum quantiles are not representable at N={trials_n}."
        ));
    };
    let sr_star = sharpe_variance_across_trials.sqrt()
        * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2);
    if !sr_star.is_finite() {
        return Err(
            "REFUSED: the expected maximum Sharpe evaluated to a non-finite value.".to_string(),
        );
    }
    Ok(sr_star)
}

/// `DSR = Φ( (SR − SR*)·√(T−1) / √(1 − γ3·SR + ((γ4−1)/4)·SR²) )`
///
/// The champion's own moments, and the selection-adjusted benchmark it must
/// beat. One implementation, for the same reason as [`expected_max_sharpe`].
fn dsr_from_moments(
    raw_sr: f64,
    sr_star: f64,
    skew: f64,
    kurt: f64,
    periods: usize,
) -> Result<f64, String> {
    let t = periods as f64;
    let denom_sq = 1.0 - skew * raw_sr + ((kurt - 1.0) / 4.0) * raw_sr * raw_sr;
    if denom_sq <= 0.0 || !denom_sq.is_finite() {
        return Err(format!(
            "REFUSED: the DSR variance term is {denom_sq:.6} (≤ 0) at skew={skew:.4}, \
             kurtosis={kurt:.4}, Sharpe={raw_sr:.4}. The statistic's asymptotic variance is \
             undefined there; a value printed anyway would be a square root of a negative number \
             dressed up as a probability."
        ));
    }
    let z = (raw_sr - sr_star) * (t - 1.0).sqrt() / denom_sq.sqrt();
    let dsr = normal_cdf(z);
    if !dsr.is_finite() {
        return Err("REFUSED: the DSR evaluated to a non-finite value.".to_string());
    }
    Ok(dsr)
}

/// Re-deflate an existing report against a LARGER trial count, **without the
/// matrix**.
///
/// `N` enters the DSR in exactly one place — through `SR*`, the expected maximum
/// Sharpe over `N` draws. Nothing else in the statistic depends on it: the
/// champion's raw Sharpe, its skew, its kurtosis, the period count and the
/// cross-trial Sharpe variance are all properties of the matrix alone and are
/// carried on the report. So the honest N can be installed after the fact, by
/// arithmetic, and the answer is bit-for-bit what re-reading the matrix would
/// have produced.
///
/// This exists because **the party that runs a search cannot know the honest
/// N.** In a research loop the denominator is every candidate the whole session
/// offered, and that number is not final until the session is. A search
/// deflated against its own trial count is not deflated; it is flattered by
/// orders of magnitude. The runner therefore re-deflates every report against
/// the session-wide fold once that fold is known.
///
/// `trials_offered` is a FLOOR, never a ceiling: the result uses
/// `max(trials_offered, report.trials_n)`, so this can only ever deflate
/// further. There is deliberately no path here that makes a number look better.
pub fn redeflate_sharpe(
    report: &DeflatedSharpeReport,
    trials_offered: usize,
) -> Result<DeflatedSharpeReport, String> {
    let trials_n = trials_offered.max(report.trials_n);
    if trials_n == report.trials_n {
        return Ok(report.clone());
    }
    let sr_star = expected_max_sharpe(report.sharpe_variance_across_trials, trials_n)?;
    let dsr = dsr_from_moments(
        report.raw_sharpe_per_period,
        sr_star,
        report.skewness,
        report.kurtosis,
        report.periods,
    )?;

    let ann = PERIODS_PER_YEAR.sqrt();
    let mut out = report.clone();
    let was = report.trials_n;
    out.trials_n = trials_n;
    out.trials_n_source = format!(
        "trials_offered={trials_n}, installed after the fact by redeflate_sharpe (the search \
         itself was analysed at N={was}, which is only its OWN trial count and not the honest \
         denominator)"
    );
    out.expected_max_sharpe_per_period = sr_star;
    out.expected_max_sharpe_annualised = sr_star * ann;
    out.excess_over_expected_max_per_period = report.raw_sharpe_per_period - sr_star;
    out.deflated_sharpe_ratio = dsr;
    out.assumptions.push(format!(
        "RE-DEFLATED from N={was} to N={trials_n}. N enters only through SR*, so the raw Sharpe, \
         skew, kurtosis, period count and cross-trial Sharpe variance are unchanged and this is \
         exactly the number a re-analysis of the matrix at N={trials_n} would give. SR* moved \
         {:+.6} per period and the DSR moved {:+.6}.",
        sr_star - report.expected_max_sharpe_per_period,
        dsr - report.deflated_sharpe_ratio
    ));
    Ok(out)
}

/// Compute the DSR, or say why not.
pub fn deflated_sharpe(
    matrix: &DecodedTrialMatrix,
    trials_offered: usize,
) -> Result<DeflatedSharpeReport, String> {
    let periods = matrix.periods();
    let rows = matrix.trials();
    if periods < MIN_PERIODS_FOR_DSR {
        return Err(format!(
            "REFUSED: the matrix spans {periods} monthly period(s); the Deflated Sharpe Ratio \
             needs at least {MIN_PERIODS_FOR_DSR}. Its correction is driven by the skew and \
             kurtosis of the champion's own returns, and those two moments are noise at this \
             length — the 'deflated' number could come out above or below the raw one for \
             reasons that have nothing to do with the strategy."
        ));
    }
    if rows < MIN_TRIALS_FOR_DSR {
        return Err(format!(
            "REFUSED: the matrix holds {rows} trial(s); the selection adjustment needs the \
             VARIANCE of the Sharpe estimates across at least {MIN_TRIALS_FOR_DSR} trials. With \
             fewer, that variance is not estimated, it is invented, and the whole deflation term \
             rests on it."
        ));
    }

    // Per-trial Sharpe, for both the champion and the cross-trial variance.
    let mut sharpes: Vec<f64> = Vec::with_capacity(rows);
    let mut best: Option<(usize, f64)> = None;
    for (i, row) in matrix.rows.iter().enumerate() {
        if let Some(sr) = per_period_sharpe(&row.returns) {
            sharpes.push(sr);
            if best.map(|(_, b)| sr > b).unwrap_or(true) {
                best = Some((i, sr));
            }
        }
    }
    if sharpes.len() < MIN_TRIALS_FOR_DSR {
        return Err(format!(
            "REFUSED: only {} of {rows} trials have a computable Sharpe (the rest have a constant \
             or non-finite return series — typically candidates that closed no trade inside the \
             period grid). {MIN_TRIALS_FOR_DSR} are needed for the cross-trial variance.",
            sharpes.len()
        ));
    }
    let Some((best_idx, raw_sr)) = best else {
        return Err("REFUSED: no trial in the matrix has a computable Sharpe.".to_string());
    };

    let champion = &matrix.rows[best_idx];
    let Some((_mean, _sd, skew, kurt)) = moments(&champion.returns) else {
        return Err(format!(
            "REFUSED: the champion ({}) has a degenerate or non-finite return series, so its skew \
             and kurtosis — which the DSR corrects with — do not exist.",
            champion.strategy_id
        ));
    };

    // Variance of the Sharpe estimates across trials (sample, n−1).
    let nf = sharpes.len() as f64;
    let sr_mean = sharpes.iter().sum::<f64>() / nf;
    let sr_var = sharpes.iter().map(|s| (s - sr_mean).powi(2)).sum::<f64>() / (nf - 1.0);
    if sr_var <= 0.0 || !sr_var.is_finite() {
        return Err(format!(
            "REFUSED: the Sharpe estimates across the {} trials have zero (or non-finite) \
             variance, so the expected maximum from the search is undefined. Every trial scoring \
             identically usually means the matrix rows are duplicates, not that the search was \
             free of selection.",
            sharpes.len()
        ));
    }

    // Honest N: never smaller than the rows present. When a retention stride
    // sub-sampled the file, the search really ran `trials_offered` trials and
    // using the smaller number would UNDER-deflate — the flattering direction.
    let (trials_n, trials_n_source) = if trials_offered > rows {
        (
            trials_offered,
            format!(
                "trials_offered={trials_offered} from the writer (larger than the {rows} rows on \
                 disk: a retention stride sub-sampled the file; using the smaller count would \
                 under-deflate)"
            ),
        )
    } else {
        (rows, format!("rows in the matrix ({rows})"))
    };

    let sr_star = expected_max_sharpe(sr_var, trials_n)?;
    let dsr = dsr_from_moments(raw_sr, sr_star, skew, kurt, periods)?;

    let ann = PERIODS_PER_YEAR.sqrt();
    let assumptions = vec![
        format!(
            "N = {trials_n} trials ({trials_n_source}). This is the honest denominator: every \
             screened candidate, not the survivors."
        ),
        format!(
            "Periods are {periods} CALENDAR MONTHS. Sharpe is computed per month and annualised \
             by ×√12, which assumes months are independent and identically distributed — they \
             are not, so the annualised figure is a convention for comparison, not an estimate."
        ),
        "Benchmark Sharpe is 0: the returns in this matrix are already net of spread, commission \
         and swap, and no risk-free rate is subtracted."
            .to_string(),
        format!(
            "The cross-trial Sharpe variance ({sr_var:.6e}) is estimated from the {} trials in \
             this file that had a computable Sharpe. If the file was sub-sampled by a retention \
             stride, that variance comes from the sub-sample while N uses the full offered count.",
            sharpes.len()
        ),
        "The DSR is a PROBABILITY that the true Sharpe exceeds the selection-adjusted benchmark \
         SR*. It is not a Sharpe ratio and must not be compared against a Sharpe floor."
            .to_string(),
        "Φ and Φ⁻¹ are closed-form approximations: |error| ≤ 1.5e-7 absolute for the CDF, \
         ≤ 1.15e-9 relative for the quantile. Both are far below the sampling error at any \
         realistic period count."
            .to_string(),
        "Trials are treated as independent draws. Genetic search produces correlated trials, \
         which makes the EFFECTIVE trial count lower than N and this DSR therefore CONSERVATIVE \
         (it deflates more than a perfectly correlated search would warrant)."
            .to_string(),
    ];

    Ok(DeflatedSharpeReport {
        trials_n,
        trials_n_source,
        trials_with_sharpe: sharpes.len(),
        periods,
        period_unit: "calendar_month".to_string(),
        champion_strategy_id: champion.strategy_id.clone(),
        champion_candidate_index: champion.candidate_index,
        raw_sharpe_per_period: raw_sr,
        raw_sharpe_annualised: raw_sr * ann,
        skewness: skew,
        kurtosis: kurt,
        sharpe_variance_across_trials: sr_var,
        expected_max_sharpe_per_period: sr_star,
        expected_max_sharpe_annualised: sr_star * ann,
        excess_over_expected_max_per_period: raw_sr - sr_star,
        deflated_sharpe_ratio: dsr,
        assumptions,
    })
}

// ---------------------------------------------------------------------------
// PBO via CSCV
// ---------------------------------------------------------------------------

/// Probability of Backtest Overfitting, and everything needed to check it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PboReport {
    /// The statistic: fraction of splits whose in-sample champion landed in the
    /// BOTTOM half out-of-sample. `> 0.5` = worse than random selection.
    pub pbo: f64,
    /// Balanced splits evaluated — `C(groups, groups/2)`, all of them.
    pub combinations: usize,
    pub groups: usize,
    /// Why this group count and not another.
    pub group_choice_reason: String,
    pub periods: usize,
    pub smallest_group_periods: usize,
    pub largest_group_periods: usize,
    /// Trials ranked in each split.
    pub trials_ranked: usize,
    /// Trials excluded because their whole return series is constant.
    pub trials_excluded_constant: usize,
    /// Splits in which the champion's out-of-sample series was constant, so its
    /// rank rests on the sign-only stand-in rather than on a Sharpe. Non-zero
    /// here means that many of the splits below rank on less information than
    /// the statistic assumes.
    pub degenerate_oos_evaluations: usize,
    /// Mean logit of the champion's out-of-sample relative rank. Negative means
    /// the champion is typically below the out-of-sample median.
    pub mean_logit: f64,
    /// Median out-of-sample relative rank of the champion, in `[0, 1]`.
    pub median_relative_rank: f64,
    pub assumptions: Vec<String>,
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i as u128 + 1);
    }
    acc
}

/// Every `k`-subset of `0..n`, in lexicographic order. Deterministic.
fn for_each_combination<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
    if k == 0 || k > n {
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        f(&idx);
        let mut i = k;
        let mut advanced = false;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                advanced = true;
                break;
            }
        }
        if !advanced {
            return;
        }
    }
}

/// Contiguous, near-equal group sizes covering `periods`. Time order preserved:
/// group `g` holds a contiguous block of months, which is what makes the
/// out-of-sample half an out-of-TIME half.
fn group_bounds(periods: usize, groups: usize) -> Vec<(usize, usize)> {
    let base = periods / groups;
    let rem = periods % groups;
    let mut out = Vec::with_capacity(groups);
    let mut start = 0usize;
    for g in 0..groups {
        let len = base + usize::from(g < rem);
        out.push((start, start + len));
        start += len;
    }
    out
}

/// Pooled Sharpe over a set of groups, from pre-aggregated moments.
/// Order-independent, so pooling is exact rather than an approximation of
/// concatenation.
fn pooled_performance(sum: f64, sumsq: f64, count: f64) -> (f64, bool) {
    if count <= 0.0 {
        return (0.0, true);
    }
    let mean = sum / count;
    let var = (sumsq / count) - mean * mean;
    if var <= 0.0 || !var.is_finite() {
        // Constant sub-sample: rank by sign, never by a division by zero.
        let perf = if mean > 0.0 {
            CONSTANT_SERIES_PERFORMANCE
        } else if mean < 0.0 {
            -CONSTANT_SERIES_PERFORMANCE
        } else {
            0.0
        };
        return (perf, true);
    }
    (mean / var.sqrt(), false)
}

/// Compute PBO by CSCV, or say why not.
pub fn pbo_cscv(matrix: &DecodedTrialMatrix) -> Result<PboReport, String> {
    let periods = matrix.periods();
    if periods < MIN_PERIODS_FOR_PBO {
        return Err(format!(
            "REFUSED: the matrix spans {periods} monthly period(s); CSCV needs at least \
             {MIN_PERIODS_FOR_PBO} so it can form {MIN_CSCV_GROUPS} groups of \
             {MIN_PERIODS_PER_GROUP} months. Below that the 'out-of-sample' half is a handful of \
             months and the resulting fraction is a coin flip with a decimal point."
        ));
    }

    // Constant rows are candidates that closed no trade inside the grid. They
    // can never be the in-sample champion, and keeping them would only inflate
    // the rank denominator — which pushes the champion's relative rank UP and
    // PBO DOWN, i.e. towards the flattering answer. Excluded and counted.
    let usable: Vec<&DecodedTrialRow> = matrix
        .rows
        .iter()
        .filter(|r| {
            r.returns.iter().all(|v| v.is_finite()) && r.returns.windows(2).any(|w| w[0] != w[1])
        })
        .collect();
    let excluded = matrix.rows.len() - usable.len();
    let n_trials = usable.len();
    if n_trials < MIN_TRIALS_FOR_PBO {
        return Err(format!(
            "REFUSED: {n_trials} trial(s) have a non-constant, finite return series ({excluded} of \
             {} were excluded as constant/non-finite). CSCV needs at least {MIN_TRIALS_FOR_PBO} to \
             rank: with fewer, the champion's relative rank can only take a few discrete values \
             and the resulting probability is an artefact of the grid, not of the search.",
            matrix.rows.len()
        ));
    }

    // Largest group count that fits the work budget — finer on small searches,
    // bounded on huge ones.
    let mut chosen: Option<(usize, usize)> = None; // (groups, combinations)
    let mut candidate_groups = MAX_CSCV_GROUPS - (MAX_CSCV_GROUPS % 2);
    while candidate_groups >= MIN_CSCV_GROUPS {
        if periods / candidate_groups >= MIN_PERIODS_PER_GROUP {
            let combos = binomial(candidate_groups, candidate_groups / 2);
            let work = combos * n_trials as u128 * candidate_groups as u128;
            if work <= CSCV_WORK_BUDGET {
                chosen = Some((candidate_groups, combos as usize));
                break;
            }
        }
        candidate_groups -= 2;
    }
    let (groups, combinations) = match chosen {
        Some(v) => v,
        None => {
            // The floor case: MIN_CSCV_GROUPS always fits (C(8,4)=70), so this
            // only triggers when the grid is too short for even that.
            let combos = binomial(MIN_CSCV_GROUPS, MIN_CSCV_GROUPS / 2);
            if periods / MIN_CSCV_GROUPS < MIN_PERIODS_PER_GROUP {
                return Err(format!(
                    "REFUSED: {periods} periods cannot be split into {MIN_CSCV_GROUPS} groups of \
                     at least {MIN_PERIODS_PER_GROUP}."
                ));
            }
            (MIN_CSCV_GROUPS, combos as usize)
        }
    };
    let group_choice_reason = format!(
        "largest even group count in [{MIN_CSCV_GROUPS}, {MAX_CSCV_GROUPS}] with at least \
         {MIN_PERIODS_PER_GROUP} periods per group whose sweep cost \
         (C(S,S/2) × trials × S = {} × {n_trials} × {groups}) fits the {CSCV_WORK_BUDGET}-unit \
         budget",
        combinations
    );

    let bounds = group_bounds(periods, groups);
    let smallest = bounds.iter().map(|(a, b)| b - a).min().unwrap_or(0);
    let largest = bounds.iter().map(|(a, b)| b - a).max().unwrap_or(0);

    // Pre-aggregate each trial's moments per group: the sweep then costs
    // O(combinations × trials × groups) instead of O(… × periods).
    let mut g_sum = vec![0.0f64; n_trials * groups];
    let mut g_sq = vec![0.0f64; n_trials * groups];
    let mut g_n = vec![0.0f64; groups];
    for (g, (start, end)) in bounds.iter().enumerate() {
        g_n[g] = (end - start) as f64;
    }
    for (t, row) in usable.iter().enumerate() {
        for (g, (start, end)) in bounds.iter().enumerate() {
            let mut s = 0.0;
            let mut q = 0.0;
            for v in &row.returns[*start..*end] {
                s += *v;
                q += *v * *v;
            }
            g_sum[t * groups + g] = s;
            g_sq[t * groups + g] = q;
        }
    }

    let half = groups / 2;
    let mut logits: Vec<f64> = Vec::with_capacity(combinations);
    let mut ranks: Vec<f64> = Vec::with_capacity(combinations);
    let mut below_median = 0usize;
    let mut degenerate_oos = 0usize;
    let mut oos_perf = vec![0.0f64; n_trials];

    for_each_combination(groups, half, |chosen_groups| {
        let mut in_mask = [false; MAX_CSCV_GROUPS];
        let mut is_count = 0.0;
        let mut oos_count = 0.0;
        for &g in chosen_groups {
            in_mask[g] = true;
            is_count += g_n[g];
        }
        for g in 0..groups {
            if !in_mask[g] {
                oos_count += g_n[g];
            }
        }

        let mut champion = 0usize;
        let mut champion_perf = f64::NEG_INFINITY;
        for t in 0..n_trials {
            let (mut is_s, mut is_q, mut oos_s, mut oos_q) = (0.0, 0.0, 0.0, 0.0);
            for g in 0..groups {
                let s = g_sum[t * groups + g];
                let q = g_sq[t * groups + g];
                if in_mask[g] {
                    is_s += s;
                    is_q += q;
                } else {
                    oos_s += s;
                    oos_q += q;
                }
            }
            let (ip, _) = pooled_performance(is_s, is_q, is_count);
            let (op, _) = pooled_performance(oos_s, oos_q, oos_count);
            oos_perf[t] = op;
            // Strict `>` keeps ties resolved by lowest index: deterministic.
            if ip > champion_perf {
                champion_perf = ip;
                champion = t;
            }
        }

        let cp = oos_perf[champion];
        if cp.abs() >= CONSTANT_SERIES_PERFORMANCE {
            degenerate_oos += 1;
        }
        let mut below = 0usize;
        let mut equal = 0usize;
        for (t, p) in oos_perf.iter().enumerate() {
            if t == champion {
                continue;
            }
            if *p < cp {
                below += 1;
            } else if *p == cp {
                equal += 1;
            }
        }
        // Mid-rank for ties, so a field of identical trials does not hand the
        // champion an undeserved top or bottom placement.
        let rank = 1.0 + below as f64 + 0.5 * equal as f64;
        let omega = rank / (n_trials as f64 + 1.0);
        let omega = omega.clamp(1.0e-12, 1.0 - 1.0e-12);
        let lambda = (omega / (1.0 - omega)).ln();
        if lambda <= 0.0 {
            below_median += 1;
        }
        logits.push(lambda);
        ranks.push(omega);
    });

    if logits.is_empty() {
        return Err(format!(
            "REFUSED: the CSCV sweep produced no splits at {groups} groups — nothing to average."
        ));
    }

    let pbo = below_median as f64 / logits.len() as f64;
    let mean_logit = logits.iter().sum::<f64>() / logits.len() as f64;
    let mut sorted = ranks.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_rank = sorted[sorted.len() / 2];

    let assumptions = vec![
        format!(
            "{combinations} balanced splits of {groups} CONTIGUOUS groups over {periods} calendar \
             months ({smallest}–{largest} months per group). Groups are contiguous so the \
             out-of-sample half is genuinely out of TIME, not a shuffled sample."
        ),
        format!(
            "{n_trials} trials ranked. {excluded} row(s) were excluded because their whole return \
             series is constant or non-finite — candidates that closed no trade inside the grid. \
             Keeping them would enlarge the rank denominator, push the champion's relative rank \
             up and PBO DOWN, i.e. toward the flattering answer."
        ),
        "Performance is the pooled per-month Sharpe over the selected groups. Pooling moments is \
         exact for mean and variance, which do not depend on the order the groups are \
         concatenated in."
            .to_string(),
        "The champion is the highest in-sample performer; ties resolve to the lowest trial index, \
         so the sweep is deterministic across runs."
            .to_string(),
        "PBO is the fraction of splits whose in-sample champion fell in the BOTTOM HALF \
         out-of-sample. PBO > 0.5 means the selection procedure is worse than picking at random. \
         It says nothing about whether any single strategy is profitable."
            .to_string(),
        "A champion sitting exactly ON the out-of-sample median (logit λ = 0) is counted as \
         below it. That is the pessimistic convention: it can only raise PBO, never lower it."
            .to_string(),
        format!(
            "CSCV assumes the trials are comparable draws over one period grid. All {n_trials} \
             rows here were screened on the same symbol, timeframe and month span."
        ),
    ];

    Ok(PboReport {
        pbo,
        combinations: logits.len(),
        groups,
        group_choice_reason,
        periods,
        smallest_group_periods: smallest,
        largest_group_periods: largest,
        trials_ranked: n_trials,
        trials_excluded_constant: excluded,
        degenerate_oos_evaluations: degenerate_oos,
        mean_logit,
        median_relative_rank: median_rank,
        assumptions,
    })
}

// ---------------------------------------------------------------------------
// The report that reaches the manifest, the ledger and the log
// ---------------------------------------------------------------------------

/// Both statistics, or both refusals, with everything a reader needs to check
/// them. Embedded in [`crate::trial_returns::TrialReturnsManifest`], which the
/// discovery ledger embeds in turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialStatisticsReport {
    pub schema: String,
    /// Rows actually decoded from the payload.
    pub matrix_rows: usize,
    /// Rows the writer offered — larger than `matrix_rows` under a stride.
    pub trials_offered: usize,
    pub matrix_periods: usize,
    /// True when the payload was cut mid-row (a killed run).
    pub matrix_truncated: bool,

    pub deflated_sharpe: Option<DeflatedSharpeReport>,
    /// Present exactly when `deflated_sharpe` is absent.
    pub deflated_sharpe_refusal: Option<String>,

    pub pbo: Option<PboReport>,
    /// Present exactly when `pbo` is absent.
    pub pbo_refusal: Option<String>,

    /// Anything observed about the matrix itself that a reader must know before
    /// trusting either statistic.
    pub notes: Vec<String>,
}

impl TrialStatisticsReport {
    /// The matrix could not be read at all. Both statistics refuse, with the
    /// same named cause, so "unreadable" can never be mistaken for "not run".
    pub fn unreadable(reason: String) -> Self {
        Self {
            schema: TRIAL_STATISTICS_SCHEMA.to_string(),
            matrix_rows: 0,
            trials_offered: 0,
            matrix_periods: 0,
            matrix_truncated: false,
            deflated_sharpe: None,
            deflated_sharpe_refusal: Some(format!("REFUSED: {reason}")),
            pbo: None,
            pbo_refusal: Some(format!("REFUSED: {reason}")),
            notes: vec![
                "The trial-return matrix could not be decoded, so NEITHER statistic was \
                 attempted. Any Sharpe quoted for this run is un-deflated and its overfitting \
                 probability is unknown."
                    .to_string(),
            ],
        }
    }

    /// Re-deflate this report against a larger, later-known trial count.
    ///
    /// The writer of a single search knows only how many trials IT offered. In
    /// a research loop the honest denominator is every candidate the whole
    /// session offered — a number that is not final until the session is — so
    /// the party that owns that fold installs it here, afterwards.
    ///
    /// What changes and what does not:
    /// * `deflated_sharpe` is recomputed at the new N (see
    ///   [`redeflate_sharpe`]); if the recomputation refuses, the value is
    ///   REPLACED BY THAT REFUSAL rather than left standing at the smaller N —
    ///   a stale DSR is the flattering direction.
    /// * `pbo` is untouched: CSCV ranks the rows of THIS matrix against each
    ///   other and does not contain N at all.
    /// * `trials_offered` becomes the N the report was deflated against, which
    ///   is what a reader of this field needs in order to check the statistic.
    ///   The per-search offered count is not lost — it lives on the search's own
    ///   record and in the sweep's outcome record, where it is what feeds the
    ///   fold in the first place.
    ///
    /// A report that carries no DSR (unreadable, or a named refusal) is stamped
    /// and returned unchanged, so the failure a reader is shown is the real one
    /// and not a downstream complaint about N.
    pub fn redeflated_against(&self, trials_offered: usize) -> Self {
        let mut out = self.clone();
        let was = self.trials_offered;
        out.trials_offered = trials_offered.max(self.trials_offered);
        if out.trials_offered == was {
            // N did not move, so neither does anything else. Returned without a
            // note: a note claiming a re-deflation that did not happen is the
            // same lie as a deflation that did not happen.
            return out;
        }
        if let Some(d) = self.deflated_sharpe.as_ref() {
            match redeflate_sharpe(d, out.trials_offered) {
                Ok(redeflated) => {
                    out.deflated_sharpe = Some(redeflated);
                    out.deflated_sharpe_refusal = None;
                }
                Err(refusal) => {
                    out.deflated_sharpe = None;
                    out.deflated_sharpe_refusal = Some(format!(
                        "{refusal} (raised while re-deflating from N={} to N={}; the value \
                         computed at the smaller N is NOT reported, because a Sharpe deflated \
                         against the wrong denominator is not a deflated Sharpe)",
                        d.trials_n, out.trials_offered
                    ));
                }
            }
        }
        out.notes.push(format!(
            "Re-deflated against N={} — the trial count of the whole selection procedure, not of \
             this search alone (this search offered {was}). PBO is unchanged: CSCV ranks this \
             matrix's own rows and does not contain N.",
            out.trials_offered
        ));
        out
    }

    /// Convenience for a caller that only wants to log one number each.
    pub fn dsr_value(&self) -> Option<f64> {
        self.deflated_sharpe
            .as_ref()
            .map(|d| d.deflated_sharpe_ratio)
    }
    pub fn pbo_value(&self) -> Option<f64> {
        self.pbo.as_ref().map(|p| p.pbo)
    }

    /// Multi-line human rendering. Assumptions are printed, not summarised —
    /// they are part of the output, and a value shown without them is exactly
    /// the bare number this module exists to stop producing.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("=== TRIAL STATISTICS (from the per-trial return matrix) ===\n");
        s.push_str(&format!(
            "  matrix: {} rows decoded, {} offered by the screen, {} monthly periods{}\n",
            self.matrix_rows,
            self.trials_offered,
            self.matrix_periods,
            if self.matrix_truncated {
                " [TRUNCATED — the run was cut mid-write]"
            } else {
                ""
            }
        ));
        for note in &self.notes {
            s.push_str(&format!("  note: {note}\n"));
        }

        s.push_str("\n  -- DEFLATED SHARPE RATIO --\n");
        match (&self.deflated_sharpe, &self.deflated_sharpe_refusal) {
            (Some(d), _) => {
                s.push_str(&format!(
                    "  champion            : {} (candidate #{})\n",
                    d.champion_strategy_id, d.champion_candidate_index
                ));
                s.push_str(&format!(
                    "  RAW Sharpe          : {:.4} / month   ({:.4} annualised)\n",
                    d.raw_sharpe_per_period, d.raw_sharpe_annualised
                ));
                s.push_str(&format!(
                    "  expected max by luck: {:.4} / month   ({:.4} annualised) over N={} trials\n",
                    d.expected_max_sharpe_per_period, d.expected_max_sharpe_annualised, d.trials_n
                ));
                s.push_str(&format!(
                    "  excess over luck    : {:+.4} / month\n",
                    d.excess_over_expected_max_per_period
                ));
                s.push_str(&format!(
                    "  DEFLATED SHARPE     : {:.4}   (probability the true Sharpe beats the \
                     selection-adjusted bar)\n",
                    d.deflated_sharpe_ratio
                ));
                s.push_str(&format!(
                    "  moments             : skew {:.4}, kurtosis {:.4} over {} months; \
                     cross-trial Sharpe variance {:.6e} from {} trials\n",
                    d.skewness,
                    d.kurtosis,
                    d.periods,
                    d.sharpe_variance_across_trials,
                    d.trials_with_sharpe
                ));
                for a in &d.assumptions {
                    s.push_str(&format!("    assumes: {a}\n"));
                }
            }
            (None, Some(r)) => s.push_str(&format!("  {r}\n")),
            (None, None) => s.push_str(
                "  (no value and no refusal recorded — this is a defect in the reporting path)\n",
            ),
        }

        s.push_str("\n  -- PROBABILITY OF BACKTEST OVERFITTING (CSCV) --\n");
        match (&self.pbo, &self.pbo_refusal) {
            (Some(p), _) => {
                s.push_str(&format!(
                    "  PBO                 : {:.4}   ({})\n",
                    p.pbo,
                    if p.pbo > 0.5 {
                        "ABOVE 0.5 — this selection procedure is worse than choosing at random"
                    } else {
                        "at or below 0.5"
                    }
                ));
                s.push_str(&format!(
                    "  splits              : {} balanced splits of {} contiguous groups \
                     ({}-{} months each) over {} months\n",
                    p.combinations,
                    p.groups,
                    p.smallest_group_periods,
                    p.largest_group_periods,
                    p.periods
                ));
                s.push_str(&format!(
                    "  trials ranked       : {} ({} excluded as constant/non-finite)\n",
                    p.trials_ranked, p.trials_excluded_constant
                ));
                s.push_str(&format!(
                    "  champion OOS rank   : median {:.4} of 1.0, mean logit {:+.4}\n",
                    p.median_relative_rank, p.mean_logit
                ));
                if p.degenerate_oos_evaluations > 0 {
                    s.push_str(&format!(
                        "  degenerate OOS      : {} split(s) ranked the champion on a constant \
                         out-of-sample series\n",
                        p.degenerate_oos_evaluations
                    ));
                }
                s.push_str(&format!(
                    "  group choice        : {}\n",
                    p.group_choice_reason
                ));
                for a in &p.assumptions {
                    s.push_str(&format!("    assumes: {a}\n"));
                }
            }
            (None, Some(r)) => s.push_str(&format!("  {r}\n")),
            (None, None) => s.push_str(
                "  (no value and no refusal recorded — this is a defect in the reporting path)\n",
            ),
        }
        s
    }
}

/// Decode `bytes` and compute both statistics.
///
/// NEVER fails: a decode error becomes a refusal on both statistics, because the
/// caller is the writer's own `finish` and a statistics failure must not take
/// down a discovery run that otherwise produced its artifacts.
pub fn analyse_bytes(bytes: &[u8], trials_offered: usize) -> TrialStatisticsReport {
    let matrix = match decode(bytes) {
        Ok(m) => m,
        Err(e) => return TrialStatisticsReport::unreadable(e.to_string()),
    };
    analyse_matrix(&matrix, trials_offered)
}

/// Compute both statistics from an already-decoded matrix.
pub fn analyse_matrix(matrix: &DecodedTrialMatrix, trials_offered: usize) -> TrialStatisticsReport {
    let mut notes = Vec::new();
    if matrix.is_truncated() {
        notes.push(format!(
            "The payload holds {} complete row(s) while its header declares {} and {} byte(s) \
             remain after the last complete row: the run was killed mid-write. Both statistics \
             below are computed from the rows that reached the disk.",
            matrix.rows.len(),
            matrix.header_trials,
            matrix.truncated_tail_bytes
        ));
    }
    if trials_offered > matrix.rows.len() {
        notes.push(format!(
            "The screen offered {trials_offered} trials but only {} rows were retained (disk \
             budget stride). The DSR uses the OFFERED count as N — the honest denominator — while \
             PBO can only rank the rows that are present.",
            matrix.rows.len()
        ));
    }
    if matrix.flags & 1 == 0 {
        notes.push(
            "The payload's unit flag is not set to 'fraction of initial balance'; the return \
             scale is unverified, which affects the Sharpe magnitude but not PBO (rank-based)."
                .to_string(),
        );
    }

    let (deflated_sharpe, deflated_sharpe_refusal) = match deflated_sharpe(matrix, trials_offered) {
        Ok(d) => (Some(d), None),
        Err(r) => (None, Some(r)),
    };
    let (pbo, pbo_refusal) = match pbo_cscv(matrix) {
        Ok(p) => (Some(p), None),
        Err(r) => (None, Some(r)),
    };

    TrialStatisticsReport {
        schema: TRIAL_STATISTICS_SCHEMA.to_string(),
        matrix_rows: matrix.rows.len(),
        trials_offered,
        matrix_periods: matrix.periods(),
        matrix_truncated: matrix.is_truncated(),
        deflated_sharpe,
        deflated_sharpe_refusal,
        pbo,
        pbo_refusal,
        notes,
    }
}

/// Read the exact receipt/config-bound payload and compute both statistics.
///
/// The entry point for a consumer outside the writer — a CLI subcommand, a UI
/// panel, or a re-analysis of an old run's matrix.
pub fn analyse_run(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
    expected_receipt: &CanonicalSearchInputReceiptV2,
    expected_config_hash: &str,
) -> anyhow::Result<TrialStatisticsReport> {
    let manifest = crate::trial_returns::load_manifest(
        cache_dir,
        symbol,
        tf,
        expected_receipt,
        expected_config_hash,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "no receipt-bound trial-returns manifest for {symbol}/{tf} and config {expected_config_hash}"
        )
    })?;
    let path = binary_path(cache_dir, expected_receipt, expected_config_hash)?;
    let bytes = std::fs::read(&path)
        .map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
    Ok(analyse_bytes(&bytes, manifest.trials_offered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial_returns::{TrialReturnMatrix, TrialReturnRow, encode};

    /// A deterministic, seeded return pattern. No RNG crate, no flakiness.
    fn pattern(seed: u64, j: usize) -> f64 {
        // Wrapping throughout: an overflow panic in debug would make these
        // tests fail for a reason that has nothing to do with the statistics.
        let x = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add((j as u64).wrapping_mul(1_442_695_040_888_963_407));
        let u = ((x >> 33) as f64) / ((1u64 << 31) as f64); // [0, 1)
        (u - 0.5) * 0.04
    }

    fn build(
        rows: usize,
        periods: usize,
        mean_of_row: impl Fn(usize) -> f64,
    ) -> DecodedTrialMatrix {
        let matrix = TrialReturnMatrix {
            period_keys: (0..periods as i64).map(|i| 24_000 + i).collect(),
            initial_balance: 10_000.0,
            rows: (0..rows)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("gene_{i}"),
                    returns: (0..periods)
                        .map(|j| mean_of_row(i) + pattern(i as u64 + 1, j))
                        .collect(),
                    trades_outside_grid: 0,
                })
                .collect(),
        };
        let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        decode(&encode(&matrix, &refs)).expect("decode round-trip")
    }

    #[test]
    fn the_reader_round_trips_what_the_writer_encodes() {
        let m = build(5, 30, |_| 0.0);
        assert_eq!(m.version, 1);
        assert_eq!(m.flags & 1, 1, "unit flag: fraction of initial balance");
        assert_eq!(m.initial_balance, 10_000.0);
        assert_eq!(m.header_trials, 5);
        assert_eq!(m.rows.len(), 5);
        assert_eq!(m.periods(), 30);
        assert_eq!(m.rows[3].strategy_id, "gene_3");
        assert_eq!(m.rows[3].candidate_index, 3);
        assert_eq!(m.rows[3].returns.len(), 30);
        assert!(!m.is_truncated());
    }

    #[test]
    fn a_matrix_cut_mid_row_decodes_to_the_rows_that_reached_the_disk() {
        let matrix = TrialReturnMatrix {
            period_keys: vec![1, 2, 3, 4],
            initial_balance: 1.0,
            rows: (0..4)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: "abcdef".to_string(),
                    returns: vec![0.1, 0.2, 0.3, 0.4],
                    trades_outside_grid: 0,
                })
                .collect(),
        };
        let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        let mut bytes = encode(&matrix, &refs);
        bytes.truncate(bytes.len() - 9); // cut the last row mid-way
        let m = decode(&bytes).expect("a partial matrix is still readable");
        assert_eq!(m.rows.len(), 3, "three complete rows survived");
        assert_eq!(m.header_trials, 4, "the header still claims four");
        assert!(m.is_truncated());
        assert!(m.truncated_tail_bytes > 0);
    }

    #[test]
    fn the_reader_refuses_a_file_that_is_not_a_trial_matrix() {
        assert!(matches!(
            decode(&[0u8; 8]),
            Err(DecodeError::TooShortForHeader { .. })
        ));
        let mut junk = vec![0u8; 64];
        junk[..8].copy_from_slice(b"NOTATRIA");
        assert!(matches!(decode(&junk), Err(DecodeError::BadMagic { .. })));
        let mut wrong = vec![0u8; 64];
        wrong[..8].copy_from_slice(TRIAL_RETURNS_MAGIC);
        wrong[8..12].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            decode(&wrong),
            Err(DecodeError::UnsupportedVersion { version: 7 })
        ));
    }

    #[test]
    fn four_periods_is_a_refusal_not_a_confident_number() {
        let m = build(40, 4, |i| 0.001 * i as f64);
        let report = analyse_matrix(&m, 40);
        assert!(report.deflated_sharpe.is_none());
        assert!(report.pbo.is_none());
        let d = report.deflated_sharpe_refusal.unwrap();
        assert!(d.starts_with("REFUSED:"), "{d}");
        assert!(d.contains("24"), "the refusal names the threshold: {d}");
        let p = report.pbo_refusal.unwrap();
        assert!(p.contains("16"), "the refusal names the threshold: {p}");
    }

    #[test]
    fn too_few_trials_is_a_refusal_that_names_the_variance_it_cannot_estimate() {
        let m = build(3, 60, |i| 0.001 * i as f64);
        let r = analyse_matrix(&m, 3);
        let d = r.deflated_sharpe_refusal.unwrap();
        assert!(d.contains("VARIANCE"), "{d}");
        let p = r.pbo_refusal.unwrap();
        assert!(p.contains("rank"), "{p}");
    }

    #[test]
    fn a_dominant_trial_is_never_flagged_as_overfitting() {
        // Trial 0 beats every other trial in every single month, and all trials
        // share the same dispersion. It must win in-sample in every split and
        // top the out-of-sample ranking in every split, so PBO is exactly 0.
        let periods = 36;
        let rows = 20;
        let matrix = TrialReturnMatrix {
            period_keys: (0..periods as i64).map(|i| 24_000 + i).collect(),
            initial_balance: 10_000.0,
            rows: (0..rows)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("gene_{i}"),
                    returns: (0..periods)
                        .map(|j| {
                            let base = pattern(7, j);
                            if i == 0 {
                                base + 0.05
                            } else {
                                base + 0.0001 * i as f64
                            }
                        })
                        .collect(),
                    trades_outside_grid: 0,
                })
                .collect(),
        };
        let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        let m = decode(&encode(&matrix, &refs)).unwrap();
        let p = pbo_cscv(&m).expect("computable");
        assert_eq!(p.pbo, 0.0, "a genuinely dominant trial cannot be overfit");
        assert!(p.mean_logit > 0.0);
        assert_eq!(p.trials_ranked, rows);
        assert!(p.combinations >= 70, "all balanced splits are evaluated");
        assert!(
            !p.assumptions.is_empty(),
            "the value states its assumptions"
        );
    }

    #[test]
    fn pure_noise_produces_a_computable_pbo_in_range() {
        // Every trial is independent noise with the same zero mean: whichever
        // one wins in-sample won it by luck, so its out-of-sample rank is
        // arbitrary. The point is not the exact value but that the statistic is
        // finite, bounded, and derived from every split.
        let m = build(30, 48, |_| 0.0);
        let p = pbo_cscv(&m).expect("computable");
        assert!((0.0..=1.0).contains(&p.pbo), "pbo = {}", p.pbo);
        assert!(p.mean_logit.is_finite());
        assert!((0.0..=1.0).contains(&p.median_relative_rank));
        assert_eq!(p.trials_excluded_constant, 0);
        assert_eq!(
            p.combinations,
            binomial(p.groups, p.groups / 2) as usize,
            "every balanced split is used, none sampled"
        );
    }

    #[test]
    fn constant_trials_are_excluded_and_counted_not_silently_dropped() {
        let periods = 40;
        let rows = 24;
        let matrix = TrialReturnMatrix {
            period_keys: (0..periods as i64).map(|i| 24_000 + i).collect(),
            initial_balance: 10_000.0,
            rows: (0..rows)
                .map(|i| TrialReturnRow {
                    candidate_index: i,
                    strategy_id: format!("gene_{i}"),
                    // Every third trial never closed a trade inside the grid.
                    returns: (0..periods)
                        .map(|j| {
                            if i % 3 == 0 {
                                0.0
                            } else {
                                pattern(i as u64 + 1, j)
                            }
                        })
                        .collect(),
                    trades_outside_grid: 0,
                })
                .collect(),
        };
        let refs: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
        let m = decode(&encode(&matrix, &refs)).unwrap();
        let p = pbo_cscv(&m).expect("computable");
        assert_eq!(
            p.trials_excluded_constant, 8,
            "24 rows, every third is flat"
        );
        assert_eq!(p.trials_ranked, 16);
        assert!(
            p.assumptions.iter().any(|a| a.contains("PBO DOWN")),
            "the exclusion states which way it biases the answer"
        );
    }

    #[test]
    fn the_deflated_value_is_reported_beside_the_raw_one() {
        let m = build(200, 60, |i| 0.002 + 0.00002 * i as f64);
        let d = deflated_sharpe(&m, 200).expect("computable");
        assert_eq!(d.trials_n, 200);
        assert!(d.trials_n_source.contains("rows in the matrix"));
        assert!((0.0..=1.0).contains(&d.deflated_sharpe_ratio));
        assert!(d.raw_sharpe_per_period.is_finite());
        assert!(
            (d.raw_sharpe_annualised - d.raw_sharpe_per_period * PERIODS_PER_YEAR.sqrt()).abs()
                < 1e-9
        );
        assert!(
            d.expected_max_sharpe_per_period > 0.0,
            "N trials raise the bar"
        );
        assert!(
            (d.excess_over_expected_max_per_period
                - (d.raw_sharpe_per_period - d.expected_max_sharpe_per_period))
                .abs()
                < 1e-12
        );
        assert!(d.sharpe_variance_across_trials > 0.0);
        assert!(d.assumptions.len() >= 5);
    }

    #[test]
    fn the_honest_n_is_the_offered_count_when_the_file_was_sub_sampled() {
        let m = build(60, 60, |i| 0.002 + 0.00002 * i as f64);
        let small = deflated_sharpe(&m, 60).unwrap();
        let honest = deflated_sharpe(&m, 15_360).unwrap();
        assert_eq!(honest.trials_n, 15_360);
        assert!(honest.trials_n_source.contains("under-deflate"));
        assert!(
            honest.expected_max_sharpe_per_period > small.expected_max_sharpe_per_period,
            "more trials must raise the bar, never lower it"
        );
        assert!(
            honest.deflated_sharpe_ratio <= small.deflated_sharpe_ratio,
            "the honest N can only deflate further"
        );
    }

    #[test]
    fn the_normal_helpers_agree_with_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.959_963_985) - 0.975).abs() < 1e-6);
        assert!((normal_cdf(-1.959_963_985) - 0.025).abs() < 1e-6);
        assert!((normal_ppf(0.975).unwrap() - 1.959_963_985).abs() < 1e-6);
        assert!((normal_ppf(0.5).unwrap()).abs() < 1e-9);
        assert!((normal_ppf(0.01).unwrap() + 2.326_347_874).abs() < 1e-6);
        assert!(normal_ppf(0.0).is_none(), "no infinity smuggled in");
        assert!(normal_ppf(1.0).is_none());
        assert!(normal_ppf(f64::NAN).is_none());
    }

    #[test]
    fn every_balanced_split_is_enumerated_exactly_once() {
        let mut seen: Vec<Vec<usize>> = Vec::new();
        for_each_combination(8, 4, |c| seen.push(c.to_vec()));
        assert_eq!(seen.len(), 70);
        assert_eq!(seen.len() as u128, binomial(8, 4));
        let mut sorted = seen.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 70, "no duplicates");
        assert_eq!(seen[0], vec![0, 1, 2, 3]);
        assert_eq!(seen[69], vec![4, 5, 6, 7]);
    }

    #[test]
    fn groups_are_contiguous_and_cover_every_period() {
        let b = group_bounds(50, 8);
        assert_eq!(b.len(), 8);
        assert_eq!(b[0].0, 0);
        assert_eq!(b[7].1, 50);
        for w in b.windows(2) {
            assert_eq!(w[0].1, w[1].0, "contiguous, no gap and no overlap");
        }
        let sizes: Vec<usize> = b.iter().map(|(a, c)| c - a).collect();
        assert_eq!(sizes.iter().sum::<usize>(), 50);
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    }

    #[test]
    fn an_unreadable_matrix_refuses_both_statistics_rather_than_looking_unrun() {
        let r = analyse_bytes(&[0u8; 3], 0);
        assert!(r.deflated_sharpe.is_none() && r.pbo.is_none());
        assert!(
            r.deflated_sharpe_refusal
                .as_ref()
                .unwrap()
                .contains("REFUSED")
        );
        assert!(r.pbo_refusal.as_ref().unwrap().contains("REFUSED"));
        assert!(r.render().contains("un-deflated"));
    }

    #[test]
    fn the_rendering_prints_the_assumptions_not_just_the_numbers() {
        let m = build(200, 60, |i| 0.002 + 0.00002 * i as f64);
        let r = analyse_matrix(&m, 200);
        let text = r.render();
        assert!(text.contains("DEFLATED SHARPE RATIO"));
        assert!(text.contains("PROBABILITY OF BACKTEST OVERFITTING"));
        assert!(text.contains("RAW Sharpe"), "raw sits beside deflated");
        assert!(
            text.contains("assumes:"),
            "assumptions are part of the output"
        );
        assert!(text.contains("honest denominator"));
    }

    #[test]
    fn the_report_survives_a_json_round_trip_into_the_manifest() {
        let m = build(200, 60, |i| 0.002 + 0.00002 * i as f64);
        let r = analyse_matrix(&m, 200);
        let json = serde_json::to_string(&r).unwrap();
        let back: TrialStatisticsReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r, "the ledger must carry the exact values");
    }

    #[test]
    fn redeflating_a_report_equals_re_analysing_the_matrix_at_the_larger_n() {
        // The whole justification for the analytic shortcut. A research loop
        // cannot know its honest N until the session is over, so it installs
        // that N afterwards — by arithmetic on the report, without the matrix.
        // That is only legitimate if the answer is IDENTICAL to what re-reading
        // the matrix would have produced. Asserted bit for bit, not approximated.
        let m = build(200, 60, |i| 0.002 + 0.00002 * i as f64);
        let at_own_count = analyse_matrix(&m, 200);
        let at_session = analyse_matrix(&m, 25_000);

        let redeflated = redeflate_sharpe(
            at_own_count
                .deflated_sharpe
                .as_ref()
                .expect("a DSR at N=200"),
            25_000,
        )
        .expect("re-deflation at a larger N");
        let truth = at_session
            .deflated_sharpe
            .as_ref()
            .expect("a DSR at N=25000");

        assert_eq!(redeflated.trials_n, truth.trials_n);
        assert_eq!(
            redeflated.expected_max_sharpe_per_period, truth.expected_max_sharpe_per_period,
            "SR* is the ONLY place N enters; it must land on the same bits"
        );
        assert_eq!(
            redeflated.excess_over_expected_max_per_period,
            truth.excess_over_expected_max_per_period
        );
        assert_eq!(
            redeflated.deflated_sharpe_ratio, truth.deflated_sharpe_ratio,
            "the deflated Sharpe itself must be the same number"
        );
        // And the direction is never flattering: more trials searched means a
        // higher bar to have cleared by luck.
        let before = at_own_count.deflated_sharpe.as_ref().unwrap();
        assert!(
            truth.expected_max_sharpe_per_period > before.expected_max_sharpe_per_period,
            "SR* must RISE with N"
        );
        assert!(
            truth.deflated_sharpe_ratio <= before.deflated_sharpe_ratio,
            "the DSR must never improve because the search turned out to be larger"
        );
    }

    #[test]
    fn a_redeflated_report_carries_the_session_n_and_leaves_pbo_alone() {
        // `trials_offered` is what a reader checks the deflation against, so it
        // must BE the N used. PBO ranks this matrix's own rows and contains no
        // N at all — re-deflating must not perturb it.
        let m = build(200, 60, |i| 0.002 + 0.00002 * i as f64);
        let own = analyse_matrix(&m, 200);
        let session = own.redeflated_against(25_000);

        assert_eq!(session.trials_offered, 25_000);
        assert_eq!(session.pbo, own.pbo, "PBO does not depend on N");
        assert_eq!(session.matrix_rows, own.matrix_rows);
        assert!(
            session.notes.iter().any(|n| n.contains("25000")),
            "the report must say out loud which N it was deflated against"
        );

        // A floor, never a ceiling: a SMALLER offered count cannot un-deflate a
        // report. There is deliberately no path here that improves a number.
        let smaller = own.redeflated_against(10);
        assert_eq!(smaller.trials_offered, own.trials_offered);
        assert_eq!(smaller.deflated_sharpe, own.deflated_sharpe);
    }

    #[test]
    fn an_unreadable_report_is_stamped_with_the_session_n_and_stays_unreadable() {
        // The judge checks `trials_offered == n_session` before it looks at the
        // statistics. An unreadable report that kept `trials_offered = 0` would
        // be reported as an N-contract violation — a disk finding filed under
        // the wrong cause. It is stamped, and it stays refused.
        let unreadable = TrialStatisticsReport::unreadable("no such file".to_string());
        let stamped = unreadable.redeflated_against(25_000);
        assert_eq!(stamped.trials_offered, 25_000);
        assert!(stamped.deflated_sharpe.is_none());
        assert!(stamped.pbo.is_none());
        assert_eq!(stamped.matrix_rows, 0);
        assert!(
            stamped
                .deflated_sharpe_refusal
                .as_deref()
                .unwrap()
                .contains("no such file"),
            "the named cause survives"
        );
    }
}
