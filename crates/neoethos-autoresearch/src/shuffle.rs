//! **The shuffle control — the falsification, built in rather than bolted on.**
//!
//! `docs/autoresearch-loop.md` §9.
//!
//! Every [`crate::SHUFFLE_PERIOD`] live sweeps form a **block**. At each block
//! boundary the loop re-runs the block's best-screening sweep's exact 100
//! proposals, byte for byte, against **time-rotated** features. If the loop's
//! winners score like the shuffle's winners, the loop is mining noise — and
//! [`ShuffleSummary`] is rendered immediately under every headline number so it
//! says so in the same breath.
//!
//! ## Why a circular rotation and not an i.i.d. row shuffle
//!
//! This was a real choice, and it is the difference between a control that means
//! something and one that is trivially beaten:
//!
//! | | marginals | cross-feature structure | each feature's autocorrelation | feature → future alignment |
//! |---|---|---|---|---|
//! | i.i.d. row permutation | preserved | destroyed per-column | **destroyed** | destroyed |
//! | **circular rotation** | preserved | preserved | **preserved** | destroyed |
//!
//! An i.i.d. permutation destroys autocorrelation, so signals flicker bar to
//! bar, trade counts explode, and the control pays a completely different cost
//! per trade than the live sweep. The comparison then measures trade FREQUENCY
//! rather than predictability, and the live sweep wins for a reason that has
//! nothing to do with edge. Rotation is the conservative control: it leaves
//! everything intact — prices, labels, costs, exit geometry, gene encoding, the
//! GA seed — except the one thing under test.
//!
//! The independent per-column permutation is kept as a **weaker diagnostic**
//! ([`ControlKind::ColumnPermutation`]): reported, never entering an inequality.
//!
//! ## Cost
//!
//! One control sweep per four live ones = **20% of the budget**. Stated plainly:
//! the alternative is a session whose entire output is unfalsifiable, and that is
//! worth 0% of the budget.

use neoethos_core::utils::hashing::fnv1a64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::session::{BlockId, BlockOutcome, Session, SweepId};

/// The rotation offset is drawn uniformly from
/// `[TAU_LO_FRAC·T, TAU_HI_FRAC·T]`.
///
/// The ends are excluded because a rotation of ~0 (or ~T) leaves the alignment
/// almost intact, which would produce a "control" that is really a second live
/// run and would make the null look strong for the wrong reason.
pub const TAU_LO_FRAC: f64 = 0.05;
/// See [`TAU_LO_FRAC`].
pub const TAU_HI_FRAC: f64 = 0.95;

/// The quantile of the accumulated null that screen conjunct S6 compares
/// against.
pub const NULL_QUANTILE: f64 = 0.95;

// ─────────────────────────────────────────────────────────────────────────────
// The two controls
// ─────────────────────────────────────────────────────────────────────────────

/// Which control produced an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlKind {
    /// **The one in S6 and in R2.** A joint circular rotation of the entire
    /// feature block by a single offset.
    CircularRotation,
    /// A weaker, once-per-session diagnostic: an independent permutation per
    /// column. Reported, never used in an inequality — it destroys
    /// autocorrelation and cross-feature structure together, so beating it
    /// proves less than beating a rotation.
    ColumnPermutation,
}

impl ControlKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::CircularRotation => "circular_rotation",
            Self::ColumnPermutation => "column_permutation",
        }
    }

    /// Only the rotation control feeds the session null, and therefore S6.
    pub fn feeds_the_null(self) -> bool {
        matches!(self, Self::CircularRotation)
    }
}

/// How a feature block is laid out in memory, so the permutation can move
/// **rows** (bars) whichever way the caller stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLayout {
    /// `src[r * cols + c]`
    RowMajor,
    /// `src[c * rows + r]` — the layout the feature cube uses.
    ColumnMajor,
}

/// The permutation a control sweep applies to the feature block, and nothing
/// else.
///
/// It is drawn from `(session_seed, block, kind)` alone, so a control sweep is
/// replayable exactly like a live one — **a control nobody can re-run is not
/// evidence**. `tau` is carried as a FRACTION of the series length rather than a
/// row count, because the permutation is decided at the block boundary while the
/// row count belongs to whatever block of features the executor loads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeaturePermutation {
    kind: ControlKind,
    tau_fraction: f64,
    seed: u64,
}

impl FeaturePermutation {
    /// Draw this block's permutation, deterministically.
    pub fn draw(kind: ControlKind, session: &Session, block: BlockId) -> Self {
        let session_seed = session.opened.as_ref().map(|h| h.session_seed).unwrap_or(0);
        let mut key = Vec::with_capacity(32);
        key.extend_from_slice(&session_seed.to_le_bytes());
        key.extend_from_slice(&u64::from(block.0).to_le_bytes());
        key.extend_from_slice(kind.label().as_bytes());
        let seed = fnv1a64(&key);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Self {
            kind,
            tau_fraction: rng.random_range(TAU_LO_FRAC..=TAU_HI_FRAC),
            seed,
        }
    }

    pub fn kind(&self) -> ControlKind {
        self.kind
    }

    /// The rotation offset as a fraction of the series length, in
    /// `[TAU_LO_FRAC, TAU_HI_FRAC]`. Recorded in the journal.
    pub fn tau(&self) -> f64 {
        self.tau_fraction
    }

    /// The offset in rows for a concrete feature block.
    ///
    /// Refuses rather than clamping when the block is too short to hold a legal
    /// offset: **a rotation of zero is not a control** — it is a second live run
    /// wearing the control's label, and it would make the null look strong for
    /// the wrong reason.
    pub fn offset_rows(&self, rows: usize) -> Result<usize, String> {
        let lo = (TAU_LO_FRAC * rows as f64).ceil() as usize;
        let hi = (TAU_HI_FRAC * rows as f64).floor() as usize;
        if rows == 0 || lo == 0 || hi < lo {
            return Err(format!(
                "REFUSED: a feature block of {rows} row(s) admits no rotation offset in \
                 [{TAU_LO_FRAC}*T, {TAU_HI_FRAC}*T] = [{lo}, {hi}]. A rotation of zero is not a \
                 control."
            ));
        }
        let tau = (self.tau_fraction * rows as f64).round() as usize;
        Ok(tau.clamp(lo, hi))
    }

    /// Apply the permutation to a feature block.
    ///
    /// The caller must NOT permute prices, labels, costs or exit geometry —
    /// only the feature block. That asymmetry **is** the experiment: everything
    /// the candidate is charged and measured on stays where it was, and only the
    /// features' alignment with the future is destroyed.
    pub fn apply<T: Copy>(
        &self,
        src: &[T],
        rows: usize,
        cols: usize,
        layout: BlockLayout,
    ) -> Result<Vec<T>, String> {
        match self.kind {
            ControlKind::CircularRotation => {
                rotate_block(src, rows, cols, self.offset_rows(rows)?, layout)
            }
            ControlKind::ColumnPermutation => {
                permute_columns_independently(src, rows, cols, layout, self.seed)
            }
        }
    }
}

/// Rotate a feature block by `tau` rows, wrapping.
///
/// **Joint**: every column moves by the same offset, which is what preserves
/// cross-feature structure. Output row `i` reads source row `(i + tau) % rows`.
pub fn rotate_block<T: Copy>(
    src: &[T],
    rows: usize,
    cols: usize,
    tau: usize,
    layout: BlockLayout,
) -> Result<Vec<T>, String> {
    let expected = check_shape(src.len(), rows, cols)?;
    if tau == 0 || tau % rows == 0 {
        return Err(format!(
            "REFUSED: tau = {tau} is a no-op rotation for {rows} row(s), which would make the \
             'control' a second live run."
        ));
    }
    let tau = tau % rows;
    let mut out = Vec::with_capacity(expected);
    match layout {
        BlockLayout::RowMajor => {
            for r in 0..rows {
                let s = (r + tau) % rows;
                out.extend_from_slice(&src[s * cols..s * cols + cols]);
            }
        }
        BlockLayout::ColumnMajor => {
            for c in 0..cols {
                let base = c * rows;
                for r in 0..rows {
                    out.push(src[base + (r + tau) % rows]);
                }
            }
        }
    }
    Ok(out)
}

/// The weaker diagnostic: permute each column independently.
///
/// Destroys autocorrelation as well as alignment, so a live sweep beats it for
/// reasons that include trade frequency. Reported, never gated on.
pub fn permute_columns_independently<T: Copy>(
    src: &[T],
    rows: usize,
    cols: usize,
    layout: BlockLayout,
    seed: u64,
) -> Result<Vec<T>, String> {
    let expected = check_shape(src.len(), rows, cols)?;
    let mut out: Vec<T> = vec![src[0]; expected];
    let mut idx: Vec<usize> = (0..rows).collect();
    for c in 0..cols {
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&seed.to_le_bytes());
        key.extend_from_slice(&(c as u64).to_le_bytes());
        let mut rng = ChaCha8Rng::seed_from_u64(fnv1a64(&key));
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = i;
        }
        // Fisher-Yates, explicit rather than via a shuffle extension trait, so
        // the RNG stream consumed is fixed and the permutation is replayable.
        for i in (1..rows).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }
        match layout {
            BlockLayout::RowMajor => {
                for r in 0..rows {
                    out[r * cols + c] = src[idx[r] * cols + c];
                }
            }
            BlockLayout::ColumnMajor => {
                let base = c * rows;
                for r in 0..rows {
                    out[base + r] = src[base + idx[r]];
                }
            }
        }
    }
    Ok(out)
}

fn check_shape(len: usize, rows: usize, cols: usize) -> Result<usize, String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        format!(
            "REFUSED: rows {rows} * cols {cols} overflows a usize; the block is not addressable"
        )
    })?;
    if rows == 0 || cols == 0 {
        return Err(
            "REFUSED: a feature block with zero rows or zero columns cannot be permuted."
                .to_string(),
        );
    }
    if len != expected {
        return Err(format!(
            "REFUSED: the block holds {len} value(s) but {rows} row(s) x {cols} column(s) needs \
             {expected}. Permuting a block whose shape is not what the caller believes would \
             scramble columns into each other, and the control would then measure corruption \
             rather than the absence of signal."
        ));
    }
    Ok(expected)
}

// ─────────────────────────────────────────────────────────────────────────────
// Block bookkeeping
// ─────────────────────────────────────────────────────────────────────────────

/// The best `E_screen_pess` any live sweep in `block` recorded, and which sweep.
///
/// **Best, not average.** The loop reports maxima, so a null built from average
/// runs would be too easy to beat and every block would look distinguishable.
///
/// Ranked on the sweeps' own recorded per-search expectancies rather than on
/// PASSING screens: during the first `MIN_NULL_OBS` blocks no screen can pass at
/// all (S6 is unavailable by construction), so a passing-screen ranking would
/// return `None` for exactly the blocks whose controls create the null in the
/// first place — and the session would die at block 1 with no control ever run.
pub fn block_best(session: &Session, block: BlockId) -> Option<(SweepId, f64)> {
    session
        .live_sweeps_in_block(block)
        .into_iter()
        .filter_map(|s| {
            s.per_search
                .iter()
                .filter_map(|p| p.e_screen_pess)
                .filter(|e| e.is_finite())
                .reduce(f64::max)
                .map(|e| (s.sweep, e))
        })
        .reduce(|a, b| if b.1 > a.1 { b } else { a })
}

/// The sweep whose exact proposals the block's control re-runs.
pub fn best_sweep_of_block(session: &Session, block: BlockId) -> Option<SweepId> {
    block_best(session, block).map(|(sweep, _)| sweep)
}

/// Compare the block's best live sweep with its control.
///
/// A block is **INDISTINGUISHABLE** iff `delta_expectancy ≤ 0` **or**
/// `p_block = 1` **or** `dsr_gap ≤ 0`. The first two are the same condition
/// written twice (`delta ≤ 0` ⟺ `control ≥ best_live`); both are kept because
/// both numbers are printed and a reader must be able to check the verdict
/// against either.
///
/// **A missing number on either side means INDISTINGUISHABLE.** No number is
/// never good news: a control that produced no statistic has not demonstrated
/// that the live sweep beat it, and treating an absent control as a pass would
/// let a broken control sweep manufacture confidence.
///
/// When a side is missing, `delta_expectancy` and `dsr_gap` are recorded as
/// `0.0` — a finite sentinel meaning "not shown to differ", never `NaN`, because
/// these three numbers go into the journal as JSON and a `NaN` there would
/// serialise to `null` and fail to read back.
pub fn judge_block(
    session: &Session,
    block: BlockId,
    control_best: Option<f64>,
    control_dsr: Option<f64>,
) -> BlockOutcome {
    // BOTH sides of both comparisons come from the SAME population: the source
    // sweep, whose exact proposals the control re-ran byte for byte.
    //
    // `live_best` already did — the block's maximum expectancy IS the source
    // sweep's maximum, since the source sweep is the one that attained it — but
    // `live_dsr` maxed over EVERY sweep in the block: up to 400 live searches
    // against the control's 100, journalled as if like-for-like. A maximum over
    // four times the draws is larger for that reason alone, so `dsr_gap` read
    // positive on the strength of the sample size and the block passed a
    // falsification it had not survived.
    let best = block_best(session, block);
    let live_best = best.map(|(_, e)| e);
    let live_dsr = best.and_then(|(source, _)| {
        session
            .live_sweeps_in_block(block)
            .into_iter()
            .filter(|s| s.sweep == source)
            .filter_map(|s| {
                s.per_search
                    .iter()
                    .filter_map(|p| p.dsr)
                    .filter(|d| d.is_finite())
                    .reduce(f64::max)
            })
            .reduce(f64::max)
    });

    let control_best = control_best.filter(|e| e.is_finite());
    let control_dsr = control_dsr.filter(|d| d.is_finite());

    let delta = match (live_best, control_best) {
        (Some(l), Some(c)) => l - c,
        _ => 0.0,
    };
    let p_block = match (live_best, control_best) {
        (Some(l), Some(c)) => f64::from(u8::from(c >= l)),
        // Unknown, and an unknown control cannot count as beaten.
        _ => 1.0,
    };
    let dsr_gap = match (live_dsr, control_dsr) {
        (Some(l), Some(c)) => l - c,
        _ => 0.0,
    };

    BlockOutcome {
        block,
        delta_expectancy: delta,
        p_block,
        dsr_gap,
        indistinguishable: delta <= 0.0 || p_block >= 1.0 || dsr_gap <= 0.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The session null
// ─────────────────────────────────────────────────────────────────────────────

/// The accumulated multiset `{ control_E_screen_pess(b) : b ∈ blocks so far }`,
/// and the quantile screen conjunct S6 compares against.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShuffleNull {
    observations: Vec<f64>,
    /// Rotation controls that produced no number — counted, never dropped.
    refused: usize,
}

impl ShuffleNull {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the null from the session.
    ///
    /// A fold, like everything else that must survive a crash: the null is never
    /// carried as mutable state a resume could get wrong.
    pub fn from_session(session: &Session) -> Self {
        Self {
            observations: session
                .null_observations
                .iter()
                .copied()
                .filter(|e| e.is_finite())
                .collect(),
            // BOTH kinds of refusal. A control that produced a non-finite
            // number is refused here; a control that produced NO number at all
            // never reaches `null_observations` — the fold counts it in
            // `Session::null_refusals` instead — and forgetting that second
            // kind is what pinned this counter at zero.
            refused: session.null_refusals
                + session
                    .null_observations
                    .iter()
                    .filter(|e| !e.is_finite())
                    .count(),
        }
    }

    /// Record one control observation. Only the rotation control feeds the null.
    pub fn observe(&mut self, kind: ControlKind, e_screen_pess: Option<f64>) {
        if !kind.feeds_the_null() {
            return;
        }
        match e_screen_pess.filter(|e| e.is_finite()) {
            Some(e) => self.observations.push(e),
            None => self.refused += 1,
        }
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Controls that produced no number.
    pub fn refused(&self) -> usize {
        self.refused
    }

    /// Is S6 available? False until [`crate::MIN_NULL_OBS`] observations exist.
    pub fn is_available(&self) -> bool {
        self.observations.len() >= crate::MIN_NULL_OBS
    }

    /// `Q_shuffle_0.95(session)`, or `None` while the null is unavailable.
    ///
    /// Nearest-rank on the sorted observations, matching
    /// `goal_report::percentile_sorted`. At the minimum of three observations
    /// this returns the MAXIMUM of the three — the conservative reading, and the
    /// intended one: with three points the honest 95th percentile is *"no
    /// observation was higher than this"*.
    pub fn quantile_95(&self) -> Option<f64> {
        if !self.is_available() {
            return None;
        }
        let mut sorted = self.observations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 - 1.0) * NULL_QUANTILE).round() as usize;
        sorted.get(idx.min(sorted.len() - 1)).copied()
    }

    /// The best control observation ever seen, for the report.
    pub fn best_observation(&self) -> Option<f64> {
        self.observations.iter().copied().reduce(f64::max)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The summary that rides under every headline
// ─────────────────────────────────────────────────────────────────────────────

/// §9.4. Rendered immediately under the headline of **every** report the loop
/// emits — success, refutation, or inconclusive.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShuffleSummary {
    pub blocks_judged: usize,
    pub indistinguishable: usize,
    /// Mean of `p_block` over judged blocks. Resolution `1/blocks_judged`.
    pub session_p: Option<f64>,
    pub best_live_e_screen_pess: Option<f64>,
    pub best_shuffle_e_screen_pess: Option<f64>,
    pub q_shuffle_95: Option<f64>,
    pub null_observations: usize,
    pub null_refusals: usize,
    pub null_available: bool,
}

impl ShuffleSummary {
    /// Build the summary from the session.
    pub fn from_session(session: &Session) -> Self {
        let null = ShuffleNull::from_session(session);
        let blocks_judged = session.blocks.len();
        let indistinguishable = session
            .blocks
            .iter()
            .filter(|b| b.indistinguishable)
            .count();
        let session_p = (blocks_judged > 0)
            .then(|| session.blocks.iter().map(|b| b.p_block).sum::<f64>() / blocks_judged as f64);
        Self {
            blocks_judged,
            indistinguishable,
            session_p,
            // The headline number a reader compares against must be the SAME one
            // the report quotes elsewhere, so it is taken from `best_ever`.
            best_live_e_screen_pess: session.best_ever.as_ref().map(|b| b.e_screen_pess),
            best_shuffle_e_screen_pess: null.best_observation(),
            q_shuffle_95: null.quantile_95(),
            null_observations: null.len(),
            null_refusals: null.refused(),
            null_available: null.is_available(),
        }
    }

    /// The fraction R2/U3 reads.
    pub fn indistinguishable_fraction(&self) -> Option<f64> {
        (self.blocks_judged > 0).then(|| self.indistinguishable as f64 / self.blocks_judged as f64)
    }

    /// The block that sits directly under every headline number.
    ///
    /// *"If the loop's winners score like the shuffle's winners, the loop is
    /// mining noise and must say so in the same breath as its best result"* —
    /// the closing sentence below is that instruction, rendered mechanically so
    /// it cannot be omitted by whoever quotes the headline.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "SHUFFLE CONTROL (joint circular rotation — the falsification)"
        );
        let _ = writeln!(
            s,
            "  blocks: {}   indistinguishable: {} ({})   session p: {}",
            self.blocks_judged,
            self.indistinguishable,
            match self.indistinguishable_fraction() {
                Some(f) => format!("{:.0}%", f * 100.0),
                None => "n/a".to_string(),
            },
            match self.session_p {
                Some(p) => format!("{p:.2} (resolution 1/{})", self.blocks_judged.max(1)),
                None => "n/a".to_string(),
            }
        );
        let _ = writeln!(
            s,
            "  best live    E_screen_pess: {}",
            num(self.best_live_e_screen_pess)
        );
        let _ = writeln!(
            s,
            "  best shuffle E_screen_pess: {}     Q_0.95(null): {}",
            num(self.best_shuffle_e_screen_pess),
            num(self.q_shuffle_95)
        );
        let _ = writeln!(
            s,
            "  null: {} observation(s), {} refusal(s) — S6 {}",
            self.null_observations,
            self.null_refusals,
            if self.null_available {
                "AVAILABLE".to_string()
            } else {
                format!(
                    "UNAVAILABLE (needs {}); every screen returns Unavailable",
                    crate::MIN_NULL_OBS
                )
            }
        );
        match (self.best_live_e_screen_pess, self.q_shuffle_95) {
            (Some(live), Some(q)) if live <= q => {
                let _ = writeln!(
                    s,
                    "  READ THIS WITH THE HEADLINE: the loop's best result does NOT beat its own \
                     shuffle control ({live:+.4} <= {q:+.4}). On this evidence the loop is mining \
                     noise."
                );
            }
            (Some(live), Some(q)) => {
                let _ = writeln!(
                    s,
                    "  the loop's best result beats its own shuffle control by {:+.4} pips.",
                    live - q
                );
            }
            _ => {
                let _ = writeln!(
                    s,
                    "  READ THIS WITH THE HEADLINE: there is no usable shuffle null yet, so \
                     NOTHING above has been falsified."
                );
            }
        }
        s
    }
}

fn num(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:+.4} pips"),
        Some(_) => "<non-finite>".to_string(),
        None => "<none>".to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── the rotation ────────────────────────────────────────────────────────

    #[test]
    fn a_rotation_preserves_every_column_marginal() {
        // The property that makes it a control at all: the marginal
        // distribution of every feature is untouched.
        let (rows, cols) = (10usize, 3usize);
        let src: Vec<i32> = (0..(rows * cols) as i32).collect();
        let out = rotate_block(&src, rows, cols, 3, BlockLayout::ColumnMajor).unwrap();
        for c in 0..cols {
            let mut a: Vec<i32> = src[c * rows..(c + 1) * rows].to_vec();
            let mut b: Vec<i32> = out[c * rows..(c + 1) * rows].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "column {c} marginal changed");
        }
    }

    #[test]
    fn a_rotation_moves_every_column_by_the_same_offset() {
        // Joint, not per-column: cross-feature structure survives. If columns
        // moved independently the control would destroy the correlations
        // between features too, and would be measuring two things at once.
        let (rows, cols, tau) = (8usize, 4usize, 3usize);
        let src: Vec<usize> = (0..rows * cols).collect();
        let out = rotate_block(&src, rows, cols, tau, BlockLayout::ColumnMajor).unwrap();
        for c in 0..cols {
            for r in 0..rows {
                assert_eq!(out[c * rows + r], src[c * rows + (r + tau) % rows]);
            }
        }
    }

    #[test]
    fn row_major_and_column_major_rotate_the_same_bars() {
        let (rows, cols, tau) = (6usize, 2usize, 2usize);
        let col_major: Vec<usize> = (0..rows * cols).collect();
        let mut row_major = vec![0usize; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                row_major[r * cols + c] = col_major[c * rows + r];
            }
        }
        let a = rotate_block(&col_major, rows, cols, tau, BlockLayout::ColumnMajor).unwrap();
        let b = rotate_block(&row_major, rows, cols, tau, BlockLayout::RowMajor).unwrap();
        for c in 0..cols {
            for r in 0..rows {
                assert_eq!(a[c * rows + r], b[r * cols + c]);
            }
        }
    }

    #[test]
    fn a_zero_rotation_is_refused_rather_than_run() {
        // A rotation of zero is a second live run wearing the control's label.
        let src: Vec<u8> = vec![0; 12];
        assert!(
            rotate_block(&src, 6, 2, 0, BlockLayout::ColumnMajor)
                .unwrap_err()
                .contains("no-op")
        );
        assert!(
            rotate_block(&src, 6, 2, 6, BlockLayout::ColumnMajor)
                .unwrap_err()
                .contains("no-op")
        );
    }

    #[test]
    fn a_block_whose_shape_disagrees_with_its_length_is_refused() {
        let src: Vec<u8> = vec![0; 11];
        assert!(
            rotate_block(&src, 6, 2, 3, BlockLayout::ColumnMajor)
                .unwrap_err()
                .contains("REFUSED")
        );
    }

    #[test]
    fn the_permutation_is_reproducible_and_inside_the_band() {
        let session = Session::default();
        let a = FeaturePermutation::draw(ControlKind::CircularRotation, &session, BlockId(3));
        let b = FeaturePermutation::draw(ControlKind::CircularRotation, &session, BlockId(3));
        assert_eq!(a, b, "a control nobody can re-run is not evidence");
        assert!(a.tau() >= TAU_LO_FRAC && a.tau() <= TAU_HI_FRAC);
        let rows = 100_000;
        let off = a.offset_rows(rows).unwrap();
        assert!(off >= (TAU_LO_FRAC * rows as f64).ceil() as usize);
        assert!(off <= (TAU_HI_FRAC * rows as f64).floor() as usize);
    }

    #[test]
    fn different_blocks_and_kinds_get_different_permutations() {
        let s = Session::default();
        let b3 = FeaturePermutation::draw(ControlKind::CircularRotation, &s, BlockId(3));
        let b4 = FeaturePermutation::draw(ControlKind::CircularRotation, &s, BlockId(4));
        let other = FeaturePermutation::draw(ControlKind::ColumnPermutation, &s, BlockId(3));
        assert_ne!(b3.tau(), b4.tau());
        assert_ne!(b3.kind(), other.kind());
    }

    /// The band is `τ ∈ [0.05·T, 0.95·T]` (design §9, `TAU_LO_FRAC` /
    /// `TAU_HI_FRAC`), so "too short" means *the band contains no integer* — and
    /// under this band that is `T ≤ 1`, not any small `T`.
    ///
    /// Asserted in BOTH directions on purpose. A one-sided test would pass just
    /// as happily against a function that refused everything, which is the
    /// mirror-image bug: a control that never runs makes the shuffle null empty,
    /// S6 permanently `Unavailable`, and the loop unfalsifiable — the exact
    /// failure this module exists to prevent.
    #[test]
    fn a_window_too_short_to_rotate_is_named_not_clamped() {
        let s = Session::default();
        let p = FeaturePermutation::draw(ControlKind::CircularRotation, &s, BlockId(0));

        // No integer offset exists in [0.05·T, 0.95·T]: named, not clamped to 0.
        for rows in [0usize, 1] {
            let err = p.offset_rows(rows).expect_err(
                "a block the band admits no integer offset for must be refused, not clamped",
            );
            assert!(err.contains("REFUSED"), "rows={rows}: got {err}");
        }

        // A block that DOES admit an offset gets one, strictly inside the band.
        for rows in [4usize, 100, 5_000] {
            let tau = p
                .offset_rows(rows)
                .unwrap_or_else(|e| panic!("rows={rows} admits an offset but was refused: {e}"));
            assert!(tau > 0, "rows={rows}: a rotation of zero is not a control");
            assert!(tau < rows, "rows={rows}: tau={tau} would wrap to a no-op");
            assert!(
                tau as f64 >= TAU_LO_FRAC * rows as f64 && tau as f64 <= TAU_HI_FRAC * rows as f64,
                "rows={rows}: tau={tau} left the band"
            );
        }
    }

    #[test]
    fn the_column_permutation_diagnostic_preserves_column_marginals() {
        let (rows, cols) = (12usize, 3usize);
        let src: Vec<i32> = (0..(rows * cols) as i32).collect();
        let out =
            permute_columns_independently(&src, rows, cols, BlockLayout::ColumnMajor, 7).unwrap();
        for c in 0..cols {
            let mut a: Vec<i32> = src[c * rows..(c + 1) * rows].to_vec();
            let mut b: Vec<i32> = out[c * rows..(c + 1) * rows].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
    }

    // ── the null ────────────────────────────────────────────────────────────

    #[test]
    fn s6_is_unavailable_until_three_observations_exist() {
        let mut null = ShuffleNull::new();
        assert!(!null.is_available());
        assert_eq!(null.quantile_95(), None);
        null.observe(ControlKind::CircularRotation, Some(0.1));
        null.observe(ControlKind::CircularRotation, Some(0.2));
        assert!(!null.is_available());
        null.observe(ControlKind::CircularRotation, Some(0.3));
        assert!(null.is_available());
        // With exactly three points the conservative reading is the maximum.
        assert_eq!(null.quantile_95(), Some(0.3));
    }

    #[test]
    fn the_weaker_diagnostic_never_enters_the_null() {
        let mut null = ShuffleNull::new();
        for _ in 0..5 {
            null.observe(ControlKind::ColumnPermutation, Some(9.9));
        }
        assert_eq!(null.len(), 0);
        assert!(!null.is_available());
    }

    #[test]
    fn a_refused_control_is_counted_and_never_silently_dropped() {
        let mut null = ShuffleNull::new();
        null.observe(ControlKind::CircularRotation, None);
        null.observe(ControlKind::CircularRotation, Some(f64::NAN));
        assert_eq!(null.len(), 0);
        assert_eq!(null.refused(), 2);
    }

    // ── the block comparison ────────────────────────────────────────────────

    #[test]
    fn a_missing_control_number_is_a_failure_not_a_pass() {
        // "No number is never good news." A broken control sweep must not be
        // able to manufacture confidence in the live one.
        let s = Session::default();
        let j = judge_block(&s, BlockId(0), None, None);
        assert!(j.indistinguishable);
        assert_eq!(j.p_block, 1.0);
        assert!(j.delta_expectancy.is_finite() && j.dsr_gap.is_finite());
    }

    /// C6. `dsr_gap` compares the LIVE side against the CONTROL side, and the
    /// control re-runs ONE sweep — the block's best — byte for byte. Maxing the
    /// live DSR over every sweep in the block put up to four times as many draws
    /// on the live side of a maximum, so the gap read positive on sample size
    /// alone and the block passed a falsification it had not survived.
    #[test]
    fn the_live_dsr_comes_from_the_sweep_the_control_re_ran_not_from_the_block() {
        use crate::journal::{SearchRecord, SweepKind};
        use crate::session::{SweepOutcome, SweepSummary};

        let search = |e: Option<f64>, dsr: Option<f64>| SearchRecord {
            slot: 0,
            config_hash: "fnv64:h".to_string(),
            trials_offered: 10,
            survivors: 1,
            e_screen_pess: e,
            n_trades: 400,
            dsr,
            pbo: Some(0.2),
            dsr_refusal: None,
            pbo_refusal: None,
            cost_band: crate::journal::CostBandCounts::default(),
            rejections: Vec::new(),
            streamed: true,
            batch_columns: 0,
            next_cursor: 0,
            wall_ms: 1,
            error: None,
        };
        let sweep = |id: u64, e: f64, dsr: f64| SweepSummary {
            sweep: SweepId(id),
            kind: SweepKind::Live,
            sweep_hash: format!("fnv64:{id}"),
            trials_offered: 10,
            outcome: SweepOutcome::Completed,
            per_search: vec![search(Some(e), Some(dsr))],
            wall_ms: 1,
        };

        let mut s = Session::default();
        // Sweep 1 attains the block's best EXPECTANCY, so it is the sweep the
        // control re-runs. Sweep 2 has a higher DSR and is NOT re-run.
        s.sweeps.push(sweep(1, 0.90, 0.55));
        s.sweeps.push(sweep(2, 0.10, 0.99));
        assert_eq!(best_sweep_of_block(&s, BlockId(0)), Some(SweepId(1)));

        // Control DSR 0.70: it beat sweep 1, which is the honest comparison.
        let j = judge_block(&s, BlockId(0), Some(0.20), Some(0.70));
        assert!(
            (j.dsr_gap - (0.55 - 0.70)).abs() < 1e-12,
            "dsr_gap must be the SOURCE sweep's 0.55 against 0.70, got {}",
            j.dsr_gap
        );
        assert!(
            j.indistinguishable,
            "a negative dsr_gap is a block the control was not beaten in"
        );
    }

    #[test]
    fn an_empty_block_cannot_be_beaten_by_its_control() {
        // A block with no live sweep recorded has nothing to compare, and
        // "nothing" must never read as a win.
        let s = Session::default();
        assert!(judge_block(&s, BlockId(0), Some(0.9), Some(0.9)).indistinguishable);
        assert!(best_sweep_of_block(&s, BlockId(0)).is_none());
    }

    // ── the summary ─────────────────────────────────────────────────────────

    #[test]
    fn the_summary_admits_when_nothing_has_been_falsified_yet() {
        let text = ShuffleSummary::from_session(&Session::default()).render();
        assert!(text.starts_with("SHUFFLE CONTROL"));
        assert!(text.contains("NOTHING above has been falsified"));
        assert!(text.contains("UNAVAILABLE"));
    }

    #[test]
    fn the_summary_says_the_loop_is_mining_noise_when_it_does_not_beat_the_null() {
        let sum = ShuffleSummary {
            blocks_judged: 4,
            indistinguishable: 4,
            session_p: Some(1.0),
            best_live_e_screen_pess: Some(0.30),
            best_shuffle_e_screen_pess: Some(0.35),
            q_shuffle_95: Some(0.35),
            null_observations: 4,
            null_refusals: 0,
            null_available: true,
        };
        let text = sum.render();
        assert!(text.contains("mining noise"), "{text}");
        assert_eq!(sum.indistinguishable_fraction(), Some(1.0));
    }

    #[test]
    fn the_summary_reports_the_margin_when_the_loop_does_beat_the_null() {
        let sum = ShuffleSummary {
            blocks_judged: 4,
            indistinguishable: 1,
            session_p: Some(0.25),
            best_live_e_screen_pess: Some(0.90),
            best_shuffle_e_screen_pess: Some(0.35),
            q_shuffle_95: Some(0.35),
            null_observations: 4,
            null_refusals: 0,
            null_available: true,
        };
        assert!(sum.render().contains("beats its own shuffle control"));
    }
}
