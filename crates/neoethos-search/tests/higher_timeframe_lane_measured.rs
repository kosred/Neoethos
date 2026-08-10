//! THE HIGHER-TIMEFRAME PREFILTER LANE, RE-MEASURED ON REAL BARS.
//!
//! # Why this test exists
//!
//! A number was in circulation and was being used to decide which timeframes
//! are worth searching:
//!
//! > "base keeps 217/217, H1 keeps 40/217, H4 keeps 8/217, and none of the 8
//! > earned their place."
//!
//! **That number is VOID.** It was produced by the pre-2026-08-09 f32
//! `pearson_correlation`, which returned exactly `0.0` whenever its denominator
//! was non-finite, meeting `core::features::align_features_by_ns`, which
//! initialises every aligned higher-timeframe cell to `NaN` by construction.
//! One NaN anywhere in a column made the whole column score exactly `0.0`, the
//! stable sort broke the resulting mass tie by original column index, and base
//! columns — emitted first in the cube — swept the top-K. The prefilter was not
//! ranking the higher timeframes badly. It was not ranking them at all.
//!
//! That correlation has been replaced (`neoethos_data::core::stats_f64`, f64,
//! two-pass, pairwise-complete). This test is the re-measurement, made
//! repeatable so that ANY future change to the prefilter — a new early-reject
//! predicate, a different `top_k`, a change to `min_per_tf`, a change to
//! `MIN_PAIRWISE_SAMPLES` — can be re-run against a comparable answer instead
//! of against a remembered anecdote.
//!
//! Written up in `docs/higher-timeframe-lane-2026-08-09.md`. If you are about
//! to cite a higher-timeframe keep-rate, cite that document, and re-run this
//! test first.
//!
//! # What it reports, per timeframe
//!
//! * columns OFFERED to the prefilter
//! * columns KEPT, and the MECHANISM that kept each one — rank, the `regime_`
//!   INFINITY exemption, the per-timeframe quota, or the seed-template
//!   force-keep. "Kept" without a mechanism is how "H4 keeps 8" was read as
//!   "H4 has 8 useful features" when in fact 6 were quota and 2 were force-keep.
//! * the SCORE DISTRIBUTION (min/p10/p25/median/p75/p90/max of |r|)
//! * the USED / SKIPPED row counts the repaired correlation itself returns —
//!   its own accounting, not an estimate
//! * the EFFECTIVE SAMPLE SIZE behind each score: distinct runs of the
//!   bit-identical value, i.e. the number of times the column actually changed.
//!   A higher-TF column is forward-filled onto base bars, so one H4 observation
//!   becomes 48 identical M5 rows and `used` counts it 48 times. Without this
//!   column the measurement reads as "H4 wins" and it is not true.
//!
//! and it scores every column FOUR ways over the SAME rows so the correlation
//! function is isolated from the label change and the fold change:
//!
//! | variant         | correlation | target                | windows      |
//! |-----------------|-------------|-----------------------|--------------|
//! | `legacy_prefix` | OLD f32     | 1-bar forward return  | 80% prefix   |
//! | `new_prefix`    | repaired    | 1-bar forward return  | 80% prefix   |
//! | `legacy_cpcv`   | OLD f32     | triple-barrier        | CPCV folds   |
//! | `new_cpcv`      | repaired    | triple-barrier        | CPCV folds   |
//!
//! `legacy_prefix` is the exact regime in which the void number was produced,
//! and reproducing it is one of this test's assertions.
//!
//! # Running it
//!
//! ```text
//! cargo test -p neoethos-search --release --test higher_timeframe_lane_measured \
//!     -- --ignored --nocapture
//! ```
//!
//! `--release` is not optional in practice: this builds the full multi-timeframe
//! feature cube and correlates every column over seven CPCV folds twice.
//!
//! # It is `#[ignore]`d, and a missing store is a HARD FAILURE
//!
//! Same discipline as `neoethos-data/tests/vocabulary_restoration_measured.rs`.
//! A measurement test that prints "SKIPPED" and passes green is the defect this
//! whole workstream exists to kill, wearing a test's clothing. So it never runs
//! where the store is absent — and when it IS run, a missing store panics with
//! the resolved path in the message.

use neoethos_data::core::stats_f64::{PearsonOutcome, pearson_pairwise, pearson_pairwise_f32};
use neoethos_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

// ─── measurement parameters ──────────────────────────────────────────────────
//
// These are CONSTANTS, deliberately, with no environment or CLI override. The
// point of the test is that two people on two days get a comparable answer;
// a knob is how "comparable" quietly stops being true. Every one of them is
// either the production value or is justified in place.

/// The operator's primary symbol, and the one every prior higher-timeframe
/// claim was made on.
const SYMBOL: &str = "EURUSD";
/// Production base timeframe for discovery.
const BASE_TF: &str = "M5";
/// The two higher lanes the void number was about.
const HIGHER_TFS: [&str; 2] = ["H1", "H4"];

/// Base bars retained (the TAIL, so the fit windows see the most recent bars).
///
/// Two independent reasons, both measured:
///
/// 1. `PREFILTER_CPCV_MAX_ROWS` is 200,000. The prefilter's fit windows are cut
///    from the LAST 200,000 rows regardless, so truncating here changes which
///    bars the INDICATORS warm up on and nothing else about the fit.
/// 2. On full history (823,801 M5 bars) the run does not reach the prefilter at
///    all: the vocabulary budget shrinks with frame width until the floor in
///    `indicator_ledger::enforce_floor` equals exactly what the budget afforded,
///    leaving zero tolerance for the routine per-frame drops, and the run aborts
///    with `INDICATOR VOCABULARY COLLAPSE at stage 'classic-ta'`. That is a real
///    open defect (see the document), not something this test should paper over
///    — but it means the full-history measurement is currently unobtainable and
///    this test would be permanently red rather than informative.
const MAX_BASE_BARS: usize = 200_000;

/// `models.discovery_runtime.prefilter_top_k`, shipped value.
const TOP_K: usize = 240;
/// `models.discovery_runtime.prefilter_min_per_timeframe`, shipped value.
const MIN_PER_TF: usize = 6;

/// `models.data_runtime.normalize_features`.
///
/// THE SINGLE MOST IMPORTANT SWITCH IN THIS MEASUREMENT, so it is pinned rather
/// than inherited. `false` is the operator's installed setting and
/// `neoethos-data`'s own default, and it is the regime in which the alignment
/// NaN survives all the way to the prefilter — i.e. the regime the void number
/// was produced in. With `true` the NaN becomes `0.0` upstream, every column is
/// 100% dense, and the whole finding changes shape. `neoethos-core`'s default is
/// `true`, so which regime a real run lands in currently depends on which
/// default wins; that disagreement is reported in the document.
const NORMALIZE_FEATURES: bool = false;

/// Label geometry, matching what `discovery.rs` derives from the shipped gene
/// stop band: `sl_atr_mult` = midpoint of (`sl_min_atr` 1.0, `sl_max_atr` 4.0),
/// `rr` = midpoint of (`rr_min` 1.5, `rr_max` 4.0).
const SL_ATR_MULT: f64 = 2.5;
const RR: f64 = 2.75;
const ATR_PERIOD: usize = 14;
const MAX_HOLD_BARS: usize = 35;
const SPREAD_PIPS: f64 = 1.5;
const PIP: f64 = 0.0001;

/// `(n_splits, n_test_groups, embargo_pct, purge_pct, max_rows)` — the
/// production CPCV configuration the prefilter refits on.
const CPCV: (usize, usize, f64, f64, usize) = (8, 2, 0.01, 0.02, 200_000);

/// `discovery::PREFILTER_MAX_REFIT_FOLDS`.
const PREFILTER_MAX_REFIT_FOLDS: usize = 8;
/// `discovery::MIN_DECIDED_FIRST_PASSAGE_LABELS` — below this the production
/// code abandons the triple-barrier label and falls back to a 1-bar return, in
/// which case this measurement would not be measuring the production label.
const MIN_DECIDED_FIRST_PASSAGE_LABELS: usize = 100;

// ─── store resolution: hard failure, path named ──────────────────────────────

/// Resolve the operator's real vortex store, or FAIL naming the resolved path.
fn store_root() -> String {
    let base = std::env::var("LOCALAPPDATA")
        .expect("LOCALAPPDATA is unset — cannot resolve the neoethos data store");
    let root = format!("{base}/neoethos/data");
    assert!(
        std::path::Path::new(&root).is_dir(),
        "no neoethos data store at {root} — this measurement is only meaningful on the real \
         store, so it fails rather than skipping. Import the bars first."
    );
    for tf in std::iter::once(BASE_TF).chain(HIGHER_TFS) {
        let part = format!("{root}/symbol={SYMBOL}/timeframe={tf}/data.vortex");
        assert!(
            std::path::Path::new(&part).is_file(),
            "the store at {root} has no {SYMBOL} {tf} partition (expected {part}). The \
             higher-timeframe lane cannot be measured without every requested timeframe present \
             — a partial cube would silently under-report the lane, which is exactly the class \
             of error this test retracts."
        );
    }
    root
}

// ─── verbatim copies of the private discovery internals ──────────────────────
//
// `prefilter_features`, `first_passage_labels`, `prefilter_fit_windows`,
// `timeframe_group` and `rolling_atr_f64` are private to `discovery.rs`, and the
// mechanism attribution this measurement needs does not exist in them at all.
// They are reproduced here verbatim. If any of them changes in `discovery.rs`
// and is not changed here, the measurement stops describing the shipped
// prefilter — so treat a divergence as a bug in this file.

/// `discovery::timeframe_group`, verbatim.
fn timeframe_group(name: &str) -> Option<&str> {
    let head = name.split('_').next()?;
    if head.len() < 2 || head.len() > 3 {
        return None;
    }
    let digits = if let Some(rest) = head.strip_prefix("MN") {
        rest
    } else if head.starts_with(['M', 'H', 'D', 'W']) {
        &head[1..]
    } else {
        return None;
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(head)
}

fn tf_of(name: &str) -> String {
    timeframe_group(name)
        .map(|g| g.to_string())
        .unwrap_or_else(|| "base".to_string())
}

/// `discovery::rolling_atr_f64`, same arithmetic with a running sum so a
/// 200,000-bar series does not spend minutes in an O(n·p) double loop. Same
/// window `[i+1-period, i]`, same finite filter, same "no finite true range in
/// the window yields NaN and the labeller COUNTS it" behaviour.
fn rolling_atr_f64(ohlcv: &Ohlcv, period: usize) -> Vec<f64> {
    let n = ohlcv.close.len();
    let period = period.max(1);
    let mut tr = vec![0.0f64; n];
    for i in 0..n {
        let hi = ohlcv.high[i];
        let lo = ohlcv.low[i];
        let prev_close = if i > 0 { ohlcv.close[i - 1] } else { ohlcv.close[i] };
        if !hi.is_finite() || !lo.is_finite() || !prev_close.is_finite() {
            tr[i] = f64::NAN;
            continue;
        }
        tr[i] = (hi - lo)
            .max((hi - prev_close).abs())
            .max((lo - prev_close).abs());
    }
    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        if tr[i].is_finite() {
            sum += tr[i];
            count += 1;
        }
        if i >= period {
            let drop = tr[i - period];
            if drop.is_finite() {
                sum -= drop;
                count -= 1;
            }
        }
        if count > 0 {
            out[i] = sum / count as f64;
        }
    }
    out
}

#[derive(Default, Debug)]
struct LabelCensus {
    up: usize,
    down: usize,
    vertical: usize,
    ambiguous: usize,
    short_win: usize,
    short_loss: usize,
    vertical_short: usize,
    ambiguous_short: usize,
    undefined: usize,
}

/// `discovery::first_passage_labels`, verbatim.
fn first_passage_labels(ohlcv: &Ohlcv, cost_px: f64) -> (Vec<f32>, Vec<f32>, LabelCensus) {
    let n = ohlcv.close.len();
    let mut long_labels = vec![f32::NAN; n];
    let mut short_labels = vec![f32::NAN; n];
    let mut census = LabelCensus::default();
    if n < 2 {
        census.undefined = n;
        return (long_labels, short_labels, census);
    }
    let atr = rolling_atr_f64(ohlcv, ATR_PERIOD);
    let hold = MAX_HOLD_BARS.max(1);

    for i in 0..n {
        let entry = ohlcv.close[i];
        let a = atr[i];
        if !entry.is_finite() || !a.is_finite() || a <= 0.0 || i + 1 >= n {
            census.undefined += 1;
            continue;
        }
        let stop_distance = SL_ATR_MULT * a;
        let take_distance = RR * stop_distance;
        let long_tp = entry + take_distance + cost_px;
        let long_sl = entry - stop_distance + cost_px;
        let short_tp = entry - take_distance - cost_px;
        let short_sl = entry + stop_distance - cost_px;
        let horizon_end = (i + hold).min(n - 1);

        let mut long_label = 0.0f32;
        let mut short_label = 0.0f32;
        let mut long_decided = false;
        let mut short_decided = false;
        for f in (i + 1)..=horizon_end {
            let hi = ohlcv.high[f];
            let lo = ohlcv.low[f];
            let hi_ok = hi.is_finite();
            let lo_ok = lo.is_finite();
            if !long_decided {
                match (hi_ok && hi >= long_tp, lo_ok && lo <= long_sl) {
                    (true, true) => {
                        census.ambiguous += 1;
                        long_decided = true;
                    }
                    (true, false) => {
                        long_label = 1.0;
                        census.up += 1;
                        long_decided = true;
                    }
                    (false, true) => {
                        long_label = -1.0;
                        census.down += 1;
                        long_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if !short_decided {
                match (lo_ok && lo <= short_tp, hi_ok && hi >= short_sl) {
                    (true, true) => {
                        census.ambiguous_short += 1;
                        short_decided = true;
                    }
                    (true, false) => {
                        short_label = 1.0;
                        census.short_win += 1;
                        short_decided = true;
                    }
                    (false, true) => {
                        short_label = -1.0;
                        census.short_loss += 1;
                        short_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if long_decided && short_decided {
                break;
            }
        }
        if !long_decided {
            census.vertical += 1;
        }
        if !short_decided {
            census.vertical_short += 1;
        }
        long_labels[i] = long_label;
        short_labels[i] = short_label;
    }
    (long_labels, short_labels, census)
}

/// `discovery::prefilter_fit_windows` with CPCV, verbatim.
fn cpcv_fit_windows(n_rows: usize) -> (Vec<Vec<usize>>, usize) {
    let (n_splits, n_test_groups, embargo_pct, purge_pct, max_rows) = CPCV;
    let capped = if max_rows > 0 { max_rows.min(n_rows) } else { n_rows };
    let offset = n_rows.saturating_sub(capped);
    let cv = neoethos_search::validation::CombinatorialPurgedCV::new(
        n_splits,
        n_test_groups,
        embargo_pct,
        purge_pct,
    );
    let splits = cv.split(capped);
    let available = splits.len();
    if available > 0 {
        let step = available.div_ceil(PREFILTER_MAX_REFIT_FOLDS).max(1);
        let windows: Vec<Vec<usize>> = splits
            .into_iter()
            .step_by(step)
            .take(PREFILTER_MAX_REFIT_FOLDS)
            .map(|(train, _test)| train.into_iter().map(|i| i + offset).collect())
            .filter(|w: &Vec<usize>| !w.is_empty())
            .collect();
        if !windows.is_empty() {
            return (windows, available);
        }
    }
    (Vec::new(), available)
}

/// `discovery::prefilter_fit_windows`' non-CPCV branch: the single leading
/// in-sample prefix the void number was measured over.
fn prefix_fit_window(n_rows: usize, insample_frac: f64) -> Vec<usize> {
    let train_end = ((n_rows as f64) * insample_frac).floor() as usize;
    let train_end = train_end.clamp(2, n_rows.saturating_sub(1)).max(2);
    (0..train_end.saturating_sub(1)).collect()
}

/// The PRE-REPAIR statistic, verbatim: single pass, f32 accumulators, `n` as
/// f32, and the `!den.is_finite() -> 0.0` guard that scored every NaN-carrying
/// column exactly zero. This function IS the void number's cause; it lives here
/// only so the void number can be reproduced and retracted with evidence.
fn legacy_pearson_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 || n != y.len() {
        return 0.0;
    }
    let n_f = n as f32;
    let (mut sx, mut sy, mut sxy, mut sx2, mut sy2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let (a, b) = (x[i], y[i]);
        sx += a;
        sy += b;
        sxy += a * b;
        sx2 += a * a;
        sy2 += b * b;
    }
    let num = n_f * sxy - sx * sy;
    let den = ((n_f * sx2 - sx * sx) * (n_f * sy2 - sy * sy)).sqrt();
    if den == 0.0 || !den.is_finite() {
        0.0
    } else {
        num / den
    }
}

// ─── the four variants and the mechanism attribution ─────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    LegacyPrefix,
    NewPrefix,
    LegacyCpcv,
    NewCpcv,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::LegacyPrefix => {
                "legacy_prefix (OLD f32 corr | 1-bar fwd ret | 80% prefix)  <- THE VOID NUMBER'S REGIME"
            }
            Variant::NewPrefix => "new_prefix    (repaired corr | 1-bar fwd ret | 80% prefix)",
            Variant::LegacyCpcv => "legacy_cpcv   (OLD f32 corr | triple-barrier | CPCV folds)",
            Variant::NewCpcv => {
                "new_cpcv      (repaired corr | triple-barrier | CPCV folds)  <- WHAT SHIPS TODAY"
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ColumnScore {
    score: f64,
    rankable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mechanism {
    Rank,
    RegimeExemption,
    Quota,
    SeedTemplate,
}

impl Mechanism {
    fn label(self) -> &'static str {
        match self {
            Mechanism::Rank => "rank",
            Mechanism::RegimeExemption => "regime-exemption",
            Mechanism::Quota => "per-TF-quota",
            Mechanism::SeedTemplate => "seed-template",
        }
    }
}

/// `discovery::prefilter_features`' selection half, verbatim, but returning WHY
/// each column was kept instead of only the surviving frame.
///
/// The order is the shipped order and it matters: global top-K (with `regime_`
/// columns riding the INFINITY slots at the head so they do not consume the
/// operator's budget), THEN the per-timeframe quota, THEN the seed-template
/// force-keep. A column is attributed to the FIRST mechanism that would have
/// kept it, which is the only attribution that answers "would this column have
/// survived on its own merit".
fn select_with_mechanisms(names: &[String], scores: &[ColumnScore]) -> HashMap<usize, Mechanism> {
    let n_cols = names.len();
    let mut correlations: Vec<(usize, f64)> = Vec::with_capacity(n_cols);
    for (idx, s) in scores.iter().enumerate() {
        if !s.rankable {
            continue;
        }
        correlations.push((idx, s.score));
    }
    let regime_forced = names.iter().filter(|n| n.starts_with("regime_")).count();

    correlations.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let actual_top_k = (TOP_K + regime_forced).min(n_cols);

    let mut kept: HashMap<usize, Mechanism> = HashMap::new();
    for (idx, _) in correlations.iter().take(actual_top_k) {
        let m = if names[*idx].starts_with("regime_") {
            Mechanism::RegimeExemption
        } else {
            Mechanism::Rank
        };
        kept.insert(*idx, m);
    }

    if MIN_PER_TF > 0 {
        let mut per_group: HashMap<&str, usize> = HashMap::new();
        for idx in kept.keys() {
            if let Some(group) = timeframe_group(&names[*idx]) {
                *per_group.entry(group).or_insert(0) += 1;
            }
        }
        for &(idx, _) in &correlations {
            let Some(group) = timeframe_group(&names[idx]) else {
                continue;
            };
            let count = per_group.entry(group).or_insert(0);
            if *count >= MIN_PER_TF {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = kept.entry(idx) {
                e.insert(Mechanism::Quota);
                *count += 1;
            }
        }
        for idx in neoethos_search::genetic::seed_templates::template_feature_indices(names) {
            kept.entry(idx).or_insert(Mechanism::SeedTemplate);
        }
    }
    kept
}

// ─── effective sample size ───────────────────────────────────────────────────

/// Collapse a fold's `(x, y)` rows into ONE POINT PER RUN of identical `x`.
///
/// A higher-timeframe column is forward-filled onto base bars, so one H4
/// observation appears as 48 bit-identical M5 rows and the correlation counts it
/// 48 times. A run ends when `x` changes or becomes non-finite; `y` is averaged
/// over the run's finite labels. The returned length is the number of times the
/// column actually CHANGED — the honest denominator for a significance test.
fn collapse_runs(xs: &[f32], ys: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut rx: Vec<f64> = Vec::new();
    let mut ry: Vec<f64> = Vec::new();
    let mut cur: Option<f32> = None;
    let mut acc = 0.0f64;
    let mut cnt = 0usize;
    let flush = |cur: &mut Option<f32>,
                 acc: &mut f64,
                 cnt: &mut usize,
                 rx: &mut Vec<f64>,
                 ry: &mut Vec<f64>| {
        if let Some(v) = cur.take() {
            if *cnt > 0 {
                rx.push(v as f64);
                ry.push(*acc / *cnt as f64);
            }
        }
        *acc = 0.0;
        *cnt = 0;
    };
    for i in 0..xs.len().min(ys.len()) {
        let x = xs[i];
        if !x.is_finite() {
            flush(&mut cur, &mut acc, &mut cnt, &mut rx, &mut ry);
            continue;
        }
        match cur {
            // `to_bits` equality: two values are the same OBSERVATION only when
            // they are bit-identical, which is exactly what a forward fill
            // produces and what a genuine re-computation almost never does.
            Some(v) if v.to_bits() == x.to_bits() => {}
            _ => {
                flush(&mut cur, &mut acc, &mut cnt, &mut rx, &mut ry);
                cur = Some(x);
            }
        }
        if ys[i].is_finite() {
            acc += ys[i] as f64;
            cnt += 1;
        }
    }
    flush(&mut cur, &mut acc, &mut cnt, &mut rx, &mut ry);
    (rx, ry)
}

/// `t = |r|·sqrt(n_eff - 2) / sqrt(1 - r²)` — the comparable quantity across
/// timeframes, because `|r|` alone is not: a base column's `|r|` rests on
/// ~138,000 bars, an H4 column's on ~2,800, and a sparse H4 event column's on
/// fifteen.
fn t_statistic(r: f64, n_eff: usize) -> f64 {
    if !r.is_finite() || n_eff < 3 || r.abs() >= 1.0 {
        return 0.0;
    }
    r.abs() * ((n_eff - 2) as f64).sqrt() / (1.0 - r * r).sqrt()
}

/// Two-sided Bonferroni |z| for `n_tests` simultaneous tests at family-wise 5%.
///
/// The prefilter runs one hypothesis test per column and keeps the winners,
/// which is precisely the setting where an uncorrected maximum is meaningless.
fn bonferroni_z(n_tests: usize) -> f64 {
    let target = 0.05 / (2.0 * n_tests.max(1) as f64);
    let (mut lo, mut hi) = (0.0f64, 12.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let tail = 0.5 * (1.0 - statrs::function::erf::erf(mid / std::f64::consts::SQRT_2));
        if tail > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ─── reporting helpers ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Quant {
    min: f64,
    p10: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    p90: f64,
    max: f64,
}

fn quantiles(mut v: Vec<f64>) -> Option<Quant> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 { v[(((v.len() - 1) as f64) * p).round() as usize] };
    Some(Quant {
        min: v[0],
        p10: q(0.10),
        p25: q(0.25),
        p50: q(0.50),
        p75: q(0.75),
        p90: q(0.90),
        max: v[v.len() - 1],
    })
}

/// Per-column results, all four variants plus the evidence behind them.
struct Row {
    legacy_prefix: ColumnScore,
    new_prefix: ColumnScore,
    legacy_cpcv: ColumnScore,
    new_cpcv: ColumnScore,
    /// Pairwise-complete rows the repaired function used, summed over folds.
    used_total: u64,
    /// Rows it skipped because the feature or the label was non-finite.
    skipped_total: u64,
    /// Rows offered to the correlation across the folds.
    offered_total: u64,
    /// Minimum `used` over the folds — the fold the worst-fold rule ranks on.
    used_min: u64,
    /// Minimum distinct-run count over the folds: the effective sample size.
    runs_min: u64,
    /// `|r|` recomputed one point per run, worst over folds.
    r_runs: f64,
    r_runs_rankable: bool,
    /// Rows the repaired function used in the single 80% prefix window.
    prefix_used: u64,
}

impl Row {
    fn score(&self, v: Variant) -> ColumnScore {
        match v {
            Variant::LegacyPrefix => self.legacy_prefix,
            Variant::NewPrefix => self.new_prefix,
            Variant::LegacyCpcv => self.legacy_cpcv,
            Variant::NewCpcv => self.new_cpcv,
        }
    }
}

// ─── the measurement ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires the real EURUSD M5/H1/H4 vortex store and builds the full multi-timeframe \
            feature cube. Re-run it after ANY change to the prefilter, the correlation, or \
            MIN_PAIRWISE_SAMPLES — `cargo test -p neoethos-search --release --test \
            higher_timeframe_lane_measured -- --ignored --nocapture`"]
fn measure_the_higher_timeframe_lane_on_real_bars() {
    let root = store_root();

    // Pin the runtime switch rather than inherit it. See NORMALIZE_FEATURES.
    neoethos_data::install_data_runtime_overrides(NORMALIZE_FEATURES, false);

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  HIGHER-TIMEFRAME PREFILTER LANE — MEASURED ON REAL BARS                 ║");
    println!("║  The old \"H4 keeps 8/217, none earned it\" number is VOID. See            ║");
    println!("║  docs/higher-timeframe-lane-2026-08-09.md before citing anything here.   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
    println!("store              {root}");
    println!("symbol             {SYMBOL}   base {BASE_TF}   higher {HIGHER_TFS:?}");
    println!("prefilter          top_k={TOP_K}  min_per_tf={MIN_PER_TF}");
    println!(
        "normalize_features {NORMALIZE_FEATURES}   (true = the alignment NaN becomes 0.0 BEFORE \
         the prefilter sees it)"
    );

    let t0 = std::time::Instant::now();
    let mut tfs: Vec<&str> = vec![BASE_TF];
    tfs.extend(HIGHER_TFS);
    let mut dataset = neoethos_data::load_symbol_dataset_with_timeframes(&root, SYMBOL, &tfs)
        .unwrap_or_else(|e| panic!("failed to load {SYMBOL} {tfs:?} from {root}: {e:#}"));
    let mut loaded: Vec<(String, usize)> = dataset
        .frames
        .iter()
        .map(|(tf, o)| (tf.clone(), o.close.len()))
        .collect();
    loaded.sort();
    for (tf, rows) in &loaded {
        println!("loaded             {tf:<4} rows={rows}");
    }

    // Truncate the BASE only, keeping the tail. The higher frames stay whole so
    // the alignment still has bars to bind to.
    {
        let o = dataset
            .frames
            .get_mut(BASE_TF)
            .unwrap_or_else(|| panic!("base timeframe {BASE_TF} missing from the loaded dataset"));
        let n = o.close.len();
        assert!(
            n >= MAX_BASE_BARS,
            "the {SYMBOL} {BASE_TF} partition holds {n} bars, fewer than the {MAX_BASE_BARS} this \
             measurement needs"
        );
        let start = n - MAX_BASE_BARS;
        o.open = o.open[start..].to_vec();
        o.high = o.high[start..].to_vec();
        o.low = o.low[start..].to_vec();
        o.close = o.close[start..].to_vec();
        if let Some(v) = o.volume.as_mut() {
            *v = v[start..].to_vec();
        }
        if let Some(ts) = o.timestamp.as_mut() {
            *ts = ts[start..].to_vec();
        }
        println!("TRUNCATED base to the last {MAX_BASE_BARS} bars (see MAX_BASE_BARS for why)");
    }

    let ohlcv = dataset
        .frames
        .get(BASE_TF)
        .cloned()
        .unwrap_or_else(|| panic!("base timeframe {BASE_TF} missing"));

    let higher_refs: Vec<&str> = HIGHER_TFS.to_vec();
    let features: FeatureFrame =
        neoethos_data::prepare_multitimeframe_features(&dataset, BASE_TF, &higher_refs)
            .unwrap_or_else(|e| {
                panic!(
                    "the multi-timeframe feature cube failed to build: {e:#}\n\
                     If this is an INDICATOR VOCABULARY COLLAPSE, that is the open \
                     `enforce_floor` defect described in docs/higher-timeframe-lane-2026-08-09.md \
                     — the floor is clamped to exactly what the budget afforded, leaving zero \
                     tolerance for the routine per-frame drops."
                )
            });

    let names = features.names.clone();
    let n_rows = features.n_samples();
    let n_cols = features.n_features();
    println!(
        "feature cube       rows={n_rows} cols={n_cols}   built in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let mut by_tf: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, n) in names.iter().enumerate() {
        by_tf.entry(tf_of(n)).or_default().push(i);
    }
    println!(
        "offered by TF      {:?}",
        by_tf
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect::<BTreeMap<_, _>>()
    );

    // A cube with no higher-timeframe columns would make every number below
    // vacuously "H4 keeps 0" — the exact shape of the claim being retracted.
    for tf in HIGHER_TFS {
        let offered = by_tf.get(tf).map(|v| v.len()).unwrap_or(0);
        assert!(
            offered > 0,
            "the cube offers ZERO {tf} columns. The higher-timeframe lane cannot be measured, and \
             a run reporting '{tf} keeps 0' from this cube would be reporting the cube, not the \
             lane."
        );
    }

    // ── labels ──────────────────────────────────────────────────────────────
    let cost_px = SPREAD_PIPS.max(0.0) * PIP;
    let atr_pips =
        neoethos_search::stop_target::median_atr_pips(&ohlcv.high, &ohlcv.low, &ohlcv.close, PIP, 14);
    let t1 = std::time::Instant::now();
    let (long_labels, short_labels, lc) = first_passage_labels(&ohlcv, cost_px);
    println!(
        "label geometry     sl={SL_ATR_MULT} ATR  rr={RR}  cost_px={cost_px:.6}  \
         hold={MAX_HOLD_BARS}  median_atr_pips={atr_pips:?}"
    );
    println!(
        "labels             long up={} down={} vert={} amb={} | short win={} loss={} vert={} \
         amb={} | undefined={}   ({:.1}s)",
        lc.up,
        lc.down,
        lc.vertical,
        lc.ambiguous,
        lc.short_win,
        lc.short_loss,
        lc.vertical_short,
        lc.ambiguous_short,
        lc.undefined,
        t1.elapsed().as_secs_f64()
    );
    let decided = (lc.up + lc.down).max(lc.short_win + lc.short_loss);
    assert!(
        decided >= MIN_DECIDED_FIRST_PASSAGE_LABELS,
        "only {decided} decided first-passage labels — below \
         MIN_DECIDED_FIRST_PASSAGE_LABELS={MIN_DECIDED_FIRST_PASSAGE_LABELS}, at which point \
         discovery abandons the triple-barrier label for a 1-bar forward return. This measurement \
         would then not be measuring the production label at all."
    );

    // The OLD target: 1-bar forward return.
    let mut fwd = vec![f32::NAN; ohlcv.close.len()];
    for i in 0..ohlcv.close.len().saturating_sub(1) {
        let d = ohlcv.close[i];
        if d.abs() > 1e-12 {
            fwd[i] = ((ohlcv.close[i + 1] - d) / d) as f32;
        }
    }

    // ── fit windows ─────────────────────────────────────────────────────────
    let (windows, folds_available) = cpcv_fit_windows(n_rows);
    assert!(
        !windows.is_empty(),
        "CPCV produced no usable fit windows over {n_rows} rows — nothing below would be measured \
         on the production folds"
    );
    let prefix_window = prefix_fit_window(n_rows, 0.80);
    println!(
        "cpcv windows       {} of {folds_available} folds; rows per fold {:?}",
        windows.len(),
        windows.iter().map(|w| w.len()).collect::<Vec<_>>()
    );
    println!(
        "prefix window      1 window, {} rows (leading 80%)",
        prefix_window.len()
    );

    // ── score every column, four ways plus effective-n, in one traversal ─────
    let t2 = std::time::Instant::now();
    let rows: Vec<Row> = (0..n_cols)
        .into_par_iter()
        .map(|col_idx| {
            let name = &names[col_idx];
            let is_regime = name.starts_with("regime_");
            let col: Vec<f32> = features.feature_column(col_idx).iter().copied().collect();

            let mut used_total = 0u64;
            let mut skipped_total = 0u64;
            let mut offered_total = 0u64;
            let mut used_min = u64::MAX;
            let mut runs_min = u64::MAX;

            let mut new_worst = f64::INFINITY;
            let mut legacy_worst = f64::INFINITY;
            let mut runs_worst = f64::INFINITY;
            let mut new_rankable = true;
            let mut runs_rankable = true;

            for window in &windows {
                let mut xs: Vec<f32> = Vec::with_capacity(window.len());
                let mut yl: Vec<f32> = Vec::with_capacity(window.len());
                let mut ys: Vec<f32> = Vec::with_capacity(window.len());
                for &row in window {
                    if row >= n_rows || row >= long_labels.len() {
                        continue;
                    }
                    xs.push(col[row]);
                    yl.push(long_labels[row]);
                    ys.push(short_labels.get(row).copied().unwrap_or(f32::NAN));
                }

                let ol: PearsonOutcome = pearson_pairwise_f32(&xs, &yl);
                let os: PearsonOutcome = pearson_pairwise_f32(&xs, &ys);
                // The repaired function's OWN accounting must close. This is the
                // "no silent drops" invariant at the level of a single column.
                assert_eq!(
                    ol.used + ol.skipped,
                    xs.len(),
                    "column '{name}': the correlation used {} + skipped {} rows but was offered \
                     {} — its own row accounting does not close, so every used/skipped number \
                     below is untrustworthy",
                    ol.used,
                    ol.skipped,
                    xs.len()
                );
                offered_total += xs.len() as u64;
                used_total += ol.used as u64;
                skipped_total += ol.skipped as u64;
                used_min = used_min.min(ol.used as u64);

                // Worst fold, best direction — the shipped rule.
                let mut a: Option<f64> = if ol.is_rankable() { Some(ol.abs()) } else { None };
                if os.is_rankable() {
                    let s = os.abs();
                    a = Some(a.map_or(s, |l: f64| l.max(s)));
                }
                match a {
                    Some(v) => new_worst = new_worst.min(v),
                    None => new_rankable = false,
                }

                let l = (legacy_pearson_f32(&xs, &yl) as f64)
                    .abs()
                    .max((legacy_pearson_f32(&xs, &ys) as f64).abs());
                legacy_worst = legacy_worst.min(l);

                // Same statistic at the column's OWN resolution.
                let (rxl, ryl) = collapse_runs(&xs, &yl);
                let (rxs, rys) = collapse_runs(&xs, &ys);
                runs_min = runs_min.min(rxl.len() as u64);
                let rl = pearson_pairwise(&rxl, &ryl);
                let rs = pearson_pairwise(&rxs, &rys);
                let mut r: Option<f64> = if rl.is_rankable() { Some(rl.abs()) } else { None };
                if rs.is_rankable() {
                    let s = rs.abs();
                    r = Some(r.map_or(s, |l: f64| l.max(s)));
                }
                match r {
                    Some(v) => runs_worst = runs_worst.min(v),
                    None => runs_rankable = false,
                }
            }
            if !new_rankable || !new_worst.is_finite() {
                new_worst = f64::NEG_INFINITY;
            }
            if !legacy_worst.is_finite() {
                legacy_worst = 0.0;
            }
            if !runs_rankable || !runs_worst.is_finite() {
                runs_worst = f64::NEG_INFINITY;
            }
            if used_min == u64::MAX {
                used_min = 0;
            }
            if runs_min == u64::MAX {
                runs_min = 0;
            }

            // The OLD regime: 1-bar forward return over the single leading prefix.
            let mut xs: Vec<f32> = Vec::with_capacity(prefix_window.len());
            let mut ys: Vec<f32> = Vec::with_capacity(prefix_window.len());
            for &row in &prefix_window {
                if row >= n_rows || row >= fwd.len() {
                    continue;
                }
                xs.push(col[row]);
                ys.push(fwd[row]);
            }
            let o = pearson_pairwise_f32(&xs, &ys);
            let legacy_prefix_score = (legacy_pearson_f32(&xs, &ys) as f64).abs();

            Row {
                legacy_prefix: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { legacy_prefix_score },
                    // The legacy function had no rankability concept at all:
                    // every column competed, including the ones it scored 0.0.
                    rankable: true,
                },
                new_prefix: ColumnScore {
                    score: if is_regime {
                        f64::INFINITY
                    } else if o.is_rankable() {
                        o.abs()
                    } else {
                        f64::NEG_INFINITY
                    },
                    rankable: is_regime || o.is_rankable(),
                },
                legacy_cpcv: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { legacy_worst },
                    rankable: true,
                },
                new_cpcv: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { new_worst },
                    rankable: is_regime || new_rankable,
                },
                used_total,
                skipped_total,
                offered_total,
                used_min,
                runs_min,
                r_runs: runs_worst,
                r_runs_rankable: runs_rankable && runs_worst.is_finite(),
                prefix_used: o.used as u64,
            }
        })
        .collect();
    println!(
        "scored             {n_cols} columns x 4 variants + effective-n in {:.1}s\n",
        t2.elapsed().as_secs_f64()
    );

    // ── ROW EVIDENCE: what each timeframe actually contributes ───────────────
    println!("═══ ROW EVIDENCE — used / skipped, as the repaired correlation reports them ═══");
    println!(
        "{:<6} {:>7} {:>14} {:>14} {:>9} {:>16} {:>13}",
        "TF", "cols", "used/fold avg", "skipped/fold", "used %", "worst-fold min", "prefix used"
    );
    let nw = windows.len().max(1) as f64;
    for (tf, idxs) in &by_tf {
        let cols = idxs.len() as f64;
        let used: f64 = idxs.iter().map(|i| rows[*i].used_total as f64).sum::<f64>() / cols / nw;
        let skipped: f64 =
            idxs.iter().map(|i| rows[*i].skipped_total as f64).sum::<f64>() / cols / nw;
        let offered: f64 =
            idxs.iter().map(|i| rows[*i].offered_total as f64).sum::<f64>() / cols / nw;
        let used_min = idxs.iter().map(|i| rows[*i].used_min).min().unwrap_or(0);
        let prefix_used: f64 =
            idxs.iter().map(|i| rows[*i].prefix_used as f64).sum::<f64>() / cols;
        println!(
            "{:<6} {:>7} {:>14.0} {:>14.0} {:>8.2}% {:>16} {:>13.0}",
            tf,
            idxs.len(),
            used,
            skipped,
            if offered > 0.0 { 100.0 * used / offered } else { 0.0 },
            used_min,
            prefix_used
        );
    }
    println!();

    // ── EFFECTIVE SAMPLE SIZE — the forward-fill inflation ──────────────────
    println!("═══ EFFECTIVE SAMPLE SIZE — `used` rows vs DISTINCT OBSERVATIONS (worst fold) ═══");
    println!(
        "A forward-filled H4 column repeats one observation across 48 M5 rows and the correlation\n\
         counts all 48. `runs` is the number of times the column actually changed.\n"
    );
    println!(
        "{:<6} {:>7} {:>12} {:>12} {:>12} {:>12} {:>11}",
        "TF", "cols", "used p10", "used med", "runs p10", "runs med", "runs/used"
    );
    for (tf, idxs) in &by_tf {
        let used: Vec<f64> = idxs.iter().map(|i| rows[*i].used_min as f64).collect();
        let runs: Vec<f64> = idxs.iter().map(|i| rows[*i].runs_min as f64).collect();
        let ratio: Vec<f64> = idxs
            .iter()
            .filter(|i| rows[**i].used_min > 0)
            .map(|i| rows[*i].runs_min as f64 / rows[*i].used_min as f64)
            .collect();
        let (Some(u), Some(r)) = (quantiles(used), quantiles(runs)) else {
            continue;
        };
        let rat = quantiles(ratio).map(|q| q.p50).unwrap_or(f64::NAN);
        println!(
            "{:<6} {:>7} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>11.4}",
            tf,
            idxs.len(),
            u.p10,
            u.p50,
            r.p10,
            r.p50,
            rat
        );
    }
    // Arithmetic invariant: a run cannot contain fewer than one used row.
    for (i, r) in rows.iter().enumerate() {
        assert!(
            r.runs_min <= r.used_min,
            "column '{}': {} distinct runs over {} used rows — a run is a block of used rows, so \
             runs can never exceed used",
            names[i],
            r.runs_min,
            r.used_min
        );
    }
    println!();

    // ── per variant: distribution, exactly-0.0 census, selection, mechanism ──
    let mut zero_by_variant: BTreeMap<&'static str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut rank_kept_by_variant: BTreeMap<&'static str, BTreeMap<String, usize>> = BTreeMap::new();

    for variant in [
        Variant::LegacyPrefix,
        Variant::NewPrefix,
        Variant::LegacyCpcv,
        Variant::NewCpcv,
    ] {
        let scores: Vec<ColumnScore> = rows.iter().map(|r| r.score(variant)).collect();
        println!("═══ VARIANT: {} ═══", variant.label());

        println!(
            "{:<6} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "TF", "cols", "unrank", "min", "p10", "p25", "median", "p75", "p90", "max"
        );
        for (tf, idxs) in &by_tf {
            let vals: Vec<f64> = idxs
                .iter()
                .filter(|i| !names[**i].starts_with("regime_"))
                .filter(|i| scores[**i].rankable && scores[**i].score.is_finite())
                .map(|i| scores[*i].score)
                .collect();
            let unrank = idxs
                .iter()
                .filter(|i| !names[**i].starts_with("regime_"))
                .filter(|i| !scores[**i].rankable || !scores[**i].score.is_finite())
                .count();
            match quantiles(vals) {
                Some(q) => println!(
                    "{:<6} {:>6} {:>7} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6}",
                    tf, idxs.len(), unrank, q.min, q.p10, q.p25, q.p50, q.p75, q.p90, q.max
                ),
                None => println!(
                    "{:<6} {:>6} {:>7}   (no rankable columns)",
                    tf,
                    idxs.len(),
                    unrank
                ),
            }
        }

        // Columns scoring EXACTLY 0.0 — the legacy bug's fingerprint.
        let mut zero_by_tf: BTreeMap<String, usize> = BTreeMap::new();
        for (tf, idxs) in &by_tf {
            let z = idxs
                .iter()
                .filter(|i| !names[**i].starts_with("regime_"))
                .filter(|i| scores[**i].score == 0.0)
                .count();
            zero_by_tf.insert(tf.clone(), z);
        }
        println!("exactly-0.0        {zero_by_tf:?}   <- the fingerprint of the void number");

        // Where the higher-TF columns sit in the global ranking. Under the
        // legacy function the median rank per timeframe collapses onto the
        // cube's index midpoints, which is the proof the "ranking" was column
        // order and nothing else.
        let mut ranked: Vec<(usize, f64)> = (0..n_cols)
            .filter(|i| scores[*i].rankable)
            .map(|i| (i, scores[i].score))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut rank_of: HashMap<usize, usize> = HashMap::new();
        for (pos, (idx, _)) in ranked.iter().enumerate() {
            rank_of.insert(*idx, pos);
        }
        let mut median_rank: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for (tf, idxs) in &by_tf {
            let mut r: Vec<usize> = idxs.iter().filter_map(|i| rank_of.get(i).copied()).collect();
            if r.is_empty() {
                continue;
            }
            r.sort_unstable();
            median_rank.insert(tf.clone(), (r[r.len() / 2], r[0]));
        }
        println!(
            "(median, best) rank by TF among {} rankable: {median_rank:?}",
            ranked.len()
        );

        // Selection + mechanism attribution.
        let kept = select_with_mechanisms(&names, &scores);
        let mut per_tf: BTreeMap<String, BTreeMap<&'static str, usize>> = BTreeMap::new();
        for (idx, mech) in &kept {
            *per_tf
                .entry(tf_of(&names[*idx]))
                .or_default()
                .entry(mech.label())
                .or_insert(0) += 1;
        }
        println!(
            "{:<6} {:>8} {:>7} {:>7} {:>9} {:>18} {:>15}",
            "TF", "offered", "kept", "rank", "quota", "regime-exemption", "seed-template"
        );
        let mut total_kept = 0usize;
        let mut rank_kept: BTreeMap<String, usize> = BTreeMap::new();
        for (tf, idxs) in &by_tf {
            let m = per_tf.get(tf).cloned().unwrap_or_default();
            let k: usize = m.values().sum();
            total_kept += k;
            rank_kept.insert(tf.clone(), m.get("rank").copied().unwrap_or(0));
            println!(
                "{:<6} {:>8} {:>7} {:>7} {:>9} {:>18} {:>15}",
                tf,
                idxs.len(),
                k,
                m.get("rank").copied().unwrap_or(0),
                m.get("per-TF-quota").copied().unwrap_or(0),
                m.get("regime-exemption").copied().unwrap_or(0),
                m.get("seed-template").copied().unwrap_or(0)
            );
        }
        assert_eq!(
            total_kept,
            kept.len(),
            "the per-timeframe mechanism table sums to {total_kept} but {} columns were kept — a \
             kept column is unattributed, which is the reporting failure that let '8 kept' be \
             read as '8 useful'",
            kept.len()
        );
        println!("TOTAL kept {total_kept} of {n_cols}");

        // Name the higher-TF survivors that earned their place ON RANK. The
        // claim "none of the 8 earned it" is exactly this list being empty.
        for tf in HIGHER_TFS {
            let mut earned: Vec<(usize, f64, usize)> = kept
                .iter()
                .filter(|(i, m)| **m == Mechanism::Rank && tf_of(&names[**i]) == tf)
                .map(|(i, _)| {
                    (
                        *i,
                        scores[*i].score,
                        rank_of.get(i).copied().unwrap_or(usize::MAX),
                    )
                })
                .collect();
            earned.sort_by_key(|(_, _, r)| *r);
            println!(
                "  {tf} kept ON RANK: {} — top 10: {:?}",
                earned.len(),
                earned
                    .iter()
                    .take(10)
                    .map(|(i, s, r)| format!("{}({s:.6} @#{r})", names[*i]))
                    .collect::<Vec<_>>()
            );
        }

        // HOW MANY ROWS DID THE RANK SURVIVORS EARN THEIR RANK ON? An |r| over
        // 700 of 138,000 rows outranking one over all of them is not a better
        // feature, it is a smaller sample.
        let fold_rows = windows.iter().map(|w| w.len()).min().unwrap_or(1) as f64;
        println!(
            "  row evidence of the RANK survivors (worst-fold `used`; smallest fold = \
             {fold_rows:.0} rows):"
        );
        for (tf, _) in &by_tf {
            let mins: Vec<f64> = kept
                .iter()
                .filter(|(i, m)| **m == Mechanism::Rank && tf_of(&names[**i]) == *tf)
                .map(|(i, _)| rows[*i].used_min as f64)
                .collect();
            if mins.is_empty() {
                println!("    {tf:<6} (none kept on rank)");
                continue;
            }
            let below_50 = mins.iter().filter(|v| **v < 0.50 * fold_rows).count();
            let below_5 = mins.iter().filter(|v| **v < 0.05 * fold_rows).count();
            let below_1 = mins.iter().filter(|v| **v < 0.01 * fold_rows).count();
            if let Some(q) = quantiles(mins.clone()) {
                println!(
                    "    {tf:<6} n={:<4} used_min min={:.0} p10={:.0} p25={:.0} median={:.0} \
                     max={:.0} | <50% of fold: {below_50}, <5%: {below_5}, <1%: {below_1}",
                    mins.len(),
                    q.min,
                    q.p10,
                    q.p25,
                    q.p50,
                    q.max
                );
            }
        }
        println!();

        zero_by_variant.insert(variant.label(), zero_by_tf);
        rank_kept_by_variant.insert(variant.label(), rank_kept);
    }

    // ── the fair comparison: t against the column's OWN effective n ──────────
    let z_bonf = bonferroni_z(n_cols);
    println!("═══ EVIDENCE-WEIGHTED — t = |r|·sqrt(n_eff-2)/sqrt(1-r²), n_eff = distinct runs ═══");
    println!(
        "Bonferroni |z| for {n_cols} simultaneous tests at family-wise 5% = {z_bonf:.3}\n\
         The prefilter runs one test per column and keeps the winners, so an uncorrected maximum\n\
         is not evidence of anything.\n"
    );
    let t_lanes: [(&str, bool); 2] = [
        ("t from r_dense (what the prefilter ranks on)", false),
        ("t from r_runs  (attenuation removed: one point per observation)", true),
    ];
    for (label, use_runs) in t_lanes {
        println!("-- {label} --");
        println!(
            "{:<6} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
            "TF", "tested", "t p50", "t p90", "t max", "n_eff p50", "t>=thresh", "share"
        );
        let mut ts_all: Vec<(usize, f64)> = Vec::new();
        for (tf, idxs) in &by_tf {
            let mut ts: Vec<f64> = Vec::new();
            let mut neff: Vec<f64> = Vec::new();
            let mut pass = 0usize;
            for i in idxs {
                if names[*i].starts_with("regime_") {
                    continue;
                }
                let (r, ok) = if use_runs {
                    (rows[*i].r_runs, rows[*i].r_runs_rankable)
                } else {
                    (rows[*i].new_cpcv.score, rows[*i].new_cpcv.rankable)
                };
                if !ok || !r.is_finite() {
                    continue;
                }
                let t = t_statistic(r, rows[*i].runs_min as usize);
                if t >= z_bonf {
                    pass += 1;
                }
                ts.push(t);
                neff.push(rows[*i].runs_min as f64);
                ts_all.push((*i, t));
            }
            let n = ts.len();
            match (quantiles(ts), quantiles(neff)) {
                (Some(q), Some(nq)) => println!(
                    "{:<6} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>12.0} {:>12} {:>9.1}%",
                    tf,
                    n,
                    q.p50,
                    q.p90,
                    q.max,
                    nq.p50,
                    pass,
                    100.0 * pass as f64 / n.max(1) as f64
                ),
                _ => println!("{tf:<6} (none testable)"),
            }
        }
        ts_all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut top_by_tf: BTreeMap<String, usize> = BTreeMap::new();
        for (i, _) in ts_all.iter().take(TOP_K) {
            *top_by_tf.entry(tf_of(&names[*i])).or_insert(0) += 1;
        }
        println!("top-{TOP_K} membership by TF when ranked on t: {top_by_tf:?}");
        for (i, t) in ts_all.iter().take(10) {
            println!(
                "  {:<50} t={:>9.3}  n_eff={}",
                names[*i], t, rows[*i].runs_min
            );
        }
        println!();
    }

    // ── how far the ranking moved ───────────────────────────────────────────
    let ranked_set = |variant: Variant| -> HashSet<usize> {
        let mut v: Vec<(usize, f64)> = (0..n_cols)
            .filter(|i| {
                let s = rows[*i].score(variant);
                s.rankable && s.score.is_finite()
            })
            .map(|i| (i, rows[i].score(variant).score))
            .collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        v.into_iter().take(TOP_K).map(|(i, _)| i).collect()
    };
    let legacy_set = ranked_set(Variant::LegacyPrefix);
    let new_set = ranked_set(Variant::NewCpcv);
    let overlap = new_set.intersection(&legacy_set).count();
    println!(
        "═══ top-{TOP_K} overlap between legacy_prefix and new_cpcv: {overlap} of {TOP_K} ═══\n"
    );

    // ═══ THE ASSERTIONS ══════════════════════════════════════════════════════
    //
    // Everything above is the measurement and is reported whatever it says.
    // These are the claims the measurement must support for the retraction to
    // stand, plus the sanity gates that stop a vacuous run reading as a result.

    // 1. THE RETRACTION. In the void number's own regime, with the operator's
    //    normalize_features=false, the legacy correlation scored EVERY
    //    higher-timeframe column exactly 0.0. Not "ranked them low" — scored
    //    them all identically zero, so the sort fell back to column index.
    let legacy_zero = &zero_by_variant[Variant::LegacyPrefix.label()];
    for tf in HIGHER_TFS {
        let offered_non_regime = by_tf[tf]
            .iter()
            .filter(|i| !names[**i].starts_with("regime_"))
            .count();
        let zeros = legacy_zero.get(tf).copied().unwrap_or(0);
        assert_eq!(
            zeros, offered_non_regime,
            "the legacy correlation scored {zeros} of {offered_non_regime} {tf} columns exactly \
             0.0. The void number requires ALL of them. Either the alignment no longer emits NaN \
             (check align_features_by_ns), or normalize_features is no longer being honoured \
             (this test pins it to {NORMALIZE_FEATURES}). Whichever it is, the historical figure \
             now has a DIFFERENT explanation than the one on record, and \
             docs/higher-timeframe-lane-2026-08-09.md must be re-written before anyone acts on it."
        );
    }

    // 2. THE REPAIR. The shipped correlation must not reproduce that fingerprint.
    let new_zero = &zero_by_variant[Variant::NewCpcv.label()];
    for tf in HIGHER_TFS {
        let offered_non_regime = by_tf[tf]
            .iter()
            .filter(|i| !names[**i].starts_with("regime_"))
            .count();
        let zeros = new_zero.get(tf).copied().unwrap_or(0);
        assert!(
            zeros < offered_non_regime,
            "the REPAIRED correlation still scores all {offered_non_regime} {tf} columns exactly \
             0.0. The prefilter is not ranking the {tf} lane, it is excluding it, and every \
             higher-timeframe keep-rate produced by this build is meaningless."
        );
    }

    // 3. THE OVERTURN IS REAL, NOT COSMETIC. Under the repair at least one
    //    higher-timeframe lane must earn keeps ON RANK — otherwise the repair
    //    changed the arithmetic without changing the outcome, and the standing
    //    belief survives.
    let new_rank_kept = &rank_kept_by_variant[Variant::NewCpcv.label()];
    let legacy_rank_kept = &rank_kept_by_variant[Variant::LegacyPrefix.label()];
    let higher_rank_now: usize = HIGHER_TFS
        .iter()
        .map(|tf| new_rank_kept.get(*tf).copied().unwrap_or(0))
        .sum();
    let higher_rank_before: usize = HIGHER_TFS
        .iter()
        .map(|tf| legacy_rank_kept.get(*tf).copied().unwrap_or(0))
        .sum();
    println!(
        "higher-TF columns kept ON RANK: {higher_rank_before} under the void regime, \
         {higher_rank_now} under the repair"
    );
    assert!(
        higher_rank_now > higher_rank_before,
        "the repaired correlation keeps {higher_rank_now} higher-timeframe columns on rank versus \
         {higher_rank_before} before — no improvement. Re-read the document before concluding \
         anything about higher timeframes from this run."
    );

    // 4. SANITY: the ranking must actually have moved. If the top-K is
    //    unchanged, the repair did not reach the selection and none of the
    //    numbers above describe a behaviour change.
    assert!(
        overlap < TOP_K,
        "the repaired top-{TOP_K} is IDENTICAL to the legacy top-{TOP_K}. The repair did not \
         reach the selection."
    );

    println!(
        "\nMeasured. Write any conclusion into docs/higher-timeframe-lane-2026-08-09.md rather \
         than into a comment — a number in a comment is what got retracted here."
    );
}
