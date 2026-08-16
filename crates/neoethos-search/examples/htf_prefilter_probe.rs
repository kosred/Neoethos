//! THROWAWAY MEASUREMENT PROBE — re-measure the higher-timeframe prefilter lane
//! with the REPAIRED correlation (`stats_f64::pearson_pairwise_f32`).
//!
//! Usage:
//!   cargo run -p neoethos-search --release --example htf_prefilter_probe -- \
//!       <data-root> [--symbol EURUSD] [--base M5] [--higher H1,H4] \
//!       [--top-k 240] [--min-per-tf 6] [--normalize true|false] [--max-bars N]
//!
//! It REPLICATES `discovery::prefilter_features` verbatim (that function is
//! private, and the mechanism attribution this measurement needs does not exist
//! in it), and scores every column FOUR ways over the SAME rows:
//!
//!   new_cpcv     — today's pipeline: triple-barrier label, both directions,
//!                  worst-of-8-CPCV-folds, `pearson_pairwise_f32` (f64 two-pass,
//!                  pairwise-complete).
//!   legacy_cpcv  — identical labels and identical fold windows, but the OLD
//!                  f32 single-pass correlation with the `!den.is_finite() -> 0.0`
//!                  guard. Isolates the correlation function ALONE.
//!   new_prefix   — repaired correlation against the OLD target (1-bar forward
//!                  return) over the OLD single 80% leading prefix.
//!   legacy_prefix— the FULL pre-repair regime. This is the configuration the
//!                  historical "base 217/217, H1 40/217, H4 8/217" was measured in.
//!
//! For each of the four it runs the real selection (global top-K including the
//! `regime_` INFINITY exemption, then the per-timeframe quota, then the
//! seed-template force-keep) and attributes every kept column to the FIRST
//! mechanism that would have kept it.

use neoethos_data::core::stats_f64::pearson_pairwise_f32;
use neoethos_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

// ── copied verbatim from discovery.rs (private there) ────────────────────────

const PREFILTER_MAX_REFIT_FOLDS: usize = 8;
const MIN_DECIDED_FIRST_PASSAGE_LABELS: usize = 100;

#[derive(Debug, Clone)]
struct PrefilterSpec {
    top_k: usize,
    insample_frac: f64,
    min_per_tf: usize,
    max_hold_bars: usize,
    atr_period: usize,
    sl_atr_mult: f64,
    rr: f64,
    round_trip_cost_px: f64,
    cpcv: Option<(usize, usize, f64, f64, usize)>,
}

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

/// `discovery::rolling_atr_f64`, verbatim.
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
    // Rolling mean over the finite entries. The discovery version is an O(n*p)
    // double loop; this is the same arithmetic with a running sum so a 750k-bar
    // series does not spend minutes here. Verified identical below by
    // construction: same window `[i+1-period, i]`, same finite filter.
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
    label_up: usize,
    label_down: usize,
    label_vertical: usize,
    label_ambiguous: usize,
    label_short_win: usize,
    label_short_loss: usize,
    label_vertical_short: usize,
    label_ambiguous_short: usize,
    label_undefined: usize,
}

/// `discovery::first_passage_labels`, verbatim.
fn first_passage_labels(ohlcv: &Ohlcv, spec: &PrefilterSpec) -> (Vec<f32>, Vec<f32>, LabelCensus) {
    let n = ohlcv.close.len();
    let mut long_labels = vec![f32::NAN; n];
    let mut short_labels = vec![f32::NAN; n];
    let mut census = LabelCensus::default();
    if n < 2 {
        census.label_undefined = n;
        return (long_labels, short_labels, census);
    }
    let atr = rolling_atr_f64(ohlcv, spec.atr_period);
    let hold = spec.max_hold_bars.max(1);
    let sl_mult = if spec.sl_atr_mult.is_finite() && spec.sl_atr_mult > 0.0 {
        spec.sl_atr_mult
    } else {
        1.0
    };
    let rr = if spec.rr.is_finite() && spec.rr > 0.0 {
        spec.rr
    } else {
        2.0
    };
    let cost = if spec.round_trip_cost_px.is_finite() && spec.round_trip_cost_px >= 0.0 {
        spec.round_trip_cost_px
    } else {
        0.0
    };

    for i in 0..n {
        let entry = ohlcv.close[i];
        let a = atr[i];
        if !entry.is_finite() || !a.is_finite() || a <= 0.0 || i + 1 >= n {
            census.label_undefined += 1;
            continue;
        }
        let stop_distance = sl_mult * a;
        let take_distance = rr * stop_distance;
        let long_tp = entry + take_distance + cost;
        let long_sl = entry - stop_distance + cost;
        let short_tp = entry - take_distance - cost;
        let short_sl = entry + stop_distance - cost;
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
                        census.label_ambiguous += 1;
                        long_decided = true;
                    }
                    (true, false) => {
                        long_label = 1.0;
                        census.label_up += 1;
                        long_decided = true;
                    }
                    (false, true) => {
                        long_label = -1.0;
                        census.label_down += 1;
                        long_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if !short_decided {
                match (lo_ok && lo <= short_tp, hi_ok && hi >= short_sl) {
                    (true, true) => {
                        census.label_ambiguous_short += 1;
                        short_decided = true;
                    }
                    (true, false) => {
                        short_label = 1.0;
                        census.label_short_win += 1;
                        short_decided = true;
                    }
                    (false, true) => {
                        short_label = -1.0;
                        census.label_short_loss += 1;
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
            census.label_vertical += 1;
        }
        if !short_decided {
            census.label_vertical_short += 1;
        }
        long_labels[i] = long_label;
        short_labels[i] = short_label;
    }
    (long_labels, short_labels, census)
}

/// `discovery::prefilter_fit_windows`, verbatim.
fn prefilter_fit_windows(n_rows: usize, spec: &PrefilterSpec) -> (Vec<Vec<usize>>, usize) {
    if let Some((n_splits, n_test_groups, embargo_pct, purge_pct, max_rows)) = spec.cpcv {
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
    }
    let train_end = ((n_rows as f64) * spec.insample_frac).floor() as usize;
    let train_end = train_end.clamp(2, n_rows.saturating_sub(1)).max(2);
    (vec![(0..train_end.saturating_sub(1)).collect()], 0)
}

/// The PRE-REPAIR statistic, verbatim from the doc comment in
/// `neoethos-data/src/core/stats_f64.rs` (which quotes discovery.rs:3835 as it
/// stood before 2026-08-09). Single pass, f32 accumulators, `n` as f32, and the
/// `!den.is_finite() -> 0.0` guard that scored every NaN-carrying column zero.
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

// ── the four score variants ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Variant {
    NewCpcv,
    LegacyCpcv,
    NewPrefix,
    LegacyPrefix,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::NewCpcv => "new_cpcv      (repaired corr | triple-barrier | 8 CPCV folds)",
            Variant::LegacyCpcv => "legacy_cpcv   (OLD f32 corr | triple-barrier | 8 CPCV folds)",
            Variant::NewPrefix => "new_prefix    (repaired corr | 1-bar fwd ret | 80% prefix)",
            Variant::LegacyPrefix => "legacy_prefix (OLD f32 corr | 1-bar fwd ret | 80% prefix) <- the historical regime",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ColumnScore {
    score: f64,
    rankable: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct RowEvidence {
    /// Pairwise-complete rows the repaired function actually used, summed over
    /// the CPCV folds.
    used_total: u64,
    /// Rows it skipped because the feature or the label was non-finite.
    skipped_total: u64,
    /// Rows offered to the correlation across the folds.
    offered_total: u64,
    /// Minimum `used` over the folds — the weakest fold is what the worst-fold
    /// rule ranks on.
    used_min: u64,
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
fn select_with_mechanisms(
    names: &[String],
    scores: &[ColumnScore],
    spec: &PrefilterSpec,
) -> HashMap<usize, Mechanism> {
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

    let actual_top_k = (spec.top_k + regime_forced).min(n_cols);

    let mut kept: HashMap<usize, Mechanism> = HashMap::new();
    for (idx, _) in correlations.iter().take(actual_top_k) {
        let m = if names[*idx].starts_with("regime_") {
            Mechanism::RegimeExemption
        } else {
            Mechanism::Rank
        };
        kept.insert(*idx, m);
    }

    if spec.min_per_tf > 0 {
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
            if *count >= spec.min_per_tf {
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

// ── reporting helpers ────────────────────────────────────────────────────────

fn quantiles(mut v: Vec<f64>) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 {
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        v[idx]
    };
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    Some((v[0], q(0.10), q(0.25), q(0.50), q(0.75), q(0.90), v[v.len() - 1], mean))
}

fn tf_of(name: &str) -> String {
    timeframe_group(name)
        .map(|g| g.to_string())
        .unwrap_or_else(|| "base".to_string())
}

fn flag(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = args
        .first()
        .cloned()
        .filter(|a| !a.starts_with("--"))
        .expect("usage: htf_prefilter_probe <data-root> [--symbol EURUSD] ...");
    let symbol = flag(&args, "--symbol").unwrap_or_else(|| "EURUSD".to_string());
    let base_tf = flag(&args, "--base").unwrap_or_else(|| "M5".to_string());
    let higher_csv = flag(&args, "--higher").unwrap_or_else(|| "H1,H4".to_string());
    let top_k: usize = flag(&args, "--top-k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(240);
    let min_per_tf: usize = flag(&args, "--min-per-tf")
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let normalize: bool = flag(&args, "--normalize")
        .and_then(|v| v.parse().ok())
        .unwrap_or(true);
    let max_bars: usize = flag(&args, "--max-bars")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let spread_pips: f64 = flag(&args, "--spread-pips")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5);
    let pip: f64 = flag(&args, "--pip").and_then(|v| v.parse().ok()).unwrap_or(0.0001);

    let higher: Vec<String> = higher_csv
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher.iter().map(|s| s.as_str()).collect();

    // `normalize_features` is the operator's shipped `models.data_runtime`
    // setting. It DECIDES whether the higher-TF alignment NaN survives to the
    // prefilter or is turned into 0.0 upstream, so it is the single most
    // important switch in this measurement and it is printed, not assumed.
    neoethos_data::install_data_runtime_overrides(normalize, false);

    println!("═══ SETUP ═══");
    println!("root            {root}");
    println!("symbol          {symbol}   base {base_tf}   higher {higher:?}");
    println!("prefilter       top_k={top_k}  min_per_tf={min_per_tf}");
    println!("normalize_feats {normalize}   (true = NaN cells become 0.0 BEFORE the prefilter)");

    let t0 = std::time::Instant::now();
    let mut tfs: Vec<&str> = vec![base_tf.as_str()];
    tfs.extend(higher_refs.iter().copied());
    let mut dataset = neoethos_data::load_symbol_dataset_with_timeframes(&root, &symbol, &tfs)?;
    for (tf, o) in &dataset.frames {
        println!("loaded          {tf:<4} rows={}", o.close.len());
    }

    if max_bars > 0 {
        // Truncate the BASE only, keeping the tail. The higher frames stay whole
        // so the alignment still has bars to bind to.
        if let Some(o) = dataset.frames.get_mut(base_tf.as_str()) {
            let n = o.close.len();
            if n > max_bars {
                let start = n - max_bars;
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
                println!("TRUNCATED base to the last {max_bars} bars (--max-bars)");
            }
        }
    }

    let ohlcv = dataset
        .frames
        .get(base_tf.as_str())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("base timeframe {base_tf} missing"))?;

    let features: FeatureFrame =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs)?;
    println!(
        "feature cube    rows={} cols={}  built in {:.1}s",
        features.n_samples(),
        features.n_features(),
        t0.elapsed().as_secs_f64()
    );

    let names = features.names.clone();
    let n_rows = features.n_samples();
    let n_cols = features.n_features();

    let mut offered_by_tf: BTreeMap<String, usize> = BTreeMap::new();
    for name in &names {
        *offered_by_tf.entry(tf_of(name)).or_insert(0) += 1;
    }
    println!("offered by TF   {offered_by_tf:?}");

    // ── the spec, assembled the way discovery.rs assembles it ────────────────
    // Gene stop band -> label geometry. discovery.rs takes the MIDPOINT of the
    // ATR-scaled band: sl_mult = mid(sl_min_atr, sl_max_atr), rr = mid(rr_min,
    // rr_max). config.yaml ships 1.0/4.0 and 1.5/4.0.
    let atr_pips = neoethos_search::stop_target::median_atr_pips(
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        pip,
        14,
    );
    let (sl_atr_mult, rr) = (2.5f64, 2.75f64);
    let round_trip_cost_px = spread_pips.max(0.0) * pip;
    let spec = PrefilterSpec {
        top_k,
        insample_frac: 0.80,
        min_per_tf,
        max_hold_bars: 35,
        atr_period: 14,
        sl_atr_mult,
        rr,
        round_trip_cost_px,
        cpcv: Some((8, 2, 0.01, 0.02, 200_000)),
    };
    println!(
        "label geometry  sl={sl_atr_mult} ATR  rr={rr}  cost_px={round_trip_cost_px:.6}  \
         hold=35  median_atr_pips={atr_pips:?}"
    );

    // ── labels ───────────────────────────────────────────────────────────────
    let t1 = std::time::Instant::now();
    let (long_labels, short_labels, lc) = first_passage_labels(&ohlcv, &spec);
    let decided_long = lc.label_up + lc.label_down;
    let decided_short = lc.label_short_win + lc.label_short_loss;
    println!(
        "labels          long up={} down={} vert={} amb={} | short win={} loss={} vert={} amb={} | undefined={} ({:.1}s)",
        lc.label_up,
        lc.label_down,
        lc.label_vertical,
        lc.label_ambiguous,
        lc.label_short_win,
        lc.label_short_loss,
        lc.label_vertical_short,
        lc.label_ambiguous_short,
        lc.label_undefined,
        t1.elapsed().as_secs_f64()
    );
    let fell_back = decided_long.max(decided_short) < MIN_DECIDED_FIRST_PASSAGE_LABELS;
    if fell_back {
        println!("!! label degenerate — discovery would fall back to the 1-bar forward return");
    }

    // 1-bar forward return — the OLD target.
    let mut fwd = vec![f32::NAN; ohlcv.close.len()];
    for i in 0..ohlcv.close.len().saturating_sub(1) {
        let d = ohlcv.close[i];
        if d.abs() > 1e-12 {
            fwd[i] = ((ohlcv.close[i + 1] - d) / d) as f32;
        }
    }

    // ── fit windows ──────────────────────────────────────────────────────────
    let (windows, folds_available) = prefilter_fit_windows(n_rows, &spec);
    let prefix_spec = PrefilterSpec {
        cpcv: None,
        ..spec.clone()
    };
    let (prefix_windows, _) = prefilter_fit_windows(n_rows, &prefix_spec);
    println!(
        "cpcv windows    {} of {} folds; rows per fold {:?}",
        windows.len(),
        folds_available,
        windows.iter().map(|w| w.len()).collect::<Vec<_>>()
    );
    println!(
        "prefix window   1 window, {} rows (leading {:.0}%)",
        prefix_windows[0].len(),
        spec.insample_frac * 100.0
    );

    // ── score every column, four ways, in one traversal ──────────────────────
    let t2 = std::time::Instant::now();
    struct Row {
        new_cpcv: ColumnScore,
        legacy_cpcv: ColumnScore,
        new_prefix: ColumnScore,
        legacy_prefix: ColumnScore,
        evidence: RowEvidence,
        /// Rows the repaired function used in the single 80% prefix window.
        prefix_used: u64,
    }

    let rows: Vec<Row> = (0..n_cols)
        .into_par_iter()
        .map(|col_idx| {
            let name = &names[col_idx];
            let is_regime = name.starts_with("regime_");
            let col: Vec<f32> = features.feature_column(col_idx).iter().copied().collect();

            let mut ev = RowEvidence {
                used_min: u64::MAX,
                ..RowEvidence::default()
            };

            // The regime exemption is applied by discovery BEFORE any
            // correlation is computed, so it short-circuits identically here —
            // but the row evidence is still collected so the report can say how
            // many rows a regime column would have had.
            let mut new_worst = f64::INFINITY;
            let mut legacy_worst = f64::INFINITY;
            let mut new_rankable = true;

            for window in &windows {
                let mut xs: Vec<f32> = Vec::with_capacity(window.len());
                let mut ys: Vec<f32> = Vec::with_capacity(window.len());
                let mut ys_short: Vec<f32> = Vec::with_capacity(window.len());
                for &row in window {
                    if row >= n_rows || row >= long_labels.len() {
                        continue;
                    }
                    xs.push(col[row]);
                    ys.push(long_labels[row]);
                    ys_short.push(short_labels.get(row).copied().unwrap_or(f32::NAN));
                }
                let o_long = pearson_pairwise_f32(&xs, &ys);
                let o_short = pearson_pairwise_f32(&xs, &ys_short);
                ev.offered_total += xs.len() as u64;
                ev.used_total += o_long.used as u64;
                ev.skipped_total += o_long.skipped as u64;
                ev.used_min = ev.used_min.min(o_long.used as u64);

                let mut a: Option<f64> = if o_long.is_rankable() {
                    Some(o_long.abs())
                } else {
                    None
                };
                if o_short.is_rankable() {
                    let s = o_short.abs();
                    a = Some(a.map_or(s, |l: f64| l.max(s)));
                }
                match a {
                    Some(a) => new_worst = new_worst.min(a),
                    None => {
                        new_rankable = false;
                    }
                }

                let l = (legacy_pearson_f32(&xs, &ys) as f64)
                    .abs()
                    .max((legacy_pearson_f32(&xs, &ys_short) as f64).abs());
                legacy_worst = legacy_worst.min(l);
            }
            if !new_rankable || !new_worst.is_finite() {
                new_worst = f64::NEG_INFINITY;
            }
            if !legacy_worst.is_finite() {
                legacy_worst = 0.0;
            }
            if ev.used_min == u64::MAX {
                ev.used_min = 0;
            }

            // Old regime: 1-bar forward return over the single leading prefix.
            let w = &prefix_windows[0];
            let mut xs: Vec<f32> = Vec::with_capacity(w.len());
            let mut ys: Vec<f32> = Vec::with_capacity(w.len());
            for &row in w {
                if row >= n_rows || row >= fwd.len() {
                    continue;
                }
                xs.push(col[row]);
                ys.push(fwd[row]);
            }
            let o = pearson_pairwise_f32(&xs, &ys);
            let legacy_prefix_score = (legacy_pearson_f32(&xs, &ys) as f64).abs();

            Row {
                new_cpcv: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { new_worst },
                    rankable: is_regime || new_rankable,
                },
                legacy_cpcv: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { legacy_worst },
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
                legacy_prefix: ColumnScore {
                    score: if is_regime { f64::INFINITY } else { legacy_prefix_score },
                    rankable: true,
                },
                evidence: ev,
                prefix_used: o.used as u64,
            }
        })
        .collect();
    println!("scored          {n_cols} columns x 4 variants in {:.1}s\n", t2.elapsed().as_secs_f64());

    // ── ROW EVIDENCE per timeframe ───────────────────────────────────────────
    println!("═══ ROW EVIDENCE — how many rows each timeframe actually contributes ═══");
    println!(
        "{:<6} {:>7} {:>14} {:>14} {:>9} {:>14} {:>12}",
        "TF", "cols", "used/fold avg", "skipped/fold", "used %", "worst-fold min", "prefix used"
    );
    let mut by_tf_rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, n) in names.iter().enumerate() {
        by_tf_rows.entry(tf_of(n)).or_default().push(i);
    }
    let nw = windows.len().max(1) as f64;
    for (tf, idxs) in &by_tf_rows {
        let used: f64 = idxs.iter().map(|i| rows[*i].evidence.used_total as f64).sum::<f64>()
            / idxs.len() as f64
            / nw;
        let skipped: f64 = idxs
            .iter()
            .map(|i| rows[*i].evidence.skipped_total as f64)
            .sum::<f64>()
            / idxs.len() as f64
            / nw;
        let offered: f64 = idxs
            .iter()
            .map(|i| rows[*i].evidence.offered_total as f64)
            .sum::<f64>()
            / idxs.len() as f64
            / nw;
        let used_min: u64 = idxs.iter().map(|i| rows[*i].evidence.used_min).min().unwrap_or(0);
        let prefix_used: f64 =
            idxs.iter().map(|i| rows[*i].prefix_used as f64).sum::<f64>() / idxs.len() as f64;
        println!(
            "{:<6} {:>7} {:>14.0} {:>14.0} {:>8.2}% {:>14} {:>12.0}",
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

    // ── per variant: distribution + selection + mechanism ────────────────────
    for variant in [
        Variant::LegacyPrefix,
        Variant::NewPrefix,
        Variant::LegacyCpcv,
        Variant::NewCpcv,
    ] {
        let scores: Vec<ColumnScore> = rows
            .iter()
            .map(|r| match variant {
                Variant::NewCpcv => r.new_cpcv,
                Variant::LegacyCpcv => r.legacy_cpcv,
                Variant::NewPrefix => r.new_prefix,
                Variant::LegacyPrefix => r.legacy_prefix,
            })
            .collect();

        println!("═══ VARIANT: {} ═══", variant.label());

        // Distribution per TF over FINITE, rankable, non-regime scores.
        println!(
            "{:<6} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "TF", "cols", "unrank", "min", "p10", "p25", "median", "p75", "p90", "max"
        );
        for (tf, idxs) in &by_tf_rows {
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
                Some((mn, p10, p25, p50, p75, p90, mx, _mean)) => println!(
                    "{:<6} {:>6} {:>7} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6} {:>9.6}",
                    tf,
                    idxs.len(),
                    unrank,
                    mn,
                    p10,
                    p25,
                    p50,
                    p75,
                    p90,
                    mx
                ),
                None => println!("{:<6} {:>6} {:>7}   (no rankable columns)", tf, idxs.len(), unrank),
            }
        }

        // How many columns score EXACTLY 0.0 — the legacy bug's signature.
        let mut zero_by_tf: BTreeMap<&String, usize> = BTreeMap::new();
        for (tf, idxs) in &by_tf_rows {
            let z = idxs
                .iter()
                .filter(|i| !names[**i].starts_with("regime_"))
                .filter(|i| scores[**i].score == 0.0)
                .count();
            zero_by_tf.insert(tf, z);
        }
        println!("exactly-0.0     {zero_by_tf:?}");

        // Where do the higher-TF columns sit in the global ranking?
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
        for (tf, idxs) in &by_tf_rows {
            let mut r: Vec<usize> = idxs.iter().filter_map(|i| rank_of.get(i).copied()).collect();
            if r.is_empty() {
                continue;
            }
            r.sort_unstable();
            median_rank.insert(tf.clone(), (r[r.len() / 2], r[0]));
        }
        println!("(median rank, best rank) by TF among {} rankable: {median_rank:?}", ranked.len());

        // Selection + mechanism attribution.
        let kept = select_with_mechanisms(&names, &scores, &spec);
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
        for (tf, idxs) in &by_tf_rows {
            let m = per_tf.get(tf).cloned().unwrap_or_default();
            let k: usize = m.values().sum();
            total_kept += k;
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
        println!("TOTAL kept {total_kept} of {n_cols}");

        // Name the higher-TF survivors that earned their place ON RANK, with
        // their score and their global rank — the claim "none of the 8 earned
        // it" is exactly this list being empty.
        for (tf, _) in &by_tf_rows {
            if tf == "base" {
                continue;
            }
            let mut earned: Vec<(usize, f64, usize)> = kept
                .iter()
                .filter(|(i, m)| **m == Mechanism::Rank && tf_of(&names[**i]) == *tf)
                .map(|(i, _)| (*i, scores[*i].score, rank_of.get(i).copied().unwrap_or(usize::MAX)))
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

        // HOW MANY ROWS DID THE SURVIVORS EARN THEIR RANK ON? A |r| computed
        // over 700 of 138,000 rows outranking one computed over all of them is
        // not a better feature, it is a smaller sample. Reported per TF for the
        // columns kept ON RANK, using the worst fold's `used` count — the fold
        // the worst-fold rule actually ranks by.
        let fold_rows = windows.iter().map(|w| w.len()).min().unwrap_or(1) as f64;
        println!(
            "  row evidence of the RANK survivors (worst-fold `used`, smallest fold = {fold_rows:.0} rows):"
        );
        for (tf, _) in &by_tf_rows {
            let mins: Vec<f64> = kept
                .iter()
                .filter(|(i, m)| **m == Mechanism::Rank && tf_of(&names[**i]) == *tf)
                .map(|(i, _)| rows[*i].evidence.used_min as f64)
                .collect();
            if mins.is_empty() {
                println!("    {tf:<6} (none kept on rank)");
                continue;
            }
            let below_50pct = mins.iter().filter(|v| **v < 0.50 * fold_rows).count();
            let below_5pct = mins.iter().filter(|v| **v < 0.05 * fold_rows).count();
            let below_1pct = mins.iter().filter(|v| **v < 0.01 * fold_rows).count();
            match quantiles(mins.clone()) {
                Some((mn, p10, p25, p50, _p75, _p90, mx, _mean)) => println!(
                    "    {tf:<6} n={:<4} used_min min={mn:.0} p10={p10:.0} p25={p25:.0} \
                     median={p50:.0} max={mx:.0} | <50% of fold: {below_50pct}, <5%: \
                     {below_5pct}, <1%: {below_1pct}",
                    mins.len()
                ),
                None => {}
            }
        }
        println!();
    }

    // ── the overall top of the repaired ranking ──────────────────────────────
    let mut top: Vec<(usize, f64)> = (0..n_cols)
        .filter(|i| rows[*i].new_cpcv.rankable && rows[*i].new_cpcv.score.is_finite())
        .map(|i| (i, rows[i].new_cpcv.score))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("═══ TOP 30 of the repaired (new_cpcv) ranking ═══");
    for (i, s) in top.iter().take(30) {
        println!(
            "  {:<40} {:>10.6}  used_min={}",
            names[*i], s, rows[*i].evidence.used_min
        );
    }

    // Rank-agreement between the old and the repaired ranking, so "the ranking
    // moved" is a number.
    let mut new_set: HashSet<usize> = HashSet::new();
    for (i, _) in top.iter().take(top_k) {
        new_set.insert(*i);
    }
    let mut legacy_top: Vec<(usize, f64)> = (0..n_cols)
        .map(|i| (i, rows[i].legacy_prefix.score))
        .filter(|(_, s)| s.is_finite())
        .collect();
    legacy_top.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let legacy_set: HashSet<usize> = legacy_top.iter().take(top_k).map(|(i, _)| *i).collect();
    println!(
        "\ntop-{top_k} overlap between legacy_prefix and new_cpcv: {} of {top_k}",
        new_set.intersection(&legacy_set).count()
    );

    Ok(())
}
