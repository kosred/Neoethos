//! THROWAWAY MEASUREMENT PROBE #2 — effective sample size behind each
//! higher-timeframe correlation.
//!
//! `htf_prefilter_probe` answers "does the repaired correlation rank higher-TF
//! columns". This one answers the follow-up the operator must not be denied:
//! a higher-TF column is FORWARD-FILLED onto base bars, so one H4 observation
//! becomes 48 identical M5 rows. `pearson_pairwise_f32` counts those 48 rows as
//! 48 `used` samples. They are one observation.
//!
//! For every column, per CPCV fold, this reports:
//!   used  — the pairwise-complete row count the repaired function reports.
//!   runs  — contiguous blocks of the SAME value among those rows: the number of
//!           times the column actually changed, i.e. the independent-ish
//!           observations behind `used`.
//!   r_dense — |r| exactly as the prefilter computes it (row-wise).
//!   r_runs  — |r| recomputed one point per run: x = the run's constant value,
//!             y = the MEAN label over the run. This is the same statistic at
//!             the column's own resolution.
//! Scores are worst-over-folds and max-over-direction, matching the prefilter.
//!
//! Usage:
//!   cargo run -p neoethos-search --release --example htf_effective_n_probe -- \
//!       <data-root> --identity <d1...> [--higher H1,H4] [--max-bars N]

use anyhow::Context;
use neoethos_data::core::stats_f64::{pearson_pairwise, pearson_pairwise_f32};
use neoethos_data::{CanonicalDatasetIdentity, CanonicalTimeframe, Ohlcv};
use neoethos_search::data_selection::ExactCanonicalSeries;
use rayon::prelude::*;
use std::collections::BTreeMap;

const PREFILTER_MAX_REFIT_FOLDS: usize = 8;

#[derive(Debug, Clone)]
struct PrefilterSpec {
    insample_frac: f64,
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

fn tf_of(name: &str) -> String {
    timeframe_group(name)
        .map(|g| g.to_string())
        .unwrap_or_else(|| "base".to_string())
}

fn rolling_atr_f64(ohlcv: &Ohlcv, period: usize) -> Vec<f64> {
    let n = ohlcv.close.len();
    let period = period.max(1);
    let mut tr = vec![0.0f64; n];
    for i in 0..n {
        let hi = ohlcv.high[i];
        let lo = ohlcv.low[i];
        let prev_close = if i > 0 {
            ohlcv.close[i - 1]
        } else {
            ohlcv.close[i]
        };
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

/// `discovery::first_passage_labels`, verbatim (census dropped — probe #1
/// already printed it).
fn first_passage_labels(ohlcv: &Ohlcv, spec: &PrefilterSpec) -> (Vec<f32>, Vec<f32>) {
    let n = ohlcv.close.len();
    let mut long_labels = vec![f32::NAN; n];
    let mut short_labels = vec![f32::NAN; n];
    if n < 2 {
        return (long_labels, short_labels);
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
                    (true, true) => long_decided = true,
                    (true, false) => {
                        long_label = 1.0;
                        long_decided = true;
                    }
                    (false, true) => {
                        long_label = -1.0;
                        long_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if !short_decided {
                match (lo_ok && lo <= short_tp, hi_ok && hi >= short_sl) {
                    (true, true) => short_decided = true,
                    (true, false) => {
                        short_label = 1.0;
                        short_decided = true;
                    }
                    (false, true) => {
                        short_label = -1.0;
                        short_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if long_decided && short_decided {
                break;
            }
        }
        long_labels[i] = long_label;
        short_labels[i] = short_label;
    }
    (long_labels, short_labels)
}

/// `discovery::prefilter_fit_windows`, verbatim.
fn prefilter_fit_windows(n_rows: usize, spec: &PrefilterSpec) -> (Vec<Vec<usize>>, usize) {
    if let Some((n_splits, n_test_groups, embargo_pct, purge_pct, max_rows)) = spec.cpcv {
        let capped = if max_rows > 0 {
            max_rows.min(n_rows)
        } else {
            n_rows
        };
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

/// Collapse a fold's (x, y) rows into ONE POINT PER RUN of identical x.
///
/// A run ends when x changes or becomes non-finite. `y` is averaged over the
/// run's finite labels; a run with no finite label is dropped (and counted by
/// the caller through the returned length).
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
            // they are bit-identical, which is what a forward fill produces.
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

fn quantiles(mut v: Vec<f64>) -> Option<(f64, f64, f64, f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 { v[(((v.len() - 1) as f64) * p).round() as usize] };
    Some((v[0], q(0.10), q(0.25), q(0.50), q(0.90), v[v.len() - 1]))
}

fn flag(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn exact_identity(args: &[String]) -> anyhow::Result<CanonicalDatasetIdentity> {
    let encoded = flag(args, "--identity").context(
        "--identity <d1...> is required; display symbol/timeframe fields are not an identity",
    )?;
    CanonicalDatasetIdentity::from_path_component(&encoded)
        .with_context(|| format!("invalid canonical dataset identity {encoded:?}"))
}

fn direct_timeframes(csv: &str) -> anyhow::Result<Vec<CanonicalTimeframe>> {
    csv.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<CanonicalTimeframe>()
                .with_context(|| format!("unsupported direct canonical timeframe {value:?}"))
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = args
        .first()
        .cloned()
        .filter(|a| !a.starts_with("--"))
        .expect("usage: htf_effective_n_probe <data-root> --identity <d1...> ...");
    let identity = exact_identity(&args)?;
    let symbol = identity.symbol_name().to_owned();
    let base_tf = identity.timeframe();
    let higher_csv = flag(&args, "--higher").unwrap_or_else(|| "H1,H4".to_string());
    let top_k: usize = flag(&args, "--top-k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(240);
    let normalize: bool = flag(&args, "--normalize")
        .and_then(|v| v.parse().ok())
        .unwrap_or(false);
    let max_bars: usize = flag(&args, "--max-bars")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let spread_pips: f64 = flag(&args, "--spread-pips")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5);
    let pip: f64 = flag(&args, "--pip")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0001);

    let higher = direct_timeframes(&higher_csv)?;

    neoethos_data::install_data_runtime_overrides(normalize);

    println!(
        "exact series identity={} symbol={symbol} base={base_tf} direct_higher={higher:?}",
        identity.to_path_component()
    );
    let series = ExactCanonicalSeries::open(&root, identity)
        .context("opening the exact current canonical dataset generation")?;
    let input = series
        .load_search_input(&higher)
        .context("loading direct base/higher timeframes and building their feature cube")?;
    let full_ohlcv = input.base_frame().ohlcv();
    let full_features = input.features();
    let rows = full_ohlcv.close.len();
    let tail_start = if max_bars > 0 && rows > max_bars {
        rows - max_bars
    } else {
        0
    };
    let sliced_ohlcv =
        (tail_start > 0).then(|| neoethos_data::slice_ohlcv(full_ohlcv, tail_start, rows, None));
    let sliced_features = if tail_start > 0 {
        Some(full_features.row_slice(tail_start, rows)?)
    } else {
        None
    };
    let ohlcv = sliced_ohlcv.as_ref().unwrap_or(full_ohlcv);
    let features = sliced_features.as_ref().unwrap_or(full_features);
    let names = features.names.clone();
    let n_rows = features.n_samples();
    let n_cols = features.n_features();
    println!("cube rows={n_rows} cols={n_cols} normalize={normalize}");

    let spec = PrefilterSpec {
        insample_frac: 0.80,
        max_hold_bars: 35,
        atr_period: 14,
        sl_atr_mult: 2.5,
        rr: 2.75,
        round_trip_cost_px: spread_pips.max(0.0) * pip,
        cpcv: Some((8, 2, 0.01, 0.02, 200_000)),
    };
    let (long_labels, short_labels) = first_passage_labels(&ohlcv, &spec);
    let (windows, _) = prefilter_fit_windows(n_rows, &spec);
    println!(
        "folds={} rows/fold={:?}",
        windows.len(),
        windows.iter().map(|w| w.len()).collect::<Vec<_>>()
    );

    struct Row {
        r_dense: f64,
        r_runs: f64,
        dense_rankable: bool,
        runs_rankable: bool,
        used_min: usize,
        runs_min: usize,
    }

    let rows: Vec<Row> = (0..n_cols)
        .into_par_iter()
        .map(|col_idx| {
            // This probe intentionally narrows a copy to reproduce the retired
            // f32 correlation lane; the shared FeatureFrame remains f64.
            let col: Vec<f32> = features
                .feature_column(col_idx)
                .expect("feature projection succeeds")
                .values
                .iter()
                .map(|&value| value as f32)
                .collect();
            let mut dense_worst = f64::INFINITY;
            let mut runs_worst = f64::INFINITY;
            let mut dense_ok = true;
            let mut runs_ok = true;
            let mut used_min = usize::MAX;
            let mut runs_min = usize::MAX;
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
                let ol = pearson_pairwise_f32(&xs, &yl);
                let os = pearson_pairwise_f32(&xs, &ys);
                used_min = used_min.min(ol.used);
                let mut d: Option<f64> = if ol.is_rankable() {
                    Some(ol.abs())
                } else {
                    None
                };
                if os.is_rankable() {
                    let s = os.abs();
                    d = Some(d.map_or(s, |l: f64| l.max(s)));
                }
                match d {
                    Some(v) => dense_worst = dense_worst.min(v),
                    None => dense_ok = false,
                }

                let (rxl, ryl) = collapse_runs(&xs, &yl);
                let (rxs, rys) = collapse_runs(&xs, &ys);
                runs_min = runs_min.min(rxl.len());
                let rl = pearson_pairwise(&rxl, &ryl);
                let rs = pearson_pairwise(&rxs, &rys);
                let mut r: Option<f64> = if rl.is_rankable() {
                    Some(rl.abs())
                } else {
                    None
                };
                if rs.is_rankable() {
                    let s = rs.abs();
                    r = Some(r.map_or(s, |l: f64| l.max(s)));
                }
                match r {
                    Some(v) => runs_worst = runs_worst.min(v),
                    None => runs_ok = false,
                }
            }
            Row {
                r_dense: if dense_ok && dense_worst.is_finite() {
                    dense_worst
                } else {
                    f64::NEG_INFINITY
                },
                r_runs: if runs_ok && runs_worst.is_finite() {
                    runs_worst
                } else {
                    f64::NEG_INFINITY
                },
                dense_rankable: dense_ok && dense_worst.is_finite(),
                runs_rankable: runs_ok && runs_worst.is_finite(),
                used_min: if used_min == usize::MAX { 0 } else { used_min },
                runs_min: if runs_min == usize::MAX { 0 } else { runs_min },
            }
        })
        .collect();

    let mut by_tf: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, n) in names.iter().enumerate() {
        by_tf.entry(tf_of(n)).or_default().push(i);
    }

    println!("\n═══ EFFECTIVE SAMPLE SIZE — `used` rows vs distinct RUNS, worst fold ═══");
    println!(
        "{:<6} {:>6} {:>12} {:>12} {:>12} {:>12} {:>10}",
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
        let (_, up10, _, umed, _, _) = quantiles(used).unwrap();
        let (_, rp10, _, rmed, _, _) = quantiles(runs).unwrap();
        let (_, _, _, ratmed, _, _) = quantiles(ratio).unwrap();
        println!(
            "{:<6} {:>6} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>10.4}",
            tf,
            idxs.len(),
            up10,
            umed,
            rp10,
            rmed,
            ratmed
        );
    }

    // Score distribution at the two resolutions.
    for (label, pick) in [
        ("r_dense (what the prefilter ranks on)", 0usize),
        (
            "r_runs  (same statistic, one point per observation)",
            1usize,
        ),
    ] {
        println!("\n═══ SCORE DISTRIBUTION — {label} ═══");
        println!(
            "{:<6} {:>6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "TF", "cols", "unrank", "p10", "p25", "median", "p90", "max"
        );
        for (tf, idxs) in &by_tf {
            let vals: Vec<f64> = idxs
                .iter()
                .filter(|i| {
                    if pick == 0 {
                        rows[**i].dense_rankable
                    } else {
                        rows[**i].runs_rankable
                    }
                })
                .map(|i| {
                    if pick == 0 {
                        rows[*i].r_dense
                    } else {
                        rows[*i].r_runs
                    }
                })
                .collect();
            let unrank = idxs.len() - vals.len();
            match quantiles(vals) {
                Some((_, p10, p25, p50, p90, mx)) => println!(
                    "{:<6} {:>6} {:>7} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
                    tf,
                    idxs.len(),
                    unrank,
                    p10,
                    p25,
                    p50,
                    p90,
                    mx
                ),
                None => println!("{:<6} {:>6} {:>7}  (none rankable)", tf, idxs.len(), unrank),
            }
        }

        // Global top-K membership by TF under this statistic.
        let mut ranked: Vec<(usize, f64)> = (0..n_cols)
            .filter(|i| {
                if pick == 0 {
                    rows[*i].dense_rankable
                } else {
                    rows[*i].runs_rankable
                }
            })
            .filter(|i| !names[*i].starts_with("regime_"))
            .map(|i| {
                (
                    i,
                    if pick == 0 {
                        rows[i].r_dense
                    } else {
                        rows[i].r_runs
                    },
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut top_by_tf: BTreeMap<String, usize> = BTreeMap::new();
        for (i, _) in ranked.iter().take(top_k) {
            *top_by_tf.entry(tf_of(&names[*i])).or_insert(0) += 1;
        }
        println!("top-{top_k} membership by TF: {top_by_tf:?}");
        println!("top 15:");
        for (i, s) in ranked.iter().take(15) {
            println!(
                "  {:<48} {:>10.6}  used_min={:<8} runs_min={}",
                names[*i], s, rows[*i].used_min, rows[*i].runs_min
            );
        }
    }

    // ── EVIDENCE-WEIGHTED COMPARISON ─────────────────────────────────────────
    //
    // |r| alone cannot compare timeframes: a base column's |r| rests on ~138,000
    // independent bars, an H4 column's on ~2,800 (or, for a sparse event column,
    // on 15). The comparable quantity is the t statistic against the column's
    // OWN effective n, and the bar it must clear is a Bonferroni threshold over
    // the whole cube — because the prefilter is running 5,838 simultaneous
    // hypothesis tests and keeping the winners, which is precisely the setting
    // where an uncorrected max is meaningless.
    let z_bonf = {
        // Two-sided 5% family-wise over n_cols tests. Inverse normal via a
        // bisection on erfc — no new dependency, and 40 iterations is exact to
        // well past the digits printed.
        let target = 0.05 / (2.0 * n_cols as f64); // upper-tail probability
        let (mut lo, mut hi) = (0.0f64, 12.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            // upper tail = 0.5 * erfc(mid / sqrt(2)); erfc via continued
            // fraction is overkill — use the complementary error function from
            // statrs, already a dependency of this crate.
            let tail = 1.0 - statrs::function::erf::erf(mid / std::f64::consts::SQRT_2);
            let tail = 0.5 * tail;
            if tail > target { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    println!(
        "\n═══ EVIDENCE-WEIGHTED — t = |r|·sqrt(n_eff-2)/sqrt(1-r²), n_eff = distinct runs ═══"
    );
    println!(
        "Bonferroni |z| threshold for {n_cols} simultaneous tests at family-wise 5% = {z_bonf:.3}"
    );

    let t_of = |r: f64, n: usize| -> f64 {
        if !(r.is_finite()) || n < 3 || r.abs() >= 1.0 {
            return 0.0;
        }
        r.abs() * ((n - 2) as f64).sqrt() / (1.0 - r * r).sqrt()
    };

    for (label, pick) in [("r_dense", 0usize), ("r_runs", 1usize)] {
        println!("\n-- t from {label} --");
        println!(
            "{:<6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>14}",
            "TF", "cols", "t p50", "t p90", "t max", "n_eff p50", "t>=thresh", "share"
        );
        let mut ts_all: Vec<(usize, f64)> = Vec::new();
        for (tf, idxs) in &by_tf {
            let mut ts: Vec<f64> = Vec::new();
            let mut neff: Vec<f64> = Vec::new();
            let mut pass = 0usize;
            for i in idxs {
                let (r, ok) = if pick == 0 {
                    (rows[*i].r_dense, rows[*i].dense_rankable)
                } else {
                    (rows[*i].r_runs, rows[*i].runs_rankable)
                };
                if !ok || names[*i].starts_with("regime_") {
                    continue;
                }
                let t = t_of(r, rows[*i].runs_min);
                if t >= z_bonf {
                    pass += 1;
                }
                ts.push(t);
                neff.push(rows[*i].runs_min as f64);
                ts_all.push((*i, t));
            }
            let n = ts.len();
            match (quantiles(ts), quantiles(neff)) {
                (Some((_, _, _, p50, p90, mx)), Some((_, _, _, nmed, _, _))) => println!(
                    "{:<6} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.0} {:>12} {:>13.1}%",
                    tf,
                    n,
                    p50,
                    p90,
                    mx,
                    nmed,
                    pass,
                    100.0 * pass as f64 / n.max(1) as f64
                ),
                _ => println!("{tf:<6} (none)"),
            }
        }
        ts_all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut top_by_tf: BTreeMap<String, usize> = BTreeMap::new();
        for (i, _) in ts_all.iter().take(top_k) {
            *top_by_tf.entry(tf_of(&names[*i])).or_insert(0) += 1;
        }
        println!("top-{top_k} membership by TF when ranked on t: {top_by_tf:?}");
        println!("top 15 by t:");
        for (i, t) in ts_all.iter().take(15) {
            let r = if pick == 0 {
                rows[*i].r_dense
            } else {
                rows[*i].r_runs
            };
            println!(
                "  {:<48} t={:>9.3}  |r|={:>8.6}  n_eff={}",
                names[*i], t, r, rows[*i].runs_min
            );
        }
    }

    Ok(())
}
