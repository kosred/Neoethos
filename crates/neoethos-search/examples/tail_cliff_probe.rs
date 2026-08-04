//! Measures the 300 000-bar expected-shortfall cliff on REAL bars.
//!
//! Usage:
//!   cargo run -p neoethos-search --release --example tail_cliff_probe -- <vortex-file> [pip_size]
//!
//! Prints the median adaptive base stop (pips) for a sweep of series lengths,
//! under BOTH the old hardcoded `tail_max_bars = 300_000` and the new default
//! of no cap, so the discontinuity is two columns of numbers rather than a
//! claim. The lengths are the ones the three production callers actually pass:
//! ~1 000 (live `warmup_bars`), ~70 000 (a walk-forward window at
//! `walkforward_splits = 15`), and the full series (discovery scoring).

use neoethos_search::{StopTargetSettings, compute_stop_distance_series};

fn median(v: &[f64]) -> f64 {
    let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if w.is_empty() {
        return f64::NAN;
    }
    w[w.len() / 2]
}

/// Largest relative bar-for-bar difference between a base series computed over
/// the trailing `slice.len()` bars and the same bars inside the FULL-series
/// base. `start` is where the slice begins in the full series. Skips the
/// slice's own warmup (the vol/tail windows), which genuinely cannot match.
///
/// ~0 means the two spans agree bar-for-bar and any difference in their MEDIANS
/// is only which bars each one averages — not a divergence.
fn max_rel_vs_full(slice: &[f64], full: &[f64], start: usize) -> f64 {
    let warmup = 200usize;
    let mut worst = 0.0_f64;
    for i in warmup..slice.len() {
        let (a, b) = (slice[i], full[start + i]);
        let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-12);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

/// Exactly what `adaptive_base_pips_series` does, with the cap made explicit.
fn base_pips(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    pip: f64,
    tail_max_bars: usize,
) -> Option<(Vec<f64>, f64)> {
    base_pips_step(high, low, close, pip, tail_max_bars, 5)
}

fn base_pips_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    pip: f64,
    tail_max_bars: usize,
    tail_step: usize,
) -> Option<(Vec<f64>, f64)> {
    let mut settings = StopTargetSettings::default();
    settings.vol_estimator = "parkinson".to_string();
    settings.tail_max_bars = tail_max_bars;
    settings.tail_step = tail_step;
    let t0 = std::time::Instant::now();
    let dist = compute_stop_distance_series(close, high, low, close, &settings).ok()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    Some((dist.iter().map(|d| (d / pip).max(1e-9)).collect(), ms))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: tail_cliff_probe <vortex-file> [pip_size]");
    let pip: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0001);

    let o = neoethos_data::load_vortex(&path)?;
    println!("file      : {path}");
    println!("bars      : {}", o.close.len());
    println!("pip_size  : {pip}");
    println!();
    // The full-series base, to answer the question the median column raises:
    // once the cliff is gone, do two spans still DISAGREE bar-for-bar, or do
    // their medians differ only because they average different bars?
    let full = base_pips(&o.high, &o.low, &o.close, pip, 0)
        .map(|(v, _)| v)
        .expect("full-series base builds with no cap");

    // Same thing with the expected-shortfall subsampling turned OFF
    // (`tail_step = 1`). The step-5 grid is anchored at the series START, so
    // two callers whose slices differ by an offset that is not a multiple of 5
    // sample DIFFERENT bars and carry different values forward. step = 1 is
    // position-invariant by construction, so these columns price the fix.
    let (full_s1, full_s1_ms) = base_pips_step(&o.high, &o.low, &o.close, pip, 0, 1)
        .expect("full-series base builds at step 1");

    println!(
        "{:>10}  {:>16}  {:>16}  {:>8}  {:>9}  {:>13}  {:>13}  {:>13}",
        "n_bars",
        "cap=300k (old)",
        "cap=0 (new)",
        "ratio",
        "ms(new)",
        "max_rel_vs_full",
        "step1_median",
        "step1_max_rel"
    );

    for n in [
        1_000usize, // live_trading.rs default warmup_bars
        5_000,
        70_000, // ~ one walk-forward window at walkforward_splits = 15
        250_000,
        299_000,
        300_000,
        300_001,
        301_000,
        400_000,
        o.close.len(),
    ] {
        if n > o.close.len() || n < 200 {
            continue;
        }
        let s = o.close.len() - n;
        let (h, l, c) = (&o.high[s..], &o.low[s..], &o.close[s..]);
        let old = base_pips(h, l, c, pip, 300_000).map(|(v, _)| median(&v));
        let new = base_pips(h, l, c, pip, 0);
        let s1 = base_pips_step(h, l, c, pip, 0, 1).map(|(v, _)| v);
        let (s1_med, s1_rel) = match &s1 {
            Some(v) => (median(v), max_rel_vs_full(v, &full_s1, s)),
            None => (f64::NAN, f64::NAN),
        };
        match (old, new) {
            (Some(a), Some((b, ms))) => {
                let mb = median(&b);
                println!(
                    "{n:>10}  {a:>16.4}  {mb:>16.4}  {:>8.2}  {ms:>9.1}  {:>13.2e}  {s1_med:>13.4}  {s1_rel:>13.2e}",
                    mb / a,
                    max_rel_vs_full(&b, &full, s)
                );
            }
            (None, Some((b, ms))) => println!(
                "{n:>10}  {:>16}  {:>16.4}  {:>8}  {ms:>9.1}  {:>13.2e}  {s1_med:>13.4}  {s1_rel:>13.2e}",
                "REFUSED",
                median(&b),
                "-",
                max_rel_vs_full(&b, &full, s)
            ),
            _ => println!("{n:>10}  {:>16}", "n/a"),
        }
    }
    println!();
    println!(
        "full series at tail_step = 1: {:.1} ms (vs {:.1} ms at step 5) — the cost of \
         making the tail term position-invariant, paid ONCE per combo.",
        full_s1_ms,
        base_pips(&o.high, &o.low, &o.close, pip, 0)
            .map(|(_, ms)| ms)
            .unwrap_or(f64::NAN)
    );
    Ok(())
}
