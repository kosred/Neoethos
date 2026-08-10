//! SLICE-2 MEASUREMENT (2026-08-08): what changes when every selection-bearing
//! serial stage evaluates the SAME stop regime GA scoring scored.
//!
//! Before this slice, 9 of 13 `discovery_backtest_settings` call sites —
//! including the quality screen's base backtest, the prop-firm window gate,
//! the forward-test/prop-firm tails and the walk-forward risk diagnostics —
//! ran adaptive genes on their unused FIXED `sl_pips`/`tp_pips`, while GA
//! scoring (and the population validation helpers) ran
//! `sl = stop_vol_mult × base[i]`. This example holds the SIGNALS constant
//! (a deterministic SMA-cross grid — the stand-in for a candidate pool) and
//! varies ONLY the stop regime, so every difference reported is exactly the
//! selection consequence of the fix, nothing else.
//!
//! It also measures the walk-forward day-bucket fix: the risk diagnostics
//! bucketed trades by epoch-ms exit time into a YYYYMMDD-keyed map, so every
//! trade was its own "day" and same-day losses never accumulated into a
//! daily-loss breach.
//!
//! Usage:
//!   cargo run -p neoethos-search --release --example stage_stop_agreement \
//!       -- <vortex-file> [pip_size]

use neoethos_search::{BacktestSettings, adaptive_base_pips_series, simulate_trades_core};
use std::collections::BTreeMap;

fn sma_cross_signals(close: &[f64], fast: usize, slow: usize) -> Vec<i8> {
    let sma = |w: usize| -> Vec<f64> {
        let mut out = vec![f64::NAN; close.len()];
        let mut sum = 0.0;
        for i in 0..close.len() {
            sum += close[i];
            if i >= w {
                sum -= close[i - w];
            }
            if i + 1 >= w {
                out[i] = sum / w as f64;
            }
        }
        out
    };
    let f = sma(fast);
    let s = sma(slow);
    (0..close.len())
        .map(|i| {
            if !f[i].is_finite() || !s[i].is_finite() {
                0
            } else if f[i] > s[i] {
                1
            } else if f[i] < s[i] {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// One synthetic candidate: a fixed signal vector + the gene's stop params.
struct Candidate {
    label: String,
    signals: Vec<i8>,
    sl_pips: f64,
    stop_vol_mult: f64,
}

struct LaneStats {
    trades: usize,
    net: f64,
    pf: f64,
    win_rate: f64,
}

fn lane_stats(trades: &[neoethos_search::Trade]) -> LaneStats {
    let net: f64 = trades.iter().map(|t| t.pnl).sum();
    let wins: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let losses: f64 = trades
        .iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| -t.pnl)
        .sum();
    let win_n = trades.iter().filter(|t| t.pnl > 0.0).count();
    LaneStats {
        trades: trades.len(),
        net,
        pf: if losses > 1e-9 { wins / losses } else { 0.0 },
        win_rate: win_n as f64 / trades.len().max(1) as f64,
    }
}

/// Screen-shaped survivor rule (stands in for passes_strict_quality):
/// enough trades, profitable, PF above water.
fn survives(s: &LaneStats, min_trades: usize) -> bool {
    s.trades >= min_trades && s.net > 0.0 && s.pf >= 1.05
}

/// Max daily loss fraction, bucketing trades per CALENDAR DAY (the fixed
/// arithmetic) or per TRADE (the old broken bucketing), at `initial_balance`.
fn max_daily_loss(
    trades: &[neoethos_search::Trade],
    per_day: bool,
    initial_balance: f64,
) -> f64 {
    let mut buckets: BTreeMap<i64, f64> = BTreeMap::new();
    for (i, t) in trades.iter().enumerate() {
        let key = if per_day {
            t.exit_time.unwrap_or(t.entry_time) / 86_400_000
        } else {
            i as i64 // the old behaviour: every trade its own bucket
        };
        *buckets.entry(key).or_insert(0.0) += t.pnl;
    }
    buckets
        .values()
        .filter(|p| **p < 0.0)
        .map(|p| p.abs() / initial_balance)
        .fold(0.0, f64::max)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: stage_stop_agreement <vortex-file> [pip_size]");
    let pip: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0001);

    let o = neoethos_data::load_vortex(&path)?;
    let n = o.close.len();
    let ts: Vec<i64> = o
        .timestamp
        .clone()
        .unwrap_or_else(|| (0..n as i64).map(|i| i * 300_000).collect());

    println!("file : {path}");
    println!("bars : {n}");

    // Candidate pool: 3 SMA-cross signal shapes × 5 stop parameterisations,
    // matching gene-generation ranges (fixed sl ~ U(6,20), mult ~ U(0.5,3.0)).
    let sma_grid = [(10usize, 50usize), (20, 100), (40, 200)];
    let stop_grid = [(6.0f64, 0.5f64), (10.0, 1.0), (13.0, 1.75), (16.0, 2.5), (20.0, 3.0)];
    let mut candidates = Vec::new();
    for (fast, slow) in sma_grid {
        let signals = sma_cross_signals(&o.close, fast, slow);
        for (sl, mult) in stop_grid {
            candidates.push(Candidate {
                label: format!("sma{fast}/{slow} sl{sl:.0} x{mult}"),
                signals: signals.clone(),
                sl_pips: sl,
                stop_vol_mult: mult,
            });
        }
    }

    let common = BacktestSettings {
        pip_value: pip,
        pip_value_per_lot: 10.0,
        spread_pips: 1.5,
        commission_per_trade: 7.0,
        risk_based_sizing: false,
        // The shipped value of `risk.kill_zones_enabled` (config.rs:671), which
        // since 2026-08-10 is what `discovery_backtest_settings` reads instead
        // of a literal (#75/#217). Pinned here so this example compares the two
        // lanes under one weekend policy rather than under whatever the local
        // config says.
        kill_zones_enabled: true,
        ..BacktestSettings::default()
    };
    let initial_balance = 100_000.0;

    // Full-series base (quality-screen lane).
    let full_base: std::sync::Arc<[f64]> =
        adaptive_base_pips_series(&o.high, &o.low, &o.close, pip)?.into();
    let median_base = {
        let mut w = full_base.to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w[w.len() / 2]
    };
    println!("median base : {median_base:.3} pips (mult = 1)\n");

    let settings_fixed = |c: &Candidate| BacktestSettings {
        sl_pips: c.sl_pips,
        tp_pips: c.sl_pips * 2.0,
        ..common.clone()
    };
    let settings_adaptive = |c: &Candidate, base: &std::sync::Arc<[f64]>| BacktestSettings {
        sl_pips: c.sl_pips,
        tp_pips: c.sl_pips * 2.0,
        adaptive_base_pips: Some(base.clone()),
        adaptive_vol_mult: c.stop_vol_mult,
        adaptive_rr: 2.0,
        ..common.clone()
    };

    // ── A. Quality-screen base backtest (full series) ───────────────────────
    println!("A. QUALITY-SCREEN LANE (full series, signal constant, stop regime varies)");
    println!(
        "{:>22}  {:>8} {:>12} {:>6} {:>6}   {:>8} {:>12} {:>6} {:>6}",
        "candidate", "tr_fix", "net_fix", "pf", "win%", "tr_ada", "net_ada", "pf", "win%"
    );
    let min_trades = 100;
    let (mut surv_fix, mut surv_ada, mut flips) = (0usize, 0usize, 0usize);
    let mut ratios = Vec::new();
    for c in &candidates {
        let t_fix = simulate_trades_core(&o.close, &o.high, &o.low, &ts, &c.signals, &settings_fixed(c));
        let t_ada = simulate_trades_core(
            &o.close, &o.high, &o.low, &ts, &c.signals,
            &settings_adaptive(c, &full_base),
        );
        let (sf, sa) = (lane_stats(&t_fix), lane_stats(&t_ada));
        if sa.trades > 0 {
            ratios.push(sf.trades as f64 / sa.trades as f64);
        }
        let (vf, va) = (survives(&sf, min_trades), survives(&sa, min_trades));
        surv_fix += vf as usize;
        surv_ada += va as usize;
        flips += (vf != va) as usize;
        println!(
            "{:>22}  {:>8} {:>12.0} {:>6.2} {:>5.1}%   {:>8} {:>12.0} {:>6.2} {:>5.1}%",
            c.label, sf.trades, sf.net, sf.pf, 100.0 * sf.win_rate,
            sa.trades, sa.net, sa.pf, 100.0 * sa.win_rate
        );
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\n  candidates {} | survivors fixed {} vs adaptive {} | verdict flips {} | \
         trade-count ratio fix/ada median {:.2} (min {:.2}, max {:.2})\n",
        candidates.len(), surv_fix, surv_ada, flips,
        ratios.get(ratios.len() / 2).copied().unwrap_or(0.0),
        ratios.first().copied().unwrap_or(0.0),
        ratios.last().copied().unwrap_or(0.0),
    );

    // ── B. Prop-firm 30-day window gate ─────────────────────────────────────
    println!("B. PROP-FIRM WINDOW GATE (40 x 30-day windows, window-local base)");
    let window_ms: i64 = 30 * 86_400_000;
    let n_windows = 40usize;
    let first_ts = ts[0];
    let last_ts = ts[n - 1];
    let span = ((last_ts - window_ms) - first_ts).max(1) as f64;
    let stride = span / (n_windows as f64 - 1.0);
    let mut windows: Vec<(usize, usize, Option<std::sync::Arc<[f64]>>)> = Vec::new();
    for i in 0..n_windows {
        let start_ts = first_ts + stride.mul_add(i as f64, 0.0) as i64;
        let end_ts = start_ts + window_ms;
        let s = ts.partition_point(|&t| t < start_ts);
        let e = ts.partition_point(|&t| t < end_ts).min(n);
        if e <= s + 1 {
            continue;
        }
        let base = adaptive_base_pips_series(&o.high[s..e], &o.low[s..e], &o.close[s..e], pip)
            .ok()
            .map(std::sync::Arc::from);
        windows.push((s, e, base));
    }
    let pass = |trades: &[neoethos_search::Trade]| -> bool {
        // FTMO-shaped: no 5% daily loss, no 10% overall loss (per-day buckets).
        let net: f64 = trades.iter().map(|t| t.pnl).sum();
        max_daily_loss(trades, true, initial_balance) < 0.05 && net > -0.10 * initial_balance
    };
    let floor = 0.65;
    let (mut gate_fix, mut gate_ada, mut gate_flips) = (0usize, 0usize, 0usize);
    let (mut sum_fix, mut sum_ada) = (0.0f64, 0.0f64);
    for c in &candidates {
        let (mut p_fix, mut p_ada) = (0usize, 0usize);
        for (s, e, base) in &windows {
            let (s, e) = (*s, *e);
            let t_fix = simulate_trades_core(
                &o.close[s..e], &o.high[s..e], &o.low[s..e], &ts[s..e],
                &c.signals[s..e], &settings_fixed(c),
            );
            p_fix += pass(&t_fix) as usize;
            let mut ada = settings_adaptive(c, &full_base);
            ada.adaptive_base_pips = base.clone(); // window-local, as landed
            let t_ada = simulate_trades_core(
                &o.close[s..e], &o.high[s..e], &o.low[s..e], &ts[s..e],
                &c.signals[s..e], &ada,
            );
            p_ada += pass(&t_ada) as usize;
        }
        let (r_fix, r_ada) = (
            p_fix as f64 / windows.len().max(1) as f64,
            p_ada as f64 / windows.len().max(1) as f64,
        );
        sum_fix += r_fix;
        sum_ada += r_ada;
        let (vf, va) = (r_fix >= floor, r_ada >= floor);
        gate_fix += vf as usize;
        gate_ada += va as usize;
        gate_flips += (vf != va) as usize;
    }
    println!(
        "  windows {} | mean pass-rate fixed {:.2} vs adaptive {:.2} | gate survivors \
         (floor {floor}) fixed {} vs adaptive {} | verdict flips {}\n",
        windows.len(),
        sum_fix / candidates.len() as f64,
        sum_ada / candidates.len() as f64,
        gate_fix, gate_ada, gate_flips,
    );

    // ── C. Walk-forward daily-loss diagnostics ──────────────────────────────
    println!("C. WALK-FORWARD RISK DIAGNOSTICS (12 splits, 5% daily-loss limit)");
    let n_splits = 12usize;
    let window = (n / n_splits).max(1);
    let limit = 0.05;
    let (mut breach_old_bucketing, mut breach_per_day_fix, mut breach_per_day_ada) =
        (0usize, 0usize, 0usize);
    let mut split_count = 0usize;
    for c in &candidates {
        for i in 0..n_splits {
            let start = i * window;
            let end = ((i + 1) * window).min(n);
            let train_end = start + ((window as f64) * 0.70) as usize;
            if train_end >= end || end - train_end < 40 {
                continue;
            }
            let (s, e) = (train_end, end);
            split_count += 1;
            let t_fix = simulate_trades_core(
                &o.close[s..e], &o.high[s..e], &o.low[s..e], &ts[s..e],
                &c.signals[s..e], &settings_fixed(c),
            );
            let base = adaptive_base_pips_series(&o.high[s..e], &o.low[s..e], &o.close[s..e], pip)
                .ok()
                .map(std::sync::Arc::from);
            let mut ada = settings_adaptive(c, &full_base);
            ada.adaptive_base_pips = base;
            let t_ada = simulate_trades_core(
                &o.close[s..e], &o.high[s..e], &o.low[s..e], &ts[s..e],
                &c.signals[s..e], &ada,
            );
            // The verdict the OLD code produced: fixed stops AND per-trade buckets.
            breach_old_bucketing +=
                (max_daily_loss(&t_fix, false, initial_balance) >= limit) as usize;
            // Corrected bucketing, fixed stops (isolates the day-key fix).
            breach_per_day_fix +=
                (max_daily_loss(&t_fix, true, initial_balance) >= limit) as usize;
            // Corrected bucketing, adaptive stops (what the code now runs).
            breach_per_day_ada +=
                (max_daily_loss(&t_ada, true, initial_balance) >= limit) as usize;
        }
    }
    println!(
        "  (candidate,split) cells {split_count} | daily-loss breaches: \
         OLD (fixed stops, per-trade buckets) {breach_old_bucketing} | \
         fixed stops, per-day buckets {breach_per_day_fix} | \
         ADAPTIVE stops, per-day buckets (landed) {breach_per_day_ada}",
    );

    Ok(())
}
