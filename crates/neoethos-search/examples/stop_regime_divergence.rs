//! Quantifies the SECOND divergence found while fixing the 300 000-bar cliff:
//! the discovery pipeline evaluated the same gene under TWO different stop
//! regimes depending on which call site was running.
//!
//! **FIXED 2026-08-08 (slice 2):** every serial discovery stage now routes
//! through `discovery.rs::GeneEvalSettingsResolver`, which installs the same
//! `sl = stop_vol_mult × base[i]` regime scoring uses; the guard test
//! `discovery_backtest_settings_has_no_callers_outside_the_resolvers` keeps it
//! that way. The measurement companion is `stage_stop_agreement.rs`. This
//! example is kept as the minimal reproduction of the divergence itself.
//!
//! History: `search_engine.rs::resolve_adaptive_stops` installed the per-bar
//! volatility-scaled stop for scoring, the walk-forward metric half and the
//! CPCV half, while `discovery.rs` never set `adaptive_base_pips` /
//! `adaptive_vol_mult` anywhere, so every path that went through
//! `discovery_backtest_settings` — `faithful_oos_eval`, the walk-forward RISK
//! diagnostics, the forward-test and prop-firm artifacts, the
//! permutation/plateau robustness filters — ran the gene's unused FIXED
//! `sl_pips`/`tp_pips` instead.
//!
//! This holds the SIGNAL constant and varies only the stop regime, so the
//! difference reported here is exactly the divergence, nothing else.
//!
//! Usage:
//!   cargo run -p neoethos-search --release --example stop_regime_divergence \
//!       -- <vortex-file> [pip_size]

use neoethos_search::{BacktestSettings, adaptive_base_pips_series, simulate_trades_core};

/// Deterministic SMA-cross signal — a stand-in for a gene's direction vector.
/// Its only job is to be IDENTICAL across the two runs below.
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

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: stop_regime_divergence <vortex-file> [pip_size]");
    let pip: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0001);

    let o = neoethos_data::load_vortex(&path)?;
    let n = o.close.len();
    let ts: Vec<i64> = o
        .timestamp
        .clone()
        .unwrap_or_else(|| (0..n as i64).map(|i| i * 300_000).collect());
    let signals = sma_cross_signals(&o.close, 20, 50);

    let base = adaptive_base_pips_series(&o.high, &o.low, &o.close, pip)
        .expect("base series builds with the cap removed");
    let median_base = {
        let mut w = base.clone();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w[w.len() / 2]
    };

    println!("file        : {path}");
    println!("bars        : {n}");
    println!("median base : {median_base:.4} pips (stop_vol_mult = 1)");
    println!();
    println!(
        "{:>26}  {:>9}  {:>14}  {:>12}  {:>12}",
        "regime", "trades", "net", "avg_sl_pips", "gross_win%"
    );

    let mut common = BacktestSettings {
        pip_value: pip,
        pip_value_per_lot: 10.0,
        spread_pips: 1.5,
        commission_per_trade: 7.0,
        risk_based_sizing: false,
        ..BacktestSettings::default()
    };

    // The gene-generation ranges from `evolution_math.rs::new_random_gene`:
    // fixed  sl ~ U(6, 20) pips (or exactly 15 with p = 0.2)
    // adaptive stop_vol_mult ~ U(0.5, 3.0)
    let report = |label: &str, s: &BacktestSettings| {
        let trades = simulate_trades_core(&o.close, &o.high, &o.low, &ts, &signals, s);
        let net: f64 = trades.iter().map(|t| t.pnl).sum();
        let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
        let avg_sl = if s.adaptive_vol_mult > 0.0 {
            s.adaptive_vol_mult * median_base
        } else {
            s.sl_pips
        };
        println!(
            "{label:>26}  {:>9}  {net:>14.2}  {avg_sl:>12.2}  {:>11.1}%",
            trades.len(),
            100.0 * wins as f64 / trades.len().max(1) as f64
        );
    };

    // What `discovery.rs` runs: the gene's FIXED pips.
    for sl in [6.0_f64, 13.0, 20.0] {
        common.sl_pips = sl;
        common.tp_pips = sl * 2.0;
        common.adaptive_base_pips = None;
        common.adaptive_vol_mult = 0.0;
        report(&format!("discovery.rs fixed {sl:.0}p"), &common);
    }

    // What `search_engine.rs` scores on: the volatility-scaled stop.
    common.adaptive_base_pips = Some(base.clone().into());
    common.adaptive_rr = 2.0;
    for mult in [0.5_f64, 1.75, 3.0] {
        common.adaptive_vol_mult = mult;
        report(&format!("search_engine adaptive x{mult}"), &common);
    }

    // ── Second divergence on the same struct: kill zones ───────────────────
    // HALF FIXED 2026-08-10 (#75/#217). `risk.kill_zones_enabled` used to reach
    // the LIVE loop and NOT discovery, which hardcoded `true`. Discovery now
    // reads the same field, so live and the discovery lanes agree.
    //
    // What this example still measures is the OTHER half: `search_engine.rs`'s
    // scoring settings take `..BacktestSettings::default()`, which is `false`.
    // So the GA holds through the weekend while every downstream validation
    // force-exits at Friday 20:00 — same gene, same bars, different exit rule.
    // The two rows below are that gap, in money.
    println!();
    println!(
        "{:>26}  {:>9}  {:>14}  {:>12}  {:>12}",
        "kill-zone regime", "trades", "net", "avg_sl_pips", "gross_win%"
    );
    common.adaptive_base_pips = None;
    common.adaptive_vol_mult = 0.0;
    common.sl_pips = 13.0;
    common.tp_pips = 26.0;
    for (label, on) in [
        ("search_engine (default) off", false),
        ("discovery.rs hardcoded on", true),
    ] {
        common.kill_zones_enabled = on;
        report(label, &common);
    }

    Ok(())
}
