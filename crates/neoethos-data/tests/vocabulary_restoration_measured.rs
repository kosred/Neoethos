//! What the vocabulary restoration actually bought, measured on the real
//! on-disk store rather than estimated.
//!
//! Non-negotiable #4: restoring the vocabulary CHANGES WHAT THE SEARCH
//! EXPLORES, so the before/after column counts must be reported, and every
//! historical discovery result — produced on the 66-column vocabulary — is not
//! comparable to anything produced after this lands. This test is where that
//! number comes from.
//!
//! Skips (does not fail) when the store is absent, so a machine without the
//! data still runs the suite.
//!
//! ```text
//! cargo test -p neoethos-data --test vocabulary_restoration_measured -- --nocapture
//! ```

use std::collections::BTreeMap;

use neoethos_data::core::feature_budget::{VocabularyBudget, column_bytes};
use neoethos_data::core::hpc_ta::{
    ALT_PERIODS, IndicatorComputePolicy, MULTI_PERIOD_IDS, compute_classic_ta_columns_with_policy,
};

/// Bars to measure on. Long enough to clear every period in `ALT_PERIODS`
/// (200 * 1.25 = 250) by three orders of magnitude, so nothing is
/// warmup-limited, and short enough to run in seconds.
const BARS: usize = 200_000;

/// Load the real bars, or FAIL naming the resolved path.
///
/// This used to return `Option` and the test printed "SKIPPED" and passed. That
/// is the defect this whole workstream exists to kill, wearing a test's
/// clothing: a green result that distinguishes "measured 825 columns on real
/// bars" from "measured nothing" only in stdout nobody reads. The test is
/// `#[ignore]`d instead, so it never runs where the store is absent — and when
/// it IS run, a missing store is a hard failure with the path in the message.
fn real_ohlcv() -> neoethos_data::Ohlcv {
    let base = std::env::var("LOCALAPPDATA")
        .expect("LOCALAPPDATA is unset — cannot resolve the neoethos data store");
    let path = format!("{base}/neoethos/data/symbol=EURUSD/timeframe=M5/data.vortex");
    let o = neoethos_data::load_vortex(&path)
        .unwrap_or_else(|e| panic!("no EURUSD M5 vortex store at {path}: {e:#}"));
    let total = o.close.len();
    assert!(
        total >= BARS,
        "the store at {path} holds {total} bars, fewer than the {BARS} this measurement needs"
    );
    let lo = total - BARS;
    neoethos_data::Ohlcv {
        timestamp: o.timestamp.as_ref().map(|t| t[lo..].to_vec()),
        open: o.open[lo..].to_vec(),
        high: o.high[lo..].to_vec(),
        low: o.low[lo..].to_vec(),
        close: o.close[lo..].to_vec(),
        volume: o.volume.as_ref().map(|v| v[lo..].to_vec()),
    }
}

/// A column belongs to the multi-period sweep iff its name starts with a swept
/// id followed by `_<period>`. Everything else came from the base
/// ALL_INDICATORS pass.
fn is_sweep_column(name: &str) -> bool {
    MULTI_PERIOD_IDS.iter().any(|id| {
        ALT_PERIODS.iter().any(|p| {
            let prefix = format!("{id}_{p}");
            name == prefix || name.starts_with(&format!("{prefix}_"))
        })
    })
}

#[test]
#[ignore = "requires the real EURUSD M5 vortex store; run it explicitly before any production \
            discovery run — `cargo test -p neoethos-data --test vocabulary_restoration_measured \
            -- --ignored --nocapture`"]
fn measure_the_restored_vocabulary_on_real_bars() {
    let ohlcv = real_ohlcv();
    let n = ohlcv.close.len();

    let t0 = std::time::Instant::now();
    let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu)
        .expect("the repaired base pass must clear its own vocabulary floor on real bars");
    let elapsed = t0.elapsed();

    let sweep: Vec<&String> = cols
        .iter()
        .map(|(k, _)| k)
        .filter(|k| is_sweep_column(k))
        .collect();
    let base_count = cols.len() - sweep.len();

    // Distinct producing ids in the base pass: the column name for a
    // single-output indicator IS the id; a multi-output one is `<id>_<output>`.
    let mut per_id: BTreeMap<String, usize> = BTreeMap::new();
    for (name, _) in cols.iter().filter(|(k, _)| !is_sweep_column(k)) {
        let id = neoethos_data::core::all_indicators::ALL_INDICATORS
            .iter()
            .filter(|id| name == *id || name.starts_with(&format!("{id}_")))
            // Longest match wins, so `adaptive_macd_signal` is not attributed
            // to `adaptive_macd`'s shorter cousins.
            .max_by_key(|id| id.len())
            .map(|s| s.to_string())
            .unwrap_or_else(|| name.clone());
        *per_id.entry(id).or_insert(0) += 1;
    }

    let bytes = column_bytes(n) * cols.len() as u64;
    let budget = VocabularyBudget::for_frame(n);

    eprintln!("\n=== RESTORED INDICATOR VOCABULARY, MEASURED ===");
    eprintln!("bars                    : {n}");
    eprintln!("wall clock              : {:.1}s", elapsed.as_secs_f64());
    eprintln!(
        "ALL_INDICATORS ids      : {}",
        neoethos_data::core::all_indicators::ALL_INDICATORS.len()
    );
    eprintln!("producing ids (base)    : {}   (was 1: ttm_trend)", per_id.len());
    eprintln!("base columns            : {base_count}   (was 1)");
    eprintln!("sweep columns           : {}   (was 65)", sweep.len());
    eprintln!("TOTAL columns           : {}   (was 66)", cols.len());
    eprintln!(
        "f64 staging cost        : {:.2} GB at {n} bars ({:.2} MB per column)",
        bytes as f64 / 1e9,
        column_bytes(n) as f64 / 1e6
    );
    eprintln!(
        "budget on this machine  : {} columns max ({:.1} GB free)",
        budget.max_columns,
        budget.available_bytes as f64 / 1e9
    );
    eprintln!(
        "at 843,456 bars         : {:.2} GB for {} columns",
        (column_bytes(843_456) * cols.len() as u64) as f64 / 1e9,
        cols.len()
    );
    eprintln!("===============================================\n");

    // The measurement is the point, but these are the claims it must support.
    assert!(
        cols.len() > 400,
        "the restored vocabulary is {} columns — the whole exercise was to get past 66",
        cols.len()
    );
    assert!(
        per_id.len() > 250,
        "only {} of {} indicator ids produced a column",
        per_id.len(),
        neoethos_data::core::all_indicators::ALL_INDICATORS.len()
    );
    // Every column full length: the cube copy in lib.rs zero-pads otherwise.
    for (name, v) in &cols {
        assert_eq!(v.len(), n, "column '{name}' is {} values", v.len());
    }
}
