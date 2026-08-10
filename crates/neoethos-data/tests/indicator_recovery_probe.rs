//! THROWAWAY DIAGNOSTIC #2 — how many DISTINCT base-timeframe columns are
//! actually recoverable, once the two dispatch mistakes are fixed?
//!
//! Probe 1 (`indicator_dispatch_census`) established the causes. This one
//! establishes the SIZE of the prize, honestly:
//!   * registered ids  -> every output the registry declares;
//!   * unregistered ids -> every output name harvested from the dispatch
//!     source (NEOETHOS_OUTPUT_NAMES json), because the registry does not
//!     know about them;
//!   * every produced series is HASHED and de-duplicated per id, so alias
//!     output names ("hist"/"histogram") count once, and an output that is
//!     bit-identical to another is not counted twice;
//!   * all-NaN / constant series are counted separately — a column with no
//!     variance is not vocabulary, it is ballast the prefilter will drop.
//!
//! Run:
//!   cargo test -p neoethos-data --test indicator_recovery_probe -- --nocapture

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

use neoethos_data::core::all_indicators::ALL_INDICATORS;
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, compute_cpu,
};
use vector_ta::indicators::registry::get_indicator;
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

const BARS: usize = 6000;

fn real_candles() -> (Candles, usize) {
    let base = std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA");
    let path = format!("{base}/neoethos/data/symbol=EURUSD/timeframe=M5/data.vortex");
    let o = neoethos_data::load_vortex(&path).expect("real bars");
    let total = o.close.len();
    let lo = total - BARS;
    let n = BARS;
    let ts = o
        .timestamp
        .as_ref()
        .map(|t| t[lo..].to_vec())
        .unwrap_or_else(|| vec![0i64; n]);
    let v = o
        .volume
        .as_ref()
        .map(|v| v[lo..].to_vec())
        .unwrap_or_else(|| vec![0.0; n]);
    (
        Candles::new(
            ts,
            o.open[lo..].to_vec(),
            o.high[lo..].to_vec(),
            o.low[lo..].to_vec(),
            o.close[lo..].to_vec(),
            v,
        ),
        n,
    )
}

fn hash_series(v: &[f64]) -> u64 {
    // FNV-1a over the raw bits, NaN-canonicalised so two all-NaN series hash
    // the same. f64 bits — no narrowing anywhere.
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        let bits = if x.is_nan() { 0x7ff8_0000_0000_0000u64 } else { x.to_bits() };
        for b in bits.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// (finite_count, distinct_finite_values)
fn variance_profile(v: &[f64]) -> (usize, usize) {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut finite = 0usize;
    for &x in v {
        if x.is_finite() {
            finite += 1;
            seen.insert(x.to_bits());
        }
    }
    (finite, seen.len())
}

#[test]
fn recoverable_vocabulary_size() {
    let (candles, n) = real_candles();
    let names_path = std::env::var("NEOETHOS_OUTPUT_NAMES").expect("NEOETHOS_OUTPUT_NAMES");
    let harvested: BTreeMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(&names_path).expect("read names"))
            .expect("parse names");

    let mut report = String::new();
    let mut total_distinct = 0usize;
    let mut total_useful = 0usize; // distinct AND has >1 finite value
    let mut ids_with_any = 0usize;
    let mut per_id: Vec<(String, usize, usize, String)> = Vec::new();
    let mut dead: Vec<&str> = Vec::new();

    for &id in ALL_INDICATORS {
        let info = get_indicator(id);
        // Candidate output ids: registry first, harvested source names as the
        // fallback for the ids the registry does not know.
        let mut candidates: Vec<Option<String>> = Vec::new();
        match info {
            Some(i) if i.outputs.len() == 1 => candidates.push(None),
            Some(i) if i.outputs.len() > 1 => {
                for o in i.outputs.iter() {
                    candidates.push(Some(o.id.to_string()));
                }
            }
            _ => {
                if let Some(names) = harvested.get(id) {
                    for nm in names {
                        candidates.push(Some(nm.clone()));
                    }
                }
                candidates.push(None);
            }
        }

        let mut hashes: BTreeSet<u64> = BTreeSet::new();
        let mut distinct = 0usize;
        let mut useful = 0usize;
        let mut note = String::new();

        for c in &candidates {
            let data = IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            };
            let req = IndicatorComputeRequest {
                indicator_id: id,
                output_id: c.as_deref(),
                data,
                params: &[],
                kernel: Kernel::Auto,
            };
            let r = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
            match r {
                Ok(Ok(o)) => {
                    // Normalise every series shape to f64 columns of length n.
                    // rows = param combos (1 here), cols = series length.
                    let vals: Vec<f64> = match o.series {
                        IndicatorSeries::F64(v) => v,
                        IndicatorSeries::I32(v) => v.into_iter().map(|x| x as f64).collect(),
                        IndicatorSeries::Bool(v) => {
                            v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect()
                        }
                    };
                    if vals.len() < n {
                        if note.is_empty() {
                            note = format!("short len={} (rows={} cols={})", vals.len(), o.rows, o.cols);
                        }
                        continue;
                    }
                    // A matrix output (e.g. pattern_recognition) is rows*cols;
                    // split it into `rows` columns of length cols=n.
                    let chunks: Vec<&[f64]> = if o.cols == n && vals.len() == o.rows * o.cols {
                        vals.chunks(n).collect()
                    } else {
                        vec![&vals[..n]]
                    };
                    for ch in chunks {
                        if hashes.insert(hash_series(ch)) {
                            distinct += 1;
                            let (finite, uniq) = variance_profile(ch);
                            if finite > n / 2 && uniq > 1 {
                                useful += 1;
                            }
                        }
                    }
                }
                Ok(Err(e)) => {
                    if note.is_empty() {
                        note = format!("{e}");
                    }
                }
                Err(_) => {
                    if note.is_empty() {
                        note = "PANIC".into();
                    }
                }
            }
        }

        if distinct == 0 {
            dead.push(id);
        } else {
            ids_with_any += 1;
        }
        total_distinct += distinct;
        total_useful += useful;
        per_id.push((id.to_string(), distinct, useful, note));
    }

    let _ = writeln!(
        report,
        "RECOVERABLE VOCABULARY — {} ids, {n} REAL EURUSD M5 bars\n\
         ids yielding >=1 distinct series : {ids_with_any}\n\
         ids yielding NOTHING             : {}\n\
         DISTINCT base columns            : {total_distinct}\n\
         of which non-degenerate          : {total_useful}   (>50% finite AND >1 unique value)\n",
        ALL_INDICATORS.len(),
        dead.len()
    );
    let _ = writeln!(report, "DEAD IDS ({}): {dead:?}\n", dead.len());
    let _ = writeln!(report, "{:<52} {:>8} {:>8}  note", "id", "distinct", "useful");
    for (id, d, u, note) in &per_id {
        let _ = writeln!(report, "{id:<52} {d:>8} {u:>8}  {note}");
    }

    let out = std::env::var("NEOETHOS_PROBE_OUT").unwrap_or_else(|_| "recovery_probe.txt".into());
    std::fs::write(&out, &report).expect("write");
    eprintln!("{}", &report[..report.len().min(4000)]);
    eprintln!("--- full report at {out} ---");
}
