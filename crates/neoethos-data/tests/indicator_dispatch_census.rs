//! THROWAWAY DIAGNOSTIC — not a regression test. Delete after the finding lands.
//!
//! Answers one question with measurement rather than reasoning: WHY do 341 of
//! the 342 ids in `ALL_INDICATORS` produce no column in
//! `hpc_ta::compute_classic_ta_columns`?
//!
//! It makes, for every id, the EXACT call the production loop makes
//! (`params: &[]`, `output_id: None`), classifies the outcome, and then
//! replays the production accept/drop logic byte-for-byte to separate
//! "the library said Err" from "the library said Ok and we threw it away".
//!
//! Run:
//!   cargo test -p neoethos-data --test indicator_dispatch_census -- --nocapture
//! Report is written to NEOETHOS_CENSUS_OUT (default: ./indicator_census.txt).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

use neoethos_data::core::all_indicators::ALL_INDICATORS;
use neoethos_data::core::hpc_ta::{ALT_PERIODS, MULTI_PERIOD_IDS};
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorDispatchError, IndicatorSeries, ParamKV,
    ParamValue, compute_cpu,
};
use vector_ta::indicators::registry::{IndicatorValueType, get_indicator};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Bars of REAL EURUSD M5 to run the census on. Long enough that every
/// warmup in the library clears (longest declared lookback is ~200) and that
/// a length mismatch is unambiguous; short enough that 342 indicators finish.
const CENSUS_BARS: usize = 6000;

fn real_candles() -> (Candles, usize) {
    let base = std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA");
    let path = format!("{base}/neoethos/data/symbol=EURUSD/timeframe=M5/data.vortex");
    let ohlcv = neoethos_data::load_vortex(&path)
        .unwrap_or_else(|e| panic!("could not load real bars from {path}: {e:#}"));
    let total = ohlcv.close.len();
    assert!(total > CENSUS_BARS, "only {total} real bars available");
    // Take a slice from the RECENT end — the zero-price corruption documented
    // for 2014-12-08 lives in the early history and would poison the census
    // with spurious ComputeFailed.
    let lo = total - CENSUS_BARS;
    let n = CENSUS_BARS;
    let ts = ohlcv
        .timestamp
        .as_ref()
        .map(|t| t[lo..].to_vec())
        .unwrap_or_else(|| vec![0i64; n]);
    let vol = ohlcv
        .volume
        .as_ref()
        .map(|v| v[lo..].to_vec())
        .unwrap_or_else(|| vec![0.0; n]);
    let c = Candles::new(
        ts,
        ohlcv.open[lo..].to_vec(),
        ohlcv.high[lo..].to_vec(),
        ohlcv.low[lo..].to_vec(),
        ohlcv.close[lo..].to_vec(),
        vol,
    );
    (c, n)
}

/// Coarse bucket for an error, so 341 messages collapse into a handful of
/// named causes.
fn classify(e: &IndicatorDispatchError) -> &'static str {
    match e {
        IndicatorDispatchError::UnknownIndicator { .. } => "UnknownIndicator",
        IndicatorDispatchError::UnknownOutput { .. } => "UnknownOutput",
        IndicatorDispatchError::MissingRequiredInput { .. } => "MissingRequiredInput",
        IndicatorDispatchError::InvalidParam { key, reason, .. } => {
            if key == "output_id" && reason.contains("multi-output") {
                "InvalidParam/output_id-required-for-multi-output"
            } else {
                "InvalidParam/other"
            }
        }
        IndicatorDispatchError::UnsupportedCapability { .. } => "UnsupportedCapability",
        IndicatorDispatchError::DataLengthMismatch { .. } => "DataLengthMismatch",
        IndicatorDispatchError::KernelUnavailable { .. } => "KernelUnavailable",
        IndicatorDispatchError::ComputeFailed { .. } => "ComputeFailed",
        IndicatorDispatchError::CudaF64KernelMissing { .. } => "CudaF64KernelMissing",
    }
}

/// EXACT replay of the accept/drop logic in
/// `hpc_ta::compute_classic_ta_columns_with_policy` (the ALL_INDICATORS loop),
/// as of the file read today. Returns (columns_emitted, why).
fn hpc_ta_verdict(rows: usize, cols: usize, series: &IndicatorSeries, n: usize) -> (usize, String) {
    match series {
        IndicatorSeries::F64(v) => {
            if cols <= 1 {
                if v.len() == n {
                    (1, "F64 single-output, exact length".into())
                } else if v.len() > n {
                    (1, "F64 single-output, truncated".into())
                } else {
                    (
                        0,
                        format!("DROPPED: cols<=1 but v.len()={} < n={n}", v.len()),
                    )
                }
            } else if v.len() == rows * cols && rows >= n {
                (cols, "F64 multi-output decomposed".into())
            } else {
                (
                    0,
                    format!(
                        "DROPPED: took the multi-output branch (cols={cols}>1) and failed \
                         `v.len()=={rows}*{cols} && rows({rows})>=n({n})`"
                    ),
                )
            }
        }
        IndicatorSeries::I32(v) => {
            if v.len() == n {
                (1, "I32 exact length".into())
            } else {
                (0, format!("DROPPED: I32 len {} != n {n}", v.len()))
            }
        }
        IndicatorSeries::Bool(v) => {
            if v.len() == n {
                (1, "Bool exact length".into())
            } else {
                (0, format!("DROPPED: Bool len {} != n {n}", v.len()))
            }
        }
    }
}

struct Row {
    id: &'static str,
    registered: bool,
    n_outputs: usize,
    input_kind: String,
    value_types: String,
    /// outcome of the EXACT production call
    outcome: String,
    detail: String,
    emitted: usize,
    verdict: String,
    /// outcome of the REPAIRED call (explicit output_id per registered output)
    repaired_ok_outputs: usize,
    repaired_first_err: String,
}

#[test]
fn census_of_all_indicators() {
    let (candles, n) = real_candles();
    let out_path =
        std::env::var("NEOETHOS_CENSUS_OUT").unwrap_or_else(|_| "indicator_census.txt".into());

    let mut rows: Vec<Row> = Vec::with_capacity(ALL_INDICATORS.len());

    for &id in ALL_INDICATORS {
        let info = get_indicator(id);
        let n_outputs = info.map(|i| i.outputs.len()).unwrap_or(0);
        let input_kind = info
            .map(|i| format!("{:?}", i.input_kind))
            .unwrap_or_else(|| "-".into());
        let value_types = info
            .map(|i| {
                i.outputs
                    .iter()
                    .map(|o| match o.value_type {
                        IndicatorValueType::F64 => "f64",
                        IndicatorValueType::F32 => "f32",
                        IndicatorValueType::I32 => "i32",
                        IndicatorValueType::Bool => "bool",
                    })
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .unwrap_or_else(|| "-".into());

        // ---- Call A: byte-for-byte the production call site ---------------
        let call = |output_id: Option<&'static str>, params: &[ParamKV<'static>]| {
            let data = IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            };
            let req = IndicatorComputeRequest {
                indicator_id: id,
                output_id,
                data,
                params,
                kernel: Kernel::Auto,
            };
            catch_unwind(AssertUnwindSafe(|| compute_cpu(req)))
        };

        let (outcome, detail, emitted, verdict) = match call(None, &[]) {
            Err(_) => ("PANIC".to_string(), String::new(), 0, "panic".to_string()),
            Ok(Err(e)) => (
                classify(&e).to_string(),
                format!("{e}"),
                0,
                "err".to_string(),
            ),
            Ok(Ok(o)) => {
                let (emitted, why) = hpc_ta_verdict(o.rows, o.cols, &o.series, n);
                let kind = match &o.series {
                    IndicatorSeries::F64(v) => format!("F64 len={}", v.len()),
                    IndicatorSeries::I32(v) => format!("I32 len={}", v.len()),
                    IndicatorSeries::Bool(v) => format!("Bool len={}", v.len()),
                };
                (
                    if emitted > 0 {
                        "OK_KEPT".to_string()
                    } else {
                        "OK_BUT_DROPPED".to_string()
                    },
                    format!("rows={} cols={} {kind}", o.rows, o.cols),
                    emitted,
                    why,
                )
            }
        };

        // ---- Call B: the repair — explicit output_id per registered output -
        let mut repaired_ok_outputs = 0usize;
        let mut repaired_first_err = String::new();
        if let Some(info) = info {
            for o in info.outputs.iter() {
                let oid: Option<&'static str> = if info.outputs.len() <= 1 {
                    None
                } else {
                    Some(o.id)
                };
                match call(oid, &[]) {
                    Err(_) => {
                        if repaired_first_err.is_empty() {
                            repaired_first_err = format!("{}: PANIC", o.id);
                        }
                    }
                    Ok(Err(e)) => {
                        if repaired_first_err.is_empty() {
                            repaired_first_err = format!("{}: [{}] {e}", o.id, classify(&e));
                        }
                    }
                    Ok(Ok(out)) => {
                        let len = match &out.series {
                            IndicatorSeries::F64(v) => v.len(),
                            IndicatorSeries::I32(v) => v.len(),
                            IndicatorSeries::Bool(v) => v.len(),
                        };
                        if len >= n {
                            repaired_ok_outputs += 1;
                        } else if repaired_first_err.is_empty() {
                            repaired_first_err = format!("{}: SHORT len={len} n={n}", o.id);
                        }
                    }
                }
            }
        }

        rows.push(Row {
            id,
            registered: info.is_some(),
            n_outputs,
            input_kind,
            value_types,
            outcome,
            detail,
            emitted,
            verdict,
            repaired_ok_outputs,
            repaired_first_err,
        });
    }

    // ------------------------------------------------------------------
    // Report
    // ------------------------------------------------------------------
    let mut r = String::new();
    let _ = writeln!(
        r,
        "INDICATOR DISPATCH CENSUS — {} ids, {n} REAL EURUSD M5 bars\n",
        ALL_INDICATORS.len()
    );

    let mut by_outcome: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        by_outcome.entry(row.outcome.as_str()).or_default().push(row);
    }
    let _ = writeln!(r, "== OUTCOME DISTRIBUTION (production call, params=[], output_id=None) ==");
    for (k, v) in &by_outcome {
        let _ = writeln!(r, "{:>6}  {}", v.len(), k);
    }
    let kept: usize = rows.iter().filter(|x| x.emitted > 0).count();
    let cols_today: usize = rows.iter().map(|x| x.emitted).sum();
    let _ = writeln!(
        r,
        "\nids that produce >=1 column TODAY: {kept}  (columns: {cols_today})"
    );

    // Cross-tab: single vs multi output
    let multi_err = rows
        .iter()
        .filter(|x| x.n_outputs > 1 && x.verdict == "err")
        .count();
    let single_dropped = rows
        .iter()
        .filter(|x| x.n_outputs <= 1 && x.outcome == "OK_BUT_DROPPED")
        .count();
    let _ = writeln!(
        r,
        "\nmulti-output ids that Err: {multi_err}\nsingle-output ids that succeed then get DROPPED by hpc_ta: {single_dropped}"
    );

    // Repair potential
    let repair_ids = rows.iter().filter(|x| x.repaired_ok_outputs > 0).count();
    let repair_cols: usize = rows.iter().map(|x| x.repaired_ok_outputs).sum();
    let _ = writeln!(
        r,
        "\n== REPAIR POTENTIAL (explicit output_id, library default params, correct shape handling) ==\n\
         ids producing >=1 usable series: {repair_ids} / {}\n\
         total base-timeframe columns recoverable: {repair_cols}",
        ALL_INDICATORS.len()
    );

    // Who still fails after repair, and why
    let mut still_bad: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for row in &rows {
        if row.repaired_ok_outputs < row.n_outputs.max(1) {
            let bucket = if !row.registered {
                "not in registry".to_string()
            } else if row.repaired_first_err.is_empty() {
                "unknown".to_string()
            } else {
                let msg = &row.repaired_first_err;
                if msg.contains("PANIC") {
                    "PANIC".into()
                } else if let Some(s) = msg.find('[') {
                    let e = msg[s..].find(']').map(|e| s + e + 1).unwrap_or(msg.len());
                    msg[s..e].to_string()
                } else if msg.contains("SHORT") {
                    "SHORT series".into()
                } else {
                    "other".into()
                }
            };
            still_bad.entry(bucket).or_default().push(row.id);
        }
    }
    let _ = writeln!(r, "\n== RESIDUAL FAILURES AFTER REPAIR ==");
    for (k, v) in &still_bad {
        let _ = writeln!(r, "{:>6}  {}   e.g. {:?}", v.len(), k, &v[..v.len().min(12)]);
    }

    // Full per-id table
    let _ = writeln!(
        r,
        "\n== PER-ID DETAIL ==\n{:<52} {:>4} {:>4} {:<12} {:<10} {:<42} {:<34} {}",
        "id", "outs", "emit", "value_types", "input", "outcome", "detail", "verdict"
    );
    for row in &rows {
        let _ = writeln!(
            r,
            "{:<52} {:>4} {:>4} {:<12} {:<10} {:<42} {:<34} {} | repaired_ok={} {}",
            row.id,
            row.n_outputs,
            row.emitted,
            row.value_types,
            row.input_kind,
            row.outcome,
            row.detail,
            row.verdict,
            row.repaired_ok_outputs,
            row.repaired_first_err
        );
    }

    // ------------------------------------------------------------------
    // The parameterised sweep path, for comparison
    // ------------------------------------------------------------------
    let _ = writeln!(
        r,
        "\n== MULTI-PERIOD SWEEP PATH (params=[period], output_id=None) — the 18 that 'work' =="
    );
    for &ind in MULTI_PERIOD_IDS.iter() {
        let info = get_indicator(ind);
        let outs = info.map(|i| i.outputs.len()).unwrap_or(0);
        let mut oks = 0;
        let mut first_err = String::new();
        for &p in ALT_PERIODS.iter() {
            let params = [ParamKV {
                key: "period",
                value: ParamValue::Int(p as i64),
            }];
            let data = IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            };
            let req = IndicatorComputeRequest {
                indicator_id: ind,
                output_id: None,
                data,
                params: &params,
                kernel: Kernel::Auto,
            };
            match catch_unwind(AssertUnwindSafe(|| compute_cpu(req))) {
                Ok(Ok(o)) => {
                    let len = match &o.series {
                        IndicatorSeries::F64(v) => v.len(),
                        IndicatorSeries::I32(v) => v.len(),
                        IndicatorSeries::Bool(v) => v.len(),
                    };
                    if len == n {
                        oks += 1;
                    } else if first_err.is_empty() {
                        first_err = format!("p={p}: len={len} n={n} rows={} cols={}", o.rows, o.cols);
                    }
                }
                Ok(Err(e)) => {
                    if first_err.is_empty() {
                        first_err = format!("p={p}: [{}] {e}", classify(&e));
                    }
                }
                Err(_) => {
                    if first_err.is_empty() {
                        first_err = format!("p={p}: PANIC");
                    }
                }
            }
        }
        let _ = writeln!(
            r,
            "{:<20} outputs={outs:<3} periods_ok={oks}/5   {first_err}",
            ind
        );
    }

    std::fs::write(&out_path, &r).expect("write census");
    eprintln!("{r}");
    eprintln!("--- census written to {out_path} ---");
}
