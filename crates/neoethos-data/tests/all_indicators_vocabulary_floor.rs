//! The test that would have caught the historical all-but-one indicator collapse.
//!
//! # What went wrong, and why nothing noticed for sixteen months
//!
//! `hpc_ta` iterated every id in [`ALL_INDICATORS`] and kept whatever came
//! back through an `if let Ok(output)` with no `else` and an `if v.len() == n`
//! with no `else`. Exactly ONE id (`ttm_trend`) produced a column. Every other
//! declared id was discarded with no log line, no counter, and no way for any
//! run to know. Every discovery result the project has ever produced was
//! searched over 66 columns instead of ~800.
//!
//! There was no test that could fail, because the only assertion anyone could
//! have written from the outside — "the feature build succeeded" — was TRUE.
//! The build did succeed. It succeeded at producing one column.
//!
//! # What this test asserts
//!
//! The floor is **derived from the declared list**, not a constant somebody can
//! quietly lower:
//!
//! ```text
//! attemptable = ALL_INDICATORS.len() - EXPECTED_NON_PRODUCING.len()
//! ```
//!
//! Every id in `EXPECTED_NON_PRODUCING` is named there with a measured reason
//! (three are moving-average family dispatch selectors, two have no dispatch
//! arm in vector-ta 0.2.9, four have no `cpu_batch` arm). Everything else is
//! expected to produce. So if someone deletes an indicator from
//! `ALL_INDICATORS`, the floor drops with it and this test does not become a
//! liar; but if the dispatch breaks again, the ratio collapses and CI fails
//! with the id census attached.
//!
//! This is deliberately independent of `hpc_ta::MIN_PRODUCING_INDICATOR_IDS`.
//! That constant guards the production path at run time; this guards the
//! constant. Lowering `MIN_PRODUCING_INDICATOR_IDS` to 1 to make a red build
//! green would leave this test red.
//!
//! # Why synthetic bars
//!
//! It must run in CI on a machine with no data store, so it generates its own
//! deterministic random walk. That is a departure from "verify only on real
//! data", and the reason is specific: this test does not measure MARKET
//! behaviour, it measures DISPATCH — whether each indicator returns a series of
//! the right length at all. `tests/vocabulary_restoration_measured.rs` does the
//! real-bars measurement and reports the vocabulary size; it SKIPS without the
//! store, so it cannot be the CI gate. This one can.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use neoethos_data::Ohlcv;
use neoethos_data::core::all_indicators::ALL_INDICATORS;
use neoethos_data::core::hpc_ta::{
    IndicatorComputePolicy, VOCABULARY_FLOOR_MIN_ROWS, compute_classic_ta_columns_with_policy,
};
use neoethos_data::core::indicator_ledger::{
    EXPECTED_NON_PRODUCING, PRODUCTION_OUTPUT_EXCLUSIONS, expected_non_producing,
    production_output_exclusion,
};
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, compute_cpu,
};
use vector_ta::indicators::sar::{SarInput, SarParams, sar_with_kernel};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Enough bars to clear the 200-period sweep warmup with room to spare, and to
/// pass `VOCABULARY_FLOOR_MIN_ROWS` so the production floor is armed too.
const BARS: usize = 6_000;

/// Deterministic OHLCV: a bounded random walk in f64 with a real high/low
/// spread and a positive volume, seeded by a constant so the column count is
/// reproducible on every machine and every run.
fn synthetic_ohlcv(n: usize) -> Ohlcv {
    let mut state: u64 = 0x2026_0809_0000_0001;
    let mut next = || {
        // xorshift64* — deterministic, no dependency, good enough for a walk.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamp = Vec::with_capacity(n);

    let mut px = 1.1000f64;
    // EURUSD M5, five minutes per bar, starting 2020-01-01T00:00:00Z.
    let mut ts = 1_577_836_800_000i64;
    for _ in 0..n {
        let o = px;
        // ±5 pips per bar, plus a slow sine so trend/cycle indicators have
        // something other than white noise to find.
        let step = (next() - 0.5) * 0.0010;
        let c = (o + step).clamp(0.5000, 2.0000);
        let wick = next() * 0.0006;
        let h = o.max(c) + wick;
        let l = (o.min(c) - wick).max(0.0001);
        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(100.0 + next() * 900.0);
        timestamp.push(ts);
        px = c;
        ts += 300_000;
    }

    Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

/// Attribute a produced column back to the id that made it. Column names are
/// `<id>`, `<id>_<output>`, or `<id>_<period>[_<output>]`; the longest matching
/// id wins so `adaptive_macd_signal` is not credited to a shorter cousin.
fn owning_id(name: &str) -> Option<&'static str> {
    ALL_INDICATORS
        .iter()
        .filter(|id| name == **id || name.starts_with(&format!("{id}_")))
        .max_by_key(|id| id.len())
        .copied()
}

#[test]
fn all_indicators_produce_columns_or_the_build_fails() {
    assert!(
        BARS >= VOCABULARY_FLOOR_MIN_ROWS,
        "this test must be long enough to arm the production floor too"
    );

    let ohlcv = synthetic_ohlcv(BARS);
    let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
        .expect(
            "the indicator pass refused its own vocabulary floor — read the \
             neoethos_data::indicator_ledger census lines above, they name the drop bucket",
        );

    let mut per_id: BTreeMap<&str, usize> = BTreeMap::new();
    let mut unattributed = Vec::new();
    for (name, values) in &cols {
        // A short column is a build defect: the cube copy in lib.rs pre-zeroes
        // its buffer, so the tail would silently become real zeros the GA can
        // threshold against.
        assert_eq!(
            values.len(),
            BARS,
            "column '{name}' has {} values, expected {BARS} — a short column becomes zero-padded \
             data downstream",
            values.len()
        );
        match owning_id(name) {
            Some(id) => *per_id.entry(id).or_insert(0) += 1,
            None => unattributed.push(name.clone()),
        }
    }

    // The floor, derived from the declared list rather than pinned to a number.
    let declared = ALL_INDICATORS.len();
    let excluded = EXPECTED_NON_PRODUCING.len();
    let statically_redundant_ids = ALL_INDICATORS
        .iter()
        .filter(|id| production_output_exclusion(id, None).is_some())
        .count();
    let attemptable = declared - excluded - statically_redundant_ids;
    let producing = per_id.len();
    // 80% leaves room for vector-ta version drift and for indicators that
    // legitimately need more warmup than 6,000 bars, while making the failure
    // this test exists for — the collapse to ONE producing id — impossible to
    // miss.
    let floor_ids = attemptable * 4 / 5;
    // THE COLUMN FLOOR MUST BE COUNTED IN OUTPUTS, NOT IN IDS.
    //
    // It used to be `attemptable` — one column per indicator. That is the
    // original defect in miniature: the all-but-one drop was an OUTPUT-level
    // failure (92 requests failed on `output_id: None`, and every multi-output
    // indicator — stoch, macd, bollinger_bands, keltner, supertrend — was in
    // that bucket), so a floor that cannot see outputs cannot see the bug. With
    // the id-counted floor, `output_ids_for` collapsing to its first output
    // drops 825 columns to 435 — 47% of the vocabulary — while `producing`
    // stays 329 and `silent` stays empty. Demonstrated by mutation.
    //
    // Derived INDEPENDENTLY of `output_ids_for` — straight off vector-ta's
    // registry — so mutating the resolver cannot co-mutate the floor that
    // guards it. Unregistered ids are floored at 1 (their output override table
    // is this crate's, not the registry's, and this floor must not depend on
    // the thing it is guarding).
    let floor_columns: usize = ALL_INDICATORS
        .iter()
        .filter(|id| expected_non_producing(id).is_none())
        .map(|id| {
            vector_ta::indicators::registry::get_indicator(id).map_or_else(
                || usize::from(production_output_exclusion(id, None).is_none()),
                |info| {
                    if info.outputs.len() <= 1 {
                        usize::from(production_output_exclusion(id, None).is_none())
                    } else {
                        info.outputs
                            .iter()
                            .filter(|output| {
                                production_output_exclusion(id, Some(output.id)).is_none()
                            })
                            .count()
                    }
                },
            )
        })
        .sum();

    let silent: Vec<&&str> = ALL_INDICATORS
        .iter()
        .filter(|id| {
            !per_id.contains_key(**id)
                && expected_non_producing(id).is_none()
                && production_output_exclusion(id, None).is_none()
        })
        .collect();

    // THE EXCLUSION LIST IS THE ONLY THING THAT LOWERS THE FLOOR, SO IT IS THE
    // ONLY PLACE A COLLAPSE CAN HIDE — AND IT HAD NO STALENESS CHECK.
    //
    // Nothing verified that an excluded id actually fails. Adding a working id
    // (measured: `rsi`) to EXPECTED_NON_PRODUCING left this test green while
    // producing (329) EXCEEDED attemptable (328) — an arithmetically impossible
    // state that nothing asserted on. The production ledger already detects
    // this (`IndicatorLedger::stale_exclusion`, warned in `log_summary`), but a
    // WARN is not a gate and this test never consulted it.
    let stale: Vec<&str> = EXPECTED_NON_PRODUCING
        .iter()
        .map(|(id, _)| *id)
        .filter(|id| per_id.contains_key(id))
        .collect();

    // Every BASE column name the registry says must exist. `<id>` for a
    // single-output indicator, `<id>_<output>` for each declared output of a
    // multi-output one — the naming `dispatch_indicator_outputs` emits. Read
    // from the registry directly so this floor is independent of the crate's
    // own output resolver. Unregistered ids are skipped: their outputs come
    // from this crate's override table, which is the thing under test.
    let emitted: std::collections::HashSet<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    let mut missing_outputs: Vec<String> = Vec::new();
    for &id in ALL_INDICATORS {
        if expected_non_producing(id).is_some() {
            continue;
        }
        let Some(info) = vector_ta::indicators::registry::get_indicator(id) else {
            continue;
        };
        // `pattern_recognition` is the one MATRIX output in vector-ta 0.2.9: it
        // declares a single output called `matrix` and returns 62 x n booleans,
        // one row per candlestick pattern. The feature build decomposes it into
        // one column per pattern using the library's own `pattern_ids`, so the
        // names to expect come from the library's own pattern list — not from a
        // number written down here.
        if id == "pattern_recognition" {
            for spec in vector_ta::indicators::pattern_recognition::list_patterns() {
                let name = format!("{id}_{}", spec.id);
                if !emitted.contains(name.as_str()) {
                    missing_outputs.push(name);
                }
            }
            continue;
        }
        if info.outputs.len() <= 1 {
            if production_output_exclusion(id, None).is_some() {
                continue;
            }
            if !emitted.contains(id) {
                missing_outputs.push(id.to_string());
            }
        } else {
            for o in &info.outputs {
                if production_output_exclusion(id, Some(o.id)).is_some() {
                    continue;
                }
                let name = format!("{id}_{}", o.id);
                if !emitted.contains(name.as_str()) {
                    missing_outputs.push(name);
                }
            }
        }
    }

    // Formula-level exclusions are never dispatched. Seeing one means the
    // production resolver stopped enforcing the static uniqueness contract.
    let stale_production_outputs: Vec<String> = PRODUCTION_OUTPUT_EXCLUSIONS
        .iter()
        .filter_map(|(id, output, _)| {
            let name = output.map_or_else(|| (*id).to_string(), |output| format!("{id}_{output}"));
            emitted.contains(name.as_str()).then_some(name)
        })
        .collect();

    eprintln!("\n=== ALL_INDICATORS VOCABULARY FLOOR ({BARS} synthetic bars) ===");
    eprintln!("declared ids          : {declared}");
    eprintln!("excluded by name      : {excluded}  (EXPECTED_NON_PRODUCING, each with a reason)");
    eprintln!("static redundant ids : {statically_redundant_ids}");
    eprintln!("attemptable           : {attemptable}");
    eprintln!("produced a column     : {producing}   (floor {floor_ids})");
    eprintln!(
        "total columns         : {}   (floor {floor_columns})",
        cols.len()
    );
    eprintln!("silently absent ids   : {}", silent.len());
    if !silent.is_empty() {
        let head: Vec<&str> = silent.iter().take(20).map(|s| **s).collect();
        eprintln!("  {}", head.join(", "));
    }
    eprintln!("stale exclusions      : {}", stale.len());
    if !stale.is_empty() {
        eprintln!("  {}", stale.join(", "));
    }
    eprintln!("missing declared outs : {}", missing_outputs.len());
    if !missing_outputs.is_empty() {
        let head: Vec<&str> = missing_outputs
            .iter()
            .take(20)
            .map(|s| s.as_str())
            .collect();
        eprintln!("  {}", head.join(", "));
    }
    // WHAT THE VOCABULARY IS MADE OF. A single "825 columns" number says
    // nothing about whether the search gained breadth or just more of the same
    // thing, so the census reports the registry's own category for every
    // producing id alongside the columns it contributed.
    let mut by_family: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (id, count) in &per_id {
        let family = vector_ta::indicators::registry::get_indicator(id)
            .map(|i| i.category)
            .unwrap_or("unregistered");
        let e = by_family.entry(family).or_insert((0, 0));
        e.0 += 1;
        e.1 += count;
    }
    eprintln!("-- by registry family (ids / columns) --");
    for (family, (ids, cols_n)) in &by_family {
        eprintln!("  {family:<24} {ids:>4} ids  {cols_n:>5} columns");
    }
    eprintln!("=============================================================\n");

    // Arithmetic that cannot be true: more ids produced than were attemptable.
    assert!(
        producing <= attemptable,
        "{producing} ids produced a column but only {attemptable} were attemptable — an id on \
         EXPECTED_NON_PRODUCING is producing, so the exclusion list is lying about the floor"
    );
    assert!(
        stale_production_outputs.is_empty(),
        "{} output(s) are statically excluded as formula aliases/disabled guides but were \
         emitted: {:?}",
        stale_production_outputs.len(),
        stale_production_outputs
    );
    assert!(
        stale.is_empty(),
        "{} id(s) are named in EXPECTED_NON_PRODUCING but DID produce a column: {:?}\n\n\
         The exclusion list is stale. It is the only thing that lowers this test's floor, so a \
         stale entry is a hole in the guard, not a harmless leftover — delete the entry (and its \
         reason) now that the id works.",
        stale.len(),
        stale
    );

    // NO PARTIALLY-SILENT IDS EITHER.
    //
    // `silent` is computed per ID, and the failure this whole workstream exists
    // to kill was per OUTPUT: an indicator that drops from eight declared series
    // to one still "produces", so the id-level check cannot see it. The expected
    // base-column names are derived straight from vector-ta's registry — not
    // from this crate's `output_ids_for` — so the resolver cannot vouch for
    // itself.
    assert!(
        missing_outputs.is_empty(),
        "{} declared indicator OUTPUT(S) produced no column: {:?}\n\n\
         Each of these is an output vector-ta's registry declares and the feature build did not \
         emit. That is a silent drop at output granularity — the exact shape of the original \
         defect, where every multi-output indicator collapsed. Fix the dispatch or record the \
         reason.",
        missing_outputs.len(),
        missing_outputs.iter().take(40).collect::<Vec<_>>()
    );

    assert!(
        producing >= floor_ids,
        "INDICATOR VOCABULARY COLLAPSE: {producing} of {attemptable} attemptable ids produced a \
         column, below the derived floor of {floor_ids} (80% of the declared list minus the {excluded} \
         ids EXPECTED_NON_PRODUCING names with a reason).\n\
         This is the 2026-08 regression returning: the pass discards results with no error, so \
         everything downstream still 'succeeds' on a fraction of the vocabulary.\n\
         Ids that produced nothing and are not on the exclusion list ({}): {:?}",
        silent.len(),
        silent.iter().take(40).collect::<Vec<_>>()
    );
    assert!(
        cols.len() >= floor_columns,
        "only {} columns from {attemptable} attemptable indicators — multi-output indicators are \
         being collapsed to one series, or dropped",
        cols.len()
    );

    // NO UNEXPLAINED ABSENCES.
    //
    // The 80% band above is a tolerance, and a tolerance is a cap wearing a
    // different hat: it lets an id vanish without anyone writing down why, which
    // is precisely how every other id vanished for sixteen months. The band stays
    // as a coarse collapse detector, but it is NOT allowed to absorb a specific,
    // nameable absence. Either an id produces, or it is on EXPECTED_NON_PRODUCING
    // with a measured reason. There is no third state.
    assert!(
        silent.is_empty(),
        "{} indicator(s) produced NO column and are not named in EXPECTED_NON_PRODUCING: {:?}\n\n\
         Do not lower the floor and do not widen the band. Find out why each one is silent and \
         either fix the dispatch or add it to EXPECTED_NON_PRODUCING with the measured reason \
         (the error variant and the file:line in the vendored crate that produces it). An \
         unexplained absence is the defect this test exists to catch.",
        silent.len(),
        silent
    );
    assert!(
        unattributed.is_empty(),
        "column(s) produced that belong to no id in ALL_INDICATORS: {unattributed:?}"
    );

    // ── PROOF THAT THIS FLOOR WOULD HAVE BEEN RED ──────────────────────────
    //
    // A guard is worth exactly as much as its demonstrated ability to fail.
    // Replay the historical call site and accept logic on the SAME bars:
    // `output_id: None` for every id, and the shape test that read vector-ta's
    // `rows`/`cols` metadata (a 1-D series is reported `rows=1 x cols=n`, so
    // `cols <= 1` was false and the multi-output branch then demanded
    // `rows >= n`, i.e. `1 >= 6000`).
    let historical = replay_historical_dispatch(&ohlcv);
    assert!(
        historical < floor_ids,
        "the historical dispatch produced {historical} columns on these bars, which is ABOVE this \
         test's floor of {floor_ids} — so this test would NOT have caught the all-but-one \
         collapse and its floor is too low to be worth anything"
    );
    eprintln!(
        "falsification: the pre-fix dispatch yields {historical} column(s) on the same bars \
         (floor {floor_ids}) — this test would have been red.\n"
    );
}

/// The ALL_INDICATORS loop exactly as it stood before the 2026-08-09 repair:
/// `params: &[]`, `output_id: None`, and an accept test keyed off
/// `rows`/`cols` instead of the value count. Returns the columns it would emit.
fn replay_historical_dispatch(ohlcv: &Ohlcv) -> usize {
    let n = ohlcv.close.len();
    let candles = Candles::new(
        ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]),
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]),
    );

    let mut kept = 0usize;
    for &id in ALL_INDICATORS {
        let result = catch_unwind(AssertUnwindSafe(|| {
            compute_cpu(IndicatorComputeRequest {
                indicator_id: id,
                output_id: None,
                data: IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                },
                params: &[],
                kernel: Kernel::Auto,
            })
        }));
        let Ok(Ok(o)) = result else { continue };
        match &o.series {
            IndicatorSeries::F64(v) => {
                if o.cols <= 1 {
                    if v.len() >= n {
                        kept += 1;
                    }
                } else if v.len() == o.rows * o.cols && o.rows >= n {
                    kept += o.cols;
                }
            }
            IndicatorSeries::I32(v) if v.len() == n => kept += 1,
            IndicatorSeries::Bool(v) if v.len() == n => kept += 1,
            _ => {}
        }
    }
    kept
}

#[test]
fn the_exclusion_list_names_a_reason_for_every_id() {
    // The exclusion list is the only thing that lowers the floor, so it is the
    // only place a future collapse could hide. Every entry must be a real id
    // and must carry a reason.
    for (id, why) in EXPECTED_NON_PRODUCING {
        assert!(
            ALL_INDICATORS.contains(id),
            "`{id}` is excluded from the vocabulary floor but is not in ALL_INDICATORS — drop \
             the stale exclusion"
        );
        assert!(
            why.trim().len() > 20,
            "`{id}` must say WHY it cannot produce a column, not just that it cannot"
        );
    }
    assert!(
        EXPECTED_NON_PRODUCING.len() * 10 < ALL_INDICATORS.len(),
        "{} of {} indicators are excluded from the floor — an exclusion list that large is a \
         way of not fixing the dispatch",
        EXPECTED_NON_PRODUCING.len(),
        ALL_INDICATORS.len()
    );
}

#[test]
fn vdubus_declared_output_reaches_the_cpu_dispatcher() {
    let ohlcv = synthetic_ohlcv(512);
    let n = ohlcv.close.len();
    let candles = Candles::new(
        ohlcv.timestamp.expect("fixture timestamps"),
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume.expect("fixture volume"),
    );

    let output = compute_cpu(IndicatorComputeRequest {
        indicator_id: "vdubus_divergence_wave_pattern_generator",
        output_id: Some("fast_standard"),
        data: IndicatorDataRef::Candles {
            candles: &candles,
            source: None,
        },
        params: &[],
        kernel: Kernel::Scalar,
    })
    .expect("a registry-declared vdubus output must be dispatchable");

    match output.series {
        IndicatorSeries::F64(values) => assert_eq!(values.len(), n),
        other => panic!("fast_standard must be an f64 series, got {other:?}"),
    }
}

#[test]
fn cvi_existing_f64_implementation_reaches_the_cpu_dispatcher() {
    let ohlcv = synthetic_ohlcv(512);
    let n = ohlcv.close.len();
    let candles = Candles::new(
        ohlcv.timestamp.expect("fixture timestamps"),
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume.expect("fixture volume"),
    );

    let output = compute_cpu(IndicatorComputeRequest {
        indicator_id: "cvi",
        output_id: None,
        data: IndicatorDataRef::Candles {
            candles: &candles,
            source: None,
        },
        params: &[],
        kernel: Kernel::Scalar,
    })
    .expect("the existing f64 cvi implementation must be connected to compute_cpu");

    match output.series {
        IndicatorSeries::F64(values) => assert_eq!(values.len(), n),
        other => panic!("cvi must be an f64 series, got {other:?}"),
    }
}

#[test]
fn marketefi_existing_f64_implementation_reaches_the_cpu_dispatcher() {
    let ohlcv = synthetic_ohlcv(512);
    let n = ohlcv.close.len();
    let candles = Candles::new(
        ohlcv.timestamp.expect("fixture timestamps"),
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume.expect("fixture volume"),
    );

    let output = compute_cpu(IndicatorComputeRequest {
        indicator_id: "marketefi",
        output_id: None,
        data: IndicatorDataRef::Candles {
            candles: &candles,
            source: None,
        },
        params: &[],
        kernel: Kernel::Scalar,
    })
    .expect("the existing f64 marketefi implementation must be connected to compute_cpu");

    match output.series {
        IndicatorSeries::F64(values) => assert_eq!(values.len(), n),
        other => panic!("marketefi must be an f64 series, got {other:?}"),
    }
}

#[test]
fn sar_existing_f64_implementation_reaches_the_cpu_dispatcher() {
    let ohlcv = synthetic_ohlcv(512);
    let n = ohlcv.close.len();
    let candles = Candles::new(
        ohlcv.timestamp.expect("fixture timestamps"),
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume.expect("fixture volume"),
    );
    let expected = sar_with_kernel(
        &SarInput::from_candles(&candles, SarParams::default()),
        Kernel::Scalar,
    )
    .expect("the existing direct f64 sar implementation must remain valid");

    let output = compute_cpu(IndicatorComputeRequest {
        indicator_id: "sar",
        output_id: None,
        data: IndicatorDataRef::Candles {
            candles: &candles,
            source: None,
        },
        params: &[],
        kernel: Kernel::Scalar,
    })
    .expect("the existing f64 sar implementation must be connected to compute_cpu");

    match output.series {
        IndicatorSeries::F64(values) => {
            assert_eq!(values.len(), n);
            assert!(values[0].is_nan(), "SAR must preserve its warmup NaN");
            assert!(values.iter().skip(1).any(|value| value.is_finite()));
            for (index, (actual, direct)) in values.iter().zip(&expected.values).enumerate() {
                assert!(
                    (actual.is_nan() && direct.is_nan()) || actual.to_bits() == direct.to_bits(),
                    "dispatcher SAR differs from direct f64 SAR at index {index}: actual={actual:?}, direct={direct:?}"
                );
            }
        }
        other => panic!("sar must be an f64 series, got {other:?}"),
    }
}
