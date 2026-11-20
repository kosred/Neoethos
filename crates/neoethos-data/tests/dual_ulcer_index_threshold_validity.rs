//! Regression for the Classic v9 dual-ulcer repair.
//!
//! The rolling square sum is maintained by add/subtract. It can round a few
//! ulps below zero, and sqrt() then returned NaN. In the cumulative threshold
//! lane that one NaN was added to diff_sum and every later row became NaN, so
//! the archived M5 run fitted only 95 of 633,010 training cells for
//! dual_ulcer_index_threshold. The repair clamps the impossible negative
//! before the square root; this test reproduces the original failure on the
//! committed real EURUSD M15 export and proves the repaired lane stays finite.
//!
//! cargo test -p neoethos-data --test dual_ulcer_index_threshold_validity

use std::fs;

use vector_ta::indicators::dual_ulcer_index::{
    DualUlcerIndexInput, DualUlcerIndexParams, DualUlcerIndexStream, dual_ulcer_index,
};

const PERIOD: usize = 5;
const WARMUP: usize = 2 * PERIOD - 2;

fn committed_m15_close() -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../neoethos-models/tests/fixtures/eurusd_m15_real.csv");
    let raw = fs::read_to_string(&path).expect("committed real EURUSD M15 export");
    let mut close = Vec::new();
    for (row, line) in raw.lines().enumerate() {
        if row == 0 {
            assert!(
                line.starts_with("timestamp_ms,"),
                "fixture must keep the audited timestamp_ms,open,high,low,close header"
            );
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 5, "fixture row {row} must be OHLCV-shaped");
        close.push(
            fields[4]
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("fixture row {row} close: {error}")),
        );
    }
    assert!(
        close.len() > 6_000,
        "the export must be long enough to expose the rounded-negative sum"
    );
    close
}

#[test]
fn real_m15_dual_ulcer_threshold_stays_finite_after_warmup() {
    let close = committed_m15_close();
    let output = dual_ulcer_index(&DualUlcerIndexInput::from_slice(
        &close,
        DualUlcerIndexParams::default(),
    ))
    .expect("audited dual ulcer computation");

    assert_eq!(output.long_ulcer.len(), close.len());
    assert_eq!(output.short_ulcer.len(), close.len());
    assert_eq!(output.threshold.len(), close.len());

    for row in 0..WARMUP {
        assert!(output.long_ulcer[row].is_nan(), "long warmup row {row}");
        assert!(output.short_ulcer[row].is_nan(), "short warmup row {row}");
        assert!(output.threshold[row].is_nan(), "threshold warmup row {row}");
    }
    for row in WARMUP..close.len() {
        assert!(
            output.long_ulcer[row].is_finite(),
            "long ulcer row {row} must not be poisoned by a rounded-negative sum"
        );
        assert!(
            output.short_ulcer[row].is_finite(),
            "short ulcer row {row} must stay finite"
        );
        assert!(
            output.threshold[row].is_finite(),
            "cumulative threshold row {row} must not inherit a NaN diff"
        );
    }

    // The original defect first appeared here on this export; keep the exact
    // row named so a regression reopens this file instead of searching again.
    let reproducer_row = 5_900;
    assert!(reproducer_row < close.len());
    assert!(output.long_ulcer[reproducer_row].is_finite());
    assert!(output.threshold[reproducer_row].is_finite());
    assert!(
        output.threshold[reproducer_row] >= 0.0,
        "the cumulative absolute diff mean is never negative"
    );
}

#[test]
fn streaming_dual_ulcer_threshold_keeps_the_same_repaired_values() {
    let close = committed_m15_close();
    let params = DualUlcerIndexParams::default();
    let mut stream = DualUlcerIndexStream::try_new(params.clone()).expect("stream params");
    let mut long_ulcer = Vec::with_capacity(close.len());
    let mut short_ulcer = Vec::with_capacity(close.len());
    let mut threshold = Vec::with_capacity(close.len());
    for &value in &close {
        match stream.update(value) {
            Some((long, short, threshold_value)) => {
                long_ulcer.push(long);
                short_ulcer.push(short);
                threshold.push(threshold_value);
            }
            None => {
                long_ulcer.push(f64::NAN);
                short_ulcer.push(f64::NAN);
                threshold.push(f64::NAN);
            }
        }
    }

    let batch = dual_ulcer_index(&DualUlcerIndexInput::from_slice(&close, params))
        .expect("audited batch dual ulcer");

    for row in WARMUP..close.len() {
        assert!(
            threshold[row].is_finite(),
            "stream threshold row {row} must not be poisoned"
        );
        assert_eq!(
            threshold[row].to_bits(),
            batch.threshold[row].to_bits(),
            "stream and batch threshold row {row} must stay operation-identical"
        );
        assert_eq!(long_ulcer[row].to_bits(), batch.long_ulcer[row].to_bits());
        assert_eq!(short_ulcer[row].to_bits(), batch.short_ulcer[row].to_bits());
    }
}
